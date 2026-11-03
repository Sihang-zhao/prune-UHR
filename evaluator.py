"""Dataset evaluation, worker processes, checkpoints, and reports for testv."""
from __future__ import annotations

import argparse
import shutil
import time
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from utils.data import Case
from utils.metrics import evaluate, load_jsonl, save_jsonl, save_markdown


DEFAULT_DATA = os.environ.get("XLRS_DATA_ROOT")
_EVAL_MAX_IMAGE_PIXELS = 4_194_304
_EARLY_MIDDLE_MODES = {"middle_text_early", "middle_text_early_grid"}
_MIDDLE_TEXT_MODES = {"middle_text", *_EARLY_MIDDLE_MODES}


def _validate_pruning_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    """Validate one full pruning record before its large metadata is discarded."""

    if row.get("status") == "error" or row.get("selection_verified"):
        return False
    row_mode = str(row.get("selection_mode", args.pruning_mode))
    if row_mode != str(args.pruning_mode):
        raise RuntimeError(
            f"case {row.get('index')} selection mode mismatch: "
            f"row={row_mode}, requested={args.pruning_mode}"
        )
    row_strategy = str(row.get("selection_strategy", getattr(args, "pruning_strategy", "topk")))
    requested_strategy = str(getattr(args, "pruning_strategy", "topk"))
    if row_strategy != requested_strategy:
        raise RuntimeError(
            f"case {row.get('index')} selection strategy mismatch: "
            f"row={row_strategy}, requested={requested_strategy}"
        )
    if row_mode in _MIDDLE_TEXT_MODES and int(row.get("selection_layer", args.pruning_layer)) != int(args.pruning_layer):
        raise RuntimeError(
            f"case {row.get('index')} selection layer mismatch: "
            f"row={row.get('selection_layer')}, requested={args.pruning_layer}"
        )
    if row_mode in _EARLY_MIDDLE_MODES:
        layer_counts = row.get("vision_layer_token_counts", [])
        merge_unit = int(row.get("vision_merge_unit", 0))
        if merge_unit <= 0 or len(layer_counts) <= int(args.pruning_layer):
            raise RuntimeError(f"case {row.get('index')} is missing the early ViT execution trace")
        full_count = int(row["original_image_tokens"]) * merge_unit
        compact_count = int(row["retained_tokens"]) * merge_unit
        full_blocks = int(args.pruning_layer) + int(args.compression_mode != "hard18")
        expected = [full_count] * full_blocks + [compact_count] * (len(layer_counts) - full_blocks)
        if layer_counts != expected:
            raise RuntimeError(f"case {row.get('index')} violated the requested ViT execution trace")
    budgets = row.get("selection_budgets", []) or []
    indices = row.get("selected_indices", []) or []
    if len(budgets) != len(indices):
        raise RuntimeError(f"selection metadata mismatch for case {row.get('index')}")
    groups = row.get("selection_groups", []) or []
    if len(groups) != len(budgets):
        raise RuntimeError(f"selection group metadata mismatch for case {row.get('index')}")
    for budget, selected, group in zip(budgets, indices, groups):
        budget = tuple(int(value) for value in budget)
        selected = [int(value) for value in selected]
        if sum(budget) != len(selected):
            raise RuntimeError(f"case {row.get('index')} has inconsistent percentage budget")
        if args.visual_keep_tokens is not None and len(selected) != int(args.visual_keep_tokens):
            raise RuntimeError(f"case {row.get('index')} violated override K={args.visual_keep_tokens}")
        if selected != sorted(set(selected)):
            raise RuntimeError(f"case {row.get('index')} visual tokens are not in original patch order")
        final = [int(value) for value in group.get("final", [])]
        middle = [int(value) for value in group.get("middle", [])]
        shallow = [int(value) for value in group.get("shallow", [])]
        if (len(final), len(middle), len(shallow)) != (budget[0], budget[1], budget[2]):
            raise RuntimeError(f"case {row.get('index')} violated final/middle/shallow token budgets")
        if any(values != sorted(set(values)) for values in (final, middle, shallow)):
            raise RuntimeError(f"case {row.get('index')} has an out-of-order category selection")
        if set(final) & set(middle) or set(final) & set(shallow) or set(middle) & set(shallow):
            raise RuntimeError(f"case {row.get('index')} has overlapping token categories")
        if sorted(final + middle + shallow) != selected:
            raise RuntimeError(f"case {row.get('index')} category union differs from selected tokens")
    return True


def _check_pruning_rows(rows: list[dict[str, Any]], args: argparse.Namespace, devices: list[str], logger: "_RunLogger") -> None:
    """Validate fixed-K/percentage budgets, order, and worker sharding."""

    if args.visual_pruning:
        checked = 0
        for row in rows:
            if row.get("status") == "error":
                continue
            _validate_pruning_row(row, args)
            checked += 1
        mode = (
            f"K={args.visual_keep_tokens}"
            if args.visual_keep_tokens is not None
            else f"keep_ratio={args.visual_keep_ratio:.4f}"
        )
        if args.pruning_mode in _MIDDLE_TEXT_MODES:
            mode += f"; layer={args.pruning_layer}"
            budget_label = f"{args.prune_method}; execution={args.compression_mode}"
        else:
            budget_label = "budget=6:2:2"
        logger.write(f"pruning checks passed: cases={checked}; {mode}; {budget_label}")
    if len(devices) > 1:
        active = {devices[index] for index in range(min(len(devices), len(rows)))}
        observed = {str(row.get("device")) for row in rows if row.get("device")}
        if not active.issubset(observed):
            raise RuntimeError(f"multi-card worker check failed: expected={sorted(active)}, observed={sorted(observed)}")
        logger.write(f"multi-card inference check passed: devices={','.join(sorted(observed))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Q-FlowCover on Qwen3-VL VQA data", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        required=not bool(DEFAULT_DATA),
        help="JSON/JSONL/Parquet annotations or local HuggingFace dataset; also accepts XLRS_DATA_ROOT",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="Hub model id or local checkpoint directory")
    parser.add_argument("--image-root", default=None, help="Root for relative image paths")
    parser.add_argument("--split", default="train", help="HuggingFace dataset split")
    parser.add_argument("--output-dir", default="outputs", help="Directory containing timestamped run folders")
    parser.add_argument("--output", default=None, help="Optional additional JSONL output path")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="Resume from the latest run under --output-dir, retrying only missing/error cases")
    parser.add_argument("--resume-from", default=None, help="Existing JSONL result file to resume; successful case indices are skipped")
    parser.add_argument("--log-file", default=None, help="Optional log path; default is the timestamped run folder")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids, for example 0,1")
    parser.add_argument("--num-gpus", type=int, default=None, help="Use the first n visible GPUs")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp16", "bf16", "fp32"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False, help="Enable Qwen thinking mode; disabled by default for benchmark-style VQA")
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=_EVAL_MAX_IMAGE_PIXELS,
        help="Total visual pixel budget passed to the Qwen processor; use 16777216 for the reference 16M cap",
    )
    parser.add_argument(
        "--visual-keep-ratio",
        type=float,
        default=0.5,
        help="Fraction of visual tokens retained per image view",
    )
    parser.add_argument(
        "--visual-prune-percent",
        type=float,
        default=None,
        help="Percentage of visual tokens removed; 50/70/90 means retain 50%%/30%%/10%% (must be below 100)",
    )
    parser.add_argument(
        "--visual-keep-tokens",
        type=int,
        default=None,
        help="Optional absolute per-view token override; when set it takes precedence over --visual-keep-ratio",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--worker-cases",
        type=int,
        default=128,
        help=(
            "Restart each inference worker after at most this many cases "
            "to bound long-run CUDA/CPU allocator growth; "
            "use 0 to process all cases in one worker"
        ),
    )
    parser.add_argument("--max-image-patches", type=int, default=1, help="Total Qwen image views, including the global view (kept low to bound multimodal context memory)")
    parser.add_argument(
        "--decode-hf-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decode HuggingFace Image features before load_image; disabled to limit memory use",
    )
    parser.add_argument("--patch-selection", choices=("uniform", "head", "tail"), default="uniform")
    parser.add_argument("--include-global-view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-single-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-size", type=int, default=448)
    parser.add_argument("--patch-stride", type=int, default=336)
    parser.add_argument("--global-view-size", type=int, default=448)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=False)
    return parser


class _RunLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")

    def write(self, message: str, *, error: bool = False) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(message, file=sys.stderr if error else sys.stdout, flush=True)
        self.file.write(f"{timestamp} {message}\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def _new_run_dir(root: str | Path) -> Path:
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d_%H%M")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _worker_dtype(requested: str, device: str) -> str:
    value = str(requested or "auto").lower()
    if value != "auto":
        return value
    if not str(device).startswith("cuda"):
        return "fp32"
    import torch
    try:
        bf16_ok = bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        bf16_ok = False
    return "bf16" if bf16_ok else "fp16"


def _process_exit_detail(code: int) -> str:
    """Explain a multiprocessing negative exit code as a Unix signal."""

    if code >= 0:
        return str(code)
    try:
        name = signal.Signals(-code).name
    except ValueError:
        name = "unknown signal"
    return f"{code} ({name})"


def _run_parallel(cases: list[Case], args: argparse.Namespace, devices: list[str], logger: _RunLogger) -> list[dict[str, Any]]:
    if not cases:
        return []
    from utils.data import case_dict
    from compression.worker import worker

    context = mp.get_context("spawn")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    case_file = output_path.open("w", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    completed = correct_count = error_count = 0
    requested_batch = int(getattr(args, "worker_cases", 128) or 0)
    batch_size = len(cases) if requested_batch <= 0 else max(1, requested_batch)
    config = vars(args).copy()

    # Restarting bounds allocator/decoder growth that can survive per-case
    # cleanup.  One process previously handled all 3080 cases and was killed
    # only after hundreds of successful requests.
    def run_batch(batch: list[Case], progress: Any) -> None:
        nonlocal completed, correct_count, error_count
        result_queue = context.Queue()
        shards = [batch[index::len(devices)] for index in range(len(devices))]
        processes = [
            context.Process(
                target=worker,
                args=(device, [case_dict(case) for case in shard], config, result_queue),
            )
            for device, shard in zip(devices, shards) if shard
        ]
        worker_errors: list[str] = []
        finished = 0
        for process in processes:
            process.start()
        try:
            while finished < len(processes):
                try:
                    kind, value = result_queue.get(timeout=1.0)
                except queue.Empty:
                    if all(not process.is_alive() for process in processes):
                        try:
                            kind, value = result_queue.get(timeout=2.0)
                        except queue.Empty:
                            break
                    else:
                        continue
                if kind == "row":
                    # Validate while full pruning metadata is still available;
                    # retain only a compact row in memory after writing it.
                    if args.visual_pruning and value.get("status") != "error":
                        try:
                            _validate_pruning_row(value, args)
                        except Exception as exc:
                            worker_errors.append(f"case {value.get('index')} pruning validation failed: {exc}")
                            value = dict(value)
                            value["status"] = "error"
                            value["correct"] = False
                            value["error"] = f"pruning validation failed: {exc}"
                    completed += 1
                    correct_count += int(bool(value.get("correct")))
                    error_count += int(value.get("status") == "error")
                    case_file.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
                    case_file.flush()
                    compact_value = dict(value)
                    if args.visual_pruning and compact_value.get("status") != "error":
                        compact_value.pop("selected_indices", None)
                        compact_value.pop("selection_groups", None)
                        compact_value["selection_verified"] = True
                    rows.append(compact_value)
                    if progress:
                        progress.update(1)
                        progress.set_postfix(acc=f"{correct_count / completed:.4f}", errors=error_count)
                    elif completed % 10 == 0 or completed == len(cases):
                        logger.write(f"Testing {completed}/{len(cases)} acc={correct_count / completed:.4f} errors={error_count}")
                    if value.get("status") == "error" and error_count <= 3:
                        logger.write(f"case {value.get('index')} failed: {value.get('error', 'unknown error')}", error=True)
                    value = None
                elif kind == "worker_error":
                    worker_errors.append(str(value))
                    logger.write(f"worker error: {value}", error=True)
                else:
                    finished += 1
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            try:
                result_queue.close()
                result_queue.join_thread()
            except (AttributeError, OSError):
                pass

        failed_codes = [process.exitcode for process in processes if process.exitcode not in (0, None)]
        if failed_codes:
            details = list(worker_errors)
            if any(code == -9 for code in failed_codes):
                details.append(
                    "worker received SIGKILL (-9), usually GPU/CPU OOM; "
                    "use --dtype bf16 or --dtype fp16, reduce --max-image-patches, "
                    "or lower --max-image-pixels"
                )
            if any(code == -15 for code in failed_codes):
                details.append(
                    "worker received SIGTERM (-15), usually the cluster wall-time limit, "
                    "preemption, cancellation, or an administrator action; "
                    "partial results were flushed and can be continued with --resume"
                )
            exit_detail = ", ".join(_process_exit_detail(code) for code in failed_codes)
            detail = "; ".join(details) or "process exit codes: " + exit_detail
            if details:
                detail += f"; process exit codes: {exit_detail}"
            raise RuntimeError("worker process failed: " + detail)
        batch_indices = {int(case.index) for case in batch}
        batch_rows = {int(row.get("index", -1)) for row in rows if int(row.get("index", -1)) in batch_indices}
        if len(batch_rows) != len(batch):
            raise RuntimeError(
                f"inference produced {len(batch_rows)}/{len(batch)} rows in worker batch; "
                f"{len(batch) - len(batch_rows)} cases are missing"
            )
        if worker_errors:
            raise RuntimeError("worker/row validation failed: " + "; ".join(worker_errors[:8]))

    progress = None
    try:
        try:
            from tqdm import tqdm
            progress = tqdm(total=len(cases), desc="Testing", unit="case", dynamic_ncols=True)
        except ImportError:
            progress = None
            logger.write(f"Testing 0/{len(cases)}")
        for start in range(0, len(cases), batch_size):
            batch = cases[start : start + batch_size]
            if len(cases) > batch_size:
                logger.write(
                    f"Starting worker batch {start // batch_size + 1}/"
                    f"{(len(cases) + batch_size - 1) // batch_size} "
                    f"({len(batch)} cases; workers restart after batch)"
                )
            run_batch(batch, progress)
    finally:
        if progress is not None:
            progress.close()
        case_file.close()
    if completed != len(cases):
        missing = len(cases) - completed
        raise RuntimeError(f"inference produced {completed}/{len(cases)} rows; {missing} cases are missing")
    return sorted(rows, key=lambda row: int(row.get("index", 0)))


def _gpu_ids(args: argparse.Namespace) -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for testing") from exc
    if not torch.cuda.is_available():
        if args.gpus or (args.num_gpus and args.num_gpus > 1):
            raise RuntimeError("CUDA is unavailable but multiple GPUs were requested")
        return ["cpu"]
    count = torch.cuda.device_count()
    ids = [part.strip() for part in str(args.gpus).split(",") if part.strip()] if args.gpus else [str(i) for i in range(args.num_gpus or 1)]
    if not ids or len(set(ids)) != len(ids) or len(ids) > count or any(not item.isdigit() or int(item) >= count for item in ids):
        raise ValueError(f"GPU ids {ids} are invalid; visible device count is {count}")
    return [f"cuda:{item}" for item in ids]


def _resolve_model(name: str) -> str:
    requested = Path(name).expanduser()
    if requested.is_dir():
        return str(requested.resolve())
    return name


def _resume_source(args: argparse.Namespace) -> Path | None:
    if args.resume_from:
        return Path(args.resume_from).expanduser()
    if not args.resume:
        return None
    candidates = sorted(Path(args.output_dir).expanduser().glob("*/results.jsonl"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _select_resume_cases(cases: list[Case], source: Path | None, logger: _RunLogger, requested: bool = False) -> tuple[list[Case], list[dict[str, Any]]]:
    if source is None:
        if requested:
            logger.write("resume requested but no previous run was found; starting a fresh run")
        return cases, []
    previous = load_jsonl(source)
    current = {int(case.index) for case in cases}
    successful = {int(row["index"]) for row in previous if row.get("status") != "error" and int(row["index"]) in current}
    pending = [case for case in cases if int(case.index) not in successful]
    logger.write(f"Resuming from {source}: reused={len(successful)} pending={len(pending)}")
    return pending, [row for row in previous if int(row.get("index", -1)) in successful]


def _validate_resume_config(source: Path, args: argparse.Namespace) -> None:
    config_path = source.parent / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Cannot verify resume settings: missing {config_path}")
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    keys = (
        "data", "image_root", "split", "model", "dtype", "prune_method", "compression_mode",
        "visual_keep_ratio", "visual_keep_tokens", "pruning_layer", "q_flowcover_radius",
        "q_flowcover_query_chunk", "q_flowcover_candidate_chunk", "max_image_pixels",
        "max_image_patches", "patch_selection", "include_global_view", "tile_single_images",
        "patch_size", "patch_stride", "global_view_size", "enable_thinking", "max_new_tokens",
    )
    mismatches = [key for key in keys if key not in previous or previous[key] != getattr(args, key)]
    if mismatches:
        raise ValueError("Resume settings differ for: " + ", ".join(mismatches) + "; use a matching run or start a new run")


def test(args: argparse.Namespace) -> dict[str, Any]:
    from utils.data import load_cases

    if args.max_image_patches <= 0:
        raise ValueError("max_image_patches must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.max_image_pixels <= 0:
        raise ValueError("max_image_pixels must be positive")
    if args.visual_prune_percent is not None:
        if not 0.0 <= float(args.visual_prune_percent) < 100.0:
            raise ValueError("visual_prune_percent must be in [0, 100)")
        if args.visual_keep_tokens is not None:
            raise ValueError("visual_prune_percent cannot be combined with visual_keep_tokens")
        args.visual_keep_ratio = 1.0 - float(args.visual_prune_percent) / 100.0
    if args.visual_keep_tokens is not None and args.visual_keep_tokens <= 0:
        raise ValueError("visual_keep_tokens must be positive")
    if args.visual_pruning and (
        args.visual_keep_tokens is None and not 0.0 < args.visual_keep_ratio <= 1.0
    ):
        raise ValueError("visual_keep_ratio must be in (0, 1]")
    if args.pruning_mode in _MIDDLE_TEXT_MODES:
        layers = tuple(int(value.strip()) for value in str(args.vision_layers).split(",") if value.strip())
        if args.pruning_layer not in layers:
            if len(layers) == 3 and layers[0] < args.pruning_layer < layers[-1]:
                # Keep the legacy multi-layer default (6,12,24), but make the
                # single-layer mode select the requested fixed middle block.
                args.vision_layers = f"{layers[0]},{args.pruning_layer},{layers[-1]}"
            else:
                raise ValueError(
                    "pruning_layer must lie between the first and last --vision-layers "
                    f"for {args.pruning_mode} mode; got {args.pruning_layer}, available={layers}"
                )
        if args.pruning_mode in _EARLY_MIDDLE_MODES and args.pruning_layer >= layers[-1]:
            raise ValueError(
                f"{args.pruning_mode} requires --pruning-layer before the final --vision-layers entry; "
                f"got layer={args.pruning_layer}, final={layers[-1]}"
            )
    run_started = time.perf_counter()
    run_dir = _new_run_dir(getattr(args, "output_dir", "outputs"))
    requested_output = getattr(args, "output", None)
    args.output = str(run_dir / "results.jsonl")
    args.log_file = getattr(args, "log_file", None) or str(run_dir / "test.log")
    logger = _RunLogger(args.log_file)
    try:
        args.data = str(Path(args.data).expanduser().resolve())
        logger.write(f"Starting test: data={args.data}")
        args.vision_layers = ",".join(str(int(value.strip())) for value in str(args.vision_layers).split(",") if value.strip())
        if args.image_root is None:
            data_path = Path(args.data).expanduser()
            args.image_root = str(data_path if data_path.is_dir() else data_path.parent)
        args.image_root = str(Path(args.image_root).expanduser().resolve())
        original_model = args.model
        args.model = _resolve_model(args.model)
        if args.model != original_model:
            args.offline = True
            logger.write(f"Using local model: {args.model}")
        cases = load_cases(args.data, args.limit, args.split)
        if not cases:
            raise ValueError("No test cases found")
        devices = _gpu_ids(args)
        effective_dtypes = sorted({_worker_dtype(args.dtype, device) for device in devices})
        _write_json(run_dir / "config.json", {**vars(args), "effective_dtypes": effective_dtypes, "requested_output": requested_output, "run_dir": str(run_dir), "devices": devices, "case_count": len(cases)})
        keep_mode = (
            f"K={args.visual_keep_tokens}"
            if args.visual_keep_tokens is not None
            else f"keep_ratio={args.visual_keep_ratio:.4f}"
        )
        if args.visual_prune_percent is not None:
            keep_mode = f"prune_percent={args.visual_prune_percent:.1f}; {keep_mode}"
        logger.write(
            f"Loaded {len(cases)} cases; devices={','.join(devices)}; dtype={','.join(effective_dtypes)}; "
            f"max_image_patches={args.max_image_patches}; max_image_pixels={args.max_image_pixels}; "
            f"decode_hf_images={args.decode_hf_images}; "
            f"visual_pruning={args.visual_pruning}; {keep_mode}; "
            f"pruning_mode={args.pruning_mode}; pruning_layer={args.pruning_layer}; "
            f"pruning_strategy={args.pruning_strategy}; "
            f"vision_layers={args.vision_layers}"
        )
        resume_source = _resume_source(args)
        if resume_source is not None:
            _validate_resume_config(resume_source, args)
        active_cases, reused_rows = _select_resume_cases(cases, resume_source, logger, bool(args.resume or args.resume_from))
        generated_rows = _run_parallel(active_cases, args, devices, logger)
        rows = reused_rows + generated_rows
        rows = sorted({int(row.get("index", -1)): row for row in rows}.values(), key=lambda row: int(row.get("index", 0)))
        if resume_source is not None:
            # _run_parallel writes full pruning metadata for active cases, but
            # keeps a compact in-memory row.  Rebuild the checkpoint from the
            # on-disk full rows so a second resume never loses selection data.
            generated_disk = load_jsonl(args.output) if Path(args.output).is_file() else []
            merged_disk = sorted(
                {int(row.get("index", -1)): row for row in reused_rows + generated_disk}.values(),
                key=lambda row: int(row.get("index", 0)),
            )
            save_jsonl(args.output, merged_disk)
        _check_pruning_rows(rows, args, devices, logger)
        # _run_parallel already streamed the complete rows (including full
        # pruning metadata) to the run JSONL.  The in-memory rows are compact
        # on purpose, so never rewrite the run file with those reduced rows.
        if requested_output:
            requested_path = Path(requested_output).expanduser()
            requested_path.parent.mkdir(parents=True, exist_ok=True)
            if requested_path.resolve() != Path(args.output).resolve():
                shutil.copyfile(args.output, requested_path)
        summary = evaluate(rows)
        retained_values = [int(row.get("retained_tokens", 0) or 0) for row in rows]
        retention_ratios = [float(row.get("retention_ratio", 0.0) or 0.0) for row in rows if row.get("status") != "error"]
        pruning_values = [float(row.get("pruning_latency", 0.0) or 0.0) for row in rows if row.get("status") != "error"]
        prefill_values = [float(row.get("prefill_latency", 0.0) or 0.0) for row in rows]
        summary["retained_tokens"] = sum(retained_values)
        summary["avg_retained_tokens"] = sum(retained_values) / len(retained_values) if retained_values else 0.0
        summary["avg_retention_ratio"] = sum(retention_ratios) / len(retention_ratios) if retention_ratios else 0.0
        summary["avg_pruning_latency"] = sum(pruning_values) / len(pruning_values) if pruning_values else 0.0
        summary["avg_prefill_latency"] = sum(prefill_values) / len(prefill_values) if prefill_values else 0.0
        summary["total_test_time"] = time.perf_counter() - run_started
        _write_json(run_dir / "summary.json", summary)
        # save_markdown predates fixed-K pruning and only uses this value for a
        # display label; the detailed fixed-K metrics are in summary/results.
        markdown_config = {
            **vars(args),
            "visual_keep_ratio": args.visual_keep_ratio
            if args.visual_keep_tokens is None
            else 1.0,
        }
        markdown_path = run_dir / "results.md"
        save_markdown(markdown_path, summary, config=markdown_config)
        budget_label = (
            f"retained {args.visual_keep_ratio:.1%} per view"
            if args.visual_keep_tokens is None
            else f"fixed K={args.visual_keep_tokens} per view"
        )
        pruning_label = f"{args.prune_method}; {args.compression_mode}; layer 18; {budget_label}"
        markdown_text = markdown_path.read_text(encoding="utf-8")
        method_start = markdown_text.find("<tbody>\n<tr><td>")
        if method_start >= 0:
            method_start += len("<tbody>\n<tr><td>")
            method_end = markdown_text.find("</td>", method_start)
            if method_end > method_start:
                markdown_text = (
                    markdown_text[:method_start]
                    + pruning_label
                    + markdown_text[method_end:]
                )
        markdown_path.write_text(markdown_text, encoding="utf-8")
        logger.write(
            f"overall micro AC: {summary['micro_acc']:.4f}; "
            f"macro task AC: {summary['macro_acc']:.4f} "
            f"({summary['correct']}/{summary['total']}) errors: {summary['errors']}"
        )
        logger.write(f"retained tokens: total={summary['retained_tokens']} avg={summary['avg_retained_tokens']:.2f}")
        logger.write(f"retention ratio: avg={summary['avg_retention_ratio']:.4f}")
        logger.write(f"pruning latency: avg={summary['avg_pruning_latency']:.4f}s")
        logger.write(f"prefill latency: avg={summary['avg_prefill_latency']:.4f}s")
        logger.write(f"avg testtime: {summary['avg_testtime']:.4f}s avg estimated TFLOPs: {summary['avg_tflops']:.6f} total estimated TFLOPs: {summary['total_tflops']:.6f}")
        for name, value in summary["taskL3"].items():
            logger.write(f"taskL3 {name}: {value['acc']:.4f} ({value['correct']}/{value['total']})")
        logger.write(f"total test time: {summary['total_test_time']:.4f}s")
        logger.write(f"saved: {args.output}")
        logger.write(f"saved: {run_dir / 'results.md'}")
        if requested_output:
            logger.write(f"copied: {requested_output}")
        return summary
    except Exception as exc:
        logger.write(f"test failed: {type(exc).__name__}: {exc}", error=True)
        raise
    finally:
        logger.close()
