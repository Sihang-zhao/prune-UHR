"""Spawn-safe worker for the independent ``testv.py`` entry point."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.data import Case, case_dict, load_cases, load_image
from utils.metrics import score_prediction

from .qwen3_vl_modes import Qwen3VLCompressionRunner


def _path_value(value: Any, root: str | None) -> str | list[str]:
    if isinstance(value, Mapping) and "__hf__" in value:
        return f"{value['__hf__']}::{value.get('split', 'train')}[{value.get('index', '?')}]"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_path_value(item, root) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return "<embedded-bytes>"
    if not isinstance(value, (str, Path)):
        return "<embedded-image>"
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and root:
        path = Path(root).expanduser() / path
    return str(path)


def _worker_dtype(requested: str, device: str) -> str:
    value = str(requested or "auto").lower()
    if value != "auto":
        return value
    import torch

    if not str(device).startswith("cuda"):
        return "fp32"
    try:
        bf16_ok = bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        bf16_ok = False
    return "bf16" if bf16_ok else "fp16"


def _centrality_topk(att_stats: Mapping[str, Any], k: int = 10) -> list[dict[str, Any]]:
    values = att_stats.get("source_centrality", [])
    rows: list[dict[str, Any]] = []
    for view_index, vector in enumerate(values):
        try:
            import torch

            count = min(int(k), int(vector.numel()))
            if count <= 0:
                continue
            scores, indices = torch.topk(vector.float(), k=count, largest=True, sorted=True)
            rows.extend(
                {
                    "view": view_index,
                    "source_index": int(index),
                    "centrality": float(score),
                }
                for score, index in zip(scores.detach().cpu().tolist(), indices.detach().cpu().tolist())
            )
        except Exception:
            continue
    return rows


def _extra_record(output: Any, case: Case, cfg: Mapping[str, Any], details: Mapping[str, Any]) -> dict[str, Any]:
    metadata = getattr(output, "__dict__", {})
    stats = getattr(output, "att_stats", {}) or {}
    stage_timings = dict(getattr(output, "stage_timings", {}) or {})
    stage_timings.setdefault("llm_prefill", float(getattr(output, "prefill_latency", 0.0)))
    stage_timings.setdefault("llm_decode", float(getattr(output, "decode_seconds", 0.0)))
    stage_timings.setdefault("end_to_end", float(getattr(output, "seconds", 0.0)))
    result = {
        "sample_id": case.id,
        "task_category": case.category,
        "dataset_split": cfg.get("split"),
        "model_revision": cfg.get("model"),
        "run_id": Path(str(cfg.get("output", ""))).parent.name if cfg.get("output") else None,
        "commit": "unknown",
        "compression_mode": cfg.get("compression_mode"),
        "prune_method": cfg.get("prune_method", "text_cos"),
        "decision_layer_human": int(cfg.get("decision_layer_human", 18)),
        "decision_block_index": metadata.get("decision_block_index"),
        "bridge_block_index": metadata.get("bridge_block_index"),
        "raw_patch_count": metadata.get("raw_patch_count", output.original_image_tokens),
        "merged_group_count": metadata.get("merged_group_count", output.original_image_tokens),
        "retained_group_count": metadata.get("retained_group_count", output.retained_tokens),
        "actual_llm_visual_tokens": metadata.get("actual_llm_visual_tokens", output.image_tokens),
        "image_grid_thw": metadata.get("image_grid_thw"),
        "input_resolution": [
            [int(row[1]), int(row[2])] for row in metadata.get("image_grid_thw", [])
        ],
        "keep_idx": metadata.get("keep_idx", output.selected_indices),
        "parser_status": details.get("status", "unknown"),
        "Gamma": stats.get("Gamma"),
        "source_centrality_topk": _centrality_topk(stats),
        "readout_gap": stats.get("readout_gap"),
        "att_stats": stats if cfg.get("save_att_stats") else {},
        "stage_timings": stage_timings,
        "bridge_time": float(stage_timings.get("bridge", 0.0)),
        "peak_memory": getattr(output, "peak_memory", 0),
        "exception_count": 0,
        "GPU": cfg.get("device"),
        "dtype": cfg.get("dtype"),
        "attention_backend": (
            "eager_reference" if cfg.get("att_stats") != "none" else "sdpa/native"
        ),
        "decoding_config": {
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": int(cfg.get("max_new_tokens", 0)),
        },
        "total_wall_time": float(getattr(output, "seconds", 0.0)),
        "selector_time": float(stage_timings.get("selector", 0.0)),
        "selector_flops": int(getattr(output, "selector_flops", stage_timings.get("selector_flops", 0.0)) or 0),
        "prefill_time": float(getattr(output, "prefill_latency", 0.0)),
        "decode_time": float(getattr(output, "decode_seconds", 0.0)),
        "total_test_time": float(stage_timings.get("end_to_end", getattr(output, "seconds", 0.0))),
        "visual_prefix_time": float(stage_timings.get("visual_prefix", stage_timings.get("prefix", 0.0))),
        "visual_suffix_time": float(stage_timings.get("visual_suffix", 0.0)),
        "index_expand_time": float(stage_timings.get("index_expand", 0.0)),
    }
    q_flowcover = metadata.get("q_flowcover", getattr(output, "q_flowcover_debug", [])) or []
    result["q_flowcover"] = q_flowcover
    if q_flowcover:
        result.update(
            {
                "text_topk": [row.get("text_topk", []) for row in q_flowcover],
                "final_group_index": [row.get("final_group_index", []) for row in q_flowcover],
                "N": [int(row.get("N", 0)) for row in q_flowcover],
                "K": [int(row.get("K", 0)) for row in q_flowcover],
                "pi": [row.get("pi", []) for row in q_flowcover],
                "support": [row.get("support", []) for row in q_flowcover],
                "omega": [row.get("omega", []) for row in q_flowcover],
                "local_radius": [int(row.get("local_radius", 0)) for row in q_flowcover],
                "edge_work": [int(row.get("edge_work", 0)) for row in q_flowcover],
                "replacement_group_count": int(
                    sum(int(row.get("replacement_group_count", 0)) for row in q_flowcover)
                ),
                "raw_keep_idx_hash": metadata.get("raw_keep_idx_hash"),
            }
        )
    keep_payload = json.dumps(
        result.get("keep_idx", output.selected_indices),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    result["keep_idx_hash"] = hashlib.sha256(keep_payload).hexdigest()
    if not cfg.get("save_per_sample", True):
        # The canonical results.jsonl remains per-question so the inherited
        # evaluator can score it; this flag controls extra diagnostic payloads.
        result["keep_idx"] = None
        result["att_stats"] = {}
    return result


def worker(device: str, rows: list[dict[str, Any]], cfg: dict[str, Any], out: Any) -> None:
    """Evaluate one shard and stream rows to the evaluator process."""

    try:
        import torch

        cfg = dict(cfg)
        cfg["device"] = device
        dtype = _worker_dtype(cfg["dtype"], device)
        runner = Qwen3VLCompressionRunner(
            name=cfg["model"],
            device=device,
            dtype=dtype,
            max_new_tokens=cfg["max_new_tokens"],
            trust_remote_code=cfg["trust_remote_code"],
            local_files_only=cfg["offline"],
            visual_pruning=True,
            visual_keep_ratio=cfg["visual_keep_ratio"],
            visual_keep_tokens=cfg.get("visual_keep_tokens"),
            vision_layers=tuple(int(value.strip()) for value in str(cfg["vision_layers"]).split(",") if value.strip()),
            pruning_mode="middle_text_early",
            pruning_layer=int(cfg["decision_layer_human"]),
            pruning_strategy="topk",
            max_image_pixels=cfg["max_image_pixels"],
            enable_thinking=cfg["enable_thinking"],
            compression_mode=cfg["compression_mode"],
            prune_method=cfg.get("prune_method", "text_cos"),
            decision_layer_human=int(cfg["decision_layer_human"]),
            att_stats=cfg["att_stats"],
            save_att_stats=cfg.get("save_att_stats", False),
            max_debug_samples=cfg.get("max_debug_samples", 0),
            q_flowcover_query_chunk=cfg.get("q_flowcover_query_chunk", 256),
            q_flowcover_candidate_chunk=cfg.get("q_flowcover_candidate_chunk", 64),
            q_flowcover_radius=cfg.get("q_flowcover_radius", 1),
        )
        for row in rows:
            case = Case(**row)
            image = None
            output = None
            try:
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                image = load_image(
                    case,
                    cfg.get("image_root"),
                    max_patches=cfg["max_image_patches"],
                    include_global=cfg["include_global_view"],
                    selection=cfg["patch_selection"],
                    tile_single=cfg["tile_single_images"],
                    patch_size=cfg["patch_size"],
                    patch_stride=cfg["patch_stride"],
                    global_size=cfg["global_view_size"],
                    max_source_pixels=cfg["max_image_pixels"],
                    decode_hf_images=cfg.get("decode_hf_images", False),
                )
                output = runner.answer(image, case.eval_prompt(), scoring_text=case.question)
                answer, correct, details = score_prediction(
                    case.gt, output.answer, case.options, case.question, case.category
                )
                extra = _extra_record(output, case, cfg, details)
                item = case.record(
                    answer=answer,
                    seconds=output.seconds,
                    flops=output.flops,
                    raw_model_answer=output.answer,
                    result=answer,
                    path=_path_value(case.image, cfg.get("image_root")),
                    patch_count=len(image),
                    status=details["status"],
                    correct=correct,
                    input_tokens=output.input_tokens,
                    output_tokens=output.output_tokens,
                    image_tokens=output.image_tokens,
                    original_image_tokens=output.original_image_tokens,
                    pruned_image_tokens=output.pruned_image_tokens,
                    retained_tokens=output.retained_tokens,
                    retention_ratio=output.retention_ratio,
                    pruning_latency=output.pruning_latency,
                    prefill_latency=output.prefill_latency,
                    pruning_seconds=output.pruning_seconds,
                    generation_seconds=output.generation_seconds,
                    decode_seconds=output.decode_seconds,
                    timing_scope="visual_encoding_to_answer_decode",
                    selection_budgets=output.selection_budgets,
                    selected_indices=output.selected_indices,
                    selection_groups=output.selection_groups,
                    selection_mode=output.selection_mode,
                    selection_strategy=output.selection_strategy,
                    selection_layer=output.selection_layer,
                    vision_layer_token_counts=output.vision_layer_token_counts,
                    vision_merge_unit=output.vision_merge_unit,
                    device=device,
                    stages=list(runner.stages()),
                    **extra,
                    **{key: value for key, value in details.items() if key != "status"},
                )
            except Exception as exc:
                item = case.record(
                    answer="",
                    seconds=0.0,
                    flops=0,
                    result="",
                    path=_path_value(case.image, cfg.get("image_root")),
                    status="error",
                    correct=False,
                    error=f"{type(exc).__name__}: {exc}",
                    sample_id=case.id,
                    task_category=case.category,
                    compression_mode=cfg.get("compression_mode"),
                    prune_method=cfg.get("prune_method", "text_cos"),
                    decision_layer_human=cfg.get("decision_layer_human", 18),
                    parser_status="error",
                    exception_count=1,
                    device=device,
                    stages=list(runner.stages()),
                )
            out.put(("row", item))
            image = None
            output = None
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    except Exception as exc:
        out.put(("worker_error", f"{type(exc).__name__}: {exc}"))
    finally:
        gc.collect()
        out.put(("done", device))


__all__ = ["worker"]
