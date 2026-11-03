"""Answer normalization, accuracy, and compact evaluation summaries."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .xlrs_oluc import evaluate_oluc, extract_prediction_set, normalise_options
from .tasks import ALL_TASKS, PERCEPTION_TASKS, REASONING_TASKS


_SINGLE_CHOICE_PATTERN = re.compile(r"(?<![A-Z0-9])\(?([A-H])\)?(?![A-Z0-9])", re.IGNORECASE)


def _extract_single_choice(prediction: Any, options: Any) -> str:
    """Follow RS-OPSD's single-choice rule: use the first standalone label."""

    text = str(prediction if prediction is not None else "").strip()
    allowed = {option.label for option in normalise_options(options)}
    normalized = re.sub(r"\s+", " ", text.casefold().strip(" \t\r\n.,;:!?\"'"))
    for option in normalise_options(options):
        if normalized == re.sub(r"\s+", " ", option.text.casefold().strip(" \t\r\n.,;:!?\"'")):
            return option.label
    if allowed:
        # Prefer a label that is explicitly at the beginning or follows the
        # benchmark's answer anchor.  Searching arbitrary prose first can
        # misread an option text such as ``A small lake`` as label A.
        candidates = [text]
        lowered = text.casefold()
        for marker in ("the answer is", "answer:", "final answer:"):
            position = lowered.rfind(marker)
            if position >= 0:
                candidates.insert(0, text[position + len(marker):])
        for candidate in candidates:
            match = _SINGLE_CHOICE_PATTERN.search(candidate.upper())
            if match and match.group(1).upper() in allowed:
                return match.group(1).upper()
    return ""


def _labels(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(item) for item in value)
    text = str(value if value is not None else "").upper()
    found = re.findall(r"(?<![A-Z0-9])([A-H])(?![A-Z0-9])", text)
    if found:
        return list(dict.fromkeys(found))
    fallback = re.sub(r"\s+", " ", text).strip()
    return [fallback] if fallback else []


def normalize_prediction(prediction: Any, options: Any = None, *, multiple: bool = False) -> str:
    """Convert a model response to canonical option letters."""
    if multiple:
        return " ".join(extract_prediction_set(prediction, options))
    single = _extract_single_choice(prediction, options)
    if single:
        return single
    if options:
        return ""
    return str(prediction if prediction is not None else "").strip()


def correct(gt: Any, pred: Any) -> bool:
    # Only interpret A-H as labels when the complete response is label-shaped.
    # Searching arbitrary prose for standalone letters mis-scored free-form
    # answers such as ``a cat`` as the option ``A``.
    def text(value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            value = " ".join(str(item) for item in value)
        return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()

    gt_text, pred_text = text(gt), text(pred)
    if not gt_text or not pred_text:
        return False
    choice_shape = re.compile(
        r"^\s*\(?[A-H]\)?(?:\s*(?:[,/;&]|\band\b|\s)\s*\(?[A-H]\)?)*\s*[.:]?\s*$",
        re.IGNORECASE,
    )
    if choice_shape.fullmatch(gt_text) and choice_shape.fullmatch(pred_text):
        gt_labels, pred_labels = _labels(gt_text), _labels(pred_text)
        return bool(gt_labels) and set(gt_labels) == set(pred_labels)
    normalise = lambda value: re.sub(r"\s+", " ", value.casefold().strip(" \t\r\n.,;:!?\"'"))
    return normalise(gt_text) == normalise(pred_text)


def score_prediction(
    gt: Any,
    prediction: Any,
    options: Any,
    question: Any,
    task_l3: Any,
) -> tuple[str, bool, dict[str, Any]]:
    """Return the canonical answer, exact-match result, and diagnostics."""
    is_oluc = str(task_l3 or "").upper() == "OLUC" or isinstance(gt, (list, tuple, set))
    if is_oluc:
        evaluation = evaluate_oluc(gt, prediction, options, question)
        return " ".join(evaluation.pred), evaluation.correct, {
            "status": evaluation.status,
            "oluc_gt": evaluation.gt,
            "oluc_pred": evaluation.pred,
            "oluc_precision": evaluation.precision,
            "oluc_recall": evaluation.recall,
            "oluc_f1": evaluation.f1,
            "oluc_jaccard": evaluation.jaccard,
        }
    parsed_options = normalise_options(options)
    canonical = normalize_prediction(prediction, options)
    canonical_gt = normalize_prediction(gt, options)
    is_correct = canonical.upper() == canonical_gt.upper() if parsed_options else correct(gt, canonical)
    return canonical, is_correct, {"status": "correct" if is_correct else "incorrect"}


def _group_stats(groups: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, items in sorted(groups.items()):
        hits = sum(
            bool(item.get("correct", correct(item.get("gt"), item.get("model_answer"))))
            and item.get("status") != "error"
            for item in items
        )
        valid = [item for item in items if item.get("status") != "error"]
        times = [float(item["testtime"]) for item in valid if item.get("testtime") is not None]
        flops = [int(item["flops"]) for item in valid if item.get("flops") is not None]
        throughputs = [float(item["flops"]) / float(item["testtime"]) / 1e12 for item in valid if item.get("flops") is not None and float(item.get("testtime") or 0.0) > 0]
        avg_flops = sum(flops) / len(flops) if flops else 0.0
        result[name] = {
            "correct": hits,
            "total": len(items),
            "acc": hits / len(items) if items else 0.0,
            "avg_testtime": sum(times) / len(times) if times else 0.0,
            "avg_flops": avg_flops,
            "avg_tflops": avg_flops / 1e12,
            "avg_tflops_per_sec": sum(throughputs) / len(throughputs) if throughputs else 0.0,
        }
    return result


def evaluate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_subcategory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("taskL3", row.get("category", "UNKNOWN"))).strip() or "UNKNOWN"].append(row)
        subcategory = str(row.get("l2_category", "")).strip()
        if subcategory:
            by_subcategory[subcategory].append(row)
    hits = sum(
        bool(row.get("correct", correct(row.get("gt"), row.get("model_answer"))))
        and row.get("status") != "error"
        for row in rows
    )
    valid_rows = [row for row in rows if row.get("status") != "error"]
    times = [float(row["testtime"]) for row in valid_rows if row.get("testtime") is not None]
    flops = [int(row["flops"]) for row in valid_rows if row.get("flops") is not None]
    throughputs = [float(row["flops"]) / float(row["testtime"]) / 1e12 for row in valid_rows if row.get("flops") is not None and float(row.get("testtime") or 0.0) > 0]
    avg_flops = sum(flops) / len(flops) if flops else 0.0
    task_stats = _group_stats(by_task)

    # XLRS reports the mean of the 13 L3 task accuracies (macro average).
    # ``hits / len(rows)`` below is still useful as an overall sample-level
    # accuracy, but it is a micro average and is biased toward large tasks
    # such as OCC/OCL/OSR.  Keep both metrics explicit instead of presenting
    # the micro value under the benchmark's ``Avg.`` column.
    def mean_task_accuracy(tasks: Iterable[str]) -> tuple[float, int]:
        values = [
            float(task_stats[name]["acc"])
            for name in tasks
            if name in task_stats and task_stats[name].get("total", 0)
        ]
        return (sum(values) / len(values) if values else 0.0), len(values)

    macro_acc, macro_task_count = mean_task_accuracy(ALL_TASKS)
    perception_acc, perception_task_count = mean_task_accuracy(PERCEPTION_TASKS)
    reasoning_acc, reasoning_task_count = mean_task_accuracy(REASONING_TASKS)
    return {
        "total": len(rows),
        "correct": hits,
        "errors": sum(row.get("status") == "error" or "error" in row for row in rows),
        # Backward-compatible overall sample-level (micro) accuracy.
        "acc": hits / len(rows) if rows else 0.0,
        "micro_acc": hits / len(rows) if rows else 0.0,
        # Official task-balanced summary values.
        "macro_acc": macro_acc,
        "macro_task_count": macro_task_count,
        "perception_acc": perception_acc,
        "perception_task_count": perception_task_count,
        "reasoning_acc": reasoning_acc,
        "reasoning_task_count": reasoning_task_count,
        "valid_total": len(valid_rows),
        "coverage": len(valid_rows) / len(rows) if rows else 0.0,
        "error_rate": (len(rows) - len(valid_rows)) / len(rows) if rows else 0.0,
        "avg_testtime": sum(times) / len(times) if times else 0.0,
        "avg_flops": avg_flops,
        "avg_tflops": avg_flops / 1e12,
        "avg_tflops_per_sec": sum(throughputs) / len(throughputs) if throughputs else 0.0,
        "total_flops": sum(flops),
        "total_tflops": sum(flops) / 1e12,
        "taskL3": task_stats,
        "categories": task_stats,
        "subcategories": _group_stats(by_subcategory),
    }


def save_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load the latest record for each case from a resumable run file."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"resume file does not exist: {source}")
    latest: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
        if not isinstance(row, Mapping) or row.get("index") is None:
            raise ValueError(f"resume row at {source}:{line_number} has no numeric index")
        try:
            index = int(row["index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"resume row at {source}:{line_number} has invalid index") from exc
        latest[index] = dict(row)
    return [latest[index] for index in sorted(latest)]


def save_markdown(path: str | Path, summary: Mapping[str, Any], *, config: Mapping[str, Any] | None = None) -> None:
    """Write the XLRS summary using the two-level benchmark table header."""
    config = config or {}
    stats = summary.get("taskL3", {})
    perception = list(PERCEPTION_TASKS)
    reasoning = list(REASONING_TASKS)

    def cell(task: str) -> str:
        value = stats.get(task)
        return "--" if not value or not value.get("total") else f"{float(value['acc']) * 100:.1f}"

    def header_cells(tasks: list[str]) -> str:
        return "".join(f"<th>{task}</th>" for task in tasks)

    def value_cells(tasks: list[str]) -> str:
        return "".join(f"<td>{cell(task)}</td>" for task in tasks)

    method = "Reported"
    if config.get("visual_pruning"):
        keep_percent = float(config.get("visual_keep_ratio", 1.0)) * 100.0
        removed_percent = 100.0 - keep_percent
        method = f"Retained top {keep_percent:.0f}% (removed {removed_percent:.0f}%)"
    note = (
        f"patches={config.get('max_image_patches', '--')}; "
        f"pruning={bool(config.get('visual_pruning', False))}; "
        f"errors={summary.get('errors', 0)}"
    )
    html = f"""<table>
<thead>
<tr><th>Method</th><th colspan={len(perception)}>Perception</th><th colspan={len(reasoning)}>Reasoning</th><th>Avg.</th><th>time</th><th>TFLOPs</th><th>备注</th></tr>
<tr><th>sub-task</th>{header_cells(perception)}{header_cells(reasoning)}<th></th><th></th><th></th><th></th></tr>
</thead>
<tbody>
<tr><td>{method}</td>{value_cells(perception)}{value_cells(reasoning)}<td>{float(summary.get('macro_acc', summary.get('acc', 0.0))) * 100:.1f}</td><td>{float(summary.get('avg_testtime', 0.0)):.2f}s</td><td>{float(summary.get('avg_tflops', 0.0)):.4f}</td><td>{note}; Avg.=macro</td></tr>
</tbody>
</table>

| 指标 | 数值 |
|---|---:|
| cases | {summary.get('total', 0)} |
| correct | {summary.get('correct', 0)} |
| micro accuracy (sample-weighted) | {float(summary.get('micro_acc', summary.get('acc', 0.0))) * 100:.2f}% |
| macro accuracy (13 L3 tasks) | {float(summary.get('macro_acc', summary.get('acc', 0.0))) * 100:.2f}% |
| perception macro accuracy | {float(summary.get('perception_acc', 0.0)) * 100:.2f}% |
| reasoning macro accuracy | {float(summary.get('reasoning_acc', 0.0)) * 100:.2f}% |
| errors | {summary.get('errors', 0)} |
| total TFLOPs | {float(summary.get('total_tflops', 0.0)):.4f} |
"""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("# XLRS-Bench Evaluation\n\n" + html, encoding="utf-8")


__all__ = ["correct", "evaluate", "load_jsonl", "normalize_prediction", "save_jsonl", "save_markdown", "score_prediction"]
