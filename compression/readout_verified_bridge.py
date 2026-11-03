"""Numerical comparison helper for native attention-bridge diagnostics."""
from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - metadata-only hosts
    torch = None  # type: ignore[assignment]


def _is_tensor(value: Any) -> bool:
    return torch is not None and torch.is_tensor(value)


def compare_rows(reference: Any, candidate: Any, *, eps: float = 1e-12) -> dict[str, float]:
    """Return max/mean absolute and relative errors for bridge diagnostics."""

    if _is_tensor(reference) and _is_tensor(candidate):
        diff = (candidate.float() - reference.float()).abs()
        denom = reference.float().abs().clamp_min(float(eps))
        rel = diff / denom
        return {
            "max_abs_error": float(diff.max().detach().item()),
            "mean_abs_error": float(diff.mean().detach().item()),
            "max_relative_error": float(rel.max().detach().item()),
            "mean_relative_error": float(rel.mean().detach().item()),
        }
    ref_rows = list(reference)
    cand_rows = list(candidate)
    if len(ref_rows) != len(cand_rows):
        raise ValueError("reference and candidate row counts differ")
    values: list[tuple[float, float]] = []
    for ref, cand in zip(ref_rows, cand_rows):
        for left, right in zip(ref, cand):
            absolute = abs(float(right) - float(left))
            values.append((absolute, absolute / max(abs(float(left)), eps)))
    if not values:
        return {"max_abs_error": 0.0, "mean_abs_error": 0.0, "max_relative_error": 0.0, "mean_relative_error": 0.0}
    return {
        "max_abs_error": max(value[0] for value in values),
        "mean_abs_error": sum(value[0] for value in values) / len(values),
        "max_relative_error": max(value[1] for value in values),
        "mean_relative_error": sum(value[1] for value in values) / len(values),
    }



__all__ = ["compare_rows"]
