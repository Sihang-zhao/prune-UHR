"""Canonical XLRS L3 task normalization."""
from __future__ import annotations

import re
from typing import Any, Mapping

PERCEPTION_TASKS = ("OC", "RC", "OLUC", "RLUC", "OCC", "OCL", "OMS", "OSR")
REASONING_TASKS = ("AD", "ECR", "RP", "RCCD", "CCR")
ALL_TASKS = PERCEPTION_TASKS + REASONING_TASKS

_TASK_ALIASES = {
    "OC": ("oc", "overall counting", "overall count", "object counting"),
    "RC": ("rc", "regional counting", "regional count", "region counting"),
    "OLUC": ("oluc", "overall land use classification", "overall land-use classification", "object location"),
    "RLUC": ("rluc", "regional land use classification", "regional land-use classification", "relative location"),
    "OCC": ("occ", "object classification", "object class"),
    "OCL": ("ocl", "object color", "object colour", "object color classification"),
    "OMS": ("oms", "object material", "object motion state", "motion state"),
    "OSR": ("osr", "object size", "object spatial relationship", "spatial relationship"),
    "AD": ("ad", "attribute description", "anomaly detection", "anomaly detection and interpretation"),
    "ECR": ("ecr", "existence comparison", "environmental condition reasoning"),
    "RP": ("rp", "relation prediction", "route planning", "routing planning"),
    "RCCD": ("rccd", "relation comparison", "regional counting with change detection", "counting with change detection", "counting with changing detection"),
    "CCR": ("ccr", "complex counting", "complex counting reasoning", "counting with complex reasoning"),
}

_TASK_KEYS = {
    "l3_task",
    "l3_category",
    "l3-category",
    "l3",
    "task",
    "task_type",
    "question_type",
    "subtask",
    "sub_task",
    "benchmark_task",
    "benchmark_category",
    "task_name",
    "question_category",
    "category",
    "l2_category",
    "l2-category",
}

_CONTAINER_KEYS = {"metadata", "meta", "task_info", "category_info"}


def _norm(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalise_task(value: Any) -> str:
    """Map an XLRS task name/path/code to one of the 13 official codes."""
    text = str(value if value is not None else "").strip().upper()
    if text in ALL_TASKS:
        return text
    norm = _norm(value)
    compact = re.sub(r"[^a-z0-9]", "", norm)
    for code, aliases in _TASK_ALIASES.items():
        for alias in aliases:
            alias_norm = _norm(alias)
            alias_compact = re.sub(r"[^a-z0-9]", "", alias_norm)
            if norm == alias_norm or norm.endswith("/" + alias_norm) or compact == alias_compact or compact.endswith(alias_compact):
                return code
    return text or "UNKNOWN"


def _task_from_mapping(value: Mapping[str, Any]) -> tuple[str, str] | None:
    found = [(normalise_task(value.get(key)), str(value.get(key))) for key in _TASK_KEYS if value.get(key) is not None]
    for code, raw in found:
        if code in ALL_TASKS:
            return code, raw
    for container_key in _CONTAINER_KEYS:
        container = value.get(container_key)
        if isinstance(container, Mapping):
            nested = _task_from_mapping(container)
            if nested is not None:
                return nested
    return None


def task_from_sample(sample: Any) -> str:
    """Prefer a resolvable L3 field over a broad top-level category."""
    raw = getattr(sample, "source", {})
    code = normalise_task(getattr(sample, "task_type", None))
    if isinstance(raw, Mapping):
        resolved = _task_from_mapping(raw)
        if resolved is not None:
            code = resolved[0]
    return code


__all__ = [
    "ALL_TASKS",
    "PERCEPTION_TASKS",
    "REASONING_TASKS",
    "normalise_task",
    "task_from_sample",
]
