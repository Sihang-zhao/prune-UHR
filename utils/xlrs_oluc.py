"""XLRS OLUC option parsing and set-based evaluation.

The public data note says OLUC is a multiple-answer task but does not provide
a complete official partial-credit formula. Exact set accuracy is therefore
the primary metric. Precision/recall/F1 and Jaccard are retained as explicit
diagnostics and must not be reported as the official score.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_INLINE_MARKER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\(([A-Ha-h])\)|\b([A-Ha-h])\s*[.):：\-])\s*"
)
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:\bfinal\s+answers?|\banswers?|\bselected\s+options?)\s*"
    r"(?:(?:is|are)\s*)?[:=\-]?\s*"
    r"([A-H](?:\s*(?:,|/|;|\band\b|&|\s)\s*[A-H])*)",
    re.IGNORECASE,
)
_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])", re.IGNORECASE)
_OPTION_TAIL_RE = re.compile(
    r"\s*(?:Select\s+the\s+(?:best\s+answer\(s\)?|all\s+correct\s+answers?|single\s+best\s+option)|"
    r"Only\s+respond\s+with|The\s+answers?\s+are|The\s+answer\s+is).*$",
    re.IGNORECASE | re.DOTALL,
)
_INSTRUCTION_FRAGMENT_RE = re.compile(
    r"^\s*(?:with\s+multiple\s+choices\s+separated\s+by\s+spaces|"
    r"select\s+|only\s+respond|corresponding\s+to\s+the\s+correct|"
    r"the\s+answers?\s+(?:are|is)|multiple\s+choices\s+separated\s+by\s+spaces)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Option:
    label: str
    text: str


@dataclass(frozen=True)
class OLUCEvaluation:
    status: str
    correct: bool
    partial: bool
    gt: list[str]
    pred: list[str]
    precision: float
    recall: float
    f1: float
    jaccard: float


def _clean_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    text = re.sub(r"^[\s*#>-]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean_text(value).casefold()).strip()


def _strip_option_marker(value: Any) -> str:
    return re.sub(
        r"^\s*(?:\(\s*[A-Ha-h]\s*\)|[A-Ha-h]\s*[.):：\-])\s*",
        "",
        _clean_text(value),
    ).strip()


def _clean_option_text(value: Any) -> str:
    text = _strip_option_marker(value)
    text = _OPTION_TAIL_RE.sub("", text)
    return text.strip()


def _is_instruction_fragment(value: Any) -> bool:
    text = _clean_option_text(value)
    if not text or text in {".", ":", "-"}:
        return True
    return bool(_INSTRUCTION_FRAGMENT_RE.search(text))


def normalise_options(options: Any) -> list[Option]:
    """Return ordered options from list, dict, JSON, or marker-form text."""
    if options is None or options == "":
        return []
    if isinstance(options, str):
        raw = options.strip()
        try:
            decoded = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return parse_inline_options(raw).options
        if decoded == options:
            return parse_inline_options(raw).options
        return normalise_options(decoded)
    if isinstance(options, Mapping):
        result: list[Option] = []
        for index, (key, value) in enumerate(options.items()):
            label = str(key).strip().upper()
            if len(label) != 1 or not label.isalpha():
                label = chr(65 + index)
            if isinstance(value, Mapping):
                value = value.get("text", value.get("value", value.get("label", value)))
            if _is_instruction_fragment(value):
                continue
            result.append(Option(label, _clean_option_text(value)))
        return result
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes, bytearray)):
        result = []
        for index, value in enumerate(options):
            label = chr(65 + index)
            text = value
            if isinstance(value, Mapping):
                label = str(value.get("label", value.get("key", label))).strip().upper()
                text = value.get("text", value.get("value", value.get("content", value)))
            if _is_instruction_fragment(text):
                continue
            result.append(Option(label, _clean_option_text(text)))
        return result
    return []


@dataclass(frozen=True)
class InlineOptions:
    question: str
    options: list[Option]


def parse_inline_options(text: Any) -> InlineOptions:
    """Split ``Question ... (A) ... (B) ...`` into question and options."""
    raw = str(text if text is not None else "")
    matches = list(_INLINE_MARKER_RE.finditer(raw))
    if len(matches) < 2:
        return InlineOptions(raw.strip(), [])
    options: list[Option] = []
    for index, match in enumerate(matches):
        label = (match.group(1) or match.group(2)).upper()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        option_text = raw[match.end():end].strip(" \t\r\n,;")
        options.append(Option(label, _clean_option_text(option_text)))
    prefix = raw[: matches[0].start()].rstrip()
    return InlineOptions(prefix, options)


def _allowed_labels(options: Sequence[Option]) -> set[str]:
    labels = {option.label for option in options}
    return labels or set("ABCDEFGH")


def _label_from_choice_text(value: Any, options: Sequence[Option]) -> list[str]:
    target = _normalise_text(value)
    if not target:
        return []
    matches = [option.label for option in options if _normalise_text(option.text) == target]
    return matches if len(matches) == 1 else []


def _labels_from_text_parts(value: Any, options: Sequence[Option]) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    parts = [
        part.strip()
        for part in re.split(r"\s*(?:,|;|/|\band\b|&)\s*", text, flags=re.IGNORECASE)
        if part.strip()
    ]
    if len(parts) < 2:
        return []
    labels = set()
    for part in parts:
        matches = _label_from_choice_text(part, options)
        if len(matches) != 1:
            return []
        labels.update(matches)
    return _ordered_labels(labels, options)


def _ordered_labels(labels: set[str], options: Sequence[Option]) -> list[str]:
    order = {option.label: index for index, option in enumerate(options)}
    return sorted(labels, key=lambda label: (order.get(label, 10_000), label))


def _extract_labels(value: Any, options: Sequence[Option], multiple: bool) -> list[str]:
    allowed = _allowed_labels(options)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        found: set[str] = set()
        for item in value:
            if isinstance(item, Option):
                item = item.label
            labels = _extract_labels(item, options, multiple=True)
            text_matches = _label_from_choice_text(item, options)
            found.update(labels)
            found.update(text_matches)
        if found:
            return _ordered_labels(found, options)
        return []

    text = _clean_text(value).upper()
    if not text:
        return []
    explicit = _EXPLICIT_ANSWER_RE.search(text)
    source = explicit.group(1) if explicit else text
    labels = {match.group(1).upper() for match in _LETTER_RE.finditer(source)}
    labels &= allowed
    if labels:
        return _ordered_labels(labels, options)

    # OLUC references and model answers are often serialized compactly as
    # ``ABD``.  The boundary-aware letter regex intentionally ignores adjacent
    # letters, so handle this exact compact form separately.
    compact_multi = re.fullmatch(r"\s*\(?([A-H]{2,})\)?\s*[.):：\-]?\s*", source)
    if compact_multi:
        compact_labels = set(compact_multi.group(1)) & allowed
        if len(compact_labels) == len(compact_multi.group(1)):
            return _ordered_labels(compact_labels, options)

    if not multiple:
        compact = re.fullmatch(r"\s*\(?\s*([A-H])\s*\)?[.):：\-]?\s*", text)
        if compact and compact.group(1) in allowed:
            return [compact.group(1)]
    return _label_from_choice_text(value, options) or _labels_from_text_parts(value, options)


def extract_reference_set(answer: Any, options: Any, question: Any = None) -> list[str]:
    """Convert an OLUC reference answer into a canonical label set."""
    parsed_options = normalise_options(options)
    if not parsed_options and question is not None:
        parsed_options = parse_inline_options(question).options
    return _extract_labels(answer, parsed_options, multiple=True)


def _candidate_prediction_texts(raw_prediction: Any) -> list[str]:
    text = str(raw_prediction if raw_prediction is not None else "").strip()
    if not text:
        return []
    candidates = [text]
    explicit = list(_EXPLICIT_ANSWER_RE.finditer(text))
    if explicit:
        candidates.insert(0, explicit[-1].group(1))
    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty:
        candidates.insert(0, nonempty[-1])
    return candidates


def extract_prediction_set(raw_prediction: Any, options: Any, question: Any = None) -> list[str]:
    """Parse the model's OLUC response without selecting all letters in prose."""
    parsed_options = normalise_options(options)
    if not parsed_options and question is not None:
        parsed_options = parse_inline_options(question).options
    for candidate in _candidate_prediction_texts(raw_prediction):
        labels = _extract_labels(candidate, parsed_options, multiple=True)
        if labels:
            return labels
    return []


def evaluate_oluc(
    reference: Any,
    prediction: Any,
    options: Any,
    question: Any = None,
) -> OLUCEvaluation:
    """Evaluate one OLUC sample using exact-set accuracy plus diagnostics."""
    gt = extract_reference_set(reference, options, question)
    pred = extract_prediction_set(prediction, options, question)
    gt_set = set(gt)
    pred_set = set(pred)
    exact = bool(gt_set) and gt_set == pred_set
    intersection = len(gt_set & pred_set)
    precision = intersection / len(pred_set) if pred_set else 0.0
    recall = intersection / len(gt_set) if gt_set else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(gt_set | pred_set)
    jaccard = intersection / union if union else 0.0
    if exact:
        status = "correct"
    elif intersection:
        status = "partial"
    else:
        status = "incorrect"
    return OLUCEvaluation(
        status=status,
        correct=exact,
        partial=status == "partial",
        gt=gt,
        pred=pred,
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
    )


__all__ = [
    "InlineOptions",
    "OLUCEvaluation",
    "Option",
    "evaluate_oluc",
    "extract_prediction_set",
    "extract_reference_set",
    "normalise_options",
    "parse_inline_options",
]
