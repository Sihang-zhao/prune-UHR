"""Format-tolerant VQA loading with lazy HuggingFace image access."""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .images import prepare_views
from .tasks import normalise_task
from .xlrs_oluc import normalise_options, parse_inline_options

_HF_CACHE: dict[tuple[str, str, bool], Any] = {}
_OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-Ha-h])\s*[.)。：:-]\s*(.+?)\s*$")


def _is_multiple_choice(case: "Case") -> bool:
    """Return the answer protocol for a normalized case.

    RL/OPD datasets use several field names for this bit of metadata.  Keep
    the decision in one place so data loading, prompting and scoring cannot
    silently disagree about whether a response is a set of labels.
    """

    raw = case.raw if isinstance(case.raw, Mapping) else {}
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
    for value in (
        raw.get("is_multi_choice"),
        raw.get("is_multiple_choice"),
        raw.get("multiple_choice"),
        raw.get("multi_choice"),
        raw.get("question_type"),
        raw.get("answer_type"),
        metadata.get("is_multi_choice"),
        metadata.get("is_multiple_choice"),
        metadata.get("multiple_choice"),
        metadata.get("question_type"),
    ):
        if isinstance(value, bool) and value:
            return True
        if str(value or "").strip().casefold() in {"multi", "multiple", "multiple_choice", "multi_choice", "multiple-choice"}:
            return True
    if isinstance(case.gt, (list, tuple, set)) or str(case.category or "").upper() == "OLUC":
        return True
    # Some JSONL exports serialize multiple labels as ``"A B"`` or
    # ``"A,C"`` instead of a list.  Do not treat ordinary free-form text as
    # multiple choice.
    return bool(re.fullmatch(r"\s*\(?[A-Ha-h]\)?(?:\s*(?:[,/;&]|and|\s)\s*\(?[A-Ha-h]\)?)+\s*", str(case.gt or ""), re.IGNORECASE))


@dataclass
class Case:
    id: Any
    gt: Any
    question: str
    options: list[dict[str, str]] = field(default_factory=list)
    image: Any = None
    category: str = "UNKNOWN"
    l2_category: str = ""
    image_id: Any = None
    raw: dict[str, Any] = field(default_factory=dict)
    index: int = 0

    def prompt(self) -> str:
        """Compatibility alias for the canonical benchmark prompt."""

        return self.eval_prompt()

    def eval_prompt(self) -> str:
        # Match the RS-OPSD benchmark protocol: the explicit answer anchor
        # makes greedy generation start with the option label instead of an
        # explanation, while keeping the question/choices available to the
        # visual-token scorer.
        question = self.question.strip()
        if not self.options:
            return question
        choices = "\n".join(f"{item['label']}. {item['text']}" for item in self.options)
        multiple = _is_multiple_choice(self)
        if multiple:
            instruction = (
                "This is a multiple-choice question with more than one correct answer. "
                "Respond with all correct letters only.\nThe answer is:"
            )
        else:
            instruction = (
                "Select the answer for the multiple-choice question based on the image. "
                "Respond with only the letter corresponding to the correct answer.\nThe answer is:"
            )
        return f"{question}\nThe choices are listed below:\n{choices}\n{instruction}"

    def record(self, answer: str, seconds: float, flops: int, **extra: Any) -> dict[str, Any]:
        result = {
            "id": self.id,
            "index": self.index,
            "image_id": self.image_id,
            "gt": self.gt,
            "model_answer": answer,
            "question": self.question,
            "options": self.options,
            "category": self.category,
            "taskL3": self.category,
            "l2_category": self.l2_category,
            "testtime": seconds,
            "flops": flops,
        }
        result.update(extra)
        return result


def _read(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(dict(value))
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            for key in ("data", "items", "records", "questions", "annotations", "train", "test", "validation"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError(f"Expected a list of records in {path}")
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if suffix in {".parquet", ".feather"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install pandas and pyarrow to read parquet or feather data") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported data file: {path}")


def _pick(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    normalised = {re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_"): value for key, value in row.items()}
    for key in keys:
        value = normalised.get(re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_"))
        if value is not None:
            return value
    return default


def _conversation_value(value: Any, roles: set[str], reverse: bool = False) -> Any:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    for item in (reversed(value) if reverse else value):
        if isinstance(item, Mapping) and str(item.get("from", item.get("role", ""))).lower() in roles:
            content = item.get("value", item.get("content"))
            if isinstance(content, str):
                return content
            if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
                text_parts = []
                for part in content:
                    if isinstance(part, Mapping):
                        text = part.get("text", part.get("value", ""))
                        if text:
                            text_parts.append(str(text))
                    elif isinstance(part, str):
                        text_parts.append(part)
                if text_parts:
                    return "".join(text_parts)
            if content is not None:
                return str(content)
    return None


def _conversation_images(value: Any) -> list[Any]:
    """Collect images embedded in RS-OPSD/verl ``messages`` or ``prompt``."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    images: list[Any] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content", item.get("value"))
        parts = content if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)) else [content]
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            candidate = None
            for key in ("image", "image_url", "image_path", "image_file", "path", "file", "bytes"):
                if part.get(key) is not None:
                    candidate = part[key]
                    if isinstance(candidate, Mapping):
                        candidate = candidate.get("url", candidate.get("path", candidate.get("bytes", candidate)))
                    break
            if candidate is not None:
                images.append(candidate)
    return images


def _clean_prompt_text(value: Any) -> str | None:
    """Extract text from RS-OPSD-style prompt/message fields."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        text = _conversation_value(value, {"user", "human", "question"}, reverse=True)
        if text is None:
            text = _conversation_value(value, {"assistant", "answer"}, reverse=False)
        if text is None:
            return None
    elif isinstance(value, Mapping):
        content = value.get("text", value.get("value", value.get("content")))
        return _clean_prompt_text(content)
    else:
        text = str(value)
    # Images are supplied separately to the processor. Leaving literal
    # ``<image>`` markers in the question would create duplicate placeholders.
    text = re.sub(r"<\s*(?:image|video)\s*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _options(value: Any) -> list[dict[str, str]]:
    parsed = normalise_options(value)
    return [{"label": option.label, "text": option.text} for option in parsed]


def _options_from_prompt(value: Any) -> tuple[str, list[dict[str, str]]]:
    """Extract ``A. ...`` option lines from RS-OPSD prompt text."""

    text = _clean_prompt_text(value) or ""
    matches = list(_OPTION_LINE_RE.finditer(text))
    if len(matches) < 2:
        return text, []
    options = [{"label": match.group(1).upper(), "text": match.group(2).strip()} for match in matches]
    # Keep only the question prefix. The evaluator appends its own canonical
    # answer protocol through Case.eval_prompt().
    return text[: matches[0].start()].strip(), options


def _normalise(row: Mapping[str, Any], index: int) -> Case:
    conversations = row.get("conversations", row.get("messages"))
    image = _pick(row, "image", "images", "image_path", "image_file", "img", "filename", "file_name", "student_image", "teacher_image")
    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), Mapping) else {}
    reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), Mapping) else {}
    if image is None:
        image = _pick(extra_info, "student_image_path", "full_image_path", "image", "images")
    if image is None:
        embedded_images = _conversation_images(conversations)
        image = embedded_images if len(embedded_images) > 1 else (embedded_images[0] if embedded_images else None)
    question = _pick(row, "question", "query", "instruction", "text", "human", "user")
    prompt_value = _pick(row, "prompt")
    if image is None:
        embedded_images = _conversation_images(prompt_value)
        image = embedded_images if len(embedded_images) > 1 else (embedded_images[0] if embedded_images else None)
    if question is None:
        question = _pick(extra_info, "question", "query", "prompt")
    gt = _pick(row, "gt", "ground_truth", "gt_answer", "label", "target", "answer", "response", "assistant", "gpt")
    if gt is None:
        gt = _pick(extra_info, "answer", "ground_truth", "gt", "label")
    if gt is None:
        gt = _pick(reward_model, "ground_truth", "answer", "label")
    if gt is None:
        gt = _pick(extra_info, "response", "target", "solution")
    question = question if question is not None else _conversation_value(conversations, {"human", "user", "question"})
    question = question if question is not None else _clean_prompt_text(prompt_value)
    gt = gt if gt is not None else _conversation_value(conversations, {"assistant", "answer", "gpt"}, reverse=True)
    if isinstance(image, Mapping) and "__hf__" not in image:
        image = image.get("path", image.get("file", image.get("filename", image.get("bytes"))))
    if isinstance(image, Sequence) and not isinstance(image, (str, bytes, bytearray)):
        image = list(image) or None
    category_value = _pick(row, "l3_task", "l3_category", "l3", "task", "task_type", "category", "type", default="unknown")
    if str(category_value).casefold() in {"", "unknown", "none"}:
        category_value = _pick(extra_info, "l3_task", "l3_category", "l3", "task", "task_type", "category", "type", default=category_value)
    if str(category_value).casefold() in {"", "unknown", "none"}:
        category_value = _pick(reward_model, "l3_task", "l3_category", "l3", "task", "task_type", "category", "type", default=category_value)
    prompt_question, prompt_options = _options_from_prompt(prompt_value)
    option_value = _pick(row, "options", "choices", "multi_choice_options", "multi-choice options")
    if option_value is None:
        option_value = _pick(extra_info, "options", "choices", "multi_choice_options")
    if option_value is None:
        option_value = _pick(reward_model, "options", "choices", "multi_choice_options")
    parsed_options = _options(option_value)
    if not parsed_options:
        parsed_options = prompt_options
    if question is None or (prompt_value is not None and _clean_prompt_text(question) == _clean_prompt_text(prompt_value)):
        question = prompt_question or question
    case = Case(
        id=_pick(row, "id", "sample_id", "question_id", "index", default=index),
        image_id=_pick(row, "image_id", "img_id", default=None),
        gt=gt,
        question="" if question is None else (_clean_prompt_text(question) or ""),
        options=parsed_options,
        image=image,
        category=normalise_task(category_value),
        l2_category=str(_pick(row, "l2_category", "l2-category", "subcategory", default=_pick(extra_info, "l2_category", "l2-category", "subcategory", default="")) or "").strip(),
        raw=dict(row),
        index=index,
    )
    if not case.options:
        inline = parse_inline_options(case.question)
        case.question = inline.question
        case.options = [{"label": option.label, "text": option.text} for option in inline.options]
    if not case.question.strip() or case.gt is None or case.image is None:
        raise ValueError(f"case {index} must contain image, question and ground truth")
    return case


def _load_hf_dataset(path: Path, split: str, *, decode_images: bool = True) -> Any:
    try:
        from datasets import Dataset, DatasetDict, Image as HFImage, concatenate_datasets, load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("Install datasets to read HuggingFace Arrow data") from exc
    try:
        dataset = load_from_disk(str(path))
    except (FileNotFoundError, ValueError, OSError):
        shards = sorted((path / split).glob("*.arrow"))
        if shards:
            dataset = concatenate_datasets([Dataset.from_file(str(shard)) for shard in shards])
        else:
            dataset = load_dataset(str(path), split=split)
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"split={split!r} is unavailable; available splits: {', '.join(dataset.keys())}")
        dataset = dataset[split]
    if not decode_images and "image" in getattr(dataset, "column_names", ()):
        try:
            dataset = dataset.cast_column("image", HFImage(decode=False))
        except (AttributeError, TypeError, ValueError):
            # Nested List(Image) columns cannot be cast by some datasets
            # versions.  _resolve_hf_images handles those through the raw
            # Arrow column, so this fallback remains decode-safe.
            pass
    return dataset


def _load_hf_cases(path: Path, split: str, limit: int | None) -> list[Case]:
    # Metadata discovery must not decode every image in the dataset.  The
    # previous default (decode_images=True) caused the parent process to
    # materialize large PIL objects before workers even started.  Keep image
    # features as references and decode one sample lazily in load_image().
    dataset = _load_hf_dataset(path, split, decode_images=False)
    if "image" not in dataset.column_names:
        raise ValueError("HuggingFace dataset must contain an embedded 'image' column")
    count = min(len(dataset), int(limit)) if limit is not None else len(dataset)
    metadata = dataset.select(range(count)).remove_columns("image")
    return [
        _normalise({**dict(metadata[index]), "image": {"__hf__": str(path.resolve()), "split": split, "index": index}}, index)
        for index in range(count)
    ]


def load_cases(path: str, limit: int | None = None, split: str = "train") -> list[Case]:
    """Read JSON/JSONL/Parquet or an offline HuggingFace Arrow directory."""
    file = Path(path).expanduser()
    if file.is_dir():
        names = [f"{split.strip().lower()}.jsonl", f"{split.strip().lower()}.json", "questions.jsonl", "questions.json", "annotations.json", "data.jsonl", "data.json"]
        candidates = [file / name for name in names if (file / name).is_file()]
        if not candidates:
            return _load_hf_cases(file, split, limit)
        file = candidates[0]
    if not file.is_file():
        raise FileNotFoundError(f"data file does not exist: {file}")
    cases = [_normalise(row, index) for index, row in enumerate(_read(file))]
    return cases[: int(limit)] if limit is not None else cases


def _resolve_hf_images(reference: Mapping[str, Any], *, decode_images: bool = True) -> Any:
    key = (str(reference["__hf__"]), str(reference.get("split", "train")), bool(decode_images))
    dataset = _HF_CACHE.get(key)
    if dataset is None:
        dataset = _load_hf_dataset(Path(key[0]), key[1], decode_images=decode_images)
        _HF_CACHE[key] = dataset
    index = int(reference["index"])
    if not decode_images:
        # XLRS stores ``image`` as List(Image).  On several datasets releases,
        # indexing that nested feature decodes every JPEG into a PIL object even
        # after cast_column(..., decode=False).  Read the Arrow scalar instead;
        # ``as_py`` returns the original {bytes, path} records and keeps decode
        # lazy until prepare_views applies its source-pixel cap.
        table = getattr(dataset, "_data", None)
        try:
            column = table.column("image") if table is not None else None
            if column is not None:
                return column[index].as_py()
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    return dataset[index]["image"]


def load_image(
    case: Case,
    root: str | None = None,
    *,
    max_patches: int = 64,
    include_global: bool = True,
    selection: str = "uniform",
    tile_single: bool = True,
    patch_size: int = 448,
    patch_stride: int = 336,
    global_size: int = 448,
    max_source_pixels: int | None = None,
    decode_hf_images: bool = True,
) -> list[Any]:
    """Resolve a case and construct the ordered image views sent to Qwen."""
    import PIL.Image

    # HF image features may already be decoded as PIL objects before our image
    # normalizer sees them. Suppress Pillow's advisory warning here and apply
    # the evaluator-specific area cap during the subsequent resize.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PIL.Image.DecompressionBombWarning)
        value = (
            _resolve_hf_images(case.image, decode_images=decode_hf_images)
            if isinstance(case.image, Mapping) and "__hf__" in case.image
            else case.image
        )
    return prepare_views(
        value,
        root=root,
        max_patches=max_patches,
        include_global=include_global,
        selection=selection,
        tile_single=tile_single,
        patch_size=patch_size,
        patch_stride=patch_stride,
        global_size=global_size,
        **({"max_source_pixels": int(max_source_pixels)} if max_source_pixels is not None else {}),
    )


def case_dict(case: Case) -> dict[str, Any]:
    payload = asdict(case)
    payload["raw"] = {}
    return payload


__all__ = ["Case", "case_dict", "load_cases", "load_image"]
