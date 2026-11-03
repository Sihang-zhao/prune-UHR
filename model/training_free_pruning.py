"""Qwen3-VL inference and text-cosine selection used by Q-FlowCover.

The checkpoint's frozen merger projects visual hidden states, and the
single-layer text-cosine selector provides the reference keep set. Shared
processor, DeepStack, and compact-prompt helpers preserve the original
evaluation behavior.
"""
from __future__ import annotations

import math
import time
import unicodedata
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:  # Keep the command-line help usable on a metadata-only machine.
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only without runtime deps
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from .qwen_vl import ModelOutput, QwenVL, _get, estimate_flops

_EARLY_MIDDLE_MODES = frozenset({"middle_text_early", "middle_text_early_grid"})
_MIDDLE_TEXT_MODES = frozenset({"middle_text", *_EARLY_MIDDLE_MODES})


def _is_early_middle_mode(mode: str) -> bool:
    return str(mode) in _EARLY_MIDDLE_MODES


def _inference_mode():
    if torch is None:  # pragma: no cover - metadata-only environment
        def decorator(function):
            return function

        return decorator
    return torch.inference_mode()


@dataclass
class SelectionResult:
    """Indices selected from one image's merged visual-token sequence."""

    indices: torch.Tensor
    final_indices: torch.Tensor
    middle_indices: torch.Tensor
    shallow_indices: torch.Tensor
    budget: tuple[int, int, int]


@dataclass
class TrainingFreeModelOutput(ModelOutput):
    """Baseline output plus metrics specific to the pruning experiment."""

    retained_tokens: int = 0
    retention_ratio: float = 0.0
    pruning_latency: float = 0.0
    prefill_latency: float = 0.0
    selection_budgets: list[tuple[int, int, int]] = field(default_factory=list)
    selected_indices: list[list[int]] = field(default_factory=list)
    selection_groups: list[dict[str, list[int]]] = field(default_factory=list)
    selection_mode: str = "multi_layer"
    selection_strategy: str = "topk"
    selection_layer: int | None = None
    vision_layer_token_counts: list[int] = field(default_factory=list)
    vision_merge_unit: int = 0


@dataclass
class _SelectedVisionOutput:
    """Minimal encoder-output contract consumed by Qwen3-VL's multimodal path."""

    # Qwen3-VL 4.57.x returns one tuple entry per image view.  Keeping this
    # exact structure is important: the model concatenates the tuple itself;
    # passing one tensor makes ``torch.cat`` iterate over tensor rows and can
    # surface the opaque ``string indices must be integers``/shape errors seen
    # through ``generate``.
    pooler_output: tuple[torch.Tensor, ...]
    deepstack_features: list[tuple[torch.Tensor, ...]]

    # Compatibility with the tuple return used by some 4.57.x point
    # releases.  Newer Qwen3-VL code reads attributes, while older code
    # unpacks ``(image_embeds, deepstack_features)``.
    def __iter__(self):
        yield self.pooler_output
        yield self.deepstack_features

    def __getitem__(self, index: int):
        return (self.pooler_output, self.deepstack_features)[index]

    def __len__(self) -> int:
        return 2


class _FeatureTuple(tuple):
    """Tuple of per-view features compatible with tensor-only older models.

    Current Qwen3-VL code concatenates the per-view tuple itself.  Some nearby
    Transformers revisions instead call ``.to()`` on the returned feature
    value.  This keeps tuple iteration for the former and supplies the tensor
    conversion expected by the latter, without changing feature values or
    patch order.
    """

    def _cat(self) -> torch.Tensor:
        if not self:
            raise RuntimeError("cannot materialize an empty visual feature tuple")
        return torch.cat(tuple(self), dim=0)

    def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._cat().to(*args, **kwargs)

    @property
    def shape(self) -> torch.Size:
        return self._cat().shape

    @property
    def device(self) -> torch.device:
        return self._cat().device

    @property
    def dtype(self) -> torch.dtype:
        return self._cat().dtype


@contextmanager
def _temporary_method(module: Any, name: str, replacement: Any):
    """Temporarily replace a model method and restore its exact state."""

    had_value = name in getattr(module, "__dict__", {})
    old_value = getattr(module, "__dict__", {}).get(name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        if had_value:
            setattr(module, name, old_value)
        else:
            try:
                delattr(module, name)
            except AttributeError:
                pass


def _require_torch() -> None:
    if torch is None or F is None:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for training-free visual pruning")


def _normalize_model_answer(text: str) -> str:
    """Strip Qwen/RS-OPSD thinking and answer wrappers before scoring."""

    answer = str(text or "").strip()
    think_end = answer.rfind("</think>")
    if think_end != -1:
        answer = answer[think_end + len("</think>") :].strip()
    start = answer.rfind("<answer>")
    end = answer.find("</answer>", start + len("<answer>")) if start != -1 else -1
    if start != -1 and end > start:
        return answer[start + len("<answer>") : end].strip()
    lowered = answer.casefold()
    for marker in ("the answer is:", "answer:"):
        marker_start = lowered.rfind(marker)
        if marker_start != -1:
            return answer[marker_start + len(marker) :].strip()
    return answer


def _move_processor_value(value: Any, device: Any) -> Any:
    """Move a processor result without assuming it is a BatchFeature.

    ``ProcessorMixin.apply_chat_template`` normally returns a ``BatchFeature``
    when ``return_dict=True``.  A few Qwen/Transformers combinations wrap that
    object in a tuple, or return a plain mapping without ``BatchFeature.to``.
    Moving recursively keeps all image fields (including ``image_grid_thw``)
    while preserving the container expected by the model.
    """

    if isinstance(value, Mapping):
        return {key: _move_processor_value(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_processor_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_processor_value(item, device) for item in value]
    move = getattr(value, "to", None)
    if callable(move):
        try:
            return move(device)
        except (AttributeError, TypeError):
            # Some custom containers expose ``to(device=...)`` only.
            try:
                return move(device=device)
            except (AttributeError, TypeError):
                pass
    return value


def _find_processor_inputs(value: Any, device: Any) -> Any | None:
    """Find the dict-like processor payload inside version-specific wrappers."""

    if isinstance(value, (str, bytes, bytearray)):
        return None
    if isinstance(value, Mapping):
        if "input_ids" in value:
            return _move_processor_value(value, device)
        for item in value.values():
            found = _find_processor_inputs(item, device)
            if found is not None:
                return found
        return None
    # BatchFeature-like custom objects may not register as Mapping but expose
    # the same input_ids attribute and item access.
    try:
        has_input_ids = getattr(value, "input_ids", None) is not None
    except Exception:
        has_input_ids = False
    if has_input_ids:
        return _move_processor_value(value, device)
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _find_processor_inputs(item, device)
            if found is not None:
                return found
    return None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return a mapping view for BatchFeature-like objects."""

    if isinstance(value, Mapping):
        return value
    try:
        keys = value.keys()
        return {key: value[key] for key in keys}
    except (AttributeError, KeyError, TypeError):
        return None


def _tensor_leaves(value: Any) -> list[torch.Tensor] | None:
    """Collect tensors from a tuple/list wrapper without guessing field names."""

    if torch is None:
        return None
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        leaves: list[torch.Tensor] = []
        for item in value:
            nested = _tensor_leaves(item)
            if nested is None:
                return None
            leaves.extend(nested)
        return leaves
    return None


def _normalize_tensor_field(name: str, value: Any, device: Any) -> Any:
    """Normalize version-specific processor fields to the model's tensor API.

    A few development Transformers/Qwen processors return a tuple of tensors
    for one named field (most often ``pixel_values`` or ``image_grid_thw``).
    The native Qwen forward path expects a Tensor, so passing that tuple onward
    eventually produces ``AttributeError: 'tuple' object has no attribute
    'to'``.  Only unambiguous concatenations are accepted here; ambiguous
    structures fail with a field-specific diagnostic rather than silently
    changing token order.
    """

    if torch is None:
        return value
    if torch.is_tensor(value):
        return value.to(device)
    leaves = _tensor_leaves(value)
    if not leaves:
        return value
    tensors = [item.to(device) for item in leaves]
    if len(tensors) == 1:
        return tensors[0]
    try:
        if name in {"image_grid_thw", "video_grid_thw"}:
            if any(item.numel() % 3 != 0 for item in tensors):
                raise ValueError("grid tensor does not contain triples")
            return torch.cat([item.reshape(-1, 3) for item in tensors], dim=0)
        if name == "pixel_values" or name == "pixel_values_videos":
            ranks = {int(item.ndim) for item in tensors}
            if len(ranks) != 1 or any(tuple(item.shape[1:]) != tuple(tensors[0].shape[1:]) for item in tensors):
                raise ValueError("pixel tensors have incompatible shapes")
            return torch.cat(tensors, dim=0)
        if name in {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}:
            if all(item.ndim == 1 for item in tensors):
                lengths = {int(item.shape[0]) for item in tensors}
                if len(lengths) != 1:
                    raise ValueError("sequence tensors have different lengths")
                return torch.stack(tensors, dim=0)
            if all(item.ndim == 2 for item in tensors):
                if any(tuple(item.shape[1:]) != tuple(tensors[0].shape[1:]) for item in tensors):
                    raise ValueError("batched sequence tensors have incompatible shapes")
                return torch.cat(tensors, dim=0)
        if name == "position_ids":
            if all(item.ndim == 3 for item in tensors):
                if any(item.shape[0] != tensors[0].shape[0] or item.shape[2] != tensors[0].shape[2] for item in tensors):
                    raise ValueError("position tensors have incompatible modal/sequence dimensions")
                return torch.cat(tensors, dim=1)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"processor field {name!r} contains multiple tensors that cannot be normalized; "
            f"shapes={[tuple(item.shape) for item in tensors]}"
        ) from exc
    raise RuntimeError(
        f"processor field {name!r} was returned as a nested tensor sequence; "
        f"unsupported shapes={[tuple(item.shape) for item in tensors]}"
    )


def _normalize_processor_mapping(mapping: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Move and normalize all tensor-valued multimodal processor fields."""

    tensor_fields = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "mm_token_type_ids",
        "position_ids",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    }
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in tensor_fields:
            result[key] = _normalize_tensor_field(key, value, device)
        else:
            result[key] = _move_processor_value(value, device)
    return result


def _infer_tensor_processor_inputs(value: Any, device: Any) -> Mapping[str, Any] | None:
    """Recover named multimodal fields from a legacy tuple of tensors.

    This is intentionally conservative and is used only when a processor
    returns no mapping at all.  ``image_grid_thw`` is unambiguous (last
    dimension 3), image pixels are floating point, and ``input_ids`` is the
    integer 2-D tensor whose values are not restricted to the mask range.
    """

    if torch is None or not isinstance(value, (tuple, list)):
        return None
    tensors = [item for item in value if torch.is_tensor(item)]
    if len(tensors) != len(value):
        return None
    grid_candidates = [item for item in tensors if item.ndim >= 1 and int(item.shape[-1]) == 3]
    pixel_candidates = [item for item in tensors if torch.is_floating_point(item) and item.ndim >= 2]
    integer_2d = [item for item in tensors if not torch.is_floating_point(item) and item.ndim == 2]
    if len(grid_candidates) != 1 or len(pixel_candidates) != 1 or not integer_2d:
        return None
    grid = grid_candidates[0]
    pixels = pixel_candidates[0]
    integer_2d = [item for item in integer_2d if item is not grid]
    if not integer_2d:
        return None
    # Token IDs usually contain values well above the binary attention mask;
    # choose the highest-range candidate if several integer fields exist.
    ids = max(integer_2d, key=lambda item: int(item.detach().max().item()) if item.numel() else -1)
    result: dict[str, Any] = {
        "input_ids": ids.to(device),
        "pixel_values": pixels.to(device),
        "image_grid_thw": grid.to(device),
    }
    for item in integer_2d:
        if item is ids or tuple(item.shape) != tuple(ids.shape):
            continue
        maximum = int(item.detach().max().item()) if item.numel() else 0
        minimum = int(item.detach().min().item()) if item.numel() else 0
        if minimum >= 0 and maximum <= 1:
            result.setdefault("attention_mask", item.to(device))
        elif minimum >= 0 and maximum <= 2:
            result.setdefault("mm_token_type_ids", item.to(device))
    return result


def _stable_argsort(values: torch.Tensor, *, descending: bool) -> torch.Tensor:
    """Sort deterministically, using stable sorting when the runtime supports it."""

    try:
        return torch.argsort(values, descending=descending, stable=True)
    except TypeError:  # older torch releases
        return torch.argsort(values, descending=descending)


def percentile_rank(scores: torch.Tensor) -> torch.Tensor:
    """Convert one image's scores to ranks in ``(0, 1]``.

    Ranking is performed independently for each image.  Stable ordering makes
    ties deterministic while preserving the required order-statistic meaning
    of a percentile score.
    """

    _require_torch()
    values = scores.detach().flatten().float()
    if values.numel() == 0:
        raise ValueError("cannot rank an empty visual-token score vector")
    order = _stable_argsort(values, descending=False)
    ranks = torch.empty(values.shape, dtype=torch.float32, device=values.device)
    ranks[order] = torch.arange(1, values.numel() + 1, device=values.device, dtype=torch.float32)
    return ranks / float(values.numel())


def topk_cosine_mean(
    visual_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    *,
    top_k: int = 4,
    text_is_normalized: bool = False,
) -> torch.Tensor:
    """Score every visual token by the mean of its top-k text cosine values."""

    _require_torch()
    if visual_tokens.ndim != 2 or text_tokens.ndim != 2:
        raise ValueError("visual_tokens and text_tokens must both be rank-2 tensors")
    if visual_tokens.shape[-1] != text_tokens.shape[-1]:
        raise RuntimeError(
            "projected visual/text dimensions differ: "
            f"visual={visual_tokens.shape[-1]}, text={text_tokens.shape[-1]}"
        )
    if text_tokens.shape[0] == 0:
        raise ValueError("no valid text tokens remain after filtering")
    k = min(max(1, int(top_k)), int(text_tokens.shape[0]))
    visual = F.normalize(visual_tokens.float(), dim=-1)
    text = text_tokens.float() if text_is_normalized else F.normalize(text_tokens.float(), dim=-1)
    similarity = visual @ text.transpose(0, 1)
    return similarity.topk(k=k, dim=-1, largest=True, sorted=True).values.mean(dim=-1)


def _is_punctuation_only(decoded: str) -> bool:
    stripped = "".join(char for char in str(decoded) if not char.isspace())
    stripped = stripped.lstrip("▁Ġ")
    if stripped.startswith("##"):
        stripped = stripped[2:]
    if not stripped:
        return True
    # Chat-template markers are special even on tokenizers whose
    # ``all_special_ids`` list is incomplete.
    if stripped.startswith("<|") or (stripped.startswith("<") and stripped.endswith(">")):
        return True
    return all(unicodedata.category(char).startswith("P") for char in stripped)


def _decode_token(tokenizer: Any, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise RuntimeError("tokenizer must expose decode() for text-token filtering")
    try:
        return str(decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    except TypeError:
        return str(decode([int(token_id)], skip_special_tokens=False))


def valid_text_token_mask(
    input_ids: torch.Tensor,
    tokenizer: Any = None,
    attention_mask: torch.Tensor | None = None,
    *,
    pad_token_id: int | None = None,
    special_token_ids: Sequence[int] = (),
    extra_special_token_ids: Sequence[int] = (),
) -> torch.Tensor:
    """Return a mask excluding padding, special tokens, and pure punctuation.

    ``input_ids`` may be one-dimensional or batched.  The runtime path passes
    the tokenizer from the loaded processor so punctuation is checked using
    the model's actual tokenization rather than token-id heuristics.
    """

    _require_torch()
    if input_ids.ndim not in (1, 2):
        raise ValueError("input_ids must have shape [sequence] or [batch, sequence]")
    ids = input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)
    if attention_mask is None:
        active = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
    else:
        mask = attention_mask if attention_mask.ndim == 2 else attention_mask.unsqueeze(0)
        if mask.shape != ids.shape:
            raise ValueError("attention_mask and input_ids must have matching shapes")
        active = mask.to(dtype=torch.bool)
    special = {int(value) for value in special_token_ids}
    special.update(int(value) for value in extra_special_token_ids)
    if pad_token_id is not None:
        special.add(int(pad_token_id))
    if tokenizer is not None:
        special.update(int(value) for value in getattr(tokenizer, "all_special_ids", ()) or ())

    result = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
    for row in range(ids.shape[0]):
        for col in range(ids.shape[1]):
            token_id = int(ids[row, col].item())
            if not bool(active[row, col]) or token_id in special:
                continue
            decoded = _decode_token(tokenizer, token_id)
            if not _is_punctuation_only(decoded):
                result[row, col] = True
    return result[0] if input_ids.ndim == 1 else result


def _budget_counts(keep_k: int) -> tuple[int, int, int]:
    """Allocate an exact integer 6:2:2 budget with largest-remainder rounding."""

    if int(keep_k) <= 0:
        raise ValueError("keep_k must be positive")
    target = [0.6 * int(keep_k), 0.2 * int(keep_k), 0.2 * int(keep_k)]
    counts = [math.floor(value) for value in target]
    remaining = int(keep_k) - sum(counts)
    order = sorted(range(3), key=lambda index: (target[index] - counts[index], -index), reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def select_budgeted_tokens(
    q_shallow: torch.Tensor,
    q_middle: torch.Tensor,
    q_final: torch.Tensor,
    keep_k: int,
) -> SelectionResult:
    """Apply the disjoint final/middle/shallow 6:2:2 selection rule."""

    _require_torch()
    q_s, q_m, q_f = (value.detach().flatten().float() for value in (q_shallow, q_middle, q_final))
    if not (q_s.numel() == q_m.numel() == q_f.numel()):
        raise ValueError("all three percentile vectors must have the same length")
    total = int(q_f.numel())
    if int(keep_k) > total:
        raise ValueError(f"keep_k={keep_k} exceeds available visual tokens={total}")
    final_k, middle_k, shallow_k = _budget_counts(int(keep_k))
    all_indices = torch.arange(total, device=q_f.device, dtype=torch.long)
    final_indices = _stable_argsort(q_f, descending=True)[:final_k]
    remaining_after_final = torch.ones(total, dtype=torch.bool, device=q_f.device)
    remaining_after_final[final_indices] = False

    middle_scores = q_m * torch.relu(q_m - q_f)
    middle_candidates = all_indices[remaining_after_final]
    middle_order = _stable_argsort(middle_scores[middle_candidates], descending=True)
    middle_indices = middle_candidates[middle_order[:middle_k]]
    remaining_after_middle = remaining_after_final.clone()
    remaining_after_middle[middle_indices] = False

    shallow_scores = q_s * torch.relu(q_s - torch.maximum(q_m, q_f))
    shallow_candidates = all_indices[remaining_after_middle]
    shallow_order = _stable_argsort(shallow_scores[shallow_candidates], descending=True)
    shallow_indices = shallow_candidates[shallow_order[:shallow_k]]

    selected = torch.cat((final_indices, middle_indices, shallow_indices), dim=0)
    if int(selected.numel()) != int(keep_k) or int(torch.unique(selected).numel()) != int(keep_k):
        raise RuntimeError("internal token-budget error: selected indices are not exactly K and disjoint")
    selected = selected.sort().values
    return SelectionResult(
        indices=selected,
        final_indices=final_indices.sort().values,
        middle_indices=middle_indices.sort().values,
        shallow_indices=shallow_indices.sort().values,
        budget=(final_k, middle_k, shallow_k),
    )


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch is not None and torch.is_tensor(value):
        return value
    for key in ("last_hidden_state", "hidden_states", "pooler_output"):
        candidate = getattr(value, key, None)
        if candidate is not None and candidate is not value:
            found = _first_tensor(candidate)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for key in ("last_hidden_state", "image_embeds", "image_features", "hidden_states"):
            if key in value:
                found = _first_tensor(value[key])
                if found is not None:
                    return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _extract_hidden_states(output: Any) -> list[torch.Tensor]:
    value: Any = getattr(output, "hidden_states", None)
    if value is None and isinstance(output, Mapping):
        value = output.get("hidden_states")
    if value is None and isinstance(output, (tuple, list)):
        # Prefer a sequence of tensors over the final tensor/deepstack tuple.
        for item in output:
            if isinstance(item, (tuple, list)) and item and all(torch.is_tensor(part) for part in item):
                value = item
                break
    if not isinstance(value, (tuple, list)) or not value or not all(torch.is_tensor(item) for item in value):
        raise RuntimeError(
            "the complete ViT output did not expose hidden_states; "
            "multi-layer training-free pruning cannot fall back to hidden-state distance"
        )
    return list(value)


def _extract_deepstack_features(output: Any) -> list[torch.Tensor]:
    """Extract native Qwen DeepStack streams when the ViT exposes them."""

    value: Any = getattr(output, "deepstack_features", None)
    if value is None:
        value = getattr(output, "deepstack_visual_embeds", None)
    if value is None and isinstance(output, Mapping):
        value = output.get("deepstack_features", output.get("deepstack_visual_embeds"))
    if value is None and isinstance(output, (tuple, list)) and len(output) >= 2:
        value = output[1]
    if value is None:
        return []
    if not isinstance(value, (tuple, list)) or not all(torch.is_tensor(item) for item in value):
        return []
    return list(value)


def _flatten_tokens(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise RuntimeError(f"ViT hidden state must have at least 2 dimensions, got {tuple(value.shape)}")
    return value.reshape(-1, int(value.shape[-1]))


def _split_tokens(value: torch.Tensor, counts: Sequence[int], *, layer: int) -> list[torch.Tensor]:
    flat = _flatten_tokens(value)
    expected = sum(int(item) for item in counts)
    if int(flat.shape[0]) != expected:
        raise RuntimeError(
            f"projected ViT layer {layer} has {flat.shape[0]} tokens, expected {expected}; "
            "the checkpoint projector did not produce one merged token per image patch"
        )
    return list(flat.split([int(item) for item in counts], dim=0))


def _decode_captured_vision_qkv(
    raw_qkv: torch.Tensor,
    attention_module: Any,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    rotated_qk: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one native vision qkv projection without invoking it twice."""

    if raw_qkv.ndim != 2:
        raise RuntimeError(f"captured vision qkv must be rank-2, got {tuple(raw_qkv.shape)}")
    heads = int(getattr(attention_module, "num_heads", 0) or 0)
    if heads <= 0:
        raise RuntimeError("captured vision attention exposes no positive num_heads")
    hidden = int(raw_qkv.shape[-1])
    if hidden % 3 != 0 or (hidden // 3) % heads != 0:
        raise RuntimeError(
            f"captured vision qkv shape is incompatible with heads={heads}: {tuple(raw_qkv.shape)}"
        )
    head_dim = (hidden // 3) // heads
    q, k, value = (
        raw_qkv.reshape(raw_qkv.shape[0], 3, heads, head_dim)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    if rotated_qk is not None:
        q, k = rotated_qk
    else:
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

            q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        except (ImportError, AttributeError):
            cos, sin = position_embeddings
            q_float, k_float = q.float(), k.float()
            cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
            q_half = torch.cat((-q_float[..., q_float.shape[-1] // 2 :], q_float[..., : q_float.shape[-1] // 2]), dim=-1)
            k_half = torch.cat((-k_float[..., k_float.shape[-1] // 2 :], k_float[..., : k_float.shape[-1] // 2]), dim=-1)
            q = ((q_float * cos) + (q_half * sin)).to(dtype=q.dtype)
            k = ((k_float * cos) + (k_half * sin)).to(dtype=k.dtype)
    return q, k, value


class TrainingFreeQwenVL(QwenVL):
    """Qwen-VL runner with multi-layer or single-layer text-guided pruning."""

    # Keep the system prompt identical to the baseline runner.  A different
    # system message changes the answer distribution and makes pruning vs.
    # baseline comparisons invalid.
    SYSTEM_INSTRUCTION = QwenVL.SYSTEM_INSTRUCTION

    def __init__(
        self,
        name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda:0",
        dtype: str = "auto",
        max_new_tokens: int = 32,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        visual_pruning: bool = True,
        visual_keep_ratio: float = 0.5,
        visual_keep_tokens: int | None = None,
        vision_layers: Sequence[int] = (6, 12, 24),
        pruning_mode: str = "multi_layer",
        pruning_layer: int | None = None,
        pruning_strategy: str = "topk",
        max_image_pixels: int = 4_194_304,
        enable_thinking: bool = False,
    ) -> None:
        _require_torch()
        super().__init__(
            name=name,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            max_image_pixels=max_image_pixels,
            enable_thinking=enable_thinking,
        )
        layers = tuple(int(value) for value in vision_layers)
        if len(layers) != 3 or tuple(sorted(set(layers))) != layers or layers[0] <= 0:
            raise ValueError("vision_layers must contain three strictly increasing positive layer numbers")
        if not 0.0 < float(visual_keep_ratio) <= 1.0:
            raise ValueError("visual_keep_ratio must be in (0, 1]")
        if visual_keep_tokens is not None and int(visual_keep_tokens) <= 0:
            raise ValueError("visual_keep_tokens must be positive when supplied")
        mode = str(pruning_mode).strip().lower().replace("-", "_")
        if mode not in {"multi_layer", "middle_text", *_EARLY_MIDDLE_MODES}:
            raise ValueError(
                "pruning_mode must be 'multi_layer', 'middle_text', "
                "'middle_text_early', or 'middle_text_early_grid'"
            )
        selected_layer = layers[1] if pruning_layer is None else int(pruning_layer)
        if mode in _MIDDLE_TEXT_MODES and selected_layer not in layers:
            if len(layers) == 3 and layers[0] < selected_layer < layers[-1]:
                layers = (layers[0], selected_layer, layers[-1])
            else:
                raise ValueError(
                    "middle_text pruning_layer must lie between the first and last "
                    f"--vision-layers; got {selected_layer}, available={layers}"
                )
        if _is_early_middle_mode(mode) and selected_layer >= layers[-1]:
            raise ValueError(
                "middle_text_early requires pruning_layer to be before the final "
                f"vision layer; got layer={selected_layer}, final={layers[-1]}"
            )
        strategy = str(pruning_strategy).strip().lower().replace("-", "_")
        if strategy != "topk":
            raise ValueError("pruning_strategy must be 'topk'")
        self.vision_layers = layers
        self.pruning_mode = mode
        self.pruning_layer = selected_layer
        self.pruning_strategy = strategy
        self.visual_keep_ratio = float(visual_keep_ratio)
        self.visual_keep_tokens = (
            int(visual_keep_tokens) if visual_keep_tokens is not None else None
        )
        self.visual_pruning = bool(visual_pruning)
        inner_model = getattr(self.model, "model", None)
        # Qwen3-VL keeps ``visual`` and ``language_model`` on the nested
        # Qwen3VLModel in current Transformers releases; older releases also
        # expose top-level aliases that QwenVL already discovered.
        if self.vit is None and inner_model is not None:
            self.vit = getattr(inner_model, "visual", getattr(inner_model, "vision_tower", None))
        if self.projection is None and self.vit is not None:
            self.projection = getattr(self.vit, "merger", getattr(self.vit, "proj", None))
        self.llm = getattr(inner_model, "language_model", self.llm)
        self._last_selection: dict[str, Any] = {}
        self._last_rope_deltas: torch.Tensor | None = None
        self._last_compact_rope_deltas: torch.Tensor | None = None
        self._last_native_deepstack: list[list[torch.Tensor]] = []
        # The vision tower already applies its native final merger.  Reuse that
        # result for the final-layer score instead of projecting layer 24 a
        # second time after the ViT forward.
        self._last_native_final_features: list[torch.Tensor] = []
        # ``middle_text_early`` fills these during the single custom ViT pass;
        # the compact prompt builder consumes the original merged-token indices
        # after the remaining vision blocks have finished.
        self._last_early_selections: list[SelectionResult] = []
        self._last_early_original_counts: list[int] = []
        self._last_vision_layer_token_counts: list[int] = []
        self._last_cached_visual: _SelectedVisionOutput | None = None
        self._freeze_projection()
        # The checkpoint's generation_config carries sampling-only fields even
        # though this runner always uses greedy decoding.  Clearing them once
        # prevents Transformers from warning on every case and avoids keeping
        # an invalid sampling configuration in the generation hot path.
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            for field_name in ("temperature", "top_p", "top_k"):
                if hasattr(generation_config, field_name):
                    setattr(generation_config, field_name, None)

    def _inputs(self, image: Any, prompt: str) -> Mapping[str, Any]:
        """Build the LLM prompt with a system instruction and user image/question."""

        images = image if isinstance(image, (list, tuple)) else [image]
        content = [{"type": "image", "image": item} for item in images]
        content.append({"type": "text", "text": prompt})
        messages = [
            # Qwen3-VL's processor expects every message content to be a list
            # of typed parts.  A bare system string is iterated character by
            # character by some releases and raises ``string indices must be
            # integers`` before generation starts.
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_INSTRUCTION}]},
            {"role": "user", "content": content},
        ]
        # Follow the RS-OPSD/VL processor contract: render the multimodal chat
        # template first, then call the processor with the rendered text and
        # ordered images.  This avoids version-specific tuple wrappers from
        # ``apply_chat_template(..., tokenize=True)`` while preserving the
        # exact image placeholder order used for ViT scoring.
        inputs = None
        normalized = None
        render = getattr(self.processor, "apply_chat_template", None)
        process = self.processor if callable(self.processor) else None
        rendered = None
        if callable(render):
            try:
                rendered = render(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                rendered = render(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, (tuple, list)):
                rendered = next((item for item in rendered if isinstance(item, str)), None)
        if isinstance(rendered, str) and process is not None:
            per_image_pixels = max(65_536, self.max_image_pixels // max(1, len(images)))
            try:
                inputs = process(
                    text=[rendered],
                    images=images,
                    padding=True,
                    return_tensors="pt",
                    size={
                        "longest_edge": per_image_pixels,
                        "shortest_edge": min(per_image_pixels, 65_536),
                    },
                    max_pixels=per_image_pixels,
                    do_sample_frames=False,
                )
            except TypeError:
                try:
                    inputs = process(
                        text=[rendered], images=images, padding=True,
                        return_tensors="pt",
                        size={
                            "longest_edge": per_image_pixels,
                            "shortest_edge": min(per_image_pixels, 65_536),
                        },
                        max_pixels=per_image_pixels,
                    )
                except TypeError:
                    try:
                        inputs = process(text=[rendered], images=images, return_tensors="pt")
                    except TypeError:
                        inputs = None
            normalized = _find_processor_inputs(inputs, self.device)
        # Compatibility fallback for processors whose chat template already
        # returns a complete BatchFeature when tokenization is requested.
        if normalized is None and callable(render):
            inputs = render(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            normalized = _find_processor_inputs(inputs, self.device)
            if normalized is None:
                normalized = _infer_tensor_processor_inputs(inputs, self.device)
        if normalized is None:
            raise RuntimeError(
                "processor.apply_chat_template returned no payload containing input_ids; "
                f"type={type(inputs).__name__}, value_types="
                f"{[type(item).__name__ for item in inputs] if isinstance(inputs, (tuple, list)) else 'n/a'}"
            )
        mapping = _as_mapping(normalized)
        if mapping is None:
            raise RuntimeError(
                "processor payload containing input_ids is not mapping-like; "
                f"type={type(normalized).__name__}"
            )
        try:
            mapping = _normalize_processor_mapping(mapping, self.device)
        except Exception as exc:
            raise RuntimeError(
                "processor tensor normalization failed before ViT pruning: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # ``token_type_ids`` is emitted by some tokenizer versions but is not
        # a Qwen-VL forward argument (the multimodal field is
        # ``mm_token_type_ids``).  Drop it before ``generate`` validates
        # model kwargs; retain all native image/position fields.
        mapping = {
            key: value
            for key, value in mapping.items()
            if key != "token_type_ids"
        }
        missing = [key for key in ("input_ids", "pixel_values", "image_grid_thw") if mapping.get(key) is None]
        if missing:
            present = list(mapping.keys())
            raise RuntimeError(
                "processor output is missing required multimodal fields "
                f"{missing}; present={present}; prompt/image construction cannot continue"
            )
        for key in ("input_ids", "pixel_values", "image_grid_thw"):
            if torch is not None and not torch.is_tensor(mapping[key]):
                raise RuntimeError(
                    f"processor field {key!r} is {type(mapping[key]).__name__}, expected a Tensor; "
                    "the multimodal model cannot call .to() on this value"
                )
        if mapping["input_ids"].ndim != 2:
            raise RuntimeError(
                f"processor input_ids must have shape [batch, sequence], got {tuple(mapping['input_ids'].shape)}"
            )
        if mapping["image_grid_thw"].numel() == 0 or mapping["image_grid_thw"].numel() % 3:
            raise RuntimeError(
                "processor image_grid_thw must contain one (T,H,W) triple per image view"
            )
        return mapping

    def _freeze_projection(self) -> None:
        projection = self.projection
        if projection is None or not callable(projection):
            raise RuntimeError(
                "the loaded ViT exposes no callable visual projector/merger; "
                "intermediate hidden states cannot be mapped to text space"
            )
        parameters = getattr(projection, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad_(False)

    def stages(self) -> tuple[str, ...]:
        stages = ["ViT", "Projection"]
        if self.visual_pruning:
            scoring_stage = (
                "MiddleLayerTextScoringEarly"
                if self.pruning_mode == "middle_text_early_grid"
                else "MiddleLayerTextScoringEarly"
                if self.pruning_mode == "middle_text_early"
                else "MiddleLayerTextScoring"
                if self.pruning_mode == "middle_text"
                else "MultiLayerTokenScoring"
            )
            stages.extend((scoring_stage, "VisualTokenPruning"))
        stages.extend(("LLM", "Answer"))
        return tuple(stages)

    def _vision_forward(
        self,
        inputs: Mapping[str, Any],
        *,
        text_tokens: torch.Tensor | None = None,
    ) -> Mapping[int, torch.Tensor] | list[torch.Tensor]:
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError("training-free pruning requires pixel_values and image_grid_thw")
        if not torch.is_tensor(pixel_values) or not torch.is_tensor(image_grid_thw):
            raise RuntimeError(
                "ViT input normalization failed: pixel_values and image_grid_thw must be tensors; "
                f"got pixel_values={type(pixel_values).__name__}, "
                f"image_grid_thw={type(image_grid_thw).__name__}"
            )
        visual_dtype = getattr(self.vit, "dtype", None)
        if visual_dtype is not None and torch.is_floating_point(pixel_values):
            pixel_values = pixel_values.to(dtype=visual_dtype)
        if _is_early_middle_mode(self.pruning_mode):
            if text_tokens is None:
                raise RuntimeError("middle_text_early requires text tokens before the ViT forward")
            return self._vision_forward_middle_early(pixel_values, image_grid_thw, text_tokens)
        if hasattr(self.vit, "window_size") and not hasattr(self.vit, "deepstack_merger_list"):
            raise RuntimeError(
                "this ViT uses an internal window permutation without exposing a reliable inverse for "
                "intermediate hidden states; refusing to score hidden-state distance in permuted order"
            )
        # Qwen3-VL's vision submodule calls its image tensor ``hidden_states``;
        # older Qwen-VL implementations use ``pixel_values``.  Both variants
        # are supported, but every attempt still requests the complete block
        # trace explicitly.
        no_trace_attempts = (
            {"hidden_states": pixel_values, "grid_thw": image_grid_thw, "return_dict": True},
            {"hidden_states": pixel_values, "grid_thw": image_grid_thw},
            {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "return_dict": True},
            {"pixel_values": pixel_values, "grid_thw": image_grid_thw},
        )
        trace_attempts = (
            {"hidden_states": pixel_values, "grid_thw": image_grid_thw, "output_hidden_states": True, "return_dict": True},
            {"hidden_states": pixel_values, "grid_thw": image_grid_thw, "output_hidden_states": True},
            {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "output_hidden_states": True, "return_dict": True},
            {"pixel_values": pixel_values, "grid_thw": image_grid_thw, "output_hidden_states": True},
        )
        # Some Transformers releases expose only ``(last_hidden_state,
        # pooler_output)`` even when ``output_hidden_states=True`` is passed.
        # Capture each requested ViT block directly without using a
        # hidden-state-distance fallback.
        block_container = getattr(self.vit, "blocks", None)
        if block_container is None:
            encoder = getattr(self.vit, "encoder", None)
            block_container = getattr(encoder, "layers", None) if encoder is not None else None
            if block_container is None and encoder is not None:
                block_container = getattr(encoder, "layer", None)
        captured: dict[int, torch.Tensor] = {}
        # The previous case's selected features are only needed during its
        # generation call.  Release them before allocating the next complete
        # ViT trace; otherwise the next vision pass can briefly retain both
        # cases and trigger a late-run CUDA/OS OOM.
        self._last_cached_visual = None
        self._last_native_deepstack = []
        self._last_native_final_features = []
        requested_layers = set(self.vision_layers)
        if self.pruning_mode == "middle_text":
            # A single-layer selector only needs its scoring block. The
            # native final pooler output is used for generation and does not
            # need a second block hook.
            requested_layers = {self.pruning_layer}
        handles = []
        if block_container is not None:
            try:
                blocks = list(block_container)
            except TypeError:
                blocks = []
            for index, block in enumerate(blocks, start=1):
                if index not in requested_layers:
                    continue
                register = getattr(block, "register_forward_hook", None)
                if not callable(register):
                    continue

                def _capture(_module: Any, _args: Any, result: Any, *, layer_index: int = index) -> None:
                    tensor = _first_tensor(result)
                    if tensor is not None:
                        captured[layer_index] = tensor

                handles.append(register(_capture))

        last_error: Exception | None = None
        output: Any = None
        try:
            attempts = no_trace_attempts + trace_attempts if handles else trace_attempts
            for kwargs in attempts:
                try:
                    output = self.vit(**kwargs)
                    break
                except TypeError as exc:
                    last_error = exc
        finally:
            for handle in handles:
                handle.remove()
        if output is None:
            raise RuntimeError(
                "the complete ViT cannot return hidden_states with this checkpoint; "
                "intermediate layers cannot be reliably projected to text space"
            ) from last_error
        merge_size = int(
            _get(_get(self.config, "vision_config", None), "spatial_merge_size", 1) or 1
        )
        counts = self._grid_counts(image_grid_thw, merge_size)
        # Qwen3-VL returns the native final projected stream as
        # ``pooler_output`` in newer releases and as the first tensor of the
        # vision tuple in older releases.  Both are exactly the frozen
        # projector output required for the final score and LLM replay.
        native_final: Any = getattr(output, "pooler_output", None)
        if native_final is None and isinstance(output, (tuple, list)) and output:
            candidate = output[0]
            if torch.is_tensor(candidate) or (
                isinstance(candidate, (tuple, list))
                and all(torch.is_tensor(item) for item in candidate)
            ):
                native_final = candidate
        try:
            if torch.is_tensor(native_final):
                self._last_native_final_features = _split_tokens(
                    native_final, counts, layer=self.vision_layers[-1]
                )
            elif isinstance(native_final, (tuple, list)) and all(
                torch.is_tensor(item) for item in native_final
            ):
                if len(native_final) != len(counts):
                    raise RuntimeError("native final feature view count does not match image_grid_thw")
                if any(int(item.shape[0]) != int(count) for item, count in zip(native_final, counts)):
                    raise RuntimeError("native final feature lengths do not match image_grid_thw")
                self._last_native_final_features = list(native_final)
        except (RuntimeError, ValueError):
            # A checkpoint with an incompatible output layout must use the
            # explicit frozen projector path below; never substitute a
            # hidden-state distance or silently alter token semantics.
            self._last_native_final_features = []
        native_deepstack = _extract_deepstack_features(output)
        for feature in native_deepstack:
            try:
                pieces = _split_tokens(feature, counts, layer=0)
            except (RuntimeError, ValueError):
                self._last_native_deepstack = []
                break
            self._last_native_deepstack.append(pieces)
        if captured:
            if requested_layers.issubset(captured) and (
                self.pruning_mode not in _EARLY_MIDDLE_MODES or self._last_native_final_features
            ):
                return captured
        states = _extract_hidden_states(output)
        required = max(self.vision_layers)
        if len(states) < required:
            raise RuntimeError(
                f"requested ViT layer {required}, but complete hidden_states expose only "
                f"{len(states)} layers; block-level capture was unavailable or incomplete, "
                "so intermediate layers cannot be reliably projected to text space"
            )
        return states

    def _vision_forward_middle_early(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Mapping[int, torch.Tensor]:
        """Run Qwen3-VL blocks while dropping selected merged tokens at layer 18.

        Qwen3-VL's visual sequence is ordered as groups of
        ``spatial_merge_size ** 2`` raw patch tokens.  The layer-18 selector
        scores the merged groups, then this method keeps all raw patches in a
        selected group, updates the variable-length attention boundaries and
        rotary rows, and lets only those groups enter the remaining blocks.
        DeepStack features produced before the cut are sliced with the same
        merged-token indices; features produced after the cut naturally have
        the compact length.

        This is intentionally a Qwen3-VL-specific path.  The original
        ``middle_text`` mode remains the compatibility-safe one-forward
        selector for other visual implementations.
        """

        vit = self.vit
        required = (
            "patch_embed",
            "fast_pos_embed_interpolate",
            "rot_pos_emb",
            "blocks",
            "merger",
            "deepstack_merger_list",
        )
        missing = [name for name in required if not hasattr(vit, name)]
        if missing:
            raise RuntimeError(
                "middle_text_early requires Qwen3-VL visual internals; "
                f"missing={missing}"
            )

        merge_size = int(
            getattr(vit, "spatial_merge_size", None)
            or _get(_get(self.config, "vision_config", None), "spatial_merge_size", 1)
            or 1
        )
        merge_unit = max(1, merge_size * merge_size)
        # Materialize metadata once on the host.  The selector itself stays
        # on-device; this avoids one CUDA scalar synchronization per view when
        # constructing its real merged grid shape.
        grid_rows = image_grid_thw.detach().reshape(-1, 3).cpu().tolist()
        counts = [
            int(row[0]) * (int(row[1]) // max(1, merge_size)) * (int(row[2]) // max(1, merge_size))
            for row in grid_rows
        ]
        # Image views have one temporal slice. Keeping one attention boundary
        # per view would mix video frames; reject that unsupported contract.
        if any(int(row[0]) != 1 for row in grid_rows):
            raise RuntimeError("middle_text_early currently supports image views with T=1 only")
        raw_counts = [int(count) * merge_unit for count in counts]
        raw_total = sum(raw_counts)

        self._last_cached_visual = None
        self._last_native_deepstack = []
        self._last_native_final_features = []
        self._last_early_selections = []
        self._last_early_original_counts = list(counts)
        self._last_vision_layer_token_counts = []

        hidden_states = vit.patch_embed(pixel_values)
        if int(hidden_states.shape[0]) != raw_total:
            raise RuntimeError(
                "middle_text_early received a visual sequence whose length does not "
                f"match image_grid_thw: hidden={hidden_states.shape[0]}, expected={raw_total}"
            )
        hidden_states = hidden_states + vit.fast_pos_embed_interpolate(image_grid_thw)
        rotary_pos_emb = vit.rot_pos_emb(image_grid_thw)
        seq_len = int(hidden_states.shape[0])
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        ).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_indexes = tuple(
            int(value) for value in getattr(vit, "deepstack_visual_indexes", ())
        )
        deepstack_feature_lists: list[torch.Tensor] = []
        compact_counts: list[int] | None = None
        prune_block = int(self.pruning_layer) - 1
        blocks = list(vit.blocks)
        if prune_block < 0 or prune_block >= len(blocks) - 1:
            raise RuntimeError(
                f"middle_text_early pruning layer {self.pruning_layer} must precede "
                f"the last visual block ({len(blocks)})"
            )

        for layer_num, block in enumerate(blocks):
            self._last_vision_layer_token_counts.append(int(hidden_states.shape[0]))
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in deepstack_indexes:
                merger_index = deepstack_indexes.index(layer_num)
                deepstack_feature_lists.append(
                    vit.deepstack_merger_list[merger_index](hidden_states)
                )

            if layer_num != prune_block:
                continue

            projected_middle = _flatten_tokens(
                self._project_layer(hidden_states, self.pruning_layer, image_grid_thw)
            )
            expected_projected = sum(counts)
            if int(projected_middle.shape[0]) != expected_projected:
                raise RuntimeError(
                    f"layer {self.pruning_layer} projector returned {projected_middle.shape[0]} "
                    f"merged tokens, expected {expected_projected}"
                )
            projected_views = _split_tokens(
                projected_middle, counts, layer=self.pruning_layer
            )
            if self.pruning_mode == "middle_text_early_grid":
                selections = self._select_middle_text_tokens_grid(
                    {self.pruning_layer: projected_views},
                    text_tokens,
                    counts,
                    image_grid_thw,
                    merge_size,
                )
            else:
                selections = self._select_middle_text_tokens(
                    {self.pruning_layer: projected_views},
                    text_tokens,
                    counts,
                )
            compact_counts = [int(selection.indices.numel()) for selection in selections]

            raw_indices: list[torch.Tensor] = []
            offset = 0
            for selection, raw_count in zip(selections, raw_counts):
                group_offsets = torch.arange(
                    merge_unit, device=selection.indices.device, dtype=torch.long
                )
                local = (
                    selection.indices.to(dtype=torch.long).unsqueeze(-1) * merge_unit
                    + group_offsets
                ).reshape(-1)
                raw_indices.append(local + offset)
                offset += int(raw_count)
            keep_raw = torch.cat(raw_indices, dim=0)
            hidden_states = hidden_states.index_select(0, keep_raw)
            position_embeddings = tuple(
                value.index_select(0, keep_raw) for value in position_embeddings
            )
            raw_compact_counts = [count * merge_unit for count in compact_counts]
            cu_seqlens = F.pad(torch.tensor(
                raw_compact_counts,
                dtype=cu_seqlens.dtype,
                device=cu_seqlens.device,
            ).cumsum(0, dtype=cu_seqlens.dtype), (1, 0), value=0)

            # DeepStack mergers before the cut emitted merged-token features
            # at the original length.  Slice those features before later
            # language-model fusion; post-cut features are already compact.
            sliced_deepstack: list[torch.Tensor] = []
            for feature in deepstack_feature_lists:
                feature_views = _split_tokens(feature, counts, layer=0)
                sliced_deepstack.append(
                    torch.cat(
                        [
                            view.index_select(0, selection.indices)
                            for view, selection in zip(feature_views, selections)
                        ],
                        dim=0,
                    )
                )
            deepstack_feature_lists = sliced_deepstack
            self._last_early_selections = selections
            # The full layer-18 projected matrix is only needed for this one
            # top-K operation. Release it before layers 19+ allocate their
            # activations.
            del projected_middle, projected_views

        if compact_counts is None:
            raise RuntimeError(
                f"middle_text_early did not reach pruning layer {self.pruning_layer}"
            )
        hidden_states = vit.merger(hidden_states)
        if int(hidden_states.shape[0]) != sum(compact_counts):
            raise RuntimeError(
                "final visual merger returned an unexpected compact token count: "
                f"got={hidden_states.shape[0]}, expected={sum(compact_counts)}"
            )
        self._last_native_final_features = list(
            hidden_states.split(compact_counts, dim=0)
        )
        self._last_native_deepstack = [
            list(feature.split(compact_counts, dim=0))
            for feature in deepstack_feature_lists
        ]
        # The compact final/deepstack features and selection indices are stored
        # on the runner; the full layer-18 projection is no longer needed by
        # the prompt builder.
        return {}

    def _layer_state(self, states: Mapping[int, torch.Tensor] | Sequence[torch.Tensor], layer: int) -> torch.Tensor:
        if isinstance(states, Mapping):
            if layer not in states:
                raise RuntimeError(f"requested ViT layer {layer} is unavailable in captured block outputs")
            return states[layer]
        depth = int(_get(_get(self.config, "vision_config", None), "depth", 0) or 0)
        if depth <= 0:
            depth = max(self.vision_layers[-1], len(states) - 1)
        if len(states) >= depth + 1:
            index = layer
        elif len(states) >= depth:
            index = layer - 1
        else:
            raise RuntimeError(
                f"requested ViT layer {layer}, but complete hidden_states expose only "
                f"{len(states) - 1 if len(states) > 1 else len(states)} layers"
            )
        if index < 0 or index >= len(states):
            raise RuntimeError(f"requested ViT layer {layer} is unavailable in hidden_states")
        return states[index]

    def _project_layer(self, state: torch.Tensor, layer: int, image_grid_thw: torch.Tensor) -> torch.Tensor:
        projection = self.projection
        try:
            projected = projection(state)
        except (TypeError, RuntimeError):
            try:
                projected = projection(state, image_grid_thw)
            except Exception as second_error:
                raise RuntimeError(
                    f"visual projector cannot map ViT layer {layer} to text space; "
                    "refusing a hidden-state-distance fallback"
                ) from second_error
        tensor = _first_tensor(projected)
        if tensor is None:
            raise RuntimeError(f"visual projector returned no tensor for ViT layer {layer}")
        return tensor

    def _text_tokens(self, inputs: Mapping[str, Any], scoring_text: str | None = None) -> torch.Tensor:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        config = self.config
        image_id = int(_get(config, "image_token_id", -1))
        video_id = int(_get(config, "video_token_id", -1))
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if scoring_text is not None:
            encode = getattr(tokenizer, "__call__", None)
            if not callable(encode):
                raise RuntimeError("the processor tokenizer cannot encode the question for visual scoring")
            encoded = encode(scoring_text, add_special_tokens=False, return_tensors="pt")
            encoded_mapping = _as_mapping(encoded)
            if encoded_mapping is not None:
                raw_ids = encoded_mapping.get("input_ids")
                raw_attention = encoded_mapping.get("attention_mask")
            else:
                leaves = _tensor_leaves(encoded)
                if not leaves:
                    raise RuntimeError(
                        "question tokenizer returned no tensor input_ids; "
                        f"type={type(encoded).__name__}"
                    )
                integer_leaves = [item for item in leaves if not torch.is_floating_point(item)]
                if not integer_leaves:
                    raise RuntimeError(
                        "question tokenizer returned no integer input_ids; "
                        f"shapes={[tuple(item.shape) for item in leaves]}"
                    )
                raw_ids = max(integer_leaves, key=lambda item: int(item.numel()))
                raw_attention = next(
                    (
                        item
                        for item in integer_leaves
                        if item is not raw_ids
                        and tuple(item.shape) == tuple(raw_ids.shape)
                        and item.numel() > 0
                        and int(item.min().item()) >= 0
                        and int(item.max().item()) <= 1
                    ),
                    None,
                )
            if raw_ids is None:
                raise RuntimeError("question tokenizer output is missing input_ids")
            input_ids = _normalize_tensor_field("input_ids", raw_ids, self.device)
            attention_mask = (
                _normalize_tensor_field("attention_mask", raw_attention, self.device)
                if raw_attention is not None
                else None
            )
            if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
                raise RuntimeError(
                    "question tokenizer input_ids must be a rank-2 Tensor; "
                    f"got {type(input_ids).__name__}"
                )
        else:
            input_ids = inputs["input_ids"]
            if input_ids.shape[0] != 1:
                raise RuntimeError("training-free token scoring currently requires a single prompt batch")
            attention_mask = inputs.get("attention_mask")
        mask = valid_text_token_mask(
            input_ids[0],
            tokenizer,
            attention_mask[0] if attention_mask is not None else None,
            pad_token_id=pad_id,
            extra_special_token_ids=(image_id, video_id),
        )
        ids = input_ids[0][mask]
        if ids.numel() == 0:
            raise RuntimeError("no valid question/option text tokens remain after filtering")
        embeddings = self.model.get_input_embeddings()(ids.unsqueeze(0)).squeeze(0)
        if embeddings.ndim != 2:
            embeddings = embeddings.reshape(-1, int(embeddings.shape[-1]))
        return embeddings

    @staticmethod
    def _grid_counts(image_grid_thw: torch.Tensor, merge_size: int) -> list[int]:
        values = image_grid_thw.detach().reshape(-1, 3)
        counts = (values.prod(dim=-1) // max(1, int(merge_size)) ** 2).tolist()
        counts = [int(value) for value in counts]
        if any(value <= 0 for value in counts):
            raise RuntimeError(f"invalid image_grid_thw for visual pruning: {image_grid_thw}")
        return counts

    def _keep_count(self, total: int) -> int:
        """Compute the per-view budget from the requested retention ratio."""

        if self.visual_keep_tokens is not None:
            keep = self.visual_keep_tokens
        else:
            keep = int(round(float(total) * self.visual_keep_ratio))
        keep = max(1, int(keep))
        if keep > int(total):
            raise RuntimeError(
                "requested visual retention exceeds available tokens: "
                f"keep={keep}, available={total}, ratio={self.visual_keep_ratio:.4f}"
            )
        return keep

    @staticmethod
    def _middle_text_score_views(
        target_views: Sequence[torch.Tensor],
        text_tokens: torch.Tensor,
    ) -> list[torch.Tensor]:
        text_vector = F.normalize(text_tokens.float().mean(dim=0), dim=0)
        return [
            F.cosine_similarity(view.float(), text_vector.unsqueeze(0), dim=-1)
            for view in target_views
        ]

    def _select_middle_text_tokens(
        self,
        projected_layers: Mapping[int, list[torch.Tensor]],
        text_tokens: torch.Tensor,
        counts: Sequence[int],
        score_views: Sequence[torch.Tensor] | None = None,
    ) -> list[SelectionResult]:
        """Select top-K tokens from one projected layer using mean text cosine.

        This mirrors the simple selector in ``model/pruning.py``: the selected
        layer is compared with one L2-normalized mean text embedding, then the
        highest-scoring tokens are kept in their original patch order.  The
        returned indices are later applied to the native final-layer stream so
        the LLM still receives features in its expected embedding space.
        """

        target_layer = int(self.pruning_layer)
        target_views = projected_layers[target_layer]
        scores_by_view = (
            list(score_views)
            if score_views is not None
            else self._middle_text_score_views(target_views, text_tokens)
        )
        # ``vision_layers`` is ordered shallow,middle,final, while the
        # selection metadata is ordered final,middle,shallow.
        layer_role = self.vision_layers.index(target_layer)
        role = {0: 2, 1: 1, 2: 0}[layer_role]
        selections: list[SelectionResult] = []
        for image_index, total in enumerate(counts):
            keep = self._keep_count(int(total))
            scores = scores_by_view[image_index]
            indices = torch.topk(scores, k=keep, largest=True, sorted=False).indices.sort().values
            groups = [
                torch.empty(0, dtype=torch.long, device=indices.device)
                for _ in range(3)
            ]
            groups[role] = indices
            budget = [0, 0, 0]
            budget[role] = keep
            selections.append(
                SelectionResult(
                    indices=indices,
                    final_indices=groups[0],
                    middle_indices=groups[1],
                    shallow_indices=groups[2],
                    budget=tuple(budget),
                )
            )
        return selections

    def _select_middle_text_tokens_grid(
        self,
        projected_layers: Mapping[int, list[torch.Tensor]],
        text_tokens: torch.Tensor,
        counts: Sequence[int],
        image_grid_thw: torch.Tensor,
        merge_size: int,
    ) -> list[SelectionResult]:
        """Select layer-18 tokens with one representative per coarse grid cell.

        The cell representatives are mandatory only when the budget permits
        them. Any remaining budget is filled by the same global score order as
        ``middle_text``; this is deliberately not an equal per-cell quota.
        """

        target_layer = int(self.pruning_layer)
        target_views = projected_layers[target_layer]
        text_vector = F.normalize(text_tokens.float().mean(dim=0), dim=0)
        layer_role = self.vision_layers.index(target_layer)
        role = {0: 2, 1: 1, 2: 0}[layer_role]
        grids = image_grid_thw.reshape(-1, 3)
        merge = max(1, int(merge_size))
        merged_hw = grids[:, 1:3] // merge
        selections: list[SelectionResult] = []
        for image_index, total in enumerate(counts):
            keep = self._keep_count(int(total))
            scores = F.cosine_similarity(
                target_views[image_index].float(), text_vector.unsqueeze(0), dim=-1
            )
            # Keep grid dimensions on the same device as scores. Converting
            # image_grid_thw to Python lists here would synchronize CUDA.
            height = merged_hw[image_index, 0].to(device=scores.device)
            width = merged_hw[image_index, 1].to(device=scores.device)
            token_rows = torch.arange(int(total), device=scores.device) // width
            token_cols = torch.arange(int(total), device=scores.device) % width
            cell_rows = (token_rows * 4 // height).clamp_max(3)
            cell_cols = (token_cols * 4 // width).clamp_max(3)
            cell_ids = cell_rows * 4 + cell_cols
            # Compute all cell maxima and their first argmax representatives in
            # one device-side reduction. The previous per-cell where/argmax
            # loop launched many tiny CUDA kernels and made the bookkeeping
            # visible in end-to-end timing.
            cell_max = torch.full(
                (16,), -torch.inf, dtype=scores.dtype, device=scores.device
            )
            cell_max.scatter_reduce_(0, cell_ids, scores, reduce="amax", include_self=True)
            token_indices = torch.arange(int(total), device=scores.device, dtype=torch.long)
            max_tokens = torch.where(
                scores == cell_max[cell_ids], token_indices, torch.full_like(token_indices, int(total))
            )
            representative_indices_all = torch.full(
                (16,), int(total), dtype=torch.long, device=scores.device
            )
            representative_indices_all.scatter_reduce_(
                0, cell_ids, max_tokens, reduce="amin", include_self=True
            )
            valid_cells = torch.where(representative_indices_all < int(total))[0]
            cell_count = int(valid_cells.numel())
            mandatory_count = min(keep, cell_count)
            if mandatory_count < cell_count:
                # When K<16, retain the K cells whose representatives have
                # the highest cell maxima. This is the specified reduced-grid
                # behavior; no equal per-cell quota is introduced.
                chosen_cells = torch.topk(
                    cell_max[valid_cells], k=mandatory_count, largest=True, sorted=False
                ).indices
                mandatory_cells = torch.zeros(16, dtype=torch.bool, device=scores.device)
                mandatory_cells[valid_cells[chosen_cells]] = True
            else:
                mandatory_cells = torch.zeros(16, dtype=torch.bool, device=scores.device)
                mandatory_cells[valid_cells] = True

            # Every mandatory representative is promoted into one final
            # global Top-K. This avoids a second Top-K over non-representatives
            # and keeps all score/index operations on the same device.
            mandatory = token_indices == representative_indices_all.index_select(0, cell_ids)
            mandatory = mandatory & mandatory_cells.index_select(0, cell_ids)
            priority = scores.masked_fill(mandatory, torch.inf)
            indices = torch.topk(priority, k=keep, largest=True, sorted=False).indices.sort().values
            groups = [
                torch.empty(0, dtype=torch.long, device=indices.device)
                for _ in range(3)
            ]
            groups[role] = indices
            budget = [0, 0, 0]
            budget[role] = keep
            selections.append(
                SelectionResult(
                    indices=indices,
                    final_indices=groups[0],
                    middle_indices=groups[1],
                    shallow_indices=groups[2],
                    budget=tuple(budget),
                )
            )
        return selections

    @staticmethod
    def _replace_groups(
        row: torch.Tensor,
        counts: Sequence[int],
        replacements: Sequence[torch.Tensor],
        image_token_id: int,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = reference if reference is not None else row
        positions = torch.where(source == int(image_token_id))[0]
        if int(positions.numel()) != sum(int(value) for value in counts):
            raise RuntimeError(
                "image placeholder count does not match projected visual tokens: "
                f"placeholders={positions.numel()}, features={sum(counts)}"
            )
        chunks: list[torch.Tensor] = []
        cursor = 0
        last = 0
        for count, replacement in zip(counts, replacements):
            count = int(count)
            start = int(positions[cursor].item())
            end = int(positions[cursor + count - 1].item()) + 1
            chunks.extend((row[last:start], replacement))
            last = end
            cursor += count
        chunks.append(row[last:])
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _compact_grid(image_grid_thw: torch.Tensor, keep_counts: Sequence[int], merge_size: int) -> torch.Tensor:
        # The explicit position_ids below retain each selected token's original
        # 2-D coordinates.  This rectangular metadata only keeps the model's
        # visual-token count checks consistent after placeholders are compacted.
        rows = [(1, int(merge_size), max(1, int(count)) * int(merge_size)) for count in keep_counts]
        return torch.tensor(rows, device=image_grid_thw.device, dtype=image_grid_thw.dtype)

    def _original_position_ids(self, inputs: Mapping[str, Any], counts: Sequence[int], merge_size: int) -> torch.Tensor:
        self._last_rope_deltas = None
        for target in (self.model, getattr(self.model, "model", None)):
            if target is not None and hasattr(target, "rope_deltas"):
                target.rope_deltas = None
        if inputs.get("position_ids") is not None:
            return inputs["position_ids"]
        method_candidates = (
            getattr(self.model, "get_rope_index", None),
            getattr(self.model, "compute_3d_position_ids", None),
            getattr(getattr(self.model, "model", None), "get_rope_index", None),
            getattr(getattr(self.model, "model", None), "compute_3d_position_ids", None),
            getattr(self.llm, "get_rope_index", None),
            getattr(self.llm, "compute_3d_position_ids", None),
        )
        for method in method_candidates:
            if not callable(method):
                continue
            attempts = (
                {
                    "input_ids": inputs["input_ids"],
                    "mm_token_type_ids": inputs.get("mm_token_type_ids"),
                    "image_grid_thw": inputs["image_grid_thw"],
                    "video_grid_thw": inputs.get("video_grid_thw"),
                    "attention_mask": inputs.get("attention_mask"),
                },
                {"input_ids": inputs["input_ids"], "mm_token_type_ids": inputs.get("mm_token_type_ids"), "image_grid_thw": inputs["image_grid_thw"], "attention_mask": inputs.get("attention_mask")},
            )
            if _is_early_middle_mode(self.pruning_mode):
                # Transformers 4.57.6's native get_rope_index does not accept
                # mm_token_type_ids. Use that native image-grid contract for
                # the early mode before falling back to the compatibility API.
                attempts = ({
                    "input_ids": inputs["input_ids"],
                    "image_grid_thw": inputs["image_grid_thw"],
                    "video_grid_thw": inputs.get("video_grid_thw"),
                    "attention_mask": inputs.get("attention_mask"),
                },) + attempts
            for kwargs in attempts:
                try:
                    value = method(**kwargs)
                    if isinstance(value, (tuple, list)):
                        self._last_rope_deltas = value[1] if len(value) > 1 else None
                        return value[0]
                    self._last_rope_deltas = None
                    return value
                except Exception:
                    continue
        # A deterministic fallback still preserves the original 2-D patch
        # coordinates.  It is used only when a Transformers release does not
        # expose get_rope_index; it never changes scoring to hidden-state distance.
        return self._fallback_position_ids(inputs, counts, merge_size)

    def _fallback_position_ids(self, inputs: Mapping[str, Any], counts: Sequence[int], merge_size: int) -> torch.Tensor:
        ids = inputs["input_ids"]
        image_id = int(_get(self.config, "image_token_id", 151655))
        grid = inputs["image_grid_thw"].detach().reshape(-1, 3)
        output = torch.zeros((3, ids.shape[0], ids.shape[1]), dtype=torch.long, device=ids.device)
        for row in range(ids.shape[0]):
            positions = torch.where(ids[row] == image_id)[0]
            cursor = 0
            max_coord = 0
            for image_index, count in enumerate(counts):
                image_positions = positions[cursor : cursor + int(count)]
                t, h, w = (int(value) for value in grid[image_index].tolist())
                hm, wm = h // max(1, merge_size), w // max(1, merge_size)
                for local, position in enumerate(image_positions.tolist()):
                    per_frame = max(1, hm * wm)
                    tt, rem = divmod(local, per_frame)
                    hh, ww = divmod(rem, max(1, wm))
                    output[:, row, position] = torch.tensor((tt, hh, ww), device=ids.device)
                    max_coord = max(max_coord, tt, hh, ww)
                cursor += int(count)
            text_position = max_coord + 1
            for position in range(ids.shape[1]):
                if int(ids[row, position]) != image_id:
                    output[:, row, position] = text_position
                    text_position += 1
        return output

    @staticmethod
    def _compact_positions(
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        counts: Sequence[int],
        selected: Sequence[torch.Tensor],
        image_token_id: int,
    ) -> torch.Tensor:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == input_ids.shape[0] and position_ids.shape[1] in (3, 4):
            position_ids = position_ids.transpose(0, 1)
        if position_ids.ndim != 3 or position_ids.shape[1] != input_ids.shape[0]:
            raise RuntimeError(
                "position_ids must have shape [modal_dims, batch, sequence] to preserve original visual coordinates"
            )
        rows: list[torch.Tensor] = []
        for row in range(input_ids.shape[0]):
            source = input_ids[row]
            positions = torch.where(source == int(image_token_id))[0]
            chunks: list[torch.Tensor] = []
            cursor = 0
            last = 0
            for count, indices in zip(counts, selected):
                count = int(count)
                start = int(positions[cursor].item())
                end = int(positions[cursor + count - 1].item()) + 1
                chunks.append(position_ids[:, row, last:start])
                chunks.append(position_ids[:, row, positions[cursor : cursor + count][indices]])
                cursor += count
                last = end
            chunks.append(position_ids[:, row, last:])
            rows.append(torch.cat(chunks, dim=-1))
        return torch.stack(rows, dim=1)

    @staticmethod
    def _rope_delta(position_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Derive Qwen's compact-sequence mRoPE delta from position IDs."""

        if position_ids.ndim != 3:
            raise RuntimeError(f"compact position_ids must be rank 3, got {tuple(position_ids.shape)}")
        batch = int(input_ids.shape[0])
        if position_ids.shape[0] == batch and position_ids.shape[1] <= 4:
            maximum = position_ids.amax(dim=(1, 2))
        elif position_ids.shape[1] == batch:
            maximum = position_ids.amax(dim=(0, 2))
        else:
            raise RuntimeError(
                "unsupported compact position_ids layout: "
                f"position_ids={tuple(position_ids.shape)} input_ids={tuple(input_ids.shape)}"
            )
        return (maximum + 1 - int(input_ids.shape[1])).to(input_ids.dtype).view(-1, 1)

    def _pruned_inputs(
        self,
        inputs: Mapping[str, Any],
        *,
        scoring_text: str | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        for key in ("input_ids", "image_grid_thw"):
            value = inputs.get(key)
            if not torch.is_tensor(value):
                raise RuntimeError(
                    f"visual pruning received non-tensor {key}: {type(value).__name__}; "
                    "check processor field normalization before token selection"
                )
        original_tokens = self._image_tokens(inputs, self.config)
        if not self.visual_pruning:
            self._last_cached_visual = None
            self._last_compact_rope_deltas = None
            self._last_native_deepstack = []
            self._last_selection = {
                "retained_tokens": original_tokens,
                "original_tokens": original_tokens,
                "mode": self.pruning_mode,
                "strategy": getattr(self, "pruning_strategy", "topk"),
                "layer": self.pruning_layer
                if self.pruning_mode in _MIDDLE_TEXT_MODES
                else None,
            }
            return dict(inputs), original_tokens, original_tokens
        image_grid = inputs.get("image_grid_thw")
        if image_grid is None:
            raise RuntimeError("training-free pruning requires image_grid_thw")
        merge_size = int(_get(_get(self.config, "vision_config", None), "spatial_merge_size", 1) or 1)
        counts = self._grid_counts(image_grid, merge_size)
        if _is_early_middle_mode(self.pruning_mode):
            text_tokens = self._text_tokens(inputs, scoring_text=scoring_text)
            states = self._vision_forward(inputs, text_tokens=text_tokens)
        else:
            states = self._vision_forward(inputs)
            text_tokens = self._text_tokens(inputs, scoring_text=scoring_text)
        final_layer = self.vision_layers[-1]
        projected_layers: dict[int, list[torch.Tensor]] = {}
        native_final = self._last_native_final_features
        early_mode = _is_early_middle_mode(self.pruning_mode)
        if early_mode:
            selections = list(self._last_early_selections)
            keep_counts = [int(selection.indices.numel()) for selection in selections]
            if len(native_final) != len(counts) or len(selections) != len(counts):
                raise RuntimeError(
                    "middle_text_early did not return one compact visual stream per image view"
                )
            expected_dim = int(native_final[0].shape[-1]) if native_final else 0
            if expected_dim <= 0 or any(
                int(piece.shape[0]) != int(keep)
                or int(piece.shape[-1]) != expected_dim
                for piece, keep in zip(native_final, keep_counts)
            ):
                raise RuntimeError(
                    "middle_text_early final visual features do not match selected token counts"
                )
            projected_layers[final_layer] = list(native_final)
            requested = (self.pruning_layer,)
            layer_states: list[torch.Tensor] = []
            projected_chunks: list[torch.Tensor] = []
        else:
            requested = (
                (self.pruning_layer,)
                if self.pruning_mode == "middle_text"
                else self.vision_layers
            )
            use_native_final = len(native_final) == len(counts)
            if use_native_final:
                expected_dim = int(native_final[0].shape[-1]) if native_final else 0
                use_native_final = expected_dim > 0 and all(
                    int(piece.shape[0]) == int(count)
                    and int(piece.shape[-1]) == expected_dim
                    for piece, count in zip(native_final, counts)
                )
            if self.pruning_mode == "middle_text":
                # Qwen3-VL normally exposes the native final stream. Only the
                # selected middle state is projected for scoring. The explicit
                # final-state fallback is retained for older checkpoints.
                projection_layers = requested if use_native_final else (requested + (final_layer,))
                layer_states = [self._layer_state(states, layer) for layer in projection_layers]
                projection_states = layer_states
            else:
                layer_states = [self._layer_state(states, layer) for layer in requested]
                projection_states = layer_states[:-1] if use_native_final else layer_states
                projection_layers = requested[:-1] if use_native_final else requested
            state_lengths = [int(_flatten_tokens(state).shape[0]) for state in projection_states]
            merge_unit = max(1, merge_size * merge_size)
            projected_lengths = [
                length // merge_unit if length % merge_unit == 0 else -1
                for length in state_lengths
            ]
            batch_projected: torch.Tensor | None = None
            if self.pruning_mode == "middle_text":
                # The middle selector has only one requested scoring layer on
                # the normal Qwen3-VL path. Project each requested layer
                # exactly once instead of trying a concatenated projection and
                # then silently retrying per-layer if its shape does not match.
                projected_chunks = []
                for state, layer, expected_length in zip(
                    projection_states, projection_layers, projected_lengths
                ):
                    projected = _flatten_tokens(self._project_layer(state, layer, image_grid))
                    if int(projected.shape[0]) != int(expected_length):
                        raise RuntimeError(
                            f"projected ViT layer {layer} has {projected.shape[0]} tokens, "
                            f"expected {expected_length} after spatial merge"
                        )
                    projected_chunks.append(projected)
            elif projection_states and all(length > 0 for length in projected_lengths):
                try:
                    combined_states = torch.cat([_flatten_tokens(state) for state in projection_states], dim=0)
                    candidate = self._project_layer(combined_states, projection_layers[-1], image_grid)
                    candidate = _flatten_tokens(candidate)
                    if int(candidate.shape[0]) == sum(projected_lengths):
                        batch_projected = candidate
                except (TypeError, RuntimeError, ValueError):
                    batch_projected = None
            if use_native_final:
                projected_layers[final_layer] = list(native_final)
            if self.pruning_mode != "middle_text":
                if batch_projected is not None:
                    projected_chunks = list(batch_projected.split(projected_lengths, dim=0))
                else:
                    projected_chunks = [
                        _flatten_tokens(self._project_layer(state, layer, image_grid))
                        for state, layer in zip(projection_states, projection_layers)
                    ]
            for layer, projected in zip(projection_layers, projected_chunks):
                projected_layers[layer] = _split_tokens(projected, counts, layer=layer)
            required_layers = tuple(dict.fromkeys((*requested, final_layer)))
            for layer in required_layers:
                if layer not in projected_layers or not projected_layers[layer]:
                    raise RuntimeError(f"projected ViT layer {layer} produced no per-image features")
                if projected_layers[layer][0].shape[-1] != text_tokens.shape[-1]:
                    raise RuntimeError(
                        f"projected ViT layer {layer} cannot be mapped to text space: "
                        f"visual dim={projected_layers[layer][0].shape[-1]}, text dim={text_tokens.shape[-1]}"
                    )
        selections: list[SelectionResult]
        selected_features: list[torch.Tensor] = []
        if early_mode:
            # The custom ViT pass has already selected and compacted the
            # native final stream; do not score or select a second time.
            selections = list(self._last_early_selections)
        elif self.pruning_mode == "middle_text":
            # Match the simple test.py selector: one selected layer versus
            # one mean text vector, followed by direct top-K selection.
            selections = self._select_middle_text_tokens(
                projected_layers,
                text_tokens,
                counts,
            )
        else:
            # Normalize the text embedding matrix once, then score all three
            # layers and all image views in one similarity/top-k operation.
            normalized_text = F.normalize(text_tokens.float(), dim=-1)
            score_chunks: list[torch.Tensor] = []
            score_sizes: list[int] = []
            for image_index in range(len(counts)):
                for layer in requested:
                    chunk = projected_layers[layer][image_index]
                    score_chunks.append(chunk)
                    score_sizes.append(int(chunk.shape[0]))
            all_scores = topk_cosine_mean(
                torch.cat(score_chunks, dim=0),
                normalized_text,
                top_k=4,
                text_is_normalized=True,
            )
            score_parts = all_scores.split(score_sizes)
            cursor = 0
            q_by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in requested}
            for _image_index in range(len(counts)):
                for layer in requested:
                    q_by_layer[layer].append(percentile_rank(score_parts[cursor]))
                    cursor += 1

            selections = []
            for image_index, total in enumerate(counts):
                keep = self._keep_count(int(total))
                selection = select_budgeted_tokens(
                    q_by_layer[requested[0]][image_index],
                    q_by_layer[requested[1]][image_index],
                    q_by_layer[final_layer][image_index],
                    keep,
                )
                selections.append(selection)
            del normalized_text, score_chunks, all_scores, score_parts, q_by_layer

        if not early_mode:
            keep_counts = []
        for image_index, selection in enumerate(selections):
            keep = int(selection.indices.numel())
            if not early_mode:
                keep_counts.append(keep)
            # Selection is scored in the requested layer, but generation uses
            # the native final projected stream at the same patch indices.
            if early_mode:
                selected_features.append(projected_layers[final_layer][image_index])
            else:
                selected_features.append(projected_layers[final_layer][image_index][selection.indices])

        # Only final-layer features are needed by the LLM after selection;
        # release shallow/middle tensors before rebuilding the compact prompt.
        projected_layers = {final_layer: projected_layers[final_layer]}
        del states, text_tokens, layer_states, projected_chunks

        config_image_id = int(_get(self.config, "image_token_id", 151655))
        selected_replacements = [inputs["input_ids"].new_full((keep,), config_image_id) for keep in keep_counts]
        compact_ids = torch.stack([
            self._replace_groups(
                inputs["input_ids"][row],
                counts,
                selected_replacements,
                config_image_id,
            )
            for row in range(inputs["input_ids"].shape[0])
        ])
        compact_attention = None
        if inputs.get("attention_mask") is not None:
            compact_attention = torch.stack([
                self._replace_groups(
                    inputs["attention_mask"][row],
                    counts,
                    [inputs["attention_mask"].new_ones((keep,)) for keep in keep_counts],
                    config_image_id,
                    reference=inputs["input_ids"][row],
                )
                for row in range(inputs["attention_mask"].shape[0])
            ])
        compact_grid = self._compact_grid(image_grid, keep_counts, merge_size)
        model_inputs = dict(inputs)
        # Transformers' multimodal forward contract requires exactly one of
        # input_ids or inputs_embeds.  The compact ids are used to build the
        # selected encoder-output contract above; generation keeps these ids
        # so it can track the prompt length and decode only the continuation.
        model_inputs["input_ids"] = compact_ids
        model_inputs["image_grid_thw"] = compact_grid
        # Keep compact input_ids and the original pixel tensor.  The answer
        # path temporarily replaces the model's *native* get_image_features
        # method with the selected per-view feature tuple.  This follows the
        # reference test.py generation contract and avoids passing the custom
        # ``mm_encoder_outputs`` kwarg through older Transformers generate
        # validators that reject it before the first forward call.
        model_inputs.pop("inputs_embeds", None)
        if compact_attention is not None:
            model_inputs["attention_mask"] = compact_attention
        original_positions = self._original_position_ids(inputs, counts, merge_size)
        compact_positions = self._compact_positions(
            original_positions,
            inputs["input_ids"],
            counts,
            [selection.indices for selection in selections],
            config_image_id,
        )
        model_inputs["position_ids"] = compact_positions
        self._last_compact_rope_deltas = self._rope_delta(compact_positions, compact_ids)
        if "mm_token_type_ids" in inputs:
            model_inputs["mm_token_type_ids"] = torch.stack([
                self._replace_groups(
                    inputs["mm_token_type_ids"][row],
                    counts,
                    [inputs["mm_token_type_ids"].new_full((keep,), 1) for keep in keep_counts],
                    config_image_id,
                    reference=inputs["input_ids"][row],
                )
                for row in range(inputs["mm_token_type_ids"].shape[0])
            ])
        model_inputs.pop("pixel_values", None)
        model_inputs.pop("pixel_values_videos", None)
        # The native-feature replay needs pixel_values present so Qwen's
        # forward path calls get_image_features.  The temporary method ignores
        # the tensor, so this does not run a second ViT pass.
        if "pixel_values" in inputs:
            model_inputs["pixel_values"] = inputs["pixel_values"]
        self._last_selection = {
            "retained_tokens": sum(keep_counts),
            "original_tokens": sum(counts),
            "keep_counts": keep_counts,
            "mode": self.pruning_mode,
            "strategy": getattr(self, "pruning_strategy", "topk"),
            "layer": self.pruning_layer
            if self.pruning_mode in _MIDDLE_TEXT_MODES
            else None,
            "budgets": [selection.budget for selection in selections],
            "indices": [selection.indices.detach().cpu().tolist() for selection in selections],
            "groups": [
                {
                    "final": selection.final_indices.detach().cpu().tolist(),
                    "middle": selection.middle_indices.detach().cpu().tolist(),
                    "shallow": selection.shallow_indices.detach().cpu().tolist(),
                }
                for selection in selections
            ],
        }
        self._last_cached_visual = _SelectedVisionOutput(
            pooler_output=_FeatureTuple(selected_features),
            # Preserve native DeepStack streams when the checkpoint exposes
            # them.  They are sliced by the same original patch indices; no
            # extra network or LLM-side token recall is introduced.
            deepstack_features=[
                _FeatureTuple(
                    branch[image_index]
                    if early_mode
                    else branch[image_index][selections[image_index].indices]
                    for image_index in range(len(counts))
                )
                for branch in self._last_native_deepstack
            ],
        )
        return model_inputs, sum(keep_counts), sum(counts)

    @_inference_mode()
    def answer(
        self,
        image: Any,
        prompt: str,
        *,
        scoring_text: str | None = None,
    ) -> TrainingFreeModelOutput:
        try:
            inputs = self._inputs(image, prompt)
        except Exception as exc:
            raise RuntimeError(
                f"prompt/processor stage failed before visual scoring: {type(exc).__name__}: {exc}"
            ) from exc
        # Match QwenVL.answer: processor work is outside the measured scope;
        # timing starts before visual encoding/pruning and stops after the
        # generated answer has been decoded.
        start = time.perf_counter()
        try:
            pruning_started = time.perf_counter()
            model_inputs, image_tokens, original_image_tokens = self._pruned_inputs(
                inputs,
                scoring_text=scoring_text,
            )
            # The middle-layer ViT hook, projection, cosine scoring, and
            # compaction launch CUDA kernels asynchronously.  Synchronize at
            # this boundary so pruning_seconds contains all visual-selection
            # work and generation_seconds starts with a clean stream.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            pruning_latency = time.perf_counter() - pruning_started
        except Exception as exc:
            raise RuntimeError(
                f"ViT/projector/text-token/pruning stage failed: {type(exc).__name__}: {exc}"
            ) from exc
        input_tokens = int(model_inputs["input_ids"].shape[-1])
        prefill_started = time.perf_counter()
        prefill_holder: dict[str, float] = {}
        hook_handle = None
        register = getattr(self.llm, "register_forward_hook", None)
        if callable(register):
            def _capture_prefill(*_: Any, **__: Any) -> None:
                if "seconds" not in prefill_holder:
                    prefill_holder["seconds"] = time.perf_counter() - prefill_started

            hook_handle = register(_capture_prefill)
        # Replay the native Qwen multimodal path.  Transformers 4.57.x expects
        # get_image_features() to return ``(tuple(per_image_features),
        # deepstack_features)`` (or the equivalent structured output).  The
        # compact token sequence and the selected per-view final features are
        # therefore injected by a scoped method replacement, exactly where the
        # reference test.py calls model.generate().  This avoids the rejected
        # ``mm_encoder_outputs`` generate kwarg and keeps incremental decoding
        # on the model's own prepare_inputs_for_generation implementation.
        cached_visual = self._last_cached_visual if self.visual_pruning else None
        feature_owners: list[Any] = []
        for owner in (getattr(self.model, "model", None), self.model):
            if (
                owner is not None
                and callable(getattr(owner, "get_image_features", None))
                and all(owner is not existing for existing in feature_owners)
            ):
                feature_owners.append(owner)

        def _cached_features(*_: Any, **__: Any) -> _SelectedVisionOutput:
            if cached_visual is None:
                raise RuntimeError("selected visual features are unavailable for generation")
            return cached_visual

        def _compact_rope_index(*_: Any, **__: Any):
            # Native prepare_inputs_for_generation clears explicit position_ids
            # in Transformers 4.57.x. Supply the selected original coordinates
            # at the point where the multimodal prefill recomputes them, then
            # let native decoding reuse the compact sequence's rope delta.
            return model_inputs["position_ids"], self._last_compact_rope_deltas

        @contextmanager
        def _cached_feature_scope():
            if cached_visual is None or not feature_owners:
                yield
                return
            with ExitStack() as stack:
                for owner in feature_owners:
                    stack.enter_context(_temporary_method(owner, "get_image_features", _cached_features))
                    if _is_early_middle_mode(self.pruning_mode) and callable(
                        getattr(owner, "get_rope_index", None)
                    ):
                        stack.enter_context(
                            _temporary_method(owner, "get_rope_index", _compact_rope_index)
                        )
                yield

        # Older compatible checkpoints may not expose the native image-feature
        # hook.  Their baseline path accepts inputs_embeds, so use that exact
        # compact embedding route as a compatibility fallback.
        use_embeds_fallback = cached_visual is not None and not feature_owners
        if use_embeds_fallback:
            selected = torch.cat(cached_visual.pooler_output, dim=0)
            model_inputs["inputs_embeds"] = self._compose_inputs_embeds(
                model_inputs["input_ids"], selected
            )
            model_inputs.pop("pixel_values", None)
            model_inputs.pop("pixel_values_videos", None)

        # Keep only arguments accepted by the Qwen multimodal forward path.
        # Processor versions occasionally attach auxiliary tuple-valued
        # metadata; forwarding it through ``generate`` lets the generation
        # helper call ``.to()`` on that tuple before the model is entered.
        allowed_inputs = {
            "input_ids",
            "inputs_embeds",
            "attention_mask",
            "position_ids",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "mm_token_type_ids",
        }
        model_inputs = {
            key: value for key, value in model_inputs.items() if key in allowed_inputs
        }

        def _generate() -> Any:
            return self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=False,
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False,
            )

        generation_started = time.perf_counter()
        try:
            if cached_visual is not None and feature_owners:
                # Keep compact position_ids in the regular model input.  This
                # avoids returning a version-specific ``(position_ids,
                # rope_delta)`` tuple from a monkey-patched rope method;
                # Transformers' own generation code slices the tensor for
                # incremental decoding.
                with _cached_feature_scope():
                    generated = _generate()
            else:
                generated = _generate()
        except Exception as exc:
            input_summary = {
                key: (
                    f"{type(value).__name__}{tuple(value.shape)}"
                    if torch.is_tensor(value)
                    else type(value).__name__
                )
                for key, value in model_inputs.items()
            }
            raise RuntimeError(
                f"LLM generation stage failed: {type(exc).__name__}: {exc}; "
                f"model_inputs={input_summary}"
            ) from exc
        finally:
            if hook_handle is not None:
                hook_handle.remove()
            # Do not retain per-case CUDA feature tuples on the long-lived
            # runner.  The returned record contains CPU-only selection
            # metadata, while the next case will run a fresh ViT pass.
            self._last_cached_visual = None
            self._last_native_final_features = []
            self._last_native_deepstack = []
            self._last_early_selections = []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_started
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        if generated.ndim == 1:
            generated = generated.unsqueeze(0)
        prompt_ids = model_inputs["input_ids"]
        prompt_length = int(prompt_ids.shape[-1])
        # Decoder-only generation normally returns prompt + continuation, but
        # some versions return continuation only.  Compare the prefix instead
        # of relying only on sequence length, which can otherwise erase the
        # answer when the returned tensor has exactly the prompt length.
        has_prompt_prefix = (
            generated.ndim == 2
            and int(generated.shape[-1]) >= prompt_length
            and torch.equal(
                generated[:, :prompt_length],
                prompt_ids.to(device=generated.device, dtype=generated.dtype),
            )
        )
        new_ids = generated[:, prompt_length:] if has_prompt_prefix else generated
        decode_started = time.perf_counter()
        text = self.processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        text = _normalize_model_answer(text)
        output_tokens = int(new_ids.shape[-1]) if torch.is_tensor(new_ids) else int(new_ids[0].numel())
        if not text:
            raise RuntimeError(
                "the model returned no decodable answer tokens; "
                f"generated_shape={tuple(generated.shape)}, continuation_tokens={output_tokens}"
            )
        decode_seconds = time.perf_counter() - decode_started
        seconds = time.perf_counter() - start
        prefill_latency = float(prefill_holder.get("seconds", seconds))
        return TrainingFreeModelOutput(
            answer=text,
            seconds=seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            image_tokens=image_tokens,
            flops=estimate_flops(
                self.config, input_tokens, output_tokens, image_tokens,
                vision_tokens=original_image_tokens,
                vision_prune_layer=self.pruning_layer
                if self.visual_pruning and _is_early_middle_mode(self.pruning_mode)
                else None,
            ),
            original_image_tokens=original_image_tokens,
            pruned_image_tokens=image_tokens,
            retained_tokens=image_tokens,
            retention_ratio=(float(image_tokens) / float(original_image_tokens))
            if original_image_tokens
            else 0.0,
            pruning_latency=pruning_latency,
            prefill_latency=prefill_latency,
            pruning_seconds=pruning_latency,
            generation_seconds=generation_seconds,
            decode_seconds=decode_seconds,
            selection_budgets=list(self._last_selection.get("budgets", [])),
            selected_indices=list(self._last_selection.get("indices", [])),
            selection_groups=list(self._last_selection.get("groups", [])),
            selection_mode=str(self._last_selection.get("mode", self.pruning_mode)),
            selection_strategy=str(
                self._last_selection.get("strategy", getattr(self, "pruning_strategy", "topk"))
            ),
            selection_layer=self._last_selection.get("layer"),
            vision_layer_token_counts=list(self._last_vision_layer_token_counts)
            if self.visual_pruning and _is_early_middle_mode(self.pruning_mode)
            else [],
            vision_merge_unit=int(
                _get(_get(self.config, "vision_config", None), "spatial_merge_size", 1) or 1
            ) ** 2 if _is_early_middle_mode(self.pruning_mode) else 0,
        )


__all__ = [
    "SelectionResult",
    "TrainingFreeModelOutput",
    "TrainingFreeQwenVL",
    "percentile_rank",
    "select_budgeted_tokens",
    "topk_cosine_mean",
    "valid_text_token_mask",
]
