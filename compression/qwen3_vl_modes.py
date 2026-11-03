"""Native Qwen3-VL block modes used by ``testv.py``.

The implementation is deliberately outside ``evaluator.py``.  It reuses
``TrainingFreeQwenVL`` for loading, prompt preparation, text-cosine selection,
merger replay, DeepStack slicing, LLM slot replacement, and generation.  Only
the visual block loop is specialised here.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - metadata-only hosts
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from .readout_verified_bridge import compare_rows

if torch is not None:
    from model.training_free_pruning import TrainingFreeQwenVL as _TrainingFreeQwenVL
else:  # pragma: no cover - allows inspection without PyTorch
    class _TrainingFreeQwenVL:  # type: ignore[no-redef]
        pass


def _require_runtime() -> None:
    if torch is None or F is None:
        raise RuntimeError("Q-FlowCover visual modes require PyTorch")


def _sync(device: Any) -> None:
    if torch is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _split(value: torch.Tensor, counts: Sequence[int]) -> list[torch.Tensor]:
    return list(value.split([int(v) for v in counts], dim=0))


class Qwen3VLCompressionRunner(_TrainingFreeQwenVL):
    """TrainingFreeQwenVL subclass with verified hard/full/bridge block paths."""

    def __init__(
        self,
        *args: Any,
        compression_mode: str = "hard18",
        prune_method: str = "text_cos",
        decision_layer_human: int = 18,
        att_stats: str = "none",
        save_att_stats: bool = False,
        max_debug_samples: int = 0,
        q_flowcover_query_chunk: int = 256,
        q_flowcover_candidate_chunk: int = 64,
        q_flowcover_radius: int = 1,
        **kwargs: Any,
    ) -> None:
        _require_runtime()
        mode = str(compression_mode).strip().lower()
        if mode not in {"hard18", "full19_gather", "att_bridge19_fullkv"}:
            raise ValueError(f"unsupported compression_mode={compression_mode!r}")
        if int(decision_layer_human) <= 0:
            raise ValueError("decision_layer_human must be positive")
        stats_mode = str(att_stats).strip().lower()
        if stats_mode not in {"none", "summary", "debug"}:
            raise ValueError("att_stats must be none, summary, or debug")
        self.compression_mode = mode
        self.prune_method = str(prune_method).strip().lower()
        if self.prune_method not in {
            "text_cos",
            "flow_topk",
            "text_local_cover",
            "q_flowcover",
            "q_flowcover_global",
        }:
            raise ValueError(f"unsupported prune_method={prune_method!r}")
        self.decision_layer_human = int(decision_layer_human)
        self.att_stats_mode = stats_mode
        self.save_att_stats = bool(save_att_stats)
        self.max_debug_samples = int(max_debug_samples)
        self.q_flowcover_query_chunk = max(1, int(q_flowcover_query_chunk))
        self.q_flowcover_candidate_chunk = max(1, int(q_flowcover_candidate_chunk))
        self.q_flowcover_radius = max(0, int(q_flowcover_radius))
        kwargs.setdefault("visual_pruning", True)
        kwargs.setdefault("pruning_mode", "middle_text_early")
        kwargs.setdefault("pruning_layer", self.decision_layer_human)
        kwargs.setdefault("pruning_strategy", "topk")
        # TrainingFreeQwenVL performs the canonical checkpoint/processor load.
        _TrainingFreeQwenVL.__init__(self, *args, **kwargs)
        config_depth = int(_get(_get(self.config, "vision_config"), "depth", 0) or 0)
        blocks = getattr(self.vit, "blocks", None)
        if blocks is None:
            raise RuntimeError("loaded visual model exposes no blocks container")
        block_count = len(blocks)
        if config_depth and config_depth != block_count:
            raise RuntimeError(
                f"vision config depth={config_depth} but loaded blocks={block_count}; "
                "refusing to guess the human layer mapping"
            )
        self.decision_block_index = self.decision_layer_human - 1
        self.bridge_block_index = self.decision_block_index + 1
        if self.decision_block_index < 0 or self.bridge_block_index >= block_count:
            raise ValueError(
                f"decision layer {self.decision_layer_human} maps to block {self.decision_block_index}, "
                f"but loaded visual depth is {block_count}"
            )
        self._last_compression_metadata: dict[str, Any] = {}
        self._last_stage_timings: dict[str, float] = {}
        self._last_att_stats: dict[str, Any] = {}
        self._debug_samples_seen = 0
        self._last_q_flowcover_debug: list[dict[str, Any]] = []

    def stages(self) -> tuple[str, ...]:
        base = list(super().stages())
        if self.compression_mode == "full19_gather":
            base.insert(-2, "Full19Gather")
        elif self.compression_mode == "att_bridge19_fullkv":
            base.insert(-2, "AttentionBridge19FullKV")
        else:
            base.insert(-2, "Hard18")
        if self.prune_method != "text_cos":
            base.insert(-2, f"Q-FlowCover:{self.prune_method}")
        return tuple(base)

    @staticmethod
    def _raw_indices(
        selections: Sequence[Any], raw_counts: Sequence[int], merge_unit: int, device: Any
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        parts: list[torch.Tensor] = []
        global_parts: list[torch.Tensor] = []
        offset = 0
        for selection, raw_count in zip(selections, raw_counts):
            groups = selection.indices.to(device=device, dtype=torch.long)
            offsets = torch.arange(merge_unit, device=device, dtype=torch.long)
            local = (groups.unsqueeze(-1) * merge_unit + offsets).reshape(-1)
            # The rectangular bridge indexes within each image view, while
            # the compact suffix gathers from the concatenated raw sequence.
            parts.append(local)
            global_parts.append(local + offset)
            offset += int(raw_count)
        return parts, torch.cat(global_parts, dim=0)

    @staticmethod
    def _slice_deepstack(
        features: Sequence[torch.Tensor],
        counts: Sequence[int],
        selections: Sequence[Any],
    ) -> list[torch.Tensor]:
        pieces: list[torch.Tensor] = []
        for feature in features:
            views = _split(feature, counts)
            pieces.append(torch.cat([view.index_select(0, sel.indices) for view, sel in zip(views, selections)], dim=0))
        return pieces

    def _bridge_block_views(
        self,
        block: Any,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        view_counts: Sequence[int],
        keep_views: Sequence[torch.Tensor],
        *,
        att_stats: bool,
        source_mode: str = "all",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the native Qwen block with per-view rectangular attention."""

        attn = block.attn
        norm1 = block.norm1
        norm2 = block.norm2
        mlp = block.mlp
        heads = int(attn.num_heads)
        scale = float(getattr(attn, "scaling", (attn.head_dim ** -0.5)))
        q_parts: list[torch.Tensor] = []
        gamma_parts: list[torch.Tensor] = []
        centrality_parts: list[torch.Tensor] = []
        gap_parts: list[torch.Tensor] = []
        qkv_seconds = 0.0
        attention_seconds = 0.0
        mlp_seconds = 0.0
        cos, sin = position_embeddings
        cursor = 0
        source_mode = str(source_mode).strip().lower()
        if source_mode not in {"all", "selected"}:
            raise ValueError("source_mode must be all or selected")
        for count, keep in zip(view_counts, keep_views):
            count = int(count)
            local_keep = keep.to(device=hidden.device, dtype=torch.long)
            hidden_view = hidden[cursor : cursor + count]
            started = time.perf_counter()
            normalized = norm1(hidden_view)
            qkv = attn.qkv(normalized)
            qkv_seconds += time.perf_counter() - started
            q, k, v = (
                qkv.reshape(count, 3, heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

            # The native helper uses one position tensor for Q and K.  A
            # rectangular bridge has different Q/K lengths, so apply the
            # exact same RoPE formula once to each side with its own rows.
            q_selected = q.index_select(0, local_keep)
            q_rot, _ = apply_rotary_pos_emb_vision(
                q_selected,
                q_selected,
                cos[cursor : cursor + count].index_select(0, local_keep),
                sin[cursor : cursor + count].index_select(0, local_keep),
            )
            if source_mode == "selected":
                k = k.index_select(0, local_keep)
                v = v.index_select(0, local_keep)
                k_cos = cos[cursor : cursor + count].index_select(0, local_keep)
                k_sin = sin[cursor : cursor + count].index_select(0, local_keep)
            else:
                k_cos = cos[cursor : cursor + count]
                k_sin = sin[cursor : cursor + count]
            _, k_rot = apply_rotary_pos_emb_vision(k, k, k_cos, k_sin)
            q, k = q_rot, k_rot
            q_b = q.transpose(0, 1).unsqueeze(0)
            k_b = k.transpose(0, 1).unsqueeze(0)
            v_b = v.transpose(0, 1).unsqueeze(0)
            started = time.perf_counter()
            if att_stats:
                logits = torch.matmul(q_b, k_b.transpose(-1, -2)) * scale
                weights = torch.softmax(logits.float(), dim=-1).to(q_b.dtype)
                full_attention = torch.matmul(weights, v_b)
                if source_mode == "selected":
                    keep_k, keep_v = k_b, v_b
                else:
                    keep_k = k_b.index_select(-2, local_keep)
                    keep_v = v_b.index_select(-2, local_keep)
                keep_logits = torch.matmul(q_b, keep_k.transpose(-1, -2)) * scale
                keep_weights = torch.softmax(keep_logits.float(), dim=-1).to(q_b.dtype)
                hard_attention = torch.matmul(keep_weights, keep_v)
                if source_mode == "selected":
                    deleted = torch.zeros(k_b.shape[-2], dtype=torch.bool, device=hidden.device)
                else:
                    deleted = torch.ones(k_b.shape[-2], dtype=torch.bool, device=hidden.device)
                    deleted[local_keep] = False
                gamma_parts.append(weights[..., deleted].sum(dim=-1).flatten())
                centrality_parts.append(weights.mean(dim=(0, 1, 2)))
                gap_parts.append(
                    (full_attention - hard_attention).float().norm(dim=-1).flatten()
                    / (full_attention.float().norm(dim=-1).flatten() + 1e-8)
                )
                attended = full_attention
            else:
                attended = F.scaled_dot_product_attention(
                    q_b,
                    k_b,
                    v_b,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                )
            attention_seconds += time.perf_counter() - started
            attention_output = attended.transpose(1, 2).reshape(local_keep.numel(), -1).contiguous()
            projected = attn.proj(attention_output)
            residual = hidden_view.index_select(0, local_keep) + projected
            started = time.perf_counter()
            residual = residual + mlp(norm2(residual))
            mlp_seconds += time.perf_counter() - started
            q_parts.append(residual)
            cursor += count
        summary: dict[str, Any] = {
            "bridge_qkv_time": qkv_seconds,
            "bridge_attention_time": attention_seconds,
            "bridge_mlp_time": mlp_seconds,
        }
        if att_stats:
            gamma = torch.cat(gamma_parts) if gamma_parts else hidden.new_empty(0)
            gap = torch.cat(gap_parts) if gap_parts else hidden.new_empty(0)
            summary["Gamma"] = float(gamma.float().mean().detach().cpu()) if gamma.numel() else 0.0
            summary["readout_gap"] = float(gap.float().mean().detach().cpu()) if gap.numel() else 0.0
            # Keep a per-view centrality vector; this preserves image isolation
            # and lets the worker reduce it to source-group top-k metadata.
            summary["source_centrality"] = [item.detach().cpu() for item in centrality_parts]
        return torch.cat(q_parts, dim=0), summary

    def _vision_forward_middle_early(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, text_tokens: torch.Tensor):
        _require_runtime()
        if self.compression_mode == "hard18" and self.prune_method == "text_cos":
            self._last_compression_metadata["image_grid_thw"] = image_grid_thw.detach().reshape(-1, 3).cpu().tolist()
            result = super()._vision_forward_middle_early(pixel_values, image_grid_thw, text_tokens)
            self._last_compression_metadata.update(
                {"decision_block_index": self.decision_block_index, "compression_mode": self.compression_mode}
            )
            return result
        vit = self.vit
        required = ("patch_embed", "fast_pos_embed_interpolate", "rot_pos_emb", "blocks", "merger", "deepstack_merger_list")
        missing = [name for name in required if not hasattr(vit, name)]
        if missing:
            raise RuntimeError(f"Q-FlowCover requires Qwen3-VL visual internals; missing={missing}")
        merge_size = int(getattr(vit, "spatial_merge_size", 0) or _get(_get(self.config, "vision_config"), "spatial_merge_size", 1) or 1)
        merge_unit = merge_size * merge_size
        grid_rows = image_grid_thw.detach().reshape(-1, 3).cpu().tolist()
        if any(int(row[0]) != 1 for row in grid_rows):
            raise RuntimeError("Q-FlowCover currently requires T=1 image views")
        counts = [int(t) * (int(h) // merge_size) * (int(w) // merge_size) for t, h, w in grid_rows]
        raw_counts = [count * merge_unit for count in counts]
        raw_total = sum(raw_counts)
        grid_shapes = [(int(row[1]) // merge_size, int(row[2]) // merge_size) for row in grid_rows]
        stage: dict[str, float] = {}
        _sync(self.device)
        started = time.perf_counter()
        hidden = vit.patch_embed(pixel_values)
        hidden = hidden + vit.fast_pos_embed_interpolate(image_grid_thw)
        rotary = vit.rot_pos_emb(image_grid_thw)
        hidden = hidden.reshape(hidden.shape[0], -1)
        rotary = rotary.reshape(rotary.shape[0], -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(0, dtype=torch.int32)
        cu = F.pad(cu, (1, 0), value=0)
        _sync(self.device)
        stage["prefix"] = time.perf_counter() - started
        stage["visual_prefix"] = stage["prefix"]
        deepstack_indexes = tuple(int(value) for value in getattr(vit, "deepstack_visual_indexes", ()))
        deepstack: list[tuple[torch.Tensor, bool]] = []
        selections = None
        keep_raw_views = None
        keep_raw = None
        compact_counts = None
        layer_counts: list[int] = []
        bridge_stats: dict[str, Any] = {}
        q_flowcover_debug: list[dict[str, Any]] = []
        collect_att_stats = self.att_stats_mode != "none" and (
            self.max_debug_samples <= 0 or self._debug_samples_seen < self.max_debug_samples
        )
        if collect_att_stats:
            self._debug_samples_seen += 1
        suffix_started: float | None = None
        blocks = list(vit.blocks)
        for layer_num, block in enumerate(blocks):
            layer_counts.append(int(hidden.shape[0]))
            if layer_num < self.decision_block_index:
                hidden = block(hidden, cu_seqlens=cu, position_embeddings=position_embeddings)
                if layer_num in deepstack_indexes:
                    idx = deepstack_indexes.index(layer_num)
                    deepstack.append((vit.deepstack_merger_list[idx](hidden), False))
                continue
            if layer_num == self.decision_block_index:
                captured_rotated_qk: tuple[torch.Tensor, torch.Tensor] | None = None
                capture_flow_qk = self.prune_method in {
                    "flow_topk",
                    "q_flowcover",
                    "q_flowcover_global",
                }
                q_flow_selector = self.prune_method in {
                    "flow_topk",
                    "text_local_cover",
                    "q_flowcover",
                    "q_flowcover_global",
                }
                rotary_module = None
                original_rotary = None
                # Capture the Q/K produced by the native layer-18 RoPE helper.
                if capture_flow_qk:
                    try:
                        from transformers.models.qwen3_vl import modeling_qwen3_vl as rotary_module

                        original_rotary = rotary_module.apply_rotary_pos_emb_vision

                        def _capture_rotary(
                            query: torch.Tensor,
                            key: torch.Tensor,
                            cos: torch.Tensor,
                            sin: torch.Tensor,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
                            nonlocal captured_rotated_qk
                            captured_rotated_qk = original_rotary(query, key, cos, sin)
                            return captured_rotated_qk

                        rotary_module.apply_rotary_pos_emb_vision = _capture_rotary
                    except (ImportError, AttributeError) as exc:
                        raise RuntimeError(
                            "Q-FlowCover requires the native Qwen3-VL RoPE helper"
                        ) from exc
                try:
                    hidden = block(hidden, cu_seqlens=cu, position_embeddings=position_embeddings)
                finally:
                    if rotary_module is not None and original_rotary is not None:
                        rotary_module.apply_rotary_pos_emb_vision = original_rotary
                if layer_num in deepstack_indexes:
                    idx = deepstack_indexes.index(layer_num)
                    deepstack.append((vit.deepstack_merger_list[idx](hidden), False))
                _sync(self.device)
                score_started = time.perf_counter()
                projected = self._project_layer(hidden, self.pruning_layer, image_grid_thw)
                projected_views = _split(projected, counts)
                if self.prune_method == "text_cos":
                    selections = self._select_middle_text_tokens({self.pruning_layer: projected_views}, text_tokens, counts)
                elif q_flow_selector:
                    if capture_flow_qk and captured_rotated_qk is None:
                        raise RuntimeError("native block 17 did not expose RoPE-transformed Q/K")
                    if capture_flow_qk:
                        q_raw, k_raw = captured_rotated_qk
                        if q_raw.ndim != 3 or k_raw.ndim != 3:
                            raise RuntimeError(
                                "native RoPE Q/K must have shape [raw_tokens, heads, head_dim], "
                                f"got q={tuple(q_raw.shape)}, k={tuple(k_raw.shape)}"
                            )
                        q_views = list(q_raw.split(raw_counts, dim=0))
                        k_views = list(k_raw.split(raw_counts, dim=0))
                    else:
                        q_views = [None] * len(raw_counts)
                        k_views = [None] * len(raw_counts)
                    hidden_views = _split(hidden, raw_counts)
                    from .q_flowcover import select_q_flowcover_views

                    selections, q_flowcover_debug = select_q_flowcover_views(
                        projected_views,
                        text_tokens,
                        q_views,
                        k_views,
                        hidden_views,
                        [self._keep_count(int(count)) for count in counts],
                        raw_counts,
                        merge_unit,
                        self.prune_method,
                        grid_shapes,
                        scale=float(getattr(block.attn, "scaling", block.attn.head_dim ** -0.5)),
                        query_chunk_size=self.q_flowcover_query_chunk,
                        candidate_chunk_size=self.q_flowcover_candidate_chunk,
                        local_radius=self.q_flowcover_radius,
                    )
                else:
                    raise RuntimeError(f"unsupported selector {self.prune_method!r}")
                _sync(self.device)
                stage["selector"] = time.perf_counter() - score_started
                if q_flowcover_debug:
                    stage["selector_flops"] = float(
                        sum(int(row.get("selector_flops", 0)) for row in q_flowcover_debug)
                    )
                # The selector has consumed the native RoPE Q/K.  Drop those
                # full-length views before compact suffix blocks allocate new
                # activations; no second Q/K projection is performed.
                if q_flow_selector:
                    captured_rotated_qk = None
                    q_views = None
                    k_views = None
                    hidden_views = None
                projected = None
                projected_views = None
                index_started = time.perf_counter()
                keep_counts = [int(sel.indices.numel()) for sel in selections]
                compact_counts = keep_counts
                keep_raw_views, keep_raw = self._raw_indices(selections, raw_counts, merge_unit, hidden.device)
                _sync(self.device)
                stage["index_expand"] = time.perf_counter() - index_started
                if self.compression_mode == "hard18":
                    hidden = hidden.index_select(0, keep_raw)
                    compacted_deepstack = []
                    for feature, already_compact in deepstack:
                        if already_compact:
                            compacted_deepstack.append((feature, True))
                        else:
                            compacted_deepstack.append((torch.cat([view.index_select(0, sel.indices) for view, sel in zip(_split(feature, counts), selections)], dim=0), True))
                    deepstack = compacted_deepstack
                    position_embeddings = tuple(value.index_select(0, keep_raw) for value in position_embeddings)
                    compact_raw_counts = [count * merge_unit for count in compact_counts]
                    cu = F.pad(torch.tensor(compact_raw_counts, dtype=cu.dtype, device=cu.device).cumsum(0, dtype=cu.dtype), (1, 0), value=0)
                continue
            if selections is None or keep_raw is None or keep_raw_views is None:
                raise RuntimeError("decision layer did not produce keep indices")
            if layer_num == self.bridge_block_index:
                if self.compression_mode == "hard18":
                    hidden = block(hidden, cu_seqlens=cu, position_embeddings=position_embeddings)
                    if layer_num in deepstack_indexes:
                        idx = deepstack_indexes.index(layer_num)
                        deepstack.append((vit.deepstack_merger_list[idx](hidden), True))
                    if suffix_started is None:
                        suffix_started = time.perf_counter()
                    continue
                if self.compression_mode == "full19_gather":
                    full_started = time.perf_counter()
                    full_hidden = block(hidden, cu_seqlens=cu, position_embeddings=position_embeddings)
                    stage["full19_block"] = time.perf_counter() - full_started
                    if layer_num in deepstack_indexes:
                        idx = deepstack_indexes.index(layer_num)
                        deepstack.append((vit.deepstack_merger_list[idx](full_hidden), False))
                    hidden = full_hidden.index_select(0, keep_raw)
                else:
                    bridge_started = time.perf_counter()
                    bridge_input = hidden
                    hidden, bridge_stats = self._bridge_block_views(
                        block,
                        hidden,
                        position_embeddings,
                        raw_counts,
                        keep_raw_views,
                        att_stats=collect_att_stats,
                    )
                    stage["bridge"] = time.perf_counter() - bridge_started
                    if layer_num in deepstack_indexes:
                        idx = deepstack_indexes.index(layer_num)
                        compact_deepstack = vit.deepstack_merger_list[idx](hidden)
                        if self.att_stats_mode == "debug":
                            debug_started = time.perf_counter()
                            full_bridge = block(bridge_input, cu_seqlens=cu, position_embeddings=position_embeddings)
                            full_deepstack = vit.deepstack_merger_list[idx](full_bridge)
                            # DeepStack mergers return merged group rows, so
                            # gather with the same per-view group selections.
                            selected_full = self._slice_deepstack(
                                [full_deepstack], counts, selections
                            )[0]
                            bridge_stats["deepstack_full_vs_bridge"] = compare_rows(selected_full, compact_deepstack)
                            bridge_stats["deepstack_debug_compare_time"] = time.perf_counter() - debug_started
                        deepstack.append((compact_deepstack, True))
                    bridge_stats["bridge_block_index"] = layer_num
                # All pre-bridge DeepStack streams were full length and must be
                # sliced by exactly the same raw merge-group selection.
                compacted_deepstack: list[tuple[torch.Tensor, bool]] = []
                for feature, already_compact in deepstack:
                    if already_compact:
                        compacted_deepstack.append((feature, True))
                    else:
                        compacted_deepstack.append((torch.cat([view.index_select(0, sel.indices) for view, sel in zip(_split(feature, counts), selections)], dim=0), True))
                deepstack = compacted_deepstack
                position_embeddings = tuple(value.index_select(0, keep_raw) for value in position_embeddings)
                compact_raw_counts = [count * merge_unit for count in compact_counts]
                cu = F.pad(torch.tensor(compact_raw_counts, dtype=cu.dtype, device=cu.device).cumsum(0, dtype=cu.dtype), (1, 0), value=0)
                continue
            if suffix_started is None:
                suffix_started = time.perf_counter()
            hidden = block(hidden, cu_seqlens=cu, position_embeddings=position_embeddings)
            if layer_num in deepstack_indexes:
                idx = deepstack_indexes.index(layer_num)
                deepstack.append((vit.deepstack_merger_list[idx](hidden), True))
        if selections is None or compact_counts is None:
            raise RuntimeError("Q-FlowCover did not reach the decision block")
        merge_started = time.perf_counter()
        if suffix_started is not None:
            _sync(self.device)
            stage["visual_suffix"] = time.perf_counter() - suffix_started
        _sync(self.device)
        hidden = vit.merger(hidden)
        _sync(self.device)
        stage["merger"] = time.perf_counter() - merge_started
        self._last_native_final_features = list(hidden.split(compact_counts, dim=0))
        self._last_native_deepstack = [list(feature.split(compact_counts, dim=0)) for feature, _ in deepstack]
        self._last_early_selections = list(selections)
        self._last_early_original_counts = list(counts)
        self._last_vision_layer_token_counts = layer_counts
        self._last_compression_metadata = {
            "compression_mode": self.compression_mode,
            "prune_method": self.prune_method,
            "decision_layer_human": self.decision_layer_human,
            "decision_block_index": self.decision_block_index,
            "bridge_block_index": self.bridge_block_index,
            "raw_patch_count": raw_total,
            "merged_group_count": sum(counts),
            "retained_group_count": sum(compact_counts),
            "actual_llm_visual_tokens": sum(compact_counts),
            "keep_idx": [sel.indices.detach().cpu().tolist() for sel in selections],
            "image_grid_thw": [list(row) for row in grid_rows],
            "layer_token_counts": list(layer_counts),
            "raw_keep_idx_hash": hashlib.sha256(
                json.dumps(keep_raw.detach().cpu().tolist(), separators=(",", ":")).encode()
            ).hexdigest(),
        }
        if q_flowcover_debug:
            self._last_compression_metadata["q_flowcover"] = q_flowcover_debug
            self._last_compression_metadata["text_topk"] = [
                row.get("text_topk", []) for row in q_flowcover_debug
            ]
            self._last_compression_metadata["final_group_index"] = [
                row.get("final_group_index", []) for row in q_flowcover_debug
            ]
            self._last_compression_metadata["replacement_group_count"] = sum(
                int(row.get("replacement_group_count", 0)) for row in q_flowcover_debug
            )
        self._last_stage_timings = stage
        if bridge_stats:
            self._last_stage_timings.update(
                {
                    "bridge_qkv": bridge_stats.get("bridge_qkv_time", 0.0),
                    "bridge_attention": bridge_stats.get("bridge_attention_time", 0.0),
                    "bridge_mlp": bridge_stats.get("bridge_mlp_time", 0.0),
                }
            )
        self._last_att_stats = bridge_stats
        self._last_q_flowcover_debug = q_flowcover_debug
        return {}

    def answer(self, *args: Any, **kwargs: Any):
        output = super().answer(*args, **kwargs)
        for key, value in self._last_compression_metadata.items():
            setattr(output, key, value)
        output.stage_timings = dict(self._last_stage_timings)
        output.att_stats = dict(self._last_att_stats)
        output.q_flowcover_debug = list(self._last_q_flowcover_debug)
        output.selector_flops = float(
            sum(int(row.get("selector_flops", 0)) for row in self._last_q_flowcover_debug)
        )
        if output.selector_flops:
            output.flops = int(output.flops) + int(output.selector_flops)
        output.peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if getattr(self.device, "type", None) == "cuda"
            else None
        )
        return output


__all__ = ["Qwen3VLCompressionRunner"]
