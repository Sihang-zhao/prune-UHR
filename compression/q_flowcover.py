"""Q-FlowCover group selectors for the Qwen3-VL layer-18 cut.

The implementation is deliberately independent of the canonical evaluator.  It
operates on one image view at a time and only keeps query chunks and a vector
of source-group support.  In particular, it never materializes the raw
``N x N`` attention or a global ``N x N`` cosine matrix.
"""
from __future__ import annotations

import math
import time
from typing import Any, Sequence

try:  # Keep ``testv.py --help`` importable on metadata-only hosts.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from model.training_free_pruning import SelectionResult
except Exception:  # pragma: no cover
    from dataclasses import dataclass

    @dataclass
    class SelectionResult:  # type: ignore[no-redef]
        indices: Any
        final_indices: Any
        middle_indices: Any
        shallow_indices: Any
        budget: tuple[int, int, int]


Q_FLOWCOVER_METHODS = {
    "flow_topk",
    "text_local_cover",
    "q_flowcover",
    "q_flowcover_global",
}


def _require() -> None:
    if torch is None or F is None:
        raise RuntimeError("Q-FlowCover requires PyTorch")


def _stable_topk(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return the same sorted-index Top-K contract as the baseline selector."""

    if int(k) <= 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    return torch.topk(values, k=int(k), largest=True, sorted=False).indices.sort().values


def _text_cosine(projected: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
    """Match ``TrainingFreeQwenVL._middle_text_score_views`` exactly."""

    text_vector = F.normalize(text_tokens.float().mean(dim=0), dim=0)
    return F.cosine_similarity(projected.float(), text_vector.unsqueeze(0), dim=-1)


def _group_source_support(
    q: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
    pi: torch.Tensor,
    merge_unit: int,
    *,
    scale: float,
    query_chunk_size: int,
) -> tuple[torch.Tensor, int]:
    """Compute ``support[j]`` from P queries using chunked raw attention.

    ``q`` and ``k`` are the *already RoPE-transformed* native tensors with
    shape ``[raw_tokens, heads, head_dim]``.  Each P group contributes the
    average of its ``merge_unit`` raw queries, and source keys are summed into
    groups.  Therefore each row of the implicit group attention sums to one and
    ``support`` is a probability vector up to floating point error.
    """

    _require()
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"expected q/k=[N,H,D], got {tuple(q.shape)}, {tuple(k.shape)}")
    n, heads, _ = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]))
    if int(k.shape[0]) != n or int(k.shape[1]) != heads:
        raise ValueError("q and k raw lengths/head counts must match")
    unit = int(merge_unit)
    if unit <= 0 or n % unit:
        raise ValueError(f"raw token count {n} is not divisible by merge_unit={unit}")
    groups = n // unit
    p = p.to(device=q.device, dtype=torch.long)
    if p.ndim != 1 or int(p.numel()) == 0:
        raise ValueError("P must contain at least one group index")
    if bool((p < 0).any()) or bool((p >= groups).any()) or int(torch.unique(p).numel()) != int(p.numel()):
        raise ValueError("P contains invalid or duplicate group indices")
    if pi.shape != (groups,):
        raise ValueError(f"pi must have shape [{groups}], got {tuple(pi.shape)}")

    # Native Qwen raw order is contiguous within each merged 2x2 group.
    raw_queries = (p.unsqueeze(1) * unit + torch.arange(unit, device=q.device)).reshape(-1)
    raw_weights = pi.index_select(0, p).repeat_interleave(unit) / float(unit)
    support = torch.zeros(groups, dtype=torch.float32, device=q.device)
    q_heads = q.transpose(0, 1).contiguous()  # [heads, N, D]
    k_heads = k.transpose(0, 1).contiguous()
    chunk = max(1, int(query_chunk_size))
    for start in range(0, int(raw_queries.numel()), chunk):
        local = raw_queries[start : start + chunk]
        # Flattening the leading singleton used by ``torch.matmul`` into an
        # explicit batched matmul avoids an extra broadcasted view and maps to
        # one efficient H-way GEMM on CUDA: [H,C,D] @ [H,D,N] -> [H,C,N].
        query = q_heads.index_select(1, local)
        logits = torch.bmm(query, k_heads.transpose(1, 2)) * float(scale)
        # Match the native Qwen attention path: normalize in fp32 for stable
        # bf16/fp16 inference, then cast the probabilities back to the query
        # dtype before reducing heads.  The temporary is [heads, chunk, N],
        # never an N x N attention matrix, and the cast substantially lowers
        # bandwidth/peak memory for bf16 checkpoints.
        attention = torch.softmax(logits.float(), dim=-1).to(q.dtype).mean(dim=0)
        grouped = attention.reshape(int(local.numel()), groups, unit).sum(dim=-1)
        support += (grouped * raw_weights[start : start + int(local.numel())].unsqueeze(1)).sum(dim=0)
    # Casting probabilities back to bf16/fp16 (as the native attention path
    # does) introduces a small row-sum drift.  Restore the probability-mass
    # invariant without changing fp32 results or adding another attention pass.
    if q.dtype != torch.float32:
        support = support / support.sum().clamp_min(torch.finfo(support.dtype).tiny)
    return support, int(raw_queries.numel())


def _group_hidden(hidden: torch.Tensor, merge_unit: int) -> torch.Tensor:
    if hidden.ndim != 2:
        hidden = hidden.reshape(-1, int(hidden.shape[-1]))
    n = int(hidden.shape[0])
    unit = int(merge_unit)
    if unit <= 0 or n % unit:
        raise ValueError(f"hidden token count {n} is not divisible by merge_unit={unit}")
    # Preserve the native merger's within-group ordering.  Flattening each
    # complete raw group keeps the UHR descriptor sensitive to the 2x2 layout
    # without adding a learned projection or a second ViT pass.
    z = hidden.reshape(n // unit, unit * int(hidden.shape[-1])).float()
    return F.normalize(z, dim=-1)


def _neighbor_indices(index: int, height: int, width: int, radius: int) -> torch.Tensor:
    row, col = divmod(int(index), int(width))
    values = [
        rr * int(width) + cc
        for rr in range(max(0, row - radius), min(int(height), row + radius + 1))
        for cc in range(max(0, col - radius), min(int(width), col + radius + 1))
        if max(abs(rr - row), abs(cc - col)) <= int(radius)
    ]
    return torch.tensor(values, dtype=torch.long)


def _local_cover(
    z: torch.Tensor,
    omega: torch.Tensor,
    keep: int,
    height: int,
    width: int,
    radius: int,
) -> tuple[torch.Tensor, int]:
    """Greedy weighted max coverage over local cosine edges only."""

    groups = int(z.shape[0])
    if height * width != groups:
        raise ValueError(f"grid ({height},{width}) does not match group count {groups}")
    neighbor_rows = [_neighbor_indices(i, height, width, radius) for i in range(groups)]
    degree = max(int(row.numel()) for row in neighbor_rows)
    # Build the compact spatial graph on CPU and transfer it once.  Filling a
    # CUDA tensor row by row launches 2*G tiny kernels and can dominate the
    # selector for large patch grids.
    neighbor_table_cpu = torch.zeros((groups, degree), dtype=torch.long)
    neighbor_valid_cpu = torch.zeros((groups, degree), dtype=torch.bool)
    for index, row in enumerate(neighbor_rows):
        count = int(row.numel())
        neighbor_table_cpu[index, :count] = row
        neighbor_valid_cpu[index, :count] = True
    neighbor_table = neighbor_table_cpu.to(z.device)
    neighbor_valid = neighbor_valid_cpu.to(z.device)
    # Compute the sparse kappa edges once.  This stores only E=G*degree
    # scalars, never a dense G x G similarity matrix; greedy iterations then
    # update coverage by indexing these local edges.
    edge_similarity = torch.zeros((groups, degree), dtype=torch.float32, device=z.device)
    candidate_chunk = 256
    for start in range(0, groups, candidate_chunk):
        local = torch.arange(start, min(groups, start + candidate_chunk), device=z.device)
        source = neighbor_table.index_select(0, local)
        source_hidden = z.index_select(0, source.reshape(-1)).reshape(source.shape[0], source.shape[1], -1)
        edge_similarity[local] = (source_hidden * z.index_select(0, local).unsqueeze(1)).sum(dim=-1).clamp_min(0.0)
    selected = torch.zeros(groups, dtype=torch.bool, device=z.device)
    coverage = torch.zeros(groups, dtype=torch.float32, device=z.device)
    # Keep the selected indices on-device.  Converting the argmax to a Python
    # scalar at every greedy step forces a CUDA synchronization and was a
    # significant part of the runtime for high-retention settings.
    chosen: list[torch.Tensor] = []
    edge_evals = groups * degree
    for _ in range(int(keep)):
        candidates = torch.where(~selected)[0]
        # All local candidates share the same compact E=G*degree table.  A
        # single vectorized gain evaluation avoids a Python loop over chunks
        # for every greedy step while retaining sparse O(E) work and memory;
        # no GxG similarity matrix is constructed.  ``candidates`` is sorted
        # by torch.where, so torch.max's first-index tie rule matches the
        # previous explicit ``candidate < best`` tie break.
        source = neighbor_table.index_select(0, candidates)
        valid = neighbor_valid.index_select(0, candidates)
        similarity = edge_similarity.index_select(0, candidates)
        current = coverage.index_select(0, source.reshape(-1)).reshape(source.shape)
        gain_values = omega.index_select(0, source.reshape(-1)).reshape(source.shape)
        gains = (gain_values * torch.relu(similarity - current).masked_fill(~valid, 0.0)).sum(dim=1)
        if int(candidates.numel()) == 0:
            raise RuntimeError("local Q-FlowCover ran out of candidates before K")
        # candidates is sorted, so argmax's first occurrence preserves the
        # baseline's lowest-index tie break without a host-side comparison.
        best = candidates[torch.argmax(gains)]
        selected[best] = True
        chosen.append(best)
        source = neighbor_table[best][neighbor_valid[best]]
        similarity = edge_similarity[best][neighbor_valid[best]]
        coverage[source] = torch.maximum(coverage[source], similarity)
    return torch.stack(chosen).sort().values, int(edge_evals)


def _global_cover(
    z: torch.Tensor,
    omega: torch.Tensor,
    keep: int,
    *,
    candidate_chunk_size: int,
) -> tuple[torch.Tensor, int]:
    """Greedy global cover without materializing an N x N similarity matrix."""

    groups = int(z.shape[0])
    selected = torch.zeros(groups, dtype=torch.bool, device=z.device)
    coverage = torch.zeros(groups, dtype=torch.float32, device=z.device)
    # As in the local selector, retain argmax results as scalar CUDA tensors
    # until the final sort instead of synchronizing once per selected group.
    chosen: list[torch.Tensor] = []
    pair_work = 0
    for _ in range(int(keep)):
        candidates = torch.where(~selected)[0]
        chunk = max(1, int(candidate_chunk_size))
        best_score = torch.tensor(-float("inf"), dtype=omega.dtype, device=z.device)
        best_candidate = torch.zeros((), dtype=torch.long, device=z.device)
        for start in range(0, int(candidates.numel()), chunk):
            local = candidates[start : start + chunk]
            similarity = (z @ z.index_select(0, local).transpose(0, 1)).clamp_min(0.0)
            gains = (omega.unsqueeze(1) * torch.relu(similarity - coverage.unsqueeze(1))).sum(dim=0)
            pair_work += groups * int(local.numel())
            value, position = torch.max(gains, dim=0)
            candidate = local[position]
            better = (value > best_score) | ((value == best_score) & (candidate < best_candidate))
            best_score = torch.where(better, value, best_score)
            best_candidate = torch.where(better, candidate, best_candidate)
        if int(candidates.numel()) == 0:
            raise RuntimeError("global Q-FlowCover ran out of candidates before K")
        best = best_candidate
        selected[best] = True
        chosen.append(best)
        coverage = torch.maximum(coverage, (z @ z[best]).clamp_min(0.0))
    return torch.stack(chosen).sort().values, pair_work


def _select_one_view(
    projected: torch.Tensor,
    text_tokens: torch.Tensor,
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    hidden: torch.Tensor,
    keep: int,
    raw_count: int,
    merge_unit: int,
    grid: tuple[int, int],
    method: str,
    *,
    scale: float,
    query_chunk_size: int,
    candidate_chunk_size: int,
    local_radius: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _require()
    method = str(method).strip().lower()
    if method not in Q_FLOWCOVER_METHODS:
        raise ValueError(f"unsupported Q-FlowCover method {method!r}")
    groups = int(projected.shape[0])
    if int(raw_count) != groups * int(merge_unit):
        raise ValueError("raw_count/projected group count/merge_unit are inconsistent")
    if int(keep) <= 0 or int(keep) > groups:
        raise ValueError(f"keep={keep} must be in [1,{groups}]")
    height, width = (int(grid[0]), int(grid[1]))
    if height <= 0 or width <= 0 or height * width != groups:
        raise ValueError(f"grid ({height},{width}) does not match group count {groups}")

    r = _text_cosine(projected, text_tokens).float()
    p = _stable_topk(r, int(keep))
    pi = torch.zeros(groups, dtype=torch.float32, device=projected.device)
    pi[p] = torch.softmax(r.index_select(0, p), dim=0)
    support = torch.zeros_like(pi)
    source_query_count = 0
    need_flow = method in {"flow_topk", "q_flowcover", "q_flowcover_global"}
    if need_flow:
        if q is None or k is None:
            raise ValueError(f"{method} requires native RoPE-transformed Q/K")
        support, source_query_count = _group_source_support(
            q,
            k,
            p,
            pi,
            int(merge_unit),
            scale=float(scale),
            query_chunk_size=int(query_chunk_size),
        )
    omega = pi if method == "text_local_cover" else 0.5 * (pi + support)

    if method == "flow_topk" or (
        int(local_radius) == 0 and method in {"q_flowcover", "text_local_cover"}
    ):
        # R=0 means kappa[j,i] is one only for j=i. This is exactly omega
        # Top-K, with the same deterministic ordering as the baseline.
        selected = _stable_topk(omega, int(keep))
        edge_work = 0
    elif method == "q_flowcover_global":
        selected, edge_work = _global_cover(
            _group_hidden(hidden, int(merge_unit)),
            omega,
            int(keep),
            candidate_chunk_size=int(candidate_chunk_size),
        )
    else:
        selected, edge_work = _local_cover(
            _group_hidden(hidden, int(merge_unit)),
            omega,
            int(keep),
            height,
            width,
            int(local_radius),
        )
    if int(selected.numel()) != int(keep) or int(torch.unique(selected).numel()) != int(keep):
        raise RuntimeError("Q-FlowCover did not return exactly K unique groups")
    diagnostics = {
        "method": method,
        "N": groups,
        "K": int(keep),
        "text_topk": p.detach().cpu().tolist(),
        "final_group_index": selected.detach().cpu().tolist(),
        "pi": pi.detach().cpu().tolist(),
        "support": support.detach().cpu().tolist(),
        "omega": omega.detach().cpu().tolist(),
        "grid": [height, width],
        "merge_unit": int(merge_unit),
        "raw_count": int(raw_count),
        "source_query_count": int(source_query_count),
        "edge_work": int(edge_work),
        "local_radius": int(local_radius),
        "text_topk_overlap": float(
            torch.isin(selected, p).float().mean().detach().cpu().item()
        ),
        "replacement_group_count": int((~torch.isin(p, selected)).sum().detach().cpu().item()),
    }
    # FLOPs are an accounting estimate for the selector itself.  They are
    # reported separately and then added to the runner's end-to-end estimate;
    # temporary chunk tensors are intentionally excluded from the estimate.
    selector_flops = 2 * int(projected.shape[-1]) * groups
    if need_flow and q is not None:
        selector_flops += 2 * int(q.shape[1]) * int(q.shape[2]) * int(raw_count) * int(source_query_count)
    group_width = int(merge_unit) * int(hidden.shape[-1])
    selector_flops += 2 * group_width * int(edge_work)
    diagnostics["selector_flops"] = int(selector_flops)
    return selected, diagnostics


def select_q_flowcover_views(
    projected_views: Sequence[torch.Tensor],
    text_tokens: torch.Tensor,
    q_views: Sequence[torch.Tensor | None],
    k_views: Sequence[torch.Tensor | None],
    hidden_views: Sequence[torch.Tensor],
    keep_counts: Sequence[int],
    raw_counts: Sequence[int],
    merge_unit: int,
    method: str,
    grid_shapes: Sequence[tuple[int, int]],
    *,
    scale: float,
    query_chunk_size: int = 256,
    candidate_chunk_size: int = 64,
    local_radius: int = 1,
) -> tuple[list[SelectionResult], list[dict[str, Any]]]:
    """Select independently per image/view and return baseline-compatible results."""

    _require()
    if not (
        len(projected_views)
        == len(q_views)
        == len(k_views)
        == len(hidden_views)
        == len(keep_counts)
        == len(raw_counts)
        == len(grid_shapes)
    ):
        raise ValueError("Q-FlowCover received mismatched per-view inputs")
    if int(local_radius) < 0:
        raise ValueError("local_radius must be non-negative")
    results: list[SelectionResult] = []
    diagnostics: list[dict[str, Any]] = []
    for projected, q, k, hidden, keep, raw_count, grid in zip(
        projected_views,
        q_views,
        k_views,
        hidden_views,
        keep_counts,
        raw_counts,
        grid_shapes,
    ):
        started = time.perf_counter()
        selected, debug = _select_one_view(
            projected,
            text_tokens,
            q,
            k,
            hidden,
            int(keep),
            int(raw_count),
            int(merge_unit),
            grid,
            method,
            scale=float(scale),
            query_chunk_size=int(query_chunk_size),
            candidate_chunk_size=int(candidate_chunk_size),
            local_radius=int(local_radius),
        )
        debug["selector_seconds"] = time.perf_counter() - started
        empty = torch.empty(0, dtype=torch.long, device=selected.device)
        # Layer 18 is the middle role in the existing 6,18,24 metadata layout.
        results.append(
            SelectionResult(
                indices=selected,
                final_indices=empty,
                middle_indices=selected,
                shallow_indices=empty,
                budget=(0, int(selected.numel()), 0),
            )
        )
        diagnostics.append(debug)
    return results, diagnostics


__all__ = ["Q_FLOWCOVER_METHODS", "select_q_flowcover_views"]
