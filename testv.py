#!/usr/bin/env python3
"""Portable Q-FlowCover evaluator, extracted from the original testv entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluator


def _parser() -> argparse.ArgumentParser:
    parser = evaluator.build_parser()
    parser.add_argument(
        "--prune_method", "--prune-method",
        choices=("text_cos", "flow_topk", "text_local_cover", "q_flowcover", "q_flowcover_global"),
        default="q_flowcover", help="Visual group selector; text_cos reproduces the baseline",
    )
    parser.add_argument(
        "--exec_mode", "--exec-mode",
        choices=("hard18", "full19_gather", "att_bridge19_fullkv"),
        default="hard18", help="Execution of the visual blocks after the layer-18 decision",
    )
    parser.add_argument("--prune_fraction", "--prune-fraction", type=float, default=None,
                        help="Fraction removed, in [0,1); alternative to --visual-prune-percent")
    parser.add_argument("--qflow_radius", "--qflow-radius", type=int, default=1)
    parser.add_argument("--qflow_query_chunk", "--qflow-query-chunk", type=int, default=256)
    parser.add_argument("--qflow_candidate_chunk", "--qflow-candidate-chunk", type=int, default=64)
    parser.add_argument("--save_per_sample", "--save-per-sample",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Include additional per-sample index diagnostics")
    parser.add_argument("--att_stats", "--att-stats", choices=("none", "summary", "debug"), default="none")
    parser.add_argument("--save_att_stats", "--save-att-stats", action="store_true")
    parser.add_argument("--max_debug_samples", "--max-debug-samples", type=int, default=0)
    parser.add_argument("--decision_layer_human", "--decision-layer-human", type=int, choices=(18,), default=18,
                        help="Fixed decision layer; 18 corresponds to visual block index 17")
    return parser


def resolve_decision_block(model: str) -> dict[str, Any]:
    """Validate the fixed layer boundary when a local checkpoint is supplied."""
    model_path = Path(model).expanduser()
    config_path = model_path / "config.json" if model_path.is_dir() else None
    details: dict[str, Any] = {
        "decision_layer_human": 18,
        "decision_block_index": 17,
        "config_path": str(config_path) if config_path else None,
        "config_verified": False,
    }
    if config_path is not None and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        vision = config.get("vision_config", config)
        depth = int(vision.get("depth", vision.get("num_hidden_layers", 0)) or 0)
        if depth <= 18:
            raise ValueError(f"layer 18 must precede the final visual block; checkpoint depth={depth}")
        details.update(
            config_verified=True,
            vision_depth=depth,
            deepstack_visual_indexes=list(vision.get("deepstack_visual_indexes", [])),
            spatial_merge_size=int(vision.get("spatial_merge_size", 1)),
        )
    return details


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.qflow_radius < 0:
        parser.error("--qflow-radius must be non-negative")
    if args.qflow_query_chunk <= 0 or args.qflow_candidate_chunk <= 0:
        parser.error("Q-FlowCover chunk sizes must be positive")
    if args.prune_fraction is not None:
        if not 0.0 <= args.prune_fraction < 1.0:
            parser.error("--prune-fraction must be in [0,1)")
        if args.visual_prune_percent is not None:
            parser.error("--prune-fraction cannot be combined with --visual-prune-percent")
        args.visual_prune_percent = args.prune_fraction * 100.0
    if args.att_stats != "none" and args.exec_mode != "att_bridge19_fullkv":
        parser.error("--att-stats requires --exec-mode att_bridge19_fullkv")
    if args.save_att_stats and args.att_stats == "none":
        parser.error("--save-att-stats requires --att-stats summary or debug")
    if args.max_debug_samples < 0:
        parser.error("--max-debug-samples must be non-negative")
    args.compression_mode = args.exec_mode
    args.pruning_mode = "middle_text_early"
    args.pruning_layer = 18
    args.pruning_strategy = "topk"
    args.vision_layers = "6,18,24"
    args.visual_pruning = True
    args.q_flowcover_radius = args.qflow_radius
    args.q_flowcover_query_chunk = args.qflow_query_chunk
    args.q_flowcover_candidate_chunk = args.qflow_candidate_chunk
    for key, value in resolve_decision_block(args.model).items():
        setattr(args, key, value)
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return evaluator.test(get_args(argv))


if __name__ == "__main__":
    main()
