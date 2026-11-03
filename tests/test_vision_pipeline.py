"""Exercise the whole visual loop with native tiny Qwen3-VL modules, offline."""
from __future__ import annotations

from types import SimpleNamespace
import unittest


class VisionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
            from transformers.models.qwen3_vl import modeling_qwen3_vl
        except ImportError as exc:
            raise unittest.SkipTest(f"Native Qwen3-VL runtime unavailable: {exc}") from exc
        from compression.qwen3_vl_modes import Qwen3VLCompressionRunner

        cls.torch = torch
        cls.native_module = modeling_qwen3_vl
        cls.runner_cls = Qwen3VLCompressionRunner
        cls.config_cls = Qwen3VLVisionConfig
        cls.vision_cls = Qwen3VLVisionModel
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "previous_threads"):
            cls.torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch = self.torch
        torch.manual_seed(41)
        self.config = self.config_cls(
            depth=20, hidden_size=16, intermediate_size=32, out_hidden_size=24,
            num_heads=2, spatial_merge_size=2, patch_size=2,
            temporal_patch_size=2, num_position_embeddings=64,
            deepstack_visual_indexes=[5, 17, 18], _attn_implementation="eager",
        )
        self.vit = self.vision_cls(self.config).eval()
        self.grid = torch.tensor([[1, 4, 4], [1, 4, 8]])
        self.pixels = torch.randn(48, 24)
        self.text = torch.randn(6, 24)

    def _runner(self, mode="hard18", method="q_flowcover", keep=2, stats="none"):
        runner = object.__new__(self.runner_cls)
        runner.model = SimpleNamespace(config=SimpleNamespace(vision_config=self.config))
        runner.vit = self.vit
        runner.projection = self.vit.merger
        runner.device = self.torch.device("cpu")
        runner.pruning_layer = runner.decision_layer_human = 18
        runner.decision_block_index = 17
        runner.bridge_block_index = 18
        runner.vision_layers = (6, 18, 20)
        runner.visual_keep_ratio = 1.0 if keep is None else 0.5
        runner.visual_keep_tokens = keep
        runner.pruning_mode = "middle_text_early"
        runner.pruning_strategy = "topk"
        runner.visual_pruning = True
        runner.compression_mode = mode
        runner.prune_method = method
        runner.att_stats_mode = stats
        runner.max_debug_samples = 0
        runner._debug_samples_seen = 0
        runner.q_flowcover_query_chunk = 3
        runner.q_flowcover_candidate_chunk = 2
        runner.q_flowcover_radius = 1
        runner._last_compression_metadata = {}
        runner._last_stage_timings = {}
        runner._last_att_stats = {}
        runner._last_q_flowcover_debug = []
        runner._last_native_deepstack = []
        runner._last_native_final_features = []
        runner._last_early_selections = []
        runner._last_early_original_counts = []
        runner._last_vision_layer_token_counts = []
        runner._last_selection = {}
        runner._last_cached_visual = None
        return runner

    def _run(self, runner, pixels=None, grid=None):
        original_rotary = self.native_module.apply_rotary_pos_emb_vision
        with self.torch.inference_mode():
            runner._vision_forward_middle_early(
                self.pixels if pixels is None else pixels,
                self.grid if grid is None else grid, self.text,
            )
        self.assertIs(self.native_module.apply_rotary_pos_emb_vision, original_rotary)
        return runner

    def test_fixed_k_compacts_at_the_expected_layer_for_all_modes(self):
        torch = self.torch
        for mode, method in (
            ("hard18", "q_flowcover"),
            ("full19_gather", "q_flowcover"),
            ("att_bridge19_fullkv", "q_flowcover"),
            ("hard18", "text_cos"),
        ):
            with self.subTest(mode=mode, method=method):
                traces = {}

                def capture(index):
                    def record(_module, args, kwargs):
                        traces[index] = (
                            args[0].shape[0], kwargs["cu_seqlens"].clone(),
                            tuple(part.clone() for part in kwargs["position_embeddings"]),
                        )
                    return record

                handles = [
                    self.vit.blocks[index].register_forward_pre_hook(capture(index), with_kwargs=True)
                    for index in (0, 19)
                ]
                try:
                    runner = self._run(self._runner(mode, method))
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(runner._last_early_original_counts, [4, 8])
                self.assertEqual([part.shape[0] for part in runner._last_native_final_features], [2, 2])
                self.assertEqual(len(runner._last_native_deepstack), 3)
                for branch in runner._last_native_deepstack:
                    self.assertEqual([part.shape[0] for part in branch], [2, 2])
                full_layers = 18 if mode == "hard18" else 19
                self.assertEqual(runner._last_vision_layer_token_counts, [48] * full_layers + [16] * (20 - full_layers))
                for selection, count in zip(runner._last_early_selections, [4, 8]):
                    self.assertEqual(selection.indices.numel(), 2)
                    self.assertEqual(torch.unique(selection.indices).numel(), 2)
                    self.assertTrue(bool((selection.indices < count).all()))
                raw_keep = torch.cat([
                    (runner._last_early_selections[0].indices[:, None] * 4 + torch.arange(4)).flatten(),
                    (runner._last_early_selections[1].indices[:, None] * 4 + torch.arange(4)).flatten() + 16,
                ])
                self.assertEqual(traces[19][0], 16)
                self.assertEqual(traces[19][1].tolist(), [0, 8, 16])
                for original, compact in zip(traces[0][2], traces[19][2]):
                    torch.testing.assert_close(compact, original[raw_keep])
                if method == "q_flowcover":
                    self.assertEqual([row["source_query_count"] for row in runner._last_q_flowcover_debug], [8, 8])
                    self.assertEqual(runner._last_compression_metadata["retained_group_count"], 4)

    def test_keep_all_matches_native_final_and_every_deepstack_branch(self):
        torch = self.torch
        with torch.inference_mode():
            expected_final, expected_deepstack = self.vit(self.pixels, grid_thw=self.grid)
        for mode, method in (
            ("hard18", "q_flowcover"),
            ("full19_gather", "q_flowcover"),
            ("att_bridge19_fullkv", "q_flowcover"),
            ("hard18", "text_cos"),
        ):
            with self.subTest(mode=mode, method=method):
                runner = self._run(self._runner(mode, method, keep=None))
                torch.testing.assert_close(torch.cat(runner._last_native_final_features), expected_final, atol=1e-5, rtol=1e-5)
                self.assertEqual(runner._last_vision_layer_token_counts, [48] * 20)
                self.assertEqual(len(runner._last_native_deepstack), len(expected_deepstack))
                for branch, reference in zip(runner._last_native_deepstack, expected_deepstack):
                    torch.testing.assert_close(torch.cat(branch), reference, atol=1e-5, rtol=1e-5)
                for selection, count in zip(runner._last_early_selections, [4, 8]):
                    self.assertTrue(torch.equal(selection.indices, torch.arange(count)))

    def test_multiple_views_match_independent_vision_calls(self):
        torch = self.torch
        combined = self._run(self._runner())
        for view, (start, end) in enumerate(((0, 16), (16, 48))):
            with self.subTest(view=view):
                separate = self._run(self._runner(), self.pixels[start:end], self.grid[view : view + 1])
                self.assertTrue(torch.equal(combined._last_early_selections[view].indices, separate._last_early_selections[0].indices))
                torch.testing.assert_close(combined._last_native_final_features[view], separate._last_native_final_features[0], atol=1e-5, rtol=1e-5)
                for combined_branch, separate_branch in zip(combined._last_native_deepstack, separate._last_native_deepstack):
                    torch.testing.assert_close(combined_branch[view], separate_branch[0], atol=1e-5, rtol=1e-5)

    def test_full19_and_full_kv_bridge_produce_the_same_visual_features(self):
        torch = self.torch
        full = self._run(self._runner("full19_gather"))
        bridge = self._run(self._runner("att_bridge19_fullkv", stats="summary"))
        for first, second in zip(full._last_early_selections, bridge._last_early_selections):
            self.assertTrue(torch.equal(first.indices, second.indices))
        for first, second in zip(full._last_native_final_features, bridge._last_native_final_features):
            torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)
        for first_branch, second_branch in zip(full._last_native_deepstack, bridge._last_native_deepstack):
            for first, second in zip(first_branch, second_branch):
                torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)

    def test_debug_bridge_compares_the_correct_merged_deepstack_rows(self):
        runner = self._run(self._runner("att_bridge19_fullkv", stats="debug"))
        errors = runner._last_att_stats["deepstack_full_vs_bridge"]
        self.assertLess(errors["max_abs_error"], 1e-5)
        self.assertEqual([part.shape[0] for part in runner._last_native_final_features], [2, 2])

    def test_rotary_capture_is_restored_if_the_decision_block_raises(self):
        original_rotary = self.native_module.apply_rotary_pos_emb_vision
        runner = self._runner()

        def fail(_module, _args):
            self.assertIsNot(self.native_module.apply_rotary_pos_emb_vision, original_rotary)
            raise RuntimeError("deliberate decision-block failure")

        handle = self.vit.blocks[17].register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "deliberate decision-block failure"):
                self._run(runner)
        finally:
            handle.remove()
        self.assertIs(self.native_module.apply_rotary_pos_emb_vision, original_rotary)


if __name__ == "__main__":
    unittest.main()
