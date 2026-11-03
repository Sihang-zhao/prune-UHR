"""Compare the bridge against tiny, randomly initialized native Qwen blocks."""
from __future__ import annotations

import unittest


class NativeBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock
        except ImportError as exc:
            raise unittest.SkipTest(f"Native Qwen3-VL runtime unavailable: {exc}") from exc
        from compression.qwen3_vl_modes import Qwen3VLCompressionRunner

        cls.torch = torch
        cls.config_cls = Qwen3VLVisionConfig
        cls.block_cls = Qwen3VLVisionBlock
        cls.runner_cls = Qwen3VLCompressionRunner

    def setUp(self):
        torch = self.torch
        torch.manual_seed(7)
        config = self.config_cls(
            hidden_size=32, num_heads=4, intermediate_size=64,
            depth=2, spatial_merge_size=2, _attn_implementation="eager",
        )
        self.block = self.block_cls(config).eval()
        self.runner = object.__new__(self.runner_cls)
        self.counts = [8, 12]
        self.hidden = torch.randn(sum(self.counts), 32)
        angles = torch.randn(sum(self.counts), 8)
        self.position = (angles.cos(), angles.sin())
        self.keep = [torch.tensor([0, 3, 7]), torch.tensor([1, 4, 9])]

    def _native(self, hidden=None, selected=False):
        torch = self.torch
        hidden = self.hidden if hidden is None else hidden
        outputs = []
        cursor = 0
        for count, keep in zip(self.counts, self.keep):
            view = hidden[cursor : cursor + count]
            cos, sin = (part[cursor : cursor + count] for part in self.position)
            if selected:
                view, cos, sin = (part.index_select(0, keep) for part in (view, cos, sin))
            cu = torch.tensor([0, view.shape[0]], dtype=torch.int32)
            outputs.append(self.block(view, cu_seqlens=cu, position_embeddings=(cos, sin)))
            cursor += count
        return torch.cat(outputs, dim=0)

    def test_keep_all_degenerates_to_full_native_block(self):
        torch = self.torch
        for collect_stats in (False, True):
            with self.subTest(att_stats=collect_stats):
                bridge, stats = self.runner._bridge_block_views(
                    self.block, self.hidden, self.position, self.counts,
                    [torch.arange(count) for count in self.counts], att_stats=collect_stats,
                )
                torch.testing.assert_close(bridge, self._native(), atol=1e-5, rtol=1e-5)
                if collect_stats:
                    self.assertAlmostEqual(stats["Gamma"], 0.0, places=6)
                    self.assertAlmostEqual(stats["readout_gap"], 0.0, places=6)

    def test_selected_sources_degenerate_to_hard_pruned_native_block(self):
        torch = self.torch
        before = self.hidden.clone()
        bridge, stats = self.runner._bridge_block_views(
            self.block, self.hidden, self.position, self.counts, self.keep,
            att_stats=True, source_mode="selected",
        )
        torch.testing.assert_close(bridge, self._native(selected=True), atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(stats["Gamma"], 0.0, places=6)
        self.assertAlmostEqual(stats["readout_gap"], 0.0, places=6)
        self.assertTrue(torch.equal(self.hidden, before))

    def test_full_kv_bridge_equals_gathered_full_native_rows(self):
        torch = self.torch
        full = self._native()
        global_keep = torch.cat([self.keep[0], self.keep[1] + self.counts[0]])
        expected = full.index_select(0, global_keep)
        for collect_stats in (False, True):
            with self.subTest(att_stats=collect_stats):
                bridge, stats = self.runner._bridge_block_views(
                    self.block, self.hidden, self.position, self.counts,
                    self.keep, att_stats=collect_stats,
                )
                torch.testing.assert_close(bridge, expected, atol=1e-5, rtol=1e-5)
                if collect_stats:
                    self.assertGreater(stats["Gamma"], 0.0)
                    self.assertLess(stats["Gamma"], 1.0)
                    self.assertEqual([part.numel() for part in stats["source_centrality"]], self.counts)

    def test_bridge_keeps_views_isolated(self):
        torch = self.torch
        original, _ = self.runner._bridge_block_views(
            self.block, self.hidden, self.position, self.counts, self.keep, att_stats=False,
        )
        changed = self.hidden.clone()
        changed[self.counts[0] :] = torch.randn_like(changed[self.counts[0] :]) * 100
        altered, _ = self.runner._bridge_block_views(
            self.block, changed, self.position, self.counts, self.keep, att_stats=False,
        )
        torch.testing.assert_close(original[: len(self.keep[0])], altered[: len(self.keep[0])])


if __name__ == "__main__":
    unittest.main()
