"""Small CPU references for the selector; no checkpoint or dataset is needed."""
from __future__ import annotations

import unittest


class QFlowCoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest(f"PyTorch unavailable: {exc}") from exc
        from compression import q_flowcover

        cls.torch = torch
        cls.selector = q_flowcover

    def setUp(self):
        self.generator = self.torch.Generator().manual_seed(29)

    def _randn(self, *shape):
        return self.torch.randn(*shape, generator=self.generator)

    def _fixture(self):
        return {
            "projected": [self._randn(6, 12), self._randn(8, 12)],
            "text": self._randn(5, 12),
            "q": [self._randn(24, 2, 4), self._randn(32, 2, 4)],
            "k": [self._randn(24, 2, 4), self._randn(32, 2, 4)],
            "hidden": [self._randn(24, 5), self._randn(32, 5)],
            "keep": [3, 5],
            "raw": [24, 32],
            "grid": [(2, 3), (2, 4)],
        }

    def _select(self, data, method="q_flowcover", **kwargs):
        return self.selector.select_q_flowcover_views(
            data["projected"], data["text"], data["q"], data["k"],
            data["hidden"], data["keep"], data["raw"], 4, method,
            data["grid"], scale=0.5, **kwargs,
        )

    def test_all_methods_preserve_budget_sorted_indices_and_role_metadata(self):
        torch = self.torch
        data = self._fixture()
        for method in sorted(self.selector.Q_FLOWCOVER_METHODS):
            with self.subTest(method=method):
                results, debug = self._select(data, method)
                self.assertEqual(len(results), 2)
                for result, record, keep, projected in zip(
                    results, debug, data["keep"], data["projected"]
                ):
                    indices = result.indices
                    self.assertEqual(indices.dtype, torch.long)
                    self.assertEqual(indices.numel(), keep)
                    self.assertEqual(torch.unique(indices).numel(), keep)
                    self.assertTrue(bool((indices >= 0).all()))
                    self.assertTrue(bool((indices < projected.shape[0]).all()))
                    self.assertTrue(bool((indices[1:] > indices[:-1]).all()))
                    self.assertTrue(torch.equal(result.middle_indices, indices))
                    self.assertEqual(result.final_indices.numel(), 0)
                    self.assertEqual(result.shallow_indices.numel(), 0)
                    self.assertEqual(result.budget, (0, keep, 0))
                    self.assertEqual(record["final_group_index"], indices.tolist())
                    self.assertEqual(record["K"], keep)

    def test_views_are_independent_and_match_individual_calls(self):
        torch = self.torch
        data = self._fixture()
        for method in sorted(self.selector.Q_FLOWCOVER_METHODS):
            with self.subTest(method=method):
                combined, _ = self._select(data, method)
                for view in (0, 1):
                    single = {
                        key: value if key == "text" else [value[view]]
                        for key, value in data.items()
                    }
                    separate, _ = self._select(single, method)
                    self.assertTrue(torch.equal(combined[view].indices, separate[0].indices))
                changed = dict(data)
                for key in ("projected", "q", "k", "hidden"):
                    changed[key] = [data[key][0], self._randn(*data[key][1].shape)]
                altered, _ = self._select(changed, method)
                self.assertTrue(torch.equal(combined[0].indices, altered[0].indices))

    def test_chunked_attention_support_matches_dense_group_reference(self):
        torch = self.torch
        unit, groups, heads, dim = 4, 6, 2, 4
        q = self._randn(unit * groups, heads, dim)
        k = self._randn(unit * groups, heads, dim)
        p = torch.tensor([0, 2, 5])
        pi = torch.tensor([0.2, 0.0, 0.3, 0.0, 0.0, 0.5])
        # Deliberately form the full attention only in this tiny reference.
        raw_attention = torch.softmax(torch.einsum("qhd,khd->hqk", q, k) * 0.5, dim=-1)
        group_attention = raw_attention.mean(dim=0).reshape(groups, unit, groups, unit).sum(dim=3).mean(dim=1)
        expected = (group_attention[p] * pi[p, None]).sum(dim=0)
        for chunk in (1, 3, 4, 7, 128):
            with self.subTest(chunk=chunk):
                actual, query_count = self.selector._group_source_support(
                    q, k, p, pi, unit, scale=0.5, query_chunk_size=chunk,
                )
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
                self.assertEqual(query_count, unit * p.numel())
                self.assertAlmostEqual(float(actual.sum()), 1.0, places=6)

    def test_radius_zero_degenerates_to_weight_topk(self):
        torch = self.torch
        data = self._fixture()
        for method in ("q_flowcover", "text_local_cover"):
            with self.subTest(method=method):
                results, debug = self._select(data, method, local_radius=0)
                for result, record, keep in zip(results, debug, data["keep"]):
                    weight = torch.tensor(record["omega"])
                    expected = torch.topk(weight, keep).indices.sort().values
                    self.assertTrue(torch.equal(result.indices, expected))
        flow, _ = self._select(data, "flow_topk")
        zero, _ = self._select(data, "q_flowcover", local_radius=0)
        for first, second in zip(flow, zero):
            self.assertTrue(torch.equal(first.indices, second.indices))

    def test_keep_all_returns_every_group(self):
        torch = self.torch
        data = self._fixture()
        data["keep"] = [6, 8]
        for method in sorted(self.selector.Q_FLOWCOVER_METHODS):
            with self.subTest(method=method):
                results, _ = self._select(data, method)
                for result, count in zip(results, data["keep"]):
                    self.assertTrue(torch.equal(result.indices, torch.arange(count)))

    def _dense_greedy(self, similarity, weight, keep):
        """Reference weighted max-coverage over a fully materialized graph."""
        torch = self.torch
        coverage = torch.zeros_like(weight)
        selected = []
        for _ in range(keep):
            gains = (weight[:, None] * (similarity - coverage[:, None]).clamp_min(0)).sum(dim=0)
            gains[selected] = -float("inf")
            candidate = int(gains.argmax())
            selected.append(candidate)
            coverage = torch.maximum(coverage, similarity[:, candidate])
        return torch.tensor(sorted(selected))

    def test_sparse_local_cover_matches_dense_greedy_on_small_grids(self):
        torch = self.torch
        for height, width, radius in ((2, 3, 1), (3, 3, 1), (2, 4, 2)):
            with self.subTest(grid=(height, width), radius=radius):
                count = height * width
                features = torch.nn.functional.normalize(self._randn(count, 9), dim=-1)
                weight = torch.softmax(self._randn(count), dim=0)
                similarity = (features @ features.T).clamp_min(0)
                for source in range(count):
                    for candidate in range(count):
                        sy, sx = divmod(source, width)
                        cy, cx = divmod(candidate, width)
                        if max(abs(sy - cy), abs(sx - cx)) > radius:
                            similarity[source, candidate] = 0
                expected = self._dense_greedy(similarity, weight, 4)
                actual, _ = self.selector._local_cover(features, weight, 4, height, width, radius)
                self.assertTrue(torch.equal(actual, expected))

    def test_global_chunks_match_dense_greedy(self):
        torch = self.torch
        features = torch.nn.functional.normalize(self._randn(9, 11), dim=-1)
        weight = torch.softmax(self._randn(9), dim=0)
        expected = self._dense_greedy((features @ features.T).clamp_min(0), weight, 5)
        for chunk in (1, 2, 4, 32):
            with self.subTest(chunk=chunk):
                actual, _ = self.selector._global_cover(features, weight, 5, candidate_chunk_size=chunk)
                self.assertTrue(torch.equal(actual, expected))

    def test_rejects_invalid_view_count_grid_budget_and_missing_qk(self):
        data = self._fixture()
        for key, value, method in (
            ("grid", [(2, 3)], "q_flowcover"),
            ("grid", [(3, 3), (2, 4)], "q_flowcover"),
            ("keep", [0, 5], "q_flowcover"),
            ("keep", [7, 5], "q_flowcover"),
            ("q", [None, None], "q_flowcover"),
        ):
            with self.subTest(key=key, value=value):
                invalid = {**data, key: value}
                with self.assertRaises(ValueError):
                    self._select(invalid, method)


if __name__ == "__main__":
    unittest.main()
