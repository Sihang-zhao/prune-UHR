"""CLI and evaluator contracts that do not require model weights."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluator
import testv


class CLITests(unittest.TestCase):
    def test_help_works_without_site_packages_and_lists_shared_options(self):
        result = subprocess.run(
            [sys.executable, '-S', str(Path(testv.__file__)), '--help'],
            capture_output=True, text=True, check=True,
        )
        for flag in ('--data', '--model', '--gpus', '--prune-method', '--exec-mode'):
            self.assertIn(flag, result.stdout)

    def test_fraction_and_aliases_reach_the_evaluator(self):
        with patch.object(evaluator, 'test', return_value={'total': 1}) as run:
            self.assertEqual(testv.main(['--data', 'sample.jsonl', '--prune-fraction', '0.7',
                                         '--qflow-radius', '2']), {'total': 1})
        args = run.call_args.args[0]
        self.assertEqual(args.visual_prune_percent, 70.0)
        self.assertEqual(args.q_flowcover_radius, 2)
        self.assertEqual((args.pruning_layer, args.decision_block_index), (18, 17))
        self.assertEqual(args.prune_method, 'q_flowcover')

    def test_invalid_or_conflicting_options_fail_before_loading_a_model(self):
        cases = [
            ['--decision-layer-human', '17'],
            ['--qflow-query-chunk', '0'],
            ['--qflow-radius', '-1'],
            ['--prune-fraction', '1'],
            ['--prune-fraction', '0.5', '--visual-prune-percent', '50'],
            ['--att-stats', 'summary'],
            ['--save-att-stats'],
            ['--prune-method', 'q_flowcover_seeded_all'],
        ]
        for extra in cases:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    testv.get_args(['--data', 'sample.jsonl', *extra])
                self.assertEqual(exc.exception.code, 2)

    def test_local_checkpoint_layer_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps({'vision_config': {'depth': 24, 'spatial_merge_size': 2}}))
            args = testv.get_args(['--data', 'sample.jsonl', '--model', tmp])
            self.assertTrue(args.config_verified)
            self.assertEqual(args.vision_depth, 24)
            config.write_text(json.dumps({'vision_config': {'depth': 18}}))
            with self.assertRaisesRegex(ValueError, 'depth=18'):
                testv.get_args(['--data', 'sample.jsonl', '--model', tmp])


class ExecutionTraceTests(unittest.TestCase):
    def row(self, full_blocks):
        return {
            'index': 0, 'selection_mode': 'middle_text_early', 'selection_strategy': 'topk',
            'selection_layer': 18, 'original_image_tokens': 4, 'retained_tokens': 2,
            'vision_merge_unit': 4, 'vision_layer_token_counts': [16] * full_blocks + [8] * (24 - full_blocks),
            'selection_budgets': [(0, 2, 0)], 'selected_indices': [[0, 2]],
            'selection_groups': [{'final': [], 'middle': [0, 2], 'shallow': []}],
        }

    def test_each_mode_accepts_its_own_token_trace(self):
        for mode in ('hard18', 'full19_gather', 'att_bridge19_fullkv'):
            with self.subTest(mode=mode):
                args = testv.get_args(['--data', 'sample.jsonl', '--exec-mode', mode, '--visual-keep-tokens', '2'])
                self.assertTrue(evaluator._validate_pruning_row(self.row(18 if mode == 'hard18' else 19), args))

    def test_wrong_trace_and_wrong_fixed_budget_are_rejected(self):
        args = testv.get_args(['--data', 'sample.jsonl', '--visual-keep-tokens', '2'])
        with self.assertRaisesRegex(RuntimeError, 'execution trace'):
            evaluator._validate_pruning_row(self.row(19), args)
        row = self.row(18)
        row['selected_indices'] = [[0]]
        row['selection_budgets'] = [(0, 1, 0)]
        with self.assertRaisesRegex(RuntimeError, 'override K=2'):
            evaluator._validate_pruning_row(row, args)


class ResumeTests(unittest.TestCase):
    def test_changed_selector_or_budget_cannot_reuse_old_results(self):
        args = testv.get_args(['--data', 'sample.jsonl'])
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'results.jsonl'
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps(vars(args)))
            evaluator._validate_resume_config(source, args)
            for key, value in [('prune_method', 'text_cos'), ('visual_keep_ratio', 0.3)]:
                previous = vars(args).copy()
                previous[key] = value
                config.write_text(json.dumps(previous))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    evaluator._validate_resume_config(source, args)

    def test_resume_requires_original_configuration(self):
        args = testv.get_args(['--data', 'sample.jsonl'])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'missing'):
                evaluator._validate_resume_config(Path(tmp) / 'results.jsonl', args)


if __name__ == '__main__':
    unittest.main()
