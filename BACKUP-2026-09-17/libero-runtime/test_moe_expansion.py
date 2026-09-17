"""Checks for archive reconstruction, temporal rules and comparison denominators."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_moe_expansion as expansion
from moe_joint_patterns import INDEX, motif_scores
from routing_dynamics.encoder import ReadoutEncoder


class ExpansionTests(unittest.TestCase):
    def test_two_axis_matches_online_reference(self):
        rng = np.random.default_rng(28)
        readouts = rng.uniform(.01, .4, (5, 32, 25))
        readouts[:, 0, :8] = np.nan
        lengths = np.array([7, 8, 13, 20, 32])
        valid = np.arange(32)[None] < lengths[:, None]
        encoded = expansion.encode_two(readouts[..., :8], readouts[..., 16], valid)
        for e, length in enumerate(lengths):
            encoder = ReadoutEncoder()
            expected = np.asarray([encoder.update(x).values for x in readouts[e, :length]])
            np.testing.assert_allclose(encoded[e, :length], expected[:, [4, 5]],
                                       equal_nan=True, rtol=0, atol=1e-14)
            self.assertTrue(np.isnan(encoded[e, length:]).all())

    def test_constant_and_padding_never_create_a_motif(self):
        m = np.full((1, 20, 8), .1)
        a = np.full((1, 20), .2)
        valid = np.arange(20)[None] < 14
        encoded = expansion.encode_two(m, a, valid)
        np.testing.assert_array_equal(encoded[0, 7:14], np.zeros((7, 2)))
        m[:, 14:], a[:, 14:] = 1e20, 1e-20
        np.testing.assert_array_equal(expansion.encode_two(m, a, valid), encoded)
        self.assertTrue((expansion.first_alarms(expansion.score_two(encoded), 1.1) == -1).all())

    def test_gapped_observations_rejected(self):
        valid = np.ones((1, 12), bool)
        valid[0, 3] = False
        with self.assertRaises(ValueError):
            expansion.encode_two(np.ones((1, 12, 8)), np.ones((1, 12)), valid)

    def test_temporal_motifs_match_existing_implementation(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, .3, (7, 52, 2))
        x[:, :7] = np.nan
        for row in x:
            full = np.full((52, 27), np.nan)
            full[:, INDEX['mobility_log_ratio']] = row[:, 0]
            full[:, INDEX['acceleration_log_ratio']] = row[:, 1]
            expected = motif_scores(full)[:, [0, 5]]
            actual = expansion.score_two(row[None])[0][:, [1, 2]]
            np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=0, atol=0)

    def test_transition_requires_two_completed_nonoverlapping_windows(self):
        x = np.full((1, 20, 2), np.nan)
        x[:, 7:10] = .4
        x[:, 10:] = -.4
        scores = expansion.score_two(x)
        self.assertEqual(expansion.first_alarms(scores, 1.2)[0, 2], 12)
        x[:, 9] = -.4
        self.assertEqual(expansion.first_alarms(expansion.score_two(x), 1.2)[0, 2], -1)

    def test_prefix_causality(self):
        rng = np.random.default_rng(18)
        m = rng.uniform(.01, .3, (3, 35, 8))
        a = rng.uniform(.01, .3, (3, 35))
        valid = np.ones((3, 35), bool)
        full = expansion.score_two(expansion.encode_two(m, a, valid))
        for stop in range(1, 36):
            prefix = expansion.score_two(expansion.encode_two(m[:, :stop], a[:, :stop], valid[:, :stop]))
            np.testing.assert_array_equal(prefix, full[:, :stop])

    def test_duplicate_metadata_is_not_silently_joined(self):
        frame = pd.DataFrame(dict(task=['a', 'a'], episode=[1, 1]))
        with self.assertRaises(ValueError):
            expansion.unique_lookup(frame, ['task', 'episode'])

    def test_auc_ties_and_direction(self):
        labels = np.array([True, False, True, False])
        self.assertEqual(expansion.auc(labels, np.ones(4)), .5)
        self.assertEqual(expansion.auc(labels, labels.astype(float)), 1)
        self.assertEqual(expansion.auc(labels, -labels.astype(float)), 0)

    def test_matching_does_not_count_alarm_after_success_endpoint(self):
        frame = pd.DataFrame(dict(cohort=['x', 'x'], task=['t', 't'], init_state_id=[0, 0],
                                  failure=[True, False], length=[30, 12], v82=[18, -1]))
        for name in expansion.NAMES:
            frame['first_' + name] = [18, -1]
        scores = np.full((2, 30, len(expansion.NAMES)), np.nan)
        scores[0, 9:] = -.2
        scores[0, 18:] = .4
        scores[1, 9:12] = -.2
        with tempfile.TemporaryDirectory() as directory, patch.object(expansion, 'ROOT', Path(directory)):
            result = expansion.completion_matched(frame, scores)
        self.assertEqual(len(result), len(expansion.NAMES) + 1)
        for row in result:
            self.assertEqual(row['pairs'], 1)
            self.assertEqual(row['fail_hits'], 0)
            self.assertEqual(row['success_hits'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
