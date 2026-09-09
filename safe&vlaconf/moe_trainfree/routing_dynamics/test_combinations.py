"""Fixed formula, prefix, calibration, and raw monitor equivalence checks."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from encoder import EncoderConfig
from state_action import RELATION_NAMES, SCHEMA as BASE_SCHEMA
from combinations import (COMPONENTS, METHODS, SCHEMA, CombinationMonitor, combination_scores,
                          combine_instant, component_scores, reference_evidence)
from guard import PeakBank


class CombinationTests(unittest.TestCase):
    def test_formulas_and_neutral_missing_component(self):
        evidence = dict(acceleration=np.array([2., 2., np.nan]), decoupling=np.array([1., np.nan, np.nan]),
                        gap=np.array([3., 1., 10.]), state_adjustment=np.array([4., 0., 10.]))
        scores = combine_instant(evidence)
        expected = dict(ac_recalibrated=[2., 2.], max_or_gap=[3., 2.], soft_sum_gap=[5., 3.],
                        two_of_three=[2., 1.], state_adjustment_or=[4., 2.])
        for name, values in expected.items():
            np.testing.assert_array_equal(scores[name][:2], values)
            self.assertTrue(np.isnan(scores[name][2]))

    def test_lagged_evidence_expires_after_four_queries(self):
        evidence = dict(acceleration=np.array([[2., 0., 0., 0., 0., 0.]]),
                        decoupling=np.zeros((1, 6)), gap=np.array([[0., 0., 3., 3., 3., 3.]]),
                        state_adjustment=np.zeros((1, 6)))
        values = combination_scores(evidence, np.ones((1, 6), bool))["lagged_agreement"]
        np.testing.assert_array_equal(values, [[0., 0., 2., 2., 0., 0.]])

    def test_prefix_padding_and_no_warmup_resizing(self):
        rng = np.random.default_rng(505)
        evidence = {name: rng.uniform(0., 4., (2, 20)) for name in COMPONENTS}
        for values in evidence.values():
            values[:, :8] = np.nan
        evidence["decoupling"][:, :11] = np.nan
        valid = np.arange(20)[None] < np.array([12, 20])[:, None]
        full = combination_scores(evidence, valid)
        for stop in range(1, 21):
            prefix = combination_scores({name: value[:, :stop] for name, value in evidence.items()}, valid[:, :stop])
            for name in METHODS:
                np.testing.assert_array_equal(prefix[name], full[name][:, :stop])
        for values in evidence.values():
            values[~valid] = 1e9
        padded = combination_scores(evidence, valid)
        for name in METHODS:
            np.testing.assert_array_equal(full[name], padded[name])
            self.assertTrue(np.isnan(full[name][:, :8]).all())

    def test_reference_tail_ties_and_direction(self):
        banks = {name: PeakBank([-np.inf, 0., 1., 1.]) for name in COMPONENTS}
        components = {name: np.array([-1., 0., 1., 2., np.nan]) for name in COMPONENTS}
        evidence = reference_evidence(components, banks)
        np.testing.assert_allclose(evidence["gap"], [0., 0., -np.log(3/5), -np.log(1/5), np.nan], equal_nan=True)
        values = component_scores(dict(acceleration=np.array([1., -1., np.nan]), decoupling=np.zeros(3),
                                       gap=np.zeros(3), state_relative_low=np.array([2., 3., 4.])))
        np.testing.assert_allclose(values["state_adjustment"], [1., 0., np.nan], equal_nan=True)

    def test_raw_monitor_matches_batch_and_keeps_old_alarms(self):
        base_names = ("acceleration", "decoupling", *RELATION_NAMES)
        base = dict(schema=BASE_SCHEMA, encoder_config=asdict(EncoderConfig()), checkpoints=["test"], gate_alpha=.1,
                    frozen=dict(freeze_threshold=.57, acceleration_threshold=.1, periodicity_threshold=.37,
                                periodicity_scale=.02, frontback_threshold=-100., curvature_threshold=.389, slope=.0015),
                    banks={kind: {name: np.linspace(-1, 1, 399).tolist() for name in base_names}
                           for kind in ("episode", "task_init")})
        with tempfile.TemporaryDirectory() as directory:
            base_path, path = Path(directory) / "base.json", Path(directory) / "combination.json"
            base_path.write_text(json.dumps(base))
            profile = dict(schema=SCHEMA, methods=list(METHODS), base_profile=base_path.name,
                           base_profile_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                           references={name: np.linspace(-1, 1, 500).tolist() for name in COMPONENTS},
                           banks={kind: {name: np.linspace(0, 10, 399).tolist() for name in METHODS}
                                  for kind in ("episode", "task_init")})
            path.write_text(json.dumps(profile))
            live = CombinationMonitor(path, "test")
            rng = np.random.default_rng(413)
            results = [live.update(rng.uniform(.001, .2, (8, 10, 11, 32)).astype(np.float32)) for _ in range(24)]
            evidence = {name: np.asarray([[row["evidence"][name] for row in results]]) for name in COMPONENTS}
            expected = combination_scores(evidence, np.ones((1, len(results)), bool))
            for name in METHODS:
                np.testing.assert_array_equal(expected[name][0], [row["scores"][name] for row in results])
            self.assertEqual(results[-1]["frozen_first_query"], 7)
            self.assertLessEqual(results[-1]["first_alarm_query"], 7)
            for before, after in zip(results, results[1:]):
                if before["ever_alarm"]:
                    self.assertTrue(after["ever_alarm"])
                    self.assertEqual(before["first_alarm_query"], after["first_alarm_query"])
            query = live.base.query
            with self.assertRaises(ValueError):
                live.update(np.zeros((8, 10, 11, 32)))
            self.assertEqual(live.base.query, query)
            live.reset()
            self.assertEqual(live.first_alarm, -1)


if __name__ == "__main__":
    unittest.main()
