"""Causal prefix, absolute constraints, persistence, and raw monitor checks."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from causal_models import (COMPONENTS, DECAY, DRIFT, METHODS, SCHEMA, CausalModelMonitor,
                           compose_components, instant_models, leaky_evidence, model_scores,
                           reference_evidence)
from encoder import EncoderConfig
from guard import PeakBank
from state_action import RELATION_NAMES, SCHEMA as BASE_SCHEMA


class CausalModelTests(unittest.TestCase):
    def test_absolute_constraints_and_decoupling_fallback(self):
        evidence = dict(acceleration=np.array([5., 5., np.nan]), decoupling=np.array([1., 4., np.nan]),
                        absolute_acceleration=np.array([2., 2., 8.]), absolute_mobility_low=np.array([.5, 3., 8.]),
                        joint_stop=np.array([6., 0., 8.]), state_trap=np.array([3., 0., 8.]))
        actual = instant_models(evidence)
        expected = dict(ac_reference=[5., 5.], absolute_acc=[2., 4.], absolute_both=[2., 3.],
                        joint_stop_or=[6., 5.], absolute_state_trap=[3., 4.])
        for name, values in expected.items():
            np.testing.assert_array_equal(actual[name][:2], values)
            self.assertTrue(np.isnan(actual[name][2]))

    def test_joint_stop_direction_and_zero_reference_evidence(self):
        branches = dict(acceleration=np.array([3., 3., -1., 3., np.nan]), decoupling=np.zeros(5),
                        state_relative_low=np.array([2., -1., 2., 2., 2.]))
        components = compose_components(branches, np.ones(5), -np.ones(5), np.array([1., 1., 1., -1., 1.]))
        np.testing.assert_array_equal(components["joint_stop"], [1., 0., 1., 0., 1.])
        np.testing.assert_allclose(components["state_trap"], [1., 0., 0., 0., np.nan], equal_nan=True)
        banks = {name: PeakBank([-np.inf, 0., 1., 1.]) for name in COMPONENTS}
        result = reference_evidence(components, banks)
        np.testing.assert_allclose(result["joint_stop"], [-np.log(3/5), 0., -np.log(3/5), 0., -np.log(3/5)])
        np.testing.assert_allclose(result["state_trap"], [-np.log(3/5), 0., 0., 0., np.nan], equal_nan=True)

    def test_persistence_requires_a_complete_window(self):
        evidence = {name: np.zeros((1, 8)) for name in COMPONENTS}
        evidence["acceleration"][:] = [[4., 0., 4., 4., 4., 4., 0., 4.]]
        evidence["decoupling"][:] = np.nan
        actual = model_scores(evidence, np.ones((1, 8), bool))["persistent_ac"]
        np.testing.assert_allclose(actual, [[np.nan, np.nan, np.nan, 0., 0., 4., 0., 0.]], equal_nan=True)

    def test_leaky_evidence_does_not_grow_without_input(self):
        np.testing.assert_array_equal(leaky_evidence(np.zeros((2, 500))), np.zeros((2, 500)))
        np.testing.assert_array_equal(leaky_evidence(np.full((1, 20), DRIFT)), np.zeros((1, 20)))
        values = np.array([[DRIFT + 4., DRIFT, 0., np.nan, DRIFT, DRIFT + 1.]])
        actual = leaky_evidence(values)
        np.testing.assert_allclose(actual, [[4., 3., max(0., DECAY * 3. - DRIFT), np.nan, 0., 1.]], equal_nan=True)

    def test_prefix_padding_and_missing_values(self):
        rng = np.random.default_rng(704)
        evidence = {name: rng.uniform(0., 4., (2, 20)) for name in COMPONENTS}
        for values in evidence.values():
            values[:, :8] = np.nan
        evidence["decoupling"][:, :11] = np.nan
        valid = np.arange(20)[None] < np.array([12, 20])[:, None]
        full = model_scores(evidence, valid)
        for stop in range(1, 21):
            prefix = model_scores({name: value[:, :stop] for name, value in evidence.items()}, valid[:, :stop])
            for name in METHODS:
                np.testing.assert_array_equal(prefix[name], full[name][:, :stop])
        for values in evidence.values():
            values[~valid] = 1e9
        padded = model_scores(evidence, valid)
        for name in METHODS:
            np.testing.assert_array_equal(full[name], padded[name])
        with self.assertRaises(ValueError):
            model_scores(evidence, np.array([[True, False, True]] * 2))

    def test_raw_monitor_matches_batch_preserves_alarms_and_resets(self):
        base_names = ("acceleration", "decoupling", *RELATION_NAMES)
        base = dict(schema=BASE_SCHEMA, encoder_config=asdict(EncoderConfig()), checkpoints=["test"], gate_alpha=.1,
                    frozen=dict(freeze_threshold=.57, acceleration_threshold=.1, periodicity_threshold=.37,
                                periodicity_scale=.02, frontback_threshold=-100., curvature_threshold=.389, slope=.0015),
                    banks={kind: {name: np.linspace(-1, 1, 399).tolist() for name in base_names}
                           for kind in ("episode", "task_init")})
        with tempfile.TemporaryDirectory() as directory:
            base_path, path = Path(directory) / "base.json", Path(directory) / "causal.json"
            base_path.write_text(json.dumps(base))
            profile = dict(schema=SCHEMA, methods=list(METHODS), decay=DECAY, drift=DRIFT,
                           base_profile=base_path.name, base_profile_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                           references={name: np.linspace(-1, 1, 500).tolist() for name in COMPONENTS},
                           banks={kind: {name: np.linspace(0, 10, 399).tolist() for name in METHODS}
                                  for kind in ("episode", "task_init")})
            path.write_text(json.dumps(profile))
            live = CausalModelMonitor(path, "test")
            rng = np.random.default_rng(733)
            raw = [rng.uniform(.001, .2, (8, 10, 11, 32)).astype(np.float32) for _ in range(24)]
            results = [live.update(x) for x in raw]
            evidence = {name: np.asarray([[row["evidence"][name] for row in results]]) for name in COMPONENTS}
            expected = model_scores(evidence, np.ones((1, len(results)), bool))
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
            for value in live.accumulators.values():
                self.assertEqual(value, 0.)
            self.assertEqual(len(live.recent), 0)
            replay = [live.update(x) for x in raw]
            for name in METHODS:
                np.testing.assert_array_equal([x["scores"][name] for x in results], [x["scores"][name] for x in replay])
            with self.assertRaises(ValueError):
                CausalModelMonitor(path, "wrong_checkpoint")


if __name__ == "__main__":
    unittest.main()
