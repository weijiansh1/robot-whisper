"""Relative scale definitions, temporal causality, and raw online agreement."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from encoder import EncoderConfig, FEATURE_NAMES
from guard import PeakBank
from relative_models import (COMPONENTS, METHODS, RAW_NAMES, SCHEMA, WINDOW, RelativeModelMonitor,
                             instant_components, instant_models, make_components, model_scores, reference_evidence, window_mean)
from state_action import RELATION_NAMES, SCHEMA as BASE_SCHEMA


class RelativeModelTests(unittest.TestCase):
    def test_relative_differences_and_common_scale_shift(self):
        features = np.zeros((3, len(FEATURE_NAMES)))
        relations = np.zeros((3, len(RELATION_NAMES)))
        for name, values in (("acceleration_log_ratio", [2., 1., -2.]), ("path_log_ratio", [1., 2., -1.]),
                             ("mobility_log_ratio", [0., 0., -3.]), ("layer_agreement", [1., .5, 0.])):
            features[:, FEATURE_NAMES.index(name)] = values
        relations[:, RELATION_NAMES.index("state_relative_low")] = [1., -2., 3.]
        expected = dict(relative_curvature=[1., 0., 0.], relative_path_mobility=[1., 2., 2.],
                        relative_state=[3., 0., 1.], routing_relief=[0., 0., 0.], layer_support=[1., .5, 0.])
        current = instant_components(features, relations)
        for name, values in expected.items():
            np.testing.assert_array_equal(current[name], values)
        for name in ("acceleration_log_ratio", "path_log_ratio", "mobility_log_ratio"):
            features[:, FEATURE_NAMES.index(name)] += 4.
        relations[:, RELATION_NAMES.index("state_relative_low")] -= 4.
        shifted = instant_components(features, relations)
        for name in current:
            np.testing.assert_array_equal(shifted[name], current[name])

    def test_constraints_soft_combination_and_relief(self):
        evidence = dict(acceleration=np.array([8., 8., np.nan]), decoupling=np.array([1., 4., np.nan]),
                        relative_curvature=np.array([2., 2., 8.]), relative_path_mobility=np.array([3., 1., 8.]),
                        relative_state=np.array([6., 3., 8.]), routing_relief=np.array([1., np.nan, 8.]),
                        layer_support=np.array([.5, .25, 1.]))
        actual = instant_models(evidence)
        expected = dict(ac_reference=[8., 8.], relative_curvature=[2., 4.], relative_path_mobility=[3., 4.],
                        relative_state=[6., 4.], relative_both=[2., 2.], relative_soft=[4., 4.],
                        layer_consensus=[4., 2.], relative_recovery=[1., 4.])
        for name, values in expected.items():
            np.testing.assert_array_equal(actual[name][:2], values)
            self.assertTrue(np.isnan(actual[name][2]))

    def test_zero_is_not_positive_reference_evidence(self):
        banks = {name: PeakBank([-np.inf, 0., 1., 1.]) for name in COMPONENTS}
        raw = {name: np.array([0., 1., 2., np.nan]) for name in RAW_NAMES}
        raw["layer_support"] = np.array([0., .5, 1., np.nan])
        values = reference_evidence(raw, banks)
        for name in COMPONENTS[2:]:
            np.testing.assert_allclose(values[name], [0., -np.log(3/5), -np.log(1/5), np.nan], equal_nan=True)
        np.testing.assert_array_equal(values["layer_support"], raw["layer_support"])

    def test_window_mean_expires_and_requires_complete_history(self):
        actual = window_mean(np.array([[4., 0., 0., 0., 0., np.nan, 4., 4., 4., 4.]]))
        np.testing.assert_allclose(actual, [[np.nan, np.nan, np.nan, 1., 0., np.nan, np.nan, np.nan, np.nan, 4.]], equal_nan=True)
        np.testing.assert_array_equal(window_mean(np.zeros((1, 500)))[:, WINDOW-1:], 0.)

    def test_prefix_padding_and_late_relief_availability(self):
        rng = np.random.default_rng(971)
        evidence = {name: rng.uniform(0., 4., (2, 24)) for name in RAW_NAMES}
        evidence["layer_support"] /= 4
        for values in evidence.values():
            values[:, :8] = np.nan
        evidence["decoupling"][:, :11] = np.nan
        evidence["routing_relief"][:, :15] = np.nan
        valid = np.arange(24)[None] < np.array([12, 24])[:, None]
        full = model_scores(evidence, valid)
        np.testing.assert_array_equal(full["relative_recovery"][:, :15], full["relative_curvature"][:, :15])
        for stop in range(1, 25):
            prefix = model_scores({name: value[:, :stop] for name, value in evidence.items()}, valid[:, :stop])
            for name in METHODS:
                np.testing.assert_array_equal(prefix[name], full[name][:, :stop])
        for values in evidence.values():
            values[~valid] = 1e9
        padded = model_scores(evidence, valid)
        for name in METHODS:
            np.testing.assert_array_equal(full[name], padded[name])

    def test_raw_components_batch_and_latched_alarms(self):
        base_names = ("acceleration", "decoupling", *RELATION_NAMES)
        base = dict(schema=BASE_SCHEMA, encoder_config=asdict(EncoderConfig()), checkpoints=["test"], gate_alpha=.1,
                    frozen=dict(freeze_threshold=.57, acceleration_threshold=.1, periodicity_threshold=.37,
                                periodicity_scale=.02, frontback_threshold=-100., curvature_threshold=.389, slope=.0015),
                    banks={kind: {name: np.linspace(-1, 1, 399).tolist() for name in base_names}
                           for kind in ("episode", "task_init")})
        with tempfile.TemporaryDirectory() as directory:
            base_path, path = Path(directory)/"base.json", Path(directory)/"relative.json"
            base_path.write_text(json.dumps(base))
            profile = dict(schema=SCHEMA, methods=list(METHODS), window=WINDOW, base_profile=base_path.name,
                           base_profile_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                           references={name: np.linspace(-1, 1, 500).tolist() for name in COMPONENTS},
                           banks={kind: {name: np.linspace(0, 10, 399).tolist() for name in METHODS}
                                  for kind in ("episode", "task_init")})
            path.write_text(json.dumps(profile))
            live = RelativeModelMonitor(path, "test", method="relative_recovery")
            rng = np.random.default_rng(917)
            raw = [rng.uniform(.001, .2, (8, 10, 11, 32)).astype(np.float32) for _ in range(25)]
            results, features, relations = [], [], []
            for x in raw:
                results.append(live.update(x))
                features.append(live.base.action_recent[-1].copy())
                relations.append(live.base.relation_recent[-1].copy())
            valid = np.ones((1, len(raw)), bool)
            components = make_components(np.asarray(features)[None], np.asarray(relations)[None], valid)
            for name in RAW_NAMES:
                np.testing.assert_array_equal(components[name][0], [x["components"][name] for x in results])
            evidence = {name: np.asarray([[row["evidence"][name] for row in results]]) for name in RAW_NAMES}
            expected = model_scores(evidence, valid)
            for name in METHODS:
                np.testing.assert_array_equal(expected[name][0], [row["scores"][name] for row in results])
            self.assertEqual(results[-1]["frozen_first_query"], 7)
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
            self.assertEqual(len(live.recent_core), 0)


if __name__ == "__main__":
    unittest.main()
