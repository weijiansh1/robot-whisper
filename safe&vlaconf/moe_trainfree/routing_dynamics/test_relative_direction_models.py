"""Reverse/two-sided relative constraints, confirmation, and online replay."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from encoder import EncoderConfig, FEATURE_NAMES
from guard import PeakBank
from relative_direction_models import (COMPONENTS, METHODS, SCHEMA, WINDOW, RelativeDirectionMonitor,
                                       direction_components, instant_models, make_components, model_scores, reference_evidence)
from state_action import RELATION_NAMES, SCHEMA as BASE_SCHEMA


class RelativeDirectionTests(unittest.TestCase):
    def test_direction_and_common_change_cancels(self):
        features = np.zeros((4, len(FEATURE_NAMES)))
        features[:, FEATURE_NAMES.index("acceleration_log_ratio")] = [2., 1., 0., -2.]
        features[:, FEATURE_NAMES.index("path_log_ratio")] = [1., 2., 0., -1.]
        current = direction_components(features)
        np.testing.assert_array_equal(current["curvature_low"], [0., 1., 0., 1.])
        np.testing.assert_array_equal(current["curvature_magnitude"], [1., 1., 0., 1.])
        for name in ("acceleration_log_ratio", "path_log_ratio"):
            features[:, FEATURE_NAMES.index(name)] += 4
        for name, value in direction_components(features).items():
            np.testing.assert_array_equal(value, current[name])

    def test_confirmation_precedes_direction_aggregation(self):
        features = np.zeros((1, 4, len(FEATURE_NAMES)))
        features[0, :, FEATURE_NAMES.index("path_log_ratio")] = [1., -1., 1., 1.]
        components = make_components(features, np.ones((1,4), bool))
        np.testing.assert_allclose(components["curvature_low"], [[np.nan, 0., 0., 1.]], equal_nan=True)
        np.testing.assert_allclose(components["curvature_magnitude"], [[np.nan, 1., 1., 1.]], equal_nan=True)

    def test_formulas(self):
        evidence = dict(acceleration=np.array([8., 8., np.nan]), decoupling=np.array([1., 4., np.nan]),
                        curvature_low=np.array([2., 2., 8.]), curvature_magnitude=np.array([6., 3., 8.]))
        actual = instant_models(evidence)
        for name, values in dict(ac_reference=[8.,8.], relative_low=[2.,4.], relative_magnitude=[6.,4.], relative_low_soft=[4.,4.]).items():
            np.testing.assert_array_equal(actual[name][:2], values)
            self.assertTrue(np.isnan(actual[name][2]))

    def test_zero_and_missing_reference_evidence(self):
        banks = {name: PeakBank([-np.inf, 0., 1., 1.]) for name in COMPONENTS}
        raw = {name: np.array([0., 1., 2., np.nan]) for name in COMPONENTS}
        evidence = reference_evidence(raw, banks)
        for name in COMPONENTS[2:]:
            np.testing.assert_allclose(evidence[name], [0., -np.log(3/5), -np.log(1/5), np.nan], equal_nan=True)

    def test_prefix_and_padding(self):
        rng = np.random.default_rng(187)
        evidence = {name: rng.uniform(0.,4.,(2,20)) for name in COMPONENTS}
        for value in evidence.values():
            value[:,:8] = np.nan
        evidence["decoupling"][:,:11] = np.nan
        valid = np.arange(20)[None] < np.array([12,20])[:,None]
        full = model_scores(evidence, valid)
        for stop in range(1,21):
            current = model_scores({name: value[:,:stop] for name,value in evidence.items()}, valid[:,:stop])
            for name in METHODS:
                np.testing.assert_array_equal(current[name], full[name][:,:stop])
        for value in evidence.values():
            value[~valid] = 1e9
        for name, value in model_scores(evidence, valid).items():
            np.testing.assert_array_equal(value, full[name])

    def test_raw_monitor_components_and_alarm_latch(self):
        names = ("acceleration", "decoupling", *RELATION_NAMES)
        base = dict(schema=BASE_SCHEMA, encoder_config=asdict(EncoderConfig()), checkpoints=["test"], gate_alpha=.1,
                    frozen=dict(freeze_threshold=.57, acceleration_threshold=.1, periodicity_threshold=.37,
                                periodicity_scale=.02, frontback_threshold=-100., curvature_threshold=.389, slope=.0015),
                    banks={kind:{name:np.linspace(-1,1,399).tolist() for name in names} for kind in ("episode","task_init")})
        with tempfile.TemporaryDirectory() as directory:
            base_path, path = Path(directory)/"base.json", Path(directory)/"direction.json"
            base_path.write_text(json.dumps(base))
            profile = dict(schema=SCHEMA, methods=list(METHODS), window=WINDOW, base_profile=base_path.name,
                           base_profile_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                           references={name:np.linspace(-1,1,500).tolist() for name in COMPONENTS},
                           banks={kind:{name:np.linspace(0,10,399).tolist() for name in METHODS} for kind in ("episode","task_init")})
            path.write_text(json.dumps(profile))
            live = RelativeDirectionMonitor(path,"test")
            rng = np.random.default_rng(178)
            results, features = [], []
            for _ in range(25):
                results.append(live.update(rng.uniform(.001,.2,(8,10,11,32)).astype(np.float32)))
                features.append(live.base.action_recent[-1].copy())
            valid = np.ones((1,len(results)),bool)
            components = make_components(np.asarray(features)[None], valid)
            for name in COMPONENTS:
                np.testing.assert_array_equal(components[name][0], [row["components"][name] for row in results])
            evidence = {name:np.asarray([[row["evidence"][name] for row in results]]) for name in COMPONENTS}
            for name, values in model_scores(evidence,valid).items():
                np.testing.assert_array_equal(values[0], [row["scores"][name] for row in results])
            self.assertEqual(results[-1]["frozen_first_query"],7)
            for before,after in zip(results,results[1:]):
                if before["ever_alarm"]:
                    self.assertTrue(after["ever_alarm"])
                    self.assertEqual(before["first_alarm_query"],after["first_alarm_query"])
            query = live.base.query
            with self.assertRaises(ValueError):
                live.update(np.zeros((8,10,11,32)))
            self.assertEqual(live.base.query,query)
            live.reset()
            self.assertEqual(live.first_alarm,-1)


if __name__ == "__main__":
    unittest.main()
