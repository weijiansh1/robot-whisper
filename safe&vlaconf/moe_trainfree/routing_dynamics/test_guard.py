"""Rank calibration, causal replay, and alarm bookkeeping checks."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from encoder import EncoderConfig, FEATURE_NAMES, encode_readouts
from guard import (AlarmState, BRANCHES, CONFIRMATIONS, HEADS, KINDS, PeakBank,
                   RoutingGuardMonitor, SCHEMA, combine_tails, dynamics_scores,
                   first_trigger, raw_legacy_features, v82_scores)


def example_profile():
    return dict(schema=SCHEMA, checkpoints=["test"], encoder_config=asdict(EncoderConfig()),
                confirmations=CONFIRMATIONS,
                v82_profile=dict(periodicity_scale=.02,
                                 heads={name: dict(center=0., scale=1.) for name in HEADS}),
                banks={kind: {name: np.linspace(-1, 2, 399).tolist() for name in BRANCHES} for kind in KINDS})


class GuardTests(unittest.TestCase):
    def test_tail_ties_and_missing_exposure(self):
        bank = PeakBank([-np.inf, 0, 0, 1])
        np.testing.assert_allclose(bank.tail([0, .5, 1, 2, np.nan]), [4/5, 2/5, 2/5, 1/5, np.nan])
        np.testing.assert_array_equal(PeakBank.from_list(bank.to_list()).peaks, bank.peaks)
        for peaks in ([], [np.nan], [np.inf], [[1]]):
            with self.assertRaises(ValueError):
                PeakBank(peaks)

    def test_rank_threshold_matches_p_domain_with_ties(self):
        rng = np.random.default_rng(81)
        for size in (398, 400, 3097):
            bank = PeakBank(np.r_[-np.inf, np.round(rng.normal(size=size - 1), 2)])
            candidates = np.r_[bank.peaks[1:], bank.peaks[1:] + 1e-7, 10.]
            for alpha in (.005, .01, .02, .05):
                for k in (1, 2, 3):
                    threshold, _ = bank.threshold(alpha / k)
                    np.testing.assert_array_equal(bank.tail(candidates) * k <= alpha, candidates > threshold)

    def test_resolution_and_warmup_keep_fixed_multiplicity(self):
        bank = PeakBank(np.arange(398))
        tail = float(bank.tail(10000))
        result = combine_tails(dict(v82=tail, acceleration=np.nan, decoupling=np.nan), "v82_integrated")
        self.assertAlmostEqual(float(result), 3/399)
        self.assertGreater(result, .005)
        self.assertLessEqual(result, .01)
        self.assertTrue(np.isinf(bank.threshold(.005/3)[0]))
        self.assertTrue(np.isnan(combine_tails(dict.fromkeys(BRANCHES, np.nan), "v82_integrated")))

    def test_confirmation_padding_and_prefix(self):
        rng = np.random.default_rng(31)
        readouts = np.exp(rng.normal(-4, .3, (2, 23, 25)))
        valid = np.arange(23)[None] < np.asarray([13, 23])[:, None]
        features, _ = encode_readouts(readouts, valid)
        result = dynamics_scores(features, valid)
        self.assertTrue(np.isnan(result["acceleration"][:, :8]).all())
        self.assertTrue(np.isnan(result["decoupling"][:, :11]).all())
        changed = features.copy()
        changed[~valid] = 1e9
        for name, values in dynamics_scores(changed, valid).items():
            np.testing.assert_array_equal(values, result[name])
            for q in range(1, 24):
                short = dynamics_scores(features[:, :q], valid[:, :q])
                np.testing.assert_array_equal(short[name], values[:, :q])
        changed[1, 14, FEATURE_NAMES.index("acceleration_log_ratio")] = np.nan
        score = dynamics_scores(changed, valid)["acceleration"]
        self.assertTrue(np.isnan(score[1, 14:16]).all())
        self.assertTrue(np.isfinite(score[1, 16]))

    def test_alarm_clear_keeps_historical_false_alarm(self):
        state = AlarmState(.01)
        samples = [np.nan, .05, .01, .04, .5, np.nan, .5, .5, .001]
        results = [state.update(x) for x in samples]
        self.assertEqual([x["state"] for x in results],
                         ["NORMAL", "WATCH", "ALARM", "ALARM", "ALARM", "ALARM", "ALARM", "NORMAL", "ALARM"])
        self.assertTrue(results[7]["signal_cleared"])
        self.assertTrue(all(x["ever_alarm"] and x["first_alarm_query"] == 2 for x in results[2:]))
        first = first_trigger(np.asarray(samples)[None], np.ones((1, len(samples)), bool), .01)
        self.assertEqual(first[0], 2)
        state.reset()
        self.assertFalse(state.update(.5)["ever_alarm"])

    def test_legacy_every_prefix_and_padding(self):
        rng = np.random.default_rng(73)
        cache = dict(mobility=rng.uniform(.04, .13, (2, 25, 8)).astype(np.float32),
                     acceleration=rng.uniform(.005, .02, (2, 25)).astype(np.float32),
                     periodicity=rng.normal(0, .02, (2, 25)).astype(np.float32),
                     valid=np.ones((2, 25), bool))
        cache["mobility"][:, 0] = np.nan
        cache["periodicity"][:, :2] = np.nan
        extra = rng.uniform(.01, .2, (2, 25, 2)).astype(np.float32)
        profile = example_profile()["v82_profile"]
        full = v82_scores(cache, extra, profile)
        for q in range(7, 26):
            short = v82_scores({k: v[:, :q] for k, v in cache.items()}, extra[:, :q], profile)
            np.testing.assert_array_equal(full[:, :q], short)
        cache["valid"][0, 15:] = False
        first = v82_scores(cache, extra, profile)
        for name in ("mobility", "acceleration", "periodicity"):
            cache[name][~cache["valid"]] = 1e8
        extra[~cache["valid"]] = 1e8
        second = v82_scores(cache, extra, profile)
        np.testing.assert_array_equal(first, second)

    def test_raw_api_matches_direct_raw_replay_and_reset(self):
        rng = np.random.default_rng(88)
        profile = example_profile()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile))
            live = RoutingGuardMonitor(path, "test")
            streams = {name: [] for name in ("mobility", "acceleration", "periodicity")}
            extra, history, previous = [], [], None
            observed = []
            for q in range(19):
                raw = rng.uniform(.01, .08, (8, 10, 11, 32)).astype(np.float32)
                result = live.update(raw)
                m, a, p, e, previous = raw_legacy_features(raw, previous, history)
                history.append(previous[4:].reshape(40, 32))
                history = history[-4:]
                for name, value in zip(streams, (m, a, p)):
                    streams[name].append(value)
                extra.append(e)
                observed.append(result)
            cache = {k: np.asarray(v, np.float32)[None] for k, v in streams.items()}
            cache["valid"] = np.ones((1, len(extra)), bool)
            expected = v82_scores(cache, np.asarray(extra)[None], profile["v82_profile"])[0]
            np.testing.assert_array_equal([r["branch_scores"]["v82"] for r in observed], expected)
            feature_values = np.asarray([[r["features"][name] for name in FEATURE_NAMES] for r in observed], np.float32)
            dynamics = dynamics_scores(feature_values[None], cache["valid"])
            for name in BRANCHES[1:]:
                np.testing.assert_array_equal([r["branch_scores"][name] for r in observed], dynamics[name][0])
            before = live.alert.query
            with self.assertRaises(ValueError):
                live.update(np.zeros_like(raw))
            self.assertEqual(live.alert.query, before)
            live.reset()
            self.assertFalse(live.update(raw)["ready"])


if __name__ == "__main__":
    unittest.main()
