"""Causality, direction, calibration gate, and immutable alarm history checks."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from encoder import EncoderConfig, encode_readouts, query_readout
from guard import PeakBank, dynamics_scores, raw_legacy_features
from state_action import (GATE_ALPHA, RELATION_NAMES, SCHEMA, RelationEncoder, StateActionGuardMonitor,
                          confirmation_tail, confirmed_relations, encode_relations, frozen_triggers, state_readout)


def profile():
    names = ("acceleration", "decoupling", *RELATION_NAMES)
    return dict(schema=SCHEMA, encoder_config=asdict(EncoderConfig()), checkpoints=["test"], gate_alpha=GATE_ALPHA,
                frozen=dict(freeze_threshold=.57, acceleration_threshold=.1, periodicity_threshold=.37,
                            periodicity_scale=.02, frontback_threshold=0., curvature_threshold=.389, slope=.0015),
                banks={kind: {name: np.linspace(-.5, .5, 399).tolist() for name in names}
                       for kind in ("episode", "task_init")})


class StateActionTests(unittest.TestCase):
    def test_state_token_isolation_and_known_distance(self):
        raw = np.zeros((8, 10, 11, 32), np.float32)
        raw[..., 0] = 1
        missing, previous = state_readout(raw)
        self.assertTrue(np.isnan(missing).all())
        raw[:, :, 1:, 0], raw[:, :, 1:, 1] = 0, 1
        movement, _ = state_readout(raw, previous)
        np.testing.assert_array_equal(movement, 0)
        raw[:, :, 0, 0], raw[:, :, 0, 1] = 0, 1
        movement, _ = state_readout(raw, previous)
        np.testing.assert_array_equal(movement, 1)

    def test_relation_direction_scale_and_constant(self):
        ac, st = np.full((2, 20, 8), .05), np.full((2, 20, 8), .2)
        ac[0, 5:] *= .8
        st[0, 5:] *= .3
        valid = np.ones((2, 20), bool)
        values = encode_relations(ac, st, valid)
        self.assertTrue((values[0, 7:, 0] > 0).all())
        np.testing.assert_array_equal(values[1, 7:, :2], 0.)
        zero = encode_relations(np.zeros_like(ac), np.zeros_like(st), valid)
        np.testing.assert_array_equal(zero[:, 7:], 0.)
        opposite = encode_relations(st, ac, valid)
        np.testing.assert_allclose(values[..., 0], -opposite[..., 0], equal_nan=True)

    def test_streaming_prefix_padding_and_invalid_inputs(self):
        rng = np.random.default_rng(409)
        ac = rng.uniform(.01, .1, (2, 22, 8))
        st = rng.uniform(.01, .2, (2, 22, 8))
        ac[:, 0], st[:, 0] = np.nan, np.nan
        valid = np.arange(22)[None] < np.array([13, 22])[:, None]
        batch = encode_relations(ac, st, valid)
        for i, length in enumerate(valid.sum(1)):
            live = RelationEncoder()
            observed = np.asarray([live.update(ac[i, q], st[i, q]) for q in range(length)])
            np.testing.assert_array_equal(observed, batch[i, :length])
            live.reset()
            self.assertTrue(np.isnan(live.update(ac[i, 0], st[i, 0])).all())
        for stop in range(1, 23):
            np.testing.assert_array_equal(encode_relations(ac[:, :stop], st[:, :stop], valid[:, :stop]), batch[:, :stop])
        ac[~valid], st[~valid] = 1e8, -1e8
        np.testing.assert_array_equal(encode_relations(ac, st, valid), batch)
        st[1, 10] = np.nan
        with self.assertRaises(ValueError):
            encode_relations(ac, st, valid)

    def test_confirmation_is_same_query_subset_and_sign_sensitive(self):
        tails = dict(acceleration=np.array([.001, .001, .001, .5, .001]), decoupling=np.full(5, np.nan))
        gate = np.array([.1, .10001, .001, .01, np.nan])
        score = np.array([1., 1., 0., 1., np.nan])
        out = confirmation_tail(tails, gate, score)
        np.testing.assert_allclose(out, [.002, 1., 1., 1., np.nan], equal_nan=True)
        self.assertFalse(((out <= .01) & ~(2 * tails["acceleration"] <= .01)).any())

    def test_raw_monitor_replay_reset_and_history(self):
        rng = np.random.default_rng(890)
        raw = rng.uniform(.001, .2, (22, 8, 10, 11, 32)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile()))
            live = StateActionGuardMonitor(path, "test", alpha=.05)
            results = [live.update(x) for x in raw]
            readouts, states, previous, prev_state = [], [], None, None
            legacy = {name: [] for name in ("mobility", "acceleration", "periodicity")}
            extra, history = [], []
            for x in raw:
                values, final = query_readout(x, previous)
                state, prev_state = state_readout(x, prev_state)
                m, a, p, e, _ = raw_legacy_features(x, previous, history)
                for name, value in zip(legacy, (m, a, p)):
                    legacy[name].append(value)
                history.append(final[4:].reshape(40, 32))
                history = history[-4:]
                readouts.append(values)
                states.append(state)
                extra.append(e)
                previous = final
            valid = np.ones((1, len(raw)), bool)
            features, _ = encode_readouts(np.asarray(readouts)[None], valid)
            relation = encode_relations(np.asarray(readouts)[None, :, :8], np.asarray(states)[None], valid)
            scores = dict(**dynamics_scores(features.astype(np.float32), valid), **confirmed_relations(relation, valid))
            for name, values in scores.items():
                np.testing.assert_array_equal([r["scores"][name] for r in results], values[0])
            cache = {name: np.asarray(value, np.float32)[None] for name, value in legacy.items()}
            cache["valid"] = valid
            hits = frozen_triggers(cache, np.asarray(extra, np.float32)[None], profile()["frozen"])[0]
            expected = np.flatnonzero(hits)
            self.assertEqual(live.first["v82"], int(expected[0]) if len(expected) else -1)
            tails = {name: PeakBank(profile()["banks"]["task_init"][name]).tail(value) for name, value in scores.items()}
            confirmed = confirmation_tail(tails, tails["gap"], scores["gap"])[0] <= .05
            all_hits = np.flatnonzero(hits | confirmed)
            self.assertEqual(live.first["combined"], int(all_hits[0]) if len(all_hits) else -1)
            for before, after in zip(results, results[1:]):
                if before["ever_alarm"]:
                    self.assertTrue(after["ever_alarm"])
                    self.assertEqual(before["first_alarm_query"], after["first_alarm_query"])
            before = live.query
            with self.assertRaises(ValueError):
                live.update(np.zeros_like(raw[0]))
            self.assertEqual(live.query, before)
            live.reset()
            self.assertFalse(live.update(raw[0])["ever_alarm"])


if __name__ == "__main__":
    unittest.main()
