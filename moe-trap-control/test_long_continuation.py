"""Tests for causal positions, isolated random streams and peripheral selection."""

import unittest

import numpy as np

from continuation_experiment import METHODS, events_for, noise_for, seed_for, select_edge, short_count
from audit_long_continuation import independent_edge
from collection_routes import PROBS_KEY
from analyze_long_continuation import cluster_interval, estimates


class ContinuationTest(unittest.TestCase):
    def test_positions_deduplicate_and_exclude_terminal_query(self):
        events, terminal = events_for("parent", dict(zip(METHODS, (12, 10, 10, 51))), 52)
        self.assertEqual([e["start_query"] for e in events], [11, 13])
        self.assertEqual(events[0]["methods"], [METHODS[1], METHODS[2]])
        self.assertEqual(terminal, [METHODS[3]])
        self.assertEqual(events_for("parent", dict.fromkeys(METHODS, -1), 4), ([], []))

    def test_random_streams_do_not_depend_on_branch_length(self):
        expected = noise_for("parent", "event", 2, 19)
        for q in range(100):
            for i in range(4):
                noise_for("parent", "event", 1, q, i)
                short_count("parent", "event", 2, q)
        np.testing.assert_array_equal(expected, noise_for("parent", "event", 2, 19))
        self.assertFalse(np.array_equal(expected, noise_for("parent", "event", 2, 19, 1)))
        self.assertNotEqual(seed_for("parent", "event", 2, 19, "policy"), seed_for("parent", "event", 2, 19, "environment"))
        lengths = {short_count("parent", "event", 0, q) for q in range(100)}
        self.assertEqual(lengths, set(range(1, 10)))

    def test_route_periphery_selects_outlier_not_center(self):
        p = np.ones((4, 8, 10, 11, 32), np.float32)
        p[3, 4:, :3, 1:, 0] = 100
        p /= p.sum(-1, keepdims=True)
        index, scores = select_edge(p)
        self.assertEqual(index, 3)
        scalar_index, scalar_scores = independent_edge([{PROBS_KEY: row} for row in p])
        self.assertEqual(index, scalar_index)
        np.testing.assert_allclose(scores, scalar_scores, atol=1e-12)
        p[:, 4:, :3, 1:] = 1 / 32
        self.assertEqual(select_edge(p)[0], 0)
        p[2, :4, :, :, 0] = 999
        self.assertEqual(select_edge(p)[0], 0)

    def test_degenerate_effects_do_not_claim_zero_width_confidence(self):
        self.assertIsNone(cluster_interval(np.zeros(4), ["a", "a", "b", "b"]))

    def test_cluster_interval_does_not_treat_repeats_as_new_tasks(self):
        values, groups = np.asarray([0., .5, -.25, .25]), ["a", "a", "b", "c"]
        np.testing.assert_array_equal(cluster_interval(values, groups),
            cluster_interval(np.repeat(values, 4), list(np.repeat(groups, 4))))

    def test_policy_rates_include_unalarmed_parents_and_candidate_costs(self):
        branches = [dict(event_id="e", arm=arm, queries=20, success=rep < successes)
                    for arm, successes in (("random", 1), ("edge_noise", 2)) for rep in range(4)]
        tasks = [dict(success=False, events=[dict(event_id="e", start_query=10, methods=[METHODS[0]])], branches=branches),
                 dict(success=True, events=[], branches=[], c0_queries=20)]
        rates, baseline, queries = estimates(tasks, METHODS[0], "edge_noise")
        np.testing.assert_array_equal(rates, [.5, 1])
        np.testing.assert_array_equal(baseline, [.25, 1])
        np.testing.assert_array_equal(queries, [33, 20])


if __name__ == "__main__":
    unittest.main()
