"""Tests for control causality, frozen score equivalence and execution boundaries."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from adaptive_control import (ARMS, RouteRisk, candidate_scores, chunk_limit, events_for,
                              noise_for, seed_for, select_candidate, scheduled_jobs)
from audit_adaptive_control import independent_risk
from analyze_adaptive_control import policy_rows, metric
from collection_routes import PROBS_KEY
from collection_storage import records


class AdaptiveControlTest(unittest.TestCase):
    def test_diagnostic_timing_never_claims_online_control(self):
        events = events_for("m", 18, 52, True)
        self.assertEqual([(e["start_query"], e["deployable"]) for e in events], [(19, True), (13, False)])
        terminal = events_for("m", 51, 52, True)
        self.assertEqual(len(terminal), 1)
        self.assertFalse(terminal[0]["deployable"])
        self.assertEqual(events_for("m", -1, 52, True), [])

    def test_burst_lengths_and_remaining_budget(self):
        self.assertEqual([chunk_limit("short2_5", i, 520) for i in range(7)], [2, 2, 2, 2, 2, 10, 10])
        self.assertEqual(chunk_limit("short2_20", 19, 1), 1)
        self.assertEqual(chunk_limit("short2_20", 20, 100), 10)
        self.assertTrue(all(chunk_limit(a, 30, 100) == 10 for a in ARMS))

    def test_paired_streams_do_not_depend_on_timing_or_execution_order(self):
        expected = noise_for("m", 1, 3, 2)
        for q in range(52):
            noise_for("m", 0, q, 3)
        np.testing.assert_array_equal(expected, noise_for("m", 1, 3, 2))
        self.assertFalse(np.array_equal(expected, noise_for("m", 1, 3, 1)))
        self.assertNotEqual(seed_for("m", 1, 3, "policy"), seed_for("m", 1, 3, "environment"))

    def test_guard_rejects_gripper_change_and_large_action_outlier(self):
        class FakeRisk:
            def preview(self, monitor, probabilities):
                return np.zeros((4, 10)), np.asarray([4., 3., 1., 0.])
        p = np.full((4, 8, 10, 11, 32), 1 / 32, np.float32)
        a = np.zeros((4, 10, 7))
        a[:, :, 6] = 1
        a[1, :, 0] = .1
        a[2, 0, 6] = -1
        a[3, :, :6] = 10
        scores = candidate_scores(p, a, None, FakeRisk())
        np.testing.assert_array_equal(scores["eligible"], [True, True, False, False])
        self.assertEqual(select_candidate("knn", scores), 3)
        self.assertEqual(select_candidate("guarded_knn", scores), 1)
        scores["knn"][:] = np.nan
        self.assertEqual(select_candidate("knn", scores), 0)

    def test_frozen_geometry_matches_existing_offline_algorithm(self):
        audit = json.loads(Path("design/experiment_long_batch1_audit_20260908.json").read_text())
        risk, tested = RouteRisk(), 0
        for task in audit["tasks"][:2]:
            monitor = risk.monitor()
            for row in records(Path(task["directory"]) / "main"):
                monitor.update(row[PROBS_KEY])
                expected, distance = independent_risk(monitor, risk)
                actual, score = risk.current(monitor)
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-9)
                np.testing.assert_allclose(score, distance, rtol=2e-6, atol=1e-7)
                tested += int(monitor.query >= 7)
            before = copy.deepcopy(monitor.__dict__)
            risk.preview(monitor, np.stack([row[PROBS_KEY]] * 4))
            self.assertEqual(monitor.query, before["query"])
            np.testing.assert_array_equal(monitor._previous_action_route, before["_previous_action_route"])
            self.assertEqual(len(monitor._mobility_history), len(before["_mobility_history"]))
        self.assertGreater(tested, 20)

    def test_branch_jobs_depend_on_committed_parent_replay(self):
        task = dict(main_id="parent", variant_id="variant", noise_seed=1, init_index=0,
                    max_output_bytes=1, events=events_for("parent", 10, 52, False))
        plan = dict(tasks=[task], arms=list(ARMS), replicates=2)
        jobs = scheduled_jobs(plan, {"variant": {}}, Path("/tmp/control_test"))
        self.assertEqual(len(jobs), 3)
        self.assertIsNone(jobs[0]["depends_on"])
        self.assertEqual({j["depends_on"] for j in jobs[1:]}, {jobs[0]["job_id"]})
        self.assertEqual(len({j["job_id"] for j in jobs}), 3)
        self.assertTrue(all(j["sampling"]["max_output_bytes"] > 0 for j in jobs))

    def test_candidate_oracle_and_unalarmed_denominators_are_separate(self):
        event = dict(event_id="e", timing="after_alarm", start_query=10)
        branches = [dict(event_id="e", arm="candidate%d" % i, replicate=rep, queries=20,
                        deployment_model_queries=20, success=i == 2 and rep == 0)
                    for rep in range(2) for i in range(4)]
        pools = [dict(event_id="e", replicate=rep, winners=dict(center=1, edge=2, knn=0, guarded_knn=0),
                      candidates=[dict(candidate=i, success=i == 2 and rep == 0) for i in range(4)]) for rep in range(2)]
        tasks = [dict(main_id="m", base_task="task_a", success=False, events=[event], branches=branches,
                      pools=pools, parent_queries=52),
                 dict(main_id="u", base_task="task_b", success=True, events=[], branches=[], pools=[], parent_queries=12)]
        rows = policy_rows(tasks, "after_alarm", "edge_once")
        self.assertEqual([r["success"] for r in rows], [.5, 1.])
        self.assertEqual([r["queries"] for r in rows], [33., 12])
        result = metric(tasks, "after_alarm", "edge_once", "primary")
        self.assertEqual(result["delta_vs_random_pp"], 25.)
        self.assertEqual((result["rescued_repeats"], result["failed_repeats"]), (1, 2))
        self.assertEqual(result["eligible_parents"], 1)
        oracle = metric(tasks, "after_alarm", "oracle_once", "primary")
        self.assertFalse(oracle["deployable"])
        self.assertIsNone(oracle["mean_model_queries"])
        center = metric(tasks, "after_alarm", "center_once", "primary")
        self.assertEqual(center["rescued_repeats"], 0)


if __name__ == "__main__":
    unittest.main()
