#!/usr/bin/env python3
"""Validate the curated train-free Trap experiment bundle."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from trainfree_belief_selector import candidate_gap_scores, select_min_gap
from validate_moe_invariant_alarm_cache_new import (
    validate as validate_moe_invariant_alarm,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OFFLINE = PACKAGE_ROOT / "results/trainfree_signal_matrix"
RECOVERY = PACKAGE_ROOT / "results/snapshot_fork_recovery"
FAILED_GRASP = PACKAGE_ROOT / "results/failed_grasp_moe_dynamics"
BELIEF_MISMATCH = PACKAGE_ROOT / "results/belief_state_mismatch"
BELIEF_SELECTOR = PACKAGE_ROOT / "results/trainfree_belief_selector"
ONLINE_ALARM = PACKAGE_ROOT / "results/online_belief_alarm"
MOE_ONLY_ONLINE = PACKAGE_ROOT / "results/moe_only_online_alarm"
MOE_DYNAMICS = PACKAGE_ROOT / "results/moe_dynamics_online_alarm"
TASK_DIFFICULTY = PACKAGE_ROOT / "results/task_difficulty_weighting"
SELF_REFERENCE = PACKAGE_ROOT / "results/task_free_self_reference_selector"
INPUT_VERSION = PACKAGE_ROOT / "results/input_version_counterfactual"
TRAP_PROBABILITY = PACKAGE_ROOT / "results/trainfree_trap_probability"


def read_csv(name: str) -> list[dict[str, str]]:
    with (OFFLINE / "tables" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_offline() -> list[tuple[str, str, int, str]]:
    summary = json.loads((OFFLINE / "summary.json").read_text(encoding="utf-8"))
    assert summary["training"] is False
    assert summary["feature_labels_used"] is False
    assert summary["n_permutations"] == 2000
    assert len(read_csv("fixed_time_auc.csv")) == 48
    assert len(read_csv("onset_alignment.csv")) == 1092
    rows = read_csv("type_onset_alignment.csv")
    assert len(rows) == 1092

    passed: dict[str, set[tuple[str, str, int, str]]] = {}
    for row in rows:
        if row["variant"] != "residual_route_mobility" or not row["p_maxT"]:
            continue
        if float(row["p_maxT"]) > 0.05:
            continue
        key = (
            row["event_kind"],
            row["signal"],
            int(row["relative"]),
            row["direction"],
        )
        passed.setdefault(row["corpus"], set()).add(key)
    replicated = sorted(passed["A"] & passed["B"])
    expected = sorted([
        ("loop", "late_flow_volatility", -2, "event_high"),
        ("loop", "route_acceleration", -2, "event_high"),
        ("static", "lag_periodicity", 0, "event_high"),
    ])
    assert replicated == expected
    return replicated


def validate_recovery() -> dict[str, int]:
    run = RECOVERY / "runs/init03_seed20260903_loop"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["design"]["training"] is False
    assert manifest["trunk"]["loop_onset_query"] == 36

    candidate_count = 0
    inference_count = int(manifest["trunk"]["queries"])
    noise_by_candidate: dict[int, np.ndarray] = {}
    for offset, query in [(-4, 32), (-2, 34), (0, 36)]:
        result = manifest["forks"][str(offset)]
        assert result["query"] == query
        assert result["n"] == 8 and result["successes"] == 0
        fork = run / f"fork_{offset:+d}"
        for candidate in range(8):
            stem = f"candidate_{candidate:02d}"
            record = json.loads((fork / f"{stem}.json").read_text(encoding="utf-8"))
            assert record["candidate"] == candidate
            assert record["fork_query"] == query
            assert record["noise_stream_key"] == 9731
            assert record["success"] is False
            with np.load(fork / f"{stem}.npz") as archive:
                noise = archive["flow_noise"].copy()
                assert len(noise) == record["inference_calls"]
                assert len(archive["action_chunks"]) == record["inference_calls"]
            previous = noise_by_candidate.setdefault(candidate, noise)
            common = min(len(previous), len(noise))
            assert np.array_equal(previous[:common], noise[:common])
            inference_count += int(record["inference_calls"])
            candidate_count += 1

    capture = json.loads(
        (RECOVERY / "capture/capture_summary.json").read_text(encoding="utf-8")
    )
    assert capture["control_steps"] == inference_count == 484
    assert capture["hook_verify_failures"] == []
    assert capture["store_full_probs"] is True
    assert capture["return_full_probs"] is True
    return {"candidates": candidate_count, "captured_inferences": inference_count}


def validate_legacy() -> dict[str, int]:
    root = PACKAGE_ROOT / "results/legacy_d9_baselines"
    route = json.loads(
        (root / "route_change_soft/summary.json").read_text(encoding="utf-8")
    )
    churn = json.loads(
        (root / "churn_onset/summary.json").read_text(encoding="utf-8")
    )
    onset = json.loads(
        (root / "physical_loop_onset/summary.json").read_text(encoding="utf-8")
    )
    assert len(route["per_query"]) == 24
    assert churn["pooled"]["branches"] == 352
    assert onset["branches"] == 352
    assert onset["branches_with_loop_onset"] == 172
    return {"legacy_d9_queries": 24, "legacy_onset_branches": 352}


def validate_failed_grasp() -> dict[str, int]:
    summary = json.loads((FAILED_GRASP / "summary.json").read_text(encoding="utf-8"))
    assert summary["training"] is False
    assert summary["failed_event"]["candidate"] == 6
    assert summary["failed_event"]["query"] == 31
    assert summary["failed_event"]["action_position_1based"] == 8
    assert summary["architecture"]["per_query_raw_vision_captured"] is False
    assert summary["failed_event"]["subsequent_max_lift_m"] == 0.0
    assert summary["success_controls"]["n"] == 7
    assert summary["architecture"]["hidden_state_captured"] is False
    assert summary["as_routing"]["rows_audited"] == 16180
    assert summary["as_routing"]["unique_probability_rows"] == 1
    assert summary["as_routing"]["whole_corpus_max_probability_span"] == 0.0

    table = FAILED_GRASP / "tables"
    expected_rows = {
        "physical_event_alignment.csv": 8,
        "action_event_summary.csv": 8,
        "action_token_event.csv": 80,
        "aligned_physical_state.csv": 136,
        "as_route_state.csv": 4,
        "hb_event_layer_metrics.csv": 64,
        "hb_event_group_metrics.csv": 16,
        "hb_event_group_summary.csv": 18,
        "hb_flow_token_distance.csv": 200,
        "hb_instantaneous_expert_paths.csv": 80,
        "aligned_chunk_dynamics.csv": 272,
    }
    for name, expected in expected_rows.items():
        with (table / name).open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == expected, name

    for group in ("front", "back"):
        comparison = summary["hb_event_comparison"][group]
        for metric in ("within_flow_wj", "late_flow_wj", "route_acceleration"):
            assert comparison[metric]["relation_to_success_range"] == "above_all"
        assert comparison["chunk_jump_full_h"]["relation_to_success_range"] == "below_all"
        assert comparison["event_to_success_center_h"]["relation_to_success_range"] == "above_all"
    continuous = summary["hb_continuous_comparison"]["back"]
    for relative in ("1", "2"):
        assert continuous[relative]["chunk_jump_full_h"]["relation_to_success_range"] == "above_all"

    action = summary["action_event"]["failed_gripper_command_by_token"]
    assert action[0] < 0 and action[1] < 0
    assert all(value > 0 for value in action[2:])
    return {"failed_grasp_cases": 1, "failed_grasp_success_controls": 7}


def validate_belief_mismatch() -> dict[str, int]:
    summary = json.loads(
        (BELIEF_MISMATCH / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["training"] is False
    assert summary["case"]["failed_candidate"] == 6
    assert summary["case"]["episode_id"] == 100000006
    assert summary["case"]["matched_success_controls"] == 7
    assert summary["case"]["belief_state_is_behavioral_inference"] is True
    assert summary["first_clear_mismatch_query"] == 2

    q2 = summary["query_plus_2"]
    assert q2["physics_eef_displacement"]["relation_to_success_range"] == "above_all"
    assert q2["physics_pot_displacement"]["relation_to_success_range"] == "below_all"
    assert q2["action_phase"]["relation_to_success_range"] == "inside"
    assert q2["front_state_route"]["relation_to_success_range"] == "above_all"
    assert q2["front_action_route"]["relation_to_success_range"] == "inside"
    assert q2["back_state_route"]["relation_to_success_range"] == "above_all"
    assert q2["back_action_route"]["relation_to_success_range"] == "above_all"
    assert q2["front_state_action_gap"]["relation_to_success_range"] == "above_all"

    axes = summary["query_plus_2_axis_profiles"]
    assert axes["flow_front_state"]["n_above"] == 10
    assert axes["flow_front_action"]["n_above"] == 0
    assert axes["flow_front_state_action_gap"]["n_above"] == 10
    assert axes["flow_back_state"]["n_above"] == 10
    assert axes["flow_back_action"]["n_above"] == 10
    assert axes["flow_back_state_action_gap"]["n_above"] == 0

    patterns = summary["persistent_patterns"]
    assert patterns["front_state_action_gap_above_all"] == list(range(2, 9))
    assert patterns["front_gap_above_all_queries_by_layer"]["5"] == list(range(2, 9))
    assert patterns["action_inside_success_range"][-7:] == list(range(2, 9))
    assert summary["delayed_response"]["front_chunk_jump_plus_1"][
        "relation_to_success_range"
    ] == "above_all"
    for relative in ("back_chunk_jump_plus_1", "back_chunk_jump_plus_2"):
        assert summary["delayed_response"][relative][
            "relation_to_success_range"
        ] == "above_all"

    external = summary["external_evidence_audit"]
    assert external["grasp_fail_ladder"]["global_gate"] == "FAIL"
    assert external["post_loss_recurrence"]["global_gate"] == "fail"

    table = BELIEF_MISMATCH / "tables"
    expected_rows = {
        "belief_physics_alignment.csv": 104,
        "belief_physics_summary.csv": 65,
        "action_phase_consistency.csv": 13,
        "hb_token_match_to_success.csv": 52,
        "hb_flow_token_match_to_success.csv": 520,
        "hb_flow_state_action_gap.csv": 260,
        "hb_action_position_match_to_success.csv": 260,
        "hb_state_action_gap_layer.csv": 104,
        "hb_state_action_gap_group.csv": 26,
        "hb_phase_tracking.csv": 52,
        "hb_cross_chunk_response.csv": 26,
    }
    for name, expected in expected_rows.items():
        with (table / name).open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == expected, name
    return {"belief_mismatch_cases": 1, "belief_mismatch_success_controls": 7}


def validate_belief_selector() -> dict[str, int | float]:
    summary = json.loads(
        (BELIEF_SELECTOR / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["training"] is False
    assert summary["failure_labels_used_by_selector"] is False
    assert summary["healthy_reference_used_for_calibration"] is True
    assert summary["healthy_reference_requires_known_normal_demonstrations"] is True
    trigger = summary["trigger"]
    assert trigger["name"] == "transition_split_v1"
    assert trigger["failed_case_alarm_queries"] == [2]
    assert trigger["healthy_loo_decisions"] == 56
    assert trigger["healthy_loo_alarm_count"] == 0
    assert trigger["healthy_loo_trajectories_with_alarm"] == 0
    q2 = trigger["query_plus_2"]
    assert q2["reject_stale_chunk"] is True
    assert q2["layer5_gap_ratio"] > 1.0
    assert q2["back_chunk_jump_ratio"] > 1.0
    assert q2["front_action_distance_ratio"] <= 1.0

    candidate = summary["generic_candidate_ranker"]
    assert candidate["tasks"] == 5
    assert candidate["states"] == 80
    assert candidate["pools"] == 320
    assert candidate["informative_pools"] == 86
    assert candidate["state_token_candidate_max_probability_range"] == 0.0
    assert candidate["selected_success"] < candidate["random_expected_success"]
    assert np.isclose(candidate["delta_vs_random"], -0.011328125)
    assert candidate["state_cluster_bootstrap_ci95"][0] < 0.0
    assert candidate["state_cluster_bootstrap_ci95"][1] > 0.0
    assert candidate["permutation_p_improvement"] > 0.05
    assert candidate["decision"].startswith("negative_control_failed")

    expected_rows = {
        "trigger_decisions.csv": 64,
        "candidate_pool_choices.csv": 320,
        "candidate_task_summary.csv": 5,
    }
    for name, expected in expected_rows.items():
        with (BELIEF_SELECTOR / "tables" / name).open(
            newline="", encoding="utf-8"
        ) as stream:
            assert len(list(csv.DictReader(stream))) == expected, name

    rng = np.random.default_rng(20260903)
    routes = rng.random((5, 8, 10, 11, 32), dtype=np.float64)
    ids = np.asarray([19, 3, 41, 7, 11])
    chosen = select_min_gap(routes, ids)
    permutation = np.asarray([3, 0, 4, 1, 2])
    assert select_min_gap(routes[permutation], ids[permutation]) == chosen
    scores = candidate_gap_scores(routes)
    assert scores.shape == (5,) and np.all(np.isfinite(scores))
    return {
        "belief_selector_trigger_alarms": trigger["failed_case_alarm_count"],
        "belief_selector_candidate_pools": candidate["pools"],
        "belief_selector_candidate_delta": candidate["delta_vs_random"],
    }


def validate_online_alarm() -> dict[str, int | bool]:
    summary = json.loads((ONLINE_ALARM / "summary.json").read_text(encoding="utf-8"))
    assert summary["training"] is False
    assert summary["selector_thresholds_changed_after_online_results"] is False
    assert summary["deployment_status"] == "not_reliable_as_a_standalone_online_alarm"
    fresh = summary["fresh_random_test"]
    assert fresh == {
        "episodes": 4,
        "successes": 1,
        "failures": 3,
        "closure_eligible_episodes": 2,
        "alarm_episodes": 2,
        "alarm_success_episodes": 1,
        "alarm_failure_episodes": 1,
        "out_of_scope_no_closure_failures": 2,
        "confirmed_failed_grasp_alarm_episodes": 1,
    }
    capture = summary["capture_audit"]
    assert capture["server_rows"] == capture["client_rows"] == 233
    assert capture["client_server_full_routes_exact"] is True
    assert capture["route_shape"] == [233, 8, 10, 11, 32]
    assert capture["control_steps_strictly_sequential"] is True
    assert capture["hook_verify_failures"] == []
    assert len(capture["spans"]) == 5

    expected_rows = {"episode_summary.csv": 5, "alarm_events.csv": 5}
    for name, expected in expected_rows.items():
        with (ONLINE_ALARM / "tables" / name).open(
            newline="", encoding="utf-8"
        ) as stream:
            assert len(list(csv.DictReader(stream))) == expected, name

    with np.load(BELIEF_SELECTOR / "online_healthy_reference.npz", allow_pickle=False) as archive:
        assert archive["schema"].item() == "himoe.online_healthy_reference.v1"
        assert bool(archive["training"].item()) is False
        assert bool(archive["failure_labels_used"].item()) is False
        assert archive["route_banks"].shape == (8, 7, 8, 10, 11, 32)

    return {
        "online_fresh_episodes": fresh["episodes"],
        "online_alarm_episodes": fresh["alarm_episodes"],
        "online_server_route_rows": capture["server_rows"],
        "online_routes_exact": capture["client_server_full_routes_exact"],
    }


def validate_moe_only_online() -> dict[str, int | bool]:
    audit_root = MOE_ONLY_ONLINE / "failure_type_audit"
    summary = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["training"] is False
    assert summary["selector_uses_physical_state"] is False
    assert summary["physical_data_role"] == "posthoc_failure_typing_only"
    pooled = summary["pooled"]
    target = pooled["target_missed_grasp_proxy"]
    other = pooled["other_failure"]
    success = pooled["success"]
    assert target["episodes"] == 9
    assert target["episodes_with_any_raw_reject"]["numerator"] == 7
    assert target["episodes_with_formal_alarm"]["numerator"] == 1
    assert other["episodes"] == 13
    assert other["episodes_with_any_raw_reject"]["numerator"] == 9
    assert other["episodes_with_formal_alarm"]["numerator"] == 2
    assert success["episodes"] == 18
    assert success["episodes_with_any_raw_reject"]["numerator"] == 7
    assert success["episodes_with_formal_alarm"]["numerator"] == 0
    assert summary["target_timing"] == {
        "episodes": 9,
        "raw_within_5_queries_before_or_at_miss": 1,
        "raw_within_5_queries_after_miss": 4,
        "nearest_raw_minus_missed_query": [9, 3, 1, -4, -17, 2, 1, None, None],
    }
    sensitivity = summary["pooled_sensitivity_15mm"]
    assert sensitivity["target_missed_grasp_proxy"]["episodes"] == 12
    assert sensitivity["target_missed_grasp_proxy"]["episodes_with_formal_alarm"]["numerator"] == 2
    assert summary["sensitivity_analysis"]["additional_borderline_failure_episodes"] == 3

    heldout = summary["by_run"]["heldout_gpu4"]
    assert heldout["target_missed_grasp_proxy"]["episodes"] == 2
    assert heldout["other_failure"]["episodes"] == 3
    assert heldout["success"]["episodes"] == 3
    gpu5 = summary["by_run"]["gpu5_extended"]
    assert gpu5["target_missed_grasp_proxy"]["episodes"] == 5
    assert gpu5["target_missed_grasp_proxy"]["episodes_with_formal_alarm"]["numerator"] == 1
    assert gpu5["other_failure"]["episodes"] == 9
    assert gpu5["other_failure"]["episodes_with_formal_alarm"]["numerator"] == 2
    assert gpu5["success"]["episodes"] == 10
    assert gpu5["success"]["episodes_with_formal_alarm"]["numerator"] == 0
    expected_rows = {
        "preliminary_gpu4": 349,
        "heldout_gpu4": 381,
        "gpu5_extended": 1114,
    }
    for role, expected in expected_rows.items():
        capture = summary["route_capture_audit"][role]
        assert capture["client_rows"] == capture["server_rows"] == expected
        assert capture["client_server_full_routes_exact"] is True
        assert capture["client_server_episode_ids_exact"] is True
        assert capture["server_control_steps_strictly_sequential"] is True
        assert capture["store_full_probs"] is True
        assert capture["return_full_probs"] is True
        assert capture["hook_verify_failures"] == []

    with (audit_root / "episode_audit.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    assert sum(row["failure_family"] == "target_missed_grasp_proxy" for row in rows) == 9
    assert sum(row["failure_family"] == "other_failure" for row in rows) == 13
    assert sum(row["failure_family"] == "success" for row in rows) == 18
    assert sum(1 for _ in (audit_root / "grasp_events.jsonl").open(encoding="utf-8")) == 75

    for run_name, episodes, successes, failures, alarms in (
        ("gpu4_fresh_random_seed20260905", 8, 5, 3, 0),
        ("gpu4_heldout_random_seed20260906", 8, 3, 5, 0),
        ("gpu5_extended_random_seed20260907", 24, 10, 14, 3),
    ):
        manifest = json.loads(
            (MOE_ONLY_ONLINE / run_name / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["training"] is False
        assert manifest["selector_uses_physical_state"] is False
        assert manifest["completed_episodes"] == episodes
        assert manifest["successes"] == successes
        assert manifest["failures"] == failures
        assert manifest["alarm_episodes"] == alarms

    reference_specs = (
        ("full_healthy_route_sequences.npz", 7, 46),
        ("h20_healthy_route_sequences_seed20260905.npz", 5, 41),
    )
    for name, count, padded_length in reference_specs:
        with np.load(MOE_ONLY_ONLINE / name, allow_pickle=False) as archive:
            assert archive["schema"].item() == "himoe.moe_only_healthy_sequences.v1"
            assert bool(archive["training"].item()) is False
            assert bool(archive["failure_labels_used"].item()) is False
            assert bool(archive["physical_alignment_used"].item()) is False
            assert archive["routes"].shape == (count, padded_length, 8, 10, 11, 32)

    videos = json.loads(
        (MOE_ONLY_ONLINE / "review_videos/manifest.json").read_text(encoding="utf-8")
    )
    assert videos["training"] is False
    assert videos["online_alarm_video"] is False
    assert len(videos["episodes"]) == 3
    assert [item["failure_family"] for item in videos["episodes"]] == [
        "target_missed_grasp_proxy",
        "other_failure",
        "success",
    ]
    for episode in videos["episodes"]:
        assert episode["max_abs_sim_state_error"] == 0.0
        assert episode["mean_abs_sim_state_error"] == 0.0
        assert episode["formal_alarm_queries"] == []
        for video in episode["videos"]:
            assert (PACKAGE_ROOT / video["path"]).is_file()

    gpu5_manifest = json.loads(
        (MOE_ONLY_ONLINE / "gpu5_extended_random_seed20260907/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(gpu5_manifest["videos"]) == 9
    alarm_episodes = [
        item for item in gpu5_manifest["episodes"] if item["alarm_count"] > 0
    ]
    assert [item["episode_index"] for item in alarm_episodes] == [11, 14, 22]
    assert [item["alarm_queries"] for item in alarm_episodes] == [[11], [44], [35]]
    assert all(not item["success"] for item in alarm_episodes)
    for video in gpu5_manifest["videos"]:
        assert (MOE_ONLY_ONLINE / "gpu5_extended_random_seed20260907" / video["path"]).is_file()

    diagnosis_root = MOE_ONLY_ONLINE / "method_diagnosis"
    diagnosis = json.loads((diagnosis_root / "summary.json").read_text(encoding="utf-8"))
    assert diagnosis["training"] is False
    assert diagnosis["physical_inputs_to_online_selector"] is False
    assert diagnosis["online_rule_changed_during_gpu5_collection"] is False
    conclusion = diagnosis["diagnostic_conclusion"]
    assert conclusion == {
        "frozen_rule_has_validated_pre_event_separation": False,
        "moe_telemetry_has_post_event_signal": True,
        "persistence_is_not_the_only_failure_mode": True,
        "replacement_detector_validated_online": False,
    }
    event = diagnosis["gpu5_event_aligned"]
    assert event["frozen_conjunction_pre5"]["auc_target_higher"] == 0.42
    assert event["frozen_conjunction_pre5"]["p_maxT"] == 1.0
    assert event["best_post5_feature_descriptive_not_selected_online"]["feature"] == "back_route_acceleration"
    assert event["best_post5_feature_descriptive_not_selected_online"]["auc_target_higher"] == 1.0
    assert event["best_post5_feature_descriptive_not_selected_online"]["p_maxT"] < 0.05
    with (diagnosis_root / "event_auc.csv").open(newline="", encoding="utf-8") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 96

    return {
        "moe_only_online_episodes": len(rows),
        "moe_only_target_cases": target["episodes"],
        "moe_only_target_formal_alarms": target["episodes_with_formal_alarm"]["numerator"],
        "moe_only_gpu5_alarm_videos": len(gpu5_manifest["videos"]),
        "moe_only_routes_exact": True,
    }


def validate_moe_dynamics_v2() -> dict[str, int | float | bool]:
    calibration_path = MOE_DYNAMICS / "calibration_v2/calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert calibration["schema"] == "himoe.moe_dynamics_calibration.v1"
    assert calibration["selector_version"] == "back_front_route_acceleration_v2"
    assert calibration["training"] is False
    assert calibration["gradient_optimization"] is False
    assert calibration["failure_labels_used_to_set_threshold"] is False
    assert calibration["physical_alignment_used_to_set_threshold"] is False
    assert calibration["healthy_reference_trajectories"] == 5
    assert calibration["healthy_loo_formal_alarms_at_threshold"] == 0
    assert calibration["persistence"] == 2
    threshold = calibration["normalized_excess_threshold"]
    assert threshold == 1.0457019658379743
    with (MOE_DYNAMICS / "calibration_v2/healthy_loo_episode_scores.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        calibration_rows = list(csv.DictReader(stream))
    assert len(calibration_rows) == 5
    assert max(float(row["persistence2_support_max"]) for row in calibration_rows) == threshold

    offline = json.loads(
        (MOE_DYNAMICS / "offline_replay/summary.json").read_text(encoding="utf-8")
    )
    assert offline["training"] is False
    assert offline["failure_labels_used_to_set_threshold"] is False
    assert offline["prospective_validation_complete"] is False
    assert offline["by_run"]["gpu5_extended"]["all_failures"]["hits"] == 11
    assert offline["by_run"]["gpu5_extended"]["success"]["hits"] == 0

    run = MOE_DYNAMICS / "gpu5_prospective_seed20260908"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "himoe.online_moe_dynamics_alarm.v2"
    assert manifest["selector_version"] == "back_front_route_acceleration_v2"
    assert manifest["training"] is False
    assert manifest["fixed_episode_count"] is True
    assert manifest["completed_episodes"] == 24
    assert manifest["successes"] == 10
    assert manifest["failures"] == 14
    assert manifest["alarm_episodes"] == 9
    assert len(manifest["videos"]) == 27

    calibration_digest = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    frozen_config = json.loads(
        (PACKAGE_ROOT / "configs/moe_dynamics_online_alarm_v2.json").read_text(
            encoding="utf-8"
        )
    )
    experiment_config = json.loads(
        (run / "experiment_config.json").read_text(encoding="utf-8")
    )
    assert frozen_config["calibration_sha256_before_prospective_run"] == calibration_digest
    assert experiment_config["dynamics_calibration_sha256"] == calibration_digest
    assert experiment_config["seed"] == 20260908
    assert experiment_config["init_state_schedule"] == frozen_config["prospective_gpu5_run"]["init_state_schedule"]

    audit_root = MOE_DYNAMICS / "prospective_audit"
    summary = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["training"] is False
    assert summary["selector_uses_physical_state"] is False
    assert summary["physical_data_role"] == "posthoc_failure_typing_only"
    assert summary["calibration_unchanged_after_prospective_run"] is True
    assert summary["threshold_changed_during_prospective_run"] is False
    assert summary["target_formal_alarm"]["numerator"] == 5
    assert summary["target_formal_alarm"]["denominator"] == 7
    assert summary["all_failure_formal_alarm"]["numerator"] == 8
    assert summary["all_failure_formal_alarm"]["denominator"] == 14
    assert summary["v1_all_failure_formal_alarm"]["numerator"] == 1
    assert summary["successful_episode_false_alarm"]["numerator"] == 1
    assert summary["successful_episode_false_alarm"]["denominator"] == 10
    paired = summary["paired_v2_vs_v1"]
    assert paired["all_failures"] == {
        "both": 1,
        "episodes": 14,
        "exact_mcnemar_two_sided_p": 0.015625,
        "neither": 6,
        "v1_only": 0,
        "v2_only": 7,
    }
    assert paired["target_missed_grasp_proxy"]["v2_only"] == 4
    assert paired["target_missed_grasp_proxy"]["v1_only"] == 0
    capture = summary["capture_audit"]
    assert capture["client_rows"] == capture["server_rows"] == 1129
    assert capture["route_shape"] == [1129, 8, 10, 11, 32]
    assert capture["client_server_full_routes_exact"] is True
    assert capture["client_server_episode_ids_exact"] is True
    assert capture["server_control_steps_strictly_sequential"] is True
    assert capture["hook_verify_failures"] == []

    with (audit_root / "episode_audit.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert sum(row["failure_family"] == "target_missed_grasp_proxy" for row in rows) == 7
    assert sum(row["failure_family"] == "other_failure" for row in rows) == 7
    assert sum(row["failure_family"] == "success" for row in rows) == 10
    assert [int(row["episode_index"]) for row in rows if row["success"] == "True" and row["formal_alarm"] == "True"] == [9]

    for video in manifest["videos"]:
        path = run / video["path"]
        assert path.is_file()
        assert path.stat().st_size == video["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == video["sha256"]
    for episode in manifest["episodes"]:
        assert episode["selector_uses_physical_state"] is False
        assert episode["selector_uses_action_values"] is False
        assert episode["selector_uses_gripper_event"] is False
        assert episode["selector_uses_reward_or_success"] is False

    return {
        "moe_dynamics_v2_episodes": len(rows),
        "moe_dynamics_v2_target_hits": summary["target_formal_alarm"]["numerator"],
        "moe_dynamics_v2_failure_hits": summary["all_failure_formal_alarm"]["numerator"],
        "moe_dynamics_v2_success_false_alarms": summary["successful_episode_false_alarm"]["numerator"],
        "moe_dynamics_v2_route_rows": capture["server_rows"],
        "moe_dynamics_v2_videos": len(manifest["videos"]),
    }


def validate_cache_new_moe_only_offline() -> dict[str, int | float | bool]:
    root = MOE_DYNAMICS / "cache_new_task8_replay"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "himoe.moe_dynamics_large_offline_replay.v1"
    assert summary["training"] is False
    assert summary["gradient_optimization"] is False
    assert summary["runtime_selector_dynamic_input"] == "hb_router_probs only"
    assert summary["runtime_selector_uses_action_values"] is False
    assert summary["runtime_selector_uses_gripper_events"] is False
    assert summary["runtime_selector_uses_physical_distance"] is False
    assert summary["runtime_selector_uses_reward_or_success"] is False
    assert summary["runtime_selector_uses_robot_or_sim_state"] is False
    assert summary["labels_loaded_after_predictions_written"] is True
    assert summary["healthy_reference_selection_uses_known_success"] is True
    assert summary["threshold"] == 1.0457019658379743
    assert summary["persistence"] == 2

    aggregate = summary["aggregate"]["C_cache_new"]
    assert aggregate["episodes"] == 400
    assert aggregate["queries"] == 17460
    assert aggregate["successes"] == 262
    assert aggregate["failures"] == 138
    assert aggregate["failure_episode_recall"]["numerator"] == 29
    assert aggregate["success_episode_false_alarm"]["numerator"] == 13
    unseen = summary["cache_new_condition_breakdown"]["condition_not_in_old_B"]
    assert unseen["episodes"] == 272
    assert unseen["failure_episode_recall"]["numerator"] == 20
    assert unseen["success_episode_false_alarm"]["numerator"] == 5

    posthoc = json.loads((root / "posthoc_summary.json").read_text(encoding="utf-8"))
    assert posthoc["schema"] == "himoe.cache_new_moe_dynamics_posthoc_audit.v1"
    assert posthoc["training"] is False
    assert posthoc["prediction_input"] == "HB MoE router probabilities only"
    assert posthoc["predictions_materialized_before_physics_loaded"] is True
    assert posthoc["physics_role"] == "posthoc_failure_typing_and_timing_only"
    assert posthoc["episodes"] == 400
    all_rows = posthoc["all_400"]
    assert all_rows["target_missed_grasp_proxy"]["episodes"] == 53
    assert all_rows["target_missed_grasp_proxy"]["formal_alarm_anywhere"]["numerator"] == 13
    assert all_rows["target_missed_grasp_proxy"]["formal_alarm_within_5q_before_or_at_miss"]["numerator"] == 9
    assert all_rows["other_failure"]["episodes"] == 85
    assert all_rows["other_failure"]["formal_alarm_anywhere"]["numerator"] == 16
    assert all_rows["success"]["episodes"] == 262
    assert all_rows["success"]["formal_alarm_anywhere"]["numerator"] == 13
    overlap = posthoc["old_B_overlap_audit"]
    assert overlap["conditions_not_present_in_old_B"] == 272
    assert overlap["nominally_overlapping_init_noise_conditions"] == 128
    assert overlap["whole_route_exact_duplicates"] == 0
    assert overlap["first_query_exact_duplicates"] == 0
    proxy = posthoc["query_proxy_validation_on_dense_prospective_24"]
    assert proxy["strict_true_positive"] == 7
    assert proxy["strict_false_negative"] == 0
    assert proxy["strict_false_positive"] == 2
    assert proxy["strict_true_negative"] == 15

    expected_rows = {
        "query_predictions_moe_only.csv": 17060,
        "episode_predictions_moe_only.csv": 400,
        "episode_evaluation_posthoc.csv": 400,
        "posthoc_missed_grasp_audit.csv": 400,
    }
    for name, expected in expected_rows.items():
        with (root / name).open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == expected, name
    assert sum(1 for _ in (root / "grasp_events_query_sampled.jsonl").open(
        encoding="utf-8"
    )) == 704

    return {
        "cache_new_moe_only_episodes": aggregate["episodes"],
        "cache_new_moe_only_queries": aggregate["queries"],
        "cache_new_moe_only_scored_queries": 17060,
        "cache_new_failure_hits": aggregate["failure_episode_recall"]["numerator"],
        "cache_new_success_false_alarms": aggregate["success_episode_false_alarm"]["numerator"],
        "cache_new_target_hits": all_rows["target_missed_grasp_proxy"]["formal_alarm_anywhere"]["numerator"],
    }


def validate_large_moe_only_offline() -> dict[str, int | float | bool]:
    root = MOE_DYNAMICS / "large_offline_replay"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "himoe.moe_dynamics_large_offline_replay.v1"
    assert summary["training"] is False
    assert summary["gradient_optimization"] is False
    assert summary["runtime_selector_dynamic_input"] == "hb_router_probs only"
    assert summary["runtime_selector_uses_action_values"] is False
    assert summary["runtime_selector_uses_robot_or_sim_state"] is False
    assert summary["runtime_selector_uses_physical_distance"] is False
    assert summary["runtime_selector_uses_gripper_events"] is False
    assert summary["runtime_selector_uses_reward_or_success"] is False
    assert summary["labels_loaded_after_predictions_written"] is True
    assert summary["retrospective_corpora"] == ["A", "B"]
    assert summary["independent_prospective_corpus"] == "prospective"

    expected = {
        "A": (352, 16158, 117, 235, 183, 18),
        "B": (512, 22883, 296, 216, 34, 8),
        "prospective": (24, 1129, 10, 14, 8, 1),
    }
    for corpus, values in expected.items():
        episodes, queries, successes, failures, failure_hits, false_alarms = values
        observed = summary["aggregate"][corpus]
        assert observed["episodes"] == episodes
        assert observed["queries"] == queries
        assert observed["successes"] == successes
        assert observed["failures"] == failures
        assert observed["failure_episode_recall"]["numerator"] == failure_hits
        assert observed["success_episode_false_alarm"]["numerator"] == false_alarms
    parity = summary["prospective_offline_vs_online_parity"]
    assert parity == {
        "episodes": 24,
        "exact_raw_and_formal_alarm_sequences": 24,
        "mismatch_episode_ids": [],
    }

    expected_rows = {
        "query_predictions_moe_only.csv": 39282,
        "episode_predictions_moe_only.csv": 888,
        "episode_evaluation_posthoc.csv": 888,
    }
    for name, row_count in expected_rows.items():
        with (root / name).open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == row_count, name
    return {
        "large_moe_only_offline_episodes": 888,
        "large_moe_only_offline_scored_queries": 39282,
        "prospective_offline_online_exact_parity": parity[
            "exact_raw_and_formal_alarm_sequences"
        ],
    }


def validate_task_difficulty_weighting() -> dict[str, int | float | bool]:
    summary = json.loads((TASK_DIFFICULTY / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "himoe.task_difficulty_weighting.v1"
    assert summary["training"] is False
    assert summary["gradient_optimization"] is False
    assert summary["fixed_feature_weights"] is True
    assert summary["n_bootstrap"] == 5000
    assert summary["n_permutations"] == 5000
    assert summary["tasks"] == 37
    assert summary["episodes"] == 14800
    assert summary["failures"] == 487
    assert summary["successes"] == 14313
    assert summary["first_query_prediction_inputs"] == "HB MoE router probabilities only"
    assert summary["first_query_features_materialized_before_outcomes_loaded"] is True

    reliability = summary["difficulty_reliability"]
    assert np.isclose(reliability["seed_half_spearman_rho"], 0.8372140462521269)
    assert np.isclose(
        reliability["seed_half_spearman_rho_without_hardest_task"],
        0.8230337921238435,
    )
    assert reliability["zero_failure_tasks"] == 5

    identity = summary["early_route_task_identity"]
    assert identity["episodes"] == 14800
    assert np.isclose(identity["task_accuracy"], 0.9895945945945946)
    assert identity["suite_accuracy"] == 1.0

    early = summary["early_route_difficulty"]
    assert early["protocol"] == "two-way disjoint-seed cross-fit"
    assert early["significant_early_scalars_after_bh_005"] == 0
    assert np.isclose(
        early[
            "pairwise_route_distance_vs_difficulty_difference_mean_spearman_rho"
        ],
        0.11967023213949515,
    )
    assert early["pairwise_permutation_p_positive_association"] > 0.05
    for result in early["knn"].values():
        assert result["mae"] > result["median_baseline_mae"]
        assert result["mae_improvement_over_median"] < 0

    variants = {
        row["variant"]: row for row in summary["alarm_weighting"]["variants"]
    }
    uniform = variants["uniform"]
    linear = variants["empirical_linear"]
    route = variants["route_knn5_sqrt"]
    assert uniform["true_alarms"] == 318 and uniform["false_alarms"] == 667
    assert linear["true_alarms"] == 371 and linear["false_alarms"] == 689
    assert np.isclose(linear["delta_recall_vs_uniform"], 0.10882956878850103)
    assert linear["delta_recall_ci_low"] < 0 < linear["delta_recall_ci_high"]
    assert linear["macro_task_failure_recall"] < uniform["macro_task_failure_recall"]
    assert linear["hardest_task_true_alarms"] - uniform["hardest_task_true_alarms"] == 42
    assert route["true_alarms"] == 317
    assert route["failure_recall"] < uniform["failure_recall"]
    fold_linear = [
        row
        for row in summary["alarm_weighting"]["fold_results"]
        if row["variant"] == "empirical_linear"
    ]
    assert [row["delta_true_alarms_vs_uniform"] for row in fold_linear] == [31, 22]

    expected_rows = {
        "first_query_moe_features.csv": 14800,
        "task_difficulty.csv": 37,
        "early_route_difficulty_associations.csv": 15,
        "early_route_knn_predictions.csv": 296,
        "weighting_episode_predictions.csv": 14800,
        "weighting_summary.csv": 4,
        "weighting_task_allocations.csv": 296,
        "weighting_task_evaluation.csv": 148,
        "weighting_fold_evaluation.csv": 8,
    }
    for name, row_count in expected_rows.items():
        with (TASK_DIFFICULTY / name).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == row_count, name
        if name == "first_query_moe_features.csv":
            assert "failure" not in rows[0]
            assert "outcome" not in rows[0]

    with np.load(
        TASK_DIFFICULTY / "first_query_action_route_embeddings.npz",
        allow_pickle=False,
    ) as archive:
        assert archive["embeddings"].shape == (14800, 8, 32)
        assert archive["tasks"].shape == (14800,)
        assert archive["episodes"].shape == (14800,)
        assert archive["outcome_labels_used"].item() is False
    assert (TASK_DIFFICULTY / "difficulty_weighting_audit.png").stat().st_size > 100000
    assert (TASK_DIFFICULTY / "REPORT_ZH.md").exists()
    return {
        "task_difficulty_tasks": 37,
        "task_difficulty_episodes": 14800,
        "task_identity_accuracy": identity["task_accuracy"],
        "empirical_linear_micro_recall": linear["failure_recall"],
        "route_knn_micro_recall": route["failure_recall"],
    }


def validate_self_reference_alarm() -> dict[str, int | float | bool]:
    config = json.loads(
        (PACKAGE_ROOT / "configs/self_reference_coupling_collapse_v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["selector_version"] == "self_reference_coupling_collapse_v3"
    assert config["training"] is False
    assert config["learned_parameters"] is False
    assert config["development_outcomes_inspected"] is True
    for field in (
        "runtime_normal_trajectory_reference_used",
        "runtime_outcome_labels_used",
        "runtime_physical_state_used",
        "runtime_action_values_used",
        "runtime_task_identity_used",
    ):
        assert config[field] is False

    offline = json.loads((SELF_REFERENCE / "summary.json").read_text(encoding="utf-8"))
    assert offline["schema"] == "himoe.task_free_self_reference_evaluation.v1"
    assert offline["training"] is False
    assert offline["learned_parameters"] is False
    assert offline["complete_tasks"] == 40
    heldout = offline["heldout_39_task_evaluation"]
    assert heldout["episodes"] == 15600
    assert heldout["failures"] == 530 and heldout["successes"] == 15070
    assert heldout["tp"] == 78 and heldout["fp"] == 35
    assert np.isclose(heldout["tpr"], 0.1471698113207547)
    assert np.isclose(heldout["fpr"], 0.00232249502322495)
    clock = offline["clock_controls_heldout"]["matched_or_lower_fpr"]
    assert clock["tp"] == 530 and clock["fn"] == 0
    assert np.isclose(clock["horizon_phase"], 0.81)

    expected_rows = {
        "episode_predictions.csv": 16000,
        "query_decisions.csv": 253722,
        "task_metrics.csv": 40,
        "clock_baselines.csv": 91,
    }
    for name, row_count in expected_rows.items():
        with (SELF_REFERENCE / name).open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == row_count, name

    audit_root = SELF_REFERENCE / "gpu6_audit"
    gpu6 = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    assert gpu6["schema"] == "himoe.self_reference_gpu6_audit.v1"
    assert gpu6["training"] is False
    assert gpu6["runtime_selector_inputs"] == [
        "current HB router probabilities",
        "own episode route prefix",
    ]
    prospective = gpu6["gpu6_prospective"]
    assert prospective["episodes"] == 16
    assert prospective["tp"] == 1 and prospective["fn"] == 0
    assert prospective["fp"] == 0 and prospective["tn"] == 15
    assert gpu6["gpu6_post_alarm_action_steps"] == 100
    timing = gpu6["posthoc_physical_timing"]
    assert len(timing) == 1
    assert timing[0]["plateau_query"] == 16
    assert timing[0]["first_moe_alarm_query"] == 20
    assert timing[0]["fixed_clock_query"] == 24
    assert timing[0]["marker_is_semantic_failure_onset"] is False
    replay = gpu6["cross_gpu_same_seed_replay"]
    assert replay["conditions"] == 2
    assert replay["old_failures"] == 2 and replay["new_successes"] == 2
    assert replay["q0_policy_state_exact_conditions"] == 2
    assert replay["q0_sim_state_exact_conditions"] == 2
    assert replay["runtime_identity"]["all_match"] is True
    assert replay["action_mae_min"] > 0 and replay["route_mae_min"] > 0

    for name, row_count in {
        "online_episodes.csv": 16,
        "physical_timing.csv": 1,
        "same_seed_cross_gpu.csv": 2,
    }.items():
        with (audit_root / name).open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == row_count, name
    assert (audit_root / "gpu6_self_reference_audit.png").stat().st_size > 100000
    assert (audit_root / "REPORT_ZH.md").exists()

    server_capture = json.loads(
        (
            SELF_REFERENCE
            / "gpu6_goal_server_paper_right/capture_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert server_capture["control_steps"] == 304
    assert server_capture["store_full_probs"] is True
    assert server_capture["return_full_probs"] is True
    assert server_capture["hook_verify_failures"] == []

    video_root = (
        SELF_REFERENCE
        / "gpu6_prospective_init09_seeds1000_1007"
        / "episode_000_init_09_flowseed_1000/videos"
    )
    for video in (
        "alarm_clean_full.mp4",
        "alarm_annotated_full.mp4",
        "alarm_annotated_clip.mp4",
    ):
        assert (video_root / video).stat().st_size > 100000
    return {
        "self_reference_offline_tasks": offline["complete_tasks"],
        "self_reference_heldout_episodes": heldout["episodes"],
        "self_reference_heldout_recall": heldout["tpr"],
        "self_reference_heldout_fpr": heldout["fpr"],
        "self_reference_gpu6_prospective_episodes": prospective["episodes"],
        "self_reference_gpu6_failure_alarms": prospective["tp"],
        "self_reference_gpu6_success_false_alarms": prospective["fp"],
    }


def validate_input_version_counterfactual() -> dict[str, int | float]:
    collection_config = json.loads(
        (PACKAGE_ROOT / "configs/input_version_counterfactual.json").read_text(
            encoding="utf-8"
        )
    )
    assert collection_config["training"] is False
    assert collection_config["alarm_uses_normal_trajectory_bank"] is False
    assert collection_config["alarm_uses_task_identity"] is False
    assert collection_config["alarm_uses_physics"] is False
    assert collection_config["noise_draws"] == 4

    alarm_config = json.loads(
        (PACKAGE_ROOT / "configs/counterfactual_open_loop_alarm_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert alarm_config["training"] is False
    assert alarm_config["learned_parameters"] is False
    for key in (
        "runtime_normal_trajectory_reference_used",
        "runtime_outcome_labels_used",
        "runtime_physical_state_used",
        "runtime_action_values_used",
        "runtime_task_identity_used",
    ):
        assert alarm_config[key] is False, key
    assert alarm_config["alarm_operator"] == "<"
    assert alarm_config["alarm_threshold"] == 1.0
    assert alarm_config["inference_calls_per_decision"] == 8

    formal = INPUT_VERSION / "formal_capture"
    manifest = json.loads((formal / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["training"] is False
    assert manifest["records"] == 96
    assert manifest["inference_calls"] == 768
    assert manifest["noise_draws"] == 4
    assert manifest["normal_trajectory_bank_used"] is False
    assert manifest["selection_uses_outcome"] is False
    assert manifest["selection_uses_physics"] is False
    assert manifest["task_identity_used_by_score"] is False
    assert manifest["exact_archived_policy_state_inserted"] is True
    assert manifest["paired_routes_sha256"] == (
        "2ce256ad16a530ce458d48736fb5175425d02c00f23b6e9a01325643033fe7a8"
    )

    with np.load(formal / "paired_counterfactual_routes.npz") as archive:
        assert not bool(archive["training"])
        assert archive["routes_previous"].shape == (96, 4, 8, 10, 11, 32)
        assert archive["routes_current"].shape == (96, 4, 8, 10, 11, 32)
        assert archive["actions_previous"].shape == (96, 4, 10, 7)
        assert archive["actions_current"].shape == (96, 4, 10, 7)
        assert archive["flow_noise"].shape == (96, 4, 10, 24)
        assert int(archive["failed_grasp"].sum()) == 12

    analysis = INPUT_VERSION / "analysis"
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["training"] is False
    assert summary["records"] == 96 and summary["inference_calls"] == 768
    assert summary["score_uses_normal_trajectory_bank"] is False
    assert summary["score_uses_task_identity"] is False
    assert summary["score_uses_physics"] is False
    assert summary["score_uses_outcome"] is False
    assert summary["failed_first_alarm_relative_query_m4"] == -1
    assert summary["failed_alarm_relative_queries_m4"] == [-1]
    assert summary["relative_minus_1"]["success_control_alarms_m4"] == 0
    assert summary["success_control_trajectories_with_any_alarm_m4"] == 3
    assert summary["minimal_four_call_m2"][
        "success_control_trajectories_with_any_alarm"
    ] == 5
    assert summary["architecture_audit"]["front_state_noise_effect_median_h"] == 0.0
    assert summary["architecture_audit"]["front_state_noise_effect_max_h"] < 0.001
    comparison = summary["comparison_to_self_reference_v3"]
    assert comparison["failed_counterfactual_first_alarm_relative"] == -1
    assert comparison["failed_first_alarm_relative_to_closure"] == 12
    assert comparison["successful_siblings_with_v3_alarm"] == 0

    analysis_rows = {
        "primary_scores.csv": 96,
        "counterfactual_group_metrics.csv": 2304,
        "counterfactual_layer_metrics.csv": 768,
        "replay_fidelity_by_layer.csv": 768,
        "trainfree_alarm_by_relative_query.csv": 12,
        "comparison_to_self_reference_v3.csv": 8,
    }
    for name, expected in analysis_rows.items():
        with (analysis / "tables" / name).open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == expected, name

    modality_capture = INPUT_VERSION / "modality_capture"
    modality_manifest = json.loads(
        (modality_capture / "manifest.json").read_text(encoding="utf-8")
    )
    assert modality_manifest["status"] == "complete"
    assert modality_manifest["training"] is False
    assert modality_manifest["records"] == 40
    assert modality_manifest["inference_calls"] == 320
    assert modality_manifest["selection_uses_outcome"] is False
    assert modality_manifest["selection_uses_physics"] is False
    assert modality_manifest["artifact_sha256"] == (
        "9c5274a8d1fe4b50680bc41de9ce10470333b4aacb3538362542930007b1cce5"
    )
    with np.load(modality_capture / "modality_counterfactual_routes.npz") as archive:
        assert not bool(archive["training"])
        assert archive["paired_record_index"].shape == (40,)
        assert archive["routes_vision_only"].shape == (40, 4, 8, 10, 11, 32)
        assert archive["routes_state_only"].shape == (40, 4, 8, 10, 11, 32)
        assert archive["actions_vision_only"].shape == (40, 4, 10, 7)
        assert archive["actions_state_only"].shape == (40, 4, 10, 7)

    modality_analysis = INPUT_VERSION / "modality_analysis"
    modality = json.loads(
        (modality_analysis / "summary.json").read_text(encoding="utf-8")
    )
    assert modality["status"] == "complete"
    assert modality["training"] is False
    assert modality["score_uses_normal_trajectory_bank"] is False
    assert modality["score_uses_outcome"] is False
    assert modality["score_uses_physics"] is False
    assert modality["failed_timeline"]["-1"][
        "back_action_vision_relation"
    ] == "below_all"
    assert modality["failed_timeline"]["-1"][
        "back_action_proprio_relation"
    ] == "inside"
    for relative in ("1", "2"):
        assert modality["failed_timeline"][relative][
            "back_action_vision_relation"
        ] == "above_all"
        assert modality["failed_timeline"][relative][
            "back_action_proprio_relation"
        ] == "above_all"
    assert modality["failed_timeline"]["0"]["action_output"][
        "noise_relation"
    ] == "above_all"

    novelty = json.loads(
        (modality_analysis / "input_novelty_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert novelty["training"] is False
    assert novelty["role"] == "posthoc_confound_audit_not_used_by_alarm"
    minus_one = novelty["timeline"]["-1"]
    for metric in ("base_mae_u8", "wrist_mae_u8", "policy_state_delta_l2"):
        assert minus_one[metric]["relation"] == "below_all", metric
    assert minus_one["vision_route_effect_per_pixel_mae"]["relation"] == "inside"
    assert minus_one["proprio_route_effect_per_state_l2"]["relation"] == "above_all"

    modality_rows = {
        "modality_group_metrics.csv": 480,
        "modality_layer_metrics.csv": 640,
        "modality_primary_metrics.csv": 40,
        "modality_failure_vs_success.csv": 50,
        "action_output_counterfactual.csv": 40,
        "action_output_failure_vs_success.csv": 25,
        "input_version_pixel_novelty_posthoc.csv": 40,
        "policy_state_novelty_posthoc.csv": 40,
        "route_response_per_input_novelty_posthoc.csv": 40,
    }
    for name, expected in modality_rows.items():
        with (modality_analysis / "tables" / name).open(
            newline="", encoding="utf-8"
        ) as stream:
            assert len(list(csv.DictReader(stream))) == expected, name

    for directory, calls in (("server", 816), ("server_factorial", 320)):
        capture = json.loads(
            (INPUT_VERSION / directory / "capture_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert capture["control_steps"] == calls
        assert capture["store_full_probs"] is True
        assert capture["return_full_probs"] is True
        assert capture["hook_verify_failures"] == []

    for figure in (
        analysis / "figures/counterfactual_feedback_transmission.png",
        analysis / "figures/layerwise_feedback_transmission.png",
        modality_analysis / "figures/input_modality_decomposition.png",
        modality_analysis / "figures/route_vs_action_counterfactual.png",
        modality_analysis / "figures/failed_input_version_pairs.png",
    ):
        assert figure.stat().st_size > 50000, figure

    return {
        "input_version_records": manifest["records"],
        "input_version_formal_inferences": manifest["inference_calls"],
        "input_version_failed_first_alarm_relative": summary[
            "failed_first_alarm_relative_query_m4"
        ],
        "input_version_success_control_window_alarm_trajectories": summary[
            "success_control_trajectories_with_any_alarm_m4"
        ],
        "input_modality_records": modality_manifest["records"],
        "input_modality_additional_inferences": modality_manifest[
            "inference_calls"
        ],
    }


def validate_trainfree_trap_probability() -> dict[str, int | float | bool]:
    summary = json.loads(
        (TRAP_PROBABILITY / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["schema"] == "himoe.trainfree_trap_probability.v1"
    assert summary["status"] == "complete"
    assert summary["training"] is False
    assert summary["gradient_optimization"] is False
    assert summary["learned_feature_weights"] is False
    assert summary["labeled_probability_calibration"] is True
    assert summary["runtime_task_identity_used"] is False
    assert summary["runtime_action_values_used"] is False
    assert summary["runtime_physical_state_used"] is False
    assert summary["runtime_outcome_used"] is False
    assert summary["route_states_materialized_before_labels"] is True
    assert summary["target_probabilities_written_without_target_labels"] is True
    assert summary["alarm_probability"] == 0.75
    assert summary["primary_target"]["event_type"] == "trap"
    assert summary["primary_target"]["horizon_queries"] == 2
    assert summary[
        "onset_hazard_all_probability_states_below_alarm_threshold"
    ] is True
    assert summary["onset_hazard_alarm_rows_total"] == 0

    primary = summary["primary_results"]
    expected = {
        "A_to_B": {
            "rows": 14320,
            "positive_rows": 579,
            "max_predicted_probability": 0.35833333333333334,
            "max_probability_target_observed_rate": 0.21354166666666666,
            "roc_auc": 0.8185715907124136,
            "average_precision": 0.13605424862701088,
        },
        "B_to_A": {
            "rows": 7331,
            "positive_rows": 621,
            "max_predicted_probability": 0.5131578947368421,
            "max_probability_target_observed_rate": 0.15789473684210525,
            "roc_auc": 0.7174611402694082,
            "average_precision": 0.17872662193644395,
        },
    }
    for direction, values in expected.items():
        row = primary[direction]
        assert row["alarm_rows_075"] == 0
        for key in ("rows", "positive_rows"):
            assert row[key] == values[key]
        for key in (
            "max_predicted_probability",
            "max_probability_target_observed_rate",
            "roc_auc",
            "average_precision",
        ):
            assert np.isclose(row[key], values[key]), (direction, key)

    threshold_rows = summary["primary_threshold_sweep"]
    for direction, precision, timely, false_positive in (
        ("A_to_B", 0.21354166666666666, 0.21243523316062177, 0.23824451410658307),
        ("B_to_A", 0.1597222222222222, 0.06698564593301436, 0.07746478873239436),
    ):
        row = next(
            item
            for item in threshold_rows
            if item["direction"] == direction
            and np.isclose(item["probability_threshold"], 0.25)
        )
        assert np.isclose(row["alarm_precision"], precision)
        assert np.isclose(row["event_episode_timely_recall"], timely)
        assert np.isclose(row["no_event_episode_fpr"], false_positive)

    absorbing_h2 = {
        row["direction"]: row
        for row in summary["absorbing_detection_results"]
        if row["horizon"] == 2
    }
    assert absorbing_h2["A_to_B"]["alarm_rows_075"] == 0
    assert absorbing_h2["A_to_B"]["clock_roc_auc"] > absorbing_h2["A_to_B"][
        "roc_auc"
    ]
    assert absorbing_h2["B_to_A"]["alarm_rows_075"] == 47
    assert absorbing_h2["B_to_A"]["alarm_true_rows_075"] == 31
    assert absorbing_h2["B_to_A"]["alarm_precision_075"] < 0.75
    assert absorbing_h2["B_to_A"]["clock_roc_auc"] > absorbing_h2["B_to_A"][
        "roc_auc"
    ]

    expected_table_rows = {
        "calibration_cells.csv": 576,
        "cross_corpus_probability_evaluation.csv": 18,
        "primary_episode_alarm_audit.csv": 863,
        "primary_probability_threshold_sweep.csv": 12,
        "primary_target_predictions_posthoc.csv": 21651,
        "target_reliability.csv": 224,
        "absorbing_calibration_cells.csv": 192,
        "absorbing_detection_evaluation.csv": 6,
        "absorbing_target_reliability.csv": 84,
    }
    for name, expected_rows in expected_table_rows.items():
        with (TRAP_PROBABILITY / "tables" / name).open(
            newline="", encoding="utf-8"
        ) as stream:
            assert len(list(csv.DictReader(stream))) == expected_rows, name

    prediction_files = summary["label_free_prediction_files"]
    assert len(prediction_files) == 24
    forbidden = {"label", "onset", "outcome", "success", "failure"}
    for record in prediction_files:
        assert record["target_labels_present"] is False
        directory = (
            "predictions_label_free_absorbing"
            if record.get("evaluation_mode") == "onset_absorbing_proxy"
            else "predictions_label_free"
        )
        path = TRAP_PROBABILITY / directory / Path(record["path"]).name
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            assert forbidden.isdisjoint(reader.fieldnames or []), path
            assert sum(1 for _ in reader) == record["rows"], path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256_before_target_evaluation"], path

    assert (
        TRAP_PROBABILITY / "figures/trainfree_trap_probability.png"
    ).stat().st_size > 100000

    cache_new_root = TRAP_PROBABILITY / "cache_new_40task"
    cache_summary = json.loads(
        (cache_new_root / "summary.json").read_text(encoding="utf-8")
    )
    assert cache_summary["schema"] == "himoe.trainfree_trap_probability_cache_new.v1"
    assert cache_summary["status"] == "complete"
    assert cache_summary["training"] is False
    assert cache_summary["gradient_optimization"] is False
    assert cache_summary["learned_feature_weights"] is False
    assert cache_summary["source_labels_used_for_frozen_calibration"] is True
    assert cache_summary["cache_new_outcomes_used_by_probability"] is False
    assert cache_summary["target_probabilities_written_before_outcomes"] is True
    assert cache_summary["task_identity_used_by_probability"] is False
    assert cache_summary["runtime_action_values_used"] is False
    assert cache_summary["runtime_physical_state_used"] is False
    assert cache_summary["runtime_outcome_used"] is False
    assert cache_summary["complete_tasks"] == 40
    assert cache_summary["total_episodes"] == 16000
    assert cache_summary["total_successes"] == 15468
    assert cache_summary["total_failures"] == 532
    assert cache_summary["total_route_queries"] == 253722
    assert cache_summary["eligible_query_rows"] == 75159

    cache_calibrators = cache_summary["calibrators"]
    expected_success_query = {
        "A_dense": {
            "median": 0.0685584092792046,
            "p90": 0.1532679738562091,
            "p95": 0.2038288288288288,
            "p99": 0.2276119402985074,
            "max": 0.3583333333333333,
        },
        "B_proxy": {
            "median": 0.0026664033181907,
            "p90": 0.0717098785690843,
            "p95": 0.1118769883351007,
            "p99": 0.2261904761904762,
            "max": 0.5131578947368421,
        },
    }
    for calibrator, expected_values in expected_success_query.items():
        values = cache_calibrators[calibrator]
        assert values["scoreable_episodes"] == 9542
        assert values["scoreable_failures"] == 532
        assert values["scoreable_successes"] == 9010
        assert values["unscoreable_failures_before_query_12"] == 0
        assert values["unscoreable_successes_before_query_12"] == 6458
        assert values["success_query_probability"]["count"] == 60791
        for key, expected_value in expected_values.items():
            assert np.isclose(
                values["success_query_probability"][key], expected_value
            ), (calibrator, key)
        threshold_075 = next(
            row for row in values["thresholds"] if row["threshold"] == 0.75
        )
        assert threshold_075["tp"] == 0
        assert threshold_075["fp"] == 0

    b_threshold_050 = next(
        row
        for row in cache_calibrators["B_proxy"]["thresholds"]
        if row["threshold"] == 0.5
    )
    assert b_threshold_050["tp"] == 133
    assert b_threshold_050["fp"] == 61
    assert np.isclose(b_threshold_050["failure_recall_all"], 0.25)
    assert np.isclose(b_threshold_050["endpoint_precision"], 0.6855670103092784)
    assert np.isclose(
        b_threshold_050["detected_failure_first_alarm_horizon_phase_median"],
        0.8235294117647058,
    )
    assert np.isclose(
        b_threshold_050["unseen_task_failure_recall"], 0.23604060913705585
    )
    assert np.isclose(
        b_threshold_050["unseen_task_endpoint_precision"], 0.6283783783783784
    )
    assert b_threshold_050["matched_clock_failure_recall"] == 1.0
    assert np.isclose(b_threshold_050["matched_clock_horizon_phase"], 0.79)
    assert np.isclose(
        cache_calibrators["B_proxy"]["within_task_first_query_ranking"][
            "pair_weighted_within_task_roc_auc"
        ],
        0.561412503107134,
    )
    assert np.isclose(
        cache_calibrators["B_proxy"]["within_task_duration_ranking_control"][
            "pair_weighted_within_task_roc_auc"
        ],
        0.9998990181456624,
    )

    route_manifest = json.loads(
        (cache_new_root / "route_only_manifest.json").read_text(encoding="utf-8")
    )
    assert route_manifest["schema"] == cache_summary["schema"]
    assert route_manifest["phase"] == "route_only_complete_before_outcome_load"
    assert route_manifest["training"] is False
    assert route_manifest["target_labels_loaded"] is False
    assert route_manifest["tasks"] == 40
    assert route_manifest["episodes"] == 16000
    assert route_manifest["route_queries"] == 253722
    assert route_manifest["eligible_query_rows"] == 75159
    assert len(route_manifest["task_caches"]) == 40

    cached_episodes = cached_routes = cached_eligible = 0
    cache_forbidden = {"label", "onset", "outcome", "success", "failure"}
    for record in route_manifest["task_caches"]:
        path = cache_new_root / "route_only_tasks" / Path(record["cache"]).name
        with np.load(path, allow_pickle=False) as data:
            assert data["schema"].item() == cache_summary["schema"]
            assert data["training"].item() is False
            assert data["target_labels_loaded"].item() is False
            assert cache_forbidden.isdisjoint(data.files), path
            assert len(data["all_episode_ids"]) == record["episodes"]
            assert len(data["episode"]) == record["eligible_rows"]
            assert data["route_rows"].item() == record["route_rows"]
        cached_episodes += record["episodes"]
        cached_routes += record["route_rows"]
        cached_eligible += record["eligible_rows"]
    assert cached_episodes == 16000
    assert cached_routes == 253722
    assert cached_eligible == 75159

    prediction_path = cache_new_root / "cache_new_probabilities_label_free.csv.gz"
    with gzip.open(prediction_path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert cache_forbidden.isdisjoint(reader.fieldnames or [])
        assert sum(1 for _ in reader) == 75159
    prediction_digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    assert prediction_digest == cache_summary["prediction_file_sha256"]
    assert prediction_digest == route_manifest[
        "prediction_sha256_before_outcome_load"
    ]

    cache_table_rows = {
        "endpoint_clock_baselines.csv": 91,
        "endpoint_fixed_query_ranking.csv": 80,
        "endpoint_phase_probability.csv": 16,
        "endpoint_probability_distribution.csv": 4,
        "endpoint_task_ranking.csv": 76,
        "endpoint_threshold_audit.csv": 12,
        "episode_outcome_audit.csv": 16000,
        "successful_probability_values.csv": 26,
    }
    for name, expected_rows in cache_table_rows.items():
        with (cache_new_root / "tables" / name).open(
            newline="", encoding="utf-8"
        ) as stream:
            assert len(list(csv.DictReader(stream))) == expected_rows, name
    assert (
        cache_new_root / "figures/cache_new_probability_audit.png"
    ).stat().st_size > 100000

    timing_root = TRAP_PROBABILITY / "timing_constrained_alarm"
    timing_summary = json.loads(
        (timing_root / "summary.json").read_text(encoding="utf-8")
    )
    assert timing_summary["schema"] == "himoe.timing_constrained_probability_alarm.v1"
    assert timing_summary["status"] == "complete"
    assert timing_summary["training"] is False
    assert timing_summary["gradient_optimization"] is False
    assert timing_summary["learned_feature_weights"] is False
    assert timing_summary["physical_state_used_by_alarm"] is False
    assert timing_summary["physical_state_used_for_posthoc_onset_only"] is True
    assert timing_summary["minimum_query"] == 12
    assert timing_summary["near_onset_window"] == [-2, 0]
    assert timing_summary["development"] == {
        "tasks": 37,
        "episodes": 14800,
        "failures": 487,
        "onset_events": 222,
        "eligible_onset_events": 198,
    }
    assert timing_summary["heldout"]["tasks"] == 3
    assert timing_summary["heldout"]["episodes"] == 1200
    assert timing_summary["heldout"]["failures"] == 45
    assert timing_summary["heldout"]["onset_events"] == 38
    assert timing_summary["heldout"]["eligible_onset_events"] == 24
    assert len(timing_summary["heldout"]["tasks_list"]) == 3
    assert timing_summary["suitable_onset_localized_threshold_found"] is False
    assert np.isclose(
        timing_summary[
            "best_near_first_recall_with_development_fpr_budget_le_005"
        ]["development"],
        5 / 198,
    )
    assert np.isclose(
        timing_summary[
            "best_near_first_recall_with_development_fpr_budget_le_005"
        ]["heldout"],
        2 / 24,
    )

    with (timing_root / "tables/operating_points.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        operating = list(csv.DictReader(stream))
    assert len(operating) == 20
    b_heldout_001 = next(
        row
        for row in operating
        if row["calibrator"] == "B_proxy"
        and row["split"] == "heldout_3_tasks"
        and np.isclose(float(row["success_fpr_budget"]), 0.01)
    )
    assert np.isclose(float(b_heldout_001["threshold"]), 0.3712121212121212)
    assert np.isclose(float(b_heldout_001["success_fpr"]), 15 / 1155)
    assert int(b_heldout_001["timely_events"]) == 7
    assert np.isclose(float(b_heldout_001["timely_recall"]), 7 / 24)
    assert int(b_heldout_001["near_first_events"]) == 1
    assert np.isclose(float(b_heldout_001["near_first_recall"]), 1 / 24)
    assert np.isclose(float(b_heldout_001["timely_lead_median"]), 13.0)
    assert int(b_heldout_001["clock_query"]) == 39
    assert int(b_heldout_001["clock_timely_events"]) == 7
    assert int(b_heldout_001["clock_near_first_events"]) == 2

    timing_table_rows = {
        "event_type_operating_points.csv": 60,
        "fixed_threshold_timing.csv": 28,
        "heldout_onset_aligned_probability.csv": 26,
        "physical_onset_labels.csv": 16000,
    }
    for name, expected_rows in timing_table_rows.items():
        with (timing_root / "tables" / name).open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected_rows, name
        if name == "physical_onset_labels.csv":
            assert len({row["task"] for row in rows}) == 40
            assert sum(row["failure"] == "True" for row in rows) == 532
            onset = [row for row in rows if row["onset_query"]]
            assert len(onset) == 260
            assert sum(float(row["onset_query"]) >= 12 for row in onset) == 222
    assert (
        timing_root / "figures/timing_constrained_probability_alarm.png"
    ).stat().st_size > 150000

    return {
        "trap_probability_prediction_files": len(prediction_files),
        "trap_probability_primary_alarm_rows_075": summary[
            "onset_hazard_alarm_rows_total"
        ],
        "trap_probability_a_to_b_auc": primary["A_to_B"]["roc_auc"],
        "trap_probability_b_to_a_auc": primary["B_to_A"]["roc_auc"],
        "trap_probability_cache_new_tasks": cache_summary["complete_tasks"],
        "trap_probability_cache_new_episodes": cache_summary["total_episodes"],
        "trap_probability_cache_new_eligible_rows": cache_summary[
            "eligible_query_rows"
        ],
        "trap_probability_cache_new_b050_failure_recall": b_threshold_050[
            "failure_recall_all"
        ],
        "trap_probability_timing_onsets": 260,
        "trap_probability_timing_eligible_onsets": 222,
        "trap_probability_timing_b001_heldout_timely": 7,
        "trap_probability_timing_b001_heldout_near": 1,
        "trap_probability_suitable_timing_threshold": False,
        "trap_probability_deployment_ready": False,
    }


def validate_docs() -> int:
    pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    checked = 0
    for document in [PACKAGE_ROOT / "README.md", *sorted((PACKAGE_ROOT / "docs").glob("*.md"))]:
        for target in pattern.findall(document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            path = (document.parent / target.split("#", 1)[0]).resolve()
            assert path.exists(), f"broken link in {document}: {target}"
            checked += 1
    return checked


def validate_moe_information_usage_audit() -> dict[str, object]:
    usage_root = PACKAGE_ROOT / "results/moe_information_usage_audit"
    usage = json.loads((usage_root / "summary.json").read_text(encoding="utf-8"))
    assert usage["schema"] == "himoe.moe_information_usage_audit.v1"
    assert usage["training"] is False
    assert usage["raw_hb_probabilities_per_query"] == 8 * 10 * 11 * 32
    assert (
        usage["current_detector_unique_hb_probabilities_accessed_per_query"]
        + usage["current_detector_hb_probabilities_never_accessed_per_query"]
        == usage["raw_hb_probabilities_per_query"]
    )
    causes = usage["combined_heldout_alarm_causes"]
    assert causes["persistent_recurrence"] == 163
    assert causes["convergence_and_response"] == 1
    assert causes["all_alarms"] == 164
    assert usage["as_route_audit"]["route_queries_checked"] == 508023
    assert usage["as_route_audit"]["constant_within_every_task_run"] is True
    id_audit = usage["hb_expert_id_reconstruction_audit"]
    assert id_audit["positions_checked"] == 28160
    assert id_audit["exact_top4_set_matches_from_float16_full_softmax"] == 20319
    assert id_audit["top4_set_mismatches"] == 7841
    assert id_audit["decision"] == "read stored hb_expert_ids directly"
    assert usage["position_grid"]["cells"] == 24

    two_run_root = PACKAGE_ROOT / "results/moe_invariant_alarm_two_run_audit"
    two_run = json.loads((two_run_root / "summary.json").read_text(encoding="utf-8"))
    assert two_run["schema"] == "himoe.moe_invariant_alarm_two_run_audit.v1"
    assert two_run["training"] is False
    assert two_run["data"]["task_runs"] == 80
    assert two_run["data"]["episodes"] == 32000
    assert two_run["data"]["route_queries"] == 508023
    assert two_run["formula_audit"]["values_compared"] == 3592
    assert two_run["implementation_audit"]["prediction_hash_exact_match"] is True
    heldout = two_run["combined_own_rule"]["heldout_tasks"]
    detector = heldout["detector"]
    assert heldout["episodes"] == 6400
    assert heldout["failures"] == 239
    assert detector["tp"] == 106
    assert detector["fp"] == 58
    assert detector["fn"] == 133
    assert detector["tn"] == 6103
    assert sum(detector[key] for key in ("tp", "fp", "fn", "tn")) == 6400
    assert two_run["conclusion"]["current_rule_is_valid_trap_detector"] is False
    assert (
        two_run["conclusion"]["poor_result_is_explained_by_axis_or_arithmetic_bug"]
        is False
    )

    return {
        "moe_information_raw_probabilities_per_query": usage[
            "raw_hb_probabilities_per_query"
        ],
        "moe_information_accessed_probabilities_per_query": usage[
            "current_detector_unique_hb_probabilities_accessed_per_query"
        ],
        "moe_information_unaccessed_probabilities_per_query": usage[
            "current_detector_hb_probabilities_never_accessed_per_query"
        ],
        "moe_two_run_heldout_episodes": heldout["episodes"],
        "moe_two_run_heldout_tp": detector["tp"],
        "moe_two_run_heldout_fp": detector["fp"],
    }


def validate_moe_structured_alarm() -> dict[str, object]:
    root = PACKAGE_ROOT / "results/moe_structured_alarm_two_runs"
    config = json.loads(
        (PACKAGE_ROOT / "configs/moe_structured_alarm_two_runs.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["training"] is False
    assert config["learned_feature_weights"] is False
    assert config["failure_labels_used_for_features_or_thresholds"] is False
    assert config["threshold_confirmation"]["failure_labels_used"] is False
    assert "maximum" in config["threshold_confirmation"]["selection"]

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "himoe.moe_structured_alarm_two_runs.v1"
    assert summary["training"] is False
    assert summary["learned_feature_weights"] is False
    assert summary["physical_inputs_to_detector"] is False
    assert summary["failure_labels_used_for_features_or_thresholds"] is False
    assert summary["route_queries"] == 508023
    assert summary["task_partition"] == {
        "healthy_reference": 24,
        "heldout_test": 8,
        "threshold_confirmation": 8,
    }
    heldout = summary["combined_heldout"]
    assert heldout["episodes"] == 6400
    assert heldout["failures"] == 239
    assert heldout["tp"] == 121
    assert heldout["fp"] == 17
    assert heldout["fn"] == 118
    assert heldout["tn"] == 6144
    assert sum(heldout[key] for key in ("tp", "fp", "fn", "tn")) == 6400

    partition = pd.read_csv(root / "tables/task_partition.csv")
    assert len(partition) == 40
    assert partition.groupby("suite")["role"].value_counts().to_dict() == {
        (suite, role): count
        for suite in ("libero_goal", "libero_long", "libero_object", "libero_spatial")
        for role, count in (
            ("healthy_reference", 6),
            ("heldout_test", 2),
            ("threshold_confirmation", 2),
        )
    }

    route_rows = 0
    cache_files = sorted((root / "route_only_tasks").glob("*/*.npz"))
    assert len(cache_files) == 80
    for path in cache_files:
        with np.load(path, allow_pickle=False) as archive:
            assert archive["schema"].item() == summary["schema"]
            assert archive["training"].item() is False
            assert archive["endpoint_labels_loaded"].item() is False
            assert len(archive["component_names"]) == 22
            assert archive["flow_soft_mobility"].shape[1:] == (2, 3, 4)
            assert archive["state_action_soft_gap"].shape[1:] == (2, 3, 3)
            assert archive["lag_selected_recurrence"].shape[1:] == (4, 2, 4)
            route_rows += len(archive["episode"])
    assert route_rows == 508023

    prediction_rows = {
        "seed1000_1007_to_seed1008_1015": 174301,
        "seed1008_1015_to_seed1000_1007": 173722,
    }
    for direction, expected_rows in prediction_rows.items():
        prediction = root / "predictions_label_free" / f"{direction}.csv.gz"
        manifest = json.loads(
            (root / "predictions_label_free" / f"{direction}_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["target_outcomes_loaded"] is False
        assert manifest["predictions_written_before_target_outcomes"] is True
        assert manifest["rows"] == expected_rows
        assert hashlib.sha256(prediction.read_bytes()).hexdigest() == manifest[
            "prediction_sha256"
        ]
        thresholds = json.loads(
            (root / f"thresholds_{direction}.json").read_text(encoding="utf-8")
        )
        assert thresholds["failure_labels_used"] is False
        assert thresholds["threshold_selection"] == "maximum_task_specific_threshold"
        assert all(
            values["selection"] == "maximum_task_specific_threshold"
            for values in thresholds["rules"].values()
        )

    with (root / "tables/direction_and_head_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 20
    with (root / "tables/heldout_onset_timing.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 120
    for direction in prediction_rows:
        with (root / f"tables/episode_flip_{direction}.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            assert len(list(csv.DictReader(stream))) == 16000
        with (root / f"tables/task_specific_thresholds_{direction}.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            assert len(list(csv.DictReader(stream))) == 32

    audit_root = root / "audit"
    audit = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    assert audit["schema"] == "himoe.moe_structured_alarm_audit.v1"
    assert audit["training"] is False
    assert audit["structured_task_robust"]["tp"] == 121
    assert audit["structured_task_robust"]["fp"] == 17
    assert audit["fair_scalar_task_robust"]["tp"] == 79
    assert audit["fair_scalar_task_robust"]["fp"] == 27
    assert audit["structured_minus_fair_scalar"] == {"tp": 42, "fp": -10}
    assert audit["structured_alarm_causes"] == {
        "lock_in": 134,
        "state_action_decoupling": 1,
        "instability": 3,
    }
    assert audit["onset"]["eligible_events"] == 120
    assert audit["onset"]["totals"]["combined"]["near_first_minus2_to_0"] == 10
    assert audit["onset"]["totals"]["combined"]["late"] == 38
    assert audit["conclusion"]["structured_information_improves_fair_scalar_rule"] is True
    assert audit["conclusion"]["reliable_early_detector_validated"] is False
    for manifest_path in sorted(
        (audit_root / "predictions_label_free").glob("*_manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prediction = manifest_path.with_name(
            manifest_path.name.replace("_manifest.json", ".csv.gz")
        )
        assert manifest["target_outcomes_loaded"] is False
        assert manifest["predictions_written_before_target_outcomes"] is True
        assert hashlib.sha256(prediction.read_bytes()).hexdigest() == manifest[
            "prediction_sha256"
        ]

    assert (root / "redundant_recurrence_views_v0/summary.json").exists()
    assert (root / "pooled_threshold_v1/summary.json").exists()
    return {
        "moe_structured_route_queries": summary["route_queries"],
        "moe_structured_heldout_episodes": heldout["episodes"],
        "moe_structured_heldout_tp": heldout["tp"],
        "moe_structured_heldout_fp": heldout["fp"],
        "moe_structured_fair_scalar_tp": audit["fair_scalar_task_robust"]["tp"],
        "moe_structured_fair_scalar_fp": audit["fair_scalar_task_robust"]["fp"],
        "moe_structured_near_onset_first_alarms": audit["onset"]["totals"][
            "combined"
        ]["near_first_minus2_to_0"],
        "moe_structured_reliable_early_detector": False,
    }


def validate_moe_expert_identity_ablation() -> dict[str, object]:
    root = PACKAGE_ROOT / "results/moe_expert_identity_ablation"
    config = json.loads(
        (PACKAGE_ROOT / "configs/moe_expert_identity_ablation.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["training"] is False
    assert config["gradient_optimization"] is False
    assert config["learned_feature_weights"] is False
    assert config["failure_labels_used_for_features_or_thresholds"] is False
    assert config["threshold_confirmation"]["failure_labels_used"] is False
    assert len(config["variants"]) == 8

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "himoe.moe_expert_identity_ablation.v1"
    assert summary["status"] == "complete"
    assert summary["training"] is False
    assert summary["gradient_optimization"] is False
    assert summary["learned_feature_weights"] is False
    assert summary["failure_labels_used_for_features_or_thresholds"] is False
    assert summary["route_rows"] == 508023
    assert summary["task_runs"] == 80
    assert summary["global_label_permutation_invariant"] is True
    permutation = summary["global_label_permutation_audit"]
    assert permutation["support_exact"] is True
    assert permutation["selected_max_abs_difference"] == 0.0
    assert permutation["soft_max_abs_difference"] < 1e-5

    reconstruction = summary["id_reconstruction"]
    assert reconstruction["positions"] == 508023 * 8 * 11
    assert reconstruction["stable_positions"] == 35971063
    assert reconstruction["stable_matches"] == reconstruction["stable_positions"]
    assert reconstruction["tied_positions"] == 8734961
    assert reconstruction["tied_matches"] == 0
    assert reconstruction["tie_pair_valid"] == 98016388
    assert reconstruction["tie_pair_total"] == 150664096

    expected_primary = {
        "soft_coordinate": (164, 22, 75, 6139),
        "actual_support": (103, 22, 136, 6139),
        "selected_weight": (101, 20, 138, 6141),
        "rank_shape": (33, 1, 206, 6160),
        "reconstructed_support": (80, 8, 159, 6153),
        "tie_resolved_support": (108, 21, 131, 6140),
        "chunk_relabel_soft": (0, 9, 239, 6152),
        "chunk_relabel_support": (3, 3, 236, 6158),
    }
    for variant, values in expected_primary.items():
        result = summary["combined_heldout"][variant]
        assert tuple(result[key] for key in ("tp", "fp", "fn", "tn")) == values
        assert result["episodes"] == 6400
        assert result["failures"] == 239

    conclusion = summary["conclusion"]
    assert conclusion == {
        "consistent_expert_identity_contributes_signal": True,
        "numeric_expert_label_has_semantics": False,
        "reliable_early_detector_validated": False,
        "stored_actual_top4_tiebreak_is_required": False,
    }
    assert summary["combined_onset"]["soft_coordinate"] == {
        "alarm_events": 88,
        "events": 120,
        "late": 73,
        "near_active_minus2_to_0": 12,
        "near_first_minus2_to_0": 11,
        "not_later": 15,
        "too_early": 4,
    }

    cache_files = sorted((root / "route_only_tasks").glob("*/*.npz"))
    assert len(cache_files) == 80
    route_rows = 0
    for path in cache_files:
        with np.load(path, allow_pickle=False) as archive:
            assert archive["schema"].item() == summary["schema"]
            assert archive["training"].item() is False
            assert archive["endpoint_labels_loaded"].item() is False
            assert archive["recurrence_raw"].shape[1:] == (8, 4)
            assert archive["recurrence_evidence"].shape == archive["recurrence_raw"].shape
            assert archive["variants"].tolist() == summary["variants"]
            route_rows += len(archive["episode"])
    assert route_rows == 508023

    expected_prediction_rows = {
        "seed1000_1007_to_seed1008_1015": 254301,
        "seed1008_1015_to_seed1000_1007": 253722,
    }
    for direction, expected_rows in expected_prediction_rows.items():
        prediction = root / "predictions_label_free" / f"{direction}.csv.gz"
        manifest = json.loads(
            (root / "predictions_label_free" / f"{direction}_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["target_outcomes_loaded"] is False
        assert manifest["predictions_written_before_target_outcomes"] is True
        assert manifest["rows"] == expected_rows
        digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
        assert digest == manifest["prediction_sha256"]
        assert digest == summary["prediction_hashes"][direction]
        thresholds = json.loads(
            (root / f"thresholds_{direction}.json").read_text(encoding="utf-8")
        )
        assert thresholds["failure_labels_used"] is False
        assert thresholds["selection"] == "maximum_task_specific_threshold"
        assert len(thresholds["thresholds"]) == 8

    expected_table_rows = {
        "combined_metrics.csv": 8,
        "combined_onset_metrics.csv": 8,
        "direction_metrics.csv": 16,
        "heldout_onset_timing.csv": 120,
        "posthoc_fp_budget_sweep.csv": 45,
        "task_partition.csv": 40,
        "task_thresholds_seed1000_1007_to_seed1008_1015.csv": 64,
        "task_thresholds_seed1008_1015_to_seed1000_1007.csv": 64,
        "episodes_seed1000_1007_to_seed1008_1015.csv": 16000,
        "episodes_seed1008_1015_to_seed1000_1007.csv": 16000,
    }
    for name, expected in expected_table_rows.items():
        with (root / "tables" / name).open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == expected

    sweep = pd.read_csv(root / "tables/posthoc_fp_budget_sweep.csv")
    equal_fp = sweep[sweep["fp_budget"] == 22].set_index("variant")
    expected_equal_fp = {
        "soft_coordinate": (164, 22),
        "actual_support": (116, 21),
        "rank_shape": (84, 22),
        "reconstructed_support": (122, 22),
        "tie_resolved_support": (115, 22),
        "chunk_relabel_soft": (0, 22),
        "chunk_relabel_support": (24, 22),
        "fixed_query_clock": (95, 20),
    }
    for variant, (tp, fp) in expected_equal_fp.items():
        assert int(equal_fp.loc[variant, "tp"]) == tp
        assert int(equal_fp.loc[variant, "fp"]) == fp

    return {
        "moe_identity_route_queries": summary["route_rows"],
        "moe_identity_soft_tp": summary["combined_heldout"]["soft_coordinate"]["tp"],
        "moe_identity_soft_fp": summary["combined_heldout"]["soft_coordinate"]["fp"],
        "moe_identity_equal22_rank_tp": int(equal_fp.loc["rank_shape", "tp"]),
        "moe_identity_equal22_chunk_relabel_soft_tp": int(
            equal_fp.loc["chunk_relabel_soft", "tp"]
        ),
        "moe_identity_actual_tiebreak_required": False,
        "moe_identity_reliable_early_detector": False,
    }


def validate_checksums() -> int:
    checksum_file = PACKAGE_ROOT / "SHA256SUMS"
    if not checksum_file.exists():
        return 0
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = (PACKAGE_ROOT / relative).resolve()
        assert path.is_relative_to(PACKAGE_ROOT.resolve())
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == digest, relative
        checked += 1
    return checked


def main() -> int:
    replicated = validate_offline()
    recovery = validate_recovery()
    legacy = validate_legacy()
    failed_grasp = validate_failed_grasp()
    belief_mismatch = validate_belief_mismatch()
    belief_selector = validate_belief_selector()
    online_alarm = validate_online_alarm()
    moe_only_online = validate_moe_only_online()
    moe_dynamics_v2 = validate_moe_dynamics_v2()
    large_moe_only = validate_large_moe_only_offline()
    cache_new_moe_only = validate_cache_new_moe_only_offline()
    task_difficulty = validate_task_difficulty_weighting()
    self_reference = validate_self_reference_alarm()
    input_version = validate_input_version_counterfactual()
    trap_probability = validate_trainfree_trap_probability()
    moe_invariant_alarm = validate_moe_invariant_alarm()
    moe_information_usage = validate_moe_information_usage_audit()
    moe_structured_alarm = validate_moe_structured_alarm()
    moe_expert_identity = validate_moe_expert_identity_ablation()
    doc_links = validate_docs()
    checksums = validate_checksums()
    print(json.dumps({
        "status": "VALIDATION_OK",
        "training": False,
        "replicated_cells": replicated,
        **recovery,
        **legacy,
        **failed_grasp,
        **belief_mismatch,
        **belief_selector,
        **online_alarm,
        **moe_only_online,
        **moe_dynamics_v2,
        **large_moe_only,
        **cache_new_moe_only,
        **task_difficulty,
        **self_reference,
        **input_version,
        **trap_probability,
        **moe_invariant_alarm,
        **moe_information_usage,
        **moe_structured_alarm,
        **moe_expert_identity,
        "document_links_verified": doc_links,
        "checksums_verified": checksums,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
