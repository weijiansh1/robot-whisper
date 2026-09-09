import numpy as np

from audit_post_error_dense_replays import first_true_run, max_true_run
from analyze_post_error_recovery import (
    Event,
    MAIN_RULE,
    detect_target_events,
    double_holdout_feasibility,
    event_counts,
    hidden_cosine_similarity,
    identify_targets,
    route_hellinger_similarity,
    sensitivity_summary,
    top4_jaccard_similarity,
)


def synthetic_loss(close_command: float = 1.0):
    length = 8
    position = np.zeros((length, 3), np.float32)
    eef = np.zeros((length, 3), np.float32)
    position[:, 2] = 1.0
    eef[:, 2] = 1.0
    position[2] = [0.02, 0.0, 1.02]
    position[3] = [0.04, 0.0, 1.04]
    position[4] = [0.04, 0.0, 1.015]
    eef[2:4] = position[2:4]
    eef[4] = [0.20, 0.0, 1.04]
    state = np.zeros((length, 8), np.float32)
    state[:, :3] = eef
    state[:, 6:8] = 0.01
    state[1, 6:8] = 0.04
    action = np.zeros((length, 10, 7), np.float32)
    action[:, :, 6] = close_command
    return position, state, action


def test_transport_loss_is_assigned_to_next_query():
    position, state, action = synthetic_loss()
    events, evidence, _distance = detect_target_events(
        position, state, action, 0.085, MAIN_RULE
    )
    assert np.flatnonzero(evidence).tolist() == [2, 3]
    assert len(events) == 1
    assert events[0]["bout_start"] == 2
    assert events[0]["held_end"] == 3
    assert events[0]["drop_query"] == 4


def test_open_command_rejects_transport_loss_proxy():
    position, state, action = synthetic_loss(close_command=-1.0)
    events, _evidence, _distance = detect_target_events(
        position, state, action, 0.085, MAIN_RULE
    )
    assert events == []


def test_recurrence_identity_metrics_equal_one():
    probability = np.full((8, 10, 11, 32), 1.0 / 32, np.float32)
    expert = np.tile(np.arange(4, dtype=np.uint8), (8, 10, 11, 1))
    hidden = np.arange(8 * 10 * 11 * 5, dtype=np.float32).reshape(8, 10, 11, 5)
    assert route_hellinger_similarity(probability, probability, slice(1, 11)) == 1.0
    assert top4_jaccard_similarity(expert, expert, slice(1, 11)) == 1.0
    assert np.isclose(hidden_cosine_similarity(hidden, hidden, slice(1, 11)), 1.0)


def make_event(task: str, state: int, seed: int, success: bool) -> Event:
    return Event(
        task=task,
        episode=seed,
        init_state_id=state,
        flow_noise_seed=seed,
        target="object",
        bout_start=2,
        held_end=3,
        drop_query=4,
        failed_chunk_anchor=3,
        grasp_anchor=1,
        anchor_confidence="high",
        inference_calls=10,
        remaining_queries=5,
        terminal_success=success,
        future_label=(
            "terminal_success_after_proxy"
            if success
            else "terminal_failure_after_proxy"
        ),
        later_rehold_query=6 if success else None,
        future_proxy_rehold=success,
        hold_distance_threshold_m=0.085,
        object_drop_m=0.02,
        separation_increase_m=0.05,
        max_lift_in_bout_m=0.04,
        object_distance_at_post_m=0.10,
        eef_distance_to_grasp_anchor_m=0.02,
        target_distance_to_grasp_anchor_m=0.01,
    )


def test_gate_inventory_counts_only_within_cluster_pairs():
    events = [
        make_event("a", 0, 0, False),
        make_event("a", 0, 1, True),
        make_event("a", 1, 2, True),
        make_event("b", 0, 3, False),
    ]
    counts = event_counts(events)
    assert counts["mixed_task_initial_state_clusters"] == 1
    assert counts["conditional_pairs"] == 1
    folds = double_holdout_feasibility(events)
    assert folds["all_events_tested_once"]
    assert not folds["all_folds_evaluable"]


def test_empty_double_holdout_has_complete_failed_schema():
    folds = double_holdout_feasibility([])
    assert folds["all_folds_evaluable"] is False
    assert folds["all_events_tested_once"] is False
    assert folds["details"] == []


def test_targets_come_from_frozen_task_goal_mapping():
    layout = {
        "joints": [
            {
                "joint": "akita_black_bowl_1_joint0",
                "is_robot": False,
                "state_lo": 10,
                "state_hi": 17,
            },
            {
                "joint": "plate_1_joint0",
                "is_robot": False,
                "state_lo": 17,
                "state_hi": 24,
            },
        ]
    }
    targets = identify_targets(
        "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        layout,
    )
    assert [target.name for target in targets] == ["akita_black_bowl_1_joint0"]


def test_sensitivity_summary_tracks_identity_and_query_shift():
    main = [make_event("a", 0, 0, False), make_event("a", 0, 1, True)]
    shifted = make_event("a", 0, 0, False)
    shifted.drop_query = 5
    variant = [shifted, make_event("a", 1, 2, False)]
    result = sensitivity_summary(main, variant)
    stability = result["identity_stability"]
    assert stability["retained_main_events"] == 1
    assert stability["removed_main_events"] == 1
    assert stability["added_events"] == 1
    assert stability["shifted_drop_queries"] == 1


def test_dense_contact_run_helpers_require_consecutive_samples():
    mask = np.asarray([False, True, True, False, True, True, True, False])
    assert max_true_run(mask) == 3
    assert first_true_run(mask, 3) == 4
    assert first_true_run(mask, 4) is None
