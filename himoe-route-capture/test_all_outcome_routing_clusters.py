import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from analyze_all_outcome_routing_clusters import (
    ALL_VIEWS,
    PRESERVED_ARI,
    REORGANIZED_ARI,
    _change_class,
    _conditional_entropy,
    add_bh_correction,
    add_sensitivity_bh_correction,
    compare_failure_partitions,
    maximum_unique_matches,
    minimum_cluster_size,
    per_task_outcome_associations,
)


def test_minimum_cluster_size_uses_floor_until_two_percent_exceeds_it():
    assert minimum_cluster_size(0) == 16
    assert minimum_cluster_size(1) == 16
    assert minimum_cluster_size(800) == 16
    assert minimum_cluster_size(801) == 17
    assert minimum_cluster_size(2_560) == 52


@pytest.mark.parametrize(
    ("ari", "expected"),
    [
        (PRESERVED_ARI, "preserved"),
        (np.nextafter(PRESERVED_ARI, -np.inf), "moderately_changed"),
        (REORGANIZED_ARI, "moderately_changed"),
        (np.nextafter(REORGANIZED_ARI, -np.inf), "reorganized"),
        (-1.0, "reorganized"),
        (1.0, "preserved"),
    ],
)
def test_change_class_boundaries(ari, expected):
    assert _change_class(ari) == expected


def test_conditional_entropy_has_the_documented_direction():
    coarse = np.asarray([0, 0, 1, 1], dtype=np.int32)
    fine = np.asarray([4, 7, 9, 12], dtype=np.int32)

    assert _conditional_entropy(fine, coarse) == pytest.approx(0.0)
    assert _conditional_entropy(coarse, fine) == pytest.approx(1.0)


def test_conditional_entropy_is_one_bit_for_balanced_independence():
    left = np.asarray([0, 0, 1, 1], dtype=np.int32)
    right = np.asarray([0, 1, 0, 1], dtype=np.int32)
    assert _conditional_entropy(left, right) == pytest.approx(1.0)


def _association(p_value):
    return {"permutation_p_one_sided": p_value}


def test_add_bh_correction_matches_known_step_up_values():
    p_values = dict(zip(ALL_VIEWS, [0.01, 0.04, 0.03, 0.002]))
    results = {
        view: {"posthoc_association": {"outcome": _association(p_value)}}
        for view, p_value in p_values.items()
    }

    add_bh_correction(results)

    expected = dict(zip(ALL_VIEWS, [0.02, 0.04, 0.04, 0.008]))
    for view in ALL_VIEWS:
        association = results[view]["posthoc_association"]["outcome"]
        assert association["permutation_p_one_sided"] == p_values[view]
        assert association["fdr_bh_q"] == pytest.approx(expected[view])


def test_sensitivity_bh_correction_caps_values_at_one():
    rows = [
        {"outcome_association": _association(value)}
        for value in (0.9, 0.8, 0.7)
    ]
    add_sensitivity_bh_correction(rows)
    assert [row["outcome_association"]["fdr_bh_q"] for row in rows] == pytest.approx(
        [0.9, 0.9, 0.9]
    )


def _matching_metadata():
    rows = []

    def add_group(task, initial, failures, successes):
        for failure in [True] * failures + [False] * successes:
            rows.append(
                {
                    "task": task,
                    "init_state_id": initial,
                    "failure": failure,
                    "flow_noise_seed": len(rows) % 37,
                    "episode": len(rows),
                }
            )

    add_group("task-a", 0, failures=100, successes=110)
    add_group("task-a", 1, failures=60, successes=57)
    add_group("task-b", 0, failures=3, successes=0)
    add_group("task-b", 1, failures=0, successes=4)

    permutation = np.random.default_rng(9).permutation(len(rows))
    return {
        key: np.asarray([rows[index][key] for index in permutation])
        for key in rows[0]
    }


def test_maximum_unique_matches_is_maximal_unique_and_within_stratum():
    metadata = _matching_metadata()
    failures, successes = maximum_unique_matches(metadata)

    assert len(failures) == 157
    assert len(successes) == 157
    assert len(np.unique(failures)) == 157
    assert len(np.unique(successes)) == 157
    assert np.all(metadata["failure"][failures])
    assert not np.any(metadata["failure"][successes])
    np.testing.assert_array_equal(metadata["task"][failures], metadata["task"][successes])
    np.testing.assert_array_equal(
        metadata["init_state_id"][failures], metadata["init_state_id"][successes]
    )

    matched_strata, counts = np.unique(
        np.asarray(
            [
                f"{task}::{initial}"
                for task, initial in zip(
                    metadata["task"][failures], metadata["init_state_id"][failures]
                )
            ]
        ),
        return_counts=True,
    )
    assert dict(zip(matched_strata, counts)) == {"task-a::0": 100, "task-a::1": 57}


def test_maximum_unique_matches_rejects_nonformal_pair_count():
    metadata = {
        "task": np.asarray(["task", "task"]),
        "init_state_id": np.asarray([0, 0]),
        "failure": np.asarray([True, False]),
        "flow_noise_seed": np.asarray([0, 1]),
        "episode": np.asarray([0, 1]),
    }
    with pytest.raises(ValueError, match="expected 157 unique task/init matched pairs"):
        maximum_unique_matches(metadata)


def test_failure_comparison_transition_table_matches_returned_partition():
    failure = np.asarray([True, True, True, True, True, True, False, False])
    metadata = {
        "failure": failure,
        "task": np.asarray(["task"] * len(failure)),
        "episode": np.arange(len(failure), dtype=np.int32),
    }
    old = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    old_map = {
        ("task", episode): {"geometry": int(label)}
        for episode, label in zip(np.flatnonzero(failure), old)
    }
    embedding = np.asarray(
        [
            [-10.0, 0.0],
            [-9.9, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
            [-9.8, 0.0],
            [9.8, 0.0],
            [-10.2, 0.0],
            [10.2, 0.0],
        ]
    )
    score = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    joint_labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)

    comparison, same_k_full = compare_failure_partitions(
        "geometry",
        joint_labels,
        embedding,
        metadata,
        score,
        old_map,
        {"geometry": "stable_partition"},
    )

    same_k = same_k_full[failure]
    columns = np.asarray(comparison["joint_same_k_column_labels"])
    expected_table = np.asarray(
        [
            [np.sum((old == left) & (same_k == right)) for right in columns]
            for left in range(len(np.unique(old)))
        ]
    )
    np.testing.assert_array_equal(comparison["old_by_joint_same_k_table"], expected_table)
    np.testing.assert_array_equal(expected_table.sum(axis=1), np.bincount(old))
    np.testing.assert_array_equal(
        expected_table.sum(axis=0),
        np.asarray([np.sum(same_k == value) for value in columns]),
    )
    assert expected_table.sum() == failure.sum()
    assert comparison["same_k_ari"] == pytest.approx(adjusted_rand_score(old, same_k))
    assert comparison["split_entropy_joint_given_old_bits"] == pytest.approx(
        _conditional_entropy(old, same_k)
    )
    assert comparison["merge_entropy_old_given_joint_bits"] == pytest.approx(
        _conditional_entropy(same_k, old)
    )


def test_per_task_association_remaps_missing_global_cluster_ids():
    metadata = {
        "task": np.asarray(["mixed"] * 4 + ["all-success"] * 2),
        "failure": np.asarray([True, False, True, False, False, False]),
        "init_state_id": np.asarray([0, 0, 1, 1, 0, 1]),
    }
    labels = np.asarray([1, 1, 3, 3, 0, 0], dtype=np.int32)

    result = per_task_outcome_associations(metadata, labels, 10, 7)

    assert result["mixed"]["status"] == "estimated"
    assert result["mixed"]["episodes"] == 4
    assert result["all-success"]["status"] == "no_outcome_variation"
