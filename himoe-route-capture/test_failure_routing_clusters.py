import numpy as np

from analyze_failure_routing_clusters import (
    ALL_VIEWS,
    N_DENOISE,
    N_EXPERTS,
    N_LAYERS,
    PRIMARY_VIEWS,
    PRIMARY_PHASE,
    annotate_interpretation,
    build_representations,
    cluster_color_scale,
    cluster_view,
    phase_interpolate,
    recurrence_features,
    render_report,
    selected_expert_occupancy,
)


def _interpretation_fixture(status: str = "stable_partition") -> dict:
    association = {
        "nmi": 0.0,
        "null_nmi_mean": 0.0,
        "nmi_excess_over_null": 0.0,
        "fdr_bh_q": 1.0,
        "permutation_p_one_sided": 1.0,
        "permutation": "global",
    }
    return {
        "status": status,
        "selected_k": 2 if status == "stable_partition" else None,
        "exploratory_best_k": 2,
        "reported_k": 2,
        "trials": [
            {
                "clusters": 2,
                "subsample_silhouette_mean": 0.4,
                "ari_median": 0.9,
                "ari_p10": 0.8,
                "consensus_pac_01_09": 0.1,
                "sizes": [20, 20],
            }
        ],
        "posthoc_association": {
            "task": dict(association),
            "failure_mode_proxy": dict(association),
        },
        "profiles": [
            {
                "cluster": 0,
                "episodes": 20,
                "dominant_task_fraction": 0.5,
                "task_initial_state_groups": 4,
                "task_counts": {"suite/task": 20},
                "episode_length_counts": {"22": 20},
                "checkpoint_counts": {"abc": 20},
                "failure_mode_proxy_counts": {"partial": 20},
                "diagnostic_means": {
                    "mean_soft_speed": 0.1,
                    "late_soft_speed": 0.1,
                    "middle_to_terminal_drift": 0.2,
                    "backward_return_fraction": 0.3,
                },
            }
        ],
    }


def _continuous_tapes(length: int) -> dict[str, np.ndarray]:
    phase = np.linspace(0.0, 1.0, length, dtype=np.float32)
    geometry = np.column_stack(
        [phase, 1.0 - phase, 0.25 + 0.5 * phase]
    ).astype(np.float32)

    action = np.zeros(
        (length, N_LAYERS, N_DENOISE, N_EXPERTS), dtype=np.float32
    )
    action[..., 0] = (0.8 - 0.4 * phase)[:, None, None]
    action[..., 1] = (0.2 + 0.4 * phase)[:, None, None]

    hard = np.zeros((length, N_LAYERS, N_EXPERTS), dtype=np.float32)
    hard[..., 2] = (0.7 - 0.2 * phase)[:, None]
    hard[..., 3] = (0.3 + 0.2 * phase)[:, None]
    return {"geometry": geometry, "action": action, "hard": hard}


def test_phase_interpolation_uses_relative_episode_progress():
    short = phase_interpolate(_continuous_tapes(22)["geometry"], PRIMARY_PHASE)
    long = phase_interpolate(_continuous_tapes(52)["geometry"], PRIMARY_PHASE)
    np.testing.assert_allclose(short, long, atol=1e-6)


def test_all_trajectory_views_are_independent_of_sampling_length_for_linear_path():
    short, _ = build_representations(_continuous_tapes(22), PRIMARY_PHASE)
    long, _ = build_representations(_continuous_tapes(52), PRIMARY_PHASE)
    for view in ("geometry", "change", "recurrence", "expert_occupancy"):
        np.testing.assert_allclose(short[view], long[view], atol=1e-5)


def test_selected_expert_occupancy_counts_all_action_sites_equally():
    ids = np.zeros(
        (2, N_LAYERS, N_DENOISE, 11, 4), dtype=np.uint8
    )
    ids[..., 0] = 4
    ids[..., 1] = 7
    ids[..., 2] = 9
    ids[..., 3] = 12
    occupancy = selected_expert_occupancy(ids)
    np.testing.assert_allclose(occupancy[..., [4, 7, 9, 12]], 0.25)
    np.testing.assert_allclose(occupancy.sum(axis=-1), 1.0)


def test_recurrence_is_invariant_to_checkpoint_wide_expert_permutation():
    rng = np.random.default_rng(8)
    action = rng.random((7, N_LAYERS, N_DENOISE, N_EXPERTS), dtype=np.float32)
    action /= action.sum(axis=-1, keepdims=True)
    hard = rng.random((7, N_LAYERS, N_EXPERTS), dtype=np.float32)
    hard /= hard.sum(axis=-1, keepdims=True)
    permutation = rng.permutation(N_EXPERTS)
    np.testing.assert_allclose(
        recurrence_features(action, hard),
        recurrence_features(action[..., permutation], hard[..., permutation]),
        atol=1e-6,
    )


def test_stability_protocol_recovers_well_separated_three_cluster_data():
    rng = np.random.default_rng(19)
    centers = np.asarray(
        [[-8.0, -8.0], [0.0, 8.0], [8.0, -8.0]],
        dtype=np.float64,
    )
    matrix = np.vstack(
        [center + rng.normal(scale=0.05, size=(24, 2)) for center in centers]
    )
    score = np.repeat([0.0, 1.0, 2.0], 24)
    result, labels, _embedding = cluster_view(
        matrix, "geometry", score, subsamples=20, seed=31
    )
    assert result["status"] == "stable_partition"
    assert result["selected_k"] == 3
    assert np.array_equal(np.bincount(labels), [24, 24, 24])


def test_phase_sensitivity_rejects_partition_with_undersized_cluster():
    results = {view: _interpretation_fixture() for view in ALL_VIEWS}
    cross_view = {
        left: {right: 1.0 for right in ALL_VIEWS} for left in ALL_VIEWS
    }
    sensitivity = [
        {
            "view": view,
            "ari_to_primary": 0.9,
            "minimum_size_pass": view != "recurrence",
        }
        for view in PRIMARY_VIEWS
    ]
    annotate_interpretation(results, cross_view, sensitivity)
    assert "phase_sensitive" in results["recurrence"]["interpretation_flags"]
    assert "phase_sensitive" not in results["geometry"]["interpretation_flags"]


def test_cross_view_replication_requires_a_stable_peer():
    results = {view: _interpretation_fixture() for view in ALL_VIEWS}
    results["change"] = _interpretation_fixture("no_stable_partition")
    results["recurrence"] = _interpretation_fixture("no_stable_partition")
    cross_view = {
        left: {right: 1.0 for right in ALL_VIEWS} for left in ALL_VIEWS
    }
    sensitivity = [
        {"view": view, "ari_to_primary": 1.0, "minimum_size_pass": True}
        for view in PRIMARY_VIEWS
    ]
    annotate_interpretation(results, cross_view, sensitivity)
    assert "no_stable_cross_view_replication" in results["geometry"][
        "interpretation_flags"
    ]


def test_cluster_color_scale_maps_labels_consistently():
    cmap, norm = cluster_color_scale(4)
    colors = cmap(norm(np.arange(4)))
    assert len(np.unique(colors, axis=0)) == 4
    np.testing.assert_allclose(cmap(norm([2]))[0], colors[2])


def test_report_renderer_accepts_complete_schema():
    results = {view: _interpretation_fixture() for view in ALL_VIEWS}
    for view in ALL_VIEWS:
        results[view]["interpretation_status"] = "exploratory_only"
    summary = {
        "provenance": {"run_class": "test"},
        "data_audit": {
            "coverage": [
                {
                    "task": "suite/task",
                    "failures": 40,
                    "failure_lengths": [22],
                    "checkpoint_sha256": "abc",
                }
            ],
            "failures": 40,
            "failure_queries": 880,
            "distinct_checkpoint_sha256": 1,
        },
        "views": results,
        "cross_view_ari": {
            left: {right: 1.0 for right in ALL_VIEWS} for left in ALL_VIEWS
        },
        "phase_sensitivity": [
            {
                "view": view,
                "configuration": "grid8",
                "fixed_k": 2,
                "ari_to_primary": 1.0,
                "fixed_k_silhouette": 0.4,
                "fixed_k_sizes": [20, 20],
                "minimum_size_pass": True,
                "size_valid_best_k": 2,
            }
            for view in PRIMARY_VIEWS
        ],
    }
    report = render_report(summary)
    assert "Normalized middle/late routing clusters" in report
    assert "mean subsample silhouette" in report
    assert "min-size pass" in report
