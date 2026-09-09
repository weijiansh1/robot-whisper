from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import analyze_runtime_vector_v3 as probe


def _synthetic_data(states: int = 3, seeds: int = 3) -> probe.RuntimeVectorData:
    n = states * seeds
    x = np.zeros((n, 11, 10, 7), dtype=np.float64)
    state_group = np.repeat(np.arange(states), seeds)
    seed_group = np.tile(np.arange(seeds), states)
    return probe.RuntimeVectorData(
        x_traj=x,
        input_hidden=np.zeros((n, 10, 2)),
        shared_output=np.zeros((n, 10, 2)),
        router_probs=np.full((n, 10, 32), 1.0 / 32.0),
        selected_ids=np.zeros((n, 10, 4), dtype=np.int64),
        selected_weights=np.full((n, 10, 4), 0.25),
        selected_raw=np.zeros((n, 10, 4, 2)),
        state_group=state_group,
        seed_group=seed_group,
        task_group=np.full(n, "task-0"),
        checkpoint_group=np.full(n, "checkpoint-0"),
        sample_id=np.asarray([f"sample-{index}" for index in range(n)]),
    )


def test_primary_path_energy_and_secondary_x10_minus_x1_axes() -> None:
    x = np.zeros((2, 11, 10, 9), dtype=np.float64)
    x[:, 0, :, :7] = 500.0  # Must not enter the scientific target.
    x[:, 1, :, :7] = np.arange(70).reshape(10, 7)
    for step in range(2, 11):
        x[:, step, :, :7] = x[:, step - 1, :, :7] + 2.0
    primary = probe.future_path_energy_target(x)
    secondary = probe.remaining_correction_target(x)
    np.testing.assert_allclose(primary, 2.0)
    assert secondary.shape == (2, 70)
    np.testing.assert_array_equal(secondary, 18.0)
    np.testing.assert_array_equal(probe.immediate_control_target(x)[0], -500 + np.arange(70))


def test_state_and_seed_axes_are_both_disjoint_and_cover_once() -> None:
    states = np.repeat(np.arange(4), 5)
    seeds = np.tile(np.arange(5), 4)
    folds = probe.state_seed_disjoint_folds(states, seeds, n_splits=3, split_seed=8)
    coverage = np.zeros(len(states), dtype=np.int64)
    for fold in folds:
        assert set(states[fold.train]).isdisjoint(states[fold.test])
        assert set(seeds[fold.train]).isdisjoint(seeds[fold.test])
        coverage[fold.test] += 1
    np.testing.assert_array_equal(coverage, 1)


def test_raw_expert_representation_uses_identity_not_slot() -> None:
    ids = np.asarray([[[2, 7]]])
    raw = np.asarray([[[[1.0, 2.0], [10.0, 20.0]]]])
    represented = probe.scatter_expert_vectors(ids, raw, n_experts=8)
    np.testing.assert_array_equal(represented[0, 0, 2], [1.0, 2.0])
    np.testing.assert_array_equal(represented[0, 0, 7], [10.0, 20.0])

    permuted = probe.scatter_expert_vectors(ids[..., ::-1], raw[..., ::-1, :], 8)
    np.testing.assert_array_equal(permuted, represented)
    assert represented.shape == (1, 1, 8, 2)


def test_invariant_s_loo_and_gram_ignore_slot_permutation() -> None:
    raw = np.asarray([[[[1.0], [3.0]]]])
    weight = np.asarray([[[0.25, 0.75]]])
    original = probe.expert_output_blocks(raw, weight)
    permuted = probe.expert_output_blocks(raw[:, :, ::-1], weight[:, :, ::-1])
    np.testing.assert_allclose(original["routed"], 2.5)
    np.testing.assert_allclose(original["s_loo_squared"], 1.75)
    np.testing.assert_allclose(original["s_loo"], np.sqrt(1.75))
    np.testing.assert_allclose(original["s_loo_features"], permuted["s_loo_features"])
    np.testing.assert_allclose(original["gram_geometry"], permuted["gram_geometry"])
    quantized = probe.expert_output_blocks(raw, weight * 0.994)
    np.testing.assert_allclose(original["s_loo_features"], quantized["s_loo_features"])
    np.testing.assert_allclose(original["gram_geometry"], quantized["gram_geometry"])
    assert not np.allclose(original["routed"], quantized["routed"])


def test_baselines_are_strictly_nested_and_moe_is_only_an_addition() -> None:
    tiers = probe.baseline_tiers()
    names = list(tiers)
    assert names == [
        "b0_x0_x1",
        "b0_plus_hidden",
        "b0_plus_hidden_shared",
        "b1_operational",
    ]
    for left, right in zip(names, names[1:]):
        assert tiers[left] == tiers[right][: len(tiers[left])]
        assert len(tiers[right]) > len(tiers[left])
    assert tiers["b0_x0_x1"] == ("x0", "x1")
    assert set(probe.ID_AWARE_EXPERT_FAMILIES).isdisjoint(
        tiers["b1_operational"]
    )
    assert set(probe.INVARIANT_EXPERT_FAMILIES).isdisjoint(
        tiers["b1_operational"]
    )
    assert "hb5_actual_merged_routed" in tiers["b1_operational"]
    assert "hb5_router_probs" in tiers["b1_operational"]
    assert "hb5_selected_id_one_hot" in tiers["b1_operational"]
    assert "hb5_selected_weight_by_id" in tiers["b1_operational"]


def test_k1_smoke_is_refused_for_inference() -> None:
    data = _synthetic_data(states=1, seeds=1)
    with pytest.raises(probe.AdmissionError, match="K1/v3 smoke"):
        probe.validate_admission(data, outer_splits=3)


def test_repeated_noise_identity_requires_matching_full_x0() -> None:
    data = _synthetic_data()
    data.x_traj[0, 0, 0, 0] = 0.1
    with pytest.raises(probe.AdmissionError, match="max x0 disagreement"):
        probe.validate_admission(data, outer_splits=3)


def test_held_target_cannot_change_its_own_cross_fitted_prediction() -> None:
    data = _synthetic_data()
    rng = np.random.default_rng(4)
    projected = {"x1": rng.normal(size=(9, 5))}
    target = rng.normal(size=(9, 3))
    original, _, original_fold = probe.cross_validated_predictions(
        projected,
        (("x1",),),
        target,
        data.state_group,
        data.seed_group,
        outer_splits=3,
        split_seed=19,
    )
    changed = target.copy()
    changed[0] += 1e6
    repeated, _, repeated_fold = probe.cross_validated_predictions(
        projected,
        (("x1",),),
        changed,
        data.state_group,
        data.seed_group,
        outer_splits=3,
        split_seed=19,
    )
    np.testing.assert_allclose(repeated[0], original[0], atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(repeated_fold, original_fold)


def test_appended_expert_block_keeps_b1_design_verbatim() -> None:
    rng = np.random.default_rng(7)
    projected = {
        "b1a": rng.normal(size=(9, 4)),
        "b1b": rng.normal(size=(9, 4)),
        "expert": rng.normal(size=(9, 4)),
    }
    train = np.arange(6)
    apply = np.arange(6, 9)
    b1_train, b1_apply = probe._design(
        projected, (("b1a", "b1b"),), train, apply
    )
    full_train, full_apply = probe._design(
        projected, (("b1a", "b1b"), ("expert",)), train, apply
    )
    np.testing.assert_array_equal(full_train[:, :4], b1_train)
    np.testing.assert_array_equal(full_apply[:, :4], b1_apply)
    assert full_train.shape[1] == 2 * b1_train.shape[1]


def test_small_invariant_blocks_are_exact_not_sketched() -> None:
    data = _synthetic_data()
    projected = probe.project_feature_families(data, width=5, projection_seed=3)
    assert projected[probe.S_LOO_FAMILY].shape == (9, 11)
    assert projected[probe.INVARIANT_GEOMETRY_FAMILY].shape == (9, 110)
    assert projected[probe.ROUTED_RMS_CONTROL_FAMILY].shape == (9, 11)
    assert projected["x0"].shape == (9, 5)


def test_geometry_never_compares_predictions_from_different_seed_fold_models() -> None:
    target = np.asarray([[0.0], [1.0], [3.0], [10.0], [11.0], [13.0]])
    prediction = np.asarray([[100.0], [101.0], [103.0], [0.0], [1.0], [3.0]])
    states = np.full(6, "state")
    fold_id = np.asarray([0, 0, 0, 1, 1, 1])
    result = probe.geometry_spearman(prediction, target, states, fold_id)
    assert result["pooled_within_state"] == pytest.approx(1.0)


def test_k8_uses_largest_remainder_seed_fold_allocation() -> None:
    value = np.arange(16, dtype=np.float64)[:, None]
    folds = np.asarray([0] * 6 + [1] * 5 + [2] * 5)
    result = probe.k8_geometry_coverage(
        value, value, np.full(16, "state"), folds
    )
    assert result["per_state"]["state"]["seed_fold_sizes"] == [6, 5, 5]
    assert result["per_state"]["state"]["stratified_k8_allocation"] == [3, 3, 2]


def test_grid_root_maps_each_client_to_its_suite_store(tmp_path, monkeypatch) -> None:
    benchmarks = [
        "libero_goal",
        "libero_goal",
        "libero_spatial",
        "libero_spatial",
        "libero_10",
    ]
    for suite in ("goal", "spatial", "long"):
        (tmp_path / "server" / suite / "activation_flow.zarr").mkdir(parents=True)
    for index, benchmark in enumerate(benchmarks):
        run = tmp_path / "client" / f"task-{index}"
        run.mkdir(parents=True)
        (run / "experiment_config.json").write_text(
            json.dumps({"benchmark": benchmark})
        )
        (run / "query_records.json").write_text("[]")

    calls = []

    def fake_load(run_dir, store_path):
        calls.append((run_dir.name, store_path.parent.name))
        data = _synthetic_data(states=1, seeds=1)
        data.task_group[0] = run_dir.name
        data.sample_id[0] = run_dir.name
        return data

    monkeypatch.setattr(probe, "_load_one_run", fake_load)
    loaded = probe.load_grid_root(tmp_path)
    assert len(loaded.x_traj) == 5
    assert calls == [
        ("task-0", "goal"),
        ("task-1", "goal"),
        ("task-2", "spatial"),
        ("task-3", "spatial"),
        ("task-4", "long"),
    ]


def test_five_task_decision_uses_relative_effect_and_all_projection_seeds(
    monkeypatch,
) -> None:
    tasks = []
    expected_tasks = sorted(probe.EXPECTED_TASKS)
    for task, task_name in enumerate(expected_tasks):
        data = _synthetic_data(states=8, seeds=16)
        data = replace(
            data,
            task_group=np.full(128, task_name),
            checkpoint_group=np.full(128, f"checkpoint-{task // 2}"),
            state_group=np.asarray(
                [
                    f"{task_name}:state-{state}"
                    for state in np.repeat(np.arange(8), 16)
                ]
            ),
            sample_id=np.asarray(
                [f"{task_name}:sample-{row}" for row in range(128)]
            ),
        )
        tasks.append(data)
    grid = probe.concatenate_data(tasks)

    def metric(rmse, states):
        return {
            "held_out_rmse": rmse,
            "held_out_geometry_spearman": {"pooled_within_state": 0.2},
            "per_state_paired_mse": {state: rmse**2 for state in states},
            "paired_squared_error_state_by_candidate": {
                state: {f"candidate-{seed}": rmse**2 for seed in range(16)}
                for state in states
            },
            "secondary_k8_geometry_coverage": {
                "macro_selected_target_coverage": rmse
            },
        }

    def fake_task(data, width, outer_splits, split_seed, projection_seeds):
        states = list(np.unique(data.state_group.astype(str)))
        seed_runs = []
        for seed in projection_seeds:
            baseline = {
                endpoint: metric(1.0, states)
                for endpoint in (
                    "primary_future_path_energy",
                    "secondary_x10_minus_x1_live10x7",
                )
            }
            full = {
                endpoint: metric(0.9, states)
                for endpoint in baseline
            }
            control = {
                endpoint: metric(0.95, states)
                for endpoint in baseline
            }
            seed_runs.append(
                {
                    "projection_seed": seed,
                    "b1_operational": baseline,
                    "b1_plus_matched_routed_rms_control": control,
                    "b1_plus_id_aware_contribution_secondary": full,
                    "b1_plus_s_loo_primary_commonality": full,
                    "b1_plus_s_loo_gram_secondary": full,
                    "deltas_vs_b1": {
                        arm: probe._endpoint_deltas(full, baseline)
                        for arm in (
                            "id_aware_contribution_secondary",
                            "s_loo_primary_commonality",
                            "s_loo_gram_secondary",
                        )
                    },
                }
            )
            seed_runs[-1]["deltas_vs_b1"][
                "matched_routed_rms_control"
            ] = probe._endpoint_deltas(control, baseline)
        return {
            "admission": {},
            "checkpoint": data.checkpoint_group[0],
            "projection_seed_runs": seed_runs,
            "reference_projection_diagnostics": {},
            "standalone_mechanism": {
                "features": {
                    feature: {"equal_state_macro_spearman": 0.1}
                    for feature in (
                        "s_loo_chunk",
                        "runtime_merged_routed_rms_chunk",
                        "d0_velocity_live7_rms",
                    )
                }
            },
        }

    monkeypatch.setattr(probe, "_analyze_task", fake_task)
    summary = probe.analyze(
        grid, 128, 3, 9, probe.DEFAULT_PROJECTION_SEEDS
    )
    assert summary["preregistered_decision"]["all_projection_seeds_pass"]
    assert summary["preregistered_decision"]["deployable_pruning_k8_gate"] is None
    assert summary["preregistered_decision"][
        "worst_projection_seed_macro_relative_improvement"
    ] == pytest.approx(0.1)
    for row in summary["equal_task_macro_by_projection_seed"]:
        decision = row["preregistered_decision"]
        assert decision["tasks_improved"] == 5
        assert decision["equal_task_macro_relative_improvement"] == pytest.approx(0.1)


def test_bootstrap_reuses_one_candidate_column_draw_across_tasks() -> None:
    states, candidates = probe.bootstrap_axis_indices(5, 8, 16, 7, 31)
    assert states.shape == (7, 5, 8)
    assert candidates.shape == (7, 16)
    repeated_states, repeated_candidates = probe.bootstrap_axis_indices(5, 8, 16, 7, 31)
    np.testing.assert_array_equal(states, repeated_states)
    np.testing.assert_array_equal(candidates, repeated_candidates)
