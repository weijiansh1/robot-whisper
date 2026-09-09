from __future__ import annotations

import json

import numpy as np

import analyze_offline_vector_screen as screen
from analyze_early_action_head import pairwise_rms


def test_equal_block_features_are_exact_mean_linear_kernel():
    rng = np.random.default_rng(3)
    blocks = [rng.normal(size=(3, 8, width)) for width in (2, 5, 11)]
    combined = screen.equal_block_kernel_features(blocks)
    observed = combined[1] @ combined[1].T / combined.shape[-1]
    expected = np.mean(
        [block[1] @ block[1].T / block.shape[-1] for block in blocks], axis=0
    )
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)


def test_fold_transform_is_fitted_only_on_training_values():
    rng = np.random.default_rng(5)
    train = rng.normal(size=(3, 6, 4))
    test = rng.normal(size=(2, 4))
    transformed_train, transformed_test, diagnostics = (
        screen.fit_train_fold_transform(train, test)
    )
    changed_train, changed_test, changed_diagnostics = screen.fit_train_fold_transform(
        train, test + 10_000.0
    )
    np.testing.assert_allclose(transformed_train, changed_train, atol=0.0, rtol=0.0)
    assert diagnostics == changed_diagnostics
    np.testing.assert_allclose(
        transformed_test - changed_test,
        np.full_like(transformed_test, -10_000.0 / diagnostics["train_scalar_rms_before_scaling"]),
    )
    np.testing.assert_allclose(
        diagnostics["train_kernel_trace_after_scaling"], 1.0, atol=1e-12
    )


def test_residualized_nested_kernel_preserves_full_base_kernel():
    rng = np.random.default_rng(6)
    train_base = rng.normal(size=(3, 5, 7))
    test_base = rng.normal(size=(4, 7))
    train_routed = rng.normal(size=(3, 5, 6))
    test_routed = rng.normal(size=(4, 6))
    nested_train, nested_test, diagnostics = screen.residualized_nested_features(
        train_base, test_base, train_routed, test_routed, ridge=1.0
    )
    flat_base = train_base.reshape(-1, train_base.shape[-1])
    flat_nested = nested_train.reshape(-1, nested_train.shape[-1])
    base_kernel = flat_base @ flat_base.T / flat_base.shape[-1]
    nested_kernel = flat_nested @ flat_nested.T / flat_nested.shape[-1]
    residual_kernel = nested_kernel - base_kernel
    assert np.linalg.eigvalsh((residual_kernel + residual_kernel.T) * 0.5).min() > -1e-10
    np.testing.assert_allclose(
        diagnostics["nested_kernel_trace"],
        diagnostics["base_kernel_trace"]
        + diagnostics["residual_routed_kernel_trace"],
        atol=1e-12,
    )
    assert nested_test.shape == (4, 13)


def test_evaluate_pool_uses_four_independent_k2_selections_and_exact_baseline():
    rng = np.random.default_rng(7)
    actions = rng.normal(size=(32, 10, 7))
    result = screen.evaluate_pool(actions.copy(), actions, screen.seed_folds())
    assert len(result["within_held8_spearman"]) == 4
    np.testing.assert_allclose(result["within_held8_spearman"], 1.0)
    selected = np.asarray(result["selected_seed_positions"])
    assert selected.shape == (8,)
    assert len(np.unique(selected)) == 8
    for fold in screen.seed_folds():
        assert np.isin(selected, fold).sum() == 2
    assert result["coverage_over_exact_stratified_random"] < 1.0


def _fake_rows(
    nested_gain: list[float],
    coverage_gain: list[float],
    routed_rho: float = 0.9,
):
    rows = []
    for task_axis, task in enumerate([f"task-{index}" for index in range(5)]):
        for state in range(16):
            methods = {}
            for method in screen.METHODS:
                rho = 0.2
                coverage = 1.0
                if method == "routed":
                    rho = routed_rho
                elif method == "base":
                    rho = 0.5
                elif method == "base_plus_routed":
                    rho = 0.5 + nested_gain[task_axis]
                    coverage = 1.0 - coverage_gain[task_axis]
                methods[method] = {
                    "within_held8_spearman": [rho] * 4,
                    "pool_spearman_mean": rho,
                    "coverage": coverage,
                    "exact_stratified_random_coverage": 1.0,
                    "coverage_over_exact_stratified_random": coverage,
                    "relative_coverage_improvement": 1.0 - coverage,
                    "selected_seed_positions": list(range(8)),
                }
            rows.append({"task": task, "state_id": state, "methods": methods})
    return rows


def test_decision_uses_nested_effect_thresholds_not_standalone_routed():
    tasks = [f"task-{index}" for index in range(5)]
    positive = screen.summarize(
        _fake_rows([0.03] * 5, [0.02] * 5), tasks
    )["decision"]
    assert positive["all_tasks_meet_min_spearman_gain"]
    assert positive["all_tasks_meet_min_coverage_gain"]
    assert positive["screen_positive"]

    one_failed_task = screen.summarize(
        _fake_rows([0.03, 0.03, 0.03, 0.03, 0.019], [0.02] * 5),
        tasks,
    )["decision"]
    assert not one_failed_task["all_tasks_meet_min_spearman_gain"]
    assert not one_failed_task["screen_positive"]

    standalone_cannot_rescue = screen.summarize(
        _fake_rows([0.0] * 5, [0.0] * 5, routed_rho=1.0), tasks
    )["decision"]
    assert not standalone_cannot_rescue["screen_positive"]


def test_paired_state_bootstrap_is_deterministic_and_contains_constant_effect():
    tasks = [f"task-{index}" for index in range(5)]
    first = screen.summarize(_fake_rows([0.03] * 5, [0.02] * 5), tasks)
    second = screen.summarize(_fake_rows([0.03] * 5, [0.02] * 5), tasks)
    first_nested = first["paired_comparisons"][
        "nested_residualized_routed_minus_base"
    ]
    second_nested = second["paired_comparisons"][
        "nested_residualized_routed_minus_base"
    ]
    assert first_nested == second_nested
    np.testing.assert_allclose(
        first_nested["macro"]["spearman_gain_stratified_state_bootstrap_ci95"],
        0.03,
    )
    np.testing.assert_allclose(
        first_nested["macro"]["coverage_improvement_stratified_state_bootstrap_ci95"],
        0.02,
    )


def test_source_signature_fingerprints_summary_archive_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint-a")
    archive = tmp_path / "candidate_proxy_values.npz"
    archive.write_bytes(b"archive-a")
    run = tmp_path / "run"
    (run / "server" / "hidden.zarr").mkdir(parents=True)
    (run / "server" / "routes.zarr").mkdir(parents=True)
    (run / "client").mkdir()
    (run / "meta.json").write_text("{}")
    (run / "server" / "hidden.zarr" / "chunk").write_bytes(b"hidden-a")
    (run / "server" / "routes.zarr" / "chunk").write_bytes(b"routes-a")
    (run / "client" / "episode_00.npz").write_bytes(b"action-a")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": "declared-a",
                "run": str(run),
            }
        )
    )
    first = screen.source_signature(summary)
    archive.write_bytes(b"archive-b")
    second = screen.source_signature(summary)
    assert first["summary_sha256"] == second["summary_sha256"]
    assert first["proxy_archive_sha256"] != second["proxy_archive_sha256"]
    checkpoint.write_bytes(b"checkpoint-b-is-longer")
    third = screen.source_signature(summary)
    assert second["checkpoint_size"] != third["checkpoint_size"]
    (run / "server" / "hidden.zarr" / "chunk").write_bytes(b"hidden-b-is-longer")
    fourth = screen.source_signature(summary)
    assert (
        third["run_input_stat_manifest"]["sha256"]
        != fourth["run_input_stat_manifest"]["sha256"]
    )


def test_pool_geometry_is_invariant_to_common_prediction_offset():
    rng = np.random.default_rng(11)
    actions = rng.normal(size=(32, 10, 7))
    predicted = rng.normal(size=(32, 10, 7))
    offset = rng.normal(size=(1, 10, 7)) * 100.0
    original = screen.evaluate_pool(predicted, actions, screen.seed_folds())
    shifted = screen.evaluate_pool(predicted + offset, actions, screen.seed_folds())
    upper = np.triu_indices(len(predicted), 1)
    np.testing.assert_allclose(
        pairwise_rms(predicted)[upper],
        pairwise_rms(predicted + offset)[upper],
        atol=3e-6,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        original["within_held8_spearman"], shifted["within_held8_spearman"]
    )
    assert original["selected_seed_positions"] == shifted["selected_seed_positions"]
