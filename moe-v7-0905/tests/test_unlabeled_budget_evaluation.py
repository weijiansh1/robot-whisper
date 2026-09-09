from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))
sys.path.insert(0, str(BUNDLE / "method"))

import evaluate_unlabeled_budget as evaluator  # noqa: E402
from verify_unlabeled_budget_raw import max_error, sample_rows  # noqa: E402


def test_full_seal_cannot_read_labels_and_detects_tampering(tmp_path: Path, monkeypatch) -> None:
    if not evaluator.MAIN_LAYER.exists():
        pytest.skip("full reference corpus is unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("label access during calibration/sealing is forbidden")

    monkeypatch.setattr(evaluator.pd, "read_csv", forbidden)
    monkeypatch.setattr(evaluator, "aligned_labels", forbidden)
    output = tmp_path / "sealed"
    manifest = evaluator.seal(output, 0.03)
    assert evaluator.verify_seal(output) == manifest
    assert manifest["calibration_uses_outcome_labels"] is False
    assert manifest["calibration_uses_published_thresholds"] is False
    assert manifest["calibration_uses_external_features"] is False
    assert manifest["calibration"]["union_reference_alarms"] <= 480
    assert manifest["primary_detector"] == "unlabeled_budget_guard"
    with pytest.raises(FileExistsError):
        evaluator.seal(output, 0.03)
    np.savez(output / "global_profile.npz", tampered=np.array([1]))
    with pytest.raises(ValueError, match="changed"):
        evaluator.verify_seal(output)


def test_published_new_artifact_declares_remaining_constraints() -> None:
    path = evaluator.DEFAULT_OUTPUT / "sealed_manifest.json"
    if not path.exists():
        pytest.skip("full experiment has not been generated")
    manifest = json.loads(path.read_text())
    assert manifest["runtime_monitor_unchanged"] is True
    assert manifest["new_outcome_grid_search"] is False
    assert manifest["inherited_method_design_used_outcome_feedback"] is True
    assert manifest["external_pristine_holdout"] is False
    assert manifest["partial_ablations_retain_published_label_selected_thresholds"] is True
    assert manifest["calibration"]["reference_success_fpr_guarantee"] is False
    assert manifest["calibration"]["out_of_sample_alarm_budget_guarantee"] is False


def test_raw_sampling_covers_both_decision_states_without_labels() -> None:
    first = np.array([-1, 8, -1, 12, -1, 15, -1, -1])
    rows = sample_rows(first, 2)
    np.testing.assert_array_equal(rows, [0, 1, 5, 7])
    assert (first[rows] >= 0).sum() == 2
    np.testing.assert_array_equal(sample_rows(np.array([-1, -1]), 8), [0, 1])
    with pytest.raises(ValueError):
        sample_rows(first, 0)


def test_raw_error_checks_do_not_confuse_nan_and_infinities() -> None:
    assert max_error(np.array([np.nan, 1.0]), np.array([np.nan, 1.0])) == 0
    assert max_error(np.array([np.nan]), np.array([np.inf])) == np.inf
    assert max_error(np.array([-np.inf]), np.array([np.inf])) == np.inf
    assert max_error(np.array([1.0]), np.array([1.0, 2.0])) == np.inf
