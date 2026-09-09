from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))
sys.path.insert(0, str(BUNDLE / "method"))

import evaluate_temporal_budget as evaluator  # noqa: E402
from temporal_budget_guard import SCHEDULES  # noqa: E402
from verify_temporal_budget_raw import sample_temporal_rows  # noqa: E402


def test_temporal_seal_never_opens_outcomes_and_detects_tampering(tmp_path: Path, monkeypatch) -> None:
    if not evaluator.MAIN_LAYER.exists() or not evaluator.STATIC_OUTPUT.exists():
        pytest.skip("full reference and static artifacts are unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("labels cannot be opened during temporal calibration and sealing")

    monkeypatch.setattr(evaluator.pd, "read_csv", forbidden)
    monkeypatch.setattr(evaluator, "aligned_labels", forbidden)
    output = tmp_path / "sealed"
    manifest = evaluator.seal(output, 0.03)
    assert evaluator.verify_seal(output) == manifest
    assert manifest["calibration_uses_outcome_labels"] is False
    assert manifest["calibration_uses_external_features"] is False
    assert manifest["constant_matches_previous_static_profile_and_alarms"] is True
    for name in SCHEDULES:
        assert manifest["calibration"]["variants"][name]["reference_replay_alarms"] <= 480
    assert manifest["calibration"]["time_only"]["reference_replay_alarms"] == 480
    with pytest.raises(FileExistsError):
        evaluator.seal(output, 0.03)
    np.savez(output / "early_strict_profile.npz", changed=np.asarray(1))
    with pytest.raises(ValueError, match="changed"):
        evaluator.verify_seal(output)
    with pytest.raises(ValueError, match="changed"):
        evaluator.evaluate(output, 1)


def test_raw_sample_covers_schedules_and_disagreements_without_labels() -> None:
    first = {"constant": np.array([-1, 8, -1, 12, -1, 15, -1, -1]),
             "early_strict": np.array([-1, 8, 13, 12, -1, -1, -1, -1]),
             "early_loose": np.array([7, 8, -1, 12, -1, 16, -1, -1])}
    rows = sample_temporal_rows(first, 2)
    for name in SCHEDULES:
        assert (first[name][rows] >= 0).any()
        assert (first[name][rows] < 0).any()
    assert 2 in rows and 5 in rows and 0 in rows
    with pytest.raises(ValueError):
        sample_temporal_rows(first, 0)


def test_sealed_experiment_declares_limits() -> None:
    path = evaluator.DEFAULT_OUTPUT / "sealed_manifest.json"
    if not path.exists():
        pytest.skip("temporal experiment has not yet been generated")
    manifest = json.loads(path.read_text())
    assert manifest["external_pristine_holdout"] is False
    assert manifest["related_outcomes_already_seen_by_author"] is True
    assert manifest["new_outcome_grid_search"] is False
    assert manifest["out_of_sample_alarm_budget_guarantee"] is False
    assert manifest["no_outcome_based_schedule_selection"] is True
    assert manifest["persistence_uses_each_observations_own_boundary"] is True
    assert manifest["current_episode_final_length_used_for_boundary"] is False
    assert manifest["alarms_backdated"] is False
