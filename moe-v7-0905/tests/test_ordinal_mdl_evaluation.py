from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))
sys.path.insert(0, str(BUNDLE / "method"))

import evaluate_ordinal_mdl_guard as evaluator  # noqa: E402


def test_seal_has_no_outcome_access_and_detects_changed_outputs(tmp_path: Path, monkeypatch) -> None:
    metadata = {
        "task_names": np.array(["suite/task"]), "task_index": np.array([0]),
        "episode": np.array([0]), "init_state_id": np.array([0]),
        "length": np.array([8]), "valid": np.ones((1, 8), bool),
    }
    layer, route = tmp_path / "layer.npz", tmp_path / "route.npz"
    np.savez(layer, **metadata, mobility=np.ones((1, 8, 8)))
    np.savez(route, **metadata, feature_names=np.array(["route_acceleration", "lag_periodicity"]),
             features=np.ones((1, 8, 2)))
    monkeypatch.setattr(evaluator, "COHORTS", (("sample", "sample", layer, route, "forbidden.csv"),))

    def forbidden(*args, **kwargs):
        raise AssertionError("outcomes must not be opened while sealing")

    monkeypatch.setattr(evaluator.pd, "read_csv", forbidden)
    monkeypatch.setattr(evaluator, "aligned_labels", forbidden)
    output = tmp_path / "sealed"
    evaluator.seal(output)
    manifest = evaluator.verify_seal(output)
    assert manifest["candidate_count"] == 1
    assert manifest["outcome_files_read_before_seal"] is False
    assert manifest["threshold_calibration"] is False
    with pytest.raises(FileExistsError):
        evaluator.seal(output)
    np.savez(output / "sample_scores.npz", changed=np.array([1]))
    with pytest.raises(ValueError, match="changed"):
        evaluator.verify_seal(output)


def test_frozen_experiment_manifest_is_explicit_about_boundaries() -> None:
    path = evaluator.DEFAULT_OUTPUT / "sealed_manifest.json"
    if not path.exists():
        pytest.skip("full replay has not been generated")
    manifest = json.loads(path.read_text())
    assert manifest["new_outcome_grid_search"] is False
    assert manifest["reference_corpus_used_for_new_method"] is False
    assert manifest["offline_model_fit"] is False
    assert manifest["online_adaptive_symbol_counts"] is True
    assert manifest["fixed_code_convention_and_implicit_boundary"] is True
    assert manifest["external_pristine_holdout"] is False
