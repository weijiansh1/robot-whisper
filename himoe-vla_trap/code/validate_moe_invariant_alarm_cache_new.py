#!/usr/bin/env python3
"""Validate the cache_new MoE invariant alarm experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "results/moe_invariant_alarm_cache_new"
CONFIG = PACKAGE_ROOT / "configs/moe_invariant_alarm_cache_new.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(alarm: pd.Series, failure: pd.Series) -> dict[str, float | int]:
    alarm = alarm.to_numpy(dtype=bool)
    failure = failure.to_numpy(dtype=bool)
    tp = int(np.sum(alarm & failure))
    fp = int(np.sum(alarm & ~failure))
    fn = int(np.sum(~alarm & failure))
    tn = int(np.sum(~alarm & ~failure))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "failure_recall": tp / max(tp + fn, 1),
        "success_false_alarm_rate": fp / max(fp + tn, 1),
    }


def assert_metrics(observed: dict, expected: dict) -> None:
    for key, value in expected.items():
        if isinstance(value, int):
            assert int(observed[key]) == value, key
        else:
            assert np.isclose(float(observed[key]), float(value)), key


def validate() -> dict[str, object]:
    config_text = CONFIG.read_text(encoding="utf-8")
    config = json.loads(config_text)
    assert config["training"] is False
    assert config["gradient_optimization"] is False
    assert config["learned_feature_weights"] is False
    assert config["probability_calibration"] is False
    assert config["threshold_confirmation"]["failure_labels_used"] is False
    for legacy_name in ("A_dense", "B_proxy", "A_to_B", "B_to_A"):
        assert legacy_name not in config_text

    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "himoe.moe_invariant_alarm_cache_new.v1"
    assert summary["training"] is False
    assert summary["probability_calibration"] is False
    assert summary["data"]["tasks"] == 40
    assert summary["data"]["episodes"] == 16000
    assert summary["data"]["route_queries"] == 253722
    assert summary["split"]["threshold_confirmation_tasks"] == 32
    assert summary["split"]["heldout_tasks"] == 8
    assert summary["threshold"]["confirmation_success_episodes"] == 12382
    assert summary["threshold"]["confirmation_success_episode_fpr"] <= 0.01
    assert summary["decision"]["strict_test_passed"] is False
    assert summary["decision"]["deployment_ready"] is False

    manifest = json.loads(
        (ROOT / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    prediction_path = PACKAGE_ROOT / manifest["prediction_path"]
    assert manifest["heldout_endpoint_labels_loaded"] is False
    assert manifest["predictions_written_before_heldout_outcome_load"] is True
    assert sha256(prediction_path) == manifest["prediction_sha256"]
    assert manifest["prediction_rows"] == 34935

    predictions = pd.read_csv(prediction_path)
    assert len(predictions) == 34935
    assert predictions["query"].min() == 4
    forbidden = {"success", "failure", "outcome", "episode_length", "reward"}
    assert forbidden.isdisjoint(predictions.columns)
    assert predictions.groupby(["task", "episode"])["first_alarm"].sum().max() <= 1
    threshold = float(summary["threshold"]["detector_confidence"])
    expected_alarm = predictions["detector_confidence"] >= threshold
    assert np.array_equal(expected_alarm, predictions["alarm"].to_numpy(dtype=bool))
    joint = predictions["joint_confidence"] >= threshold
    recurrence = predictions["recurrence_persistent_confidence"] >= threshold
    assert np.array_equal(expected_alarm, (joint | recurrence).to_numpy(dtype=bool))

    heldout = pd.read_csv(ROOT / "tables/heldout_episode_flip.csv")
    assert len(heldout) == 3200
    assert heldout["task"].nunique() == 8
    assert int(heldout["failure"].sum()) == 114
    assert_metrics(summary["heldout_endpoint_flip"]["detector"], metrics(heldout["alarm"], heldout["failure"]))
    assert_metrics(summary["heldout_endpoint_flip"]["fixed_clock"], metrics(heldout["clock_alarm"], heldout["failure"]))
    assert_metrics(
        summary["heldout_endpoint_flip"]["heldout_fpr_matched_clock"],
        metrics(heldout["matched_clock_alarm"], heldout["failure"]),
    )
    assert summary["heldout_alarm_structure"] == {
        "alarm_tasks": 5,
        "false_alarm_tasks": 1,
        "joint_convergence_response_alarm_episodes": 0,
        "persistent_recurrence_alarm_episodes": 83,
    }

    full = pd.read_csv(ROOT / "tables/all_episode_flip_descriptive.csv")
    assert len(full) == 16000
    assert full["task"].nunique() == 40
    assert int(full["failure"].sum()) == 532
    assert_metrics(
        summary["all_episodes_descriptive"]["detector"],
        metrics(full["alarm"], full["failure"]),
    )
    assert summary["all_episodes_descriptive"]["detector"]["tp"] == 152
    assert summary["all_episodes_descriptive"]["fixed_clock"]["tp"] == 267

    onset = pd.read_csv(ROOT / "tables/heldout_onset_proxy_audit.csv")
    assert len(onset) == 71
    assert summary["secondary_onset_proxy"]["detector_within_minus2_to_onset"] == 5
    assert summary["secondary_onset_proxy"]["detector_late"] == 27
    assert (ROOT / "figures/heldout_invariant_alarm.png").stat().st_size > 50000

    return {
        "moe_invariant_alarm_tasks": 40,
        "moe_invariant_alarm_episodes": 16000,
        "moe_invariant_alarm_failures": 532,
        "moe_invariant_alarm_heldout_episodes": 3200,
        "moe_invariant_alarm_heldout_failures": 114,
        "moe_invariant_alarm_heldout_tp": 51,
        "moe_invariant_alarm_heldout_fp": 32,
        "moe_invariant_alarm_full_tp": 152,
        "moe_invariant_alarm_prediction_sha256": manifest["prediction_sha256"],
        "moe_invariant_alarm_strict_test_passed": False,
    }


def main() -> int:
    result = validate()
    print(json.dumps({"status": "VALIDATION_OK", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
