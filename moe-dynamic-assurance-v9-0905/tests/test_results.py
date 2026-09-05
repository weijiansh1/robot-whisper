import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dynamic"))

from scoring import DynamicScoreReference  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_sealed_dynamic_profile_inventory():
    summary = json.loads(
        (ROOT / "results/dynamic_profiles/build_summary.json").read_text(encoding="utf-8")
    )
    assert summary["tasks"] == 45
    assert summary["episodes"] == 18560
    assert summary["rows"] == 305030
    assert summary["total_dynamic_axes"] == 2460
    assert summary["resumed_profiles"] == 0
    assert {worker["device"] for worker in summary["workers"]} == {0, 1}
    assert max(record["max_float16_quantization_error"] for record in summary["records"]) < 1e-3
    profiles = sorted((ROOT / "results/dynamic_profiles").glob("*/*/*.npz"))
    if not profiles:
        pytest.skip("optional dense profiles not present")
    assert len(profiles) == 45
    record = min(summary["records"], key=lambda row: row["profile_bytes"])
    path = Path(record["profile"])
    assert path.stat().st_size == record["profile_bytes"]
    assert sha256(path) == record["profile_sha256"]


def test_runtime_scorer_reproduces_offline_scores():
    profile = (
        ROOT
        / "results/dynamic_profiles/grid50x8/libero_goal/turn_on_the_stove.npz"
    )
    score_path = (
        ROOT
        / "results/dynamic_evaluation/scores/grid50x8/libero_goal/turn_on_the_stove.npz"
    )
    if not profile.exists():
        pytest.skip("optional dense profiles not present")
    with np.load(profile, allow_pickle=False) as archive:
        features = np.column_stack(
            (
                np.asarray(archive["within_features"][:32], dtype=np.float32),
                np.asarray(archive["query_features"][:32], dtype=np.float32),
            )
        )
    reference = DynamicScoreReference.load(
        ROOT / "results/dynamic_evaluation/score_reference.npz"
    )
    runtime = reference.score(features)
    with np.load(score_path, allow_pickle=False) as archive:
        names = archive["score_names"].astype(str).tolist()
        expected = np.asarray(archive["scores"][:32], dtype=np.float32)
    assert set(runtime) == set(names)
    for name, values in runtime.items():
        np.testing.assert_allclose(values, expected[:, names.index(name)], equal_nan=True)


def test_evaluation_and_risk_semantics_are_explicit():
    evaluation = json.loads(
        (ROOT / "results/dynamic_evaluation/summary.json").read_text(encoding="utf-8")
    )
    assert evaluation["dynamic_axes"] == 2460
    assert evaluation["fitted_weights"] is False
    assert evaluation["classifier_trained"] is False
    q2 = {
        (row["event"], row["score"]): row
        for row in evaluation["composite_results"]
        if row["corpus"] == "grid50x8" and row["lead"] == -2
    }
    assert q2[("loop", "discovery_full_loop")]["events"] == 306
    assert q2[("loop", "discovery_history_loop")]["events"] == 177

    risk = json.loads(
        (ROOT / "results/dynamic_risk/summary.json").read_text(encoding="utf-8")
    )
    assert risk["classifier_trained"] is False
    assert risk["semantics"] == "risk-controlled alarm, not an outcome probability"
