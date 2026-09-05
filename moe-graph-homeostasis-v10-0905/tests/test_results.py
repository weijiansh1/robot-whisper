import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_profile_inventory_and_schema():
    summary = read_json(RESULTS / "structural_profiles/build_summary.json")
    assert summary["tasks"] == 45
    assert summary["episodes"] == 18560
    assert summary["rows"] == 305030
    assert summary["within_feature_count"] == 123
    assert summary["query_feature_count"] == 95
    assert summary["total_dynamic_axes"] == 218
    assert summary["outcomes_loaded"] is False
    assert summary["fitted_weights"] is False
    assert summary["task_conditioned_features"] is False
    assert summary["cuda_visible_devices"] == "6,7"
    profiles = sorted((RESULTS / "structural_profiles").glob("*/*/*.npz"))
    if not profiles:
        pytest.skip("optional dense profiles not present")
    assert len(profiles) == 45
    rows = 0
    for path in profiles:
        with np.load(path, allow_pickle=False) as archive:
            count = len(archive["episode_id"])
            assert archive["within_features"].shape == (count, 123)
            assert archive["query_features"].shape == (count, 95)
            assert not np.isinf(archive["within_features"]).any()
            assert not np.isinf(archive["query_features"]).any()
            rows += count
    assert rows == 305030


def test_same_snapshot_layer_differentiation_and_outcome_boundary():
    summary = read_json(RESULTS / "fork_contraction/summary.json")
    assert summary["datasets"]["fork_pilot_n32"]["snapshots"] == 20
    assert summary["datasets"]["rolling_star_k16"]["snapshots"] == 22
    assert summary["datasets"]["rolling_star_k16"]["mixed_success_snapshots"] == 15

    def row(dataset, group, view):
        return next(
            item for item in summary["contraction_summary"]
            if item["dataset"] == dataset
            and item["layer_group"] == group
            and item["view"] == view
        )

    for dataset in ("fork_pilot_n32", "rolling_star_k16"):
        front = row(dataset, "front", "edge_action")
        assert front["fraction_contracting"] == 1.0
        assert front["bootstrap_95_high"] < 1.0
    rolling_back = row("rolling_star_k16", "back", "token_gram")
    assert rolling_back["fraction_contracting"] == 0.0
    assert rolling_back["bootstrap_95_low"] > 1.0
    success = [
        item for item in summary["outcome_summary"]
        if item["dataset"] == "rolling_star_k16" and item["outcome"] == "success"
    ]
    assert len(success) == 6
    assert all(item["mixed_snapshots"] == 15 for item in success)
    assert all(item["auc_wilcoxon_p_vs_half"] > 0.05 for item in success)


def test_phase_results_replicate_with_locked_directions():
    summary = read_json(RESULTS / "phase_portrait/summary.json")

    def composite(section, corpus, score):
        return next(
            item for item in summary[section]
            if item["corpus"] == corpus and item["score"] == score and item["lead"] == -2
        )

    for corpus in ("main16x32", "grid50x8"):
        assert composite("fixed_composites", corpus, "loop_flow_instability")[
            "auc_event_high"
        ] > 0.60
        assert composite("fixed_composites", corpus, "static_rigid_consensus")[
            "auc_event_high"
        ] > 0.86
    assert composite(
        "locked_discovery_composites", "grid50x8", "main_ranked_loop_top4"
    )["auc_event_high"] > 0.65
    assert composite(
        "locked_discovery_composites", "grid50x8", "main_ranked_static_top4"
    )["auc_event_high"] > 0.98

    phase = {
        (row["corpus"], row["event"], row["coordinate"]): row
        for row in summary["phase_summary"]
    }
    for corpus in ("main16x32", "grid50x8"):
        assert phase[(corpus, "loop", "responsiveness")]["bootstrap_95_low"] > 0.0
        assert phase[(corpus, "static", "responsiveness")]["bootstrap_95_high"] < 0.0
        assert phase[(corpus, "static", "coherence")]["bootstrap_95_low"] > 0.0
    assert (RESULTS / "phase_portrait/phase_portrait.png").stat().st_size > 50000
    assert (RESULTS / "fork_contraction/fork_contraction.png").stat().st_size > 50000


def test_sealed_manifest_hashes_every_declared_artifact():
    manifest = read_json(RESULTS / "sealed_manifest.json")
    assert manifest["files"] >= 60
    assert manifest["bytes"] > 130_000_000
    if any(not (ROOT / artifact["path"]).exists() for artifact in manifest["artifacts"]):
        pytest.skip("optional dense profiles not present")
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"]
