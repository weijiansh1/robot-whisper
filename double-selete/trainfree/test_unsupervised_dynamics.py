from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results/unsupervised_dynamics"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dynamics_unlabeled_seal_and_input_boundary() -> None:
    manifest = json.loads((RESULTS / "unlabeled_manifest.json").read_text())
    assert manifest["training"] is False
    assert manifest["labels_read"] == []
    assert manifest["success_read"] is False
    assert manifest["episode_length_as_feature"] is False
    assert manifest["protocol_sha256"] == sha256(
        HERE / "UNSUPERVISED_DYNAMICS_PROTOCOL.md"
    )
    assert manifest["code_sha256"] == sha256(HERE / "cluster_moe_dynamics.py")
    assert manifest["source_sha256"] == sha256(Path(manifest["source"]))
    for name, expected in manifest["artifact_sha256"].items():
        assert sha256(RESULTS / name) == expected

    source = (HERE / "cluster_moe_dynamics.py").read_text()
    for forbidden in (
        "assignment_audit_labeled",
        "B_meta.csv",
        "loop_label",
        "static_label",
        "loop_onset",
        "static_onset",
        'archive["success"]',
        'archive["length"]',
    ):
        assert forbidden not in source


def test_state_vocabulary_and_adaptive_trajectory_models() -> None:
    manifest = json.loads((RESULTS / "unlabeled_manifest.json").read_text())
    state = manifest["state_vocabulary"]
    assert state["selected_components"] == 3
    assert state["training_windows"] == 512 * 27
    assert state["hdbscan_diagnostic"]["clusters"] == 0
    assert manifest["early_primary"]["clusters"] == 0
    assert manifest["full_retrospective_primary"]["clusters"] == 0
    assert manifest["gmm_bic_selected_components"] == {"early": 5, "full": 7}
    assert manifest["positive_control"]["clusters"] == 3
    assert manifest["positive_control"]["ari_truth"] == 1.0

    assignment = pd.read_csv(RESULTS / "unlabeled_trajectory_assignments.csv")
    assert len(assignment) == 512
    assert assignment["early_hdbscan_cluster"].eq(-1).all()
    assert assignment["full_hdbscan_cluster"].eq(-1).all()
    assert assignment["early_gmm_bic_cluster"].nunique() == 5
    assert assignment["full_gmm_bic_cluster"].nunique() == 7


def test_gmm_partition_stability_is_limited() -> None:
    stability = pd.read_csv(RESULTS / "stability_sentinels.csv")
    gmm = stability[stability["algorithm"] == "GMM-BIC"]
    early_subset = gmm[
        (gmm["scope"] == "early") & (gmm["test"] == "feature_subsample_80pct")
    ]
    full_subset = gmm[
        (gmm["scope"] == "full") & (gmm["test"] == "feature_subsample_80pct")
    ]
    early_jitter = gmm[
        (gmm["scope"] == "early") & (gmm["test"] == "jitter_1pct")
    ]
    assert len(early_subset) == len(full_subset) == len(early_jitter) == 30
    assert np.isclose(early_subset["ari_vs_primary"].median(), 0.5778, atol=5e-4)
    assert np.isclose(full_subset["ari_vs_primary"].median(), 0.3739, atol=5e-4)
    assert early_jitter["ari_vs_primary"].median() < 0.25


def test_early_is_null_but_full_has_weak_retrospective_structure() -> None:
    summary = json.loads((RESULTS / "evaluation_summary.json").read_text())
    assert summary["labels_opened_after_seal"] is True
    tests = summary["key_label_tests"]
    early = tests["early_gmm_4way"]
    assert abs(early["adjusted_mutual_info"]) < 0.002
    assert early["conditional_permutation_p"] > 0.4

    full = tests["full_gmm_4way"]
    assert 0.07 < full["adjusted_mutual_info"] < 0.09
    assert full["conditional_permutation_p"] < 1e-3
    failure = tests["full_gmm_4way_failures_only"]
    assert failure["sample_n"] == 216
    assert 0.08 < failure["adjusted_mutual_info"] < 0.09
    loop_static = tests["full_gmm_loop_vs_static"]
    assert loop_static["sample_n"] == 183
    assert 0.08 < loop_static["adjusted_mutual_info"] < 0.09

    composition = pd.read_csv(RESULTS / "cluster_composition.csv")
    cluster_six = composition[
        (composition["assignment"] == "full_gmm_bic_partition")
        & (composition["cluster"] == 6)
    ].iloc[0]
    assert cluster_six["n"] == 52
    assert cluster_six["static_only_n"] == 40
    assert cluster_six["normal_n"] == 7

    onset = pd.read_csv(RESULTS / "onset_state_profiles.csv").set_index(
        ["event", "state"]
    )
    loop_s0 = onset.loc[("loop", 0)]
    static_s2 = onset.loc[("static", 2)]
    assert loop_s0["paired_episode_n"] == 37
    assert 0.17 < loop_s0["delta_mean"] < 0.20
    assert loop_s0["paired_sign_permutation_p"] < 0.01
    assert static_s2["paired_episode_n"] == 141
    assert 0.27 < static_s2["delta_mean"] < 0.30
    assert static_s2["paired_sign_permutation_p"] < 1e-3
