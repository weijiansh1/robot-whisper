from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results/unsupervised_cluster"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_unlabeled_cluster_seal_is_complete_and_label_blind() -> None:
    manifest = json.loads((RESULTS / "unlabeled_manifest.json").read_text())
    assert manifest["training"] is False
    assert manifest["labels_read"] == []
    assert manifest["outcomes_read"] == []
    assert manifest["protocol_sha256"] == sha256(
        HERE / "UNSUPERVISED_CLUSTER_PROTOCOL.md"
    )
    assert manifest["code_sha256"] == sha256(HERE / "cluster_moe_peaks.py")
    assert manifest["source_sha256"] == sha256(Path(manifest["source"]))
    for name, expected in manifest["artifact_sha256"].items():
        assert sha256(RESULTS / name) == expected

    source = (HERE / "cluster_moe_peaks.py").read_text()
    for forbidden in (
        "assignment_audit_labeled",
        "B_meta.csv",
        "loop_label",
        "static_label",
        "loop_onset",
        "static_onset",
    ):
        assert forbidden not in source


def test_primary_adaptive_result_and_controls() -> None:
    manifest = json.loads((RESULTS / "unlabeled_manifest.json").read_text())
    primary = manifest["primary"]
    assert primary["algorithm"] == "HDBSCAN"
    assert primary["clusters"] == 2
    assert primary["sizes"] == [43, 33]
    assert primary["noise"] == 436
    assert np.isclose(primary["coverage"], 76 / 512)
    assert manifest["gmm_bic_selected_components"] == 1
    assert manifest["positive_control"]["clusters"] == 3
    assert manifest["positive_control"]["ari_truth"] == 1.0

    stability = pd.read_csv(RESULTS / "stability_sentinels.csv")
    permuted = stability[
        stability["test"] == "within_pool_time_identity_permutation"
    ]
    assert len(permuted) == 30
    assert (permuted["clusters"] == 0).all()
    assert (permuted["coverage"] == 0).all()


def test_posthoc_labels_do_not_form_trap_phenotype_clusters() -> None:
    summary = json.loads((RESULTS / "evaluation_summary.json").read_text())
    assert summary["labels_opened_after_seal"] is True
    assert summary["label_inventory"] == {
        "both": 10,
        "loop_only": 47,
        "normal": 319,
        "static_only": 136,
    }
    tests = {row["target"]: row for row in summary["association_tests"]}
    primary = tests["phenotype_4way"]
    assert primary["adjusted_mutual_info"] < 0.02
    assert primary["conditional_permutation_p"] > 0.05
    exclusive = tests["phenotype_3way_excluding_both"]
    assert exclusive["adjusted_mutual_info"] < 0.02
    assert exclusive["conditional_permutation_p"] > 0.05

    composition = pd.read_csv(RESULTS / "cluster_composition.csv").set_index("cluster")
    assert composition.loc[0, "normal_n"] == 35
    assert composition.loc[0, "n"] == 43
    assert composition.loc[1, "normal_n"] == 20
    assert composition.loc[1, "n"] == 33


def test_cluster_shapes_match_reported_tail_signatures() -> None:
    profiles = pd.read_csv(RESULTS / "cluster_peak_profiles.csv")
    at_anchor = profiles[profiles["relative_query"] == 0].set_index(
        ["cluster", "component"]
    )["mean_rank"]
    assert at_anchor.loc[(0, "late_flow_volatility_w4_high")] > 0.9
    assert at_anchor.loc[(0, "route_acceleration_w4_high")] > 0.9
    assert at_anchor.loc[(1, "late_flow_volatility_w4_high")] < 0.15
    assert at_anchor.loc[(1, "route_mobility_w4_low")] > 0.75
