#!/usr/bin/env python3
"""Open labels only after MoE-only clusters are sealed, then audit separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    v_measure_score,
)


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results/unsupervised_cluster"
DEFAULT_LABELS = HERE / "results/assignment_audit_labeled.csv"
ROOT = HERE.parents[1]
DEFAULT_META = ROOT / "analysis_ssm/cache/B_meta.csv"
PROTOCOL = HERE / "UNSUPERVISED_CLUSTER_PROTOCOL.md"
CLUSTER_CODE = HERE / "cluster_moe_peaks.py"
SEED = 20260903
COMPONENTS = (
    "gate_concentration_delta2_high",
    "gate_concentration_w4_low",
    "late_flow_volatility_w4_high",
    "route_acceleration_w4_high",
    "route_mobility_w4_low",
    "deep_soft_consensus_delta2_high",
    "deep_soft_mixture_delta2_high",
    "l15_soft_mixture_now_high",
    "l15_soft_consensus_now_high",
    "l15_soft_final_jump_high",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def verify_seal(results: Path) -> dict[str, Any]:
    manifest_path = results / "unlabeled_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["labels_read"] or manifest["outcomes_read"]:
        raise ValueError("unlabeled clustering manifest reports label/outcome access")
    if manifest["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("protocol changed after cluster seal")
    if manifest["code_sha256"] != sha256(CLUSTER_CODE):
        raise ValueError("clustering code changed after cluster seal")
    source = Path(manifest["source"])
    if manifest["source_sha256"] != sha256(source):
        raise ValueError("unlabeled source changed after cluster seal")
    for name, expected in manifest["artifact_sha256"].items():
        observed = sha256(results / name)
        if observed != expected:
            raise ValueError(f"sealed artifact changed: {name}")
    return manifest


def load_episode_labels(path: Path) -> pd.DataFrame:
    columns = [
        "episode_id",
        "group",
        "loop_label",
        "static_label",
        "loop_onset",
        "static_onset",
    ]
    repeated = pd.read_csv(path, usecols=columns)
    for column in columns[1:]:
        variation = repeated.groupby("episode_id")[column].nunique(dropna=False)
        if (variation != 1).any():
            raise ValueError(f"inconsistent repeated label column: {column}")
    frame = repeated.drop_duplicates("episode_id").sort_values("episode_id")
    if not np.array_equal(frame["episode_id"].to_numpy(), np.arange(512)):
        raise ValueError("expected labels for episode IDs 0..511")
    frame["loop_label"] = frame["loop_label"].astype(bool)
    frame["static_label"] = frame["static_label"].astype(bool)
    frame["phenotype"] = np.select(
        [
            frame["loop_label"] & frame["static_label"],
            frame["loop_label"],
            frame["static_label"],
        ],
        ["both", "loop_only", "static_only"],
        default="normal",
    )
    frame["trap_label"] = frame["loop_label"] | frame["static_label"]
    return frame


def encode(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    names, codes = np.unique(np.asarray(values).astype(str), return_inverse=True)
    return codes.astype(np.int64), names.tolist()


def discrete_mi(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    n_left = int(left.max()) + 1
    n_right = int(right.max()) + 1
    count = np.bincount(
        left * n_right + right, minlength=n_left * n_right
    ).reshape(n_left, n_right)
    joint = count / max(count.sum(), 1)
    p_left = joint.sum(axis=1, keepdims=True)
    p_right = joint.sum(axis=0, keepdims=True)
    expected = p_left @ p_right
    good = joint > 0
    return float(np.sum(joint[good] * np.log(joint[good] / expected[good])))


def conditional_permutation_test(
    cluster: np.ndarray,
    target: np.ndarray,
    group: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    cluster_code, _ = encode(cluster)
    target_code, _ = encode(target)
    observed = discrete_mi(cluster_code, target_code)
    indices = [np.flatnonzero(group == value) for value in np.unique(group)]
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        shuffled = target_code.copy()
        for index in indices:
            shuffled[index] = rng.permutation(target_code[index])
        null[draw] = discrete_mi(cluster_code, shuffled)
    p = (1.0 + float(np.sum(null >= observed - 1e-15))) / (draws + 1.0)
    return observed, p, null


def association_metrics(cluster: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "adjusted_mutual_info": float(adjusted_mutual_info_score(target, cluster)),
        "normalized_mutual_info": float(normalized_mutual_info_score(target, cluster)),
        "adjusted_rand": float(adjusted_rand_score(target, cluster)),
        "homogeneity": float(homogeneity_score(target, cluster)),
        "completeness": float(completeness_score(target, cluster)),
        "v_measure": float(v_measure_score(target, cluster)),
    }


def cluster_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overall = frame["phenotype"].value_counts(normalize=True)
    for cluster, part in frame.groupby("hdbscan_cluster", sort=True):
        counts = part["phenotype"].value_counts()
        row: dict[str, Any] = {
            "cluster": int(cluster),
            "cluster_name": "noise" if cluster < 0 else f"cluster_{int(cluster)}",
            "n": len(part),
            "success_n": int(part["success"].sum()),
            "failure_n": int((~part["success"]).sum()),
            "success_rate": float(part["success"].mean()),
            "anchor_query_median": float(part["moe_anchor_query"].median()),
            "membership_probability_median": float(part["hdbscan_probability"].median()),
        }
        for label in ("normal", "loop_only", "static_only", "both"):
            count = int(counts.get(label, 0))
            rate = count / len(part)
            row[f"{label}_n"] = count
            row[f"{label}_rate"] = rate
            row[f"{label}_enrichment"] = rate / max(float(overall.get(label, 0.0)), 1e-12)
        rows.append(row)
    return pd.DataFrame(rows)


def onset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event, label_column, onset_column in (
        ("loop", "loop_label", "loop_onset"),
        ("static", "static_label", "static_onset"),
    ):
        part = frame[frame[label_column] & (frame[onset_column] >= 0)].copy()
        delta = part["moe_anchor_query"] - part[onset_column]
        rows.append(
            {
                "event": event,
                "n": len(part),
                "onset_by_q34_n": int((part[onset_column] <= 34).sum()),
                "anchor_minus_onset_median": float(delta.median()),
                "anchor_minus_onset_q25": float(delta.quantile(0.25)),
                "anchor_minus_onset_q75": float(delta.quantile(0.75)),
                "anchor_before_onset_rate": float((delta < 0).mean()),
                "anchor_within_2q_rate": float((delta.abs() <= 2).mean()),
            }
        )
    return pd.DataFrame(rows)


def peak_profiles(
    frame: pd.DataFrame, curves: np.ndarray, query_start: int = 4, radius: int = 3
) -> pd.DataFrame:
    rows = []
    anchor = frame["moe_anchor_query"].to_numpy(dtype=np.int64) - query_start
    for cluster in sorted(frame["hdbscan_cluster"].unique()):
        indices = np.flatnonzero(frame["hdbscan_cluster"].to_numpy() == cluster)
        snippets = np.stack(
            [curves[index, anchor[index] - radius : anchor[index] + radius + 1] for index in indices]
        )
        for offset_index, offset in enumerate(range(-radius, radius + 1)):
            for component_index, component in enumerate(COMPONENTS):
                values = snippets[:, offset_index, component_index]
                rows.append(
                    {
                        "cluster": int(cluster),
                        "relative_query": offset,
                        "component": component,
                        "mean_rank": float(values.mean()),
                        "median_rank": float(np.median(values)),
                        "std_rank": float(values.std()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.self_test:
        left = np.asarray([0, 0, 1, 1])
        assert discrete_mi(left, left) > 0.69
        assert abs(discrete_mi(left, np.asarray([0, 1, 0, 1]))) < 1e-12
        print("self-test passed")
        return
    if args.permutations < 1:
        raise ValueError("permutations must be positive")
    manifest = verify_seal(args.results)
    assignments = pd.read_csv(args.results / "unlabeled_cluster_assignments.csv")
    labels = load_episode_labels(args.labels)
    meta = pd.read_csv(args.meta, usecols=["episode_id", "group", "success", "T"])
    frame = assignments.merge(labels, on=["episode_id", "group"], validate="one_to_one")
    frame = frame.merge(
        meta.rename(columns={"T": "episode_length"}),
        on=["episode_id", "group"],
        validate="one_to_one",
    ).sort_values("episode_id")
    if len(frame) != 512:
        raise ValueError("evaluation merge lost episodes")

    cluster = frame["hdbscan_cluster"].to_numpy(dtype=np.int64)
    group = frame["group"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    test_rows = []
    targets = {
        "phenotype_4way": frame["phenotype"].to_numpy(),
        "trap_vs_normal": frame["trap_label"].astype(str).to_numpy(),
        "success_vs_failure": frame["success"].astype(str).to_numpy(),
    }
    for name, target in targets.items():
        observed_mi, p_value, null = conditional_permutation_test(
            cluster, target, group, args.permutations, rng
        )
        test_rows.append(
            {
                "target": name,
                **association_metrics(cluster, target),
                "mutual_info": observed_mi,
                "conditional_permutation_p": p_value,
                "null_mi_q95": float(np.quantile(null, 0.95)),
                "permutations": args.permutations,
            }
        )

    exclusive = frame[frame["phenotype"] != "both"]
    exclusive_cluster = exclusive["hdbscan_cluster"].to_numpy(dtype=np.int64)
    exclusive_target = exclusive["phenotype"].to_numpy()
    exclusive_group = exclusive["group"].to_numpy(dtype=np.int64)
    observed_mi, p_value, null = conditional_permutation_test(
        exclusive_cluster, exclusive_target, exclusive_group, args.permutations, rng
    )
    test_rows.append(
        {
            "target": "phenotype_3way_excluding_both",
            **association_metrics(exclusive_cluster, exclusive_target),
            "mutual_info": observed_mi,
            "conditional_permutation_p": p_value,
            "null_mi_q95": float(np.quantile(null, 0.95)),
            "permutations": args.permutations,
        }
    )

    clustered = frame[frame["hdbscan_cluster"] >= 0]
    clustered_cluster = clustered["hdbscan_cluster"].to_numpy(dtype=np.int64)
    clustered_group = clustered["group"].to_numpy(dtype=np.int64)
    for name, target in (
        ("phenotype_4way_clustered_only", clustered["phenotype"].to_numpy()),
        (
            "trap_vs_normal_clustered_only",
            clustered["trap_label"].astype(str).to_numpy(),
        ),
    ):
        observed_mi, p_value, null = conditional_permutation_test(
            clustered_cluster, target, clustered_group, args.permutations, rng
        )
        test_rows.append(
            {
                "target": name,
                **association_metrics(clustered_cluster, target),
                "mutual_info": observed_mi,
                "conditional_permutation_p": p_value,
                "null_mi_q95": float(np.quantile(null, 0.95)),
                "permutations": args.permutations,
            }
        )
    tests = pd.DataFrame(test_rows)

    nuisance_rows = []
    for name, target in (
        ("init_pool", frame["group"].astype(str).to_numpy()),
        ("episode_length", frame["episode_length"].astype(str).to_numpy()),
    ):
        nuisance_rows.append({"target": name, **association_metrics(cluster, target)})
    nuisances = pd.DataFrame(nuisance_rows)

    profiles = cluster_profiles(frame)
    onset = onset_summary(frame)
    with np.load(args.results / "unlabeled_embedding.npz", allow_pickle=False) as archive:
        curves = np.asarray(archive["curves"], dtype=np.float64)
        sensitivity = {
            name: np.asarray(archive[name], dtype=np.int64)
            for name in archive.files
            if name.startswith("hdb_") and name != "hdbscan_labels"
        }
    peak = peak_profiles(frame, curves)
    sensitivity_rows = []
    for name, values in sensitivity.items():
        clusters = len(set(values.tolist()) - {-1})
        sensitivity_rows.append(
            {
                "assignment": name,
                "clusters": clusters,
                "coverage": float((values >= 0).mean()),
                "ami_phenotype_4way": float(
                    adjusted_mutual_info_score(frame["phenotype"], values)
                ),
                "ami_success": float(adjusted_mutual_info_score(frame["success"], values)),
            }
        )
    sensitivity_frame = pd.DataFrame(sensitivity_rows)

    frame.to_csv(args.results / "labeled_cluster_assignments.csv", index=False)
    profiles.to_csv(args.results / "cluster_composition.csv", index=False)
    tests.to_csv(args.results / "label_association_tests.csv", index=False)
    nuisances.to_csv(args.results / "nuisance_associations.csv", index=False)
    onset.to_csv(args.results / "anchor_onset_alignment.csv", index=False)
    peak.to_csv(args.results / "cluster_peak_profiles.csv", index=False)
    sensitivity_frame.to_csv(args.results / "sensitivity_label_association.csv", index=False)

    summary = {
        "schema": "himoe.moe_peak_unsupervised_cluster_evaluation.v1",
        "unlabeled_manifest_sha256": sha256(args.results / "unlabeled_manifest.json"),
        "sealed_assignment_sha256": manifest["artifact_sha256"][
            "unlabeled_cluster_assignments.csv"
        ],
        "labels_opened_after_seal": True,
        "label_inventory": frame["phenotype"].value_counts().to_dict(),
        "outcome_inventory": {
            "success": int(frame["success"].sum()),
            "failure": int((~frame["success"]).sum()),
        },
        "primary_cluster": manifest["primary"],
        "cluster_composition": profiles.to_dict(orient="records"),
        "association_tests": tests.to_dict(orient="records"),
        "nuisance_associations": nuisances.to_dict(orient="records"),
        "anchor_onset_alignment": onset.to_dict(orient="records"),
        "interpretation_guardrails": [
            "HDBSCAN noise is retained and is not renamed normal.",
            "Cluster count and parameters were sealed before per-episode labels were read.",
            "Label association is descriptive on a previously studied corpus, not discovery-independent replication.",
            "The MoE-selected anchor is not an online alarm and must not be reported as lead time.",
            "Clustered-only tests are secondary because HDBSCAN selected that subset without labels.",
        ],
    }
    (args.results / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plain(summary), indent=2))


if __name__ == "__main__":
    main()
