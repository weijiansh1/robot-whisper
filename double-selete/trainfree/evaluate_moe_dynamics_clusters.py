#!/usr/bin/env python3
"""Open labels after sealing and audit MoE dynamics-grammar clusters."""

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
    normalized_mutual_info_score,
    v_measure_score,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results/unsupervised_dynamics"
LABELS = HERE / "results/assignment_audit_labeled.csv"
META = HERE.parents[1] / "analysis_ssm/cache/B_meta.csv"
PROTOCOL = HERE / "UNSUPERVISED_DYNAMICS_PROTOCOL.md"
CLUSTER_CODE = HERE / "cluster_moe_dynamics.py"
SEED = 20260904
ASSIGNMENTS = {
    "early_hdbscan_density": "early_hdbscan_cluster",
    "early_gmm_bic_partition": "early_gmm_bic_cluster",
    "early_optics_density": "early_optics_cluster",
    "full_hdbscan_density": "full_hdbscan_cluster",
    "full_gmm_bic_partition": "full_gmm_bic_cluster",
    "full_optics_density": "full_optics_cluster",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--meta", type=Path, default=META)
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
    if manifest["labels_read"] or manifest["success_read"]:
        raise ValueError("unlabeled dynamics stage reports label access")
    checks = (
        (PROTOCOL, manifest["protocol_sha256"], "protocol"),
        (CLUSTER_CODE, manifest["code_sha256"], "clustering code"),
        (Path(manifest["source"]), manifest["source_sha256"], "unlabeled source"),
    )
    for path, expected, description in checks:
        if sha256(path) != expected:
            raise ValueError(f"{description} changed after seal")
    for name, expected in manifest["artifact_sha256"].items():
        if sha256(results / name) != expected:
            raise ValueError(f"sealed artifact changed: {name}")
    return manifest


def load_labels(path: Path) -> pd.DataFrame:
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
        if (repeated.groupby("episode_id")[column].nunique(dropna=False) != 1).any():
            raise ValueError(f"inconsistent episode label: {column}")
    frame = repeated.drop_duplicates("episode_id").sort_values("episode_id")
    if not np.array_equal(frame["episode_id"], np.arange(512)):
        raise ValueError("expected episode IDs 0..511")
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


def encode(values: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(values).astype(str), return_inverse=True)[1].astype(np.int64)


def discrete_mi(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    width = int(right.max()) + 1
    count = np.bincount(
        left * width + right, minlength=(int(left.max()) + 1) * width
    ).reshape(int(left.max()) + 1, width)
    joint = count / count.sum()
    expected = joint.sum(axis=1, keepdims=True) @ joint.sum(axis=0, keepdims=True)
    positive = joint > 0
    return float(np.sum(joint[positive] * np.log(joint[positive] / expected[positive])))


def conditional_permutation(
    assignment: np.ndarray,
    target: np.ndarray,
    group: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    assignment_code = encode(assignment)
    target_code = encode(target)
    observed = discrete_mi(assignment_code, target_code)
    indices = [np.flatnonzero(group == value) for value in np.unique(group)]
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        shuffled = target_code.copy()
        for index in indices:
            shuffled[index] = rng.permutation(target_code[index])
        null[draw] = discrete_mi(assignment_code, shuffled)
    p_value = (1 + np.sum(null >= observed - 1e-15)) / (draws + 1)
    return observed, float(p_value), float(np.quantile(null, 0.95))


def association(assignment: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "adjusted_mutual_info": float(adjusted_mutual_info_score(target, assignment)),
        "normalized_mutual_info": float(normalized_mutual_info_score(target, assignment)),
        "adjusted_rand": float(adjusted_rand_score(target, assignment)),
        "homogeneity": float(homogeneity_score(target, assignment)),
        "completeness": float(completeness_score(target, assignment)),
        "v_measure": float(v_measure_score(target, assignment)),
    }


def composition_rows(frame: pd.DataFrame) -> pd.DataFrame:
    overall = frame["phenotype"].value_counts(normalize=True)
    rows = []
    for assignment_name, column in ASSIGNMENTS.items():
        for cluster, part in frame.groupby(column, sort=True):
            row: dict[str, Any] = {
                "assignment": assignment_name,
                "cluster": int(cluster),
                "n": len(part),
                "success_rate": float(part["success"].mean()),
                "episode_length_median": float(part["episode_length"].median()),
            }
            counts = part["phenotype"].value_counts()
            for phenotype in ("normal", "loop_only", "static_only", "both"):
                count = int(counts.get(phenotype, 0))
                rate = count / len(part)
                row[f"{phenotype}_n"] = count
                row[f"{phenotype}_rate"] = rate
                row[f"{phenotype}_enrichment"] = rate / max(
                    float(overall.get(phenotype, 0)), 1e-12
                )
            rows.append(row)
    return pd.DataFrame(rows)


def long_profile(
    matrix: np.ndarray,
    names: np.ndarray,
    frame: pd.DataFrame,
    scope: str,
    grouping: str,
    values: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for value in np.unique(values):
        take = values == value
        for index, feature in enumerate(names):
            feature_values = matrix[take, index]
            rows.append(
                {
                    "scope": scope,
                    "grouping": grouping,
                    "group_value": str(value),
                    "n": int(take.sum()),
                    "feature": str(feature),
                    "mean": float(feature_values.mean()),
                    "median": float(np.median(feature_values)),
                    "std": float(feature_values.std()),
                }
            )
    return pd.DataFrame(rows)


def onset_state_profiles(
    frame: pd.DataFrame,
    sequence: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    query = np.arange(sequence.shape[1])
    state_values = [int(value) for value in np.unique(sequence) if value != -2]
    for event in ("loop", "static"):
        episode_rows = []
        for record in frame[frame[f"{event}_label"]].itertuples(index=False):
            episode = int(record.episode_id)
            onset = int(getattr(record, f"{event}_onset"))
            valid = sequence[episode] != -2
            pre = valid & (query <= onset - 2)
            post = valid & (query >= onset + 2)
            if pre.sum() < 3 or post.sum() < 3:
                continue
            episode_rows.append(
                {
                    "episode_id": episode,
                    **{
                        f"pre_state{state}": float(
                            np.mean(sequence[episode, pre] == state)
                        )
                        for state in state_values
                    },
                    **{
                        f"post_state{state}": float(
                            np.mean(sequence[episode, post] == state)
                        )
                        for state in state_values
                    },
                }
            )
        paired = pd.DataFrame(episode_rows)
        for state in state_values:
            pre = paired[f"pre_state{state}"].to_numpy(dtype=np.float64)
            post = paired[f"post_state{state}"].to_numpy(dtype=np.float64)
            delta = post - pre
            bootstrap_index = rng.integers(0, len(delta), size=(draws, len(delta)))
            bootstrap_mean = delta[bootstrap_index].mean(axis=1)
            random_sign = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(delta)))
            null = (delta[None, :] * random_sign).mean(axis=1)
            p_value = (1 + np.sum(np.abs(null) >= abs(delta.mean()) - 1e-15)) / (
                draws + 1
            )
            rows.append(
                {
                    "event": event,
                    "state": state,
                    "paired_episode_n": len(delta),
                    "pre_occupancy_mean": float(pre.mean()),
                    "post_occupancy_mean": float(post.mean()),
                    "delta_mean": float(delta.mean()),
                    "delta_median": float(np.median(delta)),
                    "delta_bootstrap_ci_low": float(np.quantile(bootstrap_mean, 0.025)),
                    "delta_bootstrap_ci_high": float(np.quantile(bootstrap_mean, 0.975)),
                    "paired_sign_permutation_p": float(p_value),
                    "permutations": draws,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.self_test:
        left = np.asarray([0, 0, 1, 1])
        assert discrete_mi(left, left) > 0.69
        assert discrete_mi(left, np.asarray([0, 1, 0, 1])) < 1e-12
        print("self-test passed")
        return
    if args.permutations < 1:
        raise ValueError("permutations must be positive")

    manifest = verify_seal(args.results)
    assignments = pd.read_csv(args.results / "unlabeled_trajectory_assignments.csv")
    labels = load_labels(args.labels)
    meta = pd.read_csv(args.meta, usecols=["episode_id", "group", "success", "T"])
    frame = assignments.merge(labels, on=["episode_id", "group"], validate="one_to_one")
    frame = frame.merge(
        meta.rename(columns={"T": "episode_length"}),
        on=["episode_id", "group"],
        validate="one_to_one",
    ).sort_values("episode_id")
    if len(frame) != 512:
        raise ValueError("evaluation merge lost episodes")

    rng = np.random.default_rng(args.seed)
    test_rows = []
    base_targets = {
        "phenotype_4way": frame["phenotype"].to_numpy(),
        "trap_vs_normal": frame["trap_label"].astype(str).to_numpy(),
        "success_vs_failure": frame["success"].astype(str).to_numpy(),
    }
    for assignment_name, column in ASSIGNMENTS.items():
        assignment_values = frame[column].to_numpy(dtype=np.int64)
        for target_name, target in base_targets.items():
            observed, p_value, null_q95 = conditional_permutation(
                assignment_values,
                target,
                frame["group"].to_numpy(dtype=np.int64),
                args.permutations,
                rng,
            )
            test_rows.append(
                {
                    "assignment": assignment_name,
                    "target": target_name,
                    "sample_n": len(frame),
                    **association(assignment_values, target),
                    "mutual_info": observed,
                    "conditional_permutation_p": p_value,
                    "null_mi_q95": null_q95,
                    "permutations": args.permutations,
                }
            )
        exclusive = frame[frame["phenotype"] != "both"]
        assignment_values = exclusive[column].to_numpy(dtype=np.int64)
        target = exclusive["phenotype"].to_numpy()
        observed, p_value, null_q95 = conditional_permutation(
            assignment_values,
            target,
            exclusive["group"].to_numpy(dtype=np.int64),
            args.permutations,
            rng,
        )
        test_rows.append(
            {
                "assignment": assignment_name,
                "target": "phenotype_3way_excluding_both",
                "sample_n": len(exclusive),
                **association(assignment_values, target),
                "mutual_info": observed,
                "conditional_permutation_p": p_value,
                "null_mi_q95": null_q95,
                "permutations": args.permutations,
            }
        )

    subset_specs = (
        ("phenotype_4way_failures_only", frame[~frame["success"]], "phenotype"),
        ("phenotype_4way_T52", frame[frame["episode_length"] == 52], "phenotype"),
        (
            "loop_vs_static_exclusive",
            frame[frame["phenotype"].isin(["loop_only", "static_only"])],
            "phenotype",
        ),
    )
    for assignment_name, column in (
        ("early_gmm_bic_partition", "early_gmm_bic_cluster"),
        ("full_gmm_bic_partition", "full_gmm_bic_cluster"),
    ):
        for target_name, subset, target_column in subset_specs:
            assignment_values = subset[column].to_numpy(dtype=np.int64)
            target = subset[target_column].to_numpy()
            observed, p_value, null_q95 = conditional_permutation(
                assignment_values,
                target,
                subset["group"].to_numpy(dtype=np.int64),
                args.permutations,
                rng,
            )
            test_rows.append(
                {
                    "assignment": assignment_name,
                    "target": target_name,
                    "sample_n": len(subset),
                    **association(assignment_values, target),
                    "mutual_info": observed,
                    "conditional_permutation_p": p_value,
                    "null_mi_q95": null_q95,
                    "permutations": args.permutations,
                }
            )

    joint_stratum = (
        frame["group"].astype(str) + "_T" + frame["episode_length"].astype(str)
    ).to_numpy()
    assignment_values = frame["full_gmm_bic_cluster"].to_numpy(dtype=np.int64)
    target = frame["phenotype"].to_numpy()
    observed, p_value, null_q95 = conditional_permutation(
        assignment_values,
        target,
        joint_stratum,
        args.permutations,
        rng,
    )
    test_rows.append(
        {
            "assignment": "full_gmm_bic_partition",
            "target": "phenotype_4way_pool_and_length_conditioned",
            "sample_n": len(frame),
            **association(assignment_values, target),
            "mutual_info": observed,
            "conditional_permutation_p": p_value,
            "null_mi_q95": null_q95,
            "permutations": args.permutations,
        }
    )
    tests = pd.DataFrame(test_rows)

    nuisance_rows = []
    for assignment_name, column in ASSIGNMENTS.items():
        assignment_values = frame[column].to_numpy(dtype=np.int64)
        for target_name, target in (
            ("init_pool", frame["group"].astype(str).to_numpy()),
            ("episode_length", frame["episode_length"].astype(str).to_numpy()),
        ):
            nuisance_rows.append(
                {
                    "assignment": assignment_name,
                    "target": target_name,
                    **association(assignment_values, target),
                }
            )
    nuisances = pd.DataFrame(nuisance_rows)
    compositions = composition_rows(frame)

    with np.load(args.results / "unlabeled_dynamics.npz", allow_pickle=False) as archive:
        feature_names = np.asarray(archive["grammar_feature_names"])
        early_grammar = np.asarray(archive["early_grammar_q95"], dtype=np.float64)
        full_grammar = np.asarray(archive["full_grammar_q95"], dtype=np.float64)
        full_sequence = np.asarray(archive["full_sequence_q95"], dtype=np.int16)
    profiles = []
    for scope, matrix, cluster_column in (
        ("early", early_grammar, "early_gmm_bic_cluster"),
        ("full", full_grammar, "full_gmm_bic_cluster"),
    ):
        profiles.append(
            long_profile(
                matrix,
                feature_names,
                frame,
                scope,
                "phenotype",
                frame["phenotype"].to_numpy(),
            )
        )
        profiles.append(
            long_profile(
                matrix,
                feature_names,
                frame,
                scope,
                "gmm_cluster",
                frame[cluster_column].to_numpy(),
            )
        )
    profile_frame = pd.concat(profiles, ignore_index=True)
    onset_profiles = onset_state_profiles(
        frame,
        full_sequence,
        args.permutations,
        np.random.default_rng(args.seed + 91),
    )

    stability = pd.read_csv(args.results / "stability_sentinels.csv")
    stability_summary = (
        stability.groupby(["scope", "algorithm", "test"], dropna=False)
        .agg(
            repeats=("repeat", "count"),
            cluster_median=("clusters", "median"),
            cluster_min=("clusters", "min"),
            cluster_max=("clusters", "max"),
            selected_k_median=("selected_k", "median"),
            selected_k_min=("selected_k", "min"),
            selected_k_max=("selected_k", "max"),
            ari_median=("ari_vs_primary", "median"),
            ari_q25=("ari_vs_primary", lambda value: value.quantile(0.25)),
            ari_min=("ari_vs_primary", "min"),
            silhouette_median=("silhouette_clustered", "median"),
        )
        .reset_index()
    )

    frame.to_csv(args.results / "labeled_trajectory_assignments.csv", index=False)
    tests.to_csv(args.results / "label_association_tests.csv", index=False)
    nuisances.to_csv(args.results / "nuisance_associations.csv", index=False)
    compositions.to_csv(args.results / "cluster_composition.csv", index=False)
    profile_frame.to_csv(args.results / "grammar_profiles.csv", index=False)
    onset_profiles.to_csv(args.results / "onset_state_profiles.csv", index=False)
    stability_summary.to_csv(args.results / "stability_summary.csv", index=False)

    def result(assignment_name: str, target: str) -> dict[str, Any]:
        row = tests[(tests["assignment"] == assignment_name) & (tests["target"] == target)]
        return plain(row.iloc[0].to_dict())

    summary = {
        "schema": "himoe.moe_dynamics_unsupervised_evaluation.v1",
        "unlabeled_manifest_sha256": sha256(args.results / "unlabeled_manifest.json"),
        "sealed_assignment_sha256": manifest["artifact_sha256"][
            "unlabeled_trajectory_assignments.csv"
        ],
        "labels_opened_after_seal": True,
        "label_inventory": frame["phenotype"].value_counts().to_dict(),
        "outcome_inventory": {
            "success": int(frame["success"].sum()),
            "failure": int((~frame["success"]).sum()),
        },
        "unlabeled_structure": {
            "state_vocabulary": manifest["state_vocabulary"],
            "early_hdbscan": manifest["early_primary"],
            "full_hdbscan": manifest["full_retrospective_primary"],
            "gmm_bic_selected_components": manifest["gmm_bic_selected_components"],
            "positive_control": manifest["positive_control"],
        },
        "key_label_tests": {
            "early_hdbscan_4way": result("early_hdbscan_density", "phenotype_4way"),
            "early_gmm_4way": result("early_gmm_bic_partition", "phenotype_4way"),
            "early_gmm_3way": result(
                "early_gmm_bic_partition", "phenotype_3way_excluding_both"
            ),
            "full_gmm_4way": result("full_gmm_bic_partition", "phenotype_4way"),
            "full_gmm_4way_failures_only": result(
                "full_gmm_bic_partition", "phenotype_4way_failures_only"
            ),
            "full_gmm_loop_vs_static": result(
                "full_gmm_bic_partition", "loop_vs_static_exclusive"
            ),
            "full_gmm_4way_pool_and_length_conditioned": result(
                "full_gmm_bic_partition",
                "phenotype_4way_pool_and_length_conditioned",
            ),
            "early_gmm_success": result(
                "early_gmm_bic_partition", "success_vs_failure"
            ),
            "full_gmm_success": result(
                "full_gmm_bic_partition", "success_vs_failure"
            ),
        },
        "onset_state_profiles": onset_profiles.to_dict(orient="records"),
        "interpretation_guardrails": [
            "The dynamics method was designed after the earlier peak-cluster label audit and is exploratory.",
            "HDBSCAN found no natural trajectory-density clusters; GMM-BIC partitions a continuous cloud.",
            "The full scope uses the observed trajectory suffix and is retrospective, not an online alarm.",
            "No label was used to select the state count, trajectory component count, or assignment.",
        ],
    }
    (args.results / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plain(summary), indent=2))


if __name__ == "__main__":
    main()
