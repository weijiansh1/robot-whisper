#!/usr/bin/env python3
"""Evaluate the pure-MoE persistent-phenotype cascade on both route cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "ONLINE_MOE_ONLY_CASCADE_V2_PROTOCOL.md"
DEFAULT_MAIN = HERE / "results/online_multihead_hub/sealed_online_scores.npz"
DEFAULT_EXTERNAL = (
    HERE / "results/online_precision_cascade_external/sealed_online_scores.npz"
)
DEFAULT_SHARED = HERE / "results/online_precision_cascade_shared/first_alarms.npz"
DEFAULT_LABELS = HERE / "results/timeout_extension_plus10"
DEFAULT_OUTPUT = HERE / "results/online_moe_only_cascade_v2"

HEAD_GATE = 0.80
PHENOTYPE_CONFIRMATIONS = 4
QUANTILE = 0.95
HEADS = ("lock_in", "flat_narrow_support")
DETECTORS = (
    "mobility_warning_k2",
    "multihead_warning_k2_or_static_k4",
    "mobility_hard_k4",
    "multihead_hard_k4_or_static_k4",
)
EARLY_LEADS = (2, 4, 8, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
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


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def first_consecutive_above(
    values: np.ndarray, cutoff: float, count: int
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("head values must have shape [episode, query]")
    streak = np.zeros(len(values), dtype=np.int16)
    first = np.full(len(values), -1, dtype=np.int16)
    for query in range(values.shape[1]):
        crossing = np.isfinite(values[:, query]) & (values[:, query] > cutoff)
        streak = np.where(crossing, streak + 1, 0)
        new = (streak >= count) & (first < 0)
        first[new] = query
    return first


def first_or(*values: np.ndarray) -> np.ndarray:
    if not values:
        raise ValueError("first_or needs at least one input")
    stack = np.stack([np.asarray(value, dtype=np.int16) for value in values])
    available = stack >= 0
    masked = np.where(available, stack, np.iinfo(np.int16).max)
    output = masked.min(axis=0).astype(np.int16)
    output[~available.any(axis=0)] = -1
    return output


def head_arrays(sealed: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    if "route_head_scores" in sealed:
        names = sealed["route_head_names"].astype(str).tolist()
        values = np.asarray(sealed["route_head_scores"], dtype=np.float32)
    else:
        names = sealed["head_names"].astype(str).tolist()
        detector_names = sealed["detector_names"].astype(str).tolist()
        positions = [detector_names.index(name) for name in names]
        values = np.asarray(sealed["scores"][:, :, positions], dtype=np.float32)
    if set(HEADS) - set(names):
        raise ValueError("sealed scores do not contain both lock heads")
    return names, values


def aligned_labels(
    cohort: str, sealed: dict[str, np.ndarray], path: Path
) -> pd.DataFrame:
    task_names = sealed["task_names"].astype(str)
    index = pd.DataFrame(
        {
            "sealed_row": np.arange(len(sealed["episode"])),
            "task": task_names[sealed["task_index"].astype(int)],
            "episode": sealed["episode"].astype(int),
            "init_state_id": sealed["init_state_id"].astype(int),
            "flow_noise_seed": sealed["flow_noise_seed"].astype(int),
            "length": sealed["length"].astype(int),
        }
    )
    filename = (
        "development_main_clean_labels.csv"
        if cohort == "development_main"
        else "external_8b_clean_labels.csv"
    )
    labels = pd.read_csv(path / filename)[
        [
            "task",
            "episode",
            "original_failure",
            "failure",
            "late_success_plus10_queries",
        ]
    ]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("sealed_row")
    if merged["failure"].isna().any():
        raise ValueError(f"{cohort} outcome labels do not align")
    merged["cohort"] = cohort
    merged["suite"] = merged["task"].str.split("/", n=1).str[0]
    merged["category"] = np.where(
        ~merged["original_failure"],
        "timely",
        np.where(merged["late_success_plus10_queries"], "late", "persistent"),
    )
    return merged.reset_index(drop=True)


def detector_first_queries(
    cohort: str,
    sealed: dict[str, np.ndarray],
    shared: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    names, heads = head_arrays(sealed)
    lock = first_consecutive_above(
        heads[:, :, names.index("lock_in")], HEAD_GATE, PHENOTYPE_CONFIRMATIONS
    )
    flat = first_consecutive_above(
        heads[:, :, names.index("flat_narrow_support")],
        HEAD_GATE,
        PHENOTYPE_CONFIRMATIONS,
    )
    static = first_or(lock, flat)

    levels = shared["levels"].astype(str).tolist()
    quantiles = shared["quantiles"].astype(float)
    q_position = int(np.flatnonzero(np.isclose(quantiles, QUANTILE))[0])
    source = (
        shared["main_first"]
        if cohort == "development_main"
        else shared["external_first"]
    )
    warning = source[:, levels.index("mobility_w4_k2"), q_position].astype(np.int16)
    hard = source[:, levels.index("mobility_w4_k4"), q_position].astype(np.int16)
    if len(warning) != len(heads):
        raise ValueError(f"{cohort} mobility and route-head rows do not align")

    detector = {
        "mobility_warning_k2": warning,
        "multihead_warning_k2_or_static_k4": first_or(warning, static),
        "mobility_hard_k4": hard,
        "multihead_hard_k4_or_static_k4": first_or(hard, static),
    }
    branches = {"lock_in": lock, "flat_narrow_support": flat, "static": static}
    return detector, branches


def metric_row(
    cohort: str,
    detector: str,
    first: np.ndarray,
    labels: pd.DataFrame,
    group: str = "all",
) -> dict[str, Any]:
    alarm = np.asarray(first) >= 0
    risk = labels["original_failure"].to_numpy(dtype=bool)
    timely = ~risk
    late = labels["late_success_plus10_queries"].to_numpy(dtype=bool)
    persistent = labels["failure"].to_numpy(dtype=bool)
    lead = labels["length"].to_numpy(dtype=int) - 1 - np.asarray(first, dtype=int)
    tp = int((alarm & risk).sum())
    fp = int((alarm & timely).sum())
    row: dict[str, Any] = {
        "cohort": cohort,
        "group": group,
        "detector": detector,
        "episodes": len(labels),
        "risk_n": int(risk.sum()),
        "timely_n": int(timely.sum()),
        "late_n": int(late.sum()),
        "persistent_n": int(persistent.sum()),
        "tp": tp,
        "fp": fp,
        "fn": int((~alarm & risk).sum()),
        "tn": int((~alarm & timely).sum()),
        "risk_recall": ratio(tp, risk.sum()),
        "timely_fpr": ratio(fp, timely.sum()),
        "precision": ratio(tp, tp + fp),
        "late_alarm_n": int((alarm & late).sum()),
        "persistent_alarm_n": int((alarm & persistent).sum()),
        "late_recall": ratio((alarm & late).sum(), late.sum()),
        "persistent_recall": ratio((alarm & persistent).sum(), persistent.sum()),
    }
    for early in EARLY_LEADS:
        row[f"early{early}_risk_recall"] = ratio(
            (alarm & risk & (lead >= early)).sum(), risk.sum()
        )
    detected_lead = lead[alarm & risk]
    row["detected_risk_lead_median"] = (
        float(np.median(detected_lead)) if len(detected_lead) else float("nan")
    )
    return row


def grouped_metrics(
    labels: pd.DataFrame,
    detector: dict[str, np.ndarray],
    column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    values = labels[column].to_numpy(dtype=str)
    for group in np.unique(values):
        take = values == group
        block = labels.loc[take].reset_index(drop=True)
        for name, first in detector.items():
            rows.append(
                metric_row(
                    str(block.iloc[0]["cohort"]), name, first[take], block, group
                )
            )
    return pd.DataFrame(rows)


def branch_rows(
    cohort: str,
    labels: pd.DataFrame,
    detector: dict[str, np.ndarray],
    branches: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    category = labels["category"].to_numpy(dtype=str)
    pairs = (
        ("multihead_warning_k2_or_static_k4", "mobility_warning_k2"),
        ("multihead_hard_k4_or_static_k4", "mobility_hard_k4"),
    )
    for combined_name, mobility_name in pairs:
        combined = detector[combined_name]
        mobility = detector[mobility_name]
        static = branches["static"]
        alarm = combined >= 0
        mobility_first = alarm & (mobility == combined)
        static_first = alarm & (static == combined)
        labels_by_branch = np.full(len(labels), "none", dtype=object)
        labels_by_branch[mobility_first & ~static_first] = "mobility"
        labels_by_branch[static_first & ~mobility_first] = "static"
        labels_by_branch[mobility_first & static_first] = "mobility+static"
        for branch in ("mobility", "static", "mobility+static"):
            for outcome in ("timely", "late", "persistent"):
                rows.append(
                    {
                        "cohort": cohort,
                        "detector": combined_name,
                        "first_branch": branch,
                        "outcome": outcome,
                        "count": int(
                            ((labels_by_branch == branch) & (category == outcome)).sum()
                        ),
                    }
                )
    return rows


def paired_bootstrap(
    labels: pd.DataFrame,
    baseline: np.ndarray,
    multihead: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    task = labels["task"].to_numpy(dtype=str)
    tasks = np.unique(task)
    risk = labels["original_failure"].to_numpy(dtype=bool)
    timely = ~risk
    base_alarm = np.asarray(baseline) >= 0
    multi_alarm = np.asarray(multihead) >= 0
    counts = []
    for name in tasks:
        take = task == name
        counts.append(
            (
                int((base_alarm & risk & take).sum()),
                int((multi_alarm & risk & take).sum()),
                int((risk & take).sum()),
                int((base_alarm & timely & take).sum()),
                int((multi_alarm & timely & take).sum()),
                int((timely & take).sum()),
                int((base_alarm & take).sum()),
                int((multi_alarm & take).sum()),
            )
        )
    count = np.asarray(counts, dtype=np.float64)
    samples = rng.integers(0, len(tasks), size=(draws, len(tasks)))
    total = count[samples].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall_delta = total[:, 1] / total[:, 2] - total[:, 0] / total[:, 2]
        fpr_delta = total[:, 4] / total[:, 5] - total[:, 3] / total[:, 5]
        precision_delta = total[:, 1] / total[:, 7] - total[:, 0] / total[:, 6]

    result: dict[str, float] = {}
    for name, values in (
        ("risk_recall_delta", recall_delta),
        ("timely_fpr_delta", fpr_delta),
        ("precision_delta", precision_delta),
    ):
        finite = values[np.isfinite(values)]
        result[f"{name}_ci_low"] = float(np.quantile(finite, 0.025))
        result[f"{name}_ci_high"] = float(np.quantile(finite, 0.975))
    return result


def comparison_rows(
    cohort: str,
    labels: pd.DataFrame,
    detector: dict[str, np.ndarray],
    draws: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = (
        (
            "warning",
            "mobility_warning_k2",
            "multihead_warning_k2_or_static_k4",
        ),
        ("hard", "mobility_hard_k4", "multihead_hard_k4_or_static_k4"),
    )
    for level, base_name, multi_name in pairs:
        base = metric_row(cohort, base_name, detector[base_name], labels)
        multi = metric_row(cohort, multi_name, detector[multi_name], labels)
        row = {
            "cohort": cohort,
            "level": level,
            "baseline": base_name,
            "candidate": multi_name,
            "risk_recall_delta": multi["risk_recall"] - base["risk_recall"],
            "timely_fpr_delta": multi["timely_fpr"] - base["timely_fpr"],
            "precision_delta": multi["precision"] - base["precision"],
            "incremental_tp": multi["tp"] - base["tp"],
            "incremental_fp": multi["fp"] - base["fp"],
        }
        row.update(
            paired_bootstrap(
                labels,
                detector[base_name],
                detector[multi_name],
                draws,
                rng,
            )
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    main_sealed = load_npz(args.main)
    external_sealed = load_npz(args.external)
    shared = load_npz(args.shared)
    cohorts = (
        (
            "development_main",
            main_sealed,
            aligned_labels("development_main", main_sealed, args.labels),
        ),
        (
            "external_8b",
            external_sealed,
            aligned_labels("external_8b", external_sealed, args.labels),
        ),
    )

    outcome_rows: list[dict[str, Any]] = []
    suite_frames: list[pd.DataFrame] = []
    task_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    branch_output: list[dict[str, Any]] = []
    comparison_output: list[dict[str, Any]] = []
    first_output: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(args.seed)

    for cohort, sealed, labels in cohorts:
        detector, branches = detector_first_queries(cohort, sealed, shared)
        for name, first in detector.items():
            outcome_rows.append(metric_row(cohort, name, first, labels))
        suite_frames.append(grouped_metrics(labels, detector, "suite"))
        task_frames.append(grouped_metrics(labels, detector, "task"))
        branch_output.extend(branch_rows(cohort, labels, detector, branches))
        comparison_output.extend(
            comparison_rows(cohort, labels, detector, args.bootstrap, rng)
        )

        episode = labels.copy()
        for name, first in detector.items():
            episode[f"first_{name}_query"] = first
        for name, first in branches.items():
            episode[f"first_{name}_query"] = first
        episode_frames.append(episode)
        prefix = "main" if cohort == "development_main" else "external"
        for name, first in detector.items():
            first_output[f"{prefix}_{name}"] = first
        for name, first in branches.items():
            first_output[f"{prefix}_{name}"] = first

    outcome = pd.DataFrame(outcome_rows)
    comparisons = pd.DataFrame(comparison_output)
    args.output.mkdir(parents=True, exist_ok=True)
    outcome.to_csv(args.output / "outcome_metrics.csv", index=False)
    pd.concat(suite_frames, ignore_index=True).to_csv(
        args.output / "outcome_metrics_by_suite.csv", index=False
    )
    pd.concat(task_frames, ignore_index=True).to_csv(
        args.output / "outcome_metrics_by_task.csv", index=False
    )
    pd.concat(episode_frames, ignore_index=True).to_csv(
        args.output / "episode_alarms.csv", index=False
    )
    pd.DataFrame(branch_output).to_csv(
        args.output / "first_branch_composition.csv", index=False
    )
    comparisons.to_csv(args.output / "paired_comparisons.csv", index=False)
    np.savez_compressed(
        args.output / "first_alarms.npz",
        schema=np.asarray("himoe.moe_only_cascade_v2.first_alarms.v1"),
        detector_names=np.asarray(DETECTORS),
        head_gate=np.asarray(HEAD_GATE),
        phenotype_confirmations=np.asarray(PHENOTYPE_CONFIRMATIONS),
        mobility_quantile=np.asarray(QUANTILE),
        **first_output,
    )

    warning = outcome[outcome["detector"] == "multihead_warning_k2_or_static_k4"]
    hard = outcome[outcome["detector"] == "multihead_hard_k4_or_static_k4"]
    criteria = {
        "warning_useful": bool(
            (
                warning["risk_recall"].to_numpy()
                > outcome[outcome["detector"] == "mobility_warning_k2"][
                    "risk_recall"
                ].to_numpy()
            ).all()
            and (warning["timely_fpr"] <= 0.01).all()
            and (
                comparisons[comparisons["level"] == "warning"]["precision_delta"]
                >= -0.02
            ).all()
        ),
        "hard_useful": bool(
            (
                comparisons[comparisons["level"] == "hard"]["risk_recall_delta"] >= 0.05
            ).all()
            and (hard["timely_fpr"] <= 0.005).all()
            and (hard["precision"] >= 0.80).all()
        ),
    }
    artifacts = {
        "protocol_sha256": sha256(PROTOCOL),
        "evaluator_sha256": sha256(Path(__file__)),
        "runtime_monitor_sha256": sha256(HERE / "moe_only_cascade_monitor.py"),
        "main_sealed_scores_sha256": sha256(args.main),
        "external_sealed_scores_sha256": sha256(args.external),
        "shared_mobility_alarms_sha256": sha256(args.shared),
        "outcome_metrics_sha256": sha256(args.output / "outcome_metrics.csv"),
        "first_alarms_sha256": sha256(args.output / "first_alarms.npz"),
    }
    summary = {
        "schema": "himoe.moe_only_cascade_v2.evaluation.v1",
        "status": "post-hoc exploratory; neither cohort is a sealed independent confirmation for this rule",
        "runtime_inputs": ["current_router_prob", "current_expert_ids"],
        "runtime_only_moe": True,
        "runtime_future_access": False,
        "runtime_cross_rollout_access": False,
        "runtime_outcome_access": False,
        "threshold_calibration_outcome_free": True,
        "method_selection_outcome_free": False,
        "risk_target": "late_or_persistent_original_horizon_failure",
        "timely_success_is_negative": True,
        "head_gate": HEAD_GATE,
        "phenotype_confirmations": PHENOTYPE_CONFIRMATIONS,
        "mobility_quantile": QUANTILE,
        "criteria": criteria,
        "metrics": outcome.to_dict(orient="records"),
        "paired_comparisons": comparisons.to_dict(orient="records"),
        "artifacts": artifacts,
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        outcome[
            [
                "cohort",
                "detector",
                "tp",
                "fp",
                "risk_recall",
                "timely_fpr",
                "precision",
                "late_recall",
                "persistent_recall",
                "early4_risk_recall",
                "detected_risk_lead_median",
            ]
        ].to_string(index=False)
    )
    print("\nPaired changes:\n" + comparisons.to_string(index=False))
    print("\nDecision criteria: " + json.dumps(criteria, sort_keys=True))


if __name__ == "__main__":
    main()
