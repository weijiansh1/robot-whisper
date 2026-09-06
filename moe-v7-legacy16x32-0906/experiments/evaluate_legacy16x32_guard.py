#!/usr/bin/env python3
"""Apply the frozen v7 intrinsic guard to the legacy right-16x32 cohort.

Nothing is recalibrated: the profile scalars are the published ones (re-derived
from the same reference corpus and asserted equal), the Boolean mechanism is the
frozen ``freeze OR (acceleration AND periodicity)``, and the operating point is
not re-selected.  The 16x32 cohort contributes no thresholds.

Outcome labels for this cohort come from ``client/summaries.json`` (raw
original-horizon success), which is the same risk definition as
``original_failure`` for development_main and external_8b.  There is no
plus-10 continuation study for 16x32, so the late/persistent split is not
available here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frozen_profile import BUNDLE, SEALED_ALARMS, V7, guard_alarms, load_frozen_point, sealed
from raw_route_features import COHORTS, WORKSPACE, load_npz


DEFAULT_FEATURES = BUNDLE / "results/raw_features"
DEFAULT_OUTPUT = BUNDLE / "results/legacy16x32"
SUITE_CAP = {"libero_goal": 30, "libero_long": 52, "libero_spatial": 22}
BUDGETS = (0, 2, 4, 8, 12)
EARLY_LEADS = (2, 4, 8, 12)
PRIOR_BANDS = ((0.0, 0.25), (0.25, 0.75), (0.75, 1.01))
CONTROL_CHUNKS = (0, 4, 8, 12)
BRANCHES = ("freeze", "acceleration", "periodicity", "turbulence", "guard")
DETECTOR_LABEL = {
    "guard": "intrinsic_guard_v7 (freeze OR turbulence)",
    "freeze": "branch: relative freeze",
    "turbulence": "branch: confirmed turbulence (acceleration AND periodicity)",
    "acceleration": "sub-condition: flow acceleration (only enters via turbulence)",
    "periodicity": "sub-condition: recurrence loss (only enters via turbulence)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260906)
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


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def legacy_labels(layer: dict[str, np.ndarray]) -> pd.DataFrame:
    """Raw original-horizon outcome for every 16x32 episode."""
    root = COHORTS["legacy_16x32"].cache_root
    task_names = layer["task_names"].astype(str)
    rows: list[dict[str, Any]] = []
    for position, task in enumerate(task_names):
        summaries = json.loads(
            (root / task / "right-16x32/client/summaries.json").read_text(encoding="utf-8")
        )
        summaries.sort(key=lambda item: int(item["episode_index"]))
        table = {int(item["episode_index"]): item for item in summaries}
        for row in np.flatnonzero(layer["task_index"] == position):
            episode = int(layer["episode"][row])
            record = table[episode]
            if int(record["inference_calls"]) != int(layer["length"][row]):
                raise ValueError(f"length mismatch for {task}/{episode}")
            rows.append(
                {
                    "row": int(row),
                    "cohort": "legacy_16x32",
                    "task": task,
                    "suite": task.split("/", 1)[0],
                    "episode": episode,
                    "init_state_id": int(record["init_state_id"]),
                    "flow_noise_seed": int(record["flow_noise_seed"]),
                    "length": int(layer["length"][row]),
                    "original_failure": not bool(record["success"]),
                }
            )
    labels = pd.DataFrame(rows).sort_values("row").reset_index(drop=True)
    if not np.array_equal(labels["row"].to_numpy(), np.arange(len(layer["episode"]))):
        raise ValueError("legacy label construction is incomplete")
    return labels


def clustered_interval(
    tasks: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    names = np.unique(tasks)
    task_num = np.asarray([numerator[tasks == name].sum() for name in names])
    task_den = np.asarray([denominator[tasks == name].sum() for name in names])
    samples = rng.integers(0, len(names), size=(draws, len(names)))
    values = task_num[samples].sum(axis=1) / np.maximum(task_den[samples].sum(axis=1), 1)
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def metric_row(
    detector: str,
    first: np.ndarray,
    labels: pd.DataFrame,
    group: str,
    draws: int,
    rng: np.random.Generator,
    intervals: bool,
) -> dict[str, Any]:
    first = np.asarray(first, dtype=np.int16)
    alarm = first >= 0
    risk = labels["original_failure"].to_numpy(bool)
    timely = ~risk
    lead = labels["length"].to_numpy(int) - 1 - first
    tp = int((alarm & risk).sum())
    fp = int((alarm & timely).sum())
    row: dict[str, Any] = {
        "cohort": "legacy_16x32",
        "group": group,
        "detector": detector,
        "detector_role": DETECTOR_LABEL.get(detector, detector),
        "episodes": int(len(labels)),
        "risk_n": int(risk.sum()),
        "timely_n": int(timely.sum()),
        "alarms": int(alarm.sum()),
        "tp": tp,
        "fp": fp,
        "fn": int((~alarm & risk).sum()),
        "tn": int((~alarm & timely).sum()),
        "risk_recall": ratio(tp, risk.sum()),
        "timely_fpr": ratio(fp, timely.sum()),
        "precision": ratio(tp, tp + fp),
        "detected_risk_lead_median": float(np.median(lead[alarm & risk])) if tp else float("nan"),
    }
    for early in EARLY_LEADS:
        row[f"early{early}_risk_recall"] = ratio(
            int((alarm & risk & (lead >= early)).sum()), risk.sum()
        )
    if intervals:
        tasks = labels["task"].to_numpy(str)
        for name, numerator, denominator in (
            ("risk_recall", alarm & risk, risk),
            ("timely_fpr", alarm & timely, timely),
            ("precision", alarm & risk, alarm),
        ):
            low, high = clustered_interval(tasks, numerator, denominator, draws, rng)
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
    return row


def survival_prior(labels: pd.DataFrame, column: str) -> dict[str, dict[int, float]]:
    """P(risk | still running at chunk q), estimated within each group."""
    priors: dict[str, dict[int, float]] = {}
    for name, block in labels.groupby(column):
        risk = block["original_failure"].to_numpy(bool)
        length = block["length"].to_numpy(int)
        priors[str(name)] = {
            chunk: float(risk[length > chunk].mean()) for chunk in range(int(length.max()))
        }
    return priors


def attach_prior(
    labels: pd.DataFrame, first: np.ndarray, priors: dict[str, dict[int, float]], column: str
) -> pd.DataFrame:
    fired = labels.loc[first >= 0].copy()
    fired["chunk"] = first[first >= 0]
    fired["prior"] = [
        priors[str(key)][int(chunk)]
        for key, chunk in zip(fired[column], fired["chunk"], strict=True)
    ]
    fired["correct"] = fired["original_failure"].astype(bool)
    return fired


def survival_rows(fired: pd.DataFrame, detector: str, basis: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def record(group: str, block: pd.DataFrame) -> None:
        if block.empty:
            return
        precision = float(block["correct"].mean())
        prior = float(block["prior"].mean())
        rows.append(
            {
                "detector": detector,
                "detector_role": DETECTOR_LABEL.get(detector, detector),
                "prior_basis": basis,
                "group": group,
                "alarms": int(len(block)),
                "precision": precision,
                "matched_prior": prior,
                "net_gain": precision - prior,
                "lift": precision / prior if prior > 0 else float("nan"),
            }
        )

    record("all", fired)
    for suite, block in fired.groupby("suite"):
        record(str(suite), block)
    for task, block in fired.groupby("task"):
        record(f"task:{task}", block)
    for low, high in PRIOR_BANDS:
        band = fired[(fired["prior"] >= low) & (fired["prior"] < high)]
        if band.empty:
            # An empty band is a result, not a missing row.
            rows.append(
                {
                    "detector": detector,
                    "detector_role": DETECTOR_LABEL.get(detector, detector),
                    "prior_basis": basis,
                    "group": f"prior[{low:.2f},{high:.2f})",
                    "alarms": 0,
                    "precision": float("nan"),
                    "matched_prior": float("nan"),
                    "net_gain": float("nan"),
                    "lift": float("nan"),
                }
            )
        else:
            record(f"prior[{low:.2f},{high:.2f})", band)
    return rows


def within_episode_information(
    values: np.ndarray, valid: np.ndarray, name: str, labels: pd.DataFrame
) -> dict[str, Any]:
    """Does the channel actually move inside an episode?"""
    per_episode_distinct: list[int] = []
    within_var: list[float] = []
    within_std: list[float] = []
    episode_mean: list[float] = []
    counts: list[int] = []
    for row in range(values.shape[0]):
        series = values[row][valid[row]]
        series = series[np.isfinite(series)]
        if len(series) == 0:
            continue
        per_episode_distinct.append(int(len(np.unique(series))))
        counts.append(int(len(series)))
        episode_mean.append(float(series.mean()))
        if len(series) > 1:
            within_var.append(float(series.var(ddof=0)))
            within_std.append(float(series.std(ddof=0)))
    distinct = np.asarray(per_episode_distinct, dtype=float)
    counts_array = np.asarray(counts, dtype=float)
    within = np.asarray(within_var, dtype=float)
    between = float(np.var(np.asarray(episode_mean, dtype=float), ddof=0))
    mean_within = float(within.mean()) if len(within) else 0.0
    return {
        "feature": name,
        "episodes_with_finite_values": int(len(distinct)),
        "median_finite_queries_per_episode": float(np.median(counts_array)),
        "episodes_constant_within": int((distinct <= 1).sum()),
        "fraction_constant_within": float((distinct <= 1).mean()),
        "min_distinct_values_per_episode": float(distinct.min()),
        "median_distinct_values_per_episode": float(np.median(distinct)),
        "median_distinct_fraction_of_queries": float(np.median(distinct / counts_array)),
        "mean_within_episode_std": float(np.mean(within_std)) if within_std else 0.0,
        "mean_within_episode_variance": mean_within,
        "between_episode_variance": between,
        "within_variance_share": float(mean_within / (mean_within + between))
        if (mean_within + between) > 0
        else float("nan"),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    point = load_frozen_point()

    layer = load_npz(COHORTS["legacy_16x32"].layer_cache)
    raw = load_npz(args.features / "legacy_16x32_route_features.npz")
    valid = layer["valid"].astype(bool)
    labels = legacy_labels(layer)

    # Horizon caps: verified, not assumed.
    suite = labels["suite"].to_numpy(str)
    risk = labels["original_failure"].to_numpy(bool)
    horizon_by_suite: dict[str, int] = {}
    for name in sorted(set(suite)):
        cap = int(labels.loc[suite == name, "length"].max())
        if cap != SUITE_CAP[name]:
            raise ValueError(f"{name}: observed cap {cap} != documented {SUITE_CAP[name]}")
        observed = labels.loc[(suite == name) & risk, "length"].to_numpy(int)
        if len(observed) and not (observed == cap).all():
            raise ValueError(f"{name}: some risk episodes do not sit on the cap")
        horizon_by_suite[name] = cap
    horizon = np.asarray([horizon_by_suite[name] for name in suite], dtype=int)

    alarms = guard_alarms(
        point, layer["mobility"], raw["route_acceleration"], raw["lag_periodicity"], valid
    )
    scores = alarms.pop("_scores")
    alarms_fully_raw = guard_alarms(
        point, raw["mobility"], raw["route_acceleration"], raw["lag_periodicity"], valid
    )
    alarms_fully_raw.pop("_scores")

    np.savez_compressed(
        args.output / "legacy16x32_first_alarms.npz",
        schema=np.asarray("himoe.legacy16x32.alarms.v1"),
        **{f"legacy_{name}": alarms[name] for name in BRANCHES},
        **{f"legacy_fullyraw_{name}": alarms_fully_raw[name] for name in BRANCHES},
    )

    # ---------------------------------------------------------------- metrics
    rows: list[dict[str, Any]] = []
    for detector in BRANCHES:
        rows.append(
            metric_row(detector, alarms[detector], labels, "all", args.bootstrap, rng, True)
        )
    for column, prefix in (("suite", ""), ("task", "task:")):
        for name in sorted(set(labels[column])):
            take = labels[column].to_numpy(str) == name
            block = labels.loc[take].reset_index(drop=True)
            for detector in BRANCHES:
                rows.append(
                    metric_row(
                        detector,
                        alarms[detector][take],
                        block,
                        f"{prefix}{name}",
                        args.bootstrap,
                        rng,
                        False,
                    )
                )

    # Restatement of the label definition, kept out of every ranking.
    length_first = np.where(labels["length"].to_numpy(int) >= horizon, horizon - 1, -1).astype(
        np.int16
    )
    rows.append(
        metric_row(
            "horizon_cap_length_restatement",
            length_first,
            labels,
            "all",
            args.bootstrap,
            rng,
            True,
        )
    )
    for name in sorted(set(suite)):
        take = suite == name
        rows.append(
            metric_row(
                "horizon_cap_length_restatement",
                length_first[take],
                labels.loc[take].reset_index(drop=True),
                name,
                args.bootstrap,
                rng,
                False,
            )
        )
    metrics = pd.DataFrame(rows)
    metrics["is_baseline"] = ~metrics["detector"].eq("horizon_cap_length_restatement")
    metrics.to_csv(args.output / "outcome_metrics.csv", index=False)

    # ------------------------------------------------------- survival scoring
    priors = {"suite": survival_prior(labels, "suite"), "task": survival_prior(labels, "task")}
    prior_rows: list[dict[str, Any]] = []
    for basis, column in (("suite_matched", "suite"), ("task_matched", "task")):
        for name, curve in priors[column].items():
            crossing = next(
                (chunk for chunk in sorted(curve) if curve[chunk] >= 0.25), None
            )
            for chunk, value in sorted(curve.items()):
                prior_rows.append(
                    {
                        "prior_basis": basis,
                        "group": name,
                        "chunk": chunk,
                        "still_running": int(
                            (labels.loc[labels[column] == name, "length"] > chunk).sum()
                        ),
                        "prior": value,
                        "first_chunk_prior_ge_0.25": crossing,
                    }
                )
    pd.DataFrame(prior_rows).to_csv(args.output / "survival_prior_curves.csv", index=False)

    task_risk_rate = labels.groupby("task")["original_failure"].mean()
    suite_of_task = labels.drop_duplicates("task").set_index("task")["suite"]
    riskier = {
        task
        for task, rate in task_risk_rate.items()
        if rate > labels.loc[labels["suite"] == suite_of_task[task], "original_failure"].mean()
    }
    task_identity_first = np.where(
        labels["task"].isin(riskier).to_numpy(bool), 0, -1
    ).astype(np.int16)
    survival: list[dict[str, Any]] = []
    alarm_priors: list[pd.DataFrame] = []
    for basis, column in (("suite_matched", "suite"), ("task_matched", "task")):
        for detector in BRANCHES:
            fired = attach_prior(labels, alarms[detector], priors[column], column)
            survival.extend(survival_rows(fired, detector, basis))
            frame = fired[["suite", "task", "episode", "chunk", "prior", "correct"]].copy()
            frame["detector"] = detector
            frame["prior_basis"] = basis
            alarm_priors.append(frame)
        # Controls: a detector that fires on every episode still running at chunk c
        # carries no information beyond survival, so its task-matched lift must be
        # exactly 1.0. Any deviation under the suite-matched prior is the inflation.
        for chunk in CONTROL_CHUNKS:
            control = np.where(labels["length"].to_numpy(int) > chunk, chunk, -1).astype(np.int16)
            fired = attach_prior(labels, control, priors[column], column)
            survival.extend(
                survival_rows(fired, f"control_all_running_at_chunk{chunk}", basis)
            )
        # A pure task-identity selector: fire at chunk 0 on the higher-risk task of
        # each suite. It reads no routing state at all, so its task-matched lift is
        # exactly 1.0 by construction; whatever the suite-matched prior reports above
        # 1.0 is the within-suite task-risk inflation, measured on this cohort.
        fired = attach_prior(labels, task_identity_first, priors[column], column)
        survival.extend(survival_rows(fired, "control_task_identity_at_chunk0", basis))
        fired = attach_prior(labels, length_first, priors[column], column)
        survival.extend(survival_rows(fired, "horizon_cap_length_restatement", basis))
    survival_table = pd.DataFrame(survival)
    survival_table.to_csv(args.output / "survival_matched.csv", index=False)
    pd.concat(alarm_priors, ignore_index=True).to_csv(
        args.output / "alarm_priors.csv", index=False
    )

    control_check = survival_table[
        survival_table["detector"].str.startswith("control_all_running")
        & (survival_table["prior_basis"] == "task_matched")
        & (survival_table["group"] == "all")
    ]
    control_max_deviation = float(np.max(np.abs(control_check["lift"].to_numpy() - 1.0)))
    if control_max_deviation > 1e-9:
        raise AssertionError(
            "task-matched survival control does not have lift 1.0; the prior machinery is wrong"
        )

    # ------------------------------------------------------- budget retention
    budget_rows: list[dict[str, Any]] = []
    for detector in BRANCHES:
        first = np.asarray(alarms[detector], dtype=int)
        alarm = first >= 0
        budget_left = horizon - 1 - first
        lead = labels["length"].to_numpy(int) - 1 - first
        groups = [("all", np.ones(len(first), bool))]
        groups += [(name, suite == name) for name in sorted(set(suite))]
        for group, take in groups:
            for budget in BUDGETS:
                fired = alarm & (budget_left >= budget) & take
                tp = int((fired & risk).sum())
                fp = int((fired & ~risk).sum())
                budget_rows.append(
                    {
                        "detector": detector,
                        "detector_role": DETECTOR_LABEL.get(detector, detector),
                        "group": group,
                        "budget_chunks": budget,
                        "tp": tp,
                        "fp": fp,
                        "risk_n": int(risk[take].sum()),
                        "timely_n": int((~risk[take]).sum()),
                        "recall": ratio(tp, risk[take].sum()),
                        "precision": ratio(tp, tp + fp),
                        "timely_fpr": ratio(fp, (~risk[take]).sum()),
                        "median_lead_chunks": float(np.median(lead[fired & risk]))
                        if tp
                        else float("nan"),
                    }
                )
    pd.DataFrame(budget_rows).to_csv(args.output / "budget_retention.csv", index=False)

    # -------------------------------------------- within-episode information
    info_rows = [
        within_episode_information(raw["route_acceleration"], valid, "route_acceleration", labels),
        within_episode_information(raw["lag_periodicity"], valid, "lag_periodicity", labels),
    ]
    for index, layer_name in enumerate(layer["layer_names"].astype(str)):
        info_rows.append(
            within_episode_information(
                layer["mobility"][:, :, index], valid, f"layer_mobility[{layer_name}]", labels
            )
        )
    for score_name in ("freeze", "acceleration_persistent", "periodicity_persistent"):
        info_rows.append(
            within_episode_information(scores[score_name], valid, f"score:{score_name}", labels)
        )
    info_rows.append(
        within_episode_information(
            np.broadcast_to(
                labels["length"].to_numpy(np.float32)[:, None], valid.shape
            ).copy(),
            valid,
            "control:episode_length (constant by construction)",
            labels,
        )
    )
    information = pd.DataFrame(info_rows)
    information.to_csv(args.output / "within_episode_information.csv", index=False)

    # ------------------------------------------------------- corpus roll-up
    sealed_alarms = load_npz(SEALED_ALARMS)
    corpus_rows: list[dict[str, Any]] = []
    for cohort, prefix, layer_path, label_path in (
        (
            "development_main",
            "main",
            COHORTS["development_main"].layer_cache,
            sealed.LABEL_ROOT / "development_main_clean_labels.csv",
        ),
        (
            "external_8b",
            "external",
            COHORTS["external_8b"].layer_cache,
            sealed.LABEL_ROOT / "external_8b_clean_labels.csv",
        ),
    ):
        other_layer = load_npz(layer_path)
        other_labels = sealed.aligned_labels(other_layer, label_path, cohort)
        other_risk = other_labels["original_failure"].to_numpy(bool)
        other_alarm = sealed_alarms[f"{prefix}_guard"] >= 0
        corpus_rows.append(
            {
                "cohort": cohort,
                "source": "published sealed v7 alarms",
                "episodes": int(len(other_labels)),
                "risks": int(other_risk.sum()),
                "tp": int((other_alarm & other_risk).sum()),
                "fp": int((other_alarm & ~other_risk).sum()),
            }
        )
    guard_alarm = alarms["guard"] >= 0
    corpus_rows.append(
        {
            "cohort": "legacy_16x32",
            "source": "measured here from raw right-16x32 routes",
            "episodes": int(len(labels)),
            "risks": int(risk.sum()),
            "tp": int((guard_alarm & risk).sum()),
            "fp": int((guard_alarm & ~risk).sum()),
        }
    )
    corpus = pd.DataFrame(corpus_rows)
    total = {
        "cohort": "WHOLE_CORPUS",
        "source": "sum of the three measured cohorts",
        "episodes": int(corpus["episodes"].sum()),
        "risks": int(corpus["risks"].sum()),
        "tp": int(corpus["tp"].sum()),
        "fp": int(corpus["fp"].sum()),
    }
    corpus = pd.concat([corpus, pd.DataFrame([total])], ignore_index=True)
    corpus["precision"] = corpus["tp"] / (corpus["tp"] + corpus["fp"])
    corpus["risk_recall"] = corpus["tp"] / corpus["risks"]
    corpus["timely_fpr"] = corpus["fp"] / (corpus["episodes"] - corpus["risks"])
    corpus.to_csv(args.output / "corpus_total.csv", index=False)

    # ------------------------------------------- composition vs degradation
    # All five 16x32 tasks also appear in development_main and external_8b, so the
    # guard can be compared on exactly the same tasks. Anything left after that is
    # not task composition.
    shared_tasks = sorted(set(labels["task"]))
    comparison_rows: list[dict[str, Any]] = []
    per_task_reference: dict[str, dict[str, dict[str, float]]] = {}
    for cohort, prefix, layer_path, label_path in (
        (
            "development_main",
            "main",
            COHORTS["development_main"].layer_cache,
            sealed.LABEL_ROOT / "development_main_clean_labels.csv",
        ),
        (
            "external_8b",
            "external",
            COHORTS["external_8b"].layer_cache,
            sealed.LABEL_ROOT / "external_8b_clean_labels.csv",
        ),
    ):
        other_layer = load_npz(layer_path)
        other_labels = sealed.aligned_labels(other_layer, label_path, cohort)
        other_alarm = sealed_alarms[f"{prefix}_guard"] >= 0
        other_risk = other_labels["original_failure"].to_numpy(bool)
        take = other_labels["task"].isin(shared_tasks).to_numpy(bool)
        per_task_reference[cohort] = {}
        for task in shared_tasks:
            rows_of_task = other_labels["task"].to_numpy(str) == task
            risk_here = other_risk[rows_of_task]
            alarm_here = other_alarm[rows_of_task]
            per_task_reference[cohort][task] = {
                "risk_recall": ratio(int((alarm_here & risk_here).sum()), risk_here.sum()),
                "timely_fpr": ratio(
                    int((alarm_here & ~risk_here).sum()), int((~risk_here).sum())
                ),
                "risk_rate": float(risk_here.mean()),
            }
        for group, mask in [("shared5_all", take)] + [
            (f"shared5_{name}", take & (other_labels["suite"].to_numpy(str) == name))
            for name in sorted(set(labels["suite"]))
        ]:
            comparison_rows.append(
                {
                    "cohort": cohort,
                    "group": group,
                    "episodes": int(mask.sum()),
                    "risk_n": int((other_risk & mask).sum()),
                    "tp": int((other_alarm & other_risk & mask).sum()),
                    "fp": int((other_alarm & ~other_risk & mask).sum()),
                    "risk_recall": ratio(
                        int((other_alarm & other_risk & mask).sum()), int((other_risk & mask).sum())
                    ),
                    "timely_fpr": ratio(
                        int((other_alarm & ~other_risk & mask).sum()),
                        int((~other_risk & mask).sum()),
                    ),
                    "precision": ratio(
                        int((other_alarm & other_risk & mask).sum()),
                        int((other_alarm & mask).sum()),
                    ),
                    "risk_rate": ratio(int((other_risk & mask).sum()), int(mask.sum())),
                }
            )
    for group, mask in [("shared5_all", np.ones(len(labels), bool))] + [
        (f"shared5_{name}", suite == name) for name in sorted(set(labels["suite"]))
    ]:
        comparison_rows.append(
            {
                "cohort": "legacy_16x32",
                "group": group,
                "episodes": int(mask.sum()),
                "risk_n": int((risk & mask).sum()),
                "tp": int((guard_alarm & risk & mask).sum()),
                "fp": int((guard_alarm & ~risk & mask).sum()),
                "risk_recall": ratio(
                    int((guard_alarm & risk & mask).sum()), int((risk & mask).sum())
                ),
                "timely_fpr": ratio(
                    int((guard_alarm & ~risk & mask).sum()), int((~risk & mask).sum())
                ),
                "precision": ratio(
                    int((guard_alarm & risk & mask).sum()), int((guard_alarm & mask).sum())
                ),
                "risk_rate": ratio(int((risk & mask).sum()), int(mask.sum())),
            }
        )
    # Expected 16x32 counts if the guard behaved on each 16x32 task exactly as it does
    # on the same task elsewhere: composition held fixed, per-task behaviour imported.
    for cohort, reference in per_task_reference.items():
        expected_tp = 0.0
        expected_fp = 0.0
        for task in shared_tasks:
            rows_of_task = labels["task"].to_numpy(str) == task
            risk_here = int((risk & rows_of_task).sum())
            timely_here = int((~risk & rows_of_task).sum())
            recall = reference[task]["risk_recall"]
            fpr = reference[task]["timely_fpr"]
            # A task with no risk episodes in the reference cohort has an undefined
            # recall; it also contributes no expected true positives here.
            expected_tp += 0.0 if (risk_here == 0 or not math.isfinite(recall)) else recall * risk_here
            expected_fp += 0.0 if (timely_here == 0 or not math.isfinite(fpr)) else fpr * timely_here
        comparison_rows.append(
            {
                "cohort": f"legacy_16x32_expected_from_{cohort}_per_task_rates",
                "group": "shared5_all",
                "episodes": int(len(labels)),
                "risk_n": int(risk.sum()),
                "tp": float(expected_tp),
                "fp": float(expected_fp),
                "risk_recall": float(expected_tp / risk.sum()),
                "timely_fpr": float(expected_fp / int((~risk).sum())),
                "precision": float(expected_tp / (expected_tp + expected_fp)),
                "risk_rate": float(risk.mean()),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.output / "cohort_comparison_shared_tasks.csv", index=False)

    episode_frame = labels.copy()
    for detector in BRANCHES:
        episode_frame[f"first_{detector}_query"] = alarms[detector]
    episode_frame["horizon_cap"] = horizon
    episode_frame["budget_left_at_guard"] = horizon - 1 - np.asarray(alarms["guard"], dtype=int)
    episode_frame.to_csv(args.output / "episode_alarms.csv", index=False)

    chunk_rows: list[dict[str, Any]] = []
    guard_first = np.asarray(alarms["guard"], dtype=int)
    for name in sorted(set(suite)):
        take = (suite == name) & (guard_first >= 0)
        chunks = guard_first[take]
        chunk_rows.append(
            {
                "suite": name,
                "horizon_cap": horizon_by_suite[name],
                "alarms": int(take.sum()),
                "first_chunk_prior_ge_0.25": next(
                    (
                        chunk
                        for chunk in sorted(priors["suite"][name])
                        if priors["suite"][name][chunk] >= 0.25
                    ),
                    None,
                ),
                "alarm_chunk_min": int(chunks.min()) if len(chunks) else None,
                "alarm_chunk_p25": float(np.quantile(chunks, 0.25)) if len(chunks) else None,
                "alarm_chunk_median": float(np.median(chunks)) if len(chunks) else None,
                "alarm_chunk_max": int(chunks.max()) if len(chunks) else None,
                "alarms_before_prior_crosses_0.25": int(
                    (
                        chunks
                        < next(
                            (
                                chunk
                                for chunk in sorted(priors["suite"][name])
                                if priors["suite"][name][chunk] >= 0.25
                            ),
                            0,
                        )
                    ).sum()
                ),
            }
        )
    pd.DataFrame(chunk_rows).to_csv(args.output / "alarm_chunk_distribution.csv", index=False)

    primary = metrics[(metrics["detector"] == "guard") & (metrics["group"] == "all")].iloc[0]
    early_band = survival_table[
        (survival_table["detector"] == "guard")
        & (survival_table["group"] == "prior[0.00,0.25)")
    ]
    fully_raw_alarm = alarms_fully_raw["guard"] >= 0
    summary = {
        "schema": "himoe.legacy16x32.evaluation.v1",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "cohort": "legacy_16x32",
        "run_id": "right-16x32",
        "recalibrated_on_this_cohort": False,
        "operating_point_reselected": False,
        "frozen_point": point.as_dict(),
        "risk_target": "raw original-horizon failure (no plus-10 continuation study exists for 16x32)",
        "horizon_by_suite": horizon_by_suite,
        "horizon_cap_verified": True,
        "episodes": int(len(labels)),
        "risks": int(risk.sum()),
        "guard": {
            "tp": int(primary["tp"]),
            "fp": int(primary["fp"]),
            "precision": float(primary["precision"]),
            "risk_recall": float(primary["risk_recall"]),
            "timely_fpr": float(primary["timely_fpr"]),
        },
        "guard_fully_raw_mobility_variant": {
            "tp": int((fully_raw_alarm & risk).sum()),
            "fp": int((fully_raw_alarm & ~risk).sum()),
            "episodes_differing_from_primary": int(
                (alarms_fully_raw["guard"] != alarms["guard"]).sum()
            ),
        },
        "task_matched_survival_control_max_lift_deviation": control_max_deviation,
        "task_identity_control": plain(
            survival_table[
                (survival_table["detector"] == "control_task_identity_at_chunk0")
                & (survival_table["group"] == "all")
            ].to_dict(orient="records")
        ),
        "early_band_alarms": plain(early_band.to_dict(orient="records")),
        "corpus_total": plain(corpus.to_dict(orient="records")),
        "cohort_comparison_shared_tasks": plain(comparison.to_dict(orient="records")),
        "artifacts": {
            "outcome_metrics_sha256": sha256(args.output / "outcome_metrics.csv"),
            "survival_matched_sha256": sha256(args.output / "survival_matched.csv"),
            "survival_prior_curves_sha256": sha256(args.output / "survival_prior_curves.csv"),
            "cohort_comparison_sha256": sha256(
                args.output / "cohort_comparison_shared_tasks.csv"
            ),
            "budget_retention_sha256": sha256(args.output / "budget_retention.csv"),
            "within_episode_information_sha256": sha256(
                args.output / "within_episode_information.csv"
            ),
            "corpus_total_sha256": sha256(args.output / "corpus_total.csv"),
            "episode_alarms_sha256": sha256(args.output / "episode_alarms.csv"),
            "first_alarms_sha256": sha256(args.output / "legacy16x32_first_alarms.npz"),
            "evaluator_sha256": sha256(Path(__file__)),
        },
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        metrics[(metrics["group"] == "all")][
            ["detector", "tp", "fp", "risk_recall", "timely_fpr", "precision", "early4_risk_recall"]
        ].to_string(index=False),
        flush=True,
    )
    print()
    print(corpus.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
