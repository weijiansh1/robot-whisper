#!/usr/bin/env python3
"""Cross-corpus Bayesian calibration of route-only near-Trap probabilities.

There is no gradient fit and no learned feature weighting. Route states are
materialized before onset labels are loaded. Source onset labels then estimate
Beta-Binomial state risks; target labels are joined only after probability files
have been written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import rankdata

import analyze_trainfree_signal_matrix as signal_matrix


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/trainfree_trap_probability.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/trainfree_trap_probability"
SCHEMA = "himoe.trainfree_trap_probability.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def robust_self_normalize(
    values: np.ndarray,
    baseline_start: int,
    baseline_stop: int,
    ratio: bool,
) -> np.ndarray:
    result = np.full(values.shape, np.nan, np.float64)
    for episode in range(len(values)):
        baseline = np.asarray(
            values[episode, baseline_start:baseline_stop], np.float64
        )
        baseline = baseline[np.isfinite(baseline)]
        if len(baseline) < 4:
            continue
        median = float(np.median(baseline))
        if ratio:
            result[episode] = values[episode] / max(abs(median), 1e-8)
        else:
            mad = float(np.median(np.abs(baseline - median)) * 1.4826)
            scale = max(mad, abs(median) * 0.05, 1e-5)
            result[episode] = (values[episode] - median) / scale
    return result


def materialize_route_only(tag: str, config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Create label-free per-query components; read no success/onset/physics fields."""
    cache = signal_matrix.SSM_CACHE
    meta = pd.read_csv(cache / f"{tag}_meta.csv", usecols=["episode_id", "T"])
    rowidx = np.load(cache / f"{tag}_rowidx.npy")
    valid = rowidx >= 0
    row_features, feature_audit = signal_matrix.extract_row_features(tag, False)
    row_series = {
        name: signal_matrix.scatter_rows(values, rowidx)
        for name, values in row_features.items()
    }
    temporal = signal_matrix.route_temporal_features(tag, rowidx, valid)
    signals = {
        "late_flow_volatility_w4": signal_matrix.rolling_mean(
            row_series["late_flow_volatility"], signal_matrix.ALIGN_WINDOW
        ),
        "route_acceleration_w4": signal_matrix.rolling_mean(
            row_series["route_acceleration"], signal_matrix.ALIGN_WINDOW
        ),
        "lag_periodicity_w4": temporal["lag_periodicity_w4"],
    }
    baseline_start, baseline_stop = map(int, config["self_reference_queries"])
    normalized = {
        "late_flow": robust_self_normalize(
            signals["late_flow_volatility_w4"], baseline_start, baseline_stop, True
        ),
        "acceleration": robust_self_normalize(
            signals["route_acceleration_w4"], baseline_start, baseline_stop, True
        ),
        "periodicity": robust_self_normalize(
            signals["lag_periodicity_w4"], baseline_start, baseline_stop, False
        ),
    }
    artifact = output / "intermediate" / f"corpus_{tag}_route_only.npz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        schema=np.asarray(SCHEMA),
        training=np.asarray(False),
        labels_loaded=np.asarray(False),
        episode_id=meta.episode_id.to_numpy(np.int64),
        length=meta["T"].to_numpy(np.int16),
        valid=valid,
        **normalized,
    )
    return {
        "tag": tag,
        "artifact": str(artifact),
        "sha256": sha256_file(artifact),
        "episodes": len(meta),
        "queries": int(valid.sum()),
        "feature_cache_reused": bool(feature_audit["cache_reused"]),
        "state_flow_max_deviation": float(feature_audit["state_flow_max_deviation"]),
    }


def load_route_only(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        assert str(archive["schema"].item()) == SCHEMA
        assert not bool(archive["training"])
        assert not bool(archive["labels_loaded"])
        return {name: np.asarray(archive[name]) for name in archive.files}


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.sort(reference[np.isfinite(reference)])
    if not len(finite):
        raise ValueError("empty percentile reference")
    result = np.full(values.shape, np.nan, np.float64)
    good = np.isfinite(values)
    result[good] = np.searchsorted(finite, values[good], side="right") / len(finite)
    return result


def phenotype_score(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    minimum_query: int,
) -> dict[str, np.ndarray]:
    reference_mask = source["valid"].copy()
    reference_mask[:, :minimum_query] = False
    percentiles = {
        name: empirical_percentile(source[name][reference_mask], target[name])
        for name in ("late_flow", "acceleration", "periodicity")
    }
    loop = np.minimum(percentiles["late_flow"], percentiles["acceleration"])
    static = percentiles["periodicity"]
    return {"loop": loop, "static": static, "trap": np.maximum(loop, static)}


def persistence(score: np.ndarray, valid: np.ndarray, threshold: float, cap: int) -> np.ndarray:
    result = np.zeros(score.shape, np.int8)
    for episode in range(len(score)):
        run = 0
        for query in range(score.shape[1]):
            if valid[episode, query] and np.isfinite(score[episode, query]):
                run = run + 1 if score[episode, query] >= threshold else 0
                result[episode, query] = min(run, cap)
            else:
                run = 0
    return result


def materialize_direction_states(
    source_tag: str,
    target_tag: str,
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    minimum_query = int(config["minimum_query"])
    threshold = float(config["persistence_score_threshold"])
    cap = int(config["persistence_cap"])
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA),
        "training": np.asarray(False),
        "labels_loaded": np.asarray(False),
    }
    for role, data in (("source", source), ("target", target)):
        arrays[f"{role}_episode_id"] = data["episode_id"]
        arrays[f"{role}_length"] = data["length"]
        arrays[f"{role}_valid"] = data["valid"]
        scores = phenotype_score(source, data, minimum_query)
        for event_kind, values in scores.items():
            arrays[f"{role}_{event_kind}_score"] = values.astype(np.float32)
            arrays[f"{role}_{event_kind}_persistence"] = persistence(
                values, data["valid"], threshold, cap
            )
    path = output / "intermediate" / f"states_{source_tag}_to_{target_tag}.npz"
    np.savez_compressed(path, **arrays)
    return {
        "source": source_tag,
        "target": target_tag,
        "artifact": str(path),
        "sha256": sha256_file(path),
        "labels_loaded": False,
    }


def event_onset(corpus: signal_matrix.Corpus, event_kind: str) -> np.ndarray:
    return {
        "loop": corpus.loop_onset,
        "static": corpus.static_onset,
        "trap": corpus.trap_onset,
    }[event_kind]


def score_bin(score: float, edges: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2))


def eligible_rows(
    lengths: np.ndarray,
    valid: np.ndarray,
    score: np.ndarray,
    run: np.ndarray,
    minimum_query: int,
    onset: np.ndarray | None = None,
    horizon: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in range(len(lengths)):
        stop = int(lengths[episode])
        if onset is not None and int(onset[episode]) >= 0:
            stop = min(stop, int(onset[episode]) + 1)
        for query in range(minimum_query, stop):
            if not valid[episode, query] or not np.isfinite(score[episode, query]):
                continue
            row: dict[str, Any] = {
                "episode_index": episode,
                "query": query,
                "score": float(score[episode, query]),
                "persistence": int(run[episode, query]),
            }
            if onset is not None and horizon is not None:
                delta = int(onset[episode]) - query
                row["label"] = int(int(onset[episode]) >= 0 and 0 <= delta <= horizon)
                row["onset"] = int(onset[episode])
                row["delta_to_onset"] = delta if int(onset[episode]) >= 0 else -999
            rows.append(row)
    return rows


def posterior(k: int, n: int, alpha: float, beta: float, interval: float) -> tuple[float, float, float]:
    a = k + alpha
    b = n - k + beta
    tail = (1.0 - interval) / 2.0
    return (
        float(a / (a + b)),
        float(beta_distribution.ppf(tail, a, b)),
        float(beta_distribution.ppf(1.0 - tail, a, b)),
    )


def fit_probability_table(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    edges = np.asarray(config["score_bins"], np.float64)
    minimum_support = int(config["minimum_probability_cell_support"])
    alpha = float(config["beta_prior"]["alpha"])
    beta = float(config["beta_prior"]["beta"])
    interval = float(config["confidence_interval"])
    detailed: dict[tuple[int, int], list[int]] = {}
    by_bin: dict[int, list[int]] = {}
    total = [0, 0]
    for row in rows:
        key = (score_bin(row["score"], edges), int(row["persistence"]))
        for store, item in ((detailed, key), (by_bin, key[0])):
            counts = store.setdefault(item, [0, 0])
            counts[0] += int(row["label"])
            counts[1] += 1
        total[0] += int(row["label"])
        total[1] += 1

    mapping: dict[tuple[int, int], dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []
    all_keys = [
        (bin_index, run)
        for bin_index in range(len(edges) - 1)
        for run in range(int(config["persistence_cap"]) + 1)
    ]
    for key in all_keys:
        k, n = detailed.get(key, [0, 0])
        if n >= minimum_support:
            used_k, used_n, level = k, n, "score_bin_x_persistence"
        elif by_bin[key[0]][1] >= minimum_support:
            used_k, used_n = by_bin[key[0]]
            level = "score_bin_backoff"
        else:
            used_k, used_n = total
            level = "global_backoff"
        mean, low, high = posterior(used_k, used_n, alpha, beta, interval)
        mapping[key] = {
            "probability": mean,
            "ci_low": low,
            "ci_high": high,
            "support": used_n,
            "events": used_k,
            "level": level,
        }
        table_rows.append(
            {
                "score_bin": key[0],
                "score_low": edges[key[0]],
                "score_high": edges[key[0] + 1],
                "persistence": key[1],
                "raw_events": k,
                "raw_support": n,
                "mapping_level": level,
                "mapped_events": used_k,
                "mapped_support": used_n,
                "posterior_mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    global_mean, global_low, global_high = posterior(
        total[0], total[1], alpha, beta, interval
    )
    return mapping, table_rows, {
        "events": total[0],
        "support": total[1],
        "posterior_mean": global_mean,
        "ci_low": global_low,
        "ci_high": global_high,
    }


def apply_probability_table(
    rows: list[dict[str, Any]],
    mapping: dict[tuple[int, int], dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    edges = np.asarray(config["score_bins"], np.float64)
    fallback = max(mapping.values(), key=lambda item: item["support"])
    output: list[dict[str, Any]] = []
    for row in rows:
        key = (score_bin(row["score"], edges), int(row["persistence"]))
        estimate = mapping.get(key, fallback)
        output.append(
            {
                **row,
                "score_bin": key[0],
                "probability": estimate["probability"],
                "probability_ci_low": estimate["ci_low"],
                "probability_ci_high": estimate["ci_high"],
                "calibration_support": estimate["support"],
                "calibration_events": estimate["events"],
                "mapping_level": estimate["level"],
            }
        )
    return output


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, np.int8)
    probability = np.clip(np.asarray(probability, np.float64), 1e-9, 1.0 - 1e-9)
    if labels.min() == labels.max():
        auc = float("nan")
        average_precision = float(labels.mean())
    else:
        ranks = rankdata(probability, method="average")
        positive = labels == 1
        n_positive = int(positive.sum())
        n_negative = len(labels) - n_positive
        auc = float(
            (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
            / (n_positive * n_negative)
        )
        order = np.argsort(-probability, kind="stable")
        sorted_labels = labels[order]
        precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
        average_precision = float(
            (precision * sorted_labels).sum() / max(n_positive, 1)
        )
    calibration_error = 0.0
    rounded = np.round(probability, 12)
    for value in np.unique(rounded):
        selected = rounded == value
        calibration_error += float(selected.mean()) * abs(
            float(labels[selected].mean()) - float(probability[selected].mean())
        )
    return {
        "prevalence": float(labels.mean()),
        "brier": float(np.mean((probability - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))
        ),
        "roc_auc": auc,
        "average_precision": average_precision,
        "expected_calibration_error": calibration_error,
    }


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    onset: np.ndarray,
    horizon: int,
    alarm_probability: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluated: list[dict[str, Any]] = []
    for row in predictions:
        episode = int(row["episode_index"])
        event_onset_value = int(onset[episode])
        query = int(row["query"])
        if event_onset_value >= 0 and query > event_onset_value:
            continue
        delta = event_onset_value - query
        evaluated.append(
            {
                **row,
                "label": int(event_onset_value >= 0 and 0 <= delta <= horizon),
                "onset": event_onset_value,
                "delta_to_onset": delta if event_onset_value >= 0 else -999,
                "alarm_075": int(float(row["probability"]) >= alarm_probability),
                "confidence_alarm_075": int(
                    float(row["probability_ci_low"]) >= alarm_probability
                ),
            }
        )
    labels = np.asarray([row["label"] for row in evaluated], np.int8)
    probability = np.asarray([row["probability"] for row in evaluated], np.float64)
    metrics = probability_metrics(labels, probability)
    alarm = probability >= alarm_probability
    tp = int(labels[alarm].sum())
    alarms = int(alarm.sum())

    episode_rows: list[dict[str, Any]] = []
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in evaluated:
        by_episode.setdefault(int(row["episode_index"]), []).append(row)
    event_episodes = 0
    timely_caught = 0
    no_event_episodes = 0
    no_event_false_alarms = 0
    for episode, rows in sorted(by_episode.items()):
        event = int(onset[episode]) >= 0
        alarms_here = [row for row in rows if row["alarm_075"]]
        timely = [row for row in alarms_here if row["label"]]
        if event:
            event_episodes += 1
            timely_caught += int(bool(timely))
        else:
            no_event_episodes += 1
            no_event_false_alarms += int(bool(alarms_here))
        episode_rows.append(
            {
                "episode_index": episode,
                "event": int(event),
                "onset": int(onset[episode]),
                "alarm_count": len(alarms_here),
                "timely_alarm_count": len(timely),
                "first_alarm_query": min((int(row["query"]) for row in alarms_here), default=-1),
                "max_probability": max(float(row["probability"]) for row in rows),
            }
        )
    metrics.update(
        {
            "rows": len(evaluated),
            "positive_rows": int(labels.sum()),
            "alarm_rows_075": alarms,
            "alarm_true_rows_075": tp,
            "alarm_precision_075": float(tp / alarms) if alarms else None,
            "positive_row_recall_075": float(tp / max(labels.sum(), 1)),
            "max_predicted_probability": float(probability.max()),
            "max_probability_target_support": int(
                (probability == probability.max()).sum()
            ),
            "max_probability_target_events": int(
                labels[probability == probability.max()].sum()
            ),
            "max_probability_target_observed_rate": float(
                labels[probability == probability.max()].mean()
            ),
            "event_episodes_evaluable": event_episodes,
            "event_episode_timely_alarms_075": timely_caught,
            "event_episode_recall_075": float(timely_caught / max(event_episodes, 1)),
            "no_event_episodes": no_event_episodes,
            "no_event_episode_false_alarms_075": no_event_false_alarms,
            "no_event_episode_fpr_075": float(
                no_event_false_alarms / max(no_event_episodes, 1)
            ),
        }
    )
    return metrics, evaluated, episode_rows


def absorbing_rows(
    rows: list[dict[str, Any]], onset: np.ndarray, horizon: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        episode = int(row["episode_index"])
        query = int(row["query"])
        event_onset_value = int(onset[episode])
        delta = event_onset_value - query
        output.append(
            {
                **row,
                "label": int(event_onset_value >= 0 and query >= event_onset_value - horizon),
                "onset": event_onset_value,
                "delta_to_onset": delta if event_onset_value >= 0 else -999,
            }
        )
    return output


def evaluate_absorbing_predictions(
    predictions: list[dict[str, Any]],
    onset: np.ndarray,
    horizon: int,
    alarm_probability: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluated = absorbing_rows(predictions, onset, horizon)
    labels = np.asarray([row["label"] for row in evaluated], np.int8)
    probability = np.asarray([row["probability"] for row in evaluated], np.float64)
    alarm = probability >= alarm_probability
    confidence_alarm = np.asarray(
        [row["probability_ci_low"] for row in evaluated], np.float64
    ) >= alarm_probability
    metrics = probability_metrics(labels, probability)
    maximum = float(probability.max())
    maximum_rows = probability == maximum
    n_alarm = int(alarm.sum())
    n_true_alarm = int(labels[alarm].sum())

    episodes = np.asarray([row["episode_index"] for row in evaluated], np.int64)
    present_episode_ids = set(episodes)
    event_episode_ids = {
        episode for episode in present_episode_ids if int(onset[episode]) >= 0
    }
    no_event_episode_ids = {
        episode for episode in present_episode_ids if int(onset[episode]) < 0
    }
    caught = set(episodes[alarm & (labels == 1)])
    false_alarm = set(episodes[alarm & np.asarray(onset[episodes] < 0)])
    metrics.update(
        {
            "rows": len(evaluated),
            "positive_rows": int(labels.sum()),
            "alarm_rows_075": n_alarm,
            "alarm_true_rows_075": n_true_alarm,
            "alarm_precision_075": float(n_true_alarm / n_alarm) if n_alarm else None,
            "positive_row_recall_075": float(n_true_alarm / max(labels.sum(), 1)),
            "confidence_alarm_rows_075": int(confidence_alarm.sum()),
            "max_predicted_probability": maximum,
            "max_probability_target_support": int(maximum_rows.sum()),
            "max_probability_target_events": int(labels[maximum_rows].sum()),
            "max_probability_target_observed_rate": float(labels[maximum_rows].mean()),
            "event_episodes_evaluable": len(event_episode_ids),
            "event_episode_alarms_075": len(caught),
            "event_episode_recall_075": float(
                len(caught) / max(len(event_episode_ids), 1)
            ),
            "no_event_episodes": len(no_event_episode_ids),
            "no_event_episode_false_alarms_075": len(false_alarm),
            "no_event_episode_fpr_075": float(
                len(false_alarm) / max(len(no_event_episode_ids), 1)
            ),
        }
    )
    for row, alarm_value, confidence_value in zip(
        evaluated, alarm, confidence_alarm
    ):
        row["alarm_075"] = int(alarm_value)
        row["confidence_alarm_075"] = int(confidence_value)
    return metrics, evaluated


def threshold_sweep_rows(
    evaluated: list[dict[str, Any]],
    onset: np.ndarray,
    thresholds: list[float],
    direction: str,
) -> list[dict[str, Any]]:
    labels = np.asarray([row["label"] for row in evaluated], np.int8)
    probability = np.asarray([row["probability"] for row in evaluated], np.float64)
    episodes = np.asarray([row["episode_index"] for row in evaluated], np.int64)
    event_episode_ids = set(episodes[np.asarray([row["onset"] >= 0 for row in evaluated])])
    no_event_episode_ids = set(episodes[np.asarray([row["onset"] < 0 for row in evaluated])])
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        alarm = probability >= threshold
        true_alarm = alarm & (labels == 1)
        alarm_event_ids = set(episodes[true_alarm])
        false_alarm_episode_ids = set(
            episodes[alarm & np.asarray([row["onset"] < 0 for row in evaluated])]
        )
        n_alarm = int(alarm.sum())
        n_true = int(true_alarm.sum())
        rows.append(
            {
                "direction": direction,
                "probability_threshold": threshold,
                "alarm_rows": n_alarm,
                "true_alarm_rows": n_true,
                "alarm_precision": float(n_true / n_alarm) if n_alarm else None,
                "positive_row_recall": float(n_true / max(labels.sum(), 1)),
                "event_episode_timely_recall": float(
                    len(alarm_event_ids) / max(len(event_episode_ids), 1)
                ),
                "no_event_episode_fpr": float(
                    len(false_alarm_episode_ids) / max(len(no_event_episode_ids), 1)
                ),
            }
        )
    return rows


def clock_predictions(
    calibration_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    width = 4
    bins: dict[int, list[int]] = {}
    total = [0, 0]
    for row in calibration_rows:
        key = int(row["query"]) // width
        value = bins.setdefault(key, [0, 0])
        value[0] += int(row["label"])
        value[1] += 1
        total[0] += int(row["label"])
        total[1] += 1
    alpha = float(config["beta_prior"]["alpha"])
    beta = float(config["beta_prior"]["beta"])
    interval = float(config["confidence_interval"])
    output = []
    for row in target_rows:
        k, n = bins.get(int(row["query"]) // width, total)
        mean, low, high = posterior(k, n, alpha, beta, interval)
        output.append({**row, "probability": mean, "probability_ci_low": low, "probability_ci_high": high})
    return output


def reliability_rows(
    evaluated: list[dict[str, Any]], direction: str, event_kind: str, horizon: int
) -> list[dict[str, Any]]:
    groups: dict[float, list[int]] = {}
    for row in evaluated:
        groups.setdefault(round(float(row["probability"]), 12), []).append(
            int(row["label"])
        )
    return [
        {
            "direction": direction,
            "event_kind": event_kind,
            "horizon": horizon,
            "predicted_probability": probability,
            "observed_probability": float(np.mean(labels)),
            "target_events": int(np.sum(labels)),
            "target_support": len(labels),
            "absolute_calibration_error": abs(probability - float(np.mean(labels))),
        }
        for probability, labels in sorted(groups.items())
    ]


def plot_results(evaluations: pd.DataFrame, reliability: pd.DataFrame, output: Path) -> None:
    primary = evaluations[
        (evaluations.event_kind == "trap") & (evaluations.horizon == 2)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    colors = {"A_to_B": "#146c94", "B_to_A": "#b24c3d"}
    for direction in ("A_to_B", "B_to_A"):
        data = reliability[
            (reliability.direction == direction)
            & (reliability.event_kind == "trap")
            & (reliability.horizon == 2)
        ]
        size = np.clip(data.target_support.to_numpy() * 0.4, 18, 180)
        axes[0].scatter(
            data.predicted_probability,
            data.observed_probability,
            s=size,
            alpha=0.72,
            color=colors[direction],
            label=direction.replace("_", " "),
        )
    axes[0].plot([0, 0.75], [0, 0.75], color="#555555", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Predicted probability", ylabel="Observed target rate", title="Cross-corpus reliability")
    axes[0].legend(frameon=False)

    x = np.arange(len(primary))
    axes[1].bar(x, primary.max_predicted_probability, color=[colors[x] for x in primary.direction])
    axes[1].axhline(0.75, color="#202020", linestyle="--", linewidth=1.2, label="alarm = 0.75")
    axes[1].set_xticks(x, [x.replace("_", "\n") for x in primary.direction])
    axes[1].set(ylabel="Maximum calibrated probability", ylim=(0, 0.8), title="No onset-hazard state reaches 75%")
    axes[1].legend(frameon=False)

    width = 0.34
    axes[2].bar(x - width / 2, primary.average_precision, width, color="#3f7d5d", label="MoE probability")
    axes[2].bar(x + width / 2, primary.prevalence, width, color="#b8b8b8", label="target prevalence")
    axes[2].set_xticks(x, [x.replace("_", "\n") for x in primary.direction])
    axes[2].set(ylabel="Average precision", title="Ranking gain is limited")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    path = output / "figures/trainfree_trap_probability.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report(summary: dict[str, Any]) -> str:
    primary = summary["primary_results"]
    lines = [
        "# Train-free Trap 概率校准实验",
        "",
        "## 核心结论",
        "",
        "把 MoE 异常分数改成显式概率是正确的设计方向，但当前 A/B 路由 phenotype **没有支持 75% 报警的概率状态**。该结论来自 A→B 与 B→A 的跨语料测试，不是在同一语料上回报拟合值。",
        "",
        "主目标是 `当前处于 onset 或未来 2 个 query 内进入 loop/static Trap`。路由状态只使用 HB late-flow volatility、route acceleration 和 lag periodicity，并先相对每条 episode 的早期前缀归一化。概率由 source onset 的 Jeffreys Beta-Binomial 频率校准；无梯度、无特征权重学习，但明确使用了 source 标签做 calibration。",
        "",
        "| 方向 | target 基率 | 最大预测 / 该状态实测 | AUROC | AP | Brier / 常数基线 | 75% 报警 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for direction in ("A_to_B", "B_to_A"):
        row = primary[direction]
        lines.append(
            f"| {direction.replace('_to_', '→')} | {row['prevalence']:.3%} | "
            f"{row['max_predicted_probability']:.3%} / "
            f"{row['max_probability_target_observed_rate']:.3%} | "
            f"{row['roc_auc']:.3f} | {row['average_precision']:.3f} | "
            f"{row['brier']:.4f} / {row['constant_source_prior_brier']:.4f} | "
            f"{row['alarm_rows_075']} |"
        )
    lines.extend(
        [
            "",
            "最高 1% 自归一化综合分数加连续两次的探索性检查也只有约 18--22% 的跨语料两步发生率。离散 16-state macro-state 的最高有支持风险同样低于 75%，且多个高风险状态在另一语料中反转。正式概率表因此没有为了制造报警继续扫描阈值。",
            "",
            "## 解释",
            "",
            "75% 是一个 precision-first 要求。现有 MoE 信号能做一定排序，但不能把某个路由状态压缩成跨 capture 稳定的高置信 Trap 概率。A/B 的事件基率和路由尺度不同，导致 source 中的条件频率不能直接搬到 target。概率化没有解决低召回，反而诚实地暴露了证据不足。",
            "",
            "最高预测状态也没有正确迁移：A→B 预测约 35.8% 的状态在 B 中实测约 21.4%；B→A 预测约 51.3% 的状态在 A 中实测约 15.8%。A→B 的 Brier 甚至差于只输出 source 常数基率，B→A 仅略好。因此当前数值不能作为跨 capture 校准概率部署。",
            "",
            "把 onset 后直到 episode 结束全部视为 Trap 的宽松 absorbing proxy 也没有形成可靠的 75% detector：A→B/H2 的最大概率为 72.6%，仍为零报警；B→A/H2 虽产生 47 个 75% 报警，但 target 实测只有 31/47=66.0%，低于它声称的置信度。这个代理还会把已经恢复的时刻继续标正，因此只能作为检测上限压力测试。",
            "",
            "这不否定 `loop lead=-2` 的组级统计复现：组级均值差异可以显著，但单个 query 的 posterior 仍远低于 75%。前者回答“总体是否有信号”，后者回答“这一时刻是否足以报警”，两者不是同一问题。",
            "",
            "## 方法边界",
            "",
            "- A 使用稠密物理 onset，B 使用在 A 上验证过的 query proxy，二者并非同等级真值。",
            "- 只评价 onset 前风险集；已经陷入 Trap 后的 active-duration 标签尚不可用。",
            "- 输入噪声反事实 head 只有 1+7 个案，未纳入概率表。",
            "- 530 个 held-out failure 是 endpoint outcome，不等同于带 onset 的 Trap，不能拿来校准两步概率。",
            "- 输出概率的含义依赖部署基率；新任务或新 capture 分布仍需独立 calibration audit。",
            "",
            "## 下一步检测设计",
            "",
            "保留概率接口，但暂时不要声称 `P>75%` detector 已成立。要达到该标准，需要增加新的、互补的 MoE phenotype head，并在带物理 onset 的新任务上做 held-out calibration。建议同时输出 `P(now)`、`P(within 2)`、`P(within 4)`；正式报警仍可固定 0.75，但允许结果为零报警，直到证据真正足够。",
            "",
        ]
    )
    absorbing_h2 = {
        row["direction"]: row
        for row in summary["absorbing_detection_results"]
        if int(row["horizon"]) == 2
    }
    lines.extend(
        [
            "## 阈值与时钟审计",
            "",
            "宽松 absorbing proxy 上，相位时钟对照仍明显强于 MoE：A→B 的 AUROC 为 "
            f"{absorbing_h2['A_to_B']['clock_roc_auc']:.3f} vs. "
            f"{absorbing_h2['A_to_B']['roc_auc']:.3f}，B→A 为 "
            f"{absorbing_h2['B_to_A']['clock_roc_auc']:.3f} vs. "
            f"{absorbing_h2['B_to_A']['roc_auc']:.3f}。因此 onset 后整段检测会被 episode phase 严重混淆，不能作为 MoE 特异性证据。",
            "",
            "主 onset-hazard 目标的阈值取舍如下；这些阈值只做预先声明的描述性审计，没有据此回调概率表：",
            "",
            "| 方向 | 阈值 | 报警行精度 | 正例行召回 | 事件 episode 及时召回 | 无事件 episode 误报率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for direction in ("A_to_B", "B_to_A"):
        for threshold in (0.25, 0.5, 0.75):
            row = next(
                item
                for item in summary["primary_threshold_sweep"]
                if item["direction"] == direction
                and np.isclose(item["probability_threshold"], threshold)
            )
            precision = (
                "--"
                if row["alarm_precision"] is None
                else f"{row['alarm_precision']:.1%}"
            )
            lines.append(
                f"| {direction.replace('_to_', '→')} | {threshold:.0%} | "
                f"{precision} | {row['positive_row_recall']:.1%} | "
                f"{row['event_episode_timely_recall']:.1%} | "
                f"{row['no_event_episode_fpr']:.1%} |"
            )
    lines.extend(
        [
            "",
            "把阈值降到 25% 也不能形成可部署规则：A→B 虽覆盖 21.2% 的事件 episode，但无事件 episode 误报率达 23.8%；B→A 仅覆盖 6.7%，报警精度为 16.0%。问题不只是阈值太高，而是现有状态的条件概率没有跨语料稳定迁移。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)

    # Phase 1: route-only materialization. No onset, outcome, action or physics is loaded.
    route_manifests = {
        tag: materialize_route_only(tag, config, output) for tag in ("A", "B")
    }
    routes = {
        tag: load_route_only(Path(route_manifests[tag]["artifact"])) for tag in ("A", "B")
    }
    direction_manifests = []
    for source_tag, target_tag in (("A", "B"), ("B", "A")):
        direction_manifests.append(
            materialize_direction_states(
                source_tag,
                target_tag,
                routes[source_tag],
                routes[target_tag],
                config,
                output,
            )
        )

    # Phase 2: labels are now loaded for source calibration and target evaluation.
    corpora = {tag: signal_matrix.load_corpus(tag)[0] for tag in ("A", "B")}
    for tag in ("A", "B"):
        assert np.array_equal(
            routes[tag]["episode_id"], corpora[tag].meta.episode_id.to_numpy(np.int64)
        )

    evaluation_rows: list[dict[str, Any]] = []
    calibration_rows_all: list[dict[str, Any]] = []
    reliability_all: list[dict[str, Any]] = []
    primary_predictions_all: list[dict[str, Any]] = []
    primary_episode_all: list[dict[str, Any]] = []
    primary_threshold_sweep: list[dict[str, Any]] = []
    primary_results: dict[str, Any] = {}
    label_free_prediction_files: list[dict[str, Any]] = []
    absorbing_evaluation_rows: list[dict[str, Any]] = []
    absorbing_calibration_rows: list[dict[str, Any]] = []
    absorbing_reliability_rows: list[dict[str, Any]] = []
    alarm_probability = float(config["alarm_probability"])

    for source_tag, target_tag in (("A", "B"), ("B", "A")):
        direction = f"{source_tag}_to_{target_tag}"
        state_path = output / "intermediate" / f"states_{source_tag}_to_{target_tag}.npz"
        with np.load(state_path) as states:
            for event_kind in config["event_types"]:
                source_score = np.asarray(states[f"source_{event_kind}_score"])
                source_run = np.asarray(states[f"source_{event_kind}_persistence"])
                target_score = np.asarray(states[f"target_{event_kind}_score"])
                target_run = np.asarray(states[f"target_{event_kind}_persistence"])
                for horizon in map(int, config["horizons"]):
                    source_onset = event_onset(corpora[source_tag], event_kind)
                    target_onset = event_onset(corpora[target_tag], event_kind)
                    calibration_rows = eligible_rows(
                        routes[source_tag]["length"],
                        routes[source_tag]["valid"],
                        source_score,
                        source_run,
                        int(config["minimum_query"]),
                        source_onset,
                        horizon,
                    )
                    mapping, cells, global_probability = fit_probability_table(
                        calibration_rows, config
                    )
                    for row in cells:
                        calibration_rows_all.append(
                            {
                                "direction": direction,
                                "event_kind": event_kind,
                                "horizon": horizon,
                                **row,
                            }
                        )

                    target_unlabeled = eligible_rows(
                        routes[target_tag]["length"],
                        routes[target_tag]["valid"],
                        target_score,
                        target_run,
                        int(config["minimum_query"]),
                    )
                    predictions = apply_probability_table(
                        target_unlabeled, mapping, config
                    )
                    for row in predictions:
                        row["episode_id"] = int(
                            routes[target_tag]["episode_id"][int(row["episode_index"])]
                        )
                    prediction_path = (
                        output
                        / "predictions_label_free"
                        / f"{direction}_{event_kind}_h{horizon}.csv"
                    )
                    write_csv(prediction_path, predictions)
                    label_free_prediction_files.append(
                        {
                            "direction": direction,
                            "event_kind": event_kind,
                            "horizon": horizon,
                            "path": str(prediction_path),
                            "sha256_before_target_evaluation": sha256_file(prediction_path),
                            "rows": len(predictions),
                            "target_labels_present": False,
                        }
                    )

                    metrics, evaluated, episode_rows = evaluate_predictions(
                        predictions, target_onset, horizon, alarm_probability
                    )
                    clock = clock_predictions(calibration_rows, target_unlabeled, config)
                    clock_metrics, _, _ = evaluate_predictions(
                        clock, target_onset, horizon, alarm_probability
                    )
                    constant_probability = np.full(
                        len(evaluated), global_probability["posterior_mean"], np.float64
                    )
                    constant_metrics = probability_metrics(
                        np.asarray([row["label"] for row in evaluated]),
                        constant_probability,
                    )
                    result = {
                        "direction": direction,
                        "source": source_tag,
                        "target": target_tag,
                        "event_kind": event_kind,
                        "horizon": horizon,
                        **metrics,
                        "source_calibration_prevalence": global_probability[
                            "posterior_mean"
                        ],
                        "source_calibration_support": global_probability["support"],
                        "source_calibration_events": global_probability["events"],
                        "constant_source_prior_brier": constant_metrics["brier"],
                        "clock_brier": clock_metrics["brier"],
                        "clock_roc_auc": clock_metrics["roc_auc"],
                        "clock_average_precision": clock_metrics["average_precision"],
                        "clock_max_predicted_probability": clock_metrics[
                            "max_predicted_probability"
                        ],
                        "clock_alarm_rows_075": clock_metrics["alarm_rows_075"],
                    }
                    evaluation_rows.append(result)
                    reliability_all.extend(
                        reliability_rows(evaluated, direction, event_kind, horizon)
                    )
                    if event_kind == config["primary_evaluation"]["event_type"] and horizon == int(
                        config["primary_evaluation"]["horizon_queries"]
                    ):
                        primary_results[direction] = result
                        primary_predictions_all.extend(
                            {
                                "direction": direction,
                                "target": target_tag,
                                **row,
                            }
                            for row in evaluated
                        )
                        primary_episode_all.extend(
                            {
                                "direction": direction,
                                "target": target_tag,
                                **row,
                            }
                            for row in episode_rows
                        )
                        primary_threshold_sweep.extend(
                            threshold_sweep_rows(
                                evaluated,
                                target_onset,
                                [
                                    float(value)
                                    for value in config[
                                        "reported_probability_thresholds"
                                    ]
                                ],
                                direction,
                            )
                        )

    # Detection upper-bound audit: treat onset as absorbing through episode end.
    # This is deliberately separate from the pre-onset hazard target above.
    for source_tag, target_tag in (("A", "B"), ("B", "A")):
        direction = f"{source_tag}_to_{target_tag}"
        state_path = output / "intermediate" / f"states_{source_tag}_to_{target_tag}.npz"
        with np.load(state_path) as states:
            source_score = np.asarray(states["source_trap_score"])
            source_run = np.asarray(states["source_trap_persistence"])
            target_score = np.asarray(states["target_trap_score"])
            target_run = np.asarray(states["target_trap_persistence"])
            source_unlabeled = eligible_rows(
                routes[source_tag]["length"],
                routes[source_tag]["valid"],
                source_score,
                source_run,
                int(config["minimum_query"]),
            )
            target_unlabeled = eligible_rows(
                routes[target_tag]["length"],
                routes[target_tag]["valid"],
                target_score,
                target_run,
                int(config["minimum_query"]),
            )
            source_onset = event_onset(corpora[source_tag], "trap")
            target_onset = event_onset(corpora[target_tag], "trap")
            for horizon in map(int, config["horizons"]):
                calibration = absorbing_rows(source_unlabeled, source_onset, horizon)
                mapping, cells, global_probability = fit_probability_table(
                    calibration, config
                )
                for row in cells:
                    absorbing_calibration_rows.append(
                        {"direction": direction, "horizon": horizon, **row}
                    )
                predictions = apply_probability_table(target_unlabeled, mapping, config)
                for row in predictions:
                    row["episode_id"] = int(
                        routes[target_tag]["episode_id"][int(row["episode_index"])]
                    )
                prediction_path = (
                    output
                    / "predictions_label_free_absorbing"
                    / f"{direction}_trap_h{horizon}.csv"
                )
                write_csv(prediction_path, predictions)
                label_free_prediction_files.append(
                    {
                        "direction": direction,
                        "event_kind": "trap",
                        "evaluation_mode": "onset_absorbing_proxy",
                        "horizon": horizon,
                        "path": str(prediction_path),
                        "sha256_before_target_evaluation": sha256_file(prediction_path),
                        "rows": len(predictions),
                        "target_labels_present": False,
                    }
                )
                metrics, evaluated = evaluate_absorbing_predictions(
                    predictions, target_onset, horizon, alarm_probability
                )
                constant = probability_metrics(
                    np.asarray([row["label"] for row in evaluated]),
                    np.full(
                        len(evaluated),
                        global_probability["posterior_mean"],
                        np.float64,
                    ),
                )
                clock = clock_predictions(calibration, target_unlabeled, config)
                clock_metrics, _ = evaluate_absorbing_predictions(
                    clock, target_onset, horizon, alarm_probability
                )
                result = {
                    "direction": direction,
                    "source": source_tag,
                    "target": target_tag,
                    "event_kind": "trap",
                    "evaluation_mode": "onset_absorbing_proxy",
                    "horizon": horizon,
                    **metrics,
                    "source_calibration_prevalence": global_probability[
                        "posterior_mean"
                    ],
                    "source_calibration_support": global_probability["support"],
                    "source_calibration_events": global_probability["events"],
                    "constant_source_prior_brier": constant["brier"],
                    "clock_brier": clock_metrics["brier"],
                    "clock_roc_auc": clock_metrics["roc_auc"],
                    "clock_average_precision": clock_metrics["average_precision"],
                    "clock_max_predicted_probability": clock_metrics[
                        "max_predicted_probability"
                    ],
                    "clock_alarm_rows_075": clock_metrics["alarm_rows_075"],
                }
                absorbing_evaluation_rows.append(result)
                absorbing_reliability_rows.extend(
                    {
                        **row,
                        "evaluation_mode": "onset_absorbing_proxy",
                    }
                    for row in reliability_rows(
                        evaluated, direction, "trap", horizon
                    )
                )

    table_dir = output / "tables"
    write_csv(table_dir / "cross_corpus_probability_evaluation.csv", evaluation_rows)
    write_csv(table_dir / "calibration_cells.csv", calibration_rows_all)
    write_csv(table_dir / "target_reliability.csv", reliability_all)
    write_csv(table_dir / "primary_target_predictions_posthoc.csv", primary_predictions_all)
    write_csv(table_dir / "primary_episode_alarm_audit.csv", primary_episode_all)
    write_csv(table_dir / "primary_probability_threshold_sweep.csv", primary_threshold_sweep)
    write_csv(
        table_dir / "absorbing_detection_evaluation.csv", absorbing_evaluation_rows
    )
    write_csv(
        table_dir / "absorbing_calibration_cells.csv", absorbing_calibration_rows
    )
    write_csv(
        table_dir / "absorbing_target_reliability.csv", absorbing_reliability_rows
    )
    evaluations = pd.DataFrame(evaluation_rows)
    reliability = pd.DataFrame(reliability_all)
    plot_results(evaluations, reliability, output)

    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "training": False,
        "gradient_optimization": False,
        "learned_feature_weights": False,
        "labeled_probability_calibration": True,
        "runtime_task_identity_used": False,
        "runtime_action_values_used": False,
        "runtime_physical_state_used": False,
        "runtime_outcome_used": False,
        "route_states_materialized_before_labels": True,
        "target_probabilities_written_without_target_labels": True,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "route_only_artifacts": route_manifests,
        "direction_state_artifacts": direction_manifests,
        "label_free_prediction_files": label_free_prediction_files,
        "primary_target": config["primary_evaluation"],
        "alarm_probability": alarm_probability,
        "primary_results": primary_results,
        "primary_threshold_sweep": primary_threshold_sweep,
        "absorbing_detection_results": absorbing_evaluation_rows,
        "onset_hazard_all_probability_states_below_alarm_threshold": all(
            row["max_predicted_probability"] < alarm_probability
            for row in evaluation_rows
        ),
        "onset_hazard_alarm_rows_total": int(
            sum(row["alarm_rows_075"] for row in evaluation_rows)
        ),
        "absorbing_proxy_alarm_rows_total": int(
            sum(row["alarm_rows_075"] for row in absorbing_evaluation_rows)
        ),
        "limitations": [
            "Corpus A has dense physical onsets; corpus B uses a validated query-resolution proxy.",
            "The target is onset or future onset while at risk, not active Trap occupancy after onset.",
            "Absolute posterior risk depends on the deployment event prior and shifted substantially across corpora.",
            "The same-noise input-version feature is excluded because it currently has only one failed case.",
        ],
    }
    write_json(output / "summary.json", summary)
    (output / "REPORT_ZH.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
