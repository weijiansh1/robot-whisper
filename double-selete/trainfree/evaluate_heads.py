#!/usr/bin/env python3
"""Evaluate already-frozen train-free scores; labels enter only in this file."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
SCORES = RESULTS / "unlabeled_scores.npz"
SCORE_MANIFEST = RESULTS / "score_manifest.json"
PROTOCOL = HERE / "PROTOCOL.md"
LABEL_MODULE = ROOT / "himoe-vla_trap/code/analyze_trainfree_signal_matrix.py"
SCHEMA = "himoe.trainfree_double_selector.evaluation.v1"
SEED = 20260903
QUERIES = (30, 34)
BUDGETS = (4, 8, 16)

PRIMARY_SCORES = (
    "loop_soft",
    "static_soft",
    "double_soft_max",
    "single_soft_mean",
    "single_mobility",
)
DIAGNOSTIC_SCORES = (
    "loop_hard",
    "static_hard",
    "double_hard_max",
    "single_hard_mean",
)
TARGETS = ("loop", "static", "trap")
COUNT_COLUMNS = (
    "n_selected",
    "trap_tp",
    "trap_pos",
    "loop_tp",
    "loop_pos",
    "static_tp",
    "static_pos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=20_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_label_module():
    spec = importlib.util.spec_from_file_location("frozen_trap_labels", LABEL_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import label module: {LABEL_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, bool)
    score = np.asarray(score, np.float64)
    good = np.isfinite(score)
    target = target[good]
    score = score[good]
    n_positive = int(target.sum())
    n_negative = int((~target).sum())
    if n_positive == 0 or n_negative == 0:
        return np.nan
    ranks = rankdata(score, method="average")
    rank_sum = ranks[target].sum()
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def grouped_auc_rows(
    target: np.ndarray, score: np.ndarray, group: np.ndarray
) -> tuple[float, float, int, int, int]:
    aucs = []
    weights = []
    total_positive = 0
    total_negative = 0
    for pool in np.unique(group):
        take = group == pool
        good = take & np.isfinite(score)
        y = target[good]
        s = score[good]
        positive = int(y.sum())
        negative = int((~y).sum())
        if positive == 0 or negative == 0:
            continue
        aucs.append(binary_auc(y, s))
        weights.append(positive * negative)
        total_positive += positive
        total_negative += negative
    if not aucs:
        return np.nan, np.nan, 0, total_positive, total_negative
    return (
        float(np.mean(aucs)),
        float(np.average(aucs, weights=weights)),
        len(aucs),
        total_positive,
        total_negative,
    )


def group_auc_matrix(
    labels: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    mapping: dict[str, str],
    group: np.ndarray,
    query: int,
) -> np.ndarray:
    """Return pool x subtype AUCs for a fixed multi-label score mapping."""
    pools = np.unique(group)
    output = np.full((len(pools), 2), np.nan, np.float64)
    for pool_index, pool in enumerate(pools):
        take = group == pool
        for target_index, target_name in enumerate(("loop", "static")):
            output[pool_index, target_index] = binary_auc(
                labels[target_name][take],
                scores[mapping[target_name]][take, query],
            )
    return output


def multilabel_auc_bootstraps(
    labels: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    group: np.ndarray,
    draws: int,
) -> pd.DataFrame:
    mappings = {
        "typed_soft_heads": {"loop": "loop_soft", "static": "static_soft"},
        "typed_hard_heads": {"loop": "loop_hard", "static": "static_hard"},
        "single_soft_mean": {
            "loop": "single_soft_mean",
            "static": "single_soft_mean",
        },
        "single_hard_mean": {
            "loop": "single_hard_mean",
            "static": "single_hard_mean",
        },
        "single_mobility": {
            "loop": "single_mobility",
            "static": "single_mobility",
        },
        "collapsed_soft_max": {
            "loop": "double_soft_max",
            "static": "double_soft_max",
        },
    }
    comparisons = (
        ("typed_soft_heads", "single_soft_mean"),
        ("typed_soft_heads", "single_mobility"),
        ("typed_soft_heads", "collapsed_soft_max"),
        ("typed_hard_heads", "single_hard_mean"),
        ("typed_hard_heads", "single_mobility"),
    )
    rng = np.random.default_rng(SEED + 2)
    output = []
    for query in QUERIES:
        matrices = {
            name: group_auc_matrix(labels, scores, mapping, group, query)
            for name, mapping in mappings.items()
        }
        n_groups = next(iter(matrices.values())).shape[0]
        sample = rng.integers(0, n_groups, size=(draws, n_groups))
        for left, right in comparisons:
            left_value = matrices[left]
            right_value = matrices[right]
            informative = np.isfinite(left_value) & np.isfinite(right_value)
            left_masked = np.where(informative, left_value, np.nan)
            right_masked = np.where(informative, right_value, np.nan)
            # Average pools within subtype, then weight loop/static equally.
            left_boot = np.nanmean(np.nanmean(left_masked[sample], axis=1), axis=1)
            right_boot = np.nanmean(np.nanmean(right_masked[sample], axis=1), axis=1)
            delta = left_boot - right_boot
            point_left = float(np.nanmean(np.nanmean(left_masked, axis=0)))
            point_right = float(np.nanmean(np.nanmean(right_masked, axis=0)))
            p_two_sided = min(
                1.0,
                2.0 * min(float(np.mean(delta <= 0)), float(np.mean(delta >= 0))),
            )
            output.append(
                {
                    "query": query,
                    "left": left,
                    "right": right,
                    "left_macro_subtype_auc": point_left,
                    "right_macro_subtype_auc": point_right,
                    "delta": point_left - point_right,
                    "ci_low": float(np.quantile(delta, 0.025)),
                    "ci_high": float(np.quantile(delta, 0.975)),
                    "p_two_sided": p_two_sided,
                    "n_informative_pool_subtypes": int(informative.sum()),
                    "bootstrap_draws": draws,
                }
            )
    return pd.DataFrame(output)


def stable_topk(
    indices: np.ndarray, score: np.ndarray, episode: np.ndarray, k: int
) -> np.ndarray:
    if k <= 0:
        return np.empty(0, np.int64)
    finite = indices[np.isfinite(score[indices])]
    order = np.lexsort((episode[finite], -score[finite]))
    return finite[order[: min(k, len(finite))]]


def selector_counts(
    selected: np.ndarray,
    loop: np.ndarray,
    static: np.ndarray,
    pool_indices: np.ndarray,
) -> dict[str, int]:
    trap = loop | static
    return {
        "n_selected": int(len(selected)),
        "trap_tp": int(trap[selected].sum()),
        "trap_pos": int(trap[pool_indices].sum()),
        "loop_tp": int(loop[selected].sum()),
        "loop_pos": int(loop[pool_indices].sum()),
        "static_tp": int(static[selected].sum()),
        "static_pos": int(static[pool_indices].sum()),
    }


def append_selection_row(
    rows: list[dict],
    selector: str,
    family: str,
    query: int,
    budget: int,
    pool: int,
    selected: np.ndarray,
    loop: np.ndarray,
    static: np.ndarray,
    pool_indices: np.ndarray,
) -> None:
    row = {
        "query": query,
        "budget_per_head": budget,
        "group": pool,
        "family": family,
        "selector": selector,
    }
    row.update(selector_counts(selected, loop, static, pool_indices))
    rows.append(row)


def build_selection_rows(
    score: dict[str, np.ndarray],
    episode: np.ndarray,
    group: np.ndarray,
    valid: np.ndarray,
    loop: np.ndarray,
    static: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    families = {
        "soft": {
            "loop": "loop_soft",
            "static": "static_soft",
            "double": "double_soft_max",
            "mean": "single_soft_mean",
        },
        "hard_diagnostic": {
            "loop": "loop_hard",
            "static": "static_hard",
            "double": "double_hard_max",
            "mean": "single_hard_mean",
        },
    }
    for query in QUERIES:
        for budget in BUDGETS:
            for pool in np.unique(group):
                pool_indices = np.flatnonzero((group == pool) & valid[:, query])
                for family, names in families.items():
                    loop_selected = stable_topk(
                        pool_indices, score[names["loop"]][:, query], episode, budget
                    )
                    static_selected = stable_topk(
                        pool_indices, score[names["static"]][:, query], episode, budget
                    )
                    union_selected = np.union1d(loop_selected, static_selected)
                    union_size = len(union_selected)
                    double_selected = stable_topk(
                        pool_indices, score[names["double"]][:, query], episode, budget
                    )
                    mean_selected = stable_topk(
                        pool_indices, score[names["mean"]][:, query], episode, budget
                    )
                    mobility_selected = stable_topk(
                        pool_indices,
                        score["single_mobility"][:, query],
                        episode,
                        budget,
                    )
                    selections = {
                        f"loop_{family}_topk": loop_selected,
                        f"static_{family}_topk": static_selected,
                        f"double_{family}_union": union_selected,
                        f"double_{family}_max_topk": double_selected,
                        f"single_{family}_mean_topk": mean_selected,
                        f"single_mobility_topk_{family}": mobility_selected,
                        f"double_{family}_max_matched": stable_topk(
                            pool_indices,
                            score[names["double"]][:, query],
                            episode,
                            union_size,
                        ),
                        f"single_{family}_mean_matched": stable_topk(
                            pool_indices,
                            score[names["mean"]][:, query],
                            episode,
                            union_size,
                        ),
                        f"single_mobility_matched_{family}": stable_topk(
                            pool_indices,
                            score["single_mobility"][:, query],
                            episode,
                            union_size,
                        ),
                    }
                    for selector, selected in selections.items():
                        append_selection_row(
                            rows,
                            selector,
                            family,
                            query,
                            budget,
                            int(pool),
                            selected,
                            loop,
                            static,
                            pool_indices,
                        )
    return pd.DataFrame(rows)


def aggregate_metric(counts: np.ndarray, metric: str) -> np.ndarray:
    """Compute metrics from [..., count_column] arrays."""
    index = {name: offset for offset, name in enumerate(COUNT_COLUMNS)}

    def ratio(numerator: str, denominator: str) -> np.ndarray:
        top = counts[..., index[numerator]]
        bottom = counts[..., index[denominator]]
        return np.divide(
            top,
            bottom,
            out=np.full(top.shape, np.nan, np.float64),
            where=bottom > 0,
        )

    if metric == "trap_precision":
        return ratio("trap_tp", "n_selected")
    if metric == "trap_recall":
        return ratio("trap_tp", "trap_pos")
    if metric == "loop_recall":
        return ratio("loop_tp", "loop_pos")
    if metric == "static_recall":
        return ratio("static_tp", "static_pos")
    if metric == "macro_subtype_recall":
        return 0.5 * (
            aggregate_metric(counts, "loop_recall")
            + aggregate_metric(counts, "static_recall")
        )
    raise KeyError(metric)


def aggregate_selection(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    keys = ["query", "budget_per_head", "family", "selector"]
    for values, frame in rows.groupby(keys, sort=True):
        counts = frame.loc[:, COUNT_COLUMNS].to_numpy(np.float64).sum(axis=0)
        record = dict(zip(keys, values, strict=True))
        record["mean_selected_per_pool"] = float(frame.n_selected.mean())
        record["total_selected"] = int(frame.n_selected.sum())
        for metric in (
            "trap_precision",
            "trap_recall",
            "loop_recall",
            "static_recall",
            "macro_subtype_recall",
        ):
            record[metric] = float(aggregate_metric(counts, metric))
        output.append(record)
    return pd.DataFrame(output)


def bootstrap_comparison(
    rows: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    left_frame = rows[rows.selector == left].sort_values("group")
    right_frame = rows[rows.selector == right].sort_values("group")
    if not np.array_equal(left_frame.group.to_numpy(), right_frame.group.to_numpy()):
        raise RuntimeError(f"unpaired groups: {left} vs {right}")
    left_count = left_frame.loc[:, COUNT_COLUMNS].to_numpy(np.float64)
    right_count = right_frame.loc[:, COUNT_COLUMNS].to_numpy(np.float64)
    group_count = len(left_count)
    sample = rng.integers(0, group_count, size=(draws, group_count))
    left_boot = left_count[sample].sum(axis=1)
    right_boot = right_count[sample].sum(axis=1)
    delta = aggregate_metric(left_boot, metric) - aggregate_metric(right_boot, metric)
    delta = delta[np.isfinite(delta)]
    point = float(
        aggregate_metric(left_count.sum(axis=0), metric)
        - aggregate_metric(right_count.sum(axis=0), metric)
    )
    p_two_sided = min(
        1.0,
        2.0 * min(float(np.mean(delta <= 0)), float(np.mean(delta >= 0))),
    )
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "delta": point,
        "ci_low": float(np.quantile(delta, 0.025)),
        "ci_high": float(np.quantile(delta, 0.975)),
        "p_two_sided": p_two_sided,
        "bootstrap_draws": int(len(delta)),
    }


def selection_bootstraps(rows: pd.DataFrame, draws: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    output = []
    for query in QUERIES:
        for budget in BUDGETS:
            subset = rows[(rows["query"] == query) & (rows.budget_per_head == budget)]
            for family in ("soft", "hard_diagnostic"):
                comparisons = [
                    (
                        f"loop_{family}_topk",
                        f"single_{family}_mean_topk",
                        ("loop_recall",),
                    ),
                    (
                        f"static_{family}_topk",
                        f"single_{family}_mean_topk",
                        ("static_recall",),
                    ),
                    (
                        f"double_{family}_union",
                        f"single_{family}_mean_matched",
                        (
                            "trap_precision",
                            "trap_recall",
                            "loop_recall",
                            "static_recall",
                            "macro_subtype_recall",
                        ),
                    ),
                    (
                        f"double_{family}_max_topk",
                        f"single_{family}_mean_topk",
                        ("trap_precision", "trap_recall", "macro_subtype_recall"),
                    ),
                ]
                if family == "soft":
                    comparisons.extend(
                        [
                            (
                                "loop_soft_topk",
                                "single_mobility_topk_soft",
                                ("loop_recall",),
                            ),
                            (
                                "static_soft_topk",
                                "single_mobility_topk_soft",
                                ("static_recall",),
                            ),
                            (
                                "double_soft_union",
                                "single_mobility_matched_soft",
                                (
                                    "trap_precision",
                                    "trap_recall",
                                    "loop_recall",
                                    "static_recall",
                                    "macro_subtype_recall",
                                ),
                            ),
                            (
                                "double_soft_max_topk",
                                "single_mobility_topk_soft",
                                (
                                    "trap_precision",
                                    "trap_recall",
                                    "macro_subtype_recall",
                                ),
                            ),
                        ]
                    )
                for left, right, metrics in comparisons:
                    for metric in metrics:
                        record = bootstrap_comparison(
                            subset, left, right, metric, draws, rng
                        )
                        record.update(
                            query=query,
                            budget_per_head=budget,
                            family=family,
                        )
                        output.append(record)
    return pd.DataFrame(output)


def event_comparisons(
    event: str,
    onset: np.ndarray,
    offset: int,
    score_names: tuple[str, ...],
    scores: dict[str, np.ndarray],
    episode: np.ndarray,
    group: np.ndarray,
    valid: np.ndarray,
) -> pd.DataFrame:
    rows = []
    target = onset >= 0
    for index in np.flatnonzero(target):
        query = int(onset[index] + offset)
        if query < 0 or query >= valid.shape[1] or not valid[index, query]:
            continue
        controls_base = (group == group[index]) & ~target & valid[:, query]
        for score_name in score_names:
            value = scores[score_name][:, query]
            controls = np.flatnonzero(controls_base & np.isfinite(value))
            if not np.isfinite(value[index]) or len(controls) == 0:
                continue
            wins = float(
                np.sum(value[index] > value[controls])
                + 0.5 * np.sum(value[index] == value[controls])
            )
            rows.append(
                {
                    "event": event,
                    "offset": offset,
                    "score": score_name,
                    "episode_id": int(episode[index]),
                    "group": int(group[index]),
                    "query": query,
                    "wins": wins,
                    "n_controls": int(len(controls)),
                    "event_auc": wins / len(controls),
                }
            )
    return pd.DataFrame(rows)


def summarize_onset(events: pd.DataFrame, draws: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for (event, offset, score_name), frame in events.groupby(
        ["event", "offset", "score"], sort=True
    ):
        by_group = (
            frame.groupby("group", sort=True)[["wins", "n_controls"]]
            .sum()
            .reset_index()
        )
        values = by_group[["wins", "n_controls"]].to_numpy(np.float64)
        sample = rng.integers(0, len(values), size=(draws, len(values)))
        boot = values[sample].sum(axis=1)
        auc_boot = boot[:, 0] / boot[:, 1]
        rows.append(
            {
                "event": event,
                "offset": int(offset),
                "score": score_name,
                "n_events": int(len(frame)),
                "n_groups": int(len(by_group)),
                "pair_weighted_auc": float(frame.wins.sum() / frame.n_controls.sum()),
                "mean_event_auc": float(frame.event_auc.mean()),
                "ci_low": float(np.quantile(auc_boot, 0.025)),
                "ci_high": float(np.quantile(auc_boot, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def onset_bootstrap_deltas(events: pd.DataFrame, draws: int) -> pd.DataFrame:
    comparisons = {
        "loop": (
            ("loop_soft", "single_soft_mean"),
            ("loop_soft", "single_mobility"),
            ("loop_soft", "loop_hard"),
        ),
        "static": (
            ("static_soft", "single_soft_mean"),
            ("static_soft", "single_mobility"),
            ("static_soft", "static_hard"),
        ),
    }
    rng = np.random.default_rng(SEED + 3)
    output = []
    join = ["event", "offset", "episode_id", "group", "query"]
    for event, pairs in comparisons.items():
        event_rows = events[events.event == event]
        for left, right in pairs:
            left_rows = event_rows[event_rows.score == left]
            right_rows = event_rows[event_rows.score == right]
            paired = left_rows.merge(right_rows, on=join, suffixes=("_left", "_right"))
            grouped = paired.groupby("group", sort=True)[
                ["wins_left", "n_controls_left", "wins_right", "n_controls_right"]
            ].sum()
            value = grouped.to_numpy(np.float64)
            sample = rng.integers(0, len(value), size=(draws, len(value)))
            boot = value[sample].sum(axis=1)
            left_boot = boot[:, 0] / boot[:, 1]
            right_boot = boot[:, 2] / boot[:, 3]
            delta = left_boot - right_boot
            left_point = float(paired.wins_left.sum() / paired.n_controls_left.sum())
            right_point = float(paired.wins_right.sum() / paired.n_controls_right.sum())
            p_two_sided = min(
                1.0,
                2.0 * min(float(np.mean(delta <= 0)), float(np.mean(delta >= 0))),
            )
            output.append(
                {
                    "event": event,
                    "offset": int(paired.offset.iloc[0]),
                    "left": left,
                    "right": right,
                    "left_pair_weighted_auc": left_point,
                    "right_pair_weighted_auc": right_point,
                    "delta": left_point - right_point,
                    "ci_low": float(np.quantile(delta, 0.025)),
                    "ci_high": float(np.quantile(delta, 0.975)),
                    "p_two_sided": p_two_sided,
                    "n_paired_events": int(len(paired)),
                    "n_groups": int(len(grouped)),
                    "bootstrap_draws": draws,
                }
            )
    return pd.DataFrame(output)


def main() -> None:
    args = parse_args()
    score_manifest = json.loads(SCORE_MANIFEST.read_text())
    if score_manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise RuntimeError("protocol changed after the label-blind scores were frozen")
    if not score_manifest.get("train_free") or score_manifest.get("fit_calls") != 0:
        raise RuntimeError("score manifest does not certify a train-free build")

    with np.load(SCORES) as data:
        if str(data["schema"].item()) != "himoe.trainfree_double_selector.scores.v1":
            raise RuntimeError("unexpected score schema")
        episode = np.asarray(data["episode"], np.int64)
        group = np.asarray(data["group"], np.int64)
        valid = np.asarray(data["valid"], bool)
        score_names = PRIMARY_SCORES + DIAGNOSTIC_SCORES
        scores = {name: np.asarray(data[name], np.float64) for name in score_names}
        components = {
            name.removeprefix("component__"): np.asarray(data[name], np.float64)
            for name in data.files
            if name.startswith("component__")
        }

    if not all(np.isfinite(value[:, 34]).all() for value in scores.values()):
        raise RuntimeError("query-34 scores must be finite for every trajectory")

    # Labels are opened only after the score file and protocol hash pass validation.
    label_module = load_label_module()
    corpus, inventory = label_module.load_corpus("B")
    label_episode = corpus.meta.episode_id.to_numpy(np.int64)
    if not np.array_equal(episode, label_episode):
        raise RuntimeError("score/label episode order mismatch")
    loop_onset = np.asarray(corpus.loop_onset, np.int64)
    static_onset = np.asarray(corpus.static_onset, np.int64)
    labels = {
        "loop": loop_onset >= 0,
        "static": static_onset >= 0,
    }
    labels["trap"] = labels["loop"] | labels["static"]

    inventory.to_csv(RESULTS / "label_inventory.csv", index=False)
    assignment_audit = pd.read_csv(RESULTS / "unlabeled_assignments.csv")
    assignment_episode = assignment_audit.episode_id.to_numpy(np.int64)
    assignment_audit["loop_label"] = labels["loop"][assignment_episode]
    assignment_audit["static_label"] = labels["static"][assignment_episode]
    assignment_audit["trap_label"] = labels["trap"][assignment_episode]
    assignment_audit["loop_onset"] = loop_onset[assignment_episode]
    assignment_audit["static_onset"] = static_onset[assignment_episode]
    assignment_audit.to_csv(RESULTS / "assignment_audit_labeled.csv", index=False)
    auc_rows = []
    for query in QUERIES:
        for score_name, value in scores.items():
            for target_name, target in labels.items():
                macro, weighted, n_groups, n_positive, n_negative = grouped_auc_rows(
                    target, value[:, query], group
                )
                auc_rows.append(
                    {
                        "query": query,
                        "score": score_name,
                        "target": target_name,
                        "group_macro_auc": macro,
                        "within_group_pair_auc": weighted,
                        "n_informative_groups": n_groups,
                        "n_positive_in_informative_groups": n_positive,
                        "n_negative_in_informative_groups": n_negative,
                    }
                )
    pd.DataFrame(auc_rows).to_csv(RESULTS / "snapshot_auc.csv", index=False)
    multilabel_auc_bootstraps(labels, scores, group, args.bootstrap).to_csv(
        RESULTS / "multilabel_auc_deltas.csv", index=False
    )

    component_rows = []
    for query in QUERIES:
        for component, value in components.items():
            for target_name, target in labels.items():
                macro, weighted, n_groups, _, _ = grouped_auc_rows(
                    target, value[:, query], group
                )
                component_rows.append(
                    {
                        "query": query,
                        "component": component,
                        "target": target_name,
                        "group_macro_auc": macro,
                        "within_group_pair_auc": weighted,
                        "n_informative_groups": n_groups,
                    }
                )
    pd.DataFrame(component_rows).to_csv(RESULTS / "component_auc.csv", index=False)

    selection_rows = build_selection_rows(
        scores, episode, group, valid, labels["loop"], labels["static"]
    )
    selection_rows.to_csv(RESULTS / "selection_by_group.csv", index=False)
    aggregate_selection(selection_rows).to_csv(
        RESULTS / "selection_metrics.csv", index=False
    )
    selection_bootstraps(selection_rows, args.bootstrap).to_csv(
        RESULTS / "bootstrap_deltas.csv", index=False
    )

    loop_events = event_comparisons(
        "loop",
        loop_onset,
        -2,
        ("loop_soft", "loop_hard", "single_soft_mean", "single_mobility"),
        scores,
        episode,
        group,
        valid,
    )
    static_events = event_comparisons(
        "static",
        static_onset,
        0,
        ("static_soft", "static_hard", "single_soft_mean", "single_mobility"),
        scores,
        episode,
        group,
        valid,
    )
    onset_events = pd.concat([loop_events, static_events], ignore_index=True)
    onset_events.to_csv(RESULTS / "onset_event_auc.csv", index=False)
    summarize_onset(onset_events, args.bootstrap).to_csv(
        RESULTS / "onset_alignment.csv", index=False
    )
    onset_bootstrap_deltas(onset_events, args.bootstrap).to_csv(
        RESULTS / "onset_bootstrap_deltas.csv", index=False
    )

    manifest = {
        "schema": SCHEMA,
        "created": "2026-09-03",
        "score_manifest_sha256": sha256(SCORE_MANIFEST),
        "protocol_sha256": sha256(PROTOCOL),
        "label_definition": str(LABEL_MODULE.relative_to(ROOT)),
        "bootstrap_draws": args.bootstrap,
        "seed": SEED,
        "queries": list(QUERIES),
        "budgets_per_head": list(BUDGETS),
        "label_counts": {name: int(value.sum()) for name, value in labels.items()},
    }
    (RESULTS / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest["label_counts"], sort_keys=True))
    print("TRAINFREE_EVALUATION_OK")


if __name__ == "__main__":
    main()
