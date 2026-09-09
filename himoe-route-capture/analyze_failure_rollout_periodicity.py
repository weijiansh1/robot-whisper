#!/usr/bin/env python3
"""Test whether failed VLA_MUI_HUB rollouts repeat or cycle in MoE routing.

The analysis deliberately separates three effects that are easy to conflate:

1. persistence: adjacent control queries use similar routes;
2. repetition: a non-adjacent route resembles an earlier route;
3. periodicity: similarity has a local peak at lag k, with at least three
   observable cycles in the analysis window.

The primary comparison uses the longest equal prefix available to every
episode of a task, so timeout failures cannot win merely by being longer.
Labels are compared only within task x initial-state strata.  A same-length
terminal-window comparison and a paired early-vs-late analysis within failed
episodes are reported separately.

HB routing is measured in two independent ways:

* actual top-4 overlap over all 8 layers x 10 denoise steps x 11 tokens;
* distance between the 8 x 32 soft routing vectors for the state token at
  denoise step 0.

Action chunks and simulator state use the same recurrence metrics as behavior
controls.  AS routing is validated but not analyzed: its gate input is the
constant data mask, and all available runs have constant AS probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import zarr
from scipy.stats import fisher_exact, rankdata, wilcoxon


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
MODEL_ROOT = REPO / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/failure-rollout-periodicity"
MAX_PERIOD = 8
DEFAULT_PERMUTATIONS = 5000
BLOCK = 512


@dataclass
class RunData:
    task: str
    suite: str
    run: pathlib.Path
    summaries: list[dict]
    n_rows: np.ndarray
    offsets: np.ndarray
    failure: np.ndarray
    scene: np.ndarray
    hard_ids: np.ndarray
    soft_route: np.ndarray
    state_top1: np.ndarray
    actions: np.ndarray
    sim_state: np.ndarray
    as_constant: bool


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    sd = x.std(axis=0)
    keep = sd > 1e-6
    if not np.any(keep):
        raise ValueError("all channels are constant")
    return (x[:, keep] - x[:, keep].mean(axis=0)) / sd[keep]


def episode_path(client: pathlib.Path, episode_index: int) -> pathlib.Path:
    # %02d intentionally becomes three digits without padding once index >=100.
    return client / ("episode_%02d.npz" % episode_index)


def load_run(summary_path: pathlib.Path) -> RunData:
    run = summary_path.parents[1]
    rel = run.relative_to(MODEL_ROOT)
    suite, task = rel.parts[:2]
    summaries = sorted(json.loads(summary_path.read_text()),
                       key=lambda x: x["episode_index"])
    n_rows = np.array([x["inference_calls"] for x in summaries], np.int32)
    offsets = np.r_[0, np.cumsum(n_rows)[:-1]].astype(np.int64)
    failure = np.array([not x["success"] for x in summaries], bool)
    scene = np.array([x["init_state_id"] for x in summaries], np.int32)
    total = int(n_rows.sum())

    store = zarr.open(str(run / "server/routes.zarr"), mode="r")
    episode_id = np.asarray(store["episode_id"][:])
    expected_id = np.repeat(
        np.array([x["episode_index"] for x in summaries], np.int32), n_rows)
    if len(episode_id) != total or not np.array_equal(episode_id, expected_id):
        raise ValueError(f"episode_id alignment failed for {run}")

    hard_ids = np.asarray(store["hb_expert_ids"][:])
    if hard_ids.shape != (total, 8, 10, 11, 4):
        raise ValueError(f"unexpected hb_expert_ids shape: {hard_ids.shape}")

    soft_route = np.empty((total, 8 * 32), np.float32)
    state_top1 = np.empty((total, 8), np.uint8)
    probs = store["hb_router_probs"]
    for a in range(0, total, BLOCK):
        b = min(a + BLOCK, total)
        block = np.asarray(probs[a:b, :, 0, 0, :], np.float32)
        soft_route[a:b] = block.reshape(b - a, -1)
        state_top1[a:b] = block.argmax(axis=-1)

    as_ids = np.asarray(store["as_expert_ids"][:])
    as_probs = np.asarray(store["as_probs"][:], np.float32)
    as_constant = bool(
        np.all(as_ids == as_ids[0]) and np.all(as_probs == as_probs[0]))

    client = run / "client"
    action_parts: list[np.ndarray] = []
    sim_parts: list[np.ndarray] = []
    for summary, length in zip(summaries, n_rows):
        with np.load(episode_path(client, summary["episode_index"]),
                     allow_pickle=False) as episode:
            action = np.asarray(episode["actions"], np.float32)
            sim = np.asarray(episode["sim_state"], np.float32)
        if len(action) != length or len(sim) != length:
            raise ValueError(
                f"client/server length mismatch in episode "
                f"{summary['episode_index']} of {task}")
        action_parts.append(action.reshape(length, -1))
        sim_parts.append(sim)

    return RunData(
        task=task,
        suite=suite,
        run=run,
        summaries=summaries,
        n_rows=n_rows,
        offsets=offsets,
        failure=failure,
        scene=scene,
        hard_ids=hard_ids,
        soft_route=standardize(soft_route),
        state_top1=state_top1,
        actions=standardize(np.concatenate(action_parts)),
        sim_state=standardize(np.concatenate(sim_parts)),
        as_constant=as_constant,
    )


def hard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Each top-4 row is unique, so equality count is the set intersection size.
    intersection = (a[..., :, None] == b[..., None, :]).sum(axis=(-1, -2))
    return float(intersection.mean() / 4.0)


def continuous_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Higher is more recurrent. RMS distance is dimension-normalized.
    return float(-np.sqrt(np.mean(np.square(a - b), axis=-1)).mean())


def curve(sequence: np.ndarray, max_period: int,
          similarity: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    # One extra lag supplies the right neighbor of the last candidate period.
    return np.array([
        similarity(sequence[lag:], sequence[:-lag])
        for lag in range(1, max_period + 2)
    ], np.float64)


def curve_features(values: np.ndarray, max_period: int) -> dict[str, float]:
    candidates = np.arange(2, max_period + 1)
    prominence = (
        values[1:max_period]
        - 0.5 * (values[:max_period - 1] + values[2:max_period + 1])
    )
    returns = values[1:max_period] - values[0]
    p_i = int(np.argmax(prominence))
    r_i = int(np.argmax(returns))
    return {
        "lag1": float(values[0]),
        "nonlocal_repeat": float(values[1:max_period].max()),
        "periodic_prominence": float(prominence[p_i]),
        "prominent_period": int(candidates[p_i]),
        "return_excess": float(returns[r_i]),
        "return_period": int(candidates[r_i]),
    }


def window_metrics(data: RunData, episode: int, start: int, length: int) -> dict:
    max_period = min(MAX_PERIOD, (length - 1) // 3)
    if max_period < 2:
        raise ValueError(f"window {length} is too short for three period-2 cycles")
    a = int(data.offsets[episode] + start)
    b = a + length
    representations = {
        "hard": (data.hard_ids[a:b], hard_similarity),
        "soft": (data.soft_route[a:b], continuous_similarity),
        "action": (data.actions[a:b], continuous_similarity),
        "sim": (data.sim_state[a:b], continuous_similarity),
    }
    out: dict[str, float] = {"max_period": max_period}
    for name, (sequence, similarity) in representations.items():
        features = curve_features(curve(sequence, max_period, similarity),
                                  max_period)
        for key, value in features.items():
            out[f"{name}_{key}"] = value

    hard_cycle = (
        out["hard_return_excess"] > 0
        and out["soft_return_excess"] > 0
        and abs(out["hard_return_period"] - out["soft_return_period"]) <= 1
    )
    state_cycle = (
        hard_cycle
        and out["sim_return_excess"] > 0
        and abs(out["hard_return_period"] - out["sim_return_period"]) <= 1
    )
    action_cycle = (
        state_cycle
        and out["action_return_excess"] > 0
        and abs(out["hard_return_period"] - out["action_return_period"]) <= 1
    )
    out["route_cycle_candidate"] = float(hard_cycle)
    out["route_state_cycle_candidate"] = float(state_cycle)
    out["joint_loop_candidate"] = float(action_cycle)
    return out


def residualize(values: np.ndarray, controls: np.ndarray,
                groups: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        x = np.column_stack([np.ones(len(idx)), controls[idx]])
        result[idx] = values[idx] - x @ np.linalg.lstsq(
            x, values[idx], rcond=None)[0]
    return result


def stratified_auc(values: np.ndarray, positive: np.ndarray,
                   groups: np.ndarray) -> float:
    numerator = 0.0
    denominator = 0.0
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        y = positive[idx]
        n1, n0 = int(y.sum()), int((~y).sum())
        if not n1 or not n0:
            continue
        ranks = rankdata(values[idx])
        numerator += ranks[y].sum() - n1 * (n1 + 1) / 2.0
        denominator += n1 * n0
    return float(numerator / denominator) if denominator else float("nan")


def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def stratified_permutation_stats(
        metrics: dict[str, np.ndarray], positive: np.ndarray,
        groups: np.ndarray, n_permutations: int,
        rng: np.random.Generator) -> dict[str, dict]:
    names = list(metrics)
    values = np.column_stack([metrics[name] for name in names]).astype(np.float64)
    valid = np.zeros(len(positive), bool)
    valid_groups: list[np.ndarray] = []
    base = 0.0
    denominator = 0.0
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        y = positive[idx]
        n1, n0 = int(y.sum()), int((~y).sum())
        if not n1 or not n0:
            continue
        valid[idx] = True
        valid_groups.append(idx)
        base += n1 * (n1 + 1) / 2.0
        denominator += n1 * n0

    use = np.flatnonzero(valid)
    y = positive[use]
    local = {original: i for i, original in enumerate(use)}
    group_local = [np.array([local[i] for i in idx]) for idx in valid_groups]
    ranks = np.empty((len(use), len(names)), np.float64)
    for idx in group_local:
        for j in range(len(names)):
            ranks[idx, j] = rankdata(values[use[idx], j])

    observed = (y.astype(np.float64) @ ranks - base) / denominator
    hits = np.zeros(len(names), np.int64)
    batch_size = 100
    for first in range(0, n_permutations, batch_size):
        batch = min(batch_size, n_permutations - first)
        permuted = np.empty((batch, len(use)), np.float64)
        for idx in group_local:
            order = np.argsort(rng.random((batch, len(idx))), axis=1)
            permuted[:, idx] = y[idx][order]
        null_auc = (permuted @ ranks - base) / denominator
        hits += (np.abs(null_auc - 0.5)
                 >= np.abs(observed[None, :] - 0.5) - 1e-15).sum(axis=0)
    p_values = (hits + 1) / (n_permutations + 1)
    q_values = bh_qvalues(p_values)

    result = {}
    for j, name in enumerate(names):
        result[name] = {
            "failure_auc": float(observed[j]),
            "permutation_p_two_sided": float(p_values[j]),
            "bh_q_within_table": float(q_values[j]),
            "failure_mean_mixed_strata": float(values[use, j][y].mean()),
            "success_mean_mixed_strata": float(values[use, j][~y].mean()),
        }
    result["_sample"] = {
        "episodes_in_mixed_strata": int(len(use)),
        "failures_in_mixed_strata": int(y.sum()),
        "successes_in_mixed_strata": int((~y).sum()),
        "mixed_strata": int(len(group_local)),
        "permutations": int(n_permutations),
    }
    return result


def add_behavior_residuals(rows: dict[str, np.ndarray], groups: np.ndarray) -> None:
    for route in ("hard", "soft"):
        for feature in ("lag1", "periodic_prominence", "return_excess"):
            controls = np.column_stack([
                rows[f"action_{feature}"], rows[f"sim_{feature}"]
            ])
            rows[f"{route}_{feature}_behavior_residual"] = residualize(
                rows[f"{route}_{feature}"], controls, groups)


def metric_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    names = sorted({key for row in records for key in row
                    if key not in {"task", "suite", "episode_index", "scene",
                                   "failure", "episode_length", "window",
                                   "window_length"}})
    return {name: np.array([row[name] for row in records], np.float64)
            for name in names}


def summarize_task(records: list[dict], failure: np.ndarray,
                   scene: np.ndarray) -> dict:
    metrics = metric_arrays(records)
    keys = [
        "hard_lag1", "hard_periodic_prominence", "hard_return_excess",
        "soft_lag1", "soft_periodic_prominence", "soft_return_excess",
        "action_periodic_prominence", "sim_periodic_prominence",
        "route_cycle_candidate", "route_state_cycle_candidate",
        "joint_loop_candidate",
    ]
    out = {}
    for key in keys:
        out[key] = {
            "failure_mean": float(metrics[key][failure].mean()),
            "success_mean": float(metrics[key][~failure].mean()),
            "failure_auc_within_scene": stratified_auc(
                metrics[key], failure, scene),
        }
    return out


def paired_summary(early: list[dict], late: list[dict]) -> dict:
    e, l = metric_arrays(early), metric_arrays(late)
    keys = [
        "hard_lag1", "hard_periodic_prominence", "hard_return_excess",
        "soft_lag1", "soft_periodic_prominence", "soft_return_excess",
        "action_periodic_prominence", "sim_periodic_prominence",
        "route_cycle_candidate", "route_state_cycle_candidate",
        "joint_loop_candidate",
    ]
    out = {}
    for key in keys:
        delta = l[key] - e[key]
        if np.allclose(delta, 0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(delta, alternative="two-sided",
                                     method="approx").pvalue)
        out[key] = {
            "early_mean": float(e[key].mean()),
            "late_mean": float(l[key].mean()),
            "late_minus_early_mean": float(delta.mean()),
            "late_minus_early_median": float(np.median(delta)),
            "fraction_increased": float((delta > 0).mean()),
            "wilcoxon_p_two_sided": p_value,
        }
    return out


def candidate_episodes(records: list[dict], key: str) -> list[int]:
    return [int(row["episode_index"]) for row in records if row[key] > 0]


def audit_late_candidates(data: RunData, half: int, late_failure: list[dict]) -> dict:
    success_late: list[dict] = []
    for i in np.flatnonzero(~data.failure):
        if data.n_rows[i] < half:
            continue
        row = window_metrics(data, int(i), int(data.n_rows[i] - half), half)
        row["episode_index"] = int(data.summaries[int(i)]["episode_index"])
        success_late.append(row)

    keys = ("route_cycle_candidate", "route_state_cycle_candidate",
            "joint_loop_candidate")
    calibration = {}
    for key in keys:
        failure_ids = candidate_episodes(late_failure, key)
        success_ids = candidate_episodes(success_late, key)
        table = [
            [len(failure_ids), len(late_failure) - len(failure_ids)],
            [len(success_ids), len(success_late) - len(success_ids)],
        ]
        calibration[key] = {
            "failure_candidates": failure_ids,
            "failure_total": len(late_failure),
            "success_candidates": success_ids,
            "success_total_eligible": len(success_late),
            "fisher_p_one_sided_failure_enrichment": float(
                fisher_exact(table, alternative="greater").pvalue),
        }

    details = []
    for row in late_failure:
        if not row["route_cycle_candidate"]:
            continue
        episode = int(row["episode_index"])
        start = int(data.offsets[episode] + data.n_rows[episode] - half)
        top1 = data.state_top1[start:start + half]
        summary = data.summaries[episode]
        stable_route, stable_joint = [], []
        lower = max(9, half - 8)
        upper = min(int(data.n_rows[episode]), half + 8)
        for length in range(lower, upper + 1, 2):
            metrics = window_metrics(
                data, episode, int(data.n_rows[episode] - length), length)
            if metrics["route_cycle_candidate"]:
                stable_route.append(length)
            if metrics["joint_loop_candidate"]:
                stable_joint.append(length)
        details.append({
            "episode_index": episode,
            "init_state_id": int(summary["init_state_id"]),
            "flow_noise_seed": int(summary["flow_noise_seed"]),
            "half_window": half,
            "hard_return_excess": float(row["hard_return_excess"]),
            "hard_return_period": int(row["hard_return_period"]),
            "soft_return_excess": float(row["soft_return_excess"]),
            "soft_return_period": int(row["soft_return_period"]),
            "action_return_excess": float(row["action_return_excess"]),
            "sim_return_excess": float(row["sim_return_excess"]),
            "state_token_top1_lag1_equality": float(
                (top1[1:] == top1[:-1]).mean()),
            "state_token_top1_lag2_equality": float(
                (top1[2:] == top1[:-2]).mean()),
            "terminal_window_lengths_route_positive": stable_route,
            "terminal_window_lengths_joint_positive": stable_joint,
        })
    return {"calibration": calibration, "details": details}


STAT_KEYS = [
    "hard_lag1",
    "hard_periodic_prominence",
    "hard_return_excess",
    "soft_lag1",
    "soft_periodic_prominence",
    "soft_return_excess",
    "route_cycle_candidate",
    "route_state_cycle_candidate",
    "joint_loop_candidate",
    "action_lag1",
    "action_periodic_prominence",
    "action_return_excess",
    "sim_lag1",
    "sim_periodic_prominence",
    "sim_return_excess",
    "hard_lag1_behavior_residual",
    "hard_periodic_prominence_behavior_residual",
    "hard_return_excess_behavior_residual",
    "soft_lag1_behavior_residual",
    "soft_periodic_prominence_behavior_residual",
    "soft_return_excess_behavior_residual",
]


LABELS = {
    "hard_lag1": "HB actual top-4 lag-1 overlap",
    "hard_periodic_prominence": "HB actual top-4 periodic prominence",
    "hard_return_excess": "HB actual top-4 return above lag 1",
    "soft_lag1": "HB soft route lag-1 similarity",
    "soft_periodic_prominence": "HB soft route periodic prominence",
    "soft_return_excess": "HB soft route return above lag 1",
    "route_cycle_candidate": "hard+soft aligned route-cycle candidate",
    "route_state_cycle_candidate": "route cycle + aligned physical return",
    "joint_loop_candidate": "route+state+action aligned loop candidate",
    "action_lag1": "action-chunk lag-1 similarity",
    "action_periodic_prominence": "action-chunk periodic prominence",
    "action_return_excess": "action-chunk return above lag 1",
    "sim_lag1": "sim-state lag-1 similarity",
    "sim_periodic_prominence": "sim-state periodic prominence",
    "sim_return_excess": "sim-state return above lag 1",
    "hard_lag1_behavior_residual": "top-4 lag-1 residual after behavior",
    "hard_periodic_prominence_behavior_residual": "top-4 periodic residual after behavior",
    "hard_return_excess_behavior_residual": "top-4 return residual after behavior",
    "soft_lag1_behavior_residual": "soft lag-1 residual after behavior",
    "soft_periodic_prominence_behavior_residual": "soft periodic residual after behavior",
    "soft_return_excess_behavior_residual": "soft return residual after behavior",
}


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_report(result: dict) -> str:
    prefix = result["combined"]["prefix"]["stats"]
    suffix = result["combined"]["suffix"]["stats"]
    failure_total = sum(task["failures"] for task in result["tasks"].values())
    route_candidates = sum(
        len(task.get("late_failure_candidates", {}).get(
            "route_cycle_candidate", []))
        for task in result["tasks"].values())
    lines = [
        "# Failed-rollout MoE periodicity audit",
        "",
        "Data source: `VLA_MUI_HUB/cache/HiMoE-VLA`, run `right-16x32`.",
        "The source stores were opened read-only. AUC > 0.5 means the metric is",
        "larger on failures after comparing only within task x init-state strata.",
        "",
        "## Bottom line",
        "",
        "The available data support **failure-associated route persistence**, not a",
        "general periodic MoE loop. In the equal-prefix test, failure AUC is "
        f"{prefix['hard_lag1']['failure_auc']:.3f} for actual top-4 lag-1 overlap "
        f"and {prefix['soft_lag1']['failure_auc']:.3f} for soft-route lag-1 "
        "similarity. Both become still stronger in terminal windows "
        f"({suffix['hard_lag1']['failure_auc']:.3f} and "
        f"{suffix['soft_lag1']['failure_auc']:.3f}).",
        "",
        "Periodic prominence is lower on failures in the primary equal-prefix test "
        f"(AUC {prefix['hard_periodic_prominence']['failure_auc']:.3f} hard, "
        f"{prefix['soft_periodic_prominence']['failure_auc']:.3f} soft), and no "
        "equal-prefix or same-length terminal episode passes the aligned hard+soft "
        "cycle criterion. An exploratory failure-half scan finds only "
        f"{route_candidates}/{failure_total} route-cycle candidates, all in one "
        "task; candidate details and matched success calibration are below.",
        "",
        "## Coverage",
        "",
        "| task | episodes | failures | equal prefix | tested periods | AS gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for task in result["tasks"].values():
        lines.append(
            f"| {task['suite']}/{task['task']} | {task['episodes']} | "
            f"{task['failures']} | {task['prefix_length']} | "
            f"2-{task['max_period']} | "
            f"{'constant' if task['as_constant'] else 'variable'} |")

    selected = [
        "hard_lag1", "hard_periodic_prominence", "hard_return_excess",
        "soft_lag1", "soft_periodic_prominence", "soft_return_excess",
        "route_cycle_candidate", "route_state_cycle_candidate",
        "joint_loop_candidate",
        "hard_lag1_behavior_residual",
        "hard_periodic_prominence_behavior_residual",
        "soft_lag1_behavior_residual",
        "soft_periodic_prominence_behavior_residual",
    ]
    for window, title in (("prefix", "Equal-prefix primary test"),
                          ("suffix", "Same-length terminal-window test")):
        stats = result["combined"][window]["stats"]
        lines += [
            "",
            f"## {title}",
            "",
            "| metric | failure AUC | permutation p | BH q | raw failure mean | raw success mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for key in selected:
            row = stats[key]
            lines.append(
                f"| {LABELS[key]} | {fmt(row['failure_auc'])} | "
                f"{fmt(row['permutation_p_two_sided'], 4)} | "
                f"{fmt(row['bh_q_within_table'], 4)} | "
                f"{fmt(row['failure_mean_mixed_strata'], 4)} | "
                f"{fmt(row['success_mean_mixed_strata'], 4)} |")
        sample = stats["_sample"]
        lines += [
            "",
            f"Mixed-stratum sample: {sample['failures_in_mixed_strata']} failures + "
            f"{sample['successes_in_mixed_strata']} successes in "
            f"{sample['mixed_strata']} task-scene strata; "
            f"{sample['permutations']} within-stratum permutations.",
            "Raw means are pooled without stratification and can disagree in direction",
            "with the task-scene-stratified AUC when task scales differ.",
        ]

    lines += [
        "",
        "## Failed episodes: early vs late",
        "",
        "Each failed episode is split into two non-overlapping equal halves. Positive",
        "delta means the metric increased near timeout.",
        "",
        "| task | half window | top-4 lag1 delta (p) | top-4 periodic delta (p) | soft periodic delta (p) |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in result["tasks"].values():
        if not task["failures"]:
            continue
        pair = task["failure_early_late"]
        a = pair["hard_lag1"]
        b = pair["hard_periodic_prominence"]
        c = pair["soft_periodic_prominence"]
        lines.append(
            f"| {task['suite']}/{task['task']} | {task['failure_half_length']} | "
            f"{fmt(a['late_minus_early_mean'], 4)} ({fmt(a['wilcoxon_p_two_sided'], 4)}) | "
            f"{fmt(b['late_minus_early_mean'], 4)} ({fmt(b['wilcoxon_p_two_sided'], 4)}) | "
            f"{fmt(c['late_minus_early_mean'], 4)} ({fmt(c['wilcoxon_p_two_sided'], 4)}) |")

    lines += [
        "",
        "High-specificity aligned candidates in the late failure half:",
        "",
        "| task | route only | route + physical state | route + state + action |",
        "|---|---|---|---|",
    ]
    for task in result["tasks"].values():
        if not task["failures"]:
            continue
        audit = task["late_failure_candidate_audit"]["calibration"]
        cells = []
        for key in ("route_cycle_candidate", "route_state_cycle_candidate",
                    "joint_loop_candidate"):
            item = audit[key]
            cells.append(
                f"{item['failure_candidates']} "
                f"({len(item['failure_candidates'])}/{item['failure_total']}) "
                f"vs success {item['success_candidates']} "
                f"({len(item['success_candidates'])}/"
                f"{item['success_total_eligible']}) "
                f"(p={item['fisher_p_one_sided_failure_enrichment']:.4f})")
        lines.append(
            f"| {task['suite']}/{task['task']} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} |")

    lines += [
        "",
        "Candidate detail (`top1` equality is over the eight HB layers on the",
        "state token; stability lists are terminal window lengths that remain",
        "positive):",
        "",
        "| episode | scene / noise | hard return | soft return | top1 lag1 -> lag2 | route-stable windows | joint-stable windows |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for task in result["tasks"].values():
        audit = task.get("late_failure_candidate_audit")
        if not audit:
            continue
        for item in audit["details"]:
            lines.append(
                f"| {item['episode_index']} | {item['init_state_id']} / "
                f"{item['flow_noise_seed']} | {item['hard_return_excess']:.4f} | "
                f"{item['soft_return_excess']:.4f} | "
                f"{item['state_token_top1_lag1_equality']:.3f} -> "
                f"{item['state_token_top1_lag2_equality']:.3f} | "
                f"{item['terminal_window_lengths_route_positive']} | "
                f"{item['terminal_window_lengths_joint_positive']} |")

    lines += [
        "",
        "## Interpretation constraints",
        "",
        "- `periodic prominence` is a local autocorrelation peak at lag 2-8; the",
        "  maximum lag is shortened when an episode window cannot show three cycles.",
        "- `return above lag 1` must be positive for a route to resemble a past",
        "  state more than its immediately previous state. The aligned candidate",
        "  flags require agreement between actual top-4 and soft routing, then",
        "  optionally simulator state and action.",
        "- The terminal comparison is descriptive: successful and failed episodes",
        "  end in different task phases. The equal-prefix test is the primary one.",
        "- Association is not causal. Routing and outcome can both reflect the same",
        "  physical trajectory; the behavior-residual rows are only a control for",
        "  the measured action and simulator recurrence.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int,
                        default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()
    rng = np.random.default_rng(20260825)

    summary_paths = sorted(MODEL_ROOT.glob(
        "libero_*/**/right-16x32/client/summaries.json"))
    if not summary_paths:
        raise FileNotFoundError(f"no right-16x32 runs below {MODEL_ROOT}")

    task_results: dict[str, dict] = {}
    prefix_records: list[dict] = []
    suffix_records: list[dict] = []
    csv_rows: list[dict] = []

    for summary_path in summary_paths:
        data = load_run(summary_path)
        prefix_length = int(data.n_rows.min())
        max_period = min(MAX_PERIOD, (prefix_length - 1) // 3)
        task_prefix: list[dict] = []
        task_suffix: list[dict] = []
        for i, summary in enumerate(data.summaries):
            common = {
                "task": data.task,
                "suite": data.suite,
                "episode_index": int(summary["episode_index"]),
                "scene": int(data.scene[i]),
                "failure": bool(data.failure[i]),
                "episode_length": int(data.n_rows[i]),
                "window_length": prefix_length,
            }
            pre = common | {"window": "prefix"} | window_metrics(
                data, i, 0, prefix_length)
            start = int(data.n_rows[i] - prefix_length)
            post = common | {"window": "suffix"} | window_metrics(
                data, i, start, prefix_length)
            task_prefix.append(pre)
            task_suffix.append(post)
            prefix_records.append(pre)
            suffix_records.append(post)
            csv_rows.extend([pre, post])

        result = {
            "suite": data.suite,
            "task": data.task,
            "episodes": len(data.summaries),
            "successes": int((~data.failure).sum()),
            "failures": int(data.failure.sum()),
            "prefix_length": prefix_length,
            "max_period": max_period,
            "as_constant": data.as_constant,
        }
        if data.failure.any():
            result["prefix"] = summarize_task(
                task_prefix, data.failure, data.scene)
            result["suffix"] = summarize_task(
                task_suffix, data.failure, data.scene)
            half = int(data.n_rows[data.failure].min() // 2)
            early: list[dict] = []
            late: list[dict] = []
            for i in np.flatnonzero(data.failure):
                summary = data.summaries[int(i)]
                common = {
                    "task": data.task,
                    "suite": data.suite,
                    "episode_index": int(summary["episode_index"]),
                    "scene": int(data.scene[i]),
                    "failure": True,
                    "episode_length": int(data.n_rows[i]),
                    "window_length": half,
                }
                early_row = common | {"window": "failure_early"} | window_metrics(
                    data, int(i), 0, half)
                late_row = common | {"window": "failure_late"} | window_metrics(
                    data, int(i), int(data.n_rows[i] - half), half)
                early.append(early_row)
                late.append(late_row)
                csv_rows.extend([early_row, late_row])
            result["failure_half_length"] = half
            result["failure_early_late"] = paired_summary(early, late)
            result["late_failure_candidates"] = {
                key: candidate_episodes(late, key)
                for key in ("route_cycle_candidate",
                            "route_state_cycle_candidate",
                            "joint_loop_candidate")
            }
            result["late_failure_candidate_audit"] = audit_late_candidates(
                data, half, late)
        task_results[f"{data.suite}/{data.task}"] = result
        print(f"loaded {data.suite}/{data.task}: {len(data.summaries)} episodes, "
              f"{data.failure.sum()} failures, prefix={prefix_length}")

    def combined(records: list[dict]) -> dict:
        informative = [row for row in records
                       if task_results[f"{row['suite']}/{row['task']}"]["failures"]]
        metrics = metric_arrays(informative)
        failure = np.array([row["failure"] for row in informative], bool)
        groups = np.array([
            f"{row['suite']}/{row['task']}|{row['scene']}" for row in informative
        ])
        add_behavior_residuals(metrics, groups)
        chosen = {key: metrics[key] for key in STAT_KEYS}
        return {
            "stats": stratified_permutation_stats(
                chosen, failure, groups, args.permutations, rng)
        }

    result = {
        "schema": "failure-rollout-periodicity/1",
        "source": str(MODEL_ROOT),
        "run_id": "right-16x32",
        "definitions": {
            "failure_auc": "AUC with failure as positive, stratified by task x init_state",
            "primary_window": "longest equal prefix shared by every episode in each task",
            "period_range": "2..min(8, floor((window_length-1)/3)) control queries",
            "hard_route": "actual top-4 overlap across 8 HB layers x 10 denoise x 11 tokens",
            "soft_route": "standardized HB router probabilities at denoise=0, state token",
            "behavior_residual": "within-stratum OLS residual after action and sim-state counterpart",
        },
        "tasks": task_results,
        "combined": {
            "prefix": combined(prefix_records),
            "suffix": combined(suffix_records),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    (args.output_dir / "report.md").write_text(render_report(result))
    fieldnames = sorted({key for row in csv_rows for key in row})
    with (args.output_dir / "episode_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {args.output_dir / 'summary.json'}")
    print(f"wrote {args.output_dir / 'report.md'}")
    print(f"wrote {args.output_dir / 'episode_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
