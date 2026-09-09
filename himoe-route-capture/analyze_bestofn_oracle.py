#!/usr/bin/env python3
"""Cross-fitted oracle-headroom analysis for terminal Best-of-N continuations."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from behavior_forks_v2 import atomic_json, load_npz, verify_artifact
from bestofn_protocol import (
    CANDIDATE_SCHEMA,
    CONTINUATION_SCHEMA,
    SUMMARY_SCHEMA,
    BestOfNError,
    candidate_dir,
    continuation_dir,
    load_config,
    load_json,
    rng_from_words,
    seed_words,
    validate_plan,
)


def candidate_subsets(
    candidate_count: int,
    subset_size: int,
    *,
    maximum: int,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    if not 1 <= subset_size <= candidate_count:
        raise ValueError("subset_size must lie in [1,candidate_count]")
    values = list(itertools.combinations(range(candidate_count), subset_size))
    if len(values) <= maximum:
        return values
    selected = np.sort(rng.choice(len(values), size=maximum, replace=False))
    return [values[int(index)] for index in selected]


def cross_fitted_snapshot(
    outcomes: np.ndarray,
    candidate_counts: Sequence[int],
    *,
    maximum_subsets: int,
    rng: np.random.Generator,
) -> dict[int, dict[str, Any]]:
    success = np.asarray(outcomes, dtype=np.float64)
    if success.ndim != 2 or success.shape[0] < 2 or success.shape[1] < 4:
        raise ValueError("outcomes must have shape [K,R] with K>=2 and R>=4")
    if success.shape[1] % 2:
        raise ValueError("cross-fitting requires an even repeat count")
    if np.any((success != 0.0) & (success != 1.0)):
        raise ValueError("terminal outcomes must be binary")
    half = success.shape[1] // 2
    q_a = success[:, :half].mean(axis=1)
    q_b = success[:, half:].mean(axis=1)
    q_all = success.mean(axis=1)
    result: dict[int, dict[str, Any]] = {}
    for n_raw in candidate_counts:
        n = int(n_raw)
        if n > success.shape[0]:
            continue
        subsets = candidate_subsets(
            success.shape[0], n, maximum=maximum_subsets, rng=rng
        )
        gains = []
        selected_agree = []
        naive = []
        for subset_raw in subsets:
            subset = np.asarray(subset_raw, dtype=np.int64)
            selected_a = int(subset[int(np.argmax(q_a[subset]))])
            selected_b = int(subset[int(np.argmax(q_b[subset]))])
            gain_ab = float(q_b[selected_a] - q_b[subset].mean())
            gain_ba = float(q_a[selected_b] - q_a[subset].mean())
            gains.append(0.5 * (gain_ab + gain_ba))
            selected_agree.append(selected_a == selected_b)
            naive.append(float(q_all[subset].max() - q_all[subset].mean()))
        result[n] = {
            "cross_fitted_headroom": float(np.mean(gains)),
            "naive_in_sample_headroom": float(np.mean(naive)),
            "selection_agreement": float(np.mean(selected_agree)),
            "subsets": len(subsets),
            "candidate_q_range": float(q_all.max() - q_all.min()),
        }
    return result


def _load_outcomes(
    run_root: Path, state: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    state_id = str(state["state_id"])
    candidate_artifact = candidate_dir(run_root, state_id) / "artifact.json"
    candidate_meta = verify_artifact(candidate_artifact, CANDIDATE_SCHEMA)
    k = int(candidate_meta["candidate_count"])
    if k != int(state["candidate_count"]):
        raise BestOfNError(f"candidate count differs from plan for {state_id}")
    root = continuation_dir(run_root, state_id)
    descriptors = sorted(root.glob("repeats_*/artifact.json"))
    if not descriptors:
        raise BestOfNError(f"no terminal continuation shards for {state_id}")
    chunks: list[tuple[int, int, np.ndarray]] = []
    for descriptor in descriptors:
        metadata = verify_artifact(descriptor, CONTINUATION_SCHEMA)
        arrays = load_npz(descriptor.parent / metadata["files"]["data"]["file"])
        outcomes = np.asarray(arrays["terminal_success"], dtype=np.bool_)
        start, stop = int(metadata["repeat_start"]), int(metadata["repeat_stop"])
        if outcomes.shape != (k, stop - start):
            raise BestOfNError(f"terminal outcome shape mismatch in {descriptor}")
        chunks.append((start, stop, outcomes))
    chunks.sort(key=lambda value: value[0])
    expected = 0
    values = []
    for start, stop, outcomes in chunks:
        if start != expected:
            raise BestOfNError(f"continuation repeat shards have a gap for {state_id}")
        expected = stop
        values.append(outcomes)
    joined = np.concatenate(values, axis=1)
    if joined.shape[1] < 4 or joined.shape[1] % 2:
        raise BestOfNError(f"{state_id} does not have an even R>=4 continuation panel")
    return joined, candidate_meta


def hierarchical_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    rng: np.random.Generator,
) -> list[float]:
    by_task: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(float(row["cross_fitted_headroom"]))
    tasks = sorted(by_task)
    if not tasks:
        raise ValueError("bootstrap requires at least one task")
    distribution = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        task_values = []
        for task_index in rng.integers(0, len(tasks), size=len(tasks)):
            values = np.asarray(by_task[tasks[int(task_index)]], dtype=np.float64)
            sampled = values[rng.integers(0, len(values), size=len(values))]
            task_values.append(float(sampled.mean()))
        distribution[draw] = float(np.mean(task_values))
    return [float(value) for value in np.percentile(distribution, [2.5, 97.5])]


def analyze(
    config_path: Path,
    plan_path: Path,
    run_root: Path,
    *,
    bootstrap_draws: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    plan = load_json(plan_path)
    validate_plan(plan, config)
    maximum_subsets = int(config["oracle"]["maximum_subsets_per_snapshot"])
    candidate_counts = [int(value) for value in config["oracle"]["candidate_counts"]]
    master_seed = int(config["master_seed"])
    rows_by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    topup_states = []
    initial_repeats = int(config["continuation"]["initial_repeats"])
    topup_repeats = int(config["continuation"]["topup_repeats"])
    for state in plan["states"]:
        outcomes, candidate_meta = _load_outcomes(run_root, state)
        rng = rng_from_words(
            seed_words(
                master_seed,
                "bestofn/oracle/subsets",
                int(state["task_id"]),
                int(state["episode"]),
                int(state["fork_step"]),
            )
        )
        results = cross_fitted_snapshot(
            outcomes,
            candidate_counts,
            maximum_subsets=maximum_subsets,
            rng=rng,
        )
        for n, values in results.items():
            rows_by_n[n].append(
                {
                    "state_id": state["state_id"],
                    "task_id": int(state["task_id"]),
                    "episode": int(state["episode"]),
                    "phase": state["phase"],
                    "split": state["split"],
                    "candidate_count": int(candidate_meta["candidate_count"]),
                    "repeats": int(outcomes.shape[1]),
                    **values,
                }
            )
        if outcomes.shape[1] == initial_repeats and initial_repeats < topup_repeats:
            n8 = results[8]
            q = outcomes.mean(axis=1)
            needs_topup = bool(
                n8["selection_agreement"] < 1.0
                or np.any((q > 0.0) & (q < 1.0))
                or (q.max() - q.min()) > 0.0
            )
            if needs_topup:
                topup_states.append(str(state["state_id"]))

    draws = (
        int(config["oracle"]["bootstrap_draws"])
        if bootstrap_draws is None
        else int(bootstrap_draws)
    )
    if draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    aggregates: dict[str, Any] = {}
    for n in candidate_counts:
        rows = rows_by_n.get(n, [])
        if not rows:
            continue
        task_values = {
            str(task): float(
                np.mean([row["cross_fitted_headroom"] for row in rows if row["task_id"] == task])
            )
            for task in sorted({int(row["task_id"]) for row in rows})
        }
        point = float(np.mean(list(task_values.values())))
        rng = rng_from_words(
            seed_words(master_seed, "bestofn/oracle/bootstrap", n, 0, 0)
        )
        ci = hierarchical_bootstrap(rows, draws=draws, rng=rng)
        aggregates[str(n)] = {
            "snapshots": len(rows),
            "tasks": len(task_values),
            "task_macro_cross_fitted_headroom": point,
            "hierarchical_bootstrap_ci95": ci,
            "task_values": task_values,
            "informative_snapshot_fraction": float(
                np.mean([row["candidate_q_range"] > 0.0 for row in rows])
            ),
            "mean_selection_agreement": float(
                np.mean([row["selection_agreement"] for row in rows])
            ),
            "mean_naive_in_sample_headroom": float(
                np.mean([row["naive_in_sample_headroom"] for row in rows])
            ),
        }
    gate_n = str(int(config["oracle"]["gate_candidate_count"]))
    gate_row = aggregates[gate_n]
    gate_passed = bool(
        gate_row["task_macro_cross_fitted_headroom"] > 0.0
        and gate_row["hierarchical_bootstrap_ci95"][0] > 0.0
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "confirmatory": False,
        "estimand": (
            "one selected H=10 chunk followed by frozen base-HiMoE continuation to "
            "success or the 300-environment-step budget"
        ),
        "config_file": str(config_path),
        "plan_file": str(plan_path),
        "run_root": str(run_root),
        "cross_fit": "first repeat half selects/second evaluates, then swap and average",
        "aggregates": aggregates,
        "gate_1": {
            "candidate_count": int(gate_n),
            "passed": gate_passed,
            "rule": "point estimate > 0 and hierarchical 95% CI lower bound > 0",
            "critic_training_authorized": gate_passed,
        },
        "snapshot_rows": {str(n): rows for n, rows in sorted(rows_by_n.items())},
    }
    topup = {
        "schema": "himoe.bestofn.topup_plan.v1",
        "initial_repeats": initial_repeats,
        "target_repeats": topup_repeats,
        "selection_rule": (
            "A/B selector disagreement, non-degenerate Bernoulli q, or candidate q-range > 0"
        ),
        "states": sorted(topup_states),
    }
    return summary, topup


def render_report(summary: Mapping[str, Any], topup: Mapping[str, Any]) -> str:
    lines = [
        "# Counterfactual terminal-Q oracle gate",
        "",
        "> New Best-of-N experiment; no legacy candidate or outcome data are used.",
        "",
        "The estimand is one selected H=10 proposal followed by the frozen base HiMoE-VLA policy to terminal success or the 300-step LIBERO budget. It is not yet closed-loop Q_N.",
        "",
        "| N | snapshots | cross-fitted headroom | hierarchical 95% CI | informative | A/B agreement | naive in-sample |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, row in summary["aggregates"].items():
        ci = row["hierarchical_bootstrap_ci95"]
        lines.append(
            "| %s | %d | %+.4f | [%+.4f, %+.4f] | %.1f%% | %.1f%% | %+.4f |"
            % (
                n,
                row["snapshots"],
                row["task_macro_cross_fitted_headroom"],
                ci[0],
                ci[1],
                100.0 * row["informative_snapshot_fraction"],
                100.0 * row["mean_selection_agreement"],
                row["mean_naive_in_sample_headroom"],
            )
        )
    gate = summary["gate_1"]
    lines.extend(
        [
            "",
            "## Gate 1",
            "",
            "- Decision: **%s**." % ("PASS" if gate["passed"] else "STOP"),
            "- Critic training authorized: `%s`." % str(gate["critic_training_authorized"]).lower(),
            "- Adaptive continuation top-up states: `%d`." % len(topup["states"]),
            "- Inference units are task and snapshot; candidate pairs/subsets are never treated as independent samples.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("bestofn_experiment_config.json"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, topup = analyze(
        args.config.expanduser().resolve(),
        args.plan.expanduser().resolve(),
        args.run_root.expanduser().resolve(),
        bootstrap_draws=args.bootstrap,
    )
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "topup_plan.json", topup)
    (out / "REPORT.md").write_text(render_report(summary, topup), encoding="utf-8")
    print(
        "Gate 1 %s; critic_training_authorized=%s"
        % ("PASS" if summary["gate_1"]["passed"] else "STOP", summary["gate_1"]["critic_training_authorized"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
