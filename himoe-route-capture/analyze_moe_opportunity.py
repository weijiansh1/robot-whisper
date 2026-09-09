#!/usr/bin/env python3
"""Audit whether first-query MoE routing predicts candidate-pool opportunity.

The five available 16 x 32 grids provide one full rollout for every
``initial-state x flow-noise-seed`` cell.  They do not provide repeated common-
random-number continuations from a candidate chunk.  Accordingly, this script
uses empirical full-rollout success headroom as a *shadow* target and never
calls it a chunk value estimate.

Inference treats the initial-state pool as the row and preserves task strata.
The primary association is the within-task macro Spearman correlation between
full action-token route dispersion and empirical success headroom.  A
task-stratified permutation test and hierarchical task/state bootstrap are
reported.  Small fixed-ridge task-held-out probes are secondary diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.stats import rankdata

from route_noise_selector import normalize_router_probabilities, pairwise_hellinger


HERE = Path(__file__).resolve().parent
DEFAULT_HUB = HERE.parent / "VLA_MUI_HUB"
DEFAULT_OUT = HERE / "analysis" / "moe-opportunity"
N_EXPERTS = 32


@dataclass(frozen=True)
class StatePool:
    task: str
    suite: str
    state: int
    success_rate: float
    mixed_outcome: bool
    empirical_headroom: float
    outcome_heterogeneity: float
    action_features: np.ndarray
    state_features: np.ndarray
    diagnostics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-root", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def discover_runs(hub_root: Path) -> list[Path]:
    cache = hub_root / "cache" / "HiMoE-VLA"
    runs = sorted(
        path.parent.parent
        for path in cache.glob("**/right-16x32/server/routes.zarr")
        if (path.parent.parent / "client" / "summaries.json").exists()
    )
    if len(runs) != 5:
        raise RuntimeError(f"expected five complete right-16x32 runs, found {len(runs)}")
    return runs


def first_rows(episode_ids: np.ndarray, expected: np.ndarray) -> np.ndarray:
    unique, first = np.unique(episode_ids, return_index=True)
    lookup = {int(episode): int(row) for episode, row in zip(unique, first)}
    missing = sorted(set(map(int, expected)) - set(lookup))
    if missing:
        raise RuntimeError(f"route store is missing episodes: {missing[:8]}")
    return np.asarray([lookup[int(episode)] for episode in expected], dtype=np.int64)


def mean_upper(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or len(values) < 2:
        raise ValueError("distance must be a square matrix")
    return float(values[np.triu_indices(len(values), 1)].mean())


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = normalize_router_probabilities(probabilities)
    return -np.sum(values * np.log(np.maximum(values, 1e-300)), axis=-1) / math.log(
        N_EXPERTS
    )


def top4_mass(probabilities: np.ndarray) -> np.ndarray:
    values = normalize_router_probabilities(probabilities)
    return np.partition(values, -4, axis=-1)[..., -4:].sum(axis=-1)


def top2_margin(probabilities: np.ndarray) -> np.ndarray:
    values = normalize_router_probabilities(probabilities)
    top2 = np.partition(values, -2, axis=-1)[..., -2:]
    return top2.max(axis=-1) - top2.min(axis=-1)


def pool_features(routes: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return action-route, state-route, and contract diagnostics for one K32 pool."""

    values = normalize_router_probabilities(routes)
    if values.shape != (32, 8, 10, 11, N_EXPERTS):
        raise ValueError(f"unexpected K32 route shape: {values.shape}")
    action = values[:, :, :, 1:, :]
    state_token = values[:, :, :, 0, :]

    d0 = mean_upper(pairwise_hellinger(action[:, :, :1]))
    d03 = mean_upper(pairwise_hellinger(action[:, :, :3]))
    full = mean_upper(pairwise_hellinger(action))
    action_entropy_d0 = float(normalized_entropy(action[:, :, :1]).mean())
    action_entropy_full = float(normalized_entropy(action).mean())
    action_features = np.asarray(
        [d0, d03, full, action_entropy_d0, action_entropy_full], dtype=np.float64
    )

    # State-token routing is candidate invariant in this checkpoint.  Average over
    # candidates and denoise rounds, then retain layer identity for the probe.
    state_mean = state_token.mean(axis=(0, 2))
    state_features = np.concatenate(
        [
            normalized_entropy(state_mean),
            top4_mass(state_mean),
            top2_margin(state_mean),
        ]
    ).astype(np.float64)
    state_variation = float(pairwise_hellinger(state_token).max())
    return action_features, state_features, {
        "state_token_candidate_max_hellinger": state_variation,
        "action_route_dispersion_d0": d0,
        "action_route_dispersion_d0_d2": d03,
        "action_route_dispersion_full": full,
    }


def load_pools(hub_root: Path) -> tuple[list[StatePool], dict[str, Any]]:
    cache = hub_root / "cache" / "HiMoE-VLA"
    pools: list[StatePool] = []
    task_audit: dict[str, Any] = {}
    canonical_seeds: tuple[int, ...] | None = None
    for run in discover_runs(hub_root):
        summaries = sorted(
            json.loads((run / "client" / "summaries.json").read_text()),
            key=lambda row: int(row["episode_index"]),
        )
        if len(summaries) != 512:
            raise RuntimeError(f"{run}: expected 512 rollout summaries")
        episodes = np.asarray([int(row["episode_index"]) for row in summaries])
        states = np.asarray([int(row["init_state_id"]) for row in summaries])
        seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries])
        success = np.asarray([bool(row["success"]) for row in summaries])
        route_group = zarr.open_group(str(run / "server" / "routes.zarr"), mode="r")
        rows = first_rows(np.asarray(route_group["episode_id"][:]), episodes)
        task = str(run.relative_to(cache).parent)
        suite = task.split("/", 1)[0]
        mixed = 0
        task_successes = int(success.sum())
        for state in np.sort(np.unique(states)):
            indices = np.flatnonzero(states == state)
            indices = indices[np.argsort(seeds[indices])]
            ordered_seeds = tuple(map(int, seeds[indices]))
            if len(indices) != 32 or len(set(ordered_seeds)) != 32:
                raise RuntimeError(f"{task} state {state} is not a complete K32 pool")
            if canonical_seeds is None:
                canonical_seeds = ordered_seeds
            elif canonical_seeds != ordered_seeds:
                raise RuntimeError("the common 32-seed grid changed across tasks/states")
            raw = np.asarray(
                route_group["hb_router_probs"].oindex[rows[indices], :, :, :, :],
                dtype=np.float32,
            )
            action_features, state_features, diagnostics = pool_features(raw)
            outcome = success[indices].astype(np.float64)
            rate = float(outcome.mean())
            is_mixed = bool(0.0 < rate < 1.0)
            mixed += int(is_mixed)
            pools.append(
                StatePool(
                    task=task,
                    suite=suite,
                    state=int(state),
                    success_rate=rate,
                    mixed_outcome=is_mixed,
                    empirical_headroom=float(outcome.max() - rate),
                    outcome_heterogeneity=float(4.0 * rate * (1.0 - rate)),
                    action_features=action_features,
                    state_features=state_features,
                    diagnostics=diagnostics,
                )
            )
        task_audit[task] = {
            "states": 16,
            "rollouts": 512,
            "successes": task_successes,
            "mixed_states": mixed,
        }
    return pools, {
        "tasks": task_audit,
        "total_states": len(pools),
        "total_rollouts": sum(row["rollouts"] for row in task_audit.values()),
        "mixed_states": sum(row["mixed_states"] for row in task_audit.values()),
        "common_seed_count": len(canonical_seeds or ()),
    }


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = rankdata(np.asarray(left, dtype=np.float64))
    b = rankdata(np.asarray(right, dtype=np.float64))
    a -= a.mean()
    b -= b.mean()
    scale = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / scale) if scale > 1e-20 else float("nan")


def task_macro_spearman(
    score: np.ndarray, target: np.ndarray, tasks: np.ndarray
) -> tuple[float, dict[str, float]]:
    per_task: dict[str, float] = {}
    for task in np.unique(tasks):
        index = tasks == task
        value = spearman(score[index], target[index])
        if np.isfinite(value):
            per_task[str(task)] = float(value)
    if not per_task:
        return float("nan"), per_task
    return float(np.mean(tuple(per_task.values()))), per_task


def stratified_permutation_p(
    score: np.ndarray,
    target: np.ndarray,
    tasks: np.ndarray,
    observed: float,
    draws: int,
    rng: np.random.Generator,
) -> float:
    exceed = 0
    for _ in range(draws):
        shuffled = target.copy()
        for task in np.unique(tasks):
            index = np.flatnonzero(tasks == task)
            shuffled[index] = shuffled[rng.permutation(index)]
        value, _ = task_macro_spearman(score, shuffled, tasks)
        exceed += int(np.isfinite(value) and abs(value) >= abs(observed))
    return float((exceed + 1) / (draws + 1))


def hierarchical_bootstrap_ci(
    score: np.ndarray,
    target: np.ndarray,
    tasks: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> list[float]:
    unique_tasks = np.unique(tasks)
    estimates = []
    for _ in range(draws):
        correlations = []
        for task in rng.choice(unique_tasks, size=len(unique_tasks), replace=True):
            source = np.flatnonzero(tasks == task)
            sampled = rng.choice(source, size=len(source), replace=True)
            value = spearman(score[sampled], target[sampled])
            if np.isfinite(value):
                correlations.append(value)
        if correlations:
            estimates.append(float(np.mean(correlations)))
    if not estimates:
        return [float("nan"), float("nan")]
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def task_held_out_ridge(
    features: np.ndarray,
    target: np.ndarray,
    tasks: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    values = np.asarray(features, dtype=np.float64)
    predictions = np.empty(len(values), dtype=np.float64)
    for held in np.unique(tasks):
        train = tasks != held
        test = ~train
        mean = values[train].mean(axis=0)
        scale = np.maximum(values[train].std(axis=0), 1e-12)
        x_train = (values[train] - mean) / scale
        x_test = (values[test] - mean) / scale
        design = np.column_stack([np.ones(train.sum()), x_train])
        penalty = alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ target[train])
        predictions[test] = np.column_stack([np.ones(test.sum()), x_test]) @ beta
    residual = float(np.sum(np.square(target - predictions)))
    total = float(np.sum(np.square(target - target.mean())))
    return {
        "r2": float(1.0 - residual / max(total, 1e-20)),
        "spearman": spearman(predictions, target),
    }


def analyze(
    pools: list[StatePool],
    audit: dict[str, Any],
    bootstrap: int,
    permutations: int,
    ridge: float,
    seed: int,
) -> dict[str, Any]:
    tasks = np.asarray([pool.task for pool in pools], dtype=object)
    target = np.asarray([pool.empirical_headroom for pool in pools], dtype=np.float64)
    heterogeneity = np.asarray(
        [pool.outcome_heterogeneity for pool in pools], dtype=np.float64
    )
    action = np.stack([pool.action_features for pool in pools])
    state = np.stack([pool.state_features for pool in pools])
    rng = np.random.default_rng(seed)

    feature_names = (
        "action_route_dispersion_d0",
        "action_route_dispersion_d0_d2",
        "action_route_dispersion_full",
        "action_route_entropy_d0",
        "action_route_entropy_full",
    )
    associations: dict[str, Any] = {}
    for axis, name in enumerate(feature_names):
        observed, per_task = task_macro_spearman(action[:, axis], target, tasks)
        associations[name] = {
            "task_macro_spearman_with_empirical_headroom": observed,
            "per_task_spearman": per_task,
            "hierarchical_bootstrap_95_ci": hierarchical_bootstrap_ci(
                action[:, axis], target, tasks, bootstrap, rng
            ),
            "task_stratified_two_sided_permutation_p": stratified_permutation_p(
                action[:, axis], target, tasks, observed, permutations, rng
            ),
            "task_macro_spearman_with_outcome_heterogeneity": task_macro_spearman(
                action[:, axis], heterogeneity, tasks
            )[0],
        }

    probes = {
        "action_route": task_held_out_ridge(action, target, tasks, ridge),
        "state_token_route": task_held_out_ridge(state, target, tasks, ridge),
        "action_plus_state_route": task_held_out_ridge(
            np.column_stack([action, state]), target, tasks, ridge
        ),
    }
    state_variation = np.asarray(
        [pool.diagnostics["state_token_candidate_max_hellinger"] for pool in pools]
    )
    return {
        "schema": "himoe-moe-opportunity-shadow-v1",
        "estimand_warning": (
            "Targets use one full rollout per state/seed and are not candidate-chunk "
            "Q estimates; each seed also controls later replanning noise."
        ),
        "data_audit": audit,
        "targets": {
            "mean_success_rate": float(
                np.mean([pool.success_rate for pool in pools])
            ),
            "mean_empirical_headroom": float(target.mean()),
            "mean_outcome_heterogeneity": float(heterogeneity.mean()),
            "mixed_state_fraction": float(
                np.mean([pool.mixed_outcome for pool in pools])
            ),
        },
        "route_contract": {
            "state_token_candidate_max_hellinger": float(state_variation.max()),
            "action_feature_names": list(feature_names),
            "state_feature_definition": "per-layer entropy, top4 mass, and top2 margin",
        },
        "associations": associations,
        "task_held_out_fixed_ridge": probes,
        "decision": {
            "confirmatory": False,
            "route_opportunity_signal_established": False,
            "reason": (
                "This observational full-rollout grid cannot identify per-chunk "
                "opportunity even if an exploratory association is nonzero."
            ),
        },
    }


def fmt(value: float, signed: bool = False) -> str:
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:.3f}"


def write_report(summary: dict[str, Any], out_dir: Path) -> None:
    primary = summary["associations"]["action_route_dispersion_full"]
    audit = summary["data_audit"]
    probe = summary["task_held_out_fixed_ridge"]
    lines = [
        "# MoE routing opportunity shadow audit",
        "",
        "## Scope",
        "",
        "This is a CPU-only retrospective analysis of the five available `16 x 32` grids. "
        "The target is empirical full-rollout success headroom, not candidate-chunk Q. "
        "A flow-noise seed also controls later replanning calls, so no causal candidate-selection "
        "claim is permitted.",
        "",
        "## Data",
        "",
        f"- Tasks: `{len(audit['tasks'])}`",
        f"- Initial-state pools: `{audit['total_states']}`",
        f"- Full rollouts: `{audit['total_rollouts']}`",
        f"- Mixed-outcome pools: `{audit['mixed_states']}`",
        f"- State-token maximum candidate variation (Hellinger): "
        f"`{summary['route_contract']['state_token_candidate_max_hellinger']:.8f}`",
        "",
        "## Route dispersion versus empirical headroom",
        "",
        "| MoE observable | task-macro Spearman | 95% hierarchical bootstrap CI | stratified p |",
        "|---|---:|---:|---:|",
    ]
    for name, row in summary["associations"].items():
        ci = row["hierarchical_bootstrap_95_ci"]
        lines.append(
            f"| `{name}` | {fmt(row['task_macro_spearman_with_empirical_headroom'], True)} "
            f"| [{fmt(ci[0], True)}, {fmt(ci[1], True)}] "
            f"| {row['task_stratified_two_sided_permutation_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Primary full-route result: task-macro Spearman "
            f"`{fmt(primary['task_macro_spearman_with_empirical_headroom'], True)}`, "
            f"95% CI `[{fmt(primary['hierarchical_bootstrap_95_ci'][0], True)}, "
            f"{fmt(primary['hierarchical_bootstrap_95_ci'][1], True)}]`, "
            f"permutation `p={primary['task_stratified_two_sided_permutation_p']:.4f}`.",
            "",
            "## Task-held-out diagnostic probes",
            "",
            "| Features | OOF R2 | OOF Spearman |",
            "|---|---:|---:|",
            f"| action-token route | {fmt(probe['action_route']['r2'], True)} "
            f"| {fmt(probe['action_route']['spearman'], True)} |",
            f"| state-token route | {fmt(probe['state_token_route']['r2'], True)} "
            f"| {fmt(probe['state_token_route']['spearman'], True)} |",
            f"| action + state route | {fmt(probe['action_plus_state_route']['r2'], True)} "
            f"| {fmt(probe['action_plus_state_route']['spearman'], True)} |",
            "",
            "These probes use fixed ridge regularization and hold out a complete task. They are "
            "diagnostics, not tuned predictive models.",
            "",
            "## Decision",
            "",
            "The analysis cannot establish a MoE opportunity predictor from current data. "
            "A positive retrospective association would still require repeated CRN continuation "
            "Q before it could authorize adaptive K.",
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1 or args.permutations < 1 or args.ridge <= 0:
        raise ValueError("bootstrap/permutations must be positive and ridge must be > 0")
    pools, audit = load_pools(args.hub_root)
    summary = analyze(
        pools,
        audit,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        ridge=args.ridge,
        seed=args.seed,
    )
    write_report(summary, args.out_dir)
    print(f"wrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
