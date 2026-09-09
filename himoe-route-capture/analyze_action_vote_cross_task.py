#!/usr/bin/env python3
"""Action-space majority voting on the five-task 16x32 grid, scored by rollout success.

This is the published Best-of-N selector evaluated against real episode success
rather than an endpoint proxy.  It is a post-hoc shadow audit and it deliberately
mirrors the statistics of ``analysis/moe-current-data-audit/cross-task/
k32_route_medoid_audit.py`` so the action selector and the route selector are
directly comparable on the same pools.

The severe caveat of this substrate is stated up front: one flow-noise seed fixes
both the first policy query and every later replanning query of the same episode.
Selecting a candidate here therefore selects a whole noise stream, not just a
first action chunk, so episode success is an episode-start shadow label rather
than the causal value of the selected chunk.  Seed concentration is reported for
exactly this reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_action_majority_vote import medoid_choice, subset_index
from analyze_early_action_head import load_action_stds
from analyze_route_noise_selector import discover_runs
from behavior_geometry import action_distance_matrix


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_HUB = REPO / "VLA_MUI_HUB"
DEFAULT_OUT = HERE / "analysis/action-majority-vote/cross-task"

STATES_PER_TASK = 16
CANDIDATES_PER_STATE = 32
BUDGETS = (2, 4, 8, 16, 32)
SUBSET_DRAWS = 20_000
SEED = 20260825
BOOTSTRAP = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-root", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gripper-weight", type=float, default=0.25)
    parser.add_argument("--subset-draws", type=int, default=SUBSET_DRAWS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_task_pools(run: Path, action_std: np.ndarray, gripper_weight: float) -> dict[str, Any]:
    summaries = json.loads((run / "client" / "summaries.json").read_text())
    if len(summaries) != STATES_PER_TASK * CANDIDATES_PER_STATE:
        raise RuntimeError(f"unexpected episode count in {run}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[int(row["init_state_id"])].append(row)
    if len(grouped) != STATES_PER_TASK:
        raise RuntimeError(f"{run} does not hold {STATES_PER_TASK} init states")

    pools = []
    for state_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda value: int(value["flow_noise_seed"]))
        if len(rows) != CANDIDATES_PER_STATE:
            raise RuntimeError(f"{run} state {state_id} does not hold 32 candidates")
        seeds = np.asarray([int(row["flow_noise_seed"]) for row in rows], dtype=np.int64)
        if len(np.unique(seeds)) != CANDIDATES_PER_STATE:
            raise RuntimeError(f"{run} state {state_id} reuses a flow-noise seed")
        chunks = []
        for row in rows:
            path = run / "client" / f"episode_{int(row['episode_index']):02d}.npz"
            with np.load(path, allow_pickle=True) as store:
                actions = np.asarray(store["actions"], dtype=np.float64)
            if actions.ndim != 3 or actions.shape[1:] != (10, 7):
                raise RuntimeError(f"{path} does not hold 10x7 action chunks")
            chunks.append(actions[0])
        pools.append(
            {
                "state_id": int(state_id),
                "seeds": seeds,
                "success": np.asarray(
                    [float(bool(row["success"])) for row in rows], dtype=np.float64
                ),
                "distance": action_distance_matrix(
                    np.stack(chunks), action_std, gripper_weight
                ),
            }
        )
    return {"task_name": run.parent.name, "suite": run.parents[1].name, "pools": pools}


def task_macro_bootstrap(
    per_task_effects: list[np.ndarray], *, draws: int, seed: int
) -> dict[str, Any]:
    """Resample states inside each task, then average tasks with equal weight."""

    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        task_means = [
            float(effects[rng.integers(0, len(effects), size=len(effects))].mean())
            for effects in per_task_effects
        ]
        samples[index] = float(np.mean(task_means))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(np.mean([float(effects.mean()) for effects in per_task_effects])),
        "lower": float(lower),
        "upper": float(upper),
        "draws": int(draws),
    }


def main() -> int:
    args = parse_args()
    runs = discover_runs(args.hub_root)
    action_stds = load_action_stds(REPO)
    tasks = []
    for run in runs:
        suite = run.parents[1].name
        if suite not in action_stds:
            raise RuntimeError(f"no official action std for suite {suite}")
        tasks.append(load_task_pools(run, action_stds[suite], args.gripper_weight))

    summary: dict[str, Any] = {
        "schema": "himoe-action-vote-cross-task-v1",
        "status": "posthoc_shadow_audit",
        "protocol": {
            "outcome": "eventual rollout success of the selected candidate episode",
            "label_caveat": (
                "one seed fixes the first query and every later replanning query, so "
                "success is an episode-start shadow label, not chunk-level causal value"
            ),
            "random_baseline": "exact within-state candidate mean",
            "distance": "checkpoint-normalized mean stepwise L2 over the first action chunk",
            "gripper_weight": float(args.gripper_weight),
            "tie_break": "lowest flow-noise seed after ascending seed sort",
            "bootstrap": "states resampled within each task, task macro averaged",
            "bootstrap_draws": int(args.bootstrap),
            "subset_draws": int(args.subset_draws),
            "seed": int(args.seed),
        },
        "tasks": [
            {
                "task_name": task["task_name"],
                "suite": task["suite"],
                "pool_success_rate": float(
                    np.mean([pool["success"].mean() for pool in task["pools"]])
                ),
                "mixed_states": int(
                    sum(1 for pool in task["pools"] if 0.0 < pool["success"].mean() < 1.0)
                ),
            }
            for task in tasks
        ],
        "budgets": {},
    }

    for budget in BUDGETS:
        subsets, exhaustive = subset_index(
            CANDIDATES_PER_STATE, budget, args.subset_draws, args.seed
        )
        per_task_effects = []
        per_task_oracle = []
        seed_counter: Counter = Counter()
        for task in tasks:
            effects = []
            oracle = []
            for pool in task["pools"]:
                success = pool["success"]
                chosen = medoid_choice(pool["distance"], subsets)
                seed_counter.update(int(pool["seeds"][value]) for value in chosen)
                effects.append(float(success[chosen].mean()) - float(success.mean()))
                oracle.append(float(success.max()) - float(success.mean()))
            per_task_effects.append(np.asarray(effects, dtype=np.float64))
            per_task_oracle.append(np.asarray(oracle, dtype=np.float64))
        total = sum(seed_counter.values())
        summary["budgets"][str(budget)] = {
            "subsets_per_state": int(len(subsets)),
            "exhaustive_subsets": bool(exhaustive),
            "degenerate_by_construction": bool(budget == 2),
            "selected_minus_random": task_macro_bootstrap(
                per_task_effects, draws=args.bootstrap, seed=args.seed
            ),
            "full_pool_oracle_minus_random": float(
                np.mean([effects.mean() for effects in per_task_oracle])
            ),
            "unique_selected_seeds": len(seed_counter),
            "max_seed_share": float(max(seed_counter.values()) / total) if total else None,
        }

    lines = [
        "# Action-space majority voting on the five-task 16x32 grid",
        "",
        "Post-hoc shadow audit. The selector uses only the first action chunk of each",
        "candidate episode; the outcome is eventual rollout success. One seed fixes both",
        "the first query and every later replanning query, so a selector here chooses a",
        "noise stream and success is an episode-start shadow label.",
        "",
        "| task | suite | pool success rate | mixed states |",
        "|---|---|---:|---:|",
    ]
    for row in summary["tasks"]:
        lines.append(
            f"| {row['task_name']} | {row['suite']} | {row['pool_success_rate']:.3f} | "
            f"{row['mixed_states']}/16 |"
        )
    lines += [
        "",
        "| budget N | selected - random | 95% CI | full-pool oracle - random | unique seeds | max seed share |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for budget, row in summary["budgets"].items():
        effect = row["selected_minus_random"]
        lines.append(
            f"| {budget} | {effect['mean'] * 100:+.2f} pp | "
            f"[{effect['lower'] * 100:+.2f}, {effect['upper'] * 100:+.2f}] pp | "
            f"{row['full_pool_oracle_minus_random'] * 100:+.2f} pp | "
            f"{row['unique_selected_seeds']} | {row['max_seed_share']:.3f} |"
        )
    lines += [
        "",
        "`N=2` is degenerate by construction: the two-candidate medoid is a tie and",
        "reduces to the lowest-seed pick.",
        "",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                budget: row["selected_minus_random"]["mean"]
                for budget, row in summary["budgets"].items()
            },
            indent=2,
        )
    )
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
