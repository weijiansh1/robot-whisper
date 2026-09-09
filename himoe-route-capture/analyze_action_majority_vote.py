#!/usr/bin/env python3
"""Published-style action-space majority voting on the legacy same-state fork pools.

This measures the literature Best-of-N selector -- pick the candidate action
chunk closest to the consensus of the other sampled candidates -- as a function
of the sampling budget ``N``.  It is a retrospective endpoint-proxy audit of
already-executed candidates, not a closed-loop experiment.

Scope frozen before running:

* Substrate: the two same-state fork captures.  Each snapshot is one exact
  simulator state; every candidate chunk was executed and then continued by the
  frozen base policy with a shared continuation realization.  This is the
  structure the published method assumes.  The five-task ``right-16x32`` grid is
  deliberately NOT used here because each of its candidates is a whole episode
  under one seed, so a selector there chooses a noise stream rather than an
  action.
* Selectors are training-free, deterministic and tie-break on the lowest
  candidate id.  No outcome is used to fit anything.
* The random baseline is the exact within-snapshot candidate mean, which is also
  the exact expectation of a uniform single draw at every budget.
* The oracle is the exact expectation of the subset maximum under uniform
  ``N``-subsets, computed from order statistics rather than sampled.
* Every contrast is reduced to one number per snapshot before bootstrapping.
  Candidates, pairs and subsets are never treated as independent observations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from analyze_moe_consensus_audit import OUTCOMES, sha256_file, snapshot_bootstrap
from behavior_geometry import action_distance_matrix, action_rms_distance_matrix


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "analysis/action-majority-vote"

POOLS = {
    "k32": {
        "records": "runs/fork-pilot-n32-client/fork_records.json",
        "chunks": "runs/fork-pilot-n32-client",
        "metadata": "runs/fork-pilot-n32-client/server_metadata.json",
        "candidates": 32,
        "budgets": (2, 4, 8, 16, 32),
    },
    "k8": {
        "records": "runs/fork-pilot-client/fork_records.json",
        "chunks": "runs/fork-pilot-client",
        "metadata": "runs/fork-pilot-client/server_metadata.json",
        "candidates": 8,
        "budgets": (2, 4, 8),
    },
}

EXPECTED_SNAPSHOTS = 20
ENUMERATION_LIMIT = 50_000
SUBSET_DRAWS = 20_000
SUBSET_SEED = 20260825
PRIMARY_OUTCOME = "drawer_progress_after_continuation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=sorted(POOLS), action="append")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gripper-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=SUBSET_SEED)
    parser.add_argument("--subset-draws", type=int, default=SUBSET_DRAWS)
    return parser.parse_args()


def normalized_actions(
    actions: np.ndarray, action_std: np.ndarray, gripper_weight: float
) -> np.ndarray:
    """Return checkpoint-normalized, gripper-weighted chunks ``[K,H,7]``."""

    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 7:
        raise ValueError("action chunks must have shape [K,H,7]")
    scale = np.asarray(action_std, dtype=np.float64)
    if scale.shape != (7,) or np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("action_std must contain seven positive finite values")
    if not np.isfinite(gripper_weight) or gripper_weight < 0.0:
        raise ValueError("gripper_weight must be finite and non-negative")
    weights = np.ones(7, dtype=np.float64)
    weights[-1] = float(gripper_weight)
    return value / scale * weights


def subset_index(n_candidates: int, budget: int, draws: int, seed: int) -> tuple[np.ndarray, bool]:
    """Return ``[M,N]`` candidate subsets and whether they enumerate exhaustively."""

    if not 2 <= budget <= n_candidates:
        raise ValueError("budget must lie between two and the pool size")
    total = comb(n_candidates, budget)
    if total <= ENUMERATION_LIMIT:
        return np.asarray(list(combinations(range(n_candidates), budget)), dtype=np.int64), True
    rng = np.random.default_rng([seed, n_candidates, budget])
    subsets = np.stack(
        [rng.choice(n_candidates, size=budget, replace=False) for _ in range(draws)]
    )
    return np.sort(subsets, axis=1), False


def medoid_choice(distance: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    """Lowest-id medoid of every subset under a precomputed pool distance."""

    block = distance[subsets[:, :, None], subsets[:, None, :]]
    return subsets[np.arange(len(subsets)), np.argmin(block.sum(axis=2), axis=1)]


def centroid_choice(normalized: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    """Lowest-id nearest neighbour of the subset mean action chunk."""

    chosen = np.empty(len(subsets), dtype=np.int64)
    for start in range(0, len(subsets), 2048):
        block = subsets[start : start + 2048]
        values = normalized[block]
        centroid = values.mean(axis=1, keepdims=True)
        distance = np.linalg.norm(values - centroid, axis=-1).mean(axis=-1)
        chosen[start : start + len(block)] = block[
            np.arange(len(block)), np.argmin(distance, axis=1)
        ]
    return chosen


def exact_expected_maximum(values: np.ndarray, budget: int) -> float:
    """Exact ``E[max]`` over uniform ``N``-subsets drawn without replacement."""

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n_candidates = len(ordered)
    total = comb(n_candidates, budget)
    weights = np.asarray(
        [comb(rank, budget - 1) for rank in range(n_candidates)], dtype=np.float64
    )
    return float(np.dot(ordered, weights) / total)


def load_pool(root: Path, spec: dict[str, Any], gripper_weight: float) -> list[dict[str, Any]]:
    records = json.loads((root / spec["records"]).read_text(encoding="utf-8"))
    metadata = json.loads((root / spec["metadata"]).read_text(encoding="utf-8"))
    action_std = np.asarray(metadata["normalization_action_std"], dtype=np.float64)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(int(row["episode"]), int(row["fork_step"]))].append(row)
    if len(grouped) != EXPECTED_SNAPSHOTS:
        raise ValueError(f"expected {EXPECTED_SNAPSHOTS} snapshots, got {len(grouped)}")

    snapshots = []
    for (episode, fork_step), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda value: int(row_candidate(value)))
        candidate_ids = np.asarray([row_candidate(row) for row in rows], dtype=np.int64)
        if not np.array_equal(candidate_ids, np.arange(spec["candidates"])):
            raise ValueError(f"ep{episode}/t{fork_step} candidate ids are not contiguous")
        chunk_path = root / spec["chunks"] / f"chunks_ep{episode:02d}_t{fork_step:02d}.npy"
        actions = np.asarray(np.load(chunk_path), dtype=np.float64)[candidate_ids]
        if len(actions) != spec["candidates"]:
            raise ValueError(f"{chunk_path} does not hold {spec['candidates']} chunks")
        available = {
            name
            for name, spec_out in OUTCOMES.items()
            if all(spec_out["source"] in row for row in rows)
        }
        if PRIMARY_OUTCOME not in available:
            raise ValueError(f"ep{episode}/t{fork_step} lacks the primary outcome source")
        snapshots.append(
            {
                "snapshot": f"ep{episode}/t{fork_step}",
                "distance": {
                    "action_medoid": action_distance_matrix(
                        actions, action_std, gripper_weight
                    ),
                    "action_rms_medoid": action_rms_distance_matrix(actions, action_std),
                },
                "normalized": normalized_actions(actions, action_std, gripper_weight),
                "outcomes": {
                    name: np.asarray(
                        [
                            float(row[OUTCOMES[name]["source"]])
                            * float(OUTCOMES[name]["multiplier"])
                            for row in rows
                        ],
                        dtype=np.float64,
                    )
                    for name in sorted(available)
                },
            }
        )
    return snapshots


def row_candidate(row: dict[str, Any]) -> int:
    return int(row["candidate"])


def evaluate_pool(
    snapshots: list[dict[str, Any]],
    budgets: tuple[int, ...],
    *,
    n_candidates: int,
    draws: int,
    seed: int,
    bootstrap: int,
    confidence: float,
) -> dict[str, Any]:
    selectors = ("action_medoid", "action_centroid_nn", "action_rms_medoid")
    results: dict[str, Any] = {"budgets": {}, "diagnostics": {}}

    for budget in budgets:
        subsets, exhaustive = subset_index(n_candidates, budget, draws, seed)
        per_selector_effects: dict[str, list[float]] = {name: [] for name in selectors}
        oracle_effects: list[float] = []
        chosen_ids: dict[str, Counter] = {name: Counter() for name in selectors}
        for snapshot in snapshots:
            outcome = snapshot["outcomes"][PRIMARY_OUTCOME]
            random_mean = float(outcome.mean())
            oracle = exact_expected_maximum(outcome, budget)
            oracle_effects.append(oracle - random_mean)
            for name in selectors:
                if name == "action_centroid_nn":
                    chosen = centroid_choice(snapshot["normalized"], subsets)
                else:
                    chosen = medoid_choice(snapshot["distance"][name], subsets)
                chosen_ids[name].update(int(value) for value in chosen)
                per_selector_effects[name].append(float(outcome[chosen].mean()) - random_mean)

        budget_row: dict[str, Any] = {
            "subsets_per_snapshot": int(len(subsets)),
            "exhaustive_subsets": bool(exhaustive),
            "degenerate_by_construction": bool(budget == 2),
            "oracle_minus_random": snapshot_bootstrap(
                oracle_effects, draws=bootstrap, confidence=confidence, seed=seed
            ),
            "selectors": {},
        }
        mean_headroom = float(np.mean(oracle_effects))
        for name in selectors:
            counts = chosen_ids[name]
            total = sum(counts.values())
            mean_effect = float(np.mean(per_selector_effects[name]))
            budget_row["selectors"][name] = {
                "selected_minus_random": snapshot_bootstrap(
                    per_selector_effects[name],
                    draws=bootstrap,
                    confidence=confidence,
                    seed=seed,
                ),
                "oracle_recovery_ratio": (
                    mean_effect / mean_headroom if mean_headroom > 0 else None
                ),
                "unique_selected_candidates": len(counts),
                "max_candidate_share": float(max(counts.values()) / total) if total else None,
                "per_snapshot_effects": [float(value) for value in per_selector_effects[name]],
            }
        results["budgets"][str(budget)] = budget_row

    scaling_budgets = [value for value in budgets if value >= 4]
    results["budget_scaling"] = {
        "budgets": scaling_budgets,
        "regressor": "log2(N)",
        "note": "N=2 is excluded because its medoid is a tie by construction",
        "selectors": {},
    }
    if len(scaling_budgets) >= 2:
        axis = np.log2(np.asarray(scaling_budgets, dtype=np.float64))
        centered = axis - axis.mean()
        denominator = float(np.dot(centered, centered))
        for name in selectors:
            curves = np.asarray(
                [
                    results["budgets"][str(value)]["selectors"][name]["per_snapshot_effects"]
                    for value in scaling_budgets
                ],
                dtype=np.float64,
            )
            slopes = (centered[:, None] * curves).sum(axis=0) / denominator
            results["budget_scaling"]["selectors"][name] = snapshot_bootstrap(
                slopes, draws=bootstrap, confidence=confidence, seed=seed
            )

    dispersion = [
        float(np.median(snapshot["distance"]["action_medoid"][np.triu_indices(n_candidates, 1)]))
        for snapshot in snapshots
    ]
    full_subsets, _ = subset_index(n_candidates, n_candidates, draws, seed)
    full_effects = []
    for snapshot in snapshots:
        outcome = snapshot["outcomes"][PRIMARY_OUTCOME]
        chosen = medoid_choice(snapshot["distance"]["action_medoid"], full_subsets)
        full_effects.append(float(outcome[chosen].mean()) - float(outcome.mean()))
    rho, p_value = spearmanr(dispersion, full_effects)
    results["diagnostics"] = {
        "primary_outcome": PRIMARY_OUTCOME,
        "median_pool_action_distance": {
            "mean": float(np.mean(dispersion)),
            "min": float(np.min(dispersion)),
            "max": float(np.max(dispersion)),
        },
        "dispersion_vs_full_pool_effect_spearman": {
            "rho": float(rho),
            "p": float(p_value),
            "snapshots": len(dispersion),
        },
    }
    results["outcome_support"] = {
        name: int(
            sum(
                1
                for snapshot in snapshots
                if len(np.unique(snapshot["outcomes"][name])) > 1
            )
        )
        for name in snapshots[0]["outcomes"]
    }
    return results


def secondary_outcomes(
    snapshots: list[dict[str, Any]],
    *,
    n_candidates: int,
    budget: int,
    draws: int,
    seed: int,
    bootstrap: int,
    confidence: float,
) -> dict[str, Any]:
    """Full-pool action medoid on every declared outcome, not just the primary."""

    subsets, _ = subset_index(n_candidates, budget, draws, seed)
    rows = {}
    for name in snapshots[0]["outcomes"]:
        effects = []
        for snapshot in snapshots:
            outcome = snapshot["outcomes"][name]
            chosen = medoid_choice(snapshot["distance"]["action_medoid"], subsets)
            effects.append(float(outcome[chosen].mean()) - float(outcome.mean()))
        rows[name] = {
            "higher_is_better": bool(OUTCOMES[name]["higher_is_better"]),
            "selected_minus_random": snapshot_bootstrap(
                effects, draws=bootstrap, confidence=confidence, seed=seed
            ),
        }
    return rows


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Published-style action-space majority voting on same-state fork pools",
        "",
        "Retrospective endpoint-proxy audit of already-executed candidates. The",
        "selector picks the candidate closest to the consensus of the other sampled",
        "candidates; no outcome is used to fit anything. Every contrast is reduced",
        "within snapshot before the snapshot bootstrap.",
        "",
    ]
    for pool, result in summary["pools"].items():
        spec = POOLS[pool]
        lines += [
            f"## Pool `{pool}` ({spec['candidates']} candidates x {EXPECTED_SNAPSHOTS} snapshots)",
            "",
            f"Primary outcome: `{PRIMARY_OUTCOME}` (higher is better).",
            "",
            "| budget N | selector | selected - random | 95% CI | oracle recovery | unique picks | max share |",
            "|---:|---|---:|---|---:|---:|---:|",
        ]
        for budget, row in result["budgets"].items():
            for name, values in row["selectors"].items():
                effect = values["selected_minus_random"]
                interval = (
                    f"[{effect['lower']:+.6f}, {effect['upper']:+.6f}]"
                    if effect.get("available")
                    else "unavailable"
                )
                recovery = values["oracle_recovery_ratio"]
                lines.append(
                    f"| {budget} | {name} | {effect['mean']:+.6f} | {interval} | "
                    + (f"{recovery:+.3f}" if recovery is not None else "n/a")
                    + f" | {values['unique_selected_candidates']} | "
                    f"{values['max_candidate_share']:.3f} |"
                )
        scaling = result["budget_scaling"]
        if scaling["selectors"]:
            lines += [
                "",
                f"Budget scaling: per-snapshot slope of the effect on `log2(N)` over "
                f"`N={scaling['budgets']}`.",
                "",
                "| selector | slope per log2(N) | 95% CI |",
                "|---|---:|---|",
            ]
            for name, row in scaling["selectors"].items():
                interval = (
                    f"[{row['lower']:+.6f}, {row['upper']:+.6f}]"
                    if row.get("available")
                    else "unavailable"
                )
                lines.append(f"| {name} | {row['mean']:+.6f} | {interval} |")
        lines += ["", "| budget N | oracle - random | 95% CI | subsets | exhaustive |", "|---:|---:|---|---:|---|"]
        for budget, row in result["budgets"].items():
            oracle = row["oracle_minus_random"]
            interval = (
                f"[{oracle['lower']:+.6f}, {oracle['upper']:+.6f}]"
                if oracle.get("available")
                else "unavailable"
            )
            lines.append(
                f"| {budget} | {oracle['mean']:+.6f} | {interval} | "
                f"{row['subsets_per_snapshot']} | {row['exhaustive_subsets']} |"
            )
        lines += ["", "### Full-pool action medoid on every declared outcome", "",
                  "| outcome | higher is better | selected - random | 95% CI |", "|---|---|---:|---|"]
        for name, row in result["secondary"].items():
            effect = row["selected_minus_random"]
            interval = (
                f"[{effect['lower']:+.6f}, {effect['upper']:+.6f}]"
                if effect.get("available")
                else "unavailable"
            )
            lines.append(
                f"| `{name}` | {row['higher_is_better']} | {effect['mean']:+.6f} | {interval} |"
            )
        support = result["outcome_support"]
        lines += [
            "",
            "### Outcome support",
            "",
            "| outcome | mixed snapshots |",
            "|---|---:|",
        ]
        for name, count in support.items():
            lines.append(f"| `{name}` | {count} |")
        diagnostics = result["diagnostics"]
        rho = diagnostics["dispersion_vs_full_pool_effect_spearman"]
        lines += [
            "",
            "### Degeneracy and dispersion",
            "",
            f"Median within-pool action distance: mean `{diagnostics['median_pool_action_distance']['mean']:.4f}`, "
            f"range `{diagnostics['median_pool_action_distance']['min']:.4f}`-"
            f"`{diagnostics['median_pool_action_distance']['max']:.4f}`.",
            "",
            f"Dispersion versus full-pool medoid effect: Spearman `{rho['rho']:+.3f}` "
            f"(`p={rho['p']:.4f}`, {rho['snapshots']} snapshots).",
            "",
        ]
    lines += [
        "## Interpretation limits",
        "",
        "- `N=2` medoid is a tie by construction and reduces to the lowest-id pick;",
        "  its row is reported only to keep the budget axis complete.",
        "- Each candidate carries one shared-CRN continuation realization, not a",
        "  repeated-continuation Monte Carlo `Q`.",
        "- Binary success is mixed in almost no snapshot, so the drawer proxies carry",
        "  the entire primary signal and no closed-loop success claim is available.",
        "- Both pools come from one LIBERO-Goal task; there is no held-out task.",
        "- Subset expectations at large `N` are Monte Carlo over common subsets; the",
        "  random and oracle baselines remain exact.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    pools = args.pool or sorted(POOLS)
    summary: dict[str, Any] = {
        "schema": "himoe-action-majority-vote-v1",
        "status": "retrospective_endpoint_proxy_audit",
        "protocol": {
            "primary_outcome": PRIMARY_OUTCOME,
            "gripper_weight": float(args.gripper_weight),
            "subset_seed": int(args.seed),
            "subset_draws": int(args.subset_draws),
            "enumeration_limit": ENUMERATION_LIMIT,
            "bootstrap_draws": int(args.bootstrap),
            "confidence": float(args.confidence),
            "random_baseline": "exact within-snapshot candidate mean",
            "oracle_baseline": "exact expected subset maximum from order statistics",
            "tie_break": "lowest candidate id",
        },
        "pools": {},
    }
    for pool in pools:
        spec = POOLS[pool]
        snapshots = load_pool(HERE, spec, args.gripper_weight)
        result = evaluate_pool(
            snapshots,
            spec["budgets"],
            n_candidates=spec["candidates"],
            draws=args.subset_draws,
            seed=args.seed,
            bootstrap=args.bootstrap,
            confidence=args.confidence,
        )
        result["secondary"] = secondary_outcomes(
            snapshots,
            n_candidates=spec["candidates"],
            budget=spec["candidates"],
            draws=args.subset_draws,
            seed=args.seed,
            bootstrap=args.bootstrap,
            confidence=args.confidence,
        )
        result["source_sha256"] = {
            spec["records"]: sha256_file(HERE / spec["records"]),
            spec["metadata"]: sha256_file(HERE / spec["metadata"]),
        }
        summary["pools"][pool] = result

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    headline = {
        pool: {
            budget: row["selectors"]["action_medoid"]["selected_minus_random"]["mean"]
            for budget, row in result["budgets"].items()
        }
        for pool, result in summary["pools"].items()
    }
    print(json.dumps(headline, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
