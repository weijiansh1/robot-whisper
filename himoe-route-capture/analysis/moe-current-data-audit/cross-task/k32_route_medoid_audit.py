#!/usr/bin/env python3
"""Audit K32 route medoids against eventual rollout success.

This is a post-hoc shadow analysis of the existing five-task right-16x32
corpus.  It does not treat candidate rows as independent observations and it
does not claim that the first action chunk caused the eventual outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from analyze_route_noise_selector import discover_runs, first_rows  # noqa: E402
from route_noise_selector import (  # noqa: E402
    centrality,
    normalize_router_probabilities,
    pairwise_hellinger,
)


DEFAULT_HUB = REPO.parent / "VLA_MUI_HUB"
DEFAULT_OUT = HERE / "k32_route_medoid.json"
BOOTSTRAP_SEED = 20260825
N_BOOTSTRAP = 20_000
METHOD_WINDOWS = {
    "route_prob_d0": slice(0, 1),
    "route_prob_d0_2": slice(0, 3),
    "route_prob_full": slice(0, 10),
    "expert_id_d0": slice(0, 1),
    "expert_id_d0_2": slice(0, 3),
    "expert_id_full": slice(0, 10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-root", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite result: {result!r}")
    return result


def finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = rankdata(np.asarray(left, dtype=np.float64))
    b = rankdata(np.asarray(right, dtype=np.float64))
    a -= a.mean()
    b -= b.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / denominator) if denominator > 1e-20 else float("nan")


def topk_jaccard_distance(expert_ids: np.ndarray) -> np.ndarray:
    """Mean order-invariant top-k Jaccard distance over aligned route sites."""

    ids = np.asarray(expert_ids, dtype=np.int16).reshape(len(expert_ids), -1, 4)
    if np.any(ids < 0) or np.any(ids >= 32):
        raise ValueError("expert IDs are outside [0, 32)")
    if np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
        raise ValueError("an aligned top-4 route site contains duplicate expert IDs")
    equality = ids[:, None, :, :, None] == ids[None, :, :, None, :]
    intersection = equality.any(axis=-1).sum(axis=-1)
    union = 8 - intersection
    distance = (1.0 - intersection / union).mean(axis=-1)
    np.fill_diagonal(distance, 0.0)
    return np.asarray(distance, dtype=np.float64)


def stable_medoid(distance: np.ndarray, seeds: np.ndarray) -> tuple[int, np.ndarray]:
    score = centrality(distance)
    optimum = float(score.min())
    tied = np.flatnonzero(np.isclose(score, optimum, atol=1e-12, rtol=0.0))
    selected = int(tied[np.argmin(seeds[tied])])
    return selected, score


def task_name(run: Path, cache_root: Path) -> str:
    return str(run.relative_to(cache_root).parent)


def load_task_states(run: Path, cache_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries_path = run / "client" / "summaries.json"
    summaries = sorted(
        json.loads(summaries_path.read_text(encoding="utf-8")),
        key=lambda row: int(row["episode_index"]),
    )
    if len(summaries) != 512:
        raise RuntimeError(f"{run} is not a complete 16x32 run")
    episodes = np.asarray([int(row["episode_index"]) for row in summaries], dtype=np.int64)
    states = np.asarray([int(row["init_state_id"]) for row in summaries], dtype=np.int64)
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries], dtype=np.int64)
    success = np.asarray([bool(row["success"]) for row in summaries], dtype=np.bool_)
    route_path = run / "server" / "routes.zarr"
    route_group = zarr.open_group(str(route_path), mode="r")
    route_episode = np.asarray(route_group["episode_id"][:], dtype=np.int64)
    rows = first_rows(route_episode, episodes)
    inference_calls = np.asarray([int(row["inference_calls"]) for row in summaries])
    expected_rows = np.r_[0, np.cumsum(inference_calls)[:-1]]
    if not np.array_equal(rows, expected_rows):
        raise RuntimeError(f"route/client episode boundaries disagree in {run}")
    raw_probability = np.asarray(
        route_group["hb_router_probs"].oindex[rows, :, :, 1:, :],
        dtype=np.float32,
    )
    expert_ids = np.asarray(
        route_group["hb_expert_ids"].oindex[rows, :, :, 1:, :],
        dtype=np.int16,
    )
    if raw_probability.shape != (512, 8, 10, 10, 32):
        raise RuntimeError(f"unexpected probability shape in {run}: {raw_probability.shape}")
    if expert_ids.shape != (512, 8, 10, 10, 4):
        raise RuntimeError(f"unexpected expert-ID shape in {run}: {expert_ids.shape}")
    probability = normalize_router_probabilities(raw_probability)
    unique_states = np.unique(states)
    if len(unique_states) != 16:
        raise RuntimeError(f"expected 16 states in {run}")

    output = []
    canonical_seeds: tuple[int, ...] | None = None
    for state in unique_states:
        indices = np.flatnonzero(states == state)
        indices = indices[np.argsort(seeds[indices])]
        state_seeds = seeds[indices]
        if len(indices) != 32 or len(np.unique(state_seeds)) != 32:
            raise RuntimeError(f"state {state} is not a complete K32 pool")
        seed_tuple = tuple(map(int, state_seeds))
        if canonical_seeds is None:
            canonical_seeds = seed_tuple
        elif canonical_seeds != seed_tuple:
            raise RuntimeError(f"seed grid changes across states in {run}")
        state_success = success[indices]
        record: dict[str, Any] = {
            "task": task_name(run, cache_root),
            "state": int(state),
            "success_count": int(state_success.sum()),
            "random_pool_mean": finite(state_success.mean()),
            "opportunity": finite(float(state_success.max()) - state_success.mean()),
            "outcome_variance": finite(state_success.mean() * (1.0 - state_success.mean())),
            "methods": {},
        }
        for method, window in METHOD_WINDOWS.items():
            if method.startswith("route_prob"):
                distance = pairwise_hellinger(probability[indices, :, window, :, :])
            else:
                distance = topk_jaccard_distance(expert_ids[indices, :, window, :, :])
            selected, score = stable_medoid(distance, state_seeds)
            record["methods"][method] = {
                "selected_seed": int(state_seeds[selected]),
                "selected_success": bool(state_success[selected]),
                "delta_vs_random": finite(float(state_success[selected]) - state_success.mean()),
                "dispersion": finite(score.mean()),
            }
        output.append(record)

    source = {
        "task": task_name(run, cache_root),
        "run": str(run),
        "summaries_sha256": sha256_file(summaries_path),
        "routes_metadata_sha256": sha256_file(route_path / "zarr.json"),
        "episodes": len(summaries),
        "states": len(unique_states),
        "seeds": list(canonical_seeds or ()),
        "success_rate": finite(success.mean()),
        "raw_probability_mass_max_error": finite(
            np.max(np.abs(raw_probability.sum(axis=-1) - 1.0))
        ),
        "normalized_probability_mass_max_error": finite(
            np.max(np.abs(probability.sum(axis=-1) - 1.0))
        ),
    }
    return output, source


def percentile_interval(draws: np.ndarray) -> list[float]:
    return [finite(value) for value in np.quantile(draws, [0.025, 0.975])]


def bootstrap_method(
    records_by_task: dict[str, list[dict[str, Any]]],
    method: str,
    draws: int,
    rng: np.random.Generator,
) -> tuple[dict[str, list[float]], list[float]]:
    task_draws: dict[str, np.ndarray] = {}
    for task, records in records_by_task.items():
        values = np.asarray(
            [row["methods"][method]["delta_vs_random"] for row in records],
            dtype=np.float64,
        )
        sampled = rng.integers(0, len(values), size=(draws, len(values)))
        task_draws[task] = values[sampled].mean(axis=1)
    macro = np.stack([task_draws[task] for task in sorted(task_draws)], axis=1).mean(axis=1)
    return (
        {task: percentile_interval(values) for task, values in task_draws.items()},
        percentile_interval(macro),
    )


def fit_task_heldout_dispersion(
    records_by_task: dict[str, list[dict[str, Any]]], method: str
) -> dict[str, Any]:
    folds = {}
    all_target = []
    all_prediction = []
    all_baseline = []
    for held_task, held_records in records_by_task.items():
        train_records = [
            row
            for task, records in records_by_task.items()
            if task != held_task
            for row in records
        ]
        train_x = np.asarray(
            [row["methods"][method]["dispersion"] for row in train_records],
            dtype=np.float64,
        )
        train_y = np.asarray([row["opportunity"] for row in train_records], dtype=np.float64)
        test_x = np.asarray(
            [row["methods"][method]["dispersion"] for row in held_records],
            dtype=np.float64,
        )
        test_y = np.asarray([row["opportunity"] for row in held_records], dtype=np.float64)
        x_mean, x_scale = float(train_x.mean()), max(float(train_x.std()), 1e-12)
        design = np.column_stack([np.ones(len(train_x)), (train_x - x_mean) / x_scale])
        coefficients = np.linalg.lstsq(design, train_y, rcond=None)[0]
        prediction = np.column_stack(
            [np.ones(len(test_x)), (test_x - x_mean) / x_scale]
        ) @ coefficients
        baseline = np.full(len(test_y), train_y.mean())
        target_variance = float(np.sum(np.square(test_y - test_y.mean())))
        folds[held_task] = {
            "n_test_states": len(test_y),
            "target_mean": finite(test_y.mean()),
            "prediction_mean": finite(prediction.mean()),
            "mae": finite(np.mean(np.abs(prediction - test_y))),
            "train_mean_baseline_mae": finite(np.mean(np.abs(baseline - test_y))),
            "r2": (
                finite_or_none(1.0 - np.sum(np.square(prediction - test_y)) / target_variance)
                if target_variance > 1e-20
                else None
            ),
            "spearman": finite_or_none(rank_correlation(prediction, test_y)),
        }
        all_target.append(test_y)
        all_prediction.append(prediction)
        all_baseline.append(baseline)
    target = np.concatenate(all_target)
    prediction = np.concatenate(all_prediction)
    baseline = np.concatenate(all_baseline)
    return {
        "folds": folds,
        "pooled_state_mae": finite(np.mean(np.abs(prediction - target))),
        "pooled_train_mean_baseline_mae": finite(np.mean(np.abs(baseline - target))),
        "pooled_spearman": finite_or_none(rank_correlation(prediction, target)),
    }


def summarize(records: list[dict[str, Any]], sources: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    records_by_task = {
        task: [row for row in records if row["task"] == task]
        for task in sorted({row["task"] for row in records})
    }
    rng = np.random.default_rng(args.seed)
    methods: dict[str, Any] = {}
    for method in METHOD_WINDOWS:
        task_ci, macro_ci = bootstrap_method(
            records_by_task, method, args.bootstrap, rng
        )
        per_task = {}
        selected_seeds = []
        for task, task_records in records_by_task.items():
            selected = np.asarray(
                [row["methods"][method]["selected_success"] for row in task_records],
                dtype=np.float64,
            )
            random_mean = np.asarray(
                [row["random_pool_mean"] for row in task_records], dtype=np.float64
            )
            seeds = [row["methods"][method]["selected_seed"] for row in task_records]
            selected_seeds.extend(seeds)
            per_task[task] = {
                "states": len(task_records),
                "selected_success_rate": finite(selected.mean()),
                "random_pool_mean": finite(random_mean.mean()),
                "delta_vs_random": finite((selected - random_mean).mean()),
                "state_bootstrap_ci95": task_ci[task],
                "unique_selected_seeds": len(set(seeds)),
                "max_seed_share": finite(max(Counter(seeds).values()) / len(seeds)),
            }
        deltas = np.asarray(
            [per_task[task]["delta_vs_random"] for task in sorted(per_task)],
            dtype=np.float64,
        )
        selected_rates = np.asarray(
            [per_task[task]["selected_success_rate"] for task in sorted(per_task)]
        )
        random_rates = np.asarray(
            [per_task[task]["random_pool_mean"] for task in sorted(per_task)]
        )
        methods[method] = {
            "per_task": per_task,
            "task_macro_selected_success_rate": finite(selected_rates.mean()),
            "task_macro_random_pool_mean": finite(random_rates.mean()),
            "task_macro_delta_vs_random": finite(deltas.mean()),
            "stratified_state_bootstrap_ci95": macro_ci,
            "unique_selected_seeds": len(set(selected_seeds)),
            "max_seed_share": finite(
                max(Counter(selected_seeds).values()) / len(selected_seeds)
            ),
        }

    dispersion = {}
    for method in METHOD_WINDOWS:
        per_task = {}
        correlations = []
        for task, task_records in records_by_task.items():
            x = np.asarray(
                [row["methods"][method]["dispersion"] for row in task_records]
            )
            y = np.asarray([row["opportunity"] for row in task_records])
            rho = rank_correlation(x, y)
            per_task[task] = {
                "states": len(task_records),
                "mean_dispersion": finite(x.mean()),
                "mean_opportunity": finite(y.mean()),
                "mixed_outcome_states": int(
                    sum(0 < row["success_count"] < 32 for row in task_records)
                ),
                "spearman": finite_or_none(rho),
            }
            if math.isfinite(rho):
                correlations.append(rho)
        dispersion[method] = {
            "per_task": per_task,
            "finite_task_spearman_macro": (
                finite(np.mean(correlations)) if correlations else None
            ),
            "task_heldout_ols": fit_task_heldout_dispersion(records_by_task, method),
        }

    return {
        "schema": "himoe-k32-route-medoid-audit-v1",
        "status": "posthoc_shadow_audit",
        "protocol": {
            "source": "five right-16x32 runs; first policy query per episode",
            "tasks": len(records_by_task),
            "states_per_task": 16,
            "candidates_per_state": 32,
            "state_observations": len(records),
            "candidate_rollouts": len(records) * 32,
            "probability_distance": (
                "root-mean-square Hellinger over aligned HB layer, denoise, "
                "and action-token sites after full 32-way L1 normalization"
            ),
            "expert_id_distance": (
                "mean order-invariant top-4 Jaccard distance over aligned HB "
                "layer, denoise, and action-token sites"
            ),
            "tokens": "action tokens 1-10 only; state token excluded",
            "windows": {name: [window.start, window.stop] for name, window in METHOD_WINDOWS.items()},
            "medoid_tie_break": "lowest flow-noise seed",
            "random_baseline": "exact K32 within-state mean eventual rollout success",
            "opportunity": "max observed K32 success minus exact K32 mean success",
            "bootstrap": (
                "states resampled independently within each task; task macro then averaged"
            ),
            "bootstrap_draws": args.bootstrap,
            "random_seed": args.seed,
            "task_heldout_dispersion_model": (
                "posthoc one-feature OLS trained on four tasks and tested on the fifth"
            ),
        },
        "source_integrity": sources,
        "methods": methods,
        "dispersion_opportunity": dispersion,
        "states": records,
        "limitations": [
            "Each candidate has one eventual episode outcome and no common-random-number continuation repeats.",
            "The first-query route is associated with a full episode seed stream; later replanning noise also changes with that seed.",
            "The medoid is a shadow selector requiring all K32 routes and was not executed online.",
            "Only 80 independent state pools and five tasks are available; saturated tasks make opportunity correlations undefined.",
            "Task-heldout dispersion fits are exploratory and were not preregistered.",
        ],
    }


def report_markdown(summary: dict[str, Any]) -> str:
    labels = {
        "route_prob_d0": "route probability medoid d0",
        "route_prob_d0_2": "route probability medoid d0-2",
        "route_prob_full": "route probability medoid full",
        "expert_id_d0": "expert-ID medoid d0",
        "expert_id_d0_2": "expert-ID medoid d0-2",
        "expert_id_full": "expert-ID medoid full",
    }
    lines = [
        "# K32 full-route medoid shadow audit",
        "",
        "All values use one state as the statistical unit. Random is the exact K32 "
        "pool mean, not a sampled baseline.",
        "",
        "| method | selected success | random mean | delta | state-bootstrap 95% CI | unique seeds | max seed share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, value in summary["methods"].items():
        ci = value["stratified_state_bootstrap_ci95"]
        lines.append(
            f"| {labels[method]} | {value['task_macro_selected_success_rate']:.3f} | "
            f"{value['task_macro_random_pool_mean']:.3f} | "
            f"{value['task_macro_delta_vs_random']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{value['unique_selected_seeds']} | {value['max_seed_share']:.3f} |"
        )
    lines += ["", "## Per-task deltas", ""]
    tasks = [row["task"] for row in summary["source_integrity"]]
    lines.append("| method | " + " | ".join(tasks) + " |")
    lines.append("|---|" + "---:|" * len(tasks))
    for method, value in summary["methods"].items():
        cells = [f"{value['per_task'][task]['delta_vs_random']:+.3f}" for task in tasks]
        lines.append(f"| {labels[method]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Dispersion and opportunity",
        "",
        "The opportunity target is `max(success) - mean(success)` in each K32 state "
        "pool. Correlation is undefined for a task whose opportunity is constant.",
        "",
        "| method | finite-task Spearman macro | task-heldout MAE | train-mean baseline MAE |",
        "|---|---:|---:|---:|",
    ]
    for method, value in summary["dispersion_opportunity"].items():
        held = value["task_heldout_ols"]
        rho = value["finite_task_spearman_macro"]
        rho_text = "NA" if rho is None else f"{rho:.3f}"
        lines.append(
            f"| {labels[method]} | {rho_text} | {held['pooled_state_mae']:.3f} | "
            f"{held['pooled_train_mean_baseline_mae']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This is an episode-start shadow association. Each candidate has one outcome; "
        "the first query and all later replanning noise share the candidate seed stream. "
        "There are no continuation repeats or common random numbers, and no medoid was "
        "executed online. A selected-success delta therefore is not a first-chunk causal "
        "effect.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")
    hub = args.hub_root.resolve()
    cache_root = hub / "cache" / "HiMoE-VLA"
    records = []
    sources = []
    canonical_seeds: list[int] | None = None
    for run in discover_runs(hub):
        print(f"loading {run.relative_to(cache_root)}", flush=True)
        task_records, source = load_task_states(run, cache_root)
        if canonical_seeds is None:
            canonical_seeds = source["seeds"]
        elif canonical_seeds != source["seeds"]:
            raise RuntimeError("tasks do not share the same 32 noise seeds")
        records.extend(task_records)
        sources.append(source)
    summary = summarize(records, sources, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = args.out.with_suffix(".md")
    report.write_text(report_markdown(summary), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
