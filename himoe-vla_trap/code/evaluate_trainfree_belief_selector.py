#!/usr/bin/env python3
"""Evaluate a train-free stale-belief rejector and its candidate-ranking control."""

from __future__ import annotations

import argparse
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
import zarr

import analyze_failed_grasp_moe_dynamics as base
from trainfree_belief_selector import (
    calibrate_healthy,
    candidate_gap_scores,
    evaluate_stale_belief,
    normalize,
    select_min_gap,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/trainfree_belief_selector.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/trainfree_belief_selector"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_event_branches(config: dict[str, Any]) -> tuple[base.Branch, list[base.Branch]]:
    event_config = json.loads(
        (PACKAGE_ROOT / config["base_event_config"]).read_text(encoding="utf-8")
    )
    run_root = (WORKSPACE_ROOT / event_config["run_root"]).resolve()
    snapshot_dir = (
        run_root
        / "formal"
        / f"worker{event_config['worker']}"
        / f"snapshot_{event_config['snapshot']:03d}"
    )
    failed, controls = base.load_branches(event_config, snapshot_dir)
    store = zarr.open_group(str(run_root / "formal/server/routes.zarr"), mode="r")
    base.attach_routes([failed, *controls], store)
    return failed, controls


def routes_at(branches: list[base.Branch], relative: int) -> np.ndarray:
    values = []
    for branch in branches:
        assert branch.route is not None
        values.append(branch.route[branch.event.query + relative])
    return np.stack(values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_online_healthy_reference(
    controls: list[base.Branch],
    low: int,
    high: int,
    path: Path,
) -> dict[str, Any]:
    """Freeze healthy-only phase banks and bounds for online inference."""

    relative_queries = np.arange(low, high + 1, dtype=np.int16)
    route_banks = []
    threshold_rows = []
    for relative in relative_queries:
        current = routes_at(controls, int(relative))
        previous = routes_at(controls, int(relative) - 1)
        thresholds = calibrate_healthy(current, previous)
        route_banks.append(current.astype(np.float16))
        threshold_rows.append([
            thresholds.layer5_gap_upper,
            thresholds.back_chunk_jump_upper,
            thresholds.front_action_distance_upper,
        ])

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        schema=np.asarray("himoe.online_healthy_reference.v1"),
        training=np.asarray(False),
        failure_labels_used=np.asarray(False),
        relative_queries=relative_queries,
        route_banks=np.stack(route_banks),
        thresholds=np.asarray(threshold_rows, dtype=np.float64),
        threshold_names=np.asarray([
            "layer5_gap_upper",
            "back_chunk_jump_upper",
            "front_action_distance_upper",
        ]),
        control_candidates=np.asarray(
            [branch.candidate for branch in controls], dtype=np.int32
        ),
        control_episode_ids=np.asarray(
            [branch.episode_id for branch in controls], dtype=np.int64
        ),
        control_closure_queries=np.asarray(
            [branch.event.query for branch in controls], dtype=np.int16
        ),
    )
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": "himoe.online_healthy_reference.v1",
        "relative_queries": relative_queries.tolist(),
        "healthy_trajectories": len(controls),
        "route_bank_shape": list(np.stack(route_banks).shape),
        "training": False,
        "failure_labels_used": False,
    }


def evaluate_trigger(
    failed: base.Branch,
    controls: list[base.Branch],
    low: int,
    high: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = [failed, *controls]
    for target in targets:
        references = controls if target is failed else [
            branch for branch in controls if branch is not target
        ]
        for relative in range(low, high + 1):
            reference_current = routes_at(references, relative)
            reference_previous = routes_at(references, relative - 1)
            thresholds = calibrate_healthy(reference_current, reference_previous)
            assert target.route is not None
            query = target.event.query + relative
            decision = evaluate_stale_belief(
                target.route[query],
                target.route[query - 1],
                reference_current,
                thresholds,
            )
            rows.append({
                "candidate": target.candidate,
                "episode_id": target.episode_id,
                "evaluation_role": "belief_trap" if target is failed else "healthy_loo",
                "relative_query": relative,
                "query": query,
                **decision.to_dict(),
                "layer5_gap_upper": thresholds.layer5_gap_upper,
                "back_chunk_jump_upper": thresholds.back_chunk_jump_upper,
                "front_action_distance_upper": thresholds.front_action_distance_upper,
            })
    return pd.DataFrame(rows)


def first_route_rows(route_episode: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    unique, first = np.unique(route_episode, return_index=True)
    lookup = {int(episode): int(row) for episode, row in zip(unique, first)}
    if not set(map(int, episode_ids)).issubset(lookup):
        raise RuntimeError("route store is missing episode IDs")
    return np.asarray([lookup[int(episode)] for episode in episode_ids], np.int64)


def candidate_audit(
    audit_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    hub_root = (WORKSPACE_ROOT / audit_config["hub_root"]).resolve()
    summary_paths = sorted(hub_root.glob(audit_config["run_glob"]))
    if len(summary_paths) != 5:
        raise RuntimeError(f"expected five right-16x32 runs, found {len(summary_paths)}")
    pool_size = int(audit_config["pool_size"])
    pools_per_state = int(audit_config["pools_per_state"])
    layer_axis = int(audit_config["layer5_axis"])
    choices: list[dict[str, Any]] = []
    permutation_units: list[dict[str, Any]] = []
    max_state_route_range = 0.0

    for summary_path in summary_paths:
        run = summary_path.parent.parent
        records = sorted(
            json.loads(summary_path.read_text(encoding="utf-8")),
            key=lambda row: int(row["episode_index"]),
        )
        episode_ids = np.asarray([int(row["episode_index"]) for row in records])
        states = np.asarray([int(row["init_state_id"]) for row in records])
        seeds = np.asarray([int(row["flow_noise_seed"]) for row in records])
        success = np.asarray([bool(row["success"]) for row in records])
        task = f"{run.parents[1].name}/{run.parent.name}"
        route_store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        route_rows = first_route_rows(
            np.asarray(route_store["episode_id"][:], dtype=np.int64), episode_ids
        )
        layer_routes = np.asarray(
            route_store["hb_router_probs"].get_orthogonal_selection(
                (route_rows, [layer_axis], slice(None), slice(None), slice(None))
            ),
            dtype=np.float64,
        )
        layer_routes = layer_routes[:, 0]
        for state in sorted(np.unique(states)):
            take = np.flatnonzero(states == state)
            take = take[np.argsort(seeds[take], kind="stable")]
            if len(take) != pool_size * pools_per_state:
                raise RuntimeError(f"unexpected candidate count for {task} state {state}")
            state_token = normalize(layer_routes[take, :, 0])
            max_state_route_range = max(
                max_state_route_range,
                float(np.ptp(state_token, axis=0).max()),
            )
            unit = {
                "task": task,
                "init_state_id": int(state),
                "success": success[take].copy(),
                "selected_local_indices": [],
            }
            for fold in range(pools_per_state):
                local = np.arange(fold * pool_size, (fold + 1) * pool_size)
                pool = take[local]
                pool_routes = layer_routes[pool, None]
                scores = candidate_gap_scores(pool_routes, layer_axis=0)
                selected_seed = select_min_gap(
                    pool_routes, candidate_ids=seeds[pool], layer_axis=0
                )
                selected_in_pool = int(np.flatnonzero(seeds[pool] == selected_seed)[0])
                selected_index = int(pool[selected_in_pool])
                unit["selected_local_indices"].append(int(local[selected_in_pool]))
                choices.append({
                    "task": task,
                    "init_state_id": int(state),
                    "pool_fold": fold,
                    "selected_seed": selected_seed,
                    "selected_score": float(scores[selected_in_pool]),
                    "selected_success": bool(success[selected_index]),
                    "random_expected_success": float(success[pool].mean()),
                    "delta_vs_random": float(
                        success[selected_index] - success[pool].mean()
                    ),
                    "pool_successes": int(success[pool].sum()),
                    "pool_size": pool_size,
                })
            permutation_units.append(unit)

    choice_frame = pd.DataFrame(choices)
    task_frame = (
        choice_frame.groupby("task", sort=True)
        .agg(
            pools=("selected_success", "size"),
            selected_success=("selected_success", "mean"),
            random_expected_success=("random_expected_success", "mean"),
            delta_vs_random=("delta_vs_random", "mean"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(int(audit_config["random_seed"]))
    cluster_delta = (
        choice_frame.groupby(["task", "init_state_id"], sort=True)
        .delta_vs_random.mean()
        .reset_index()
    )
    task_clusters = [
        group.delta_vs_random.to_numpy(float)
        for _, group in cluster_delta.groupby("task", sort=True)
    ]
    bootstrap = np.empty(int(audit_config["bootstrap_draws"]), np.float64)
    for draw in range(len(bootstrap)):
        bootstrap[draw] = np.mean([
            values[rng.integers(0, len(values), len(values))].mean()
            for values in task_clusters
        ])

    observed = float(task_frame.delta_vs_random.mean())
    permutation = np.empty(int(audit_config["permutation_draws"]), np.float64)
    for draw in range(len(permutation)):
        selected_by_task: dict[str, list[float]] = {}
        random_by_task: dict[str, list[float]] = {}
        for unit in permutation_units:
            labels = rng.permutation(unit["success"])
            selected = labels[np.asarray(unit["selected_local_indices"], dtype=int)]
            selected_by_task.setdefault(unit["task"], []).extend(selected.astype(float))
            random_by_task.setdefault(unit["task"], []).append(float(labels.mean()))
        permutation[draw] = np.mean([
            np.mean(selected_by_task[task]) - np.mean(random_by_task[task])
            for task in sorted(selected_by_task)
        ])
    p_improvement = float((1 + np.sum(permutation >= observed)) / (len(permutation) + 1))

    macro = {
        "tasks": int(len(task_frame)),
        "states": int(len(cluster_delta)),
        "pools": int(len(choice_frame)),
        "selected_success": float(task_frame.selected_success.mean()),
        "random_expected_success": float(task_frame.random_expected_success.mean()),
        "delta_vs_random": observed,
        "state_cluster_bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "permutation_p_improvement": p_improvement,
        "informative_pools": int(
            ((choice_frame.pool_successes > 0) & (choice_frame.pool_successes < pool_size)).sum()
        ),
        "state_token_candidate_max_probability_range": max_state_route_range,
    }
    return choice_frame, task_frame, macro, permutation_units


def plot_results(
    trigger: pd.DataFrame,
    task_audit: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    failed = trigger[trigger.evaluation_role == "belief_trap"]
    ax = axes[0]
    ax.plot(failed.relative_query, failed.layer5_gap_ratio, marker="o", label="layer-5 gap / healthy max")
    ax.plot(failed.relative_query, failed.back_chunk_jump_ratio, marker="o", label="back jump / healthy max")
    ax.plot(failed.relative_query, failed.front_action_distance_ratio, marker="o", label="front action distance / healthy max")
    alarm = failed[failed.reject_stale_chunk]
    ax.scatter(alarm.relative_query, alarm.layer5_gap_ratio, color="#b43c3c", marker="x", s=70, label="reject")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set(title="Train-free transition-split trigger", xlabel="Queries from failed closure", ylabel="Ratio to healthy upper bound")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    x = np.arange(len(task_audit))
    width = 0.36
    ax.bar(x - width / 2, task_audit.random_expected_success, width, label="random expectation", color="#8db3c7")
    ax.bar(x + width / 2, task_audit.selected_success, width, label="min-gap selector", color="#b43c3c")
    ax.set_xticks(x, [f"task {index + 1}" for index in x])
    ax.set_ylim(0, 1.08)
    ax.set(title="Generic K8 candidate-selection negative control", ylabel="Success rate")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output, bbox_inches="tight", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve()
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    failed, controls = load_event_branches(config)
    low, high = map(int, config["trigger_relative_query_range"])
    trigger = evaluate_trigger(failed, controls, low, high)
    online_reference = write_online_healthy_reference(
        controls,
        low,
        high,
        output / "online_healthy_reference.npz",
    )
    choices, task_audit, candidate_macro, _ = candidate_audit(
        config["candidate_audit"]
    )
    trigger.to_csv(table_dir / "trigger_decisions.csv", index=False)
    choices.to_csv(table_dir / "candidate_pool_choices.csv", index=False)
    task_audit.to_csv(table_dir / "candidate_task_summary.csv", index=False)
    plot_results(
        trigger,
        task_audit,
        figure_dir / "trainfree_belief_selector_audit.png",
    )

    failed_rows = trigger[trigger.evaluation_role == "belief_trap"]
    control_rows = trigger[trigger.evaluation_role == "healthy_loo"]
    alarm_queries = failed_rows[
        failed_rows.reject_stale_chunk
    ].relative_query.astype(int).tolist()
    summary = {
        "schema": "himoe.trainfree_belief_selector.v1",
        "training": False,
        "failure_labels_used_by_selector": False,
        "healthy_reference_used_for_calibration": True,
        "healthy_reference_requires_known_normal_demonstrations": True,
        "trigger": {
            "name": config["trigger_rule"]["name"],
            "decision": "reject current chunk and invoke an external fallback",
            "healthy_calibration": config["trigger_rule"]["threshold"],
            "conditions": config["trigger_rule"]["conditions"],
            "failed_case_alarm_queries": alarm_queries,
            "first_failed_case_alarm_query": alarm_queries[0] if alarm_queries else None,
            "failed_case_alarm_count": len(alarm_queries),
            "healthy_loo_decisions": int(len(control_rows)),
            "healthy_loo_alarm_count": int(control_rows.reject_stale_chunk.sum()),
            "healthy_loo_trajectories_with_alarm": int(
                control_rows.groupby("candidate").reject_stale_chunk.any().sum()
            ),
            "query_plus_2": jsonable(
                failed_rows[failed_rows.relative_query == 2].iloc[0].to_dict()
            ),
            "evidence_status": "development_case_only_post_hoc_rule",
        },
        "generic_candidate_ranker": {
            "name": "min_layer5_full_flow_state_action_gap",
            "score_uses": "routing only; no action, reward, or outcome",
            **candidate_macro,
            "decision": "negative_control_failed_do_not_deploy_as_generic_selector",
        },
        "online_healthy_reference": online_reference,
        "interpretation": (
            "The train-free rule can reject the stale action at q+2 in this matched "
            "case without firing on the seven leave-one-out healthy controls. However, "
            "minimizing the same gap as a generic K8 action selector is worse than random "
            "on the independent five-task episode-start grid."
        ),
        "limitations": [
            "The trigger conjunction was derived after inspecting this one failed-grasp case; its zero-control-alarm result is not an independent test.",
            "Phase-matched healthy references are required online; threshold calibration is train-free but task/stage specific.",
            "Rejecting a stale chunk does not specify a successful fallback action.",
            "The K8 audit is an independent generic-selection negative control at episode start, not a causal post-grasp intervention test.",
            "A causal selector test requires multiple candidates executed from the exact q+2 simulator state with common future noise.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "OK",
        "trigger_alarm_queries": alarm_queries,
        "healthy_loo_alarms": int(control_rows.reject_stale_chunk.sum()),
        "generic_candidate_delta": candidate_macro["delta_vs_random"],
        "online_healthy_reference": online_reference["path"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
