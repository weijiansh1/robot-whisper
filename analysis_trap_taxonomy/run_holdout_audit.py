#!/usr/bin/env python3
"""Train-free trap taxonomy audit on the two independent 50x8 LIBERO runs.

Physical event definitions are imported from the frozen phenotype analysis.  The
second run (flow-noise seeds 1008--1015) is treated as a confirmation set.  MoE
features come from outcome-blind route caches produced directly from routes.zarr.
No classifier, fitted feature weight, or failure-derived threshold is used here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "VLA_MUI_HUB" / "cache_new" / "HiMoE-VLA"
EVENT_MODULE = ROOT / "analysis_moe_phenotype" / "events" / "build_events.py"
SCENE8 = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"

RUNS = {
    "seed1000_1007": {
        "run_id": "right-50x8-20260903",
        "route_manifest": ROOT
        / "himoe-vla_trap/results/moe_invariant_alarm_cache_new_recomputed_50x8/route_only_manifest.json",
    },
    "seed1008_1015": {
        "run_id": "right-50x8b-20260903",
        "route_manifest": ROOT
        / "himoe-vla_trap/results/moe_invariant_alarm_cache_new_50x8b/route_only_manifest.json",
    },
}

# The frozen free-joint loop proxy is not semantically valid for articulated
# objects in these two tasks.  Static remains valid and is retained.
LOOP_INVALID_TASKS = {
    "libero_goal/open_the_middle_drawer_of_the_cabinet",
    "libero_long/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
}

LEADS = tuple(range(-4, 3))
SIGNALS = ("late_flow_volatility", "route_acceleration", "recurrence_raw")


def load_event_module():
    spec = importlib.util.spec_from_file_location("frozen_trap_events", EVENT_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EVENT_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_event(loop: int, static: int) -> str:
    if loop < 0 and static < 0:
        return "no_event"
    if loop >= 0 and static >= 0:
        if loop == static:
            return "coincident"
        return "loop_first" if loop < static else "static_first"
    return "loop_first" if loop >= 0 else "static_first"


def collect_events(label: str, cfg: dict, event_module) -> pd.DataFrame:
    references = np.asarray(
        json.loads(event_module.REF_JSON.read_text())["success_terminal_references"]
        ["moka_pot_1_joint0"],
        dtype=np.float64,
    )
    rows: list[dict] = []
    pattern = f"libero_*/*/{cfg['run_id']}"
    runs = sorted(path for path in HUB.glob(pattern) if (path / "meta.json").exists())
    if len(runs) != 40:
        raise RuntimeError(f"{label}: expected 40 complete task runs, found {len(runs)}")

    for index, run in enumerate(runs, 1):
        suite = run.parents[1].name
        task = run.parent.name
        task_key = f"{suite}/{task}"
        raw, _ = event_module.process_task(
            run, task == SCENE8, references if task == SCENE8 else None
        )
        for episode, scene, repeat, success, n_queries, loop, static, _, grade in raw:
            valid_loop = -1 if task_key in LOOP_INVALID_TASKS else int(loop)
            category = first_event(valid_loop, int(static))
            rows.append(
                {
                    "run": label,
                    "suite": suite,
                    "task": task,
                    "task_key": task_key,
                    "episode": int(episode),
                    "scene": int(scene),
                    "repeat": int(repeat),
                    "success": bool(success),
                    "failure": not bool(success),
                    "n_queries": int(n_queries),
                    "loop_onset_raw": int(loop),
                    "loop_onset": valid_loop,
                    "static_onset": int(static),
                    "loop_proxy_valid": task_key not in LOOP_INVALID_TASKS,
                    "proxy_grade": grade,
                    "first_event": category,
                }
            )
        print(f"[{label} physical {index:02d}/40] {task_key}", flush=True)
    return pd.DataFrame(rows)


def route_cache_map(manifest_path: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text())
    result = {}
    for item in manifest["tasks"]:
        cache = Path(item["cache"])
        if not cache.is_absolute():
            cache = ROOT / cache
        result[str(item["task"])] = cache
    if len(result) != 40:
        raise RuntimeError(f"{manifest_path}: expected 40 route caches, found {len(result)}")
    return result


def load_route_features(label: str, cfg: dict) -> pd.DataFrame:
    frames = []
    for task_key, cache in sorted(route_cache_map(cfg["route_manifest"]).items()):
        with np.load(cache, allow_pickle=False) as data:
            frames.append(
                pd.DataFrame(
                    {
                        "run": label,
                        "task_key": task_key,
                        "episode": np.asarray(data["episode"], np.int32),
                        "query": np.asarray(data["query"], np.int16),
                        "late_flow_volatility": np.asarray(
                            data["late_flow_volatility"], np.float32
                        ),
                        "route_acceleration": np.asarray(
                            data["route_acceleration"], np.float32
                        ),
                        "recurrence_raw": np.asarray(data["recurrence_raw"], np.float32),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def comparison_effect(value: float, controls: np.ndarray) -> float:
    controls = controls[np.isfinite(controls)]
    if not np.isfinite(value) or len(controls) == 0:
        return math.nan
    return float(np.mean(value > controls) + 0.5 * np.mean(value == controls) - 0.5)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def exact_sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return math.nan
    tail = min(positive, negative)
    p = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, p)


def onset_effects(
    events: pd.DataFrame, routes: pd.DataFrame, draws: int
) -> pd.DataFrame:
    route_by_task = {key: value for key, value in routes.groupby("task_key", sort=False)}
    records: list[dict] = []
    rng = np.random.default_rng(20260904)

    for event_name, onset_col in (("loop", "loop_onset"), ("static", "static_onset")):
        for population in ("all_events", "failed_events"):
            selected = events[events[onset_col] >= 0]
            if population == "failed_events":
                selected = selected[selected.failure]
            cell_effects: dict[tuple, list[tuple[str, int, float]]] = {}
            for row in selected.itertuples(index=False):
                task_routes = route_by_task[row.task_key]
                task_events = events[events.task_key == row.task_key]
                control_eps = task_events[
                    (task_events.scene == row.scene) & (task_events[onset_col] < 0)
                ].episode.to_numpy()
                if len(control_eps) == 0:
                    continue
                for lead in LEADS:
                    query = int(getattr(row, onset_col)) + lead
                    if query < 0 or query >= row.n_queries:
                        continue
                    at_query = task_routes[task_routes["query"] == query]
                    event_route = at_query[at_query.episode == row.episode]
                    controls = at_query[at_query.episode.isin(control_eps)]
                    if len(event_route) != 1 or len(controls) == 0:
                        continue
                    cell = (row.task_key, int(row.scene))
                    for signal in SIGNALS:
                        effect = comparison_effect(
                            float(event_route.iloc[0][signal]), controls[signal].to_numpy()
                        )
                        if np.isfinite(effect):
                            cell_effects.setdefault(cell, []).append((signal, lead, effect))

            grouped: dict[tuple[str, int], list[float]] = {}
            event_counts: dict[tuple[str, int], int] = {}
            for values in cell_effects.values():
                local: dict[tuple[str, int], list[float]] = {}
                for signal, lead, effect in values:
                    local.setdefault((signal, lead), []).append(effect)
                for key, effects in local.items():
                    grouped.setdefault(key, []).append(float(np.mean(effects)))
                    event_counts[key] = event_counts.get(key, 0) + len(effects)

            for (signal, lead), values in sorted(grouped.items()):
                array = np.asarray(values, np.float64)
                positive = int((array > 0).sum())
                negative = int((array < 0).sum())
                low, high = bootstrap_ci(array, rng, draws)
                records.append(
                    {
                        "run": events.run.iloc[0],
                        "event": event_name,
                        "population": population,
                        "signal": signal,
                        "lead": lead,
                        "effect_auc_minus_half": float(array.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "cells": len(array),
                        "event_comparisons": event_counts[(signal, lead)],
                        "positive_cells": positive,
                        "negative_cells": negative,
                        "sign_p": exact_sign_p(positive, negative),
                    }
                )
    return pd.DataFrame(records)


def inventory(events: pd.DataFrame) -> dict:
    result: dict = {}
    for run, frame in events.groupby("run"):
        run_out: dict = {
            "episodes": int(len(frame)),
            "success": int(frame.success.sum()),
            "failure": int(frame.failure.sum()),
            "failure_first_event": {},
            "all_first_event": {},
        }
        for category, group in frame.groupby("first_event"):
            run_out["all_first_event"][category] = {
                "n": int(len(group)),
                "success": int(group.success.sum()),
                "failure": int(group.failure.sum()),
                "escape_rate": float(group.success.mean()),
            }
        for category, count in frame[frame.failure].first_event.value_counts().items():
            run_out["failure_first_event"][category] = int(count)
        run_out["failure_with_any_event"] = int(
            (frame.failure & (frame.first_event != "no_event")).sum()
        )
        run_out["right_censored_no_event"] = int(
            (frame.failure & (frame.first_event == "no_event")).sum()
        )
        run_out["right_censored_fraction_of_failure"] = float(
            (frame[frame.failure].first_event == "no_event").mean()
        )
        result[run] = run_out
    return result


def termination_audit() -> dict:
    """Establish what success=false means in the frozen evaluator outputs."""
    result: dict = {}
    for label, cfg in RUNS.items():
        failures = 0
        failures_at_horizon = 0
        successes_at_or_past_horizon = 0
        task_runs = sorted(HUB.glob(f"libero_*/*/{cfg['run_id']}"))
        if len(task_runs) != 40:
            raise RuntimeError(f"{label}: expected 40 task runs for termination audit")
        for run in task_runs:
            meta = json.loads((run / "meta.json").read_text())
            summaries = json.loads((run / "client" / "summaries.json").read_text())
            horizon = int(meta["max_steps"])
            for episode in summaries:
                success = bool(episode["success"])
                action_steps = int(episode["action_steps"])
                if success:
                    successes_at_or_past_horizon += int(action_steps >= horizon)
                else:
                    failures += 1
                    failures_at_horizon += int(action_steps == horizon)
        result[label] = {
            "failures": failures,
            "failures_ending_exactly_at_max_steps": failures_at_horizon,
            "all_failures_are_horizon_censored": failures_at_horizon == failures,
            "successes_at_or_past_max_steps": successes_at_or_past_horizon,
            "interpretation": (
                "success=false means the task was unfinished when the evaluator "
                "reached max_steps; it is not a separately observed failure cause"
            ),
        }
    return result


def candidate_overlap() -> dict:
    mismatch_path = ROOT / "analysis_moe_phenotype/E7_task_manifold/mismatch_scores.csv"
    mismatch = pd.read_csv(mismatch_path)
    mismatch = (
        mismatch[mismatch.window != "early"]
        .groupby(["suite", "task", "scene", "repeat"], as_index=False)
        .wrong_manifold.max()
    )
    event_rows = []
    base = ROOT / "analysis_moe_phenotype/events/grid50x8"
    for path in base.glob("libero_*/*/events.csv"):
        frame = pd.read_csv(path)
        frame["suite"] = path.parent.parent.name
        frame["task"] = path.parent.name
        event_rows.append(frame)
    events = pd.concat(event_rows, ignore_index=True)
    joined = mismatch.merge(events, on=["suite", "task", "scene", "repeat"])
    has_event = (joined.loop_onset_q >= 0) | (joined.static_onset_q >= 0)
    wrong = joined.wrong_manifold.astype(bool)

    atlas = pd.read_csv(
        ROOT / "himoe-vla_trap/results/hub_phenotype_atlas_50x8/episode_physical_labels.csv"
    )
    atlas = atlas[atlas.failure].copy()
    atlas_events = events.copy()
    atlas_events["task"] = atlas_events.suite + "/" + atlas_events.task
    atlas_join = atlas.merge(
        atlas_events[["task", "episode_id", "loop_onset_q", "static_onset_q"]],
        left_on=["task", "episode"],
        right_on=["task", "episode_id"],
    )
    no_event = atlas_join[(atlas_join.loop_onset_q < 0) & (atlas_join.static_onset_q < 0)]
    labels = [
        "label_stagnation",
        "label_gripper_cycling",
        "label_goal_regression",
        "label_goal_approach_leave",
        "label_subtask_undo",
        "label_regrasp_or_drop",
    ]
    return {
        "wrong_manifold_failures": int(wrong.sum()),
        "wrong_manifold_with_loop_or_static": int((wrong & has_event).sum()),
        "wrong_manifold_without_loop_or_static": int((wrong & ~has_event).sum()),
        "no_event_failures_in_atlas": int(len(no_event)),
        "no_event_candidate_label_counts": {
            label.removeprefix("label_"): int(no_event[label].sum()) for label in labels
        },
        "no_event_with_no_candidate_label": int((no_event[labels].sum(axis=1) == 0).sum()),
        "note": "candidate labels overlap and were not confirmed as independent MoE types",
    }


def architecture_audit() -> dict:
    model = ROOT / (
        "himoe-vla-cache/himoe-libero-bridge/cache/upstream/HiMoE-VLA/"
        "src/moevla/models/himoe.py"
    )
    gate = model.with_name("modeling_moe.py")
    return {
        "as_moe": "layers 0,1,16,17; 1-of-3; gate input is the action-space data_mask",
        "hb_moe": "layers 2-5 and 12-15; top-4-of-32 routed experts plus one shared expert",
        "hb_gate": "softmax(W @ RMS-normalized hidden), unsorted top-k, selected weights renormalized",
        "analysis_scope": "deep HB layers 12-15, action tokens 1-10, full 32-way probabilities",
        "source_files": [str(model.relative_to(ROOT)), str(gate.relative_to(ROOT))],
        "expert_identity_rule": "no fixed expert-ID trap switch is assumed; rules are distribution dynamics",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    event_module = load_event_module()
    event_frames = []
    effect_frames = []
    for label, cfg in RUNS.items():
        events = collect_events(label, cfg, event_module)
        routes = load_route_features(label, cfg)
        expected = int(events.n_queries.sum())
        if len(routes) != expected:
            raise RuntimeError(f"{label}: route/event rows differ: {len(routes)} != {expected}")
        event_frames.append(events)
        effect_frames.append(onset_effects(events, routes, args.bootstrap))

    all_events = pd.concat(event_frames, ignore_index=True)
    effects = pd.concat(effect_frames, ignore_index=True)
    all_events.to_csv(args.output / "events.csv.gz", index=False, compression="gzip")
    effects.to_csv(args.output / "onset_route_effects.csv", index=False)

    summary = {
        "schema": "himoe.trainfree_trap_taxonomy_holdout.v1",
        "training": False,
        "failure_thresholds_used": False,
        "inventory": inventory(all_events),
        "termination_audit": termination_audit(),
        "candidate_overlap_development_run": candidate_overlap(),
        "architecture": architecture_audit(),
        "route_effect_columns": {
            "effect_auc_minus_half": "within task+scene+absolute-query event-vs-no-same-event controls",
            "ci": "task-scene cell bootstrap",
            "sign_p": "two-sided exact sign test across non-tied task-scene cells",
        },
        "limitations": [
            "loop/static are query-boundary kinematic proxies, not contact/video truth",
            "no-event failures are right-censored, not proven behaviorally normal",
            "wrong-manifold and old physical labels are candidate modifiers, not confirmed extra types",
            "cached route features validate V/A/recurrence; entropy/consensus evidence remains development-run only",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["inventory"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
