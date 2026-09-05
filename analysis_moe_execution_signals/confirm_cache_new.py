#!/usr/bin/env python3
"""Run the frozen F2 confirmation on 800 untouched cache_new episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from analysis_moe_execution_signals.execution_features import combine_weights  # noqa: E402
from analysis_moe_execution_signals.run import (  # noqa: E402
    SEED,
    query_execution_features,
    scatter,
    stage_rank_residual_multi,
    trailing_mean,
)
from analysis_signal_matrix_trainfree import run as baseline  # noqa: E402


SCHEMA = "himoe.execution_cache_new_confirmation.v1"
TASK_ROOT = (
    ROOT
    / "VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_long/"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
)
RUNS = (
    TASK_ROOT / "right-50x8-20260903",
    TASK_ROOT / "right-50x8b-20260903",
)
SIGNALS = ("query_support_churn", "query_exec_churn")
EVENTS = ("trap", "static")
LEADS = (-4, -2)
BLOCK_ROWS = 16


def load_confirmation_corpus() -> tuple[baseline.Corpus, dict]:
    summaries_by_run = [
        json.loads((run / "client/summaries.json").read_text()) for run in RUNS
    ]
    total_episodes = sum(len(items) for items in summaries_by_run)
    max_t = max(int(item["inference_calls"]) for items in summaries_by_run for item in items)
    rowidx = np.full((total_episodes, max_t), -1, np.int32)
    actions = np.full((total_episodes, max_t, 10, 7), np.nan, np.float32)
    eef = np.full((total_episodes, max_t, 3), np.nan, np.float64)
    objects = np.full((total_episodes, max_t, 2, 3), np.nan, np.float64)
    gripper = np.full((total_episodes, max_t), np.nan, np.float64)
    meta_rows = []

    global_episode = 0
    route_offset = 0
    for run, summaries in zip(RUNS, summaries_by_run, strict=True):
        route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        route_episode = np.asarray(route["episode_id"][:], np.int32)
        route_control = np.asarray(route["control_step"][:], np.int32)
        for expected_episode, item in enumerate(summaries):
            if int(item["episode_index"]) != expected_episode:
                raise RuntimeError(f"{run.name}: summary episode order changed")
            calls = int(item["inference_calls"])
            positions = np.flatnonzero(route_episode == expected_episode)
            if len(positions) != calls:
                raise RuntimeError(
                    f"{run.name}/episode {expected_episode}: routes={len(positions)} calls={calls}"
                )
            # The v2 writer keeps control_step global within one server run;
            # episode_id is the authoritative episode boundary.
            np.testing.assert_array_equal(route_control[positions], positions)
            rowidx[global_episode, :calls] = route_offset + positions
            with np.load(run / f"client/episode_{expected_episode:02d}.npz") as data:
                state = np.asarray(data["state"], np.float64)
                sim = np.asarray(data["sim_state"], np.float64)
                action = np.asarray(data["actions"], np.float32)
            if state.shape[0] != calls or sim.shape[0] != calls or action.shape[0] != calls:
                raise RuntimeError(f"{run.name}/episode {expected_episode}: client length mismatch")
            actions[global_episode, :calls] = action
            eef[global_episode, :calls] = state[:, :3]
            objects[global_episode, :calls, 0] = sim[:, 10:13]
            objects[global_episode, :calls, 1] = sim[:, 17:20]
            gripper[global_episode, :calls] = state[:, 6:8].mean(axis=1)
            meta_rows.append(
                {
                    "episode_id": global_episode,
                    "source_run": run.name,
                    "source_episode_id": expected_episode,
                    "success": bool(item["success"]),
                    "T": calls,
                    "group": int(item["init_state_id"]),
                    "flow_noise_seed": int(item["flow_noise_seed"]),
                }
            )
            global_episode += 1
        route_offset += len(route_episode)

    if global_episode != total_episodes:
        raise RuntimeError("confirmation episode count mismatch")
    valid = rowidx >= 0
    definitions = json.loads(
        (baseline.A_ROOT / "analysis_dryrun_v10/physical_label_definitions.json").read_text()
    )
    references = np.asarray(
        definitions["success_terminal_references"]["moka_pot_1_joint0"], np.float64
    )
    loop = np.full(total_episodes, -1, np.int16)
    static = np.full(total_episodes, -1, np.int16)
    trap = np.full(total_episodes, -1, np.int16)
    goal = np.full((total_episodes, max_t), np.nan, np.float64)
    for episode, item in enumerate(meta_rows):
        length = int(item["T"])
        loop[episode], static[episode], trap[episode] = baseline.physical_onsets(
            eef[episode, :length],
            objects[episode, :length],
            gripper[episode, :length],
            references,
        )
        goal[episode, :length] = baseline.goal_distance(
            objects[episode, :length], references
        )

    meta = pd.DataFrame(meta_rows)
    corpus = baseline.Corpus(
        tag="C",
        meta=meta,
        valid=valid,
        rowidx=rowidx,
        actions=actions,
        match_group=meta.group.astype(str).to_numpy(),
        trap_onset=trap,
        loop_onset=loop,
        static_onset=static,
        trap_proxy=trap.copy(),
        goal_distance=goal,
        physical_progress=baseline.trailing_physical(
            goal, valid, baseline.ALIGN_WINDOW
        ),
        eef_motion=np.full(valid.shape, np.nan),
        loop_ratio=np.full(valid.shape, np.nan),
    )
    inventory = {
        "episodes": total_episodes,
        "route_rows": int(route_offset),
        "initial_states": int(meta.group.nunique()),
        "seeds": int(meta.flow_noise_seed.nunique()),
        "successes": int(meta.success.sum()),
        "trap_events": int((trap >= 0).sum()),
        "loop_events": int((loop >= 0).sum()),
        "static_events": int((static >= 0).sum()),
        "ground_truth": "query_proxy",
    }
    return corpus, inventory


def _soft_f0(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = baseline.normalize_probability(np.asarray(probabilities, np.float32))
    action = p[:, 4:, :, 1:, :]
    late = action[:, :, baseline.FLOW_LATE_START :]
    volatility = (
        1.0 - baseline.weighted_jaccard(late[:, :, 1:], late[:, :, :-1])
    ).mean(axis=(1, 2, 3))
    root = np.sqrt(action)
    acceleration = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
    acceleration = (
        np.linalg.norm(acceleration, axis=-1).mean(axis=(1, 2, 3)) / np.sqrt(2.0)
    )
    return volatility, acceleration


def extract_confirmation_rows(rebuild: bool) -> tuple[dict[str, np.ndarray], dict]:
    cache = HERE / "cache_new_confirmation_rows.npz"
    if cache.exists() and not rebuild:
        with np.load(cache) as data:
            if str(data["schema"].item()) == SCHEMA:
                arrays = {
                    key: np.asarray(data[key])
                    for key in data.files
                    if key not in {"schema", "audit_json"}
                }
                return arrays, {**json.loads(str(data["audit_json"].item())), "cache_reused": True}

    stores = [zarr.open_group(str(run / "server/routes.zarr"), mode="r") for run in RUNS]
    total = sum(int(store["episode_id"].shape[0]) for store in stores)
    arrays = {
        "final_ids": np.empty((total, 8, 11, 4), np.uint8),
        "final_weights": np.empty((total, 8, 11, 4), np.float16),
        "late_flow_volatility": np.empty(total, np.float32),
        "route_acceleration": np.empty(total, np.float32),
    }
    offset = 0
    for run, store in zip(RUNS, stores, strict=True):
        rows = int(store["episode_id"].shape[0])
        for start in range(0, rows, BLOCK_ROWS):
            stop = min(rows, start + BLOCK_ROWS)
            target = slice(offset + start, offset + stop)
            p = np.asarray(store["hb_router_probs"][start:stop], np.float32)
            ids = np.asarray(store["hb_expert_ids"][start:stop])
            raw = np.asarray(store["hb_selected_prob"][start:stop], np.float32)
            volatility, acceleration = _soft_f0(p)
            arrays["late_flow_volatility"][target] = volatility
            arrays["route_acceleration"][target] = acceleration
            arrays["final_ids"][target] = ids[:, :, baseline.DENOISE_FINAL]
            arrays["final_weights"][target] = combine_weights(
                raw[:, :, baseline.DENOISE_FINAL]
            ).astype(np.float16)
            if stop % 4096 < BLOCK_ROWS or stop == rows:
                print(f"[{run.name}] rows {stop}/{rows}", flush=True)
        offset += rows
    audit = {
        "rows": total,
        "stores": [str(run.relative_to(ROOT)) for run in RUNS],
        "source_open_mode": "r",
    }
    np.savez_compressed(
        cache,
        schema=np.asarray(SCHEMA),
        audit_json=np.asarray(json.dumps(audit, sort_keys=True)),
        **arrays,
    )
    return arrays, {**audit, "cache_reused": False}


def _cell_groups(
    corpus: baseline.Corpus,
    score: np.ndarray,
    event_kind: str,
    relative: int,
) -> pd.DataFrame:
    onset = corpus.trap_onset if event_kind == "trap" else corpus.static_onset
    rows, _ = baseline.matched_group_statistics(
        score,
        corpus,
        relative,
        "all_no_event",
        onset_values=onset,
    )
    return pd.DataFrame(rows)


def _maxT(
    groups: pd.DataFrame, cells: list[tuple[str, str, int]], nperm: int
) -> dict[tuple[str, str, int], tuple[float, float]]:
    names = np.asarray(sorted(groups.group.astype(str).unique()))
    index = {name: i for i, name in enumerate(names)}
    effects = np.full((len(names), len(cells)), np.nan)
    weights = np.zeros((len(names), len(cells)))
    for column, (event, signal, relative) in enumerate(cells):
        part = groups[
            (groups.event_kind == event)
            & (groups.signal == signal)
            & (groups.relative == relative)
        ]
        for row in part.itertuples():
            i = index[str(row.group)]
            effects[i, column] = float(row.auc_trap_high) - 0.5
            weights[i, column] = float(row.pairs)
    denominator = weights.sum(axis=0)
    if np.any(denominator == 0):
        raise RuntimeError("a frozen confirmation cell has no matched pairs")
    observed = np.abs(np.nansum(effects * weights, axis=0) / denominator)
    rng = np.random.default_rng(SEED + 90000)
    null = np.empty((nperm, len(cells)))
    for draw in range(nperm):
        signs = rng.choice((-1.0, 1.0), len(names))
        null[draw] = np.abs(
            np.nansum(effects * weights * signs[:, None], axis=0) / denominator
        )
    null_max = null.max(axis=1)
    point = (1 + (null >= observed).sum(axis=0)) / (nperm + 1)
    family = (1 + (null_max[:, None] >= observed).sum(axis=0)) / (nperm + 1)
    return {
        cell: (float(point[i]), float(family[i])) for i, cell in enumerate(cells)
    }


def evaluate(
    corpus: baseline.Corpus,
    arrays: dict[str, np.ndarray],
    nperm: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_scalar, _token = query_execution_features(arrays, corpus)
    raw = {
        signal: trailing_mean(query_scalar[signal], baseline.ALIGN_WINDOW)
        for signal in SIGNALS
    }
    f0 = (
        trailing_mean(scatter(arrays["late_flow_volatility"], corpus.rowidx), 4),
        trailing_mean(scatter(arrays["route_acceleration"], corpus.rowidx), 4),
    )
    residual = {
        signal: stage_rank_residual_multi(score, f0, corpus.match_group)
        for signal, score in raw.items()
    }
    cells = [(event, signal, relative) for event in EVENTS for signal in SIGNALS for relative in LEADS]
    aggregate_rows = []
    group_rows = []
    for variant, mapping in (("raw_descriptive", raw), ("residual_f0_primary", residual)):
        for event, signal, relative in cells:
            groups = _cell_groups(corpus, mapping[signal], event, relative)
            for row in groups.to_dict("records"):
                row.update(
                    event_kind=event,
                    signal=signal,
                    relative=relative,
                    variant=variant,
                )
                group_rows.append(row)
            pairs = int(groups.pairs.sum())
            auc = float(groups.concordant.sum() / pairs)
            lo, hi = baseline.bootstrap_group_auc(
                groups,
                nperm,
                np.random.default_rng(SEED + 80000 + len(aggregate_rows)),
            )
            aggregate_rows.append(
                {
                    "corpus": "C_cache_new",
                    "event_kind": event,
                    "signal": signal,
                    "relative": relative,
                    "variant": variant,
                    "auc_event_high": auc,
                    "det_auc": max(auc, 1.0 - auc),
                    "direction": "event_high" if auc >= 0.5 else "event_low",
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_groups": len(groups),
                    "n_events": int(groups.n_events.sum()),
                    "npairs": pairs,
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    group_frame = pd.DataFrame(group_rows)
    tests = _maxT(
        group_frame[group_frame.variant == "residual_f0_primary"], cells, nperm
    )
    for (event, signal, relative), (point, family) in tests.items():
        mask = (
            (aggregate.variant == "residual_f0_primary")
            & (aggregate.event_kind == event)
            & (aggregate.signal == signal)
            & (aggregate.relative == relative)
        )
        aggregate.loc[mask, ["p_point", "p_maxT"]] = point, family
    return aggregate, group_frame


def write_report(frame: pd.DataFrame, inventory: dict, nperm: int) -> None:
    primary = frame[frame.variant == "residual_f0_primary"].sort_values(
        ["event_kind", "relative", "signal"]
    )
    confirmed = primary[(primary.direction == "event_low") & (primary.p_maxT <= 0.05)]
    lines = [
        "# cache_new frozen confirmation",
        "",
        f"Primary family: 2 signals x 2 event definitions x 2 leads; {nperm} group sign flips, one maxT family.",
        "",
        "## Admission",
        "",
        f"- Episodes: {inventory['episodes']}; initial states: {inventory['initial_states']}; seeds: {inventory['seeds']}.",
        f"- Success: {inventory['successes']}/{inventory['episodes']}; Trap/static proxy events: "
        f"{inventory['trap_events']}/{inventory['static_events']}.",
        "- This is an untouched holdout relative to the A/B feature and lead selection; its onset remains a query proxy.",
        "",
        "## Primary F0-residual results",
        "",
        "| event | signal | lead | direction | det AUC | 95% CI of event-high AUC | point p | maxT p |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {row.event_kind} | `{row.signal}` | {int(row.relative)} | {row.direction} | "
            f"{row.det_auc:.3f} | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] | "
            f"{row.p_point:.4f} | {row.p_maxT:.4f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"Confirmed event-low cells: {len(confirmed)}/8.",
        "A confirmed result means actual cross-query execution support freezes before the event even after removing the two old F0 loop signals. "
        "It remains observational and does not show that forcing churn, rerouting, or truncating a chunk improves behavior.",
    ]
    (HERE / "cache_new_confirmation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nperm", type=int, default=5000)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    corpus, inventory = load_confirmation_corpus()
    print(f"admitted {inventory}", flush=True)
    arrays, audit = extract_confirmation_rows(args.rebuild_cache)
    frame, groups = evaluate(corpus, arrays, args.nperm)
    frame.to_csv(HERE / "cache_new_confirmation.csv", index=False)
    groups.to_csv(HERE / "cache_new_confirmation_group_effects.csv", index=False)
    summary = {
        "schema": SCHEMA,
        "frozen_protocol": "CACHE_NEW_CONFIRMATION.md",
        "n_permutations": args.nperm,
        "inventory": inventory,
        "capture_audit": audit,
        "primary_family_cells": len(EVENTS) * len(SIGNALS) * len(LEADS),
        "expected_direction": "event_low",
    }
    (HERE / "cache_new_confirmation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(frame, inventory, args.nperm)
    print(f"wrote cache_new confirmation to {HERE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
