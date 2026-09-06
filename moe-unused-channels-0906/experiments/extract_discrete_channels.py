#!/usr/bin/env python3
"""One pass over the raw route logs for the four channels nobody has opened.

Reads, per task and run_id:

  hb_router_probs    (query, 8, 10, 11, 32)  the exhausted channel, needed only
                                             to verify the other three against
  hb_expert_ids      (query, 8, 10, 11, 4)   the discrete top-k selection
  hb_selected_prob   (query, 8, 10, 11, 4)   its gate probabilities
  hb_entropy         (query, 8, 10, 11)      routing entropy
  as_probs           (query, 4, 3)           the second router
  as_expert_ids      (query, 4)              its selection

and emits two things.

1. Verification statistics (PREREG C1-C5), accumulated exactly, no sampling.
   The top-k test is deliberately tie-robust: `hb_expert_ids` is a valid top-4
   set if and only if the four gathered probabilities sum to the maximal
   4-subset mass of the 32-way row.  Comparing index sets against `argsort`
   instead would count float16 boundary ties as violations.

2. Per (episode, chunk, layer) quantities, in the (episode, chunk, 8) layout
   the frozen protocol expects, so they drop straight into
   `dev.representations` with no reimplementation.

  query_hard_churn      Jaccard distance between the top-4 set at chunk q and
                        q-1, averaged over the ten denoising steps and the ten
                        action tokens.  Port of
                        VLA_MUI_HUB/moe-physical-failure-dynamics `query_hard_churn`.
  query_hard_churn_d9   the same at denoising step 9 only.  This is the exact
                        discrete analogue of the frozen `mobility` cache, which
                        is also step 9 only, so the two are comparable
                        like-for-like.
  query_top1_churn      fraction of (step, action token) cells whose argmax
                        expert changed between chunks.  The coarsest possible
                        discrete change.
  denoise_hard_churn    the same Jaccard distance between consecutive denoising
                        steps inside one chunk.  Discrete analogue of flow_path.
  set_dwell             causal run length, in chunks, over which the step-9
                        top-4 set of an action token has been unchanged,
                        averaged over the ten action tokens.
  tie_margin            p_(4) - p_(5) of the 32-way gate, averaged over steps
                        and action tokens.  How close the selection boundary is
                        to flipping; small means one perturbation away.
  selected_mass         sum of the four logged selected probabilities, averaged
                        over steps and action tokens.
  hb_entropy_action     the logged hb_entropy averaged over steps and action
                        tokens.  Verification target against the frozen
                        flow-semantics `token_entropy`.

Nothing here reads an outcome label, an episode length beyond the frozen row
alignment, wall clock, or task identity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import zarr  # noqa: E402


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
V4 = PROJECT / "moe-v4-0904/results/layerwise_mobility"
DEFAULT_OUTPUT = BUNDLE / "results/channels"

COHORTS: dict[str, Path] = {
    "development_main": V4 / "main_reference.npz",
    "development_extra": V4 / "extra_reference.npz",
    "external_8b": V4 / "external_8b.npz",
}
QUANTITIES = (
    "query_hard_churn",
    "query_hard_churn_d9",
    "query_top1_churn",
    "denoise_hard_churn",
    "set_dwell",
    "tie_margin",
    "selected_mass",
    "hb_entropy_action",
)
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
N_LAYERS, N_STEPS, N_TOKENS, N_EXPERTS, TOP_K = 8, 10, 11, 32, 4
N_AS_LAYERS, N_AS_EXPERTS = 4, 3
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--cohorts", nargs="*", default=list(COHORTS), choices=list(COHORTS))
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def top4_mask(ids: np.ndarray) -> np.ndarray:
    """(..., 4) expert indices -> (...) uint32 bitmask.  Raises on duplicates."""
    mask = np.zeros(ids.shape[:-1], dtype=np.uint32)
    for position in range(TOP_K):
        mask |= np.left_shift(np.uint32(1), ids[..., position].astype(np.uint32))
    if np.any(np.bitwise_count(mask) != TOP_K):
        raise ValueError("duplicate expert index inside a top-4 set")
    return mask


def mask_churn(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Jaccard distance between two top-4 sets, identical to the two HUB bundles."""
    intersection = np.bitwise_count(left & right).astype(np.float32)
    return 1.0 - intersection / np.maximum(2.0 * TOP_K - intersection, 1.0)


def process_task(job: dict[str, Any]) -> dict[str, Any]:
    route = Path(job["route"])
    group = zarr.open_group(str(route), mode="r")
    attrs = dict(group.attrs)
    probs_store = group["hb_router_probs"]
    ids_store = group["hb_expert_ids"]
    selected_store = group["hb_selected_prob"]
    entropy_store = group["hb_entropy"]
    count = int(probs_store.shape[0])
    if tuple(probs_store.shape[1:]) != (N_LAYERS, N_STEPS, N_TOKENS, N_EXPERTS):
        raise ValueError(f"unexpected router shape at {route}")
    if tuple(ids_store.shape[1:]) != (N_LAYERS, N_STEPS, N_TOKENS, TOP_K):
        raise ValueError(f"unexpected expert-id shape at {route}")
    if tuple(entropy_store.shape[1:]) != (N_LAYERS, N_STEPS, N_TOKENS):
        raise ValueError(f"unexpected entropy shape at {route}")

    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    if np.any(np.diff(episode_id) < 0):
        raise ValueError(f"episode_id not non-decreasing in {route}")
    episodes = np.asarray(job["episodes"], dtype=np.int64)
    lengths = np.asarray(job["lengths"], dtype=np.int64)
    starts = np.searchsorted(episode_id, episodes, side="left")
    stops = np.searchsorted(episode_id, episodes, side="right")
    if not np.array_equal(stops - starts, lengths) or int(lengths.sum()) != count:
        raise ValueError(f"episode lengths in {route} disagree with the v4 cache")

    # ---- the second router, read whole; it is 16 values per row -------------
    as_probs = np.asarray(group["as_probs"][:], dtype=np.float32)
    as_ids = np.asarray(group["as_expert_ids"][:], dtype=np.int64)
    if as_probs.shape[1:] != (N_AS_LAYERS, N_AS_EXPERTS) or as_ids.shape[1:] != (N_AS_LAYERS,):
        raise ValueError(f"unexpected as_* shape at {route}")
    as_row_deviation = float(np.abs(as_probs - as_probs[0]).max())
    as_unique_prob_rows = int(len(np.unique(as_probs.round(6).reshape(count, -1), axis=0)))
    as_unique_id_rows = int(len(np.unique(as_ids, axis=0)))
    as_argmax_match = float((as_probs.argmax(axis=-1) == as_ids).mean())
    as_sum_deviation = float(np.abs(as_probs.sum(axis=-1) - 1.0).max())
    # within-episode deviation, the quantity the frame question turns on
    as_within_episode = 0.0
    for start, stop in zip(starts, stops, strict=True):
        block = as_probs[start:stop]
        as_within_episode = max(
            as_within_episode, float(np.abs(block - block[0]).max())
        )

    values = np.full((count, N_LAYERS, len(QUANTITIES)), np.nan, dtype=np.float32)
    same_d9 = np.zeros((count, N_LAYERS, N_TOKENS - 1), dtype=bool)
    verify = {
        "cells": 0,
        "gather_max_abs_delta": 0.0,
        "selected_mass_max_abs_delta": 0.0,
        "topk_set_violations": 0,
        "boundary_ties": 0,
        "argsort_set_mismatch": 0,
        "entropy_max_abs_delta": 0.0,
        "entropy_max_float16_ulp": 0.0,
        "ids_all_zero_cells": 0,
        "ids_min": 255,
        "ids_max": 0,
        "expert_seen": np.zeros(N_EXPERTS, dtype=np.int64),
        "selected_prob_max": 0.0,
        "selected_prob_min": 1.0,
    }

    batch = int(job["batch"])
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        # Read one extra leading row so the cross-chunk difference is exact at
        # every batch seam; entry j of the difference belongs to row lead+1+j.
        lead = max(start - 1, 0)
        offset = start - lead
        window_probs = np.asarray(probs_store[lead:stop], dtype=np.float32)
        window_ids = np.asarray(ids_store[lead:stop], dtype=np.int64)
        probs = window_probs[offset:]
        ids = window_ids[offset:]
        selected = np.asarray(selected_store[start:stop], dtype=np.float32)
        entropy = np.asarray(entropy_store[start:stop], dtype=np.float32)
        size = stop - start

        # ---- C1: selected probabilities are the gathered gate values -------
        gathered = np.take_along_axis(probs, ids, axis=-1)
        verify["gather_max_abs_delta"] = max(
            verify["gather_max_abs_delta"], float(np.abs(gathered - selected).max())
        )

        # ---- C2: the selected set attains the maximal 4-subset mass --------
        top5 = np.sort(np.partition(probs, -5, axis=-1)[..., -5:], axis=-1)
        top4_mass = top5[..., 1:].sum(axis=-1)
        fourth, fifth = top5[..., 1], top5[..., 0]
        deviation = np.abs(selected.sum(axis=-1) - top4_mass)
        verify["selected_mass_max_abs_delta"] = max(
            verify["selected_mass_max_abs_delta"], float(deviation.max())
        )
        verify["topk_set_violations"] += int((deviation > 0.0).sum())
        tied = fourth == fifth
        verify["boundary_ties"] += int(tied.sum())
        argsort_top4 = np.argsort(-probs, axis=-1, kind="stable")[..., :TOP_K]
        verify["argsort_set_mismatch"] += int(
            (~(np.sort(argsort_top4, axis=-1) == np.sort(ids, axis=-1)).all(-1)).sum()
        )

        # ---- C3: hb_entropy is the entropy of the full 32-way row ----------
        reference = -(probs * np.log(np.maximum(probs, EPSILON))).sum(axis=-1)
        verify["entropy_max_abs_delta"] = max(
            verify["entropy_max_abs_delta"], float(np.abs(entropy - reference).max())
        )
        verify["entropy_max_float16_ulp"] = max(
            verify["entropy_max_float16_ulp"],
            float(np.spacing(np.float16(np.abs(reference).max()))),
        )

        # ---- C4: placeholder check -----------------------------------------
        verify["ids_all_zero_cells"] += int((ids == 0).all(axis=-1).sum())
        verify["ids_min"] = min(verify["ids_min"], int(ids.min()))
        verify["ids_max"] = max(verify["ids_max"], int(ids.max()))
        verify["expert_seen"] += np.bincount(ids.ravel(), minlength=N_EXPERTS)
        verify["cells"] += int(size * N_LAYERS * N_STEPS * N_TOKENS)
        verify["selected_prob_max"] = max(verify["selected_prob_max"], float(selected.max()))
        verify["selected_prob_min"] = min(verify["selected_prob_min"], float(selected.min()))

        # ---- quantities -----------------------------------------------------
        action = slice(1, N_TOKENS)
        window_mask = top4_mask(window_ids)  # (offset + size, 8, 10, 11)
        window_top1 = window_probs.argmax(axis=-1).astype(np.int16)

        # within-chunk denoising churn
        mask = window_mask[offset:]
        values[start:stop, :, QUANTITIES.index("denoise_hard_churn")] = mask_churn(
            mask[:, :, 1:, action], mask[:, :, :-1, action]
        ).mean(axis=(2, 3))

        # boundary margin, selected mass, logged entropy
        values[start:stop, :, QUANTITIES.index("tie_margin")] = (
            (fourth - fifth)[:, :, :, action].mean(axis=(2, 3))
        )
        values[start:stop, :, QUANTITIES.index("selected_mass")] = (
            selected.sum(axis=-1)[:, :, :, action].mean(axis=(2, 3))
        )
        values[start:stop, :, QUANTITIES.index("hb_entropy_action")] = entropy[
            :, :, :, action
        ].mean(axis=(2, 3))

        # cross-chunk quantities; entry j belongs to global row lead + 1 + j
        if window_mask.shape[0] > 1:
            later, earlier = window_mask[1:], window_mask[:-1]
            values[lead + 1 : stop, :, QUANTITIES.index("query_hard_churn")] = mask_churn(
                later[:, :, :, action], earlier[:, :, :, action]
            ).mean(axis=(2, 3))
            values[lead + 1 : stop, :, QUANTITIES.index("query_hard_churn_d9")] = mask_churn(
                later[:, :, 9, action], earlier[:, :, 9, action]
            ).mean(axis=2)
            values[lead + 1 : stop, :, QUANTITIES.index("query_top1_churn")] = (
                window_top1[1:, :, :, action] != window_top1[:-1, :, :, action]
            ).mean(axis=(2, 3), dtype=np.float32)
            same_d9[lead + 1 : stop] = (
                later[:, :, 9, action] == earlier[:, :, 9, action]
            )

        del probs, window_probs, gathered, top5, reference, argsort_top4

    # The first query of an episode has no predecessor.
    cross = [
        QUANTITIES.index(name)
        for name in ("query_hard_churn", "query_hard_churn_d9", "query_top1_churn")
    ]
    values[np.ix_(starts, np.arange(N_LAYERS), np.asarray(cross))] = np.nan
    same_d9[starts] = False

    # set_dwell: consecutive chunks whose step-9 top-4 set is unchanged.  A run
    # ending at row q has length q - (last row where the set changed), and the
    # episode start always counts as a change, so the recursion resets there.
    position = np.arange(count, dtype=np.int32).reshape(count, 1, 1)
    last_change = np.maximum.accumulate(
        np.where(same_d9, np.int32(-1), position), axis=0
    )
    values[:, :, QUANTITIES.index("set_dwell")] = (
        (position - last_change).astype(np.float32).mean(axis=2)
    )

    output = Path(job["output"])
    cohort = job["cohort"]
    dense = np.lib.format.open_memmap(output / f"{cohort}_quantities.npy", mode="r+")
    rows = np.asarray(job["rows"], dtype=np.int64)
    for row, episode_start, length in zip(rows, starts, lengths, strict=True):
        block = slice(int(episode_start), int(episode_start) + int(length))
        dense[row, :length] = values[block]
    dense.flush()

    verify["expert_seen"] = verify["expert_seen"].tolist()
    return {
        "cohort": cohort,
        "task": job["task"],
        "task_position": job["task_position"],
        "queries": count,
        "attrs": attrs,
        "verify": verify,
        "as_probs_first": as_probs[0].tolist(),
        "as_expert_ids_first": as_ids[0].tolist(),
        "as_row_deviation": as_row_deviation,
        "as_within_episode_deviation": as_within_episode,
        "as_unique_prob_rows": as_unique_prob_rows,
        "as_unique_id_rows": as_unique_id_rows,
        "as_argmax_match": as_argmax_match,
        "as_sum_deviation": as_sum_deviation,
    }


def build_jobs(cohort: str, output: Path, batch: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    cache = load_npz(COHORTS[cohort])
    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    lengths = cache["length"].astype(int)
    run_id = str(cache["run_id"])
    max_query = cache["valid"].shape[1]

    array = np.lib.format.open_memmap(
        output / f"{cohort}_quantities.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(episodes), max_query, N_LAYERS, len(QUANTITIES)),
    )
    array[...] = np.nan
    array.flush()
    del array

    jobs = []
    for task_position, task in enumerate(task_names):
        rows = np.flatnonzero(task_index == task_position)
        rows = rows[np.argsort(episodes[rows], kind="stable")]
        jobs.append(
            {
                "cohort": cohort,
                "task": task,
                "task_position": task_position,
                "route": str(ROUTE_ROOT / task / run_id / "server/routes.zarr"),
                "rows": rows,
                "episodes": episodes[rows],
                "lengths": lengths[rows],
                "output": str(output),
                "batch": batch,
            }
        )
    return jobs, cache


def merge_verify(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = {
        "cells": 0,
        "gather_max_abs_delta": 0.0,
        "selected_mass_max_abs_delta": 0.0,
        "topk_set_violations": 0,
        "boundary_ties": 0,
        "argsort_set_mismatch": 0,
        "entropy_max_abs_delta": 0.0,
        "entropy_max_float16_ulp": 0.0,
        "ids_all_zero_cells": 0,
        "ids_min": 255,
        "ids_max": 0,
        "selected_prob_max": 0.0,
        "selected_prob_min": 1.0,
    }
    seen = np.zeros(N_EXPERTS, dtype=np.int64)
    for record in records:
        block = record["verify"]
        for key in ("cells", "topk_set_violations", "boundary_ties",
                    "argsort_set_mismatch", "ids_all_zero_cells"):
            total[key] += int(block[key])
        for key in ("gather_max_abs_delta", "selected_mass_max_abs_delta",
                    "entropy_max_abs_delta", "entropy_max_float16_ulp",
                    "selected_prob_max", "ids_max"):
            total[key] = max(total[key], block[key])
        for key in ("ids_min", "selected_prob_min"):
            total[key] = min(total[key], block[key])
        seen += np.asarray(block["expert_seen"], dtype=np.int64)
    total["experts_never_selected"] = int((seen == 0).sum())
    total["expert_selection_counts"] = seen.tolist()
    return total


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".gitignore").write_text("*.npy\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "schema": "himoe.unused_channels.extraction.v1",
        "source": "VLA_MUI_HUB/cache_new/HiMoE-VLA/<task>/<run_id>/server/routes.zarr",
        "arrays_read": [
            "hb_router_probs", "hb_expert_ids", "hb_selected_prob", "hb_entropy",
            "as_probs", "as_expert_ids", "episode_id",
        ],
        "outcomes_loaded": False,
        "task_identity_used_inside_any_quantity": False,
        "subsampling": "none; every query row of every task of both run_ids",
        "device": "cpu",
        "quantity_names": list(QUANTITIES),
        "layer_names": list(LAYER_NAMES),
        "cohorts": {},
    }
    context = mp.get_context("fork")
    for cohort in args.cohorts:
        jobs, cache = build_jobs(cohort, args.output, args.batch)
        with context.Pool(processes=min(args.workers, len(jobs))) as pool:
            results = []
            for done, result in enumerate(pool.imap_unordered(process_task, jobs), 1):
                results.append(result)
                print(f"[{cohort} {done}/{len(jobs)}] {result['task']}", flush=True)
        results.sort(key=lambda item: item["task_position"])

        attrs = results[0]["attrs"]
        if any(r["attrs"].get("format") != attrs.get("format") for r in results):
            raise ValueError(f"{cohort}: inconsistent route trace format")

        np.savez_compressed(
            args.output / f"{cohort}_as_router.npz",
            schema=np.asarray("himoe.unused_channels.as_router.v1"),
            cohort=np.asarray(cohort),
            run_id=cache["run_id"],
            task_names=cache["task_names"],
            as_probs_first=np.asarray([r["as_probs_first"] for r in results], np.float32),
            as_expert_ids_first=np.asarray([r["as_expert_ids_first"] for r in results], np.int64),
            as_row_deviation=np.asarray([r["as_row_deviation"] for r in results], np.float64),
            as_within_episode_deviation=np.asarray(
                [r["as_within_episode_deviation"] for r in results], np.float64
            ),
            as_unique_prob_rows=np.asarray([r["as_unique_prob_rows"] for r in results], np.int64),
            as_unique_id_rows=np.asarray([r["as_unique_id_rows"] for r in results], np.int64),
            queries=np.asarray([r["queries"] for r in results], np.int64),
        )
        np.savez_compressed(
            args.output / f"{cohort}_index.npz",
            schema=np.asarray("himoe.unused_channels.index.v1"),
            cohort=np.asarray(cohort),
            run_id=cache["run_id"],
            task_names=cache["task_names"],
            task_index=cache["task_index"],
            episode=cache["episode"],
            init_state_id=cache["init_state_id"],
            flow_noise_seed=cache["flow_noise_seed"],
            length=cache["length"],
            valid=cache["valid"],
            layer_names=np.asarray(LAYER_NAMES),
            quantity_names=np.asarray(QUANTITIES),
        )
        summary["cohorts"][cohort] = {
            "run_id": str(cache["run_id"]),
            "route_trace_attrs": attrs,
            "tasks": len(results),
            "episodes": int(len(cache["episode"])),
            "valid_queries": int(cache["valid"].sum()),
            "router_rows_read": int(sum(r["queries"] for r in results)),
            "verification": merge_verify(results),
            "as_router": {
                "max_within_episode_deviation": max(
                    r["as_within_episode_deviation"] for r in results
                ),
                "max_within_task_deviation": max(r["as_row_deviation"] for r in results),
                "max_unique_prob_rows_within_task": max(
                    r["as_unique_prob_rows"] for r in results
                ),
                "max_unique_id_rows_within_task": max(
                    r["as_unique_id_rows"] for r in results
                ),
                "argmax_matches_selected_id": min(r["as_argmax_match"] for r in results),
                "max_abs_sum_minus_one": max(r["as_sum_deviation"] for r in results),
                "distinct_prob_values_across_tasks": int(
                    len(
                        np.unique(
                            np.asarray([r["as_probs_first"] for r in results], np.float32)
                            .round(6)
                            .reshape(len(results), -1),
                            axis=0,
                        )
                    )
                ),
                "distinct_id_values_across_tasks": int(
                    len(np.unique(np.asarray([r["as_expert_ids_first"] for r in results]), axis=0))
                ),
            },
        }
        print(json.dumps(summary["cohorts"][cohort], indent=2, default=str), flush=True)

    (args.output / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print("wrote", args.output / "extraction_summary.json", flush=True)


if __name__ == "__main__":
    main()
