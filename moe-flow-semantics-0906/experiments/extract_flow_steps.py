#!/usr/bin/env python3
"""Extract per-denoising-step MoE routing quantities from hb_router_probs.

Every established detector in this project reads the routing tensor at the final
flow step only.  This extractor keeps the step axis intact: for every query, HB
layer and denoising step it emits eight within-query quantities plus two
cross-query quantities.  Nothing here reads outcomes, episode length, wall clock
or task identity; the only inputs are ``hb_router_probs`` and the row alignment
of the frozen v4 episode caches.

Quantities, and why each one is here
------------------------------------
within-query, per (query, layer, step)

  token_entropy               mean over the ten action tokens of H(p_token).
                              How undecided the router is at this step.
  load_entropy                H(mean over action tokens of p).  Concentration of
                              the aggregate expert load produced by this step.
  token_differentiation       load_entropy - token_entropy.  This is exactly the
                              mutual information I(token; expert) under a uniform
                              token prior, so it is zero when every action token
                              routes identically and grows when the ten tokens
                              split across different experts.  It is the direct
                              measurement of "differentiated relative token
                              structure" on the step axis.
  action_consensus            mean Bhattacharyya coefficient over the 45 action
                              token pairs.  Second-order (geometric) view of the
                              same question as token_differentiation.
  state_action_alignment      mean Bhattacharyya coefficient between the state
                              token and each action token.  How much this step's
                              routing is still the conditioning route.
  conditional_energy          mean diagonal of the state-projected action Gram,
                              i.e. how much action-token route is left after the
                              shared state direction is removed.
  conditional_effective_rank  entropy effective rank of the eigenvalues of that
                              state-projected Gram, over 10.  Dimensionality of
                              the token configuration at this step.
  flow_speed                  Hellinger distance between this step's action route
                              and the previous step's.  NaN at step 0.  This is
                              the per-step summand of the cached flow_path.

cross-query, per (query, layer, step)

  mobility_step               Hellinger distance between the action route at
                              query q and query q-1 at the same step, averaged
                              over the ten action tokens.  At step 9 this is
                              bit-for-bit the definition of the frozen v4
                              ``mobility`` cache; steps 0-8 have never been
                              computed.
  state_mobility_step         the same for the state token alone.  The state
                              token carries the conditioning, so it acts as an
                              observation-change probe that the action tokens can
                              be compared against.

query-0 noise / observation decomposition
-----------------------------------------
Each task holds 50 initial states x 8 flow-noise seeds, and the same eight seeds
(1000..1007) are reused for every initial state.  At query 0 the observation is
identical inside an initial-state group, so any route difference there is caused
by the flow noise alone; conversely two episodes with the same seed and different
initial states differ only in the observation.  The extractor therefore reports,
per task, layer and step, the mean pairwise Hellinger distance for

  same_state_diff_noise   pure flow-noise sensitivity
  diff_state_same_noise   pure observation sensitivity
  diff_state_diff_noise   both

which is the step-axis analogue of "front layers respond to observation change".

Outputs are memory-mapped .npy so the analysis stage never has to hold them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import zarr  # noqa: E402


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
V4 = PROJECT / "moe-v4-0904/results/layerwise_mobility"
DEFAULT_OUTPUT = BUNDLE / "results/step_profiles"

COHORTS: dict[str, Path] = {
    "development_main": V4 / "main_reference.npz",
    "development_extra": V4 / "extra_reference.npz",
    "external_8b": V4 / "external_8b.npz",
}
METRIC_NAMES = (
    "token_entropy",
    "load_entropy",
    "token_differentiation",
    "action_consensus",
    "state_action_alignment",
    "conditional_energy",
    "conditional_effective_rank",
    "flow_speed",
)
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
PAIR_NAMES = ("same_state_diff_noise", "diff_state_same_noise", "diff_state_diff_noise")
N_LAYERS, N_STEPS, N_TOKENS, N_EXPERTS = 8, 10, 11, 32
EPSILON = 1e-12
BATCH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument(
        "--cohorts", nargs="*", default=list(COHORTS), choices=list(COHORTS)
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def normalize(raw: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(raw, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), EPSILON)


def shannon(probability: np.ndarray) -> np.ndarray:
    return -(probability * np.log(np.maximum(probability, EPSILON))).sum(axis=-1)


def effective_rank(mass: np.ndarray, maximum: int) -> np.ndarray:
    mass = np.maximum(mass, 0.0)
    share = mass / np.maximum(mass.sum(axis=-1, keepdims=True), EPSILON)
    return np.exp(shannon(share)) / float(maximum)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hellinger distance between two sqrt-probability vectors."""
    return np.sqrt(np.clip(1.0 - (left * right).sum(axis=-1), 0.0, None))


def within_query_metrics(root: np.ndarray) -> np.ndarray:
    """root is (B, 8, 10, 11, 32) sqrt-probabilities; returns (B, 8, 10, 8)."""
    probability = root * root
    action_probability = probability[:, :, :, 1:, :]
    token_entropy = shannon(action_probability).mean(axis=-1)
    load = action_probability.mean(axis=3)
    load_entropy = shannon(load)

    action_root = root[:, :, :, 1:, :]
    state_root = root[:, :, :, 0, :]
    gram = np.matmul(action_root, np.swapaxes(action_root, -1, -2))
    state_action = np.einsum("blse,blste->blst", state_root, action_root, optimize=True)
    conditional = gram - state_action[..., :, None] * state_action[..., None, :]
    diagonal = np.clip(
        np.diagonal(conditional, axis1=-2, axis2=-1), 0.0, None
    )
    eigenvalues = np.clip(np.linalg.eigvalsh(conditional), 0.0, None)
    upper = np.triu_indices(10, 1)

    flow_speed = np.full(root.shape[:3], np.nan, dtype=np.float32)
    flow_speed[:, :, 1:] = hellinger(action_root[:, :, 1:], action_root[:, :, :-1]).mean(
        axis=-1
    )

    return np.stack(
        (
            token_entropy,
            load_entropy,
            load_entropy - token_entropy,
            gram[..., upper[0], upper[1]].mean(axis=-1),
            state_action.mean(axis=-1),
            diagonal.mean(axis=-1),
            effective_rank(eigenvalues, 10),
            flow_speed,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def query_zero_pairs(root_zero: np.ndarray, init: np.ndarray, seed: np.ndarray) -> dict[str, Any]:
    """Mean pairwise Hellinger distance at query 0 by what the pair shares.

    root_zero is (E, 8, 10, 11, 32) sqrt-probabilities for one task's episodes.
    """
    count = len(init)
    same_state = init[:, None] == init[None, :]
    same_seed = seed[:, None] == seed[None, :]
    offdiag = ~np.eye(count, dtype=bool)
    masks = {
        "same_state_diff_noise": same_state & ~same_seed & offdiag,
        "diff_state_same_noise": ~same_state & same_seed & offdiag,
        "diff_state_diff_noise": ~same_state & ~same_seed & offdiag,
    }
    action = np.zeros((3, N_LAYERS, N_STEPS), dtype=np.float64)
    state = np.zeros((3, N_LAYERS, N_STEPS), dtype=np.float64)
    for layer in range(N_LAYERS):
        for step in range(N_STEPS):
            block = root_zero[:, layer, step]  # (E, 11, 32)
            action_distance = np.zeros((count, count), dtype=np.float64)
            for token in range(1, N_TOKENS):
                vectors = block[:, token]
                action_distance += np.sqrt(
                    np.clip(1.0 - vectors @ vectors.T, 0.0, None)
                )
            action_distance /= float(N_TOKENS - 1)
            state_vectors = block[:, 0]
            state_distance = np.sqrt(
                np.clip(1.0 - state_vectors @ state_vectors.T, 0.0, None)
            )
            for position, name in enumerate(PAIR_NAMES):
                mask = masks[name]
                action[position, layer, step] = action_distance[mask].mean()
                state[position, layer, step] = state_distance[mask].mean()
    return {
        "action": action.astype(np.float32),
        "state": state.astype(np.float32),
        "pair_counts": np.asarray(
            [int(masks[name].sum()) // 2 for name in PAIR_NAMES], dtype=np.int64
        ),
    }


def process_task(job: dict[str, Any]) -> dict[str, Any]:
    route = Path(job["route"])
    group = zarr.open_group(str(route), mode="r")
    source = group["hb_router_probs"]
    if tuple(source.shape[1:]) != (N_LAYERS, N_STEPS, N_TOKENS, N_EXPERTS):
        raise ValueError(f"unexpected router shape at {route}: {source.shape}")
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    count = int(source.shape[0])
    if np.any(np.diff(episode_id) < 0):
        raise ValueError(f"episode_id is not non-decreasing in {route}")

    metrics = np.empty((count, N_LAYERS, N_STEPS, len(METRIC_NAMES)), dtype=np.float32)
    mobility = np.full((count, N_LAYERS, N_STEPS), np.nan, dtype=np.float32)
    state_mobility = np.full((count, N_LAYERS, N_STEPS), np.nan, dtype=np.float32)

    episodes = np.asarray(job["episodes"], dtype=np.int64)
    lengths = np.asarray(job["lengths"], dtype=np.int64)
    starts = np.searchsorted(episode_id, episodes, side="left")
    stops = np.searchsorted(episode_id, episodes, side="right")
    if not np.array_equal(stops - starts, lengths) or int(lengths.sum()) != count:
        raise ValueError(f"episode lengths in {route} disagree with the v4 cache")
    zero_rows = starts.copy()
    root_zero = np.empty(
        (len(episodes), N_LAYERS, N_STEPS, N_TOKENS, N_EXPERTS), dtype=np.float32
    )

    batch = int(job["batch"])
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        lead = max(start - 1, 0)
        root = np.sqrt(normalize(np.asarray(source[lead:stop])))
        offset = start - lead
        metrics[start:stop] = within_query_metrics(root[offset:])
        if root.shape[0] > 1:
            action = root[:, :, :, 1:, :]
            state = root[:, :, :, 0, :]
            # entry j of the difference belongs to global row lead + 1 + j
            mobility[lead + 1 : stop] = hellinger(action[1:], action[:-1]).mean(axis=-1)
            state_mobility[lead + 1 : stop] = hellinger(state[1:], state[:-1])
        take = zero_rows[(zero_rows >= start) & (zero_rows < stop)]
        if len(take):
            root_zero[np.searchsorted(zero_rows, take)] = root[take - lead]

    # A query 0 has no predecessor, and neither does the first query of an episode.
    mobility[starts] = np.nan
    state_mobility[starts] = np.nan

    pairs = query_zero_pairs(
        root_zero, np.asarray(job["init_state"]), np.asarray(job["flow_noise_seed"])
    )

    output = Path(job["output"])
    cohort = job["cohort"]
    dense_metrics = np.lib.format.open_memmap(output / f"{cohort}_metrics.npy", mode="r+")
    dense_mobility = np.lib.format.open_memmap(output / f"{cohort}_mobility.npy", mode="r+")
    dense_state = np.lib.format.open_memmap(output / f"{cohort}_state_mobility.npy", mode="r+")
    rows = np.asarray(job["rows"], dtype=np.int64)
    for row, episode_start, length in zip(rows, starts, lengths, strict=True):
        block = slice(int(episode_start), int(episode_start) + int(length))
        dense_metrics[row, :length] = metrics[block]
        dense_mobility[row, :length] = mobility[block]
        dense_state[row, :length] = state_mobility[block]
    dense_metrics.flush()
    dense_mobility.flush()
    dense_state.flush()

    return {
        "cohort": cohort,
        "task": job["task"],
        "task_position": job["task_position"],
        "rows": len(rows),
        "queries": int(count),
        "pair_action": pairs["action"],
        "pair_state": pairs["state"],
        "pair_counts": pairs["pair_counts"],
    }


def build_jobs(cohort: str, output: Path, batch: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    cache = load_npz(COHORTS[cohort])
    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    lengths = cache["length"].astype(int)
    valid = cache["valid"].astype(bool)
    run_id = str(cache["run_id"])
    max_query = valid.shape[1]

    shapes = {
        f"{cohort}_metrics.npy": (len(episodes), max_query, N_LAYERS, N_STEPS, len(METRIC_NAMES)),
        f"{cohort}_mobility.npy": (len(episodes), max_query, N_LAYERS, N_STEPS),
        f"{cohort}_state_mobility.npy": (len(episodes), max_query, N_LAYERS, N_STEPS),
    }
    for name, shape in shapes.items():
        array = np.lib.format.open_memmap(
            output / name, mode="w+", dtype=np.float32, shape=shape
        )
        array[...] = np.nan
        array.flush()
        del array

    jobs = []
    for task_position, task in enumerate(task_names):
        rows = np.flatnonzero(task_index == task_position)
        order = np.argsort(episodes[rows], kind="stable")
        rows = rows[order]
        jobs.append(
            {
                "cohort": cohort,
                "task": task,
                "task_position": task_position,
                "route": str(ROUTE_ROOT / task / run_id / "server/routes.zarr"),
                "rows": rows,
                "episodes": episodes[rows],
                "lengths": lengths[rows],
                "init_state": cache["init_state_id"].astype(int)[rows],
                "flow_noise_seed": cache["flow_noise_seed"].astype(int)[rows],
                "output": str(output),
                "batch": batch,
            }
        )
    return jobs, cache


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".gitignore").write_text("*.npy\n", encoding="utf-8")

    records: list[dict[str, Any]] = []
    context = mp.get_context("fork")
    for cohort in args.cohorts:
        jobs, cache = build_jobs(cohort, args.output, args.batch)
        with context.Pool(processes=min(args.workers, len(jobs))) as pool:
            results = []
            for done, result in enumerate(pool.imap_unordered(process_task, jobs), 1):
                results.append(result)
                print(f"[{cohort} {done}/{len(jobs)}] {result['task']}", flush=True)
        results.sort(key=lambda item: item["task_position"])

        valid = cache["valid"].astype(bool)
        mobility = np.load(args.output / f"{cohort}_mobility.npy", mmap_mode="r")
        expected = valid.copy()
        expected[:, 0] = False
        observed = np.isfinite(np.asarray(mobility[:, :, 0, 9]))
        if not np.array_equal(observed, expected):
            raise ValueError(f"{cohort}: step-9 mobility mask does not match the cache")
        reference = load_npz(COHORTS[cohort])["mobility"]
        gap = np.abs(np.asarray(mobility[:, :, :, 9])[expected] - reference[expected])
        # v4 forms sqrt(p_a * p_b) and this script forms sqrt(p_a) * sqrt(p_b);
        # the two agree analytically and differ only by float32 rounding.
        if not np.isfinite(gap).all() or float(gap.max()) > 1e-5:
            raise ValueError(
                f"{cohort}: step-9 mobility does not reproduce the v4 cache "
                f"(max |delta| = {float(np.nanmax(gap))})"
            )

        np.savez_compressed(
            args.output / f"{cohort}_query0_pairs.npz",
            schema=np.asarray("himoe.flow_semantics.query0_pairs.v1"),
            cohort=np.asarray(cohort),
            run_id=cache["run_id"],
            task_names=cache["task_names"],
            layer_names=np.asarray(LAYER_NAMES),
            pair_names=np.asarray(PAIR_NAMES),
            action=np.stack([r["pair_action"] for r in results]),
            state=np.stack([r["pair_state"] for r in results]),
            pair_counts=np.stack([r["pair_counts"] for r in results]),
        )
        np.savez_compressed(
            args.output / f"{cohort}_index.npz",
            schema=np.asarray("himoe.flow_semantics.step_profile_index.v1"),
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
            metric_names=np.asarray(METRIC_NAMES),
        )
        records.append(
            {
                "cohort": cohort,
                "run_id": str(cache["run_id"]),
                "tasks": len(results),
                "episodes": int(len(cache["episode"])),
                "valid_queries": int(valid.sum()),
                "router_rows_read": int(sum(r["queries"] for r in results)),
                "step9_mobility_max_abs_delta_vs_v4": float(gap.max()),
            }
        )
        print(json.dumps(records[-1], indent=2), flush=True)

    summary = {
        "schema": "himoe.flow_semantics.step_extraction.v1",
        "source": "VLA_MUI_HUB/cache_new/HiMoE-VLA/<task>/<run_id>/server/routes.zarr :: hb_router_probs",
        "raw_router_shape": [N_LAYERS, N_STEPS, N_TOKENS, N_EXPERTS],
        "outcomes_loaded": False,
        "task_identity_used_inside_any_quantity": False,
        "device": "cpu",
        "metric_names": list(METRIC_NAMES),
        "layer_names": list(LAYER_NAMES),
        "pair_names": list(PAIR_NAMES),
        "records": records,
    }
    (args.output / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
