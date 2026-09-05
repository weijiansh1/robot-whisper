#!/usr/bin/env python3
"""Extract outcome-blind early-query MoE fingerprints for task cold start."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
DEFAULT_LAYER_ROOT = BUNDLE / "results/layerwise_mobility"
DEFAULT_HUB = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
DEFAULT_OUTPUT = BUNDLE / "results/unknown_task/early_route_fingerprints.npz"

PREFIX = 4
FINAL_FLOW = 9
ACTION = slice(1, 11)
EXPECTED_LAYERS = 8
EXPECTED_FLOWS = 10
EXPECTED_TOKENS = 11
EXPECTED_EXPERTS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-root", type=Path, default=DEFAULT_LAYER_ROOT)
    parser.add_argument("--hub", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=("main", "extra", "external"),
        default=("main", "extra", "external"),
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def normalize(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return output / np.maximum(output.sum(axis=-1, keepdims=True), 1e-12)


def extract_cohort(
    cache: dict[str, np.ndarray], hub: Path
) -> dict[str, np.ndarray]:
    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    lengths = cache["length"].astype(int)
    run_id = str(cache["run_id"])
    n_episode = len(episodes)

    final_action_queries = np.empty(
        (n_episode, PREFIX, EXPECTED_LAYERS, EXPECTED_EXPERTS), dtype=np.float16
    )
    final_state_queries = np.empty_like(final_action_queries)
    all_flow_action = np.empty(
        (n_episode, EXPECTED_LAYERS, EXPECTED_FLOWS, EXPECTED_EXPERTS),
        dtype=np.float16,
    )

    for task_position, task in enumerate(task_names):
        take = np.flatnonzero(task_index == task_position)
        run = hub / task / run_id / "server/routes.zarr"
        group = zarr.open_group(str(run), mode="r")
        router = group["hb_router_probs"]
        episode_id = np.asarray(group["episode_id"][:], dtype=int)
        expected_shape = (
            EXPECTED_LAYERS,
            EXPECTED_FLOWS,
            EXPECTED_TOKENS,
            EXPECTED_EXPERTS,
        )
        if router.shape[1:] != expected_shape:
            raise ValueError(f"unexpected route shape in {run}: {router.shape}")

        for start in range(0, len(take), 32):
            rows = take[start : start + 32]
            query_indices: list[int] = []
            for row in rows:
                if lengths[row] < PREFIX:
                    raise ValueError(
                        f"episode shorter than prefix: {task}/{episodes[row]}"
                    )
                indices = np.flatnonzero(episode_id == episodes[row])
                if len(indices) != lengths[row]:
                    raise ValueError(
                        f"route/index mismatch: {task}/{episodes[row]}"
                    )
                query_indices.extend(indices[:PREFIX].tolist())

            probability = normalize(np.asarray(router.oindex[query_indices]))
            probability = probability.reshape(
                len(rows),
                PREFIX,
                EXPECTED_LAYERS,
                EXPECTED_FLOWS,
                EXPECTED_TOKENS,
                EXPECTED_EXPERTS,
            )
            action = probability[..., ACTION, :].mean(axis=-2)
            state = probability[..., 0, :]
            final_action_queries[rows] = action[:, :, :, FINAL_FLOW]
            final_state_queries[rows] = state[:, :, :, FINAL_FLOW]
            all_flow_action[rows] = action.mean(axis=1)

        print(
            f"[{task_position + 1}/{len(task_names)}] {task}",
            flush=True,
        )

    return {
        "task_names": task_names,
        "task_index": task_index.astype(np.int16),
        "episode": episodes.astype(np.int16),
        "final_action_queries": final_action_queries,
        "final_state_queries": final_state_queries,
        "all_flow_action": all_flow_action,
    }


def main() -> None:
    args = parse_args()
    available = {
        "main": load_npz(args.layer_root / "main_reference.npz"),
        "extra": load_npz(args.layer_root / "extra_reference.npz"),
        "external": load_npz(args.layer_root / "external_8b.npz"),
    }
    cohorts = {name: available[name] for name in args.cohorts}
    extracted = {
        name: extract_cohort(cache, args.hub) for name, cache in cohorts.items()
    }
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray("himoe.early_route_fingerprint.v1"),
        "prefix": np.asarray(PREFIX, dtype=np.int16),
    }
    for cohort, values in extracted.items():
        for key, value in values.items():
            payload[f"{cohort}_{key}"] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
