#!/usr/bin/env python3
"""Extract per-HB-layer adjacent-query action-route mobility from a route run."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import zarr


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
HUB = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
FINAL_FLOW = 9
ACTION = slice(1, 11)
N_LAYERS = 8
N_EXPERTS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def extract(cache: dict[str, np.ndarray], run_id: str) -> np.ndarray:
    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    lengths = cache["length"].astype(int)
    n_query = cache["valid"].shape[1]
    output = np.full((len(episodes), n_query, N_LAYERS), np.nan, dtype=np.float32)

    for task_position, task in enumerate(task_names):
        take = np.flatnonzero(task_index == task_position)
        if len(take) != 400:
            raise ValueError(f"expected 400 episodes for {task}, found {len(take)}")
        run = HUB / task / run_id / "server/routes.zarr"
        group = zarr.open_group(str(run), mode="r")
        source = group["hb_router_probs"]
        episode_id = np.asarray(group["episode_id"][:], dtype=int)
        route = normalize(np.asarray(source[:, :, FINAL_FLOW, ACTION, :]))
        if route.shape[1:] != (N_LAYERS, 10, N_EXPERTS):
            raise ValueError(
                f"unexpected final-action route shape in {run}: {route.shape}"
            )
        affinity = np.sqrt(route[1:] * route[:-1]).sum(axis=-1)
        adjacent = np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0)).mean(axis=-1)

        for global_row in take:
            episode = int(episodes[global_row])
            indices = np.flatnonzero(episode_id == episode)
            length = int(lengths[global_row])
            if len(indices) != length or (
                len(indices) > 1 and not np.all(np.diff(indices) == 1)
            ):
                raise ValueError(f"route/index mismatch for {task} episode {episode}")
            if length > 1:
                output[global_row, 1:length] = adjacent[indices[:-1]]
        print(
            f"[layer mobility {task_position + 1}/{len(task_names)}] {task}",
            flush=True,
        )
    return output


def main() -> None:
    args = parse_args()
    cache = load_npz(args.feature_cache)
    mobility = extract(cache, args.run_id)
    expected = cache["valid"].copy()
    expected[:, 0] = False
    if not np.array_equal(np.isfinite(mobility).all(axis=-1), expected):
        raise ValueError("layer-mobility finite mask does not match route cache")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("himoe.layerwise_mobility.v1"),
        source_feature_cache_sha256=np.asarray(sha256(args.feature_cache)),
        run_id=np.asarray(args.run_id),
        task_names=cache["task_names"],
        task_index=cache["task_index"],
        episode=cache["episode"],
        init_state_id=cache["init_state_id"],
        flow_noise_seed=cache["flow_noise_seed"],
        length=cache["length"],
        valid=cache["valid"],
        layer_names=np.asarray(("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")),
        mobility=mobility,
    )
    print(f"wrote {args.output} with shape {mobility.shape}", flush=True)


if __name__ == "__main__":
    main()
