#!/usr/bin/env python3
"""Cache multi-lag HB routing displacements for the progress ratio.

R(q, W) = d(z_q, z_{q-W}) / sum_{k=1..W} d(z_{q-k+1}, z_{q-k})

Only the numerator needs raw routes: the denominator is a rolling sum of the
adjacent-query distances, and the adjacent-query distance is exactly what the v4
layerwise-mobility cache already stores.  Lag 1 is recomputed anyway, because
reproducing the v4 ``mobility`` array bit-for-bit (atol 2e-5) is the correctness
anchor that makes this cache comparable with every earlier bundle.  The anchor is
mandatory: a mismatch aborts the build instead of being written out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "method"))

import progress_ratio  # noqa: E402


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
ROUTE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
V4 = WORKSPACE / "moe-v4-0904/results/layerwise_mobility"
DEFAULT_OUTPUT = BUNDLE / "results/progress_cache"

SCHEMA = "himoe.progress_ratio_v12.cache.v1"
ANCHOR_TOLERANCE = 2e-5
RAW_SHAPE = (len(progress_ratio.LAYER_NAMES), 10, 11, progress_ratio.N_EXPERTS)
ROUTE_SHAPE = (
    len(progress_ratio.LAYER_NAMES),
    progress_ratio.N_ACTION_TOKENS,
    progress_ratio.N_EXPERTS,
)

COHORTS: dict[str, dict[str, Any]] = {
    "development_main": {"cache": V4 / "main_reference.npz", "root": ROUTE_ROOT},
    "development_extra": {"cache": V4 / "extra_reference.npz", "root": ROUTE_ROOT},
    "external_8b": {"cache": V4 / "external_8b.npz", "root": ROUTE_ROOT},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORTS))
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


def episode_lag_distances(
    hb_router_probs: np.ndarray, lags: tuple[int, ...]
) -> np.ndarray:
    """一个 episode 的 [Q, 8, 10, 11, 32] -> [Q, len(lags), 8]。"""
    routes = progress_ratio.action_route(np.asarray(hb_router_probs, dtype=np.float32))
    return progress_ratio.lag_distances(routes, lags)


def assert_mobility_anchor(
    lag_distance: np.ndarray, mobility: np.ndarray, valid: np.ndarray
) -> None:
    """lag-1 必须复现 v4 的 mobility，否则本缓存与既往全部结果不可比。"""
    computed = lag_distance[:, :, 0, :]
    mask = valid[:, :, None] & np.isfinite(computed) & np.isfinite(mobility)
    if not mask.any():
        raise ValueError("no overlapping finite lag-1 entries to anchor against")
    deviation = float(np.abs(computed[mask] - mobility[mask]).max())
    if deviation > ANCHOR_TOLERANCE:
        raise ValueError(
            f"lag-1 distances disagree with the v4 mobility cache by {deviation:.3e}"
        )


def anchor_deviation(
    lag_distance: np.ndarray, mobility: np.ndarray, valid: np.ndarray
) -> float:
    """报告用的偏差，不作判定；判定始终走 assert_mobility_anchor。"""
    computed = lag_distance[:, :, 0, :]
    mask = valid[:, :, None] & np.isfinite(computed) & np.isfinite(mobility)
    return float(np.abs(computed[mask] - mobility[mask]).max())


def task_routes(route_path: Path, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Stream one task's zarr into [rows, 8, 10, 32] action routes."""
    if not route_path.is_dir():
        raise FileNotFoundError(route_path)
    group = zarr.open_group(str(route_path), mode="r")
    source = group["hb_router_probs"]
    if tuple(source.shape[1:]) != RAW_SHAPE:
        raise ValueError(f"unexpected HB shape at {route_path}: {source.shape}")
    episode_id = np.asarray(group["episode_id"][:], dtype=int)
    count = len(episode_id)
    if count != source.shape[0]:
        raise ValueError(
            f"episode_id/hb_router_probs row mismatch at {route_path}: "
            f"{count} != {source.shape[0]}"
        )
    if batch < 1:
        raise ValueError("batch must be positive")
    routes = np.empty((count,) + ROUTE_SHAPE, dtype=np.float32)
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        routes[start:stop] = progress_ratio.action_route(
            np.asarray(source[start:stop], dtype=np.float32)
        )
    return episode_id, routes


def validate_existing(path: Path, expected_shape: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive["lag_distance"].shape) != expected_shape:
                return False
            return np.array_equal(
                np.asarray(archive["lags"]), np.asarray(progress_ratio.LAGS, np.int16)
            )
    except Exception:
        return False


def build_cohort(
    cohort: str,
    cache_path: Path,
    route_root: Path,
    output: Path,
    batch: int,
    resume: bool,
) -> dict[str, Any]:
    cache = load_npz(cache_path)
    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    lengths = cache["length"].astype(int)
    valid = cache["valid"].astype(bool)
    mobility = np.asarray(cache["mobility"], dtype=np.float32)
    lags = progress_ratio.LAGS
    n_episode, max_query = valid.shape
    expected_shape = (n_episode, max_query, len(lags), len(progress_ratio.LAYER_NAMES))
    destination = output / f"{cohort}.npz"

    if resume and validate_existing(destination, expected_shape):
        print(f"[{cohort}] resume: {destination} already matches", flush=True)
        return {
            "cohort": cohort,
            "cache": str(destination),
            "cache_bytes": destination.stat().st_size,
            "episodes": n_episode,
            "tasks": len(task_names),
            "resumed": True,
        }

    if not np.array_equal(
        np.asarray(cache["layer_names"]).astype(str),
        np.asarray(progress_ratio.LAYER_NAMES),
    ):
        raise ValueError(f"{cohort}: layer names disagree with the method module")

    run_id = str(cache["run_id"])
    lag_distance = np.full(expected_shape, np.nan, dtype=np.float32)
    started = time.perf_counter()
    for task_position, task in enumerate(task_names):
        global_rows = np.flatnonzero(task_index == task_position)
        if len(global_rows) == 0:
            raise ValueError(f"{cohort}: no cache rows for task {task}")
        route_path = route_root / task / run_id / "server/routes.zarr"
        episode_id, routes = task_routes(route_path, batch)
        task_lookup = {
            int(episode): int(row)
            for row, episode in zip(global_rows, episodes[global_rows])
        }
        seen = set()
        for episode in np.unique(episode_id):
            positions = np.flatnonzero(episode_id == episode)
            if len(positions) > 1 and not np.all(np.diff(positions) == 1):
                raise ValueError(f"non-contiguous episode {episode} in {route_path}")
            row = task_lookup.get(int(episode))
            if row is None:
                raise ValueError(f"episode {episode} missing from cache for {task}")
            length = int(lengths[row])
            if len(positions) != length:
                raise ValueError(
                    f"length mismatch for {task} episode {episode}: "
                    f"{len(positions)} != {length}"
                )
            lag_distance[row, :length] = progress_ratio.lag_distances(
                routes[positions], lags
            )
            seen.add(row)
        missing = sorted(set(task_lookup.values()) - seen)
        if missing:
            raise ValueError(
                f"{cohort}: {len(missing)} cache rows of {task} absent from {route_path}"
            )
        print(
            f"[{cohort} {task_position + 1}/{len(task_names)}] {task} "
            f"rows={len(episode_id)} episodes={len(seen)}",
            flush=True,
        )

    expected_lag1 = valid.copy()
    expected_lag1[:, 0] = False
    observed_lag1 = np.isfinite(lag_distance[:, :, 0, :]).all(axis=2)
    if not np.array_equal(expected_lag1, observed_lag1):
        raise ValueError(f"{cohort}: lag-1 finite mask does not match the route cache")

    assert_mobility_anchor(lag_distance, mobility, valid)
    deviation = anchor_deviation(lag_distance, mobility, valid)
    elapsed = time.perf_counter() - started
    print(
        f"[{cohort}] anchor: max lag-1 deviation vs v4 mobility = {deviation:.3e} "
        f"(tolerance {ANCHOR_TOLERANCE:.1e}) in {elapsed:.1f}s",
        flush=True,
    )

    output.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(SCHEMA),
            cohort=np.asarray(cohort),
            run_id=np.asarray(run_id),
            source_cache=np.asarray(str(cache_path)),
            source_cache_sha256=np.asarray(sha256(cache_path)),
            lags=np.asarray(lags, dtype=np.int16),
            layer_names=cache["layer_names"],
            task_names=cache["task_names"],
            task_index=cache["task_index"],
            episode=cache["episode"],
            init_state_id=cache["init_state_id"],
            length=cache["length"],
            valid=cache["valid"],
            lag_distance=lag_distance,
        )
    os.replace(temporary, destination)
    return {
        "cohort": cohort,
        "cache": str(destination),
        "cache_bytes": destination.stat().st_size,
        "episodes": n_episode,
        "tasks": len(task_names),
        "queries": int(valid.sum()),
        "run_id": run_id,
        "max_lag1_deviation": deviation,
        "seconds": round(elapsed, 1),
        "resumed": False,
    }


def main() -> None:
    args = parse_args()
    selected = args.cohort if args.cohort else list(COHORTS)
    records = [
        build_cohort(
            cohort,
            COHORTS[cohort]["cache"],
            COHORTS[cohort]["root"],
            args.output,
            args.batch,
            args.resume,
        )
        for cohort in selected
    ]
    summary = {
        "schema": "himoe.progress_ratio_v12.cache_build.v1",
        "lags": list(progress_ratio.LAGS),
        "anchor_tolerance": ANCHOR_TOLERANCE,
        "outcomes_loaded": False,
        "records": records,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
