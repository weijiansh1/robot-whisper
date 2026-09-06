#!/usr/bin/env python3
"""Extract layer mobility, route acceleration and lag periodicity from raw routes.

CPU only, one worker process per task.  No outcome file is opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from raw_route_features import (
    BUNDLE,
    COHORTS,
    MAX_QUERY,
    LAYER_NAMES,
    WORKSPACE,
    load_npz,
    task_block,
)


DEFAULT_OUTPUT = BUNDLE / "results/raw_features"
PUBLISHED_ROUTE_FEATURES = {
    "external_8b": WORKSPACE
    / "double-selete/trainfree/results/online_precision_cascade_external/unlabeled_query_features.npz",
    "development_main": WORKSPACE
    / "double-selete/trainfree/results/online_multihead_hub/unlabeled_query_features.npz",
    "legacy_16x32": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=sorted(COHORTS), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker(payload):
    cohort, task, episodes, lengths = payload
    started = time.time()
    block = task_block(COHORTS[cohort], task, episodes, lengths)
    return task, block, time.time() - started


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    spec = COHORTS[args.cohort]
    layer = load_npz(spec.layer_cache)
    task_names = layer["task_names"].astype(str)
    task_index = layer["task_index"].astype(int)
    episodes = layer["episode"].astype(int)
    lengths = layer["length"].astype(int)
    count = len(episodes)

    jobs = []
    for position, task in enumerate(task_names):
        rows = np.flatnonzero(task_index == position)
        jobs.append((args.cohort, str(task), episodes[rows], lengths[rows]))

    mobility = np.full((count, MAX_QUERY, len(LAYER_NAMES)), np.nan, dtype=np.float32)
    acceleration = np.full((count, MAX_QUERY), np.nan, dtype=np.float32)
    periodicity = np.full((count, MAX_QUERY), np.nan, dtype=np.float32)

    started = time.time()
    context = mp.get_context("spawn")
    with context.Pool(processes=args.workers) as pool:
        for done, (task, block, seconds) in enumerate(
            pool.imap_unordered(_worker, jobs), start=1
        ):
            position = int(np.flatnonzero(task_names == task)[0])
            rows = np.flatnonzero(task_index == position)
            mobility[rows], acceleration[rows], periodicity[rows] = block
            print(
                f"[{args.cohort} {done}/{len(jobs)}] {task} ({seconds:.1f}s)", flush=True
            )
    elapsed = time.time() - started

    valid = layer["valid"].astype(bool)
    for name, values in (("acceleration", acceleration), ("periodicity", periodicity)):
        if np.isfinite(values[~valid]).any():
            raise ValueError(f"{name} is finite outside the valid mask")
    if np.isfinite(mobility[~valid]).any():
        raise ValueError("mobility is finite outside the valid mask")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.cohort}_route_features.npz"
    np.savez_compressed(
        path,
        schema=np.asarray("himoe.legacy16x32.raw_route_features.v1"),
        cohort=np.asarray(args.cohort),
        run_id=np.asarray(spec.run_id),
        cache_root=np.asarray(str(spec.cache_root)),
        task_names=layer["task_names"],
        task_index=layer["task_index"],
        episode=layer["episode"],
        length=layer["length"],
        valid=layer["valid"],
        layer_names=np.asarray(LAYER_NAMES),
        mobility=mobility,
        route_acceleration=acceleration,
        lag_periodicity=periodicity,
    )

    audit = {
        "schema": "himoe.legacy16x32.extraction_audit.v1",
        "cohort": args.cohort,
        "extracted_at_utc": datetime.now(UTC).isoformat(),
        "cache_root": str(spec.cache_root),
        "run_id": spec.run_id,
        "tasks": len(jobs),
        "episodes": count,
        "queries": int(valid.sum()),
        "wall_seconds": elapsed,
        "workers": args.workers,
        "gpu_used": False,
        "outcome_files_opened": False,
        "layer_cache": str(spec.layer_cache),
        "layer_cache_sha256": sha256(spec.layer_cache),
        "output_sha256": sha256(path),
    }

    published_mobility_error = _finite_max_abs(mobility, layer["mobility"])
    audit["max_abs_vs_published_mobility"] = published_mobility_error
    reference = PUBLISHED_ROUTE_FEATURES[args.cohort]
    if reference is not None:
        cache = load_npz(reference)
        names = cache["feature_names"].astype(str).tolist()
        for field, feature_name in (
            ("route_acceleration", "route_acceleration"),
            ("lag_periodicity", "lag_periodicity"),
        ):
            published = np.asarray(
                cache["features"][:, :, names.index(feature_name)], dtype=np.float32
            )
            local = acceleration if field == "route_acceleration" else periodicity
            audit[f"max_abs_vs_published_{field}"] = _finite_max_abs(local, published)
            audit[f"bit_exact_vs_published_{field}"] = _bit_exact(local, published)
            audit[f"nan_pattern_matches_published_{field}"] = bool(
                np.array_equal(np.isnan(local), np.isnan(published))
            )
        audit["published_route_feature_cache"] = str(reference)
        audit["published_route_feature_cache_sha256"] = sha256(reference)

    (args.output / f"{args.cohort}_extraction_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


def _finite_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def _bit_exact(left: np.ndarray, right: np.ndarray) -> bool:
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return False
    finite = np.isfinite(left) & np.isfinite(right)
    return bool(np.array_equal(left[finite], right[finite]))


if __name__ == "__main__":
    main()
