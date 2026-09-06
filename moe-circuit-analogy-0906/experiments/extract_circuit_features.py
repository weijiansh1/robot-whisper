#!/usr/bin/env python3
"""Extract electrical-network quantities per (episode, query, HB layer).

CPU only, deterministic, one worker process per task shard.  Reads only
`hb_router_probs` at the final flow step; no outcome, length, wall-clock or
task-identity information enters any feature.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import zarr

import circuit_lib as cl


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
GRAPHS = PROJECT / "moe-hb-front-back-0905/results/layer_graphs"
DEFAULT_OUTPUT = BUNDLE / "results/circuit_features"

COHORTS = ("development_main", "development_extra", "external_8b")
BATCH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def task_job(payload: tuple[str, str, str]) -> tuple[int, np.ndarray, np.ndarray, dict[str, float]]:
    position, task, run_id = payload
    route = ROUTE_ROOT / task / run_id / "server/routes.zarr"
    group = zarr.open_group(str(route), mode="r")
    source = group["hb_router_probs"]
    if tuple(source.shape[1:]) != (8, 10, 11, 32):
        raise ValueError(f"unexpected shape at {route}: {source.shape}")
    count = int(source.shape[0])
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    features = np.empty((count, 8, cl.N_FEATURES), dtype=np.float32)
    audit = {"cond_min_eig": np.inf, "neg_entries": 0.0, "entries": 0.0,
             "neg_abs_mass": 0.0, "abs_mass": 0.0, "foster_max": 0.0,
             "dd_rows_dominant": 0.0, "dd_rows": 0.0,
             "cells": 0.0, "cells_with_negative": 0.0, "cells_disconnected": 0.0,
             "foster_residual_connected_max": 0.0}
    for start in range(0, count, BATCH):
        stop = min(start + BATCH, count)
        raw = np.asarray(source[start:stop, :, -1])
        cond = cl.conditional_from_raw(raw)
        block = cl.circuit_features(cond)
        features[start:stop] = block.astype(np.float32)
        off = cond[..., ~np.eye(10, dtype=bool)]
        audit["cond_min_eig"] = min(audit["cond_min_eig"],
                                    float(np.linalg.eigvalsh(cond).min()))
        audit["neg_entries"] += float((off < 0).sum())
        audit["entries"] += float(off.size)
        audit["neg_abs_mass"] += float(np.abs(off[off < 0]).sum())
        audit["abs_mass"] += float(np.abs(off).sum())
        foster = block[..., cl.FEATURE_NAMES.index("foster_residual")]
        comps = block[..., cl.FEATURE_NAMES.index("n_near_zero")]
        connected = comps <= 1.0
        audit["foster_max"] = max(audit["foster_max"], float(foster.max()))
        if connected.any():
            audit["foster_residual_connected_max"] = max(
                audit["foster_residual_connected_max"], float(foster[connected].max()))
        audit["cells"] += float(comps.size)
        audit["cells_disconnected"] += float((~connected).sum())
        audit["cells_with_negative"] += float(
            (block[..., cl.FEATURE_NAMES.index("neg_count")] > 0).sum())
        deg = np.clip(off, 0, None).reshape(cond.shape[:-1] + (9,)).sum(-1)
        dia = np.diagonal(cond, axis1=-2, axis2=-1)
        audit["dd_rows_dominant"] += float((dia >= deg).sum())
        audit["dd_rows"] += float(dia.size)
    return int(position), episode_id, features, audit


def build(cohort: str, output: Path, workers: int) -> dict[str, Any]:
    graph = load_npz(GRAPHS / f"{cohort}.npz")
    task_names = graph["task_names"].astype(str)
    task_index = graph["task_index"].astype(int)
    episodes = graph["episode"].astype(int)
    lengths = graph["length"].astype(int)
    valid = graph["valid"].astype(bool)
    run_id = str(graph["run_id"])
    n_episodes, max_query = valid.shape

    row_of = np.full((n_episodes, max_query), -1, dtype=np.int64)
    flat_row, flat_query = np.nonzero(valid)
    order = np.lexsort((flat_query, flat_row))
    flat_row, flat_query = flat_row[order], flat_query[order]
    row_of[flat_row, flat_query] = np.arange(len(flat_row))
    features = np.full((len(flat_row), 8, cl.N_FEATURES), np.nan, dtype=np.float32)

    jobs = [(str(i), str(name), run_id) for i, name in enumerate(task_names)]
    audits: list[dict[str, float]] = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for position, episode_id, block, audit in pool.imap_unordered(task_job, jobs, chunksize=1):
            audits.append(audit)
            global_rows = np.flatnonzero(task_index == position)
            lookup = {int(e): int(r) for r, e in zip(global_rows, episodes[global_rows])}
            for episode in np.unique(episode_id):
                pos = np.flatnonzero(episode_id == episode)
                if len(pos) > 1 and not np.all(np.diff(pos) == 1):
                    raise ValueError(f"non-contiguous episode {episode} in task {position}")
                row = lookup[int(episode)]
                if len(pos) != int(lengths[row]):
                    raise ValueError(f"length mismatch task {position} episode {episode}")
                dest = row_of[row, : len(pos)]
                if (dest < 0).any():
                    raise ValueError("valid mask does not cover episode length")
                features[dest] = block[pos]
            print(f"[{cohort}] task {position + 1}/{len(task_names)} rows={len(episode_id)}", flush=True)

    if not np.isfinite(features).all():
        bad = np.argwhere(~np.isfinite(features))
        raise ValueError(f"non-finite feature at {bad[:5].tolist()}")

    merged = {
        "cond_min_eigenvalue": float(min(a["cond_min_eig"] for a in audits)),
        "negative_offdiag_fraction": sum(a["neg_entries"] for a in audits) / sum(a["entries"] for a in audits),
        "negative_offdiag_mass_fraction": sum(a["neg_abs_mass"] for a in audits) / sum(a["abs_mass"] for a in audits),
        "foster_residual_max": float(max(a["foster_max"] for a in audits)),
        "foster_residual_max_on_connected": float(max(a["foster_residual_connected_max"] for a in audits)),
        "diagonally_dominant_row_fraction": sum(a["dd_rows_dominant"] for a in audits) / sum(a["dd_rows"] for a in audits),
        "layer_query_cells": int(sum(a["cells"] for a in audits)),
        "cells_with_any_negative_conductance": int(sum(a["cells_with_negative"] for a in audits)),
        "cells_disconnected_after_clipping": int(sum(a["cells_disconnected"] for a in audits)),
    }

    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{cohort}.npz"
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray("himoe.circuit_analogy.features.v1"),
            cohort=np.asarray(cohort),
            run_id=np.asarray(run_id),
            feature_names=np.asarray(cl.FEATURE_NAMES),
            layer_names=np.asarray(cl.LAYER_NAMES),
            task_names=task_names,
            task_index=graph["task_index"],
            episode=graph["episode"],
            init_state_id=graph["init_state_id"],
            length=graph["length"],
            valid=valid,
            flat_row=flat_row.astype(np.int32),
            flat_query=flat_query.astype(np.int16),
            features=features,
        )
    os.replace(temporary, destination)
    return {
        "cohort": cohort,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "episodes": int(n_episodes),
        "tasks": int(len(task_names)),
        "queries": int(len(flat_row)),
        "run_id": run_id,
        **merged,
    }


def main() -> None:
    args = parse_args()
    records = [build(cohort, args.output, args.workers) for cohort in COHORTS]
    summary = {
        "schema": "himoe.circuit_analogy.extract.v1",
        "source": "hb_router_probs final flow step only",
        "outcomes_loaded": False,
        "features": list(cl.FEATURE_NAMES),
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "extract_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
