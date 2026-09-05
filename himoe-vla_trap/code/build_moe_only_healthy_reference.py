#!/usr/bin/env python3
"""Build full successful-route sequences without physical event alignment."""

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

import numpy as np
import zarr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
DEFAULT_EVENT_CONFIG = PACKAGE_ROOT / "configs/failed_grasp_moe_dynamics.json"
DEFAULT_OUTPUT = (
    PACKAGE_ROOT / "results/moe_only_online_alarm/full_healthy_route_sequences.npz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-config", type=Path, default=DEFAULT_EVENT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.event_config.read_text(encoding="utf-8"))
    run_root = (WORKSPACE_ROOT / config["run_root"]).resolve()
    snapshot_dir = (
        run_root
        / "formal"
        / ("worker%d" % int(config["worker"]))
        / ("snapshot_%03d" % int(config["snapshot"]))
    )

    records: list[dict[str, Any]] = []
    for metadata_path in sorted(snapshot_dir.glob("candidate_*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if bool(metadata["success"]):
            records.append(metadata)
    if len(records) < 3:
        raise RuntimeError("at least three successful healthy trajectories are required")

    store = zarr.open_group(str(run_root / "formal/server/routes.zarr"), mode="r")
    episode = np.asarray(store["episode_id"][:], dtype=np.int64)
    control_step = np.asarray(store["control_step"][:], dtype=np.int64)
    sequences = []
    lengths = []
    for record in records:
        episode_id = int(record["episode_id"])
        rows = np.flatnonzero(episode == episode_id)
        rows = rows[np.argsort(control_step[rows], kind="stable")]
        expected = int(record["inference_calls"])
        if len(rows) != expected:
            raise RuntimeError(
                "episode %d has %d route rows, expected %d"
                % (episode_id, len(rows), expected)
            )
        route = np.asarray(store["hb_router_probs"].oindex[rows], dtype=np.float32)
        route /= np.maximum(route.sum(axis=-1, keepdims=True), 1e-12)
        sequences.append(route.astype(np.float16))
        lengths.append(len(route))

    max_queries = max(lengths)
    padded = np.zeros(
        (len(sequences), max_queries, 8, 10, 11, 32), dtype=np.float16
    )
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        schema=np.asarray("himoe.moe_only_healthy_sequences.v1"),
        training=np.asarray(False),
        failure_labels_used=np.asarray(False),
        physical_alignment_used=np.asarray(False),
        success_labels_used_to_define_healthy_reference=np.asarray(True),
        routes=padded,
        lengths=np.asarray(lengths, dtype=np.int16),
        candidates=np.asarray([int(row["candidate"]) for row in records], dtype=np.int16),
        episode_ids=np.asarray(
            [int(row["episode_id"]) for row in records], dtype=np.int64
        ),
    )
    os.replace(temporary, args.output)
    summary = {
        "schema": "himoe.moe_only_healthy_sequences.summary.v1",
        "training": False,
        "failure_labels_used": False,
        "physical_alignment_used": False,
        "success_labels_used_to_define_healthy_reference": True,
        "healthy_trajectories": len(records),
        "candidates": [int(row["candidate"]) for row in records],
        "lengths": lengths,
        "route_archive": str(args.output.resolve()),
        "route_archive_sha256": sha256_file(args.output),
        "route_archive_shape": list(padded.shape),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
