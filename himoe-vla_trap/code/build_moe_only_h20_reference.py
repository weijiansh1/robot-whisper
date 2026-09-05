#!/usr/bin/env python3
"""Build a physics-free H20 healthy bank from completed successful episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    PACKAGE_ROOT / "results/moe_only_online_alarm/gpu4_fresh_random_seed20260905"
)
DEFAULT_OUTPUT = (
    PACKAGE_ROOT
    / "results/moe_only_online_alarm/h20_healthy_route_sequences_seed20260905.npz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["training"] is not False:
        raise RuntimeError("source run is not a complete train-free run")

    selected = [episode for episode in manifest["episodes"] if episode["success"]]
    if len(selected) < 3:
        raise RuntimeError("at least three successful calibration episodes are required")
    sequences = []
    for episode in selected:
        with np.load(run / episode["trajectory_npz"], allow_pickle=False) as archive:
            route = np.asarray(archive["hb_router_probs"], dtype=np.float32)
        route /= np.maximum(route.sum(axis=-1, keepdims=True), 1e-12)
        sequences.append(route.astype(np.float16))
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int16)
    padded = np.zeros(
        (len(sequences), int(lengths.max()), 8, 10, 11, 32), dtype=np.float16
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
        lengths=lengths,
        candidates=np.asarray(
            [int(episode["episode_id"]) for episode in selected], dtype=np.int64
        ),
        episode_ids=np.asarray(
            [int(episode["episode_id"]) for episode in selected], dtype=np.int64
        ),
        init_state_ids=np.asarray(
            [int(episode["init_state_id"]) for episode in selected], dtype=np.int16
        ),
    )
    os.replace(temporary, args.output)
    summary = {
        "schema": "himoe.moe_only_h20_healthy_sequences.summary.v1",
        "training": False,
        "failure_labels_used": False,
        "physical_alignment_used": False,
        "selection_rule": "completed successful episodes only",
        "source_run": str(run),
        "source_seed": 20260905,
        "healthy_trajectories": len(selected),
        "episode_ids": [int(episode["episode_id"]) for episode in selected],
        "init_state_ids": [int(episode["init_state_id"]) for episode in selected],
        "lengths": lengths.tolist(),
        "route_archive": str(args.output.resolve()),
        "route_archive_sha256": sha256_file(args.output),
        "route_archive_shape": list(padded.shape),
        "role": "calibration_only_not_test",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
