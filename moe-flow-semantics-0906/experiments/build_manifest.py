#!/usr/bin/env python3
"""Record what was computed, from what, with which script, and at what size."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
RESULTS = BUNDLE / "results"

INPUTS = (
    "VLA_MUI_HUB/cache_new/HiMoE-VLA/<suite>/<task>/right-50x8-20260903/server/routes.zarr",
    "VLA_MUI_HUB/cache_new/HiMoE-VLA/<suite>/<task>/right-50x8b-20260903/server/routes.zarr",
    "moe-v4-0904/results/layerwise_mobility/main_reference.npz",
    "moe-v4-0904/results/layerwise_mobility/extra_reference.npz",
    "moe-v4-0904/results/layerwise_mobility/external_8b.npz",
    "double-selete/trainfree/results/timeout_extension_plus10/development_main_clean_labels.csv",
    "double-selete/trainfree/results/timeout_extension_plus10/external_8b_clean_labels.csv",
)
HASHED_INPUTS = INPUTS[2:]

PIPELINE = (
    {
        "script": "experiments/extract_flow_steps.py",
        "reads": ["hb_router_probs", "episode_id", "the three v4 episode caches"],
        "writes": "results/step_profiles/",
        "outcomes_loaded": False,
        "note": "CPU, 24 fork workers, 81 s wall for all three cohorts",
    },
    {
        "script": "experiments/analyze_step_structure.py",
        "reads": ["results/step_profiles/"],
        "writes": "results/step_structure/",
        "outcomes_loaded": False,
    },
    {
        "script": "experiments/analyze_step_decomposition.py",
        "reads": ["results/step_structure/", "results/step_profiles/"],
        "writes": "results/step_structure/{mobility_vs_noise_floor,step_increments,mobility_episode_icc}.csv",
        "outcomes_loaded": False,
    },
    {
        "script": "experiments/detect_step_alarm.py",
        "reads": [
            "results/step_profiles/",
            "development_main_clean_labels.csv (selection)",
            "external_8b_clean_labels.csv (single scoring pass)",
        ],
        "writes": "results/step_alarm/",
        "outcomes_loaded": True,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(BUNDLE)),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        record["shape"] = list(array.shape)
        record["dtype"] = str(array.dtype)
        record["versioned_in_git"] = False
    else:
        record["sha256"] = sha256(path)
        record["versioned_in_git"] = True
    return record


def main() -> None:
    outputs = sorted(
        path for path in RESULTS.rglob("*") if path.is_file() and path.name != ".gitignore"
    )
    manifest = {
        "schema": "himoe.flow_semantics.manifest.v1",
        "bundle": "moe-flow-semantics-0906",
        "question": "do the ten flow-matching denoising steps differ functionally, the way the eight HB layers do",
        "everything_derives_from": "hb_router_probs, plus the frozen v4 row alignment",
        "task_identity_used_inside_any_quantity": False,
        "episode_length_or_outcome_used_inside_any_quantity": False,
        "causality": "a score at query q reads queries 0..q only; inside a query all ten steps are available because they all run before the action is emitted",
        "inputs": list(INPUTS),
        "input_hashes": {
            name: sha256(PROJECT / name) for name in HASHED_INPUTS if (PROJECT / name).is_file()
        },
        "pipeline": list(PIPELINE),
        "dense_profiles_not_versioned": {
            "reason": "regenerable per-query profiles, 4.7 GB; matches the repository convention for dense profiles",
            "regenerate_with": "python experiments/extract_flow_steps.py --workers 24",
        },
        "python": subprocess.run(
            ["python", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "numpy": np.__version__,
        "outputs": [describe(path) for path in outputs],
        "total_output_bytes": sum(path.stat().st_size for path in outputs),
    }
    (RESULTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{len(outputs)} output files, {manifest['total_output_bytes'] / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
