#!/usr/bin/env python3
"""Inventory every result file with its size and sha256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geometry as G


DEFAULT_OUTPUT = G.BUNDLE / "results/manifest.json"
PIPELINE = (
    ("extract_conditional_kernels.py", "cache/kernels", "reads routes.zarr, writes the PSD conditional kernels"),
    ("compute_features.py", "cache/features", "layer templates and twelve per-query geometric features"),
    ("compute_dynamic_features.py", "cache/features", "causal shape-change features"),
    ("characterise_geometry.py", "results/geometry", "shape, dimension, stability, front vs back"),
    ("detect_deformation.py", "results/detection", "alarm sweep, development selection, one external replay"),
    ("stratified_effect.py", "results/effect", "survival-conditioned AUC"),
    ("profile_shape_drift.py", "results/effect", "fixed-layer profile and redundancy checks"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    args = parse_args()
    results = G.BUNDLE / "results"
    files = [
        {
            "path": str(path.relative_to(G.BUNDLE)),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(results.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    cache = [
        {"path": str(path.relative_to(G.BUNDLE)), "bytes": path.stat().st_size}
        for path in sorted((G.BUNDLE / "cache").rglob("*"))
        if path.is_file()
    ]
    args.output.write_text(
        json.dumps(
            {
                "schema": "himoe.token_geometry.manifest.v1",
                "bundle": "moe-token-geometry-0906",
                "question": "is the MoE action-token geometry stable, what shape is it, does it deform before failure",
                "inputs": {
                    "routing": "VLA_MUI_HUB/cache_new/HiMoE-VLA/<task>/<run_id>/server/routes.zarr:hb_router_probs",
                    "run_ids": ["right-50x8-20260903", "right-50x8b-20260903"],
                    "episode_index": "moe-v4-0904/results/layerwise_mobility/*.npz",
                    "labels": "double-selete/trainfree/results/timeout_extension_plus10/*_clean_labels.csv",
                },
                "coverage": {
                    "subsampling": "none; every valid query of every cohort is processed",
                    "development_main_queries": 221781,
                    "development_extra_queries": 31941,
                    "external_8b_queries": 248255,
                    "denoising_step": "final step only, as the construction specifies",
                },
                "pipeline": [
                    {"script": f"experiments/{script}", "writes": target, "role": role}
                    for script, target, role in PIPELINE
                ],
                "cache_is_regenerable": True,
                "cache_note": "cache/ holds the bulk float32 intermediates; rebuild with the first three scripts",
                "results": files,
                "cache": cache,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(files)} result files, {sum(f['bytes'] for f in files) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
