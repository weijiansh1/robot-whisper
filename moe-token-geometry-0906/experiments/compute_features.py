#!/usr/bin/env python3
"""Turn the cached conditional kernels into per-query geometric features.

The layer templates are the pooled mean conditional kernel over every valid
query of the two development cohorts, one per HB layer.  They use no outcome
label, no task identity and no external data, so they are legitimate fixed
constants for a global detector.  Per-task templates are also stored, but only
for the stability analysis and for the explicitly task-side variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import geometry as G


DEFAULT_OUTPUT = G.FEATURE_ROOT
COHORTS = ("development_main", "development_extra", "external_8b")
REFERENCE = ("development_main", "development_extra")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def pooled_mean_kernel(cohorts: tuple[str, ...]) -> np.ndarray:
    total = np.zeros((8, 55), dtype=np.float64)
    count = 0
    for cohort in cohorts:
        packed = G.load_packed(cohort)
        for start in range(0, len(packed), 40000):
            stop = min(start + 40000, len(packed))
            total += np.asarray(packed[start:stop], dtype=np.float64).sum(axis=0)
        count += len(packed)
    return G.symmetrise((total / count).astype(np.float32))


def query_task_key(index: dict[str, np.ndarray]) -> np.ndarray:
    valid = index["valid"].astype(bool)
    task_index = index["task_index"].astype(int)
    return np.repeat(task_index, valid.sum(axis=1))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pooled = pooled_mean_kernel(REFERENCE)
    templates = G.layer_templates(pooled)
    np.savez_compressed(
        args.output / "templates.npz",
        schema=np.asarray("himoe.token_geometry.templates.v1"),
        built_from=np.asarray(",".join(REFERENCE)),
        layer_names=np.asarray(G.LAYER_NAMES),
        pooled_mean_kernel=pooled,
        pooled_centred=G.double_centre(pooled),
        template_coordinates=templates,
        template_distance=G.distances(pooled),
    )

    records = []
    for cohort in COHORTS:
        index = G.load_index(cohort)
        features = G.features_for_cohort(cohort, templates)
        if not np.isfinite(features).all():
            raise ValueError(f"{cohort}: non-finite feature")
        np.save(args.output / f"{cohort}_features.npy", features)

        keys = query_task_key(index)
        task_kernel = G.group_mean_kernel(cohort, keys, len(index["task_names"]))
        np.savez_compressed(
            args.output / f"{cohort}_task_kernels.npz",
            schema=np.asarray("himoe.token_geometry.task_kernels.v1"),
            task_names=index["task_names"],
            layer_names=np.asarray(G.LAYER_NAMES),
            task_mean_kernel=task_kernel,
        )
        records.append(
            {
                "cohort": cohort,
                "valid_queries": int(len(features)),
                "features": list(G.FEATURE_NAMES),
                "tasks": int(len(index["task_names"])),
            }
        )
        print(f"{cohort}: {len(features):,} queries", flush=True)

    (args.output / "feature_summary.json").write_text(
        json.dumps(
            {
                "schema": "himoe.token_geometry.features.v1",
                "template_source": list(REFERENCE),
                "template_uses_labels": False,
                "template_uses_task_identity": False,
                "feature_names": list(G.FEATURE_NAMES),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
