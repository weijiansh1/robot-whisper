#!/usr/bin/env python3
"""Manifest: inputs with hashes, outputs with hashes, and the survival-prior
scaffolding (horizon caps and the chunk where each suite's prior crosses 0.25)
so the early-band definition is checkable without rerunning anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import twotier_lib as lib


LOW_PRIOR = 0.25
INPUTS = {
    "external_first_alarms_published": lib.EXTERNAL_ALARMS,
    "external_detectors": lib.EXTERNAL_DETECTORS,
    "development_candidates": lib.DEVELOPMENT_CANDIDATES,
    "v7_sealed_first_alarms": lib.V7_ALARMS,
    "physical_failure_labels": lib.PHYSICAL_LABELS,
    "development_labels": lib.PROJECT
    / "double-selete/trainfree/results/timeout_extension_plus10/development_main_clean_labels.csv",
    "external_labels": lib.PROJECT
    / "double-selete/trainfree/results/timeout_extension_plus10/external_8b_clean_labels.csv",
    "layer_graph_development_main": lib.PROFILE_ROOT / "development_main.npz",
    "layer_graph_development_extra": lib.PROFILE_ROOT / "development_extra.npz",
    "layer_graph_external_8b": lib.PROFILE_ROOT / "external_8b.npz",
    "mobility_development_main": lib.PROJECT
    / "moe-v4-0904/results/layerwise_mobility/main_reference.npz",
    "mobility_development_extra": lib.PROJECT
    / "moe-v4-0904/results/layerwise_mobility/extra_reference.npz",
    "mobility_external_8b": lib.PROJECT
    / "moe-v4-0904/results/layerwise_mobility/external_8b.npz",
    "survey_reference_frames_source": lib.PROJECT
    / "moe-hb-front-back-0905/experiments/survey_reference_frames.py",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=lib.RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scaffolding: dict[str, Any] = {}
    for name in ("development_main", "external_8b"):
        cohort = lib.load_cohort(name)
        block: dict[str, Any] = {}
        for suite in sorted(set(cohort["suite"])):
            curve = cohort["priors"][suite]
            horizon = int(cohort["length"][cohort["suite"] == suite].max())
            crossing = next(
                (q for q in sorted(curve) if curve[q] >= LOW_PRIOR), None
            )
            block[suite] = {
                "horizon_cap": horizon,
                "prior_crosses_0.25_at_chunk": crossing,
                "prior_at_chunk_0": curve[0],
                "risk_episodes": int(cohort["risk"][cohort["suite"] == suite].sum()),
                "episodes": int((cohort["suite"] == suite).sum()),
            }
        scaffolding[name] = block

    outputs = sorted(
        p for p in args.output.iterdir() if p.is_file() and p.name != "manifest.json"
    )
    experiments = sorted(p for p in (lib.BUNDLE / "experiments").glob("*.py"))
    manifest = {
        "schema": "himoe.two_tier.manifest.v1",
        "bundle": "moe-two-tier-0906",
        "question": "does a two-tier WATCH/ACT monitor built on MoE routing reference "
        "frames work, and what does each tier cost",
        "moe_only": "every score derives from hb_router_probs; physical failure-mode "
        "labels are used for evaluation and, in the flagged WATCH variant (b) only, "
        "for rule selection on development. No physical signal enters a runtime decision.",
        "causal": "a k-of-n alarm fires at the k-th component alarm chunk, so a combined "
        "alarm at chunk q uses only component alarms at chunks <= q",
        "train_free": "no fitted weights; selection is a search over a finite predeclared "
        "grid on development outcomes",
        "low_prior_threshold": LOW_PRIOR,
        "survival_scaffolding": scaffolding,
        "run_order": [
            "build_alarms.py",
            "select_on_development.py",
            "evaluate_external.py",
            "analyse_two_tier.py",
            "make_manifest.py",
        ],
        "inputs": {
            name: {"path": str(path.relative_to(lib.PROJECT)), "sha256": digest(path)}
            for name, path in INPUTS.items()
        },
        "experiments": {
            p.name: {"sha256": digest(p)} for p in experiments
        },
        "outputs": {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in outputs
        },
        "numpy": np.__version__,
        "python": subprocess.run(
            ["python3", "--version"], capture_output=True, text=True
        ).stdout.strip(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(scaffolding["external_8b"], indent=2, sort_keys=True), flush=True)
    print(f"\nmanifest lists {len(outputs)} outputs and {len(experiments)} scripts", flush=True)


if __name__ == "__main__":
    main()
