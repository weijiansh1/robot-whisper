#!/usr/bin/env python3
"""Manifest: hashes of every input read and every output written."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import common as C

OUT = C.BUNDLE / "results"

INPUTS = [
    C.PHYSICAL,
    C.LABEL_PATHS["development_main"],
    C.LABEL_PATHS["external_8b"],
    C.EXTERNAL_ALARMS,
    C.EXTERNAL_DETECTORS,
    C.V7_ALARMS,
    C.GRAPH_ROOT / "development_main.npz",
    C.GRAPH_ROOT / "development_extra.npz",
    C.GRAPH_ROOT / "external_8b.npz",
    C.MOBILITY_PATHS["development_main"],
    C.MOBILITY_PATHS["development_extra"],
    C.MOBILITY_PATHS["external_8b"],
]


def main() -> None:
    manifest = {
        "schema": "himoe.failure_modes_0906.manifest.v1",
        "written_at_utc": datetime.now(UTC).isoformat(),
        "question": "do MoE routing failure detectors specialise by physical failure mode",
        "constraints": {
            "detector_inputs": ["hb_router_probs-derived routing caches only"],
            "physical_labels_used_for": "analysis and validation only, never a runtime decision",
            "query_causal": True,
            "gradient_updates": False,
            "learned_weights": False,
            "selection_cohort": "development_main (+ development_extra as reference corpus)",
            "external_outcomes_used_for_selection": False,
            "threshold_modes_reported_separately": ["global", "per_task"],
        },
        "inputs_sha256": {
            str(p.relative_to(C.PROJECT)): C.sha256(p) for p in INPUTS if p.exists()
        },
        "experiments_sha256": {
            p.name: C.sha256(p) for p in sorted((C.BUNDLE / "experiments").glob("*.py"))
        },
        "outputs_sha256": {
            p.name: C.sha256(p)
            for p in sorted(OUT.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(manifest['outputs_sha256'])} outputs, "
          f"{len(manifest['inputs_sha256'])} inputs hashed")


if __name__ == "__main__":
    main()
