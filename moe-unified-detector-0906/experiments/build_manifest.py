#!/usr/bin/env python3
"""results/manifest.json: inputs, outputs, hashes, and what was fitted where."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import protocol as P


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


INPUTS = [
    "moe-v4-0904/results/layerwise_mobility/main_reference.npz",
    "moe-v4-0904/results/layerwise_mobility/external_8b.npz",
    "moe-v4-0904/results/cache16x32_v4/layerwise_mobility.npz",
    "moe-v4-0904/results/cache16x32_v4/episode_alarms.csv",
    "moe-hb-front-back-0905/results/layer_graphs/development_main.npz",
    "moe-hb-front-back-0905/results/layer_graphs/external_8b.npz",
    "moe-hb-front-back-0905/results/layer_graphs/legacy_main16x32.npz",
    "moe-unused-channels-0906/results/channels/development_main_quantities.npy",
    "moe-unused-channels-0906/results/channels/external_8b_quantities.npy",
    "moe-flow-semantics-0906/results/step_profiles/development_main_metrics.npy",
    "moe-flow-semantics-0906/results/step_profiles/external_8b_metrics.npy",
    "double-selete/trainfree/results/timeout_extension_plus10/"
    "development_main_clean_labels.csv",
    "double-selete/trainfree/results/timeout_extension_plus10/"
    "external_8b_clean_labels.csv",
    "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz",
    "moe-v7-legacy16x32-0906/results/legacy16x32/legacy16x32_first_alarms.npz",
]

CODE = [
    "protocol.py", "arms.py", "auc_table.py", "detector.py", "sweep.py",
    "run_build.py", "analyze.py", "controls.py", "anchors.py",
    "make_figure.py", "build_manifest.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    manifest = {
        "schema": "himoe.unified_detector.manifest.v1",
        "bundle": "moe-unified-detector-0906",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "python": subprocess.run(
            ["python", "-V"], capture_output=True, text=True
        ).stdout.strip(),
        "protocol": {
            "fit_cohort": P.FIT_COHORT,
            "sealed_cohorts": list(P.SEALED_COHORTS),
            "caps": P.CAPS,
            "in_window_end": P.IN_WINDOW_END,
            "in_window_end_strict_phase": P.IN_WINDOW_END_STRICT,
            "phase": "(q+1)/cap; cap is rollout configuration, not task identity",
            "normalisations": list(P.NORMALISATIONS),
            "self_baseline_chunks": P.SELF_BASELINE_CHUNKS,
            "max_gated_chunks_per_suite": P.MAX_GATED_CHUNKS,
            "seed": P.SEED,
            "task_identity_used": False,
            "future_chunks_used": False,
            "sealed_outcomes_used_for_selection": False,
            "development_iterations": 2,
            "development_iteration_note": (
                "iteration 1 = the arms named `unified`/`ablate_*`/`null_*`; "
                "iteration 2 added `unified_plus_prior` (channel score plus the "
                "development survival-prior log-odds offset) and the "
                "`survivor_lead` baseline family, after the first pass showed "
                "the information-free survival rule was the binding reference. "
                "No threshold, weight, gate or term of the first-iteration arms "
                "was changed."
            ),
        },
        "inputs": {},
        "code": {},
        "outputs": {},
    }
    for rel in INPUTS:
        path = P.PROJECT / rel
        manifest["inputs"][rel] = (
            {"sha256": sha256(path), "bytes": path.stat().st_size}
            if path.exists() else {"missing": True}
        )
    for rel in CODE:
        path = here / rel
        if path.exists():
            manifest["code"][rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    for path in sorted(P.RESULTS.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][str(path.relative_to(P.RESULTS))] = {
                "sha256": sha256(path), "bytes": path.stat().st_size
            }
    (P.RESULTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    print("wrote", P.RESULTS / "manifest.json")


if __name__ == "__main__":
    main()
