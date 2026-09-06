#!/usr/bin/env python3
"""Hash every artifact in this bundle and record what it depends on."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path



from raw_route_features import BUNDLE, COHORTS, WORKSPACE


V7 = WORKSPACE / "moe-v7-0905"
RESULTS = BUNDLE / "results"
EXPERIMENTS = BUNDLE / "experiments"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    anchor = json.loads((RESULTS / "anchor/anchor_verification.json").read_text())
    evaluation = json.loads((RESULTS / "legacy16x32/evaluation_summary.json").read_text())

    inputs = {
        "frozen_profile": V7 / "results/intrinsic_guard_v7/global_profile.npz",
        "sealed_first_alarms": V7 / "results/intrinsic_guard_v7/sealed_first_alarms.npz",
        "monitor_module": V7 / "method/intrinsic_guard_monitor.py",
        "sealed_evaluator": V7 / "experiments/evaluate_intrinsic_guard_v7.py",
        "legacy16x32_layer_mobility": COHORTS["legacy_16x32"].layer_cache,
        "development_main_layer_mobility": COHORTS["development_main"].layer_cache,
        "external_8b_layer_mobility": COHORTS["external_8b"].layer_cache,
        "development_main_labels": WORKSPACE
        / "double-selete/trainfree/results/timeout_extension_plus10/development_main_clean_labels.csv",
        "external_8b_labels": WORKSPACE
        / "double-selete/trainfree/results/timeout_extension_plus10/external_8b_clean_labels.csv",
    }

    raw_runs = sorted(
        (COHORTS["legacy_16x32"].cache_root).glob("libero_*/*/right-16x32/server/routes.zarr")
    )
    manifest = {
        "schema": "himoe.legacy16x32.manifest.v1",
        "bundle": BUNDLE.name,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "measure the frozen v7 intrinsic guard on the legacy right-16x32 cohort, "
            "replacing the previous whole-corpus extrapolation with a measurement"
        ),
        "recalibrated": False,
        "operating_point_reselected": False,
        "gpu_used": False,
        "anchor_green": anchor["anchor_green"],
        "anchor_counts": {
            cohort: report["variants"]["published_mobility_raw_route_features"]["counts"]
            for cohort, report in anchor["cohorts"].items()
        },
        "legacy16x32_guard": evaluation["guard"],
        "corpus_total": evaluation["corpus_total"],
        "raw_route_stores": [str(path.relative_to(WORKSPACE)) for path in raw_runs],
        "inputs": {
            name: {"path": str(path.relative_to(WORKSPACE)), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "code": {
            str(path.relative_to(BUNDLE)): sha256(path)
            for path in sorted(
                list(EXPERIMENTS.rglob("*.py")) + list((BUNDLE / "tests").rglob("*.py"))
            )
            if "__pycache__" not in path.parts
        },
        "outputs": {
            str(path.relative_to(BUNDLE)): sha256(path)
            for path in sorted(RESULTS.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (RESULTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("anchor_green", "legacy16x32_guard")}, indent=2))


if __name__ == "__main__":
    main()
