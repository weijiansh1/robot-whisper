#!/usr/bin/env python3
"""Write the run manifest and the development-to-external transfer table.

The manifest restates the predeclared objective, records every artifact and its
provenance, and marks explicitly which numbers were chosen before external was
opened and which were not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import combination_core as core


DEFAULT_OUTPUT = core.BUNDLE / "results"
DEV_TIMELY = 14800 - 487
EXT_TIMELY = 15600 - 564
DEV_RISKS = 487
EXT_RISKS = 564


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    args = parse_args()
    shortlist = pd.read_csv(args.output / "external_shortlist.csv")
    transfer = pd.DataFrame(
        {
            "selected_by": shortlist["selected_by"],
            "structure": shortlist["family"]
            + "/"
            + shortlist["pool"]
            + " level="
            + shortlist["level"].astype(str)
            + "->"
            + shortlist["level_loose"].astype(str)
            + " k="
            + shortlist["k"].astype(str)
            + " W="
            + shortlist["window"].map(lambda w: "inf" if w < 0 else str(w))
            + " f="
            + shortlist["frames"].astype(str),
            "dev_recall": shortlist["dev_tp"] / DEV_RISKS,
            "ext_recall": shortlist["tp"] / EXT_RISKS,
            "dev_fpr": shortlist["dev_fp"] / DEV_TIMELY,
            "ext_fpr": shortlist["fp"] / EXT_TIMELY,
        }
    )
    transfer["recall_transfer_ratio"] = transfer["ext_recall"] / transfer["dev_recall"]
    transfer["fpr_inflation"] = transfer["ext_fpr"] / transfer["dev_fpr"].replace(0, float("nan"))
    transfer.to_csv(args.output / "development_to_external_transfer.csv", index=False)

    artifacts = {
        path.name: {"sha256_16": digest(path), "bytes": path.stat().st_size}
        for path in sorted(args.output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema": "himoe.combination_rules.manifest.v1",
        "bundle": "moe-combination-rules-0906",
        "date": "2026-09-06",
        "question": "can an explicit train-free combination rule recover substantially more of the detector union's coverage than any single detector or plain AND, at a comparable false alarm rate",
        "predeclared_objective": {
            "document": "PREREGISTRATION.json",
            "written_before_external_was_opened": True,
            "objectives": core.OBJECTIVES,
            "constraints": {
                "development_low_prior_precision_at_least": core.MIN_LOW_PRIOR_PRECISION,
                "development_timely_fpr_caps": list(core.FPR_CAPS),
            },
            "rule_families": ["quorum", "cascade"],
            "grid": {
                "levels": list(core.LEVELS),
                "k": list(core.K_VALUES),
                "window": ["inf" if w < 0 else w for w in core.W_VALUES],
                "frames": list(core.F_VALUES),
                "pools": {name: list(members) for name, members in core.POOLS.items()},
            },
        },
        "train_free": True,
        "learned_parameters": 0,
        "moe_only": "every pool member is a hb_router_probs quantity; episode length, outcome and wall-clock never enter a rule",
        "causal": "a combined alarm at chunk q uses only component alarms at chunks <= q",
        "cohorts": {
            "development_main": {"episodes": 14800, "risks": DEV_RISKS},
            "external_8b": {"episodes": 15600, "risks": EXT_RISKS},
        },
        "provenance": {
            "detector_definitions": "moe-hb-front-back-0905/results/frame_survey/external_detectors.csv",
            "alarm_reconstruction_verified_against": "moe-hb-front-back-0905/results/frame_survey/external_first_alarms.npz",
            "labels": "double-selete/trainfree/results/timeout_extension_plus10/*_clean_labels.csv",
            "routing_caches": [
                "moe-hb-front-back-0905/results/layer_graphs/*.npz",
                "moe-v4-0904/results/layerwise_mobility/*.npz",
            ],
        },
        "pipeline": [
            "experiments/build_detector_alarms.py",
            "experiments/test_combination_core.py",
            "experiments/select_on_development.py",
            "experiments/evaluate_external.py",
            "experiments/analyze_results.py",
            "experiments/make_headline_table.py",
            "experiments/ablate_window.py",
            "experiments/write_manifest.py",
        ],
        "post_hoc_items": [
            "premise.best_possible_four_detector_union in analysis.json is chosen by external coverage and is labelled post-hoc there",
            "union_coverage_context.csv sweeps the plain-OR sensitivity curve on external; it is reported as context and no rule was selected from it",
            "the suite-level reading of why the global-mode gain exists was written after seeing the external suite split",
        ],
        "artifacts": artifacts,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.set_option("display.width", 220)
    print(transfer.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
