"""Inventory every result file with its size, sha256 and provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
OUT = BUNDLE / "results"

PIPELINE = [
    ("step0_anchor.py",
     "Reproduces every published v8.3 anchor before anything new is built: "
     "per-cohort lead>=4, the whole lead profile, a bit-exact rebuild of v8.3 "
     "from flow_speed, and the reachability table.",
     ["anchor.json", "anchor_reachability.csv", "logs/step0_anchor.log"]),
    ("step1_diagnose.py",
     "Decomposes the lead>=16 shortfall into DETECTED-BUT-LATE and "
     "NEVER-DETECTED. 715 late, 110 never; in goal, object and spatial the "
     "never-detected count is zero.",
     ["shortfall.json", "shortfall_decomposition.csv",
      "goal_object_per_task.csv", "logs/step1_diagnose.log"]),
    ("step2_feasibility.py",
     "Survivor composition at each suite's lead>=16 deadline and the "
     "enrichment a head would need to close the gap.",
     ["feasibility.json", "survivor_table.csv", "logs/step2_feasibility.log"]),
    ("step3_ceiling.py",
     "Ceiling scan over the published mobility / flow_speed bank: within-task "
     "survivor-only AUC and the operational oracle conversion TP@FP, with the "
     "Gaussian prediction for comparison.",
     ["ceiling.json", "ceiling_scan_development.csv",
      "logs/step3_ceiling.log"]),
    ("extract_raw_quantities.py",
     "One pass over routes.zarr for seventeen quantities not previously "
     "computed: top-4 support set structure, rank structure, the ten action "
     "tokens jointly, the state token in its own right, and load "
     "concentration. [n, 52, 8] float32 per quantity, row-aligned with the "
     "published caches.",
     ["development_main_rawq.npz", "external_8b_rawq.npz",
      "legacy_main16x32_rawq.npz", "extract_raw.log"]),
    ("step4_ceiling_raw.py",
     "The ceiling scan again over 128 series, with a null floor computed per "
     "(suite, chunk) cell from a task-stratified label permutation at the "
     "full sweep width.",
     ["ceiling_raw.json", "ceiling_scan_raw_development.csv",
      "null_floor_by_cell.csv", "logs/step4_ceiling_raw.log"]),
    ("step5_rules.py",
     "2,520 decision rules on the shortlist: hard threshold, m-of-k excursion "
     "in time, CUSUM accumulation and opening-regime-stratified thresholds. "
     "Development only; selection rule declared in the module docstring.",
     ["rule_sweep.json", "rule_sweep_development.csv",
      "logs/step5_rules.log"]),
    ("step6_capfree_cost.py",
     "Separates 'the protocol blocks it' from 'the signal is not there': the "
     "same head with global, opening-stratified, per-suite and per-task "
     "thresholds, against an oracle handed the suite, the chunk and a "
     "labelled threshold.",
     ["capfree_cost.json", "capfree_cost_development.csv",
      "logs/step6_capfree_cost.log"]),
    ("step7_opening_regime.py",
     "How much of the suite the episode's own opening regime carries (purity "
     "0.947 at 49 cells, 0.995 at 2,002), and whether a 2-D opening x "
     "chunk-band threshold grid recovers the oracle gap.",
     ["opening_regime.json", "opening_regime_development.csv",
      "logs/step7_opening_regime.log"]),
    ("step8_multiple_looks.py",
     "The mechanism: false-alarm compounding over the chunk window a long "
     "lead forces, measured directly; plus the last compliant sweep over "
     "quantile x rule x stratification.",
     ["multiple_looks.json", "strat_sweep_development.csv",
      "logs/step8_multiple_looks.log"]),
    ("step9_freeze_and_seal.py",
     "Freezes the two pre-declared arms and scores external_8b and "
     "legacy_main16x32 ONCE. Per cohort, per suite, per task, "
     "leave-one-task-out.",
     ["sealed_summary.json", "sealed_alarms.npz", "sealed_per_cohort.csv",
      "sealed_per_suite.csv", "sealed_per_task_gain.csv",
      "logs/step9_freeze_and_seal.log"]),
    ("step10_controls.py",
     "The controls that decide the result: a fixed-chunk constant with no "
     "routing input, and a rate-matched random-episode null at the head's own "
     "alarm count and chunk distribution.",
     ["controls.json", "trivial_control.csv", "logs/step10_controls.log"]),
    ("step11_failure_modes.py",
     "Whether routing can see object_released_or_dropped_before_goal early. "
     "Detection by physical mode, and fixed-chunk separability of drop-type "
     "risks against all other risks.",
     ["failure_modes.json", "failure_mode_detection.csv",
      "drop_separability.csv", "logs/step11_failure_modes.log"]),
]

SUPPORT = {
    "common.py": "Cohort loading, scoring, union, confirmed_first, per-suite "
                 "and per-task reporting, frozen v8.3 constants and anchors.",
    "bank.py": "Named [n, 52] causal series from the cached arrays and from "
               "the newly extracted raw quantities, plus the fixed-window "
               "opening self-baseline and the after-end mask.",
}

HEADLINE = {
    "corpus": {"episodes": 32960, "risks": 1358, "successes": 31602},
    "v7": {"0": [1078, 172], "4": [870, 111], "8": [638, 82],
           "12": [564, 60], "16": [513, 29], "20": [343, 14]},
    "v8.3": {"0": [1167, 209], "4": [967, 134], "8": [711, 104],
             "12": [617, 66], "16": [533, 30], "20": [352, 14]},
    "v8.3+A+B": {"0": [1210, 258], "4": [1118, 171], "8": [778, 134],
                 "12": [647, 90], "16": [552, 41], "20": [357, 15]},
    "verdict": "lead>=4 moves a lot (+151 TP / +37 FP, and +139 TP over a "
               "rate-matched null); lead>=12 and lead>=16 move little "
               "(+30/+24 and +19/+11, margins of +18 and +9 over the null); "
               "lead>=20 does not move at all (the null wins by 12 TP). On "
               "legacy_main16x32, the only never-fitted cohort, the gain is "
               "+32 TP at lead>=4 for zero extra false alarms and exactly "
               "zero at lead>=12 and lead>=16.",
}


def digest(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def main() -> None:
    entries = []
    for script, what, files in PIPELINE:
        produced = {}
        for rel in files:
            p = OUT / rel
            produced[rel] = digest(p) if p.exists() else {"missing": True}
        entries.append({"script": f"experiments/{script}", "produces": what,
                        "files": produced})
    manifest = {
        "bundle": "moe-longlead-search-0906",
        "question": "raise long-lead detection on the short-horizon suites",
        "protocol": {
            "cap_free": "detector input is chunk index + routing only",
            "comparison": ">= and <=, never strict",
            "thresholds": "order statistics of the unlabeled development pool",
            "fit": "development_main only; external_8b and legacy_main16x32 "
                   "scored once, in step 9",
            "development_iterations_before_sealing": 9,
            "sealed_cohorts_seen_before_design_was_fixed": "only in step 0, "
                "and only to reproduce the published v8.3 anchors; no head "
                "built here was evaluated on them until step 9",
        },
        "support": SUPPORT,
        "pipeline": entries,
        "headline": HEADLINE,
        "tests": "tests/test_longlead.py -- 15 tests, all passing",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n = sum(1 for e in entries for f in e["files"].values()
            if "missing" not in f)
    miss = [f"{e['script']}:{k}" for e in entries
            for k, f in e["files"].items() if "missing" in f]
    print("manifest written: %d scripts, %d files" % (len(entries), n))
    if miss:
        print("MISSING: %s" % ", ".join(miss))


if __name__ == "__main__":
    main()
