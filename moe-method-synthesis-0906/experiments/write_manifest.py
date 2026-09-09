"""Manifest: what was read, what was written, and under what conventions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import synth_core as sc

INPUTS = [
    "moe-prior-correction-0906/experiments/recompute_task_matched_lift.py",
    "moe-v7-0905/experiments/evaluate_intrinsic_guard_v7.py",
    "moe-v7-0905/experiments/loso_folds.py",
    "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz",
    "moe-v7-0905/results/intrinsic_guard_v7/sealed_manifest.json",
    "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv",
    "moe-circuit-analogy-0906/results/detectors/external_first_alarms.npz",
    "moe-flow-semantics-0906/results/step_alarm/external_first_alarms.npz",
    "moe-state-channel-0906/results/detectors/external_first_alarms.npz",
    "moe-token-geometry-0906/results/detection/external_first_alarms.npz",
    "moe-two-tier-0906/results/external_first_alarms.npz",
    "moe-two-tier-0906/results/development_first_alarms.npz",
    "moe-failure-modes-0906/results/first_alarms_external.npz",
    "moe-failure-modes-0906/results/first_alarms_development.npz",
    "moe-combination-rules-0906/results/alarms/external_8b_alarms.npz",
    "moe-combination-rules-0906/results/alarms/development_main_alarms.npz",
    "moe-v4-0904/results/layerwise_mobility/external_8b.npz",
    "moe-v4-0904/results/layerwise_mobility/main_reference.npz",
]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    man = {
        "schema": "himoe.method_synthesis.manifest.v1",
        "bundle": "moe-method-synthesis-0906",
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "Accounting and structure analysis over already-computed alarm "
            "vectors.  No detector was built, no feature was re-derived and no "
            "threshold was re-fitted; every first-alarm vector is read exactly "
            "as its bundle published it."
        ),
        "conventions": {
            "risk": "original_failure - did not finish before the original horizon cap",
            "caps": sc.CAP,
            "in_window": "first alarm chunk < 0.65 x suite cap",
            "false_positive": "in-window alarm on a non-risk episode",
            "fpr": "FP / number of non-risk episodes",
            "lift_basis": ("per-task survival prior; the suite-matched basis is "
                           "also recorded in detector_census.csv but is known to "
                           "be anti-conservative and is not used for any claim"),
            "ties": ("no feature threshold is re-fitted here, so the strict-> "
                     "versus >= tie rule does not bind; every count threshold "
                     "used (vote k, budget B, family cut) is inclusive"),
            "length": ("negative control only.  Risk is defined as failing to "
                       "finish before the cap, so a length-at-cap alarm recalls "
                       "100% by construction; is_baseline=False in the census "
                       "and it is excluded from every ranking"),
            "selection": ("detector families, family representatives and greedy "
                          "orders are all chosen on development_main; external_8b "
                          "is scored once"),
            "in_sample_arms": ("anything tagged ORACLE_* or *_IN_SAMPLE selects on "
                               "external outcomes and is an upper bound only"),
            "nulls": ("every clustering, overlap and complementarity statistic "
                      "carries a rate-matched random arm with per-suite firing "
                      "rates preserved"),
        },
        "cohorts": {
            "external_8b": {"episodes": 15600, "risks": 564,
                            "run_id": "right-50x8b-20260903"},
            "development_main": {"episodes": 14800, "risks": 487,
                                 "run_id": "right-50x8-20260903"},
        },
        "inputs": {},
        "outputs": {},
        "experiments": [
            "synth_core.py - loading, alignment, physical-mode join, nulls",
            "ledger_core.py - development-only family representative selection",
            "census.py - per-detector scoring and duplicate detection",
            "families.py - clustering by alarm set, three similarities, null arm",
            "coverage_ledger.py - part 1, coverage ledger and saturation",
            "fp_ledger.py - part 2, false alarm ledger and vote decomposition",
            "blindspot.py - part 4, family x physical failure mode",
            "complementarity.py - part 5, combination frontiers and null",
            "irreducible_fp.py - stratified tests on multi-vote false alarms",
            "headroom.py - matched-external-FP ceiling versus honest arms",
            "summarise.py - assembles results/key_numbers.json",
        ],
        "not_touched": [
            "moe-unified-detector-0906 (concurrent work, never read or written)",
        ],
    }
    for rel in INPUTS:
        p = sc.ROOT / rel
        man["inputs"][rel] = {"exists": p.exists(),
                              "sha256": sha256(p) if p.exists() else None}
    for p in sorted(sc.RESULTS.iterdir()):
        if p.name == "manifest.json":
            continue
        man["outputs"][p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (sc.RESULTS / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"{len(man['inputs'])} inputs, {len(man['outputs'])} outputs")


if __name__ == "__main__":
    main()
