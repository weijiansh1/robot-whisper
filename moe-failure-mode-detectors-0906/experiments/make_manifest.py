#!/usr/bin/env python3
"""Write results/manifest.json: what was run, on what, producing what."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import arms as A
import modes_common as M

INPUTS = [
    "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv",
    "moe-flow-semantics-0906/results/step_profiles/development_main_metrics.npy",
    "moe-flow-semantics-0906/results/step_profiles/external_8b_metrics.npy",
    "moe-flow-semantics-0906/results/step_profiles/development_main_mobility.npy",
    "moe-flow-semantics-0906/results/step_profiles/external_8b_mobility.npy",
    "moe-flow-semantics-0906/results/step_profiles/development_main_state_mobility.npy",
    "moe-flow-semantics-0906/results/step_profiles/external_8b_state_mobility.npy",
    "moe-unused-channels-0906/results/channels/development_main_quantities.npy",
    "moe-unused-channels-0906/results/channels/external_8b_quantities.npy",
    "moe-hb-front-back-0905/results/layer_graphs/development_main.npz",
    "moe-hb-front-back-0905/results/layer_graphs/external_8b.npz",
    "moe-v4-0904/results/layerwise_mobility/main_reference.npz",
    "moe-v4-0904/results/layerwise_mobility/external_8b.npz",
    "double-selete/trainfree/results/timeout_extension_plus10/development_main_clean_labels.csv",
    "double-selete/trainfree/results/timeout_extension_plus10/external_8b_clean_labels.csv",
]
IMPORTED = [
    "moe-capfree-0906/experiments/capfree_protocol.py",
    "moe-early-window-0906/experiments/common.py",
    "moe-prior-correction-0906/experiments/recompute_task_matched_lift.py",
    "moe-v7-0905/experiments/evaluate_intrinsic_guard_v7.py",
]
PIPELINE = [
    ("harness_checks.py", "reproduce every anchor before reporting anything"),
    ("build_tables.py", "frozen operating-point tables (dev, ext frozen-value, ext rate-matched)"),
    ("select_and_score.py", "select on development, score external, LOTO"),
    ("arm_frontier.py", "arm ceilings at matched external false alarms + null floors"),
    ("union_vs_single.py", "mode-targeted pair vs one detector vs an undifferentiated pair"),
    ("within_task_auc.py", "within-task fixed-chunk survivors-only AUC, per mode"),
    ("summarise.py", "collect key numbers"),
]


def sha256(path: Path, limit: int = 1 << 30) -> str:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
            n += len(block)
            if n >= limit:
                break
    return h.hexdigest()


def main() -> None:
    out = M.RESULTS
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in out.iterdir() if p.name != "manifest.json")
    manifest = {
        "bundle": "moe-failure-mode-detectors-0906",
        "written_utc": datetime.now(UTC).isoformat(),
        "question": ("do the two dominant physical failure modes need different "
                     "detector shapes - a change point for a dropped object, a "
                     "persistent anomaly for a grasp that never formed"),
        "labels": {
            "source": "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv",
            "join": "(run_id, suite, task_name, episode_index)",
            "confidence": "both modes are failure_reason_confidence == medium; "
                          "they are annotations of the physical outcome, not "
                          "evidence about the routing mechanism",
            "modes": list(M.MODES),
        },
        "protocol": {
            "detector_input": "chunk index + routing only",
            "no_cap": True, "no_phase": True, "no_task_identity": True,
            "no_per_task_threshold": True,
            "metric": "true positives, false alarms, lead = length - alarm_chunk",
            "headline_lead": M.HEADLINE_LEAD,
            "leads_reported": list(M.LEADS),
            "baseline": "still running at chunk q0, q0 in [2, 40)",
            "banned_quantities": list(M.BANNED_QUANTITIES),
            "negative_controls": list(M.NEGATIVE_CONTROL),
            "ties": ">= throughout; strict > drops whole tie groups",
            "within_episode_filter": (
                f"headline pool excludes channels within-episode constant in "
                f"more than {A.MAX_WITHIN_EPISODE_CONSTANT:.0%} of episodes"),
            "seed": M.SEED,
        },
        "arms": {k: list(v) for k, v in A.ARMS.items()},
        "gates": list(A.GATES),
        "grid_k": list(A.GRID_K),
        "development_iterations": {
            "count": 4,
            "saw_external_before_the_final_design": True,
            "note": (
                "Stated plainly because it matters for how the external numbers "
                "should be read. Four full development-then-external passes were "
                "run, and external results were computed on all four. What "
                "changed between them was the calibration and reporting "
                "machinery, never the arms, the statistics, the lead budgets or "
                "the metric, all of which were fixed before the first pass and "
                "never touched: "
                "(1) the threshold grid was parameterised by development "
                "non-risk episodes, and was replaced because such a grid cannot "
                "be re-derived on a cohort whose outcomes are unknown; "
                "(2) the grid became label-free (alarm counts), with one "
                "external table at development's numeric thresholds; "
                "(3) a second external table was added, rate-matched, after the "
                "false-alarm count at lead >= 4 was found not to transfer "
                "between cohorts of different suite composition; "
                "(4) the control pool was split into white noise, chunk counter "
                "and episode-constant, after the chunk counter was found to "
                "reproduce the cap-free fixed-chunk baseline exactly, and the "
                "'floor' was redefined as the maximum of the baseline and the "
                "control pools. "
                "So the external numbers here are not a single sealed "
                "evaluation. They are reported as they are, and the headline "
                "conclusions are all negative or control-driven, i.e. in the "
                "direction that this kind of leakage would work against."),
        },
        "inputs": [],
        "imported_not_copied": IMPORTED,
        "pipeline": [{"script": s, "purpose": p} for s, p in PIPELINE],
        "outputs": [],
    }
    for rel in INPUTS:
        p = M.PROJECT / rel
        manifest["inputs"].append({
            "path": rel, "exists": p.exists(),
            "bytes": p.stat().st_size if p.exists() else 0,
            "sha256_first_1GiB": sha256(p) if p.exists() else None})
    for p in files:
        manifest["outputs"].append({
            "file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    manifest["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": subprocess.run(["uname", "-sr"], capture_output=True,
                                   text=True).stdout.strip(),
    }
    M.write_json(out / "manifest.json", manifest)
    print(f"manifest: {len(manifest['inputs'])} inputs, "
          f"{len(manifest['outputs'])} outputs")


if __name__ == "__main__":
    main()
