#!/usr/bin/env python3
"""Validate critical invariants and emit a compact, hashed result summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


BUNDLE = Path(__file__).resolve().parent.parent
RESULTS = BUNDLE / "results"


def load(relative: str) -> dict[str, Any]:
    return json.loads((BUNDLE / relative).read_text(encoding="utf-8"))


def one(rows: list[dict[str, Any]], **match: Any) -> dict[str, Any]:
    selected = [row for row in rows if all(row.get(key) == value for key, value in match.items())]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {match}, found {len(selected)}")
    return selected[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    build = load("results/conditional_profiles/build_summary.json")
    evaluation = load("results/conditional_evaluation/summary.json")
    locked = load("results/locked_loop_detector/summary.json")
    forks = load("results/conditional_forks/summary.json")
    state = load("results/state_anchor_audit.json")

    assert build["tasks"] == 45
    assert build["episodes"] == 18560
    assert build["rows"] == 305030
    assert build["raw_router_shape"] == [8, 10, 11, 32]
    assert build["total_dynamic_axes"] == 584
    assert build["outcomes_loaded"] is False
    assert build["fitted_weights"] is False
    assert locked["training"] is False
    assert locked["locked_axis"] == "query|action_partial|front_acceleration"
    assert state["all"]["rows"] == build["rows"]

    handoff = [
        row
        for row in forks["handoff_contrast"]
        if row["view"] in {"action_conditional", "action_partial"}
    ]
    contraction = [
        row
        for row in forks["ensemble_contraction"]
        if row["view"] in {"action_conditional", "action_partial"}
    ]
    q0_fixed = {
        event: one(rows, axis=f"fixed_composite|{event}")
        for event, rows in forks["fixed_outcome_axes"].items()
    }
    q2_loop = {
        corpus: {
            score: one(locked["q_minus_2"], corpus=corpus, score=score)
            for score in ("baseline", "conditional", "fusion")
        }
        for corpus in ("main16x32", "grid50x8")
    }
    q2_increment = {
        corpus: one(locked["q_minus_2_paired_increment"], corpus=corpus)
        for corpus in ("main16x32", "grid50x8")
    }
    q2_static = {
        corpus: {
            score: one(
                evaluation["primary_q_minus_2"],
                corpus=corpus,
                score=score,
            )
            for score in (
                "static_baseline",
                "static_conditional_arrest",
                "static_equal_fusion",
            )
        }
        for corpus in ("main16x32", "grid50x8")
    }

    final = {
        "schema": "himoe.routing_transfer_final.v1",
        "claim_status": {
            "front_absorption_back_structure": "confirmed in two same-snapshot fork datasets",
            "q0_handoff_as_trap_detector": "rejected by fixed tests",
            "q_minus_2_loop_transfer_increment": (
                "direction replicated; held-out paired increment confidence interval crosses zero"
            ),
            "static_increment": "not consistent across corpora",
        },
        "training": {
            "learned_model": False,
            "outcome_weights": False,
            "discovery_axis_selection": True,
            "probability_calibration": False,
        },
        "data": {
            "tasks": build["tasks"],
            "episodes": build["episodes"],
            "queries": build["rows"],
            "router_shape": build["raw_router_shape"],
            "dynamic_axes": build["total_dynamic_axes"],
        },
        "gpu": {
            "cuda_visible_devices": build["cuda_visible_devices"],
            "profile_workers": build["workers"],
            "fork_datasets": forks["datasets"],
        },
        "state_anchor": state["corpora"],
        "same_snapshot_contraction": contraction,
        "same_snapshot_handoff": handoff,
        "q0_fixed_outcome_tests": q0_fixed,
        "q_minus_2_loop": q2_loop,
        "q_minus_2_loop_paired_increment": q2_increment,
        "q_minus_2_static": q2_static,
        "locked_detector": {
            "baseline_axis": locked["baseline_axis"],
            "conditional_axis": locked["locked_axis"],
            "direction": locked["locked_direction"],
            "fusion": locked["fusion"],
            "selection": locked["selection"],
            "caveat": locked["caveat"],
        },
    }
    final_path = RESULTS / "final_summary.json"
    final_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = (
        "README.md",
        "REPORT_ZH.md",
        "transfer/field.py",
        "experiments/build_transfer_profiles_gpu.py",
        "experiments/evaluate_transfer.py",
        "experiments/evaluate_locked_loop_detector.py",
        "experiments/analyze_conditional_forks_gpu.py",
        "experiments/audit_state_anchor.py",
        "tests/test_field.py",
        "results/final_summary.json",
        "results/state_anchor_audit.json",
        "results/conditional_profiles/build_summary.json",
        "results/conditional_evaluation/summary.json",
        "results/conditional_evaluation/transfer_increment.png",
        "results/locked_loop_detector/summary.json",
        "results/locked_loop_detector/locked_loop_leads.png",
        "results/conditional_forks/summary.json",
        "results/conditional_forks/conditional_fork_contraction.png",
    )
    manifest = []
    for relative in artifacts:
        path = BUNDLE / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    manifest_path = RESULTS / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"schema": "himoe.routing_transfer_artifacts.v1", "artifacts": manifest},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"final_summary": str(final_path), "artifacts": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
