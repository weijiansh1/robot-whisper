#!/usr/bin/env python3
"""Locate the first trap onset query, then sweep every control step forward.

The tail experiment froze q24-q31 after the signal was seen there and predicted
a whole-episode label.  This analysis does neither.  For every branch it finds
the first query at which the frozen physical loop rule fires, then walks the
control steps from the very first one, asking at each step k, using only
routing observed at q0..qk, whether a branch that is not yet trapped will enter
the trap later.  The question is whether the activation signal leads the
physical onset, so no window may be chosen with knowledge of the answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
sys.path.insert(0, str(WORKSPACE_ROOT / "himoe-route-capture"))

import analyze_rolling_star_experiment as rolling  # noqa: E402


def loop_onset_query(
    eef: np.ndarray,
    objects: np.ndarray,
    gripper: np.ndarray,
    goal_distance: np.ndarray,
) -> tuple[int, int]:
    """First query index satisfying the frozen loop rule, and its partner.

    Identical predicate to ``rolling.query_loop_metrics`` but returns the
    earliest firing right-hand query instead of whole-episode aggregates.
    """
    n_query = len(eef)
    if n_query <= rolling.LOOP_MIN_QUERY_LAG:
        return -1, -1
    eef_step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    eef_cumulative = np.r_[0.0, np.cumsum(eef_step)]
    for right in range(rolling.LOOP_MIN_QUERY_LAG, n_query):
        for left in range(0, right - rolling.LOOP_MIN_QUERY_LAG + 1):
            eef_return = float(np.linalg.norm(eef[right] - eef[left]))
            object_return = float(
                np.linalg.norm(objects[right] - objects[left], axis=1).max()
            )
            grip_return = float(abs(gripper[right] - gripper[left]))
            path = float(eef_cumulative[right] - eef_cumulative[left])
            progress = float(goal_distance[left] - goal_distance[right])
            if (
                eef_return <= rolling.LOOP_EEF_RETURN_M
                and object_return <= rolling.LOOP_OBJECT_RETURN_M
                and grip_return <= rolling.LOOP_GRIPPER_RETURN_M
                and path >= rolling.LOOP_EEF_PATH_M
                and progress <= 0.035
            ):
                return right, left
    return -1, -1


def per_query_arrays(
    arrays: dict[str, np.ndarray],
    targets: list[dict],
    references: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rebuild the per-query physical series exactly as the main analysis does."""
    sim = np.asarray(arrays["control_sim_state"], dtype=np.float32)
    eef = np.asarray(arrays["control_eef_position"], dtype=np.float32)
    gripper = np.asarray(arrays["control_gripper_qpos"], dtype=np.float32).mean(axis=1)
    positions = []
    distances = []
    for target in targets:
        lo = int(target["state_lo"])
        position = sim[:, lo : lo + 3]
        positions.append(position)
        distances.append(
            rolling.distance_to_references(position, references[str(target["joint"])])
        )
    objects = np.stack(positions, axis=1)
    aggregate_goal = np.max(np.stack(distances, axis=1), axis=1)
    starts = np.r_[
        0, np.flatnonzero(np.diff(arrays["control_query_index"]) != 0) + 1
    ]
    return {
        "eef": eef[starts],
        "objects": objects[starts],
        "gripper": gripper[starts],
        "goal": aggregate_goal[starts],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = (args.out_dir or run_root / "analysis_trap_onset").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, _ = rolling.discover_candidates(run_root)
    targets, _ = rolling.load_layout(run_root)
    trajectories = {
        candidate.episode_id: rolling.load_trajectory(candidate)
        for candidate in candidates
    }
    references = rolling.build_goal_references(candidates, trajectories, targets)

    rows = []
    for candidate in candidates:
        series = per_query_arrays(
            trajectories[candidate.episode_id], targets, references
        )
        onset, partner = loop_onset_query(
            series["eef"], series["objects"], series["gripper"], series["goal"]
        )
        rows.append(
            {
                "worker": candidate.worker,
                "snapshot": candidate.snapshot,
                "snapshot_key": candidate.snapshot_key,
                "candidate": candidate.candidate,
                "episode_id": candidate.episode_id,
                "success": candidate.success,
                "n_query": len(series["eef"]),
                "loop_onset_query": onset,
                "loop_onset_partner": partner,
                "ever_loops": onset >= 0,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "trap_onset.csv", index=False)

    looping = frame[frame.ever_loops]
    summary = {
        "branches": int(len(frame)),
        "branches_with_loop_onset": int(len(looping)),
        "onset_query_min": int(looping.loop_onset_query.min()),
        "onset_query_p10": float(looping.loop_onset_query.quantile(0.10)),
        "onset_query_median": float(looping.loop_onset_query.median()),
        "onset_query_p90": float(looping.loop_onset_query.quantile(0.90)),
        "onset_query_max": int(looping.loop_onset_query.max()),
        "onset_before_q24": int((looping.loop_onset_query < 24).sum()),
        "onset_at_or_after_q24": int((looping.loop_onset_query >= 24).sum()),
        "success_branches_with_loop_onset": int(looping.success.sum()),
        "failure_branches_with_loop_onset": int((~looping.success).sum()),
    }
    with open(out_dir / "onset_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
