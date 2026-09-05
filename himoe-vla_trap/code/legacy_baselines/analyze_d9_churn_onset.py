#!/usr/bin/env python3
"""Replicate the trap-animation churn result across four initial states.

The trap animation established, on a single initial state, that deep-layer
route churn separates stasis from success only after the progress onset.  This
script reruns that exact pairing on the rolling-star capture, which holds four
independent initial states and 22 shared-trunk snapshots.

Definitions are taken from ``build_trap_animation_data.py`` rather than from
the rolling-star analysis, so the two results are directly comparable:

* trap onset is the first control step from which the branch never again
  improves its held-out goal distance by more than ``PROGRESS_EPS_M`` while
  still outside ``GOAL_RADIUS_M``;
* churn is ``1 - Jaccard`` of consecutive control steps' Top-4 expert sets at
  the final denoise step, averaged over the ten action tokens, per HB layer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy import stats

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
sys.path.insert(0, str(WORKSPACE_ROOT / "himoe-route-capture"))

import analyze_rolling_star_experiment as rolling  # noqa: E402

GOAL_RADIUS_M = 0.05
PROGRESS_EPS_M = 0.01
DENOISE = 9
ACTION_TOKENS = slice(1, 11)
FRONT_AXES = [0, 1, 2, 3]
DEEP_AXES = [4, 5, 6, 7]
BLOCK = 512


def progress_trap_onset(distance: np.ndarray) -> int:
    """First step after which no further goal progress is ever made."""
    best = np.minimum.accumulate(distance)
    future = np.minimum.accumulate(distance[::-1])[::-1]
    eligible = np.flatnonzero(
        (best - future <= PROGRESS_EPS_M) & (best > GOAL_RADIUS_M)
    )
    return int(eligible[0]) if len(eligible) else -1


def joint_pose_series(
    arrays: dict[str, np.ndarray], targets: list[dict]
) -> np.ndarray:
    """Per-query joint pose of every target object, concatenated.

    The trap animation scores a rollout against successful terminal poses in one
    joint space rather than per object, so a branch that abandons one object is
    not credited for progress on the other.
    """
    sim = np.asarray(arrays["control_sim_state"], dtype=np.float32)
    columns = [
        sim[:, int(target["state_lo"]) : int(target["state_lo"]) + 3]
        for target in targets
    ]
    pose = np.concatenate(columns, axis=1)
    starts = np.r_[
        0, np.flatnonzero(np.diff(arrays["control_query_index"]) != 0) + 1
    ]
    return pose[starts]


def held_out_goal_distance(
    pose: np.ndarray, reference: np.ndarray, exclude: np.ndarray
) -> np.ndarray:
    """Nearest-neighbour distance to successful terminal poses, self excluded."""
    keep = reference[~exclude]
    if not len(keep):
        return np.full(len(pose), np.inf, dtype=np.float32)
    difference = pose[:, None, :] - keep[None, :, :]
    return np.linalg.norm(difference, axis=2).min(axis=1)


def churn_series(expert_ids: np.ndarray) -> np.ndarray:
    """1 - Jaccard of consecutive steps' Top-4 sets, per layer.

    ``expert_ids`` is [query, layer, token, 4].  Top-4 sets are compared token
    by token and averaged, matching the animation's construction.
    """
    n_query, n_layer, n_token, _ = expert_ids.shape
    if n_query < 2:
        return np.zeros((0, n_layer), dtype=np.float32)
    out = np.zeros((n_query - 1, n_layer), dtype=np.float32)
    for step in range(n_query - 1):
        for layer in range(n_layer):
            values = []
            for token in range(n_token):
                left = set(expert_ids[step, layer, token].tolist())
                right = set(expert_ids[step + 1, layer, token].tolist())
                union = len(left | right)
                values.append(1.0 - (len(left & right) / union if union else 1.0))
            out[step, layer] = float(np.mean(values))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = (args.out_dir or run_root / "analysis_churn_onset").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, _ = rolling.discover_candidates(run_root)
    targets, _ = rolling.load_layout(run_root)
    trajectories = {
        candidate.episode_id: rolling.load_trajectory(candidate)
        for candidate in candidates
    }
    store = zarr.open(str(run_root / "formal/server/routes.zarr"), mode="r")
    episode_of_row = store["episode_id"][:]
    step_of_row = store["control_step"][:]
    n_row = len(episode_of_row)
    ids = np.zeros((n_row, 8, 10, 4), dtype=np.uint8)
    for start in range(0, n_row, BLOCK):
        stop = min(start + BLOCK, n_row)
        ids[start:stop] = store["hb_expert_ids"][start:stop][
            :, :, DENOISE, ACTION_TOKENS, :
        ]
        if start % (BLOCK * 8) == 0:
            print(f"  ids {stop}/{n_row}", flush=True)

    order = np.lexsort((step_of_row, episode_of_row))
    ids = ids[order]
    episode_sorted = episode_of_row[order]
    by_episode = {
        int(episode): ids[episode_sorted == episode]
        for episode in np.unique(episode_sorted)
    }

    # Successful terminal joint poses form the reference set.  A branch is
    # never scored against a success from its own snapshot.
    success_pose = []
    success_snapshot = []
    for candidate in candidates:
        if not candidate.success:
            continue
        pose = joint_pose_series(trajectories[candidate.episode_id], targets)
        success_pose.append(pose[-1])
        success_snapshot.append(candidate.snapshot_key)
    success_pose = np.stack(success_pose)
    success_snapshot = np.asarray(success_snapshot)

    rows = []
    for candidate in candidates:
        pose = joint_pose_series(trajectories[candidate.episode_id], targets)
        distance = held_out_goal_distance(
            pose, success_pose, success_snapshot == candidate.snapshot_key
        )
        onset = progress_trap_onset(distance)
        block = by_episode.get(candidate.episode_id)
        if block is None or len(block) < 2:
            continue
        churn = churn_series(block)
        n = min(len(churn), len(distance) - 1)
        churn = churn[:n]
        deep = churn[:, DEEP_AXES].mean(axis=1)
        front = churn[:, FRONT_AXES].mean(axis=1)
        cut = len(deep) if onset < 0 else min(onset, len(deep))
        rows.append(
            {
                "worker": candidate.worker,
                "snapshot_key": candidate.snapshot_key,
                "candidate": candidate.candidate,
                "episode_id": candidate.episode_id,
                "success": candidate.success,
                "n_query": len(distance),
                "onset": onset,
                "deep_churn_all": float(deep.mean()),
                "front_churn_all": float(front.mean()),
                "deep_churn_before_onset": float(deep[:cut].mean()) if cut >= 4 else np.nan,
                "front_churn_before_onset": float(front[:cut].mean()) if cut >= 4 else np.nan,
                "deep_churn_after_onset": (
                    float(deep[onset:].mean())
                    if onset >= 0 and len(deep) - onset >= 4
                    else np.nan
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "churn_onset.csv", index=False)

    print("\n=== onset coverage ===")
    trapped = frame[frame.onset >= 0]
    print(f"  branches {len(frame)}  with onset {len(trapped)}  "
          f"success {int(frame.success.sum())}")
    print(f"  onset median t{trapped.onset.median():.0f}  "
          f"range {trapped.onset.min():.0f}-{trapped.onset.max():.0f}")

    summary = {"per_worker": [], "pooled": {}}
    print("\n=== deep churn before onset: success vs will-trap ===")
    print(f"{'worker':>7} {'nS':>4} {'nT':>4} {'success':>9} {'will_trap':>10} {'diff':>9}")
    for worker in sorted(frame.worker.unique()):
        part = frame[(frame.worker == worker) & frame.deep_churn_before_onset.notna()]
        s = part[part.success].deep_churn_before_onset
        t = part[~part.success & (part.onset >= 0)].deep_churn_before_onset
        if len(s) < 3 or len(t) < 3:
            print(f"{worker:>7} {len(s):>4} {len(t):>4}   (insufficient)")
            continue
        diff = float(s.mean() - t.mean())
        summary["per_worker"].append(
            {"worker": int(worker), "n_success": len(s), "n_trap": len(t),
             "success_churn": float(s.mean()), "trap_churn": float(t.mean()),
             "difference": diff}
        )
        print(f"{worker:>7} {len(s):>4} {len(t):>4} {s.mean():>9.4f} {t.mean():>10.4f} {diff:>+9.4f}")

    part = frame[frame.deep_churn_before_onset.notna()]
    s = part[part.success].deep_churn_before_onset
    t = part[~part.success & (part.onset >= 0)].deep_churn_before_onset
    pooled_diff = float(s.mean() - t.mean())
    p_before = float(stats.mannwhitneyu(s, t, alternative="greater").pvalue)
    print(f"{'pooled':>7} {len(s):>4} {len(t):>4} {s.mean():>9.4f} {t.mean():>10.4f} "
          f"{pooled_diff:>+9.4f}   p={p_before:.4f}")

    after = frame[frame.deep_churn_after_onset.notna()].deep_churn_after_onset
    print("\n=== deep churn after onset ===")
    print(f"  trapped after onset {after.mean():.4f} (n={len(after)})  "
          f"vs success whole {s.mean():.4f}   diff {float(s.mean() - after.mean()):+.4f}")

    summary["pooled"] = {
        "before_onset_success_churn": float(s.mean()),
        "before_onset_trap_churn": float(t.mean()),
        "before_onset_difference": pooled_diff,
        "before_onset_p_one_sided": p_before,
        "after_onset_trap_churn": float(after.mean()),
        "after_onset_difference": float(s.mean() - after.mean()),
        "branches": int(len(frame)),
        "workers": int(frame.worker.nunique()),
    }
    with open(out_dir / "churn_onset_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
