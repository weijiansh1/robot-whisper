#!/usr/bin/env python3
"""Measure routing fluctuation inside a chunk as well as between chunks.

Every earlier pass in this session read routing at the final denoise step only,
so the ten iterations the router runs while flow-matching denoises one action
chunk were never examined.  Within a chunk the routing decision should settle
as the latent resolves; a decision that keeps jumping between denoise steps is
an instability that lives entirely inside the network and needs no environment
to be defined.

Two scales are reported per branch, both on the executed mixture, which is the
recorded gate weight of the selected experts renormalized into the 32-expert
basis so that a p4=p5 swap moves it only by the tie gap:

* within chunk -- Hellinger between consecutive denoise steps d0..d9, plus how
  much the late steps still move relative to the early ones;
* between chunks -- Hellinger between consecutive queries at the final step.

Branches are then compared by how far they actually moved the target objects,
paired inside a snapshot so that geometry and phase are held fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy import stats

import analyze_rolling_star_experiment as rolling
from analyze_trap_onset_sweep import per_query_arrays

DEEP_AXES = [4, 5, 6, 7]
ACTION_TOKENS = slice(1, 11)
N_DENOISE = 10
BLOCK = 128
EARLY = slice(0, 3)
LATE = slice(6, 9)


def executed_mixture(weight: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """Scatter renormalized top-4 gate weights into the 32-expert basis."""
    shape = weight.shape[:-1] + (32,)
    mixture = np.zeros(shape, dtype=np.float32)
    weight = weight / np.maximum(weight.sum(axis=-1, keepdims=True), 1e-12)
    np.put_along_axis(mixture, chosen, weight, axis=-1)
    return mixture


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.maximum(left, 0.0)) - np.sqrt(np.maximum(right, 0.0))
    return np.sqrt(np.clip(0.5 * (root ** 2).sum(axis=-1), 0.0, 1.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = run_root / "analysis_chunk_fluctuation"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, _ = rolling.discover_candidates(run_root)
    targets, _ = rolling.load_layout(run_root)
    trajectories = {
        c.episode_id: rolling.load_trajectory(c) for c in candidates
    }
    references = rolling.build_goal_references(candidates, trajectories, targets)

    store = zarr.open(str(run_root / "formal/server/routes.zarr"), mode="r")
    episode_of_row = store["episode_id"][:]
    step_of_row = store["control_step"][:]
    n_row = len(episode_of_row)

    within = np.zeros((n_row, N_DENOISE - 1), dtype=np.float32)
    final = np.zeros((n_row, len(DEEP_AXES), 10, 32), dtype=np.float32)
    for start in range(0, n_row, BLOCK):
        stop = min(start + BLOCK, n_row)
        weight = store["hb_selected_prob"][start:stop][
            :, DEEP_AXES, :, ACTION_TOKENS, :
        ].astype(np.float32)
        chosen = store["hb_expert_ids"][start:stop][
            :, DEEP_AXES, :, ACTION_TOKENS, :
        ].astype(np.int64)
        mixture = executed_mixture(weight, chosen)      # (b, layer, denoise, token, 32)
        step = hellinger(mixture[:, :, :-1], mixture[:, :, 1:])
        within[start:stop] = step.mean(axis=(1, 3))     # average layer and token
        final[start:stop] = mixture[:, :, -1]
        if start % (BLOCK * 32) == 0:
            print(f"  rows {stop}/{n_row}", flush=True)

    order = np.lexsort((step_of_row, episode_of_row))
    within = within[order]
    final = final[order]
    episode_sorted = episode_of_row[order]
    within_by = {int(e): within[episode_sorted == e] for e in np.unique(episode_sorted)}
    final_by = {int(e): final[episode_sorted == e] for e in np.unique(episode_sorted)}

    rows = []
    for candidate in candidates:
        inner = within_by[candidate.episode_id]
        outer = final_by[candidate.episode_id]
        cross = (
            float(np.mean(hellinger(outer[:-1], outer[1:])))
            if len(outer) > 1
            else np.nan
        )
        early = float(inner[:, EARLY].mean())
        late = float(inner[:, LATE].mean())
        series = per_query_arrays(trajectories[candidate.episode_id], targets, references)
        moved = float(
            np.linalg.norm(
                series["objects"][-1] - series["objects"][0], axis=1
            ).max()
        )
        rows.append(
            {
                "worker": candidate.worker,
                "snapshot_key": candidate.snapshot_key,
                "episode_id": candidate.episode_id,
                "success": candidate.success,
                "object_moved_m": moved,
                "within_chunk_path": float(inner.sum(axis=1).mean()),
                "within_chunk_early": early,
                "within_chunk_late": late,
                "within_chunk_settling": float(late / early) if early > 0 else np.nan,
                "cross_chunk": cross,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "chunk_fluctuation.csv", index=False)

    metrics = [
        "within_chunk_path",
        "within_chunk_early",
        "within_chunk_late",
        "within_chunk_settling",
        "cross_chunk",
    ]
    print("\n=== denoise-step fluctuation profile, all branches ===")
    profile = within.mean(axis=0)
    print("  d(k)->d(k+1): " + " ".join(f"{v:.4f}" for v in profile))

    print("\n=== absorbing (object <= 2mm) versus the rest, per worker ===")
    frame["ineffective"] = frame.object_moved_m <= 0.002
    for worker, part in frame.groupby("worker"):
        a = part[part.ineffective]
        b = part[~part.ineffective]
        if len(a) < 5 or len(b) < 5:
            print(f"  worker {worker}: n_ineffective={len(a)} n_effective={len(b)}  (skip)")
            continue
        line = [f"  worker {worker}: n={len(a)}/{len(b)}"]
        for metric in metrics:
            line.append(f"{metric}={a[metric].mean() - b[metric].mean():+.4f}")
        print("  " + "  ".join(line))

    print("\n=== paired inside snapshot: object movement versus fluctuation ===")
    summary = {}
    for metric in metrics:
        correlations = []
        for _, group in frame.groupby("snapshot_key"):
            if group.object_moved_m.std() < 1e-6 or group[metric].isna().any():
                continue
            rho, _ = stats.spearmanr(group.object_moved_m, group[metric])
            if np.isfinite(rho):
                correlations.append(rho)
        correlations = np.asarray(correlations)
        positive = int((correlations > 0).sum())
        p_value = float(stats.binomtest(positive, len(correlations), 0.5).pvalue)
        summary[metric] = {
            "snapshots": len(correlations),
            "median_rho": float(np.median(correlations)),
            "positive": positive,
            "sign_test_p": p_value,
        }
        print(
            f"  {metric:>22}: rho median {np.median(correlations):+.3f}  "
            f"positive {positive}/{len(correlations)}  p={p_value:.4f}"
        )

    with open(out_dir / "chunk_fluctuation_summary.json", "w") as handle:
        json.dump(
            {"denoise_profile": profile.tolist(), "paired": summary},
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
