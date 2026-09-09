#!/usr/bin/env python3
"""Sweep every control step and ask whether activation leads the trap onset.

At each step k the risk set holds only branches that are still alive and have
not yet tripped the frozen physical loop rule.  The label is whether the branch
trips it later than k + lead.  Features come from q0..qk only, so nothing after
the decision point enters the model, and a positive lead forces the model to
speak before the physical trap exists rather than describing an active one.

MoE activation is always judged as an increment over a homomorphic physical and
action control built from the same prefix, never on its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analyze_rolling_star_experiment as rolling
from analyze_trap_onset_sweep import loop_onset_query, per_query_arrays

SEED = 20260829
PCA_COMPONENTS = 12
MIN_RISK_BRANCHES = 40
MIN_POSITIVE = 8


def prefix_control_features(series: dict[str, np.ndarray], k: int) -> np.ndarray:
    """Physical and motion summary of q0..qk, with no routing information."""
    eef = series["eef"][: k + 1]
    goal = series["goal"][: k + 1]
    gripper = series["gripper"][: k + 1]
    objects = series["objects"][: k + 1]
    step = np.linalg.norm(np.diff(eef, axis=0), axis=1) if len(eef) > 1 else np.zeros(1)
    object_step = (
        np.linalg.norm(np.diff(objects, axis=0), axis=2).max(axis=1)
        if len(objects) > 1
        else np.zeros(1)
    )
    path = float(step.sum())
    net = float(np.linalg.norm(eef[-1] - eef[0]))
    recent = step[-4:] if len(step) >= 4 else step
    return np.asarray(
        [
            path,
            net,
            net / max(path, 1e-6),
            float(step.mean()),
            float(step.std()),
            float(recent.mean()),
            float(recent.std()),
            float(step[-1]) if len(step) else 0.0,
            float(goal[-1]),
            float(goal[0] - goal[-1]),
            float(np.diff(goal).mean()) if len(goal) > 1 else 0.0,
            float(np.diff(goal)[-4:].mean()) if len(goal) > 4 else 0.0,
            float(object_step.sum()),
            float(object_step[-4:].mean()),
            float(gripper[-1]),
            float(gripper.std()),
            float(np.abs(np.diff(gripper)).sum()) if len(gripper) > 1 else 0.0,
        ],
        dtype=np.float32,
    )


def prefix_activation_features(block: np.ndarray) -> np.ndarray:
    """Prefix mean, final value and recent drift of every concentration column."""
    final = block[-1]
    mean = block.mean(axis=0)
    recent = block[-4:].mean(axis=0)
    early = block[: max(1, len(block) // 2)].mean(axis=0)
    return np.concatenate([final, mean, recent - early]).astype(np.float32)


def center_within_snapshot(matrix: np.ndarray, snapshots: np.ndarray) -> np.ndarray:
    centered = matrix.astype(np.float64).copy()
    for snapshot in np.unique(snapshots):
        index = np.flatnonzero(snapshots == snapshot)
        centered[index] -= centered[index].mean(axis=0, keepdims=True)
    return centered


def worker_cv_brier(
    matrix: np.ndarray, labels: np.ndarray, workers: np.ndarray
) -> tuple[float, dict[int, float]]:
    """Leave-one-worker-out Brier error, plus the per-worker breakdown."""
    predictions = np.full(len(labels), np.nan)
    for worker in np.unique(workers):
        test = workers == worker
        train = ~test
        if len(np.unique(labels[train])) < 2:
            predictions[test] = float(labels[train].mean())
            continue
        components = min(PCA_COMPONENTS, matrix.shape[1], max(2, int(train.sum()) // 5))
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, random_state=SEED),
            LogisticRegression(C=0.1, max_iter=3000, random_state=SEED),
        )
        model.fit(matrix[train], labels[train])
        predictions[test] = model.predict_proba(matrix[test])[:, 1]
    overall = float(np.mean((predictions - labels) ** 2))
    per_worker = {
        int(worker): float(
            np.mean((predictions[workers == worker] - labels[workers == worker]) ** 2)
        )
        for worker in np.unique(workers)
    }
    return overall, per_worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--leads", type=int, nargs="+", default=[0, 3, 5, 10])
    parser.add_argument("--max-step", type=int, default=40)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = run_root / "analysis_trap_onset"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, _ = rolling.discover_candidates(run_root)
    targets, _ = rolling.load_layout(run_root)
    trajectories = {
        candidate.episode_id: rolling.load_trajectory(candidate)
        for candidate in candidates
    }
    references = rolling.build_goal_references(candidates, trajectories, targets)

    cache = np.load(out_dir / "per_query_activation.npz", allow_pickle=True)
    values = cache["values"]
    episode_of_row = cache["episode_id"]
    step_of_row = cache["control_step"]
    order = np.lexsort((step_of_row, episode_of_row))
    values = values[order]
    episode_sorted = episode_of_row[order]
    step_sorted = step_of_row[order]
    activation: dict[int, np.ndarray] = {}
    for episode in np.unique(episode_sorted):
        mask = episode_sorted == episode
        activation[int(episode)] = values[mask]

    records = []
    for candidate in candidates:
        series = per_query_arrays(
            trajectories[candidate.episode_id], targets, references
        )
        onset, _ = loop_onset_query(
            series["eef"], series["objects"], series["gripper"], series["goal"]
        )
        records.append(
            {
                "candidate": candidate,
                "series": series,
                "onset": onset,
                "n_query": len(series["eef"]),
                "activation": activation.get(candidate.episode_id),
            }
        )

    rows = []
    for lead in args.leads:
        for k in range(rolling.LOOP_MIN_QUERY_LAG, args.max_step + 1):
            risk = [
                record
                for record in records
                if record["n_query"] > k
                and (record["onset"] < 0 or record["onset"] > k)
                and record["activation"] is not None
                and len(record["activation"]) > k
            ]
            if len(risk) < MIN_RISK_BRANCHES:
                continue
            labels = np.asarray(
                [
                    1.0
                    if (record["onset"] >= 0 and record["onset"] > k + lead)
                    else 0.0
                    for record in risk
                ]
            )
            if labels.sum() < MIN_POSITIVE or (1 - labels).sum() < MIN_POSITIVE:
                continue
            workers = np.asarray([record["candidate"].worker for record in risk])
            snapshots = np.asarray(
                [record["candidate"].snapshot_key for record in risk]
            )
            if len(np.unique(workers)) < 2:
                continue
            control = np.stack(
                [prefix_control_features(record["series"], k) for record in risk]
            )
            moe = np.stack(
                [
                    prefix_activation_features(record["activation"][: k + 1])
                    for record in risk
                ]
            )
            control = center_within_snapshot(control, snapshots)
            moe = center_within_snapshot(moe, snapshots)
            rate_brier = float(np.mean((labels.mean() - labels) ** 2))
            control_brier, control_workers = worker_cv_brier(control, labels, workers)
            joint_brier, joint_workers = worker_cv_brier(
                np.hstack([control, moe]), labels, workers
            )
            moe_only_brier, _ = worker_cv_brier(moe, labels, workers)
            gains = {
                worker: control_workers[worker] - joint_workers[worker]
                for worker in control_workers
            }
            rows.append(
                {
                    "lead": lead,
                    "step": k,
                    "risk_branches": len(risk),
                    "positives": int(labels.sum()),
                    "snapshots": int(len(np.unique(snapshots))),
                    "workers": int(len(np.unique(workers))),
                    "rate_brier": rate_brier,
                    "control_brier": control_brier,
                    "moe_only_brier": moe_only_brier,
                    "control_plus_moe_brier": joint_brier,
                    "moe_increment": control_brier - joint_brier,
                    "control_gain_vs_rate": rate_brier - control_brier,
                    "workers_positive": int(sum(1 for v in gains.values() if v > 0)),
                    "worker_gain_min": float(min(gains.values())),
                    "worker_gain_max": float(max(gains.values())),
                }
            )
            print(
                f"lead={lead} k={k:2d} n={len(risk):3d} pos={int(labels.sum()):3d} "
                f"ctrl={control_brier:.4f} +moe={joint_brier:.4f} "
                f"inc={control_brier - joint_brier:+.4f} "
                f"w+={rows[-1]['workers_positive']}/4",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "onset_lead_sweep.csv", index=False)
    print(f"\nwrote {out_dir / 'onset_lead_sweep.csv'}  rows={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
