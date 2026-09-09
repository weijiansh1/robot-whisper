#!/usr/bin/env python3
"""Audit the post-hoc q0 state-route loop classifier worker by worker."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

import analyze_rolling_star_experiment as analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--seed", type=int, default=analysis.SEED)
    return parser.parse_args()


def heldout_rate(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    prediction = np.full(len(labels), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    placeholder = np.ones((len(labels), 1), dtype=np.float32)
    for train, test in splitter.split(placeholder, labels, groups):
        prediction[test] = float(labels[train].mean())
    return prediction


def exact_group_sign_flip_p(gains: np.ndarray) -> float:
    """One-sided exact paired test with each worker as one exchangeable unit."""
    observed = float(gains.mean())
    null = np.asarray(
        [
            np.mean(gains * np.asarray(signs, dtype=np.float64))
            for signs in itertools.product((-1.0, 1.0), repeat=len(gains))
        ]
    )
    return float(np.mean(null >= observed - 1e-15))


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_dir = (args.analysis_dir or run_root / "analysis").resolve()
    candidates, _ = analysis.discover_candidates(run_root)
    physical = pd.read_csv(out_dir / "candidate_physical_labels.csv")
    ordered = physical.set_index("episode_id").loc[
        [item.episode_id for item in candidates]
    ].reset_index()
    trajectories = {
        item.episode_id: analysis.load_trajectory(item) for item in candidates
    }
    hb_routes, as_routes, route_audit = analysis.load_routes(run_root, candidates)
    banks, _, _ = analysis.route_feature_banks(
        candidates, trajectories, hb_routes, as_routes
    )

    failures = np.flatnonzero(ordered["failure"].to_numpy(dtype=bool))
    labels = ordered.loc[failures, "label_loop_or_cycling"].to_numpy(dtype=np.int64)
    all_snapshots = np.asarray([item.snapshot_key for item in candidates])
    workers = np.asarray([item.worker for item in candidates], dtype=np.int64)[failures]
    bank = banks[1]
    state_all = analysis.center_within_groups(
        bank["moe_state_full"], all_snapshots
    )
    noise_action_all = analysis.center_within_groups(
        np.c_[bank["noise_raw"], bank["action_raw"]], all_snapshots
    )
    state = state_all[failures]
    noise_action = noise_action_all[failures]

    prediction = analysis.group_cv_predictions(
        state, labels, workers, use_pca=True, seed=args.seed
    )
    noise_prediction = analysis.group_cv_predictions(
        noise_action, labels, workers, use_pca=True, seed=args.seed
    )
    rate_prediction = heldout_rate(labels, workers)
    if any(
        np.any(~np.isfinite(values))
        for values in (prediction, noise_prediction, rate_prediction)
    ):
        raise ValueError("non-finite selected-loop OOF prediction")

    selected = ordered.iloc[failures].reset_index(drop=True)
    rows = pd.DataFrame(
        {
            "worker": workers,
            "snapshot_key": selected["snapshot_key"],
            "candidate": selected["candidate"],
            "episode_id": selected["episode_id"],
            "loop_or_cycling": labels.astype(bool),
            "predicted_loop_q0_state_within_snapshot": prediction,
            "predicted_loop_failure_rate_only": rate_prediction,
            "predicted_loop_noise_action_within_snapshot": noise_prediction,
        }
    )
    rows["brier_gain_vs_rate"] = (
        (labels - rate_prediction) ** 2 - (labels - prediction) ** 2
    )
    rows["brier_gain_vs_noise_action"] = (
        (labels - noise_prediction) ** 2 - (labels - prediction) ** 2
    )
    rows.to_csv(out_dir / "exploratory_loop_q0_worker_oof.csv", index=False)

    metrics = []
    for worker, group in rows.groupby("worker", sort=True):
        worker_labels = group["loop_or_cycling"].to_numpy(dtype=np.int64)
        metrics.append(
            {
                "worker": int(worker),
                "failure_branches": len(group),
                "snapshots": int(group["snapshot_key"].nunique()),
                "loop_branches": int(worker_labels.sum()),
                "loop_rate": float(worker_labels.mean()),
                "selected_model_brier": float(
                    brier_score_loss(
                        worker_labels,
                        group["predicted_loop_q0_state_within_snapshot"],
                    )
                ),
                "failure_rate_only_brier": float(
                    brier_score_loss(
                        worker_labels, group["predicted_loop_failure_rate_only"]
                    )
                ),
                "noise_action_brier": float(
                    brier_score_loss(
                        worker_labels,
                        group["predicted_loop_noise_action_within_snapshot"],
                    )
                ),
                "brier_gain_vs_rate": float(group["brier_gain_vs_rate"].mean()),
                "brier_gain_vs_noise_action": float(
                    group["brier_gain_vs_noise_action"].mean()
                ),
            }
        )
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(
        out_dir / "exploratory_loop_q0_worker_metrics.csv", index=False
    )

    overall = {
        "failure_branches": len(rows),
        "loop_branches": int(labels.sum()),
        "workers": len(metrics_frame),
        "selected_model_brier": float(brier_score_loss(labels, prediction)),
        "failure_rate_only_brier": float(brier_score_loss(labels, rate_prediction)),
        "noise_action_brier": float(brier_score_loss(labels, noise_prediction)),
        "branch_weighted_brier_gain_vs_rate": float(rows["brier_gain_vs_rate"].mean()),
        "branch_weighted_brier_gain_vs_noise_action": float(
            rows["brier_gain_vs_noise_action"].mean()
        ),
        "equal_worker_mean_brier_gain_vs_rate": float(
            metrics_frame["brier_gain_vs_rate"].mean()
        ),
        "equal_worker_mean_brier_gain_vs_noise_action": float(
            metrics_frame["brier_gain_vs_noise_action"].mean()
        ),
        "exact_worker_sign_flip_p_one_sided_vs_rate": exact_group_sign_flip_p(
            metrics_frame["brier_gain_vs_rate"].to_numpy()
        ),
        "exact_worker_sign_flip_p_one_sided_vs_noise_action": (
            exact_group_sign_flip_p(
                metrics_frame["brier_gain_vs_noise_action"].to_numpy()
            )
        ),
    }
    audit = {
        "schema": "himoe.exploratory_loop_q0_worker_audit.v1",
        "selection_status": "post_hoc_from_364_subtype_model_rows",
        "confirmatory": False,
        "target": "physical label_loop_or_cycling among failed branches",
        "feature": "q0 state-token full HB probabilities, centered within K=16 snapshot",
        "cv": "leave_one_worker_out",
        "tail_features_used": False,
        "seed": args.seed,
        "route_rows_total": route_audit["route_rows_total"],
        "overall": overall,
        "per_worker": metrics,
    }
    (out_dir / "exploratory_loop_q0_worker_audit.json").write_text(
        json.dumps(analysis.plain(audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Operational check: predict eventual loop directly on every branch. Unlike
    # the subtype audit above, this target does not assume that failure is known.
    all_labels = ordered["label_loop_or_cycling"].to_numpy(dtype=np.int64)
    all_workers = np.asarray([item.worker for item in candidates], dtype=np.int64)
    all_prediction = analysis.group_cv_predictions(
        state_all, all_labels, all_workers, use_pca=True, seed=args.seed
    )
    all_noise_prediction = analysis.group_cv_predictions(
        noise_action_all, all_labels, all_workers, use_pca=True, seed=args.seed
    )
    all_rate_prediction = heldout_rate(all_labels, all_workers)
    all_rows = pd.DataFrame(
        {
            "worker": all_workers,
            "snapshot_key": ordered["snapshot_key"],
            "candidate": ordered["candidate"],
            "episode_id": ordered["episode_id"],
            "success": ordered["success"].astype(bool),
            "loop_or_cycling": all_labels.astype(bool),
            "predicted_loop_q0_state_within_snapshot": all_prediction,
            "predicted_loop_failure_rate_only": all_rate_prediction,
            "predicted_loop_noise_action_within_snapshot": all_noise_prediction,
        }
    )
    all_rows["brier_gain_vs_rate"] = (
        (all_labels - all_rate_prediction) ** 2
        - (all_labels - all_prediction) ** 2
    )
    all_rows["brier_gain_vs_noise_action"] = (
        (all_labels - all_noise_prediction) ** 2
        - (all_labels - all_prediction) ** 2
    )
    all_rows.to_csv(out_dir / "exploratory_loop_q0_all_branch_oof.csv", index=False)
    all_metrics = []
    for worker, group in all_rows.groupby("worker", sort=True):
        worker_labels = group["loop_or_cycling"].to_numpy(dtype=np.int64)
        all_metrics.append(
            {
                "worker": int(worker),
                "branches": len(group),
                "snapshots": int(group["snapshot_key"].nunique()),
                "successes": int(group["success"].sum()),
                "loop_branches": int(worker_labels.sum()),
                "loop_rate": float(worker_labels.mean()),
                "selected_model_brier": float(
                    brier_score_loss(
                        worker_labels,
                        group["predicted_loop_q0_state_within_snapshot"],
                    )
                ),
                "failure_rate_only_brier": float(
                    brier_score_loss(
                        worker_labels, group["predicted_loop_failure_rate_only"]
                    )
                ),
                "noise_action_brier": float(
                    brier_score_loss(
                        worker_labels,
                        group["predicted_loop_noise_action_within_snapshot"],
                    )
                ),
                "brier_gain_vs_rate": float(group["brier_gain_vs_rate"].mean()),
                "brier_gain_vs_noise_action": float(
                    group["brier_gain_vs_noise_action"].mean()
                ),
            }
        )
    all_metrics_frame = pd.DataFrame(all_metrics)
    all_metrics_frame.to_csv(
        out_dir / "exploratory_loop_q0_all_branch_worker_metrics.csv", index=False
    )
    all_overall = {
        "branches": len(all_rows),
        "loop_branches": int(all_labels.sum()),
        "workers": len(all_metrics_frame),
        "selected_model_brier": float(brier_score_loss(all_labels, all_prediction)),
        "failure_rate_only_brier": float(
            brier_score_loss(all_labels, all_rate_prediction)
        ),
        "noise_action_brier": float(
            brier_score_loss(all_labels, all_noise_prediction)
        ),
        "branch_weighted_brier_gain_vs_rate": float(
            all_rows["brier_gain_vs_rate"].mean()
        ),
        "branch_weighted_brier_gain_vs_noise_action": float(
            all_rows["brier_gain_vs_noise_action"].mean()
        ),
        "equal_worker_mean_brier_gain_vs_rate": float(
            all_metrics_frame["brier_gain_vs_rate"].mean()
        ),
        "equal_worker_mean_brier_gain_vs_noise_action": float(
            all_metrics_frame["brier_gain_vs_noise_action"].mean()
        ),
        "exact_worker_sign_flip_p_one_sided_vs_rate": exact_group_sign_flip_p(
            all_metrics_frame["brier_gain_vs_rate"].to_numpy()
        ),
        "exact_worker_sign_flip_p_one_sided_vs_noise_action": (
            exact_group_sign_flip_p(
                all_metrics_frame["brier_gain_vs_noise_action"].to_numpy()
            )
        ),
    }
    all_audit = {
        "schema": "himoe.exploratory_loop_q0_all_branch_audit.v1",
        "selection_status": "post_hoc_followup_of_failure_conditioned_loop_signal",
        "confirmatory": False,
        "target": "physical label_loop_or_cycling on all branches",
        "feature": "q0 state-token full HB probabilities, centered within K=16 snapshot",
        "cv": "leave_one_worker_out",
        "tail_features_used": False,
        "seed": args.seed,
        "overall": all_overall,
        "per_worker": all_metrics,
    }
    (out_dir / "exploratory_loop_q0_all_branch_audit.json").write_text(
        json.dumps(analysis.plain(all_audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"failure_conditioned": overall, "all_branches": all_overall},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
