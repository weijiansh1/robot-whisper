"""Plot operating points and the frozen reference-PCA kNN decision regions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from knn import HERE, METHODS, PRIMARY, ReferenceScorer, dynamics
from core import digest, write_json

LABELS = {"dyn_radius": "Dynamic radius", "dyn_radius_outward": "Radius change from q7",
    "dyn_success_knn_k1": "Dynamic success kNN-1", "dyn_success_knn_k5": "Dynamic success kNN-5",
    PRIMARY: "Dynamic success kNN-20", "dyn_success_knn_k20_persist3": "Dynamic kNN-20, persistent",
    "dyn_vote_knn_k20": "Dynamic failure vote kNN-20", "dyn_pca2_success_knn_k20": "PCA-2D success kNN-20",
    "route_success_knn_k20": "Routing success kNN-20", "eef_motion_low": "Low EEF motion (from q7)",
    "clock_q7": "Clock (from q7)", "success_only__cumsum": "Earlier statistics distance, sum",
    "v7_guard_constant": "Earlier v7 continuous guard"}


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def comparison(output):
    ranks = pd.read_csv(output / "ranking_metrics.csv")
    alarms = pd.read_csv(output / "alarm_metrics.csv")
    common = ranks.loc[(ranks.scope == "unseen") & (ranks.view == "common_horizon")]
    summary = common.groupby("method")[["task_macro_auc", "within_init_macro_auc", "scorable_fraction"]].mean()
    summary.to_csv(output / "common_horizon_summary.csv")
    grouped = alarms.loc[(alarms.scope == "unseen") & (alarms.calibration == "task_init")]
    means = grouped.groupby(["method", "alpha"])[["recall", "fpr", "balanced_accuracy", "t_det"]].mean().reset_index()
    means.to_csv(output / "unseen_operating_points.csv", index=False)
    methods = list(METHODS) + ["success_only__cumsum", "v7_guard_constant"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
    colors = ["#c34836" if name == PRIMARY else "#24816f" if name == "eef_motion_low" else "#637e99" for name in methods]
    axes[0].barh(np.arange(len(methods)), summary.loc[methods].task_macro_auc, color=colors, height=.65)
    axes[0].set_yticks(np.arange(len(methods)), [LABELS[name] for name in methods], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].axvline(.5, color="#777777", ls="--", lw=.8)
    axes[0].set(xlim=(0, 1), xlabel="Task-macro AUROC", title="Same-task common observation horizon")
    for i, name in enumerate(methods):
        axes[0].text(summary.loc[name].task_macro_auc + .014, i, f"{summary.loc[name].task_macro_auc:.3f}", va="center", fontsize=8)
    selected = [(PRIMARY, "#c34836"), ("dyn_radius", "#4377a3"), ("dyn_pca2_success_knn_k20", "#8a6696"),
                ("dyn_vote_knn_k20", "#a17b27"), ("eef_motion_low", "#24816f")]
    xmax = .1
    for method, color in selected:
        group = means.loc[means.method == method].sort_values("alpha")
        axes[1].plot(group.fpr * 100, group.recall * 100, ".-", color=color, lw=1.2, label=LABELS[method])
        primary = group.loc[np.isclose(group.alpha, .05)]
        axes[1].scatter(primary.fpr * 100, primary.recall * 100, s=70, color=color, edgecolor="white", zorder=4)
        xmax = max(xmax, group.fpr.max())
    for method, label, color in (("v7_unlabeled_reference_budget", "Original v7 reference budget 5%", "#333333"),
                                  ("v7_success_calibration_budget", "Original v7 success budget 5%", "#777777")):
        point = alarms.loc[(alarms.scope == "unseen") & (alarms.method == method) & np.isclose(alarms.alpha, .05)]
        axes[1].scatter(point.fpr.mean() * 100, point.recall.mean() * 100, color=color, marker="*", s=110, label=label, zorder=5)
    axes[1].set(xlim=(0, (xmax + .02) * 100), ylim=(0, 103), xlabel="Actual false-positive rate (%)",
                ylabel="Failure recall (%)", title="Trajectory alarms; task/init calibration")
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Unseen tasks: equal-weight mean over 12 suite/split settings\n"
                 "Large circles: nominal 5% group calibration; stars: distinct historical v7 budget rules", fontsize=12)
    save(fig, output / "figures/method_comparison")


def pca_regions(output, parent):
    frame = pd.read_csv(output / "outcome_alignment.csv")
    cache = load(parent / "v7/v7_inputs.npz")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    rng = np.random.default_rng(20260907)
    records = []
    for axis, suite in zip(axes.flat, sorted(frame.suite.unique())):
        fold = f"{suite}_20260907"
        profile = load(output / "profiles" / f"{fold}.npz")
        predictions = load(output / "predictions" / f"{fold}.npz")
        part = frame.iloc[predictions["test_rows"][predictions["test_unseen"]]].copy()
        chosen = []
        for _, group in part.groupby(["task", "failure"]):
            chosen.extend(rng.choice(group.index, min(10, len(group)), replace=False).tolist())
        rows, queries = [], []
        for row in chosen:
            available = np.arange(7, int(frame.loc[row, "length"]))
            if not len(available):
                continue
            sampled = available[np.linspace(0, len(available) - 1, min(8, len(available))).astype(int)]
            rows.extend([row] * len(sampled))
            queries.extend(sampled.tolist())
        rows, queries = np.asarray(rows), np.asarray(queries)
        unique, inverse = np.unique(rows, return_inverse=True)
        d, _ = dynamics(cache["mobility"][unique], cache["acceleration"][unique], cache["periodicity"][unique],
                        float(profile["periodicity_scale"]))
        vectors = (d[inverse, queries].astype(np.float64) - profile["dynamic_center"]) / profile["dynamic_scale"]
        xy = (vectors - profile["pca_mean"]) @ profile["pca_components"].T
        reference = profile["success_pca2"]
        method = METHODS.index("dyn_pca2_success_knn_k20")
        alpha_index = int(np.flatnonzero(np.isclose(profile["alphas"], .05))[0])
        threshold = float(profile["thresholds"][1, alpha_index, method])
        combined = np.concatenate((reference, xy))
        lower = np.minimum(combined.min(0), reference.min(0) - threshold * 1.05)
        upper = np.maximum(combined.max(0), reference.max(0) + threshold * 1.05)
        center = (upper + lower) / 2
        half = float((upper - lower).max()) * .53
        x = np.linspace(center[0] - half, center[0] + half, 150)
        y = np.linspace(center[1] - half, center[1] + half, 150)
        gx, gy = np.meshgrid(x, y)
        grid = np.column_stack((gx.ravel(), gy.ravel()))
        score = ReferenceScorer(profile).neighbors("success_pca2", grid)[0].mean(1).reshape(gx.shape)
        assert (score[[0, -1]] > threshold).all() and (score[:, [0, -1]] > threshold).all()
        has_boundary = bool(score.min() < threshold < score.max())
        if has_boundary:
            axis.contour(gx, gy, score, levels=[threshold], colors=["#176b54"], linewidths=1.1)
        axis.scatter(*reference.T, color="#a5a5a5", s=3, alpha=.18, linewidths=0, rasterized=True)
        color = np.where(frame.iloc[rows].failure, queries / (frame.iloc[rows].length.to_numpy() - 1), 0.)
        artist = axis.scatter(*xy.T, c=color, cmap="coolwarm", vmin=0, vmax=1, s=9, alpha=.75, linewidths=0, rasterized=True)
        axis.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), xlabel="Reference PC1", ylabel="Reference PC2",
                 title=suite.replace("libero_", "").capitalize())
        axis.set_aspect("equal", adjustable="box")
        records.append({"suite": suite, "fold": fold, "episodes": len(np.unique(rows)), "points": len(rows),
            "reference_points": len(reference), "threshold": threshold, "boundary_visible": has_boundary,
            "test_global_rows": rows.tolist(), "queries": queries.tolist()})
    legend = [Line2D([], [], color="#a5a5a5", marker=".", ls="none", label="Successful A reference"),
              Line2D([], [], color="#176b54", lw=1.1, label="PCA-2D kNN-20 calibrated boundary")]
    axes[0, 0].legend(handles=legend, loc="upper left", fontsize=7, frameon=False)
    fig.colorbar(artist, ax=list(axes.flat), orientation="horizontal", fraction=.04, pad=.03,
                 label="Failed-rollout progress; successful rollouts stay at 0 (blue)")
    fig.suptitle("Frozen reference PCA with a train-free 2D kNN distance boundary\n"
                 "Unseen B trajectories, failure-enriched illustration; this is the 2D ablation, not the 10D primary method", fontsize=12)
    save(fig, output / "figures/pca2_knn_regions")
    write_json(output / "pca2_illustration.json", {"cases": records, "labels_used_only_for_display_sampling_and_color": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round5_knn")
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    output = args.input.resolve()
    evaluation = json.loads((output / "evaluation_summary.json").read_text())
    for name, expected in evaluation["artifacts"].items():
        assert digest(output / name) == expected
    (output / "figures").mkdir(exist_ok=True)
    comparison(output)
    pca_regions(output, args.parent.resolve())
    write_json(output / "figure_manifest.json", {"plotter_sha256": digest(Path(__file__)),
        "artifacts": {str(p.relative_to(output)): digest(p) for p in sorted((output / "figures").iterdir())
                      if p.is_file() and p.suffix in (".png", ".pdf")}})
    print(output / "figures", flush=True)


if __name__ == "__main__":
    main()
