"""Project frozen 10D kNN scores, with the original 2D ablation for comparison."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from knn import HERE, ROOT, METHODS, PRIMARY, ReferenceScorer, dynamics
from core import digest, write_json
from plot_knn import load, save

PCA_METHOD = "dyn_pca2_success_knn_k20"
ALPHA = .05
SCORE_NORM = LogNorm(vmin=.125, vmax=8)
DECISIONS = (
    ("neither", "Neither exceeds", "#788d9f", "o", 13),
    ("both", "Both exceed", "#b93636", "o", 23),
    ("10d_only", "10D only", "#d78614", "X", 42),
    ("2d_only", "2D only", "#7868a3", "+", 48),
)


def calibration_diagnostics(frame, predictions, profile, fold):
    success = predictions["calibration_labels"] == 0
    selected = frame.iloc[predictions["calibration_rows"]].loc[success]
    groups = (selected.task + "|" + selected.init_state_id.astype(str)).to_numpy()
    alpha_index = int(np.flatnonzero(np.isclose(profile["alphas"], ALPHA))[0])
    records = []
    for method in (PRIMARY, PCA_METHOD):
        mi = METHODS.index(method)
        scores = predictions["calibration_scores"][mi, success]
        points = scores[np.isfinite(scores)]
        peaks = np.where(np.isfinite(scores), scores, -np.inf).max(1)
        units = np.array([peaks[groups == key].max() for key in sorted(set(groups))])
        rank = int(np.ceil((len(units) + 1) * (1 - ALPHA)))
        threshold = float(profile["thresholds"][1, alpha_index, mi])
        np.testing.assert_equal(threshold, np.sort(units)[rank - 1])
        episode_rank = int(np.ceil((len(peaks) + 1) * (1 - ALPHA)))
        episode_threshold = float(profile["thresholds"][0, alpha_index, mi])
        np.testing.assert_equal(episode_threshold, np.sort(peaks)[episode_rank - 1])
        q95 = float(np.quantile(points, .95))
        records.append({
            "fold": fold, "suite": selected.suite.iloc[0], "method": method,
            "alpha": ALPHA, "success_episodes": len(peaks), "success_points": len(points),
            "success_task_init_groups": len(units), "group_rank": rank,
            "group_threshold": threshold, "episode_threshold": episode_threshold,
            "point_q50": float(np.median(points)), "point_q95": q95,
            "point_q99": float(np.quantile(points, .99)),
            "group_threshold_over_point_q95": threshold / q95,
            "point_fraction_below_group_threshold": float((points <= threshold).mean()),
            "episode_fraction_exceeding_group_threshold": float((peaks > threshold).mean()),
            "groups_exceeding_threshold": int((units > threshold).sum()),
        })
    return records


def collect_case(case, frame, cache, profile, predictions):
    fold = case["fold"]
    rows = np.asarray(case["test_global_rows"], dtype=int)
    queries = np.asarray(case["queries"], dtype=int)
    positions = pd.Index(predictions["test_rows"]).get_indexer(rows)
    assert (positions >= 0).all() and predictions["test_unseen"][positions].all()
    assert cache["valid"][rows, queries].all()
    assert (queries >= 7).all() and (queries < frame.iloc[rows].length.to_numpy()).all()
    np.testing.assert_array_equal(profile["methods"], METHODS)
    np.testing.assert_array_equal(predictions["methods"], METHODS)
    np.testing.assert_array_equal(profile["thresholds"], predictions["thresholds"])
    np.testing.assert_array_equal(profile["alphas"], predictions["alphas"])
    np.testing.assert_allclose(profile["pca_components"] @ profile["pca_components"].T,
                               np.eye(2), atol=1e-12)
    np.testing.assert_allclose(profile["success_pca2"],
        (profile["success_dynamic"] - profile["pca_mean"]) @ profile["pca_components"].T,
        atol=1e-12)

    unique, inverse = np.unique(rows, return_inverse=True)
    dynamic, _ = dynamics(cache["mobility"][unique], cache["acceleration"][unique],
                          cache["periodicity"][unique], float(profile["periodicity_scale"]))
    vectors = ((dynamic[inverse, queries].astype(np.float64) - profile["dynamic_center"])
               / profile["dynamic_scale"])
    xy = (vectors - profile["pca_mean"]) @ profile["pca_components"].T
    scorer = ReferenceScorer(profile)
    alpha_index = int(np.flatnonzero(np.isclose(profile["alphas"], ALPHA))[0])
    points = frame.iloc[rows][["suite", "task", "run_id", "episode", "init_state_id",
                              "length", "failure"]].reset_index(drop=True)
    points.insert(0, "fold", fold)
    points["global_row"] = rows
    points["query"] = queries
    points["pc1"], points["pc2"] = xy.T
    points["safe_progress_color"] = np.where(points.failure, queries / (points.length - 1), 0.)
    verification = {"fold": fold, "points": len(rows), "sample_identity_matches_original": True,
                    "all_sampled_points_are_unseen_test": True, "distance_checks": []}
    # Colors use sealed scores; freshly computed neighbors only verify their identities and values.
    for dimension, method, bank, values in (
            ("10d", PRIMARY, "success_dynamic", vectors),
            ("2d", PCA_METHOD, "success_pca2", xy)):
        mi = METHODS.index(method)
        scores = predictions["scores"][mi, positions, queries].astype(np.float64)
        threshold = float(profile["thresholds"][1, alpha_index, mi])
        recomputed = scorer.neighbors(bank, values)[0].mean(1)
        np.testing.assert_allclose(scores, recomputed, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(scores > threshold, recomputed > threshold)
        oracle_indices = np.unique(np.linspace(0, len(rows) - 1, 12).astype(int))
        for i in oracle_indices:
            distances = np.linalg.norm(profile[bank] - values[i], axis=1)
            direct = np.sort(distances)[:20].mean()
            np.testing.assert_allclose(scores[i], direct, rtol=2e-6, atol=2e-6)
        points[f"score_{dimension}"] = scores
        points[f"threshold_{dimension}"] = threshold
        points[f"ratio_{dimension}"] = scores / threshold
        points[f"exceeds_{dimension}"] = scores > threshold
        first = predictions["first"][1, alpha_index, mi, positions]
        full_scores = predictions["scores"][mi, positions]
        crossed = np.isfinite(full_scores) & (full_scores > threshold)
        expected_first = np.where(crossed.any(1), crossed.argmax(1), -1)
        np.testing.assert_array_equal(first, expected_first)
        points[f"first_alarm_query_{dimension}"] = first
        points[f"alarm_latched_{dimension}"] = (first >= 0) & (queries >= first)
        verification["distance_checks"].append({
            "dimension": dimension, "all_saved_scores_match": True,
            "max_abs_error": float(np.max(np.abs(scores - recomputed))),
            "all_current_decisions_and_first_alarms_match": True,
            "direct_distance_oracles": len(oracle_indices),
        })
    points["decision"] = np.select(
        [points.exceeds_10d & points.exceeds_2d, points.exceeds_10d, points.exceeds_2d],
        ["both", "10d_only", "2d_only"], default="neither")
    threshold = points.threshold_2d.iloc[0]
    np.testing.assert_equal(threshold, case["threshold"])
    reference = profile["success_pca2"]
    combined = np.concatenate((reference, xy))
    lower = np.minimum(combined.min(0), reference.min(0) - threshold * 1.05)
    upper = np.maximum(combined.max(0), reference.max(0) + threshold * 1.05)
    center, half = (upper + lower) / 2, float((upper - lower).max()) * .53
    gx, gy = np.meshgrid(np.linspace(center[0] - half, center[0] + half, 150),
                         np.linspace(center[1] - half, center[1] + half, 150))
    grid_scores = scorer.neighbors("success_pca2", np.column_stack((gx.ravel(), gy.ravel())))[0]
    grid_scores = grid_scores.mean(1).reshape(gx.shape)
    assert (grid_scores[[0, -1]] > threshold).all() and (grid_scores[:, [0, -1]] > threshold).all()
    assert grid_scores.min() < threshold < grid_scores.max()
    summary = {
        "fold": fold, "suite": case["suite"], "episodes": len(unique), "points": len(rows),
        "reference_points": len(reference),
        "pca_explained_variance": float(profile["pca_explained_variance_ratio"].sum()),
        "threshold_10d": float(points.threshold_10d.iloc[0]), "threshold_2d": float(threshold),
        "exceeds_10d": int(points.exceeds_10d.sum()), "exceeds_2d": int(points.exceeds_2d.sum()),
        **{f"points_{name}": int((points.decision == name).sum()) for name, *_ in DECISIONS},
        "display_ratio_underflow": int((points.ratio_10d < SCORE_NORM.vmin).sum()),
        "display_ratio_overflow": int((points.ratio_10d > SCORE_NORM.vmax).sum()),
    }
    return {"points": points, "xy": xy, "reference": reference, "summary": summary,
            "grid": (gx, gy, grid_scores), "verification": verification}


def projection_axes(axis, case):
    gx, gy, _ = case["grid"]
    axis.scatter(*case["reference"].T, s=3, c="#a8a8a8", alpha=.20,
                 linewidths=0, rasterized=True, zorder=1)
    axis.set(xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()),
             xlabel="Reference PC1", ylabel="Reference PC2")
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=9)
    axis.spines[["top", "right"]].set_visible(False)


def score_points(axis, case):
    points = case["points"].sort_values("ratio_10d", kind="stable")
    return axis.scatter(points.pc1, points.pc2, c=points.ratio_10d,
                        cmap="RdBu_r", norm=SCORE_NORM, s=17, edgecolors="#555555",
                        linewidths=.2, rasterized=True, zorder=2)


def score_colorbar(fig, artist, **kwargs):
    bar = fig.colorbar(artist, orientation="horizontal", extend="both", **kwargs)
    bar.set_ticks([.125, .25, .5, 1, 2, 4, 8], labels=["0.125", "0.25", "0.5", "1", "2", "4", "8"])
    bar.ax.axvline(1, color="#222222", lw=1.3)
    bar.ax.minorticks_off()
    bar.set_label("10D distance / threshold (1 = threshold; log scale)", fontsize=9)
    return bar


def plot_overview(cases, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11), layout="constrained")
    for axis, case in zip(axes.flat, cases):
        projection_axes(axis, case)
        artist = score_points(axis, case)
        summary = case["summary"]
        suite = summary["suite"].replace("libero_", "").capitalize()
        axis.set_title(f"{suite}: 10D score shown at its 2D position\n"
                       f"PCA variance {summary['pca_explained_variance']:.1%}; "
                       f"{summary['exceeds_10d']}/{summary['points']} sampled points exceed", fontsize=11)
    score_colorbar(fig, artist, ax=list(axes.flat), fraction=.035, pad=.035)
    fig.suptitle("10D kNN-20 scores projected onto frozen reference PCA\n"
                 "Distances computed in 10D; grey = successful A reference; no 10D contour inferred", fontsize=13)
    fig.supxlabel("Same failure-enriched unseen B sample as the original figure; split 20260907. "
                  "Point counts are not trajectory detection rates.", fontsize=9)
    save(fig, output / "figures/knn10_projected_scores")


def plot_comparison(case, output):
    fig = plt.figure(figsize=(16.8, 6.8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, .055])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    keys = [fig.add_subplot(grid[1, i]) for i in range(3)]
    for axis in axes:
        projection_axes(axis, case)
    points, summary = case["points"], case["summary"]
    order = points.sort_values("safe_progress_color", kind="stable")
    progress = axes[0].scatter(order.pc1, order.pc2, c=order.safe_progress_color,
                              cmap="coolwarm", vmin=0, vmax=1, s=16, alpha=.85,
                              linewidths=0, rasterized=True, zorder=2)
    axes[0].contour(*case["grid"], levels=[summary["threshold_2d"]],
                    colors=["#176b54"], linewidths=1.2)
    axes[0].set_title(f"Original 2D kNN boundary\n2D threshold = {summary['threshold_2d']:.3f}", fontsize=12)
    bar = fig.colorbar(progress, cax=keys[0], orientation="horizontal")
    bar.set_ticks([0, .5, 1], labels=["0", "0.5", "1"])
    bar.set_label("Failed-rollout progress; success stays blue (0)", fontsize=9)
    artist = score_points(axes[1], case)
    axes[1].set_title(f"Actual 10D kNN score\n10D threshold = {summary['threshold_10d']:.3f}", fontsize=12)
    score_colorbar(fig, artist, cax=keys[1])
    handles = []
    for name, label, color, marker, size in DECISIONS:
        subset = points.loc[points.decision == name]
        axes[2].scatter(subset.pc1, subset.pc2, c=color, marker=marker, s=size,
                        alpha=.85, linewidths=.9 if marker == "+" else 0,
                        rasterized=True, zorder=3 if name.endswith("only") else 2)
        handles.append(Line2D([], [], marker=marker, color=color, ls="none", markersize=6,
                              label=f"{label}: {len(subset)}"))
    keys[2].set_axis_off()
    keys[2].legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=9)
    axes[2].set_title("Current threshold decisions\nOrange X: 10D exceeds, 2D does not", fontsize=12)
    suite = summary["suite"].replace("libero_", "").capitalize()
    fig.suptitle(f"{suite}: identical points and PCA coordinates, different scoring dimensions\n"
                 f"{summary['episodes']} unseen B trajectories, {summary['points']} sampled queries; "
                 f"PCA retains {summary['pca_explained_variance']:.1%} variance", fontsize=13)
    fig.supxlabel("Green line belongs only to the 2D method. Decisions use current scores, "
                  "not latched alarms. Labels only affect illustration sampling and progress color.", fontsize=9)
    save(fig, output / "figures" / f"{summary['suite']}_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round5_knn")
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round3_safe")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source, parent = args.input.resolve(), args.parent.resolve()
    output = args.output.resolve() if args.output else source / "projection10"
    sealed = json.loads((source / "sealed_manifest.json").read_text())
    illustration = json.loads((source / "pca2_illustration.json").read_text())
    inputs = {}
    for name, expected in sealed["sources"].items():
        assert digest(ROOT / name) == expected, name
        inputs[str(ROOT / name)] = expected
    cache_path = parent / "v7/v7_inputs.npz"
    cache_sha = digest(cache_path)
    assert cache_sha == sealed["inputs"][str(cache_path.relative_to(ROOT))]
    inputs[str(cache_path)] = cache_sha
    for name in ("sealed_manifest.json", "pca2_illustration.json", "outcome_alignment.csv"):
        inputs[str(source / name)] = digest(source / name)
    frame = pd.read_csv(source / "outcome_alignment.csv")
    index = pd.read_csv(source / "index.csv")
    assert digest(source / "index.csv") == sealed["artifacts"]["index.csv"]
    pd.testing.assert_frame_equal(frame[index.columns], index)
    cache = load(cache_path)
    cases, diagnostics = [], []
    for original in illustration["cases"]:
        fold = original["fold"]
        loaded = []
        for kind in ("profiles", "predictions"):
            relative = f"{kind}/{fold}.npz"
            path = source / relative
            actual = digest(path)
            assert actual == sealed["artifacts"][relative], relative
            inputs[str(path)] = actual
            loaded.append(load(path))
        profile, predictions = loaded
        case = collect_case(original, frame, cache, profile, predictions)
        cases.append(case)
        diagnostics.extend(calibration_diagnostics(frame, predictions, profile, fold))
        print(json.dumps(case["summary"]), flush=True)
    output.mkdir(exist_ok=False)
    (output / "figures").mkdir()
    pd.concat([case["points"] for case in cases], ignore_index=True).to_csv(output / "projected_points.csv", index=False)
    pd.DataFrame([case["summary"] for case in cases]).to_csv(output / "projection_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "calibration_diagnostics.csv", index=False)
    plot_overview(cases, output)
    for case in cases:
        plot_comparison(case, output)
    write_json(output / "verification.json", {
        "cases": [case["verification"] for case in cases],
        "score_pairs_checked": int(sum(2 * len(case["points"]) for case in cases)),
        "calibration_thresholds_recomputed": 2 * len(diagnostics),
        "display_coordinates": "Existing A-mixture-reference PCA; no new fit",
        "display_sampling": "Identical to pca2_illustration.json; final-outcome enriched",
        "scoring": "Sealed 10D and 2D kNN-20 scores; unchanged task/init alpha=0.05 thresholds",
        "decision_semantics": "Current score > threshold; latched alarms saved separately in CSV",
        "exact_10d_boundary_drawn_in_2d": False,
        "new_model_training": False, "test_label_threshold_tuning": False,
    })
    write_json(output / "manifest.json", {
        "plotter_sha256": digest(Path(__file__)), "shared_plot_helpers_sha256": digest(HERE / "plot_knn.py"),
        "inputs": inputs,
        "artifacts": {str(path.relative_to(output)): digest(path)
                      for path in sorted(output.rglob("*")) if path.is_file()},
    })
    print(output, flush=True)


if __name__ == "__main__":
    main()
