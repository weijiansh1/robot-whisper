#!/usr/bin/env python3
"""Plot failed and successful rollout MoE signals over single chunks.

Every score at query k uses only HB-MoE computation from query k. Two
per-query scores reproduce the existing cross-fitted analysis. A third score
fits a routed-MoE failure axis only at k=8, freezes it, and projects each
earlier query onto that same axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze_moe_rollout_trend import (
    LOGISTIC_C,
    N_CHUNKS,
    OUT_DIR,
    PCA_COMPONENTS,
    PROJECTED_DIM,
    TrendTask,
    _load_cached,
    crossfit_query,
    evaluation_splits,
    feature_blocks,
)


ANCHOR_CHUNK = 8
BOOTSTRAPS = 5000
GROUPS = ("success", "failure")
SIGNALS = (
    "per_query_risk_percentile",
    "per_query_success_manifold_percentile",
    "frozen_k8_failure_axis",
)


@dataclass(frozen=True)
class BlockReducer:
    keep: np.ndarray
    scaler: StandardScaler
    pca: PCA | None

    def transform(self, values: np.ndarray) -> np.ndarray:
        reduced = self.scaler.transform(
            np.asarray(values, dtype=np.float64)[:, self.keep]
        )
        return self.pca.transform(reduced) if self.pca is not None else reduced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def load_tasks(out_dir: pathlib.Path) -> list[TrendTask]:
    tasks = []
    for path in sorted((out_dir / "compact").glob("*.npz")):
        task = _load_cached(path, N_CHUNKS, PROJECTED_DIM)
        if task is None:
            raise ValueError("incompatible compact cache: %s" % path)
        tasks.append(task)
    if len(tasks) != 4:
        raise ValueError("expected four compact task caches, found %d" % len(tasks))
    return tasks


def fit_reducer(values: np.ndarray, seed: int) -> tuple[BlockReducer, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    keep = values.std(axis=0) > 1e-9
    if not np.any(keep):
        raise ValueError("feature block is constant")
    scaler = StandardScaler().fit(values[:, keep])
    reduced = scaler.transform(values[:, keep])
    count = min(PCA_COMPONENTS, reduced.shape[1], reduced.shape[0] - 1)
    pca = None
    if count < reduced.shape[1]:
        pca = PCA(
            n_components=count, svd_solver="randomized", random_state=seed
        ).fit(reduced)
        reduced = pca.transform(reduced)
    return BlockReducer(keep=keep, scaler=scaler, pca=pca), reduced


def frozen_anchor_scores(
    data: TrendTask, seed: int, anchor_probability: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cross-fit one k8 routed-full axis and apply it unchanged to every k."""
    splits, strata, onehot = evaluation_splits(data, "seed_heldout")
    labels = data.failure.astype(np.int8)
    output = np.full((len(labels), N_CHUNKS), np.nan, dtype=np.float64)
    anchor_reproduction = np.full(len(labels), np.nan, dtype=np.float64)
    blocks = []
    for chunk in range(N_CHUNKS):
        current = feature_blocks(data, chunk)
        blocks.append({name: current[name] for name in ("routed", "scalar")})

    for fold, (train, test) in enumerate(splits):
        reducers = {}
        train_parts = [onehot[train]]
        for name, original_axis in (("routed", 0), ("scalar", 3)):
            reducer, reduced = fit_reducer(
                blocks[ANCHOR_CHUNK][name][train],
                seed + ANCHOR_CHUNK * 1000 + fold * 10 + original_axis,
            )
            reducers[name] = reducer
            train_parts.append(reduced)
        x_train = np.column_stack(train_parts)
        family_scaler = StandardScaler().fit(x_train)
        x_train = family_scaler.transform(x_train)
        model = LogisticRegression(
            C=LOGISTIC_C,
            solver="lbfgs",
            max_iter=3000,
            class_weight="balanced",
        ).fit(x_train, labels[train])
        train_decision = model.decision_function(x_train)

        test_indices = np.flatnonzero(test)
        train_indices = np.flatnonzero(train)
        scene_reference = {}
        for scene in np.unique(data.scenes):
            local = data.scenes[train_indices] == scene
            reference = train_decision[local]
            scene_reference[int(scene)] = (
                float(reference.mean()),
                max(float(reference.std()), 1e-8),
            )

        for chunk in range(N_CHUNKS):
            test_parts = [onehot[test]]
            for name in ("routed", "scalar"):
                test_parts.append(reducers[name].transform(blocks[chunk][name][test]))
            x_test = family_scaler.transform(np.column_stack(test_parts))
            decision = model.decision_function(x_test)
            for scene in np.unique(data.scenes[test_indices]):
                local = data.scenes[test_indices] == scene
                center, scale = scene_reference[int(scene)]
                output[test_indices[local], chunk] = (decision[local] - center) / scale
            if chunk == ANCHOR_CHUNK:
                anchor_reproduction[test] = model.predict_proba(x_test)[:, 1]

    if np.any(~np.isfinite(output)) or np.any(~np.isfinite(anchor_reproduction)):
        raise RuntimeError("frozen-axis cross-fit output is incomplete")
    maximum_difference = float(np.max(np.abs(anchor_reproduction - anchor_probability)))
    return output, strata, maximum_difference


def scene_group_values(
    labels: np.ndarray,
    scores: np.ndarray,
    scenes: np.ndarray,
    strata: np.ndarray,
    percentile: bool,
) -> np.ndarray:
    """Return equal-stratum success/failure means for each initial scene."""
    labels = np.asarray(labels, dtype=bool)
    unique_scenes = np.unique(scenes)
    result = np.full((len(unique_scenes), 2), np.nan, dtype=np.float64)
    for scene_axis, scene in enumerate(unique_scenes):
        strata_values = []
        scene_mask = scenes == scene
        for stratum in np.unique(strata[scene_mask]):
            mask = scene_mask & (strata == stratum) & np.isfinite(scores)
            if not np.any(mask & labels) or not np.any(mask & ~labels):
                continue
            values = np.asarray(scores[mask], dtype=np.float64)
            if percentile:
                values = (rankdata(values, method="average") - 0.5) / len(values)
            local_labels = labels[mask]
            strata_values.append(
                [values[~local_labels].mean(), values[local_labels].mean()]
            )
        if strata_values:
            result[scene_axis] = np.mean(strata_values, axis=0)
    return result


def calculate_task_curves(
    data: TrendTask, seed: int
) -> tuple[dict[str, np.ndarray], dict]:
    signal_scene = {
        signal: np.full((N_CHUNKS, len(np.unique(data.scenes)), 2), np.nan)
        for signal in SIGNALS
    }
    anchor_probability = None
    common_strata = None
    for chunk in range(N_CHUNKS):
        predictions, manifold, strata = crossfit_query(
            data, chunk, "seed_heldout", seed
        )
        if common_strata is None:
            common_strata = strata
        elif not np.array_equal(common_strata, strata):
            raise RuntimeError("conditional strata changed across queries")
        risk = predictions["routed_full"]
        signal_scene["per_query_risk_percentile"][chunk] = scene_group_values(
            data.failure, risk, data.scenes, strata, percentile=True
        )
        signal_scene["per_query_success_manifold_percentile"][chunk] = (
            scene_group_values(
                data.failure,
                manifold["routed_full"],
                data.scenes,
                strata,
                percentile=True,
            )
        )
        if chunk == ANCHOR_CHUNK:
            anchor_probability = risk
        print("%s: group curves query %d" % (data.task, chunk), flush=True)

    if anchor_probability is None or common_strata is None:
        raise RuntimeError("anchor query was not evaluated")
    frozen, frozen_strata, maximum_difference = frozen_anchor_scores(
        data, seed, anchor_probability
    )
    if not np.array_equal(common_strata, frozen_strata):
        raise RuntimeError("frozen-axis strata differ from per-query strata")
    for chunk in range(N_CHUNKS):
        signal_scene["frozen_k8_failure_axis"][chunk] = scene_group_values(
            data.failure,
            frozen[:, chunk],
            data.scenes,
            frozen_strata,
            percentile=False,
        )
    return signal_scene, {"anchor_probability_max_abs_diff": maximum_difference}


def summarize_curves(
    tasks: list[TrendTask],
    values: dict[tuple[str, str], np.ndarray],
    bootstrap: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    curve_rows = []
    gap_rows = []
    task_rows = []
    for signal in SIGNALS:
        task_points = []
        task_boots = []
        for task in tasks:
            matrix = values[(task.task, signal)]
            valid = np.flatnonzero(np.all(np.isfinite(matrix), axis=(0, 2)))
            if not len(valid):
                raise ValueError("no complete mixed scenes for %s %s" % (task.task, signal))
            point = matrix[:, valid].mean(axis=1)
            draws = rng.integers(0, len(valid), size=(bootstrap, len(valid)))
            sampled = matrix[:, valid][:, draws].mean(axis=2).transpose(1, 0, 2)
            task_points.append(point)
            task_boots.append(sampled)
            for chunk in range(N_CHUNKS):
                task_rows.append(
                    {
                        "task": task.task,
                        "signal": signal,
                        "chunk": chunk,
                        "success": float(point[chunk, 0]),
                        "failure": float(point[chunk, 1]),
                        "failure_minus_success": float(
                            point[chunk, 1] - point[chunk, 0]
                        ),
                        "mixed_scenes": int(len(valid)),
                    }
                )
        macro = np.mean(task_points, axis=0)
        macro_boot = np.mean(task_boots, axis=0)
        for chunk in range(N_CHUNKS):
            for group_axis, group in enumerate(GROUPS):
                distribution = macro_boot[:, chunk, group_axis]
                curve_rows.append(
                    {
                        "signal": signal,
                        "chunk": chunk,
                        "group": group,
                        "mean": float(macro[chunk, group_axis]),
                        "ci_low": float(np.percentile(distribution, 2.5)),
                        "ci_high": float(np.percentile(distribution, 97.5)),
                    }
                )
            gap_distribution = macro_boot[:, chunk, 1] - macro_boot[:, chunk, 0]
            gap_rows.append(
                {
                    "signal": signal,
                    "chunk": chunk,
                    "failure_minus_success": float(macro[chunk, 1] - macro[chunk, 0]),
                    "ci_low": float(np.percentile(gap_distribution, 2.5)),
                    "ci_high": float(np.percentile(gap_distribution, 97.5)),
                }
            )
    return curve_rows, gap_rows, task_rows


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(summary: dict, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    curves = {
        (row["signal"], row["chunk"], row["group"]): row
        for row in summary["curves"]
    }
    gaps = {(row["signal"], row["chunk"]): row for row in summary["gaps"]}
    x = np.arange(N_CHUNKS)
    colors = {"success": "#2A9D8F", "failure": "#C44536"}
    titles = {
        "per_query_risk_percentile": "Cross-fitted MoE risk percentile",
        "per_query_success_manifold_percentile": "Success-manifold anomaly percentile",
        "frozen_k8_failure_axis": "Frozen k8 MoE failure axis",
    }
    ylabels = {
        "per_query_risk_percentile": "within-state percentile",
        "per_query_success_manifold_percentile": "within-state percentile",
        "frozen_k8_failure_axis": "standardized axis score",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for axis, signal in zip(axes[:3], SIGNALS):
        for group in GROUPS:
            rows = [curves[(signal, int(chunk), group)] for chunk in x]
            mean = np.asarray([row["mean"] for row in rows])
            low = np.asarray([row["ci_low"] for row in rows])
            high = np.asarray([row["ci_high"] for row in rows])
            axis.plot(x, mean, marker="o", color=colors[group], label=group)
            axis.fill_between(x, low, high, color=colors[group], alpha=0.16)
        axis.axhline(
            0.5 if signal != "frozen_k8_failure_axis" else 0.0,
            color="black",
            linestyle="--",
            linewidth=1,
        )
        axis.set(
            title=titles[signal],
            xlabel="control query k",
            ylabel=ylabels[signal],
            xticks=x,
        )
        axis.legend(frameon=False)

    gap_colors = {
        "per_query_risk_percentile": "#C44536",
        "per_query_success_manifold_percentile": "#E09F3E",
    }
    gap_labels = {
        "per_query_risk_percentile": "per-query risk",
        "per_query_success_manifold_percentile": "success-manifold anomaly",
    }
    for signal in SIGNALS[:2]:
        rows = [gaps[(signal, int(chunk))] for chunk in x]
        mean = np.asarray([row["failure_minus_success"] for row in rows])
        low = np.asarray([row["ci_low"] for row in rows])
        high = np.asarray([row["ci_high"] for row in rows])
        axes[3].plot(
            x, mean, marker="o", color=gap_colors[signal], label=gap_labels[signal]
        )
        axes[3].fill_between(x, low, high, color=gap_colors[signal], alpha=0.12)
    axes[3].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[3].set(
        title="Percentile gap: failure minus success",
        xlabel="control query k",
        ylabel="percentile gap",
        xticks=x,
    )
    axes[3].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_task_plot(summary: dict, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    rows = {
        (row["task"], row["chunk"]): row
        for row in summary["by_task"]
        if row["signal"] == "per_query_success_manifold_percentile"
    }
    x = np.arange(N_CHUNKS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, task in zip(axes.ravel(), summary["tasks"]):
        axis.plot(
            x,
            [rows[(task, int(chunk))]["success"] for chunk in x],
            marker="o",
            color="#2A9D8F",
            label="success",
        )
        axis.plot(
            x,
            [rows[(task, int(chunk))]["failure"] for chunk in x],
            marker="o",
            color="#C44536",
            label="failure",
        )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set(
            title=task.split("/", 1)[1].replace("_", " "),
            xlabel="control query k",
            ylabel="success-manifold anomaly percentile",
            xticks=x,
        )
        axis.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_report(summary: dict) -> str:
    curves = {
        (row["signal"], row["chunk"], row["group"]): row
        for row in summary["curves"]
    }
    gaps = {(row["signal"], row["chunk"]): row for row in summary["gaps"]}
    lines = [
        "# Failed versus successful rollout MoE curves",
        "",
        "Each point reads only the current action chunk. Shaded intervals in the plot "
        "bootstrap initial states within task and average the four tasks equally.",
        "",
        "## Bottom line",
        "",
        "The clearest growing comparison is distance from the successful MoE manifold: "
        "the failure-minus-success percentile gap is not distinguishable from zero "
        "through k4, becomes +%.3f at k5 (95%% CI [%+.3f, %+.3f]), and grows to "
        "+%.3f at k8 (95%% CI [%+.3f, %+.3f])."
        % (
            gaps[(SIGNALS[1], 5)]["failure_minus_success"],
            gaps[(SIGNALS[1], 5)]["ci_low"],
            gaps[(SIGNALS[1], 5)]["ci_high"],
            gaps[(SIGNALS[1], 8)]["failure_minus_success"],
            gaps[(SIGNALS[1], 8)]["ci_low"],
            gaps[(SIGNALS[1], 8)]["ci_high"],
        ),
        "",
        "A decoder direction frozen at k8 does not separate the groups through k6 "
        "(gap %+.3f, 95%% CI [%+.3f, %+.3f]). It separates only at k7 and k8, "
        "reaching %+.3f at k8 (95%% CI [%+.3f, %+.3f]). Thus anomaly relative "
        "to the appropriate success state accumulates, but one fixed late-failure "
        "direction is not already growing in the early chunks."
        % (
            gaps[(SIGNALS[2], 6)]["failure_minus_success"],
            gaps[(SIGNALS[2], 6)]["ci_low"],
            gaps[(SIGNALS[2], 6)]["ci_high"],
            gaps[(SIGNALS[2], 8)]["failure_minus_success"],
            gaps[(SIGNALS[2], 8)]["ci_low"],
            gaps[(SIGNALS[2], 8)]["ci_high"],
        ),
        "",
        "## Population curves",
        "",
        "| k | risk success | risk failure | risk gap | manifold success | manifold failure | manifold gap | frozen-axis success | frozen-axis failure | frozen gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for chunk in range(N_CHUNKS):
        risk_success = curves[(SIGNALS[0], chunk, "success")]["mean"]
        risk_failure = curves[(SIGNALS[0], chunk, "failure")]["mean"]
        manifold_success = curves[(SIGNALS[1], chunk, "success")]["mean"]
        manifold_failure = curves[(SIGNALS[1], chunk, "failure")]["mean"]
        frozen_success = curves[(SIGNALS[2], chunk, "success")]["mean"]
        frozen_failure = curves[(SIGNALS[2], chunk, "failure")]["mean"]
        lines.append(
            "| %d | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f | %+.3f | %+.3f | %+.3f |"
            % (
                chunk,
                risk_success,
                risk_failure,
                gaps[(SIGNALS[0], chunk)]["failure_minus_success"],
                manifold_success,
                manifold_failure,
                gaps[(SIGNALS[1], chunk)]["failure_minus_success"],
                frozen_success,
                frozen_failure,
                gaps[(SIGNALS[2], chunk)]["failure_minus_success"],
            )
        )
    lines.extend(
        [
            "",
            "## Reading the signals",
            "",
            "- Per-query risk percentile uses an independently cross-fitted routed-full decoder at each k, then ranks scores only within the same initial state and held-out seed fold.",
            "- Success-manifold anomaly is label-free at scoring time: larger means farther from the successful routed-computation centroid learned from training seeds at that k.",
            "- Frozen k8 failure axis fits a routed-full decoder at k8 inside each task and fold, then applies that exact preprocessing and direction to every earlier chunk. It tests whether the same within-task late-failure direction grows over time.",
            "- Percentile curves show relative separation, not raw MoE magnitude. The frozen-axis panel retains one common score direction but remains an associative offline readout.",
            "",
            "The k8 frozen-axis implementation reproduces the ordinary cross-fitted k8 routed-full probabilities with maximum absolute difference %.2g."
            % max(
                row["anchor_probability_max_abs_diff"]
                for row in summary["validation"].values()
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary_path = args.out_dir / "success_failure_curves.json"
    if args.render_only:
        summary = json.loads(summary_path.read_text())
        (args.out_dir / "success_failure_curves.md").write_text(
            render_report(summary)
        )
        make_plot(summary, args.out_dir / "moe_success_failure_curves.png")
        make_task_plot(summary, args.out_dir / "moe_success_failure_by_task.png")
        print("regenerated direct-curve artifacts in %s" % args.out_dir, flush=True)
        return
    tasks = load_tasks(args.out_dir)
    values = {}
    validation = {}
    for task_axis, task in enumerate(tasks):
        curves, audit = calculate_task_curves(
            task, args.seed + task_axis * 10000
        )
        validation[task.task] = audit
        for signal, matrix in curves.items():
            values[(task.task, signal)] = matrix
    curve_rows, gap_rows, task_rows = summarize_curves(
        tasks, values, args.bootstrap, args.seed + 700000
    )
    summary = {
        "analysis": "failed versus successful single-chunk MoE curves",
        "tasks": [task.task for task in tasks],
        "chunks": N_CHUNKS,
        "bootstrap": args.bootstrap,
        "anchor_chunk": ANCHOR_CHUNK,
        "curves": curve_rows,
        "gaps": gap_rows,
        "by_task": task_rows,
        "validation": validation,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    write_csv(args.out_dir / "success_failure_curves.csv", curve_rows)
    write_csv(args.out_dir / "success_failure_gaps.csv", gap_rows)
    write_csv(args.out_dir / "success_failure_curves_by_task.csv", task_rows)
    (args.out_dir / "success_failure_curves.md").write_text(render_report(summary))
    make_plot(summary, args.out_dir / "moe_success_failure_curves.png")
    make_task_plot(summary, args.out_dir / "moe_success_failure_by_task.png")
    print("wrote direct success/failure curves to %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
