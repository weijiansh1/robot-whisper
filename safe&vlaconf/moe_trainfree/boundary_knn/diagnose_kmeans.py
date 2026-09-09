"""Explain the frozen K-means results without refitting or selecting thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, average_precision_score, roc_auc_score

from kmeans_reference import OUTPUT, load_npz
from core import conformal_threshold, digest, trajectory_peak, write_json

SELECTED = ("knn20", "norm_only", "c1_centroid_distance", "c4_centroid_distance",
            "c4_assigned_radius", "c8_centroid_distance", "c32_centroid_distance", "c32_union_radius")
QUANTILES = (.1, .25, .5, .75, .9)


def quantiles(values, prefix):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    result = np.quantile(finite, QUANTILES) if len(finite) else np.full(len(QUANTILES), np.nan)
    return {f"{prefix}_q{round(q * 100):02}": float(value) for q, value in zip(QUANTILES, result)}


def diagnostic_figures(output, frame, effects, ratios):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained",
                             gridspec_kw={"width_ratios": (1.25, 1, 1)})
    tasks = {
        "libero_goal/put_the_wine_bottle_on_the_rack": "Wine bottle to rack",
        "libero_goal/open_the_middle_drawer_of_the_cabinet": "Open middle drawer",
        "libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate": "Bowl on cabinet to plate",
        "libero_goal/push_the_plate_to_the_front_of_the_stove": "Push plate to stove front",
        "libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket": "BBQ sauce to basket",
        "libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate": "Bowl in drawer to plate",
    }
    part = effects.set_index("task").loc[list(tasks)]
    bars = axes[0].barh(np.arange(len(part)), part.fp_removed,
                       color=np.where(part.fp_removed >= 0, "#168579", "#bb594d"))
    axes[0].set_yticks(np.arange(len(part)), list(tasks.values()))
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#555555", linewidth=.8)
    axes[0].bar_label(bars, padding=4, fontsize=9)
    axes[0].set_xlim(-110, 730)
    axes[0].set(xlabel="Fewer false alarms (kNN minus C=4 scaled)", title="Improvement is concentrated")
    methods = ("knn20", "c4_centroid_distance", "c4_assigned_radius", "c32_centroid_distance")
    colors = ("#555555", "#c08039", "#168579", "#6c78ad")
    labels = ("kNN-20", "C=4 centroid", "C=4 scaled", "C=32 centroid")
    for ax, task in zip(axes[1:], (list(tasks)[0], list(tasks)[3])):
        selected = frame.task.eq(task) & ~frame.failure
        for method, color, label in zip(methods, colors, labels):
            values = np.sort(ratios[method][selected])
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=color, label=label)
        ax.axvline(1, color="#444444", linestyle="--", linewidth=1)
        ax.set(xlabel="Trajectory peak / calibrated threshold", ylabel="Fraction of successful trajectories",
               title=tasks[task], xlim=(0, 3), ylim=(0, 1.02))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Frozen 32,000-trajectory evaluation | nominal 5% | descriptive diagnosis", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"task_and_threshold_diagnosis.{suffix}", dpi=170)
    plt.close(fig)


def diagnose(parent, output):
    sealed = json.loads((parent / "sealed_manifest.json").read_text())
    verified = json.loads((parent / "verification.json").read_text())
    reported = json.loads((parent / "report_verification.json").read_text())
    assert verified["passed"] and reported["passed"]
    assert verified["sealed_manifest_sha256"] == digest(parent / "sealed_manifest.json")
    inputs = {}

    def checked(name):
        path = parent / name
        expected = sealed["artifacts"].get(name, reported["artifacts"].get(name))
        actual = digest(path)
        assert expected is not None and actual == expected, name
        inputs[name] = actual
        return path

    fields = ["global_row", "suite", "task", "fold", "failure", "actual_action_steps", "length",
              "primary_failure_reason", "init_state_id", "episode", "source"]
    frame = pd.read_csv(checked("trajectory_results.csv"), usecols=fields)
    np.testing.assert_array_equal(frame.global_row, np.arange(32000))
    data = load_npz(checked("all_episode_results.npz"))
    methods = list(data["methods"])
    ai = int(np.flatnonzero(np.isclose(data["alphas"], .05))[0])
    assert list(data["calibration_kinds"]) == ["episode", "task_init"]
    y = frame.failure.to_numpy()
    raw = {method: data["peak_scores"][methods.index(method)] for method in SELECTED}
    tau = {method: data["thresholds"][1, ai, methods.index(method)] for method in SELECTED}
    ratios = {method: np.where(np.isfinite(raw[method]), raw[method] / tau[method], -1) for method in SELECTED}
    for method in SELECTED:
        np.testing.assert_array_equal(ratios[method] > 1, data["first"][1, ai, methods.index(method)] >= 0)
    old, new = (data["first"][1, ai, methods.index(method)] >= 0 for method in ("knn20", "c4_assigned_radius"))
    assert ((old & y).sum(), (old & ~y).sum(), (new & y).sum(), (new & ~y).sum()) == (977, 2754, 942, 1220)
    effects, distributions, durations = [], [], []
    for task, ids in frame.groupby("task").groups.items():
        ids = np.asarray(ids)
        suite, fold = frame.loc[ids[0], ["suite", "fold"]]
        row = dict(task=task, suite=suite, fold=fold, failures=int(y[ids].sum()), successes=int((~y[ids]).sum()))
        for name, alarms in (("knn", old), ("c4", new)):
            row[f"{name}_fp"] = int((alarms[ids] & ~y[ids]).sum())
            row[f"{name}_tp"] = int((alarms[ids] & y[ids]).sum())
        row.update(fp_removed=row["knn_fp"] - row["c4_fp"], tp_change=row["c4_tp"] - row["knn_tp"])
        effects.append(row)
        duration = dict(task=task, suite=suite)
        for failure in (False, True):
            selected = ids[y[ids] == failure]
            steps = frame.loc[selected, "actual_action_steps"].to_numpy()
            label = "failure" if failure else "success"
            duration.update({f"{label}_episodes": len(steps), f"{label}_min": steps.min() if len(steps) else np.nan,
                             f"{label}_max": steps.max() if len(steps) else np.nan, **quantiles(steps, label)})
            if not len(selected):
                continue
            record = dict(task=task, suite=suite, fold=fold, failure=failure, episodes=len(selected))
            for method in SELECTED:
                record.update(quantiles(np.where(ratios[method][selected] >= 0, ratios[method][selected], np.nan), method))
            for other in ("knn20", "c32_centroid_distance"):
                valid = np.isfinite(raw[other][selected]) & (raw[other][selected] > 0)
                record.update(quantiles(raw["c4_centroid_distance"][selected][valid] / raw[other][selected][valid], f"raw_c4_over_{other}"))
                record[f"tau_c4_over_{other}"] = float(tau["c4_centroid_distance"][ids[0]] / tau[other][ids[0]])
            distributions.append(record)
        durations.append(duration)
    tables = dict(task_effects=pd.DataFrame(effects), score_distributions=pd.DataFrame(distributions), duration_by_task=pd.DataFrame(durations))
    tables["fold_effects"] = tables["task_effects"].groupby(["suite", "fold"], as_index=False).sum(numeric_only=True)

    rankings = []
    scores_for_ranking = {**ratios, "final_duration_diagnostic_only": frame.actual_action_steps.to_numpy()}
    populations = [("all", "all", np.arange(len(frame)))]
    populations += [(level, name, np.asarray(ids)) for level in ("suite", "fold", "task") for name, ids in frame.groupby(level).groups.items()]
    for level, group, ids in populations:
        if y[ids].sum() in (0, len(ids)):
            continue
        for method, values in scores_for_ranking.items():
            rankings.append(dict(level=level, group=group, method=method, episodes=len(ids), failures=int(y[ids].sum()),
                                 auc=roc_auc_score(y[ids], values[ids]), ap=average_precision_score(y[ids], values[ids])))
    tables["ranking_diagnostics"] = pd.DataFrame(rankings)
    tables["macro_task_ranking"] = tables["ranking_diagnostics"].loc[lambda x: x.level.eq("task")].groupby("method", as_index=False).agg(
        tasks=("group", "size"), mean_auc=("auc", "mean"), mean_ap=("ap", "mean"))
    reasons = []
    for reason, group in frame.loc[y].groupby("primary_failure_reason", dropna=False):
        ids = group.index.to_numpy()
        reasons.append(dict(reason=reason, failures=len(ids), knn_detected=int(old[ids].sum()), c4_detected=int(new[ids].sum()),
                            lost=int((old[ids] & ~new[ids]).sum()), gained=int((~old[ids] & new[ids]).sum()), missed=int((~new[ids]).sum())))
    tables["failure_reasons"] = pd.DataFrame(reasons)

    units, thresholds, identities, composition = [], [], [], []
    threshold_checks = 0
    for info in sealed["folds"]:
        fold = info["fold"]
        prediction = load_npz(checked(f"predictions/{fold}.npz"))
        np.testing.assert_array_equal(prediction["methods"], data["methods"])
        cal_rows = prediction["calibration_rows"]
        cal = frame.loc[cal_rows, ["global_row", "task", "init_state_id", "episode", "source"]].reset_index(drop=True)
        good = prediction["calibration_labels"] == 0
        for method in SELECTED:
            mi = methods.index(method)
            scores = prediction["calibration_scores"][mi]
            peaks = trajectory_peak(scores)
            group = cal.loc[good].copy()
            group["score"] = peaks[good]
            group["peak_query"] = np.where(np.isfinite(scores), scores, -np.inf).argmax(1)[good]
            group.loc[~np.isfinite(group.score), "peak_query"] = -1
            winners = group.groupby(["task", "init_state_id"]).score.idxmax()
            top = group.loc[winners].sort_values(["score", "global_row"], ascending=[False, True]).copy()
            top["descending_rank"] = np.arange(1, len(top) + 1)
            top["method"], top["fold"] = method, fold
            units.extend(top.to_dict("records"))
            for alpha_i, alpha in enumerate(data["alphas"]):
                expected, rank = conformal_threshold(top.score.to_numpy(), float(alpha))
                assert expected == prediction["thresholds"][1, alpha_i, mi]
                thresholds.append(dict(fold=fold, method=method, alpha=float(alpha), units=len(top), ascending_rank=rank,
                                       descending_rank=len(top) - rank + 1, threshold=expected,
                                       tau_over_alpha005=expected / prediction["thresholds"][1, ai, mi]))
                threshold_checks += 1
        for count in sealed["clusters"]:
            profile = load_npz(checked(f"profiles/{fold}_c{count}.npz"))
            task = frame.loc[profile["success_global_rows"], "task"].to_numpy()
            table = pd.crosstab(profile["assignments"], task)
            identities.append(dict(fold=fold, suite=info["suite"], clusters=count, reference_points=len(task),
                                   weighted_task_purity=table.max(axis=1).sum() / len(task),
                                   task_adjusted_mutual_information=adjusted_mutual_info_score(task, profile["assignments"]),
                                   single_task_clusters=int((table.gt(0).sum(axis=1) == 1).sum())))
            for cluster in table.index:
                for name in table.columns:
                    points = int(table.loc[cluster, name])
                    if points:
                        composition.append(dict(fold=fold, clusters=count, cluster=int(cluster), task=name, points=points))
        print(f"DIAGNOSED {fold}", flush=True)
    tables.update(calibration_units=pd.DataFrame(units), calibration_thresholds=pd.DataFrame(thresholds),
                  cluster_task_identity=pd.DataFrame(identities), cluster_task_composition=pd.DataFrame(composition))
    tables["cluster_task_summary"] = tables["cluster_task_identity"].groupby("clusters", as_index=False).agg(
        mean_task_purity=("weighted_task_purity", "mean"), min_task_purity=("weighted_task_purity", "min"),
        max_task_purity=("weighted_task_purity", "max"), mean_task_ami=("task_adjusted_mutual_information", "mean"),
        single_task_clusters=("single_task_clusters", "sum"))

    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    diagnostic_figures(output, frame, tables["task_effects"], ratios)
    effect = tables["task_effects"]
    summary = dict(passed=True, unique_trajectories=len(frame), failures=int(y.sum()), successes=int((~y).sum()),
                   refitted_models=False, retuned_thresholds=False, new_rollouts=False, descriptive_not_blind=True,
                   checked_trajectory_alarms=len(SELECTED) * len(frame), checked_thresholds=threshold_checks,
                   tasks_fp_better=int((effect.fp_removed > 0).sum()), tasks_fp_tied=int((effect.fp_removed == 0).sum()),
                   tasks_fp_worse=int((effect.fp_removed < 0).sum()), source_sha256=digest(Path(__file__)),
                   parent_manifest_sha256=digest(parent / "sealed_manifest.json"), inputs=inputs,
                   rows={name: len(table) for name, table in tables.items()},
                   artifacts={path.name: digest(path) for path in sorted(output.iterdir()) if path.suffix in (".csv", ".png", ".pdf")})
    write_json(output / "verification.json", summary)
    print(tables["macro_task_ranking"].to_string(index=False))
    print(tables["cluster_task_summary"].to_string(index=False))
    print("K-MEANS DIAGNOSIS VERIFIED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    diagnose(args.parent, args.output or args.parent / "diagnosis")
