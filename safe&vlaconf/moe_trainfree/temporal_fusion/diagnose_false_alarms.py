"""Retrospective task, phase, and feature attribution for frozen kNN alarms."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "boundary_knn"))
from knn import ReferenceScorer, dynamics, reference_pairs
from core import ROOT, digest, first_alarm, write_json

GROUPS = {"front_mobility": slice(0, 4), "back_mobility": slice(4, 8),
          "acceleration": slice(8, 9), "periodicity": slice(9, 10)}
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def tabulate(decisions, output):
    data = decisions.loc[decisions.method.eq("knn10")].copy()
    data["alarm"] = data.first_alarm_query.ge(0)
    success = data.loc[~data.failure].copy()
    for name, columns in (("scope", ["scope"]), ("task", ["scope", "task"]),
                          ("task_fold", ["scope", "task", "fold"]),
                          ("init", ["scope", "task", "fold", "init_state_id"]),
                          ("noise", ["scope", "task", "fold", "noise_seed"])):
        result = success.groupby(columns).agg(successes=("alarm", "size"),
            unique_episodes=("global_row", "nunique"), fp=("alarm", "sum"), fpr=("alarm", "mean"))
        result.to_csv(output / f"{name}_false_alarms.csv")
    paired = success.groupby(["global_row", "task", "scope"]).alarm.agg(["mean", "sum", "size"]).unstack("scope")
    paired = paired.dropna().reset_index()
    paired.columns = ["_".join(str(v) for v in key if v) for key in paired.columns]
    paired.to_csv(output / "same_episode_seen_unseen.csv", index=False)
    rows = []
    for (scope, failed), group in data.groupby(["scope", "failure"]):
        alarmed = group.loc[group.alarm]
        progress = (alarmed.first_alarm_query.to_numpy() + 1) / alarmed.length.to_numpy()
        band = np.minimum((4 * progress).astype(int), 3)
        for i, label in enumerate(("[0,25%)", "[25,50%)", "[50,75%)", "[75,100%]")):
            count = int((band == i).sum())
            rows.append(dict(scope=scope, failure=bool(failed), bin=label, count=count,
                alarmed=len(alarmed), all_episodes=len(group), fraction_of_alarms=count / len(alarmed)))
    pd.DataFrame(rows).to_csv(output / "relative_timing.csv", index=False)
    return data


def control_step_timing(frame, data, output):
    steps, sources = {}, {}
    for source, group in frame.iloc[data.global_row.unique()].groupby("source"):
        path = ROOT / "VLA_MUI_HUB" / source / "client/summaries.json"
        records = {int(row["episode_index"]): row for row in json.loads(path.read_text())}
        sources[str(path)] = digest(path)
        for row in group.itertuples():
            record = records[int(row.episode)]
            assert int(record["inference_calls"]) == row.length
            count = int(record["action_steps"])
            assert 10 * (row.length - 1) < count <= 10 * row.length
            steps[row.Index] = count
    data["actual_action_steps"] = data.global_row.map(steps)
    assert data.actual_action_steps.notna().all()
    data["control_progress_at_alarm"] = np.where(data.alarm,
        10 * data.first_alarm_query / data.actual_action_steps, np.nan)
    assert data.loc[data.alarm, "control_progress_at_alarm"].between(0, 1, inclusive="left").all()
    data.to_csv(output / "episode_control_timing.csv", index=False)
    records = []
    for (scope, failed), group in data.groupby(["scope", "failure"]):
        for suite, part in [("all", group), *list(group.groupby("suite"))]:
            alarmed = part.loc[part.alarm]
            bins = np.minimum((alarmed.control_progress_at_alarm.to_numpy()*4).astype(int), 3)
            for i, label in enumerate(("[0,25%)", "[25,50%)", "[50,75%)", "[75,100%)")):
                count = int((bins == i).sum())
                records.append(dict(scope=scope, failure=bool(failed), suite=suite, bin=label,
                    count=count, alarms=len(alarmed), all_episodes=len(part),
                    fraction_of_alarms=count/len(alarmed)))
    pd.DataFrame(records).to_csv(output / "control_step_timing.csv", index=False)
    return sources


def attribute_alarms(fold, profile, prediction, vectors, rows, decisions):
    part = decisions.loc[decisions.fold.eq(fold) & decisions.alarm].copy()
    positions = pd.Index(rows).get_indexer(part.global_row)
    queries = part.first_alarm_query.to_numpy()
    raw = vectors[positions, queries]
    normalized = (raw.astype(np.float64) - profile["dynamic_center"]) / profile["dynamic_scale"]
    distances, ids = ReferenceScorer(profile).neighbors("success_dynamic", normalized)
    test_positions = pd.Index(prediction["test_rows"]).get_indexer(part.global_row)
    method = list(prediction["methods"]).index("dyn_success_knn_k20")
    score = prediction["scores"][method]
    np.testing.assert_allclose(distances.mean(1), score[test_positions, queries], rtol=1e-6, atol=2e-7)
    delta = normalized[:, None] - profile["success_dynamic"][ids]
    # Allocate each Euclidean distance across dimensions, then average neighbors.
    # Unlike squared-distance shares, these terms sum to the actual kNN score.
    contribution = (delta ** 2 / distances[..., None]).mean(1)
    np.testing.assert_allclose(contribution.sum(1), distances.mean(1), rtol=1e-10, atol=1e-10)
    share = contribution / distances.mean(1)[:, None]
    part["score"] = distances.mean(1)
    for name, columns in GROUPS.items():
        part[f"{name}_share"] = share[:, columns].sum(1)
        part[f"{name}_value"] = raw[:, columns].mean(1)
    for i, name in enumerate((*LAYER_NAMES, "acceleration", "periodicity")):
        part[f"dimension_{name}_share"] = share[:, i]
        part[f"dimension_{name}_value"] = raw[:, i]
    part["dominant_group"] = np.asarray(list(GROUPS))[part[[f"{name}_share" for name in GROUPS]].to_numpy().argmax(1)]
    part["mobility_expansion_share"] = (share[:, :8] * (raw[:, :8] < 0)).sum(1)
    part["mobility_contraction_share"] = (share[:, :8] * (raw[:, :8] >= 0)).sum(1)
    returns, runs, remaining = [], [], []
    for row, position in zip(part.itertuples(), test_positions):
        active = score[position, row.first_alarm_query:row.length] > row.threshold
        assert active[0]
        off = np.flatnonzero(~active)
        returns.append(bool(len(off)))
        runs.append(int(off[0]) if len(off) else len(active))
        remaining.append(row.length - row.first_alarm_query - 1)
    part["returns_below_threshold"] = returns
    part["first_crossing_run"] = runs
    part["queries_after_alarm"] = remaining
    return part


def matched_bank_addition(fold, task, profile, prediction, vectors, rows, frame, output):
    """Oracle diagnosis only: extra A successes of a task absent from this fold."""
    local = frame.iloc[rows].copy()
    local["global_row"] = rows
    local = local.reset_index(drop=True)
    heldout = local.index[local.task.eq(task) & local.run_id.eq("right-50x8-20260903") & ~local.failure].to_numpy()
    bank_tasks = frame.iloc[profile["success_global_rows"]].task
    assert not bank_tasks.eq(task).any()
    success_reference = prediction["reference_rows"][~frame.iloc[prediction["reference_rows"]].failure.to_numpy()]
    control_rows = pd.Index(rows).get_indexer(success_reference)
    same_pairs = reference_pairs(vectors, heldout, cap=1000000)
    control_pairs = reference_pairs(vectors, control_rows, cap=1000000)
    existing = set(zip(profile["success_global_rows"].tolist(), profile["success_queries"].tolist()))
    control_pairs = np.asarray([pair for pair in control_pairs if (int(rows[pair[0]]), int(pair[1])) not in existing])
    assert local.iloc[np.unique(same_pairs[:, 0])].run_id.eq("right-50x8-20260903").all()
    assert local.iloc[np.unique(control_pairs[:, 0])].run_id.eq("right-50x8-20260903").all()
    tests = local.loc[local.run_id.eq("right-50x8b-20260903") & local.task.eq(task)]
    assert set(tests.global_row) <= set(prediction["test_rows"])
    method = list(prediction["methods"]).index("dyn_success_knn_k20")
    alpha = np.flatnonzero(np.isclose(prediction["alphas"], .05))[0]
    threshold = float(prediction["thresholds"][1, alpha, method])
    test_positions = pd.Index(prediction["test_rows"]).get_indexer(tests.global_row)
    normalized = (vectors[tests.index].astype(np.float64) - profile["dynamic_center"]) / profile["dynamic_scale"]
    valid = np.isfinite(normalized).all(-1)
    base_distances, _ = ReferenceScorer(profile).neighbors("success_dynamic", normalized[valid])
    original = np.full(valid.shape, np.nan, dtype=np.float32)
    original[valid] = base_distances.mean(1)
    np.testing.assert_allclose(original, prediction["scores"][method, test_positions], rtol=1e-6, atol=2e-7, equal_nan=True)
    scores = {"original": original, "plus_same_task": original.copy(), "plus_other_tasks": original.copy()}
    flat_episode = np.nonzero(valid)[0]
    selected = {}
    for init in sorted(tests.init_state_id.unique()):
        # Each B initial state is excluded from both additional A banks.
        current = tests.init_state_id.to_numpy()[flat_episode] == init
        query = normalized[valid][current]
        for name, pairs in (("plus_same_task", same_pairs), ("plus_other_tasks", control_pairs)):
            eligible = pairs[local.iloc[pairs[:, 0]].init_state_id.to_numpy() != init]
            assert len(eligible) >= 1024
            chosen = eligible[np.random.default_rng(20260911 + int(init)).choice(len(eligible), 1024, replace=False)]
            assert (local.iloc[chosen[:, 0]].init_state_id != init).all()
            selected[f"init_{init}_{name}"] = np.column_stack((rows[chosen[:, 0]], chosen[:, 1]))
            extra = (vectors[chosen[:, 0], chosen[:, 1]].astype(np.float64) - profile["dynamic_center"]) / profile["dynamic_scale"]
            distance = cdist(query, extra)
            nearest = np.partition(distance, 19, axis=1)[:, :20]
            combined = np.concatenate((base_distances[current], nearest), axis=1)
            score = np.partition(combined, 19, axis=1)[:, :20].mean(1)
            flat_scores = scores[name][valid]
            flat_scores[current] = score
            scores[name][valid] = flat_scores
    results = []
    for name, score in scores.items():
        record = tests[["global_row", "task", "episode", "init_state_id", "noise_seed", "failure", "length"]].copy()
        record["fold"], record["variant"], record["threshold"] = fold, name, threshold
        record["first_alarm_query"] = first_alarm(score, threshold)
        record["original_first_alarm_query"] = first_alarm(original, threshold)
        assert np.all(np.nan_to_num(score-original, nan=0) <= 2e-6)
        results.append(record)
    name = task.split("/")[-1]
    np.savez_compressed(output / f"bank_addition_{name}.npz", **selected, **scores,
                        global_rows=tests.global_row.to_numpy(), threshold=threshold)
    return pd.concat(results, ignore_index=True)


def make_figure(output, data, attribution, additions, top_tasks):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    short = {task: task.split("/")[-1].replace("pick_up_the_", "").replace("_and_place_it_in_the_basket", "").replace("_", " ") for task in top_tasks}
    success = data.loc[~data.failure & data.task.isin(top_tasks)]
    rates = success.groupby(["task", "scope"]).alarm.mean().unstack().loc[top_tasks]
    x = np.arange(len(rates))
    axes[0, 0].barh(x-.18, rates.seen, height=.34, label="Task seen in A", color="#378477")
    axes[0, 0].barh(x+.18, rates.unseen, height=.34, label="Task held out from A", color="#bf5667")
    axes[0, 0].set_yticks(x, [short[t] for t in top_tasks], fontsize=8)
    axes[0, 0].invert_yaxis()
    axes[0, 0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].set_title("Task coverage and false alarms", loc="left", fontsize=11)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fp = data.loc[data.scope.eq("unseen") & ~data.failure & data.alarm]
    progress = fp.control_progress_at_alarm.to_numpy()
    counts = np.bincount(np.minimum((progress*4).astype(int), 3), minlength=4)
    axes[0, 1].bar(np.arange(4), counts, color="#5687b3")
    axes[0, 1].set_xticks(np.arange(4), ["0-25%", "25-50%", "50-75%", "75-100%"])
    axes[0, 1].set_xlabel("Completed control steps / actual episode control steps")
    axes[0, 1].set_title("Where the 847 false alarms occur", loc="left", fontsize=11)
    for i, count in enumerate(counts):
        axes[0, 1].text(i, count+8, f"{count} ({count/len(fp):.1%})", ha="center", fontsize=9)
    axes[0, 1].set_ylim(0, counts.max()*1.17)
    afp = attribution.loc[attribution.scope.eq("unseen") & ~attribution.failure]
    share = np.stack([afp[f"{name}_share"].mean() for name in GROUPS])
    axes[1, 0].bar(np.arange(4), share, color=["#5687b3", "#378477", "#bf5667", "#aa8846"])
    axes[1, 0].set_xticks(np.arange(4), ["Front 4\nmobility", "Back 4\nmobility", "Acceleration", "Periodicity"])
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].set_ylim(0, max(share)*1.2)
    axes[1, 0].set_title("Mean exact-distance contribution at false alarms", loc="left", fontsize=11)
    for i, value in enumerate(share):
        axes[1, 0].text(i, value+.012, f"{value:.1%}", ha="center", fontsize=9)
    rates = additions.loc[~additions.failure].assign(alarm=lambda d: d.first_alarm_query.ge(0)).groupby(["task", "variant"]).alarm.mean().unstack()
    for i, (variant, label, color) in enumerate((("original", "Original", "#555555"), ("plus_other_tasks", "+1,024 other-task A points", "#5687b3"), ("plus_same_task", "+1,024 same-task A points", "#378477"))):
        axes[1, 1].bar(np.arange(len(rates))+(i-1)*.24, rates[variant], width=.23, label=label, color=color)
    axes[1, 1].set_xticks(np.arange(len(rates)), ["Push plate", "Wine bottle to rack"])
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 1].set_ylim(0, 1.04)
    axes[1, 1].legend(fontsize=8, frameon=False, loc="upper right")
    axes[1, 1].set_title("Reference coverage intervention (diagnosis only)", loc="left", fontsize=11)
    axes[1, 1].set_xlabel("Scale and threshold fixed; same initial state excluded")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Original kNN-10D: false-alarm diagnosis\nRetrospective A/B analysis; no model training", fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"false_alarm_diagnosis.{extension}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round7_temporal_fusion/false_alarm_diagnosis")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=True)
    original = HERE.parent / "results/round5_knn"
    source = HERE.parent / "results/round7_temporal_fusion"
    manifest = json.loads((original / "sealed_manifest.json").read_text())
    frame = pd.read_csv(original / "outcome_alignment.csv")
    data = tabulate(pd.read_csv(source / "episode_decisions.csv"), output)
    cache_path = HERE.parent / "results/round3_safe/v7/v7_inputs.npz"
    cached = load_npz(cache_path)
    inputs = {str(path): digest(path) for path in (cache_path, source / "episode_decisions.csv", original / "outcome_alignment.csv", original / "sealed_manifest.json")}
    inputs.update(control_step_timing(frame, data, output))
    top_tasks = data.loc[data.scope.eq("unseen") & ~data.failure].groupby("task").alarm.sum().nlargest(4).index.tolist()
    interventions = set(top_tasks[:2])
    parts, additions, series = [], [], []
    for info in manifest["folds"]:
        fold = info["fold"]
        loaded = {}
        for name in ("profiles", "predictions"):
            path = original / name / f"{fold}.npz"
            assert digest(path) == manifest["artifacts"][str(path.relative_to(original))]
            inputs[str(path)] = digest(path)
            loaded[name] = load_npz(path)
        profile, prediction = loaded["profiles"], loaded["predictions"]
        rows = np.flatnonzero(frame.suite.eq(info["suite"]))
        vectors, _ = dynamics(cached["mobility"][rows], cached["acceleration"][rows],
                              cached["periodicity"][rows], float(profile["periodicity_scale"]))
        vectors[~cached["valid"][rows]] = np.nan
        parts.append(attribute_alarms(fold, profile, prediction, vectors, rows, data))
        heldout_tasks = set(frame.iloc[prediction["test_rows"][prediction["test_unseen"]]].task)
        for task in sorted(interventions & heldout_tasks):
            additions.append(matched_bank_addition(fold, task, profile, prediction, vectors, rows, frame, output))
            episode_rows = np.flatnonzero(frame.task.eq(task) & frame.run_id.eq("right-50x8b-20260903"))
            positions = pd.Index(rows).get_indexer(episode_rows)
            decision = data.loc[data.fold.eq(fold)].set_index("global_row")
            method = list(prediction["methods"]).index("dyn_success_knn_k20")
            prediction_positions = pd.Index(prediction["test_rows"]).get_indexer(episode_rows)
            for j, (row, local) in enumerate(zip(episode_rows, positions)):
                for query in range(int(frame.iloc[row].length)):
                    m = cached["mobility"][row, query]
                    base = cached["mobility"][row, 1:5].mean(0)
                    v = vectors[local, query]
                    series.append(dict(fold=fold, task=task, global_row=int(row), query=query,
                        failure=bool(frame.iloc[row].failure), first_alarm_query=int(decision.loc[row].first_alarm_query),
                        length=int(frame.iloc[row].length), front_mobility=float(m[:4].mean()),
                        back_mobility=float(m[4:].mean()), front_base=float(base[:4].mean()),
                        back_base=float(base[4:].mean()), front_relative=float(v[:4].mean()),
                        back_relative=float(v[4:8].mean()), acceleration=float(v[8]), periodicity=float(v[9]),
                        score=float(prediction["scores"][method, prediction_positions[j], query]),
                        threshold=float(decision.loc[row].threshold)))
        print(f"DIAGNOSED {fold}", flush=True)
    attribution = pd.concat(parts, ignore_index=True)
    attribution.to_csv(output / "alarm_feature_attribution.csv", index=False)
    additions = pd.concat(additions, ignore_index=True)
    additions.to_csv(output / "bank_addition_episode_decisions.csv", index=False)
    additions["alarm"] = additions.first_alarm_query.ge(0)
    counts = additions.groupby(["task", "variant", "failure"]).agg(episodes=("alarm", "size"),
        alarms=("alarm", "sum"), alarm_rate=("alarm", "mean"))
    counts.to_csv(output / "bank_addition_summary.csv")
    pd.DataFrame(series).to_csv(output / "top_task_query_signals.csv", index=False)
    columns = [f"{name}_share" for name in GROUPS] + ["mobility_expansion_share", "mobility_contraction_share", "returns_below_threshold"]
    attribution.groupby(["scope", "failure"])[columns].mean().to_csv(output / "feature_summary.csv")
    attribution.groupby(["scope", "failure", "task"])[columns].mean().to_csv(output / "task_feature_summary.csv")
    make_figure(output, data, attribution, additions, top_tasks)
    write_json(output / "verification.json", dict(inputs=inputs, source_sha256=digest(Path(__file__)),
        folds=len(manifest["folds"]), reproduced_alarm_distances=len(attribution),
        exact_additive_distance_attribution=True, bank_addition_extra_points=1024,
        bank_addition_same_init_excluded=True, bank_addition_scale_and_threshold_fixed=True,
        intervention_tasks=sorted(interventions), new_model_training=False,
        protocol="Retrospective diagnosis. Same-task intervention accesses additional held-out-task A successes; not an unseen-task evaluation or calibrated deployable method.",
        artifacts={path.name: digest(path) for path in sorted(output.iterdir())
                   if path.is_file() and path.name != "verification.json"}))
    print(counts.to_string(), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
