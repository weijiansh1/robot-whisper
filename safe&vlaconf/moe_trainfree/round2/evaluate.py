"""Evaluate the frozen paired scores, and align the event pilot to physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from paired_core import ROOT, SEED, digest, write_json

HERE = Path(__file__).resolve().parent
PHYSICAL = ROOT / "VLA_MUI_HUB/physical-failure-labels/results"
DROP = ("object_released_or_dropped_before_goal", "object_released_outside_goal")


def join_labels(frame, labels):
    keys = ["task", "run_id", "episode"]
    frame = frame.copy()
    frame["order"] = np.arange(len(frame))
    needed = pd.MultiIndex.from_frame(frame[keys])
    labels = labels.loc[pd.MultiIndex.from_frame(labels[keys]).isin(needed)]
    if labels.duplicated(keys).any():
        raise ValueError("duplicate physical outcome label")
    result = frame.merge(labels[keys + ["recorded_success", "primary_failure_reason", "init_state_id"]],
                         how="left", on=keys, validate="many_to_one", suffixes=("", "_label"))
    result = result.sort_values("order").reset_index(drop=True)
    if result.recorded_success.isna().any() or not pd.api.types.is_bool_dtype(result.recorded_success):
        raise ValueError("missing or non-boolean physical outcome")
    if not np.array_equal(result.init_state_id, result.init_state_id_label):
        raise ValueError("physical outcome initial-state mismatch")
    result["failure"] = ~result.recorded_success
    return result


def auc(y, score):
    return float(roc_auc_score(y, score)) if np.unique(y).size == 2 else np.nan


def metrics(frame, score):
    y = frame.failure.to_numpy(bool)
    groups = []
    task_auc = []
    for task, task_rows in frame.groupby("task").groups.items():
        task_auc.append(auc(y[task_rows], score[task_rows]))
        init_values = []
        for _, rows in frame.loc[task_rows].groupby("init_state_id").groups.items():
            value = auc(y[rows], score[rows])
            if np.isfinite(value):
                init_values.append(value)
        if init_values:
            groups.append(float(np.mean(init_values)))
    return {"episodes": len(y), "failures": int(y.sum()), "auc": auc(y, score),
            "ap": float(average_precision_score(y, score)),
            "task_macro_auc": float(np.nanmean(task_auc)),
            "within_init_macro_auc": float(np.mean(groups)) if groups else np.nan}


def alarm_metrics(y, fired):
    p, n = int(y.sum()), int((~y).sum())
    tp, fp = int((y & fired).sum()), int((~y & fired).sum())
    return {"tp": tp, "fp": fp, "failures": p, "successes": n,
            "recall": tp / p if p else np.nan, "fpr": fp / n if n else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "coverage": float((~fired).mean())}


def paired_representation_bootstrap(frame, scores, names):
    comparison = "routed_d9__knn5"
    baselines = ("hidden_d9__knn5", "shared_d9__knn5", "action__knn5", "route_d9__knn5")
    records = []
    rng = np.random.default_rng(SEED)
    tasks = sorted(frame.task.unique())
    for baseline in baselines:
        task_differences = []
        for task in tasks:
            task_frame = frame[frame.task == task]
            delta = []
            for _, rows in task_frame.groupby("init_state_id").groups.items():
                y = frame.loc[rows, "failure"].to_numpy(bool)
                delta.append(auc(y, scores[rows, names.index(comparison)]) -
                             auc(y, scores[rows, names.index(baseline)]))
            task_differences.append(np.asarray(delta))
        boot_tasks = []
        estimates = []
        for delta in task_differences:
            sample = rng.integers(len(delta), size=(5000, len(delta)))
            values = delta[sample]
            counts = np.isfinite(values).sum(axis=1)
            boot_tasks.append(np.divide(np.nansum(values, axis=1), counts,
                                        out=np.full(5000, np.nan), where=counts > 0))
            estimates.append(np.nanmean(delta))
        boot = np.nanmean(boot_tasks, axis=0)
        lo, hi = np.nanquantile(boot, [0.025, 0.975])
        records.append({"method": comparison, "baseline": baseline,
                        "within_init_macro_auc_difference": float(np.nanmean(estimates)),
                        "lo": lo, "hi": hi})
    return records


def releases(record, kind):
    objects = record["goal_subject_physics"]
    if kind == "any_goal":
        subjects = list(objects)
    else:
        expressions = {p["id"]: p["expression"] for p in record["goal_predicates"]}
        subjects = [expressions[p["goal_id"]][1] for p in record["goal_failure_labels"]
                    if ((p["reason"] in DROP) if kind == "drop_goal" else
                        p["reason"] != "goal_satisfied_at_last_checkpoint")]
    times = [objects[s]["first_release_snapshot"] for s in subjects
             if s in objects and objects[s].get("first_release_snapshot") is not None]
    return min(times) if times else None


def physics_rows(frame, records):
    result = []
    for row in frame.itertuples():
        if not row.failure:
            continue
        key = (row.task, row.run_id, int(row.episode))
        if key not in records:
            raise ValueError("missing physical failure record")
        record = records[key]
        item = {"order": row.order, "task": row.task, "run_id": row.run_id,
                "episode": row.episode, "query": row.query,
                "reason": record["primary_failure_reason"]}
        for scope in ("any_goal", "failed_goal", "drop_goal"):
            release = releases(record, scope)
            item[scope + "_release"] = release
            item[scope + "_query_minus_release"] = row.query - release if release is not None else np.nan
        if hasattr(row, "pair_id"):
            item.update(pair_id=row.pair_id, lead=row.lead, qualified=row.qualified,
                        static_onset=row.static_onset)
        result.append(item)
    return pd.DataFrame(result)


def evaluate_pilot(output, frame):
    with np.load(output / "pilot_predictions.npz", allow_pickle=False) as archive:
        scores, names = archive["scores"], archive["method_names"].astype(str).tolist()
    frame["qualified"] = frame.qualified.astype(bool)
    pairs = sorted(frame.pair_id.unique())
    pair_index = {p: i for i, p in enumerate(pairs)}
    sampled = np.random.default_rng(SEED).integers(len(pairs), size=(10000, len(pairs)))
    results, differences, associations = [], [], []
    for lead in (-4, -2):
        take = (frame.lead.to_numpy() == lead) & frame.qualified.to_numpy()
        rows = np.flatnonzero(take)
        truth = frame.failure.to_numpy(bool)
        hits = np.full((len(pairs), len(names)), np.nan)
        raw_differences = np.full_like(hits, np.nan)
        for pair, selected in frame.loc[take].groupby("pair_id").groups.items():
            selected = np.asarray(selected)
            if len(selected) != 2 or truth[selected].sum() != 1:
                raise ValueError("pilot comparison is not an event/control pair")
            event_row, control_row = selected[truth[selected]][0], selected[~truth[selected]][0]
            delta = scores[event_row] - scores[control_row]
            hits[pair_index[pair]] = (delta > 0) + 0.5 * (delta == 0)
            raw_differences[pair_index[pair]] = delta
            if frame.loc[event_row, "query"] != frame.loc[control_row, "query"]:
                raise ValueError("pilot pair absolute queries differ")
        for m, method in enumerate(names):
            good = np.isfinite(hits[:, m])
            boot = np.nanmean(hits[sampled, m], axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            results.append({"lead": lead, "method": method, "pairs": int(good.sum()),
                            "pooled_auc": auc(truth[rows], scores[rows, m]),
                            "within_pair_accuracy": float(np.nanmean(hits[:, m])), "lo": lo, "hi": hi})
        for functional in ("cosine_high", "authority_low", "disagreement_low", "cancellation_low"):
            m = names.index(functional)
            for baseline in ("action_translation_low", "eef_motion_low", "route_entropy_low"):
                b = names.index(baseline)
                delta = hits[:, m] - hits[:, b]
                boot = np.nanmean(delta[sampled], axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                differences.append({"lead": lead, "method": functional, "baseline": baseline,
                                    "within_pair_accuracy_difference": float(np.nanmean(delta)),
                                    "lo": lo, "hi": hi})
                good = np.isfinite(raw_differences[:, m]) & np.isfinite(raw_differences[:, b])
                rho = spearmanr(raw_differences[good, m], raw_differences[good, b]).statistic
                associations.append({"lead": lead, "method": functional, "baseline": baseline,
                                     "pair_difference_spearman": float(rho)})
    pd.DataFrame(results).to_csv(output / "pilot_metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "pilot_paired_comparisons.csv", index=False)
    pd.DataFrame(associations).to_csv(output / "pilot_behavior_correlations.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round2")
    args = parser.parse_args()
    output = args.input.resolve()
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        if digest(output / name) != expected:
            raise ValueError(f"sealed predictions changed: {name}")
    for name, expected in manifest["sources"].items():
        if digest(name) != expected:
            raise ValueError(f"sealed source changed: {name}")
    labels = pd.read_csv(PHYSICAL / "episodes.csv")
    labels["task"] = labels.suite + "/" + labels.task_name
    labels = labels.rename(columns={"episode_index": "episode"})
    datasets = {name: join_labels(pd.read_csv(output / f"{name}_index.csv"), labels)
                for name in ("q0", "q34", "pilot")}
    rankings, alarms, comparisons = [], [], []
    for name, frame in datasets.items():
        frame.to_csv(output / f"{name}_label_alignment.csv", index=False)
        print(f"Aligned {name}: {len(frame)} observations, {int(frame.failure.sum())} failure rows", flush=True)
        if name == "pilot":
            continue
        settings = ("init_state_id", "flow_noise_seed") if name == "q0" else ("init_state_id",)
        for setting in settings:
            with np.load(output / f"{name}_{setting}.npz", allow_pickle=False) as archive:
                scores, flags = archive["scores"], archive["alarms"]
                methods, budgets = archive["method_names"].astype(str).tolist(), archive["budgets"]
            scopes = [("all", np.arange(len(frame)))] + list(frame.groupby("task").groups.items())
            for scope, rows in scopes:
                rows = np.asarray(rows)
                local = frame.loc[rows].reset_index(drop=True)
                y = local.failure.to_numpy(bool)
                for m, method in enumerate(methods):
                    rankings.append({"dataset": name, "setting": setting, "scope": scope, "method": method,
                                     **metrics(local, scores[rows, m])})
                    for b, budget in enumerate(budgets):
                        alarms.append({"dataset": name, "setting": setting, "scope": scope, "method": method,
                                       "budget": budget, **alarm_metrics(y, flags[b, rows, m])})
            if name == "q0":
                for comparison in paired_representation_bootstrap(frame, scores, methods):
                    comparisons.append({"setting": setting, **comparison})
    pd.DataFrame(rankings).to_csv(output / "ranking_metrics.csv", index=False)
    pd.DataFrame(alarms).to_csv(output / "alarm_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "representation_comparisons.csv", index=False)
    evaluate_pilot(output, datasets["pilot"])

    needed = {(r.task, r.run_id, int(r.episode)) for name in ("q34", "pilot")
              for r in datasets[name].itertuples() if r.failure}
    physical = {}
    with (PHYSICAL / "failures.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            key = (record["suite"] + "/" + record["task_name"], record["run_id"], record["episode_index"])
            if key in needed:
                if key in physical:
                    raise ValueError("duplicate physical timeline")
                physical[key] = record
    for name in ("q34", "pilot"):
        physics_rows(datasets[name], physical).to_csv(output / f"{name}_physical_timing.csv", index=False)
    write_json(output / "evaluation_summary.json", {
        "q0_episodes": len(datasets["q0"]), "q0_failures": int(datasets["q0"].failure.sum()),
        "q0_tasks": datasets["q0"].task.nunique(),
        "q0_checkpoints": datasets["q0"].checkpoint_sha256.nunique(),
        "q34_episodes": len(datasets["q34"]), "q34_failures": int(datasets["q34"].failure.sum()),
        "pilot_rows": len(datasets["pilot"]), "pilot_qualified_rows": int(datasets["pilot"].qualified.sum()),
        "bootstrap_scope": "init-state resampling, conditional on the captured tasks, noise seeds and frozen predictions",
        "probability_calibration": False, "new_rollouts": False,
        "labels_sha256": digest(PHYSICAL / "episodes.csv"),
        "physical_sha256": digest(PHYSICAL / "failures.jsonl"),
        "source_sha256": digest(Path(__file__)),
        "prediction_manifest_sha256": digest(output / "sealed_manifest.json"),
    })
    ranks = pd.DataFrame(rankings)
    print(ranks[(ranks.dataset == "q0") & (ranks.setting == "init_state_id") & (ranks.scope == "all") &
                ranks.method.isin(("routed_d9__knn5", "hidden_d9__knn5", "shared_d9__knn5",
                                   "cosine_high__flowmean", "action__knn5"))].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
