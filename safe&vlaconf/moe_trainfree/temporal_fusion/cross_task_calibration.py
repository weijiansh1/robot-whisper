"""Recalibrate kNN against A trajectories whose task is absent from the bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusion import (HERE, ROOT, METHODS, ALPHAS, dynamics, v8_heads, ReferenceScorer,
                    conformal_threshold, first_alarm, trajectory_peak)
from core import digest, write_json
from run_fusion import load_npz
from analyze_fusion import timing_metrics, distributions

BASES = ("knn10", "knn12_v8")
NAMES = ("knn10_cross_task", "knn12_cross_task")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round7_temporal_fusion")
    args = parser.parse_args()
    parent = args.input.resolve()
    output = parent / "cross_task_calibration"
    output.mkdir(exist_ok=False)
    for name in ("profiles", "predictions"):
        (output / name).mkdir()
    manifest = json.loads((parent / "sealed_manifest.json").read_text())
    frame = pd.read_csv(parent / "index.csv")
    v7 = load_npz(HERE.parent / "results/round3_safe/v7/v7_inputs.npz")
    raw = load_npz(parent / "v8_inputs.npz")["raw"]
    source_paths = (Path(__file__), HERE / "CROSS_TASK_CALIBRATION_ZH.md")
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    records, audits = [], []
    for info in manifest["folds"]:
        fold = info["fold"]
        profile_path, pred_path = parent / "profiles" / f"{fold}.npz", parent / "predictions" / f"{fold}.npz"
        for path in (profile_path, pred_path):
            assert digest(path) == manifest["artifacts"][str(path.relative_to(parent))]
        profile, original = load_npz(profile_path), load_npz(pred_path)
        rows = original["calibration_rows"]
        cal = frame.iloc[rows].reset_index(drop=True)
        dynamic, _ = dynamics(v7["mobility"][rows], v7["acceleration"][rows], v7["periodicity"][rows], float(profile["periodicity_scale"]))
        d = (dynamic.astype(np.float64)-profile["dynamic_center"]) / profile["dynamic_scale"]
        h = (v8_heads(raw[rows]).astype(np.float64)-profile["v8_center"]) / profile["v8_scale"]
        augmented = np.concatenate((d, h), axis=-1)
        cal_scores = np.full((2, len(rows), 52), np.nan, np.float32)
        bank_tasks = frame.iloc[profile["success_global_rows"]].task.to_numpy()
        for task, positions in cal.groupby("task").indices.items():
            keep = bank_tasks != task
            assert 20 <= keep.sum() < len(keep)
            for mi, (bank, values) in enumerate((("success_dynamic", d), ("success_augmented", augmented))):
                scorer = ReferenceScorer({"cross_task_bank": profile[bank][keep]})
                block = values[positions]
                valid = np.isfinite(block).all(-1)
                result = np.full(block.shape[:2], np.nan, np.float32)
                result[valid] = scorer.neighbors("cross_task_bank", block[valid])[0].mean(-1)
                cal_scores[mi, positions] = result
                valid_points = np.argwhere(valid)
                pi, query = valid_points[len(valid_points)//2]
                direct = np.linalg.norm(profile[bank][keep]-block[pi, query], axis=1)
                np.testing.assert_allclose(result[pi, query], np.sort(direct)[:20].mean(), rtol=2e-6, atol=2e-7)
            audits.append(dict(fold=fold, task=task, retained_reference_points=int(keep.sum()), excluded_reference_points=int((~keep).sum())))
        tests = original["scores"][[METHODS.index(name) for name in BASES]]
        thresholds = np.empty((2, len(ALPHAS), 2))
        first = np.empty((*thresholds.shape, len(original["test_rows"])), np.int16)
        success = original["calibration_labels"] == 0
        ids = (cal.loc[success, "task"] + "|" + cal.loc[success, "init_state_id"].astype(str)).to_numpy()
        for mi, name in enumerate(NAMES):
            old_mi = METHODS.index(BASES[mi])
            finite = np.isfinite(cal_scores[mi])
            assert np.all(cal_scores[mi][finite] + 2e-6 >= original["calibration_scores"][old_mi][finite])
            peaks = trajectory_peak(cal_scores[mi, success])
            grouped = np.asarray([peaks[ids == group].max() for group in np.unique(ids)])
            for ki, units in enumerate((peaks, grouped)):
                for ai, alpha in enumerate(ALPHAS):
                    tau, rank = conformal_threshold(units, alpha)
                    assert tau + 2e-6 >= original["thresholds"][ki, ai, old_mi]
                    thresholds[ki, ai, mi] = tau
                    new = first_alarm(tests[mi], tau)
                    old = original["first"][ki, ai, old_mi]
                    assert not ((new >= 0) & ((old < 0) | (new < old))).any()
                    first[ki, ai, mi] = new
                    profile["thresholds"][ki, ai, old_mi] = tau
                    records.append(dict(fold=fold, method=name, calibration=("episode", "task_init")[ki],
                        alpha=alpha, threshold=tau, original_threshold=original["thresholds"][ki, ai, old_mi],
                        rank=rank, units=len(units)))
        profile["calibration_variant"] = np.asarray("exclude calibration episode task from reference bank")
        profile["cross_task_calibrated_methods"] = np.asarray(BASES)
        np.savez_compressed(output / "profiles" / f"{fold}.npz", **profile)
        np.savez_compressed(output / "predictions" / f"{fold}.npz", scores=tests, calibration_scores=cal_scores,
            methods=np.asarray(NAMES), alphas=np.asarray(ALPHAS), thresholds=thresholds, first=first,
            test_rows=original["test_rows"], test_unseen=original["test_unseen"],
            calibration_rows=rows, calibration_labels=original["calibration_labels"])
        print(f"CROSS-TASK CALIBRATION SEALED {fold}", flush=True)
    pd.DataFrame(records).to_csv(output / "thresholds.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "bank_exclusions.csv", index=False)
    write_json(output / "sealed_manifest.json", dict(sources=sources, parent_manifest_sha256=digest(parent / "sealed_manifest.json"),
        methods=NAMES, new_model_training=False, test_outcomes_used_for_calibration=False,
        historically_explored_data=True, direct_distance_oracle_checks=len(audits)*2,
        calibration_distance_and_alarm_monotonicity_verified=True,
        artifacts={str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}))
    # Calibration is complete before outcomes and historical alarm vectors are loaded.
    outcomes = pd.read_csv(parent / "outcome_alignment.csv")
    historical = pd.read_csv(parent / "historical_episode_alarms.csv")
    historical = historical.pivot(index="global_row", columns="method", values="first_alarm_query")
    full_records, matched_records, metrics = [], [], []
    for info in manifest["folds"]:
        fold = info["fold"]
        original = load_npz(parent / "predictions" / f"{fold}.npz")
        new = load_npz(output / "predictions" / f"{fold}.npz")
        part = outcomes.iloc[new["test_rows"]].reset_index(drop=True)
        part["global_row"] = new["test_rows"]
        part["fold"] = fold
        part["scope"] = np.where(new["test_unseen"], "unseen", "seen")
        ai = int(np.flatnonzero(np.isclose(new["alphas"], .05))[0])
        firsts = {name: new["first"][1, ai, mi] for mi, name in enumerate(NAMES)}
        firsts.update({name: original["first"][1, ai, METHODS.index(name)] for name in BASES})
        covered = part.global_row.isin(historical.index).to_numpy()
        matched = {name: values[covered] for name, values in firsts.items()}
        for anchor in ("v8_frozen", "v82_frozen"):
            old = historical.loc[part.loc[covered, "global_row"], anchor].to_numpy()
            matched[anchor] = old
            for name in NAMES:
                k = firsts[name][covered]
                matched[f"{name}_or_{anchor}"] = np.where(k < 0, old, np.where(old < 0, k, np.minimum(k, old)))
        for sample, sample_part, streams, destination in (("all_test_rows", part, firsts, full_records),
                    ("historical_coverage", part.loc[covered].reset_index(drop=True), matched, matched_records)):
            for name, first in streams.items():
                rec = sample_part[["fold", "global_row", "suite", "task", "episode", "init_state_id", "noise_seed", "length", "failure", "scope"]].copy()
                rec["method"], rec["first_alarm_query"] = name, first
                destination.append(rec)
                for scope, positions in rec.groupby("scope").indices.items():
                    metrics.append(dict(sample=sample, fold=fold, suite=info["suite"], scope=scope, method=name,
                                        **timing_metrics(rec.iloc[positions], first[positions])))
    full = pd.concat(full_records, ignore_index=True)
    matched = pd.concat(matched_records, ignore_index=True)
    full.to_csv(output / "episode_decisions.csv", index=False)
    matched.to_csv(output / "matched_episode_decisions.csv", index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "alarm_metrics.csv", index=False)
    fields = ["recall", "fpr", "recall_by_q14", "recall_lead4"]
    summary = metrics.groupby(["sample", "scope", "method"])[fields].mean().reset_index()
    summary.to_csv(output / "fold_macro_summary.csv", index=False)
    distributions(full, output)
    events = pd.read_csv(parent / "physical_timing.csv")[["global_row", "drop_goal_release"]].drop_duplicates()
    physical = matched.merge(events, on="global_row", validate="many_to_one")
    physical = physical.loc[physical.scope.eq("unseen") & physical.drop_goal_release.notna()]
    phys_rows = []
    for method, group in physical.groupby("method"):
        f, e = group.first_alarm_query, group.drop_goal_release
        phys_rows.append(dict(method=method, events=len(group), before=int(((f >= 0) & (f < e)).sum()),
            at_event=int((f == e).sum()), after=int((f > e).sum()), missed=int((f < 0).sum())))
    pd.DataFrame(phys_rows).to_csv(output / "physical_summary.csv", index=False)
    print(summary.loc[summary.scope.eq("unseen")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
