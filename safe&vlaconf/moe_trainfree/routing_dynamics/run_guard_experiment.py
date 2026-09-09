"""Fixed retrospective integration experiment; calibration precedes B metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from encoder import EncoderConfig, FEATURE_NAMES
from guard import (ALPHAS, BRANCHES, CONFIRMATIONS, HEADS, KINDS, METHOD_BRANCHES,
                   SCHEMA, AlarmState, PeakBank, combine_tails, dynamics_scores, first_trigger)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v82_validation"))
from monitor import auc, union
from run_analysis import BASE, ROOT, RUN_B, bootstrap_indices, digest, rate_interval, state_splits, write_json

FEATURE_DIR = BASE / "routing_dynamics_20260908"
LEGACY_DIR = BASE / "v82_validation_20260908"
METHODS = tuple(METHOD_BRANCHES)
S05 = "libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
ACTION_CAPS = dict(libero_goal=300, libero_long=520, libero_object=280, libero_spatial=220)


def verified_inputs():
    manifests = [(FEATURE_DIR, "final_verification.json", ("features.npz", "index.csv")),
                 (LEGACY_DIR, "analysis_verification.json",
                  ("profiles.json", "crossfit_predictions.npz", "frozen_first_alarms.csv",
                   *(f"fold_{f}_calibration.npz" for f in range(5))))]
    hashes = {}
    for directory, name, files in manifests:
        manifest = json.loads((directory / name).read_text())
        for file in files:
            path = directory / file
            actual = digest(path)
            assert actual == manifest["artifacts"][file], path
            hashes[str(path)] = actual
    frame = pd.read_csv(FEATURE_DIR / "index.csv")
    np.testing.assert_array_equal(frame.global_row, np.arange(32000))
    with np.load(FEATURE_DIR / "features.npz", allow_pickle=False) as z:
        features, valid = z["features"], z["valid"]
        np.testing.assert_array_equal(z["global_rows"], frame.global_row)
        np.testing.assert_array_equal(z["names"], FEATURE_NAMES)
    np.testing.assert_array_equal(valid, np.arange(valid.shape[1])[None] < frame.length.to_numpy()[:, None])
    assert int(valid.sum()) == 508023
    return frame, features, valid, hashes


def calibrate(frame, dynamics, output):
    references = json.loads((LEGACY_DIR / "profiles.json").read_text())
    threshold_rows, assignments, profiles = [], [], []
    for fold, ref, cal, test in state_splits(frame):
        old = references[fold]
        for field, expected in (("reference_rows", ref), ("calibration_rows", cal), ("test_rows", test)):
            np.testing.assert_array_equal(old[field], expected)
        with np.load(LEGACY_DIR / f"fold_{fold}_calibration.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["rows"], cal)
            np.testing.assert_array_equal(z["failure"], frame.iloc[cal].failure)
            legacy = z["scores"][list(z["methods"]).index("v82")]
        success = ~frame.iloc[cal].failure.to_numpy(bool)
        cal_frame = frame.iloc[cal].loc[success]
        group_index, group_names = pd.factorize(pd.MultiIndex.from_frame(cal_frame[["task", "init_state_id"]]), sort=True)
        branches = dict(v82=legacy, **{name: value[cal] for name, value in dynamics.items()})
        banks = {kind: {} for kind in KINDS}
        for name, score in branches.items():
            peaks = np.where(np.isfinite(score[success]), score[success], -np.inf).max(axis=1)
            groups = np.full(len(group_names), -np.inf)
            np.maximum.at(groups, group_index, peaks)
            for kind, units in zip(KINDS, (peaks, groups)):
                bank = PeakBank(units)
                banks[kind][name] = bank.to_list()
                for method, selected in METHOD_BRANCHES.items():
                    if name not in selected:
                        continue
                    for alpha in ALPHAS:
                        threshold, rank = bank.threshold(alpha / len(selected))
                        threshold_rows.append(dict(fold=fold, calibration=kind, method=method, branch=name,
                                                   alpha=alpha, branch_alpha=alpha / len(selected),
                                                   threshold=threshold, rank=rank, units=len(units),
                                                   resolution=1 / (len(units) + 1), attainable=rank <= len(units),
                                                   no_exposure_units=int(np.isneginf(units).sum()),
                                                   calibration_exceedances=int((units > threshold).sum())))
        profile = dict(schema=SCHEMA, fold=fold, encoder_config=asdict(EncoderConfig()),
                       confirmations=CONFIRMATIONS, default_method="v82_integrated",
                       default_calibration="task_init", default_alpha=.01,
                       checkpoints=sorted(frame.iloc[ref].checkpoint.unique()),
                       v82_profile=dict(periodicity_scale=old["profile"]["periodicity_scale"],
                                        heads={name: old["profile"]["heads"][name] for name in HEADS}),
                       banks=banks, reference_rows=ref, calibration_rows=cal, test_rows=test,
                       retrospective=True, no_classifier_training=True, equal_branch_allocation=True)
        path = output / f"fold_{fold}_profile.json"
        write_json(path, profile)
        profiles.append(json.loads(path.read_text()))
        for role, rows in (("reference", ref), ("calibration", cal), ("test", test)):
            assignments.extend(dict(fold=fold, role=role, global_row=int(row)) for row in rows)
        print(f"CALIBRATED fold={fold} success_episodes={success.sum()} success_groups={len(group_names)}", flush=True)
    pd.DataFrame(threshold_rows).to_csv(output / "calibration_thresholds.csv", index=False)
    pd.DataFrame(assignments).to_csv(output / "split_assignments.csv", index=False)
    sealed = {p.name: digest(p) for p in sorted(output.glob("fold_*_profile.json"))}
    sealed["calibration_thresholds.csv"] = digest(output / "calibration_thresholds.csv")
    sealed["execution_contract.json"] = digest(output / "execution_contract.json")
    write_json(output / "calibration_seal.json", dict(phase="before_B_scores_and_metrics", artifacts=sealed))
    return profiles


def predict(frame, valid, dynamics, profiles, output):
    b_rows = np.flatnonzero(frame.run_id.eq(RUN_B))
    b = frame.iloc[b_rows].reset_index(drop=True)
    assert len(b) == 16000 and b.failure.sum() == 564 and (~b.failure).sum() == 15436
    assert int(valid[b_rows].sum()) == 254301
    inverse = np.full(len(frame), -1, int)
    inverse[b_rows] = np.arange(len(b_rows))
    with np.load(LEGACY_DIR / "crossfit_predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(b_rows, z["global_rows"])
        mi = list(z["methods"]).index("v82")
        old_scores, old_first = z["scores"][mi], z["first"][:, :, mi]
        np.testing.assert_array_equal(z["kinds"], KINDS)
        np.testing.assert_array_equal(z["alphas"], ALPHAS)
        np.testing.assert_array_equal(z["valid"], valid[b_rows])
    tails = np.full((len(KINDS), len(METHODS), len(b), valid.shape[1]), np.nan)
    branch_tails = np.full((len(KINDS), len(BRANCHES), len(b), valid.shape[1]), np.nan)
    first = np.full((len(KINDS), len(ALPHAS), len(METHODS), len(b)), -2, np.int16)
    fold_ids = np.full(len(b), -1, np.int8)
    for profile in profiles:
        rows = np.asarray(profile["test_rows"])
        positions = inverse[rows]
        assert (fold_ids[positions] < 0).all()
        fold_ids[positions] = profile["fold"]
        branches = dict(v82=old_scores[positions], **{name: value[rows] for name, value in dynamics.items()})
        for ki, kind in enumerate(KINDS):
            current = {name: PeakBank.from_list(profile["banks"][kind][name]).tail(value)
                       for name, value in branches.items()}
            for bi, name in enumerate(BRANCHES):
                branch_tails[ki, bi, positions] = current[name]
            for mi, method in enumerate(METHODS):
                tail = combine_tails(current, method)
                tails[ki, mi, positions] = tail
                for ai, alpha in enumerate(ALPHAS):
                    first[ki, ai, mi, positions] = first_trigger(tail, valid[rows], alpha)
    assert (fold_ids >= 0).all() and (first >= -1).all()
    np.testing.assert_array_equal(first[:, :, METHODS.index("v82_reference")], old_first)
    assert np.isnan(tails[:, :, ~valid[b_rows]]).all()
    frozen_table = pd.read_csv(LEGACY_DIR / "frozen_first_alarms.csv").set_index("global_row")
    frozen = frozen_table.loc[b_rows, "v82_frozen"].to_numpy(np.int16)
    assert int(((frozen >= 0) & b.failure).sum()) == 475
    assert int(((frozen >= 0) & ~b.failure).sum()) == 99
    np.savez_compressed(output / "predictions.npz", tails=tails, branch_tails=branch_tails, first=first,
                        methods=np.asarray(METHODS), branches=np.asarray(BRANCHES), kinds=np.asarray(KINDS),
                        alphas=np.asarray(ALPHAS), global_rows=b_rows, fold_ids=fold_ids,
                        valid=valid[b_rows], frozen_first=frozen, raw_v82_scores=old_scores)
    b.to_csv(output / "test_index.csv", index=False)
    print("PREDICTED B=16000; v8.2 first alarms match all 8 calibration/budget settings", flush=True)
    return b, tails, branch_tails, first, frozen, old_scores


def settings(first, frozen):
    yield "original", np.nan, "v82_frozen", frozen
    for ki, kind in enumerate(KINDS):
        for ai, alpha in enumerate(ALPHAS):
            for mi, method in enumerate(METHODS):
                yield kind, alpha, method, first[ki, ai, mi]
            yield kind, alpha, "v82_frozen_plus_dynamics", union(frozen, first[ki, ai, METHODS.index("dynamics_only")])


def alarm_tables(b, tails, branch_tails, first, frozen, output):
    metrics, curves, paired, state_rows, episode_rows = [], [], [], [], []
    y = b.failure.to_numpy(bool)
    caps = b.suite.map(ACTION_CAPS).to_numpy()
    scopes = [("all", np.arange(len(b)))]
    scopes += [(str(name), part.index.to_numpy()) for name, part in b.groupby("suite")]
    scopes += [(str(name), part.index.to_numpy()) for name, part in b.groupby("task")]
    for kind, alpha, method, alarm in settings(first, frozen):
        alarm = np.asarray(alarm, dtype=np.int64)
        for scope, positions in scopes:
            current, fired = alarm[positions], alarm[positions] >= 0
            labels = y[positions]
            tp, fp = int((fired & labels).sum()), int((fired & ~labels).sum())
            failures, successes = int(labels.sum()), int((~labels).sum())
            percent = 1000 * current[fired & labels] / caps[positions][fired & labels]
            row = dict(calibration=kind, alpha=alpha, method=method, scope=scope, tp=tp, fp=fp,
                       failures=failures, successes=successes, recall=tp / failures if failures else np.nan,
                       fpr=fp / successes if successes else np.nan, precision=tp / (tp + fp) if tp + fp else np.nan,
                       median_failure_q=float(np.median(current[fired & labels])) if tp else np.nan,
                       median_failure_cap_percent=float(np.median(percent)) if tp else np.nan)
            for fraction in (.5, .6):
                early = fired & (10 * current <= fraction * caps[positions])
                row[f"tp_by_cap_{int(100*fraction)}"] = int((early & labels).sum())
                row[f"fp_by_cap_{int(100*fraction)}"] = int((early & ~labels).sum())
            if scope == "all":
                row.update(rate_interval(b, fired, labels))
            metrics.append(row)
        for q in range(tails.shape[-1]):
            fired = (alarm >= 0) & (alarm <= q)
            curves.append(dict(calibration=kind, alpha=alpha, method=method, query=q,
                               tp=int((fired & y).sum()), fp=int((fired & ~y).sum()),
                               recall=float((fired & y).sum() / y.sum()), fpr=float((fired & ~y).sum() / (~y).sum())))
        if alpha == .01 or kind == "original":
            for i, row in b.iterrows():
                cause = ""
                if method in METHODS and alarm[i] >= 0:
                    ki = KINDS.index(kind)
                    names = METHOD_BRANCHES[method]
                    cause = "+".join(name for name in names
                                     if len(names) * branch_tails[ki, BRANCHES.index(name), i, alarm[i]] <= alpha)
                episode_rows.append(dict(global_row=row.global_row, task=row.task, suite=row.suite,
                                         episode=row.episode, init_state_id=row.init_state_id, failure=row.failure,
                                         calibration=kind, alpha=alpha, method=method, first_alarm=int(alarm[i]),
                                         first_cause=cause,
                                         alarm_cap_percent=1000 * alarm[i] / caps[i] if alarm[i] >= 0 else np.nan))
    for ki, kind in enumerate(KINDS):
        for ai, alpha in enumerate(ALPHAS):
            reference = first[ki, ai, METHODS.index("v82_reference")]
            comparisons = [("v82_integrated", first[ki, ai, METHODS.index("v82_integrated")], "v82_reference", reference),
                           ("dynamics_only", first[ki, ai, METHODS.index("dynamics_only")], "v82_reference", reference),
                           ("v82_frozen_plus_dynamics", union(frozen, first[ki, ai, METHODS.index("dynamics_only")]), "v82_frozen", frozen)]
            for method, new, baseline, old in comparisons:
                for scope, ix in scopes:
                    a, o, labels = new[ix] >= 0, old[ix] >= 0, y[ix]
                    shared = a & o & labels
                    delta = new[ix][shared] - old[ix][shared]
                    paired.append(dict(calibration=kind, alpha=alpha, method=method, baseline=baseline, scope=scope,
                                       gained_tp=int((a & ~o & labels).sum()), lost_tp=int((~a & o & labels).sum()),
                                       added_fp=int((a & ~o & ~labels).sum()), removed_fp=int((~a & o & ~labels).sum()),
                                       shared_tp=int(shared.sum()), earlier=int((delta < 0).sum()),
                                       same=int((delta == 0).sum()), later=int((delta > 0).sum()),
                                       median_query_delta=float(np.median(delta)) if len(delta) else np.nan))
        for mi, method in enumerate(METHODS):
            for i, row in b.iterrows():
                state = AlarmState(.01)
                ever_cleared, alarm_then_cleared = False, False
                for tail in tails[ki, mi, i, :row.length]:
                    result = state.update(tail)
                    ever_cleared |= result["signal_cleared"]
                    alarm_then_cleared |= result["signal_cleared"] and result["ever_alarm"]
                assert state.first_alarm_query == first[ki, ALPHAS.index(.01), mi, i]
                state_rows.append(dict(global_row=row.global_row, task=row.task, episode=row.episode,
                                       calibration=kind, method=method, failure=row.failure, final_state=state.state,
                                       ever_alarm=state.first_alarm_query >= 0, first_alarm=state.first_alarm_query,
                                       ever_signal_cleared=ever_cleared, alarm_then_cleared=alarm_then_cleared))
    for name, records in (("alarm_metrics", metrics), ("cumulative_curves", curves), ("paired_changes", paired),
                          ("episode_alarms", episode_rows), ("current_state_audit", state_rows)):
        pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)
    episodes = pd.DataFrame(episode_rows)
    episodes.loc[episodes.task.eq(S05)].to_csv(output / "s05_alarms.csv", index=False)
    return pd.DataFrame(metrics)


def auroc_tables(b, tails, raw_v82, output):
    records = []
    score = -np.log(tails)
    for (suite, task), part in b.groupby(["suite", "task"]):
        rows, y = part.index.to_numpy(), part.failure.to_numpy(bool)
        for ki, kind in enumerate(KINDS):
            reference = score[ki, METHODS.index("v82_reference"), rows]
            for mi, method in enumerate(METHODS):
                values = score[ki, mi, rows]
                for q in range(7, score.shape[-1]):
                    available = np.isfinite(values[:, q]) & np.isfinite(reference[:, q]) & np.isfinite(raw_v82[rows, q])
                    labels = y[available]
                    if not labels.any() or labels.all():
                        continue
                    new = auc(labels, values[available, q])
                    ref = auc(labels, reference[available, q])
                    records.append(dict(calibration=kind, method=method, suite=suite, task=task, query=q,
                                        auc=new, reference_auc=ref, delta=new - ref,
                                        raw_v82_auc=auc(labels, raw_v82[rows[available], q]),
                                        failures=int(labels.sum()), successes=int((~labels).sum())))
    table = pd.DataFrame(records)
    table.to_csv(output / "query_auroc.csv", index=False)
    task_tables, summary = [], []
    for window, selected in (("q7_13", table.loc[table["query"].between(7, 13)]), ("all_comparable_queries", table)):
        grouped = selected.groupby(["calibration", "method", "suite", "task"], as_index=False).agg(
            auc=("auc", "mean"), reference_auc=("reference_auc", "mean"), delta=("delta", "mean"),
            raw_v82_auc=("raw_v82_auc", "mean"), queries=("query", "nunique"),
            first_query=("query", "min"), last_query=("query", "max"))
        grouped["window"] = window
        task_tables.append(grouped)
        for (kind, method), part in grouped.groupby(["calibration", "method"]):
            for scope, rows in [("all", part), ("excluding_S05", part.loc[part.task.ne(S05)])]:
                rows = rows.sort_values(["suite", "task"])
                draws = bootstrap_indices(np.arange(len(rows)), rows.suite)
                for metric in ("auc", "reference_auc", "delta", "raw_v82_auc"):
                    values = rows[metric].to_numpy()
                    lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
                    summary.append(dict(calibration=kind, method=method, window=window, scope=scope, metric=metric,
                                        estimate=float(values.mean()), lo=float(lo), hi=float(hi), tasks=len(rows),
                                        first_query=int(rows.first_query.min())))
    pd.concat(task_tables, ignore_index=True).to_csv(output / "task_auroc.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "auroc_summary.csv", index=False)


def plots(output):
    data = pd.read_csv(output / "cumulative_curves.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = dict(v82_reference="#77598c", dynamics_only="#227d73", v82_integrated="#bd513c")
    for ki, kind in enumerate(KINDS):
        for mi, metric in enumerate(("tp", "fp")):
            ax = axes[mi, ki]
            for method in METHODS:
                part = data.loc[data.calibration.eq(kind) & data.alpha.eq(.01) & data.method.eq(method)]
                ax.step(part["query"], part[metric], where="post", color=colors[method], label=method)
            ax.set_title(f"{kind}, nominal alpha=1%")
            ax.set_ylabel("Failures detected (of 564)" if metric == "tp" else "Successes ever alarmed (of 15436)")
            ax.set_xlabel("Observed query index")
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Fixed train-free integration on B; retrospective evaluation")
    fig.tight_layout(rect=(0, 0, 1, .96))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"alarm_curves.{suffix}", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "routing_guard_20260908")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame, features, valid, hashes = verified_inputs()
    source_names = ("encoder.py", "guard.py", "run_guard_experiment.py", "INTEGRATION_PROTOCOL_ZH.md")
    write_json(output / "execution_contract.json", dict(
        retrospective=True, no_classifier_training=True, b_parameter_selection=False,
        primary=dict(method="v82_integrated", calibration="task_init", alpha=.01),
        methods=METHOD_BRANCHES, alphas=ALPHAS, kinds=KINDS,
        encoder_config=asdict(EncoderConfig()), confirmations=CONFIRMATIONS,
        input_hashes=hashes, sources={name: digest(HERE / name) for name in source_names},
        routing_only=True, inherited_v82_threshold_slope=.0015, duration_baseline_computed=False))
    dynamics = dynamics_scores(features, valid)
    del features
    profiles = calibrate(frame, dynamics, output)
    b, tails, branch_tails, first, frozen, old_scores = predict(frame, valid, dynamics, profiles, output)
    metrics = alarm_tables(b, tails, branch_tails, first, frozen, output)
    print("ALARM METRICS and historical-state audit complete", flush=True)
    auroc_tables(b, tails, old_scores, output)
    plots(output)
    seal = json.loads((output / "calibration_seal.json").read_text())
    for name, expected in seal["artifacts"].items():
        assert digest(output / name) == expected, name
    write_json(output / "experiment_verification.json", dict(
        calibration_seal_unchanged=True, baseline_first_alarm_matches=2 * 4 * 16000,
        b_episodes=len(b), b_failures=int(b.failure.sum()), b_successes=int((~b.failure).sum()),
        every_B_episode_once=True, alarm_state_replays=2 * 3 * len(b),
        artifacts={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    main_rows = metrics.loc[metrics.scope.eq("all") & (metrics.alpha.eq(.01) | metrics.calibration.eq("original"))]
    print(main_rows[["calibration", "method", "tp", "fp", "recall", "fpr", "median_failure_q"]].to_string(index=False))


if __name__ == "__main__":
    main()
