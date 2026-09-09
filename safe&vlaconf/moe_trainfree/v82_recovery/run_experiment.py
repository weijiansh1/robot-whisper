"""Select recovery guards only on A, then evaluate every B episode once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from recovery import AlertState, HERE, METHODS, fit_profile, score_streams
from monitor import ALPHAS, ROOT, auc, conformal_threshold, cumulative_counts, first_alarm
from run_analysis import RUN_A, RUN_B, archive, bootstrap_indices, digest, rate_interval, write_json

BASE = HERE.parent / "results"
VALIDATION = BASE / "v82_validation_20260908"
KINDS = ("episode", "task_init")
ALL_METHODS = (*METHODS, "selected", "clock")
PRIMARY_KIND, PRIMARY_ALPHA = 1, ALPHAS.index(.01)


def split_roles(frame, fold):
    residue = frame.init_state_id.to_numpy() % 5
    a, b = frame.run_id.eq(RUN_A).to_numpy(), frame.run_id.eq(RUN_B).to_numpy()
    reference = np.flatnonzero(a & np.isin(residue, [(fold + 2) % 5, (fold + 3) % 5]))
    calibration = np.flatnonzero(a & (residue == (fold + 1) % 5))
    development = np.flatnonzero(a & (residue == (fold + 4) % 5))
    test = np.flatnonzero(b & (residue == fold))
    roles = (reference, calibration, development, test)
    identities = [set(zip(frame.iloc[r].task, frame.iloc[r].init_state_id)) for r in roles]
    assert not any(identities[i] & identities[j] for i in range(4) for j in range(i))
    return roles


def calibration_values(scores, frame):
    success = ~frame.failure.to_numpy(bool)
    peak = np.where(np.isfinite(scores[:, success]), scores[:, success], -np.inf).max(axis=2)
    current = frame.loc[success]
    group, names = pd.factorize(current.task + "|" + current.init_state_id.astype(str), sort=True)
    grouped = np.full((len(scores), len(names)), -np.inf)
    for i in range(len(scores)):
        np.maximum.at(grouped[i], group, peak[i])
    return peak, grouped


def calibrate(scores, frame, fold):
    threshold = np.full((2, len(ALPHAS), len(scores)), np.inf)
    watch = np.full((2, len(scores)), np.inf)
    rows = []
    names = (*METHODS, "clock")
    for ki, units in enumerate(calibration_values(scores, frame)):
        for mi, method in enumerate(names):
            watch[ki, mi] = conformal_threshold(units[mi], .1)[0]
            for ai, alpha in enumerate(ALPHAS):
                value, rank = conformal_threshold(units[mi], alpha)
                threshold[ki, ai, mi] = value
                assert watch[ki, mi] <= value
                rows.append(dict(fold=fold, method=method, calibration=KINDS[ki], alpha=alpha,
                                 threshold=value, watch_threshold=watch[ki, mi], rank=rank,
                                 units=units.shape[1], exceedances=int((units[mi] > value).sum())))
    return threshold, watch, rows


def choose(first, failure):
    records = []
    for method, values in zip(METHODS, first):
        fired = values >= 0
        utility = np.where(fired & failure, np.exp(-np.maximum(values - 7, 0) / 12.), 0.)
        records.append(dict(method=method, tp=int((fired & failure).sum()), fp=int((fired & ~failure).sum()),
                            utility=float(utility[failure].mean()) if failure.any() else 0.))
    base = records[0]
    eligible = []
    for i, record in enumerate(records):
        record["eligible"] = record["tp"] >= base["tp"] and record["fp"] <= base["fp"]
        if record["eligible"]:
            eligible.append(i)
    chosen = max(eligible, key=lambda i: (records[i]["utility"], records[i]["tp"], -records[i]["fp"], -i))
    for i, record in enumerate(records):
        record["selected"] = i == chosen
    return chosen, records


def process_split(frame, cache, raw, roles, fold, output):
    reference, calibration, development, test = roles
    rows = np.concatenate(roles)
    nr, nc, nd = len(reference), len(calibration), len(development)
    local = {k: value[rows] for k, value in cache.items()}
    extra = raw[rows]
    profile = fit_profile(local, extra, np.arange(nr))
    scores = score_streams(local, extra, profile)
    clock = np.broadcast_to(np.arange(raw.shape[1])[None], local["valid"].shape).astype(np.float32).copy()
    clock[~local["valid"]] = np.nan
    bank = np.concatenate((scores, clock[None]))
    threshold, watch, calibration_rows = calibrate(bank[:, nr:nr + nc], frame.iloc[calibration], fold)
    dev_first = np.full((2, len(ALPHAS), len(METHODS), nd), -1, np.int16)
    selected = np.zeros((2, len(ALPHAS)), np.int16)
    selection_rows = []
    for ki, kind in enumerate(KINDS):
        for ai, alpha in enumerate(ALPHAS):
            for mi in range(len(METHODS)):
                dev_first[ki, ai, mi] = first_alarm(bank[mi, nr + nc:nr + nc + nd],
                                                    local["valid"][nr + nc:nr + nc + nd], threshold[ki, ai, mi])
            selected[ki, ai], records = choose(dev_first[ki, ai], frame.iloc[development].failure.to_numpy(bool))
            selection_rows.extend(dict(fold=fold, calibration=kind, alpha=alpha, **record) for record in records)
    mi = int(selected[PRIMARY_KIND, PRIMARY_ALPHA])
    online_profile = dict(profile, method=METHODS[mi],
                          threshold=threshold[PRIMARY_KIND, PRIMARY_ALPHA, mi],
                          watch_threshold=watch[PRIMARY_KIND, mi], checkpoints=sorted(frame.iloc[reference].checkpoint.unique()),
                          calibration=KINDS[PRIMARY_KIND], alpha=ALPHAS[PRIMARY_ALPHA], fold=fold,
                          reference_rows=reference, calibration_rows=calibration, development_rows=development,
                          test_rows=test, description="Current routing state, not a physical recovery classifier")
    write_json(output / ("profile_%s.json" % fold), online_profile)
    np.savez_compressed(output / ("design_%s.npz" % fold), calibration_rows=calibration, development_rows=development,
                        calibration_scores=bank[:, nr:nr + nc], development_first=dev_first,
                        thresholds=threshold, watch_thresholds=watch, selected=selected)
    return dict(scores=bank[:, nr + nc + nd:], threshold=threshold, watch=watch, selected=selected,
                calibration_rows=calibration_rows, selection_rows=selection_rows)


def summarize(frame, first, output):
    rows, curves = [], []
    for ki, kind in enumerate(KINDS):
        for ai, alpha in enumerate(ALPHAS):
            for mi, method in enumerate(ALL_METHODS):
                values = first[ki, ai, mi]
                for scope, current in [("all", frame)] + list(frame.groupby("suite", sort=True)):
                    ix = current.index.to_numpy()
                    alarm = values[ix]
                    failure, length = current.failure.to_numpy(bool), current.length.to_numpy(int)
                    counts = cumulative_counts(alarm, failure, length, 51)
                    meta = dict(method=method, calibration=kind, alpha=alpha, scope=scope)
                    rows.append(dict(meta, **counts, recall=counts["tp"] / counts["failures"] if counts["failures"] else np.nan,
                                     fpr=counts["fp"] / counts["successes"],
                                     **rate_interval(current, alarm >= 0, failure)))
                    for q in range(52):
                        c = cumulative_counts(alarm, failure, length, q)
                        curves.append(dict(meta, query=q, **c, recall=c["tp"] / c["failures"] if c["failures"] else np.nan,
                                           fpr=c["fp"] / c["successes"]))
    pd.DataFrame(rows).to_csv(output / "alarm_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "cumulative_curves.csv", index=False)
    paired = []
    for ki, kind in enumerate(KINDS):
        for ai, alpha in enumerate(ALPHAS):
            old = first[ki, ai, 0]
            for mi, method in enumerate(ALL_METHODS[1:-1], 1):
                new = first[ki, ai, mi]
                for scope, current in [("all", frame)] + list(frame.groupby("suite", sort=True)):
                    ix = current.index.to_numpy()
                    a, b, failure = old[ix], new[ix], current.failure.to_numpy(bool)
                    shared = failure & (a >= 0) & (b >= 0)
                    counts = []
                    for _, group in current.groupby("task", sort=True):
                        g = group.index.to_numpy()
                        f = group.failure.to_numpy(bool)
                        changed = (new[g] >= 0).astype(int) - (old[g] >= 0)
                        counts.append((group.suite.iloc[0], group.task.iloc[0], changed[f].sum(), f.sum(),
                                       changed[~f].sum(), (~f).sum()))
                    count = np.asarray([x[2:] for x in counts], float)
                    draws = bootstrap_indices([x[1] for x in counts], [x[0] for x in counts])
                    total = count[draws].sum(axis=1)
                    delta = np.divide(total[:, [0, 2]], total[:, [1, 3]], out=np.full((len(total), 2), np.nan), where=total[:, [1, 3]] > 0)
                    ci = np.nanquantile(delta, [.025, .975], axis=0)
                    paired.append(dict(method=method, calibration=kind, alpha=alpha, scope=scope,
                                       gained_tp=int((failure & (a < 0) & (b >= 0)).sum()),
                                       lost_tp=int((failure & (a >= 0) & (b < 0)).sum()),
                                       added_fp=int((~failure & (a < 0) & (b >= 0)).sum()),
                                       removed_fp=int((~failure & (a >= 0) & (b < 0)).sum()),
                                       shared_tp=int(shared.sum()), earlier=int((shared & (b < a)).sum()),
                                       later=int((shared & (b > a)).sum()),
                                       median_delta_q=float(np.median(b[shared] - a[shared])) if shared.any() else np.nan,
                                       recall_delta_lo=ci[0, 0], recall_delta_hi=ci[1, 0], fpr_delta_lo=ci[0, 1], fpr_delta_hi=ci[1, 1]))
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)


def early_auc(frame, scores, output):
    rows = []
    names = (*METHODS, "selected")
    for task, current in frame.groupby("task", sort=True):
        failure = current.failure.to_numpy(bool)
        ix = current.index.to_numpy()
        for q in range(7, 14):
            active = current.length.to_numpy() > q
            for mi, method in enumerate(names):
                values = scores[mi, ix, q]
                good = active & np.isfinite(values)
                value = auc(failure[good], values[good])
                if np.isfinite(value):
                    rows.append(dict(task=task, suite=current.suite.iloc[0], query=q, method=method,
                                     auc=value, failures=int((good & failure).sum()), successes=int((good & ~failure).sum())))
    data = pd.DataFrame(rows)
    data.to_csv(output / "early_auc_strata.csv", index=False)
    task = data.groupby(["suite", "task", "method"], as_index=False).auc.mean()
    summary = []
    for scope, current in [("all", task)] + list(task.groupby("suite", sort=True)):
        for method, part in current.groupby("method", sort=True):
            values = part.auc.to_numpy()
            draws = bootstrap_indices(part.task, part.suite)
            ci = np.quantile(values[draws].mean(axis=1), [.025, .975])
            summary.append(dict(scope=scope, method=method, auc=values.mean(), lo=ci[0], hi=ci[1], tasks=len(part)))
    pd.DataFrame(summary).to_csv(output / "early_auc_summary.csv", index=False)


def state_diagnostics(frame, scores, first, thresholds, watch, fold_ids, choices, output):
    records, states = [], np.zeros((len(frame), 52), np.int8)
    for i, row in enumerate(frame.itertuples()):
        f = fold_ids[i]
        mi = choices[f, PRIMARY_KIND, PRIMARY_ALPHA]
        machine = AlertState(thresholds[f, PRIMARY_KIND, PRIMARY_ALPHA, mi], watch[f, PRIMARY_KIND, mi])
        clear, cleared_alarm, entries = [], [], 0
        for q in range(row.length):
            before = machine.state
            result = machine.update(scores[-1, i, q])
            states[i, q] = ("NORMAL", "WATCH", "ALARM").index(result["state"])
            if result["signal_cleared"]:
                clear.append(q)
                if before == "ALARM":
                    cleared_alarm.append(q)
            if result["state"] == "ALARM" and before != "ALARM":
                entries += 1
        assert machine.first_alarm_query == first[PRIMARY_KIND, PRIMARY_ALPHA, len(METHODS), i]
        records.append(dict(global_row=row.global_row, failure=bool(row.failure), suite=row.suite, task=row.task,
                            length=row.length, first_alarm=machine.first_alarm_query, first_watch=machine.first_watch_query,
                            last_state=machine.state, alarm_entries=entries, signal_clearances=len(clear),
                            alarm_clearances=len(cleared_alarm), first_alarm_clearance=cleared_alarm[0] if cleared_alarm else -1))
    data = pd.DataFrame(records)
    data.to_csv(output / "state_episodes.csv", index=False)
    np.savez_compressed(output / "states.npz", global_rows=frame.global_row.to_numpy(), states=states,
                        state_names=np.asarray(["NORMAL", "WATCH", "ALARM"]))
    events = pd.read_csv(VALIDATION / "physical_primary_events.csv")
    old = pd.read_csv(VALIDATION / "frozen_first_alarms.csv").set_index("global_row")
    candidates = events.loc[events.kind.eq("release_outside_goal") & ~events.failure].copy()
    candidates["old_first"] = old.loc[candidates.global_row, "v82_frozen"].to_numpy()
    candidates = candidates.loc[(candidates["query"] <= candidates.old_first) & (candidates.regrasp_q > candidates.old_first)]
    candidates.merge(data[["global_row", "first_alarm", "first_watch", "last_state", "alarm_clearances"]],
                     on="global_row", validate="one_to_one").to_csv(output / "recovery_candidates.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "v82_recovery_20260908")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_paths = (HERE / "PROTOCOL_ZH.md", HERE / "recovery.py", Path(__file__))
    contract = dict(protocol_sha256=digest(source_paths[0]), methods=METHODS, alphas=ALPHAS,
                    primary_calibration="task_init", primary_alpha=.01, test_cohort=RUN_B,
                    historical_exploration=True, new_rollouts=False, model_training=False,
                    success_fpr_keeps_cleared_alarms=True, source_hashes={p.name: digest(p) for p in source_paths})
    write_json(output / "execution_contract.json", contract)
    frame = pd.read_csv(VALIDATION / "index.csv")
    cache_path, raw_path = BASE / "round3_safe/v7/v7_inputs.npz", BASE / "round7_temporal_fusion/v8_inputs.npz"
    provenance = json.loads((VALIDATION / "input_verification.json").read_text())
    for p in (cache_path, raw_path):
        assert digest(p) == provenance["inputs"][str(p.relative_to(ROOT))]
    cache, raw = archive(cache_path), archive(raw_path)["raw"]
    np.testing.assert_array_equal(cache["valid"], np.arange(52)[None] < frame.length.to_numpy()[:, None])
    frame.to_csv(output / "index.csv", index=False)
    write_json(output / "input_verification.json", dict(previous_audit_sha256=digest(VALIDATION / "input_verification.json"),
                index_sha256=digest(VALIDATION / "index.csv"),
                inputs={str(p.relative_to(ROOT)): digest(p) for p in (cache_path, raw_path)}))
    b_rows = np.flatnonzero(frame.run_id.eq(RUN_B))
    inverse = np.full(len(frame), -1, int)
    inverse[b_rows] = np.arange(len(b_rows))
    b = frame.iloc[b_rows].reset_index(drop=True)
    first = np.full((2, len(ALPHAS), len(ALL_METHODS), len(b)), -2, np.int16)
    scores = np.full((len(METHODS) + 1, len(b), 52), np.nan, np.float32)
    fold_ids = np.full(len(b), -1, int)
    choices, thresholds, watches = [], [], []
    selection, calibration, assignments = [], [], []
    for fold in range(5):
        roles = split_roles(frame, fold)
        assert tuple(map(len, roles)) == (6400, 3200, 3200, 3200)
        part = process_split(frame, cache, raw, roles, fold, output)
        test = roles[-1]
        ix = inverse[test]
        fold_ids[ix] = fold
        choices.append(part["selected"])
        thresholds.append(part["threshold"])
        watches.append(part["watch"])
        scores[:len(METHODS), ix] = part["scores"][:len(METHODS)]
        mi = part["selected"][PRIMARY_KIND, PRIMARY_ALPHA]
        scores[-1, ix] = part["scores"][mi]
        for ki in range(2):
            for ai in range(len(ALPHAS)):
                for method in range(len(METHODS)):
                    first[ki, ai, method, ix] = first_alarm(part["scores"][method], cache["valid"][test], part["threshold"][ki, ai, method])
                first[ki, ai, len(METHODS), ix] = first[ki, ai, part["selected"][ki, ai], ix]
                first[ki, ai, -1, ix] = first_alarm(part["scores"][-1], cache["valid"][test], part["threshold"][ki, ai, -1])
        selection.extend(part["selection_rows"])
        calibration.extend(part["calibration_rows"])
        assignments.extend(dict(fold=fold, role=role, global_row=int(i))
                           for role, group in zip(("reference", "calibration", "development", "test"), roles) for i in group)
        print("DESIGN fold %d selected=%s" % (fold, METHODS[int(mi)]), flush=True)
    a = frame.run_id.eq(RUN_A).to_numpy()
    residue = frame.init_state_id.to_numpy() % 5
    deploy_roles = (np.flatnonzero(a & (residue < 3)), np.flatnonzero(a & (residue == 3)),
                    np.flatnonzero(a & (residue == 4)), np.asarray([], int))
    deploy = process_split(frame, cache, raw, deploy_roles, "deploy", output)
    selection.extend(deploy["selection_rows"])
    calibration.extend(deploy["calibration_rows"])
    print("DESIGN deployment selected=%s" % METHODS[int(deploy["selected"][PRIMARY_KIND, PRIMARY_ALPHA])], flush=True)
    pd.DataFrame(selection).to_csv(output / "selection_grid.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "calibration_thresholds.csv", index=False)
    pd.DataFrame(assignments).to_csv(output / "split_assignments.csv", index=False)
    write_json(output / "selection_frozen.json", dict(test_metrics_not_used=True,
                design_files={p.name: digest(p) for p in output.iterdir() if p.name.startswith(("profile_", "design_"))},
                selection_grid_sha256=digest(output / "selection_grid.csv"), contract_sha256=digest(output / "execution_contract.json")))
    assert (fold_ids >= 0).all() and (first >= -1).all()
    np.savez_compressed(output / "predictions.npz", first=first, scores=scores, global_rows=b_rows,
                        methods=np.asarray(ALL_METHODS), score_methods=np.asarray([*METHODS, "selected"]),
                        fold_ids=fold_ids, choices=np.asarray(choices), thresholds=np.asarray(thresholds),
                        watch_thresholds=np.asarray(watches), valid=cache["valid"][b_rows])
    summarize(b, first, output)
    early_auc(b, scores, output)
    state_diagnostics(b, scores, first, np.asarray(thresholds), np.asarray(watches), fold_ids, np.asarray(choices), output)
    write_json(output / "verification.json", dict(sources=contract["source_hashes"],
                artifact_hashes={p.name: digest(p) for p in output.iterdir() if p.is_file()},
                b_episodes=len(b), b_failures=int(b.failure.sum()), test_each_episode_once=True,
                selection_uses_a_only=True, seconds=time.perf_counter() - started))
    print("EXPERIMENT COMPLETE %.1fs" % (time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    main()
