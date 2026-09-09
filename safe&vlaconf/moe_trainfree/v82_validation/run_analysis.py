"""Audit frozen guards, cross-fit success budgets, and evaluate temporal evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from monitor import (ALPHAS, HERE, METHODS, ROOT, auc, build_budget_streams, clean_first,
                     conformal_threshold, cumulative_counts, first_alarm, first_and,
                     intrinsic_score_arrays, persistent, prefix_peak, union, v8_streams)

BASE = HERE.parent / "results"
HUB = ROOT / "VLA_MUI_HUB"
RUN_A = "right-50x8-20260903"
RUN_B = "right-50x8b-20260903"
SEED = 20260908


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def load_inputs(output):
    index_path = BASE / "round9_full_corpus/index.csv"
    frame = pd.read_csv(index_path)
    assert len(frame) == 32000 and not frame.duplicated(["source", "episode"]).any()
    pd.testing.assert_frame_equal(frame, pd.read_csv(BASE / "round7_temporal_fusion/index.csv"))
    frame["global_row"] = frame.index
    paths = [index_path, BASE / "round3_safe/v7/v7_inputs.npz", BASE / "round7_temporal_fusion/v8_inputs.npz"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    old = json.loads((BASE / "round9_full_corpus/sealed_manifest.json").read_text())
    assert hashes[str(paths[1].relative_to(ROOT))] == old["inputs"][str(paths[1].relative_to(ROOT))]
    audit_v8 = json.loads((BASE / "round7_temporal_fusion/input_audit.json").read_text())
    assert hashes[str(paths[2].relative_to(ROOT))] == audit_v8["output_sha256"]
    cache, extra = archive(paths[1]), archive(paths[2])
    valid = np.arange(52)[None] < frame.length.to_numpy()[:, None]
    np.testing.assert_array_equal(cache["valid"], valid)
    np.testing.assert_array_equal(extra["valid"], valid)
    np.testing.assert_array_equal(np.isfinite(extra["raw"]).all(axis=-1), valid)
    frame["failure"] = False
    frame["action_steps"] = -1
    eef = np.full(valid.shape, np.nan, np.float32)
    audits = json.loads((BASE / "round3_safe/extraction_audit.json").read_text())
    by_source = {r["source"]: r for r in audits}
    records = []
    for source, part in frame.groupby("source", sort=True):
        summary_path = HUB / source / "client/summaries.json"
        summary_hash = digest(summary_path)
        assert summary_hash == by_source[source]["summary_sha256"]
        summaries = {int(r["episode_index"]): r for r in json.loads(summary_path.read_text())}
        assert len(summaries) == len(part) == 400
        for row in part.itertuples():
            entry = summaries[int(row.episode)]
            assert int(entry["inference_calls"]) == row.length
            assert int(entry["init_state_id"]) == row.init_state_id
            assert int(entry["flow_noise_seed"]) == row.noise_seed
            assert row.task == row.suite + "/" + entry["task_name"]
            frame.at[row.Index, "failure"] = not entry["success"]
            frame.at[row.Index, "action_steps"] = entry["action_steps"]
        feature_path = BASE / "round3_safe/features" / by_source[source]["output"]
        with np.load(feature_path) as data:
            saved = pd.DataFrame(json.loads(str(data["index"])))
            for field in ("episode", "length", "init_state_id", "noise_seed", "checkpoint"):
                np.testing.assert_array_equal(part[field], saved[field])
            np.testing.assert_array_equal(data["valid"], valid[part.index])
            eef[part.index] = data["direct"][:, :, 5]
        hashes[str(summary_path.relative_to(ROOT))] = summary_hash
        hashes[str(feature_path.relative_to(ROOT))] = digest(feature_path)
        records.append(dict(source=source, episodes=len(part), queries=int(valid[part.index].sum()),
                            summary_sha256=summary_hash, cached_feature=str(feature_path.relative_to(ROOT))))
    labels_path = HUB / "physical-failure-labels/results/episodes.csv"
    labels = pd.read_csv(labels_path)
    labels = labels.set_index(["source_run", "episode_index"])
    keys = pd.MultiIndex.from_frame(frame[["source", "episode"]])
    matched = labels.reindex(keys)
    assert matched.recorded_success.notna().all()
    np.testing.assert_array_equal(~frame.failure.to_numpy(), matched.recorded_success.to_numpy(bool))
    frame["primary_failure_reason"] = matched.primary_failure_reason.fillna("").to_numpy()
    hashes[str(labels_path.relative_to(ROOT))] = digest(labels_path)
    assert frame.groupby("run_id").failure.sum().to_dict() == {RUN_A: 532, RUN_B: 564}
    frame.to_csv(output / "index.csv", index=False)
    pd.DataFrame(records).to_csv(output / "source_coverage.csv", index=False)
    write_json(output / "input_verification.json", dict(inputs=hashes, episodes=len(frame), runs=len(records),
                queries=int(valid.sum()), raw_cache_provenance=audit_v8, hub_summary_and_physical_labels_agree=True))
    print("INPUTS verified 32000 episodes / %d queries / 80 HUB runs" % valid.sum(), flush=True)
    return frame, cache, extra["raw"], eef


def historical_map(frame, metadata):
    lookup = pd.MultiIndex.from_frame(frame[["task", "run_id", "episode"]])
    keys = list(zip(metadata["task_names"].astype(str)[metadata["task_index"]],
                    np.repeat(str(metadata["run_id"]), len(metadata["episode"])), metadata["episode"]))
    rows = lookup.get_indexer(keys)
    assert (rows >= 0).all()
    for key, field in (("init_state_id", "init_state_id"), ("flow_noise_seed", "noise_seed"), ("length", "length")):
        np.testing.assert_array_equal(metadata[key], frame.iloc[rows][field])
    return rows


def historical_series(raw, valid):
    # Preserve the historical partial warm-up means for a padding-only calibration audit.
    base = raw[:, 1:5, 1].mean(axis=1, dtype=np.float32)
    rel = np.log(np.maximum(raw[:, :, 1], 1e-12) / np.maximum(base[:, None], 1e-12))
    result = np.full(raw.shape, np.nan, dtype=np.float64)
    for q in range(raw.shape[1]):
        lo = max(0, q - 5)
        good = valid[:, q]
        result[good, q, 0] = raw[good, lo:q + 1, 0].mean(axis=1, dtype=np.float64)
        result[good, q, 1] = rel[good, lo:q + 1].mean(axis=1, dtype=np.float64)
    return result


def frozen_replay(frame, cache, raw, output):
    v7root = ROOT / "moe-v7-0905/results/intrinsic_guard_v7"
    profile = archive(v7root / "global_profile.npz")
    v7 = intrinsic_score_arrays(cache["mobility"], cache["acceleration"], cache["periodicity"],
                               float(profile["periodicity_scale"]))
    valid = cache["valid"]
    freeze = first_alarm(v7["freeze"], valid, float(profile["freeze_threshold"]))
    acc = first_alarm(v7["acceleration_persistent"], valid, float(profile["acceleration_threshold"]))
    period = first_alarm(v7["periodicity_persistent"], valid, float(profile["periodicity_threshold"]))
    turbulence = first_and(acc, period)
    old_guard = union(freeze, turbulence)
    meta_a = archive(ROOT / "moe-v4-0904/results/layerwise_mobility/main_reference.npz")
    meta_b = archive(ROOT / "moe-v4-0904/results/layerwise_mobility/external_8b.npz")
    rows_a, rows_b = historical_map(frame, meta_a), historical_map(frame, meta_b)
    old_v7 = archive(v7root / "sealed_first_alarms.npz")
    np.testing.assert_array_equal(old_guard[rows_a], old_v7["main_guard"])
    np.testing.assert_array_equal(old_guard[rows_b], old_v7["external_guard"])
    series = historical_series(raw, valid)
    bpool = -series[rows_a, :, 0][valid[rows_a]]
    cpool = series[rows_a, :, 1][valid[rows_a]]
    corrected = dict(frontback=-float(np.quantile(bpool, .005, method="lower")),
                     curvature=float(np.quantile(cpool, .995, method="lower")))
    historical = dict(frontback=0.0, curvature=.3890000581741333)
    active = series.copy()
    active[:, :6] = np.nan
    q = np.arange(valid.shape[1])[None]

    def new_alarms(thresholds, slope):
        return [first_alarm(persistent(active[:, :, i] + slope * q, 2), valid, thresholds[key], True)
                for i, key in enumerate(("frontback", "curvature"))]

    original_new = new_alarms(historical, .0015)
    fixed_new = new_alarms(historical, 0.0)
    corrected_new = new_alarms(corrected, .0015)
    corrected_fixed = new_alarms(corrected, 0.0)
    first = {
        "v7_frozen": old_guard,
        "v8_frozen": union(old_guard, *fixed_new),
        "v82_frozen": union(old_guard, *original_new),
        "v82_padding_corrected": union(old_guard, *corrected_new),
        "v8_padding_corrected": union(old_guard, *corrected_fixed),
        "v7_corrected_frontback": union(old_guard, corrected_new[0]),
        "v7_corrected_curvature": union(old_guard, corrected_new[1]),
    }
    checks = []
    for name, rows in (("development_main", rows_a), ("external_8b", rows_b)):
        old8 = np.load(ROOT / "moe-v8-0906/results/v8_full_corpus_alarms.npz")[name + "|v8"]
        old82 = np.load(ROOT / "moe-v8-0906/results/v82_alarms.npz")[name + "|v8.2"]
        for method, saved in (("v8_frozen", old8), ("v82_frozen", old82)):
            expected = clean_first(saved, frame.iloc[rows].length.to_numpy())
            np.testing.assert_array_equal(first[method][rows], expected)
            checks.append(dict(cohort=name, method=method, exact_matches=len(rows)))
    frame["historical_a"] = frame.index.isin(rows_a)
    frame["historical_b"] = frame.index.isin(rows_b)
    frame.to_csv(output / "index.csv", index=False)
    out = frame[["global_row", "source", "episode", "run_id", "suite", "task", "init_state_id", "length", "failure"]].copy()
    for name, values in first.items():
        out[name] = values
    out.to_csv(output / "frozen_first_alarms.csv", index=False)
    head_names = ("freeze", "turbulence", "frontback", "curvature")
    attribution = []
    for version, heads in (("frozen", original_new), ("padding_corrected", corrected_new)):
        components = [freeze, turbulence, *heads]
        guard = union(*components)
        for i in np.flatnonzero(frame.run_id.eq(RUN_B).to_numpy() & (guard >= 0)):
            cause = "+".join(n for n, a in zip(head_names, components) if a[i] == guard[i])
            attribution.append(dict(global_row=i, version=version, task=frame.iloc[i].task,
                                    suite=frame.iloc[i].suite, failure=bool(frame.iloc[i].failure), first=int(guard[i]),
                                    cause=cause, v7_first=int(old_guard[i]),
                                    added=bool(old_guard[i] < 0), earlier=bool(old_guard[i] > guard[i])))
    pd.DataFrame(attribution).to_csv(output / "head_attribution.csv", index=False)
    raw_freeze = -np.median(np.log(np.maximum(cache["mobility"][:, :, 4:], 1e-6) /
                        np.maximum(cache["mobility"][:, 1:5, 4:].mean(axis=1)[:, None], 1e-6)), axis=2)
    base = raw[:, 1:5, 1].mean(axis=1)
    relative_c = np.log(np.maximum(raw[:, :, 1], 1e-12) / np.maximum(base[:, None], 1e-12))
    event_scores = np.stack((raw_freeze, raw[:, :, 0], relative_c, v7["freeze"], series[:, :, 0], series[:, :, 1]))
    event_scores[:, :, :6] = np.nan
    event_scores[:, ~valid] = np.nan
    np.savez_compressed(output / "frozen_scores.npz", scores=event_scores.astype(np.float32),
                        names=np.asarray(("freeze_raw", "frontback_raw", "curvature_relative_raw",
                                          "freeze_smooth", "frontback_smooth", "curvature_smooth")), valid=valid)
    report = dict(exact_replays=checks, historical_thresholds=historical,
                  padding_corrected_thresholds=corrected, calibration_episodes=len(rows_a),
                  valid_calibration_queries=int(valid[rows_a].sum()),
                  padding_calibration_cells=int((~valid[rows_a]).sum()),
                  calibration_includes_historical_partial_warmup=True, historical_b_episodes=len(rows_b))
    write_json(output / "frozen_verification.json", report)
    print("FROZEN REPLAY exact; corrected thresholds %s" % corrected, flush=True)
    return first


def state_splits(frame):
    a = frame.run_id.eq(RUN_A).to_numpy()
    b = frame.run_id.eq(RUN_B).to_numpy()
    residue = frame.init_state_id.to_numpy() % 5
    for f in range(5):
        test = np.flatnonzero(b & (residue == f))
        cal = np.flatnonzero(a & (residue == (f + 1) % 5))
        ref = np.flatnonzero(a & (residue != f) & (residue != (f + 1) % 5))
        groups = [set(zip(frame.iloc[ix].task, frame.iloc[ix].init_state_id)) for ix in (ref, cal, test)]
        assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
        yield f, ref, cal, test


def crossfit(frame, cache, raw, eef, output):
    b_rows = np.flatnonzero(frame.run_id.eq(RUN_B))
    inverse = np.full(len(frame), -1, int)
    inverse[b_rows] = np.arange(len(b_rows))
    scores_all = np.full((len(METHODS), len(b_rows), 52), np.nan, np.float32)
    first_all = np.full((2, len(ALPHAS), len(METHODS), len(b_rows)), -2, np.int16)
    kinds = ("episode", "task_init")
    fold_ids = np.full(len(b_rows), -1, np.int8)
    thresholds, assignments, profiles = [], [], []
    for fold, reference, calibration, test in state_splits(frame):
        started = time.perf_counter()
        rows = np.r_[reference, calibration, test]
        nr, nc = len(reference), len(calibration)
        current = {k: v[rows] for k, v in cache.items()}
        scores, profile = build_budget_streams(current, raw[rows], eef[rows], np.arange(nr))
        positions = inverse[test]
        assert (fold_ids[positions] < 0).all()
        fold_ids[positions] = fold
        scores_all[:, positions] = scores[:, nr + nc:]
        success = ~frame.iloc[calibration].failure.to_numpy(bool)
        cal_frame = frame.iloc[calibration].loc[success]
        group_keys = (cal_frame.task + "|" + cal_frame.init_state_id.astype(str)).to_numpy()
        group_index, names = pd.factorize(group_keys, sort=True)
        cal_scores = scores[:, nr:nr + nc][:, success]
        peaks = np.where(np.isfinite(cal_scores), cal_scores, -np.inf).max(axis=2)
        for mi, method in enumerate(METHODS):
            group_peaks = np.full(len(names), -np.inf)
            np.maximum.at(group_peaks, group_index, peaks[mi])
            for ki, (kind, units) in enumerate(zip(kinds, (peaks[mi], group_peaks))):
                for ai, alpha in enumerate(ALPHAS):
                    threshold, rank = conformal_threshold(units, alpha)
                    first_all[ki, ai, mi, positions] = first_alarm(scores[mi, nr + nc:], current["valid"][nr + nc:], threshold)
                    thresholds.append(dict(fold=fold, method=method, calibration=kind, alpha=alpha,
                                           threshold=threshold, rank=rank, units=len(units),
                                           calibration_exceedances=int((units > threshold).sum())))
        for role, ix in (("reference", reference), ("calibration", calibration), ("test", test)):
            assignments.extend(dict(fold=fold, role=role, global_row=int(i)) for i in ix)
        profiles.append(dict(fold=fold, profile=profile, reference_rows=reference,
                             calibration_rows=calibration, test_rows=test))
        np.savez_compressed(output / ("fold_%d_calibration.npz" % fold),
                            rows=calibration, scores=scores[:, nr:nr + nc], failure=~success,
                            methods=np.asarray(METHODS))
        print("CROSSFIT %d reference=%d calibration=%d test=%d %.1fs" %
              (fold, nr, nc, len(test), time.perf_counter() - started), flush=True)
    assert (fold_ids >= 0).all() and (first_all >= -1).all()
    np.savez_compressed(output / "crossfit_predictions.npz", scores=scores_all, first=first_all,
                        methods=np.asarray(METHODS), kinds=np.asarray(kinds), alphas=np.asarray(ALPHAS),
                        global_rows=b_rows, fold_ids=fold_ids, valid=cache["valid"][b_rows])
    pd.DataFrame(thresholds).to_csv(output / "calibration_thresholds.csv", index=False)
    pd.DataFrame(assignments).to_csv(output / "split_assignments.csv", index=False)
    write_json(output / "profiles.json", profiles)
    return b_rows, scores_all, first_all


def bootstrap_indices(tasks, suites, samples=2000):
    rng = np.random.default_rng(SEED)
    blocks = [np.flatnonzero(np.asarray(suites) == suite) for suite in sorted(set(suites))]
    return np.concatenate([rng.choice(block, size=(samples, len(block)), replace=True) for block in blocks], axis=1)


def rate_interval(frame, fired, failure):
    tasks = sorted(frame.task.unique())
    counts = []
    for task in tasks:
        mask = frame.task.eq(task).to_numpy()
        counts.append([int((mask & fired & failure).sum()), int((mask & failure).sum()),
                       int((mask & fired & ~failure).sum()), int((mask & ~failure).sum())])
    counts = np.asarray(counts, float)
    draws = bootstrap_indices(tasks, [t.split("/")[0] for t in tasks])
    totals = counts[draws].sum(axis=1)
    out = {}
    for name, num, den in (("recall", 0, 1), ("fpr", 2, 3)):
        values = np.divide(totals[:, num], totals[:, den], out=np.full(len(totals), np.nan), where=totals[:, den] > 0)
        out[name + "_lo"], out[name + "_hi"] = np.nanquantile(values, [.025, .975]) if np.isfinite(values).any() else (np.nan, np.nan)
    return out


def summarize_alarms(frame, frozen, b_rows, budget_first, output):
    rows, curves = [], []
    b = frame.iloc[b_rows].reset_index(drop=True)
    methods = [("frozen", name, "original", np.nan, values[b_rows]) for name, values in frozen.items()]
    methods.extend(("crossfit", name, kind, alpha, budget_first[ki, ai, mi])
                   for ki, kind in enumerate(("episode", "task_init"))
                   for ai, alpha in enumerate(ALPHAS) for mi, name in enumerate(METHODS))
    scopes = [("all", np.ones(len(b), bool)), ("historical_b", b.historical_b.to_numpy(bool))]
    scopes += [(suite, b.suite.eq(suite).to_numpy()) for suite in sorted(b.suite.unique())]
    for family, method, kind, alpha, first in methods:
        for scope, mask in scopes:
            current = b.loc[mask].reset_index(drop=True)
            failure = current.failure.to_numpy(bool)
            length = current.length.to_numpy(int)
            alarm = first[mask]
            meta = dict(family=family, method=method, calibration=kind, alpha=alpha, scope=scope)
            counts = cumulative_counts(alarm, failure, length, 51)
            fired = alarm >= 0
            row = dict(meta, **counts, recall=counts["tp"] / counts["failures"] if counts["failures"] else np.nan,
                       fpr=counts["fp"] / counts["successes"],
                       precision=counts["tp"] / fired.sum() if fired.any() else np.nan,
                       median_failure_q=float(np.median(alarm[fired & failure])) if (fired & failure).any() else np.nan)
            row.update(rate_interval(current, fired, failure))
            rows.append(row)
            for q in range(52):
                c = cumulative_counts(alarm, failure, length, q)
                curves.append(dict(meta, query=q, actions=10 * q, **c,
                                   recall=c["tp"] / c["failures"] if c["failures"] else np.nan,
                                   fpr=c["fp"] / c["successes"]))
    pd.DataFrame(rows).to_csv(output / "alarm_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "cumulative_curves.csv", index=False)


def conditional_analysis(frame, b_rows, scores, output, frozen=None):
    b = frame.iloc[b_rows].reset_index(drop=True)
    names = list(METHODS)
    if frozen is not None:
        additional = []
        for name in ("v7_frozen", "v82_frozen", "v82_padding_corrected"):
            first = frozen[name][b_rows]
            values = ((first[:, None] >= 0) & (first[:, None] <= np.arange(52)[None])).astype(np.float32)
            values[np.arange(52)[None] >= b.length.to_numpy()[:, None]] = np.nan
            additional.append(values)
            names.append(name + "_binary")
        scores = np.concatenate((scores, np.stack(additional)))
    records = []
    # A score is compared only with scores at the same time in the same task/initial state.
    for level, keys in (("task", ["task"]), ("task_init", ["task", "init_state_id"])):
        for _, part in b.groupby(keys, sort=True):
            ix = part.index.to_numpy()
            failure = part.failure.to_numpy(bool)
            if not failure.any() or failure.all():
                continue
            length = part.length.to_numpy(int)
            for q in range(7, 52):
                active = length > q
                if not (active & failure).any() or not (active & ~failure).any():
                    continue
                for mi, name in enumerate(names):
                    values = scores[mi, ix, q]
                    mask = active & np.isfinite(values)
                    value = auc(failure[mask], values[mask])
                    records.append(dict(level=level, task=part.task.iloc[0], suite=part.suite.iloc[0],
                                        init_state_id=int(part.init_state_id.iloc[0]) if level == "task_init" else -1,
                                        query=q, method=name, auc=value, failures=int((mask & failure).sum()),
                                        successes=int((mask & ~failure).sum())))
    data = pd.DataFrame(records)
    data.to_csv(output / "conditional_auc_strata.csv", index=False)
    task_values = data.groupby(["level", "suite", "task", "query", "method"], as_index=False).agg(
        auc=("auc", "mean"), strata=("auc", "count"), failures=("failures", "sum"), successes=("successes", "sum"))
    summaries = []
    for (level, q, method), part in task_values.groupby(["level", "query", "method"], sort=True):
        scopes = [("all", part)] + list(part.groupby("suite", sort=True))
        for scope, current in scopes:
            current = current.loc[current.auc.notna()]
            if not len(current):
                continue
            draws = bootstrap_indices(current.task.tolist(), current.suite.tolist())
            sample = current.auc.to_numpy()[draws].mean(axis=1)
            lo, hi = np.quantile(sample, [.025, .975])
            summaries.append(dict(level=level, query=q, method=method, scope=scope,
                                  auc=float(current.auc.mean()), lo=lo, hi=hi, tasks=len(current),
                                  strata=int(current.strata.sum()), failures=int(current.failures.sum()),
                                  successes=int(current.successes.sum())))
    pd.DataFrame(summaries).to_csv(output / "conditional_auc_curves.csv", index=False)
    early = task_values.loc[task_values["query"].between(7, 13)].groupby(["level", "suite", "task", "method"], as_index=False).auc.mean()
    primary = []
    for level, group in early.groupby("level", sort=True):
        for scope, current in [("all", group)] + list(group.groupby("suite", sort=True)):
            pivot = current.pivot(index=["suite", "task"], columns="method", values="auc")
            for name in names:
                eligible = pivot[[name]].dropna()
                values = eligible[name].to_numpy()
                if not len(values):
                    continue
                draws = bootstrap_indices(eligible.index.get_level_values("task"), eligible.index.get_level_values("suite"))
                ci = np.quantile(values[draws].mean(axis=1), [.025, .975])
                primary.append(dict(level=level, scope=scope, method=name, comparison="auc",
                                    estimate=values.mean(), lo=ci[0], hi=ci[1], tasks=len(values)))
            for control in ("v7", "eef_motion_low", "clock"):
                eligible = pivot[["v82", control]].dropna()
                values = (eligible.v82 - eligible[control]).to_numpy()
                if not len(values):
                    continue
                draws = bootstrap_indices(eligible.index.get_level_values("task"), eligible.index.get_level_values("suite"))
                ci = np.quantile(values[draws].mean(axis=1), [.025, .975])
                primary.append(dict(level=level, scope=scope, method="v82", comparison="minus_" + control,
                                    estimate=values.mean(), lo=ci[0], hi=ci[1], tasks=len(values)))
    pd.DataFrame(primary).to_csv(output / "early_conditional_summary.csv", index=False)
    print("CONDITIONAL AUC %d task/time/initial-state comparisons" % len(data), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "v82_validation_20260908")
    parser.add_argument("--resume-statistics", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    started = time.perf_counter()
    if args.resume_statistics:
        frame = pd.read_csv(output / "index.csv")
        saved = pd.read_csv(output / "frozen_first_alarms.csv")
        frozen = {k: saved[k].to_numpy(np.int16) for k in saved.columns if k.startswith("v7_") or k.startswith("v8_") or k.startswith("v82_")}
        pred = archive(output / "crossfit_predictions.npz")
        b_rows, scores, first = pred["global_rows"], pred["scores"], pred["first"]
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "execution_contract.json", dict(protocol_sha256=digest(HERE / "PROTOCOL_ZH.md"),
                   methods=METHODS, alphas=ALPHAS, seed=SEED, test_cohort=RUN_B,
                   primary_calibration="task_init", primary_alpha=.01, early_queries=list(range(7, 14)),
                   calibration_uses_full_success_trajectory_peaks=True, checkpoint_recalibration=False,
                   historical_exploration=True, new_rollouts=False, model_training=False))
        frame, cache, raw, eef = load_inputs(output)
        frozen = frozen_replay(frame, cache, raw, output)
        b_rows, scores, first = crossfit(frame, cache, raw, eef, output)
    summarize_alarms(frame, frozen, b_rows, first, output)
    conditional_analysis(frame, b_rows, scores, output, frozen)
    owned = ("execution_contract.json", "index.csv", "source_coverage.csv", "input_verification.json",
             "frozen_first_alarms.csv", "head_attribution.csv", "frozen_scores.npz", "frozen_verification.json",
             "crossfit_predictions.npz", "calibration_thresholds.csv", "split_assignments.csv", "profiles.json",
             "alarm_metrics.csv", "cumulative_curves.csv", "conditional_auc_strata.csv",
             "conditional_auc_curves.csv", "early_conditional_summary.csv")
    owned = (*owned, *("fold_%d_calibration.npz" % f for f in range(5)))
    artifacts = {name: digest(output / name) for name in owned}
    write_json(output / "analysis_verification.json", dict(artifacts=artifacts,
                source_sha256={p.name: digest(p) for p in (Path(__file__), HERE / "monitor.py", HERE / "PROTOCOL_ZH.md")},
                test_episodes=len(b_rows), test_coverage_once=True, valid_execution_alarms_only=True,
                elapsed_seconds=time.perf_counter() - started))
    print("ANALYSIS COMPLETE %.1fs" % (time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    main()
