"""Separate cosine retrieval, magnitude loss, origin, and calibration effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from cosine_knn import HERE, ROOT, CASE_ROW, CASE_FOLD, load_npz, normalized_vectors
from compare_cosine import load_action_steps, metrics, pooled_counts
from core import conformal_threshold, digest, first_alarm, trajectory_peak, write_json

METHODS = ("euclidean", "cosine", "cosine_neighbors_euclidean_score",
           "euclidean_neighbors_cosine_score", "cosine_without_center", "norm_only")
OUTPUT = HERE.parent / "results/round8_cosine_knn/diagnosis"
QUANTILES = (0, .1, .25, .5, .75, .9, .95, 1)


def quantiles(values, prefix):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {f"{prefix}_p{int(q * 100):02}": float(np.quantile(values, q)) if len(values) else np.nan for q in QUANTILES}


def score_factorial(x, bank, offset):
    valid = np.isfinite(x).all(-1)
    values = x[valid]
    assert (np.linalg.norm(values, axis=1) > 1e-12).all()
    assert (np.linalg.norm(bank, axis=1) > 1e-12).all()
    assert (np.linalg.norm(values + offset, axis=1) > 1e-12).all()
    assert (np.linalg.norm(bank + offset, axis=1) > 1e-12).all()
    scores = np.full((len(METHODS), *valid.shape), np.nan, np.float32)
    ids = {key: np.full((*valid.shape, 20), -1, np.int32) for key in ("euclidean", "cosine")}
    flat_scores = np.empty((len(METHODS), len(values)), np.float32)
    flat_ids = {key: np.empty((len(values), 20), np.int32) for key in ids}
    for start in range(0, len(values), 512):
        stop = min(start + 512, len(values))
        current = values[start:stop]
        # One distance matrix per metric gives an independent scipy replay of sklearn.
        euclidean = cdist(current, bank, metric="euclidean")
        cosine = np.clip(cdist(current, bank, metric="cosine"), 0, 2)
        eids = np.argpartition(euclidean, 19, axis=1)[:, :20]
        cids = np.argpartition(cosine, 19, axis=1)[:, :20]
        for mi, matrix, identity in ((0, euclidean, eids), (1, cosine, cids),
                                     (2, euclidean, cids), (3, cosine, eids)):
            flat_scores[mi, start:stop] = np.take_along_axis(matrix, identity, axis=1).mean(1)
        absolute = np.clip(cdist(current + offset, bank + offset, metric="cosine"), 0, 2)
        flat_scores[4, start:stop] = np.partition(absolute, 19, axis=1)[:, :20].mean(1)
        flat_scores[5, start:stop] = np.linalg.norm(current, axis=1)
        flat_ids["euclidean"][start:stop], flat_ids["cosine"][start:stop] = eids, cids
    scores[:, valid] = flat_scores
    for key in ids:
        ids[key][valid] = flat_ids[key]
    return scores, ids


def calibrate(scores, labels, cal_frame, ncal):
    success = labels == 0
    group_ids = (cal_frame.loc[success, "task"] + "|" + cal_frame.loc[success, "init_state_id"].astype(str)).to_numpy()
    thresholds = np.empty((2, len(METHODS)), np.float64)
    first = np.empty((2, len(METHODS), scores.shape[1] - ncal), np.int16)
    records = []
    for mi, method in enumerate(METHODS):
        peaks = trajectory_peak(scores[mi, :ncal][success])
        grouped = np.asarray([peaks[group_ids == key].max() for key in sorted(set(group_ids))])
        for kind_i, (kind, units) in enumerate((("episode", peaks), ("task_init", grouped))):
            tau, rank = conformal_threshold(units, .05)
            thresholds[kind_i, mi] = tau
            first[kind_i, mi] = first_alarm(scores[mi, ncal:], tau)
            records.append(dict(method=method, calibration=kind, threshold=tau, units=len(units), rank=rank))
    return thresholds, first, records


def pair_geometry(current, bank, ids):
    reference = bank[ids]
    r = float(np.linalg.norm(current))
    s = np.linalg.norm(reference, axis=1)
    distance = np.linalg.norm(reference - current, axis=1)
    cosine = np.clip(1 - (reference @ current) / (s * r), 0, 2)
    radial_squared = (r - s) ** 2
    angular_squared = 2 * r * s * cosine
    np.testing.assert_allclose(distance ** 2, radial_squared + angular_squared, rtol=2e-10, atol=2e-10)
    # Divide each squared component by its distance to partition the mean distance.
    radial = np.divide(radial_squared, distance, out=np.zeros_like(distance), where=distance > 1e-12)
    angular = np.divide(angular_squared, distance, out=np.zeros_like(distance), where=distance > 1e-12)
    np.testing.assert_allclose(radial + angular, distance, rtol=2e-9, atol=2e-9)
    mean = float(distance.mean())
    return dict(norm=r, neighbor_norm_mean=float(s.mean()), neighbor_norm_min=float(s.min()),
        neighbor_norm_max=float(s.max()), norm_ratio=r / float(s.mean()),
        euclidean=mean, cosine=float(cosine.mean()), radial_contribution=float(radial.mean()),
        angular_contribution=float(angular.mean()), radial_fraction=float(radial.mean()) / mean if mean else 0), dict(
        distance=distance, cosine=cosine, norm=s, radial_contribution=radial, angular_contribution=angular)


def calibration_peaks(info, data, frame, x, scores, thresholds, steps):
    records, groups = [], []
    success_positions = np.flatnonzero(data["calibration_labels"] == 0)
    for mi, method in enumerate(METHODS[:2]):
        for position in success_positions:
            score = scores[mi, position]
            if not np.isfinite(score).any():
                continue
            q = int(np.nanargmax(score))
            row = int(data["calibration_rows"][position])
            entry = frame.loc[row]
            vector = x[position, q]
            record = dict(info, method=method, global_row=row, task=entry.task,
                init_state_id=int(entry.init_state_id), noise_seed=int(entry.noise_seed),
                episode=int(entry.episode), length=int(entry.length), query=q, score=float(score[q]),
                norm=float(np.linalg.norm(vector)), control_progress=10 * q / steps[row],
                threshold_group=float(thresholds[1, mi]), threshold_episode=float(thresholds[0, mi]),
                dominant_dimension=int(np.argmax(vector ** 2)),
                mobility_norm_fraction=float(np.sum(vector[:8] ** 2) / np.sum(vector ** 2)))
            records.append(record)
        peaks = pd.DataFrame([record for record in records if record["method"] == method])
        for (_, _), part in peaks.groupby(["task", "init_state_id"]):
            winner = part.loc[part.score.idxmax()].to_dict()
            winner["successful_repeats"] = len(part)
            groups.append(winner)
        ordered = sorted([record for record in groups if record["method"] == method], key=lambda z: z["score"])
        for rank, record in enumerate(ordered, start=1):
            record["rank"] = rank
            record["sets_group_threshold"] = bool(record["score"] == thresholds[1, mi])
    return records, groups


def distributions(info, data, frame, x, scores, thresholds, ncal):
    populations = {"cal_success": np.flatnonzero(data["calibration_labels"] == 0)}
    failures = frame.iloc[data["test_rows"]].failure.to_numpy()
    for scope, unseen in (("seen", False), ("unseen", True)):
        for outcome, failure in (("success", False), ("failure", True)):
            populations[f"test_{scope}_{outcome}"] = ncal + np.flatnonzero((data["test_unseen"] == unseen) & (failures == failure))
    records = []
    for population, rows in populations.items():
        for mi, method in enumerate(METHODS):
            raw = scores[mi, rows]
            finite = np.isfinite(raw)
            tau = float(thresholds[1, mi])
            peaks = trajectory_peak(raw)
            records.append(dict(info, method=method, population=population, episodes=len(rows),
                points=int(finite.sum()), threshold=tau, **quantiles(raw[finite], "chunk_score"),
                **quantiles(peaks, "episode_peak"), **quantiles(peaks / tau, "peak_ratio"),
                **quantiles(np.linalg.norm(x[rows][finite], axis=1), "norm")))
    return records


def analyze_case(profile, data, x, scores, ids, thresholds, first, ncal, frame):
    position = int(np.flatnonzero(data["test_rows"] == CASE_ROW)[0])
    length = int(frame.loc[CASE_ROW, "length"])
    bank = profile["success_dynamic"]
    queries, neighbors, sweep = [], [], []
    for q in range(length):
        current = x[ncal + position, q]
        record = dict(query=q, executed_actions_before_query=10 * q, norm=float(np.linalg.norm(current)))
        for mi, method in enumerate(METHODS):
            record[f"{method}_score"] = float(scores[mi, ncal + position, q])
            record[f"{method}_threshold"] = float(thresholds[1, mi])
            record[f"{method}_first_alarm"] = int(first[1, mi, position])
        if q >= 7:
            for kind in ("euclidean", "cosine"):
                identities = ids[kind][ncal + position, q]
                summary, detail = pair_geometry(current, bank, identities)
                record.update({f"{kind}_neighbors_{key}": value for key, value in summary.items()})
                order = np.argsort(detail["distance" if kind == "euclidean" else "cosine"])
                for rank, j in enumerate(order, start=1):
                    identity = identities[j]
                    ref_row = int(profile["success_global_rows"][identity])
                    neighbors.append(dict(query=q, selected_by=kind, rank=rank,
                        reference_global_row=ref_row, reference_query=int(profile["success_queries"][identity]),
                        reference_task=frame.loc[ref_row, "task"], current_norm=summary["norm"],
                        **{key: float(value[j]) for key, value in detail.items()}))
        queries.append(record)
    current = x[ncal + position, 33]
    original_cosine = float(scores[1, ncal + position, 33])
    for factor in (.25, .5, 1, 2):
        candidate = current * factor
        euclidean = np.sort(np.linalg.norm(bank - candidate, axis=1))[:20].mean()
        cosine = np.clip(1 - np.sum(bank * candidate, axis=1) /
            (np.linalg.norm(bank, axis=1) * np.linalg.norm(candidate)), 0, 2)
        score = float(np.sort(cosine)[:20].mean())
        np.testing.assert_allclose(score, original_cosine, rtol=2e-6, atol=2e-7)
        sweep.append(dict(query=33, factor=factor, norm=float(np.linalg.norm(candidate)),
                          euclidean=float(euclidean), cosine=score))
    return {"case_queries": pd.DataFrame(queries), "case_neighbors": pd.DataFrame(neighbors),
            "case_radial_rescaling": pd.DataFrame(sweep)}


def summarize_diagnostics(tables):
    phases, geometry, radial_bins = [], [], []
    for level, key in (("episode", "calibration_peaks"), ("task_init", "calibration_group_peaks")):
        for method, group in tables[key].groupby("method"):
            for lo, hi, name in ((7, 10, "q7-10"), (11, 14, "q11-14"), (15, 19, "q15-19"),
                                 (20, 29, "q20-29"), (30, 39, "q30-39"), (40, 51, "q40-51")):
                count = int(group["query"].between(lo, hi).sum())
                phases.append(dict(level=level, method=method, kind="query", bin=name, count=count,
                                   units=len(group), fraction=count / len(group)))
            for i, name in enumerate(("[0,25%)", "[25,50%)", "[50,75%)", "[75,100%)")):
                count = int(((group.control_progress >= i / 4) & (group.control_progress < (i + 1) / 4)).sum())
                phases.append(dict(level=level, method=method, kind="control_progress", bin=name,
                                   count=count, units=len(group), fraction=count / len(group)))
    for scope, all_points in tables["alarm_geometry"].groupby("scope"):
        for failure, outcome in ((True, "failure"), (False, "success")):
            for kept in (False, True):
                part = all_points.loc[all_points.failure.eq(failure) & all_points.cosine_ever_alarms.eq(kept)]
                if not len(part):
                    continue
                name = f"{outcome}_{'kept' if kept else 'lost'}"
                row = dict(scope=scope, population=name, points=len(part),
                    cosine_neighbor_distance_le005=int(part.cosine_neighbors_cosine.le(.05).sum()),
                    norm_at_least_twice_cosine_neighbors=int(part.cosine_neighbors_norm_ratio.ge(2).sum()),
                    **quantiles(part.cosine_ratio_at_euclidean_alarm, "cosine_ratio"),
                    **quantiles(part.cosine_neighbors_norm_ratio, "cosine_neighbor_norm_ratio"),
                    **quantiles(part.euclidean_neighbors_radial_fraction, "radial_fraction"),
                    **quantiles(part.neighbor_overlap, "neighbor_overlap"))
                geometry.append(row)
                for kind in ("euclidean", "cosine"):
                    share = part[f"{kind}_neighbors_radial_fraction"].clip(0, 1)
                    bands = np.minimum(np.floor(share * 4).astype(int), 3)
                    for i, label in enumerate(("[0,25%)", "[25,50%)", "[50,75%)", "[75,100%]")):
                        count = int((bands == i).sum())
                        radial_bins.append(dict(scope=scope, population=name, neighbors=kind,
                            radial_fraction_bin=label, count=count, points=len(part), fraction=count / len(part)))
    thresholds = tables["thresholds"].pivot(index=["fold", "suite", "method"], columns="calibration", values="threshold").reset_index()
    thresholds["group_over_episode_threshold"] = thresholds.task_init / thresholds.episode
    return {"calibration_phase_distribution": pd.DataFrame(phases), "geometry_summary": pd.DataFrame(geometry),
        "radial_fraction_bins": pd.DataFrame(radial_bins), "threshold_inflation": thresholds,
        "threshold_owners": tables["calibration_group_peaks"].loc[lambda x: x.sets_group_threshold].copy()}


def analyze(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    manifest = json.loads((parent / "sealed_manifest.json").read_text())
    inputs = {}
    for name, expected in manifest["sources"].items():
        assert digest(ROOT / name) == expected, name
        inputs[name] = expected
    for name, expected in manifest["artifacts"].items():
        path = parent / name
        assert digest(path) == expected, path
        inputs[str(path.relative_to(ROOT))] = expected
    frame_path = HERE.parent / "results/round5_knn/outcome_alignment.csv"
    cache_path = HERE.parent / "results/round3_safe/v7/v7_inputs.npz"
    inputs.update({str(path.relative_to(ROOT)): digest(path) for path in (frame_path, cache_path, parent / "sealed_manifest.json")})
    assert inputs[str(cache_path.relative_to(ROOT))] == manifest["inputs"][str(cache_path.relative_to(ROOT))]
    frame, cache = pd.read_csv(frame_path), load_npz(cache_path)
    data_by_fold = {info["fold"]: load_npz(parent / "predictions" / f"{info['fold']}.npz") for info in manifest["folds"]}
    needed = np.unique(np.concatenate([np.r_[d["calibration_rows"], d["test_rows"]] for d in data_by_fold.values()]))
    steps, step_inputs = load_action_steps(frame, needed)
    inputs.update(step_inputs)
    write_json(output / "diagnostic_contract.json", dict(methods=METHODS, alpha=.05, k=20,
        frozen_features_reference_and_splits=True, new_training=False, new_rollouts=False,
        posthoc_diagnostic=True, fixed_case_row=CASE_ROW, fixed_case_fold=CASE_FOLD,
        interpretation="feature-space controls; not physical interventions or new blind-test claims"))
    alarms, task_alarms, cal_peaks, group_peaks, distance_distributions, geometries, threshold_rows = [], [], [], [], [], [], []
    audits, snapshots, timings = [], [], []
    case_tables = {}
    for info in manifest["folds"]:
        started = time.perf_counter()
        fold = info["fold"]
        data = data_by_fold[fold]
        profile = load_npz(parent / "profiles" / f"{fold}.npz")
        old_path = HERE.parent / "results/round5_knn/predictions" / f"{fold}.npz"
        inputs[str(old_path.relative_to(ROOT))] = digest(old_path)
        old = load_npz(old_path)
        rows = np.r_[data["calibration_rows"], data["test_rows"]]
        ncal = len(data["calibration_rows"])
        x = normalized_vectors(cache, rows, profile)
        bank = profile["success_dynamic"]
        scores, ids = score_factorial(x, bank, profile["dynamic_center"] / profile["dynamic_scale"])
        replay_errors = {}
        for mi, method in enumerate(METHODS[:2]):
            expected = np.concatenate((data["calibration_scores"][mi], data["scores"][mi]))
            np.testing.assert_allclose(scores[mi], expected, rtol=2e-6, atol=2e-7, equal_nan=True)
            replay_errors[method] = float(np.nanmax(np.abs(scores[mi] - expected)))
            scores[mi] = expected
        ri = list(old["methods"].astype(str)).index("dyn_radius")
        radius = np.concatenate((old["calibration_scores"][ri], old["scores"][ri]))
        np.testing.assert_allclose(scores[5], radius, rtol=2e-6, atol=2e-7, equal_nan=True)
        scores[5] = radius
        thresholds, first, records = calibrate(scores, data["calibration_labels"], frame.iloc[data["calibration_rows"]].reset_index(drop=True), ncal)
        ai = int(np.flatnonzero(np.isclose(data["alphas"], .05))[0])
        np.testing.assert_array_equal(thresholds[:, :2], data["thresholds"][:, ai, :2])
        np.testing.assert_array_equal(first[:, :2], data["first"][:, ai, :2])
        np.testing.assert_array_equal(thresholds[:, 5], old["thresholds"][:, ai, ri])
        np.testing.assert_array_equal(first[:, 5], old["first"][:, ai, ri])
        threshold_rows.extend(dict(info, **record) for record in records)
        np.savez_compressed(output / "predictions" / f"{fold}.npz", methods=np.asarray(METHODS),
            scores=scores[:, ncal:], calibration_scores=scores[:, :ncal], thresholds=thresholds, first=first,
            test_rows=data["test_rows"], test_unseen=data["test_unseen"], calibration_rows=data["calibration_rows"],
            calibration_labels=data["calibration_labels"], reference_rows=data["reference_rows"])
        part = frame.iloc[data["test_rows"]].reset_index(drop=True)
        part["global_row"] = data["test_rows"]
        part["scope"] = np.where(data["test_unseen"], "unseen", "seen")
        for mi, method in enumerate(METHODS):
            for ki, kind in enumerate(("episode", "task_init")):
                for scope, mask in ((scope, part.scope.eq(scope).to_numpy()) for scope in ("seen", "unseen")):
                    alarms.append(dict(info, method=method, calibration=kind, scope=scope,
                        **metrics(part.loc[mask], first[ki, mi, mask])))
            for (scope, task), positions in part.groupby(["scope", "task"]).indices.items():
                task_alarms.append(dict(info, method=method, scope=scope, task=task,
                    **metrics(part.iloc[positions], first[1, mi, positions])))
            for (scope, failure), positions in part.groupby(["scope", "failure"]).indices.items():
                q = first[1, mi, positions]
                for query in range(-1, 52):
                    timings.append(dict(info, method=method, scope=scope, failure=bool(failure),
                        query=query, count=int((q == query).sum()), episodes=len(positions)))
        cp, gp = calibration_peaks(info, data, frame, x, scores, thresholds, steps)
        cal_peaks.extend(cp)
        group_peaks.extend(gp)
        distance_distributions.extend(distributions(info, data, frame, x, scores, thresholds, ncal))
        for position in np.flatnonzero(first[1, 0] >= 0):
            q = int(first[1, 0, position])
            row = int(data["test_rows"][position])
            current = x[ncal + position, q]
            record = dict(info, global_row=row, task=frame.loc[row, "task"], scope=part.loc[position, "scope"],
                failure=bool(frame.loc[row, "failure"]), query=q, cosine_ever_alarms=bool(first[1, 1, position] >= 0),
                cosine_at_euclidean_alarm=float(scores[1, ncal + position, q]),
                cosine_ratio_at_euclidean_alarm=float(scores[1, ncal + position, q] / thresholds[1, 1]))
            for kind in ("euclidean", "cosine"):
                summary, _ = pair_geometry(current, bank, ids[kind][ncal + position, q])
                record.update({f"{kind}_neighbors_{key}": value for key, value in summary.items()})
            record["neighbor_overlap"] = len(set(ids["euclidean"][ncal + position, q]) & set(ids["cosine"][ncal + position, q]))
            geometries.append(record)
        if fold == CASE_FOLD:
            case_tables = analyze_case(profile, data, x, scores, ids, thresholds, first, ncal, frame)
        # Verify representative factorial entries directly without matrix indexing.
        for position in (0, ncal, len(rows) - 1):
            for q in (7, int(frame.loc[rows[position], "length"]) - 1):
                if q < 7 or not np.isfinite(x[position, q]).all():
                    continue
                current = x[position, q]
                de = np.linalg.norm(bank - current, axis=1)
                dc = np.clip(1 - np.sum(bank * current, axis=1) / (np.linalg.norm(bank, axis=1) * np.linalg.norm(current)), 0, 2)
                eids, cids = np.argsort(de)[:20], np.argsort(dc)[:20]
                offset = profile["dynamic_center"] / profile["dynamic_scale"]
                u, b = current + offset, bank + offset
                du = np.clip(1 - np.sum(b * u, axis=1) / (np.linalg.norm(b, axis=1) * np.linalg.norm(u)), 0, 2)
                expected = [de[eids].mean(), dc[cids].mean(), de[cids].mean(), dc[eids].mean(), np.sort(du)[:20].mean(), np.linalg.norm(current)]
                np.testing.assert_allclose(scores[:, position, q], expected, rtol=2e-6, atol=2e-7)
                snapshots.append(dict(fold=fold, global_row=int(rows[position]), query=q))
        audits.append(dict(info, queries=int(np.isfinite(x).all(-1).sum()), **replay_errors,
            uncentered_query_norm_min=float(np.linalg.norm(x[np.isfinite(x).all(-1)] + profile["dynamic_center"] / profile["dynamic_scale"], axis=1).min()),
            seconds=time.perf_counter() - started))
        print(f"DIAGNOSED {fold}: {audits[-1]['seconds']:.1f}s", flush=True)
    tables = {"alarm_metrics": pd.DataFrame(alarms), "task_alarm_metrics": pd.DataFrame(task_alarms),
        "calibration_peaks": pd.DataFrame(cal_peaks), "calibration_group_peaks": pd.DataFrame(group_peaks),
        "distance_distributions": pd.DataFrame(distance_distributions), "alarm_geometry": pd.DataFrame(geometries),
        "thresholds": pd.DataFrame(threshold_rows), "replay_audit": pd.DataFrame(audits),
        "query_distribution": pd.DataFrame(timings), **case_tables}
    tables["pooled_metrics"] = pooled_counts(tables["alarm_metrics"], ["method", "calibration", "scope"])
    tables["suite_metrics"] = pooled_counts(tables["alarm_metrics"], ["suite", "method", "calibration", "scope"])
    tables["task_metrics"] = pooled_counts(tables["task_alarm_metrics"], ["method", "task", "scope"])
    tables.update(summarize_diagnostics(tables))
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    artifacts = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "verification.json", dict(inputs=inputs, artifacts=artifacts,
        source_sha256=digest(Path(__file__)), independent_scipy_query_replays=sum(a["queries"] for a in audits),
        original_score_max_abs_errors={method: max(a[method] for a in audits) for method in METHODS[:2]},
        direct_factorial_checks=snapshots, original_calibrations_and_first_alarms_match=True,
        distance_decomposition_points=len(geometries), posthoc_diagnostic=True))
    print("COSINE DIAGNOSIS VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round8_cosine_knn")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        analyze(args.parent.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
