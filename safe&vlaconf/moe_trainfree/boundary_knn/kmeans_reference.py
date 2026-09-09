"""Fit successful-reference clusters and evaluate frozen online readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import sklearn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from full_corpus_knn import ROOT, HERE, CACHE, OUTPUT as PARENT, validate_split
from cosine_knn import load_npz, normalized_vectors
from core import conformal_threshold, digest, first_alarm, trajectory_peak, write_json

OUTPUT = HERE.parent / "results/round11_kmeans"
CLUSTERS = (1, 2, 4, 8, 16, 32, 64)
VARIANTS = ("centroid_distance", "assigned_radius", "union_radius")
METHODS = ("knn20", "norm_only") + tuple(f"c{c}_{variant}" for c in CLUSTERS for variant in VARIANTS)
ALPHAS = np.asarray((.02, .03, .05, .10))
KINDS = ("episode", "task_init")
PRIMARY_C = 32
MIN_CLUSTER_POINTS = 20


def method_info(method):
    if method in METHODS[:2]:
        return dict(method=method, clusters=0, variant=method)
    count, variant = method.split("_", 1)
    return dict(method=method, clusters=int(count[1:]), variant=variant)


def fit_clusters(bank, count, seed):
    model = KMeans(n_clusters=count, init="k-means++", n_init=10, max_iter=300,
                   tol=1e-4, algorithm="lloyd", random_state=seed).fit(bank)
    distances = cdist(bank, model.cluster_centers_)
    assignment = distances.argmin(1)
    np.testing.assert_array_equal(assignment, model.labels_)
    nearest = distances[np.arange(len(bank)), assignment]
    pool_radius = float(np.quantile(nearest, .9))
    sizes = np.bincount(assignment, minlength=count)
    assert (sizes > 0).all()
    raw = np.asarray([np.quantile(nearest[assignment == c], .9) for c in range(count)])
    fallback = sizes < MIN_CLUSTER_POINTS
    radius = np.maximum(np.where(fallback, pool_radius, raw), 1e-6)
    return dict(centers=model.cluster_centers_, assignments=assignment, counts=sizes,
        radii=radius, raw_radii=raw, pooled_radius=np.asarray(pool_radius), fallback=fallback,
        inertia=np.asarray(model.inertia_), iterations=np.asarray(model.n_iter_), seed=np.asarray(seed))


def score_clusters(x, profile):
    valid = np.isfinite(x).all(-1)
    values = x[valid]
    out = np.full((3, *valid.shape), np.nan, np.float32)
    flat = np.empty((3, len(values)), np.float32)
    identities = np.full((2, *valid.shape), -1, np.int16)
    flat_ids = np.empty((2, len(values)), np.int16)
    for start in range(0, len(values), 4096):
        stop = min(start + 4096, len(values))
        distance = cdist(values[start:stop], profile["centers"])
        nearest = distance.argmin(1)
        relative = distance / profile["radii"]
        flat[0, start:stop] = distance[np.arange(len(nearest)), nearest]
        flat[1, start:stop] = relative[np.arange(len(nearest)), nearest]
        flat[2, start:stop] = relative.min(1)
        flat_ids[:, start:stop] = np.stack((nearest, relative.argmin(1)))
    out[:, valid], identities[:, valid] = flat, flat_ids
    return out, identities


def calibrate_all(scores, labels, frame, ncal):
    success = labels == 0
    groups = (frame.loc[success, "task"] + "|" + frame.loc[success, "init_state_id"].astype(str)).to_numpy()
    thresholds = np.empty((2, len(ALPHAS), len(METHODS)), np.float64)
    first = np.empty((*thresholds.shape, scores.shape[1] - ncal), np.int16)
    records = []
    for mi, method in enumerate(METHODS):
        peaks = trajectory_peak(scores[mi, :ncal][success])
        grouped = np.asarray([peaks[groups == group].max() for group in sorted(set(groups))])
        for kind_i, (kind, units) in enumerate(zip(KINDS, (peaks, grouped))):
            for ai, alpha in enumerate(ALPHAS):
                tau, rank = conformal_threshold(units, alpha)
                thresholds[kind_i, ai, mi] = tau
                first[kind_i, ai, mi] = first_alarm(scores[mi, ncal:], tau)
                records.append(dict(**method_info(method), calibration=kind, alpha=float(alpha),
                    threshold=tau, units=len(units), rank=rank, exceedances=int((units > tau).sum())))
    return thresholds, first, records


def cluster_summary(profile, original, frame, info, count):
    rows, queries = original["success_global_rows"], original["success_queries"]
    records = []
    for c in range(count):
        selected = profile["assignments"] == c
        task_counts = frame.loc[rows[selected], "task"].value_counts()
        records.append(dict(fold=info["fold"], suite=info["suite"], clusters=count, cluster=c,
            points=int(selected.sum()), reference_episodes=len(np.unique(rows[selected])),
            radius=float(profile["radii"][c]), raw_radius=float(profile["raw_radii"][c]),
            pooled_radius=float(profile["pooled_radius"]), fallback=bool(profile["fallback"][c]),
            reference_tasks=len(task_counts), dominant_task=task_counts.index[0],
            dominant_task_fraction=float(task_counts.iloc[0] / selected.sum()),
            first_reference_query=int(queries[selected].min()), last_reference_query=int(queries[selected].max())))
    return records


def check_baseline(data, old):
    ai = int(np.flatnonzero(np.isclose(ALPHAS, .05))[0])
    for mi, original_name in enumerate(("euclidean", "norm_only")):
        old_mi = list(old["methods"]).index(original_name)
        for key in ("scores", "calibration_scores"):
            np.testing.assert_array_equal(data[key][mi], old[key][old_mi])
        np.testing.assert_array_equal(data["thresholds"][:, ai, mi], old["thresholds"][:, old_mi])
        np.testing.assert_array_equal(data["first"][:, ai, mi], old["first"][:, old_mi])


def score_all(output):
    if output.exists():
        raise FileExistsError(output)
    parent = json.loads((PARENT / "sealed_manifest.json").read_text())
    verified = json.loads((PARENT / "verification.json").read_text())
    assert verified["passed"] and verified["sealed_manifest_sha256"] == digest(PARENT / "sealed_manifest.json")
    for category, root in (("sources", ROOT), ("inputs", ROOT), ("artifacts", PARENT)):
        for name, expected in parent[category].items():
            assert digest(root / name) == expected, name
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in
              (PARENT / "sealed_manifest.json", PARENT / "verification.json", PARENT / "index.csv", CACHE)}
    paths = (Path(__file__), HERE / "KMEANS_PROTOCOL_ZH.md", HERE / "full_corpus_knn.py",
        HERE / "cosine_knn.py", HERE / "knn.py", HERE.parent / "feature_geometry/analyze.py",
        HERE.parent / "safe_protocol/core.py", ROOT / "moe-v7-0905/method/intrinsic_guard_monitor.py")
    sources = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    frame, cache = pd.read_csv(PARENT / "index.csv"), load_npz(CACHE)
    output.mkdir(parents=True)
    for directory in ("profiles", "predictions", "calibration"):
        (output / directory).mkdir()
    frame.to_csv(output / "index.csv", index=False)
    contract = dict(sources=sources, sklearn_version=sklearn.__version__, methods=METHODS, clusters=CLUSTERS,
        variants=VARIANTS, alphas=ALPHAS, primary_clusters=PRIMARY_C, primary_alpha=.05,
        primary_calibration="task_init", min_cluster_points=MIN_CLUSTER_POINTS, radius_quantile=.9,
        radius_floor=1e-6, offline_reference_clustering=True, vla_or_moe_parameter_updates=False,
        supervised_classifier_training=False, new_rollouts=False, test_outcomes_used_for_scoring=False,
        historically_explored_data=True, clustering_input="same frozen successful reference chunks as Round 9")
    write_json(output / "scoring_contract.json", contract)
    folds, summaries = [], []
    ownership = np.zeros(len(frame), np.int16)
    for info in parent["folds"]:
        started = time.perf_counter()
        paths = [PARENT / directory / f"{info['fold']}.npz" for directory in ("profiles", "predictions")]
        for path in paths:
            inputs[str(path.relative_to(ROOT))] = digest(path)
        original, old = (load_npz(path) for path in paths)
        validate_split(frame, old["reference_rows"], old["calibration_rows"], old["test_rows"], info)
        needed = np.r_[old["calibration_rows"], old["test_rows"]]
        x = normalized_vectors(cache, needed, original)
        scores = np.full((len(METHODS), *x.shape[:2]), np.nan, np.float32)
        for mi, original_name in enumerate(("euclidean", "norm_only")):
            old_mi = list(old["methods"]).index(original_name)
            scores[mi] = np.concatenate((old["calibration_scores"][old_mi], old["scores"][old_mi]))
        ncal = len(old["calibration_rows"])
        for ci, count in enumerate(CLUSTERS):
            profile = fit_clusters(original["success_dynamic"], count, info["bank_seed"] + 10000 + count)
            values, identities = score_clusters(x, profile)
            scores[2 + 3 * ci:2 + 3 * (ci + 1)] = values
            profile.update(success_global_rows=original["success_global_rows"], success_queries=original["success_queries"],
                checkpoint=original["checkpoint"])
            np.savez_compressed(output / "profiles" / f"{info['fold']}_c{count}.npz", **profile)
            summaries.extend(cluster_summary(profile, original, frame, info, count))
            if count == PRIMARY_C:
                primary_assignments = identities[:, ncal:]
        thresholds, first, records = calibrate_all(scores, old["calibration_labels"],
            frame.loc[old["calibration_rows"]].reset_index(drop=True), ncal)
        data = {key: old[key] for key in ("reference_rows", "reference_labels", "calibration_rows", "calibration_labels", "test_rows", "checkpoint")}
        data.update(methods=np.asarray(METHODS), alphas=ALPHAS, scores=scores[:, ncal:],
            calibration_scores=scores[:, :ncal], thresholds=thresholds, first=first,
            primary_cluster_assignments=primary_assignments)
        check_baseline(data, old)
        np.savez_compressed(output / "predictions" / f"{info['fold']}.npz", **data)
        pd.DataFrame(records).to_csv(output / "calibration" / f"{info['fold']}.csv", index=False)
        ownership[old["test_rows"]] += 1
        fold = dict(info, kmeans_seconds=time.perf_counter() - started)
        folds.append(fold)
        print(f"SEALED {info['fold']}: {len(old['test_rows'])} tests, {fold['kmeans_seconds']:.1f}s", flush=True)
    np.testing.assert_array_equal(ownership, 1)
    pd.DataFrame(summaries).to_csv(output / "cluster_summary.csv", index=False)
    artifacts = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "sealed_manifest.json", dict(contract, inputs=inputs, artifacts=artifacts,
        folds=folds, unique_test_trajectories=len(frame)))
    print("ALL K-MEANS PREDICTIONS SEALED", flush=True)


def verify_all(output):
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    hashes = 0
    for category, root in (("sources", ROOT), ("inputs", ROOT), ("artifacts", output)):
        for name, expected in manifest[category].items():
            assert digest(root / name) == expected, name
            hashes += 1
    assert manifest["sklearn_version"] == sklearn.__version__
    frame, cache = pd.read_csv(output / "index.csv"), load_npz(CACHE)
    ownership = np.zeros(len(frame), np.int16)
    threshold_checks, decisions, points, profiles_checked, refits = 0, 0, 0, 0, 0
    for info in manifest["folds"]:
        data = load_npz(output / "predictions" / f"{info['fold']}.npz")
        old = load_npz(PARENT / "predictions" / f"{info['fold']}.npz")
        original = load_npz(PARENT / "profiles" / f"{info['fold']}.npz")
        np.testing.assert_array_equal(data["methods"], METHODS)
        np.testing.assert_array_equal(data["alphas"], ALPHAS)
        for key in ("reference_rows", "reference_labels", "calibration_rows", "calibration_labels", "test_rows", "checkpoint"):
            np.testing.assert_array_equal(data[key], old[key])
        validate_split(frame, data["reference_rows"], data["calibration_rows"], data["test_rows"], info)
        check_baseline(data, old)
        needed = np.r_[data["calibration_rows"], data["test_rows"]]
        x = normalized_vectors(cache, needed, original)
        scores = np.concatenate((data["calibration_scores"], data["scores"]), axis=1)
        valid = np.isfinite(x).all(-1)
        np.testing.assert_array_equal(valid, cache["valid"][needed] & (np.arange(52)[None] >= 7))
        for score in scores:
            np.testing.assert_array_equal(np.isfinite(score), valid)
        profiles = []
        for ci, count in enumerate(CLUSTERS):
            p = load_npz(output / "profiles" / f"{info['fold']}_c{count}.npz")
            np.testing.assert_array_equal(p["success_global_rows"], original["success_global_rows"])
            np.testing.assert_array_equal(p["success_queries"], original["success_queries"])
            assert set(p["success_global_rows"]).issubset(set(data["reference_rows"]))
            distance = np.linalg.norm(original["success_dynamic"][:, None] - p["centers"][None], axis=-1)
            assignment = distance.argmin(1)
            np.testing.assert_array_equal(assignment, p["assignments"])
            nearest = distance[np.arange(len(distance)), assignment]
            np.testing.assert_allclose(np.sum(nearest ** 2), p["inertia"], rtol=1e-9, atol=1e-9)
            sizes = np.bincount(assignment, minlength=count)
            np.testing.assert_array_equal(sizes, p["counts"])
            raw = np.asarray([np.quantile(nearest[assignment == c], .9) for c in range(count)])
            np.testing.assert_allclose(raw, p["raw_radii"], rtol=1e-12, atol=1e-12)
            pool = np.quantile(nearest, .9)
            np.testing.assert_allclose(pool, p["pooled_radius"], rtol=1e-12, atol=1e-12)
            radius = np.maximum(np.where(sizes < MIN_CLUSTER_POINTS, pool, raw), 1e-6)
            np.testing.assert_allclose(radius, p["radii"], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(sizes < MIN_CLUSTER_POINTS, p["fallback"])
            if count == PRIMARY_C and info["fold_index"] == 0:
                refit = fit_clusters(original["success_dynamic"], count, info["bank_seed"] + 10000 + count)
                for key in refit:
                    np.testing.assert_array_equal(p[key], refit[key])
                refits += 1
            assert (scores[2 + 3 * ci + 2, valid] <= scores[2 + 3 * ci + 1, valid] + 1e-6).all()
            profiles.append(p)
            profiles_checked += 1
        candidates = np.argwhere(valid)
        rng = np.random.default_rng(info["bank_seed"])
        selected = candidates[rng.choice(len(candidates), 12, replace=False)].tolist()
        selected += [list(np.unravel_index(np.nanargmax(scores[mi]), valid.shape)) for mi in (0, 1, METHODS.index("c32_union_radius"))]
        for position, query in selected:
            current = x[position, query]
            reference_distances = np.linalg.norm(original["success_dynamic"] - current, axis=1)
            expected = [np.sort(reference_distances)[:20].mean(), np.linalg.norm(current)]
            for p in profiles:
                distance = np.linalg.norm(p["centers"] - current, axis=1)
                c = int(distance.argmin())
                expected.extend((distance[c], distance[c] / p["radii"][c], np.min(distance / p["radii"])))
            np.testing.assert_allclose(scores[:, position, query], np.asarray(expected, np.float32), rtol=3e-6, atol=3e-7)
            points += 1
        primary_profile = profiles[CLUSTERS.index(PRIMARY_C)]
        _, identities = score_clusters(x[len(data["calibration_rows"]):], primary_profile)
        np.testing.assert_array_equal(data["primary_cluster_assignments"], identities)
        success = data["calibration_labels"] == 0
        cf = frame.loc[data["calibration_rows"]].reset_index(drop=True)
        groups = (cf.loc[success, "task"] + "|" + cf.loc[success, "init_state_id"].astype(str)).to_numpy()
        for mi in range(len(METHODS)):
            raw = data["calibration_scores"][mi, success]
            peaks = np.max(np.where(np.isfinite(raw), raw, -np.inf), axis=1)
            grouped = np.asarray([peaks[groups == group].max() for group in sorted(set(groups))])
            for kind_i, units in enumerate((peaks, grouped)):
                for ai, alpha in enumerate(ALPHAS):
                    rank = int(np.ceil((len(units) + 1) * (1 - alpha)))
                    tau = np.sort(units)[rank - 1] if rank <= len(units) else np.inf
                    assert tau == data["thresholds"][kind_i, ai, mi]
                    assert int((units > tau).sum()) <= len(units) + 1 - rank
                    crossed = np.isfinite(data["scores"][mi]) & (data["scores"][mi] > tau)
                    first = np.where(crossed.any(1), crossed.argmax(1), -1)
                    np.testing.assert_array_equal(first, data["first"][kind_i, ai, mi])
                    assert ((first == -1) | ((first >= 7) & (first < frame.loc[data["test_rows"], "length"].to_numpy()))).all()
                    threshold_checks += 1
                    decisions += len(first)
        ownership[data["test_rows"]] += 1
        print(f"VERIFIED {info['fold']}", flush=True)
    np.testing.assert_array_equal(ownership, 1)
    result = dict(passed=True, hash_checks=hashes, unique_test_trajectories=len(frame), cluster_profiles_verified=profiles_checked,
        deterministic_primary_refits=refits, independently_recomputed_distance_points=points, methods_per_point=len(METHODS),
        independently_recomputed_thresholds=threshold_checks, independently_recomputed_trajectory_decisions=decisions,
        unchanged_baselines_at_alpha005=True, sealed_manifest_sha256=digest(output / "sealed_manifest.json"))
    write_json(output / "verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (score_all if args.command == "score" else verify_all)(args.output.resolve())


if __name__ == "__main__":
    main()
