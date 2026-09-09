"""Unsupervised MoE geometry, paired projections, and cross-task diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "moe-v7-0905/method"))
from intrinsic_guard_monitor import EPSILON, causal_trailing_mean, intrinsic_score_arrays

SEED = 20260906
BATCH = "right-50x8b-20260903"
REPRESENTATIONS = ("routing_full", "routing_matched", "dynamics_matched")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def dynamics(mobility, acceleration, periodicity, scale):
    base = mobility[:, 1:5].mean(1)
    relative = -np.log(np.maximum(mobility, EPSILON) / np.maximum(base[:, None], EPSILON))
    layers = np.stack([causal_trailing_mean(relative[:, :, i], 6) for i in range(8)], axis=-1)
    heads = intrinsic_score_arrays(mobility, acceleration, periodicity, scale)
    vectors = np.concatenate((layers, heads["acceleration"][..., None], heads["periodicity"][..., None]), axis=-1)
    vectors[:, :7] = np.nan
    return vectors, heads


def sample_points(length, minimum=0, dense=False):
    available = np.arange(minimum, length)
    if dense or len(available) <= 8:
        return available
    return available[np.linspace(0, len(available) - 1, 8).astype(int)]


def choose_cases(frame, events):
    cases = []
    for suite, part in frame.loc[frame.run_id == BATCH].groupby("suite"):
        candidates = events.loc[events.global_row.isin(part.index) & events.drop_goal_release.notna()].copy()
        candidates = candidates.loc[candidates.drop_goal_release >= 7]
        feasible = []
        for row in candidates.itertuples():
            f = frame.loc[row.global_row]
            successes = part.loc[(part.task == f.task) & (part.init_state_id == f.init_state_id) & ~part.failure]
            if len(successes):
                feasible.append((row.drop_goal_release / (f.length - 1), row.global_row, successes.index.tolist()))
        if not feasible:
            raise ValueError(f"no matched physical case for {suite}")
        feasible.sort()
        _, failure_row, successes = feasible[len(feasible) // 2]
        failure = frame.loc[failure_row]
        success_row = min(successes, key=lambda i: (abs(int(frame.loc[i, "noise_seed"]) - int(failure.noise_seed)), i))
        event = float(events.set_index("global_row").loc[failure_row, "drop_goal_release"])
        for role, row in (("failure", failure_row), ("success", success_row)):
            entry = frame.loc[row]
            cases.append({"suite": suite, "role": role, "global_row": int(row), "source": entry.source,
                          "task": entry.task, "episode": int(entry.episode), "length": int(entry.length),
                          "init_state_id": int(entry.init_state_id), "noise_seed": int(entry.noise_seed),
                          "release_query": event if role == "failure" else None})
    return cases


def load_suite(parent, frame, suite, inputs):
    part = frame.loc[frame.suite == suite].copy()
    part["global_row"] = part.index
    if part.checkpoint.nunique() != 1:
        raise ValueError("different checkpoints in one geometry")
    part = part.reset_index(drop=True)
    arrays = {"routing": np.full((len(part), 52, 256), np.nan, np.float32),
              "direct": np.full((len(part), 52, 6), np.nan, np.float32)}
    audits = {a["source"]: a for a in json.loads((parent / "extraction_audit.json").read_text())}
    for source, rows in part.groupby("source").groups.items():
        path = parent / "features" / audits[source]["output"]
        inputs[str(path.relative_to(ROOT))] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            index = pd.DataFrame(json.loads(str(data["index"])))
            positions = index.set_index("episode").index.get_indexer(part.loc[rows, "episode"])
            if (positions < 0).any():
                raise ValueError("missing episode in feature cache")
            for key in ("task", "episode", "length", "init_state_id", "noise_seed", "checkpoint"):
                np.testing.assert_array_equal(index.iloc[positions][key], part.loc[rows, key])
            arrays["routing"][rows] = data["load"][positions]
            arrays["direct"][rows] = data["direct"][positions]
    return part, arrays


def reference_scaling(values, rows, global_scale=False):
    samples = []
    for row in rows:
        valid = np.flatnonzero(np.isfinite(values[row]).all(-1))
        if len(valid):
            samples.append(values[row, valid[np.linspace(0, len(valid) - 1, min(8, len(valid))).astype(int)]])
    samples = np.concatenate(samples)
    center = np.median(samples, axis=0)
    if global_scale:
        scale = np.repeat(max(float(np.median(np.linalg.norm(samples - center, axis=-1))), 1e-6), samples.shape[-1])
    else:
        scale = np.maximum(1.4826 * np.median(np.abs(samples - center), axis=0), 1e-6)
    return ((values - center) / scale).astype(np.float32), {"center": center.tolist(), "scale": scale.tolist(), "points": len(samples)}


def other_task_neighbors(values, tasks, k=20):
    if not np.isfinite(values).all():
        raise ValueError("nonfinite neighbor input")
    result, baselines = [], []
    for start in range(0, len(values), 256):
        stop = min(start + 256, len(values))
        distance = cdist(values[start:stop], values, metric="sqeuclidean")
        eligible = tasks[start:stop, None] != tasks[None]
        if (eligible.sum(1) < k).any():
            raise ValueError("not enough cross-task candidates")
        distance[~eligible] = np.inf
        neighbors = np.argpartition(distance, k - 1, axis=1)[:, :k]
        assert np.all(tasks[neighbors] != tasks[start:stop, None])
        result.append(neighbors)
        baselines.append(eligible)
    return np.concatenate(result), np.concatenate(baselines)


def neighborhood_diagnostics(part, arrays, records):
    for q in (7, 14, 21):
        eligible = (part.run_id == BATCH).to_numpy() & (part.length.to_numpy() > q)
        eligible &= np.isfinite(arrays["dynamics"][:, q]).all(-1)
        rows = np.flatnonzero(eligible)
        current = part.iloc[rows]
        y, tasks = current.failure.to_numpy(), current.task.to_numpy()
        for name in ("routing", "dynamics"):
            neighbors, allowed = other_task_neighbors(arrays[name][rows, q], tasks)
            local_failure = y[neighbors].mean(1)
            base_failure = (allowed * y[None]).sum(1) / allowed.sum(1)
            # One point per recorded trajectory at a fixed absolute query.
            for i, (_, row) in enumerate(current.iterrows()):
                records.append({"suite": row.suite, "representation": name, "query": q,
                    "global_row": int(row.global_row), "task": row.task, "failure": bool(row.failure),
                    "neighbor_failure_rate": float(local_failure[i]), "candidate_failure_rate": float(base_failure[i]),
                    "excess_failure_neighbor_rate": float(local_failure[i] - base_failure[i]),
                    "eligible_episodes": len(rows), "eligible_failures": int(y.sum()),
                    "neighbors": 20})


def nearest_ids(values, k=15):
    index = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=1).fit(values)
    ids = index.kneighbors(values, return_distance=False)
    return np.asarray([row[row != i][:k] for i, row in enumerate(ids)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round3_safe")
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round4_geometry")
    args = parser.parse_args()
    parent, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "projections").mkdir()
    frame = pd.read_csv(parent / "outcome_alignment.csv")
    index = pd.read_csv(parent / "index.csv")
    pd.testing.assert_frame_equal(frame[index.columns], index)
    events = pd.read_csv(parent / "physical_events.csv")
    cases = choose_cases(frame, events)
    write_json(output / "cases.json", cases)
    inputs = {str((parent / f).relative_to(ROOT)): digest(parent / f)
              for f in ("index.csv", "outcome_alignment.csv", "extraction_audit.json", "physical_events.csv", "v7/v7_inputs.npz")}
    with np.load(parent / "v7/v7_inputs.npz", allow_pickle=False) as z:
        mobility, acceleration, periodicity, valid = (z[k] for k in ("mobility", "acceleration", "periodicity", "valid"))
    records, checks, projected, selected_rows, coverage_rows, scalers = [], [], [], [], [], {}
    for suite_index, suite in enumerate(sorted(frame.suite.unique())):
        begin = time.perf_counter()
        part, arrays = load_suite(parent, frame, suite, inputs)
        global_rows = part.global_row.to_numpy()
        global_to_local = {int(g): i for i, g in enumerate(global_rows)}
        m, a, p = mobility[global_rows], acceleration[global_rows], periodicity[global_rows]
        fold = parent / "v7/predictions" / f"{suite}_20260907.npz"
        inputs[str(fold.relative_to(ROOT))] = digest(fold)
        with np.load(fold, allow_pickle=False) as z:
            reference = np.asarray([global_to_local[int(g)] for g in z["reference_rows"]])
        scale = float(np.quantile(np.abs(p[reference][np.isfinite(p[reference])]), .75))
        d, heads = dynamics(m, a, p, scale)
        d[~valid[global_rows]] = np.nan
        arrays["dynamics"], ds = reference_scaling(d, reference)
        arrays["routing"], rs = reference_scaling(arrays["routing"], reference, global_scale=True)
        scalers[suite] = {"routing": rs, "dynamics": ds, "periodicity_scale": scale,
                          "reference_global_rows": global_rows[reference].tolist(), "checkpoint": part.checkpoint.iloc[0]}
        rng = np.random.default_rng(SEED + suite_index)
        chosen = set()
        for (task, failed), group in part.loc[part.run_id == BATCH].groupby(["task", "failure"]):
            selected = rng.choice(group.index.to_numpy(), min(20, len(group)), replace=False)
            chosen.update(selected.tolist())
        local_cases = [c for c in cases if c["suite"] == suite]
        dense = {global_to_local[c["global_row"]] for c in local_cases}
        chosen.update(dense)
        for row in sorted(chosen):
            entry = part.iloc[row].to_dict()
            entry["dense_case"] = row in dense
            selected_rows.append(entry)
        full_points, matched_points = [], []
        for row in sorted(chosen):
            full_points.extend((row, int(q)) for q in sample_points(int(part.loc[row, "length"]), dense=row in dense))
            matched_points.extend((row, int(q)) for q in sample_points(int(part.loc[row, "length"]), 7, row in dense)
                                  if np.isfinite(arrays["dynamics"][row, q]).all())
        points_by_kind = {"full": np.asarray(full_points), "matched": np.asarray(matched_points)}
        for kind, points in points_by_kind.items():
            metadata = part.iloc[points[:, 0]].reset_index(drop=True).copy()
            metadata["query"] = points[:, 1]
            metadata["progress"] = metadata["query"] / (metadata.length - 1)
            metadata["safe_color"] = np.where(metadata.failure, metadata.progress, 0.)
            metadata["eef_motion_m"] = -arrays["direct"][points[:, 0], points[:, 1], 5]
            metadata["freeze_score"] = heads["freeze"][points[:, 0], points[:, 1]]
            metadata["drop_goal_release"] = metadata.global_row.map(events.set_index("global_row").drop_goal_release)
            metadata.to_csv(output / "projections" / f"{suite}_{kind}_points.csv", index=False)
            np.savez_compressed(output / "projections" / f"{suite}_{kind}_vectors.npz",
                routing=arrays["routing"][points[:, 0], points[:, 1]],
                **({"dynamics": arrays["dynamics"][points[:, 0], points[:, 1]]} if kind == "matched" else {}))
            coverage_rows.append({"suite": suite, "view": kind, "episodes": int(metadata.global_row.nunique()),
                "failure_episodes": int(metadata.loc[metadata.failure].global_row.nunique()),
                "points": len(points), "failure_points": int(metadata.failure.sum())})
        for row in dense:
            for stop in (8, min(12, int(part.loc[row, "length"])), int(part.loc[row, "length"])):
                replay, _ = dynamics(m[row:row + 1, :stop], a[row:row + 1, :stop], p[row:row + 1, :stop], scale)
                np.testing.assert_allclose(replay[0], d[row, :stop], equal_nan=True, rtol=1e-6, atol=1e-6)
                checks.append({"suite": suite, "global_row": int(global_rows[row]), "prefix_queries": stop})
        print(f"PREPARED {suite}: full={len(full_points)}, matched={len(matched_points)}", flush=True)
        neighborhood_diagnostics(part, arrays, records)
        for representation in REPRESENTATIONS:
            name, kind = representation.split("_", 1)
            points = points_by_kind[kind]
            x = arrays[name][points[:, 0], points[:, 1]]
            if not np.isfinite(x).all():
                raise ValueError("nonfinite projection vector")
            high_neighbors = nearest_ids(x)
            pca = PCA(n_components=2, svd_solver="full")
            xy_pca = pca.fit_transform(x).astype(np.float32)
            seeds = (SEED,) if kind == "full" else (SEED, SEED + 1, SEED + 2)
            for seed in seeds:
                started = time.perf_counter()
                model = TSNE(n_components=2, perplexity=30, init="pca", max_iter=1000,
                             learning_rate="auto", random_state=seed, n_jobs=1)
                xy = model.fit_transform(x)
                low_neighbors = nearest_ids(xy)
                overlap = np.asarray([len(set(h) & set(l)) / len(h) for h, l in zip(high_neighbors, low_neighbors)])
                name_out = f"{suite}_{representation}_seed{seed}"
                np.savez_compressed(output / "projections" / f"{name_out}.npz", xy=xy, pca=xy_pca,
                    point_global_rows=part.iloc[points[:, 0]].global_row.to_numpy(), point_queries=points[:, 1],
                    pca_explained_variance_ratio=pca.explained_variance_ratio_, neighbor_overlap=overlap)
                projected.append({"suite": suite, "representation": representation, "seed": seed,
                    "points": len(x), "dimensions": x.shape[1], "kl_divergence": float(model.kl_divergence_),
                    "neighbor_overlap_15": float(overlap.mean()), "pca_variance_2d": float(pca.explained_variance_ratio_.sum()),
                    "seconds": time.perf_counter() - started})
                print(f"PROJECTED {name_out}: {projected[-1]['seconds']:.1f}s", flush=True)
        pd.DataFrame(records).to_csv(output / "neighbor_points.csv", index=False)
        pd.DataFrame(projected).to_csv(output / "projection_metrics.csv", index=False)
        print(f"FINISHED {suite}: {time.perf_counter()-begin:.1f}s", flush=True)
    pd.DataFrame(selected_rows).to_csv(output / "selected_episodes.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "sampling_coverage.csv", index=False)
    write_json(output / "scaling.json", scalers)
    write_json(output / "verification.json", {"prefix_replays": checks, "cross_task_neighbors_checked": True,
        "fixed_query_one_point_per_episode": True, "paired_point_sets_identical": True,
        "labels_used_for_sampling_and_annotations": True, "labels_used_for_projection_objective": False,
        "normalization_uses_only_original_seen_reference": True, "new_detector_training": False})
    sources = (Path(__file__), HERE / "PROTOCOL_ZH.md", ROOT / "moe-v7-0905/method/intrinsic_guard_monitor.py")
    artifacts = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", {"inputs": inputs, "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "artifacts": artifacts, "sklearn": sklearn.__version__, "numpy": np.__version__, "post_result_exploration": True,
        "projection_primary_seed": SEED, "perplexity": 30, "max_iter": 1000})
    print(output, flush=True)


if __name__ == "__main__":
    main()
