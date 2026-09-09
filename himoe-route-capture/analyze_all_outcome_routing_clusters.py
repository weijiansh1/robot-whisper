#!/usr/bin/env python3
"""Jointly cluster normalized middle/late routing for success and failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shutil
import time
from collections import Counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from analyze_failure_routing_clusters import (
    ALL_VIEWS,
    CACHE_ROOT,
    EXPECTED_EPISODES_PER_RUN,
    EXPECTED_FORMAL_RUNS,
    PRIMARY_PHASE,
    PRIMARY_VIEWS,
    PhaseConfig,
    association_summary,
    build_representations,
    canonicalize,
    cluster_color_scale,
    discover_runs,
    matched_cluster_jaccard,
    normalize_probabilities,
    prepare_embedding,
    query_tapes,
    sha256_file,
    task_key,
    ward_labels,
)


HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "analysis/all-outcome-routing-clusters"
METHOD_PATH = OUT_DIR / "METHOD.md"
FAILURE_ASSIGNMENTS = (
    HERE / "analysis/failure-routing-clusters/assignments.csv"
)

EXPECTED_RUN_QUERIES = {
    "libero_goal/open_the_middle_drawer_of_the_cabinet": 6452,
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside": 10018,
    "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": 22883,
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": 5139,
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": 6816,
}
EXPECTED_EPISODES = 2560
EXPECTED_FAILURES = 307
EXPECTED_SUCCESSES = 2253
EXPECTED_QUERIES = 51308

CANDIDATE_K = tuple(range(2, 7))
SUBSAMPLES = 50
ASSOCIATION_PERMUTATIONS = 5000
CONSENSUS_PROBE = 512
SILHOUETTE_SAMPLE = 1000
SUBSAMPLE_SILHOUETTE_SAMPLE = 512
OUTCOME_EXCESS_NMI_MIN = 0.05
TASK_SHADOW_NMI = 0.50
TASK_SHADOW_CLUSTER_FRACTION = 0.90
PRESERVED_ARI = 0.80
REORGANIZED_ARI = 0.50

GRID5_PHASE = PhaseConfig("grid5", 0.5, 1.0, 5)
TRUNCATE90_PHASE = PhaseConfig("truncate90", 0.5, 0.9, 10)
SENSITIVITY_PHASES = (GRID5_PHASE, TRUNCATE90_PHASE)
CACHE_SCHEMA = "himoe.all_outcome_routing_features.v1"
_INPUT_FINGERPRINT_MEMO: dict[pathlib.Path, dict[str, str]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--feature-cache", type=pathlib.Path)
    parser.add_argument("--subsamples", type=int, default=SUBSAMPLES)
    parser.add_argument(
        "--association-permutations", type=int, default=ASSOCIATION_PERMUTATIONS
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def minimum_cluster_size(count: int) -> int:
    return max(16, int(math.ceil(0.02 * count)))


def _save_array(path: pathlib.Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def _write_feature_cache(
    cache_dir: pathlib.Path,
    arrays: dict[str, np.ndarray],
    audit: dict[str, Any],
    cache_root: pathlib.Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    version = str(time.time_ns())
    for name, values in arrays.items():
        filename = f"{name}.{version}.npy"
        _save_array(cache_dir / filename, np.asarray(values))
        files[name] = {
            "file": filename,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": sha256_file(cache_dir / filename),
        }
    manifest = {
        "schema": CACHE_SCHEMA,
        "cache_root": str(cache_root.resolve()),
        "extractor_sha256": sha256_file(pathlib.Path(__file__)),
        "feature_core_sha256": sha256_file(
            HERE / "analyze_failure_routing_clusters.py"
        ),
        "router_feature_sha256": sha256_file(
            HERE / "analyze_single_chunk_early_signal.py"
        ),
        "input_fingerprints": _input_fingerprints(cache_root),
        "audit": audit,
        "arrays": files,
    }
    temporary = cache_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    temporary.replace(cache_dir / "manifest.json")


def _load_feature_cache(
    cache_dir: pathlib.Path, cache_root: pathlib.Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CACHE_SCHEMA:
        raise ValueError("feature cache schema mismatch")
    if pathlib.Path(manifest["cache_root"]).resolve() != cache_root.resolve():
        raise ValueError("feature cache points to a different Hub root")
    if manifest.get("extractor_sha256") != sha256_file(pathlib.Path(__file__)):
        raise ValueError("feature extractor changed; rerun with --force-features")
    current_core = sha256_file(HERE / "analyze_failure_routing_clusters.py")
    if manifest.get("feature_core_sha256") != current_core:
        raise ValueError("feature core changed; rerun with --force-features")
    if manifest.get("router_feature_sha256") != sha256_file(
        HERE / "analyze_single_chunk_early_signal.py"
    ):
        raise ValueError("router feature implementation changed; rerun with --force-features")
    if manifest.get("input_fingerprints") != _input_fingerprints(cache_root):
        raise ValueError("formal input metadata changed; rerun with --force-features")
    arrays = {}
    for name, item in manifest["arrays"].items():
        path = cache_dir / item["file"]
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != item["shape"] or str(values.dtype) != item["dtype"]:
            raise ValueError(f"cached array metadata mismatch for {name}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"cached array hash mismatch for {name}")
        arrays[name] = values
    return arrays, manifest["audit"]


def _validate_formal_runs(cache_root: pathlib.Path) -> list[pathlib.Path]:
    runs = discover_runs(cache_root)
    discovered = {task_key(run, cache_root) for run in runs}
    if discovered != set(EXPECTED_RUN_QUERIES):
        raise ValueError(
            "formal run set drifted: missing=%s extra=%s"
            % (
                sorted(set(EXPECTED_RUN_QUERIES) - discovered),
                sorted(discovered - set(EXPECTED_RUN_QUERIES)),
            )
        )
    return runs


def _input_fingerprints(cache_root: pathlib.Path) -> dict[str, str]:
    resolved = cache_root.resolve()
    if resolved in _INPUT_FINGERPRINT_MEMO:
        return dict(_INPUT_FINGERPRINT_MEMO[resolved])
    result = {}
    for run in _validate_formal_runs(cache_root):
        for relative in (
            "meta.json",
            "client/summaries.json",
            "client/server_metadata.json",
            "server/routes.zarr/zarr.json",
            "server/routes.zarr/hb_router_probs/zarr.json",
            "server/routes.zarr/hb_expert_ids/zarr.json",
            "server/routes.zarr/episode_id/zarr.json",
            "server/routes.zarr/control_step/zarr.json",
        ):
            path = run / relative
            result[str(path.relative_to(cache_root))] = sha256_file(path)
        route_root = run / "server/routes.zarr"
        digest = hashlib.sha256()
        for path in sorted(item for item in route_root.rglob("*") if item.is_file()):
            relative = str(path.relative_to(route_root))
            digest.update(len(relative).to_bytes(4, "little"))
            digest.update(relative.encode("utf-8"))
            digest.update(bytes.fromhex(sha256_file(path)))
        result[
            str(route_root.relative_to(cache_root)) + "::content_tree_sha256"
        ] = digest.hexdigest()
    _INPUT_FINGERPRINT_MEMO[resolved] = dict(result)
    return result


def extract_all_features(
    cache_root: pathlib.Path, cache_dir: pathlib.Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    feature_lists = {view: [] for view in ALL_VIEWS}
    sensitivity_lists = {
        config.name: {view: [] for view in PRIMARY_VIEWS}
        for config in SENSITIVITY_PHASES
    }
    diagnostic_lists: dict[str, list[Any]] = {
        "soft_speed_curve": [],
        "hard_speed_curve": [],
        "mean_soft_speed": [],
        "late_soft_speed": [],
        "middle_to_terminal_drift": [],
        "backward_return_fraction": [],
    }
    metadata: dict[str, list[Any]] = {
        "task": [],
        "checkpoint_sha256": [],
        "episode": [],
        "init_state_id": [],
        "flow_noise_seed": [],
        "episode_length": [],
        "failure": [],
    }
    coverage = []
    probability_sum_min = float("inf")
    probability_sum_max = float("-inf")

    for run in _validate_formal_runs(cache_root):
        key = task_key(run, cache_root)
        run_meta = json.loads((run / "meta.json").read_text())
        sampling = run_meta.get("sampling", {})
        if (
            run_meta.get("status") != "complete"
            or not sampling.get("complete", False)
            or int(sampling.get("actual_episodes", -1)) != EXPECTED_EPISODES_PER_RUN
            or int(sampling.get("designed_episodes", -1)) != EXPECTED_EPISODES_PER_RUN
            or int(sampling.get("unique_init_states", -1)) != 16
            or int(sampling.get("unique_flow_noise_seeds", -1)) != 32
        ):
            raise ValueError(f"incomplete formal run: {key}")
        rows = sorted(
            json.loads((run / "client/summaries.json").read_text()),
            key=lambda row: int(row["episode_index"]),
        )
        episode_ids = np.asarray([int(row["episode_index"]) for row in rows])
        if len(rows) != EXPECTED_EPISODES_PER_RUN or not np.array_equal(
            episode_ids, np.arange(EXPECTED_EPISODES_PER_RUN)
        ):
            raise ValueError(f"episode IDs drifted for {key}")
        initial_states = np.asarray([int(row["init_state_id"]) for row in rows])
        flow_seeds = np.asarray([int(row["flow_noise_seed"]) for row in rows])
        design_pairs = set(zip(initial_states.tolist(), flow_seeds.tolist()))
        initial_counts = Counter(initial_states.tolist())
        seed_counts = Counter(flow_seeds.tolist())
        if (
            len(design_pairs) != EXPECTED_EPISODES_PER_RUN
            or len(initial_counts) != 16
            or set(initial_counts.values()) != {32}
            or len(seed_counts) != 32
            or set(seed_counts.values()) != {16}
        ):
            raise ValueError(f"16 x 32 task design drifted for {key}")
        lengths = np.asarray([int(row["inference_calls"]) for row in rows])
        if int(lengths.sum()) != EXPECTED_RUN_QUERIES[key]:
            raise ValueError(f"query count drifted for {key}")
        offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        expected_episode = np.repeat(episode_ids, lengths)
        if not np.array_equal(np.asarray(route["episode_id"][:]), expected_episode):
            raise ValueError(f"episode alignment failed for {key}")
        if not np.array_equal(
            np.asarray(route["control_step"][:]), np.arange(int(lengths.sum()))
        ):
            raise ValueError(f"control-step alignment failed for {key}")
        server_meta = json.loads((run / "client/server_metadata.json").read_text())
        checkpoint = str(server_meta.get("checkpoint_sha256"))
        failures = sum(not bool(row["success"]) for row in rows)
        if failures != EXPECTED_FORMAL_RUNS[key][0]:
            raise ValueError(f"failure count drifted for {key}")

        for row in rows:
            episode = int(row["episode_index"])
            start = int(offsets[episode])
            stop = start + int(lengths[episode])
            raw = np.asarray(route["hb_router_probs"][start:stop], np.float32)
            ids = np.asarray(route["hb_expert_ids"][start:stop], np.uint8)
            normalized, low, high = normalize_probabilities(raw)
            probability_sum_min = min(probability_sum_min, low)
            probability_sum_max = max(probability_sum_max, high)
            tapes = query_tapes(normalized, ids)
            primary, diagnostics = build_representations(tapes, PRIMARY_PHASE)
            for view in ALL_VIEWS:
                feature_lists[view].append(primary[view])
            for name in diagnostic_lists:
                diagnostic_lists[name].append(diagnostics[name])
            for config in SENSITIVITY_PHASES:
                variants, _unused = build_representations(tapes, config)
                for view in PRIMARY_VIEWS:
                    sensitivity_lists[config.name][view].append(variants[view])
            metadata["task"].append(key)
            metadata["checkpoint_sha256"].append(checkpoint)
            metadata["episode"].append(episode)
            metadata["init_state_id"].append(int(row["init_state_id"]))
            metadata["flow_noise_seed"].append(int(row["flow_noise_seed"]))
            metadata["episode_length"].append(int(row["inference_calls"]))
            metadata["failure"].append(not bool(row["success"]))

        coverage.append(
            {
                "task": key,
                "episodes": len(rows),
                "successes": len(rows) - failures,
                "failures": failures,
                "queries": int(lengths.sum()),
                "length_min": int(lengths.min()),
                "length_median": float(np.median(lengths)),
                "length_max": int(lengths.max()),
                "checkpoint_sha256": checkpoint,
            }
        )
        print(f"extracted {key}: {len(rows) - failures} success / {failures} failure", flush=True)

    arrays: dict[str, np.ndarray] = {}
    for view, values in feature_lists.items():
        arrays[f"feature_primary_{view}"] = np.stack(values).astype(np.float32)
    for config, views in sensitivity_lists.items():
        for view, values in views.items():
            arrays[f"feature_{config}_{view}"] = np.stack(values).astype(np.float32)
    for name, values in diagnostic_lists.items():
        arrays[f"diagnostic_{name}"] = np.stack(values).astype(np.float32)
    arrays.update(
        {
            "meta_task": np.asarray(metadata["task"]),
            "meta_checkpoint_sha256": np.asarray(metadata["checkpoint_sha256"]),
            "meta_episode": np.asarray(metadata["episode"], dtype=np.int32),
            "meta_init_state_id": np.asarray(metadata["init_state_id"], dtype=np.int32),
            "meta_flow_noise_seed": np.asarray(metadata["flow_noise_seed"], dtype=np.int32),
            "meta_episode_length": np.asarray(metadata["episode_length"], dtype=np.int32),
            "meta_failure": np.asarray(metadata["failure"], dtype=bool),
        }
    )
    failures = int(arrays["meta_failure"].sum())
    audit = {
        "coverage": coverage,
        "episodes": len(arrays["meta_failure"]),
        "successes": int(len(arrays["meta_failure"]) - failures),
        "failures": failures,
        "queries": int(sum(row["queries"] for row in coverage)),
        "probability_sum_min": probability_sum_min,
        "probability_sum_max": probability_sum_max,
        "feature_shapes": {
            name: list(values.shape)
            for name, values in arrays.items()
            if name.startswith("feature_")
        },
    }
    if (
        audit["episodes"] != EXPECTED_EPISODES
        or audit["successes"] != EXPECTED_SUCCESSES
        or audit["failures"] != EXPECTED_FAILURES
        or audit["queries"] != EXPECTED_QUERIES
    ):
        raise ValueError(f"formal all-outcome cohort drifted: {audit}")
    _write_feature_cache(cache_dir, arrays, audit, cache_root)
    return _load_feature_cache(cache_dir, cache_root)


def load_or_extract_features(
    cache_root: pathlib.Path, cache_dir: pathlib.Path, force: bool
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not force and (cache_dir / "manifest.json").exists():
        print(f"loading feature cache {cache_dir}", flush=True)
        return _load_feature_cache(cache_dir, cache_root)
    return extract_all_features(cache_root, cache_dir)


def _silhouette(
    embedding: np.ndarray, labels: np.ndarray, sample_size: int, seed: int
) -> float:
    return float(
        silhouette_score(
            embedding,
            labels,
            sample_size=min(sample_size, len(embedding)),
            random_state=seed,
        )
    )


def cluster_joint_view(
    matrix: np.ndarray,
    view: str,
    score: np.ndarray,
    subsamples: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    embedding, preprocessing = prepare_embedding(matrix, view, seed)
    count = len(embedding)
    minimum = minimum_cluster_size(count)
    full_labels = {clusters: ward_labels(embedding, clusters) for clusters in CANDIDATE_K}
    rng = np.random.default_rng(seed + 1000)
    probe = np.sort(rng.choice(count, min(CONSENSUS_PROBE, count), replace=False))
    full_silhouette_index = np.sort(
        rng.choice(count, min(SILHOUETTE_SAMPLE, count), replace=False)
    )
    trials: dict[int, dict[str, Any]] = {}
    trackers = {}
    for clusters, labels in full_labels.items():
        sizes = np.bincount(labels, minlength=clusters)
        trials[clusters] = {
            "clusters": clusters,
            "full_silhouette_sampled": float(
                silhouette_score(
                    embedding[full_silhouette_index], labels[full_silhouette_index]
                )
            ),
            "sizes": sizes.tolist(),
            "minimum_size_pass": bool(sizes.min() >= minimum),
        }
        trackers[clusters] = {
            "ari": [],
            "jaccard": [],
            "silhouette": [],
            "pair_count": np.zeros((len(probe), len(probe)), dtype=np.uint16),
            "pair_same": np.zeros((len(probe), len(probe)), dtype=np.uint16),
        }

    sample_size = int(math.ceil(0.80 * count))
    for draw in range(subsamples):
        index = np.sort(rng.choice(count, sample_size, replace=False))
        subset, _unused = prepare_embedding(
            matrix[index], view, seed + 100000 + draw
        )
        tree = linkage(subset, method="ward", optimal_ordering=False)
        silhouette_index = np.sort(
            rng.choice(
                len(subset), min(SUBSAMPLE_SILHOUETTE_SAMPLE, len(subset)), replace=False
            )
        )
        probe_mask = np.isin(probe, index)
        probe_global = probe[probe_mask]
        probe_position = np.flatnonzero(probe_mask)
        subset_probe_position = np.searchsorted(index, probe_global)
        pair_index = np.ix_(probe_position, probe_position)
        for clusters in CANDIDATE_K:
            labels = fcluster(tree, clusters, criterion="maxclust").astype(np.int32) - 1
            if len(np.unique(labels)) != clusters:
                continue
            tracker = trackers[clusters]
            reference = full_labels[clusters][index]
            tracker["ari"].append(adjusted_rand_score(reference, labels))
            tracker["jaccard"].append(matched_cluster_jaccard(reference, labels))
            tracker["silhouette"].append(
                float(
                    silhouette_score(
                        subset[silhouette_index], labels[silhouette_index]
                    )
                )
            )
            probe_labels = labels[subset_probe_position]
            tracker["pair_count"][pair_index] += 1
            tracker["pair_same"][pair_index] += (
                probe_labels[:, None] == probe_labels[None, :]
            )

    for clusters in CANDIDATE_K:
        tracker = trackers[clusters]
        ari = np.asarray(tracker["ari"], dtype=np.float64)
        jaccard = np.asarray(tracker["jaccard"], dtype=np.float64)
        silhouettes = np.asarray(tracker["silhouette"], dtype=np.float64)
        if len(ari) != subsamples:
            raise RuntimeError(f"incomplete subsampling for {view} K={clusters}")
        valid = (tracker["pair_count"] > 0) & ~np.eye(len(probe), dtype=bool)
        consensus = tracker["pair_same"][valid] / tracker["pair_count"][valid]
        trial = trials[clusters]
        trial.update(
            {
                "subsamples_completed": len(ari),
                "ari_median": float(np.median(ari)),
                "ari_p10": float(np.quantile(ari, 0.10)),
                "matched_jaccard_median": float(np.median(jaccard)),
                "matched_jaccard_p10": float(np.quantile(jaccard, 0.10)),
                "subsample_silhouette_mean": float(silhouettes.mean()),
                "subsample_silhouette_sd": float(silhouettes.std(ddof=1)),
                "consensus_probe_episodes": len(probe),
                "consensus_pac_01_09": float(
                    np.mean((consensus > 0.1) & (consensus < 0.9))
                ),
            }
        )
        trial["stable"] = bool(
            trial["minimum_size_pass"]
            and trial["ari_median"] >= 0.75
            and trial["ari_p10"] >= 0.50
            and trial["consensus_pac_01_09"] <= 0.20
        )

    stable = [trials[k] for k in CANDIDATE_K if trials[k]["stable"]]
    if stable:
        selected = max(stable, key=lambda row: row["subsample_silhouette_mean"])
        selected_k = int(selected["clusters"])
        status = "stable_partition"
    else:
        selected_k = None
        status = "no_stable_partition"
    size_valid = [row for row in trials.values() if row["minimum_size_pass"]]
    exploratory = max(
        size_valid if size_valid else list(trials.values()),
        key=lambda row: row["full_silhouette_sampled"],
    )["clusters"]
    reported_k = selected_k if selected_k is not None else int(exploratory)
    labels = canonicalize(full_labels[reported_k], score)
    return (
        {
            "view": view,
            "status": status,
            "selected_k": selected_k,
            "exploratory_best_k": int(exploratory),
            "reported_k": reported_k,
            "reported_sizes": np.bincount(
                labels, minlength=reported_k
            ).tolist(),
            "minimum_cluster_size": minimum,
            "preprocessing": preprocessing,
            "trials": [trials[k] for k in CANDIDATE_K],
        },
        labels,
        embedding,
    )


def _metadata(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name.removeprefix("meta_"): np.asarray(values)
        for name, values in arrays.items()
        if name.startswith("meta_")
    }


def _task_initial_state(metadata: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            f"{task}::{initial}"
            for task, initial in zip(metadata["task"], metadata["init_state_id"])
        ]
    )


def posthoc_associations(
    metadata: dict[str, np.ndarray],
    labels: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    tasks = metadata["task"]
    task_initial = _task_initial_state(metadata)
    outcome = np.where(metadata["failure"], "failure", "success")
    values = {
        "outcome_within_task_initial_state": (
            outcome,
            task_initial,
            "within_task_initial_state",
        ),
        "task": (tasks, tasks, "global"),
        "checkpoint": (metadata["checkpoint_sha256"], tasks, "global"),
        "episode_length": (
            metadata["episode_length"].astype(str),
            tasks,
            "global",
        ),
    }
    return {
        name: association_summary(
            categories,
            labels,
            strata,
            permutations,
            seed + index,
            permutation,
        )
        for index, (name, (categories, strata, permutation)) in enumerate(
            values.items()
        )
    }


def add_bh_correction(results: dict[str, dict[str, Any]]) -> None:
    entries = [
        association
        for view in ALL_VIEWS
        for association in results[view]["posthoc_association"].values()
    ]
    p_values = np.asarray(
        [entry["permutation_p_one_sided"] for entry in entries], dtype=np.float64
    )
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.minimum(adjusted, 1.0)
    for entry, value in zip(entries, q_values):
        entry["fdr_bh_q"] = float(value)


def outcome_profiles(
    labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    profiles = []
    for cluster in np.unique(labels):
        index = labels == cluster
        failures = int(metadata["failure"][index].sum())
        task_counts = Counter(metadata["task"][index])
        lengths = metadata["episode_length"][index]
        profiles.append(
            {
                "cluster": int(cluster),
                "episodes": int(index.sum()),
                "successes": int(index.sum() - failures),
                "failures": failures,
                "failure_rate": float(failures / index.sum()),
                "task_counts": dict(sorted(task_counts.items())),
                "dominant_task_fraction": float(max(task_counts.values()) / index.sum()),
                "length_min": int(lengths.min()),
                "length_median": float(np.median(lengths)),
                "length_max": int(lengths.max()),
                "mean_soft_speed": float(diagnostics["mean_soft_speed"][index].mean()),
                "late_soft_speed": float(diagnostics["late_soft_speed"][index].mean()),
                "middle_to_terminal_drift": float(
                    diagnostics["middle_to_terminal_drift"][index].mean()
                ),
            }
        )
    return profiles


def _conditional_entropy(left: np.ndarray, right: np.ndarray) -> float:
    result = 0.0
    for value in np.unique(left):
        subset = right[left == value]
        counts = np.bincount(subset.astype(np.int64))
        probability = counts[counts > 0] / len(subset)
        result += len(subset) / len(left) * float(
            -np.sum(probability * np.log2(probability))
        )
    return result


def _change_class(ari: float) -> str:
    if ari >= PRESERVED_ARI:
        return "preserved"
    if ari < REORGANIZED_ARI:
        return "reorganized"
    return "moderately_changed"


def load_failure_only_labels() -> tuple[dict[tuple[str, int], dict[str, int]], dict[str, str]]:
    with FAILURE_ASSIGNMENTS.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_FAILURES:
        raise ValueError("failure-only assignment cohort mismatch")
    labels = {
        (row["task"], int(row["episode"])): {
            view: int(row[f"{view}_cluster"]) for view in ALL_VIEWS
        }
        for row in rows
    }
    statuses = {view: rows[0][f"{view}_partition_status"] for view in ALL_VIEWS}
    if len(labels) != len(rows):
        raise ValueError("duplicate failure-only assignment keys")
    return labels, statuses


def compare_failure_partitions(
    view: str,
    joint_labels: np.ndarray,
    joint_embedding: np.ndarray,
    metadata: dict[str, np.ndarray],
    score: np.ndarray,
    old_map: dict[tuple[str, int], dict[str, int]],
    old_statuses: dict[str, str],
) -> tuple[dict[str, Any], np.ndarray]:
    failure_index = np.flatnonzero(metadata["failure"])
    keys = [
        (str(metadata["task"][index]), int(metadata["episode"][index]))
        for index in failure_index
    ]
    if set(keys) != set(old_map):
        raise ValueError("joint/failure-only failure keys do not match")
    old = np.asarray([old_map[key][view] for key in keys], dtype=np.int32)
    old_k = len(np.unique(old))
    selected = joint_labels[failure_index]
    same_k_full = canonicalize(ward_labels(joint_embedding, old_k), score)
    same_k = same_k_full[failure_index]
    tasks = metadata["task"][failure_index]
    within_task = {}
    for task in np.unique(tasks):
        index = tasks == task
        if len(np.unique(old[index])) >= 2 and len(np.unique(same_k[index])) >= 2:
            within_task[str(task)] = {
                "episodes": int(index.sum()),
                "ari_same_k": float(adjusted_rand_score(old[index], same_k[index])),
            }
    same_values = np.unique(same_k)
    table = np.zeros((old_k, len(same_values)), dtype=np.int64)
    for left in range(table.shape[0]):
        for column, right in enumerate(same_values):
            table[left, column] = int(np.sum((old == left) & (same_k == right)))
    selected_ari = float(adjusted_rand_score(old, selected))
    same_k_ari = float(adjusted_rand_score(old, same_k))
    return (
        {
            "failure_only_status": old_statuses[view],
            "failure_only_k": old_k,
            "joint_reported_k": len(np.unique(joint_labels)),
            "joint_failure_subset_k": len(np.unique(selected)),
            "selected_k_ari": selected_ari,
            "same_k_ari": same_k_ari,
            "same_k_matched_jaccard": matched_cluster_jaccard(old, same_k),
            "selected_k_change_class": _change_class(selected_ari),
            "same_k_change_class": _change_class(same_k_ari),
            "split_entropy_joint_given_old_bits": _conditional_entropy(old, same_k),
            "merge_entropy_old_given_joint_bits": _conditional_entropy(same_k, old),
            "joint_same_k_column_labels": same_values.tolist(),
            "old_by_joint_same_k_table": table.tolist(),
            "within_task": within_task,
        },
        same_k_full,
    )


def maximum_unique_matches(
    metadata: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    strata = _task_initial_state(metadata)
    failure_matches = []
    success_matches = []
    for stratum in np.unique(strata):
        group = np.flatnonzero(strata == stratum)
        failures = group[metadata["failure"][group]]
        successes = group[~metadata["failure"][group]]
        if len(failures) == 0 or len(successes) == 0:
            continue
        failures = failures[
            np.lexsort(
                (metadata["episode"][failures], metadata["flow_noise_seed"][failures])
            )
        ]
        successes = successes[
            np.lexsort(
                (metadata["episode"][successes], metadata["flow_noise_seed"][successes])
            )
        ]
        matched = min(len(failures), len(successes))
        failure_rank = np.floor(
            (np.arange(matched) + 0.5) * len(failures) / matched
        ).astype(int)
        success_rank = np.floor(
            (np.arange(matched) + 0.5) * len(successes) / matched
        ).astype(int)
        failure_matches.extend(failures[failure_rank].tolist())
        success_matches.extend(successes[success_rank].tolist())
    order = np.argsort(failure_matches)
    failure_array = np.asarray(failure_matches, dtype=np.int32)[order]
    success_array = np.asarray(success_matches, dtype=np.int32)[order]
    if len(failure_array) != 157 or len(np.unique(success_array)) != len(success_array):
        raise ValueError(
            f"expected 157 unique task/init matched pairs, found {len(failure_array)}"
        )
    return failure_array, success_array


def extract_matched_failure_prefixes(
    cache_root: pathlib.Path,
    cache_dir: pathlib.Path,
    metadata: dict[str, np.ndarray],
    failure_index: np.ndarray,
    success_index: np.ndarray,
    force: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    manifest_path = cache_dir / "manifest.json"
    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema") == "himoe.matched_prefix_features.v1"
            and pathlib.Path(manifest.get("cache_root", "")).resolve()
            == cache_root.resolve()
            and manifest.get("extractor_sha256")
            == sha256_file(pathlib.Path(__file__))
            and manifest.get("feature_core_sha256")
            == sha256_file(HERE / "analyze_failure_routing_clusters.py")
            and manifest.get("router_feature_sha256")
            == sha256_file(HERE / "analyze_single_chunk_early_signal.py")
            and manifest.get("input_fingerprints")
            == _input_fingerprints(cache_root)
        ):
            loaded = {}
            for name, item in manifest["arrays"].items():
                path = cache_dir / item["file"]
                if sha256_file(path) != item["sha256"]:
                    raise ValueError(f"matched-prefix cache hash mismatch: {name}")
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(values.shape) != item["shape"] or str(values.dtype) != item["dtype"]:
                    raise ValueError(f"matched-prefix cache metadata mismatch: {name}")
                loaded[name] = values
            cached_failure = loaded["failure_index"]
            cached_success = loaded["success_index"]
            if not np.array_equal(cached_failure, failure_index) or not np.array_equal(
                cached_success, success_index
            ):
                raise ValueError("matched-prefix cache pair indices changed")
            features = {
                view: loaded[f"feature_{view}"]
                for view in ALL_VIEWS
            }
            score = loaded["mean_soft_speed"]
            return features, score

    cache_dir.mkdir(parents=True, exist_ok=True)
    run_data = {}
    for run in _validate_formal_runs(cache_root):
        key = task_key(run, cache_root)
        rows = sorted(
            json.loads((run / "client/summaries.json").read_text()),
            key=lambda row: int(row["episode_index"]),
        )
        lengths = np.asarray([int(row["inference_calls"]) for row in rows])
        run_data[key] = {
            "offsets": np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64),
            "lengths": lengths,
            "route": zarr.open_group(str(run / "server/routes.zarr"), mode="r"),
        }
    feature_lists = {view: [] for view in ALL_VIEWS}
    score = []
    for position, (failure, success) in enumerate(zip(failure_index, success_index)):
        task = str(metadata["task"][failure])
        episode = int(metadata["episode"][failure])
        target_length = int(metadata["episode_length"][success])
        data = run_data[task]
        if target_length > int(data["lengths"][episode]):
            raise ValueError("matched success is longer than its paired failure")
        start = int(data["offsets"][episode])
        stop = start + target_length
        raw = np.asarray(data["route"]["hb_router_probs"][start:stop], np.float32)
        ids = np.asarray(data["route"]["hb_expert_ids"][start:stop], np.uint8)
        normalized, _low, _high = normalize_probabilities(raw)
        representations, diagnostics = build_representations(
            query_tapes(normalized, ids), PRIMARY_PHASE
        )
        for view in ALL_VIEWS:
            feature_lists[view].append(representations[view])
        score.append(diagnostics["mean_soft_speed"])
        if (position + 1) % 40 == 0:
            print(f"extracted matched prefixes: {position + 1}/{len(failure_index)}", flush=True)

    features = {
        view: np.stack(values).astype(np.float32)
        for view, values in feature_lists.items()
    }
    score_array = np.asarray(score, dtype=np.float32)
    version = str(time.time_ns())
    payload = {
        "failure_index": failure_index,
        "success_index": success_index,
        "mean_soft_speed": score_array,
        **{f"feature_{view}": values for view, values in features.items()},
    }
    files = {}
    for name, values in payload.items():
        filename = f"{name}.{version}.npy"
        path = cache_dir / filename
        _save_array(path, values)
        files[name] = {
            "file": filename,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": sha256_file(path),
        }
    temporary = cache_dir / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "schema": "himoe.matched_prefix_features.v1",
                "pairs": len(failure_index),
                "cache_root": str(cache_root.resolve()),
                "extractor_sha256": sha256_file(pathlib.Path(__file__)),
                "feature_core_sha256": sha256_file(
                    HERE / "analyze_failure_routing_clusters.py"
                ),
                "router_feature_sha256": sha256_file(
                    HERE / "analyze_single_chunk_early_signal.py"
                ),
                "input_fingerprints": _input_fingerprints(cache_root),
                "arrays": files,
            },
            indent=2,
        )
    )
    temporary.replace(manifest_path)
    return features, score_array


def fixed_partition(
    matrix: np.ndarray,
    view: str,
    clusters: int,
    score: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    embedding, preprocessing = prepare_embedding(matrix, view, seed)
    labels = canonicalize(ward_labels(embedding, clusters), score)
    sizes = np.bincount(labels, minlength=clusters)
    minimum = minimum_cluster_size(len(labels))
    return labels, {
        "episodes": len(labels),
        "fixed_k": clusters,
        "sizes": sizes.tolist(),
        "minimum_cluster_size": minimum,
        "minimum_size_pass": bool(sizes.min() >= minimum),
        "silhouette_sampled": _silhouette(
            embedding, labels, SILHOUETTE_SAMPLE, seed + 1
        ),
        "preprocessing": preprocessing,
    }


def _old_labels_for_indices(
    view: str,
    indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    old_map: dict[tuple[str, int], dict[str, int]],
) -> np.ndarray:
    return np.asarray(
        [
            old_map[(str(metadata["task"][index]), int(metadata["episode"][index]))][
                view
            ]
            for index in indices
        ],
        dtype=np.int32,
    )


def run_support_controls(
    view: str,
    matrix: np.ndarray,
    score: np.ndarray,
    metadata: dict[str, np.ndarray],
    matched_failure: np.ndarray,
    matched_success: np.ndarray,
    prefix_features: np.ndarray,
    prefix_score: np.ndarray,
    old_map: dict[tuple[str, int], dict[str, int]],
    seed: int,
) -> dict[str, Any]:
    all_success_task = "libero_goal/open_the_middle_drawer_of_the_cabinet"
    old_k = len(np.unique(_old_labels_for_indices(
        view, np.flatnonzero(metadata["failure"]), metadata, old_map
    )))

    c2_global = np.flatnonzero(metadata["task"] != all_success_task)
    c2_labels, c2_info = fixed_partition(
        matrix[c2_global], view, old_k, score[c2_global], seed + 10
    )
    c2_failure_local = np.flatnonzero(metadata["failure"][c2_global])
    c2_failure_global = c2_global[c2_failure_local]
    c2_old = _old_labels_for_indices(view, c2_failure_global, metadata, old_map)
    c2_ari = float(adjusted_rand_score(c2_old, c2_labels[c2_failure_local]))

    c3_global = np.r_[matched_failure, matched_success]
    c3f_labels, c3f_info = fixed_partition(
        matrix[matched_failure], view, old_k, score[matched_failure], seed + 20
    )
    c3_labels, c3_info = fixed_partition(
        matrix[c3_global], view, old_k, score[c3_global], seed + 21
    )
    matched_old = _old_labels_for_indices(
        view, matched_failure, metadata, old_map
    )
    support_ari = float(adjusted_rand_score(matched_old, c3f_labels))
    c3_net_ari = float(
        adjusted_rand_score(c3f_labels, c3_labels[: len(matched_failure)])
    )

    c4f_labels, c4f_info = fixed_partition(
        prefix_features, view, old_k, prefix_score, seed + 30
    )
    tail_ari = float(adjusted_rand_score(c3f_labels, c4f_labels))
    c4_matrix = np.concatenate(
        [np.asarray(prefix_features), np.asarray(matrix[matched_success])], axis=0
    )
    c4_score = np.r_[prefix_score, score[matched_success]]
    c4_labels, c4_info = fixed_partition(
        c4_matrix, view, old_k, c4_score, seed + 40
    )
    net_ari = float(
        adjusted_rand_score(c4f_labels, c4_labels[: len(matched_failure)])
    )
    return {
        "old_failure_k": old_k,
        "matched_pairs": len(matched_failure),
        "c2_shared_task_support": {
            **c2_info,
            "ari_to_old_failure_partition": c2_ari,
            "change_class": _change_class(c2_ari),
        },
        "c3_unique_task_init_matched_balanced": {
            **c3_info,
            "ari_full_failure_only_to_joint_failure_subset": c3_net_ari,
            "net_add_success_change_class": _change_class(c3_net_ari),
        },
        "c3f_matched_full_failure_only": {
            **c3f_info,
            "ari_old_restricted_to_refit_supported_failure": support_ari,
            "support_refit_change_class": _change_class(support_ari),
        },
        "c4f_matched_failure_prefix_only": {
            **c4f_info,
            "ari_matched_full_failure_to_prefix_failure": tail_ari,
            "tail_change_class": _change_class(tail_ari),
        },
        "c4_matched_prefix_joint": {
            **c4_info,
            "ari_prefix_failure_to_joint_failure_subset": net_ari,
            "net_add_success_change_class": _change_class(net_ari),
        },
    }


def sensitivity_partition(
    matrix: np.ndarray,
    view: str,
    clusters: int,
    base_labels: np.ndarray,
    score: np.ndarray,
    metadata: dict[str, np.ndarray],
    permutations: int,
    name: str,
    seed: int,
) -> dict[str, Any]:
    labels, info = fixed_partition(matrix, view, clusters, score, seed)
    outcome = np.where(metadata["failure"], "failure", "success")
    outcome_association = association_summary(
        outcome,
        labels,
        _task_initial_state(metadata),
        permutations,
        seed + 1,
        "within_task_initial_state",
    )
    return {
        "view": view,
        "configuration": name,
        **info,
        "ari_to_primary": float(adjusted_rand_score(base_labels, labels)),
        "outcome_association": outcome_association,
    }


def per_task_outcome_associations(
    metadata: dict[str, np.ndarray],
    labels: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    result = {}
    for task_index, task in enumerate(np.unique(metadata["task"])):
        index = metadata["task"] == task
        outcome = np.where(metadata["failure"][index], "failure", "success")
        if len(np.unique(outcome)) < 2:
            result[str(task)] = {
                "status": "no_outcome_variation",
                "episodes": int(index.sum()),
            }
            continue
        _cluster_values, local_labels = np.unique(labels[index], return_inverse=True)
        association = association_summary(
            outcome,
            local_labels,
            metadata["init_state_id"][index].astype(str),
            permutations,
            seed + task_index,
            "within_initial_state",
        )
        result[str(task)] = {
            "status": "estimated",
            "episodes": int(index.sum()),
            "failures": int(metadata["failure"][index].sum()),
            **association,
        }
    return result


def add_sensitivity_bh_correction(rows: list[dict[str, Any]]) -> None:
    entries = [row["outcome_association"] for row in rows]
    p_values = np.asarray(
        [entry["permutation_p_one_sided"] for entry in entries], dtype=np.float64
    )
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.minimum(adjusted, 1.0)
    for entry, value in zip(entries, q_values):
        entry["fdr_bh_q"] = float(value)


def add_per_task_bh_correction(results: dict[str, dict[str, Any]]) -> None:
    entries = [
        association
        for view in ALL_VIEWS
        for association in results[view]["per_task_outcome_association"].values()
        if association["status"] == "estimated"
    ]
    if not entries:
        return
    p_values = np.asarray(
        [entry["permutation_p_one_sided"] for entry in entries], dtype=np.float64
    )
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.minimum(adjusted, 1.0)
    for entry, value in zip(entries, q_values):
        entry["fdr_bh_q"] = float(value)


def annotate_results(
    results: dict[str, dict[str, Any]], sensitivity: list[dict[str, Any]]
) -> None:
    for view in ALL_VIEWS:
        result = results[view]
        outcome = result["posthoc_association"][
            "outcome_within_task_initial_state"
        ]
        result["outcome_structured"] = bool(
            outcome["nmi_excess_over_null"] >= OUTCOME_EXCESS_NMI_MIN
            and outcome["fdr_bh_q"] < 0.05
        )
        task_nmi = result["posthoc_association"]["task"]["nmi"]
        dominant = max(row["dominant_task_fraction"] for row in result["profiles"])
        result["task_shadowed"] = bool(
            task_nmi >= TASK_SHADOW_NMI
            or dominant >= TASK_SHADOW_CLUSTER_FRACTION
        )
        phase_rows = [row for row in sensitivity if row["view"] == view]
        result["phase_sensitive"] = bool(
            view in PRIMARY_VIEWS
            and any(
                row["ari_to_primary"] < 0.50 or not row["minimum_size_pass"]
                for row in phase_rows
            )
        )
        flags = []
        if result["status"] != "stable_partition":
            flags.append("exploratory_only")
        if result["outcome_structured"]:
            flags.append("outcome_structured")
        if result["task_shadowed"]:
            flags.append("task_shadowed")
        if result["phase_sensitive"]:
            flags.append("phase_sensitive")
        if view == "expert_occupancy":
            flags.append("secondary_expert_id_view")
        result["interpretation_flags"] = flags
        result["interpretation_status"] = "+".join(flags) if flags else "route_blocks_only"


def write_assignments(
    path: pathlib.Path,
    metadata: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    same_k_labels: dict[str, np.ndarray],
    results: dict[str, dict[str, Any]],
) -> None:
    fieldnames = [
        "task",
        "checkpoint_sha256",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "episode_length",
        "outcome",
    ]
    for view in ALL_VIEWS:
        fieldnames.extend(
            [
                f"{view}_cluster",
                f"{view}_same_failure_k_cluster",
                f"{view}_partition_status",
                f"{view}_interpretation_status",
            ]
        )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(metadata["episode"])):
            row = {
                "task": str(metadata["task"][index]),
                "checkpoint_sha256": str(metadata["checkpoint_sha256"][index]),
                "episode": int(metadata["episode"][index]),
                "init_state_id": int(metadata["init_state_id"][index]),
                "flow_noise_seed": int(metadata["flow_noise_seed"][index]),
                "episode_length": int(metadata["episode_length"][index]),
                "outcome": "failure" if metadata["failure"][index] else "success",
            }
            for view in ALL_VIEWS:
                row[f"{view}_cluster"] = int(labels[view][index])
                row[f"{view}_same_failure_k_cluster"] = int(
                    same_k_labels[view][index]
                )
                row[f"{view}_partition_status"] = results[view]["status"]
                row[f"{view}_interpretation_status"] = results[view][
                    "interpretation_status"
                ]
            writer.writerow(row)


def plot_model_selection(
    path: pathlib.Path, results: dict[str, dict[str, Any]]
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, view in zip(axes.ravel(), ALL_VIEWS):
        rows = results[view]["trials"]
        k = [row["clusters"] for row in rows]
        axis.plot(
            k,
            [row["subsample_silhouette_mean"] for row in rows],
            "o-",
            color="C0",
            label="mean sampled silhouette",
        )
        axis.plot(k, [row["ari_median"] for row in rows], "s-", color="C1", label="median ARI")
        axis.plot(k, [row["ari_p10"] for row in rows], "^-", color="C2", label="ARI p10")
        axis.plot(
            k,
            [row["consensus_pac_01_09"] for row in rows],
            "x-",
            color="C3",
            label="probe PAC",
        )
        axis.axhline(0.75, color="C1", linestyle=":", linewidth=1)
        axis.axhline(0.50, color="C2", linestyle=":", linewidth=1)
        axis.axhline(0.20, color="C3", linestyle=":", linewidth=1)
        axis.set_title(view.replace("_", " "))
        axis.set_xlabel("K")
        axis.set_xticks(k)
        axis.set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_embeddings(
    path: pathlib.Path,
    embeddings: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    results: dict[str, dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, view in zip(axes.ravel(), ALL_VIEWS):
        embedding = embeddings[view]
        cmap, norm = cluster_color_scale(results[view]["reported_k"])
        success = ~metadata["failure"]
        failure = metadata["failure"]
        axis.scatter(
            embedding[success, 0],
            embedding[success, 1],
            c=labels[view][success],
            cmap=cmap,
            norm=norm,
            marker="o",
            s=9,
            alpha=0.45,
            edgecolors="none",
            label="success",
        )
        axis.scatter(
            embedding[failure, 0],
            embedding[failure, 1],
            c=labels[view][failure],
            cmap=cmap,
            norm=norm,
            marker="x",
            s=24,
            alpha=0.9,
            linewidths=0.9,
            label="failure",
        )
        qualifier = "stable" if results[view]["selected_k"] is not None else "exploratory"
        axis.set_title(f"{view.replace('_', ' ')} ({qualifier} K={results[view]['reported_k']})")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    axes[0, 0].legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_outcome_rates(
    path: pathlib.Path, results: dict[str, dict[str, Any]], global_rate: float
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, view in zip(axes.ravel(), ALL_VIEWS):
        profiles = results[view]["profiles"]
        clusters = [row["cluster"] for row in profiles]
        rates = [row["failure_rate"] for row in profiles]
        cmap, norm = cluster_color_scale(results[view]["reported_k"])
        colors = cmap(norm(clusters))
        axis.bar(clusters, rates, color=colors)
        axis.axhline(global_rate, color="black", linestyle=":", linewidth=1)
        for row in profiles:
            axis.text(
                row["cluster"],
                row["failure_rate"] + 0.015,
                f"n={row['episodes']}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        axis.set_title(view.replace("_", " "))
        axis.set_xlabel("joint cluster")
        axis.set_ylabel("failure rate")
        axis.set_xticks(clusters)
        axis.set_ylim(0.0, min(1.0, max(rates + [global_rate]) + 0.15))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_failure_comparisons(
    path: pathlib.Path,
    results: dict[str, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
) -> None:
    metrics = [
        ("joint selected", lambda view: results[view]["failure_comparison"]["selected_k_ari"]),
        ("joint same K", lambda view: results[view]["failure_comparison"]["same_k_ari"]),
        ("shared tasks", lambda view: controls[view]["c2_shared_task_support"]["ari_to_old_failure_partition"]),
        (
            "support refit",
            lambda view: controls[view]["c3f_matched_full_failure_only"][
                "ari_old_restricted_to_refit_supported_failure"
            ],
        ),
        (
            "balanced +success",
            lambda view: controls[view]["c3_unique_task_init_matched_balanced"][
                "ari_full_failure_only_to_joint_failure_subset"
            ],
        ),
        (
            "remove tail",
            lambda view: controls[view]["c4f_matched_failure_prefix_only"][
                "ari_matched_full_failure_to_prefix_failure"
            ],
        ),
        (
            "prefix +success",
            lambda view: controls[view]["c4_matched_prefix_joint"][
                "ari_prefix_failure_to_joint_failure_subset"
            ],
        ),
    ]
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(ALL_VIEWS))
    width = 0.11
    for index, (name, accessor) in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        axis.bar(x + offset, [accessor(view) for view in ALL_VIEWS], width, label=name)
    axis.axhline(PRESERVED_ARI, color="0.25", linestyle=":", linewidth=1)
    axis.axhline(REORGANIZED_ARI, color="0.5", linestyle=":", linewidth=1)
    axis.set_xticks(x, [view.replace("_", " ") for view in ALL_VIEWS])
    axis.set_ylabel("ARI")
    axis.set_ylim(-0.05, 1.05)
    axis.legend(fontsize=7, ncol=4)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _short_task(task: str) -> str:
    return task.split("/", 1)[-1]


def render_report(summary: dict[str, Any]) -> str:
    direct = summary["direct_answer"]
    supported = direct["stable_primary_outcome_structured_views"]
    if supported:
        outcome_sentence = (
            "Yes under the frozen criterion: stable outcome-separated blocks occur in "
            + ", ".join(f"`{view}`" for view in supported)
            + "."
        )
    else:
        outcome_sentence = (
            "No primary view simultaneously passed the frozen stability and "
            "conditional outcome-structure gates."
        )
    comparable = direct["confirmatory_same_k_failure_change"]
    comparison_sentence = "; ".join(
        f"`{view}`={details['change_class']} (ARI {fmt(details['ari'])})"
        for view, details in comparable.items()
    )
    if not comparison_sentence:
        comparison_sentence = "no view had stable old and joint partitions"

    lines = [
        "# Joint success/failure routing blocks",
        "",
        f"Run class: `{summary['provenance']['run_class']}`.",
        "",
        outcome_sentence,
        "",
        "For the stronger question of whether adding successes changes the old failure",
        "taxonomy, the confirmatory same-K comparisons are: " + comparison_sentence + ".",
        "Outcome is tested only after route-only clustering, with permutations inside",
        "task x initial-state strata. This is an association, not a causal outcome signal.",
        "",
        "## Coverage",
        "",
        "| task | success | failure | queries | length min/median/max | checkpoint |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in summary["data_audit"]["coverage"]:
        lines.append(
            "| %s | %d | %d | %d | %d/%s/%d | `%s` |"
            % (
                _short_task(row["task"]),
                row["successes"],
                row["failures"],
                row["queries"],
                row["length_min"],
                fmt(row["length_median"], 1),
                row["length_max"],
                row["checkpoint_sha256"][:12],
            )
        )
    lines.extend(
        [
            "",
            "Total: %d episodes (%d successes, %d failures), %d queries. Every task"
            % (
                summary["data_audit"]["episodes"],
                summary["data_audit"]["successes"],
                summary["data_audit"]["failures"],
                summary["data_audit"]["queries"],
            ),
            "passes the complete 16 initial-state x 32 flow-seed design guard.",
            "",
            "## Joint Blocks",
            "",
            "The reported K is selected only among partitions passing all frozen",
            "subsample/PAC/size gates. Otherwise it is explicitly exploratory.",
            "",
            "| view | status | K | sizes | silhouette | ARI med/p10 | PAC | outcome NMI | null NMI | excess | p / BH q | outcome blocks | task NMI | flags |",
            "|---|---|---:|---|---:|---|---:|---:|---:|---:|---|---|---:|---|",
        ]
    )
    for view in ALL_VIEWS:
        result = summary["views"][view]
        trial = next(
            row for row in result["trials"] if row["clusters"] == result["reported_k"]
        )
        outcome = result["posthoc_association"][
            "outcome_within_task_initial_state"
        ]
        lines.append(
            "| %s | %s | %d | %s | %s | %s/%s | %s | %s | %s | %s | %s/%s | %s | %s | %s |"
            % (
                view,
                result["status"],
                result["reported_k"],
                "/".join(map(str, result["reported_sizes"])),
                fmt(trial["subsample_silhouette_mean"]),
                fmt(trial["ari_median"]),
                fmt(trial["ari_p10"]),
                fmt(trial["consensus_pac_01_09"]),
                fmt(outcome["nmi"]),
                fmt(outcome["null_nmi_mean"]),
                fmt(outcome["nmi_excess_over_null"]),
                fmt(outcome["permutation_p_one_sided"], 4),
                fmt(outcome["fdr_bh_q"], 4),
                "yes" if result["outcome_structured"] else "no",
                fmt(result["posthoc_association"]["task"]["nmi"]),
                result["interpretation_status"],
            )
        )

    lines.extend(
        [
            "",
            "## Block Composition",
            "",
            "| view | block | n | success | failure | failure rate | dominant-task fraction | length min/median/max |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for view in ALL_VIEWS:
        for row in summary["views"][view]["profiles"]:
            lines.append(
                "| %s | C%d | %d | %d | %d | %s | %s | %d/%s/%d |"
                % (
                    view,
                    row["cluster"],
                    row["episodes"],
                    row["successes"],
                    row["failures"],
                    fmt(row["failure_rate"]),
                    fmt(row["dominant_task_fraction"]),
                    row["length_min"],
                    fmt(row["length_median"], 1),
                    row["length_max"],
                )
            )

    lines.extend(
        [
            "",
            "## Failure Taxonomy Change",
            "",
            "`same K` is the primary estimand: the all-outcome dendrogram is cut at the",
            "old failure-only K before restricting to the same 307 failures. `selected K`",
            "is auxiliary because it also includes model-selection changes.",
            "",
            "| view | old status/K | joint status/K | selected-K ARI | same-K ARI | Jaccard | split bits | merge bits | same-K class |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for view in ALL_VIEWS:
        result = summary["views"][view]
        comparison = result["failure_comparison"]
        lines.append(
            "| %s | %s/%d | %s/%d | %s | %s | %s | %s | %s | %s |"
            % (
                view,
                comparison["failure_only_status"],
                comparison["failure_only_k"],
                result["status"],
                result["reported_k"],
                fmt(comparison["selected_k_ari"]),
                fmt(comparison["same_k_ari"]),
                fmt(comparison["same_k_matched_jaccard"]),
                fmt(comparison["split_entropy_joint_given_old_bits"]),
                fmt(comparison["merge_entropy_old_given_joint_bits"]),
                comparison["same_k_change_class"],
            )
        )

    lines.extend(["", "### Same-K transition details", ""])
    for view in ALL_VIEWS:
        comparison = summary["views"][view]["failure_comparison"]
        columns = comparison["joint_same_k_column_labels"]
        lines.extend(
            [
                "#### %s" % view,
                "",
                "| old failure block | "
                + " | ".join(f"joint C{value}" for value in columns)
                + " |",
                "|---|" + "---:|" * len(columns),
            ]
        )
        for old, counts in enumerate(comparison["old_by_joint_same_k_table"]):
            lines.append(
                "| old C%d | %s |" % (old, " | ".join(map(str, counts)))
            )
        lines.extend(["", "Within-task same-K ARI:", ""])
        if comparison["within_task"]:
            lines.extend(["| task | failures | ARI |", "|---|---:|---:|"])
            for task, row in comparison["within_task"].items():
                lines.append(
                    "| %s | %d | %s |"
                    % (_short_task(task), row["episodes"], fmt(row["ari_same_k"]))
                )
        else:
            lines.append(
                "Not estimable: no task contains at least two blocks in both partitions."
            )

    lines.extend(
        [
            "",
            "### Support and termination controls",
            "",
            "The matched controls use 157 unique failure-success pairs with identical",
            "task x initial-state support. They are descriptive for this supported subset.",
            "",
            "| view | C2 shared tasks | support/refit | full +success | remove failure tail | prefix +success |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for view in ALL_VIEWS:
        control = summary["support_controls"][view]
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                view,
                fmt(
                    control["c2_shared_task_support"][
                        "ari_to_old_failure_partition"
                    ]
                ),
                fmt(
                    control["c3f_matched_full_failure_only"][
                        "ari_old_restricted_to_refit_supported_failure"
                    ]
                ),
                fmt(
                    control["c3_unique_task_init_matched_balanced"][
                        "ari_full_failure_only_to_joint_failure_subset"
                    ]
                ),
                fmt(
                    control["c4f_matched_failure_prefix_only"][
                        "ari_matched_full_failure_to_prefix_failure"
                    ]
                ),
                fmt(
                    control["c4_matched_prefix_joint"][
                        "ari_prefix_failure_to_joint_failure_subset"
                    ]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Per-task Outcome Association",
            "",
            "| view | task | failures | outcome NMI excess | p | BH q |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for view in ALL_VIEWS:
        for task, row in summary["views"][view][
            "per_task_outcome_association"
        ].items():
            if row["status"] != "estimated":
                continue
            lines.append(
                "| %s | %s | %d | %s | %s | %s |"
                % (
                    view,
                    _short_task(task),
                    row["failures"],
                    fmt(row["nmi_excess_over_null"]),
                    fmt(row["permutation_p_one_sided"], 4),
                    fmt(row["fdr_bh_q"], 4),
                )
            )

    lines.extend(
        [
            "",
            "## Relative-phase Sensitivity",
            "",
            "| view | configuration | K | ARI to primary | sizes | min-size pass | outcome excess | BH q |",
            "|---|---|---:|---:|---|---|---:|---:|",
        ]
    )
    for row in summary["phase_sensitivity"]:
        outcome = row["outcome_association"]
        lines.append(
            "| %s | %s | %d | %s | %s | %s | %s | %s |"
            % (
                row["view"],
                row["configuration"],
                row["fixed_k"],
                fmt(row["ari_to_primary"]),
                "/".join(map(str, row["sizes"])),
                "yes" if row["minimum_size_pass"] else "no",
                fmt(outcome["nmi_excess_over_null"]),
                fmt(outcome["fdr_bh_q"], 4),
            )
        )

    lines.extend(["", "## Cross-view Agreement", ""])
    lines.extend(
        [
            "| view | " + " | ".join(ALL_VIEWS) + " |",
            "|---|" + "---:|" * len(ALL_VIEWS),
        ]
    )
    for left in ALL_VIEWS:
        values = [fmt(summary["cross_view_ari"][left][right]) for right in ALL_VIEWS]
        lines.append("| %s | %s |" % (left, " | ".join(values)))

    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- Relative phase removes absolute chunk index and episode length from the feature vector, but it does not make success completion and failure timeout semantically equivalent.",
            "- Failure horizons are almost perfectly outcome-linked in this corpus. The matched-prefix controls reduce this confound only on 157 supported pairs; they do not identify a causal outcome effect for all failures.",
            "- A task-shadowed partition can still have conditioned outcome association, but it is not a task-independent routing taxonomy.",
            "- `change` results remain descriptive if either the old or joint partition fails the frozen stability gate. `expert_occupancy` is an expert-ID-based secondary view.",
            "- Stability is conditional on episode subsampling and does not establish generalization to unseen tasks, states, or noise draws.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    assert minimum_cluster_size(307) == 16
    assert minimum_cluster_size(2560) == 52
    assert _change_class(0.80) == "preserved"
    assert _change_class(0.50) == "moderately_changed"
    assert _change_class(0.499) == "reorganized"
    labels = np.asarray([0, 0, 1, 1])
    assert _conditional_entropy(labels, labels) == 0.0
    assert math.isclose(
        _conditional_entropy(np.zeros(4, dtype=np.int32), labels), 1.0
    )
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.subsamples < 10 or args.association_permutations < 100:
        raise ValueError("too few stability/permutation draws")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "completion.json").unlink(missing_ok=True)
    method_source = OUT_DIR / "METHOD.md"
    method_target = args.out_dir / "METHOD.md"
    if method_source.resolve() != method_target.resolve():
        shutil.copyfile(method_source, method_target)
    feature_cache = args.feature_cache or args.out_dir / "feature_cache"
    prefix_cache = args.out_dir / "matched_prefix_cache"
    formal_run = bool(
        args.subsamples == SUBSAMPLES
        and args.association_permutations == ASSOCIATION_PERMUTATIONS
        and args.cache_root.resolve() == CACHE_ROOT.resolve()
        and args.seed == 20260828
    )

    with threadpool_limits(limits=4):
        arrays, audit = load_or_extract_features(
            args.cache_root, feature_cache, args.force_features
        )
        metadata = _metadata(arrays)
        diagnostics = {
            name.removeprefix("diagnostic_"): np.asarray(values)
            for name, values in arrays.items()
            if name.startswith("diagnostic_")
        }
        score = diagnostics["mean_soft_speed"]
        if len(metadata["failure"]) != EXPECTED_EPISODES:
            raise ValueError("feature cache episode count drifted")

        results: dict[str, dict[str, Any]] = {}
        labels: dict[str, np.ndarray] = {}
        embeddings: dict[str, np.ndarray] = {}
        for view_index, view in enumerate(ALL_VIEWS):
            print(f"clustering joint {view}", flush=True)
            result, view_labels, embedding = cluster_joint_view(
                arrays[f"feature_primary_{view}"],
                view,
                score,
                args.subsamples,
                args.seed + 10000 * view_index,
            )
            results[view] = result
            labels[view] = view_labels
            embeddings[view] = embedding

        cross_view = {
            left: {
                right: float(adjusted_rand_score(labels[left], labels[right]))
                for right in ALL_VIEWS
            }
            for left in ALL_VIEWS
        }

        print("joint labels frozen; revealing outcome/task metadata", flush=True)
        for view_index, view in enumerate(ALL_VIEWS):
            results[view]["posthoc_association"] = posthoc_associations(
                metadata,
                labels[view],
                args.association_permutations,
                args.seed + 50000 + 1000 * view_index,
            )
            results[view]["per_task_outcome_association"] = (
                per_task_outcome_associations(
                    metadata,
                    labels[view],
                    args.association_permutations,
                    args.seed + 60000 + 1000 * view_index,
                )
            )
            results[view]["profiles"] = outcome_profiles(
                labels[view], metadata, diagnostics
            )
            print(f"posthoc associations {view} complete", flush=True)
        add_bh_correction(results)
        add_per_task_bh_correction(results)

        old_map, old_statuses = load_failure_only_labels()
        same_k_labels: dict[str, np.ndarray] = {}
        for view in ALL_VIEWS:
            comparison, same_k = compare_failure_partitions(
                view,
                labels[view],
                embeddings[view],
                metadata,
                score,
                old_map,
                old_statuses,
            )
            results[view]["failure_comparison"] = comparison
            same_k_labels[view] = same_k

        phase_sensitivity = []
        for view_index, view in enumerate(PRIMARY_VIEWS):
            for config_index, config in enumerate(SENSITIVITY_PHASES):
                print(f"phase sensitivity {view}/{config.name}", flush=True)
                phase_sensitivity.append(
                    sensitivity_partition(
                        arrays[f"feature_{config.name}_{view}"],
                        view,
                        results[view]["reported_k"],
                        labels[view],
                        score,
                        metadata,
                        args.association_permutations,
                        config.name,
                        args.seed + 70000 + 1000 * view_index + config_index,
                    )
                )
        add_sensitivity_bh_correction(phase_sensitivity)
        annotate_results(results, phase_sensitivity)

        matched_failure, matched_success = maximum_unique_matches(metadata)
        prefix_features, prefix_score = extract_matched_failure_prefixes(
            args.cache_root,
            prefix_cache,
            metadata,
            matched_failure,
            matched_success,
            args.force_features,
        )
        controls = {}
        for view_index, view in enumerate(ALL_VIEWS):
            print(f"support controls {view}", flush=True)
            controls[view] = run_support_controls(
                view,
                arrays[f"feature_primary_{view}"],
                score,
                metadata,
                matched_failure,
                matched_success,
                prefix_features[view],
                prefix_score,
                old_map,
                args.seed + 80000 + 1000 * view_index,
            )

    feature_manifest_source = feature_cache / "manifest.json"
    prefix_manifest_source = prefix_cache / "manifest.json"
    feature_manifest_snapshot = args.out_dir / "feature_cache_manifest.json"
    prefix_manifest_snapshot = args.out_dir / "matched_prefix_cache_manifest.json"
    shutil.copyfile(feature_manifest_source, feature_manifest_snapshot)
    shutil.copyfile(prefix_manifest_source, prefix_manifest_snapshot)
    feature_manifest = json.loads(feature_manifest_snapshot.read_text())

    stable_primary_outcome = [
        view
        for view in PRIMARY_VIEWS
        if results[view]["status"] == "stable_partition"
        and results[view]["outcome_structured"]
    ]
    confirmatory_change = {
        view: {
            "ari": results[view]["failure_comparison"]["same_k_ari"],
            "change_class": results[view]["failure_comparison"][
                "same_k_change_class"
            ],
        }
        for view in ALL_VIEWS
        if results[view]["status"] == "stable_partition"
        and old_statuses[view] == "stable_partition"
    }
    direct_answer = {
        "stable_primary_outcome_structured_views": stable_primary_outcome,
        "stable_all_view_outcome_structured_views": [
            view
            for view in ALL_VIEWS
            if results[view]["status"] == "stable_partition"
            and results[view]["outcome_structured"]
        ],
        "confirmatory_same_k_failure_change": confirmatory_change,
        "same_k_failure_change_all_views": {
            view: {
                "ari": results[view]["failure_comparison"]["same_k_ari"],
                "change_class": results[view]["failure_comparison"][
                    "same_k_change_class"
                ],
                "joint_status": results[view]["status"],
                "old_status": old_statuses[view],
            }
            for view in ALL_VIEWS
        },
    }
    pair_audit = {
        "pairs": int(len(matched_failure)),
        "unique_failures": int(len(np.unique(matched_failure))),
        "unique_successes": int(len(np.unique(matched_success))),
        "task_initial_state_match": bool(
            np.array_equal(
                _task_initial_state(metadata)[matched_failure],
                _task_initial_state(metadata)[matched_success],
            )
        ),
        "success_no_longer_than_failure": bool(
            np.all(
                metadata["episode_length"][matched_success]
                <= metadata["episode_length"][matched_failure]
            )
        ),
        "failure_indices": matched_failure.tolist(),
        "success_indices": matched_success.tolist(),
    }
    summary = {
        "schema": "himoe.all_outcome_routing_clusters.v1",
        "method": str(method_target.resolve()),
        "provenance": {
            "run_class": "formal" if formal_run else "nonformal",
            "cache_root": str(args.cache_root.resolve()),
            "feature_cache": str(feature_cache.resolve()),
            "seed": args.seed,
            "subsamples": args.subsamples,
            "association_permutations": args.association_permutations,
            "numeric_thread_limit": 4,
            "analysis_script_sha256": sha256_file(pathlib.Path(__file__)),
            "method_sha256": sha256_file(method_target),
            "failure_only_assignments_sha256": sha256_file(FAILURE_ASSIGNMENTS),
            "failure_cluster_core_sha256": sha256_file(
                HERE / "analyze_failure_routing_clusters.py"
            ),
            "router_feature_sha256": sha256_file(
                HERE / "analyze_single_chunk_early_signal.py"
            ),
            "feature_cache_manifest_sha256": sha256_file(
                feature_manifest_snapshot
            ),
            "matched_prefix_cache_manifest_sha256": sha256_file(
                prefix_manifest_snapshot
            ),
            "input_fingerprints": feature_manifest["input_fingerprints"],
        },
        "scope": {
            "outcome": "all successes and failures in five complete right-16x32 LIBERO runs",
            "episode_is_statistical_unit": True,
            "primary_phase": {
                "coordinate": "query_index/(episode_queries-1)",
                "start": PRIMARY_PHASE.start,
                "end": PRIMARY_PHASE.end,
                "anchors": PRIMARY_PHASE.bins,
            },
            "absolute_chunk_used_as_feature": False,
            "episode_length_used_as_feature": False,
            "task_used_as_feature": False,
            "outcome_used_for_clustering_or_k_selection": False,
        },
        "stability_protocol": {
            "candidate_k": list(CANDIDATE_K),
            "minimum_cluster_size_rule": "max(16, ceil(0.02*N))",
            "joint_minimum_cluster_size": minimum_cluster_size(EXPECTED_EPISODES),
            "subsamples": args.subsamples,
            "subsample_fraction": 0.80,
            "preprocessing_refit_within_each_subsample": True,
            "consensus_probe_episodes": CONSENSUS_PROBE,
            "selection_rule": "highest mean sampled silhouette among stable K",
            "stable_requirements": {
                "ari_median_min": 0.75,
                "ari_p10_min": 0.50,
                "pac_max": 0.20,
            },
        },
        "interpretation_gates": {
            "outcome_excess_nmi_min": OUTCOME_EXCESS_NMI_MIN,
            "outcome_fdr_bh_q_max": 0.05,
            "failure_partition_preserved_ari_min": PRESERVED_ARI,
            "failure_partition_reorganized_ari_below": REORGANIZED_ARI,
            "task_nmi_shadow_min": TASK_SHADOW_NMI,
            "single_cluster_task_fraction_shadow_min": TASK_SHADOW_CLUSTER_FRACTION,
        },
        "data_audit": audit,
        "matched_pair_audit": pair_audit,
        "direct_answer": direct_answer,
        "views": results,
        "cross_view_ari": cross_view,
        "phase_sensitivity": phase_sensitivity,
        "support_controls": controls,
    }

    write_assignments(
        args.out_dir / "assignments.csv", metadata, labels, same_k_labels, results
    )
    np.savez_compressed(
        args.out_dir / "embeddings_and_labels.npz",
        **{
            f"embedding_{view}": embeddings[view].astype(np.float32)
            for view in ALL_VIEWS
        },
        **{f"label_{view}": labels[view].astype(np.int16) for view in ALL_VIEWS},
        **{
            f"same_failure_k_label_{view}": same_k_labels[view].astype(np.int16)
            for view in ALL_VIEWS
        },
        failure=metadata["failure"].astype(bool),
        episode_length=metadata["episode_length"].astype(np.int16),
        soft_speed_curve=diagnostics["soft_speed_curve"].astype(np.float32),
        hard_speed_curve=diagnostics["hard_speed_curve"].astype(np.float32),
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    plot_model_selection(args.out_dir / "model_selection.png", results)
    plot_embeddings(
        args.out_dir / "embeddings.png", embeddings, labels, metadata, results
    )
    plot_outcome_rates(
        args.out_dir / "outcome_rates.png",
        results,
        float(metadata["failure"].mean()),
    )
    plot_failure_comparisons(
        args.out_dir / "failure_partition_comparisons.png", results, controls
    )
    output_names = [
        "METHOD.md",
        "feature_cache_manifest.json",
        "matched_prefix_cache_manifest.json",
        "assignments.csv",
        "embeddings_and_labels.npz",
        "summary.json",
        "report.md",
        "model_selection.png",
        "embeddings.png",
        "outcome_rates.png",
        "failure_partition_comparisons.png",
    ]
    completion = {
        "schema": "himoe.all_outcome_routing_clusters.completion.v1",
        "run_class": "formal" if formal_run else "nonformal",
        "expected_outputs": output_names,
        "output_sha256": {
            name: sha256_file(args.out_dir / name) for name in output_names
        },
    }
    completion_temporary = args.out_dir / "completion.json.tmp"
    completion_temporary.write_text(json.dumps(completion, indent=2, allow_nan=False))
    completion_temporary.replace(args.out_dir / "completion.json")
    print(f"wrote {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
