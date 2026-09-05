#!/usr/bin/env python3
"""Build checkpoint-local PCA-10 fingerprints for every action-token route.

One raw token fingerprint is the dense, normalized Top-4 combine-weight route
over 8 HB layers, 10 denoising forwards, and 32 experts: 8 * 10 * 32 = 2560
features.  All T1..T10 positions share one PCA basis within a checkpoint so
their coordinates remain comparable.  Different checkpoints are never pooled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
DEFAULT_HUB = HERE.parent
DEFAULT_OUTPUT = HERE / "results"

HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
N_LAYERS = len(HB_LAYERS)
N_DENOISE = 10
N_SUFFIX = 11
N_ACTION_TOKENS = 10
N_EXPERTS = 32
TOP_K = 4
N_COMPONENTS = 10
N_ROUTE_SITES = N_LAYERS * N_DENOISE
RAW_FEATURES = N_ROUTE_SITES * N_EXPERTS
LAGS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class RunInfo:
    key: str
    suite: str
    task: str
    prompt: str
    path: Path
    relative_path: str
    checkpoint_sha256: str
    checkpoint_group: str
    rows: int
    episodes: int


@dataclass
class TransformedRun:
    info: RunInfo
    episode_id: np.ndarray
    episode_step: np.ndarray
    control_step: np.ndarray
    scores: np.ndarray
    whitened: np.ndarray
    pair_starts: np.ndarray
    raw_distance: np.ndarray
    raw_cosine: np.ndarray
    top4_jaccard: np.ndarray
    top4_site_exact: np.ndarray
    whole_route_exact: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--run-id", default="right-16x32")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fit-chunks-per-episode", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--stability-seed", type=int, default=20260904)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(hub: Path, run_id: str) -> list[RunInfo]:
    root = hub / "cache" / "HiMoE-VLA"
    candidates = sorted(root.glob(f"libero_*/*/{run_id}"))
    provisional: list[dict[str, Any]] = []
    for path in candidates:
        required = (
            path / "meta.json",
            path / "client" / "summaries.json",
            path / "client" / "server_metadata.json",
            path / "server" / "routes.zarr",
        )
        if not all(item.exists() for item in required):
            continue
        meta = read_json(path / "meta.json")
        if not bool(meta.get("sampling", {}).get("complete", False)):
            continue
        server = read_json(path / "client" / "server_metadata.json")
        summaries = read_json(path / "client" / "summaries.json")
        store = zarr.open_group(str(path / "server" / "routes.zarr"), mode="r")
        shape = tuple(store["hb_expert_ids"].shape)
        expected_tail = (N_LAYERS, N_DENOISE, N_SUFFIX, TOP_K)
        if shape[1:] != expected_tail:
            raise ValueError(f"{path}: unexpected hb_expert_ids shape {shape}")
        if tuple(store["hb_selected_prob"].shape) != shape:
            raise ValueError(f"{path}: ids/probability shapes differ")
        suite = str(meta["suite"])
        task = str(meta["task_name"])
        provisional.append(
            {
                "key": f"{suite}/{task}",
                "suite": suite,
                "task": task,
                "prompt": str(meta["prompt"]),
                "path": path,
                "relative_path": str(path.relative_to(hub)),
                "checkpoint_sha256": str(server["checkpoint_sha256"]),
                "rows": shape[0],
                "episodes": len(summaries),
            }
        )

    if not provisional:
        raise ValueError(f"no complete {run_id!r} route runs under {root}")

    suites_by_sha: dict[str, set[str]] = {}
    shas_by_suite: dict[str, set[str]] = {}
    for item in provisional:
        suites_by_sha.setdefault(item["checkpoint_sha256"], set()).add(item["suite"])
        shas_by_suite.setdefault(item["suite"], set()).add(item["checkpoint_sha256"])

    result = []
    for item in provisional:
        sha = item["checkpoint_sha256"]
        suite = item["suite"]
        if len(suites_by_sha[sha]) == 1 and len(shas_by_suite[suite]) == 1:
            group = suite
        else:
            group = f"{suite}-{sha[:8]}"
        result.append(RunInfo(**item, checkpoint_group=group))
    return result


def load_episode_axes(run: RunInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    store = zarr.open_group(str(run.path / "server" / "routes.zarr"), mode="r")
    episode_id = np.asarray(store["episode_id"][:], dtype=np.int32)
    control_step = np.asarray(store["control_step"][:], dtype=np.int32)
    if len(episode_id) != run.rows or len(control_step) != run.rows:
        raise ValueError(f"{run.key}: route index arrays have the wrong length")
    if len(np.unique(control_step)) != run.rows:
        raise ValueError(f"{run.key}: control_step is not unique")

    boundaries = np.r_[0, np.flatnonzero(episode_id[1:] != episode_id[:-1]) + 1, run.rows]
    segment_ids = episode_id[boundaries[:-1]]
    if len(segment_ids) != len(np.unique(segment_ids)):
        raise ValueError(f"{run.key}: an episode appears in multiple segments")
    if len(segment_ids) != run.episodes:
        raise ValueError(
            f"{run.key}: {len(segment_ids)} route episodes != {run.episodes} summaries"
        )

    episode_step = np.empty(run.rows, dtype=np.int16)
    observed_lengths: dict[int, int] = {}
    for start, stop, episode in zip(boundaries[:-1], boundaries[1:], segment_ids):
        episode_step[start:stop] = np.arange(stop - start, dtype=np.int16)
        observed_lengths[int(episode)] = int(stop - start)

    summaries = sorted(
        read_json(run.path / "client" / "summaries.json"),
        key=lambda row: int(row["episode_index"]),
    )
    for row in summaries:
        episode = int(row["episode_index"])
        expected = int(row["inference_calls"])
        if observed_lengths.get(episode) != expected:
            raise ValueError(
                f"{run.key}: episode {episode} has {observed_lengths.get(episode)} "
                f"route rows, expected {expected}"
            )
    return episode_id, episode_step, control_step


def take_rows(array: zarr.Array, rows: slice | np.ndarray) -> np.ndarray:
    if isinstance(rows, slice):
        return np.asarray(array[rows])
    return np.asarray(array.oindex[rows])


def encode_route_arrays(
    ids: np.ndarray, selected_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    expected = (N_LAYERS, N_DENOISE, N_SUFFIX, TOP_K)
    if ids.ndim != 5 or tuple(ids.shape[1:]) != expected:
        raise ValueError(f"unexpected route batch shape {ids.shape}")
    if selected_prob.shape != ids.shape:
        raise ValueError("ids and selected probabilities must have equal shapes")

    ids = np.asarray(ids[:, :, :, 1:, :], dtype=np.int64)
    weights = np.maximum(
        np.asarray(selected_prob[:, :, :, 1:, :], dtype=np.float32), 0.0
    )
    denominator = weights.sum(axis=-1, keepdims=True)
    if np.any(denominator <= 0.0):
        raise ValueError("selected Top-4 probabilities contain a zero-mass site")
    weights /= denominator

    # [chunk, layer, denoise, token, topk] -> [chunk, token, layer, denoise, topk]
    ids = np.transpose(ids, (0, 3, 1, 2, 4))
    weights = np.transpose(weights, (0, 3, 1, 2, 4))
    if np.any((ids < 0) | (ids >= N_EXPERTS)):
        raise ValueError("expert id outside [0, 31]")
    if np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
        raise ValueError("duplicate expert id inside a Top-4 site")

    dense = np.zeros((*ids.shape[:-1], N_EXPERTS), dtype=np.float32)
    np.put_along_axis(dense, ids, weights, axis=-1)
    if not np.allclose(dense.sum(axis=-1), 1.0, atol=2e-6):
        raise ValueError("normalized combine weights do not sum to one")
    return dense.reshape(len(ids), N_ACTION_TOKENS, RAW_FEATURES), ids


def read_route_features(
    store: zarr.Group, rows: slice | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ids = take_rows(store["hb_expert_ids"], rows)
    selected = take_rows(store["hb_selected_prob"], rows)
    return encode_route_arrays(ids, selected)


def stable_run_seed(seed: int, run: RunInfo) -> int:
    digest = hashlib.sha256(run.relative_path.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "little")


def sample_fit_rows(run: RunInfo, per_episode: int, seed: int) -> np.ndarray:
    if per_episode <= 0:
        raise ValueError("fit-chunks-per-episode must be positive")
    episode_id, _, _ = load_episode_axes(run)
    rng = np.random.default_rng(stable_run_seed(seed, run))
    rows: list[int] = []
    for episode in np.unique(episode_id):
        candidates = np.flatnonzero(episode_id == episode)
        take = min(per_episode, len(candidates))
        rows.extend(rng.choice(candidates, size=take, replace=False).tolist())
    return np.asarray(sorted(rows), dtype=np.int64)


def fit_checkpoint_pca(
    runs: list[RunInfo], per_episode: int, seed: int
) -> tuple[PCA, dict[str, int]]:
    samples = []
    sampled_controls: dict[str, int] = {}
    for run in runs:
        rows = sample_fit_rows(run, per_episode, seed)
        store = zarr.open_group(str(run.path / "server" / "routes.zarr"), mode="r")
        features, _ = read_route_features(store, rows)
        samples.append(features.reshape(-1, RAW_FEATURES))
        sampled_controls[run.key] = len(rows)
        print(
            f"  PCA sample {run.key}: {len(rows)} chunks / "
            f"{len(rows) * N_ACTION_TOKENS} token vectors",
            flush=True,
        )
    matrix = np.concatenate(samples, axis=0)
    pca = PCA(
        n_components=N_COMPONENTS,
        svd_solver="randomized",
        iterated_power=4,
        n_oversamples=20,
        random_state=seed,
    )
    pca.fit(matrix)
    del matrix, samples
    return pca, sampled_controls


def validate_pca_subspace(
    reference: PCA,
    runs: list[RunInfo],
    per_episode: int,
    seed: int,
) -> dict[str, Any]:
    alternate, sampled_controls = fit_checkpoint_pca(runs, per_episode, seed)
    canonical = np.linalg.svd(
        reference.components_ @ alternate.components_.T, compute_uv=False
    )
    return {
        "alternate_seed": seed,
        "alternate_fit_sampled_controls": sampled_controls,
        "alternate_cumulative_explained_variance": float(
            alternate.explained_variance_ratio_.sum()
        ),
        "canonical_correlations": canonical,
        "mean_squared_subspace_overlap": float(np.mean(np.square(canonical))),
        "minimum_canonical_correlation": float(canonical.min()),
    }


def pair_route_metrics(
    a: np.ndarray, b: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if a.shape != b.shape or ids_a.shape != ids_b.shape:
        raise ValueError("pair tensors must have equal shapes")
    distance = np.linalg.norm(a - b, axis=-1) / math.sqrt(N_ROUTE_SITES)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.sum(a * b, axis=-1) / np.maximum(norms, 1e-12)

    equality = ids_a[..., :, None] == ids_b[..., None, :]
    intersection = equality.any(axis=-1).sum(axis=-1)
    jaccard_site = intersection / np.maximum(2 * TOP_K - intersection, 1)
    site_exact = intersection == TOP_K
    jaccard = jaccard_site.mean(axis=(2, 3))
    site_exact_rate = site_exact.mean(axis=(2, 3))
    whole_exact = site_exact.all(axis=(2, 3))
    return distance, cosine, jaccard, site_exact_rate, whole_exact


def transform_run(run: RunInfo, pca: PCA, batch_size: int) -> TransformedRun:
    episode_id, episode_step, control_step = load_episode_axes(run)
    pair_starts = np.flatnonzero(episode_id[1:] == episode_id[:-1]).astype(np.int64)
    pair_lookup = np.full(run.rows, -1, dtype=np.int64)
    pair_lookup[pair_starts] = np.arange(len(pair_starts), dtype=np.int64)

    scores = np.empty((run.rows, N_ACTION_TOKENS, N_COMPONENTS), dtype=np.float32)
    whitened = np.empty_like(scores)
    raw_distance = np.full((len(pair_starts), N_ACTION_TOKENS), np.nan, np.float32)
    raw_cosine = np.full_like(raw_distance, np.nan)
    top4_jaccard = np.full_like(raw_distance, np.nan)
    top4_site_exact = np.full_like(raw_distance, np.nan)
    whole_route_exact = np.zeros(
        (len(pair_starts), N_ACTION_TOKENS), dtype=np.bool_
    )

    scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-12)).astype(np.float32)
    store = zarr.open_group(str(run.path / "server" / "routes.zarr"), mode="r")
    previous_features: np.ndarray | None = None
    previous_ids: np.ndarray | None = None
    previous_episode: int | None = None

    for start in range(0, run.rows, batch_size):
        stop = min(start + batch_size, run.rows)
        features, ids = read_route_features(store, slice(start, stop))
        flat_scores = pca.transform(features.reshape(-1, RAW_FEATURES))
        block_scores = flat_scores.reshape(-1, N_ACTION_TOKENS, N_COMPONENTS)
        scores[start:stop] = block_scores
        whitened[start:stop] = block_scores / scale

        if previous_features is None:
            joined_features = features
            joined_ids = ids
            joined_episode = episode_id[start:stop]
            base = start
        else:
            joined_features = np.concatenate((previous_features[None], features), axis=0)
            joined_ids = np.concatenate((previous_ids[None], ids), axis=0)
            joined_episode = np.r_[previous_episode, episode_id[start:stop]]
            base = start - 1

        valid = np.flatnonzero(joined_episode[1:] == joined_episode[:-1])
        if len(valid):
            destination = pair_lookup[base + valid]
            if np.any(destination < 0):
                raise ValueError(f"{run.key}: failed to map an adjacent pair")
            metrics = pair_route_metrics(
                joined_features[valid],
                joined_features[valid + 1],
                joined_ids[valid],
                joined_ids[valid + 1],
            )
            raw_distance[destination] = metrics[0]
            raw_cosine[destination] = metrics[1]
            top4_jaccard[destination] = metrics[2]
            top4_site_exact[destination] = metrics[3]
            whole_route_exact[destination] = metrics[4]

        previous_features = features[-1].copy()
        previous_ids = ids[-1].copy()
        previous_episode = int(episode_id[stop - 1])
        print(f"  transform {run.key}: {stop}/{run.rows}", flush=True)

    for name, values in (
        ("raw_distance", raw_distance),
        ("raw_cosine", raw_cosine),
        ("top4_jaccard", top4_jaccard),
        ("top4_site_exact", top4_site_exact),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{run.key}: incomplete {name}")

    return TransformedRun(
        info=run,
        episode_id=episode_id,
        episode_step=episode_step,
        control_step=control_step,
        scores=scores,
        whitened=whitened,
        pair_starts=pair_starts,
        raw_distance=raw_distance,
        raw_cosine=raw_cosine,
        top4_jaccard=top4_jaccard,
        top4_site_exact=top4_site_exact,
        whole_route_exact=whole_route_exact,
    )


def component_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    numerator = np.mean(a_centered * b_centered, axis=0)
    denominator = np.std(a, axis=0) * np.std(b, axis=0)
    return numerator / np.maximum(denominator, 1e-12)


def eta_squared_by_position(scores: np.ndarray) -> tuple[np.ndarray, float]:
    # scores: [chunk, token, component]
    overall = scores.reshape(-1, scores.shape[-1]).mean(axis=0)
    total = np.square(scores - overall).sum(axis=(0, 1))
    between = np.zeros(scores.shape[-1], dtype=np.float64)
    for token in range(N_ACTION_TOKENS):
        center = scores[:, token].mean(axis=0)
        between += len(scores) * np.square(center - overall)
    component_eta = between / np.maximum(total, 1e-30)
    return component_eta, float(between.sum() / np.maximum(total.sum(), 1e-30))


def analyze_transformed(runs: list[TransformedRun]) -> dict[str, Any]:
    pairs = sum(len(run.pair_starts) for run in runs)
    if pairs == 0:
        raise ValueError("no within-episode adjacent chunk pairs")

    matrix_sum = np.zeros((N_ACTION_TOKENS, N_ACTION_TOKENS), np.float64)
    aligned_values = []
    nearest_correct = 0
    aligned_closer = 0
    source_tokens = 0
    component_a = []
    component_b = []

    for run in runs:
        starts = run.pair_starts
        aligned_run = np.empty((len(starts), N_ACTION_TOKENS), np.float32)
        for offset in range(0, len(starts), 512):
            selected = starts[offset : offset + 512]
            a = run.whitened[selected]
            b = run.whitened[selected + 1]
            distance = np.sqrt(
                np.mean(np.square(a[:, :, None, :] - b[:, None, :, :]), axis=-1)
            )
            matrix_sum += distance.sum(axis=0)
            diagonal = np.diagonal(distance, axis1=1, axis2=2)
            aligned_run[offset : offset + len(selected)] = diagonal
            nearest = np.argmin(distance, axis=2)
            expected = np.arange(N_ACTION_TOKENS)[None, :]
            nearest_correct += int(np.sum(nearest == expected))
            offdiag_mean = (
                distance.sum(axis=2) - diagonal
            ) / (N_ACTION_TOKENS - 1)
            aligned_closer += int(np.sum(diagonal < offdiag_mean))
            source_tokens += distance.shape[0] * N_ACTION_TOKENS
        aligned_values.append(aligned_run)
        component_a.append(run.whitened[starts])
        component_b.append(run.whitened[starts + 1])

    matrix = matrix_sum / pairs
    aligned = np.concatenate(aligned_values, axis=0)
    a_all = np.concatenate(component_a, axis=0)
    b_all = np.concatenate(component_b, axis=0)
    diagonal_mean = float(np.trace(matrix) / N_ACTION_TOKENS)
    offdiag_mean = float(
        (matrix.sum() - np.trace(matrix))
        / (N_ACTION_TOKENS * (N_ACTION_TOKENS - 1))
    )

    lag_summary: dict[str, Any] = {}
    for lag in LAGS:
        lag_values = []
        for run in runs:
            valid = np.flatnonzero(run.episode_id[lag:] == run.episode_id[:-lag])
            delta = run.whitened[valid + lag] - run.whitened[valid]
            lag_values.append(np.sqrt(np.mean(np.square(delta), axis=-1)))
        values = np.concatenate(lag_values, axis=0)
        lag_summary[str(lag)] = {
            "pairs": len(values),
            "mean": float(values.mean()),
            "by_token": values.mean(axis=0),
        }

    raw_distance = np.concatenate([run.raw_distance for run in runs], axis=0)
    raw_cosine = np.concatenate([run.raw_cosine for run in runs], axis=0)
    top4_jaccard = np.concatenate([run.top4_jaccard for run in runs], axis=0)
    top4_site_exact = np.concatenate(
        [run.top4_site_exact for run in runs], axis=0
    )
    whole_route_exact = np.concatenate(
        [run.whole_route_exact for run in runs], axis=0
    )
    projected_distance = np.concatenate(
        [
            np.linalg.norm(
                run.scores[run.pair_starts + 1] - run.scores[run.pair_starts],
                axis=-1,
            )
            / math.sqrt(N_ROUTE_SITES)
            for run in runs
        ],
        axis=0,
    )
    correlation = float(
        np.corrcoef(raw_distance.reshape(-1), projected_distance.reshape(-1))[0, 1]
    )
    projection_energy = float(
        np.square(projected_distance).sum()
        / np.maximum(np.square(raw_distance).sum(), 1e-30)
    )

    all_scores = np.concatenate([run.scores for run in runs], axis=0)
    position_eta, total_position_eta = eta_squared_by_position(all_scores)
    component_delta = b_all - a_all
    component_corr = component_correlation(a_all, b_all)

    return {
        "chunks": sum(run.info.rows for run in runs),
        "episodes": sum(run.info.episodes for run in runs),
        "adjacent_pairs": pairs,
        "whitened_distance": {
            "matrix_from_token_to_next_chunk_token": matrix,
            "aligned_mean": diagonal_mean,
            "aligned_by_token": aligned.mean(axis=0),
            "off_diagonal_mean": offdiag_mean,
            "aligned_to_offdiag_ratio": diagonal_mean / offdiag_mean,
            "same_position_nearest_accuracy": nearest_correct / source_tokens,
            "aligned_closer_than_mean_offdiag_fraction": aligned_closer
            / source_tokens,
            "lag": lag_summary,
        },
        "component_dynamics": {
            "delta_rms_by_token_component": np.sqrt(
                np.mean(np.square(component_delta), axis=0)
            ),
            "adjacent_correlation_by_token_component": component_corr,
        },
        "raw_route": {
            "distance_by_token": raw_distance.mean(axis=0),
            "cosine_by_token": raw_cosine.mean(axis=0),
            "top4_jaccard_by_token": top4_jaccard.mean(axis=0),
            "top4_site_exact_rate_by_token": top4_site_exact.mean(axis=0),
            "whole_80_site_top4_exact_rate_by_token": whole_route_exact.mean(axis=0),
            "whole_80_site_top4_exact_rate": float(whole_route_exact.mean()),
            "whole_80_site_top4_exact_matches": int(whole_route_exact.sum()),
            "whole_80_site_top4_comparisons": int(whole_route_exact.size),
        },
        "pca_fidelity_on_adjacent_changes": {
            "raw_vs_projected_distance_pearson": correlation,
            "projected_squared_distance_fraction": projection_energy,
        },
        "token_position_effect": {
            "eta_squared_by_component": position_eta,
            "eta_squared_across_10d": total_position_eta,
        },
    }


def summarize_group(
    name: str,
    pca: PCA,
    sampled_controls: dict[str, int],
    runs: list[TransformedRun],
    stability: dict[str, Any],
) -> dict[str, Any]:
    analysis = analyze_transformed(runs)
    analysis.update(
        {
            "checkpoint_group": name,
            "checkpoint_sha256": runs[0].info.checkpoint_sha256,
            "runs": [run.info.key for run in runs],
            "fit_sampled_controls": sampled_controls,
            "fit_token_vectors": sum(sampled_controls.values())
            * N_ACTION_TOKENS,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
            "pca_subspace_stability": stability,
        }
    )
    return analysis


def macro_summary(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(groups.values())

    def mean_at(path: tuple[str, ...]) -> np.ndarray:
        selected = []
        for item in values:
            current: Any = item
            for key in path:
                current = current[key]
            selected.append(np.asarray(current, dtype=np.float64))
        return np.mean(selected, axis=0)

    return {
        "aggregation": "equal weight per checkpoint group",
        "checkpoint_groups": len(values),
        "whitened_distance": {
            "matrix_from_token_to_next_chunk_token": mean_at(
                (
                    "whitened_distance",
                    "matrix_from_token_to_next_chunk_token",
                )
            ),
            "aligned_mean": float(
                mean_at(("whitened_distance", "aligned_mean"))
            ),
            "aligned_by_token": mean_at(
                ("whitened_distance", "aligned_by_token")
            ),
            "off_diagonal_mean": float(
                mean_at(("whitened_distance", "off_diagonal_mean"))
            ),
            "aligned_to_offdiag_ratio": float(
                mean_at(("whitened_distance", "aligned_to_offdiag_ratio"))
            ),
            "same_position_nearest_accuracy": float(
                mean_at(
                    ("whitened_distance", "same_position_nearest_accuracy")
                )
            ),
            "aligned_closer_than_mean_offdiag_fraction": float(
                mean_at(
                    (
                        "whitened_distance",
                        "aligned_closer_than_mean_offdiag_fraction",
                    )
                )
            ),
            "lag": {
                str(lag): {
                    "mean": float(
                        mean_at(("whitened_distance", "lag", str(lag), "mean"))
                    ),
                    "by_token": mean_at(
                        ("whitened_distance", "lag", str(lag), "by_token")
                    ),
                }
                for lag in LAGS
            },
        },
        "raw_route": {
            key: mean_at(("raw_route", key))
            for key in (
                "distance_by_token",
                "cosine_by_token",
                "top4_jaccard_by_token",
                "top4_site_exact_rate_by_token",
                "whole_80_site_top4_exact_rate_by_token",
            )
        }
        | {
            "whole_80_site_top4_exact_rate": float(
                mean_at(("raw_route", "whole_80_site_top4_exact_rate"))
            ),
            "whole_80_site_top4_exact_matches": int(
                sum(
                    item["raw_route"]["whole_80_site_top4_exact_matches"]
                    for item in values
                )
            ),
            "whole_80_site_top4_comparisons": int(
                sum(
                    item["raw_route"]["whole_80_site_top4_comparisons"]
                    for item in values
                )
            ),
        },
        "pca_fidelity_on_adjacent_changes": {
            key: float(mean_at(("pca_fidelity_on_adjacent_changes", key)))
            for key in (
                "raw_vs_projected_distance_pearson",
                "projected_squared_distance_fraction",
            )
        },
        "token_position_effect": {
            "eta_squared_across_10d": float(
                mean_at(("token_position_effect", "eta_squared_across_10d"))
            )
        },
    }


def save_models(output: Path, models: dict[str, PCA]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for group, pca in models.items():
        prefix = group.replace("-", "_")
        arrays[f"{prefix}_mean"] = pca.mean_.astype(np.float32)
        arrays[f"{prefix}_components"] = pca.components_.astype(np.float32)
        arrays[f"{prefix}_explained_variance"] = pca.explained_variance_.astype(
            np.float32
        )
        arrays[f"{prefix}_explained_variance_ratio"] = (
            pca.explained_variance_ratio_.astype(np.float32)
        )
    np.savez_compressed(output / "pca_models.npz", **arrays)


def save_coordinates(
    output: Path,
    ordered_runs: list[RunInfo],
    transformed: dict[str, TransformedRun],
) -> list[dict[str, Any]]:
    run_names = np.asarray([run.key for run in ordered_runs])
    group_names = np.asarray(
        sorted({run.checkpoint_group for run in ordered_runs})
    )
    group_lookup = {name: index for index, name in enumerate(group_names)}

    scores = []
    whitened = []
    run_index = []
    checkpoint_group_index = []
    episode_id = []
    episode_step = []
    control_step = []
    run_slices = []
    offset = 0
    for index, info in enumerate(ordered_runs):
        run = transformed[info.key]
        stop = offset + info.rows
        scores.append(run.scores)
        whitened.append(run.whitened)
        run_index.append(np.full(info.rows, index, np.int16))
        checkpoint_group_index.append(
            np.full(info.rows, group_lookup[info.checkpoint_group], np.int8)
        )
        episode_id.append(run.episode_id)
        episode_step.append(run.episode_step)
        control_step.append(run.control_step)
        run_slices.append(
            {
                "run_index": index,
                "run": info.key,
                "checkpoint_group": info.checkpoint_group,
                "start": offset,
                "stop": stop,
            }
        )
        offset = stop

    np.savez_compressed(
        output / "token_pca10.npz",
        pca_scores=np.concatenate(scores, axis=0),
        pca_whitened=np.concatenate(whitened, axis=0),
        run_index=np.concatenate(run_index),
        checkpoint_group_index=np.concatenate(checkpoint_group_index),
        episode_id=np.concatenate(episode_id),
        episode_step=np.concatenate(episode_step),
        control_step=np.concatenate(control_step),
        run_names=run_names,
        checkpoint_group_names=group_names,
        token_positions=np.arange(1, N_ACTION_TOKENS + 1, dtype=np.int8),
        component_indices=np.arange(1, N_COMPONENTS + 1, dtype=np.int8),
    )
    return run_slices


def save_adjacent_deltas(
    output: Path,
    ordered_runs: list[RunInfo],
    transformed: dict[str, TransformedRun],
) -> None:
    deltas = []
    from_row = []
    to_row = []
    run_index = []
    episode_id = []
    from_episode_step = []
    raw_distance = []
    raw_cosine = []
    top4_jaccard = []
    top4_site_exact = []
    whole_route_exact = []
    offset = 0
    for index, info in enumerate(ordered_runs):
        run = transformed[info.key]
        starts = run.pair_starts
        deltas.append(run.whitened[starts + 1] - run.whitened[starts])
        from_row.append(offset + starts)
        to_row.append(offset + starts + 1)
        run_index.append(np.full(len(starts), index, np.int16))
        episode_id.append(run.episode_id[starts])
        from_episode_step.append(run.episode_step[starts])
        raw_distance.append(run.raw_distance)
        raw_cosine.append(run.raw_cosine)
        top4_jaccard.append(run.top4_jaccard)
        top4_site_exact.append(run.top4_site_exact)
        whole_route_exact.append(run.whole_route_exact)
        offset += info.rows

    np.savez_compressed(
        output / "adjacent_chunk_deltas.npz",
        delta_pca10_whitened=np.concatenate(deltas, axis=0),
        from_coordinate_row=np.concatenate(from_row),
        to_coordinate_row=np.concatenate(to_row),
        run_index=np.concatenate(run_index),
        episode_id=np.concatenate(episode_id),
        from_episode_step=np.concatenate(from_episode_step),
        raw_route_distance=np.concatenate(raw_distance, axis=0),
        raw_route_cosine=np.concatenate(raw_cosine, axis=0),
        top4_jaccard=np.concatenate(top4_jaccard, axis=0),
        top4_site_exact_rate=np.concatenate(top4_site_exact, axis=0),
        whole_80_site_top4_exact=np.concatenate(whole_route_exact, axis=0),
    )


def make_examples(
    groups: dict[str, list[TransformedRun]],
) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for group, runs in groups.items():
        run = runs[0]
        start = int(run.pair_starts[0])
        examples[group] = {
            "run": run.info.key,
            "episode_id": int(run.episode_id[start]),
            "from_episode_step": int(run.episode_step[start]),
            "to_episode_step": int(run.episode_step[start + 1]),
            "T1": {
                "from_pca_scores": run.scores[start, 0],
                "from_pca_whitened": run.whitened[start, 0],
                "to_pca_whitened": run.whitened[start + 1, 0],
                "delta_pca_whitened": run.whitened[start + 1, 0]
                - run.whitened[start, 0],
            },
        }
    return examples


def make_plots(output: Path, summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = summary["checkpoint_groups"]
    macro = summary["macro_across_checkpoints"]
    tokens = np.arange(1, N_ACTION_TOKENS + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    matrix = np.asarray(
        macro["whitened_distance"]["matrix_from_token_to_next_chunk_token"]
    )
    image = axes[0, 0].imshow(matrix, cmap="viridis", aspect="equal")
    axes[0, 0].set_title("Adjacent chunks: whitened PCA-10 RMS")
    axes[0, 0].set_xlabel("Token in next chunk")
    axes[0, 0].set_ylabel("Token in current chunk")
    axes[0, 0].set_xticks(np.arange(10), tokens)
    axes[0, 0].set_yticks(np.arange(10), tokens)
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046)

    for name, item in groups.items():
        axes[0, 1].plot(
            tokens,
            item["whitened_distance"]["aligned_by_token"],
            marker="o",
            label=name,
        )
    axes[0, 1].set_title("Same-position change by action token")
    axes[0, 1].set_xlabel("Action token position")
    axes[0, 1].set_ylabel("Whitened PCA-10 RMS")
    axes[0, 1].set_xticks(tokens)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    for name, item in groups.items():
        lag_values = [item["whitened_distance"]["lag"][str(lag)]["mean"] for lag in LAGS]
        axes[1, 0].plot(LAGS, lag_values, marker="o", label=name)
    axes[1, 0].set_title("Same-position route separation by chunk lag")
    axes[1, 0].set_xlabel("Chunk lag")
    axes[1, 0].set_ylabel("Whitened PCA-10 RMS")
    axes[1, 0].set_xticks(LAGS)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    for name, item in groups.items():
        axes[1, 1].plot(
            np.arange(1, N_COMPONENTS + 1),
            item["cumulative_explained_variance"],
            marker="o",
            label=name,
        )
    axes[1, 1].set_title("PCA cumulative explained variance")
    axes[1, 1].set_xlabel("Number of components")
    axes[1, 1].set_ylabel("Explained variance ratio")
    axes[1, 1].set_xticks(np.arange(1, N_COMPONENTS + 1))
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    fig.savefig(output / "overview.png", dpi=180)
    plt.close(fig)

    names = list(groups)
    fig, axes = plt.subplots(
        1, len(names), figsize=(5 * len(names), 4.5), constrained_layout=True
    )
    if len(names) == 1:
        axes = [axes]
    vmax = max(
        np.asarray(groups[name]["component_dynamics"]["delta_rms_by_token_component"]).max()
        for name in names
    )
    for axis, name in zip(axes, names):
        values = np.asarray(
            groups[name]["component_dynamics"]["delta_rms_by_token_component"]
        )
        image = axis.imshow(values, cmap="magma", aspect="auto", vmin=0, vmax=vmax)
        axis.set_title(name)
        axis.set_xlabel("PCA component")
        axis.set_ylabel("Action token position")
        axis.set_xticks(np.arange(10), np.arange(1, 11))
        axis.set_yticks(np.arange(10), np.arange(1, 11))
    fig.colorbar(image, ax=axes, fraction=0.02, label="Adjacent delta RMS")
    fig.savefig(output / "component_change.png", dpi=180)
    plt.close(fig)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def vector_text(values: list[float] | np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):+.3f}" for value in values) + "]"


def write_report(output: Path, summary: dict[str, Any]) -> None:
    groups = summary["checkpoint_groups"]
    macro = summary["macro_across_checkpoints"]
    route = macro["raw_route"]
    white = macro["whitened_distance"]
    exact = float(route["whole_80_site_top4_exact_rate"])
    advantage = 1.0 - float(white["aligned_to_offdiag_ratio"])
    fidelity = macro["pca_fidelity_on_adjacent_changes"]
    overlap = [
        float(item["pca_subspace_stability"]["mean_squared_subspace_overlap"])
        for item in groups.values()
    ]
    example_group = next(iter(summary["examples"]))
    example = summary["examples"][example_group]

    lines = [
        "# 按 token 位置检查 chunk 间 MoE 路由变化",
        "",
        "## 结论",
        "",
        f"- 相邻 chunk 的同位置 token 路由重排很大：完整 80 个路由站点的 Top-4 集合完全一致为 **{route['whole_80_site_top4_exact_matches']:,} / {route['whole_80_site_top4_comparisons']:,}**（{pct(exact)}）。",
        f"- 同时存在可测的位置骨架：在下一个 chunk 的 T1–T10 中做最近邻，同位置命中率为 **{pct(float(white['same_position_nearest_accuracy']))}**（随机基线 10%），但这不等于位置固定。",
        f"- 同位置距离比错位位置平均低 **{pct(advantage)}**；lag 1→4 的距离逐步增大，之后基本饱和，表现为位置骨架上的连续变化叠加显著路由重排。",
        f"- PCA-10 只保留相邻变化约 **{pct(float(fidelity['projected_squared_distance_fraction']))}** 的平方距离能量，距离相关为 **r={float(fidelity['raw_vs_projected_distance_pearson']):.3f}**；它适合可视化和粗比较，不适合精确同路由判定。",
        "",
        "## 数据与表示",
        "",
        f"共分析 {summary['corpus']['chunks']:,} 个 control chunks、{summary['corpus']['episodes']:,} 个 episodes、{summary['corpus']['adjacent_pairs']:,} 对 episode 内相邻 chunks。",
        "每个 action token 先构造成 2560 维稀疏路由向量：`8 HB layers × 10 denoise steps × 32 experts`。每个站点使用实际保存的 Top-4 expert id，并将 `hb_selected_prob` 在 Top-4 内归一化为真实 combine weight。",
        "T1–T10 共用同一个 PCA 基底；PCA 分量是特征轴，token 位置是样本轴。三个 checkpoint（goal / spatial / long）分别拟合。用于跨 checkpoint 汇总的距离采用白化 PC 坐标，并对三个 checkpoint 等权平均。",
        "",
        "## Checkpoint 结果",
        "",
        "| checkpoint | chunks | adjacent pairs | PCA-10 variance | raw/PCA distance r | subspace overlap | aligned RMS | off-position RMS | nearest-position |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in groups.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{item['chunks']:,}",
                    f"{item['adjacent_pairs']:,}",
                    pct(float(item["cumulative_explained_variance"][-1])),
                    f"{item['pca_fidelity_on_adjacent_changes']['raw_vs_projected_distance_pearson']:.3f}",
                    pct(float(item["pca_subspace_stability"]["mean_squared_subspace_overlap"])),
                    f"{item['whitened_distance']['aligned_mean']:.3f}",
                    f"{item['whitened_distance']['off_diagonal_mean']:.3f}",
                    pct(float(item["whitened_distance"]["same_position_nearest_accuracy"])),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "`PCA-10 variance` 衡量对所有静态路由差异的保留量；`raw/PCA distance r` 衡量对相邻 chunk 变化大小排序的保真度。`subspace overlap` 是换一套 episode 抽样与随机种子后，十个主方向的平均 cos² 重合度。",
            f"三套 PCA-10 子空间复拟合重合度为 {pct(min(overlap))}–{pct(max(overlap))}，说明主子空间稳定；但累计方差仍低，10 维不应替代原始 Top-4 做“完全相同”判定。",
            "",
            "## 逐 token 的相邻变化",
            "",
            "下表是三个 checkpoint 的等权宏平均。白化距离单位是每个 PC 的标准差；Top-4 Jaccard 和 cosine 均在原始 2560 维表示上计算。",
            "",
            "| token | PCA-10 RMS | raw cosine | Top-4 Jaccard | site Top-4 exact | whole-token exact |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for token in range(N_ACTION_TOKENS):
        lines.append(
            f"| T{token + 1} | "
            f"{white['aligned_by_token'][token]:.3f} | "
            f"{route['cosine_by_token'][token]:.3f} | "
            f"{route['top4_jaccard_by_token'][token]:.3f} | "
            f"{pct(float(route['top4_site_exact_rate_by_token'][token]))} | "
            f"{pct(float(route['whole_80_site_top4_exact_rate_by_token'][token]))} |"
        )

    lines.extend(
        [
            "",
            "## 随 chunk 间隔的变化",
            "",
            "| lag | same-position PCA-10 RMS |",
            "|---:|---:|",
        ]
    )
    for lag in LAGS:
        lines.append(
            f"| {lag} | {white['lag'][str(lag)]['mean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 一个实际 10 维向量",
            "",
            f"示例来自 `{example['run']}`，episode {example['episode_id']}，q{example['from_episode_step']} → q{example['to_episode_step']} 的 T1：",
            "",
            f"- q{example['from_episode_step']}: `{vector_text(example['T1']['from_pca_whitened'])}`",
            f"- q{example['to_episode_step']}: `{vector_text(example['T1']['to_pca_whitened'])}`",
            f"- delta: `{vector_text(example['T1']['delta_pca_whitened'])}`",
            "",
            "这些正负数只表示 checkpoint-local PCA 坐标，不分别对应前移、旋转或夹爪等动作语义。",
            "",
            "## 产物",
            "",
            "- `token_pca10.npz`: `pca_scores` / `pca_whitened` 的 shape 均为 `[chunk, 10 token positions, 10 PCs]`。",
            "- `adjacent_chunk_deltas.npz`: episode 内每对相邻 chunk 的白化 10 维 delta 及原始路由指标。",
            "- `pca_models.npz`: 各 checkpoint 的 mean、components、explained variance，可复算新样本。",
            "- `summary.json`: 全部汇总矩阵、逐位置指标和方法元数据。",
            "- `overview.png` / `component_change.png`: 位置距离矩阵、lag 曲线、累计方差和逐 PC 变化。",
            "",
            "读取示例：",
            "",
            "```python",
            "import numpy as np",
            "",
            "data = np.load('token_pca10.npz')",
            "z = data['pca_whitened']       # [51308, 10, 10]",
            "one_token = z[0, 0]            # 第 0 个 chunk 的 T1，10 维",
            "```",
            "",
            "## 边界",
            "",
            "这里只比较同一 episode 内的 control chunks；episode 边界没有连成相邻对。PCA 拟合样本对每个 episode 等量抽取，所有 token 位置等权。不同 checkpoint 的专家槽位没有共同身份，因此没有把三套原始路由拼成一个 PCA。",
        ]
    )
    (output / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    ids = np.zeros((2, N_LAYERS, N_DENOISE, N_SUFFIX, TOP_K), dtype=np.uint8)
    ids[...] = np.arange(TOP_K, dtype=np.uint8)
    selected = np.zeros_like(ids, dtype=np.float32)
    selected[...] = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    features, token_ids = encode_route_arrays(ids, selected)
    assert features.shape == (2, N_ACTION_TOKENS, RAW_FEATURES)
    assert token_ids.shape == (
        2,
        N_ACTION_TOKENS,
        N_LAYERS,
        N_DENOISE,
        TOP_K,
    )
    first_site = features[0, 0].reshape(N_LAYERS, N_DENOISE, N_EXPERTS)[0, 0]
    np.testing.assert_allclose(first_site[:4], [0.1, 0.2, 0.3, 0.4])
    metrics = pair_route_metrics(features[:1], features[1:], token_ids[:1], token_ids[1:])
    np.testing.assert_allclose(metrics[0], 0.0)
    np.testing.assert_allclose(metrics[1], 1.0)
    np.testing.assert_allclose(metrics[2], 1.0)
    assert np.all(metrics[4])

    shifted_ids = token_ids[1:].copy() + TOP_K
    shifted_features = np.roll(features[1:], TOP_K, axis=-1)
    shifted = pair_route_metrics(
        features[:1], shifted_features, token_ids[:1], shifted_ids
    )
    np.testing.assert_allclose(shifted[2], 0.0)
    assert not np.any(shifted[4])
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.hub.resolve(), args.run_id)
    grouped_info: dict[str, list[RunInfo]] = {}
    for run in runs:
        grouped_info.setdefault(run.checkpoint_group, []).append(run)
    print(
        f"discovered {len(runs)} runs in {len(grouped_info)} checkpoint groups",
        flush=True,
    )

    models: dict[str, PCA] = {}
    sampled: dict[str, dict[str, int]] = {}
    transformed: dict[str, TransformedRun] = {}
    grouped_transformed: dict[str, list[TransformedRun]] = {}
    group_summaries: dict[str, Any] = {}

    for group, group_runs in grouped_info.items():
        sha_values = {run.checkpoint_sha256 for run in group_runs}
        if len(sha_values) != 1:
            raise ValueError(f"{group}: checkpoint group contains multiple hashes")
        print(f"fit PCA for {group}", flush=True)
        pca, sampled_controls = fit_checkpoint_pca(
            group_runs, args.fit_chunks_per_episode, args.seed
        )
        models[group] = pca
        sampled[group] = sampled_controls
        grouped_transformed[group] = []
        print(
            f"  {group} PCA-10 cumulative variance: "
            f"{pca.explained_variance_ratio_.sum():.4f}",
            flush=True,
        )
        print(f"validate PCA subspace for {group}", flush=True)
        stability = validate_pca_subspace(
            pca,
            group_runs,
            args.fit_chunks_per_episode,
            args.stability_seed,
        )
        print(
            f"  {group} mean squared subspace overlap: "
            f"{stability['mean_squared_subspace_overlap']:.4f}",
            flush=True,
        )
        for run in group_runs:
            result = transform_run(run, pca, args.batch_size)
            transformed[run.key] = result
            grouped_transformed[group].append(result)
        group_summaries[group] = summarize_group(
            group,
            pca,
            sampled_controls,
            grouped_transformed[group],
            stability,
        )

    examples = make_examples(grouped_transformed)
    total_pairs = sum(len(item.pair_starts) for item in transformed.values())
    summary = {
        "schema_version": "himoe-token-route-pca10/1",
        "method": {
            "raw_token_vector": (
                "dense Top-4 combine weights over "
                "[8 HB layers, 10 denoise steps, 32 experts]"
            ),
            "raw_features": RAW_FEATURES,
            "components": N_COMPONENTS,
            "shared_basis_across_token_positions": True,
            "separate_basis_per_checkpoint": True,
            "comparison_vector": "PCA scores divided by sqrt(explained_variance)",
            "fit_chunks_per_episode": args.fit_chunks_per_episode,
            "random_seed": args.seed,
            "stability_seed": args.stability_seed,
            "hb_layers": HB_LAYERS,
            "denoise_steps": N_DENOISE,
            "action_token_positions": N_ACTION_TOKENS,
            "experts": N_EXPERTS,
            "top_k": TOP_K,
            "lags": LAGS,
        },
        "corpus": {
            "hub": str(args.hub.resolve()),
            "run_id": args.run_id,
            "runs": len(runs),
            "checkpoint_groups": len(grouped_info),
            "chunks": sum(run.rows for run in runs),
            "episodes": sum(run.episodes for run in runs),
            "adjacent_pairs": total_pairs,
        },
        "checkpoint_groups": group_summaries,
        "macro_across_checkpoints": macro_summary(group_summaries),
        "examples": examples,
    }

    run_slices = save_coordinates(output, runs, transformed)
    save_adjacent_deltas(output, runs, transformed)
    save_models(output, models)
    index = {
        "schema_version": summary["schema_version"],
        "coordinate_shape": [sum(run.rows for run in runs), 10, 10],
        "axis_order": ["control_chunk", "action_token_position", "pca_component"],
        "run_slices": run_slices,
        "files": {
            "token_pca10.npz": {
                "pca_scores": "checkpoint-local unwhitened PCA scores",
                "pca_whitened": "main 10D comparison vector",
            },
            "adjacent_chunk_deltas.npz": {
                "delta_pca10_whitened": "next chunk minus current chunk"
            },
            "pca_models.npz": {
                "*_components": "[10, 2560] PCA basis",
                "*_mean": "[2560] fit mean",
            },
        },
    }
    (output / "index.json").write_text(
        json.dumps(plain(index), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(plain(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output, plain(summary))
    make_plots(output, plain(summary))
    print(f"wrote analysis to {output}", flush=True)


if __name__ == "__main__":
    main()
