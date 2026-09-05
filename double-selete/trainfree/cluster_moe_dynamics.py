#!/usr/bin/env python3
"""Label-blind clustering of MoE routing-state sequence grammar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "results/unlabeled_scores.npz"
DEFAULT_OUTPUT = HERE / "results/unsupervised_dynamics"
PROTOCOL = HERE / "UNSUPERVISED_DYNAMICS_PROTOCOL.md"
SEED = 20260904
WINDOW_RADIUS = 2
EARLY_START = 6
EARLY_STOP = 33
STATE_MIN_CLUSTER_SIZE = 118
STATE_MIN_SAMPLES = 14
TRAJECTORY_MIN_CLUSTER_SIZE = 23
TRAJECTORY_MIN_SAMPLES = 9
RADIUS_QUANTILES = (0.90, 0.95, 0.99)
COMPONENTS = (
    "gate_concentration_delta2_high",
    "gate_concentration_w4_low",
    "late_flow_volatility_w4_high",
    "route_acceleration_w4_high",
    "route_mobility_w4_low",
    "deep_soft_consensus_delta2_high",
    "deep_soft_mixture_delta2_high",
    "l15_soft_mixture_now_high",
    "l15_soft_consensus_now_high",
    "l15_soft_final_jump_high",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stability-repeats", type=int, default=30)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_curves(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        episode = np.asarray(archive["episode"], dtype=np.int64)
        group = np.asarray(archive["group"], dtype=np.int64)
        curves = np.stack(
            [
                np.asarray(archive[f"component__{component}"], dtype=np.float64)
                for component in COMPONENTS
            ],
            axis=-1,
        )
    if not np.array_equal(episode, np.arange(512)):
        raise ValueError("expected episode IDs 0..511")
    _, group_counts = np.unique(group, return_counts=True)
    if len(group_counts) != 16 or not np.all(group_counts == 32):
        raise ValueError("expected 16 pools with 32 trajectories each")
    if curves.shape != (512, 52, len(COMPONENTS)):
        raise ValueError(f"unexpected curve shape: {curves.shape}")
    if not np.isfinite(curves[:, 4:35]).all():
        raise ValueError("common q4..q34 prefix must be complete")
    return episode, group, curves


def robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    center = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, (0.25, 0.75), axis=0)
    scale = q75 - q25
    fallback = matrix.std(axis=0)
    scale = np.where(scale > 1e-10, scale, np.where(fallback > 1e-10, fallback, 1.0))
    values = np.clip((matrix - center) / scale, -10.0, 10.0)
    keep = values.std(axis=0) > 1e-10
    return values[:, keep], {"center": center, "scale": scale, "keep": keep}


def apply_scale(matrix: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    values = np.clip(
        (np.asarray(matrix, dtype=np.float64) - scaler["center"]) / scaler["scale"],
        -10.0,
        10.0,
    )
    return values[:, scaler["keep"]]


def embed(matrix: np.ndarray) -> tuple[np.ndarray, PCA, dict[str, np.ndarray]]:
    standardized, scaler = robust_scale(matrix)
    pca = PCA(n_components=0.90, svd_solver="full")
    return pca.fit_transform(standardized), pca, scaler


def window_feature_names() -> list[str]:
    names = [
        f"level_t{offset:+d}_{component}"
        for offset in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1)
        for component in COMPONENTS
    ]
    names.extend(
        f"delta_t{offset:+d}_{component}"
        for offset in range(-WINDOW_RADIUS + 1, WINDOW_RADIUS + 1)
        for component in COMPONENTS
    )
    return names


def collect_windows(
    curves: np.ndarray, scope: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    episodes: list[int] = []
    queries: list[int] = []
    for episode in range(len(curves)):
        if scope == "early":
            centers = range(EARLY_START, EARLY_STOP)
        elif scope == "full":
            finite = np.isfinite(curves[episode]).all(axis=1)
            centers = [
                query
                for query in range(WINDOW_RADIUS, curves.shape[1] - WINDOW_RADIUS)
                if finite[query - WINDOW_RADIUS : query + WINDOW_RADIUS + 1].all()
            ]
        else:
            raise ValueError(scope)
        for query in centers:
            window = curves[
                episode,
                query - WINDOW_RADIUS : query + WINDOW_RADIUS + 1,
            ]
            if not np.isfinite(window).all():
                raise ValueError(f"non-finite {scope} window at episode={episode}, q={query}")
            rows.append(np.concatenate([window.reshape(-1), np.diff(window, axis=0).reshape(-1)]))
            episodes.append(episode)
            queries.append(query)
    matrix = np.stack(rows)
    if matrix.shape[1] != len(window_feature_names()):
        raise AssertionError("window feature width mismatch")
    return matrix, np.asarray(episodes, np.int64), np.asarray(queries, np.int64)


def canonicalize(labels: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    clusters = [value for value in np.unique(labels) if value >= 0]
    order = sorted(clusters, key=lambda value: float(embedding[labels == value, 0].mean()))
    mapping = {old: new for new, old in enumerate(order)}
    return np.asarray([mapping.get(int(value), -1) for value in labels], dtype=np.int64)


def cluster_summary(labels: np.ndarray, embedding: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    assigned = labels >= 0
    clusters = [value for value in np.unique(labels) if value >= 0]
    silhouette = float("nan")
    if len(clusters) >= 2 and assigned.sum() > len(clusters):
        silhouette = float(silhouette_score(embedding[assigned], labels[assigned]))
    return {
        "clusters": len(clusters),
        "coverage": float(assigned.mean()),
        "noise": int((~assigned).sum()),
        "sizes": [int((labels == value).sum()) for value in clusters],
        "silhouette_clustered": silhouette,
    }


def fit_hdbscan(
    embedding: np.ndarray, min_cluster_size: int, min_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=False,
            copy=True,
            n_jobs=-1,
        ).fit(embedding)
    labels = canonicalize(model.labels_, embedding)
    return labels, np.asarray(model.probabilities_, dtype=np.float64)


def state_vocabulary(
    windows: np.ndarray, seed: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[float, np.ndarray],
    np.ndarray,
    PCA,
    dict[str, np.ndarray],
    GaussianMixture,
    np.ndarray,
    pd.DataFrame,
]:
    embedding, pca, scaler = embed(windows)
    diagnostic_hdb_labels, _ = fit_hdbscan(
        embedding, STATE_MIN_CLUSTER_SIZE, STATE_MIN_SAMPLES
    )
    bic_rows = []
    models = []
    for components in range(1, 13):
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            n_init=5,
            random_state=seed,
            reg_covar=1e-6,
        ).fit(embedding)
        bic_rows.append(
            {
                "components": components,
                "bic": float(model.bic(embedding)),
                "selected": False,
            }
        )
        models.append(model)
    best_index = int(np.argmin([row["bic"] for row in bic_rows]))
    bic_rows[best_index]["selected"] = True
    model = models[best_index]
    raw_labels = model.predict(embedding).astype(np.int64)
    probabilities = model.predict_proba(embedding).max(axis=1)
    order = np.argsort(model.means_[:, 0])
    raw_to_canonical = np.empty(model.n_components, dtype=np.int64)
    raw_to_canonical[order] = np.arange(model.n_components)
    member_labels = raw_to_canonical[raw_labels]
    prototypes = []
    for raw_state in order:
        indices = np.flatnonzero(raw_labels == raw_state)
        distance = np.linalg.norm(embedding[indices] - model.means_[raw_state], axis=1)
        prototypes.append(embedding[indices[int(np.argmin(distance))]])
    prototypes = np.stack(prototypes)

    assigned_mean = model.means_[raw_labels]
    assigned_covariance = model.covariances_[raw_labels]
    mahalanobis = np.sqrt(
        np.sum((embedding - assigned_mean) ** 2 / assigned_covariance, axis=1)
    )

    radii: dict[float, np.ndarray] = {}
    for quantile in RADIUS_QUANTILES:
        values = []
        for state in range(len(prototypes)):
            values.append(float(np.quantile(mahalanobis[member_labels == state], quantile)))
        radii[quantile] = np.asarray(values, dtype=np.float64)
    return (
        embedding,
        member_labels,
        probabilities,
        diagnostic_hdb_labels,
        radii,
        prototypes,
        pca,
        scaler,
        model,
        raw_to_canonical,
        pd.DataFrame(bic_rows),
    )


def assign_states(
    embedding: np.ndarray,
    model: GaussianMixture,
    raw_to_canonical: np.ndarray,
    radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_labels = model.predict(embedding).astype(np.int64)
    labels = raw_to_canonical[raw_labels]
    distance = np.sqrt(
        np.sum(
            (embedding - model.means_[raw_labels]) ** 2
            / model.covariances_[raw_labels],
            axis=1,
        )
    )
    labels[distance > radius[labels]] = -1
    return labels, distance


def sequence_matrix(
    row_episode: np.ndarray,
    row_query: np.ndarray,
    labels: np.ndarray,
    scope: str,
) -> np.ndarray:
    output = np.full((512, 52), -2, dtype=np.int16)
    if scope == "early":
        take = (row_query >= EARLY_START) & (row_query < EARLY_STOP)
    elif scope == "full":
        take = np.ones(len(row_query), dtype=bool)
    else:
        raise ValueError(scope)
    output[row_episode[take], row_query[take]] = labels[take].astype(np.int16)
    expected = 27 if scope == "early" else None
    if expected is not None and not np.all((output != -2).sum(axis=1) == expected):
        raise AssertionError("early sequence length mismatch")
    return output


def grammar_feature_names(state_count: int) -> list[str]:
    names = []
    for family in ("occupancy", "visit_rate", "longest_dwell", "late_minus_early"):
        names.extend(f"{family}_state{state}" for state in range(-1, state_count))
    names.extend(
        f"transition_state{left}_to_state{right}"
        for left in range(-1, state_count)
        for right in range(-1, state_count)
    )
    names.extend(
        [
            "switch_rate",
            "run_count_rate",
            "run_length_mean_fraction",
            "run_length_max_fraction",
            "run_length_std_fraction",
            "occupancy_entropy",
        ]
    )
    names.extend(f"lag{lag}_exact_recurrence" for lag in range(1, 9))
    names.extend(f"lag{lag}_known_recurrence" for lag in range(1, 9))
    names.extend(f"lag{lag}_known_pair_fraction" for lag in range(1, 9))
    names.extend(f"lag{lag}_return_after_change" for lag in range(2, 9))
    return names


def grammar_features(
    sequences: np.ndarray, state_count: int
) -> tuple[np.ndarray, list[str]]:
    category_count = state_count + 1
    rows = []
    for sequence_row in sequences:
        sequence = sequence_row[sequence_row != -2].astype(np.int64)
        if len(sequence) < 9:
            raise ValueError("sequence too short for lag grammar")
        encoded = sequence + 1
        n = len(encoded)
        occupancy = np.bincount(encoded, minlength=category_count) / n
        change = np.r_[True, encoded[1:] != encoded[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], n]
        run_states = encoded[starts]
        run_lengths = ends - starts
        visits = np.bincount(run_states, minlength=category_count) / n
        longest = np.zeros(category_count, dtype=np.float64)
        for state in range(category_count):
            selected = run_lengths[run_states == state]
            longest[state] = (selected.max() / n) if len(selected) else 0.0
        midpoint = n // 2
        early = np.bincount(encoded[:midpoint], minlength=category_count) / midpoint
        late = np.bincount(encoded[midpoint:], minlength=category_count) / (n - midpoint)
        transition = np.zeros((category_count, category_count), dtype=np.float64)
        np.add.at(transition, (encoded[:-1], encoded[1:]), 1.0)
        transition /= n - 1
        entropy = -np.sum(occupancy[occupancy > 0] * np.log(occupancy[occupancy > 0]))
        entropy /= math.log(category_count) if category_count > 1 else 1.0
        global_features = np.asarray(
            [
                np.mean(encoded[1:] != encoded[:-1]),
                len(run_lengths) / n,
                run_lengths.mean() / n,
                run_lengths.max() / n,
                run_lengths.std() / n,
                entropy,
            ],
            dtype=np.float64,
        )
        exact = []
        known_recurrence = []
        known_fraction = []
        for lag in range(1, 9):
            left, right = encoded[:-lag], encoded[lag:]
            exact.append(float(np.mean(left == right)))
            known = (left > 0) & (right > 0)
            known_fraction.append(float(known.mean()))
            known_recurrence.append(float(np.mean(left[known] == right[known])) if known.any() else 0.0)
        returns = []
        for lag in range(2, 9):
            matches = encoded[:-lag] == encoded[lag:]
            changed = np.asarray(
                [np.any(encoded[index + 1 : index + lag] != encoded[index]) for index in range(n - lag)]
            )
            returns.append(float(np.mean(matches & changed)))
        rows.append(
            np.concatenate(
                [
                    occupancy,
                    visits,
                    longest,
                    late - early,
                    transition.reshape(-1),
                    global_features,
                    exact,
                    known_recurrence,
                    known_fraction,
                    returns,
                ]
            )
        )
    matrix = np.stack(rows)
    names = grammar_feature_names(state_count)
    if matrix.shape[1] != len(names) or not np.isfinite(matrix).all():
        raise AssertionError("grammar feature mismatch")
    return matrix, names


def cluster_grammar(
    matrix: np.ndarray, min_cluster_size: int, min_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA, dict[str, np.ndarray]]:
    embedding, pca, scaler = embed(matrix)
    labels, probability = fit_hdbscan(embedding, min_cluster_size, min_samples)
    return labels, probability, embedding, pca, scaler


def fit_optics(embedding: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = OPTICS(
            min_samples=TRAJECTORY_MIN_SAMPLES,
            min_cluster_size=TRAJECTORY_MIN_CLUSTER_SIZE,
            cluster_method="xi",
            xi=0.05,
            metric="euclidean",
            n_jobs=-1,
        ).fit_predict(embedding)
    return canonicalize(labels, embedding)


def fit_gmm_bic(
    embedding: np.ndarray, scope: str, seed: int
) -> tuple[pd.DataFrame, np.ndarray, int]:
    rows = []
    models = []
    for components in range(1, 13):
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            n_init=5,
            random_state=seed,
            reg_covar=1e-6,
        ).fit(embedding)
        rows.append({"scope": scope, "components": components, "bic": float(model.bic(embedding))})
        models.append(model)
    best_index = int(np.argmin([row["bic"] for row in rows]))
    labels = canonicalize(models[best_index].predict(embedding), embedding)
    return pd.DataFrame(rows), labels, best_index + 1


def grammar_model_suite(
    matrix: np.ndarray, scope: str, seed: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    PCA,
    dict[str, np.ndarray],
]:
    primary_labels, probability, embedding, pca, scaler = cluster_grammar(
        matrix, TRAJECTORY_MIN_CLUSTER_SIZE, TRAJECTORY_MIN_SAMPLES
    )
    model_rows = []
    sensitivity = {}
    for min_cluster_size in (16, 23, 32):
        for min_samples in (8, 9, 10):
            labels, _, candidate_embedding, _, _ = cluster_grammar(
                matrix, min_cluster_size, min_samples
            )
            name = f"{scope}_hdb_mcs{min_cluster_size}_ms{min_samples}"
            sensitivity[name] = labels
            model_rows.append(
                {
                    "scope": scope,
                    "method": "HDBSCAN",
                    "variant": f"mcs{min_cluster_size}_ms{min_samples}",
                    "selected_primary": min_cluster_size == 23 and min_samples == 9,
                    **cluster_summary(labels, candidate_embedding),
                }
            )
    optics = fit_optics(embedding)
    model_rows.append(
        {
            "scope": scope,
            "method": "OPTICS-Xi",
            "variant": "mcs23_ms9_xi0.05",
            "selected_primary": False,
            **cluster_summary(optics, embedding),
        }
    )
    bic, gmm_labels, gmm_k = fit_gmm_bic(embedding, scope, seed)
    model_rows.append(
        {
            "scope": scope,
            "method": "GMM-BIC",
            "variant": f"K{gmm_k}",
            "selected_primary": False,
            **cluster_summary(gmm_labels, embedding),
        }
    )
    return (
        primary_labels,
        probability,
        embedding,
        optics,
        gmm_labels,
        pd.DataFrame(model_rows),
        bic,
        sensitivity,
        pca,
        scaler,
    )


def permute_early_curves(
    curves: np.ndarray, group: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    output = curves.copy()
    for pool in np.unique(group):
        indices = np.flatnonzero(group == pool)
        for query in range(4, 35):
            for component in range(curves.shape[2]):
                source = rng.permutation(indices)
                output[indices, query, component] = curves[source, query, component]
    return output


def synthetic_positive_control(_seed: int) -> dict[str, Any]:
    truth = np.arange(512, dtype=np.int64) % 3
    sequences = np.full((512, 27), -2, dtype=np.int16)
    for index, category in enumerate(truth):
        if category == 0:
            row = np.zeros(27, dtype=np.int16)
        elif category == 1:
            row = (np.arange(27) % 2).astype(np.int16)
        else:
            row = np.r_[np.zeros(8), np.full(11, 2), np.zeros(8)].astype(np.int16)
        sequences[index] = row
    matrix, _ = grammar_features(sequences, state_count=3)
    labels, _, embedding, _, _ = cluster_grammar(
        matrix, TRAJECTORY_MIN_CLUSTER_SIZE, TRAJECTORY_MIN_SAMPLES
    )
    return {**cluster_summary(labels, embedding), "ari_truth": float(adjusted_rand_score(truth, labels))}


def main() -> None:
    args = parse_args()
    if args.self_test:
        stable = np.zeros((1, 27), dtype=np.int16)
        alternating = (np.arange(27) % 2)[None, :].astype(np.int16)
        matrix, names = grammar_features(np.concatenate([stable, alternating]), 2)
        switch = names.index("switch_rate")
        lag2 = names.index("lag2_exact_recurrence")
        return2 = names.index("lag2_return_after_change")
        assert matrix[0, switch] == 0.0
        assert matrix[1, switch] == 1.0
        assert matrix[1, lag2] == 1.0
        assert matrix[1, return2] == 1.0
        print("self-test passed")
        return
    if args.stability_repeats < 1:
        raise ValueError("stability repeats must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    episode, group, curves = load_curves(args.source)
    early_windows, early_episode, early_query = collect_windows(curves, "early")
    full_windows, full_episode, full_query = collect_windows(curves, "full")
    (
        state_embedding,
        state_labels,
        state_probability,
        state_hdb_labels,
        state_radii,
        prototypes,
        state_pca,
        state_scaler,
        state_model,
        state_raw_to_canonical,
        state_bic,
    ) = state_vocabulary(early_windows, args.seed)
    state_count = len(prototypes)
    full_state_embedding = state_pca.transform(apply_scale(full_windows, state_scaler))

    sequences: dict[tuple[str, int], np.ndarray] = {}
    state_distances: dict[int, np.ndarray] = {}
    for quantile in RADIUS_QUANTILES:
        key = int(round(quantile * 100))
        labels, distance = assign_states(
            full_state_embedding,
            state_model,
            state_raw_to_canonical,
            state_radii[quantile],
        )
        state_distances[key] = distance
        sequences[("early", key)] = sequence_matrix(
            full_episode, full_query, labels, "early"
        )
        sequences[("full", key)] = sequence_matrix(
            full_episode, full_query, labels, "full"
        )

    grammar: dict[tuple[str, int], np.ndarray] = {}
    grammar_names: list[str] | None = None
    for sequence_key, values in sequences.items():
        matrix, names = grammar_features(values, state_count)
        grammar[sequence_key] = matrix
        if grammar_names is None:
            grammar_names = names
        elif names != grammar_names:
            raise AssertionError("grammar names changed across scopes")
    assert grammar_names is not None

    suites = {}
    model_frames = []
    bic_frames = []
    sensitivity: dict[str, np.ndarray] = {}
    pca_models = {}
    grammar_scalers = {}
    for scope in ("early", "full"):
        suite = grammar_model_suite(grammar[(scope, 95)], scope, args.seed)
        suites[scope] = suite
        model_frames.append(suite[5])
        bic_frames.append(suite[6])
        sensitivity.update(suite[7])
        pca_models[scope] = suite[8]
        grammar_scalers[scope] = suite[9]
        for radius in (90, 99):
            labels, _, candidate_embedding, _, _ = cluster_grammar(
                grammar[(scope, radius)],
                TRAJECTORY_MIN_CLUSTER_SIZE,
                TRAJECTORY_MIN_SAMPLES,
            )
            name = f"{scope}_radius{radius}_hdb_mcs23_ms9"
            sensitivity[name] = labels
            model_frames.append(
                pd.DataFrame(
                    [
                        {
                            "scope": scope,
                            "method": "HDBSCAN-radius",
                            "variant": f"q{radius}_mcs23_ms9",
                            "selected_primary": False,
                            **cluster_summary(labels, candidate_embedding),
                        }
                    ]
                )
            )

    early_labels, early_probability, early_embedding = suites["early"][:3]
    full_labels, full_probability, full_embedding = suites["full"][:3]
    assignments = pd.DataFrame(
        {
            "episode_id": episode,
            "group": group,
            "early_hdbscan_cluster": early_labels,
            "early_hdbscan_probability": early_probability,
            "early_optics_cluster": suites["early"][3],
            "early_gmm_bic_cluster": suites["early"][4],
            "full_hdbscan_cluster": full_labels,
            "full_hdbscan_probability": full_probability,
            "full_optics_cluster": suites["full"][3],
            "full_gmm_bic_cluster": suites["full"][4],
        }
    )

    state_rows = []
    center_offset = WINDOW_RADIUS * len(COMPONENTS)
    for state in range(state_count):
        members = state_labels == state
        row = {
            "state": state,
            "gmm_member_n": int(members.sum()),
            "radius_q90": state_radii[0.90][state],
            "radius_q95": state_radii[0.95][state],
            "radius_q99": state_radii[0.99][state],
        }
        if members.any():
            central = early_windows[members, center_offset : center_offset + len(COMPONENTS)]
            for component_index, component in enumerate(COMPONENTS):
                row[f"center_mean__{component}"] = float(central[:, component_index].mean())
        state_rows.append(row)
    state_frame = pd.DataFrame(state_rows)

    rng = np.random.default_rng(args.seed)
    stability_rows = []
    for scope, labels, base_matrix in (
        ("early", early_labels, grammar[("early", 95)]),
        ("full", full_labels, grammar[("full", 95)]),
    ):
        feature_std = base_matrix.std(axis=0)
        for repeat in range(args.stability_repeats):
            jittered = base_matrix + rng.normal(size=base_matrix.shape) * feature_std * 0.01
            candidate, _, candidate_embedding, _, _ = cluster_grammar(
                jittered, TRAJECTORY_MIN_CLUSTER_SIZE, TRAJECTORY_MIN_SAMPLES
            )
            stability_rows.append(
                {
                    "scope": scope,
                    "algorithm": "HDBSCAN",
                    "test": "jitter_1pct",
                    "repeat": repeat,
                    "ari_vs_primary": float(adjusted_rand_score(labels, candidate)),
                    **cluster_summary(candidate, candidate_embedding),
                }
            )
            jitter_embedding, _, _ = embed(jittered)
            _, jitter_gmm, jitter_k = fit_gmm_bic(
                jitter_embedding, scope, args.seed + 1000 + repeat
            )
            stability_rows.append(
                {
                    "scope": scope,
                    "algorithm": "GMM-BIC",
                    "test": "jitter_1pct",
                    "repeat": repeat,
                    "ari_vs_primary": float(
                        adjusted_rand_score(suites[scope][4], jitter_gmm)
                    ),
                    "selected_k": jitter_k,
                    **cluster_summary(jitter_gmm, jitter_embedding),
                }
            )
            width = max(2, int(math.ceil(base_matrix.shape[1] * 0.8)))
            columns = np.sort(rng.choice(base_matrix.shape[1], size=width, replace=False))
            candidate, _, candidate_embedding, _, _ = cluster_grammar(
                base_matrix[:, columns],
                TRAJECTORY_MIN_CLUSTER_SIZE,
                TRAJECTORY_MIN_SAMPLES,
            )
            stability_rows.append(
                {
                    "scope": scope,
                    "algorithm": "HDBSCAN",
                    "test": "feature_subsample_80pct",
                    "repeat": repeat,
                    "ari_vs_primary": float(adjusted_rand_score(labels, candidate)),
                    **cluster_summary(candidate, candidate_embedding),
                }
            )
            subset_embedding, _, _ = embed(base_matrix[:, columns])
            _, subset_gmm, subset_k = fit_gmm_bic(
                subset_embedding, scope, args.seed + 2000 + repeat
            )
            stability_rows.append(
                {
                    "scope": scope,
                    "algorithm": "GMM-BIC",
                    "test": "feature_subsample_80pct",
                    "repeat": repeat,
                    "ari_vs_primary": float(
                        adjusted_rand_score(suites[scope][4], subset_gmm)
                    ),
                    "selected_k": subset_k,
                    **cluster_summary(subset_gmm, subset_embedding),
                }
            )

    for repeat in range(args.stability_repeats):
        permuted = permute_early_curves(curves, group, rng)
        permuted_windows, permuted_episode, permuted_query = collect_windows(permuted, "early")
        permuted_embedding = state_pca.transform(apply_scale(permuted_windows, state_scaler))
        permuted_state, _ = assign_states(
            permuted_embedding,
            state_model,
            state_raw_to_canonical,
            state_radii[0.95],
        )
        permuted_sequence = sequence_matrix(
            permuted_episode, permuted_query, permuted_state, "early"
        )
        permuted_grammar, _ = grammar_features(permuted_sequence, state_count)
        candidate, _, candidate_embedding, _, _ = cluster_grammar(
            permuted_grammar,
            TRAJECTORY_MIN_CLUSTER_SIZE,
            TRAJECTORY_MIN_SAMPLES,
        )
        stability_rows.append(
            {
                "scope": "early",
                "algorithm": "HDBSCAN",
                "test": "within_pool_time_component_identity_permutation",
                "repeat": repeat,
                "ari_vs_primary": float(adjusted_rand_score(early_labels, candidate)),
                **cluster_summary(candidate, candidate_embedding),
            }
        )
        permuted_grammar_embedding, _, _ = embed(permuted_grammar)
        _, permuted_gmm, permuted_k = fit_gmm_bic(
            permuted_grammar_embedding, "early", args.seed + 3000 + repeat
        )
        stability_rows.append(
            {
                "scope": "early",
                "algorithm": "GMM-BIC",
                "test": "within_pool_time_component_identity_permutation",
                "repeat": repeat,
                "ari_vs_primary": float(
                    adjusted_rand_score(suites["early"][4], permuted_gmm)
                ),
                "selected_k": permuted_k,
                **cluster_summary(permuted_gmm, permuted_grammar_embedding),
            }
        )
    stability = pd.DataFrame(stability_rows)

    assignments.to_csv(args.output / "unlabeled_trajectory_assignments.csv", index=False)
    state_frame.to_csv(args.output / "state_vocabulary.csv", index=False)
    pd.concat(model_frames, ignore_index=True).to_csv(
        args.output / "model_selection.csv", index=False
    )
    pd.concat(bic_frames, ignore_index=True).to_csv(args.output / "gmm_bic.csv", index=False)
    state_bic.to_csv(args.output / "state_gmm_bic.csv", index=False)
    stability.to_csv(args.output / "stability_sentinels.csv", index=False)
    np.savez_compressed(
        args.output / "unlabeled_dynamics.npz",
        episode=episode,
        group=group,
        components=np.asarray(COMPONENTS),
        state_feature_names=np.asarray(window_feature_names()),
        state_embedding=state_embedding.astype(np.float32),
        state_gmm_bic_labels=state_labels.astype(np.int16),
        state_gmm_bic_probability=state_probability.astype(np.float32),
        state_hdbscan_diagnostic_labels=state_hdb_labels.astype(np.int16),
        state_prototypes=prototypes.astype(np.float32),
        state_gmm_weights=state_model.weights_[np.argsort(state_model.means_[:, 0])].astype(np.float32),
        state_gmm_means=state_model.means_[np.argsort(state_model.means_[:, 0])].astype(np.float32),
        state_gmm_covariances=state_model.covariances_[np.argsort(state_model.means_[:, 0])].astype(np.float32),
        state_radius_q90=state_radii[0.90].astype(np.float32),
        state_radius_q95=state_radii[0.95].astype(np.float32),
        state_radius_q99=state_radii[0.99].astype(np.float32),
        full_window_episode=full_episode.astype(np.int16),
        full_window_query=full_query.astype(np.int16),
        full_state_distance_q95=state_distances[95].astype(np.float32),
        early_sequence_q90=sequences[("early", 90)],
        early_sequence_q95=sequences[("early", 95)],
        early_sequence_q99=sequences[("early", 99)],
        full_sequence_q90=sequences[("full", 90)],
        full_sequence_q95=sequences[("full", 95)],
        full_sequence_q99=sequences[("full", 99)],
        grammar_feature_names=np.asarray(grammar_names),
        early_grammar_q95=grammar[("early", 95)].astype(np.float32),
        full_grammar_q95=grammar[("full", 95)].astype(np.float32),
        early_embedding=early_embedding.astype(np.float32),
        full_embedding=full_embedding.astype(np.float32),
        early_explained_variance_ratio=pca_models["early"].explained_variance_ratio_.astype(np.float32),
        full_explained_variance_ratio=pca_models["full"].explained_variance_ratio_.astype(np.float32),
        **{name: value.astype(np.int16) for name, value in sensitivity.items()},
    )

    artifact_names = (
        "unlabeled_trajectory_assignments.csv",
        "state_vocabulary.csv",
        "model_selection.csv",
        "gmm_bic.csv",
        "stability_sentinels.csv",
        "state_gmm_bic.csv",
        "unlabeled_dynamics.npz",
    )
    manifest = {
        "schema": "himoe.moe_dynamics_unsupervised.v1",
        "created": "2026-09-04",
        "training": False,
        "exploratory_after_peak_cluster_audit": True,
        "labels_read": [],
        "success_read": False,
        "episode_length_as_feature": False,
        "termination_mask_used": "retrospective full scope only",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "protocol_sha256": sha256(PROTOCOL),
        "code_sha256": sha256(Path(__file__)),
        "episodes": len(episode),
        "init_pools": len(np.unique(group)),
        "components": list(COMPONENTS),
        "state_window": {"radius": WINDOW_RADIUS, "dimensions": early_windows.shape[1]},
        "state_vocabulary": {
            "training_windows": len(early_windows),
            "algorithm": "diagonal GMM selected by BIC",
            "selected_components": state_count,
            "hdbscan_diagnostic": cluster_summary(state_hdb_labels, state_embedding),
            "prototype_count": state_count,
            "pca_dimensions": state_embedding.shape[1],
            "pca_variance": float(state_pca.explained_variance_ratio_.sum()),
            "primary_radius_quantile": 0.95,
        },
        "early_primary": {
            **cluster_summary(early_labels, early_embedding),
            "pca_dimensions": early_embedding.shape[1],
            "pca_variance": float(pca_models["early"].explained_variance_ratio_.sum()),
            "query_centers": [EARLY_START, EARLY_STOP - 1],
        },
        "full_retrospective_primary": {
            **cluster_summary(full_labels, full_embedding),
            "pca_dimensions": full_embedding.shape[1],
            "pca_variance": float(pca_models["full"].explained_variance_ratio_.sum()),
        },
        "gmm_bic_selected_components": {
            "early": int(suites["early"][6].loc[suites["early"][6]["bic"].idxmin(), "components"]),
            "full": int(suites["full"][6].loc[suites["full"][6]["bic"].idxmin(), "components"]),
        },
        "positive_control": synthetic_positive_control(args.seed),
        "stability_repeats": args.stability_repeats,
        "artifact_sha256": {
            name: sha256(args.output / name) for name in artifact_names
        },
    }
    (args.output / "unlabeled_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plain(manifest), indent=2))


if __name__ == "__main__":
    main()
