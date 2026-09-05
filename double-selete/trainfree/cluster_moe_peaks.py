#!/usr/bin/env python3
"""Label-blind adaptive clustering of MoE peak/valley trajectories."""

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
DEFAULT_SCORES = HERE / "results/unlabeled_scores.npz"
DEFAULT_OUTPUT = HERE / "results/unsupervised_cluster"
PROTOCOL = HERE / "UNSUPERVISED_CLUSTER_PROTOCOL.md"
SEED = 20260903
QUERY_START = 4
QUERY_STOP = 35
ANCHOR_RADIUS = 3
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
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
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
                np.asarray(
                    archive[f"component__{name}"][:, QUERY_START:QUERY_STOP],
                    dtype=np.float64,
                )
                for name in COMPONENTS
            ],
            axis=-1,
        )
    if not np.array_equal(episode, np.arange(512)):
        raise ValueError("expected episode IDs 0..511")
    _, counts = np.unique(group, return_counts=True)
    if len(counts) != 16 or not np.all(counts == 32):
        raise ValueError("expected 16 init pools with 32 trajectories each")
    if curves.shape != (512, 31, len(COMPONENTS)) or not np.isfinite(curves).all():
        raise ValueError(f"unexpected curve tensor: {curves.shape}")
    return episode, group, curves


def select_anchor(curves: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "extreme":
        salience = np.sqrt(np.mean((curves - 0.5) ** 2, axis=2))
    elif mode == "novelty":
        salience = np.zeros(curves.shape[:2], dtype=np.float64)
        for query in range(ANCHOR_RADIUS, curves.shape[1]):
            baseline = curves[:, query - ANCHOR_RADIUS : query].mean(axis=1)
            salience[:, query] = np.sqrt(
                np.mean((curves[:, query] - baseline) ** 2, axis=1)
            )
    else:
        raise ValueError(mode)
    local = salience[:, ANCHOR_RADIUS:-ANCHOR_RADIUS]
    anchor = ANCHOR_RADIUS + np.argmax(local, axis=1)
    return anchor, salience[np.arange(len(curves)), anchor]


def build_features(
    curves: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if mode == "full_curve":
        anchor, salience = select_anchor(curves, "extreme")
        names = [
            f"curve_q{query + QUERY_START}_{component}"
            for query in range(curves.shape[1])
            for component in COMPONENTS
        ]
        return curves.reshape(len(curves), -1), anchor, salience, names

    anchor, salience = select_anchor(curves, mode)
    window = np.stack(
        [
            curves[index, value - ANCHOR_RADIUS : value + ANCHOR_RADIUS + 1]
            for index, value in enumerate(anchor)
        ]
    )
    delta = np.diff(window, axis=1)
    summaries = (window.mean(axis=1), window.std(axis=1), window.min(axis=1), window.max(axis=1))
    matrix = np.concatenate(
        [window.reshape(len(window), -1), delta.reshape(len(window), -1), *summaries],
        axis=1,
    )
    names = [
        f"level_t{offset:+d}_{component}"
        for offset in range(-ANCHOR_RADIUS, ANCHOR_RADIUS + 1)
        for component in COMPONENTS
    ]
    names.extend(
        f"delta_t{offset:+d}_{component}"
        for offset in range(-ANCHOR_RADIUS + 1, ANCHOR_RADIUS + 1)
        for component in COMPONENTS
    )
    names.extend(
        f"{stat}_{component}"
        for stat in ("mean", "std", "min", "max")
        for component in COMPONENTS
    )
    if matrix.shape[1] != len(names):
        raise AssertionError("feature name mismatch")
    return matrix, anchor, salience, names


def robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(matrix, dtype=np.float64)
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = q75 - q25
    fallback = values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    standardized = np.clip((values - center) / scale, -10.0, 10.0)
    keep = standardized.std(axis=0) > 1e-9
    return standardized[:, keep], {"center": center, "scale": scale, "keep": keep}


def embed(matrix: np.ndarray) -> tuple[np.ndarray, PCA, dict[str, np.ndarray]]:
    standardized, scaler = robust_scale(matrix)
    model = PCA(n_components=0.90, svd_solver="full")
    embedding = model.fit_transform(standardized)
    return embedding, model, scaler


def canonicalize(labels: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    clusters = [value for value in np.unique(labels) if value >= 0]
    order = sorted(clusters, key=lambda value: float(embedding[labels == value, 0].mean()))
    mapping = {old: new for new, old in enumerate(order)}
    return np.asarray([mapping.get(int(value), -1) for value in labels], dtype=np.int64)


def cluster_summary(labels: np.ndarray, embedding: np.ndarray) -> dict[str, Any]:
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
    matrix: np.ndarray, min_cluster_size: int, min_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA, dict[str, np.ndarray]]:
    embedding, pca, scaler = embed(matrix)
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
    return labels, np.asarray(model.probabilities_), embedding, pca, scaler


def permute_curves(
    curves: np.ndarray, group: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    output = np.empty_like(curves)
    for value in np.unique(group):
        index = np.flatnonzero(group == value)
        for query in range(curves.shape[1]):
            for component in range(curves.shape[2]):
                source = rng.permutation(index)
                output[index, query, component] = curves[source, query, component]
    return output


def positive_control(
    count: int, dimensions: int, min_cluster_size: int, min_samples: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    truth = np.arange(count, dtype=np.int64) % 3
    values = rng.normal(0.0, 0.35, size=(count, dimensions))
    values[truth == 0, 0] -= 4.0
    values[truth == 1, 0] += 4.0
    values[truth == 2, 1] += 4.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            copy=True,
            n_jobs=-1,
        ).fit_predict(values)
    return {
        **cluster_summary(labels, values),
        "ari_truth": float(adjusted_rand_score(truth, labels)),
    }


def main() -> None:
    args = parse_args()
    if args.self_test:
        synthetic = positive_control(510, 20, 23, 9, args.seed)
        assert synthetic["clusters"] == 3
        assert synthetic["coverage"] > 0.95
        assert synthetic["ari_truth"] > 0.95
        print("self-test passed")
        return
    if args.stability_repeats < 1:
        raise ValueError("stability-repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    episode, group, curves = load_curves(args.scores)
    matrix, anchor, salience, feature_names = build_features(curves, "extreme")
    n = len(matrix)
    primary_mcs = math.ceil(math.sqrt(n))
    primary_ms = math.ceil(math.log2(n))
    labels, probability, embedding, pca, scaler = fit_hdbscan(
        matrix, primary_mcs, primary_ms
    )

    model_rows: list[dict[str, Any]] = []
    sensitivity_labels: dict[str, np.ndarray] = {}
    for representation in ("extreme", "novelty", "full_curve"):
        candidate_matrix, _, _, _ = build_features(curves, representation)
        settings = (
            [(primary_mcs, primary_ms)]
            if representation != "extreme"
            else [
                (mcs, ms)
                for mcs in (16, primary_mcs, 32)
                for ms in (8, primary_ms, 10)
            ]
        )
        for mcs, ms in settings:
            candidate, _, candidate_embedding, _, _ = fit_hdbscan(
                candidate_matrix, mcs, ms
            )
            key = f"hdb_{representation}_mcs{mcs}_ms{ms}"
            sensitivity_labels[key] = candidate
            model_rows.append(
                {
                    "method": "HDBSCAN",
                    "representation": representation,
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "selected_primary": representation == "extreme"
                    and mcs == primary_mcs
                    and ms == primary_ms,
                    **cluster_summary(candidate, candidate_embedding),
                }
            )

    optics = OPTICS(
        min_samples=primary_ms,
        min_cluster_size=primary_mcs,
        xi=0.05,
        cluster_method="xi",
        n_jobs=-1,
    ).fit_predict(embedding)
    optics = canonicalize(optics, embedding)
    model_rows.append(
        {
            "method": "OPTICS-Xi",
            "representation": "extreme",
            "min_cluster_size": primary_mcs,
            "min_samples": primary_ms,
            "selected_primary": False,
            **cluster_summary(optics, embedding),
        }
    )

    bic_rows = []
    fitted_gmm: dict[int, GaussianMixture] = {}
    for components in range(1, 13):
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            n_init=5,
            random_state=args.seed,
            reg_covar=1e-5,
        ).fit(embedding)
        fitted_gmm[components] = model
        bic_rows.append({"components": components, "bic": float(model.bic(embedding))})
    selected_components = min(bic_rows, key=lambda row: row["bic"])["components"]
    gmm = fitted_gmm[int(selected_components)]
    gmm_labels = canonicalize(gmm.predict(embedding), embedding)
    model_rows.append(
        {
            "method": "diagonal-GMM-BIC",
            "representation": "extreme",
            "min_cluster_size": np.nan,
            "min_samples": np.nan,
            "selected_primary": False,
            **cluster_summary(gmm_labels, embedding),
        }
    )

    rng = np.random.default_rng(args.seed)
    stability_rows = []
    for repeat in range(args.stability_repeats):
        jittered = matrix + rng.normal(0.0, 0.01, size=matrix.shape)
        jitter_labels, _, jitter_embedding, _, _ = fit_hdbscan(
            jittered, primary_mcs, primary_ms
        )
        stability_rows.append(
            {
                "test": "jitter_1pct",
                "repeat": repeat,
                "ari_vs_primary": float(adjusted_rand_score(labels, jitter_labels)),
                **cluster_summary(jitter_labels, jitter_embedding),
            }
        )

        feature_index = np.sort(
            rng.choice(matrix.shape[1], size=math.ceil(0.8 * matrix.shape[1]), replace=False)
        )
        subset_labels, _, subset_embedding, _, _ = fit_hdbscan(
            matrix[:, feature_index], primary_mcs, primary_ms
        )
        stability_rows.append(
            {
                "test": "feature_subsample_80pct",
                "repeat": repeat,
                "ari_vs_primary": float(adjusted_rand_score(labels, subset_labels)),
                **cluster_summary(subset_labels, subset_embedding),
            }
        )

        null_curves = permute_curves(curves, group, rng)
        null_matrix, _, _, _ = build_features(null_curves, "extreme")
        null_labels, _, null_embedding, _, _ = fit_hdbscan(
            null_matrix, primary_mcs, primary_ms
        )
        stability_rows.append(
            {
                "test": "within_pool_time_identity_permutation",
                "repeat": repeat,
                "ari_vs_primary": float(adjusted_rand_score(labels, null_labels)),
                **cluster_summary(null_labels, null_embedding),
            }
        )

    primary = cluster_summary(labels, embedding)
    positive = positive_control(n, embedding.shape[1], primary_mcs, primary_ms, args.seed + 1)
    assignments = pd.DataFrame(
        {
            "episode_id": episode,
            "group": group,
            "moe_anchor_query": anchor + QUERY_START,
            "moe_anchor_salience": salience,
            "hdbscan_cluster": labels,
            "hdbscan_probability": probability,
            "optics_cluster": optics,
            "gmm_bic_cluster": gmm_labels,
        }
    )
    assignments.to_csv(args.output / "unlabeled_cluster_assignments.csv", index=False)
    pd.DataFrame(model_rows).to_csv(args.output / "model_selection.csv", index=False)
    pd.DataFrame(bic_rows).to_csv(args.output / "gmm_bic.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(args.output / "stability_sentinels.csv", index=False)
    np.savez_compressed(
        args.output / "unlabeled_embedding.npz",
        episode=episode,
        group=group,
        curves=curves.astype(np.float32),
        feature_matrix=matrix.astype(np.float32),
        feature_names=np.asarray(feature_names),
        anchor_query=(anchor + QUERY_START).astype(np.int16),
        anchor_salience=salience.astype(np.float32),
        embedding=embedding.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        hdbscan_labels=labels.astype(np.int16),
        hdbscan_probability=probability.astype(np.float32),
        optics_labels=optics.astype(np.int16),
        gmm_bic_labels=gmm_labels.astype(np.int16),
        **{key: value.astype(np.int16) for key, value in sensitivity_labels.items()},
    )
    artifact_names = (
        "unlabeled_cluster_assignments.csv",
        "model_selection.csv",
        "gmm_bic.csv",
        "stability_sentinels.csv",
        "unlabeled_embedding.npz",
    )
    manifest = {
        "schema": "himoe.moe_peak_unsupervised_cluster.v1",
        "created": "2026-09-03",
        "training": False,
        "labels_read": [],
        "outcomes_read": [],
        "source": str(args.scores),
        "source_sha256": sha256(args.scores),
        "protocol_sha256": sha256(PROTOCOL),
        "code_sha256": sha256(Path(__file__)),
        "episodes": n,
        "init_pools": len(np.unique(group)),
        "query_window": [QUERY_START, QUERY_STOP - 1],
        "components": list(COMPONENTS),
        "raw_dimensions": matrix.shape[1],
        "pca_dimensions": embedding.shape[1],
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
        "primary": {
            "algorithm": "HDBSCAN",
            "representation": "extreme_anchor_radius3",
            "min_cluster_size": primary_mcs,
            "min_samples": primary_ms,
            **primary,
        },
        "gmm_bic_selected_components": int(selected_components),
        "positive_control": positive,
        "stability_repeats": args.stability_repeats,
        "artifact_sha256": {
            name: sha256(args.output / name) for name in artifact_names
        },
    }
    (args.output / "unlabeled_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plain(manifest), indent=2))


if __name__ == "__main__":
    main()
