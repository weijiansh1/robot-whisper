"""Do the 256 (layer, expert) cells form eight layer blobs or cross-layer modules?

Descriptive only. Nothing here is a detector and no failure label is touched;
`success` and `init_state_id < 30` merely select the train split this repository
always fits on.

The pipeline: per-layer CLR on the router load, a 256x256 cell-by-cell
correlation across queries, three 2D embeddings of the cells, a data-driven
partition, and three nulls - a cell-label permutation for the layer partition, a
per-layer circular shift for cross-layer coupling, and a per-cell shift with
re-closure so that within-layer and cross-layer blocks share a baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from moe_grammar.cell_coactivation import (
    CELL_FRONT,
    CELL_LAYER,
    LAYER_NAMES,
    N_CELLS,
    N_EXPERTS,
    N_LAYERS,
    block_means,
    cell_names,
    cell_shift_closed_surrogate,
    classical_mds,
    closure_corrected,
    clr_matrix,
    correlation,
    coupling_distance,
    episode_differences,
    group_center,
    layer_shift_surrogate,
    load_cell_loads,
    modularity,
    neighbour_overlap,
    procrustes_disparity,
    signed_distance,
    split_half_reliability,
    sqrt_matrix,
    standardize,
    top_edge_graph,
    upper_cross_mask,
)

CHANNELS = ("action", "state")
THRESHOLD_GRID = np.round(np.arange(0.04, 0.90, 0.005), 4)
NAMES = cell_names()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-dir", type=Path, default=Path("artifacts/layer-expert-load"))
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results-cell-coactivation"))
    parser.add_argument("--max-init-state-id", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th train query")
    parser.add_argument("--shift-surrogates", type=int, default=200)
    parser.add_argument("--closure-surrogates", type=int, default=40)
    parser.add_argument("--difference-surrogates", type=int, default=50)
    parser.add_argument("--louvain-null-surrogates", type=int, default=40)
    parser.add_argument("--label-permutations", type=int, default=2000)
    parser.add_argument("--louvain-seeds", type=int, default=40)
    parser.add_argument("--tsne-perplexity", type=float, default=20.0)
    parser.add_argument("--sparse-density", type=float, default=0.05)
    parser.add_argument("--fdr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="redraw the figure from an existing summary.json and matrices.npz",
    )
    return parser.parse_args()


def redraw(output_dir: Path) -> None:
    results = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    stored = np.load(output_dir / "matrices.npz", allow_pickle=False)
    arrays: dict[str, dict[str, np.ndarray]] = {channel: {} for channel in CHANNELS}
    for key in stored.files:
        if "__" not in key:
            continue
        channel, name = key.split("__", 1)
        if channel in arrays:
            arrays[channel][name] = stored[key]
    make_figure(results, arrays, output_dir / "cell_coactivation.png")


def louvain_partition(weights: np.ndarray, resolution: float, seed: int) -> np.ndarray:
    import networkx as nx

    graph = np.array(weights, dtype=np.float64, copy=True)
    np.fill_diagonal(graph, 0.0)
    communities = nx.community.louvain_communities(
        nx.from_numpy_array(graph), weight="weight", resolution=resolution, seed=seed
    )
    labels = np.zeros(len(graph), dtype=np.int64)
    for index, community in enumerate(sorted(communities, key=len, reverse=True)):
        labels[list(community)] = index
    return labels


def consensus_partition(
    weights: np.ndarray, resolution: float, seeds: int, rng: np.random.Generator
) -> tuple[np.ndarray, float, list[int]]:
    """Louvain over many seeds, then cut the co-assignment dendrogram at the modal k."""

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    partitions = [
        louvain_partition(weights, resolution, int(rng.integers(0, 2**31 - 1)))
        for _ in range(seeds)
    ]
    coassign = np.zeros((N_CELLS, N_CELLS))
    for labels in partitions:
        coassign += (labels[:, None] == labels[None, :]).astype(np.float64)
    coassign /= len(partitions)

    pairwise = [
        adjusted_rand_score(partitions[i], partitions[j])
        for i in range(len(partitions))
        for j in range(i + 1, len(partitions))
    ]
    modal_k = int(np.argmax(np.bincount([len(np.unique(p)) for p in partitions])))

    distance = 1.0 - coassign
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    consensus = fcluster(tree, modal_k, criterion="maxclust") - 1
    sizes = np.bincount(consensus)
    remap = np.zeros(len(sizes), dtype=np.int64)
    remap[np.argsort(sizes)[::-1]] = np.arange(len(sizes))
    consensus = remap[consensus]
    return consensus, float(np.mean(pairwise)), [int(x) for x in np.bincount(consensus)]


def partition_scores(distance: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> dict:
    unique = len(np.unique(labels))
    silhouette = float("nan")
    if unique > 1:
        silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
    return {
        "n_clusters": unique,
        "silhouette": silhouette,
        "modularity": modularity(weights, labels),
    }


def label_permutation_null(
    distance: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
    include_silhouette: bool = True,
) -> dict:
    """Shuffle which cell carries which label, keeping the group sizes fixed."""

    silhouettes = np.empty(permutations)
    modularities = np.empty(permutations)
    for index in range(permutations):
        shuffled = rng.permutation(labels)
        if include_silhouette:
            silhouettes[index] = silhouette_score(distance, shuffled, metric="precomputed")
        modularities[index] = modularity(weights, shuffled)
    observed = partition_scores(distance, weights, labels)
    draw_sets = [("modularity", modularities)]
    if include_silhouette:
        draw_sets.insert(0, ("silhouette", silhouettes))
    else:
        observed.pop("silhouette")
    out: dict[str, Any] = dict(observed)
    for name, draws in draw_sets:
        value = observed[name]
        mean = float(draws.mean())
        std = float(draws.std(ddof=1))
        out[f"{name}_null_mean"] = mean
        out[f"{name}_null_std"] = std
        out[f"{name}_z"] = float((value - mean) / std) if std > 0 else float("nan")
        out[f"{name}_p_two_sided"] = float(
            (np.sum(np.abs(draws - mean) >= abs(value - mean)) + 1) / (permutations + 1)
        )
    return out


def front_back_separation(embedding: np.ndarray) -> float:
    """How well one linear direction in the 2D layout splits front cells from back cells."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    scaled = (embedding - embedding.mean(0)) / (embedding.std(0) + 1e-12)
    model = LogisticRegression(max_iter=2000).fit(scaled, CELL_FRONT)
    return float(roc_auc_score(CELL_FRONT, model.decision_function(scaled)))


def coupling_summary(excess: np.ndarray) -> dict:
    """Average excess coupling inside the front stack, inside the back stack, and between."""

    diagonal = np.eye(N_LAYERS, dtype=bool)
    front = np.zeros((N_LAYERS, N_LAYERS), dtype=bool)
    front[:4, :4] = True
    back = np.zeros((N_LAYERS, N_LAYERS), dtype=bool)
    back[4:, 4:] = True
    between = ~(front | back)
    return {
        "front_within_layer": float(excess[front & diagonal].mean()),
        "front_cross_layer": float(excess[front & ~diagonal].mean()),
        "back_within_layer": float(excess[back & diagonal].mean()),
        "back_cross_layer": float(excess[back & ~diagonal].mean()),
        "front_to_back": float(excess[between].mean()),
    }


def significant_edges(matrix: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(threshold):
        return np.array([], dtype=int), np.array([], dtype=int)
    rows, cols = np.where(np.triu(np.abs(matrix), 1) >= threshold)
    keep = CELL_LAYER[rows] != CELL_LAYER[cols]
    return rows[keep], cols[keep]


def edge_composition(rows: np.ndarray, cols: np.ndarray) -> dict:
    left = CELL_LAYER[rows] < 4
    right = CELL_LAYER[cols] < 4
    return {
        "front_front": int(np.sum(left & right)),
        "front_back": int(np.sum(left != right)),
        "back_back": int(np.sum(~left & ~right)),
    }


def run_channel(
    channel: str,
    loads: np.ndarray,
    run_index: np.ndarray,
    episode_index: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    from sklearn.manifold import SpectralEmbedding, TSNE

    report: dict[str, Any] = {"channel": channel, "n_queries": int(len(loads))}
    arrays: dict[str, np.ndarray] = {}
    cross = upper_cross_mask()
    within = np.triu(CELL_LAYER[:, None] == CELL_LAYER[None, :], 1)
    off_diagonal = ~np.eye(N_CELLS, dtype=bool)

    clr = clr_matrix(loads)
    standardized = standardize(clr)
    matrix = correlation(standardized)
    arrays["correlation_clr"] = matrix

    # ---- transform robustness --------------------------------------------------
    hellinger = correlation(standardize(sqrt_matrix(loads)))
    arrays["correlation_sqrt"] = hellinger
    report["transform_agreement_clr_vs_sqrt"] = {
        "pearson_offdiag": float(np.corrcoef(matrix[off_diagonal], hellinger[off_diagonal])[0, 1]),
        "cross_layer_mean_abs_clr": float(np.abs(matrix[cross]).mean()),
        "cross_layer_mean_abs_sqrt": float(np.abs(hellinger[cross]).mean()),
        "block_abs_sqrt": block_means(hellinger, absolute=True).tolist(),
    }

    # ---- how much of the matrix is reproducible at all? ------------------------
    report["split_half_reliability"] = split_half_reliability(clr, episode_index, rng)

    # ---- is the coupling only task identity or episode phase? ------------------
    conditioned = {}
    for name, groups in (("task", run_index), ("episode", episode_index)):
        centred = correlation(standardize(group_center(clr, groups)))
        arrays[f"correlation_clr_{name}_centred"] = centred
        conditioned[name] = {
            "cross_layer_mean_abs": float(np.abs(centred[cross]).mean()),
            "within_layer_mean_r": float(centred[within].mean()),
            "block_abs": block_means(centred, absolute=True).tolist(),
            "pearson_offdiag_vs_pooled": float(
                np.corrcoef(matrix[off_diagonal], centred[off_diagonal])[0, 1]
            ),
        }
    report["conditioned"] = conditioned

    report["raw_summary"] = {
        "within_layer_mean_r": float(matrix[within].mean()),
        "within_layer_mean_abs_r": float(np.abs(matrix[within]).mean()),
        "cross_layer_mean_r": float(matrix[cross].mean()),
        "cross_layer_mean_abs_r": float(np.abs(matrix[cross]).mean()),
        "cross_layer_max_r": float(matrix[cross].max()),
        "cross_layer_min_r": float(matrix[cross].min()),
        "clr_std_by_layer": [
            float(x) for x in clr.std(axis=0).reshape(N_LAYERS, N_EXPERTS).mean(axis=1)
        ],
        "mean_load_by_layer": [
            float(x) for x in loads.reshape(-1, N_LAYERS, N_EXPERTS).mean(axis=0).max(axis=1)
        ],
    }
    report["block_abs"] = block_means(matrix, absolute=True).tolist()
    report["block_signed"] = block_means(matrix, absolute=False).tolist()

    # ---- null 2: per-layer circular shift --------------------------------------
    # The chance level for |r| is far above zero because queries are massively
    # serially dependent, and it differs per layer pair because front cells drift
    # much more slowly than back cells. Every comparison below is therefore made
    # against a block-specific baseline, never against zero.
    started = time.time()
    rows_cross, cols_cross = np.where(cross)
    left_front = CELL_LAYER[rows_cross] < 4
    right_front = CELL_LAYER[cols_cross] < 4
    class_id = np.where(
        left_front & right_front, 0, np.where(left_front != right_front, 1, 2)
    )
    class_names = ("front_front", "front_back", "back_back")

    exceed = np.zeros(len(THRESHOLD_GRID))
    exceed_class = np.zeros((3, len(THRESHOLD_GRID)))
    null_mean_abs = np.empty(args.shift_surrogates)
    null_max_abs = np.empty(args.shift_surrogates)
    null_block = np.zeros((N_LAYERS, N_LAYERS))
    null_block_sq = np.zeros((N_LAYERS, N_LAYERS))
    for index in range(args.shift_surrogates):
        surrogate = correlation(layer_shift_surrogate(standardized, rng))
        values = np.abs(surrogate[cross])
        null_mean_abs[index] = values.mean()
        null_max_abs[index] = values.max()
        block = block_means(surrogate, absolute=True)
        null_block += block
        null_block_sq += block**2
        above = values[:, None] >= THRESHOLD_GRID[None, :]
        exceed += above.sum(axis=0)
        for code in range(3):
            exceed_class[code] += above[class_id == code].sum(axis=0)
    null_block /= args.shift_surrogates
    null_block_sd = np.sqrt(
        np.maximum(null_block_sq / args.shift_surrogates - null_block**2, 0.0)
    )
    exceed /= args.shift_surrogates
    exceed_class /= args.shift_surrogates
    arrays["shift_null_block_abs"] = null_block

    observed_abs = np.abs(matrix[cross])
    observed_counts = (observed_abs[:, None] >= THRESHOLD_GRID[None, :]).sum(axis=0)
    ratio = np.where(observed_counts > 0, exceed / np.maximum(observed_counts, 1), 1.0)
    usable = np.flatnonzero((ratio <= args.fdr) & (observed_counts > 0))
    threshold = float(THRESHOLD_GRID[usable[0]]) if usable.size else float("nan")
    n_significant = int(observed_counts[usable[0]]) if usable.size else 0

    # A pooled threshold is set by the noisiest block class, which is unfair to the
    # back stack. Repeat the FDR inside each class against its own null.
    diagonal_mask = np.eye(N_LAYERS, dtype=bool)
    front_block = np.zeros((N_LAYERS, N_LAYERS), dtype=bool)
    front_block[:4, :4] = True
    back_block = np.zeros((N_LAYERS, N_LAYERS), dtype=bool)
    back_block[4:, 4:] = True
    class_blocks = (
        front_block & ~diagonal_mask,
        ~(front_block | back_block),
        back_block & ~diagonal_mask,
    )
    per_class: dict[str, Any] = {}
    for code, name in enumerate(class_names):
        counts = (observed_abs[class_id == code][:, None] >= THRESHOLD_GRID[None, :]).sum(axis=0)
        class_ratio = np.where(counts > 0, exceed_class[code] / np.maximum(counts, 1), 1.0)
        ok = np.flatnonzero((class_ratio <= args.fdr) & (counts > 0))
        total = int(np.sum(class_id == code))
        per_class[name] = {
            "n_pairs": total,
            "observed_mean_abs_r": float(observed_abs[class_id == code].mean()),
            "null_mean_abs_r": float(null_block[class_blocks[code]].mean()),
            "threshold": float(THRESHOLD_GRID[ok[0]]) if ok.size else float("nan"),
            "n_significant": int(counts[ok[0]]) if ok.size else 0,
            "share_significant": float(counts[ok[0]] / total) if ok.size else 0.0,
        }

    report["shift_null"] = {
        "n_surrogates": args.shift_surrogates,
        "cross_layer_mean_abs_r": float(null_mean_abs.mean()),
        "cross_layer_mean_abs_r_std": float(null_mean_abs.std(ddof=1)),
        "cross_layer_max_abs_r": float(null_max_abs.mean()),
        "cross_layer_max_abs_r_std": float(null_max_abs.std(ddof=1)),
        "mean_abs_z": float(
            (np.abs(matrix[cross]).mean() - null_mean_abs.mean()) / null_mean_abs.std(ddof=1)
        ),
        "effective_sample_size": float(2.0 / (np.pi * null_mean_abs.mean() ** 2)),
        "block_abs": null_block.tolist(),
        "edge_threshold_at_fdr": threshold,
        "target_fdr": args.fdr,
        "n_significant_cross_layer_edges": n_significant,
        "n_cross_layer_edges": int(cross.sum()),
        "per_block_class_fdr": per_class,
        "seconds": round(time.time() - started, 1),
    }

    excess = block_means(matrix, absolute=True) - null_block
    with np.errstate(divide="ignore", invalid="ignore"):
        excess_z = np.where(null_block_sd > 0, excess / null_block_sd, np.nan)
    report["block_abs_excess_shift"] = excess.tolist()
    report["block_abs_excess_shift_z"] = np.nan_to_num(excess_z, nan=0.0).tolist()
    report["coupling_summary_excess_shift"] = coupling_summary(excess)
    arrays["block_abs_excess_shift"] = excess

    # ---- timescale control: correlate query-to-query differences ---------------
    started = time.time()
    differenced = standardize(episode_differences(clr, episode_index))
    difference_matrix = correlation(differenced)
    arrays["correlation_clr_differenced"] = difference_matrix
    difference_null = np.zeros((N_LAYERS, N_LAYERS))
    for _ in range(args.difference_surrogates):
        difference_null += block_means(
            correlation(layer_shift_surrogate(differenced, rng)), absolute=True
        )
    difference_null /= args.difference_surrogates
    difference_excess = block_means(difference_matrix, absolute=True) - difference_null
    difference_weights = np.abs(difference_matrix).copy()
    np.fill_diagonal(difference_weights, 0.0)
    difference_distance = coupling_distance(difference_matrix)
    report["differenced_view"] = {
        "n_rows": int(len(differenced)),
        "n_surrogates": args.difference_surrogates,
        "block_abs": block_means(difference_matrix, absolute=True).tolist(),
        "shift_null_block_abs": difference_null.tolist(),
        "block_abs_excess": difference_excess.tolist(),
        "coupling_summary_excess": coupling_summary(difference_excess),
        "layer": label_permutation_null(
            difference_distance, difference_weights, CELL_LAYER, args.label_permutations, rng
        ),
        "front_back": label_permutation_null(
            difference_distance, difference_weights, CELL_FRONT, args.label_permutations, rng
        ),
        "seconds": round(time.time() - started, 1),
    }

    # ---- null 3: per-cell shift with re-closure --------------------------------
    started = time.time()
    closure_block = np.zeros((N_LAYERS, N_LAYERS))
    for _ in range(args.closure_surrogates):
        closure_block += block_means(
            correlation(cell_shift_closed_surrogate(loads, rng)), absolute=True
        )
    closure_block /= args.closure_surrogates
    arrays["closure_null_block_abs"] = closure_block
    closure_excess = block_means(matrix, absolute=True) - closure_block
    report["closure_null"] = {
        "n_surrogates": args.closure_surrogates,
        "block_abs": closure_block.tolist(),
        "within_layer_baseline": float(np.diag(closure_block).mean()),
        "cross_layer_baseline": float(closure_block[~np.eye(N_LAYERS, dtype=bool)].mean()),
        "analytic_closure_offset": -1.0 / (N_EXPERTS - 1),
        "seconds": round(time.time() - started, 1),
    }
    report["block_abs_excess_closure"] = closure_excess.tolist()
    report["coupling_summary_excess_closure"] = coupling_summary(closure_excess)
    arrays["block_abs_excess_closure"] = closure_excess

    # ---- the significant cross-layer coupling network --------------------------
    rows, cols = significant_edges(matrix, threshold)
    order = np.argsort(np.abs(matrix[rows, cols]))[::-1] if rows.size else np.array([], dtype=int)
    report["significant_cross_layer_edges_top50"] = [
        {"a": NAMES[rows[i]], "b": NAMES[cols[i]], "r": round(float(matrix[rows[i], cols[i]]), 4)}
        for i in order[:50]
    ]
    if rows.size:
        report["significant_edge_composition"] = edge_composition(rows, cols)
        report["significant_edge_composition_expected_share"] = {
            "front_front": 6 / 28,
            "front_back": 16 / 28,
            "back_back": 6 / 28,
        }
        report["significant_edge_sign"] = {
            "positive": int(np.sum(matrix[rows, cols] > 0)),
            "negative": int(np.sum(matrix[rows, cols] < 0)),
        }
    degree = np.zeros(N_CELLS)
    np.add.at(degree, rows, 1.0)
    np.add.at(degree, cols, 1.0)
    arrays["cross_layer_degree"] = degree
    report["cross_layer_degree_by_layer"] = [
        float(x) for x in degree.reshape(N_LAYERS, N_EXPERTS).sum(axis=1)
    ]
    report["hub_cells"] = [
        {"cell": NAMES[i], "significant_cross_layer_edges": int(degree[i])}
        for i in np.argsort(degree)[::-1][:20]
    ]

    # ---- embeddings -------------------------------------------------------------
    distance = coupling_distance(matrix)
    weights = np.abs(matrix).copy()
    np.fill_diagonal(weights, 0.0)
    arrays["coupling_distance"] = distance

    mds = classical_mds(distance, 2)
    spectral = SpectralEmbedding(
        n_components=2, affinity="precomputed", random_state=args.seed
    ).fit_transform(weights)
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        init=(mds / (np.abs(mds).max() + 1e-12)) * 1e-4,
        perplexity=args.tsne_perplexity,
        random_state=args.seed,
        max_iter=2000,
    ).fit_transform(distance)
    arrays["embed_mds"] = mds
    arrays["embed_tsne"] = tsne
    arrays["embed_spectral"] = spectral
    agreement = {
        "procrustes_mds_vs_tsne": procrustes_disparity(mds, tsne),
        "procrustes_mds_vs_spectral": procrustes_disparity(mds, spectral),
        "procrustes_tsne_vs_spectral": procrustes_disparity(tsne, spectral),
    }
    for name, embedding in (("mds", mds), ("tsne", tsne), ("spectral", spectral)):
        agreement[f"knn10_overlap_{name}"] = neighbour_overlap(distance, embedding, 10)
        agreement[f"front_back_auc_{name}"] = front_back_separation(embedding)
    report["embedding_agreement"] = agreement

    # ---- layer partition versus a data-driven partition -------------------------
    sweep = []
    for resolution in [round(float(x), 2) for x in np.arange(0.6, 2.01, 0.1)]:
        labels = louvain_partition(weights, resolution, args.seed)
        sweep.append(
            {
                "resolution": resolution,
                "n_clusters": int(len(np.unique(labels))),
                "modularity": modularity(weights, labels),
                "ari_layer": float(adjusted_rand_score(CELL_LAYER, labels)),
                "ari_front_back": float(adjusted_rand_score(CELL_FRONT, labels)),
            }
        )
    report["louvain_resolution_sweep"] = sweep
    best_resolution = max(sweep, key=lambda item: item["modularity"])["resolution"]
    consensus, stability, sizes = consensus_partition(
        weights, best_resolution, args.louvain_seeds, rng
    )
    arrays["consensus_labels"] = consensus
    report["data_driven_partition"] = {
        "resolution": best_resolution,
        "n_clusters": len(sizes),
        "sizes": sizes,
        "mean_pairwise_ari_across_seeds": stability,
        "ari_layer": float(adjusted_rand_score(CELL_LAYER, consensus)),
        "ari_front_back": float(adjusted_rand_score(CELL_FRONT, consensus)),
        "composition": [
            {
                LAYER_NAMES[layer]: int(count)
                for layer, count in enumerate(
                    np.bincount(CELL_LAYER[consensus == cluster], minlength=N_LAYERS)
                )
                if count
            }
            for cluster in range(len(sizes))
        ],
        "members": [
            [NAMES[i] for i in np.flatnonzero(consensus == cluster)] for cluster in range(len(sizes))
        ],
    }

    report["partition_tests"] = {
        "layer": label_permutation_null(
            distance, weights, CELL_LAYER, args.label_permutations, rng
        ),
        "front_back": label_permutation_null(
            distance, weights, CELL_FRONT, args.label_permutations, rng
        ),
        "data_driven": partition_scores(distance, weights, consensus),
    }

    # Same tests after undoing the mechanical within-layer negativity of the CLR.
    corrected = closure_corrected(matrix)
    corrected_distance = coupling_distance(corrected)
    corrected_weights = np.abs(corrected).copy()
    np.fill_diagonal(corrected_weights, 0.0)
    report["closure_corrected_partition_tests"] = {
        "note": "within-layer off-diagonal r shifted by +1/31 before |r| and 1-|r|",
        "layer": label_permutation_null(
            corrected_distance, corrected_weights, CELL_LAYER, args.label_permutations, rng
        ),
        "front_back": label_permutation_null(
            corrected_distance, corrected_weights, CELL_FRONT, args.label_permutations, rng
        ),
        "sparse_modularity_layer": modularity(
            top_edge_graph(corrected, args.sparse_density), CELL_LAYER
        ),
        "sparse_modularity_front_back": modularity(
            top_edge_graph(corrected, args.sparse_density), CELL_FRONT
        ),
    }

    # ---- density-matched sparse graph -------------------------------------------
    sparse = top_edge_graph(matrix, args.sparse_density)
    arrays["sparse_graph"] = sparse
    sparse_consensus, sparse_stability, sparse_sizes = consensus_partition(
        sparse, 1.0, args.louvain_seeds, rng
    )
    arrays["sparse_consensus_labels"] = sparse_consensus
    upper_sparse = np.triu(sparse, 1)
    block_edges = np.zeros((N_LAYERS, N_LAYERS))
    rows_s, cols_s = np.nonzero(upper_sparse)
    np.add.at(block_edges, (CELL_LAYER[rows_s], CELL_LAYER[cols_s]), 1.0)
    block_edges = block_edges + block_edges.T
    report["sparse_graph"] = {
        "density": args.sparse_density,
        "n_edges": int(upper_sparse.sum()),
        "share_of_edges_by_block": (block_edges / max(block_edges.sum(), 1.0)).tolist(),
        "modularity_layer": modularity(sparse, CELL_LAYER),
        "modularity_front_back": modularity(sparse, CELL_FRONT),
        "modularity_louvain": modularity(sparse, sparse_consensus),
        "louvain_n_clusters": len(sparse_sizes),
        "louvain_sizes": sparse_sizes,
        "louvain_mean_pairwise_ari_across_seeds": sparse_stability,
        "louvain_ari_layer": float(adjusted_rand_score(CELL_LAYER, sparse_consensus)),
        "louvain_ari_front_back": float(adjusted_rand_score(CELL_FRONT, sparse_consensus)),
        "louvain_composition": [
            {
                LAYER_NAMES[layer]: int(count)
                for layer, count in enumerate(
                    np.bincount(CELL_LAYER[sparse_consensus == cluster], minlength=N_LAYERS)
                )
                if count
            }
            for cluster in range(len(sparse_sizes))
        ],
        "louvain_members": [
            [NAMES[i] for i in np.flatnonzero(sparse_consensus == cluster)]
            for cluster in range(len(sparse_sizes))
        ],
    }
    report["sparse_partition_tests"] = {
        "layer": label_permutation_null(
            distance, sparse, CELL_LAYER, args.label_permutations, rng, include_silhouette=False
        ),
        "front_back": label_permutation_null(
            distance, sparse, CELL_FRONT, args.label_permutations, rng, include_silhouette=False
        ),
    }

    # Louvain maximises modularity by construction, so shuffling its labels cannot
    # test it. The honest null is Louvain re-run on shift surrogate graphs, at the
    # same edge density so that the graphs are comparable.
    started = time.time()
    dense_q: list[float] = []
    sparse_q: list[float] = []
    sparse_k: list[int] = []
    for _ in range(args.louvain_null_surrogates):
        surrogate = correlation(layer_shift_surrogate(standardized, rng))
        dense_surrogate = np.abs(surrogate)
        np.fill_diagonal(dense_surrogate, 0.0)
        dense_q.append(
            modularity(
                dense_surrogate,
                louvain_partition(dense_surrogate, best_resolution, int(rng.integers(0, 2**31 - 1))),
            )
        )
        sparse_surrogate = top_edge_graph(surrogate, args.sparse_density)
        labels = louvain_partition(sparse_surrogate, 1.0, int(rng.integers(0, 2**31 - 1)))
        sparse_q.append(modularity(sparse_surrogate, labels))
        sparse_k.append(int(len(np.unique(labels))))
    observed_q = report["partition_tests"]["data_driven"]["modularity"]
    report["partition_tests"]["data_driven_shift_null"] = {
        "n_surrogates": len(dense_q),
        "modularity_null_mean": float(np.mean(dense_q)),
        "modularity_null_std": float(np.std(dense_q, ddof=1)),
        "modularity_z": float((observed_q - np.mean(dense_q)) / np.std(dense_q, ddof=1)),
        "caveat": "dense weighted graphs: an all-noise surrogate can score high, read the sparse test",
        "seconds": round(time.time() - started, 1),
    }
    observed_sparse_q = report["sparse_graph"]["modularity_louvain"]
    report["sparse_partition_tests"]["data_driven_shift_null"] = {
        "n_surrogates": len(sparse_q),
        "modularity_null_mean": float(np.mean(sparse_q)),
        "modularity_null_std": float(np.std(sparse_q, ddof=1)),
        "modularity_z": float((observed_sparse_q - np.mean(sparse_q)) / np.std(sparse_q, ddof=1)),
        "n_clusters_null_median": float(np.median(sparse_k)),
    }

    # ---- signed (positive co-activation) view -----------------------------------
    positive = np.maximum(matrix, 0.0)
    np.fill_diagonal(positive, 0.0)
    signed_labels = louvain_partition(positive, 1.0, args.seed)
    report["signed_view"] = {
        "note": "positive-r graph; the CLR closure biases within-layer edges down by about 1/31",
        "modularity_layer": modularity(positive, CELL_LAYER),
        "modularity_front_back": modularity(positive, CELL_FRONT),
        "modularity_louvain": modularity(positive, signed_labels),
        "louvain_n_clusters": int(len(np.unique(signed_labels))),
        "louvain_ari_layer": float(adjusted_rand_score(CELL_LAYER, signed_labels)),
        "louvain_ari_front_back": float(adjusted_rand_score(CELL_FRONT, signed_labels)),
        "silhouette_layer_signed_distance": float(
            silhouette_score(signed_distance(matrix), CELL_LAYER, metric="precomputed")
        ),
    }
    arrays["signed_louvain_labels"] = signed_labels
    return report, arrays


def _scatter_by_layer(ax, embedding: np.ndarray, palette, markers, legend: bool) -> None:
    scaled = (embedding - embedding.mean(0)) / (embedding.std(0).mean() + 1e-12)
    for layer in range(N_LAYERS):
        keep = CELL_LAYER == layer
        ax.scatter(
            scaled[keep, 0],
            scaled[keep, 1],
            s=32,
            c=palette[layer],
            marker=markers[layer],
            edgecolors="k",
            linewidths=0.3,
            label=LAYER_NAMES[layer],
        )
    if legend:
        ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.9, title="layer (circle=front)")


def make_figure(results: dict, arrays: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    palette = [
        "#08306b", "#2171b5", "#4292c6", "#9ecae1",
        "#7f2704", "#d94801", "#f16913", "#fdd0a2",
    ]
    markers = ["o", "o", "o", "o", "^", "^", "^", "^"]
    figure, axes = plt.subplots(3, 3, figsize=(16.5, 15.0))

    for row, channel in enumerate(CHANNELS):
        block = arrays[channel]
        matrix = block["correlation_clr"]
        strength = np.abs(matrix) * (1.0 - np.eye(N_CELLS))
        limit = float(np.percentile(strength, 99.5))

        ax = axes[row, 0]
        image = ax.imshow(strength, cmap="magma", vmin=0.0, vmax=limit)
        for edge in range(N_EXPERTS, N_CELLS, N_EXPERTS):
            ax.axhline(edge - 0.5, color="w", lw=0.4)
            ax.axvline(edge - 0.5, color="w", lw=0.4)
        ax.axhline(4 * N_EXPERTS - 0.5, color="#00e5ff", lw=1.8)
        ax.axvline(4 * N_EXPERTS - 0.5, color="#00e5ff", lw=1.8)
        ax.set_xticks(np.arange(16, N_CELLS, N_EXPERTS))
        ax.set_xticklabels(LAYER_NAMES, fontsize=8)
        ax.set_yticks(np.arange(16, N_CELLS, N_EXPERTS))
        ax.set_yticklabels(LAYER_NAMES, fontsize=8)
        ax.set_title(f"({'ad'[row]}) {channel}_load: |r| between the 256 cells", fontsize=11)
        figure.colorbar(image, ax=ax, fraction=0.046)

        ax = axes[row, 1]
        excess = block["block_abs_excess_shift"]
        span = max(float(np.abs(excess).max()), 1e-6)
        image = ax.imshow(
            excess, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span)
        )
        for i in range(N_LAYERS):
            for j in range(N_LAYERS):
                label = "n/a" if i == j else f"{excess[i, j]:.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7)
        ax.axhline(3.5, color="k", lw=1.6)
        ax.axvline(3.5, color="k", lw=1.6)
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels(LAYER_NAMES, fontsize=8)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels(LAYER_NAMES, fontsize=8)
        ax.set_title(
            f"({'be'[row]}) {channel}_load: cross-layer coupling above chance\n"
            "mean |r| minus the shift null; the diagonal is preserved by that null",
            fontsize=10,
        )
        figure.colorbar(image, ax=ax, fraction=0.046)

        ax = axes[row, 2]
        _scatter_by_layer(ax, block["embed_mds"], palette, markers, legend=row == 0)
        auc = results[channel]["embedding_agreement"]["front_back_auc_mds"]
        ax.set_title(
            f"({'cf'[row]}) {channel}_load: classical MDS on 1-|r|  (front/back AUC {auc:.2f})",
            fontsize=11,
        )
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")

    for column, channel in enumerate(CHANNELS):
        ax = axes[2, column]
        _scatter_by_layer(ax, arrays[channel]["embed_tsne"], palette, markers, legend=False)
        auc = results[channel]["embedding_agreement"]["front_back_auc_tsne"]
        ax.set_title(
            f"({'gh'[column]}) {channel}_load: t-SNE on 1-|r|  (front/back AUC {auc:.2f})",
            fontsize=11,
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    ax = axes[2, 2]
    labels: list[str] = []
    observed: list[float] = []
    null_mean: list[float] = []
    null_std: list[float] = []
    annotations: list[str] = []
    for channel in CHANNELS:
        tests = results[channel]["partition_tests"]
        for key, pretty in (("layer", "layer (8)"), ("front_back", "front/back (2)")):
            labels.append(f"{channel[:3]} {pretty}")
            observed.append(tests[key]["modularity"])
            null_mean.append(tests[key]["modularity_null_mean"])
            null_std.append(tests[key]["modularity_null_std"])
        partition = results[channel]["data_driven_partition"]
        annotations.append(
            f"{channel}: Louvain k={partition['n_clusters']} "
            f"sizes {partition['sizes']}\n"
            f"    ARI vs layer {partition['ari_layer']:.3f}, "
            f"ARI vs front/back {partition['ari_front_back']:.3f}"
        )
    positions = np.arange(len(labels))
    ax.bar(positions - 0.2, observed, width=0.4, color="#2171b5", label="observed")
    ax.bar(
        positions + 0.2,
        null_mean,
        width=0.4,
        yerr=np.asarray(null_std) * 3.0,
        color="#bdbdbd",
        capsize=3,
        label="cell-label shuffle (mean, 3 sd)",
    )
    for index, (value, mean, std) in enumerate(zip(observed, null_mean, null_std)):
        ax.text(
            index - 0.2,
            value + 0.002,
            f"z={((value - mean) / std):.0f}",
            ha="center",
            fontsize=7,
        )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Newman modularity on the |r| graph")
    ax.set_ylim(top=max(observed) * 2.4)
    ax.set_title("(i) layer partition vs front/back partition", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    ax.text(
        0.02,
        0.82,
        "\n".join(annotations),
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#999999"},
    )

    figure.suptitle(
        "Co-activation of the 256 (layer, expert) cells: they do not form eight layer blobs.\n"
        "The split the data makes is front stack (L2-L5) vs back stack (L12-L15), and inside\n"
        "each stack a cell couples to other layers at least as strongly as to its own.",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        redraw(args.output_dir)
        print(f"Redrew the figure in {args.output_dir}", flush=True)
        return
    started = time.time()
    rng = np.random.default_rng(args.seed)

    print("Loading cell loads...", flush=True)
    data = load_cell_loads(args.load_dir, args.features_dir, args.max_init_state_id, args.stride)
    print(
        f"  {len(data.action)} train queries, {len(np.unique(data.episode_index))} episodes, "
        f"{len(data.run_keys)} run files",
        flush=True,
    )

    results: dict[str, Any] = {
        "config": {
            "load_dir": str(args.load_dir),
            "features_dir": str(args.features_dir),
            "split": f"success and init_state_id < {args.max_init_state_id}",
            "stride": args.stride,
            "n_queries": int(len(data.action)),
            "n_episodes": int(len(np.unique(data.episode_index))),
            "n_run_files": len(data.run_keys),
            "transform": "per-layer centred log-ratio (CLR)",
            "shift_surrogates": args.shift_surrogates,
            "closure_surrogates": args.closure_surrogates,
            "label_permutations": args.label_permutations,
            "louvain_seeds": args.louvain_seeds,
            "target_fdr": args.fdr,
            "seed": args.seed,
        },
        "layer_names": list(LAYER_NAMES),
    }
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for channel in CHANNELS:
        print(f"Channel {channel}...", flush=True)
        report, channel_arrays = run_channel(
            channel, data.channel(channel), data.run_index, data.episode_index, args, rng
        )
        results[channel] = report
        arrays[channel] = channel_arrays
        print(f"  done at {round(time.time() - started, 1)}s", flush=True)

    results["runtime_seconds"] = round(time.time() - started, 1)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "matrices.npz",
        cell_names=np.array(NAMES),
        cell_layer=CELL_LAYER,
        cell_expert=np.tile(np.arange(N_EXPERTS), N_LAYERS),
        **{
            f"{channel}__{key}": value
            for channel, block in arrays.items()
            for key, value in block.items()
        },
    )
    make_figure(results, arrays, args.output_dir / "cell_coactivation.png")
    print(f"Wrote {args.output_dir} in {results['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
