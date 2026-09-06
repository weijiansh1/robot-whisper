#!/usr/bin/env python3
"""What shape do the ten action tokens make, and how stable is it?

Four questions, all answered from routing tensors alone and without any outcome
label:

1. shape      -- is the configuration a curve, a blob, or something else?
2. dimension  -- how many directions does the shape spectrum actually occupy?
3. stability  -- does the same configuration reappear across tasks, seeds,
                 sampling designs and layers?
4. front/back -- does the geometry reproduce the published finding that back
                 layers carry more differentiated relative token structure?

Shape distance is always the full Procrustes disparity, so nothing here can be
explained by the cloud simply growing or shrinking.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import geometry as G


DEFAULT_OUTPUT = G.BUNDLE / "results/geometry"
FEATURES = G.FEATURE_ROOT
TOKENS = np.arange(1, 11, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sign_fixed(coordinates: np.ndarray) -> np.ndarray:
    """Flip each axis so that PC1 increases with token index."""
    out = coordinates.copy()
    orientation = np.sign(
        ((out[:, 0] - out[:, 0].mean()) * (TOKENS - TOKENS.mean())).sum()
    )
    return out * (orientation if orientation != 0 else 1.0)


def robinson(distance: np.ndarray) -> bool:
    """True if d(i, j) increases with |i - j| along every row and column."""
    for i in range(10):
        if i + 2 < 10 and not np.all(np.diff(distance[i, i + 1 :]) > 0):
            return False
        if i > 1 and not np.all(np.diff(distance[i, : i][::-1]) > 0):
            return False
    return True


def shape_rows(name: str, kernels: np.ndarray) -> list[dict[str, Any]]:
    centred = G.double_centre(kernels)
    values, coordinates = G.configuration(centred)
    distance = G.distances(kernels)
    lag = np.abs(G.OFFDIAG[0] - G.OFFDIAG[1]).astype(np.float64)
    rows = []
    for layer, layer_name in enumerate(G.LAYER_NAMES):
        share = values[layer] / values[layer].sum()
        c = sign_fixed(coordinates[layer])
        centred_token = TOKENS - TOKENS.mean()
        edge = distance[layer][G.OFFDIAG]
        gaps = np.diff(c[:, 0])
        rows.append(
            {
                "source": name,
                "layer": layer_name,
                "group": "front" if layer < 4 else "back",
                "kernel_trace": float(np.trace(kernels[layer])),
                "shape_trace": float(values[layer].sum()),
                "shape_pr": float(G.participation_ratio(values[layer])),
                "shape_erank": float(G.entropy_rank(values[layer])),
                **{f"lam{k + 1}_share": float(share[k]) for k in range(5)},
                "dim_90": int(np.searchsorted(np.cumsum(share), 0.90) + 1),
                "dim_99": int(np.searchsorted(np.cumsum(share), 0.99) + 1),
                "r_pc1_index": float(np.corrcoef(c[:, 0], TOKENS)[0, 1]),
                "r_pc2_index_sq": float(np.corrcoef(c[:, 1], centred_token**2)[0, 1]),
                "r_pc3_index_cu": float(np.corrcoef(c[:, 2], centred_token**3)[0, 1]),
                "r_dist_lag": float(np.corrcoef(edge, lag)[0, 1]),
                "spearman_dist_lag": float(spearmanr(edge, lag).statistic),
                "robinson": bool(robinson(distance[layer])),
                "stretch_max_over_min": float(edge.max() / edge.min()),
                "edge_cv": float(edge.std() / edge.mean()),
                "pc1_gap_cv": float(gaps.std() / gaps.mean()),
                "d_first_second_over_median_interior": float(
                    distance[layer][0, 1] / np.median(np.diagonal(distance[layer], 1)[1:-1])
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    templates = np.load(FEATURES / "templates.npz")
    pooled = templates["pooled_mean_kernel"]
    template_coordinates = templates["template_coordinates"]

    # ---- 1-3: shape of the pooled template and of every task template -------
    rows = shape_rows("pooled_development", pooled)
    task_kernels: dict[str, np.ndarray] = {}
    task_names: dict[str, np.ndarray] = {}
    for cohort in ("development_main", "development_extra", "external_8b"):
        archive = np.load(FEATURES / f"{cohort}_task_kernels.npz", allow_pickle=False)
        task_kernels[cohort] = archive["task_mean_kernel"]
        task_names[cohort] = archive["task_names"].astype(str)
        rows.extend(shape_rows(f"{cohort}_mean", task_kernels[cohort].mean(axis=0)))
    shape = pd.DataFrame(rows)
    shape.to_csv(args.output / "template_shape.csv", index=False)

    # per-layer lag profile of the pooled template
    distance = G.distances(pooled)
    lag_rows = []
    for layer, layer_name in enumerate(G.LAYER_NAMES):
        for k in range(1, 10):
            band = np.diagonal(distance[layer], k)
            lag_rows.append(
                {
                    "layer": layer_name,
                    "group": "front" if layer < 4 else "back",
                    "lag": k,
                    "mean_distance": float(band.mean()),
                    "min_distance": float(band.min()),
                    "max_distance": float(band.max()),
                }
            )
    pd.DataFrame(lag_rows).to_csv(args.output / "template_lag_profile.csv", index=False)

    # ---- stability of the configuration ------------------------------------
    stability: list[dict[str, Any]] = []
    dev_tasks = np.concatenate(
        [task_names["development_main"], task_names["development_extra"]]
    )
    dev_task_kernels = np.concatenate(
        [task_kernels["development_main"], task_kernels["development_extra"]], axis=0
    )
    external_tasks = task_names["external_8b"]
    shared = np.array([t for t in external_tasks if t in set(dev_tasks.tolist())])
    dev_lookup = {name: i for i, name in enumerate(dev_tasks)}
    ext_lookup = {name: i for i, name in enumerate(external_tasks)}

    # seed split inside development_main: even vs odd flow noise seed
    index = G.load_index("development_main")
    valid = index["valid"].astype(bool)
    episode_seed = index["flow_noise_seed"].astype(int)
    episode_task = index["task_index"].astype(int)
    query_seed = np.repeat(episode_seed, valid.sum(axis=1))
    query_task = np.repeat(episode_task, valid.sum(axis=1))
    n_task = len(task_names["development_main"])
    half = (query_seed % 2).astype(int)
    split_key = query_task * 2 + half
    split_kernels = G.group_mean_kernel("development_main", split_key, n_task * 2)

    for layer, layer_name in enumerate(G.LAYER_NAMES):
        pooled_template = template_coordinates[layer]
        dev_values, dev_coordinates = G.configuration(
            G.double_centre(dev_task_kernels[:, layer])
        )
        dev_trace = dev_values.sum(axis=-1)
        to_pooled = G.procrustes_disparity(dev_coordinates, dev_trace, pooled_template)
        pairwise = G.pairwise_procrustes(G.double_centre(dev_task_kernels[:, layer]))
        upper = pairwise[np.triu_indices(len(pairwise), 1)]

        ext_values, ext_coordinates = G.configuration(
            G.double_centre(task_kernels["external_8b"][:, layer])
        )
        ext_trace = ext_values.sum(axis=-1)
        cross = np.array(
            [
                G.procrustes_disparity(
                    dev_coordinates[dev_lookup[name]][None],
                    dev_trace[dev_lookup[name]][None],
                    ext_coordinates[ext_lookup[name]],
                )[0]
                for name in shared
            ]
        )

        split_values, split_coordinates = G.configuration(
            G.double_centre(split_kernels[:, layer])
        )
        split_trace = split_values.sum(axis=-1)
        seed_gap = np.array(
            [
                G.procrustes_disparity(
                    split_coordinates[2 * t][None],
                    split_trace[2 * t][None],
                    split_coordinates[2 * t + 1],
                )[0]
                for t in range(n_task)
            ]
        )

        edges_dev = G.distances(dev_task_kernels[:, layer])[:, G.OFFDIAG[0], G.OFFDIAG[1]]
        edges_ext = G.distances(task_kernels["external_8b"][:, layer])[
            :, G.OFFDIAG[0], G.OFFDIAG[1]
        ]
        pooled_edges = np.array(
            [
                spearmanr(
                    edges_dev[dev_lookup[name]], edges_ext[ext_lookup[name]]
                ).statistic
                for name in shared
            ]
        )
        stability.append(
            {
                "layer": layer_name,
                "group": "front" if layer < 4 else "back",
                "tasks_development": int(len(dev_task_kernels)),
                "tasks_shared": int(len(shared)),
                "task_to_pooled_median": float(np.median(to_pooled)),
                "task_to_pooled_p90": float(np.quantile(to_pooled, 0.90)),
                "task_to_pooled_max": float(to_pooled.max()),
                "task_pair_median": float(np.median(upper)),
                "task_pair_max": float(upper.max()),
                "dev_vs_external_same_task_median": float(np.median(cross)),
                "dev_vs_external_same_task_max": float(cross.max()),
                "seed_split_median": float(np.median(seed_gap)),
                "seed_split_max": float(seed_gap.max()),
                "spearman_distance_dev_vs_external_median": float(np.median(pooled_edges)),
                "spearman_distance_dev_vs_external_min": float(pooled_edges.min()),
            }
        )
    pd.DataFrame(stability).to_csv(args.output / "configuration_stability.csv", index=False)

    # ---- per-query dispersion around the template --------------------------
    query_rows: list[dict[str, Any]] = []
    for cohort in ("development_main", "external_8b"):
        features = np.load(FEATURES / f"{cohort}_features.npy")
        name_to_column = {name: i for i, name in enumerate(G.FEATURE_NAMES)}
        for layer, layer_name in enumerate(G.LAYER_NAMES):
            block = features[:, layer]
            record = {
                "cohort": cohort,
                "layer": layer_name,
                "group": "front" if layer < 4 else "back",
                "queries": int(len(block)),
            }
            for feature in (
                "shape_pr",
                "shape_erank",
                "lam1_share",
                "size",
                "procrustes_layer",
                "bandedness",
                "d_cv",
                "kernel_erank",
            ):
                column = block[:, name_to_column[feature]]
                record[f"{feature}_median"] = float(np.median(column))
                record[f"{feature}_p05"] = float(np.quantile(column, 0.05))
                record[f"{feature}_p95"] = float(np.quantile(column, 0.95))
                record[f"{feature}_cv"] = float(column.std() / abs(column.mean()))
            query_rows.append(record)
    pd.DataFrame(query_rows).to_csv(args.output / "query_dispersion.csv", index=False)

    # ---- how much does knowing the task buy on a single query? -------------
    # If a per-task template fits an individual query barely better than the one
    # pooled template, the geometry is task-independent and a global threshold
    # is the honest calibration.
    gain_rows: list[dict[str, Any]] = []
    for cohort in ("development_main", "external_8b"):
        cohort_index = G.load_index(cohort)
        cohort_valid = cohort_index["valid"].astype(bool)
        keys = np.repeat(cohort_index["task_index"].astype(int), cohort_valid.sum(axis=1))
        task_template = np.load(FEATURES / f"{cohort}_task_kernels.npz", allow_pickle=False)[
            "task_mean_kernel"
        ]
        packed = G.load_packed(cohort)
        pooled_disparity = np.load(FEATURES / f"{cohort}_features.npy")[
            :, :, G.FEATURE_NAMES.index("procrustes_layer")
        ]
        own = np.empty_like(pooled_disparity)
        for task_position in range(task_template.shape[0]):
            take = np.flatnonzero(keys == task_position)
            _, own_template = G.configuration(
                G.double_centre(task_template[task_position])
            )
            for start in range(0, len(take), 40000):
                rows_here = take[start : start + 40000]
                kernel = G.symmetrise(np.asarray(packed[rows_here]))
                values, coordinates = G.configuration(G.double_centre(kernel))
                trace = values.sum(axis=-1)
                for layer in range(8):
                    own[rows_here, layer] = G.procrustes_disparity(
                        coordinates[:, layer], trace[:, layer], own_template[layer]
                    )
        for layer, layer_name in enumerate(G.LAYER_NAMES):
            gain_rows.append(
                {
                    "cohort": cohort,
                    "layer": layer_name,
                    "group": "front" if layer < 4 else "back",
                    "to_pooled_template_median": float(np.median(pooled_disparity[:, layer])),
                    "to_own_task_template_median": float(np.median(own[:, layer])),
                    "task_template_gain": float(
                        1.0
                        - np.median(own[:, layer]) / np.median(pooled_disparity[:, layer])
                    ),
                }
            )
    pd.DataFrame(gain_rows).to_csv(args.output / "task_template_gain.csv", index=False)

    # ---- is the cloud an arc or a loop? ------------------------------------
    arc_rows: list[dict[str, Any]] = []
    for cohort in ("development_main", "development_extra", "external_8b"):
        kernels = task_kernels[cohort]
        distance = G.distances(kernels)
        for layer, layer_name in enumerate(G.LAYER_NAMES):
            ends_are_diameter = 0
            for task in range(len(kernels)):
                flat = distance[task, layer]
                i, j = np.unravel_index(np.argmax(flat), flat.shape)
                ends_are_diameter += int({int(i), int(j)} == {0, 9})
            arc_rows.append(
                {
                    "cohort": cohort,
                    "layer": layer_name,
                    "tasks": int(len(kernels)),
                    "diameter_is_first_last_token": ends_are_diameter,
                    "robinson_tasks": int(
                        sum(robinson(distance[task, layer]) for task in range(len(kernels)))
                    ),
                }
            )
    pd.DataFrame(arc_rows).to_csv(args.output / "arc_not_loop.csv", index=False)

    # ---- front vs back, task by task, on the per-query geometry ------------
    contrast_rows: list[dict[str, Any]] = []
    for cohort in ("development_main", "external_8b"):
        cohort_index = G.load_index(cohort)
        cohort_valid = cohort_index["valid"].astype(bool)
        keys = np.repeat(cohort_index["task_index"].astype(int), cohort_valid.sum(axis=1))
        block = np.load(FEATURES / f"{cohort}_features.npy")
        names = cohort_index["task_names"].astype(str)
        column = {name: i for i, name in enumerate(G.FEATURE_NAMES)}
        for task_position, task in enumerate(names):
            take = keys == task_position
            record = {"cohort": cohort, "task": task, "queries": int(take.sum())}
            for feature in ("procrustes_layer", "bandedness", "shape_pr", "lam1_share", "d_cv"):
                values = block[take][:, :, column[feature]]
                record[f"{feature}_front"] = float(np.median(values[:, G.FRONT]))
                record[f"{feature}_back"] = float(np.median(values[:, G.BACK]))
            contrast_rows.append(record)
    contrast = pd.DataFrame(contrast_rows)
    contrast.to_csv(args.output / "front_back_by_task.csv", index=False)
    contrast_summary = []
    for cohort, block in contrast.groupby("cohort"):
        for feature in ("procrustes_layer", "bandedness", "shape_pr", "lam1_share", "d_cv"):
            back = block[f"{feature}_back"].to_numpy()
            front = block[f"{feature}_front"].to_numpy()
            contrast_summary.append(
                {
                    "cohort": cohort,
                    "feature": feature,
                    "tasks": int(len(block)),
                    "back_median": float(np.median(back)),
                    "front_median": float(np.median(front)),
                    "tasks_back_greater": int((back > front).sum()),
                }
            )
    pd.DataFrame(contrast_summary).to_csv(
        args.output / "front_back_summary.csv", index=False
    )

    # ---- is the shape spectrum a rediscovery of an existing metric? --------
    features = np.load(FEATURES / "development_main_features.npy")
    columns = {name: i for i, name in enumerate(G.FEATURE_NAMES)}
    graph = np.load(
        G.PROJECT / "moe-hb-front-back-0905/results/layer_graphs/development_main.npz",
        allow_pickle=False,
    )
    metric_names = graph["metric_names"].astype(str).tolist()
    graph_valid = graph["valid"].astype(bool)
    overlap_rows = []
    sample = np.random.default_rng(0).choice(len(features), 40000, replace=False)
    for layer, layer_name in enumerate(G.LAYER_NAMES):
        existing = {
            name: graph["metrics"][:, :, layer, metric_names.index(name)][graph_valid]
            for name in ("conditional_effective_rank", "conditional_energy", "partial_edge_std")
        }
        for new in ("shape_pr", "shape_erank", "lam1_share", "size", "procrustes_layer", "bandedness"):
            mine = features[:, layer, columns[new]]
            for old, values in existing.items():
                overlap_rows.append(
                    {
                        "layer": layer_name,
                        "new": new,
                        "existing": old,
                        "pearson": float(np.corrcoef(mine, values)[0, 1]),
                        "spearman_40k": float(
                            spearmanr(mine[sample], values[sample]).statistic
                        ),
                    }
                )
    pd.DataFrame(overlap_rows).to_csv(args.output / "overlap_with_existing.csv", index=False)

    # the small template artefacts belong with the results, the bulk caches do not
    for name in (
        "templates.npz",
        "development_main_task_kernels.npz",
        "development_extra_task_kernels.npz",
        "external_8b_task_kernels.npz",
    ):
        shutil.copyfile(FEATURES / name, args.output / name)

    summary = {
        "schema": "himoe.token_geometry.characterisation.v1",
        "shape_distance": "full Procrustes disparity, translation/rotation/reflection/scale invariant",
        "template": "pooled mean conditional kernel over development_main + development_extra",
        "labels_used": False,
        "pooled_template": {
            row["layer"]: {
                key: row[key]
                for key in (
                    "shape_pr",
                    "shape_erank",
                    "lam1_share",
                    "lam2_share",
                    "dim_90",
                    "r_pc1_index",
                    "r_pc2_index_sq",
                    "spearman_dist_lag",
                    "robinson",
                    "stretch_max_over_min",
                    "pc1_gap_cv",
                )
            }
            for row in rows
            if row["source"] == "pooled_development"
        },
    }
    (args.output / "characterisation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pd.set_option("display.width", 220)
    print(shape[shape["source"] == "pooled_development"].to_string(index=False, float_format="%.4f"))
    print()
    print(pd.DataFrame(stability).to_string(index=False, float_format="%.5f"))


if __name__ == "__main__":
    main()
