#!/usr/bin/env python3
"""Build the browser-sized data bundle for the HiMoE HB activation explorer.

The source stores every router input for every rollout control step.  This
builder uses only the first control row of each episode and groups the 32 common
flow-noise seeds by (task, init_state).  It never modifies source captures.

Two representations deliberately coexist in the output:

* PCA is fitted to layer/action-token means and is only a 2-D visual projection.
* cloud dispersion is computed on the unprojected, unpooled hidden tensor.  It
  is therefore not a PCA distance and retains every hidden coordinate.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.stats import rankdata
from sklearn.decomposition import PCA


SCHEMA_NAME = "himoe_hb_activation_v1"
LAYER_IDS = (2, 3, 4, 5, 12, 13, 14, 15)
VIEWS = {
    "early": {"positions": slice(0, 4), "layers": (2, 3, 4, 5)},
    "late": {"positions": slice(4, 8), "layers": (12, 13, 14, 15)},
}
FEATURE_FAMILIES = (
    "rms",
    "token_adj_cosdist",
    "flow_endpoint_cosdist",
    "pool_centroid_d9_rms",
    "pool_centroid_convergence",
    "pool_knn4_d9_rms",
)
FEATURE_NAMES = tuple(
    f"{view}_{family}" for view in VIEWS for family in FEATURE_FAMILIES
)
PERMUTATION_SEED = 20260821


def finite_float(value: float | np.floating) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"non-finite value in output: {out}")
    return out


def rounded(value: np.ndarray, decimals: int) -> list:
    return np.round(np.asarray(value, dtype=np.float64), decimals=decimals).tolist()


def discover_runs(hub: Path, expected_tasks: int) -> list[Path]:
    cache = hub / "cache" / "HiMoE-VLA"
    runs = sorted(
        path.parent.parent
        for path in cache.glob("**/right-16x32/server/hidden.zarr")
        if (path.parent.parent / "client" / "summaries.json").exists()
    )
    if len(runs) != expected_tasks:
        raise RuntimeError(
            f"expected {expected_tasks} complete right-16x32 runs, found {len(runs)}"
        )
    return runs


def first_rows(run: Path, summaries: list[dict[str, Any]], hidden: Any) -> np.ndarray:
    episode_id = np.asarray(hidden["episode_id"][:], dtype=np.int64)
    starts = np.r_[0, 1 + np.flatnonzero(episode_id[1:] != episode_id[:-1])]
    start_ids = episode_id[starts]
    if len(starts) != len(summaries) or len(np.unique(start_ids)) != len(summaries):
        raise RuntimeError(
            f"episode segmentation mismatch in {run}: {len(starts)} starts for "
            f"{len(summaries)} summaries"
        )
    by_id = {int(ep): int(row) for ep, row in zip(start_ids, starts)}
    expected = {int(row["episode_index"]) for row in summaries}
    if set(by_id) != expected:
        raise RuntimeError(f"episode IDs do not match summaries in {run}")
    return np.asarray([by_id[int(row["episode_index"])] for row in summaries])


def centroid_dispersion(x: np.ndarray) -> np.ndarray:
    """RMS distance to the K-candidate centroid for every denoise step.

    x has shape [K, layer, denoise, token, hidden].  Only the candidate axis is
    used to form the centroid; the reported RMS retains all other coordinates.
    """
    centroid = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    residual = x - centroid[None]
    return np.sqrt(np.mean(np.square(residual), axis=(0, 1, 3, 4), dtype=np.float64))


def layer_dispersion(x: np.ndarray) -> np.ndarray:
    """Per-layer version of centroid_dispersion, returning [layer, denoise]."""
    centroid = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    residual = x - centroid[None]
    return np.sqrt(np.mean(np.square(residual), axis=(0, 3, 4), dtype=np.float64))


def fit_pca_trajectory(x: np.ndarray, decimals: int) -> dict[str, Any]:
    """Fit one fixed PCA to per-denoise candidate residuals."""
    # x: [K, layer, denoise, action_token, hidden]
    mean_hidden = x.mean(axis=(1, 3), dtype=np.float32)  # [K, denoise, 1024]
    # Remove the hidden-state centroid separately at every denoise step.  The
    # projection then shows candidate-cloud geometry and contraction instead
    # of the much larger trajectory of the shared centroid.
    mean_hidden -= mean_hidden.mean(axis=0, keepdims=True, dtype=np.float32)
    k, denoise, width = mean_hidden.shape
    matrix = mean_hidden.reshape(k * denoise, width)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0, iterated_power=5)
    coords = pca.fit_transform(matrix).reshape(k, denoise, 2)

    # Make component orientation reproducible even if the linear algebra backend
    # chooses the opposite valid sign for a singular vector.
    for component in range(2):
        pivot = int(np.argmax(np.abs(pca.components_[component])))
        if pca.components_[component, pivot] < 0:
            coords[:, :, component] *= -1

    return {
        "fit_samples": int(k * denoise),
        "input_width": int(width),
        "candidate_centered_per_denoise": True,
        "explained_variance_ratio": [
            finite_float(value) for value in pca.explained_variance_ratio_
        ],
        "steps": rounded(coords.transpose(1, 0, 2), decimals),
    }


def cosdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.einsum("...d,...d->...", a, b, optimize=True)
    aa = np.einsum("...d,...d->...", a, a, optimize=True)
    bb = np.einsum("...d,...d->...", b, b, optimize=True)
    return 1.0 - dot / np.sqrt(np.maximum(aa * bb, 1e-20))


def candidate_rms_to_centroid(x: np.ndarray) -> np.ndarray:
    centroid = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    return np.sqrt(np.mean(np.square(x - centroid[None]), axis=tuple(range(1, x.ndim))))


def knn4_rms(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(len(x), -1)
    squared_norm = np.einsum("nd,nd->n", flat, flat, optimize=True)
    distance2 = (
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * (flat @ flat.T)
    ) / flat.shape[1]
    np.maximum(distance2, 0.0, out=distance2)
    np.fill_diagonal(distance2, np.inf)
    return np.sqrt(np.partition(distance2, 3, axis=1)[:, :4]).mean(axis=1)


def scalar_success_features(h: np.ndarray) -> np.ndarray:
    """The twelve predeclared, success-blind scalar features for one K32 pool."""
    values: dict[str, np.ndarray] = {}
    for view, spec in VIEWS.items():
        x = h[:, spec["positions"], :, 1:, :]
        values[f"{view}_rms"] = np.sqrt(np.mean(np.square(x), axis=(1, 2, 3, 4)))
        values[f"{view}_token_adj_cosdist"] = cosdist(
            x[:, :, :, :-1], x[:, :, :, 1:]
        ).mean((1, 2, 3))
        values[f"{view}_flow_endpoint_cosdist"] = cosdist(
            x[:, :, 0], x[:, :, -1]
        ).mean((1, 2))
        d0 = candidate_rms_to_centroid(x[:, :, 0])
        d9 = candidate_rms_to_centroid(x[:, :, -1])
        values[f"{view}_pool_centroid_d9_rms"] = d9
        values[f"{view}_pool_centroid_convergence"] = d0 - d9
        values[f"{view}_pool_knn4_d9_rms"] = knn4_rms(x[:, :, -1])
    return np.column_stack([values[name] for name in FEATURE_NAMES]).astype(np.float64)


def success_screen(
    features: np.ndarray,
    labels: np.ndarray,
    task_ids: np.ndarray,
    permutations: int,
) -> dict[str, Any]:
    """Within-pool AUC with one shared 32-seed permutation and two-sided maxT."""
    ranks = np.empty_like(features, dtype=np.float64)
    for pool in range(len(features)):
        for feature in range(features.shape[2]):
            ranks[pool, :, feature] = rankdata(
                features[pool, :, feature], method="average"
            )

    n_success = labels.sum(axis=1)
    mixed = (n_success > 0) & (n_success < 32)
    balanced = (n_success >= 4) & (n_success <= 28)
    non_long = np.asarray([not task.startswith("libero_long/") for task in task_ids])
    subset_masks = {
        "all_mixed": mixed,
        "balanced_min4": balanced,
        "non_long_mixed": mixed & non_long,
        "non_long_balanced_min4": balanced & non_long,
    }

    rng = np.random.default_rng(PERMUTATION_SEED)
    perms = np.stack([rng.permutation(32) for _ in range(permutations)])
    subsets = []
    for name, keep in subset_masks.items():
        rr = ranks[keep]
        yy = labels[keep]
        n1 = yy.sum(axis=1).astype(np.float64)
        n0 = 32 - n1
        correction_by_pool = n1 * (n1 + 1) / 2.0
        u_by_pool = (
            np.einsum("pkf,pk->pf", rr, yy, optimize=True)
            - correction_by_pool[:, None]
        )
        pairs_by_pool = n1 * n0
        observed = u_by_pool.sum(axis=0) / pairs_by_pool.sum()
        macro = np.mean(u_by_pool / pairs_by_pool[:, None], axis=0)

        null = np.empty((permutations, len(FEATURE_NAMES)), dtype=np.float32)
        batch = 250
        correction = correction_by_pool.sum()
        denominator = pairs_by_pool.sum()
        for start in range(0, permutations, batch):
            p = perms[start : start + batch]
            permuted_y = np.transpose(yy[:, p], (1, 0, 2))
            rank_sums = np.einsum("bpk,pkf->bf", permuted_y, rr, optimize=True)
            null[start : start + len(p)] = (rank_sums - correction) / denominator

        deviation = np.abs(observed - 0.5)
        null_deviation = np.abs(null - 0.5)
        raw_p = (1 + (null_deviation >= deviation[None]).sum(axis=0)) / (
            permutations + 1
        )
        null_max = null_deviation.max(axis=1)
        max_p = (1 + (null_max[:, None] >= deviation[None]).sum(axis=0)) / (
            permutations + 1
        )
        per_pool_auc = u_by_pool / pairs_by_pool[:, None]
        subsets.append(
            {
                "id": name,
                "n_pools": int(keep.sum()),
                "n_candidates": int(32 * keep.sum()),
                "discordant_pairs": int(denominator),
                "null_max_abs_auc_minus_half_p95": finite_float(
                    np.quantile(null_max, 0.95)
                ),
                "feature": [
                    {
                        "id": FEATURE_NAMES[index],
                        "auc": finite_float(observed[index]),
                        "macro_auc": finite_float(macro[index]),
                        "p_raw_two_sided": finite_float(raw_p[index]),
                        "p_maxT": finite_float(max_p[index]),
                        "pools_auc_above_half": int(
                            (per_pool_auc[:, index] > 0.5).sum()
                        ),
                        "pools_auc_below_half": int(
                            (per_pool_auc[:, index] < 0.5).sum()
                        ),
                    }
                    for index in range(len(FEATURE_NAMES))
                ],
            }
        )
    return {
        "analysis_unit": "first control row; comparisons only within task/init-state pools",
        "features": list(FEATURE_NAMES),
        "permutation": {
            "n": int(permutations),
            "seed": PERMUTATION_SEED,
            "scheme": "one common permutation of the 32 flow-noise seed positions across every pool",
            "test": "two-sided abs(AUC-0.5); maxT over the 12 scalar features within each subset",
        },
        "subsets": subsets,
    }


def curve_summary(curves: np.ndarray) -> dict[str, Any]:
    median = np.median(curves, axis=0)
    retention = curves / curves[:, :1]
    return {
        "dispersion_median_by_pool": [finite_float(x) for x in median],
        "dispersion_q25_by_pool": [finite_float(x) for x in np.quantile(curves, 0.25, axis=0)],
        "dispersion_q75_by_pool": [finite_float(x) for x in np.quantile(curves, 0.75, axis=0)],
        "retention_median_by_pool": [finite_float(x) for x in np.median(retention, axis=0)],
        "endpoint": {
            "d0": finite_float(median[0]),
            "d9": finite_float(median[-1]),
            "d9_over_d0_pool_median": finite_float(np.median(retention[:, -1])),
        },
    }


def schema(pca_decimals: int) -> dict[str, Any]:
    return {
        "name": SCHEMA_NAME,
        "version": 1,
        "semantics": {
            "tensor": "hb_hidden: the 1024-D input read by each HB router/block; not an expert or block output",
            "first_row": "the first policy inference of each episode",
            "candidate_pool": "one task and init_state, sorted by the 32 common flow_noise_seed values",
            "pca": "for each pool/view, average action tokens and view layers to [K,denoise,1024], subtract the K-candidate mean separately at every denoise step, then fit one label-blind fixed 2-D PCA to all K*denoise residuals; each step's cloud is centered near (0,0), so the plot emphasizes candidate contraction rather than shared-centroid drift",
            "dispersion_rms": "sqrt(mean((h-candidate_centroid)^2)) at each denoise step on raw, unprojected [K,view_layer,action_token,1024] hidden",
            "retention": "dispersion_rms[d] / dispersion_rms[0]",
            "matched_site_norm_cv": "at each matched (layer,denoise,action-token) site, std_K(||h||_RMS)/mean_K(||h||_RMS); a view first takes each layer's site median, then the median of its four equally weighted layers; values are unitless ratios",
            "state_token_range": "candidate-wise max-min of raw suffix token 0; max and RMS are reduced only after taking that range",
        },
        "axes": {
            "candidate": 32,
            "denoise": 10,
            "suffix_token": 11,
            "state_token_index": 0,
            "action_token_indices": list(range(1, 11)),
            "hidden": 1024,
            "hb_layer_ids": list(LAYER_IDS),
        },
        "views": {name: {"layers": list(spec["layers"])} for name, spec in VIEWS.items()},
        "field_shapes": {
            "pools[].candidates": "[32] objects aligned to the PCA candidate axis",
            "pools[].views.<view>.pca.steps": "[10 denoise][32 candidate][2 coordinate]",
            "pools[].views.<view>.pca.candidate_centered_per_denoise": "true; PCA input subtracts mean_K independently for every denoise step",
            "pools[].views.<view>.raw_cloud.dispersion_rms": "[10 denoise]",
            "pools[].views.<view>.raw_cloud.matched_site_norm_cv_median_by_denoise": "[10 denoise]",
            "pools[].layers": "[8 layer] with [10] dispersion and retention curves",
            "pools[].state_token_range.*_by_denoise": "[10 denoise]",
        },
        "encoding": {
            "success": "JSON boolean",
            "pca_coordinate_decimals": int(pca_decimals),
            "raw_metrics": "unrounded JSON float64 reductions from float16 source decoded to float32",
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    hub = args.hub_root.resolve()
    plan_path = hub / "_pipeline" / "right-16x32" / "plan.json"
    plan = json.loads(plan_path.read_text())
    runs = discover_runs(hub, int(plan["n_tasks"]))

    pools: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    all_view_curves: dict[str, list[np.ndarray]] = {name: [] for name in VIEWS}
    all_view_norm_cv: dict[str, list[float]] = {name: [] for name in VIEWS}
    all_layer_curves: list[np.ndarray] = []
    scalar_features: list[np.ndarray] = []
    scalar_labels: list[np.ndarray] = []
    scalar_tasks: list[str] = []
    canonical_seeds: tuple[int, ...] | None = None

    for run_index, run in enumerate(runs, start=1):
        relative = run.relative_to(hub / "cache" / "HiMoE-VLA")
        suite, task_name = relative.parts[:2]
        task_id = f"{suite}/{task_name}"
        summaries = sorted(
            json.loads((run / "client" / "summaries.json").read_text()),
            key=lambda row: int(row["episode_index"]),
        )
        hidden = zarr.open_group(str(run / "server" / "hidden.zarr"), mode="r")
        if hidden["hb_hidden"].shape[1:] != (8, 10, 11, 1024):
            raise RuntimeError(
                f"unexpected hb_hidden shape in {run}: {hidden['hb_hidden'].shape}"
            )
        rows = first_rows(run, summaries, hidden)
        by_state: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(summaries):
            by_state[int(row["init_state_id"])].append(index)

        task_pool_ids: list[str] = []
        task_view_curves: dict[str, list[np.ndarray]] = {name: [] for name in VIEWS}
        task_success = int(sum(bool(row["success"]) for row in summaries))
        mixed_count = balanced_count = 0
        print(
            f"[{run_index}/{len(runs)}] {task_id}: {len(by_state)} pools, "
            f"{task_success}/{len(summaries)} success",
            flush=True,
        )

        for state in sorted(by_state):
            indices = np.asarray(by_state[state], dtype=np.int64)
            indices = indices[
                np.argsort(
                    [int(summaries[index]["flow_noise_seed"]) for index in indices]
                )
            ]
            seeds = tuple(
                int(summaries[index]["flow_noise_seed"]) for index in indices
            )
            if len(indices) != 32 or len(set(seeds)) != 32:
                raise RuntimeError(f"{task_id} state {state} is not a complete K=32 pool")
            if canonical_seeds is None:
                canonical_seeds = seeds
            elif seeds != canonical_seeds:
                raise RuntimeError(f"seed ordering differs in {task_id} state {state}")

            labels = np.asarray(
                [bool(summaries[index]["success"]) for index in indices], dtype=bool
            )
            n_success = int(labels.sum())
            mixed_count += int(0 < n_success < 32)
            balanced_count += int(4 <= n_success <= 28)

            h = np.asarray(
                hidden["hb_hidden"].oindex[rows[indices], :, :, :, :],
                dtype=np.float32,
            )
            if h.shape != (32, 8, 10, 11, 1024) or not np.isfinite(h).all():
                raise RuntimeError(f"invalid first-row tensor in {task_id} state {state}")

            state_range = np.ptp(h[:, :, :, 0, :], axis=0)
            state_max_by_denoise = state_range.max(axis=(0, 2))
            state_rms_by_denoise = np.sqrt(
                np.mean(np.square(state_range), axis=(0, 2), dtype=np.float64)
            )

            action_hidden = h[:, :, :, 1:, :]
            per_layer = layer_dispersion(action_hidden)
            all_layer_curves.append(per_layer)
            view_payload: dict[str, Any] = {}
            for view, spec in VIEWS.items():
                view_hidden = h[:, spec["positions"], :, 1:, :]
                dispersion = centroid_dispersion(view_hidden)
                retention = dispersion / dispersion[0]
                # Keep this float32 reduction aligned with the audited RMSNorm
                # diagnostic in analyze_hb_hidden_success.py.
                vector_rms = np.sqrt(np.mean(np.square(view_hidden), axis=-1))
                site_norm_cv = vector_rms.std(axis=0) / vector_rms.mean(axis=0)
                layer_norm_cv = np.median(site_norm_cv, axis=(1, 2))
                norm_cv_median = finite_float(np.median(layer_norm_cv))
                norm_cv_by_denoise = np.median(
                    np.median(site_norm_cv, axis=2), axis=0
                )
                all_view_curves[view].append(dispersion)
                all_view_norm_cv[view].append(norm_cv_median)
                task_view_curves[view].append(dispersion)
                view_payload[view] = {
                    "layers": list(spec["layers"]),
                    "pca": fit_pca_trajectory(view_hidden, args.pca_decimals),
                    "raw_cloud": {
                        "dispersion_rms": [finite_float(value) for value in dispersion],
                        "retention": [finite_float(value) for value in retention],
                        "matched_site_norm_cv_median": norm_cv_median,
                        "matched_site_norm_cv": norm_cv_median,
                        "matched_site_norm_cv_median_by_denoise": [
                            finite_float(value) for value in norm_cv_by_denoise
                        ],
                    },
                }

            pool_id = f"{task_id}::state-{state}"
            task_pool_ids.append(pool_id)
            pools.append(
                {
                    "id": pool_id,
                    "task_id": task_id,
                    "init_state_id": int(state),
                    "counts": {"success": n_success, "failure": 32 - n_success},
                    "candidates": [
                        {"seed": int(seed), "success": bool(success)}
                        for seed, success in zip(seeds, labels)
                    ],
                    "state_token_range": {
                        "max": finite_float(state_range.max()),
                        "max_by_denoise": [
                            finite_float(value) for value in state_max_by_denoise
                        ],
                        "rms_by_denoise": [
                            finite_float(value) for value in state_rms_by_denoise
                        ],
                    },
                    "views": view_payload,
                    "layers": [
                        {
                            "layer": int(layer),
                            "dispersion_rms": [
                                finite_float(value) for value in per_layer[position]
                            ],
                            "retention": [
                                finite_float(value)
                                for value in per_layer[position] / per_layer[position, 0]
                            ],
                        }
                        for position, layer in enumerate(LAYER_IDS)
                    ],
                }
            )
            scalar_features.append(scalar_success_features(h))
            scalar_labels.append(labels)
            scalar_tasks.append(task_id)
            del h, action_hidden

        tasks.append(
            {
                "id": task_id,
                "suite": suite,
                "task_name": task_name,
                "display_name": task_name.replace("_", " "),
                "source_run": str(relative),
                "episodes": len(summaries),
                "success": task_success,
                "failure": len(summaries) - task_success,
                "mixed_pools": mixed_count,
                "balanced_min4_pools": balanced_count,
                "pool_ids": task_pool_ids,
                "views": {
                    view: curve_summary(np.stack(curves))
                    for view, curves in task_view_curves.items()
                },
            }
        )

    view_arrays = {view: np.stack(curves) for view, curves in all_view_curves.items()}
    layer_array = np.stack(all_layer_curves)  # [80,8,10]
    global_views = {view: curve_summary(curves) for view, curves in view_arrays.items()}
    for view in VIEWS:
        norm_cv_median = finite_float(np.median(all_view_norm_cv[view]))
        global_views[view]["matched_site_norm_cv_median"] = norm_cv_median
        # Compatibility alias for the initial front-end schema draft.
        global_views[view]["matched_site_norm_cv"] = norm_cv_median
    global_layers = [
        {
            "layer": int(layer),
            **curve_summary(layer_array[:, position]),
        }
        for position, layer in enumerate(LAYER_IDS)
    ]

    # These assertions pin the raw metric definition to the audited reference.
    expected_endpoints = {
        "early": (0.4848240242, 0.1561376803),
        "late": (0.3520347055, 0.1685039502),
    }
    expected_norm_cv = {"early": 0.0005795878, "late": 0.0006641507}
    for view, expected in expected_endpoints.items():
        actual = global_views[view]["endpoint"]
        if not (
            abs(actual["d0"] - expected[0]) < 2e-6
            and abs(actual["d9"] - expected[1]) < 2e-6
        ):
            raise RuntimeError(
                f"{view} endpoint definition drifted: {actual} vs {expected}"
            )
        actual_cv = global_views[view]["matched_site_norm_cv_median"]
        if abs(actual_cv - expected_norm_cv[view]) >= 2e-7:
            raise RuntimeError(
                f"{view} matched-site norm CV drifted: {actual_cv} vs "
                f"{expected_norm_cv[view]}"
            )

    state_max = max(pool["state_token_range"]["max"] for pool in pools)
    payload = {
        "schema": schema(args.pca_decimals),
        "source": {
            "pipeline_plan": str(plan_path.relative_to(hub)),
            "run_id": plan["run_id"],
            "wrist_layout": plan["wrist_layout"],
            "scene_ids": plan["scene_ids"],
            "common_flow_noise_seeds": list(canonical_seeds or ()),
            "task_count": len(tasks),
            "pool_count": len(pools),
            "episode_count": int(sum(task["episodes"] for task in tasks)),
        },
        "global": {
            "counts": {
                "tasks": len(tasks),
                "pools": len(pools),
                "candidates": 32 * len(pools),
                "mixed_pools": int(sum(task["mixed_pools"] for task in tasks)),
                "balanced_min4_pools": int(
                    sum(task["balanced_min4_pools"] for task in tasks)
                ),
            },
            "state_token_max_candidate_range": finite_float(state_max),
            "views": global_views,
            "layers": global_layers,
        },
        "tasks": tasks,
        "pools": pools,
        "success_probe": success_screen(
            np.stack(scalar_features),
            np.stack(scalar_labels),
            np.asarray(scalar_tasks),
            args.permutations,
        ),
    }
    return payload


def validate(payload: dict[str, Any], output: Path, max_bytes: int) -> None:
    if payload["schema"]["name"] != SCHEMA_NAME:
        raise RuntimeError("schema name mismatch")
    counts = payload["global"]["counts"]
    if counts != {
        "tasks": 5,
        "pools": 80,
        "candidates": 2560,
        "mixed_pools": 40,
        "balanced_min4_pools": 16,
    }:
        raise RuntimeError(f"unexpected corpus counts: {counts}")
    if payload["global"]["state_token_max_candidate_range"] != 0.0:
        raise RuntimeError("the first-row state token unexpectedly differs across candidates")
    for pool in payload["pools"]:
        if len(pool["candidates"]) != 32:
            raise RuntimeError(f"bad candidate count in {pool['id']}")
        for view in VIEWS:
            value = pool["views"][view]
            coordinates = np.asarray(value["pca"]["steps"], dtype=np.float64)
            if coordinates.shape != (10, 32, 2):
                raise RuntimeError(f"bad PCA shape in {pool['id']} / {view}")
            if value["pca"]["candidate_centered_per_denoise"] is not True:
                raise RuntimeError(f"PCA is not candidate-centered in {pool['id']} / {view}")
            decimals = payload["schema"]["encoding"]["pca_coordinate_decimals"]
            rounding_tolerance = 0.5 * 10.0 ** (-decimals) + 1e-8
            if np.max(np.abs(coordinates.mean(axis=1))) > rounding_tolerance:
                raise RuntimeError(
                    f"PCA step centroid is not zero in {pool['id']} / {view}"
                )
            if len(value["raw_cloud"]["dispersion_rms"]) != 10:
                raise RuntimeError(f"bad dispersion shape in {pool['id']} / {view}")
            if value["raw_cloud"]["retention"][0] != 1.0:
                raise RuntimeError(f"bad retention origin in {pool['id']} / {view}")
        if [row["layer"] for row in pool["layers"]] != list(LAYER_IDS):
            raise RuntimeError(f"bad layer order in {pool['id']}")

    reparsed = json.loads(output.read_text())
    if reparsed["global"]["counts"] != counts:
        raise RuntimeError("JSON round-trip changed the payload")
    size = output.stat().st_size
    if size > max_bytes:
        raise RuntimeError(f"data.json is {size:,} bytes, above limit {max_bytes:,}")


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-root", type=Path, default=here.parent)
    ap.add_argument("--output", type=Path, default=here / "data.json")
    ap.add_argument("--pca-decimals", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--max-bytes", type=int, default=8_000_000)
    args = ap.parse_args()

    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n"
    )
    os.replace(temporary, args.output)
    validate(payload, args.output, args.max_bytes)

    early = payload["global"]["views"]["early"]["endpoint"]
    late = payload["global"]["views"]["late"]["endpoint"]
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    print(
        "validated: 5 tasks / 80 pools / 2560 candidates; "
        f"early {early['d0']:.6f}->{early['d9']:.6f}; "
        f"late {late['d0']:.6f}->{late['d9']:.6f}; "
        f"state-token max range {payload['global']['state_token_max_candidate_range']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
