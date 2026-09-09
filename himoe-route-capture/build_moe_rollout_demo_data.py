#!/usr/bin/env python3
"""Build the compact JavaScript payload for the rollout-trend demo."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
RESULT_DIR = HERE / "analysis/moe-rollout-trend"
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
FAMILIES = (
    "routed_identity",
    "routed_scalar",
    "routed_full",
    "hidden_identity",
    "shared_identity",
    "base",
    "base_routed",
)
DYNAMIC_METRICS = {
    "routed_rms": ("scalar", 8),
    "cancellation": ("scalar", 10),
    "shared_conflict": ("scalar", 13),
    "commitment": ("routed_commitment", 1),
}
ROUTE_ATLAS_VERSION = 3
N_HB_EXPERTS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=pathlib.Path, default=RESULT_DIR)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument(
        "--output", type=pathlib.Path, default=RESULT_DIR / "demo-data.js"
    )
    parser.add_argument(
        "--rebuild-route-atlas",
        action="store_true",
        help="recompute the full variable-length route atlas from raw capture",
    )
    return parser.parse_args()


def task_counts(result_dir: pathlib.Path) -> dict[str, dict]:
    counts = {}
    for path in sorted((result_dir / "compact").glob("*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            task = str(stored["task"])
            failure = np.asarray(stored["failure"], dtype=bool)
            scenes = np.asarray(stored["scenes"])
            counts[task] = {
                "rollouts": int(len(failure)),
                "failure": int(failure.sum()),
                "success": int((~failure).sum()),
                "initial_states": int(len(np.unique(scenes))),
                "mixed_states": int(
                    sum(
                        len(np.unique(failure[scenes == scene])) == 2
                        for scene in np.unique(scenes)
                    )
                ),
            }
    return counts


def _same_state_group_mean(
    values: np.ndarray, failure: np.ndarray, scenes: np.ndarray
) -> np.ndarray:
    """Return equally weighted mixed-state success/failure standardized means."""
    values = np.asarray(values, dtype=np.float32)
    center = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    standardized = (values - center) / np.maximum(scale, 1e-6)
    state_points = []
    for scene in np.unique(scenes):
        state = scenes == scene
        if not np.any(state & failure) or not np.any(state & ~failure):
            continue
        state_points.append(
            np.stack(
                (
                    standardized[state & ~failure].mean(axis=0),
                    standardized[state & failure].mean(axis=0),
                )
            )
        )
    if not state_points:
        raise ValueError("task contains no mixed initial states")
    return np.mean(state_points, axis=0)


def task_dynamics(result_dir: pathlib.Path, tasks: list[dict]) -> dict:
    by_scope = {}
    task_arrays = {}
    for path in sorted((result_dir / "compact").glob("*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            task = str(stored["task"])
            failure = np.asarray(stored["failure"], dtype=bool)
            scenes = np.asarray(stored["scenes"])
            metrics = {}
            for metric, (source, index) in DYNAMIC_METRICS.items():
                source_values = np.asarray(stored[source], dtype=np.float32)
                values = source_values[..., index]
                point = _same_state_group_mean(values, failure, scenes)
                metrics[metric] = point
            task_arrays[task] = metrics

    expected_tasks = [task["id"] for task in tasks]
    if set(task_arrays) != set(expected_tasks):
        raise ValueError("dynamic task set does not match curve task set")
    for task in expected_tasks:
        by_scope[task] = {
            metric: {
                "success": np.round(values[0], 5).tolist(),
                "failure": np.round(values[1], 5).tolist(),
                "gap": np.round(values[1] - values[0], 5).tolist(),
            }
            for metric, values in task_arrays[task].items()
        }
    by_scope["aggregate"] = {}
    for metric in DYNAMIC_METRICS:
        values = np.mean(
            [task_arrays[task][metric] for task in expected_tasks], axis=0
        )
        by_scope["aggregate"][metric] = {
            "success": np.round(values[0], 5).tolist(),
            "failure": np.round(values[1], 5).tolist(),
            "gap": np.round(values[1] - values[0], 5).tolist(),
        }
    return {
        "metrics": list(DYNAMIC_METRICS),
        "layers": [2, 5, 12, 15],
        "denoise_rounds": 10,
        "normalization": "within-task per-cell z score; mixed initial states equal weight",
        "by_scope": by_scope,
    }


def _balanced_mixed_scene(rows: list[dict]) -> int:
    candidates = []
    for scene in sorted({int(row["init_state_id"]) for row in rows}):
        local = [row for row in rows if int(row["init_state_id"]) == scene]
        success = sum(bool(row["success"]) for row in local)
        failure = len(local) - success
        if success and failure:
            candidates.append((min(success, failure), -abs(success - failure), scene))
    if not candidates:
        raise ValueError("task contains no mixed initial state")
    return max(candidates)[2]


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return (ranks + 0.5) / len(values)


def _route_moe_percentiles(
    compact_path: pathlib.Path, episodes: np.ndarray, success: np.ndarray
) -> np.ndarray | None:
    if int(success.sum()) < 2:
        return None
    with np.load(compact_path, allow_pickle=False) as stored:
        compact_episodes = np.asarray(stored["episodes"], dtype=np.int64)
        episode_axis = {int(episode): axis for axis, episode in enumerate(compact_episodes)}
        axes = np.asarray([episode_axis[int(episode)] for episode in episodes])
        routed = np.asarray(stored["routed"][axes], dtype=np.float32)
    output = np.empty((len(episodes), routed.shape[1]), dtype=np.float64)
    for chunk in range(routed.shape[1]):
        features = routed[:, chunk].reshape(len(episodes), -1)
        success_sum = features[success].sum(axis=0)
        success_count = int(success.sum())
        distances = np.empty(len(features), dtype=np.float64)
        for axis, feature in enumerate(features):
            if success[axis] and success_count > 1:
                center = (success_sum - feature) / (success_count - 1)
            else:
                center = success_sum / success_count
            distances[axis] = np.sqrt(np.mean(np.square(feature - center)))
        output[:, chunk] = _rank_percentiles(distances)
    return output


def _routing_metrics(
    ids: np.ndarray, raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return HB routing signatures, denoise churn, and d0-to-d9 revision."""
    ids = np.asarray(ids, dtype=np.int64)
    raw = np.asarray(raw, dtype=np.float32)
    alpha = raw / np.maximum(raw.sum(axis=-1, keepdims=True), 1e-12)
    cells = int(np.prod(ids.shape[:-1]))
    occupancy = np.zeros((cells, N_HB_EXPERTS), dtype=np.float32)
    cell_axis = np.repeat(np.arange(cells, dtype=np.int64), ids.shape[-1])
    np.add.at(
        occupancy,
        (cell_axis, ids.reshape(-1)),
        alpha.reshape(-1),
    )
    occupancy = occupancy.reshape(*ids.shape[:-1], N_HB_EXPERTS)
    grouped = np.stack(
        (occupancy[:, :, :, 0], occupancy[:, :, :, 1:].mean(axis=3)),
        axis=3,
    )
    adjacent = np.sqrt(
        np.mean(np.square(grouped[:, :, 1:] - grouped[:, :, :-1]), axis=(1, 3, 4))
    )
    churn = adjacent.mean(axis=1)
    revision = np.sqrt(
        np.mean(np.square(grouped[:, :, -1] - grouped[:, :, 0]), axis=(1, 2, 3))
    )
    return grouped.reshape(len(grouped), -1), churn, revision


def _leave_one_out_distances(features: np.ndarray, success: np.ndarray) -> np.ndarray:
    success_count = int(success.sum())
    if success_count < 2:
        raise ValueError("at least two active success routes are required")
    success_sum = features[success].sum(axis=0)
    centers = np.repeat((success_sum / success_count)[None], len(features), axis=0)
    centers[success] = (success_sum - features[success]) / (success_count - 1)
    return np.sqrt(np.mean(np.square(features - centers), axis=1))


def _optional_round(values: np.ndarray | list, digits: int) -> list:
    array = np.asarray(values, dtype=np.float64)
    return [None if not np.isfinite(value) else round(float(value), digits) for value in array]


def _load_full_task_routes(
    result_dir: pathlib.Path, cache_root: pathlib.Path, task: str
) -> dict:
    import zarr

    run = cache_root / task / "right-16x32"
    client = run / "client"
    rows = sorted(
        json.loads((client / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    episodes = np.asarray([int(row["episode_index"]) for row in rows])
    scenes = np.asarray([int(row["init_state_id"]) for row in rows])
    success = np.asarray([bool(row["success"]) for row in rows])
    counts = np.asarray([int(row["inference_calls"]) for row in rows])
    offsets = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    states = []
    for row, count in zip(rows, counts):
        episode = int(row["episode_index"])
        with np.load(client / ("episode_%02d.npz" % episode), allow_pickle=False) as stored:
            state = np.asarray(stored["state"][:, :3], dtype=np.float64)
        if len(state) != count:
            raise ValueError("state/query length mismatch for %s episode %d" % (task, episode))
        states.append(state)

    route_group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    route_episodes = np.asarray(route_group["episode_id"][:])
    if len(route_episodes) != int(counts.sum()):
        raise ValueError("summary and route capture row counts differ for %s" % task)
    if not np.all(route_episodes[offsets] == episodes):
        raise ValueError("episode offsets do not align for %s" % task)

    routing = [np.full(count, np.nan, dtype=np.float64) for count in counts]
    routing_churn = [np.full(count, np.nan, dtype=np.float64) for count in counts]
    routing_revision = [np.full(count, np.nan, dtype=np.float64) for count in counts]
    routing_gap_by_scene = {
        int(scene): np.full(int(counts[scenes == scene].max()), np.nan)
        for scene in np.unique(scenes)
    }
    active_by_scene = {
        int(scene): np.zeros((int(counts[scenes == scene].max()), 2), dtype=np.int64)
        for scene in np.unique(scenes)
    }
    for chunk in range(int(counts.max())):
        active_axes = np.flatnonzero(counts > chunk)
        capture_rows = offsets[active_axes] + chunk
        selection = (capture_rows, slice(None), slice(None), slice(None), slice(None))
        ids = route_group["hb_expert_ids"].get_orthogonal_selection(selection)
        raw = route_group["hb_selected_prob"].get_orthogonal_selection(selection)
        signatures, churn, revision = _routing_metrics(ids, raw)
        for scene in np.unique(scenes[active_axes]):
            scene = int(scene)
            local = scenes[active_axes] == scene
            local_axes = active_axes[local]
            local_success = success[local_axes]
            active_by_scene[scene][chunk] = (
                int(local_success.sum()),
                int((~local_success).sum()),
            )
            churn_percentiles = _rank_percentiles(churn[local])
            revision_percentiles = _rank_percentiles(revision[local])
            for axis, churn_value, revision_value in zip(
                local_axes, churn_percentiles, revision_percentiles
            ):
                routing_churn[axis][chunk] = churn_value
                routing_revision[axis][chunk] = revision_value
            if int(local_success.sum()) < 2:
                continue
            distances = _leave_one_out_distances(signatures[local], local_success)
            percentiles = _rank_percentiles(distances)
            for axis, percentile in zip(local_axes, percentiles):
                routing[axis][chunk] = percentile
            if np.any(~local_success):
                routing_gap_by_scene[scene][chunk] = (
                    percentiles[~local_success].mean()
                    - percentiles[local_success].mean()
                )

    compact_path = result_dir / "compact" / (task.replace("/", "__") + ".npz")
    scene_entries = []
    default_scene = _balanced_mixed_scene(rows)
    for scene in sorted(np.unique(scenes)):
        scene = int(scene)
        scene_axes = np.flatnonzero(scenes == scene)
        local_rows = [rows[axis] for axis in scene_axes]
        local_states = [states[axis] for axis in scene_axes]
        local_success = success[scene_axes]
        local_episodes = episodes[scene_axes]
        flat = np.concatenate(local_states, axis=0)
        center = flat.mean(axis=0)
        _, singular, axes = np.linalg.svd(flat - center, full_matrices=False)
        projected = [(state - center) @ axes[:2].T for state in local_states]
        start_x = np.mean([points[0, 0] for points in projected])
        terminal_x = np.mean([points[-1, 0] for points in projected])
        if terminal_x < start_x:
            for points in projected:
                points[:, 0] *= -1
        if np.any(local_success) and np.any(~local_success):
            success_y = np.mean([projected[i][-1, 1] for i in np.flatnonzero(local_success)])
            failure_y = np.mean([projected[i][-1, 1] for i in np.flatnonzero(~local_success)])
            if failure_y < success_y:
                for points in projected:
                    points[:, 1] *= -1

        contribution = _route_moe_percentiles(
            compact_path, local_episodes, local_success
        )
        max_chunks = int(max(counts[scene_axes]))
        endpoint_gap = np.full(max_chunks, np.nan)
        contribution_gap = np.full(9, np.nan)
        if np.any(local_success) and np.any(~local_success):
            for chunk in range(max_chunks):
                held = np.stack(
                    [state[min(chunk, len(state) - 1)] for state in local_states]
                )
                endpoint_gap[chunk] = np.linalg.norm(
                    held[local_success].mean(axis=0)
                    - held[~local_success].mean(axis=0)
                )
            if contribution is not None:
                contribution_gap = (
                    contribution[~local_success].mean(axis=0)
                    - contribution[local_success].mean(axis=0)
                )

        highlighted = {}
        for group, mask in (("success", local_success), ("failure", ~local_success)):
            candidates = np.flatnonzero(mask)
            if len(candidates):
                chosen = max(
                    candidates,
                    key=lambda axis: (len(local_states[axis]), -int(local_episodes[axis])),
                )
                highlighted[group] = int(local_episodes[chosen])

        routes = []
        for local_axis, (global_axis, row, points) in enumerate(
            zip(scene_axes, local_rows, projected)
        ):
            contribution_values = (
                [None] * min(9, len(points))
                if contribution is None
                else np.round(contribution[local_axis, : min(9, len(points))], 5).tolist()
            )
            routes.append(
                {
                    "episode": int(episodes[global_axis]),
                    "seed": int(row["flow_noise_seed"]),
                    "success": bool(success[global_axis]),
                    "chunks": int(len(points)),
                    "action_steps": int(row["action_steps"]),
                    "points": np.round(points, 6).tolist(),
                    "routing_percentile": _optional_round(routing[global_axis], 5),
                    "routing_churn_percentile": _optional_round(
                        routing_churn[global_axis], 5
                    ),
                    "routing_revision_percentile": _optional_round(
                        routing_revision[global_axis], 5
                    ),
                    "contribution_percentile": contribution_values,
                }
            )
        scene_entries.append(
            {
                "scene": scene,
                "success": int(local_success.sum()),
                "failure": int((~local_success).sum()),
                "min_chunks": int(counts[scene_axes].min()),
                "max_chunks": max_chunks,
                "routes": routes,
                "physical_centroid_gap_m": _optional_round(endpoint_gap, 6),
                "routing_gap": _optional_round(routing_gap_by_scene[scene], 5),
                "contribution_gap": _optional_round(contribution_gap, 5),
                "active_counts": active_by_scene[scene].tolist(),
                "highlighted": highlighted,
                "pca_explained_fraction": round(
                    float(np.square(singular[:2]).sum() / np.square(singular).sum()),
                    5,
                ),
            }
        )
    return {
        "task": task,
        "default_scene": int(default_scene),
        "min_chunks": int(counts.min()),
        "max_chunks": int(counts.max()),
        "scenes": scene_entries,
    }


def trajectory_routes(
    result_dir: pathlib.Path,
    cache_root: pathlib.Path,
    tasks: list[dict],
    rebuild: bool = False,
) -> dict:
    cache = result_dir / "full-route-atlas.json"
    if cache.exists() and not rebuild:
        stored = json.loads(cache.read_text())
        if int(stored.get("version", -1)) == ROUTE_ATLAS_VERSION:
            return stored
    atlas = {
        "version": ROUTE_ATLAS_VERSION,
        "fixed_contribution_chunks": 9,
        "state": "end-effector xyz before each action chunk",
        "projection": "per-task/per-initial-state PCA from full xyz routes to two dimensions",
        "physical_gap": "success/failure centroid distance with completed routes held at their final recorded state",
        "routing_signal": "within-state active-risk-set percentile of leave-one-out distance from the successful HB top-4 routing occupancy centroid",
        "routing_churn": "within-state active-risk-set percentile of mean adjacent-denoise RMS change in HB top-4 routing occupancy",
        "routing_revision": "within-state active-risk-set percentile of d0-to-d9 RMS change in HB top-4 routing occupancy",
        "contribution_signal": "k0-k8 leave-one-out distance from the successful routed-contribution centroid",
        "tasks": [
            _load_full_task_routes(result_dir, cache_root, task["id"])
            for task in tasks
        ],
    }
    cache.write_text(json.dumps(atlas, ensure_ascii=False, separators=(",", ":")) + "\n")
    return atlas


def build_payload(
    result_dir: pathlib.Path,
    cache_root: pathlib.Path = CACHE_ROOT,
    rebuild_route_atlas: bool = False,
) -> dict:
    trend = json.loads((result_dir / "summary.json").read_text())
    curves = json.loads((result_dir / "success_failure_curves.json").read_text())
    counts = task_counts(result_dir)
    tasks = []
    for task in curves["tasks"]:
        if task not in counts:
            raise ValueError("missing compact count data for %s" % task)
        tasks.append({"id": task, **counts[task]})
    validation = [
        row for task_rows in trend["validation"].values() for row in task_rows
    ]
    payload = {
        "meta": {
            "analysis": trend["analysis"],
            "exploratory": bool(trend["exploratory"]),
            "chunks": int(curves["chunks"]),
            "anchor_chunk": int(curves["anchor_chunk"]),
            "bootstrap": int(curves["bootstrap"]),
            "layers": trend["layers"],
            "projected_dim": int(trend["projected_dim"]),
            "pca_components": int(trend["pca_components"]),
            "tasks": tasks,
            "rollouts": int(sum(task["rollouts"] for task in tasks)),
            "failure": int(sum(task["failure"] for task in tasks)),
            "success": int(sum(task["success"] for task in tasks)),
            "max_selected_probability_mae": float(
                max(row["selected_probability_mae"] for row in validation)
            ),
            "min_top4_set_match": float(
                min(row["top4_set_match"] for row in validation)
            ),
            "frozen_anchor_max_abs_diff": float(
                max(
                    row["anchor_probability_max_abs_diff"]
                    for row in curves["validation"].values()
                )
            ),
        },
        "curves": curves["curves"],
        "gaps": curves["gaps"],
        "by_task": curves["by_task"],
        "classifier_macro": [
            row for row in trend["classifier_macro"] if row["family"] in FAMILIES
        ],
        "centroid_macro": [
            row for row in trend["manifold_macro"] if row["family"] in FAMILIES
        ],
        "trends": trend["trends"],
        "descriptive_trends": trend["descriptive_trends"],
    }
    payload["dynamics"] = task_dynamics(result_dir, tasks)
    payload["trajectory_routes"] = trajectory_routes(
        result_dir, cache_root, tasks, rebuild=rebuild_route_atlas
    )
    if payload["meta"]["rollouts"] != 2048 or payload["meta"]["chunks"] != 9:
        raise ValueError("unexpected fixed-cohort dimensions")
    return payload


def main() -> None:
    args = parse_args()
    payload = build_payload(
        args.result_dir,
        args.cache_root,
        rebuild_route_atlas=args.rebuild_route_atlas,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "window.HIMOE_ROLLOUT_DEMO = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
