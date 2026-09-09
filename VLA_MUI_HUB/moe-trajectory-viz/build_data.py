#!/usr/bin/env python3
"""Build the static data bundle for the MoE control-step trajectory viewer.

The viewer compares one fixed action-token position across control queries.
Only the final denoise step is exported.  Each authoritative Top-4 expert set
is encoded as one uint32 bit mask, reducing the browser payload substantially
without changing set-overlap calculations.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
HUB = HERE.parent
TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
RUN = HUB / "cache" / "HiMoE-VLA" / "libero_long" / TASK / "right-16x32"
LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
DENOISE_STEP = 9
TOKEN_POSITIONS = tuple(range(1, 11))
WINDOW = 16
PCA_COMPONENTS = 2
PCA_FIT_LAST_STEP = 34


def episode_rows(summaries: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([row["inference_calls"] for row in summaries], dtype=np.int32)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    return offsets, lengths


def top4_masks(ids: np.ndarray) -> np.ndarray:
    if ids.dtype != np.uint8 or ids.shape[-1] != 4:
        raise ValueError(f"unexpected expert-id array: {ids.shape} {ids.dtype}")
    bits = np.left_shift(np.uint32(1), ids.astype(np.uint32))
    masks = np.bitwise_or.reduce(bits, axis=-1).astype("<u4", copy=False)
    if not np.all(np.bitwise_count(masks) == 4):
        raise ValueError("a stored Top-4 row does not contain four unique experts")
    return masks


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    common = np.bitwise_and(a, b)
    counts = np.bitwise_count(common)
    return counts.astype(np.float32) / 4.0


def rolling_persistence(path: np.ndarray, end: int) -> float:
    """Mean lag-1..5 overlap in a trailing 16-control window."""
    start = end - WINDOW + 1
    segment = path[start : end + 1]
    scores = [overlap(segment[lag:], segment[:-lag]).mean() for lag in range(1, 6)]
    return float(np.mean(scores))


def quantile(value: np.ndarray, q: float) -> float:
    return round(float(np.quantile(value, q)), 6)


def fit_pca_coordinates(fit_matrix: np.ndarray, project_matrix: np.ndarray) -> dict[str, Any]:
    """Fit one label-blind PCA on a balanced prefix and project the full group."""
    expected_width = len(LAYERS) * len(TOKEN_POSITIONS) * 32
    if fit_matrix.ndim != 2 or fit_matrix.shape[1] != expected_width:
        raise ValueError(f"unexpected PCA fit matrix: {fit_matrix.shape}")
    if project_matrix.ndim != 2 or project_matrix.shape[1] != expected_width:
        raise ValueError(f"unexpected PCA projection matrix: {project_matrix.shape}")
    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=0,
        iterated_power=5,
    )
    pca.fit(fit_matrix.astype(np.float32, copy=False))
    coordinates = pca.transform(project_matrix.astype(np.float32, copy=False))
    for component in range(PCA_COMPONENTS):
        pivot = int(np.argmax(np.abs(pca.components_[component])))
        if pca.components_[component, pivot] < 0:
            coordinates[:, component] *= -1

    max_abs = np.maximum(np.abs(coordinates).max(axis=0), 1e-8)
    scale = max_abs / 30000.0
    quantized = np.rint(coordinates / scale).astype("<i2")
    restored = quantized.astype(np.float32) * scale
    max_error = np.abs(restored - coordinates).max(axis=0)
    return {
        "fit_rows": int(len(fit_matrix)),
        "projected_rows": int(len(project_matrix)),
        "explained_variance_ratio": [
            round(float(value), 8) for value in pca.explained_variance_ratio_
        ],
        "quantization": "little-endian int16 multiplied by scale[component]",
        "scale": [float(value) for value in scale],
        "max_abs_error": [float(value) for value in max_error],
        "coords_base64": base64.b64encode(quantized.tobytes(order="C")).decode("ascii"),
    }


def fit_grouped_pca(
    matrix: np.ndarray,
    summaries: list[dict[str, Any]],
    offsets: np.ndarray,
    lengths: np.ndarray,
    representation: str,
) -> dict[str, Any]:
    """Fit one PCA per initial state, each over its 32 flow-noise rollouts."""
    scene_ids = np.asarray([row["init_state_id"] for row in summaries], dtype=np.int32)
    groups: dict[str, dict[str, Any]] = {}
    for scene in sorted(np.unique(scene_ids)):
        episode_indexes = np.flatnonzero(scene_ids == scene)
        if len(episode_indexes) != 32:
            raise ValueError(f"initial state {scene} has {len(episode_indexes)} episodes, not 32")
        first = int(episode_indexes[0])
        last = int(episode_indexes[-1])
        if not np.array_equal(episode_indexes, np.arange(first, last + 1)):
            raise ValueError(f"initial state {scene} episodes are not contiguous")
        row_offset = int(offsets[first])
        row_end = int(offsets[last] + lengths[last])
        if row_end - row_offset != int(lengths[episode_indexes].sum()):
            raise ValueError(f"initial state {scene} control rows are not contiguous")
        fit_steps = PCA_FIT_LAST_STEP + 1
        if np.any(lengths[episode_indexes] < fit_steps):
            raise ValueError(f"initial state {scene} has a trajectory ending before t{PCA_FIT_LAST_STEP}")
        fit_indexes = np.concatenate(
            [
                np.arange(offsets[index], offsets[index] + fit_steps, dtype=np.int64)
                for index in episode_indexes
            ]
        )
        group = fit_pca_coordinates(matrix[fit_indexes], matrix[row_offset:row_end])
        group.update(
            {
                "scene": int(scene),
                "episode_count": int(len(episode_indexes)),
                "row_offset": row_offset,
                "fit_control_steps": [0, PCA_FIT_LAST_STEP],
            }
        )
        groups[str(int(scene))] = group

    return {
        "representation": representation,
        "input_width": int(matrix.shape[1]),
        "grouping": (
            "one label-blind PCA per initial state; fit equally on each rollout's "
            f"t0-t{PCA_FIT_LAST_STEP} prefix, then project complete trajectories"
        ),
        "groups": groups,
    }


def hard_route_matrix(ids: np.ndarray) -> np.ndarray:
    """Convert authoritative Top-4 IDs to one multi-hot feature per expert."""
    one_hot = np.zeros(ids.shape[:-1] + (32,), dtype=np.uint8)
    np.put_along_axis(one_hot, ids.astype(np.int64), 1, axis=-1)
    return one_hot.reshape(len(ids), -1)


def build(output: Path) -> None:
    summaries_path = RUN / "client" / "summaries.json"
    summaries = sorted(
        json.loads(summaries_path.read_text()), key=lambda row: row["episode_index"]
    )
    offsets, lengths = episode_rows(summaries)
    if len(summaries) != 512:
        raise ValueError(f"expected 512 episodes, found {len(summaries)}")

    store = zarr.open(str(RUN / "server" / "routes.zarr"), mode="r")
    episode_id = np.asarray(store["episode_id"][:], dtype=np.int32)
    expected = np.repeat(
        np.asarray([row["episode_index"] for row in summaries], dtype=np.int32), lengths
    )
    if not np.array_equal(episode_id, expected):
        raise ValueError("episode_id does not align with summaries.json")

    ids = np.asarray(
        store["hb_expert_ids"][:, :, DENOISE_STEP, 1:11, :], dtype=np.uint8
    )
    if ids.shape != (int(lengths.sum()), len(LAYERS), len(TOKEN_POSITIONS), 4):
        raise ValueError(f"unexpected selected route shape: {ids.shape}")
    masks = top4_masks(ids)

    pca_top4 = fit_grouped_pca(
        hard_route_matrix(ids),
        summaries,
        offsets,
        lengths,
        "authoritative Top-4 multi-hot",
    )
    soft = np.asarray(
        store["hb_router_probs"][:, :, DENOISE_STEP, 1:11, :], dtype=np.float32
    )
    if soft.shape != (int(lengths.sum()), len(LAYERS), len(TOKEN_POSITIONS), 32):
        raise ValueError(f"unexpected selected soft-route shape: {soft.shape}")
    pca_soft = fit_grouped_pca(
        soft.reshape(len(soft), -1),
        summaries,
        offsets,
        lengths,
        "32-way soft probabilities",
    )

    references: dict[str, dict[str, dict[str, float | int]]] = {}
    success = np.asarray([row["success"] for row in summaries], dtype=bool)
    scene_ids = np.asarray([row["init_state_id"] for row in summaries], dtype=np.int32)
    for scene in sorted(np.unique(scene_ids)):
        scene_reference: dict[str, dict[str, float | int]] = {}
        for end in range(WINDOW - 1, int(lengths.max())):
            eligible = np.flatnonzero((scene_ids == scene) & success & (lengths > end))
            if not len(eligible):
                continue
            values = np.asarray(
                [
                    rolling_persistence(
                        masks[offsets[index] : offsets[index] + lengths[index]], end
                    )
                    for index in eligible
                ],
                dtype=np.float32,
            )
            scene_reference[str(end)] = {
                "n": int(len(values)),
                "p50": quantile(values, 0.50),
                "p75": quantile(values, 0.75),
                "p95": quantile(values, 0.95),
            }
        references[str(int(scene))] = scene_reference

    episodes = []
    for row, offset, length in zip(summaries, offsets, lengths):
        episodes.append(
            {
                "episode": int(row["episode_index"]),
                "scene": int(row["init_state_id"]),
                "noise": int(row["flow_noise_seed"]),
                "success": bool(row["success"]),
                "length": int(length),
                "action_steps": int(row["action_steps"]),
                "offset": int(offset),
            }
        )

    payload = {
        "schema": {
            "name": "himoe_moe_control_trajectory_v1",
            "version": 2,
            "encoding": "little-endian uint32 bit mask; one bit per selected expert",
            "axis_order": ["control_row", "hb_layer", "action_token"],
            "comparison": "same action-token position across control steps",
            "pca": (
                "one label-blind PCA per initial state, fit on the balanced 32 x t0-t34 "
                "pre-length-split prefix and used to project complete trajectories"
            ),
        },
        "source": {
            "task": TASK,
            "prompt": summaries[0]["prompt"],
            "run": "right-16x32",
            "suite": "libero_long",
            "route_store": "server/routes.zarr/hb_expert_ids",
            "denoise_step": DENOISE_STEP,
            "rows": int(lengths.sum()),
            "episodes": len(summaries),
            "initial_states": int(len(np.unique(scene_ids))),
            "rollouts_per_initial_state": 32,
        },
        "axes": {
            "layers": list(LAYERS),
            "tokens": list(TOKEN_POSITIONS),
            "experts": 32,
            "top_k": 4,
            "window": WINDOW,
            "lags": [1, 2, 3, 4, 5],
        },
        "landmarks": [
            {"step": 8, "label": "pose split"},
            {"step": 13, "label": "route split"},
            {"step": 35, "label": "survivor split"},
        ],
        "defaults": {"scene": 16, "failure_episode": 161, "success_episode": 163},
        "episodes": episodes,
        "success_reference": references,
        "pca": {"soft": pca_soft, "top4": pca_top4},
        "masks_base64": base64.b64encode(masks.tobytes(order="C")).decode("ascii"),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    print(
        f"wrote {output} ({output.stat().st_size:,} bytes; "
        f"{len(episodes)} episodes, {masks.size:,} route masks)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "data.json")
    args = parser.parse_args()
    build(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
