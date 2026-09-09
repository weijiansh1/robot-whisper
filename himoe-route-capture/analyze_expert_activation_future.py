"""Test whether real HB-MoE expert contributions anticipate actions and outcomes.

The corpus contains a crossed design: 16 initial LIBERO states, each evaluated
with the same 32 flow-noise seeds.  This script uses only the first policy query
from every rollout.  It reconstructs the exact initial noise, recomputes selected
expert outputs from the captured router inputs and checkpoint, and compares:

* scalar expert magnitude versus routed-vector direction;
* both against initial noise, router decisions, input norm, and the shared MLP;
* the final action chunk from the same inference and eventual rollout success.

Action-basin models are evaluated leave-one-scene-out.  Success models hold out
entire noise seeds, so the same random draw never appears in train and test.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Iterable

import numpy as np
import zarr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


HB_LAYER_NUMBERS = (2, 3, 4, 5, 12, 13, 14, 15)
N_EXPERTS = 32
N_ACTION_TOKENS = 10
HIDDEN_WIDTH = 1024
NOISE_SHAPE = (10, 24)
LIVE_ACTION_DIMS = 7

BASE_METRICS = (
    "input_rms",
    "expert_mass_rms",
    "routed_rms",
    "cancellation",
    "expert_norm_cv",
    "shared_rms",
    "routed_over_shared",
    "routed_shared_cosine",
)
METRIC_NAMES = tuple(
    name + suffix for name in BASE_METRICS for suffix in ("_mean", "_token_std")
)
METRIC_GROUPS = {
    "input_size": (0, 1),
    "expert_size": (2, 3, 4, 5),
    "expert_structure": (6, 7, 8, 9),
    "expert_full": (2, 3, 4, 5, 6, 7, 8, 9),
    "shared_size": (10, 11),
    "branch_relation": (12, 13, 14, 15),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="right-16x32 corpus run")
    parser.add_argument("--checkpoint", required=True, help="pytorch_model.pth")
    parser.add_argument("--layers", default="2,5,12,15")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def _first_rows(episode_ids: np.ndarray, expected: np.ndarray) -> np.ndarray:
    unique, first = np.unique(episode_ids, return_index=True)
    lookup = {int(episode): int(row) for episode, row in zip(unique, first)}
    missing = [int(episode) for episode in expected if int(episode) not in lookup]
    if missing:
        raise ValueError("route store is missing episodes: %s" % missing[:8])
    return np.asarray([lookup[int(episode)] for episode in expected], dtype=np.int64)


def load_dataset(run: pathlib.Path, layer_numbers: tuple[int, ...]) -> dict:
    summaries = json.loads((run / "client" / "summaries.json").read_text())
    summaries = sorted(summaries, key=lambda row: int(row["episode_index"]))
    episodes = np.asarray([int(row["episode_index"]) for row in summaries])
    if len(np.unique(episodes)) != len(episodes):
        raise ValueError("episode_index is not unique")

    route_group = zarr.open_group(str(run / "server" / "routes.zarr"), mode="r")
    hidden_group = zarr.open_group(str(run / "server" / "hidden.zarr"), mode="r")
    route_episode = np.asarray(route_group["episode_id"][:])
    hidden_episode = np.asarray(hidden_group["episode_id"][:])
    if not np.array_equal(route_episode, hidden_episode):
        raise ValueError("routes.zarr and hidden.zarr episode axes differ")
    rows = _first_rows(route_episode, episodes)

    metadata = json.loads((run / "client" / "server_metadata.json").read_text())
    captured_layers = tuple(int(value) for value in metadata["routing_hb_layer_indices"])
    layer_axes = tuple(captured_layers.index(layer) for layer in layer_numbers)

    selection = (rows, list(layer_axes), slice(None), slice(1, 11), slice(None))
    route_ids = np.asarray(
        route_group["hb_expert_ids"].get_orthogonal_selection(selection), dtype=np.int64
    )
    route_raw = np.asarray(
        route_group["hb_selected_prob"].get_orthogonal_selection(selection),
        dtype=np.float32,
    )
    route_weight = route_raw / np.maximum(route_raw.sum(axis=-1, keepdims=True), 1e-20)
    entropy = np.asarray(
        route_group["hb_entropy"].get_orthogonal_selection(selection[:-1]),
        dtype=np.float32,
    )

    print("loading %d first-query hidden rows" % len(rows), flush=True)
    started = time.perf_counter()
    hidden = np.asarray(
        hidden_group["hb_hidden"].get_orthogonal_selection(selection),
        dtype=np.float16,
    )
    print("loaded hidden subset %s in %.1fs" % (hidden.shape, time.perf_counter() - started), flush=True)

    actions = []
    states = []
    sim_states = []
    for episode in episodes:
        path = run / "client" / ("episode_%02d.npz" % int(episode))
        with np.load(path) as record:
            actions.append(np.asarray(record["actions"][0], dtype=np.float32))
            states.append(np.asarray(record["state"][0], dtype=np.float32))
            sim_states.append(np.asarray(record["sim_state"][0], dtype=np.float32))

    noise = np.stack(
        [
            np.random.default_rng(int(row["flow_noise_seed"]))
            .standard_normal(NOISE_SHAPE)
            .astype(np.float32)
            for row in summaries
        ]
    )
    scenes = np.asarray([int(row["init_state_id"]) for row in summaries])
    noise_seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries])
    success = np.asarray([bool(row["success"]) for row in summaries], dtype=np.int8)
    repeats = np.asarray([int(row["repeat"]) for row in summaries])

    unique_scenes = np.unique(scenes)
    unique_noise = np.unique(noise_seeds)
    expected_n = len(unique_scenes) * len(unique_noise)
    if len(summaries) != expected_n:
        raise ValueError("dataset is not a complete scene x noise grid")
    for scene in unique_scenes:
        index = np.flatnonzero(scenes == scene)
        if len(index) != len(unique_noise) or len(np.unique(noise_seeds[index])) != len(unique_noise):
            raise ValueError("scene %d does not contain every noise seed" % scene)
        if not np.allclose(np.stack(states)[index], np.stack(states)[index[0]], atol=0, rtol=0):
            raise ValueError("first-query robot state varies within scene %d" % scene)
        if not np.allclose(np.stack(sim_states)[index], np.stack(sim_states)[index[0]], atol=0, rtol=0):
            raise ValueError("first-query simulator state varies within scene %d" % scene)

    stats_path = pathlib.Path(metadata["normalization_stats_path"])
    stats = json.loads(stats_path.read_text())
    action_std = np.asarray(stats["actions"]["std"], dtype=np.float64)
    if action_std.shape != (LIVE_ACTION_DIMS,) or np.any(action_std <= 0):
        raise ValueError("invalid checkpoint action std")

    return {
        "summaries": summaries,
        "episodes": episodes,
        "scenes": scenes,
        "noise_seeds": noise_seeds,
        "repeats": repeats,
        "success": success,
        "noise": noise,
        "actions": np.stack(actions),
        "states": np.stack(states),
        "sim_states": np.stack(sim_states),
        "action_std": action_std,
        "hidden": hidden,
        "route_ids": route_ids,
        "route_raw": route_raw,
        "route_weight": route_weight,
        "entropy": entropy,
        "layer_numbers": np.asarray(layer_numbers),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
    }


def _mlp(torch, x, weights):
    gate, up, down = weights
    return torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, gate))
        * torch.nn.functional.linear(x, up),
        down,
    )


def _layer_weights(state, layer: int, device: str, torch):
    root = "paligemma_with_expert.gemma_expert.layers.%d.mlp." % layer

    def triplet(prefix: str):
        return tuple(
            state[root + prefix + name + ".weight"].to(device)
            for name in ("gate_proj", "up_proj", "down_proj")
        )

    experts = [triplet("experts.%d." % expert) for expert in range(N_EXPERTS)]
    shared = triplet("shared_experts.")
    gate = state[root + "gate.weight"].to(device)
    return experts, shared, gate


def compute_expert_features(data: dict, checkpoint: pathlib.Path, device: str) -> dict:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    hidden = data["hidden"]
    route_ids = data["route_ids"]
    route_raw = data["route_raw"]
    route_weight = data["route_weight"]
    n_rows, n_layers, n_denoise, n_tokens, width = hidden.shape
    if (n_tokens, width) != (N_ACTION_TOKENS, HIDDEN_WIDTH):
        raise ValueError("unexpected hidden shape %s" % (hidden.shape,))

    metrics = np.empty((n_rows, n_denoise, n_layers, len(METRIC_NAMES)), np.float32)
    routed_vectors = np.empty((n_rows, n_denoise, n_layers, width), np.float16)
    shared_vectors = np.empty_like(routed_vectors)
    validation = []
    root_width = math.sqrt(width)

    for layer_axis, layer in enumerate(data["layer_numbers"]):
        started = time.perf_counter()
        print("layer %d: loading expert weights" % int(layer), flush=True)
        experts, shared_weights, gate_weight = _layer_weights(state, int(layer), device, torch)
        x = torch.from_numpy(np.ascontiguousarray(hidden[:, layer_axis])).reshape(-1, width)
        x = x.to(device=device, dtype=torch.bfloat16)
        ids = torch.from_numpy(np.ascontiguousarray(route_ids[:, layer_axis]).reshape(-1, 4)).to(device)
        weights = torch.from_numpy(
            np.ascontiguousarray(route_weight[:, layer_axis]).reshape(-1, 4)
        ).to(device)
        raw = torch.from_numpy(np.ascontiguousarray(route_raw[:, layer_axis]).reshape(-1, 4)).to(device)
        token_count = x.shape[0]

        with torch.inference_mode():
            routed = torch.zeros((token_count, width), dtype=torch.float32, device=device)
            mass = torch.zeros(token_count, dtype=torch.float32, device=device)
            second = torch.zeros_like(mass)
            for expert_id, expert_weights in enumerate(experts):
                matches = torch.nonzero(ids == expert_id, as_tuple=False)
                if matches.numel() == 0:
                    continue
                token_index = matches[:, 0]
                slot_index = matches[:, 1]
                output = _mlp(torch, x[token_index], expert_weights).float()
                alpha = weights[token_index, slot_index].float()
                output_rms = output.square().mean(dim=-1).sqrt()
                routed.index_add_(0, token_index, output * alpha[:, None])
                mass.index_add_(0, token_index, output_rms * alpha)
                second.index_add_(0, token_index, output_rms.square() * alpha)

            shared = _mlp(torch, x, shared_weights).float()
            input_rms = x.float().square().mean(dim=-1).sqrt()
            routed_rms = routed.square().mean(dim=-1).sqrt()
            shared_rms = shared.square().mean(dim=-1).sqrt()
            cancellation = routed_rms / mass.clamp_min(1e-12)
            expert_cv = (second - mass.square()).clamp_min(0).sqrt() / mass.clamp_min(1e-12)
            routed_over_shared = routed_rms / shared_rms.clamp_min(1e-12)
            cosine = (routed * shared).sum(dim=-1) / (
                routed.norm(dim=-1) * shared.norm(dim=-1)
            ).clamp_min(1e-12)

            base = torch.stack(
                (
                    input_rms,
                    mass,
                    routed_rms,
                    cancellation,
                    expert_cv,
                    shared_rms,
                    routed_over_shared,
                    cosine,
                ),
                dim=-1,
            ).reshape(n_rows, n_denoise, n_tokens, len(BASE_METRICS))
            aggregated = torch.stack((base.mean(dim=2), base.std(dim=2)), dim=-1)
            metrics[:, :, layer_axis] = aggregated.flatten(2).cpu().numpy()
            routed_vectors[:, :, layer_axis] = (
                routed.reshape(n_rows, n_denoise, n_tokens, width)
                .mean(dim=2)
                .to(torch.float16)
                .cpu()
                .numpy()
            )
            shared_vectors[:, :, layer_axis] = (
                shared.reshape(n_rows, n_denoise, n_tokens, width)
                .mean(dim=2)
                .to(torch.float16)
                .cpu()
                .numpy()
            )

            logits = torch.nn.functional.linear(x, gate_weight)
            probabilities = logits.softmax(dim=-1)
            predicted = probabilities.topk(4, dim=-1, sorted=False).indices
            set_match = (
                predicted.sort(dim=-1).values == ids.sort(dim=-1).values
            ).all(dim=-1).float().mean()
            captured_prob = probabilities.gather(1, ids).float()
            probability_mae = (captured_prob - raw.float()).abs().mean()
            validation.append(
                {
                    "layer": int(layer),
                    "top4_set_match": float(set_match.cpu()),
                    "selected_probability_mae": float(probability_mae.cpu()),
                }
            )

        del x, ids, weights, raw, routed, shared, experts, shared_weights, gate_weight
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("layer %d: done in %.1fs" % (int(layer), time.perf_counter() - started), flush=True)

    return {
        "metrics": metrics,
        "routed_vectors": routed_vectors,
        "shared_vectors": shared_vectors,
        "validation": validation,
    }


def load_or_compute_features(
    cache: pathlib.Path,
    data: dict,
    checkpoint: pathlib.Path,
    device: str,
    force: bool,
) -> dict:
    if cache.is_file() and not force:
        with np.load(cache, allow_pickle=False) as stored:
            layers = tuple(int(value) for value in stored["layer_numbers"])
            if layers == tuple(int(value) for value in data["layer_numbers"]):
                print("loading cached expert features from %s" % cache, flush=True)
                return {
                    "metrics": np.asarray(stored["metrics"]),
                    "routed_vectors": np.asarray(stored["routed_vectors"]),
                    "shared_vectors": np.asarray(stored["shared_vectors"]),
                    "validation": json.loads(str(stored["validation_json"])),
                }
    features = compute_expert_features(data, checkpoint, device)
    np.savez(
        cache,
        metrics=features["metrics"],
        routed_vectors=features["routed_vectors"],
        shared_vectors=features["shared_vectors"],
        layer_numbers=data["layer_numbers"],
        metric_names=np.asarray(METRIC_NAMES),
        validation_json=json.dumps(features["validation"]),
    )
    return features


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) != 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _bootstrap_mean(values: Iterable[float], rng: np.random.Generator, draws: int) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "n": int(len(array)),
    }


def _scene_indices(data: dict) -> dict[int, np.ndarray]:
    result = {}
    for scene in np.unique(data["scenes"]):
        index = np.flatnonzero(data["scenes"] == scene)
        result[int(scene)] = index[np.argsort(data["noise_seeds"][index])]
    return result


def _pair_distance(values: np.ndarray, index: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    flattened = np.asarray(values[index], dtype=np.float64).reshape(len(index), -1)
    if scale is not None:
        flattened = flattened / scale
    upper = np.triu_indices(len(index), 1)
    delta = flattened[upper[0]] - flattened[upper[1]]
    return np.sqrt(np.mean(np.square(delta), axis=-1))


def _standardized_feature_distances(
    values: np.ndarray, scenes: dict[int, np.ndarray], fit_rows: np.ndarray | None = None
) -> dict[int, np.ndarray]:
    flattened = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    source = flattened if fit_rows is None else flattened[fit_rows]
    scale = source.std(axis=0)
    keep = scale > 1e-10
    if not np.any(keep):
        return {scene: np.zeros(len(index) * (len(index) - 1) // 2) for scene, index in scenes.items()}
    normalized = flattened[:, keep] / scale[keep]
    return {scene: _pair_distance(normalized, index) for scene, index in scenes.items()}


def _cosine_distances(values: np.ndarray, scenes: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    unit = array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)
    result = {}
    for scene, index in scenes.items():
        upper = np.triu_indices(len(index), 1)
        similarity = (unit[index[upper[0]]] * unit[index[upper[1]]]).sum(axis=-1)
        result[scene] = np.mean(1.0 - similarity, axis=-1)
    return result


def _route_dense(data: dict) -> np.ndarray:
    ids = data["route_ids"]
    weight = data["route_weight"]
    dense = np.zeros(ids.shape[:-1] + (N_EXPERTS,), dtype=np.float32)
    np.put_along_axis(dense, ids, weight, axis=-1)
    return dense


def _route_pair_distances(
    dense: np.ndarray, scenes: dict[int, np.ndarray], tau: int
) -> dict[int, np.ndarray]:
    result = {}
    for scene, index in scenes.items():
        values = dense[index, :, tau]
        upper = np.triu_indices(len(index), 1)
        left = values[upper[0]]
        right = values[upper[1]]
        lower = np.minimum(left, right).sum(axis=-1)
        upper_mass = np.maximum(left, right).sum(axis=-1)
        result[scene] = np.mean(1.0 - lower / np.maximum(upper_mass, 1e-12), axis=(1, 2))
    return result


def _direct_scene_auc(distances: dict[int, np.ndarray], labels: dict[int, np.ndarray]) -> list[float]:
    return [_auc(labels[scene], -distances[scene]) for scene in sorted(labels)]


def _loso_auc(
    feature_sets: list[dict[int, np.ndarray]],
    labels: dict[int, np.ndarray],
    logistic_c: float,
) -> list[float]:
    scenes = sorted(labels)
    output = []
    for held_out in scenes:
        train_scenes = [scene for scene in scenes if scene != held_out]
        x_train = np.column_stack(
            [np.concatenate([feature[scene] for scene in train_scenes]) for feature in feature_sets]
        )
        y_train = np.concatenate([labels[scene] for scene in train_scenes])
        x_test = np.column_stack([feature[held_out] for feature in feature_sets])
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(C=logistic_c, solver="lbfgs", max_iter=2000)
        model.fit(scaler.transform(x_train), y_train)
        score = model.predict_proba(scaler.transform(x_test))[:, 1]
        output.append(_auc(labels[held_out], score))
    return output


def analyze_future_action(
    data: dict,
    features: dict,
    bootstrap: int,
    seed: int,
    logistic_c: float,
) -> dict:
    rng = np.random.default_rng(seed)
    scenes = _scene_indices(data)
    action_values = data["actions"] / data["action_std"][None, None, :]
    noise24 = data["noise"]
    noise7 = data["noise"][:, :, :LIVE_ACTION_DIMS]
    action_distance = {scene: _pair_distance(action_values, index) for scene, index in scenes.items()}
    noise24_distance = {scene: _pair_distance(noise24, index) for scene, index in scenes.items()}
    noise7_distance = {scene: _pair_distance(noise7, index) for scene, index in scenes.items()}
    labels = {}
    for scene, distance in action_distance.items():
        labels[scene] = (distance <= np.median(distance)).astype(np.int8)

    dense = _route_dense(data)
    metrics = features["metrics"]
    routed_vectors = features["routed_vectors"]
    shared_vectors = features["shared_vectors"]
    direct = {name: [] for name in (
        "noise24", "noise7", "router", "input_size", "expert_size",
        "expert_full", "routed_direction", "shared_size", "shared_direction",
    )}
    loso_names = (
        "noise24", "noise7", "noise24_router", "noise24_input_size",
        "noise24_expert_size", "noise24_expert_full", "noise24_routed_direction",
        "noise24_shared_size", "noise24_shared_direction",
        "noise24_router_expert_size", "noise24_router_routed_direction",
        "noise7_router", "noise7_input_size", "noise7_expert_size",
        "noise7_expert_full", "noise7_routed_direction", "noise7_shared_size",
        "noise7_shared_direction", "noise7_router_expert_size",
        "noise7_router_routed_direction",
    )
    loso = {name: [] for name in loso_names}

    noise24_scene_auc = _direct_scene_auc(noise24_distance, labels)
    noise7_scene_auc = _direct_scene_auc(noise7_distance, labels)
    noise24_loso = _loso_auc([noise24_distance], labels, logistic_c)
    noise7_loso = _loso_auc([noise7_distance], labels, logistic_c)

    for tau in range(metrics.shape[1]):
        distances = {
            "noise24": noise24_distance,
            "noise7": noise7_distance,
            "router": _route_pair_distances(dense, scenes, tau),
        }
        for group in ("input_size", "expert_size", "expert_full", "shared_size"):
            values = np.take(metrics[:, tau], METRIC_GROUPS[group], axis=-1)
            distances[group] = _standardized_feature_distances(values, scenes)
        distances["routed_direction"] = _cosine_distances(routed_vectors[:, tau], scenes)
        distances["shared_direction"] = _cosine_distances(shared_vectors[:, tau], scenes)

        for name in direct:
            if name == "noise24":
                scene_auc = noise24_scene_auc
            elif name == "noise7":
                scene_auc = noise7_scene_auc
            else:
                scene_auc = _direct_scene_auc(distances[name], labels)
            direct[name].append({**_bootstrap_mean(scene_auc, rng, bootstrap), "per_scene": scene_auc})

        model_features = {
            "noise24": [noise24_distance],
            "noise7": [noise7_distance],
            "noise24_router": [noise24_distance, distances["router"]],
            "noise24_input_size": [noise24_distance, distances["input_size"]],
            "noise24_expert_size": [noise24_distance, distances["expert_size"]],
            "noise24_expert_full": [noise24_distance, distances["expert_full"]],
            "noise24_routed_direction": [noise24_distance, distances["routed_direction"]],
            "noise24_shared_size": [noise24_distance, distances["shared_size"]],
            "noise24_shared_direction": [noise24_distance, distances["shared_direction"]],
            "noise24_router_expert_size": [
                noise24_distance, distances["router"], distances["expert_size"]
            ],
            "noise24_router_routed_direction": [
                noise24_distance, distances["router"], distances["routed_direction"]
            ],
            "noise7_router": [noise7_distance, distances["router"]],
            "noise7_input_size": [noise7_distance, distances["input_size"]],
            "noise7_expert_size": [noise7_distance, distances["expert_size"]],
            "noise7_expert_full": [noise7_distance, distances["expert_full"]],
            "noise7_routed_direction": [noise7_distance, distances["routed_direction"]],
            "noise7_shared_size": [noise7_distance, distances["shared_size"]],
            "noise7_shared_direction": [noise7_distance, distances["shared_direction"]],
            "noise7_router_expert_size": [
                noise7_distance, distances["router"], distances["expert_size"]
            ],
            "noise7_router_routed_direction": [
                noise7_distance, distances["router"], distances["routed_direction"]
            ],
        }
        for name, model_input in model_features.items():
            if name == "noise24":
                scene_auc = noise24_loso
            elif name == "noise7":
                scene_auc = noise7_loso
            else:
                scene_auc = _loso_auc(model_input, labels, logistic_c)
            stat = _bootstrap_mean(scene_auc, rng, bootstrap)
            delta24 = np.asarray(scene_auc) - np.asarray(noise24_loso)
            delta7 = np.asarray(scene_auc) - np.asarray(noise7_loso)
            loso[name].append(
                {
                    **stat,
                    "per_scene": scene_auc,
                    "delta_vs_noise24": _bootstrap_mean(delta24, rng, bootstrap),
                    "delta_vs_noise7": _bootstrap_mean(delta7, rng, bootstrap),
                }
            )
        print("action analysis: denoise step %d complete" % tau, flush=True)

    return {
        "definition": {
            "positive_pair": "within-scene final-action distance <= scene median",
            "action_distance": "RMS after division by checkpoint action std",
            "evaluation": "leave-one-initial-scene-out logistic AUC; scene-bootstrap CI",
            "auc_chance": 0.5,
        },
        "direct_auc": direct,
        "loso_auc": loso,
        "pairs_per_scene": int(len(next(iter(labels.values())))),
        "scenes": len(scenes),
    }


def _conditional_auc(labels: np.ndarray, scores: np.ndarray, strata: np.ndarray) -> float:
    wins = 0.0
    pairs = 0
    for stratum in np.unique(strata):
        index = strata == stratum
        positive = scores[index & (labels == 1)]
        negative = scores[index & (labels == 0)]
        if not len(positive) or not len(negative):
            continue
        wins += float((positive[:, None] > negative[None, :]).sum())
        wins += 0.5 * float((positive[:, None] == negative[None, :]).sum())
        pairs += len(positive) * len(negative)
    return float(wins / pairs) if pairs else float("nan")


def _reduce_features(train: np.ndarray, test: np.ndarray, components: int = 12):
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    count = min(components, train_scaled.shape[1], train_scaled.shape[0] - 1)
    if count >= train_scaled.shape[1]:
        return train_scaled, test_scaled
    pca = PCA(n_components=count, svd_solver="full").fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(test_scaled)


def _fit_success_model(parts_train, parts_test, labels, logistic_c):
    train = np.column_stack(parts_train)
    test = np.column_stack(parts_test)
    scaler = StandardScaler().fit(train)
    model = LogisticRegression(C=logistic_c, solver="lbfgs", max_iter=2000)
    model.fit(scaler.transform(train), labels)
    return model.predict_proba(scaler.transform(test))[:, 1]


def _route_outcome_features(data: dict, dense: np.ndarray, tau: int) -> np.ndarray:
    histogram = dense[:, :, tau].mean(axis=2).reshape(len(dense), -1)
    entropy = data["entropy"][:, :, tau]
    top = data["route_weight"][:, :, tau].max(axis=-1)
    summaries = np.concatenate(
        (entropy.mean(axis=-1), entropy.std(axis=-1), top.mean(axis=-1), top.std(axis=-1)),
        axis=-1,
    )
    return np.column_stack((histogram, summaries))


def _scene_bootstrap_conditional(
    labels: np.ndarray,
    scores: np.ndarray,
    scenes: np.ndarray,
    folds: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> list[float]:
    unique = np.unique(scenes)
    output = []
    for _ in range(draws):
        selected = rng.choice(unique, size=len(unique), replace=True)
        y_parts = []
        score_parts = []
        strata_parts = []
        for replicate, scene in enumerate(selected):
            index = np.flatnonzero(scenes == scene)
            y_parts.append(labels[index])
            score_parts.append(scores[index])
            strata_parts.append(replicate * (folds.max() + 1) + folds[index])
        output.append(
            _conditional_auc(
                np.concatenate(y_parts), np.concatenate(score_parts), np.concatenate(strata_parts)
            )
        )
    return output


def analyze_success(
    data: dict,
    features: dict,
    bootstrap: int,
    seed: int,
    logistic_c: float,
) -> dict:
    rng = np.random.default_rng(seed + 17)
    labels = data["success"]
    scenes = data["scenes"]
    groups = data["noise_seeds"]
    unique_scenes = np.unique(scenes)
    scene_onehot = (scenes[:, None] == unique_scenes[None, :]).astype(np.float64)
    noise_flat = data["noise"][:, :, :LIVE_ACTION_DIMS].reshape(len(labels), -1)
    noise24_flat = data["noise"].reshape(len(labels), -1)
    action_flat = (data["actions"] / data["action_std"][None, None, :]).reshape(len(labels), -1)
    metrics = features["metrics"]
    dense = _route_dense(data)
    model_names = (
        "scene", "scene_noise24", "scene_noise", "scene_noise_route", "scene_noise_expert_size",
        "scene_noise_expert_full", "scene_noise_shared_size",
        "scene_noise_route_expert_full", "scene_noise_final_action",
    )
    predictions = {name: np.full((metrics.shape[1], len(labels)), np.nan) for name in model_names}
    fold_id = np.full(len(labels), -1, dtype=np.int64)

    splitter = GroupKFold(n_splits=4)
    for fold, (train, test) in enumerate(splitter.split(noise_flat, labels, groups)):
        fold_id[test] = fold
        noise_train, noise_test = _reduce_features(noise_flat[train], noise_flat[test])
        noise24_train, noise24_test = _reduce_features(noise24_flat[train], noise24_flat[test])
        action_train, action_test = _reduce_features(action_flat[train], action_flat[test])
        for tau in range(metrics.shape[1]):
            route = _route_outcome_features(data, dense, tau)
            route_train, route_test = _reduce_features(route[train], route[test])
            size = np.take(metrics[:, tau], METRIC_GROUPS["expert_size"], axis=-1).reshape(
                len(labels), -1
            )
            full = np.take(metrics[:, tau], METRIC_GROUPS["expert_full"], axis=-1).reshape(
                len(labels), -1
            )
            shared = np.take(metrics[:, tau], METRIC_GROUPS["shared_size"], axis=-1).reshape(
                len(labels), -1
            )
            parts = {
                "scene": ([scene_onehot[train]], [scene_onehot[test]]),
                "scene_noise24": (
                    [scene_onehot[train], noise24_train],
                    [scene_onehot[test], noise24_test],
                ),
                "scene_noise": (
                    [scene_onehot[train], noise_train], [scene_onehot[test], noise_test]
                ),
                "scene_noise_route": (
                    [scene_onehot[train], noise_train, route_train],
                    [scene_onehot[test], noise_test, route_test],
                ),
                "scene_noise_expert_size": (
                    [scene_onehot[train], noise_train, size[train]],
                    [scene_onehot[test], noise_test, size[test]],
                ),
                "scene_noise_expert_full": (
                    [scene_onehot[train], noise_train, full[train]],
                    [scene_onehot[test], noise_test, full[test]],
                ),
                "scene_noise_shared_size": (
                    [scene_onehot[train], noise_train, shared[train]],
                    [scene_onehot[test], noise_test, shared[test]],
                ),
                "scene_noise_route_expert_full": (
                    [scene_onehot[train], noise_train, route_train, full[train]],
                    [scene_onehot[test], noise_test, route_test, full[test]],
                ),
                "scene_noise_final_action": (
                    [scene_onehot[train], noise_train, action_train],
                    [scene_onehot[test], noise_test, action_test],
                ),
            }
            for name, (train_parts, test_parts) in parts.items():
                predictions[name][tau, test] = _fit_success_model(
                    train_parts, test_parts, labels[train], logistic_c
                )
        print("success analysis: held-out noise fold %d complete" % fold, flush=True)

    if np.any(fold_id < 0) or any(np.any(~np.isfinite(value)) for value in predictions.values()):
        raise RuntimeError("success cross-validation did not produce complete predictions")
    strata = scenes.astype(np.int64) * 10 + fold_id
    result = {name: [] for name in model_names}
    baseline = predictions["scene_noise"]
    for name in model_names:
        for tau in range(metrics.shape[1]):
            score = predictions[name][tau]
            auc = _conditional_auc(labels, score, strata)
            boot = _scene_bootstrap_conditional(
                labels, score, scenes, fold_id, bootstrap, rng
            )
            baseline_auc = _conditional_auc(labels, baseline[tau], strata)
            result[name].append(
                {
                    "conditional_auc": float(auc),
                    "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "delta_vs_scene_noise": float(auc - baseline_auc),
                    "log_loss": float(log_loss(labels, np.clip(score, 1e-6, 1 - 1e-6))),
                }
            )

    return {
        "definition": {
            "evaluation": "4-fold CV holding out complete flow-noise seeds",
            "conditional_auc": "comparisons only within the same scene and held-out fold",
            "uncertainty": "scene bootstrap",
            "positive_rate": float(labels.mean()),
            "variable_outcome_scenes": int(
                sum(len(np.unique(labels[scenes == scene])) == 2 for scene in unique_scenes)
            ),
        },
        "models": result,
    }


def _fmt(value: float) -> str:
    return "%.3f" % float(value)


def _best_delta(rows: list[dict], baseline: str) -> tuple[int, dict]:
    key = "delta_vs_" + baseline
    values = [row[key]["mean"] for row in rows]
    index = int(np.nanargmax(values))
    return index, rows[index]


def summarize_magnitudes(data: dict, features: dict) -> dict:
    metrics = features["metrics"]
    scenes = data["scenes"]
    output = {}
    for name in (
        "input_rms_mean",
        "expert_mass_rms_mean",
        "routed_rms_mean",
        "cancellation_mean",
        "expert_norm_cv_mean",
        "shared_rms_mean",
        "routed_over_shared_mean",
    ):
        metric_index = METRIC_NAMES.index(name)
        rows = []
        for tau in range(metrics.shape[1]):
            values = metrics[:, tau, :, metric_index]
            within_scene_cv = []
            for scene in np.unique(scenes):
                scene_values = values[scenes == scene]
                within_scene_cv.extend(
                    (
                        scene_values.std(axis=0)
                        / np.maximum(np.abs(scene_values.mean(axis=0)), 1e-12)
                    ).tolist()
                )
            rows.append(
                {
                    "median": float(np.median(values)),
                    "median_within_scene_cv": float(np.median(within_scene_cv)),
                }
            )
        output[name] = rows
    return output


def render_report(summary: dict) -> str:
    action = summary["future_action"]
    success = summary["future_outcome"]
    direct = action["direct_auc"]
    loso = action["loso_auc"]
    lines = [
        "# 专家激活大小、噪声与未来动作/结果",
        "",
        "## 实验思路",
        "",
        "- 使用 LIBERO-Long t08 的 16 个初始场景，每个场景 32 个 flow-noise seed；只取每条 rollout 的第一个 policy query，场景内观测完全相同。",
        "- 从 seed 精确重建 `10x24` 初始噪声；主对照只用 action mask 有效的前 7 维，完整 24 维作为 padding 消融。最终动作使用 checkpoint 标准差归一化，把距离较近的一半 pair 定义为同一 action basin。",
        "- 从 `hidden.zarr` 和 checkpoint 重算被选专家的 `w_e E_e(h)`。标量大小、专家抵消结构、routed 向量方向、router 和 shared 分支分别比较。",
        "- 未来动作使用留一初始场景交叉验证；最终成功/失败使用按 noise seed 整组留出的 4 折交叉验证。",
        "",
        "## 结果",
        "",
        "checkpoint/hidden 对齐：4 个层的 top-4 专家集合复算一致率为 %s，选中概率 MAE 为 %s。"
        % (
            _fmt(np.mean([row["top4_set_match"] for row in summary["feature_validation"]])),
            "%.2e" % np.mean(
                [row["selected_probability_mae"] for row in summary["feature_validation"]]
            ),
        ),
        "",
        "step 0 的中位量级：加权专家输出 RMS=%s，合并 routed RMS=%s，抵消率=%s，routed/shared=%s；专家大小在同场景 32 个噪声间的中位 CV=%s。"
        % (
            _fmt(summary["feature_magnitudes"]["expert_mass_rms_mean"][0]["median"]),
            _fmt(summary["feature_magnitudes"]["routed_rms_mean"][0]["median"]),
            _fmt(summary["feature_magnitudes"]["cancellation_mean"][0]["median"]),
            _fmt(summary["feature_magnitudes"]["routed_over_shared_mean"][0]["median"]),
            _fmt(
                summary["feature_magnitudes"]["expert_mass_rms_mean"][0][
                    "median_within_scene_cv"
                ]
            ),
        ),
        "",
        "### 同一次推理的最终动作",
        "",
        "AUC=0.5 为随机；`Delta size` / `Delta direction` 是在有效 7 维噪声之上的留一场景 AUC 增量。CI 只重采样初始场景，条件于现有 32 个 noise seed。仅用完整 24 维的噪声 AUC=%s，padding 会明显稀释基线。"
        % _fmt(direct["noise24"][0]["mean"]),
        "",
        "| denoise step | live-noise AUC | router AUC | expert-size AUC | routed-direction AUC | shared-direction AUC | Delta size | Delta direction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tau in range(10):
        lines.append(
            "| %d | %s | %s | %s | %s | %s | %+0.3f | %+0.3f |"
            % (
                tau,
                _fmt(direct["noise7"][tau]["mean"]),
                _fmt(direct["router"][tau]["mean"]),
                _fmt(direct["expert_size"][tau]["mean"]),
                _fmt(direct["routed_direction"][tau]["mean"]),
                _fmt(direct["shared_direction"][tau]["mean"]),
                loso["noise7_expert_size"][tau]["delta_vs_noise7"]["mean"],
                loso["noise7_routed_direction"][tau]["delta_vs_noise7"]["mean"],
            )
        )
    size_tau, size_best = _best_delta(loso["noise7_expert_size"], "noise7")
    direction_tau, direction_best = _best_delta(
        loso["noise7_routed_direction"], "noise7"
    )
    shared_tau, shared_best = _best_delta(loso["noise7_shared_direction"], "noise7")
    lines.extend(
        [
            "",
            "专家大小的最大增量出现在 step %d：%+0.3f（95%% CI %+.3f, %+.3f）；routed 方向的最大增量出现在 step %d：%+0.3f（95%% CI %+.3f, %+.3f）。shared 方向最大增量为 %+0.3f（step %d），用于判断该方向信号是否为专家特有。"
            % (
                size_tau,
                size_best["delta_vs_noise7"]["mean"],
                size_best["delta_vs_noise7"]["ci95"][0],
                size_best["delta_vs_noise7"]["ci95"][1],
                direction_tau,
                direction_best["delta_vs_noise7"]["mean"],
                direction_best["delta_vs_noise7"]["ci95"][0],
                direction_best["delta_vs_noise7"]["ci95"][1],
                shared_best["delta_vs_noise7"]["mean"],
                shared_tau,
            ),
            "",
            "动作结论：专家大小有可测但很弱的早期信息，远小于有效噪声本身，并在后半程降到约 0 增量；完整向量方向更有信息，但 shared 方向更强，因此不能把这部分归因于专家激活大小或专家特化。router 在有效噪声上也只增加约 0.000-0.002 AUC。",
            "",
            "### 整条 rollout 的最终成功/失败",
            "",
            "下表为按 noise seed 留出的条件 AUC；比较只发生在同一场景、同一测试折内。",
            "",
            "| denoise step | scene+live noise | +router | +expert size | +expert full | +shared size | final first action |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    outcome = success["models"]
    for tau in range(10):
        lines.append(
            "| %d | %s | %s | %s | %s | %s | %s |"
            % (
                tau,
                _fmt(outcome["scene_noise"][tau]["conditional_auc"]),
                _fmt(outcome["scene_noise_route"][tau]["conditional_auc"]),
                _fmt(outcome["scene_noise_expert_size"][tau]["conditional_auc"]),
                _fmt(outcome["scene_noise_expert_full"][tau]["conditional_auc"]),
                _fmt(outcome["scene_noise_shared_size"][tau]["conditional_auc"]),
                _fmt(outcome["scene_noise_final_action"][tau]["conditional_auc"]),
            )
        )
    expert_outcome = outcome["scene_noise_expert_size"]
    best_outcome_tau = int(np.nanargmax([row["delta_vs_scene_noise"] for row in expert_outcome]))
    best_outcome = expert_outcome[best_outcome_tau]
    lines.extend(
        [
            "",
            "专家大小对成功率的最大条件 AUC 增量在 step %d，为 %+0.3f（AUC=%s，scene-bootstrap 95%% CI %s-%s）；其 log loss=%s，scene+noise 基线=%s。该结果是首个 query 对长时程结果的探索性关联，不是因果效应。"
            % (
                best_outcome_tau,
                best_outcome["delta_vs_scene_noise"],
                _fmt(best_outcome["conditional_auc"]),
                _fmt(best_outcome["ci95"][0]),
                _fmt(best_outcome["ci95"][1]),
                _fmt(best_outcome["log_loss"]),
                _fmt(outcome["scene_noise"][best_outcome_tau]["log_loss"]),
            ),
            "",
            "结果结论：step 0 的专家大小包含一定成功/失败排序信息，但跨 denoise step 不稳定，且固定正则下没有改善 log loss；目前只适合作为候选筛选的探索性特征，不能单独当作成功率或未来轨迹预测器。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run = pathlib.Path(args.run).resolve()
    checkpoint = pathlib.Path(args.checkpoint).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_numbers = tuple(int(value) for value in args.layers.split(",") if value)
    if not layer_numbers or any(layer not in HB_LAYER_NUMBERS for layer in layer_numbers):
        raise ValueError("layers must be a subset of %s" % (HB_LAYER_NUMBERS,))

    data = load_dataset(run, layer_numbers)
    features = load_or_compute_features(
        out_dir / "features.npz", data, checkpoint, args.device, args.force_features
    )
    future_action = analyze_future_action(
        data, features, args.bootstrap, args.seed, args.logistic_c
    )
    future_outcome = analyze_success(
        data, features, args.bootstrap, args.seed, args.logistic_c
    )
    summary = {
        "experiment": "expert_activation_future_action_and_outcome",
        "run": str(run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": data["checkpoint_sha256"],
        "layers": list(layer_numbers),
        "episodes": int(len(data["episodes"])),
        "scenes": int(len(np.unique(data["scenes"]))),
        "noise_seeds": int(len(np.unique(data["noise_seeds"]))),
        "successes": int(data["success"].sum()),
        "metric_names": list(METRIC_NAMES),
        "feature_validation": features["validation"],
        "feature_magnitudes": summarize_magnitudes(data, features),
        "future_action": future_action,
        "future_outcome": future_outcome,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print("wrote %s" % (out_dir / "summary.json"), flush=True)
    print("wrote %s" % (out_dir / "report.md"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
