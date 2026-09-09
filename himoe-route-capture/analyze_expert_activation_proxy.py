"""Retrospective hidden-matched and block-sensitivity MoE proxy.

This deliberately does not modify the frozen expert-activation-v2 outputs.
See EXPERT_ACTIVATION_PROXY_DESIGN.md for the frozen exploratory protocol and
the boundary between an offline proxy and a runtime causal intervention.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np
import zarr
from scipy.stats import rankdata

from analyze_expert_activation_future import _first_rows, _layer_weights, _mlp


LIVE_ACTION_DIMS = 7
N_EXPERTS = 32
N_ACTION_TOKENS = 10
HIDDEN_WIDTH = 1024
DISCOVERY_TWIN = (
    "dcq_level",
    "disagreement_level",
    "cancellation_level",
    "conflict_level",
    "dcq_diff",
    "disagreement_diff",
    "cancellation_diff",
    "conflict_diff",
)
DISCOVERY_FRAGILITY = ("dcq", "disagreement", "cancellation", "conflict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--denoise", type=int, default=0)
    parser.add_argument("--near-fraction", type=float, default=0.10)
    parser.add_argument("--tail-fraction", type=float, default=0.20)
    parser.add_argument("--perms", type=int, default=5000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def _rms(value: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(value, dtype=np.float64), axis=axis))


def _pair_rms(value: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, value.ndim))
    return _rms(value[pair_i] - value[pair_j], axis=axes)


def _pair_cosine(value: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    vector /= np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)
    cosine = np.sum(vector[pair_i] * vector[pair_j], axis=-1)
    return np.mean(1.0 - cosine, axis=tuple(range(1, cosine.ndim)))


def _fast_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    """P(score_positive > score_negative), with half credit for ties."""
    negative = np.asarray(negative, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))


def _scene_bootstrap(
    scene_values: np.ndarray, draws: int, rng: np.random.Generator
) -> list[float]:
    n_scene = scene_values.shape[-1]
    index = rng.integers(0, n_scene, size=(draws, n_scene))
    sampled = np.take(scene_values, index, axis=-1)
    mean = np.nanmean(sampled, axis=-1)
    return [float(np.nanpercentile(mean, 2.5)), float(np.nanpercentile(mean, 97.5))]


def _rank_rows(value: np.ndarray) -> np.ndarray:
    return np.stack([rankdata(row, method="average") for row in value]).astype(np.float64)


def _row_standardize(value: np.ndarray) -> np.ndarray:
    centered = value - value.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def _resolve_episode_file(client: pathlib.Path, episode: int) -> pathlib.Path:
    for width in (2, 4):
        path = client / (f"episode_{episode:0{width}d}.npz")
        if path.is_file():
            return path
    raise FileNotFoundError(f"no episode archive for {episode} in {client}")


def load_primary_cell(run: pathlib.Path, layer: int, denoise: int) -> dict[str, Any]:
    client = run / "client"
    server = run / "server"
    summaries = sorted(
        json.loads((client / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    episodes = np.asarray([int(row["episode_index"]) for row in summaries], dtype=np.int64)
    scenes = np.asarray([int(row["init_state_id"]) for row in summaries], dtype=np.int64)
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries], dtype=np.int64)
    metadata = json.loads((client / "server_metadata.json").read_text())
    captured_layers = tuple(int(value) for value in metadata["routing_hb_layer_indices"])
    if layer not in captured_layers:
        raise ValueError(f"layer {layer} not captured; available={captured_layers}")
    layer_axis = captured_layers.index(layer)

    routes = zarr.open_group(str(server / "routes.zarr"), mode="r")
    hidden = zarr.open_group(str(server / "hidden.zarr"), mode="r")
    route_episode = np.asarray(routes["episode_id"][:], dtype=np.int64)
    hidden_episode = np.asarray(hidden["episode_id"][:], dtype=np.int64)
    if not np.array_equal(route_episode, hidden_episode):
        raise ValueError("routes.zarr and hidden.zarr identity axes differ")
    rows = _first_rows(route_episode, episodes)
    selection_hidden = (rows, [layer_axis], [denoise], slice(1, 11), slice(None))
    selection_topk = (rows, [layer_axis], [denoise], slice(1, 11), slice(None))
    selection_probs = (rows, [layer_axis], [denoise], slice(1, 11), slice(None))
    hidden_value = np.asarray(
        hidden["hb_hidden"].get_orthogonal_selection(selection_hidden), dtype=np.float32
    )[:, 0, 0]
    route_ids = np.asarray(
        routes["hb_expert_ids"].get_orthogonal_selection(selection_topk), dtype=np.int64
    )[:, 0, 0]
    route_raw = np.asarray(
        routes["hb_selected_prob"].get_orthogonal_selection(selection_topk),
        dtype=np.float32,
    )[:, 0, 0]
    route_probs = np.asarray(
        routes["hb_router_probs"].get_orthogonal_selection(selection_probs),
        dtype=np.float32,
    )[:, 0, 0]
    route_probs = np.maximum(route_probs, 0.0)
    route_probs /= np.maximum(route_probs.sum(axis=-1, keepdims=True), 1e-20)
    route_weight = route_raw / np.maximum(route_raw.sum(axis=-1, keepdims=True), 1e-20)

    actions = []
    states = []
    sim_states = []
    for episode in episodes:
        with np.load(_resolve_episode_file(client, int(episode))) as payload:
            actions.append(np.asarray(payload["actions"][0], dtype=np.float32))
            states.append(np.asarray(payload["state"][0], dtype=np.float32))
            sim_states.append(np.asarray(payload["sim_state"][0], dtype=np.float32))
    actions = np.stack(actions)
    states = np.stack(states)
    sim_states = np.stack(sim_states)

    unique_scenes = np.unique(scenes)
    unique_seeds = np.unique(seeds)
    if len(summaries) != len(unique_scenes) * len(unique_seeds):
        raise ValueError("capture is not a complete scene x seed grid")
    order = []
    for scene in unique_scenes:
        index = np.flatnonzero(scenes == scene)
        index = index[np.argsort(seeds[index], kind="stable")]
        if not np.array_equal(seeds[index], unique_seeds):
            raise ValueError(f"scene {scene} does not contain the common seed grid")
        if not np.all(states[index] == states[index[0]]):
            raise ValueError(f"robot state varies inside scene {scene}")
        if not np.all(sim_states[index] == sim_states[index[0]]):
            raise ValueError(f"sim state varies inside scene {scene}")
        order.append(index)
    order = np.stack(order)

    stats = json.loads(pathlib.Path(metadata["normalization_stats_path"]).read_text())
    action_std = np.asarray(stats["actions"]["std"], dtype=np.float64)
    noise = np.stack(
        [
            np.random.default_rng(int(seed)).standard_normal((10, 24)).astype(np.float32)
            for seed in seeds
        ]
    )
    return {
        "summaries": summaries,
        "episodes": episodes,
        "scenes": scenes,
        "seeds": seeds,
        "scene_ids": unique_scenes,
        "seed_ids": unique_seeds,
        "order": order,
        "hidden": hidden_value,
        "route_ids": route_ids,
        "route_raw": route_raw,
        "route_weight": route_weight,
        "route_probs": route_probs,
        "actions": actions / action_std[None, None, :],
        "noise": noise,
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "layer_axis": layer_axis,
    }


def _mixture(
    torch,
    x,
    ids,
    alpha,
    experts,
    with_moments: bool,
):
    output = torch.zeros_like(x, dtype=torch.float32)
    s1 = torch.zeros(len(x), dtype=torch.float32) if with_moments else None
    s2 = torch.zeros(len(x), dtype=torch.float32) if with_moments else None
    for expert_id, weights in enumerate(experts):
        matches = torch.nonzero(ids == expert_id, as_tuple=False)
        if matches.numel() == 0:
            continue
        token = matches[:, 0]
        slot = matches[:, 1]
        value = _mlp(torch, x[token], weights)
        mix_weight = alpha[token, slot]
        output.index_add_(0, token, value * mix_weight[:, None])
        if with_moments:
            norm = value.square().mean(dim=-1).sqrt()
            s1.index_add_(0, token, norm * mix_weight)
            s2.index_add_(0, token, norm.square() * mix_weight)
    return output, s1, s2


def reconstruct_primary_cell(
    data: dict[str, Any], checkpoint: pathlib.Path, layer: int, threads: int
) -> dict[str, np.ndarray]:
    import torch

    torch.set_num_threads(threads)
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    experts, shared_weights, gate_weight = _layer_weights(
        checkpoint_state, layer, "cpu", torch
    )
    experts = [tuple(weight.float() for weight in triplet) for triplet in experts]
    shared_weights = tuple(weight.float() for weight in shared_weights)
    gate_weight = gate_weight.float()

    n_candidate = len(data["episodes"])
    x = torch.from_numpy(np.ascontiguousarray(data["hidden"])).reshape(-1, HIDDEN_WIDTH)
    ids = torch.from_numpy(np.ascontiguousarray(data["route_ids"])).reshape(-1, 4)
    alpha = torch.from_numpy(np.ascontiguousarray(data["route_weight"])).reshape(-1, 4)
    probabilities = data["route_probs"].reshape(-1, N_EXPERTS)
    order = np.argsort(-probabilities, axis=-1, kind="stable")
    selected = data["route_ids"].reshape(-1, 4)
    is_selected = (order[:, :, None] == selected[:, None, :]).any(axis=-1)
    alternative = order[~is_selected].reshape(len(order), N_EXPERTS - 4)[:, :4]
    alt_raw = np.take_along_axis(probabilities, alternative, axis=-1)
    alt_weight = alt_raw / np.maximum(alt_raw.sum(axis=-1, keepdims=True), 1e-20)
    alt_ids = torch.from_numpy(np.ascontiguousarray(alternative))
    alt_alpha = torch.from_numpy(np.ascontiguousarray(alt_weight)).float()

    with torch.inference_mode():
        routed, s1, s2 = _mixture(torch, x, ids, alpha.float(), experts, True)
        routed_alt, _, _ = _mixture(torch, x, alt_ids, alt_alpha, experts, False)
        shared = _mlp(torch, x, shared_weights)
        logits = torch.nn.functional.linear(x, gate_weight)
        recomputed = logits.softmax(dim=-1)

        routed_rms = routed.square().mean(dim=-1).sqrt()
        shared_rms = shared.square().mean(dim=-1).sqrt()
        input_rms = x.square().mean(dim=-1).sqrt()
        post_rms = (routed + shared).square().mean(dim=-1).sqrt()
        delta_rms = (routed_alt - routed).square().mean(dim=-1).sqrt()
        disagreement = 1.0 - routed_rms.square() / s2.clamp_min(1e-12)
        cancellation = 1.0 - routed_rms / s1.clamp_min(1e-12)
        conflict = 1.0 - (routed * shared).sum(dim=-1) / (
            routed.norm(dim=-1) * shared.norm(dim=-1)
        ).clamp_min(1e-12)

        top4 = recomputed.topk(4, dim=-1, sorted=False).indices
        set_match = (
            top4.sort(dim=-1).values == ids.sort(dim=-1).values
        ).all(dim=-1).float().mean()
        selected_raw = torch.from_numpy(np.ascontiguousarray(data["route_raw"])).reshape(-1, 4)
        probability_mae = (
            recomputed.gather(1, ids).float() - selected_raw.float()
        ).abs().mean()

    def shaped(tensor) -> np.ndarray:
        return tensor.numpy().reshape(n_candidate, N_ACTION_TOKENS, *tensor.shape[1:])

    result = {
        "routed": shaped(routed),
        "shared": shaped(shared),
        "routed_alt": shaped(routed_alt),
        "disagreement": shaped(disagreement),
        "cancellation": shaped(cancellation),
        "conflict": shaped(conflict),
        "expert_mass": shaped(s1),
        "routed_rms": shaped(routed_rms),
        "shared_rms": shaped(shared_rms),
        "block_sensitivity": shaped(delta_rms / post_rms.clamp_min(1e-12)),
        "routed_sensitivity": shaped(delta_rms / routed_rms.clamp_min(1e-12)),
        "input_sensitivity": shaped(delta_rms / input_rms.clamp_min(1e-12)),
        "top4_margin": (
            np.sort(data["route_probs"], axis=-1)[..., -4]
            - np.sort(data["route_probs"], axis=-1)[..., -5]
        ),
        "router_entropy": -np.sum(
            data["route_probs"] * np.log(np.maximum(data["route_probs"], 1e-20)), axis=-1
        ),
        "validation": {
            "top4_set_match": float(set_match),
            "selected_probability_mae": float(probability_mae),
        },
    }
    return result


def _weighted_router_tv(
    probabilities: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray
) -> np.ndarray:
    return 0.5 * np.abs(probabilities[pair_i] - probabilities[pair_j]).sum(axis=-1).mean(axis=-1)


def build_scene_pair_data(
    data: dict[str, Any], reconstructed: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    k = data["order"].shape[1]
    pair_i, pair_j = np.triu_indices(k, 1)
    scalar_names = ("disagreement", "cancellation", "conflict")
    scenes = []
    for scene_axis, index in enumerate(data["order"]):
        hidden = data["hidden"][index]
        action = data["actions"][index]
        noise = data["noise"][index, :, :LIVE_ACTION_DIMS]
        routed = reconstructed["routed"][index]
        shared = reconstructed["shared"][index]
        probabilities = data["route_probs"][index]
        candidate = {
            name: reconstructed[name][index].mean(axis=1) for name in scalar_names
        }
        z = np.stack([_row_standardize(candidate[name][None])[0] for name in scalar_names])
        dcq = z.mean(axis=0)
        score = {
            "hidden_rms": _pair_rms(hidden, pair_i, pair_j),
            "hidden_cosine": _pair_cosine(hidden, pair_i, pair_j),
            "noise_rms": _pair_rms(noise, pair_i, pair_j),
            "router_tv": _weighted_router_tv(probabilities, pair_i, pair_j),
            "routed_cosine": _pair_cosine(routed, pair_i, pair_j),
            "shared_cosine": _pair_cosine(shared, pair_i, pair_j),
            "dcq_level": 0.5 * (dcq[pair_i] + dcq[pair_j]),
            "dcq_diff": np.abs(dcq[pair_i] - dcq[pair_j]),
        }
        for name in scalar_names:
            score[f"{name}_level"] = 0.5 * (
                candidate[name][pair_i] + candidate[name][pair_j]
            )
            score[f"{name}_diff"] = np.abs(
                candidate[name][pair_i] - candidate[name][pair_j]
            )
        scenes.append(
            {
                "scene_id": int(data["scene_ids"][scene_axis]),
                "pair_i": pair_i,
                "pair_j": pair_j,
                "action_matrix": np.sqrt(
                    np.mean(
                        np.square(action[:, None] - action[None, :], dtype=np.float64),
                        axis=(2, 3),
                    )
                ),
                "score": score,
            }
        )
    return scenes


def analyze_twins(
    scenes: list[dict[str, Any]], args: argparse.Namespace, rng: np.random.Generator
) -> dict[str, Any]:
    score_names = tuple(scenes[0]["score"])
    n_pairs = len(scenes[0]["pair_i"])
    near_count = int(round(n_pairs * args.near_fraction))
    tail_count = int(round(near_count * args.tail_fraction))
    if near_count < 2 or tail_count < 1 or 2 * tail_count > near_count:
        raise ValueError("near/tail fractions yield invalid matched sets")

    observed_scene = np.empty((len(score_names), len(scenes)), dtype=np.float64)
    selections = []
    exact_stable = 0
    exact_divergent = 0
    matched_hidden = []
    all_hidden = []
    stable_action = []
    divergent_action = []
    for scene_axis, scene in enumerate(scenes):
        hidden_distance = scene["score"]["hidden_rms"]
        near = np.argsort(hidden_distance, kind="stable")[:near_count]
        pair_i, pair_j = scene["pair_i"], scene["pair_j"]
        action_distance = scene["action_matrix"][pair_i, pair_j]
        near_order = near[np.argsort(action_distance[near], kind="stable")]
        stable = near_order[:tail_count]
        divergent = near_order[-tail_count:]
        selections.append(near)
        for metric_axis, name in enumerate(score_names):
            score = scene["score"][name]
            observed_scene[metric_axis, scene_axis] = _fast_auc(
                score[stable], score[divergent]
            )
        global_order = np.argsort(action_distance, kind="stable")
        global_tail = max(1, int(round(len(action_distance) * 0.10)))
        exact_stable += int(np.intersect1d(near, global_order[:global_tail]).size)
        exact_divergent += int(np.intersect1d(near, global_order[-global_tail:]).size)
        matched_hidden.extend(hidden_distance[near].tolist())
        all_hidden.extend(hidden_distance.tolist())
        stable_action.extend(action_distance[stable].tolist())
        divergent_action.extend(action_distance[divergent].tolist())

    observed = np.nanmean(observed_scene, axis=1)
    null = np.empty((args.perms, len(score_names)), dtype=np.float64)
    for permutation in range(args.perms):
        # The same 32 seed ids recur in every pool. A common column
        # permutation preserves that repeated-seed structure under the null.
        candidate_perm = rng.permutation(scenes[0]["action_matrix"].shape[0])
        value = np.empty((len(score_names), len(scenes)), dtype=np.float64)
        for scene_axis, scene in enumerate(scenes):
            pair_action = scene["action_matrix"][
                candidate_perm[scene["pair_i"]], candidate_perm[scene["pair_j"]]
            ]
            near = selections[scene_axis]
            near_order = near[np.argsort(pair_action[near], kind="stable")]
            stable = near_order[:tail_count]
            divergent = near_order[-tail_count:]
            for metric_axis, name in enumerate(score_names):
                score = scene["score"][name]
                value[metric_axis, scene_axis] = _fast_auc(
                    score[stable], score[divergent]
                )
        null[permutation] = np.nanmean(value, axis=1)

    name_to_axis = {name: axis for axis, name in enumerate(score_names)}
    discovery_axis = np.asarray([name_to_axis[name] for name in DISCOVERY_TWIN])
    null_max = np.max(null[:, discovery_axis] - 0.5, axis=1)
    rows = {}
    for metric_axis, name in enumerate(score_names):
        row = {
            "macro_auc": float(observed[metric_axis]),
            "scene_bootstrap_ci95": _scene_bootstrap(
                observed_scene[metric_axis], args.bootstrap, rng
            ),
            "per_scene_auc": observed_scene[metric_axis].tolist(),
            "unadjusted_permutation_p": float(
                (1 + np.sum(null[:, metric_axis] >= observed[metric_axis]))
                / (1 + args.perms)
            ),
        }
        if name in DISCOVERY_TWIN:
            row["maxT_fwer_p"] = float(
                (1 + np.sum(null_max >= observed[metric_axis] - 0.5))
                / (1 + args.perms)
            )
        rows[name] = row

    contrast = {}
    dcq_axis = name_to_axis["dcq_level"]
    for control in ("hidden_rms", "hidden_cosine", "noise_rms", "router_tv", "shared_cosine"):
        control_axis = name_to_axis[control]
        difference = observed_scene[dcq_axis] - observed_scene[control_axis]
        contrast[control] = {
            "auc_difference": float(np.nanmean(difference)),
            "scene_bootstrap_ci95": _scene_bootstrap(difference, args.bootstrap, rng),
        }
    return {
        "near_pairs_per_scene": near_count,
        "tail_pairs_per_class_per_scene": tail_count,
        "selected_pairs": int(2 * tail_count * len(scenes)),
        "exact_global_q10_intersections": {
            "stable": exact_stable,
            "divergent": exact_divergent,
        },
        "distance_summary": {
            "matched_hidden_median": float(np.median(matched_hidden)),
            "all_hidden_median": float(np.median(all_hidden)),
            "matched_over_all_hidden_median": float(
                np.median(matched_hidden) / np.median(all_hidden)
            ),
            "stable_final_action_median": float(np.median(stable_action)),
            "divergent_final_action_median": float(np.median(divergent_action)),
        },
        "metrics": rows,
        "dcq_level_contrasts": contrast,
        "null_max_auc_excess_quantiles": {
            "q50": float(np.quantile(null_max, 0.50)),
            "q95": float(np.quantile(null_max, 0.95)),
            "q99": float(np.quantile(null_max, 0.99)),
        },
    }


def _macro_spearman(predictor_rank: np.ndarray, target_rank: np.ndarray) -> np.ndarray:
    predictor = predictor_rank - predictor_rank.mean(axis=-1, keepdims=True)
    target = target_rank - target_rank.mean(axis=-1, keepdims=True)
    numerator = np.sum(predictor * target, axis=-1)
    denominator = np.sqrt(
        np.sum(np.square(predictor), axis=-1) * np.sum(np.square(target), axis=-1)
    )
    return numerator / np.maximum(denominator, 1e-20)


def _analyze_fragility_target(
    target: np.ndarray,
    matrices: dict[str, np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    target_rank = _rank_rows(target)
    names = tuple(matrices)
    predictor_rank = np.stack([_rank_rows(matrices[name]) for name in names])
    scene_rho = _macro_spearman(predictor_rank, target_rank[None])
    observed = scene_rho.mean(axis=1)

    null = np.empty((args.perms, len(names)), dtype=np.float64)
    for permutation in range(args.perms):
        seed_perm = rng.permutation(target.shape[1])
        permuted_target = target_rank[:, seed_perm]
        null_scene = _macro_spearman(predictor_rank, permuted_target[None])
        null[permutation] = null_scene.mean(axis=1)
    name_to_axis = {name: axis for axis, name in enumerate(names)}
    discovery_axis = np.asarray([name_to_axis[name] for name in DISCOVERY_FRAGILITY])
    null_max = np.max(null[:, discovery_axis], axis=1)
    rows = {}
    for axis, name in enumerate(names):
        row = {
            "scene_macro_spearman": float(observed[axis]),
            "scene_bootstrap_ci95": _scene_bootstrap(scene_rho[axis], args.bootstrap, rng),
            "per_scene_spearman": scene_rho[axis].tolist(),
            "unadjusted_permutation_p": float(
                (1 + np.sum(null[:, axis] >= observed[axis])) / (1 + args.perms)
            ),
        }
        if name in DISCOVERY_FRAGILITY:
            row["maxT_fwer_p"] = float(
                (1 + np.sum(null_max >= observed[axis])) / (1 + args.perms)
            )
        rows[name] = row
    contrast = {}
    for discovery in ("conflict", "dcq"):
        discovery_axis = name_to_axis[discovery]
        for control in ("routed_over_shared", "hidden_rms", "negative_top4_margin"):
            control_axis = name_to_axis[control]
            difference = scene_rho[discovery_axis] - scene_rho[control_axis]
            contrast[f"{discovery}_minus_{control}"] = {
                "spearman_difference": float(np.nanmean(difference)),
                "scene_bootstrap_ci95": _scene_bootstrap(
                    difference, args.bootstrap, rng
                ),
            }
    return {
        "target_summary": {
            "median": float(np.median(target)),
            "q25": float(np.quantile(target, 0.25)),
            "q75": float(np.quantile(target, 0.75)),
        },
        "metrics": rows,
        "contrasts": contrast,
        "null_max_spearman_quantiles": {
            "q50": float(np.quantile(null_max, 0.50)),
            "q95": float(np.quantile(null_max, 0.95)),
            "q99": float(np.quantile(null_max, 0.99)),
        },
    }


def analyze_fragility(
    data: dict[str, Any],
    reconstructed: dict[str, np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    order = data["order"]
    scalar = {
        "disagreement": reconstructed["disagreement"].mean(axis=1),
        "cancellation": reconstructed["cancellation"].mean(axis=1),
        "conflict": reconstructed["conflict"].mean(axis=1),
        "negative_top4_margin": -reconstructed["top4_margin"].mean(axis=1),
        "router_entropy": reconstructed["router_entropy"].mean(axis=1),
        "hidden_rms": _rms(data["hidden"], axis=(1, 2)),
        "expert_mass": reconstructed["expert_mass"].mean(axis=1),
        "routed_over_shared": (
            reconstructed["routed_rms"].mean(axis=1)
            / np.maximum(reconstructed["shared_rms"].mean(axis=1), 1e-12)
        ),
    }
    matrices = {name: value[order] for name, value in scalar.items()}
    dcq_stack = np.stack(
        [_row_standardize(matrices[name]) for name in ("disagreement", "cancellation", "conflict")]
    )
    matrices["dcq"] = dcq_stack.mean(axis=0)
    targets = {
        "post_relative": reconstructed["block_sensitivity"].mean(axis=1)[order],
        "routed_relative": reconstructed["routed_sensitivity"].mean(axis=1)[order],
        "input_relative": reconstructed["input_sensitivity"].mean(axis=1)[order],
    }
    return {
        "primary_target": "post_relative",
        "target_definition": {
            "post_relative": "RMS(r_alt-r_true)/RMS(r_true+shared)",
            "routed_relative": "RMS(r_alt-r_true)/RMS(r_true)",
            "input_relative": "RMS(r_alt-r_true)/RMS(h)",
        },
        "targets": {
            name: _analyze_fragility_target(target, matrices, args, rng)
            for name, target in targets.items()
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    twins = summary["hidden_matched_twins"]
    fragility = summary["offline_block_fragility"]
    primary_fragility = fragility["targets"][fragility["primary_target"]]
    lines = [
        "# Hidden-matched expert-activation proxy",
        "",
        "This is a retrospective proxy on one captured task, not a runtime causal test.",
        "Source capture: `%s`." % summary["run"],
        "The primary cell was fixed to HB L%d/d%d." % (summary["layer"], summary["denoise"]),
        "",
        "## Hidden-matched future-action divergence",
        "",
        "Each exact-observation K32 pool contributes its 50 nearest pre-MoE-hidden pairs;",
        "the lowest/highest ten final-action distances are the stable/divergent classes.",
        "The literal current-q10/final-global-q10-q90 rule produced %d stable and %d divergent pairs,"
        % (
            twins["exact_global_q10_intersections"]["stable"],
            twins["exact_global_q10_intersections"]["divergent"],
        ),
        "so it is reported as underpowered rather than used as the primary balanced comparison.",
        "",
        "| score | macro AUC | pool-bootstrap 95% CI | maxT FWER p |",
        "|---|---:|---:|---:|",
    ]
    table_names = (
        "hidden_rms",
        "hidden_cosine",
        "noise_rms",
        "router_tv",
        "shared_cosine",
        "routed_cosine",
        "dcq_level",
        "dcq_diff",
        "disagreement_level",
        "cancellation_level",
        "conflict_level",
    )
    for name in table_names:
        row = twins["metrics"][name]
        lines.append(
            "| %s | %.3f | [%.3f, %.3f] | %s |"
            % (
                name,
                row["macro_auc"],
                row["scene_bootstrap_ci95"][0],
                row["scene_bootstrap_ci95"][1],
                "%.4f" % row["maxT_fwer_p"] if "maxT_fwer_p" in row else "control",
            )
        )
    lines += [
        "",
        "DCQ-level AUC minus controls:",
        "",
    ]
    for name, row in twins["dcq_level_contrasts"].items():
        lines.append(
            "- %s: %+.3f [%.3f, %.3f]"
            % (
                name,
                row["auc_difference"],
                row["scene_bootstrap_ci95"][0],
                row["scene_bootstrap_ci95"][1],
            )
        )
    lines += [
        "",
        "## Offline rank-5-8 block sensitivity",
        "",
        "F_block is the immediate relative change in routed+shared output after replacing",
        "the recorded top-4 with the four highest-probability unselected experts.",
        "It is not a final-action intervention.",
        "",
        "| score | scene-macro Spearman | pool-bootstrap 95% CI | maxT FWER p |",
        "|---|---:|---:|---:|",
    ]
    fragility_names = (
        "negative_top4_margin",
        "router_entropy",
        "hidden_rms",
        "expert_mass",
        "routed_over_shared",
        "disagreement",
        "cancellation",
        "conflict",
        "dcq",
    )
    for name in fragility_names:
        row = primary_fragility["metrics"][name]
        lines.append(
            "| %s | %+.3f | [%.3f, %.3f] | %s |"
            % (
                name,
                row["scene_macro_spearman"],
                row["scene_bootstrap_ci95"][0],
                row["scene_bootstrap_ci95"][1],
                "%.4f" % row["maxT_fwer_p"] if "maxT_fwer_p" in row else "control",
            )
        )
    lines += [
        "",
        "Primary-target Spearman contrasts:",
        "",
    ]
    for name, row in primary_fragility["contrasts"].items():
        lines.append(
            "- %s: %+.3f [%.3f, %.3f]"
            % (
                name,
                row["spearman_difference"],
                row["scene_bootstrap_ci95"][0],
                row["scene_bootstrap_ci95"][1],
            )
        )
    lines += [
        "",
        "Denominator sensitivity (scene-macro Spearman):",
        "",
        "| target | disagreement | cancellation | conflict | DCQ | routed/shared |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target_name, target_result in fragility["targets"].items():
        metric = target_result["metrics"]
        lines.append(
            "| %s | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f |"
            % (
                target_name,
                metric["disagreement"]["scene_macro_spearman"],
                metric["cancellation"]["scene_macro_spearman"],
                metric["conflict"]["scene_macro_spearman"],
                metric["dcq"]["scene_macro_spearman"],
                metric["routed_over_shared"]["scene_macro_spearman"],
            )
        )
    lines += [
        "",
        "## Boundary",
        "",
        "The expert outputs were reconstructed from fp16 stored hidden values with CPU fp32",
        "checkpoint evaluation. Recorded ids and weights are authoritative, but vectors are",
        "not runtime-exact. Exact x_tau matching, AS outputs, and final-action sensitivity",
        "require a new instrumented rollout with common-future interventions.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.layer != 5 or args.denoise != 0:
        raise ValueError("the frozen primary analysis is HB layer 5, denoise 0")
    run = pathlib.Path(args.run).resolve()
    checkpoint = pathlib.Path(args.checkpoint).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()

    data = load_primary_cell(run, args.layer, args.denoise)
    reconstructed = reconstruct_primary_cell(data, checkpoint, args.layer, args.threads)
    scenes = build_scene_pair_data(data, reconstructed)
    twins = analyze_twins(scenes, args, rng)
    fragility = analyze_fragility(data, reconstructed, args, rng)
    summary = {
        "experiment": "expert_activation_hidden_matched_proxy_v1",
        "design": "EXPERT_ACTIVATION_PROXY_DESIGN.md",
        "status": "retrospective_proxy_not_runtime_causal",
        "run": str(run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": data["checkpoint_sha256"],
        "layer": args.layer,
        "denoise": args.denoise,
        "pools": int(len(data["scene_ids"])),
        "candidates_per_pool": int(len(data["seed_ids"])),
        "permutations": args.perms,
        "bootstrap_draws": args.bootstrap,
        "offline_reconstruction_validation": reconstructed["validation"],
        "hidden_matched_twins": twins,
        "offline_block_fragility": fragility,
        "elapsed_seconds": float(time.perf_counter() - started),
        "limitations": [
            "No intermediate flow latent x_tau was captured.",
            "Expert vectors are offline fp32 reconstructions from fp16 hidden values.",
            "F_block is immediate block-output sensitivity, not final-action sensitivity.",
            "One previously inspected task; results are exploratory and need fresh-task replication.",
        ],
    }
    np.savez_compressed(
        out_dir / "candidate_proxy_values.npz",
        scene_ids=data["scene_ids"],
        seed_ids=data["seed_ids"],
        block_sensitivity=reconstructed["block_sensitivity"].mean(axis=1)[data["order"]],
        routed_sensitivity=reconstructed["routed_sensitivity"].mean(axis=1)[data["order"]],
        input_sensitivity=reconstructed["input_sensitivity"].mean(axis=1)[data["order"]],
        disagreement=reconstructed["disagreement"].mean(axis=1)[data["order"]],
        cancellation=reconstructed["cancellation"].mean(axis=1)[data["order"]],
        conflict=reconstructed["conflict"].mean(axis=1)[data["order"]],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print("wrote", out_dir / "summary.json", flush=True)
    print("wrote", out_dir / "REPORT.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
