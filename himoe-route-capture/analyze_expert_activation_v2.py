"""v2 redo of expert-activation-future under the frozen prereg.

Fixes over v1 (analysis/expert-activation-future):
* seed-disjoint 4-fold pair evaluation replaces the leak-exposed scene-LOSO;
* all 11 suffix tokens kept (v1 dropped the state token and averaged the rest);
* directional disagreement D and pair-mean (level) features added;
* shared-branch and raw-hidden readouts run as matched-capacity control arms;
* FWER by max-statistic over the frozen discovery family, null = within-scene
  permutation of the rollout->internal-feature assignment.

Prereg: analysis/expert-activation-v2/long-t08/prereg.md (frozen before runs).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import zarr
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analyze_expert_activation_future import (
    _conditional_auc,
    _fit_success_model,
    _first_rows,
    _layer_weights,
    _mlp,
    _pair_distance,
    _reduce_features,
    _scene_bootstrap_conditional,
)

HB_LAYER_NUMBERS = (2, 3, 4, 5, 12, 13, 14, 15)
N_EXPERTS = 32
N_TOKENS = 11
HIDDEN_WIDTH = 1024
NOISE_SHAPE = (10, 24)
LIVE_ACTION_DIMS = 7
N_SEED_FOLDS = 4
N_SCENE_FOLDS = 4

SCALAR_NAMES = (
    "input_rms",
    "expert_mass",
    "routed_rms",
    "disagreement",
    "cancellation",
    "expert_norm_cv",
    "routed_over_shared",
    "q_conflict",
    "shared_rms",
)
C2_IDX = (1, 2)          # size: expert_mass, routed_rms
C3_IDX = (3, 4, 7)       # DCQ: disagreement, cancellation, q_conflict
FAMILY_CHANNELS = ("c1_router", "c2_size", "c3_dcq", "c4_routed_dir", "c7_all")
CONTROL_CHANNELS = ("c5_shared_dir", "c6_h_dir")
ALL_CHANNELS = FAMILY_CHANNELS + CONTROL_CHANNELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", default="2,5,12,15")
    parser.add_argument("--perms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--jobs", type=int, default=48)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-success", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------
# data loading (v1's loader with the token axis widened to all 11 tokens)
# --------------------------------------------------------------------------

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

    selection = (rows, list(layer_axes), slice(None), slice(None), slice(None))
    route_ids = np.asarray(
        route_group["hb_expert_ids"].get_orthogonal_selection(selection), dtype=np.int64
    )
    route_raw = np.asarray(
        route_group["hb_selected_prob"].get_orthogonal_selection(selection),
        dtype=np.float32,
    )
    route_weight = route_raw / np.maximum(route_raw.sum(axis=-1, keepdims=True), 1e-20)

    print("loading %d first-query hidden rows (all 11 tokens)" % len(rows), flush=True)
    started = time.perf_counter()
    hidden = np.asarray(
        hidden_group["hb_hidden"].get_orthogonal_selection(selection), dtype=np.float16
    )
    print(
        "loaded hidden subset %s in %.1fs" % (hidden.shape, time.perf_counter() - started),
        flush=True,
    )
    if hidden.shape[-2:] != (N_TOKENS, HIDDEN_WIDTH):
        raise ValueError("unexpected hidden shape %s" % (hidden.shape,))

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

    unique_scenes = np.unique(scenes)
    unique_noise = np.unique(noise_seeds)
    if len(summaries) != len(unique_scenes) * len(unique_noise):
        raise ValueError("dataset is not a complete scene x noise grid")
    stacked_states = np.stack(states)
    stacked_sim = np.stack(sim_states)
    for scene in unique_scenes:
        index = np.flatnonzero(scenes == scene)
        if len(index) != len(unique_noise) or len(np.unique(noise_seeds[index])) != len(
            unique_noise
        ):
            raise ValueError("scene %d does not contain every noise seed" % scene)
        if not np.allclose(stacked_states[index], stacked_states[index[0]], atol=0, rtol=0):
            raise ValueError("first-query robot state varies within scene %d" % scene)
        if not np.allclose(stacked_sim[index], stacked_sim[index[0]], atol=0, rtol=0):
            raise ValueError("first-query simulator state varies within scene %d" % scene)

    stats = json.loads(pathlib.Path(metadata["normalization_stats_path"]).read_text())
    action_std = np.asarray(stats["actions"]["std"], dtype=np.float64)
    if action_std.shape != (LIVE_ACTION_DIMS,) or np.any(action_std <= 0):
        raise ValueError("invalid checkpoint action std")

    return {
        "episodes": episodes,
        "scenes": scenes,
        "noise_seeds": noise_seeds,
        "success": success,
        "noise": noise,
        "actions": np.stack(actions),
        "action_std": action_std,
        "hidden": hidden,
        "route_ids": route_ids,
        "route_raw": route_raw,
        "route_weight": route_weight,
        "layer_numbers": np.asarray(layer_numbers),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
    }


# --------------------------------------------------------------------------
# feature extraction: per-token scalars + routed/shared vectors
# --------------------------------------------------------------------------

def compute_features(data: dict, checkpoint: pathlib.Path, threads: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    hidden = data["hidden"]
    n_rows, n_layers, n_denoise, n_tokens, width = hidden.shape

    scalars = np.empty((n_rows, n_denoise, n_layers, n_tokens, len(SCALAR_NAMES)), np.float32)
    routed_vec = np.empty((n_rows, n_denoise, n_layers, n_tokens, width), np.float16)
    shared_vec = np.empty_like(routed_vec)
    validation = []
    chunk = 8192

    for layer_axis, layer in enumerate(data["layer_numbers"]):
        started = time.perf_counter()
        experts, shared_weights, gate_weight = _layer_weights(state, int(layer), "cpu", torch)
        experts = [tuple(w.float() for w in trip) for trip in experts]
        shared_weights = tuple(w.float() for w in shared_weights)
        gate_weight = gate_weight.float()

        x = torch.from_numpy(
            np.ascontiguousarray(hidden[:, layer_axis])
        ).reshape(-1, width).float()
        ids = torch.from_numpy(
            np.ascontiguousarray(data["route_ids"][:, layer_axis]).reshape(-1, 4)
        )
        alpha = torch.from_numpy(
            np.ascontiguousarray(data["route_weight"][:, layer_axis]).reshape(-1, 4)
        ).float()
        raw = torch.from_numpy(
            np.ascontiguousarray(data["route_raw"][:, layer_axis]).reshape(-1, 4)
        ).float()
        token_count = x.shape[0]

        with torch.inference_mode():
            routed = torch.zeros((token_count, width), dtype=torch.float32)
            s1 = torch.zeros(token_count, dtype=torch.float32)   # sum alpha * rms(v)
            s2 = torch.zeros(token_count, dtype=torch.float32)   # sum alpha * rms(v)^2
            for expert_id, expert_weights in enumerate(experts):
                matches = torch.nonzero(ids == expert_id, as_tuple=False)
                if matches.numel() == 0:
                    continue
                token_index = matches[:, 0]
                slot_index = matches[:, 1]
                for lo in range(0, len(token_index), chunk):
                    ti = token_index[lo : lo + chunk]
                    si = slot_index[lo : lo + chunk]
                    v = _mlp(torch, x[ti], expert_weights)
                    a = alpha[ti, si]
                    rms = v.square().mean(dim=-1).sqrt()
                    routed.index_add_(0, ti, v * a[:, None])
                    s1.index_add_(0, ti, rms * a)
                    s2.index_add_(0, ti, rms.square() * a)

            shared = torch.empty((token_count, width), dtype=torch.float32)
            for lo in range(0, token_count, chunk):
                shared[lo : lo + chunk] = _mlp(torch, x[lo : lo + chunk], shared_weights)

            input_rms = x.square().mean(dim=-1).sqrt()
            routed_rms = routed.square().mean(dim=-1).sqrt()
            shared_rms = shared.square().mean(dim=-1).sqrt()
            disagreement = 1.0 - routed_rms.square() / s2.clamp_min(1e-12)
            cancellation = 1.0 - routed_rms / s1.clamp_min(1e-12)
            expert_cv = (s2 - s1.square()).clamp_min(0).sqrt() / s1.clamp_min(1e-12)
            routed_over_shared = routed_rms / shared_rms.clamp_min(1e-12)
            q_conflict = 1.0 - (routed * shared).sum(dim=-1) / (
                routed.norm(dim=-1) * shared.norm(dim=-1)
            ).clamp_min(1e-12)

            stacked = torch.stack(
                (
                    input_rms,
                    s1,
                    routed_rms,
                    disagreement,
                    cancellation,
                    expert_cv,
                    routed_over_shared,
                    q_conflict,
                    shared_rms,
                ),
                dim=-1,
            ).reshape(n_rows, n_denoise, n_tokens, len(SCALAR_NAMES))
            scalars[:, :, layer_axis] = stacked.numpy()
            routed_vec[:, :, layer_axis] = (
                routed.reshape(n_rows, n_denoise, n_tokens, width).to(torch.float16).numpy()
            )
            shared_vec[:, :, layer_axis] = (
                shared.reshape(n_rows, n_denoise, n_tokens, width).to(torch.float16).numpy()
            )

            logits = torch.nn.functional.linear(x, gate_weight)
            probabilities = logits.softmax(dim=-1)
            predicted = probabilities.topk(4, dim=-1, sorted=False).indices
            set_match = (
                predicted.sort(dim=-1).values == ids.sort(dim=-1).values
            ).all(dim=-1).float().mean()
            probability_mae = (probabilities.gather(1, ids).float() - raw).abs().mean()
            validation.append(
                {
                    "layer": int(layer),
                    "top4_set_match": float(set_match),
                    "selected_probability_mae": float(probability_mae),
                }
            )
        print(
            "layer %d features done in %.1fs" % (int(layer), time.perf_counter() - started),
            flush=True,
        )

    return {"scalars": scalars, "routed_vec": routed_vec, "shared_vec": shared_vec,
            "validation": validation}


def load_or_compute_features(
    cache: pathlib.Path, data: dict, checkpoint: pathlib.Path, threads: int, force: bool
) -> dict:
    if cache.is_file() and not force:
        with np.load(cache, allow_pickle=False) as stored:
            if tuple(int(v) for v in stored["layer_numbers"]) == tuple(
                int(v) for v in data["layer_numbers"]
            ):
                print("loading cached features from %s" % cache, flush=True)
                return {
                    "scalars": np.asarray(stored["scalars"]),
                    "routed_vec": np.asarray(stored["routed_vec"]),
                    "shared_vec": np.asarray(stored["shared_vec"]),
                    "validation": json.loads(str(stored["validation_json"])),
                }
    features = compute_features(data, checkpoint, threads)
    np.savez(
        cache,
        scalars=features["scalars"],
        routed_vec=features["routed_vec"],
        shared_vec=features["shared_vec"],
        layer_numbers=data["layer_numbers"],
        scalar_names=np.asarray(SCALAR_NAMES),
        validation_json=json.dumps(features["validation"]),
    )
    return features


# --------------------------------------------------------------------------
# pair construction
# --------------------------------------------------------------------------

def build_pair_frame(data: dict) -> dict:
    """Scene-major pair bookkeeping. Local slots are sorted by noise seed."""
    scenes = data["scenes"]
    order = {}
    for scene in np.unique(scenes):
        index = np.flatnonzero(scenes == scene)
        order[int(scene)] = index[np.argsort(data["noise_seeds"][index])]
    scene_ids = sorted(order)
    n_slots = len(order[scene_ids[0]])
    iu = np.triu_indices(n_slots, 1)
    n_local = len(iu[0])

    pairid = np.full((n_slots, n_slots), -1, dtype=np.int64)
    pairid[iu[0], iu[1]] = np.arange(n_local)
    pairid[iu[1], iu[0]] = np.arange(n_local)

    seed_fold_local = np.arange(n_slots) // (n_slots // N_SEED_FOLDS)
    fold_i = np.tile(seed_fold_local[iu[0]], len(scene_ids))
    fold_j = np.tile(seed_fold_local[iu[1]], len(scene_ids))
    scene_of_pair = np.repeat(np.arange(len(scene_ids)), n_local)
    scene_fold = np.arange(len(scene_ids)) // (len(scene_ids) // N_SCENE_FOLDS)

    action_values = data["actions"] / data["action_std"][None, None, :]
    labels = np.empty(len(scene_ids) * n_local, dtype=np.int8)
    for s, scene in enumerate(scene_ids):
        dist = _pair_distance(action_values, order[scene])
        labels[s * n_local : (s + 1) * n_local] = dist <= np.median(dist)

    noise7 = data["noise"][:, :, :LIVE_ACTION_DIMS]
    b1 = np.empty((len(labels), 12), dtype=np.float32)
    for s, scene in enumerate(scene_ids):
        index = order[scene]
        lo = s * n_local
        b1[lo : lo + n_local, 0] = _pair_distance(noise7, index)
        b1[lo : lo + n_local, 1] = _pair_distance(data["noise"], index)
        flat = noise7[index]
        delta = flat[iu[0]] - flat[iu[1]]
        b1[lo : lo + n_local, 2:] = np.sqrt(np.mean(np.square(delta), axis=-1))

    return {
        "scene_ids": scene_ids,
        "order": order,
        "iu": iu,
        "n_local": n_local,
        "pairid": pairid,
        "fold_i": fold_i,
        "fold_j": fold_j,
        "scene_of_pair": scene_of_pair,
        "scene_fold": scene_fold,
        "labels": labels,
        "b1": b1,
        "n_slots": n_slots,
    }


def _pair_scalar_block(values: np.ndarray, iu) -> np.ndarray:
    """values [slots, tokens] -> [pairs, 4]: state absdiff/mean, action absdiff/mean."""
    vi, vj = values[iu[0]], values[iu[1]]
    absdiff = np.abs(vi - vj)
    mean = 0.5 * (vi + vj)
    return np.column_stack(
        (absdiff[:, 0], mean[:, 0], absdiff[:, 1:].mean(axis=1), mean[:, 1:].mean(axis=1))
    )


def _pair_cos_block(vectors: np.ndarray, iu) -> tuple[np.ndarray, np.ndarray]:
    """vectors [slots, tokens, width] -> ([pairs, 2] state/action cosdist, [pairs, tokens])."""
    v = np.asarray(vectors, dtype=np.float32)
    unit = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    sims = np.einsum("itw,jtw->ijt", unit, unit)
    cosdist = 1.0 - sims[iu[0], iu[1]]
    return np.column_stack((cosdist[:, 0], cosdist[:, 1:].mean(axis=1))), cosdist


def _pair_router_block(dense: np.ndarray, iu) -> np.ndarray:
    """dense [slots, tokens, experts] -> [pairs, 2] overlap distance state/action."""
    di, dj = dense[iu[0]], dense[iu[1]]
    lower = np.minimum(di, dj).sum(axis=-1)
    upper = np.maximum(di, dj).sum(axis=-1)
    dist = 1.0 - lower / np.maximum(upper, 1e-12)
    return np.column_stack((dist[:, 0], dist[:, 1:].mean(axis=1)))


def build_cell_features(data: dict, features: dict, frame: dict) -> dict:
    """channel feature matrices per (layer_axis, tau), plus per-token direct AUC input."""
    scalars = features["scalars"]
    n_layers = scalars.shape[2]
    n_denoise = scalars.shape[1]
    scene_ids = frame["scene_ids"]
    order = frame["order"]
    iu = frame["iu"]
    n_local = frame["n_local"]
    n_pairs = len(frame["labels"])

    dense_weight = np.zeros(
        data["route_ids"].shape[:-1] + (N_EXPERTS,), dtype=np.float32
    )
    np.put_along_axis(dense_weight, data["route_ids"], data["route_weight"], axis=-1)

    cells = {}
    per_token_cos = {}
    for la in range(n_layers):
        for tau in range(n_denoise):
            c1 = np.empty((n_pairs, 2), np.float32)
            c2 = np.empty((n_pairs, 4 * len(C2_IDX)), np.float32)
            c3 = np.empty((n_pairs, 4 * len(C3_IDX)), np.float32)
            c4 = np.empty((n_pairs, 2), np.float32)
            c5 = np.empty((n_pairs, 2), np.float32)
            c6 = np.empty((n_pairs, 2), np.float32)
            tok_cos = np.empty((3, n_pairs, N_TOKENS), np.float32)
            for s, scene in enumerate(scene_ids):
                index = order[scene]
                lo = s * n_local
                sl = slice(lo, lo + n_local)
                scal = scalars[index, tau, la]                       # [slots, tokens, metrics]
                c2[sl] = np.column_stack(
                    [_pair_scalar_block(scal[:, :, m], iu) for m in C2_IDX]
                )
                c3[sl] = np.column_stack(
                    [_pair_scalar_block(scal[:, :, m], iu) for m in C3_IDX]
                )
                c4[sl], tok_cos[0, sl] = _pair_cos_block(
                    features["routed_vec"][index, tau, la], iu
                )
                c5[sl], tok_cos[1, sl] = _pair_cos_block(
                    features["shared_vec"][index, tau, la], iu
                )
                c6[sl], tok_cos[2, sl] = _pair_cos_block(
                    data["hidden"][index, la, tau], iu
                )
                c1[sl] = _pair_router_block(dense_weight[index, la, tau], iu)
            cells[(la, tau)] = {
                "c1_router": c1,
                "c2_size": c2,
                "c3_dcq": c3,
                "c4_routed_dir": c4,
                "c5_shared_dir": c5,
                "c6_h_dir": c6,
                "c7_all": np.column_stack((c2, c3, c4)),
            }
            per_token_cos[(la, tau)] = tok_cos
    return {"cells": cells, "per_token_cos": per_token_cos}


# --------------------------------------------------------------------------
# fitting and statistics
# --------------------------------------------------------------------------

def _fit_fold_scene_auc(
    x: np.ndarray,
    labels: np.ndarray,
    frame: dict,
    logistic_c: float,
    return_scores: bool = False,
):
    """Seed-disjoint 4-fold; returns [folds, scenes] within-scene test AUC."""
    fold_i, fold_j = frame["fold_i"], frame["fold_j"]
    scene_of_pair = frame["scene_of_pair"]
    n_scenes = len(frame["scene_ids"])
    out = np.full((N_SEED_FOLDS, n_scenes), np.nan)
    scores = np.full(len(labels), np.nan) if return_scores else None
    for k in range(N_SEED_FOLDS):
        train = (fold_i != k) & (fold_j != k)
        test = (fold_i == k) & (fold_j == k)
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=logistic_c, solver="lbfgs", max_iter=2000)
        model.fit(scaler.transform(x[train]), labels[train])
        prob = model.predict_proba(scaler.transform(x[test]))[:, 1]
        if scores is not None:
            scores[test] = prob
        test_scene = scene_of_pair[test]
        test_label = labels[test]
        for s in range(n_scenes):
            mask = test_scene == s
            if len(np.unique(test_label[mask])) == 2:
                out[k, s] = roc_auc_score(test_label[mask], prob[mask])
    return (out, scores) if return_scores else out


def _double_disjoint_value(
    x: np.ndarray, labels: np.ndarray, frame: dict, logistic_c: float
) -> float:
    fold_i, fold_j = frame["fold_i"], frame["fold_j"]
    scene_of_pair = frame["scene_of_pair"]
    scene_fold = frame["scene_fold"]
    pair_scene_fold = scene_fold[scene_of_pair]
    values = []
    for sf in range(N_SCENE_FOLDS):
        for k in range(N_SEED_FOLDS):
            train = (pair_scene_fold != sf) & (fold_i != k) & (fold_j != k)
            test = (pair_scene_fold == sf) & (fold_i == k) & (fold_j == k)
            if not test.any() or not train.any():
                continue
            scaler = StandardScaler().fit(x[train])
            model = LogisticRegression(C=logistic_c, solver="lbfgs", max_iter=2000)
            model.fit(scaler.transform(x[train]), labels[train])
            prob = model.predict_proba(scaler.transform(x[test]))[:, 1]
            test_scene = scene_of_pair[test]
            test_label = labels[test]
            cell = [
                roc_auc_score(test_label[m], prob[m])
                for m in (test_scene == s for s in np.unique(test_scene))
                if len(np.unique(test_label[m])) == 2
            ]
            if cell:
                values.append(float(np.mean(cell)))
    return float(np.mean(values)) if values else float("nan")


def _direct_scene_auc(dist: np.ndarray, labels: np.ndarray, frame: dict) -> float:
    n_local = frame["n_local"]
    values = []
    for s in range(len(frame["scene_ids"])):
        sl = slice(s * n_local, (s + 1) * n_local)
        if len(np.unique(labels[sl])) == 2:
            values.append(roc_auc_score(labels[sl], -dist[sl]))
    return float(np.mean(values)) if values else float("nan")


def _perm_gather_index(frame: dict, perm_slots: np.ndarray) -> np.ndarray:
    """perm_slots [scenes, slots] -> global gather index for internal pair rows."""
    iu = frame["iu"]
    pairid = frame["pairid"]
    n_local = frame["n_local"]
    gather = np.empty(len(frame["labels"]), dtype=np.int64)
    for s in range(perm_slots.shape[0]):
        pi = perm_slots[s]
        gather[s * n_local : (s + 1) * n_local] = s * n_local + pairid[pi[iu[0]], pi[iu[1]]]
    return gather


def _perm_worker(
    perm_batch: np.ndarray,
    family_feats: dict,
    b1: np.ndarray,
    labels: np.ndarray,
    frame_small: dict,
    logistic_c: float,
    b1_value: float,
) -> np.ndarray:
    """Returns [batch, n_family_cells] permuted delta-AUC values."""
    out = np.empty((perm_batch.shape[0], len(family_feats)), dtype=np.float64)
    keys = sorted(family_feats)
    for b in range(perm_batch.shape[0]):
        gather = _perm_gather_index(frame_small, perm_batch[b])
        for c, key in enumerate(keys):
            x = np.column_stack((b1, family_feats[key][gather]))
            auc = _fit_fold_scene_auc(x, labels, frame_small, logistic_c)
            out[b, c] = np.nanmean(auc) - b1_value
    return out


def analyze_pairs(
    data: dict,
    features: dict,
    frame: dict,
    cell_feats: dict,
    args: argparse.Namespace,
) -> dict:
    labels = frame["labels"]
    b1 = frame["b1"]
    n_layers = len(data["layer_numbers"])
    n_denoise = features["scalars"].shape[1]

    b1_matrix, b1_scores = _fit_fold_scene_auc(
        b1, labels, frame, args.logistic_c, return_scores=True
    )
    b1_value = float(np.nanmean(b1_matrix))
    print("B1 baseline fold-scene AUC %.4f" % b1_value, flush=True)

    observed = {}
    matrices = {}
    oof_scores = {}
    for (la, tau), channels in sorted(cell_feats["cells"].items()):
        for name in ALL_CHANNELS:
            x = np.column_stack((b1, channels[name]))
            keep_scores = name in ("c7_all", "c4_routed_dir", "c5_shared_dir", "c6_h_dir")
            result = _fit_fold_scene_auc(
                x, labels, frame, args.logistic_c, return_scores=keep_scores
            )
            matrix, scores = result if keep_scores else (result, None)
            matrices[(name, la, tau)] = matrix
            observed[(name, la, tau)] = float(np.nanmean(matrix)) - b1_value
            if scores is not None:
                oof_scores[(name, la, tau)] = scores
        print("fits done for layer_axis=%d tau=%d" % (la, tau), flush=True)

    # permutation null over the frozen discovery family
    rng = np.random.default_rng(args.seed)
    n_scenes = len(frame["scene_ids"])
    perms = np.stack(
        [
            np.stack([rng.permutation(frame["n_slots"]) for _ in range(n_scenes)])
            for _ in range(args.perms)
        ]
    )
    family_feats = {
        (name, la, tau): cell_feats["cells"][(la, tau)][name]
        for name in FAMILY_CHANNELS
        for la in range(n_layers)
        for tau in range(n_denoise)
    }
    frame_small = {
        key: frame[key]
        for key in ("iu", "pairid", "n_local", "fold_i", "fold_j", "scene_of_pair",
                     "scene_ids", "labels", "n_slots")
    }
    batches = np.array_split(np.arange(args.perms), min(args.jobs, args.perms))
    started = time.perf_counter()
    results = Parallel(n_jobs=args.jobs, verbose=1)(
        delayed(_perm_worker)(
            perms[batch], family_feats, b1, labels, frame_small, args.logistic_c, b1_value
        )
        for batch in batches
        if len(batch)
    )
    perm_delta = np.concatenate(results, axis=0)          # [perms, n_family_cells]
    print("permutations done in %.1fs" % (time.perf_counter() - started), flush=True)
    family_keys = sorted(family_feats)
    null_max = perm_delta.max(axis=1)
    fwer_p = {
        key: float((1 + np.sum(null_max >= observed[key])) / (1 + args.perms))
        for key in family_keys
    }
    unadjusted_p = {
        key: float(
            (1 + np.sum(perm_delta[:, c] >= observed[key])) / (1 + args.perms)
        )
        for c, key in enumerate(family_keys)
    }

    # scene-jackknife contrasts vs the two control arms for every family cell
    contrasts = {}
    for key in family_keys:
        name, la, tau = key
        base = matrices[key]
        for control in CONTROL_CHANNELS:
            ctrl = matrices[(control, la, tau)]
            full = np.nanmean(base) - np.nanmean(ctrl)
            loo = np.asarray(
                [
                    np.nanmean(np.delete(base, s, axis=1))
                    - np.nanmean(np.delete(ctrl, s, axis=1))
                    for s in range(n_scenes)
                ]
            )
            se = float(np.sqrt((n_scenes - 1) / n_scenes * np.sum((loo - loo.mean()) ** 2)))
            contrasts["%s_minus_%s_l%d_t%d" % (name, control, la, tau)] = {
                "delta": float(full),
                "ci95": [float(full - 1.96 * se), float(full + 1.96 * se)],
            }

    # double-disjoint robustness (real data only)
    double = {}
    for (la, tau) in sorted(cell_feats["cells"]):
        for name in ("c4_routed_dir", "c5_shared_dir", "c6_h_dir", "c7_all"):
            x = np.column_stack((b1, cell_feats["cells"][(la, tau)][name]))
            double["%s_l%d_t%d" % (name, la, tau)] = _double_disjoint_value(
                x, labels, frame, args.logistic_c
            )
    double["b1"] = _double_disjoint_value(b1, labels, frame, args.logistic_c)

    # matched-subset echo: test pairs with noise7 distance below the scene median
    n_local = frame["n_local"]
    matched = np.zeros(len(labels), dtype=bool)
    for s in range(n_scenes):
        sl = slice(s * n_local, (s + 1) * n_local)
        matched[sl] = b1[sl, 0] <= np.median(b1[sl, 0])
    echo = {}
    for key, scores in oof_scores.items():
        name, la, tau = key
        values = []
        for s in range(n_scenes):
            sl = slice(s * n_local, (s + 1) * n_local)
            mask = matched[sl] & np.isfinite(scores[sl])
            lab = labels[sl][mask]
            if len(np.unique(lab)) == 2:
                values.append(roc_auc_score(lab, scores[sl][mask]))
        echo["%s_l%d_t%d" % (name, la, tau)] = (
            float(np.mean(values)) if values else float("nan")
        )
    echo_b1_values = []
    for s in range(n_scenes):
        sl = slice(s * n_local, (s + 1) * n_local)
        mask = matched[sl] & np.isfinite(b1_scores[sl])
        lab = labels[sl][mask]
        if len(np.unique(lab)) == 2:
            echo_b1_values.append(roc_auc_score(lab, b1_scores[sl][mask]))
    echo["b1"] = float(np.mean(echo_b1_values)) if echo_b1_values else float("nan")

    # descriptive direct AUCs (no fitting -> unaffected by any split choice)
    direct = {
        "noise7": _direct_scene_auc(b1[:, 0], labels, frame),
        "noise24": _direct_scene_auc(b1[:, 1], labels, frame),
    }
    per_token_direct = np.full((3, n_layers, n_denoise, N_TOKENS), np.nan)
    for (la, tau), tok_cos in cell_feats["per_token_cos"].items():
        for fam in range(3):
            for t in range(N_TOKENS):
                per_token_direct[fam, la, tau, t] = _direct_scene_auc(
                    tok_cos[fam, :, t], labels, frame
                )
        for fam, fam_name in enumerate(("routed_dir", "shared_dir", "h_dir")):
            direct["%s_action_l%d_t%d" % (fam_name, la, tau)] = _direct_scene_auc(
                tok_cos[fam, :, 1:].mean(axis=1), labels, frame
            )
            direct["%s_state_l%d_t%d" % (fam_name, la, tau)] = _direct_scene_auc(
                tok_cos[fam, :, 0], labels, frame
            )

    # v1-comparability: token-mean vectors over action tokens, layer-pooled cosine
    v1_style = {}
    for fam_name, vec in (
        ("routed_dir", features["routed_vec"]),
        ("shared_dir", features["shared_vec"]),
    ):
        for tau in range(n_denoise):
            token_mean = np.asarray(vec[:, tau, :, 1:], np.float32).mean(axis=2)
            unit = token_mean / np.maximum(
                np.linalg.norm(token_mean, axis=-1, keepdims=True), 1e-12
            )
            dist = np.empty(len(labels), np.float32)
            for s, scene in enumerate(frame["scene_ids"]):
                index = frame["order"][scene]
                u = unit[index]
                sims = np.einsum("ilw,jlw->ijl", u, u)
                dist[s * n_local : (s + 1) * n_local] = (
                    1.0 - sims[frame["iu"][0], frame["iu"][1]]
                ).mean(axis=-1)
            v1_style["%s_t%d" % (fam_name, tau)] = _direct_scene_auc(dist, labels, frame)

    return {
        "b1_value": b1_value,
        "b1_fold_scene": b1_matrix.tolist(),
        "observed_delta": {"%s_l%d_t%d" % key: value for key, value in observed.items()},
        "fwer_p": {"%s_l%d_t%d" % key: value for key, value in fwer_p.items()},
        "unadjusted_p": {"%s_l%d_t%d" % key: value for key, value in unadjusted_p.items()},
        "null_max_quantiles": {
            "q50": float(np.quantile(null_max, 0.50)),
            "q95": float(np.quantile(null_max, 0.95)),
            "q99": float(np.quantile(null_max, 0.99)),
        },
        "contrasts": contrasts,
        "double_disjoint": double,
        "matched_echo": echo,
        "direct": direct,
        "per_token_direct": per_token_direct.tolist(),
        "v1_style_direct": v1_style,
    }


# --------------------------------------------------------------------------
# secondary, descriptive: episode success under v1's seed-fold protocol
# --------------------------------------------------------------------------

def analyze_success(
    data: dict, features: dict, args: argparse.Namespace
) -> dict:
    rng = np.random.default_rng(args.seed + 17)
    labels = data["success"]
    scenes = data["scenes"]
    unique_scenes = np.unique(scenes)
    scene_onehot = (scenes[:, None] == unique_scenes[None, :]).astype(np.float64)
    noise_flat = data["noise"][:, :, :LIVE_ACTION_DIMS].reshape(len(labels), -1)
    action_flat = (data["actions"] / data["action_std"][None, None, :]).reshape(
        len(labels), -1
    )
    scalars = features["scalars"]
    n_denoise = scalars.shape[1]

    def group(indices):
        return scalars[:, :, :, :, indices].reshape(len(labels), n_denoise, -1)

    size_feats = group(list(C2_IDX))
    dcq_feats = group(list(C3_IDX))
    shared_feats = group([SCALAR_NAMES.index("shared_rms")])

    model_names = (
        "scene", "scene_noise", "scene_noise_size", "scene_noise_dcq",
        "scene_noise_shared_size", "scene_noise_final_action",
    )
    predictions = {name: np.full((n_denoise, len(labels)), np.nan) for name in model_names}
    fold_id = np.full(len(labels), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=4)
    for fold, (train, test) in enumerate(
        splitter.split(noise_flat, labels, data["noise_seeds"])
    ):
        fold_id[test] = fold
        noise_train, noise_test = _reduce_features(noise_flat[train], noise_flat[test])
        action_train, action_test = _reduce_features(action_flat[train], action_flat[test])
        for tau in range(n_denoise):
            size_train, size_test = _reduce_features(
                size_feats[train, tau], size_feats[test, tau]
            )
            dcq_train, dcq_test = _reduce_features(dcq_feats[train, tau], dcq_feats[test, tau])
            shared_train, shared_test = _reduce_features(
                shared_feats[train, tau], shared_feats[test, tau]
            )
            parts = {
                "scene": ([scene_onehot[train]], [scene_onehot[test]]),
                "scene_noise": (
                    [scene_onehot[train], noise_train], [scene_onehot[test], noise_test]
                ),
                "scene_noise_size": (
                    [scene_onehot[train], noise_train, size_train],
                    [scene_onehot[test], noise_test, size_test],
                ),
                "scene_noise_dcq": (
                    [scene_onehot[train], noise_train, dcq_train],
                    [scene_onehot[test], noise_test, dcq_test],
                ),
                "scene_noise_shared_size": (
                    [scene_onehot[train], noise_train, shared_train],
                    [scene_onehot[test], noise_test, shared_test],
                ),
                "scene_noise_final_action": (
                    [scene_onehot[train], noise_train, action_train],
                    [scene_onehot[test], noise_test, action_test],
                ),
            }
            for name, (train_parts, test_parts) in parts.items():
                predictions[name][tau, test] = _fit_success_model(
                    train_parts, test_parts, labels[train], args.logistic_c
                )
        print("success fold %d complete" % fold, flush=True)

    strata = scenes.astype(np.int64) * 10 + fold_id
    result = {name: [] for name in model_names}
    for name in model_names:
        for tau in range(n_denoise):
            score = predictions[name][tau]
            auc = _conditional_auc(labels, score, strata)
            boot = _scene_bootstrap_conditional(
                labels, score, scenes, fold_id, args.bootstrap, rng
            )
            baseline_auc = _conditional_auc(labels, predictions["scene_noise"][tau], strata)
            result[name].append(
                {
                    "conditional_auc": float(auc),
                    "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "delta_vs_scene_noise": float(auc - baseline_auc),
                    "log_loss": float(log_loss(labels, np.clip(score, 1e-6, 1 - 1e-6))),
                }
            )
    return {"models": result, "positive_rate": float(labels.mean())}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def render_report(summary: dict) -> str:
    pairs = summary["pair_analysis"]
    layers = summary["layers"]
    lines = [
        "# expert-activation v2:同 token、异内部(种子干净重做)",
        "",
        "预注册:`prereg.md`(先冻结后运行)。v1 对照:`../expert-activation-future/long-t08`。",
        "",
        "## 对齐与量级",
        "",
        "门控复算(fp32,对全部 11 token):"
        + " ".join(
            "L%d set-match %.3f MAE %.1e;" % (
                row["layer"], row["top4_set_match"], row["selected_probability_mae"]
            )
            for row in summary["feature_validation"]
        ),
        "",
        "## 主任务:同 basin pair(种子双开 4 折,B1=12 维噪声基线)",
        "",
        "B1 基线折-场景 AUC = %.4f;置换零分布 max-ΔAUC 分位:q50 %.4f / q95 %.4f / q99 %.4f。"
        % (
            pairs["b1_value"],
            pairs["null_max_quantiles"]["q50"],
            pairs["null_max_quantiles"]["q95"],
            pairs["null_max_quantiles"]["q99"],
        ),
        "",
        "发现族 200 格(5 通道 x 4 层 x 10 轮)中 FWER p<0.05 的格数:%d。"
        % sum(1 for v in pairs["fwer_p"].values() if v < 0.05),
        "",
        "各通道最佳格(ΔAUC vs B1,附 FWER p):",
        "",
        "| 通道 | 最佳格 | ΔAUC | FWER p | 未校正 p |",
        "|---|---|---:|---:|---:|",
    ]
    for name in FAMILY_CHANNELS:
        best_key, best_val = max(
            (
                (key, value)
                for key, value in pairs["observed_delta"].items()
                if key.startswith(name + "_l")
            ),
            key=lambda item: item[1],
        )
        lines.append(
            "| %s | %s | %+.4f | %.3f | %.3f |"
            % (
                name,
                best_key[len(name) + 1 :],
                best_val,
                pairs["fwer_p"][best_key],
                pairs["unadjusted_p"][best_key],
            )
        )
    lines += [
        "",
        "对照臂(不属于发现族)最佳 ΔAUC:",
        "",
    ]
    for name in CONTROL_CHANNELS:
        best_key, best_val = max(
            (
                (key, value)
                for key, value in pairs["observed_delta"].items()
                if key.startswith(name + "_l")
            ),
            key=lambda item: item[1],
        )
        lines.append("- %s:%+.4f(%s)" % (name, best_val, best_key[len(name) + 1 :]))
    lines += [
        "",
        "## 描述性 direct AUC(无拟合,与 v1 direct 列可比)",
        "",
        "noise7 %.3f / noise24 %.3f。" % (pairs["direct"]["noise7"], pairs["direct"]["noise24"]),
        "",
        "v1 风格(action token 均值向量、4 层池化余弦)按轮:",
        "",
        "| tau | routed_dir | shared_dir |",
        "|---:|---:|---:|",
    ]
    for tau in range(10):
        lines.append(
            "| %d | %.3f | %.3f |"
            % (
                tau,
                pairs["v1_style_direct"]["routed_dir_t%d" % tau],
                pairs["v1_style_direct"]["shared_dir_t%d" % tau],
            )
        )
    lines += [
        "",
        "(v1 报告的对应列:routed 0.699 / shared 0.708(tau0),shared 峰值 0.727(tau4)。)",
        "",
        "## 双重双开稳健性(场景+种子)",
        "",
        "B1 = %.4f;" % pairs["double_disjoint"]["b1"]
        + " ".join(
            "%s %.4f;" % (key, value)
            for key, value in sorted(pairs["double_disjoint"].items())
            if key != "b1" and key.startswith(("c7_all_l", "c4_routed", "c5_shared", "c6_h"))
            and key.endswith("_t0")
        ),
        "",
        "## 匹配子集回声(噪声距离低于场景中位数的测试 pair)",
        "",
        "B1 = %.4f;c7_all 各格见 summary.json(matched_echo)。" % pairs["matched_echo"]["b1"],
        "",
    ]
    if "success" in summary:
        success = summary["success"]["models"]
        lines += [
            "## 次任务(纯描述):成败,按 seed 4 折条件 AUC",
            "",
            "| tau | scene+noise | +size | +DCQ | +shared_size | +final_action |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for tau in range(10):
            lines.append(
                "| %d | %.3f | %.3f | %.3f | %.3f | %.3f |"
                % (
                    tau,
                    success["scene_noise"][tau]["conditional_auc"],
                    success["scene_noise_size"][tau]["conditional_auc"],
                    success["scene_noise_dcq"][tau]["conditional_auc"],
                    success["scene_noise_shared_size"][tau]["conditional_auc"],
                    success["scene_noise_final_action"][tau]["conditional_auc"],
                )
            )
        lines += [
            "",
            "(v1 对应:+expert size 在 tau0 达 0.585,基线 0.467。预承诺:该表不得重开 selector 线。)",
            "",
        ]
    lines += [
        "## 判定(按 prereg 规则)",
        "",
        summary["verdict"],
        "",
    ]
    return "\n".join(lines)


def build_verdict(pairs: dict) -> str:
    passing = [key for key, value in pairs["fwer_p"].items() if value < 0.05]
    if not passing:
        return (
            "R1 未通过:发现族 200 格无一在 FWER p<0.05 下超过噪声基线 B1。"
            "按预注册止损规则,basin 线与 selector 线一并关闭;仅剩脆弱性臂(A 臂)。"
        )
    lines = ["R1 通过格:%s。" % ", ".join(sorted(passing))]
    for key in sorted(passing):
        channel, cell = key.split("_l", 1)
        for control in CONTROL_CHANNELS:
            ckey = "%s_minus_%s_l%s" % (channel, control, cell)
            if ckey in pairs["contrasts"]:
                entry = pairs["contrasts"][ckey]
                low, high = entry["ci95"]
                won = low > 0
                lines.append(
                    "%s vs %s:Δ=%+.4f CI[%+.4f,%+.4f] → %s。"
                    % (
                        key,
                        control,
                        entry["delta"],
                        low,
                        high,
                        "R2 通过(CI>0)" if won else "R2 未通过",
                    )
                )
    lines.append(
        "措辞阶梯:仅 R1 → 分支通用接口;+R2a(胜 shared)→ 专家分解特有;"
        "+R2b(胜 h)→ 分解预提取。任何结果均为提取差距,"
        "不得表述为超出完整 hidden state 的信息。"
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run = pathlib.Path(args.run).resolve()
    checkpoint = pathlib.Path(args.checkpoint).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_numbers = tuple(int(v) for v in args.layers.split(",") if v)
    if any(layer not in HB_LAYER_NUMBERS for layer in layer_numbers):
        raise ValueError("layers must be a subset of %s" % (HB_LAYER_NUMBERS,))

    data = load_dataset(run, layer_numbers)
    features = load_or_compute_features(
        out_dir / "features_v2.npz", data, checkpoint, args.threads, args.force_features
    )
    frame = build_pair_frame(data)
    print(
        "pairs: %d total, %d per scene, labels balanced at %.3f"
        % (len(frame["labels"]), frame["n_local"], float(frame["labels"].mean())),
        flush=True,
    )
    cell_feats = build_cell_features(data, features, frame)
    pair_analysis = analyze_pairs(data, features, frame, cell_feats, args)
    summary = {
        "experiment": "expert_activation_v2_same_token_internals",
        "prereg": "prereg.md",
        "run": str(run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": data["checkpoint_sha256"],
        "layers": list(layer_numbers),
        "episodes": int(len(data["episodes"])),
        "successes": int(data["success"].sum()),
        "scalar_names": list(SCALAR_NAMES),
        "n_perms": args.perms,
        "feature_validation": features["validation"],
        "pair_analysis": pair_analysis,
    }
    if not args.skip_success:
        summary["success"] = analyze_success(data, features, args)
    summary["verdict"] = build_verdict(pair_analysis)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print("wrote %s" % (out_dir / "summary.json"), flush=True)
    print("wrote %s" % (out_dir / "report.md"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
