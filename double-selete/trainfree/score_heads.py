#!/usr/bin/env python3
"""Build frozen train-free loop/static scores without opening Trap labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import zarr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
CACHE = HERE / "cache"
PROTOCOL = HERE / "PROTOCOL.md"
META = ROOT / "analysis_ssm/cache/B_meta.csv"
ROWIDX = ROOT / "analysis_ssm/cache/B_rowidx.npy"
ROW_FEATURES = (
    ROOT / "himoe-vla_trap/results/trainfree_signal_matrix/intermediate/"
    "corpus_B_route_features.npz"
)
HARD_CACHE_SOURCE = ROOT / "double-selete/cache/support_features_v1.npz"
DIST_CACHE = ROOT / "analysis_online_2026-08-29/data/distB.pkl"
ROUTE_STORE = (
    ROOT / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32/"
    "server/routes.zarr"
)

N_EXPERTS = 32
HISTORY = 4
SCHEMA = "himoe.trainfree_double_selector.scores.v1"
SOFT_SCHEMA = "himoe.trainfree_double_selector.soft_support.v1"

PRIMARY_FORMULAS = {
    "loop_precursor": (
        "mean(rank_high(gate_concentration_delta2), "
        "rank_high(late_flow_volatility_w4), "
        "rank_high(route_acceleration_w4))"
    ),
    "loop_lock": (
        "mean(rank_high(deep_soft_consensus_delta2), "
        "rank_high(deep_soft_mixture_concentration_delta2))"
    ),
    "loop_head": "max(loop_precursor, loop_lock)",
    "static_head": (
        "mean(rank_low(gate_concentration_w4), "
        "rank_low(gate_concentration_delta2), "
        "rank_high(l15_soft_mixture_concentration_now), "
        "rank_high(l15_soft_consensus_now), "
        "rank_high(l15_soft_final_denoise_jump_now))"
    ),
    "double_max": "max(loop_head, static_head)",
    "single_mean": "mean(loop_head, static_head)",
    "single_mobility": "rank_low(route_mobility_w4)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-soft-cache", action="store_true")
    return parser.parse_args()


def normalize_probability(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(value.sum(axis=-1, keepdims=True), 1e-12)


def concentration(value: np.ndarray) -> np.ndarray:
    entropy = -(value * np.log(np.maximum(value, 1e-12))).sum(axis=-1)
    return 1.0 - entropy / math.log(N_EXPERTS)


def hellinger(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    coefficient = np.sqrt(np.maximum(value, 0.0) * np.maximum(reference, 0.0)).sum(
        axis=-1
    )
    return np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))


def extract_soft_support(rebuild: bool = False) -> dict[str, np.ndarray]:
    """Reduce probabilities to label-free soft support/consensus descriptors."""
    path = CACHE / "soft_support_v1.npz"
    CACHE.mkdir(parents=True, exist_ok=True)
    if path.exists() and not rebuild:
        with np.load(path) as data:
            if str(data["schema"].item()) == SOFT_SCHEMA:
                return {
                    key: np.asarray(data[key]) for key in data.files if key != "schema"
                }

    route = zarr.open_group(str(ROUTE_STORE), mode="r")
    source = route["hb_router_probs"]
    n_rows = int(source.shape[0])
    names = (
        "deep_soft_mixture_concentration_d9",
        "deep_soft_consensus_d9",
        "l15_soft_mixture_concentration_d9",
        "l15_soft_consensus_d9",
        "l15_soft_final_denoise_jump",
    )
    output = {name: np.empty(n_rows, np.float32) for name in names}
    block_size = 256
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        # row, deep layer (L12-L15), denoise, action token, expert
        probability = np.asarray(source[start:stop, 4:8, :, 1:11, :], np.float32)
        probability = normalize_probability(probability)
        mixture = normalize_probability(probability.mean(axis=3))
        mix_concentration = concentration(mixture)
        consensus = 1.0 - hellinger(probability, mixture[:, :, :, None, :]).mean(axis=3)
        output["deep_soft_mixture_concentration_d9"][start:stop] = mix_concentration[
            :, :, 9
        ].mean(axis=1)
        output["deep_soft_consensus_d9"][start:stop] = consensus[:, :, 9].mean(axis=1)
        output["l15_soft_mixture_concentration_d9"][start:stop] = mix_concentration[
            :, 3, 9
        ]
        output["l15_soft_consensus_d9"][start:stop] = consensus[:, 3, 9]
        output["l15_soft_final_denoise_jump"][start:stop] = (
            mix_concentration[:, 3, 9] - mix_concentration[:, 3, 8]
        )
        if stop % 4096 < block_size or stop == n_rows:
            print(f"soft probability rows {stop}/{n_rows}", flush=True)

    np.savez_compressed(path, schema=np.asarray(SOFT_SCHEMA), **output)
    return output


def scatter_rows(value: np.ndarray, rowidx: np.ndarray) -> np.ndarray:
    output = np.full(rowidx.shape, np.nan, np.float64)
    valid = rowidx >= 0
    output[valid] = value[rowidx[valid]]
    return output


def trailing_mean(value: np.ndarray, width: int) -> np.ndarray:
    output = np.full(value.shape, np.nan, np.float64)
    for query in range(width - 1, value.shape[1]):
        block = value[:, query - width + 1 : query + 1]
        good = np.isfinite(block).all(axis=1)
        output[good, query] = block[good].mean(axis=1)
    return output


def lag_delta(value: np.ndarray, lag: int) -> np.ndarray:
    output = np.full(value.shape, np.nan, np.float64)
    good = np.isfinite(value[:, lag:]) & np.isfinite(value[:, :-lag])
    delta = value[:, lag:] - value[:, :-lag]
    output[:, lag:][good] = delta[good]
    return output


def route_mobility(rowidx: np.ndarray) -> np.ndarray:
    with DIST_CACHE.open("rb") as handle:
        stored = pickle.load(handle)
    output = np.full(rowidx.shape, np.nan, np.float64)
    for episode, values in enumerate(stored["H"]["ac"]["hellinger"]):
        values = np.asarray(values, np.float64)
        output[episode, 1 : len(values) + 1] = values
    return output


def support_metrics(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(ids, np.int64)
    n_site, n_token, top_k = ids.shape
    histogram = np.zeros((n_site, N_EXPERTS), np.int16)
    row = np.repeat(np.arange(n_site), n_token * top_k)
    np.add.at(histogram, (row, ids.reshape(-1)), 1)
    probability = histogram / float(n_token * top_k)
    log_probability = np.zeros_like(probability)
    positive = probability > 0
    log_probability[positive] = np.log(probability[positive])
    entropy = -(probability * log_probability).sum(axis=1)
    return (
        1.0 - entropy / math.log(N_EXPERTS),
        histogram.max(axis=1) / float(n_token),
        (histogram > 0).sum(axis=1).astype(np.float64),
    )


def extract_hard_support() -> dict[str, np.ndarray]:
    """Load the prior label-free cache, or rebuild it directly from route IDs."""
    if HARD_CACHE_SOURCE.exists():
        with np.load(HARD_CACHE_SOURCE) as data:
            return {key: np.asarray(data[key]) for key in data.files if key != "schema"}

    route = zarr.open_group(str(ROUTE_STORE), mode="r")
    source = route["hb_expert_ids"]
    n_rows = int(source.shape[0])
    names = (
        "deep_support_concentration_d9",
        "deep_max_occupancy_d9",
        "deep_support_size_d9",
        "l15_support_concentration_d9",
        "l15_max_occupancy_d9",
        "l15_support_size_d9",
        "l15_final_denoise_jump",
    )
    output = {name: np.empty(n_rows, np.float32) for name in names}
    for start in range(0, n_rows, 256):
        stop = min(start + 256, n_rows)
        ids = np.asarray(source[start:stop, 4:8, :, 1:11, :], np.int64)
        count = stop - start
        conc, occupancy, size = support_metrics(ids.reshape(-1, 10, 4))
        conc = conc.reshape(count, 4, 10)
        occupancy = occupancy.reshape(count, 4, 10)
        size = size.reshape(count, 4, 10)
        output["deep_support_concentration_d9"][start:stop] = conc[:, :, 9].mean(1)
        output["deep_max_occupancy_d9"][start:stop] = occupancy[:, :, 9].mean(1)
        output["deep_support_size_d9"][start:stop] = size[:, :, 9].mean(1)
        output["l15_support_concentration_d9"][start:stop] = conc[:, 3, 9]
        output["l15_max_occupancy_d9"][start:stop] = occupancy[:, 3, 9]
        output["l15_support_size_d9"][start:stop] = size[:, 3, 9]
        output["l15_final_denoise_jump"][start:stop] = conc[:, 3, 9] - conc[:, 3, 8]
    return output


def pool_rank(
    value: np.ndarray, group: np.ndarray, valid: np.ndarray, high: bool = True
) -> np.ndarray:
    """Rank each query within its candidate pool; no labels or global scale used."""
    output = np.full(value.shape, np.nan, np.float64)
    for query in range(value.shape[1]):
        for pool in np.unique(group):
            take = (group == pool) & valid[:, query] & np.isfinite(value[:, query])
            count = int(take.sum())
            if count == 0:
                continue
            oriented = value[take, query] if high else -value[take, query]
            if count == 1:
                output[take, query] = 0.5
            else:
                output[take, query] = (rankdata(oriented, method="average") - 1) / (
                    count - 1
                )
    return output


def finite_mean(*values: np.ndarray) -> np.ndarray:
    stack = np.stack(values, axis=0)
    output = np.full(values[0].shape, np.nan, np.float64)
    good = np.isfinite(stack).all(axis=0)
    output[good] = stack[:, good].mean(axis=0)
    return output


def stable_topk(
    indices: np.ndarray, score: np.ndarray, episode: np.ndarray, budget: int
) -> np.ndarray:
    finite = indices[np.isfinite(score[indices])]
    order = np.lexsort((episode[finite], -score[finite]))
    return finite[order[: min(budget, len(finite))]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Explicit usecols excludes the success column from this label-blind process.
    meta = pd.read_csv(META, usecols=["episode_id", "group", "T"])
    episode = meta.episode_id.to_numpy(np.int64)
    group = meta.group.to_numpy(np.int64)
    length = meta["T"].to_numpy(np.int64)
    rowidx = np.load(ROWIDX)
    valid = rowidx >= 0
    if not np.array_equal(episode, np.arange(512)):
        raise RuntimeError("expected corpus-B episode order 0..511")
    if len(np.unique(group)) != 16 or not np.all(
        np.unique(group, return_counts=True)[1] == 32
    ):
        raise RuntimeError("expected 16 pools with 32 candidates each")

    with np.load(ROW_FEATURES) as data:
        raw = {
            key: np.asarray(data[key])
            for key in ("gate_entropy", "late_flow_volatility", "route_acceleration")
        }
    raw.update(extract_soft_support(args.rebuild_soft_cache))
    hard = extract_hard_support()
    series = {
        key: scatter_rows(value, rowidx) for key, value in {**raw, **hard}.items()
    }

    gate_concentration = 1.0 - series["gate_entropy"]
    mobility = route_mobility(rowidx)
    features = {
        "gate_concentration_w4": trailing_mean(gate_concentration, HISTORY),
        "gate_concentration_delta2": lag_delta(gate_concentration, 2),
        "late_flow_volatility_w4": trailing_mean(
            series["late_flow_volatility"], HISTORY
        ),
        "route_acceleration_w4": trailing_mean(series["route_acceleration"], HISTORY),
        "route_mobility_w4": trailing_mean(mobility, HISTORY),
        "deep_soft_consensus_delta2": lag_delta(series["deep_soft_consensus_d9"], 2),
        "deep_soft_mixture_concentration_delta2": lag_delta(
            series["deep_soft_mixture_concentration_d9"], 2
        ),
        "l15_soft_mixture_concentration_now": series[
            "l15_soft_mixture_concentration_d9"
        ],
        "l15_soft_consensus_now": series["l15_soft_consensus_d9"],
        "l15_soft_final_denoise_jump_now": series["l15_soft_final_denoise_jump"],
        "deep_hard_concentration_delta2": lag_delta(
            series["deep_support_concentration_d9"], 2
        ),
        "deep_hard_occupancy_delta2": lag_delta(series["deep_max_occupancy_d9"], 2),
        "deep_hard_support_size_delta2": lag_delta(series["deep_support_size_d9"], 2),
        "l15_hard_concentration_now": series["l15_support_concentration_d9"],
        "l15_hard_occupancy_now": series["l15_max_occupancy_d9"],
        "l15_hard_support_size_now": series["l15_support_size_d9"],
        "l15_hard_final_denoise_jump_now": series["l15_final_denoise_jump"],
    }

    directions = {
        "gate_concentration_delta2_high": ("gate_concentration_delta2", True),
        "gate_concentration_delta2_low": ("gate_concentration_delta2", False),
        "gate_concentration_w4_low": ("gate_concentration_w4", False),
        "late_flow_volatility_w4_high": ("late_flow_volatility_w4", True),
        "route_acceleration_w4_high": ("route_acceleration_w4", True),
        "route_mobility_w4_low": ("route_mobility_w4", False),
        "deep_soft_consensus_delta2_high": ("deep_soft_consensus_delta2", True),
        "deep_soft_mixture_delta2_high": (
            "deep_soft_mixture_concentration_delta2",
            True,
        ),
        "l15_soft_mixture_now_high": ("l15_soft_mixture_concentration_now", True),
        "l15_soft_consensus_now_high": ("l15_soft_consensus_now", True),
        "l15_soft_final_jump_high": ("l15_soft_final_denoise_jump_now", True),
        "deep_hard_concentration_delta2_high": (
            "deep_hard_concentration_delta2",
            True,
        ),
        "deep_hard_occupancy_delta2_high": ("deep_hard_occupancy_delta2", True),
        "deep_hard_support_size_delta2_low": ("deep_hard_support_size_delta2", False),
        "l15_hard_concentration_now_high": ("l15_hard_concentration_now", True),
        "l15_hard_occupancy_now_high": ("l15_hard_occupancy_now", True),
        "l15_hard_support_size_now_low": ("l15_hard_support_size_now", False),
        "l15_hard_final_jump_high": ("l15_hard_final_denoise_jump_now", True),
    }
    ranks = {
        name: pool_rank(features[feature], group, valid, high)
        for name, (feature, high) in directions.items()
    }

    loop_precursor = finite_mean(
        ranks["gate_concentration_delta2_high"],
        ranks["late_flow_volatility_w4_high"],
        ranks["route_acceleration_w4_high"],
    )
    loop_soft_lock = finite_mean(
        ranks["deep_soft_consensus_delta2_high"],
        ranks["deep_soft_mixture_delta2_high"],
    )
    loop_hard_lock = finite_mean(
        ranks["deep_hard_concentration_delta2_high"],
        ranks["deep_hard_occupancy_delta2_high"],
        ranks["deep_hard_support_size_delta2_low"],
    )
    static_soft = finite_mean(
        ranks["gate_concentration_w4_low"],
        ranks["gate_concentration_delta2_low"],
        ranks["l15_soft_mixture_now_high"],
        ranks["l15_soft_consensus_now_high"],
        ranks["l15_soft_final_jump_high"],
    )
    static_hard = finite_mean(
        ranks["gate_concentration_w4_low"],
        ranks["gate_concentration_delta2_low"],
        ranks["l15_hard_concentration_now_high"],
        ranks["l15_hard_occupancy_now_high"],
        ranks["l15_hard_support_size_now_low"],
        ranks["l15_hard_final_jump_high"],
    )
    scores = {
        "loop_precursor": loop_precursor,
        "loop_soft_lock": loop_soft_lock,
        "loop_hard_lock": loop_hard_lock,
        "loop_soft": np.maximum(loop_precursor, loop_soft_lock),
        "loop_hard": np.maximum(loop_precursor, loop_hard_lock),
        "static_soft": static_soft,
        "static_hard": static_hard,
        "single_mobility": ranks["route_mobility_w4_low"],
    }
    scores["double_soft_max"] = np.maximum(scores["loop_soft"], scores["static_soft"])
    scores["single_soft_mean"] = finite_mean(scores["loop_soft"], scores["static_soft"])
    scores["double_hard_max"] = np.maximum(scores["loop_hard"], scores["static_hard"])
    scores["single_hard_mean"] = finite_mean(scores["loop_hard"], scores["static_hard"])

    np.savez_compressed(
        RESULTS / "unlabeled_scores.npz",
        schema=np.asarray(SCHEMA),
        episode=episode,
        group=group,
        length=length,
        valid=valid,
        **scores,
        **{f"component__{name}": value for name, value in ranks.items()},
    )
    snapshot_rows = []
    for query in (30, 34):
        for index in range(len(episode)):
            row = {
                "episode_id": int(episode[index]),
                "group": int(group[index]),
                "query": query,
            }
            row.update(
                {name: float(value[index, query]) for name, value in scores.items()}
            )
            snapshot_rows.append(row)
    pd.DataFrame(snapshot_rows).to_csv(
        RESULTS / "unlabeled_snapshot_scores.csv", index=False
    )

    assignment_rows = []
    for query in (30, 34):
        for budget in (4, 8, 16):
            for pool in np.unique(group):
                pool_indices = np.flatnonzero((group == pool) & valid[:, query])
                loop_selected = stable_topk(
                    pool_indices, scores["loop_soft"][:, query], episode, budget
                )
                static_selected = stable_topk(
                    pool_indices, scores["static_soft"][:, query], episode, budget
                )
                loop_flag = np.isin(pool_indices, loop_selected)
                static_flag = np.isin(pool_indices, static_selected)
                for local_index, index in enumerate(pool_indices):
                    if loop_flag[local_index] and static_flag[local_index]:
                        decision = "both"
                    elif loop_flag[local_index]:
                        decision = "loop"
                    elif static_flag[local_index]:
                        decision = "static"
                    else:
                        decision = "none"
                    assignment_rows.append(
                        {
                            "episode_id": int(episode[index]),
                            "group": int(pool),
                            "query": query,
                            "budget_per_head": budget,
                            "loop_score": float(scores["loop_soft"][index, query]),
                            "static_score": float(scores["static_soft"][index, query]),
                            "loop_selected": bool(loop_flag[local_index]),
                            "static_selected": bool(static_flag[local_index]),
                            "route_decision": decision,
                        }
                    )
    pd.DataFrame(assignment_rows).to_csv(
        RESULTS / "unlabeled_assignments.csv", index=False
    )

    manifest = {
        "schema": SCHEMA,
        "created": "2026-09-03",
        "train_free": True,
        "label_columns_read": [],
        "fit_calls": 0,
        "calibrated_parameters": 0,
        "normalization": "within-query, within-initial-state percentile rank",
        "protocol_sha256": sha256(PROTOCOL),
        "formulas": PRIMARY_FORMULAS,
        "n_trajectories": int(len(episode)),
        "n_initial_state_pools": int(len(np.unique(group))),
        "score_file": "unlabeled_scores.npz",
        "assignment_file": "unlabeled_assignments.csv",
    }
    (RESULTS / "score_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(RESULTS / "unlabeled_scores.npz")
    print("LABEL_BLIND_SCORE_BUILD_OK")


if __name__ == "__main__":
    main()
