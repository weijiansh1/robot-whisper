#!/usr/bin/env python3
"""LOIO comparison of single- and double-head physical-Trap selectors.

The script reads the frozen corpus-B caches and route store, builds only causal
routing features, and writes every derived artifact below ``double-selete``.
See PROTOCOL.md for the frozen primary comparison and interpretation limits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import pickle
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import zarr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAINFREE_CODE = ROOT / "himoe-vla_trap/code"
sys.path.insert(0, str(TRAINFREE_CODE))
import analyze_trainfree_signal_matrix as trainfree  # noqa: E402


B_ROOT = (
    ROOT
    / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
)
DIST_CACHE = ROOT / "analysis_online_2026-08-29/data/distB.pkl"
ROW_FEATURE_CACHE = (
    ROOT
    / "himoe-vla_trap/results/trainfree_signal_matrix/intermediate/"
    "corpus_B_route_features.npz"
)
SUPPORT_CACHE = HERE / "cache/support_features_v1.npz"
RESULTS = HERE / "results"

N_EXPERTS = 32
N_POOL = 32
HISTORY = 4
FIXED_C = 0.1
RANK_TIMES = (30, 34)
BUDGETS = (4, 8, 16)
SCAN_QUERIES = tuple(range(12, 35))
PRIMARY_TIME = 34
PRIMARY_BUDGET = 8
FALSE_ALARM_TARGET = 0.08
SEED = 20260903
SCHEMA = "himoe.double_selector.v1"


LOOP_FEATURES = (
    "route_mobility_now",
    "route_mobility_w4",
    "gate_concentration_now",
    "gate_concentration_delta2",
    "gate_concentration_range4",
    "late_flow_volatility_w4",
    "route_acceleration_w4",
    "deep_support_concentration_now",
    "deep_max_occupancy_now",
    "deep_support_concentration_delta2",
)

STATIC_FEATURES = (
    "gate_concentration_now",
    "gate_concentration_w4",
    "gate_concentration_delta2",
    "top12_margin_w4",
    "token_disagreement_w4",
    "l15_support_concentration_now",
    "l15_support_concentration_w4",
    "l15_max_occupancy_now",
    "l15_support_size_now",
    "l15_final_denoise_jump_now",
)

ALL_FEATURES = tuple(dict.fromkeys((*LOOP_FEATURES, *STATIC_FEATURES)))

HARD_ID_FEATURES = {
    "deep_support_concentration_now",
    "deep_max_occupancy_now",
    "deep_support_concentration_delta2",
    "l15_support_concentration_now",
    "l15_support_concentration_w4",
    "l15_max_occupancy_now",
    "l15_support_size_now",
    "l15_final_denoise_jump_now",
}

LOOP_SOFT_FEATURES = tuple(name for name in LOOP_FEATURES if name not in HARD_ID_FEATURES)
STATIC_SOFT_FEATURES = tuple(
    name for name in STATIC_FEATURES if name not in HARD_ID_FEATURES
)
ALL_SOFT_FEATURES = tuple(name for name in ALL_FEATURES if name not in HARD_ID_FEATURES)


@dataclass(frozen=True)
class Dataset:
    episode: np.ndarray
    group: np.ndarray
    success: np.ndarray
    length: np.ndarray
    loop_onset: np.ndarray
    static_onset: np.ndarray
    trap_onset: np.ndarray
    features: dict[str, np.ndarray]

    @property
    def loop(self) -> np.ndarray:
        return self.loop_onset >= 0

    @property
    def static(self) -> np.ndarray:
        return self.static_onset >= 0

    @property
    def trap(self) -> np.ndarray:
        return self.loop | self.static


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--rebuild-support-cache", action="store_true")
    parser.add_argument("--skip-sequential", action="store_true")
    return parser.parse_args()


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, np.float64)
    for t in range(width - 1, values.shape[1]):
        block = values[:, t - width + 1 : t + 1]
        good = np.isfinite(block).all(axis=1)
        out[good, t] = block[good].mean(axis=1)
    return out


def trailing_range(values: np.ndarray, width: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, np.float64)
    for t in range(width - 1, values.shape[1]):
        block = values[:, t - width + 1 : t + 1]
        good = np.isfinite(block).all(axis=1)
        out[good, t] = np.ptp(block[good], axis=1)
    return out


def lag_delta(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, np.float64)
    good = np.isfinite(values[:, lag:]) & np.isfinite(values[:, :-lag])
    delta = values[:, lag:] - values[:, :-lag]
    out[:, lag:][good] = delta[good]
    return out


def scatter_rows(values: np.ndarray, rowidx: np.ndarray) -> np.ndarray:
    out = np.full(rowidx.shape, np.nan, np.float64)
    valid = rowidx >= 0
    out[valid] = values[rowidx[valid]]
    return out


def support_metrics(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concentration, maximum token occupancy, and unique support size.

    ``ids`` has shape [site, token, top-k]. A selected expert can appear at
    most once per token, so max histogram count / n_token is the fraction of
    tokens sharing the most common selected expert.
    """

    ids = np.asarray(ids, np.int64)
    n_site, n_token, top_k = ids.shape
    hist = np.zeros((n_site, N_EXPERTS), np.int16)
    row = np.repeat(np.arange(n_site), n_token * top_k)
    np.add.at(hist, (row, ids.reshape(-1)), 1)
    prob = hist / float(n_token * top_k)
    log_prob = np.zeros_like(prob)
    positive = prob > 0
    log_prob[positive] = np.log(prob[positive])
    entropy = -(prob * log_prob).sum(axis=1)
    concentration = 1.0 - entropy / math.log(N_EXPERTS)
    max_occupancy = hist.max(axis=1) / float(n_token)
    support_size = (hist > 0).sum(axis=1).astype(np.float64)
    return concentration, max_occupancy, support_size


def extract_support_features(rebuild: bool = False) -> dict[str, np.ndarray]:
    SUPPORT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if SUPPORT_CACHE.exists() and not rebuild:
        with np.load(SUPPORT_CACHE) as data:
            if str(data["schema"].item()) == SCHEMA:
                return {k: np.asarray(data[k]) for k in data.files if k != "schema"}

    route = zarr.open_group(str(B_ROOT / "server/routes.zarr"), mode="r")
    store = route["hb_expert_ids"]
    n_rows = int(store.shape[0])
    output = {
        "deep_support_concentration_d9": np.empty(n_rows, np.float32),
        "deep_max_occupancy_d9": np.empty(n_rows, np.float32),
        "deep_support_size_d9": np.empty(n_rows, np.float32),
        "l15_support_concentration_d9": np.empty(n_rows, np.float32),
        "l15_max_occupancy_d9": np.empty(n_rows, np.float32),
        "l15_support_size_d9": np.empty(n_rows, np.float32),
        "l15_final_denoise_jump": np.empty(n_rows, np.float32),
    }
    block_size = 256
    for start in range(0, n_rows, block_size):
        stop = min(n_rows, start + block_size)
        # [row, deep layer, denoise, action token, selected expert]
        ids = np.asarray(store[start:stop, 4:8, :, 1:11, :], np.int64)
        n = stop - start
        conc, occ, size = support_metrics(ids.reshape(-1, 10, 4))
        conc = conc.reshape(n, 4, 10)
        occ = occ.reshape(n, 4, 10)
        size = size.reshape(n, 4, 10)
        output["deep_support_concentration_d9"][start:stop] = conc[:, :, 9].mean(1)
        output["deep_max_occupancy_d9"][start:stop] = occ[:, :, 9].mean(1)
        output["deep_support_size_d9"][start:stop] = size[:, :, 9].mean(1)
        output["l15_support_concentration_d9"][start:stop] = conc[:, 3, 9]
        output["l15_max_occupancy_d9"][start:stop] = occ[:, 3, 9]
        output["l15_support_size_d9"][start:stop] = size[:, 3, 9]
        output["l15_final_denoise_jump"][start:stop] = conc[:, 3, 9] - conc[:, 3, 8]

    np.savez_compressed(SUPPORT_CACHE, schema=np.asarray(SCHEMA), **output)
    return output


def route_mobility_series(rowidx: np.ndarray) -> np.ndarray:
    with DIST_CACHE.open("rb") as handle:
        cache = pickle.load(handle)
    series = cache["H"]["ac"]["hellinger"]
    out = np.full(rowidx.shape, np.nan, np.float64)
    for i, values in enumerate(series):
        values = np.asarray(values, np.float64)
        out[i, 1 : len(values) + 1] = values
    return out


def load_dataset(rebuild_support: bool = False) -> tuple[Dataset, pd.DataFrame]:
    corpus, inventory = trainfree.load_corpus("B")
    rowidx = corpus.rowidx
    with np.load(ROW_FEATURE_CACHE) as raw:
        row_features = {
            key: np.asarray(raw[key])
            for key in (
                "gate_entropy",
                "top12_margin",
                "late_flow_volatility",
                "route_acceleration",
                "token_disagreement",
            )
        }
    row_features.update(extract_support_features(rebuild_support))
    series = {key: scatter_rows(value, rowidx) for key, value in row_features.items()}

    gate_concentration = 1.0 - series["gate_entropy"]
    mobility = route_mobility_series(rowidx)
    features = {
        "route_mobility_now": mobility,
        "route_mobility_w4": trailing_mean(mobility, HISTORY),
        "gate_concentration_now": gate_concentration,
        "gate_concentration_w4": trailing_mean(gate_concentration, HISTORY),
        "gate_concentration_delta2": lag_delta(gate_concentration, 2),
        "gate_concentration_range4": trailing_range(gate_concentration, HISTORY),
        "late_flow_volatility_w4": trailing_mean(
            series["late_flow_volatility"], HISTORY
        ),
        "route_acceleration_w4": trailing_mean(series["route_acceleration"], HISTORY),
        "top12_margin_w4": trailing_mean(series["top12_margin"], HISTORY),
        "token_disagreement_w4": trailing_mean(
            series["token_disagreement"], HISTORY
        ),
        "deep_support_concentration_now": series[
            "deep_support_concentration_d9"
        ],
        "deep_max_occupancy_now": series["deep_max_occupancy_d9"],
        "deep_support_concentration_delta2": lag_delta(
            series["deep_support_concentration_d9"], 2
        ),
        "l15_support_concentration_now": series[
            "l15_support_concentration_d9"
        ],
        "l15_support_concentration_w4": trailing_mean(
            series["l15_support_concentration_d9"], HISTORY
        ),
        "l15_max_occupancy_now": series["l15_max_occupancy_d9"],
        "l15_support_size_now": series["l15_support_size_d9"],
        "l15_final_denoise_jump_now": series["l15_final_denoise_jump"],
    }

    meta = corpus.meta
    episode = meta.episode_id.to_numpy(np.int64)
    if not np.array_equal(episode, np.arange(len(meta))):
        raise RuntimeError("corpus-B episode order is not 0..511")
    dataset = Dataset(
        episode=episode,
        group=meta.group.to_numpy(np.int64),
        success=meta.success.to_numpy(bool),
        length=meta["T"].to_numpy(np.int64),
        loop_onset=np.asarray(corpus.loop_onset, np.int64),
        static_onset=np.asarray(corpus.static_onset, np.int64),
        trap_onset=np.asarray(corpus.trap_onset, np.int64),
        features=features,
    )
    validate_dataset(dataset)
    return dataset, inventory


def validate_dataset(data: Dataset) -> None:
    if len(data.episode) != 512 or len(np.unique(data.group)) != 16:
        raise RuntimeError("expected 512 trajectories and 16 initial states")
    counts = np.unique(data.group, return_counts=True)[1]
    if not np.all(counts == N_POOL):
        raise RuntimeError("each initial state must contain exactly 32 trajectories")
    if data.length.min() < 35:
        raise RuntimeError("query 34 is not an equal-opportunity snapshot")
    for name in ALL_FEATURES:
        values = data.features[name][:, SCAN_QUERIES]
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite causal feature in clean window: {name}")


def feature_matrix(
    data: Dataset, names: Iterable[str], episodes: np.ndarray, queries: np.ndarray
) -> np.ndarray:
    names = tuple(names)
    return np.column_stack([data.features[name][episodes, queries] for name in names])


def fit_probability(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_test), float(y_train.mean()))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=FIXED_C,
            class_weight="balanced",
            max_iter=2_000,
            random_state=SEED,
            solver="lbfgs",
        ),
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def predict_bundle(
    data: Dataset,
    train_episode: np.ndarray,
    test_episode: np.ndarray,
    train_query: np.ndarray,
    test_query: np.ndarray,
) -> dict[str, np.ndarray]:
    all_train = feature_matrix(data, ALL_FEATURES, train_episode, train_query)
    all_test = feature_matrix(data, ALL_FEATURES, test_episode, test_query)
    loop_train = feature_matrix(data, LOOP_FEATURES, train_episode, train_query)
    loop_test = feature_matrix(data, LOOP_FEATURES, test_episode, test_query)
    static_train = feature_matrix(data, STATIC_FEATURES, train_episode, train_query)
    static_test = feature_matrix(data, STATIC_FEATURES, test_episode, test_query)
    soft_all_train = feature_matrix(
        data, ALL_SOFT_FEATURES, train_episode, train_query
    )
    soft_all_test = feature_matrix(data, ALL_SOFT_FEATURES, test_episode, test_query)
    soft_loop_train = feature_matrix(
        data, LOOP_SOFT_FEATURES, train_episode, train_query
    )
    soft_loop_test = feature_matrix(
        data, LOOP_SOFT_FEATURES, test_episode, test_query
    )
    soft_static_train = feature_matrix(
        data, STATIC_SOFT_FEATURES, train_episode, train_query
    )
    soft_static_test = feature_matrix(
        data, STATIC_SOFT_FEATURES, test_episode, test_query
    )
    return {
        "single": fit_probability(all_train, data.trap[train_episode], all_test),
        "single_soft": fit_probability(
            soft_all_train, data.trap[train_episode], soft_all_test
        ),
        "shared_loop": fit_probability(all_train, data.loop[train_episode], all_test),
        "shared_static": fit_probability(
            all_train, data.static[train_episode], all_test
        ),
        "typed_loop": fit_probability(loop_train, data.loop[train_episode], loop_test),
        "typed_static": fit_probability(
            static_train, data.static[train_episode], static_test
        ),
        "typed_soft_loop": fit_probability(
            soft_loop_train, data.loop[train_episode], soft_loop_test
        ),
        "typed_soft_static": fit_probability(
            soft_static_train, data.static[train_episode], soft_static_test
        ),
    }


def within_group_percentile(score: np.ndarray, group: np.ndarray) -> np.ndarray:
    out = np.empty(len(score), np.float64)
    for value in np.unique(group):
        idx = np.flatnonzero(group == value)
        if len(idx) == 1:
            out[idx] = 0.5
        else:
            out[idx] = (rankdata(score[idx], method="average") - 1.0) / (
                len(idx) - 1.0
            )
    return out


def fixed_time_oof(data: Dataset, query: int) -> pd.DataFrame:
    score_names = (
        "single",
        "single_soft",
        "shared_loop",
        "shared_static",
        "typed_loop",
        "typed_static",
        "typed_soft_loop",
        "typed_soft_static",
    )
    scores = {name: np.full(len(data.episode), np.nan) for name in score_names}
    for held_out in np.unique(data.group):
        train = np.flatnonzero(data.group != held_out)
        test = np.flatnonzero(data.group == held_out)
        pred = predict_bundle(
            data,
            train,
            test,
            np.full(len(train), query),
            np.full(len(test), query),
        )
        for name in score_names:
            scores[name][test] = pred[name]

    shared_loop_rank = within_group_percentile(scores["shared_loop"], data.group)
    shared_static_rank = within_group_percentile(scores["shared_static"], data.group)
    typed_loop_rank = within_group_percentile(scores["typed_loop"], data.group)
    typed_static_rank = within_group_percentile(scores["typed_static"], data.group)
    typed_soft_loop_rank = within_group_percentile(
        scores["typed_soft_loop"], data.group
    )
    typed_soft_static_rank = within_group_percentile(
        scores["typed_soft_static"], data.group
    )
    return pd.DataFrame(
        {
            "episode_id": data.episode,
            "init_state_id": data.group,
            "query": query,
            "success": data.success,
            "loop": data.loop,
            "static": data.static,
            "trap": data.trap,
            "loop_onset": data.loop_onset,
            "static_onset": data.static_onset,
            "trap_onset": data.trap_onset,
            **scores,
            "double_shared_max": np.maximum(shared_loop_rank, shared_static_rank),
            "double_shared_noisy_or": 1.0
            - (1.0 - scores["shared_loop"]) * (1.0 - scores["shared_static"]),
            "double_typed_max": np.maximum(typed_loop_rank, typed_static_rank),
            "double_typed_noisy_or": 1.0
            - (1.0 - scores["typed_loop"]) * (1.0 - scores["typed_static"]),
            "double_typed_soft_max": np.maximum(
                typed_soft_loop_rank, typed_soft_static_rank
            ),
            "double_typed_soft_noisy_or": 1.0
            - (1.0 - scores["typed_soft_loop"])
            * (1.0 - scores["typed_soft_static"]),
        }
    )


def stable_topk(score: np.ndarray, episode: np.ndarray, k: int) -> np.ndarray:
    order = np.lexsort((episode, -np.asarray(score)))
    return order[:k]


def quota_topk(
    loop_score: np.ndarray,
    static_score: np.ndarray,
    episode: np.ndarray,
    k: int,
) -> np.ndarray:
    n_loop = k // 2
    n_static = k - n_loop
    loop_pick = stable_topk(loop_score, episode, n_loop)
    static_pick = stable_topk(static_score, episode, n_static)
    chosen = list(dict.fromkeys([*loop_pick.tolist(), *static_pick.tolist()]))
    if len(chosen) < k:
        loop_rank = (rankdata(loop_score, method="average") - 1.0) / (len(episode) - 1)
        static_rank = (rankdata(static_score, method="average") - 1.0) / (
            len(episode) - 1
        )
        fill_order = stable_topk(np.maximum(loop_rank, static_rank), episode, len(episode))
        for candidate in fill_order:
            if int(candidate) not in chosen:
                chosen.append(int(candidate))
            if len(chosen) == k:
                break
    return np.asarray(chosen, np.int64)


def selection_indices(frame: pd.DataFrame, method: str, k: int) -> np.ndarray:
    episode = frame.episode_id.to_numpy(np.int64)
    if method == "single":
        return stable_topk(frame.single.to_numpy(), episode, k)
    if method == "single_soft":
        return stable_topk(frame.single_soft.to_numpy(), episode, k)
    if method == "double_shared_max":
        return stable_topk(frame.double_shared_max.to_numpy(), episode, k)
    if method == "double_shared_noisy_or":
        return stable_topk(frame.double_shared_noisy_or.to_numpy(), episode, k)
    if method == "double_typed_max":
        return stable_topk(frame.double_typed_max.to_numpy(), episode, k)
    if method == "double_typed_noisy_or":
        return stable_topk(frame.double_typed_noisy_or.to_numpy(), episode, k)
    if method == "double_typed_soft_max":
        return stable_topk(frame.double_typed_soft_max.to_numpy(), episode, k)
    if method == "double_typed_soft_noisy_or":
        return stable_topk(
            frame.double_typed_soft_noisy_or.to_numpy(), episode, k
        )
    if method == "double_shared_quota":
        return quota_topk(
            frame.shared_loop.to_numpy(), frame.shared_static.to_numpy(), episode, k
        )
    if method == "double_typed_quota":
        return quota_topk(
            frame.typed_loop.to_numpy(), frame.typed_static.to_numpy(), episode, k
        )
    if method == "double_typed_soft_quota":
        return quota_topk(
            frame.typed_soft_loop.to_numpy(),
            frame.typed_soft_static.to_numpy(),
            episode,
            k,
        )
    raise KeyError(method)


METHODS = (
    "single",
    "single_soft",
    "double_shared_max",
    "double_shared_noisy_or",
    "double_shared_quota",
    "double_typed_max",
    "double_typed_noisy_or",
    "double_typed_quota",
    "double_typed_soft_max",
    "double_typed_soft_noisy_or",
    "double_typed_soft_quota",
)


def scoped_labels(data: Dataset, query: int, scope: str) -> tuple[np.ndarray, ...]:
    if scope == "eventual":
        loop, static = data.loop, data.static
    elif scope == "onset_by_query":
        loop = (data.loop_onset >= 0) & (data.loop_onset <= query)
        static = (data.static_onset >= 0) & (data.static_onset <= query)
    else:
        raise KeyError(scope)
    return loop | static, loop, static


def group_selection_rows(
    data: Dataset, score_frame: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict] = []
    query = int(score_frame["query"].iloc[0])
    for budget in BUDGETS:
        for method in METHODS:
            for group in np.unique(data.group):
                pool = score_frame[score_frame.init_state_id == group].reset_index(drop=True)
                chosen_local = selection_indices(pool, method, budget)
                chosen_episode = pool.iloc[chosen_local].episode_id.to_numpy(np.int64)
                selected = np.isin(data.episode, chosen_episode)
                for scope in ("eventual", "onset_by_query"):
                    trap, loop, static = scoped_labels(data, query, scope)
                    in_group = data.group == group
                    rows.append(
                        {
                            "query": query,
                            "budget": budget,
                            "method": method,
                            "label_scope": scope,
                            "init_state_id": int(group),
                            "pool_n": int(in_group.sum()),
                            "selected_n": int((selected & in_group).sum()),
                            "positive_trap": int((trap & in_group).sum()),
                            "positive_loop": int((loop & in_group).sum()),
                            "positive_static": int((static & in_group).sum()),
                            "selected_trap": int((selected & trap & in_group).sum()),
                            "selected_loop": int((selected & loop & in_group).sum()),
                            "selected_static": int((selected & static & in_group).sum()),
                            "selected_episode_ids": ";".join(map(str, chosen_episode)),
                        }
                    )
    return pd.DataFrame(rows)


def exact_random_hit(n_positive: int, k: int, n: int = N_POOL) -> float:
    if n_positive <= 0:
        return np.nan
    if n - n_positive < k:
        return 1.0
    return 1.0 - math.comb(n - n_positive, k) / math.comb(n, k)


def metrics_from_group_rows(rows: pd.DataFrame) -> dict[str, float]:
    selected = float(rows.selected_n.sum())
    n_trap = float(rows.positive_trap.sum())
    n_loop = float(rows.positive_loop.sum())
    n_static = float(rows.positive_static.sum())
    recall_loop = rows.selected_loop.sum() / n_loop if n_loop else np.nan
    recall_static = rows.selected_static.sum() / n_static if n_static else np.nan
    eligible_loop = rows.positive_loop > 0
    eligible_static = rows.positive_static > 0
    eligible_both = eligible_loop & eligible_static
    typed_recalls = np.asarray([recall_loop, recall_static], np.float64)
    macro_type_recall = (
        float(np.nanmean(typed_recalls))
        if np.isfinite(typed_recalls).any()
        else np.nan
    )
    return {
        "precision_trap": rows.selected_trap.sum() / selected if selected else np.nan,
        "recall_trap": rows.selected_trap.sum() / n_trap if n_trap else np.nan,
        "recall_loop": recall_loop,
        "recall_static": recall_static,
        "macro_type_recall": macro_type_recall,
        "state_macro_precision": float((rows.selected_trap / rows.selected_n).mean()),
        "pool_hit_loop": float((rows.loc[eligible_loop, "selected_loop"] > 0).mean())
        if eligible_loop.any()
        else np.nan,
        "pool_hit_static": float(
            (rows.loc[eligible_static, "selected_static"] > 0).mean()
        )
        if eligible_static.any()
        else np.nan,
        "pool_hit_both_types": float(
            (
                (rows.loc[eligible_both, "selected_loop"] > 0)
                & (rows.loc[eligible_both, "selected_static"] > 0)
            ).mean()
        )
        if eligible_both.any()
        else np.nan,
    }


def summarise_selection(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["query", "budget", "method", "label_scope"]
    for key, part in detail.groupby(keys, sort=False):
        values = dict(zip(keys, key))
        values.update(metrics_from_group_rows(part))
        values["selected_n"] = int(part.selected_n.sum())
        values["positive_trap_n"] = int(part.positive_trap.sum())
        values["selection_lift"] = values["precision_trap"] / (
            part.positive_trap.sum() / part.pool_n.sum()
        ) if part.positive_trap.sum() else np.nan
        rows.append(values)

    for query in sorted(detail["query"].unique()):
        for budget in BUDGETS:
            for scope in ("eventual", "onset_by_query"):
                part = detail[
                    (detail["query"] == query)
                    & (detail.budget == budget)
                    & (detail.method == "single")
                    & (detail.label_scope == scope)
                ]
                n_trap = part.positive_trap.sum()
                n_loop = part.positive_loop.sum()
                n_static = part.positive_static.sum()
                eligible_loop = part[part.positive_loop > 0]
                eligible_static = part[part.positive_static > 0]
                eligible_both = part[(part.positive_loop > 0) & (part.positive_static > 0)]
                rows.append(
                    {
                        "query": query,
                        "budget": budget,
                        "method": "random_expectation",
                        "label_scope": scope,
                        "precision_trap": n_trap / part.pool_n.sum() if n_trap else np.nan,
                        "recall_trap": budget / N_POOL if n_trap else np.nan,
                        "recall_loop": budget / N_POOL if n_loop else np.nan,
                        "recall_static": budget / N_POOL if n_static else np.nan,
                        "macro_type_recall": budget / N_POOL
                        if n_loop and n_static
                        else np.nan,
                        "state_macro_precision": float(
                            (part.positive_trap / part.pool_n).mean()
                        ),
                        "pool_hit_loop": float(
                            np.mean(
                                [
                                    exact_random_hit(int(v), budget)
                                    for v in eligible_loop.positive_loop
                                ]
                            )
                        )
                        if len(eligible_loop)
                        else np.nan,
                        "pool_hit_static": float(
                            np.mean(
                                [
                                    exact_random_hit(int(v), budget)
                                    for v in eligible_static.positive_static
                                ]
                            )
                        )
                        if len(eligible_static)
                        else np.nan,
                        "pool_hit_both_types": np.nan
                        if not len(eligible_both)
                        else np.nan,
                        "selected_n": int(len(part) * budget),
                        "positive_trap_n": int(n_trap),
                        "selection_lift": 1.0,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["query", "budget", "label_scope", "method"]
    )


BOOT_METRICS = (
    "precision_trap",
    "recall_trap",
    "recall_loop",
    "recall_static",
    "macro_type_recall",
)


def bootstrap_selection_deltas(detail: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 71)
    output = []
    keys = ["query", "budget", "label_scope"]
    for key, block in detail.groupby(keys, sort=False):
        baseline = block[block.method == "single"].sort_values("init_state_id")
        n_group = len(baseline)
        take = rng.integers(0, n_group, size=(n_boot, n_group))

        def draw_metrics(rows: pd.DataFrame) -> dict[str, np.ndarray]:
            def total(column: str) -> np.ndarray:
                values = rows[column].to_numpy(np.float64)
                return values[take].sum(axis=1)

            selected = total("selected_n")
            positive_trap = total("positive_trap")
            positive_loop = total("positive_loop")
            positive_static = total("positive_static")
            selected_trap = total("selected_trap")
            selected_loop = total("selected_loop")
            selected_static = total("selected_static")
            with np.errstate(divide="ignore", invalid="ignore"):
                precision = selected_trap / selected
                recall_trap = selected_trap / positive_trap
                recall_loop = selected_loop / positive_loop
                recall_static = selected_static / positive_static
            recall_trap[positive_trap == 0] = np.nan
            recall_loop[positive_loop == 0] = np.nan
            recall_static[positive_static == 0] = np.nan
            count = np.isfinite(recall_loop).astype(int) + np.isfinite(
                recall_static
            ).astype(int)
            macro = np.nansum(
                np.column_stack([recall_loop, recall_static]), axis=1
            ) / np.maximum(count, 1)
            macro[count == 0] = np.nan
            return {
                "precision_trap": precision,
                "recall_trap": recall_trap,
                "recall_loop": recall_loop,
                "recall_static": recall_static,
                "macro_type_recall": macro,
            }

        baseline_draws = draw_metrics(baseline)
        for method in METHODS:
            if method == "single":
                continue
            candidate = block[block.method == method].sort_values("init_state_id")
            if not np.array_equal(
                baseline.init_state_id.to_numpy(), candidate.init_state_id.to_numpy()
            ):
                raise RuntimeError("bootstrap rows are not paired by initial state")
            observed_base = metrics_from_group_rows(baseline)
            observed_candidate = metrics_from_group_rows(candidate)
            candidate_draws = draw_metrics(candidate)
            for metric in BOOT_METRICS:
                values = candidate_draws[metric] - baseline_draws[metric]
                values = values[np.isfinite(values)]
                observed = observed_candidate[metric] - observed_base[metric]
                if len(values):
                    ci_low = float(np.quantile(values, 0.025))
                    ci_high = float(np.quantile(values, 0.975))
                    p_two_sided = float(
                        min(
                            1.0,
                            2.0
                            * min(
                                (1 + (values <= 0).sum()) / (len(values) + 1),
                                (1 + (values >= 0).sum()) / (len(values) + 1),
                            ),
                        )
                    )
                else:
                    ci_low = ci_high = p_two_sided = np.nan
                output.append(
                    {
                        **dict(zip(keys, key)),
                        "method": method,
                        "baseline": "single",
                        "metric": metric,
                        "delta": observed,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "bootstrap_p_two_sided": p_two_sided,
                        "n_groups": n_group,
                        "n_bootstrap": n_boot,
                        "n_bootstrap_effective": len(values),
                    }
                )
    return pd.DataFrame(output)


def within_group_auc(
    score: np.ndarray, positive: np.ndarray, group: np.ndarray
) -> tuple[float, int]:
    concordant = 0.0
    pairs = 0
    for value in np.unique(group):
        idx = group == value
        a = score[idx & positive]
        b = score[idx & ~positive]
        if not len(a) or not len(b):
            continue
        concordant += (a[:, None] > b).sum() + 0.5 * (a[:, None] == b).sum()
        pairs += len(a) * len(b)
    return (concordant / pairs if pairs else np.nan), pairs


def score_auc_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    continuous = (
        "single",
        "single_soft",
        "shared_loop",
        "shared_static",
        "typed_loop",
        "typed_static",
        "typed_soft_loop",
        "typed_soft_static",
        "double_shared_max",
        "double_shared_noisy_or",
        "double_typed_max",
        "double_typed_noisy_or",
        "double_typed_soft_max",
        "double_typed_soft_noisy_or",
    )
    for query, block in scores.groupby("query"):
        group = block.init_state_id.to_numpy()
        for score_name in continuous:
            score = block[score_name].to_numpy()
            for target in ("trap", "loop", "static"):
                auc, pairs = within_group_auc(score, block[target].to_numpy(bool), group)
                rows.append(
                    {
                        "query": int(query),
                        "score": score_name,
                        "target": target,
                        "within_group_auc": auc,
                        "pairs": pairs,
                    }
                )
    return pd.DataFrame(rows)


INDEPENDENT_FAMILIES = {
    "shared": ("shared_loop", "shared_static"),
    "typed": ("typed_loop", "typed_static"),
    "typed_soft": ("typed_soft_loop", "typed_soft_static"),
}


def independent_head_rows(data: Dataset, score_frame: pd.DataFrame) -> pd.DataFrame:
    """Each subtype head gets its own complete K budget; union is not capped."""

    query = int(score_frame["query"].iloc[0])
    rows = []
    for budget in BUDGETS:
        for family, (loop_column, static_column) in INDEPENDENT_FAMILIES.items():
            for group in np.unique(data.group):
                pool = score_frame[score_frame.init_state_id == group].reset_index(
                    drop=True
                )
                episode = pool.episode_id.to_numpy(np.int64)
                loop_pick = stable_topk(
                    pool[loop_column].to_numpy(), episode, budget
                )
                static_pick = stable_topk(
                    pool[static_column].to_numpy(), episode, budget
                )
                single_pick = stable_topk(pool.single.to_numpy(), episode, budget)
                union_pick = np.union1d(loop_pick, static_pick)
                matched_single_pick = stable_topk(
                    pool.single.to_numpy(), episode, len(union_pick)
                )
                loop = pool.loop.to_numpy(bool)
                static = pool.static.to_numpy(bool)
                trap = loop | static
                rows.append(
                    {
                        "query": query,
                        "per_head_budget": budget,
                        "family": family,
                        "init_state_id": int(group),
                        "pool_n": len(pool),
                        "positive_trap": int(trap.sum()),
                        "positive_loop": int(loop.sum()),
                        "positive_static": int(static.sum()),
                        "loop_head_selected_n": len(loop_pick),
                        "loop_head_selected_loop": int(loop[loop_pick].sum()),
                        "static_head_selected_n": len(static_pick),
                        "static_head_selected_static": int(static[static_pick].sum()),
                        "single_k_selected_n": len(single_pick),
                        "single_k_selected_loop": int(loop[single_pick].sum()),
                        "single_k_selected_static": int(static[single_pick].sum()),
                        "head_overlap_n": int(2 * budget - len(union_pick)),
                        "union_selected_n": len(union_pick),
                        "union_selected_trap": int(trap[union_pick].sum()),
                        "union_selected_loop": int(loop[union_pick].sum()),
                        "union_selected_static": int(static[union_pick].sum()),
                        "matched_single_selected_n": len(matched_single_pick),
                        "matched_single_selected_trap": int(
                            trap[matched_single_pick].sum()
                        ),
                        "matched_single_selected_loop": int(
                            loop[matched_single_pick].sum()
                        ),
                        "matched_single_selected_static": int(
                            static[matched_single_pick].sum()
                        ),
                        "loop_head_episode_ids": ";".join(map(str, episode[loop_pick])),
                        "static_head_episode_ids": ";".join(
                            map(str, episode[static_pick])
                        ),
                        "union_episode_ids": ";".join(map(str, episode[union_pick])),
                    }
                )
    return pd.DataFrame(rows)


def independent_metrics(rows: pd.DataFrame) -> dict[str, float]:
    n_loop = rows.positive_loop.sum()
    n_static = rows.positive_static.sum()
    n_trap = rows.positive_trap.sum()
    loop_recall = rows.loop_head_selected_loop.sum() / n_loop
    static_recall = rows.static_head_selected_static.sum() / n_static
    union_loop_recall = rows.union_selected_loop.sum() / n_loop
    union_static_recall = rows.union_selected_static.sum() / n_static
    matched_loop_recall = rows.matched_single_selected_loop.sum() / n_loop
    matched_static_recall = rows.matched_single_selected_static.sum() / n_static
    return {
        "loop_head_precision": rows.loop_head_selected_loop.sum()
        / rows.loop_head_selected_n.sum(),
        "loop_head_recall": loop_recall,
        "single_k_loop_precision": rows.single_k_selected_loop.sum()
        / rows.single_k_selected_n.sum(),
        "single_k_loop_recall": rows.single_k_selected_loop.sum() / n_loop,
        "static_head_precision": rows.static_head_selected_static.sum()
        / rows.static_head_selected_n.sum(),
        "static_head_recall": static_recall,
        "single_k_static_precision": rows.single_k_selected_static.sum()
        / rows.single_k_selected_n.sum(),
        "single_k_static_recall": rows.single_k_selected_static.sum() / n_static,
        "independent_macro_own_recall": np.mean([loop_recall, static_recall]),
        "union_selected_total": int(rows.union_selected_n.sum()),
        "union_budget_mean": rows.union_selected_n.mean(),
        "union_overlap_mean": rows.head_overlap_n.mean(),
        "union_trap_precision": rows.union_selected_trap.sum()
        / rows.union_selected_n.sum(),
        "union_trap_recall": rows.union_selected_trap.sum() / n_trap,
        "union_loop_recall": union_loop_recall,
        "union_static_recall": union_static_recall,
        "union_macro_type_recall": np.mean(
            [union_loop_recall, union_static_recall]
        ),
        "matched_single_trap_precision": rows.matched_single_selected_trap.sum()
        / rows.matched_single_selected_n.sum(),
        "matched_single_trap_recall": rows.matched_single_selected_trap.sum()
        / n_trap,
        "matched_single_loop_recall": matched_loop_recall,
        "matched_single_static_recall": matched_static_recall,
        "matched_single_macro_type_recall": np.mean(
            [matched_loop_recall, matched_static_recall]
        ),
    }


def summarise_independent_heads(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["query", "per_head_budget", "family"]
    for key, part in detail.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        row.update(independent_metrics(part))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def bootstrap_independent_deltas(detail: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Pair head-vs-single and union-vs-size-matched-single by initial state."""

    rng = np.random.default_rng(SEED + 157)
    rows = []
    keys = ["query", "per_head_budget", "family"]
    for key, part in detail.groupby(keys, sort=False):
        part = part.sort_values("init_state_id")
        n_group = len(part)
        take = rng.integers(0, n_group, size=(n_boot, n_group))

        def total(column: str) -> np.ndarray:
            values = part[column].to_numpy(np.float64)
            return values[take].sum(axis=1)

        positive_loop = total("positive_loop")
        positive_static = total("positive_static")
        positive_trap = total("positive_trap")
        loop_head_n = total("loop_head_selected_n")
        static_head_n = total("static_head_selected_n")
        single_k_n = total("single_k_selected_n")
        union_n = total("union_selected_n")
        matched_n = total("matched_single_selected_n")
        with np.errstate(divide="ignore", invalid="ignore"):
            draws = {
                "loop_precision_head_minus_single_k": total(
                    "loop_head_selected_loop"
                )
                / loop_head_n
                - total("single_k_selected_loop") / single_k_n,
                "loop_recall_head_minus_single_k": total("loop_head_selected_loop")
                / positive_loop
                - total("single_k_selected_loop") / positive_loop,
                "static_precision_head_minus_single_k": total(
                    "static_head_selected_static"
                )
                / static_head_n
                - total("single_k_selected_static") / single_k_n,
                "static_recall_head_minus_single_k": total(
                    "static_head_selected_static"
                )
                / positive_static
                - total("single_k_selected_static") / positive_static,
                "union_trap_precision_minus_matched_single": total(
                    "union_selected_trap"
                )
                / union_n
                - total("matched_single_selected_trap") / matched_n,
                "union_trap_recall_minus_matched_single": total(
                    "union_selected_trap"
                )
                / positive_trap
                - total("matched_single_selected_trap") / positive_trap,
                "union_loop_recall_minus_matched_single": total(
                    "union_selected_loop"
                )
                / positive_loop
                - total("matched_single_selected_loop") / positive_loop,
                "union_static_recall_minus_matched_single": total(
                    "union_selected_static"
                )
                / positive_static
                - total("matched_single_selected_static") / positive_static,
            }
        draws["union_macro_recall_minus_matched_single"] = 0.5 * (
            draws["union_loop_recall_minus_matched_single"]
            + draws["union_static_recall_minus_matched_single"]
        )
        observed = independent_metrics(part)
        observed_delta = {
            "loop_precision_head_minus_single_k": observed["loop_head_precision"]
            - observed["single_k_loop_precision"],
            "loop_recall_head_minus_single_k": observed["loop_head_recall"]
            - observed["single_k_loop_recall"],
            "static_precision_head_minus_single_k": observed["static_head_precision"]
            - observed["single_k_static_precision"],
            "static_recall_head_minus_single_k": observed["static_head_recall"]
            - observed["single_k_static_recall"],
            "union_trap_precision_minus_matched_single": observed[
                "union_trap_precision"
            ]
            - observed["matched_single_trap_precision"],
            "union_trap_recall_minus_matched_single": observed["union_trap_recall"]
            - observed["matched_single_trap_recall"],
            "union_loop_recall_minus_matched_single": observed["union_loop_recall"]
            - observed["matched_single_loop_recall"],
            "union_static_recall_minus_matched_single": observed[
                "union_static_recall"
            ]
            - observed["matched_single_static_recall"],
            "union_macro_recall_minus_matched_single": observed[
                "union_macro_type_recall"
            ]
            - observed["matched_single_macro_type_recall"],
        }
        for metric, values in draws.items():
            values = values[np.isfinite(values)]
            p_two_sided = min(
                1.0,
                2.0
                * min(
                    (1 + (values <= 0).sum()) / (len(values) + 1),
                    (1 + (values >= 0).sum()) / (len(values) + 1),
                ),
            )
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "metric": metric,
                    "delta": observed_delta[metric],
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "bootstrap_p_two_sided": p_two_sided,
                    "n_groups": n_group,
                    "n_bootstrap": n_boot,
                    "n_bootstrap_effective": len(values),
                }
            )
    return pd.DataFrame(rows)


def stacked_coordinates(
    episodes: np.ndarray, queries: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(episodes, len(queries)), np.tile(np.asarray(queries), len(episodes))


def temporal_predict_bundle(
    data: Dataset,
    train_episodes: np.ndarray,
    test_episodes: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    train_ep, train_q = stacked_coordinates(train_episodes, SCAN_QUERIES)
    test_ep, test_q = stacked_coordinates(test_episodes, SCAN_QUERIES)
    pred = predict_bundle(data, train_ep, test_ep, train_q, test_q)
    return pred, test_ep, test_q


def temporal_scores_and_thresholds(
    data: Dataset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_names = (
        "single",
        "double_shared_max",
        "double_shared_noisy_or",
        "double_typed_max",
        "double_typed_noisy_or",
    )
    records = []
    threshold_rows = []
    unique_groups = np.unique(data.group)
    for held_out in unique_groups:
        outer_train = np.flatnonzero(data.group != held_out)
        outer_test = np.flatnonzero(data.group == held_out)

        inner_store = {
            name: np.full((len(data.episode), len(SCAN_QUERIES)), np.nan)
            for name in ("single", "shared_loop", "shared_static", "typed_loop", "typed_static")
        }
        for inner_held in unique_groups:
            if inner_held == held_out:
                continue
            inner_train = np.flatnonzero(
                (data.group != held_out) & (data.group != inner_held)
            )
            inner_test = np.flatnonzero(data.group == inner_held)
            pred, test_ep, test_q = temporal_predict_bundle(data, inner_train, inner_test)
            q_col = test_q - SCAN_QUERIES[0]
            for name in inner_store:
                inner_store[name][test_ep, q_col] = pred[name]

        inner_method = {
            "single": inner_store["single"],
            "double_shared_max": np.maximum(
                inner_store["shared_loop"], inner_store["shared_static"]
            ),
            "double_shared_noisy_or": 1.0
            - (1.0 - inner_store["shared_loop"])
            * (1.0 - inner_store["shared_static"]),
            "double_typed_max": np.maximum(
                inner_store["typed_loop"], inner_store["typed_static"]
            ),
            "double_typed_noisy_or": 1.0
            - (1.0 - inner_store["typed_loop"])
            * (1.0 - inner_store["typed_static"]),
        }
        thresholds = {}
        controls = outer_train[~data.trap[outer_train]]
        for method in score_names:
            maxima = np.nanmax(inner_method[method][controls], axis=1)
            if not np.isfinite(maxima).all():
                raise RuntimeError("inner cross-fitted calibration scores are incomplete")
            threshold = float(
                np.quantile(maxima, 1.0 - FALSE_ALARM_TARGET, method="higher")
            )
            thresholds[method] = threshold
            threshold_rows.append(
                {
                    "held_out_init_state": int(held_out),
                    "method": method,
                    "threshold": threshold,
                    "calibration_control_n": len(maxima),
                    "nominal_false_alarm": FALSE_ALARM_TARGET,
                    "inner_control_fa": float((maxima >= threshold).mean()),
                }
            )

        pred, test_ep, test_q = temporal_predict_bundle(data, outer_train, outer_test)
        method_score = {
            "single": pred["single"],
            "double_shared_max": np.maximum(pred["shared_loop"], pred["shared_static"]),
            "double_shared_noisy_or": 1.0
            - (1.0 - pred["shared_loop"]) * (1.0 - pred["shared_static"]),
            "double_typed_max": np.maximum(pred["typed_loop"], pred["typed_static"]),
            "double_typed_noisy_or": 1.0
            - (1.0 - pred["typed_loop"]) * (1.0 - pred["typed_static"]),
        }
        for row_idx, (episode, query) in enumerate(zip(test_ep, test_q)):
            for method in score_names:
                records.append(
                    {
                        "episode_id": int(episode),
                        "init_state_id": int(held_out),
                        "query": int(query),
                        "method": method,
                        "score": float(method_score[method][row_idx]),
                        "threshold": thresholds[method],
                        "fired": bool(method_score[method][row_idx] >= thresholds[method]),
                    }
                )
        print(f"  sequential outer init {held_out} complete", flush=True)
    return pd.DataFrame(records), pd.DataFrame(threshold_rows)


def first_fire_table(data: Dataset, temporal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (episode, method), part in temporal.groupby(["episode_id", "method"]):
        fired = part[part.fired].sort_values("query")
        q = int(fired["query"].iloc[0]) if len(fired) else -1
        i = int(episode)
        rows.append(
            {
                "episode_id": i,
                "init_state_id": int(data.group[i]),
                "method": method,
                "fire_query": q,
                "selected": q >= 0,
                "loop": bool(data.loop[i]),
                "static": bool(data.static[i]),
                "trap": bool(data.trap[i]),
                "trap_onset": int(data.trap_onset[i]),
                "lead_to_trap_onset": int(data.trap_onset[i] - q)
                if q >= 0 and data.trap_onset[i] >= 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def sequential_group_detail(first_fire: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, group), part in first_fire.groupby(["method", "init_state_id"]):
        selected = part.selected.to_numpy(bool)
        trap = part.trap.to_numpy(bool)
        loop = part.loop.to_numpy(bool)
        static = part.static.to_numpy(bool)
        by_stop = trap & (part.trap_onset.to_numpy() <= SCAN_QUERIES[-1])
        rows.append(
            {
                "method": method,
                "init_state_id": int(group),
                "n": len(part),
                "control_n": int((~trap).sum()),
                "false_alarm": int((selected & ~trap).sum()),
                "selected_n": int(selected.sum()),
                "positive_trap": int(trap.sum()),
                "positive_loop": int(loop.sum()),
                "positive_static": int(static.sum()),
                "positive_onset_by_stop": int(by_stop.sum()),
                "selected_trap": int((selected & trap).sum()),
                "selected_loop": int((selected & loop).sum()),
                "selected_static": int((selected & static).sum()),
                "selected_onset_by_stop": int((selected & by_stop).sum()),
            }
        )
    return pd.DataFrame(rows)


def sequential_metrics(detail: pd.DataFrame, first_fire: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, part in detail.groupby("method"):
        fired = first_fire[first_fire.method == method]
        detected_event = fired[fired.selected & fired.trap]
        n_loop = part.positive_loop.sum()
        n_static = part.positive_static.sum()
        recall_loop = part.selected_loop.sum() / n_loop
        recall_static = part.selected_static.sum() / n_static
        rows.append(
            {
                "method": method,
                "false_alarm_rate": part.false_alarm.sum() / part.control_n.sum(),
                "precision_trap": part.selected_trap.sum() / part.selected_n.sum(),
                "recall_trap": part.selected_trap.sum() / part.positive_trap.sum(),
                "recall_loop": recall_loop,
                "recall_static": recall_static,
                "macro_type_recall": np.mean([recall_loop, recall_static]),
                "recall_onset_by_q34": part.selected_onset_by_stop.sum()
                / part.positive_onset_by_stop.sum(),
                "selected_n": int(part.selected_n.sum()),
                "false_alarm_n": int(part.false_alarm.sum()),
                "detected_trap_n": int(part.selected_trap.sum()),
                "median_lead_detected": float(detected_event.lead_to_trap_onset.median())
                if len(detected_event)
                else np.nan,
                "fraction_detected_on_or_before_onset": float(
                    (detected_event.lead_to_trap_onset >= 0).mean()
                )
                if len(detected_event)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


SEQ_BOOT_METRICS = (
    "false_alarm_rate",
    "precision_trap",
    "recall_trap",
    "recall_loop",
    "recall_static",
    "macro_type_recall",
    "recall_onset_by_q34",
)


def bootstrap_sequential_deltas(detail: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Paired initial-state bootstrap for sequential selector differences."""

    rng = np.random.default_rng(SEED + 113)
    baseline = detail[detail.method == "single"].sort_values("init_state_id")
    n_group = len(baseline)
    take = rng.integers(0, n_group, size=(n_boot, n_group))

    def draw_metrics(rows: pd.DataFrame) -> dict[str, np.ndarray]:
        def total(column: str) -> np.ndarray:
            values = rows[column].to_numpy(np.float64)
            return values[take].sum(axis=1)

        control = total("control_n")
        false_alarm = total("false_alarm")
        selected = total("selected_n")
        positive_trap = total("positive_trap")
        positive_loop = total("positive_loop")
        positive_static = total("positive_static")
        positive_onset = total("positive_onset_by_stop")
        selected_trap = total("selected_trap")
        selected_loop = total("selected_loop")
        selected_static = total("selected_static")
        selected_onset = total("selected_onset_by_stop")
        with np.errstate(divide="ignore", invalid="ignore"):
            output = {
                "false_alarm_rate": false_alarm / control,
                "precision_trap": selected_trap / selected,
                "recall_trap": selected_trap / positive_trap,
                "recall_loop": selected_loop / positive_loop,
                "recall_static": selected_static / positive_static,
                "recall_onset_by_q34": selected_onset / positive_onset,
            }
        output["precision_trap"][selected == 0] = np.nan
        output["recall_trap"][positive_trap == 0] = np.nan
        output["recall_loop"][positive_loop == 0] = np.nan
        output["recall_static"][positive_static == 0] = np.nan
        output["recall_onset_by_q34"][positive_onset == 0] = np.nan
        count = np.isfinite(output["recall_loop"]).astype(int) + np.isfinite(
            output["recall_static"]
        ).astype(int)
        macro = np.nansum(
            np.column_stack([output["recall_loop"], output["recall_static"]]),
            axis=1,
        ) / np.maximum(count, 1)
        macro[count == 0] = np.nan
        output["macro_type_recall"] = macro
        return output

    baseline_draws = draw_metrics(baseline)
    rows = []
    for method in sorted(set(detail.method) - {"single"}):
        candidate = detail[detail.method == method].sort_values("init_state_id")
        if not np.array_equal(
            baseline.init_state_id.to_numpy(), candidate.init_state_id.to_numpy()
        ):
            raise RuntimeError("sequential bootstrap rows are not paired")
        candidate_draws = draw_metrics(candidate)

        def aggregate(part: pd.DataFrame) -> dict[str, float]:
            n_loop = part.positive_loop.sum()
            n_static = part.positive_static.sum()
            loop_recall = part.selected_loop.sum() / n_loop
            static_recall = part.selected_static.sum() / n_static
            return {
                "false_alarm_rate": part.false_alarm.sum() / part.control_n.sum(),
                "precision_trap": part.selected_trap.sum() / part.selected_n.sum(),
                "recall_trap": part.selected_trap.sum() / part.positive_trap.sum(),
                "recall_loop": loop_recall,
                "recall_static": static_recall,
                "macro_type_recall": np.mean([loop_recall, static_recall]),
                "recall_onset_by_q34": part.selected_onset_by_stop.sum()
                / part.positive_onset_by_stop.sum(),
            }

        observed_base = aggregate(baseline)
        observed_candidate = aggregate(candidate)
        for metric in SEQ_BOOT_METRICS:
            values = candidate_draws[metric] - baseline_draws[metric]
            values = values[np.isfinite(values)]
            p_two_sided = min(
                1.0,
                2.0
                * min(
                    (1 + (values <= 0).sum()) / (len(values) + 1),
                    (1 + (values >= 0).sum()) / (len(values) + 1),
                ),
            )
            rows.append(
                {
                    "method": method,
                    "baseline": "single",
                    "metric": metric,
                    "delta": observed_candidate[metric] - observed_base[metric],
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "bootstrap_p_two_sided": p_two_sided,
                    "n_groups": n_group,
                    "n_bootstrap": n_boot,
                    "n_bootstrap_effective": len(values),
                }
            )
    return pd.DataFrame(rows)


def feature_audit_table() -> pd.DataFrame:
    descriptions = {
        "route_mobility_now": "current inter-query action-token Hellinger mobility",
        "route_mobility_w4": "four-query trailing mobility mean",
        "gate_concentration_now": "1 - normalized gate entropy at deep layers/d9",
        "gate_concentration_w4": "four-query trailing gate concentration mean",
        "gate_concentration_delta2": "current minus two-query-earlier concentration",
        "gate_concentration_range4": "four-query concentration peak-to-valley range",
        "late_flow_volatility_w4": "four-query mean of within-flow d6:d9 volatility",
        "route_acceleration_w4": "four-query mean of within-flow route acceleration",
        "top12_margin_w4": "four-query mean soft top1-top2 probability margin",
        "token_disagreement_w4": "four-query mean soft cross-token disagreement",
        "deep_support_concentration_now": "hard Top-4 support entropy concentration, L12-L15",
        "deep_max_occupancy_now": "largest hard-support token occupancy, L12-L15",
        "deep_support_concentration_delta2": "two-query change in deep hard-support concentration",
        "l15_support_concentration_now": "hard Top-4 support entropy concentration, L15/d9",
        "l15_support_concentration_w4": "four-query mean L15/d9 support concentration",
        "l15_max_occupancy_now": "largest hard-support token occupancy, L15/d9",
        "l15_support_size_now": "number of unique selected experts, L15/d9",
        "l15_final_denoise_jump_now": "L15 support concentration d9 minus d8",
    }
    return pd.DataFrame(
        [
            {
                "feature": name,
                "loop_head": name in LOOP_FEATURES,
                "static_head": name in STATIC_FEATURES,
                "single_head": True,
                "uses_hard_top4_ids": name in HARD_ID_FEATURES,
                "causal": True,
                "description": descriptions[name],
            }
            for name in ALL_FEATURES
        ]
    )


def write_manifest(data: Dataset, inventory: pd.DataFrame, args: argparse.Namespace) -> None:
    payload = {
        "schema": SCHEMA,
        "corpus": str(B_ROOT.relative_to(ROOT)),
        "n_trajectories": len(data.episode),
        "n_initial_states": int(len(np.unique(data.group))),
        "n_loop": int(data.loop.sum()),
        "n_static": int(data.static.sum()),
        "n_trap_union": int(data.trap.sum()),
        "n_loop_static_overlap": int((data.loop & data.static).sum()),
        "rank_times": list(RANK_TIMES),
        "budgets": list(BUDGETS),
        "primary_time": PRIMARY_TIME,
        "primary_budget": PRIMARY_BUDGET,
        "scan_queries": [SCAN_QUERIES[0], SCAN_QUERIES[-1]],
        "false_alarm_target": FALSE_ALARM_TARGET,
        "fixed_logistic_C": FIXED_C,
        "bootstrap": args.bootstrap,
        "feature_count_union": len(ALL_FEATURES),
        "feature_count_loop_head": len(LOOP_FEATURES),
        "feature_count_static_head": len(STATIC_FEATURES),
        "feature_count_soft_union": len(ALL_SOFT_FEATURES),
        "post_protocol_diagnostics": ["noisy_or_fusion", "soft_only_ablation"],
        "clarified_primary": {
            "query": PRIMARY_TIME,
            "per_head_budget": PRIMARY_BUDGET,
            "comparison": "each typed head vs single-K on its own subtype; union vs single at matched union size",
        },
        "inventory": inventory.to_dict(orient="records"),
    }
    (RESULTS / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("loading corpus and causal routing features", flush=True)
    data, inventory = load_dataset(args.rebuild_support_cache)
    feature_audit_table().to_csv(RESULTS / "feature_audit.csv", index=False)
    write_manifest(data, inventory, args)

    print("fixed-time leave-one-init-state-out ranking", flush=True)
    score_frames = [fixed_time_oof(data, query) for query in RANK_TIMES]
    scores = pd.concat(score_frames, ignore_index=True)
    scores.to_csv(RESULTS / "fixed_time_oof_scores.csv", index=False)
    score_auc_table(scores).to_csv(RESULTS / "fixed_time_auc.csv", index=False)

    independent_detail = pd.concat(
        [independent_head_rows(data, frame) for frame in score_frames],
        ignore_index=True,
    )
    independent_detail.to_csv(
        RESULTS / "independent_heads_by_init_state.csv", index=False
    )
    independent_summary = summarise_independent_heads(independent_detail)
    independent_summary.to_csv(RESULTS / "independent_heads_metrics.csv", index=False)
    bootstrap_independent_deltas(independent_detail, args.bootstrap).to_csv(
        RESULTS / "independent_heads_bootstrap_deltas.csv", index=False
    )

    detail = pd.concat(
        [group_selection_rows(data, frame) for frame in score_frames], ignore_index=True
    )
    detail.to_csv(RESULTS / "selection_by_init_state.csv", index=False)
    metrics = summarise_selection(detail)
    metrics.to_csv(RESULTS / "selection_metrics.csv", index=False)
    deltas = bootstrap_selection_deltas(detail, args.bootstrap)
    deltas.to_csv(RESULTS / "selection_bootstrap_deltas.csv", index=False)

    if not args.skip_sequential:
        print("nested-LOIO sequential calibration", flush=True)
        temporal, thresholds = temporal_scores_and_thresholds(data)
        temporal.to_csv(RESULTS / "sequential_oof_scores.csv", index=False)
        thresholds.to_csv(RESULTS / "sequential_thresholds.csv", index=False)
        first_fire = first_fire_table(data, temporal)
        first_fire.to_csv(RESULTS / "sequential_first_fire.csv", index=False)
        seq_detail = sequential_group_detail(first_fire)
        seq_detail.to_csv(RESULTS / "sequential_by_init_state.csv", index=False)
        sequential_metrics(seq_detail, first_fire).to_csv(
            RESULTS / "sequential_metrics.csv", index=False
        )
        bootstrap_sequential_deltas(seq_detail, args.bootstrap).to_csv(
            RESULTS / "sequential_bootstrap_deltas.csv", index=False
        )

    primary = metrics[
        (metrics["query"] == PRIMARY_TIME)
        & (metrics.budget == PRIMARY_BUDGET)
        & (metrics.label_scope == "eventual")
    ]
    print("\nprimary q34 / K8", flush=True)
    print(
        primary[
            [
                "method",
                "precision_trap",
                "recall_trap",
                "recall_loop",
                "recall_static",
                "macro_type_recall",
            ]
        ].to_string(index=False),
        flush=True,
    )
    clarified = independent_summary[
        (independent_summary["query"] == PRIMARY_TIME)
        & (independent_summary.per_head_budget == PRIMARY_BUDGET)
        & (independent_summary.family == "typed")
    ]
    print("\nclarified independent heads q34 / K8 each", flush=True)
    print(
        clarified[
            [
                "loop_head_recall",
                "single_k_loop_recall",
                "static_head_recall",
                "single_k_static_recall",
                "union_budget_mean",
                "union_trap_recall",
                "matched_single_trap_recall",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"\nwrote results to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
