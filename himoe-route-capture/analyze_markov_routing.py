#!/usr/bin/env python3
"""Fit and audit a finite-state Markov model of state-token routing.

The observable at one time step is the state-token router distribution from the
eight HB layers at denoise round zero.  Each per-layer distribution is mapped
through the square-root (Hellinger) embedding and quantized into K cells.  The
cell sequence is then modeled as a discrete-time Markov chain.

This analysis is deliberately comparative:

* folds hold out the same flow-noise seeds in every scene, and whole episodes
  are kept intact;
* an equally resolved proprioception chain is fit alongside the routing chain;
* order-0, order-1, order-2, and phase-conditioned order-1 predictors are
  evaluated out of sample;
* every episode is truncated to the shortest observed length, so duration does
  not leak the outcome.

The output is a report, a JSON summary, and fitted transition matrices.  A
first-order gain establishes temporal dependence, not a routing-specific
mechanism; the matched proprioception model is the relevant control for that
claim.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.cluster import MiniBatchKMeans


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "himoe-routing-rules-20260819" / "data"
DEFAULT_TASK = "libero_long__KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.npz"
DEFAULT_OUT = HERE / "analysis" / "markov-routing-t08"
REPRESENTATIONS = ("routing", "proprio")


@dataclass(frozen=True)
class EpisodeData:
    suite: str
    task: str
    source: Path
    routing: np.ndarray
    proprio: np.ndarray
    success: np.ndarray
    scene: np.ndarray
    seed: np.ndarray
    window: int

    @property
    def episodes(self) -> int:
        return int(len(self.success))


@dataclass(frozen=True)
class Quantizer:
    model: MiniBatchKMeans
    mean: np.ndarray
    scale: np.ndarray
    representation: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        flat = values.reshape(-1, values.shape[-1])
        normalized = (flat - self.mean) / self.scale
        return self.model.predict(normalized).reshape(values.shape[:2])


@dataclass(frozen=True)
class MarkovModels:
    marginal: np.ndarray
    first: np.ndarray
    second: np.ndarray
    phase: np.ndarray
    transition_counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--task",
        action="append",
        help="bundle filename; repeat for more tasks (default: LIBERO Long t08)",
    )
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clusters", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--primary-k", type=int, default=16)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="hierarchical prior strength in pseudo-transitions",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def _flow_seed_index(scene: np.ndarray) -> np.ndarray:
    """Return the within-scene draw index used by the 16x32 capture."""
    scene = np.asarray(scene)
    out = np.empty(len(scene), dtype=np.int64)
    counts: dict[int, int] = {}
    for index, value in enumerate(scene.tolist()):
        key = int(value)
        out[index] = counts.get(key, 0)
        counts[key] = int(out[index]) + 1
    sizes = set(counts.values())
    if len(sizes) != 1:
        raise ValueError("every scene must contain the same number of noise draws")
    return out


def _take_common_window(values: np.ndarray, lengths: np.ndarray, window: int) -> np.ndarray:
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    take = (offsets[:, None] + np.arange(window)[None, :]).ravel()
    return values[take].reshape(len(lengths), window, *values.shape[1:])


def normalize_router_probabilities(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim < 2 or probabilities.shape[-1] != 32:
        raise ValueError("router probabilities must end in a 32-expert axis")
    if not np.all(np.isfinite(probabilities)) or np.min(probabilities) < -1e-6:
        raise ValueError("router probabilities must be finite and nonnegative")
    probabilities = np.maximum(probabilities, 0.0)
    total = probabilities.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("router probability vector has zero mass")
    return probabilities / total


def load_bundle(path: Path) -> EpisodeData:
    with np.load(path, allow_pickle=True) as archive:
        required = {
            "n_rows",
            "success",
            "scene",
            "proprio",
            "state_token_probs",
            "suite",
            "task",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError("%s is missing keys: %s" % (path, sorted(missing)))
        lengths = np.asarray(archive["n_rows"], dtype=np.int64)
        if len(lengths) == 0 or np.any(lengths < 3):
            raise ValueError("each episode needs at least three control steps")
        window = int(lengths.min())
        raw_routing = _take_common_window(
            np.asarray(archive["state_token_probs"]), lengths, window
        )
        if raw_routing.shape[2:] != (8, 32):
            raise ValueError("expected state-token probabilities shaped [episode,time,8,32]")
        probabilities = normalize_router_probabilities(raw_routing)
        # Concatenated square-root probabilities make squared Euclidean distance
        # proportional to mean per-layer squared Hellinger distance.
        routing = np.sqrt(probabilities).reshape(len(lengths), window, 8 * 32)
        proprio = _take_common_window(
            np.asarray(archive["proprio"], dtype=np.float64), lengths, window
        )
        success = np.asarray(archive["success"], dtype=bool)
        scene = np.asarray(archive["scene"], dtype=np.int64)
        suite = str(archive["suite"])
        task = str(archive["task"])
    if not (len(success) == len(scene) == len(lengths)):
        raise ValueError("episode metadata lengths disagree")
    return EpisodeData(
        suite=suite,
        task=task,
        source=path.resolve(),
        routing=routing,
        proprio=proprio,
        success=success,
        scene=scene,
        seed=_flow_seed_index(scene),
        window=window,
    )


def seed_disjoint_folds(seed_ids: np.ndarray, folds: int, random_seed: int) -> list[np.ndarray]:
    unique = np.unique(seed_ids)
    if folds < 2 or folds > len(unique):
        raise ValueError("fold count must be between 2 and the number of seeds")
    shuffled = np.random.default_rng(random_seed).permutation(unique)
    return [np.isin(seed_ids, part) for part in np.array_split(shuffled, folds)]


def fit_quantizer(
    values: np.ndarray,
    train_episodes: np.ndarray,
    clusters: int,
    representation: str,
    random_seed: int,
) -> Quantizer:
    if representation not in REPRESENTATIONS:
        raise ValueError("unknown representation %r" % representation)
    train = values[train_episodes].reshape(-1, values.shape[-1]).astype(np.float64)
    if representation == "proprio":
        mean = train.mean(axis=0)
        scale = train.std(axis=0)
        scale[scale < 1e-9] = 1.0
    else:
        mean = np.zeros(values.shape[-1], dtype=np.float64)
        scale = np.ones(values.shape[-1], dtype=np.float64)
    normalized = (train - mean) / scale
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=random_seed,
        n_init=10,
        batch_size=min(2048, len(normalized)),
        max_iter=200,
        reassignment_ratio=0.01,
    ).fit(normalized)
    return Quantizer(model=model, mean=mean, scale=scale, representation=representation)


def transition_phases(length: int, phases: int = 3) -> np.ndarray:
    if length < 2:
        return np.empty(0, dtype=np.int64)
    midpoint = (np.arange(length - 1, dtype=np.float64) + 0.5) / (length - 1)
    return np.minimum((midpoint * phases).astype(np.int64), phases - 1)


def _posterior_rows(counts: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    numerator = counts.astype(np.float64) + alpha * prior
    denominator = numerator.sum(axis=-1, keepdims=True)
    return numerator / denominator


def fit_markov_models(
    sequences: np.ndarray,
    clusters: int,
    alpha: float = 5.0,
    marginal_pseudocount: float = 0.5,
) -> MarkovModels:
    sequences = np.asarray(sequences, dtype=np.int64)
    if sequences.ndim != 2 or sequences.shape[1] < 3:
        raise ValueError("expected fixed-length sequences shaped [episode,time>=3]")
    if np.min(sequences) < 0 or np.max(sequences) >= clusters:
        raise ValueError("state id outside the declared state space")

    targets = sequences[:, 1:].ravel()
    marginal_counts = np.bincount(targets, minlength=clusters).astype(np.float64)
    marginal = (marginal_counts + marginal_pseudocount) / (
        marginal_counts.sum() + marginal_pseudocount * clusters
    )

    first_counts = np.zeros((clusters, clusters), dtype=np.int64)
    np.add.at(first_counts, (sequences[:, :-1].ravel(), targets), 1)
    first = _posterior_rows(first_counts, marginal[None, :], alpha)

    second_counts = np.zeros((clusters, clusters, clusters), dtype=np.int64)
    np.add.at(
        second_counts,
        (
            sequences[:, :-2].ravel(),
            sequences[:, 1:-1].ravel(),
            sequences[:, 2:].ravel(),
        ),
        1,
    )
    second_prior = np.broadcast_to(first[None, :, :], second_counts.shape)
    second = _posterior_rows(second_counts, second_prior, alpha)

    phase_counts = np.zeros((3, clusters, clusters), dtype=np.int64)
    phases = transition_phases(sequences.shape[1])
    for phase in range(3):
        take = phases == phase
        left = sequences[:, :-1][:, take].ravel()
        right = sequences[:, 1:][:, take].ravel()
        np.add.at(phase_counts[phase], (left, right), 1)
    phase = _posterior_rows(phase_counts, first[None, :, :], alpha)
    return MarkovModels(
        marginal=marginal,
        first=first,
        second=second,
        phase=phase,
        transition_counts=first_counts,
    )


def sequence_log_probabilities(models: MarkovModels, sequences: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-episode mean log2 predictive probabilities."""
    sequences = np.asarray(sequences, dtype=np.int64)
    left, right = sequences[:, :-1], sequences[:, 1:]
    zero = np.log2(models.marginal[right])
    first = np.log2(models.first[left, right])

    second = first.copy()
    second[:, 1:] = np.log2(
        models.second[sequences[:, :-2], sequences[:, 1:-1], sequences[:, 2:]]
    )

    phases = transition_phases(sequences.shape[1])
    phase = np.empty_like(first)
    for index, value in enumerate(phases):
        phase[:, index] = np.log2(models.phase[value, left[:, index], right[:, index]])
    return {
        "zero": zero.mean(axis=1),
        "first": first.mean(axis=1),
        "second": second.mean(axis=1),
        "phase": phase.mean(axis=1),
    }


def stationary_distribution(matrix: np.ndarray, tolerance: float = 1e-13) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition matrix must be square")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("transition rows must sum to one")
    state = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    for _ in range(100_000):
        updated = state @ matrix
        if np.max(np.abs(updated - state)) < tolerance:
            return updated / updated.sum()
        state = updated
    raise RuntimeError("stationary distribution did not converge")


def mean_run_length(sequences: np.ndarray) -> float:
    runs: list[int] = []
    for sequence in np.asarray(sequences):
        boundaries = np.flatnonzero(np.diff(sequence)) + 1
        runs.extend(np.diff(np.concatenate([[0], boundaries, [len(sequence)]])).tolist())
    return float(np.mean(runs))


def stratified_auc(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """Pair-weighted AUC using only positive/negative pairs inside a group."""
    return stratified_auc_details(scores, labels, groups)["auc"]


def stratified_auc_details(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    groups = np.asarray(groups)
    wins = pairs = 0.0
    valid_groups = 0
    for group in np.unique(groups):
        take = groups == group
        positive = scores[take & labels]
        negative = scores[take & ~labels]
        if len(positive) == 0 or len(negative) == 0:
            continue
        valid_groups += 1
        delta = positive[:, None] - negative[None, :]
        wins += float(np.sum(delta > 0) + 0.5 * np.sum(delta == 0))
        pairs += float(delta.size)
    return {
        "auc": wins / pairs if pairs else float("nan"),
        "pair_count": int(pairs),
        "valid_groups": valid_groups,
    }


def _scene_success_prior(
    train_mask: np.ndarray, success: np.ndarray, scene: np.ndarray
) -> dict[int, float]:
    prior: dict[int, float] = {}
    for value in np.unique(scene):
        labels = success[train_mask & (scene == value)]
        prior[int(value)] = float((labels.sum() + 1.0) / (len(labels) + 2.0))
    return prior


def outcome_scores(
    train_sequences: np.ndarray,
    train_success: np.ndarray,
    test_sequences: np.ndarray,
    test_scene: np.ndarray,
    scene_prior: dict[int, float],
    clusters: int,
    alpha: float,
) -> dict[str, np.ndarray] | None:
    if np.unique(train_success).size < 2:
        return None
    by_class = {
        label: fit_markov_models(train_sequences[train_success == label], clusters, alpha)
        for label in (False, True)
    }
    logp = {label: sequence_log_probabilities(by_class[label], test_sequences) for label in by_class}
    transitions = test_sequences.shape[1] - 1
    log_prior = np.array(
        [math.log2(scene_prior[int(value)] / (1.0 - scene_prior[int(value)])) for value in test_scene]
    )
    return {
        name: log_prior + transitions * (logp[True][name] - logp[False][name])
        for name in ("zero", "first")
    }


def crossed_bootstrap_interval(
    values: np.ndarray,
    scene: np.ndarray,
    seed: np.ndarray,
    draws: int,
    random_seed: int,
) -> list[float] | None:
    if draws <= 0:
        return None
    scenes, scene_index = np.unique(scene, return_inverse=True)
    seeds, seed_index = np.unique(seed, return_inverse=True)
    grid = np.full((len(scenes), len(seeds)), np.nan)
    grid[scene_index, seed_index] = values
    if np.isnan(grid).any():
        return None
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        rows = rng.integers(0, len(scenes), len(scenes))
        columns = rng.integers(0, len(seeds), len(seeds))
        estimates[draw] = grid[np.ix_(rows, columns)].mean()
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def _values_for(data: EpisodeData, representation: str) -> np.ndarray:
    return data.routing if representation == "routing" else data.proprio


def cross_validate(
    data: EpisodeData,
    representation: str,
    clusters: int,
    folds: int,
    alpha: float,
    bootstrap: int,
    random_seed: int,
) -> dict:
    masks = seed_disjoint_folds(data.seed, folds, random_seed)
    losses = {name: np.full(data.episodes, np.nan) for name in ("zero", "first", "second", "phase")}
    scores = {name: np.full(data.episodes, np.nan) for name in ("zero", "first")}
    fold_of = np.full(data.episodes, -1, dtype=np.int64)
    self_transition = np.full(data.episodes, np.nan)
    values = _values_for(data, representation)

    for fold, test_mask in enumerate(masks):
        train_mask = ~test_mask
        quantizer = fit_quantizer(
            values,
            train_mask,
            clusters,
            representation,
            random_seed + fold,
        )
        train_sequences = quantizer.transform(values[train_mask])
        test_sequences = quantizer.transform(values[test_mask])
        models = fit_markov_models(train_sequences, clusters, alpha)
        logp = sequence_log_probabilities(models, test_sequences)
        for name in losses:
            losses[name][test_mask] = -logp[name]
        self_transition[test_mask] = np.mean(
            test_sequences[:, 1:] == test_sequences[:, :-1], axis=1
        )
        fold_of[test_mask] = fold

        prior = _scene_success_prior(train_mask, data.success, data.scene)
        class_scores = outcome_scores(
            train_sequences,
            data.success[train_mask],
            test_sequences,
            data.scene[test_mask],
            prior,
            clusters,
            alpha,
        )
        if class_scores is not None:
            for name in scores:
                scores[name][test_mask] = class_scores[name]

    if np.any(fold_of < 0) or any(np.isnan(value).any() for value in losses.values()):
        raise RuntimeError("cross-validation did not score every episode exactly once")
    first_gain = losses["zero"] - losses["first"]
    second_gain = losses["first"] - losses["second"]
    phase_gain = losses["first"] - losses["phase"]
    result = {
        "zero_bits": float(losses["zero"].mean()),
        "first_bits": float(losses["first"].mean()),
        "second_bits": float(losses["second"].mean()),
        "phase_bits": float(losses["phase"].mean()),
        "first_over_zero_gain_bits": float(first_gain.mean()),
        "second_over_first_gain_bits": float(second_gain.mean()),
        "phase_over_first_gain_bits": float(phase_gain.mean()),
        "self_transition_rate": float(self_transition.mean()),
        "confidence_intervals": {
            "first_over_zero_gain_bits": crossed_bootstrap_interval(
                first_gain, data.scene, data.seed, bootstrap, random_seed + 101
            ),
            "second_over_first_gain_bits": crossed_bootstrap_interval(
                second_gain, data.scene, data.seed, bootstrap, random_seed + 102
            ),
            "phase_over_first_gain_bits": crossed_bootstrap_interval(
                phase_gain, data.scene, data.seed, bootstrap, random_seed + 103
            ),
        },
    }

    if not any(np.isnan(value).any() for value in scores.values()):
        # AUC comparisons are restricted to scene x fold cells.  This prevents
        # fold-specific likelihood scales and scene base rates from contributing.
        groups = np.array(["%s/%d" % (scene, fold) for scene, fold in zip(data.scene, fold_of)])
        zero_auc = stratified_auc_details(scores["zero"], data.success, groups)
        first_auc = stratified_auc_details(scores["first"], data.success, groups)
        if zero_auc["pair_count"] != first_auc["pair_count"]:
            raise RuntimeError("outcome models were evaluated on different pairs")
        result["outcome"] = {
            "comparison_groups": "scene x held-out-seed-fold",
            "pair_count": zero_auc["pair_count"],
            "valid_groups": zero_auc["valid_groups"],
            "zero_order_auc": float(zero_auc["auc"]),
            "first_order_auc": float(first_auc["auc"]),
            "first_minus_zero_auc": float(first_auc["auc"] - zero_auc["auc"]),
        }
    else:
        result["outcome"] = None
    return result


def final_model(
    data: EpisodeData,
    representation: str,
    clusters: int,
    alpha: float,
    random_seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    values = _values_for(data, representation)
    all_episodes = np.ones(data.episodes, dtype=bool)
    quantizer = fit_quantizer(values, all_episodes, clusters, representation, random_seed)
    sequences = quantizer.transform(values)
    models = fit_markov_models(sequences, clusters, alpha)
    stationary = stationary_distribution(models.first)
    occupancy = np.bincount(sequences[:, 1:].ravel(), minlength=clusters).astype(np.float64)
    occupancy /= occupancy.sum()
    entropy_rate = float(
        -np.sum(stationary[:, None] * models.first * np.log2(models.first))
    )
    eigenvalues = np.linalg.eigvals(models.first)
    magnitudes = np.sort(np.abs(eigenvalues))[::-1]
    second_eigenvalue = float(magnitudes[1]) if len(magnitudes) > 1 else 0.0
    implied_timescale = (
        float(-1.0 / np.log(second_eigenvalue)) if 0.0 < second_eigenvalue < 1.0 else None
    )
    observed_edges = int(np.sum(models.transition_counts > 0))
    diagnostics = {
        "states": clusters,
        "observed_edges": observed_edges,
        "observed_edge_density": observed_edges / float(clusters * clusters),
        "empirical_mean_dwell_steps": mean_run_length(sequences),
        "empirical_self_transition_rate": float(
            np.mean(sequences[:, 1:] == sequences[:, :-1])
        ),
        "stationary_self_transition_rate": float(np.sum(stationary * np.diag(models.first))),
        "stationary_vs_occupancy_tv": float(0.5 * np.abs(stationary - occupancy).sum()),
        "entropy_rate_bits_per_step": entropy_rate,
        "second_eigenvalue_magnitude": second_eigenvalue,
        "implied_timescale_steps": implied_timescale,
        "state_occupancy": occupancy.tolist(),
        "stationary_distribution": stationary.tolist(),
    }
    artifact = {
        "transition": models.first,
        "second_order_transition": models.second,
        "phase_transition": models.phase,
        "transition_counts": models.transition_counts,
        "marginal": models.marginal,
        "stationary": stationary,
        "occupancy": occupancy,
        "cluster_centers": quantizer.model.cluster_centers_,
        "feature_mean": quantizer.mean,
        "feature_scale": quantizer.scale,
        "state_sequences": sequences,
    }
    return diagnostics, artifact


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return ("%%.%df" % digits) % value


def _task_slug(data: EpisodeData) -> str:
    return "%s__%s" % (data.suite, data.task)


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# State-token 路由的马尔科夫链模型",
        "",
        "## 协议",
        "",
        "- 时间单位：一个 policy query / control step。",
        "- 路由状态：8 个 HB 层在 denoise-0 的 32 路完整概率，经逐层归一化和平方根映射后做 K-means 量化。",
        "- 数据划分：整条 rollout 不拆分；同一个 flow-noise seed 在所有 scene 中同时留出。",
        "- 对照：同 K 的 8 维 proprioception 量化链。所有量化器只在训练折拟合。",
        "- 序列窗口：逐任务截到最短 episode，所有样本在窗口内仍在运行，避免长度/终止泄漏。",
        "- 平滑：一阶、二阶和阶段链均使用下阶分布作为层级先验。",
        "",
        "`first-over-zero` 大于零表示当前状态帮助预测下一状态；`second-over-first` 大于零表示一阶 Markov 假设遗漏历史；`phase-over-first` 大于零表示齐次转移矩阵遗漏了早/中/晚阶段。不同 K 或不同表示的绝对 log-loss 目标不同，不能直接横比，只比较各自相对零阶的增益。",
        "",
    ]
    for task_key, task in summary["tasks"].items():
        meta = task["metadata"]
        lines += [
            "## %s" % meta["task"],
            "",
            "`%d` rollouts，`%d` scenes x `%d` shared seeds，公共窗口 `%d` 步，成功率 `%.1f%%`。"
            % (
                meta["episodes"],
                meta["scenes"],
                meta["seeds"],
                meta["window"],
                100 * meta["success_rate"],
            ),
            "",
            "| K | 表示 | 零阶 loss | 一阶 loss | first-over-zero | second-over-first | phase-over-first | 自环率 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for key in sorted(task["cv"], key=lambda item: (int(item.split("/")[0]), item)):
            k_string, representation = key.split("/")
            row = task["cv"][key]
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    k_string,
                    representation,
                    _fmt(row["zero_bits"]),
                    _fmt(row["first_bits"]),
                    _fmt(row["first_over_zero_gain_bits"]),
                    _fmt(row["second_over_first_gain_bits"]),
                    _fmt(row["phase_over_first_gain_bits"]),
                    _fmt(row["self_transition_rate"]),
                )
            )
        lines += ["", "### 结局条件链（探索性）", ""]
        primary = str(summary["protocol"]["primary_k"])
        lines += [
            "| K | 表示 | 零阶状态占用 AUC | 一阶转移 AUC | 增量 | 有效成败对 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for clusters in summary["protocol"]["clusters"]:
            for representation in REPRESENTATIONS:
                outcome = task["cv"]["%s/%s" % (clusters, representation)]["outcome"]
                if outcome is None:
                    lines.append("| %s | %s | n/a | n/a | n/a | 0 |" % (clusters, representation))
                else:
                    lines.append(
                        "| %s | %s | %s | %s | %s | %d |"
                        % (
                            clusters,
                            representation,
                            _fmt(outcome["zero_order_auc"], 3),
                            _fmt(outcome["first_order_auc"], 3),
                            _fmt(outcome["first_minus_zero_auc"], 3),
                            outcome["pair_count"],
                        )
                    )
        lines += [
            "",
            "AUC 只比较同一 scene、同一 held-out seed fold 内的成败对；它不使用 episode 长度。这里用于判断转移顺序是否比状态占用多带来结局区分，不作在线预警性能声明。",
            "",
            "### 全数据最终模型诊断（K=%s）" % primary,
            "",
            "| 表示 | 平均 dwell | 观测边密度 | 平稳/占用 TV | 熵率(bit/步) | 隐含时标(步) |"
            ,
            "|---|---:|---:|---:|---:|---:|",
        ]
        for representation in REPRESENTATIONS:
            diagnostic = task["final"][representation]
            lines.append(
                "| %s | %s | %s | %s | %s | %s |"
                % (
                    representation,
                    _fmt(diagnostic["empirical_mean_dwell_steps"], 3),
                    _fmt(diagnostic["observed_edge_density"], 3),
                    _fmt(diagnostic["stationary_vs_occupancy_tv"], 3),
                    _fmt(diagnostic["entropy_rate_bits_per_step"], 3),
                    _fmt(diagnostic["implied_timescale_steps"], 2),
                )
            )
        route = task["cv"]["%s/routing" % primary]
        pose = task["cv"]["%s/proprio" % primary]
        lines += ["", "### 自动判读", ""]
        if route["first_over_zero_gain_bits"] > 0:
            lines.append(
                "- 路由序列存在可泛化的一步转移结构：相对零阶模型改善 `%s` bit/step。"
                % _fmt(route["first_over_zero_gain_bits"])
            )
        else:
            lines.append("- 路由一阶链未优于零阶占用模型。")
        if route["second_over_first_gain_bits"] > 0.01:
            lines.append(
                "- 二阶历史仍改善 `%s` bit/step，一阶 Markov 假设只能作为近似。"
                % _fmt(route["second_over_first_gain_bits"])
            )
        else:
            lines.append("- 二阶历史没有带来大于 0.01 bit/step 的改善，一阶近似在当前分辨率下足够。")
        if route["phase_over_first_gain_bits"] > 0.01:
            lines.append(
                "- 阶段条件链改善 `%s` bit/step，过程不宜解释为严格齐次链。"
                % _fmt(route["phase_over_first_gain_bits"])
            )
        else:
            lines.append("- 早/中/晚阶段条件没有带来大于 0.01 bit/step 的改善。")
        ratio = route["first_over_zero_gain_bits"] / max(
            pose["first_over_zero_gain_bits"], 1e-12
        )
        lines.append(
            "- 在同 K 对照中，路由与 proprio 的 first-over-zero 增益比为 `%s`。这只是各自归一化后的动力学强度比较，不是信息量的直接差。"
            % _fmt(ratio, 3)
        )
        lines.append("")
    if len(summary["tasks"]) > 1:
        lines += [
            "## 跨任务汇总",
            "",
            "下表对任务等权，只比较同一个 K 内各表示相对自身零阶模型的增益。`routing wins` 是路由增益高于 proprio 的任务数。",
            "",
            "| K | routing first-over-zero | proprio first-over-zero | routing - proprio | routing wins |",
            "|---:|---:|---:|---:|---:|",
        ]
        for clusters in summary["protocol"]["clusters"]:
            route = np.array(
                [
                    task["cv"]["%s/routing" % clusters]["first_over_zero_gain_bits"]
                    for task in summary["tasks"].values()
                ]
            )
            pose = np.array(
                [
                    task["cv"]["%s/proprio" % clusters]["first_over_zero_gain_bits"]
                    for task in summary["tasks"].values()
                ]
            )
            lines.append(
                "| %s | %s | %s | %s | %d/%d |"
                % (
                    clusters,
                    _fmt(float(route.mean())),
                    _fmt(float(pose.mean())),
                    _fmt(float((route - pose).mean())),
                    int(np.sum(route > pose)),
                    len(route),
                )
            )
        lines += [""]
    lines += [
        "## 边界",
        "",
        "有限状态来自分析者选择的量化分辨率，不是已发现的自然语义阶段；因此结论应在多个 K 上同向才引用。state-token 路由已知主要是 proprioception 的向量量化读出，所以即使马尔科夫链拟合良好，也不能把吸引子或转移结构归因给 router 本身，除非它稳定超过匹配的 proprio 对照。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_tasks(args: argparse.Namespace) -> list[Path]:
    if args.all_tasks:
        tasks = sorted(args.data_dir.glob("*.npz"))
    else:
        names = args.task or [DEFAULT_TASK]
        tasks = [Path(name) if Path(name).is_absolute() else args.data_dir / name for name in names]
    missing = [path for path in tasks if not path.is_file()]
    if missing:
        raise SystemExit("missing bundle files: %s" % ", ".join(map(str, missing)))
    if not tasks:
        raise SystemExit("no bundle files found")
    return tasks


def _jsonable_protocol(args: argparse.Namespace, tasks: Iterable[Path]) -> dict:
    return {
        "data_files": [str(path.resolve()) for path in tasks],
        "time_unit": "policy query / control step",
        "routing_feature": "sqrt(normalized state-token probability), 8 HB layers x 32 experts",
        "quantizer": "training-fold MiniBatchKMeans in Hellinger embedding",
        "control": "training-fold standardized 8D proprioception at matched K",
        "split": "whole episodes, flow-noise-seed-disjoint across all scenes",
        "window": "task-specific shortest episode",
        "clusters": args.clusters,
        "primary_k": args.primary_k,
        "folds": args.folds,
        "transition_prior_strength": args.alpha,
        "marginal_pseudocount_per_state": 0.5,
        "crossed_scene_seed_bootstrap_draws": args.bootstrap,
        "seed": args.seed,
    }


def main() -> int:
    args = parse_args()
    if args.primary_k not in args.clusters:
        raise SystemExit("--primary-k must also appear in --clusters")
    if any(value < 2 for value in args.clusters):
        raise SystemExit("every K must be at least 2")
    task_paths = resolve_tasks(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": "state_token_markov_state_model", "protocol": _jsonable_protocol(args, task_paths), "tasks": {}}

    for path in task_paths:
        data = load_bundle(path)
        slug = _task_slug(data)
        print(
            "\n=== %s: %d episodes, %d x %d, window=%d, success=%.1f%% ==="
            % (
                data.task,
                data.episodes,
                len(np.unique(data.scene)),
                len(np.unique(data.seed)),
                data.window,
                100 * data.success.mean(),
            ),
            flush=True,
        )
        task_summary = {
            "metadata": {
                "suite": data.suite,
                "task": data.task,
                "source": str(data.source),
                "episodes": data.episodes,
                "scenes": int(len(np.unique(data.scene))),
                "seeds": int(len(np.unique(data.seed))),
                "window": data.window,
                "success_rate": float(data.success.mean()),
            },
            "cv": {},
            "final": {},
        }
        for clusters in args.clusters:
            for representation in REPRESENTATIONS:
                print("  CV K=%d %-7s" % (clusters, representation), flush=True)
                result = cross_validate(
                    data,
                    representation,
                    clusters,
                    args.folds,
                    args.alpha,
                    args.bootstrap,
                    args.seed + 10 * clusters,
                )
                task_summary["cv"]["%d/%s" % (clusters, representation)] = result
                print(
                    "    zero %.3f -> first %.3f (gain %+.3f); second %+.3f; phase %+.3f bit"
                    % (
                        result["zero_bits"],
                        result["first_bits"],
                        result["first_over_zero_gain_bits"],
                        result["second_over_first_gain_bits"],
                        result["phase_over_first_gain_bits"],
                    ),
                    flush=True,
                )

        artifacts: dict[str, np.ndarray] = {}
        for representation in REPRESENTATIONS:
            diagnostics, model_artifact = final_model(
                data,
                representation,
                args.primary_k,
                args.alpha,
                args.seed + (0 if representation == "routing" else 1),
            )
            task_summary["final"][representation] = diagnostics
            for name, value in model_artifact.items():
                artifacts["%s_%s" % (representation, name)] = value
        artifact_path = args.out_dir / (slug + "__models.npz")
        np.savez_compressed(artifact_path, **artifacts)
        task_summary["model_artifact"] = str(artifact_path.resolve())
        summary["tasks"][slug] = task_summary

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    report_path = args.out_dir / "REPORT.md"
    write_report(summary, report_path)
    print("\nwrote %s" % report_path, flush=True)
    print("wrote %s" % summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
