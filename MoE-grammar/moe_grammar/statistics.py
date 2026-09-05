from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import rankdata, spearmanr


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    if len(reference) == 0:
        raise ValueError("empty calibration reference")
    return (np.searchsorted(reference, values, side="right") + 0.5) / (len(reference) + 1.0)


def cusum(values: np.ndarray, kappa: float = 0.8) -> np.ndarray:
    output = np.empty(len(values), dtype=np.float64)
    current = 0.0
    for index, value in enumerate(values):
        current = max(0.0, current + float(value) - kappa)
        output[index] = current
    return output


def auc_pairwise(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    difference = positive[:, None] - negative[None, :]
    return float((np.sum(difference > 0.0) + 0.5 * np.sum(difference == 0.0)) / difference.size)


def stratified_pair_auc(
    labels: np.ndarray, scores: np.ndarray, strata: np.ndarray
) -> tuple[float, dict[str, float], int]:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    strata = np.asarray(strata)
    wins = 0.0
    pairs = 0
    by_stratum: dict[str, float] = {}
    for value in np.unique(strata):
        selected = strata == value
        positive = scores[selected & labels]
        negative = scores[selected & ~labels]
        if len(positive) == 0 or len(negative) == 0:
            continue
        difference = positive[:, None] - negative[None, :]
        local_wins = np.sum(difference > 0.0) + 0.5 * np.sum(difference == 0.0)
        local_pairs = difference.size
        wins += local_wins
        pairs += local_pairs
        by_stratum[str(value)] = float(local_wins / local_pairs)
    return (float(wins / pairs) if pairs else float("nan"), by_stratum, pairs)


def state_blocked_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    task: np.ndarray,
    state: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    unique_states = np.unique(state)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(unique_states, size=len(unique_states), replace=True)
        sample_labels: list[np.ndarray] = []
        sample_scores: list[np.ndarray] = []
        sample_strata: list[np.ndarray] = []
        for copy_index, selected_state in enumerate(sampled):
            mask = state == selected_state
            sample_labels.append(labels[mask])
            sample_scores.append(scores[mask])
            sample_strata.append(
                np.asarray([f"{item}|{copy_index}" for item in task[mask]], dtype=object)
            )
        values[draw] = stratified_pair_auc(
            np.concatenate(sample_labels),
            np.concatenate(sample_scores),
            np.concatenate(sample_strata),
        )[0]
    finite = values[np.isfinite(values)]
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def state_blocked_auc_difference(
    labels: np.ndarray,
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    task: np.ndarray,
    state: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float | list[float]]:
    labels = np.asarray(labels, dtype=np.bool_)
    left_scores = np.asarray(left_scores, dtype=np.float64)
    right_scores = np.asarray(right_scores, dtype=np.float64)
    task = np.asarray(task)
    state = np.asarray(state)
    strata = np.asarray(
        [f"{item_task}|{item_state}" for item_task, item_state in zip(task, state)],
        dtype=object,
    )
    left_auc = stratified_pair_auc(labels, left_scores, strata)[0]
    right_auc = stratified_pair_auc(labels, right_scores, strata)[0]
    unique_states = np.unique(state)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(unique_states, size=len(unique_states), replace=True)
        sample_labels: list[np.ndarray] = []
        sample_left: list[np.ndarray] = []
        sample_right: list[np.ndarray] = []
        sample_strata: list[np.ndarray] = []
        for copy_index, selected_state in enumerate(sampled):
            mask = state == selected_state
            sample_labels.append(labels[mask])
            sample_left.append(left_scores[mask])
            sample_right.append(right_scores[mask])
            sample_strata.append(
                np.asarray([f"{item}|{copy_index}" for item in task[mask]], dtype=object)
            )
        joined_labels = np.concatenate(sample_labels)
        joined_strata = np.concatenate(sample_strata)
        bootstrap_left = stratified_pair_auc(
            joined_labels, np.concatenate(sample_left), joined_strata
        )[0]
        bootstrap_right = stratified_pair_auc(
            joined_labels, np.concatenate(sample_right), joined_strata
        )[0]
        null[draw] = bootstrap_left - bootstrap_right
    finite = null[np.isfinite(null)]
    return {
        "auc_left_minus_right": float(left_auc - right_auc),
        "state_blocked_ci95": [float(value) for value in np.quantile(finite, [0.025, 0.975])],
    }


def paired_state_test(
    left: np.ndarray, right: np.ndarray, states: np.ndarray, seed: int = 0
) -> dict[str, float | list[float]]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    states = np.asarray(states)
    unique = np.unique(states)
    differences = np.asarray(
        [np.mean(left[states == value] - right[states == value]) for value in unique]
    )
    observed = float(differences.mean())
    if len(differences) <= 20:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(differences))))
        null = (signs * differences).mean(axis=1)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(200_000, len(differences)))
        null = (signs * differences).mean(axis=1)
    p_less = float((np.sum(null <= observed) + 1.0) / (len(null) + 1.0))
    rng = np.random.default_rng(seed + 1)
    bootstrap = np.asarray(
        [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(10_000)]
    )
    return {
        "mean_left_minus_right": observed,
        "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "one_sided_p_left_less": p_less,
        "states": int(len(unique)),
    }


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(left, right, nan_policy="omit")
    return float(result.statistic)


def rank_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (rankdata(values, method="average") - 0.5) / len(values)
