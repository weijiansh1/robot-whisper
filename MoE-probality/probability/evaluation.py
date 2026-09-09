"""Risk-set evaluation, first-alarm statistics, and cluster uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


MODELS = ("prior", "budget_raw", "budget", "moe_raw", "moe")


def wilson(successes, total):
    if total == 0:
        return None, None
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total ** 2)) / (1 + z * z / total)
    return float(max(0, center - half)), float(min(1, center + half))


def cluster_draws(frame, columns, repeats=1000, population=None):
    """Resample task/init clusters within each task, preserving its seed siblings."""
    groups = frame.groupby(["task", "cluster"])[columns].sum()
    if population is not None:
        keys = pd.MultiIndex.from_frame(population[["task", "cluster"]].drop_duplicates())
        groups = groups.reindex(keys, fill_value=0)
    groups = groups.reset_index()
    rng = np.random.default_rng(20260907)
    draws = np.zeros((repeats, len(columns)))
    for _, group in groups.groupby("task", sort=True):
        values = group[columns].to_numpy(float)
        indices = rng.integers(len(values), size=(repeats, len(values)))
        draws += values[indices].sum(axis=1)
    return draws


def alarm_statistics(episodes):
    rows = []
    for split in ("all_natural", "test_unseen_init", "test_new_noise_seen_init"):
        subset = episodes if split == "all_natural" else episodes[episodes.split == split]
        for suite in ("all", *sorted(episodes.suite.unique())):
            block = subset if suite == "all" else subset[subset.suite == suite]
            alarmed = block[block.first_alarm_q >= 0].copy()
            n, k = len(alarmed), int(alarmed.success.sum())
            low, high = wilson(k, n)
            ci = (None, None)
            if 0 < k < n:
                alarmed["numerator"] = alarmed.success.astype(int)
                alarmed["denominator"] = 1
                draws = cluster_draws(alarmed, ["numerator", "denominator"], population=block)
                ratios = draws[draws[:, 1] > 0, 0] / draws[draws[:, 1] > 0, 1]
                ci = tuple(float(v) for v in np.quantile(ratios, [0.025, 0.975]))
            rows.append(dict(split=split, suite=suite, episodes=len(block), alarmed_episodes=n,
                             alarmed_successes=k, alarmed_failures=n-k,
                             success_after_first_alarm=k/n if n else None,
                             wilson_low=low, wilson_high=high, cluster_low=ci[0], cluster_high=ci[1],
                             cluster_interval_status="available" if 0 < k < n else "undefined_or_degenerate",
                             first_alarm_q_median=float(alarmed.first_alarm_q.median()) if n else None))
    return pd.DataFrame(rows)


def alarm_sensitivity(episodes):
    rows = []
    for name in [c for c in episodes if c.startswith("first_q_")]:
        for split in ("all_natural", "test_unseen_init", "test_new_noise_seen_init"):
            block = episodes if split == "all_natural" else episodes[episodes.split == split]
            alarmed = block[block[name] >= 0]
            n, k = len(alarmed), int(alarmed.success.sum())
            low, high = wilson(k, n)
            rows.append(dict(rule=name.removeprefix("first_q_"), split=split, episodes=len(block),
                             alarmed=n, successes=k, failures=n-k, success_rate=k/n if n else None,
                             wilson_low=low, wilson_high=high))
    return pd.DataFrame(rows)


def calibration_bins(y, p, weights=None):
    weights = np.ones(len(y)) if weights is None else np.asarray(weights)
    bins = np.minimum((p * 10).astype(int), 9)
    rows = []
    for b in range(10):
        mask = bins == b
        if mask.any():
            rows.append(dict(bin=b, low=b / 10, high=(b + 1) / 10, rows=int(mask.sum()),
                             weight=float(weights[mask].sum()),
                             predicted=float(np.average(p[mask], weights=weights[mask])),
                             observed=float(np.average(y[mask], weights=weights[mask]))))
    return rows


def metrics(y, p, weights=None):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    bins = calibration_bins(y, p, weights)
    return dict(rows=len(y), successes=int(y.sum()),
                success_rate=float(np.average(y, weights=weights)),
                brier=float(np.average((p-y)**2, weights=weights)),
                log_loss=float(np.average(-(y*np.log(p)+(1-y)*np.log1p(-p)), weights=weights)),
                auroc=float(roc_auc_score(y, p, sample_weight=weights)) if len(np.unique(y)) == 2 else None,
                ece10=sum(abs(r["predicted"]-r["observed"])*r["weight"] for r in bins)
                / sum(r["weight"] for r in bins))


def evaluate(predictions, episodes):
    index = episodes.set_index("episode_row")
    p = predictions.copy()
    p["suite"] = index.loc[p.episode_row, "suite"].to_numpy()
    p["split"] = index.loc[p.episode_row, "split"].to_numpy()
    p["success"] = index.loc[p.episode_row, "success"].to_numpy(int)
    p["first_alarm"] = p["query"].to_numpy() == index.loc[p.episode_row, "first_alarm_q"].to_numpy()
    p["first_alarm_half_k4"] = p["query"].to_numpy() == index.loc[p.episode_row, "first_q_freeze_back_half_k4"].to_numpy()
    p["length"] = index.loc[p.episode_row, "length"].to_numpy()
    rows, reliability = [], []
    for split in ("calibration", "test_unseen_init", "test_new_noise_seen_init"):
        for suite in ("all", *sorted(episodes.suite.unique())):
            base = p[(p.split == split) & ((p.suite == suite) if suite != "all" else True)]
            for selection in ("all_queries", "first_alarm", "first_alarm_half_k4", "episode_equal", "q0", "q7", "q15", "q25"):
                if selection.startswith("first_alarm"):
                    block = base[base[selection]]
                elif selection.startswith("q"):
                    block = base[base["query"] == int(selection[1:])]
                else:
                    block = base
                if block.empty:
                    continue
                weights = 1 / block.length.to_numpy() if selection == "episode_equal" else None
                for model in MODELS:
                    y, v = block.success.to_numpy(), block[model].to_numpy()
                    rows.append(dict(split=split, suite=suite, selection=selection, model=model,
                                     episodes=int(block.episode_row.nunique()), **metrics(y, v, weights)))
                    if suite == "all" and selection in ("all_queries", "first_alarm", "first_alarm_half_k4"):
                        reliability.extend(dict(split=split, selection=selection, model=model, **r)
                                           for r in calibration_bins(y, v))
    uncertainty = []
    primary = p[p.split == "test_unseen_init"].copy()
    primary["cluster"] = index.loc[primary.episode_row, "cluster"].to_numpy()
    primary["task"] = index.loc[primary.episode_row, "task"].to_numpy()
    for selection in ("all_queries", "first_alarm", "first_alarm_half_k4"):
        block = primary if selection == "all_queries" else primary[primary[selection]].copy()
        block = block.copy()
        for model in ("moe", "budget"):
            v = np.clip(block[model].to_numpy(), 1e-6, 1 - 1e-6)
            y = block.success.to_numpy()
            block[model + "_brier"] = (v-y)**2
            block[model + "_log_loss"] = -(y*np.log(v)+(1-y)*np.log1p(-v))
        block["count"] = 1
        cols = [f"{m}_{s}" for m in ("moe", "budget") for s in ("brier", "log_loss")] + ["count"]
        draws = cluster_draws(block, cols, population=primary)
        draws = draws[draws[:, -1] > 0]
        for i, metric in enumerate(cols[:-1]):
            low, high = np.quantile(draws[:, i]/draws[:, -1], [0.025, 0.975])
            uncertainty.append(dict(selection=selection, metric=metric,
                                    estimate=float(block[metric].mean()), low=float(low), high=float(high)))
        for i, metric in enumerate(("brier", "log_loss")):
            diff = (draws[:, i] - draws[:, i+2])/draws[:, -1]
            low, high = np.quantile(diff, [0.025, 0.975])
            uncertainty.append(dict(selection=selection, metric="moe_minus_budget_" + metric,
                                    estimate=float((block[cols[i]]-block[cols[i+2]]).mean()),
                                    low=float(low), high=float(high)))
    return pd.DataFrame(rows), pd.DataFrame(reliability), pd.DataFrame(uncertainty)


def matched_alarm_controls(episodes):
    """Exact task, query and cap matching; task stage remains unobserved."""
    rows = []
    for alarm in episodes[(episodes.split == "test_unseen_init") & (episodes.first_alarm_q >= 0)].itertuples():
        q = alarm.first_alarm_q
        controls = episodes[(episodes.split == "test_unseen_init") & (episodes.task == alarm.task)
                            & (episodes.run_id == alarm.run_id) & (episodes.length > q)
                            & ((episodes.first_alarm_q < 0) | (episodes.first_alarm_q > q))]
        rows.append(dict(episode_row=alarm.episode_row, query=q, task=alarm.task,
                         success=int(alarm.success), controls=len(controls),
                         matched_control_success=controls.success.mean() if len(controls) else np.nan))
    return pd.DataFrame(rows)
