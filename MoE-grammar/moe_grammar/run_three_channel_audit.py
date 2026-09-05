from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from moe_grammar.open_world import (
    HealthyPrefixGrammar,
    HealthyReturnTable,
    PhaseConditionalCDF,
    TaskRobustScaler,
    causal_dwell,
)
from moe_grammar.run_open_world_audit import load_dataset
from moe_grammar.three_channel import (
    ContinuousVARGrammar,
    causal_recurrence_features,
    combine_overregularity,
    episode_balanced_positions,
)


METHODS = (
    "current_innovation",
    "var_innovation",
    "hmm_innovation",
    "innovation_persistent",
    "overregularity",
    "overregularity_persistent",
    "return_failure",
    "three_channel",
)

OVERREGULARITY_COMPONENTS = (
    "freeze",
    "recurrence",
    "low_surprise",
)

SCORED_METHODS = METHODS + OVERREGULARITY_COMPONENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/open-world-phenotypes-v2.npz")
    )
    parser.add_argument(
        "--fold-summary", type=Path, default=Path("results-open-world-global-k12/summary.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results-three-channel-v2"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase-states", type=int, default=12)
    parser.add_argument("--var-order", type=int, default=4)
    parser.add_argument("--return-horizon", type=int, default=3)
    parser.add_argument("--healthy-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def episode_rows(
    starts: np.ndarray, lengths: np.ndarray, episode_indexes: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(starts[index], starts[index] + lengths[index], dtype=np.int64)
            for index in np.asarray(episode_indexes, dtype=np.int64)
        ]
    )


def select_episodes(
    state: np.ndarray,
    states: np.ndarray,
    success: np.ndarray | None = None,
) -> np.ndarray:
    selected = np.isin(state, states)
    if success is not None:
        selected &= success
    return np.flatnonzero(selected)


def balance_episodes_by_task(
    episode_indexes: np.ndarray,
    episode_task: np.ndarray,
    seed: int,
) -> np.ndarray:
    episode_indexes = np.asarray(episode_indexes, dtype=np.int64)
    tasks = np.unique(episode_task[episode_indexes])
    groups = [episode_indexes[episode_task[episode_indexes] == task] for task in tasks]
    count = min(map(len, groups))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(group, count, replace=False) for group in groups]))


def phase_velocity(
    phase: np.ndarray, starts: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    output = np.zeros(len(phase), dtype=np.float32)
    for episode_start, episode_length in zip(starts, lengths):
        start = int(episode_start)
        stop = start + int(episode_length)
        if stop - start > 1:
            output[start + 1 : stop] = np.diff(phase[start:stop])
    return output


def balanced_rows(
    starts: np.ndarray,
    lengths: np.ndarray,
    episodes: np.ndarray,
    samples_per_episode: int = 12,
) -> np.ndarray:
    return episode_balanced_positions(
        starts, lengths, episodes, samples_per_episode=samples_per_episode
    )


def fit_percentile(
    values: np.ndarray,
    phase: np.ndarray,
    rows: np.ndarray,
) -> PhaseConditionalCDF:
    task = np.zeros(len(values), dtype=np.int16)
    return PhaseConditionalCDF().fit(values, task, phase, rows)


def percentile(
    calibrator: PhaseConditionalCDF,
    values: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    return calibrator.transform(values, np.zeros(len(values), dtype=np.int16), phase)


def raw_scores(
    base: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    model: dict[str, Any],
    device: str,
    include_clock: bool = False,
) -> dict[str, np.ndarray]:
    scaler = model["scaler"]
    scaled = scaler.transform(base, np.zeros(len(base), dtype=np.int16))
    grammar = model["prefix_grammar"]
    hmm, phase, belief_entropy = grammar.score(scaled, starts, lengths, device=device)
    var = model["var_grammar"].score(scaled, starts, lengths, device=device)
    current = model["var_grammar"].marginal_score(scaled, device=device)
    speed, periodic, nearest = causal_recurrence_features(scaled, starts, lengths)
    output = {
        "scaled": scaled,
        "hmm": hmm,
        "phase": phase,
        "belief_entropy": belief_entropy,
        "var": var,
        "current": current,
        "speed": speed,
        "periodic": periodic,
        "nearest": nearest,
        "phase_velocity": phase_velocity(phase, starts, lengths),
    }
    if include_clock:
        output["hmm_clock"] = grammar.score(
            scaled,
            starts,
            lengths,
            update_with_observations=False,
            device=device,
        )[0]
    return output


def channel_scores(
    raw: dict[str, np.ndarray],
    starts: np.ndarray,
    lengths: np.ndarray,
    model: dict[str, Any],
) -> dict[str, np.ndarray]:
    phase = raw["phase"]
    calibrators = model["calibrators"]
    current = percentile(calibrators["current"], raw["current"], phase)
    var = percentile(calibrators["var"], raw["var"], phase)
    innovation = percentile(calibrators["hmm"], raw["hmm"], phase)
    freeze = percentile(calibrators["freeze"], -raw["speed"], phase)
    recurrence = percentile(calibrators["recurrence"], -raw["periodic"], phase)
    low_surprise = percentile(calibrators["low_surprise"], -raw["hmm"], phase)
    overregularity = combine_overregularity(freeze, recurrence, low_surprise)
    deviation = np.maximum(innovation, overregularity)
    innovation_dwell = causal_dwell(innovation, starts, lengths, threshold=0.9)
    overregularity_dwell = causal_dwell(overregularity, starts, lengths, threshold=0.9)
    innovation_persistent = innovation * np.minimum(innovation_dwell / 3.0, 1.0)
    overregularity_persistent = overregularity * np.minimum(
        overregularity_dwell / 3.0, 1.0
    )
    return_probability, deviation_dwell = model["return_model"].predict(
        deviation, raw["phase_velocity"], starts, lengths
    )
    return_failure = deviation * (1.0 - return_probability)
    three_channel = np.maximum.reduce(
        [innovation_persistent, overregularity_persistent, return_failure]
    )
    return {
        "current_innovation": current.astype(np.float32),
        "var_innovation": var.astype(np.float32),
        "hmm_innovation": innovation.astype(np.float32),
        "innovation_persistent": innovation_persistent.astype(np.float32),
        "overregularity": overregularity.astype(np.float32),
        "overregularity_persistent": overregularity_persistent.astype(np.float32),
        "return_failure": return_failure.astype(np.float32),
        "three_channel": three_channel.astype(np.float32),
        "freeze": freeze.astype(np.float32),
        "recurrence": recurrence.astype(np.float32),
        "low_surprise": low_surprise.astype(np.float32),
        "return_probability": return_probability.astype(np.float32),
        "deviation": deviation.astype(np.float32),
        "deviation_dwell": deviation_dwell,
        "freeze_percentile": freeze,
        "recurrence_percentile": recurrence,
        "low_surprise_percentile": low_surprise,
    }


def fit_fold(
    arrays: dict[str, np.ndarray],
    fold: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    base = np.asarray(arrays["full_base"], dtype=np.float32)
    starts = arrays["full_starts"]
    lengths = arrays["full_lengths"]
    state = arrays["full_state"]
    success = arrays["full_success"]
    episode_task = arrays["full_task"]
    progress = np.asarray(arrays["full_query_progress"], dtype=np.float32)
    density = select_episodes(state, np.asarray(fold["density_states"]), success)
    return_episodes = select_episodes(state, np.asarray(fold["return_states"]), success)
    calibration = select_episodes(state, np.asarray(fold["calibration_states"]), success)
    density = balance_episodes_by_task(density, episode_task, args.seed + int(fold["fold"]))
    return_episodes = balance_episodes_by_task(
        return_episodes, episode_task, args.seed + 100 + int(fold["fold"])
    )
    calibration = balance_episodes_by_task(
        calibration, episode_task, args.seed + 200 + int(fold["fold"])
    )
    density_rows = balanced_rows(starts, lengths, density)
    calibration_rows = balanced_rows(starts, lengths, calibration)
    task = np.zeros(len(base), dtype=np.int16)
    scaler = TaskRobustScaler().fit(base, task, density_rows, 1)
    scaled = scaler.transform(base, task)
    grammar = HealthyPrefixGrammar(
        phase_states=args.phase_states,
        min_support=64,
        samples_per_episode=args.phase_states,
        equal_episode_transitions=True,
    ).fit(scaled, progress, starts, lengths, density)
    var = ContinuousVARGrammar(order=args.var_order).fit(
        scaled, starts, lengths, density, device=args.device
    )
    model: dict[str, Any] = {
        "scaler": scaler,
        "prefix_grammar": grammar,
        "var_grammar": var,
        "conditioning": "global",
        "phase_states": args.phase_states,
        "density_states": np.asarray(fold["density_states"], dtype=np.int16),
        "return_states": np.asarray(fold["return_states"], dtype=np.int16),
        "calibration_states": np.asarray(fold["calibration_states"], dtype=np.int16),
        "test_states": np.asarray(fold["test_states"], dtype=np.int16),
        "feature_definition": "independent-layer-permutation-invariant-moe-phenotype-v2",
    }
    raw = raw_scores(base, starts, lengths, model, args.device, include_clock=True)
    calibrators = {
        "current": fit_percentile(raw["current"], raw["phase"], calibration_rows),
        "var": fit_percentile(raw["var"], raw["phase"], calibration_rows),
        "hmm": fit_percentile(raw["hmm"], raw["phase"], calibration_rows),
        "freeze": fit_percentile(-raw["speed"], raw["phase"], calibration_rows),
        "recurrence": fit_percentile(-raw["periodic"], raw["phase"], calibration_rows),
        "low_surprise": fit_percentile(-raw["hmm"], raw["phase"], calibration_rows),
    }
    model["calibrators"] = calibrators
    preliminary = channel_scores_without_return(raw, starts, lengths, calibrators)
    return_model = HealthyReturnTable(horizon=args.return_horizon).fit(
        preliminary["deviation"],
        raw["phase_velocity"],
        starts,
        lengths,
        return_episodes,
    )
    model["return_model"] = return_model
    scores = channel_scores(raw, starts, lengths, model)
    model["balanced_counts"] = {
        "density_episodes": len(density),
        "return_episodes": len(return_episodes),
        "calibration_episodes": len(calibration),
        "density_rows": len(density_rows),
        "calibration_rows": len(calibration_rows),
    }
    return model, raw, scores


def channel_scores_without_return(
    raw: dict[str, np.ndarray],
    starts: np.ndarray,
    lengths: np.ndarray,
    calibrators: dict[str, PhaseConditionalCDF],
) -> dict[str, np.ndarray]:
    phase = raw["phase"]
    innovation = percentile(calibrators["hmm"], raw["hmm"], phase)
    freeze = percentile(calibrators["freeze"], -raw["speed"], phase)
    recurrence = percentile(calibrators["recurrence"], -raw["periodic"], phase)
    low_surprise = percentile(calibrators["low_surprise"], -raw["hmm"], phase)
    overregularity = combine_overregularity(freeze, recurrence, low_surprise)
    return {
        "innovation": innovation,
        "overregularity": overregularity,
        "deviation": np.maximum(innovation, overregularity),
    }


def episode_maxima(
    values: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    episodes: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            np.max(values[int(starts[index]) : int(starts[index] + lengths[index])])
            for index in episodes
        ],
        dtype=np.float64,
    )


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return [center - radius / denominator, center + radius / denominator]


def physical_metrics(
    arrays: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    assigned_fold: np.ndarray,
    thresholds: dict[int, dict[str, float]],
) -> dict[str, Any]:
    starts = arrays["anchor_starts"]
    lengths = arrays["anchor_lengths"]
    state = arrays["anchor_state"]
    success = arrays["anchor_success"]
    onset = arrays["anchor_stasis_onset"]
    output: dict[str, Any] = {}
    for method in SCORED_METHODS:
        false_alarms = 0
        success_total = 0
        before3 = 0
        by_onset = 0
        by_post3 = 0
        event_total = 0
        leads: list[int] = []
        matched_wins = 0.0
        matched_pairs = 0
        for episode_index in range(len(starts)):
            start = int(starts[episode_index])
            length = int(lengths[episode_index])
            fold = int(assigned_fold[episode_index])
            threshold = thresholds[fold][method]
            sequence = scores[method][start : start + length]
            crossing = np.flatnonzero(sequence > threshold)
            first = int(crossing[0]) if len(crossing) else None
            if success[episode_index]:
                success_total += 1
                false_alarms += int(first is not None)
                continue
            if onset[episode_index] < 0:
                continue
            event_total += 1
            event_onset = int(onset[episode_index])
            if first is not None:
                before3 += int(first <= event_onset - 3)
                by_onset += int(first <= event_onset)
                by_post3 += int(first <= event_onset + 3)
                if first <= event_onset:
                    leads.append(event_onset - first)
            event_value = sequence[min(event_onset, length - 1)]
            controls = np.flatnonzero(success & (state == state[episode_index]))
            for control in controls:
                control_length = int(lengths[control])
                if control_length <= event_onset:
                    continue
                control_row = int(starts[control]) + event_onset
                difference = event_value - scores[method][control_row]
                matched_wins += float(difference > 0) + 0.5 * float(difference == 0)
                matched_pairs += 1
        output[method] = {
            "success_episode_false_alarms": false_alarms,
            "success_episodes": success_total,
            "success_episode_fpr": false_alarms / success_total,
            "success_episode_fpr_ci": wilson(false_alarms, success_total),
            "stasis_events": event_total,
            "recall_onset_minus3": before3 / event_total,
            "recall_by_onset": by_onset / event_total,
            "recall_by_onset_plus3": by_post3 / event_total,
            "recall_by_onset_ci": wilson(by_onset, event_total),
            "median_lead_if_pre_onset": float(np.median(leads)) if leads else None,
            "state_and_query_matched_auc": (
                matched_wins / matched_pairs if matched_pairs else None
            ),
            "matched_pairs": matched_pairs,
        }
    return output


def render_report(summary: dict[str, Any]) -> str:
    next_query = summary["healthy_next_query"]
    physical = summary["physical_stasis_matched_5pct_operating_point"]
    deployment = summary["physical_stasis_anchor_calibrated"]
    return_eval = summary["healthy_excursion_return"]

    def pct(value: float) -> str:
        return f"{100.0 * value:.1f}%"

    lines = [
        "# 三通道 MoE routing observer（v2）",
        "",
        "## 核心结论",
        "",
        (
            f"在 held-out 成功序列上，soft HMM 的完整前缀相对仅按内部时钟推进的 "
            f"next-query 改善为 {next_query['hmm_prefix_gain_bits_per_phenotype']:.4f} "
            "bits/phenotype；连续 VAR(4) 相对无历史 Gaussian 的改善为 "
            f"{next_query['var_gain_bits_per_phenotype']:.4f} bits/phenotype。"
        ),
        "",
        "这只证明 routing history 可预测下一 query，不自动证明高 surprisal 等于 Trap。",
        "",
        (
            f"健康 excursion 的局部 return 任务含 {return_eval['rows']} 个 held-out 异常前缀；"
            f"三步不返回的 AUC 为 {return_eval['persistent_auc']:.3f}，"
            f"return-probability Brier 为 {return_eval['return_brier']:.3f}。"
        ),
        "",
        "Scene8 的物理 stasis 是当前唯一可靠 onset 锚点；没有可靠 loop onset，因此本报告不伪造 loop 结论。",
        "",
        "## 物理 onset 结果（固定 5% success-episode FPR）",
        "",
        "| channel | success episode FPR | recall q-3 | recall by onset | recall onset+3 | matched AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = physical[method]
        auc = row["state_and_query_matched_auc"]
        lines.append(
            f"| {method} | {pct(row['success_episode_fpr'])} | "
            f"{pct(row['recall_onset_minus3'])} | {pct(row['recall_by_onset'])} | "
            f"{pct(row['recall_by_onset_plus3'])} | "
            f"{'n/a' if auc is None else f'{auc:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "### Over-regularity components",
            "",
            "| component | success episode FPR | recall q-3 | recall by onset | matched AUC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in OVERREGULARITY_COMPONENTS:
        row = physical[method]
        auc = row["state_and_query_matched_auc"]
        lines.append(
            f"| {method} | {pct(row['success_episode_fpr'])} | "
            f"{pct(row['recall_onset_minus3'])} | {pct(row['recall_by_onset'])} | "
            f"{'n/a' if auc is None else f'{auc:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "所有分数均来自 init-state held-out 模型。这里仅用 pooled out-of-fold Scene8 success controls "
            "读取 5% FPR operating point，再评估 stasis episodes；它用于公平比较通道，不冒充可部署阈值。",
            "",
            "## Scene8 跨状态阈值部署",
            "",
            "| channel | Scene8 success FPR | recall by onset | matched AUC |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = deployment[method]
        auc = row["state_and_query_matched_auc"]
        lines.append(
            f"| {method} | {pct(row['success_episode_fpr'])} | "
            f"{pct(row['recall_by_onset'])} | "
            f"{'n/a' if auc is None else f'{auc:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "每折阈值由其他 Scene8 calibration states 的 success episodes 决定，再原样部署到 test states。"
            "实际 FPR 偏离 5% 表示 init-state threshold shift。",
            "",
            "full40 阈值直接迁移到 Scene8 的完整诊断保存在 `summary.json` 的 "
            "`physical_stasis_global_threshold_transfer`；它用于度量更强的跨任务阈值漂移。",
            "",
            "## 表征修正",
            "",
            "- 删除跨层同编号 expert distribution 的直接比较。新量只比较每层置换不变的标量 profile。",
            "- 新主模型使用 22 维 query phenotype，不经过 2187→PCA→hard GMM 链路。",
            "- VAR/HMM 都是连续 emission；HMM 用 forward marginalization，不使用硬 word context。",
            "- density episode 按 task 等量抽取，每 episode 均匀抽相同数量 phase positions。",
            "- 在线输入不含 task ID、绝对 query index、物理状态或最终 outcome。",
            "",
            "## 三个通道",
            "",
            "- `hmm_innovation`：完整 routing prefix 下的上尾 surprisal。",
            "- `overregularity`：lag-1 freeze、lag-2..4 recurrence、异常低 surprisal 中至少两项共同升高。",
            "- `return_failure`：健康数据中相似 excursion 在未来 3 query 不返回的经验风险。",
            "- `three_channel`：三个可解释风险通道的并集，不使用固定 drift 的单边 CUSUM。",
            "",
            "## 边界",
            "",
            "return 标签由健康 routing excursion 自身定义，用于验证局部序列目标；它不是新的物理 Trap 标签。"
            "物理有效性仍以独立 stasis onset 为准。若三通道不能在相同 episode FPR 下超过 current-query，"
            "就应把 grammar 降级为机制分析，而不是主 detector。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    started = time.time()
    arrays, metadata = load_dataset(args.dataset)
    if metadata.get("schema_version") != 2:
        raise ValueError("three-channel audit requires the corrected v2 phenotype artifact")
    fold_summary = json.loads(args.fold_summary.read_text(encoding="utf-8"))
    folds = fold_summary["folds"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)

    anchor_size = len(arrays["anchor_base"])
    anchor_scores = {
        method: np.full(anchor_size, np.nan, dtype=np.float32)
        for method in SCORED_METHODS
    }
    anchor_assigned_fold = np.full(len(arrays["anchor_starts"]), -1, dtype=np.int8)
    global_thresholds: dict[int, dict[str, float]] = {}
    anchor_thresholds: dict[int, dict[str, float]] = {}
    next_query_records: list[dict[str, float]] = []
    return_predictions: list[float] = []
    return_targets: list[bool] = []
    persistent_scores: list[float] = []
    deviation_scores: list[float] = []

    full_starts = arrays["full_starts"]
    full_lengths = arrays["full_lengths"]
    full_state = arrays["full_state"]
    full_success = arrays["full_success"]
    for fold in folds:
        fold_index = int(fold["fold"])
        print(f"fold {fold_index}: fitting balanced continuous models", flush=True)
        model, raw, scores = fit_fold(arrays, fold, args)
        joblib.dump(model, args.output_dir / "models" / f"fold_{fold_index}.joblib", compress=3)

        calibration = select_episodes(
            full_state, np.asarray(fold["calibration_states"]), full_success
        )
        global_thresholds[fold_index] = {
            method: float(
                np.quantile(
                    episode_maxima(
                        scores[method], full_starts, full_lengths, calibration
                    ),
                    1.0 - args.healthy_fpr,
                    method="higher",
                )
            )
            for method in SCORED_METHODS
        }
        test_success = select_episodes(
            full_state, np.asarray(fold["test_states"]), full_success
        )
        test_success = balance_episodes_by_task(
            test_success, arrays["full_task"], args.seed + 300 + fold_index
        )
        test_rows = balanced_rows(full_starts, full_lengths, test_success)
        next_query_records.append(
            {
                "rows": len(test_rows),
                "hmm": float(raw["hmm"][test_rows].sum()),
                "hmm_clock": float(raw["hmm_clock"][test_rows].sum()),
                "var": float(raw["var"][test_rows].sum()),
                "current": float(raw["current"][test_rows].sum()),
            }
        )
        for episode_index in test_success:
            start = int(full_starts[episode_index])
            length = int(full_lengths[episode_index])
            for offset in range(max(0, length - args.return_horizon)):
                row = start + offset
                if scores["deviation"][row] < 0.8:
                    continue
                future = scores["deviation"][row + 1 : row + args.return_horizon + 1]
                returned = bool(np.any(future < 0.8))
                return_predictions.append(float(scores["return_probability"][row]))
                return_targets.append(returned)
                persistent_scores.append(float(scores["return_failure"][row]))
                deviation_scores.append(float(scores["deviation"][row]))

        anchor_raw = raw_scores(
            np.asarray(arrays["anchor_base"], dtype=np.float32),
            arrays["anchor_starts"],
            arrays["anchor_lengths"],
            model,
            args.device,
        )
        current_anchor_scores = channel_scores(
            anchor_raw, arrays["anchor_starts"], arrays["anchor_lengths"], model
        )
        anchor_calibration = select_episodes(
            arrays["anchor_state"],
            np.asarray(fold["calibration_states"]),
            arrays["anchor_success"],
        )
        if len(anchor_calibration) == 0:
            raise ValueError(f"fold {fold_index} has no Scene8 calibration successes")
        anchor_thresholds[fold_index] = {
            method: float(
                np.quantile(
                    episode_maxima(
                        current_anchor_scores[method],
                        arrays["anchor_starts"],
                        arrays["anchor_lengths"],
                        anchor_calibration,
                    ),
                    1.0 - args.healthy_fpr,
                    method="higher",
                )
            )
            for method in SCORED_METHODS
        }
        selected_episodes = np.flatnonzero(
            np.isin(arrays["anchor_state"], np.asarray(fold["test_states"]))
        )
        selected_rows = episode_rows(
            arrays["anchor_starts"], arrays["anchor_lengths"], selected_episodes
        )
        for method in SCORED_METHODS:
            anchor_scores[method][selected_rows] = current_anchor_scores[method][selected_rows]
        anchor_assigned_fold[selected_episodes] = fold_index

    if np.any(anchor_assigned_fold < 0) or any(
        not np.isfinite(values).all() for values in anchor_scores.values()
    ):
        raise AssertionError("anchor fold assignment is incomplete")
    total_rows = sum(record["rows"] for record in next_query_records)
    totals = {
        key: sum(record[key] for record in next_query_records)
        for key in ("hmm", "hmm_clock", "var", "current")
    }
    return_probability = np.asarray(return_predictions)
    returned = np.asarray(return_targets, dtype=bool)
    persistent_score = np.asarray(persistent_scores)
    deviation_score = np.asarray(deviation_scores)
    if not returned.any() or returned.all():
        raise ValueError("held-out return target has only one class")
    healthy_next_query = {
        "balanced_test_rows": total_rows,
        "hmm_nll_nats_per_phenotype": totals["hmm"] / total_rows,
        "hmm_clock_nll_nats_per_phenotype": totals["hmm_clock"] / total_rows,
        "hmm_prefix_gain_bits_per_phenotype": (
            (totals["hmm_clock"] - totals["hmm"]) / total_rows / np.log(2.0)
        ),
        "var_nll_nats_per_phenotype": totals["var"] / total_rows,
        "current_nll_nats_per_phenotype": totals["current"] / total_rows,
        "var_gain_bits_per_phenotype": (
            (totals["current"] - totals["var"]) / total_rows / np.log(2.0)
        ),
    }
    return_evaluation = {
        "rows": len(returned),
        "returned_fraction": float(returned.mean()),
        "return_brier": float(np.square(return_probability - returned).mean()),
        "persistent_auc": float(roc_auc_score(~returned, persistent_score)),
        "deviation_only_persistent_auc": float(roc_auc_score(~returned, deviation_score)),
    }
    physical_anchor_calibrated = physical_metrics(
        arrays, anchor_scores, anchor_assigned_fold, anchor_thresholds
    )
    physical_global_transfer = physical_metrics(
        arrays, anchor_scores, anchor_assigned_fold, global_thresholds
    )
    anchor_success_episodes = np.flatnonzero(arrays["anchor_success"])
    pooled_thresholds = {
        method: float(
            np.quantile(
                episode_maxima(
                    anchor_scores[method],
                    arrays["anchor_starts"],
                    arrays["anchor_lengths"],
                    anchor_success_episodes,
                ),
                1.0 - args.healthy_fpr,
                method="higher",
            )
        )
        for method in SCORED_METHODS
    }
    matched_thresholds = {
        int(fold["fold"]): pooled_thresholds for fold in folds
    }
    physical_matched_fpr = physical_metrics(
        arrays, anchor_scores, anchor_assigned_fold, matched_thresholds
    )
    summary = {
        "schema_version": 3,
        "protocol": {
            "feature_definition": metadata["definition"],
            "inference_inputs": "MoE routing only; no task id, query index, or physical state",
            "density_sampling": "equal task count, equal episode weight, phase-uniform positions",
            "innovation_models": ["continuous VAR(4)", "12-state soft Gaussian HMM"],
            "overregularity": "top-two mean of freeze, lag-2..4 recurrence, low surprisal",
            "return_horizon": args.return_horizon,
            "primary_threshold_calibration": (
                "Scene8 calibration-state success episode maxima; test states disjoint"
            ),
            "matched_fpr_operating_point": (
                "pooled out-of-fold Scene8 success controls; descriptive comparison only"
            ),
            "transfer_threshold_calibration": (
                "full40 calibration-state success episode maxima; applied unchanged to Scene8"
            ),
            "state_split": "five-fold disjoint init states",
            "physical_anchor": metadata["anchor_definition"],
        },
        "counts": {
            "full_episodes": len(arrays["full_starts"]),
            "full_queries": len(arrays["full_base"]),
            "anchor_episodes": len(arrays["anchor_starts"]),
            "anchor_stasis": int(np.sum(arrays["anchor_stasis_onset"] >= 0)),
        },
        "healthy_next_query": healthy_next_query,
        "healthy_excursion_return": return_evaluation,
        "physical_stasis_matched_5pct_operating_point": physical_matched_fpr,
        "physical_stasis_anchor_calibrated": physical_anchor_calibrated,
        "physical_stasis_global_threshold_transfer": physical_global_transfer,
        "matched_5pct_thresholds": pooled_thresholds,
        "anchor_thresholds_by_fold": anchor_thresholds,
        "global_thresholds_by_fold": global_thresholds,
        "elapsed_seconds": time.time() - started,
    }
    np.savez_compressed(
        args.output_dir / "anchor_scores.npz",
        **anchor_scores,
        assigned_fold=anchor_assigned_fold,
        anchor_starts=arrays["anchor_starts"],
        anchor_lengths=arrays["anchor_lengths"],
        anchor_state=arrays["anchor_state"],
        anchor_success=arrays["anchor_success"],
        anchor_stasis_onset=arrays["anchor_stasis_onset"],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.zh.md").write_text(render_report(summary), encoding="utf-8")
    print(
        f"wrote {args.output_dir} in {time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
