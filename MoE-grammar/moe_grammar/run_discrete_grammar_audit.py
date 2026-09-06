from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from moe_grammar.discrete_phenotype import (
    TRACK_NAMES,
    TRACK_SYMBOLS,
    FactorialCategoricalHMM,
    FactorialContextGrammar,
    FactorialDurationModel,
    TernaryChordTokenizer,
    categorical_recurrence,
    chord_codes,
    chord_label,
    episode_uniform_rows,
    phenotype_track_scores,
)
from moe_grammar.open_world import (
    HealthyReturnTable,
    PhaseConditionalCDF,
    TaskRobustScaler,
    causal_dwell,
)
from moe_grammar.run_open_world_audit import load_dataset
from moe_grammar.run_three_channel_audit import (
    balance_episodes_by_task,
    balanced_rows,
    episode_maxima,
    episode_rows,
    phase_velocity,
    select_episodes,
    wilson,
)
from moe_grammar.three_channel import combine_overregularity


INNOVATION_METHODS = (
    "unigram_innovation",
    "clock_innovation",
    "prefix_innovation",
    "ordered4_innovation",
    "bag4_innovation",
)
MAIN_METHODS = INNOVATION_METHODS + (
    "prefix_persistent",
    "overregularity",
    "overregularity_persistent",
    "return_failure",
    "discrete_three_channel",
)
COMPONENTS = ("duration", "freeze", "periodic", "low_surprise")
SCORED_METHODS = MAIN_METHODS + COMPONENTS
DIRECT_MOTIFS = (
    "fl_fl_sy_sy",
    "fl_parallel_sy",
    "switching_pair",
    "lock_streak",
    "abab",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/open-world-phenotypes-v2.npz")
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=Path("results-open-world-global-k12/summary.json"),
    )
    parser.add_argument(
        "--continuous-summary",
        type=Path,
        default=Path("results-three-channel-v2/summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results-discrete-grammar-v1")
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--phase-states", type=int, default=12)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--return-horizon", type=int, default=3)
    parser.add_argument("--healthy-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="regenerate REPORT.zh.md from an existing summary without refitting",
    )
    return parser.parse_args()


def fit_percentile(
    values: np.ndarray, phase: np.ndarray, rows: np.ndarray
) -> PhaseConditionalCDF:
    return PhaseConditionalCDF().fit(
        values,
        np.zeros(len(values), dtype=np.int8),
        phase,
        rows,
    )


def percentile(
    calibrator: PhaseConditionalCDF, values: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    return calibrator.transform(
        values,
        np.zeros(len(values), dtype=np.int8),
        phase,
    )


def raw_scores(
    base: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    model: dict[str, Any],
    device: str,
) -> dict[str, np.ndarray]:
    task = np.zeros(len(base), dtype=np.int8)
    scaled = model["scaler"].transform(base, task)
    tracks = phenotype_track_scores(scaled)
    chords = model["tokenizer"].transform(tracks)
    prefix, phase, belief_entropy = model["hmm"].score(
        chords, starts, lengths, device=device
    )
    clock, clock_phase, _ = model["hmm"].score(
        chords,
        starts,
        lengths,
        update_with_observations=False,
        device=device,
    )
    ordered, ordered_depth = model["ordered"].score(chords, starts, lengths)
    bag, bag_depth = model["bag"].score(chords, starts, lengths)
    duration, duration_tracks = model["duration"].score(chords, starts, lengths)
    freeze, periodic, winning_lag = categorical_recurrence(chords, starts, lengths)
    return {
        "scaled": scaled,
        "tracks": tracks,
        "chords": chords,
        "unigram": model["hmm"].global_nll(chords),
        "clock": clock,
        "clock_phase": clock_phase,
        "prefix": prefix,
        "phase": phase,
        "belief_entropy": belief_entropy,
        "ordered": ordered,
        "bag": bag,
        "ordered_depth": ordered_depth,
        "bag_depth": bag_depth,
        "duration": duration,
        "duration_tracks": duration_tracks,
        "freeze": freeze,
        "periodic": periodic,
        "winning_lag": winning_lag,
        "phase_velocity": phase_velocity(phase, starts, lengths),
    }


def channel_scores_without_return(
    raw: dict[str, np.ndarray],
    starts: np.ndarray,
    lengths: np.ndarray,
    calibrators: dict[str, PhaseConditionalCDF],
) -> dict[str, np.ndarray]:
    zeros = np.zeros(len(raw["phase"]), dtype=np.float32)
    unigram = percentile(calibrators["unigram"], raw["unigram"], zeros)
    clock = percentile(calibrators["clock"], raw["clock"], raw["clock_phase"])
    prefix = percentile(calibrators["prefix"], raw["prefix"], raw["phase"])
    ordered = percentile(calibrators["ordered"], raw["ordered"], zeros)
    bag = percentile(calibrators["bag"], raw["bag"], zeros)
    duration = percentile(calibrators["duration"], raw["duration"], raw["phase"])
    freeze = percentile(calibrators["freeze"], raw["freeze"], raw["phase"])
    periodic = percentile(calibrators["periodic"], raw["periodic"], raw["phase"])
    recurrence = np.maximum(freeze, periodic)
    low_surprise = percentile(
        calibrators["low_surprise"], -raw["prefix"], raw["phase"]
    )
    overregularity = combine_overregularity(duration, recurrence, low_surprise)
    deviation = np.maximum(prefix, overregularity)
    prefix_dwell = causal_dwell(prefix, starts, lengths, threshold=0.9)
    overregularity_dwell = causal_dwell(
        overregularity, starts, lengths, threshold=0.9
    )
    return {
        "unigram_innovation": unigram.astype(np.float32),
        "clock_innovation": clock.astype(np.float32),
        "prefix_innovation": prefix.astype(np.float32),
        "ordered4_innovation": ordered.astype(np.float32),
        "bag4_innovation": bag.astype(np.float32),
        "prefix_persistent": (
            prefix * np.minimum(prefix_dwell / 3.0, 1.0)
        ).astype(np.float32),
        "overregularity": overregularity.astype(np.float32),
        "overregularity_persistent": (
            overregularity * np.minimum(overregularity_dwell / 3.0, 1.0)
        ).astype(np.float32),
        "duration": duration.astype(np.float32),
        "freeze": freeze.astype(np.float32),
        "periodic": periodic.astype(np.float32),
        "low_surprise": low_surprise.astype(np.float32),
        "deviation": deviation.astype(np.float32),
    }


def channel_scores(
    raw: dict[str, np.ndarray],
    starts: np.ndarray,
    lengths: np.ndarray,
    model: dict[str, Any],
) -> dict[str, np.ndarray]:
    output = channel_scores_without_return(
        raw, starts, lengths, model["calibrators"]
    )
    return_probability, deviation_dwell = model["return_model"].predict(
        output["deviation"], raw["phase_velocity"], starts, lengths
    )
    return_failure = output["deviation"] * (1.0 - return_probability)
    output["return_failure"] = return_failure.astype(np.float32)
    output["discrete_three_channel"] = np.maximum.reduce(
        [
            output["prefix_persistent"],
            output["overregularity_persistent"],
            output["return_failure"],
        ]
    ).astype(np.float32)
    output["return_probability"] = return_probability.astype(np.float32)
    output["deviation_dwell"] = deviation_dwell
    return output


def fit_fold(
    arrays: dict[str, np.ndarray],
    fold: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    base = np.asarray(arrays["full_base"], dtype=np.float32)
    starts = arrays["full_starts"]
    lengths = arrays["full_lengths"]
    state = arrays["full_state"]
    success = arrays["full_success"]
    episode_task = arrays["full_task"]
    progress = np.asarray(arrays["full_query_progress"], dtype=np.float32)
    fold_index = int(fold["fold"])

    density = select_episodes(state, np.asarray(fold["density_states"]), success)
    return_episodes = select_episodes(
        state, np.asarray(fold["return_states"]), success
    )
    calibration = select_episodes(
        state, np.asarray(fold["calibration_states"]), success
    )
    density = balance_episodes_by_task(density, episode_task, args.seed + fold_index)
    return_episodes = balance_episodes_by_task(
        return_episodes, episode_task, args.seed + 100 + fold_index
    )
    calibration = balance_episodes_by_task(
        calibration, episode_task, args.seed + 200 + fold_index
    )
    density_rows = balanced_rows(starts, lengths, density)
    calibration_rows = balanced_rows(starts, lengths, calibration)

    task = np.zeros(len(base), dtype=np.int8)
    scaler = TaskRobustScaler().fit(base, task, density_rows, 1)
    scaled = scaler.transform(base, task)
    tracks = phenotype_track_scores(scaled)
    tokenizer = TernaryChordTokenizer().fit(tracks, starts, lengths, density)
    chords = tokenizer.transform(tracks)
    hmm = FactorialCategoricalHMM(phase_states=args.phase_states).fit(
        chords, progress, starts, lengths, density
    )
    ordered = FactorialContextGrammar(
        max_order=args.max_order, ordered=True
    ).fit(chords, starts, lengths, density)
    bag = FactorialContextGrammar(max_order=args.max_order, ordered=False).fit(
        chords, starts, lengths, density
    )
    duration = FactorialDurationModel().fit(chords, starts, lengths, density)
    model: dict[str, Any] = {
        "scaler": scaler,
        "tokenizer": tokenizer,
        "hmm": hmm,
        "ordered": ordered,
        "bag": bag,
        "duration": duration,
        "density_states": np.asarray(fold["density_states"], dtype=np.int16),
        "return_states": np.asarray(fold["return_states"], dtype=np.int16),
        "calibration_states": np.asarray(fold["calibration_states"], dtype=np.int16),
        "test_states": np.asarray(fold["test_states"], dtype=np.int16),
    }
    raw = raw_scores(base, starts, lengths, model, args.device)
    zeros = np.zeros(len(base), dtype=np.float32)
    calibrators = {
        "unigram": fit_percentile(raw["unigram"], zeros, calibration_rows),
        "clock": fit_percentile(raw["clock"], raw["clock_phase"], calibration_rows),
        "prefix": fit_percentile(raw["prefix"], raw["phase"], calibration_rows),
        "ordered": fit_percentile(raw["ordered"], zeros, calibration_rows),
        "bag": fit_percentile(raw["bag"], zeros, calibration_rows),
        "duration": fit_percentile(raw["duration"], raw["phase"], calibration_rows),
        "freeze": fit_percentile(raw["freeze"], raw["phase"], calibration_rows),
        "periodic": fit_percentile(raw["periodic"], raw["phase"], calibration_rows),
        "low_surprise": fit_percentile(
            -raw["prefix"], raw["phase"], calibration_rows
        ),
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

    hysteretic_chords = tokenizer.transform(
        tracks, starts, lengths, hysteresis=True
    )
    hysteretic_hmm = FactorialCategoricalHMM(phase_states=args.phase_states).fit(
        hysteretic_chords, progress, starts, lengths, density
    )
    hysteretic_prefix = hysteretic_hmm.score(
        hysteretic_chords, starts, lengths, device=args.device
    )[0]
    hysteretic_clock = hysteretic_hmm.score(
        hysteretic_chords,
        starts,
        lengths,
        update_with_observations=False,
        device=args.device,
    )[0]
    extra = {
        "density_episodes": int(len(density)),
        "return_episodes": int(len(return_episodes)),
        "calibration_episodes": int(len(calibration)),
        "hysteresis_changed_fraction": float(np.mean(hysteretic_chords != chords)),
    }
    raw["hysteretic_prefix"] = hysteretic_prefix
    raw["hysteretic_clock"] = hysteretic_clock
    raw["density_episodes"] = density
    return model, raw, scores, extra


def physical_metrics(
    arrays: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    assigned_fold: np.ndarray,
    thresholds: dict[int, dict[str, float]],
    methods: tuple[str, ...],
) -> dict[str, Any]:
    starts = arrays["anchor_starts"]
    lengths = arrays["anchor_lengths"]
    state = arrays["anchor_state"]
    success = arrays["anchor_success"]
    onset = arrays["anchor_stasis_onset"]
    output: dict[str, Any] = {}
    for method in methods:
        false_alarms = success_total = before3 = by_onset = by_post3 = event_total = 0
        matched_wins = 0.0
        matched_pairs = 0
        leads: list[int] = []
        for episode_index in range(len(starts)):
            start = int(starts[episode_index])
            length = int(lengths[episode_index])
            threshold = thresholds[int(assigned_fold[episode_index])][method]
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
                if int(lengths[control]) <= event_onset:
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


def motif_indicators(
    chords: np.ndarray, starts: np.ndarray, lengths: np.ndarray
) -> dict[str, np.ndarray]:
    symbols = np.asarray(chords, dtype=np.int8)
    codes = chord_codes(symbols)
    output = {name: np.zeros(len(symbols), dtype=np.float32) for name in DIRECT_MOTIFS}
    for start, length in zip(starts, lengths, strict=True):
        start = int(start)
        stop = start + int(length)
        for row in range(start, stop):
            offset = row - start
            if offset >= 3:
                fl = symbols[row - 3 : row + 1, 0] == 0
                sy = symbols[row - 3 : row + 1, 1] == 2
                output["fl_fl_sy_sy"][row] = float(
                    fl[0] and fl[1] and sy[2] and sy[3]
                )
                output["fl_parallel_sy"][row] = float(
                    np.sum(fl) >= 3 and sy[2] and sy[3]
                )
                output["abab"][row] = float(
                    codes[row] == codes[row - 2]
                    and codes[row - 1] == codes[row - 3]
                )
            if offset >= 1:
                output["switching_pair"][row] = float(
                    np.all(symbols[row - 1 : row + 1, 3] == 2)
                )
            if offset >= 2:
                output["lock_streak"][row] = float(
                    np.all(symbols[row - 2 : row + 1, 3] == 0)
                )
    return output


def descriptive_language(
    raw: dict[str, np.ndarray], model: dict[str, Any]
) -> dict[str, Any]:
    chords = raw["chords"]
    starts = raw["starts"]
    lengths = raw["lengths"]
    density = raw["density_episodes"]
    rows = episode_uniform_rows(starts, lengths, density, samples_per_episode=12)
    codes = chord_codes(chords)
    unique, counts = np.unique(codes[rows], return_counts=True)
    order = np.argsort(counts)[::-1][:12]
    top_chords = []
    for index in order:
        code = int(unique[index])
        representative = chords[np.flatnonzero(codes == code)[0]]
        top_chords.append(
            {
                "code": code,
                "label": chord_label(representative),
                "fraction": float(counts[index] / counts.sum()),
            }
        )

    transitions: defaultdict[tuple[int, int], float] = defaultdict(float)
    for episode in density:
        start = int(starts[episode])
        length = int(lengths[episode])
        if length <= 1:
            continue
        weight = 1.0 / (length - 1)
        for row in range(start + 1, start + length):
            transitions[(int(codes[row - 1]), int(codes[row]))] += weight
    top_transitions = []
    for (left, right), weight in sorted(
        transitions.items(), key=lambda item: item[1], reverse=True
    )[:12]:
        left_row = np.flatnonzero(codes == left)[0]
        right_row = np.flatnonzero(codes == right)[0]
        top_transitions.append(
            {
                "left": chord_label(chords[left_row]),
                "right": chord_label(chords[right_row]),
                "episode_balanced_weight": float(weight),
            }
        )

    track_fractions = {}
    for track, name in enumerate(TRACK_NAMES):
        track_fractions[name] = {
            TRACK_SYMBOLS[track][state] or "normal": float(
                np.mean(chords[rows, track] == state)
            )
            for state in range(3)
        }
    assert model["hmm"].emission_ is not None
    phase_prototypes = []
    for phase_index, emission in enumerate(model["hmm"].emission_):
        active = []
        for track in range(len(TRACK_NAMES)):
            state = int(np.argmax(emission[track]))
            symbol = TRACK_SYMBOLS[track][state] or "normal"
            active.append(
                {
                    "track": TRACK_NAMES[track],
                    "symbol": symbol,
                    "probability": float(emission[track, state]),
                }
            )
        phase_prototypes.append({"phase": phase_index, "modal_tracks": active})
    return {
        "source": "fold-0 density-state successes only",
        "track_symbol_fractions": track_fractions,
        "unique_chords": int(len(unique)),
        "top_chords": top_chords,
        "top_transitions": top_transitions,
        "phase_prototypes": phase_prototypes,
    }


def render_report(summary: dict[str, Any]) -> str:
    next_query = summary["healthy_next_chord"]
    physical = summary["physical_stasis_matched_5pct_operating_point"]
    direct = summary["direct_motif_physical_stasis"]

    def pct(value: float) -> str:
        return f"{100.0 * value:.1f}%"

    lines = [
        "# 多轨离散 MoE 语法审计",
        "",
        "## 结论",
        "",
        (
            "在完全不带记忆的五轨三值 tokenizer 上，ordered-4 相对同一历史词袋的 "
            f"held-out next-symbol 增益为 {next_query['ordered_vs_bag_bits_per_track']:.4f} "
            "bits/track；全前缀 HMM 相对仅按内部时钟推进的增益为 "
            f"{next_query['prefix_vs_clock_bits_per_track']:.4f} bits/track。"
        ),
        "",
        (
            "滞回 tokenizer 的对应前缀增益为 "
            f"{next_query['hysteretic_prefix_vs_clock_bits_per_track']:.4f} bits/track。"
            "它只作为敏感性分析，因为滞回会把连续性写进字母本身。"
        ),
        "",
        "这两个量回答是否存在可泛化的离散顺序结构；物理 onset 表回答这种结构是否真的能提前发现错误。",
        "",
        (
            "物理结论是 no-go：固定 "
            f"{pct(physical['ordered4_innovation']['success_episode_fpr'])} success-episode FPR 时，"
            "ordered-4 在 q-3 的召回为 "
            f"{pct(physical['ordered4_innovation']['recall_onset_minus3'])}，"
            "全前缀 HMM 为 "
            f"{pct(physical['prefix_innovation']['recall_onset_minus3'])}。"
            "健康顺序可预测，不等于 stasis 是违反该顺序的句子。"
        ),
        "",
        (
            "预定义 `Fl Fl Sy Sy` 在 "
            f"{pct(direct['fl_fl_sy_sy']['success_episode_fpr'])} 的成功 episode 中也出现，"
            "state/query-matched AUC 仅 "
            f"{direct['fl_fl_sy_sy']['state_and_query_matched_auc']:.3f}；"
            "它是常见 routing motif，不是 Trap 特异规则。"
        ),
        "",
        "## 字母与和弦",
        "",
        "每个 query 同时发出五个三值轨道：`Fl/normal/Sh`、`Ds/normal/Sy`、"
        "`LR/normal/HR`、`Lk/normal/Sw`、`Ff/normal/Fs`。阈值只由 density-state "
        "成功 episode 的 20/80 分位数拟合，且每 episode 等权。主 tokenizer 无滞回。",
        "",
        "## Held-out 健康 next chord",
        "",
        "| model | nats / track |",
        "|---|---:|",
    ]
    for key, label in (
        ("unigram_nll_nats_per_track", "unigram / no history"),
        ("bag_nll_nats_per_track", "bag of last 4 chords"),
        ("ordered_nll_nats_per_track", "ordered last 4 chords"),
        ("clock_nll_nats_per_track", "latent clock only"),
        ("prefix_nll_nats_per_track", "full-prefix categorical HMM"),
    ):
        lines.append(f"| {label} | {next_query[key]:.4f} |")
    lines.extend(
        [
            "",
            "所有 headline NLL 均不含 `<END>`，模型在线不读取 task ID、绝对 query index、"
            "物理状态或 outcome。HMM 的训练相位由成功轨迹归一化进度初始化，因此 HMM 是"
            "全前缀滤波检验，不被解释为无监督发现的任务语义。",
        ]
    )
    if "sequence_stability" in summary:
        stability = summary["sequence_stability"]
        lines.extend(
            [
                "",
                "## 折间与 init-state 稳定性",
                "",
                "| contrast | gain bits/track | state-cluster 95% CI | positive states | sign-flip p |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for key, label in (
            ("ordered_vs_bag", "ordered-4 vs bag-4"),
            ("order4_vs_order1", "ordered-4 vs ordered-1"),
            ("prefix_vs_clock", "full prefix vs clock"),
            ("prefix_vs_last1", "full prefix vs clock + last chord"),
            ("ordered_vs_unigram", "ordered-4 vs unigram"),
        ):
            row = stability["clustered_inference"][key]
            interval = row["state_cluster_bootstrap_95ci"]
            lines.append(
                f"| {label} | {row['bits_per_track']:.4f} | "
                f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
                f"{pct(row['positive_state_fraction'])} | "
                f"{row['state_sign_flip_two_sided_p']:.4g} |"
            )
        lines.extend(
            [
                "",
                "置信区间以 held-out init state 为 cluster 重采样，而不是把同一 episode 的 query "
                "误当独立样本。",
            ]
        )
        curve = stability["context_order_curve_nats_per_track"]
        inference = stability["clustered_inference"]
        lines.extend(
            [
                "",
                "### 历史长度分解",
                "",
                "| maximum ordered context | nats/track | incremental gain bits/track |",
                "|---:|---:|---:|",
                f"| 1 | {curve['order_1']:.4f} | - |",
                (
                    f"| 2 | {curve['order_2']:.4f} | "
                    f"{inference['order2_vs_order1']['bits_per_track']:.5f} |"
                ),
                (
                    f"| 3 | {curve['order_3']:.4f} | "
                    f"{inference['order3_vs_order2']['bits_per_track']:.5f} |"
                ),
                (
                    f"| 4 | {curve['order_4']:.4f} | "
                    f"{inference['order4_vs_order3']['bits_per_track']:.5f} |"
                ),
                "",
                (
                    "绝大多数多步增益来自 q-2；q-3 较小，q-4 已接近零。更严格地，"
                    "full-prefix HMM 相对 clock+last-chord 的增益为 "
                    f"{inference['prefix_vs_last1']['bits_per_track']:.5f} bits/track，"
                    "即完整前缀在这个模型中反而更差。因此当前证据支持 2-3 query 的局部"
                    "离散句法，不支持长程全局句法。"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Scene8 物理 stasis（固定 pooled OOF success-episode FPR）",
            "",
            "| score | success FPR | recall q-3 | recall onset | recall onset+3 | matched AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in MAIN_METHODS:
        row = physical[method]
        lines.append(
            f"| {method} | {pct(row['success_episode_fpr'])} | "
            f"{pct(row['recall_onset_minus3'])} | {pct(row['recall_by_onset'])} | "
            f"{pct(row['recall_by_onset_plus3'])} | "
            f"{row['state_and_query_matched_auc']:.3f} |"
        )
    continuous = summary.get("continuous_v2_matched_fpr_comparison")
    if continuous:
        lines.extend(
            [
                "",
                "### 与连续 v2 的同 operating-point 对照",
                "",
                "| representation / score | recall q-3 | recall onset | matched AUC |",
                "|---|---:|---:|---:|",
            ]
        )
        comparison_rows = (
            ("discrete ordered-4", physical["ordered4_innovation"]),
            ("discrete prefix HMM", physical["prefix_innovation"]),
            ("discrete overregularity", physical["overregularity"]),
            ("continuous current query", continuous["current_innovation"]),
            ("continuous VAR(4)", continuous["var_innovation"]),
            ("continuous prefix HMM", continuous["hmm_innovation"]),
        )
        for label, row in comparison_rows:
            lines.append(
                f"| {label} | {pct(row['recall_onset_minus3'])} | "
                f"{pct(row['recall_by_onset'])} | "
                f"{row['state_and_query_matched_auc']:.3f} |"
            )
    prefix_ablation = summary.get("prefix_history_ablation")
    if prefix_ablation:
        last = prefix_ablation["last1_physical_stasis"]
        full = prefix_ablation["full_prefix_physical_stasis"]
        lines.extend(
            [
                "",
                "### 全前缀的物理增量",
                "",
                "| history | success FPR | recall q-3 | recall onset | matched AUC |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| clock + last chord | {pct(last['success_episode_fpr'])} | "
                    f"{pct(last['recall_onset_minus3'])} | {pct(last['recall_by_onset'])} | "
                    f"{last['state_and_query_matched_auc']:.3f} |"
                ),
                (
                    f"| complete prefix | {pct(full['success_episode_fpr'])} | "
                    f"{pct(full['recall_onset_minus3'])} | {pct(full['recall_by_onset'])} | "
                    f"{full['state_and_query_matched_auc']:.3f} |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "该阈值只用于同 FPR 比较。可部署的跨 init-state calibration 结果保存在 "
            "`summary.json` 的 `physical_stasis_anchor_calibrated`。",
            "",
            "## 直接检验 `Fl Fl Sy Sy`",
            "",
            "| predefined motif | success episode rate | recall q-3 | recall onset | matched AUC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for motif in DIRECT_MOTIFS:
        row = direct[motif]
        lines.append(
            f"| {motif} | {pct(row['success_episode_fpr'])} | "
            f"{pct(row['recall_onset_minus3'])} | {pct(row['recall_by_onset'])} | "
            f"{row['state_and_query_matched_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`fl_fl_sy_sy` 要求四个连续 query 中前两次为 Fl、后两次为 Sy，允许并发；"
            "`fl_parallel_sy` 更接近 static 假设，要求四次中至少三次 Fl 且末两次 Sy。"
            "这些规则在看结果前固定，没有从 stasis 标签调参。",
            "",
            "## 语言样本",
            "",
            f"fold-0 健康训练子集出现 {summary['descriptive_language']['unique_chords']} 个和弦。"
            "最高频和弦为：",
            "",
        ]
    )
    for item in summary["descriptive_language"]["top_chords"][:8]:
        lines.append(f"- `{item['label']}`: {pct(item['fraction'])}")
    lines.extend(
        [
            "",
            "最高权重的 episode-balanced 转移为：",
            "",
        ]
    )
    for item in summary["descriptive_language"]["top_transitions"][:8]:
        lines.append(
            f"- `{item['left']} -> {item['right']}`: "
            f"weight {item['episode_balanced_weight']:.2f}"
        )
    lines.extend(
        [
            "",
            "## 判定边界",
            "",
            "离散顺序增益若为正，只能证明健康和弦的排列不是词袋；是否可称为 Trap grammar，"
            "仍取决于 prefix/ordered、duration、periodicity 和 return 通道能否在相同 episode FPR "
            "下稳定超过无历史分数及连续 VAR/HMM。Scene8 目前仍只有可靠 stasis onset，没有可靠 loop onset。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.render_only:
        summary_path = args.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        (args.output_dir / "REPORT.zh.md").write_text(
            render_report(summary), encoding="utf-8"
        )
        print(f"regenerated {args.output_dir / 'REPORT.zh.md'}", flush=True)
        return
    started = time.time()
    arrays, metadata = load_dataset(args.dataset)
    if metadata.get("schema_version") != 2:
        raise ValueError("discrete grammar audit requires corrected v2 phenotypes")
    fold_summary = json.loads(args.fold_summary.read_text(encoding="utf-8"))
    folds = fold_summary["folds"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)

    anchor_size = len(arrays["anchor_base"])
    anchor_scores = {
        method: np.full(anchor_size, np.nan, dtype=np.float32)
        for method in SCORED_METHODS
    }
    anchor_chords = np.full((anchor_size, len(TRACK_NAMES)), -1, dtype=np.int8)
    assigned_fold = np.full(len(arrays["anchor_starts"]), -1, dtype=np.int8)
    global_thresholds: dict[int, dict[str, float]] = {}
    anchor_thresholds: dict[int, dict[str, float]] = {}
    next_records: list[dict[str, float]] = []
    return_predictions: list[float] = []
    return_targets: list[bool] = []
    persistent_scores: list[float] = []
    deviation_scores: list[float] = []
    fold_diagnostics: list[dict[str, Any]] = []
    language: dict[str, Any] | None = None

    full_starts = arrays["full_starts"]
    full_lengths = arrays["full_lengths"]
    full_state = arrays["full_state"]
    full_success = arrays["full_success"]
    for fold in folds:
        fold_index = int(fold["fold"])
        print(f"fold {fold_index}: fitting discrete chords and grammars", flush=True)
        model, raw, scores, diagnostics = fit_fold(arrays, fold, args)
        raw["starts"] = full_starts
        raw["lengths"] = full_lengths
        if fold_index == 0:
            language = descriptive_language(raw, model)
        joblib.dump(
            model,
            args.output_dir / "models" / f"fold_{fold_index}.joblib",
            compress=3,
        )

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
        record = {
            "rows": float(len(test_rows)),
            "unigram": float(raw["unigram"][test_rows].sum()),
            "clock": float(raw["clock"][test_rows].sum()),
            "prefix": float(raw["prefix"][test_rows].sum()),
            "ordered": float(raw["ordered"][test_rows].sum()),
            "bag": float(raw["bag"][test_rows].sum()),
            "hysteretic_prefix": float(raw["hysteretic_prefix"][test_rows].sum()),
            "hysteretic_clock": float(raw["hysteretic_clock"][test_rows].sum()),
            "ordered_depth": float(raw["ordered_depth"][test_rows].sum()),
            "bag_depth": float(raw["bag_depth"][test_rows].sum()),
        }
        next_records.append(record)
        diagnostics.update(
            {
                "fold": fold_index,
                "test_rows": len(test_rows),
                "mean_ordered_context_depth": record["ordered_depth"] / len(test_rows),
                "mean_bag_context_depth": record["bag_depth"] / len(test_rows),
            }
        )
        fold_diagnostics.append(diagnostics)

        for episode_index in test_success:
            start = int(full_starts[episode_index])
            length = int(full_lengths[episode_index])
            for offset in range(max(0, length - args.return_horizon)):
                row = start + offset
                if scores["deviation"][row] < 0.8:
                    continue
                future = scores["deviation"][
                    row + 1 : row + args.return_horizon + 1
                ]
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
            anchor_scores[method][selected_rows] = current_anchor_scores[method][
                selected_rows
            ]
        anchor_chords[selected_rows] = anchor_raw["chords"][selected_rows]
        assigned_fold[selected_episodes] = fold_index

    if language is None:
        raise AssertionError("no descriptive language was produced")
    if np.any(assigned_fold < 0) or np.any(anchor_chords < 0):
        raise AssertionError("anchor fold assignment is incomplete")
    if any(not np.isfinite(values).all() for values in anchor_scores.values()):
        raise AssertionError("anchor scores contain missing fold predictions")

    total_rows = int(sum(record["rows"] for record in next_records))
    totals = {
        key: sum(record[key] for record in next_records)
        for key in (
            "unigram",
            "clock",
            "prefix",
            "ordered",
            "bag",
            "hysteretic_prefix",
            "hysteretic_clock",
        )
    }
    healthy_next = {
        "balanced_test_rows": total_rows,
        **{
            f"{key}_nll_nats_per_track": value / total_rows
            for key, value in totals.items()
        },
        "prefix_vs_clock_bits_per_track": (
            (totals["clock"] - totals["prefix"]) / total_rows / np.log(2.0)
        ),
        "ordered_vs_bag_bits_per_track": (
            (totals["bag"] - totals["ordered"]) / total_rows / np.log(2.0)
        ),
        "ordered_vs_unigram_bits_per_track": (
            (totals["unigram"] - totals["ordered"]) / total_rows / np.log(2.0)
        ),
        "hysteretic_prefix_vs_clock_bits_per_track": (
            (totals["hysteretic_clock"] - totals["hysteretic_prefix"])
            / total_rows
            / np.log(2.0)
        ),
    }

    returned = np.asarray(return_targets, dtype=bool)
    return_probability = np.asarray(return_predictions)
    persistent_score = np.asarray(persistent_scores)
    deviation_score = np.asarray(deviation_scores)
    return_evaluation = {
        "rows": int(len(returned)),
        "returned_fraction": float(returned.mean()),
        "return_brier": float(np.square(return_probability - returned).mean()),
        "persistent_auc": float(roc_auc_score(~returned, persistent_score)),
        "deviation_only_persistent_auc": float(
            roc_auc_score(~returned, deviation_score)
        ),
    }

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
    pooled_by_fold = {int(fold["fold"]): pooled_thresholds for fold in folds}
    physical_matched = physical_metrics(
        arrays, anchor_scores, assigned_fold, pooled_by_fold, SCORED_METHODS
    )
    physical_anchor = physical_metrics(
        arrays, anchor_scores, assigned_fold, anchor_thresholds, SCORED_METHODS
    )
    physical_transfer = physical_metrics(
        arrays, anchor_scores, assigned_fold, global_thresholds, SCORED_METHODS
    )

    motif_scores = motif_indicators(
        anchor_chords, arrays["anchor_starts"], arrays["anchor_lengths"]
    )
    motif_thresholds = {
        int(fold["fold"]): {motif: 0.5 for motif in DIRECT_MOTIFS}
        for fold in folds
    }
    direct_motif_metrics = physical_metrics(
        arrays, motif_scores, assigned_fold, motif_thresholds, DIRECT_MOTIFS
    )

    continuous_comparison = None
    if args.continuous_summary.exists():
        continuous = json.loads(args.continuous_summary.read_text(encoding="utf-8"))
        continuous_physical = continuous["physical_stasis_matched_5pct_operating_point"]
        continuous_comparison = {
            method: continuous_physical[method]
            for method in (
                "current_innovation",
                "var_innovation",
                "hmm_innovation",
                "overregularity",
                "three_channel",
            )
        }

    summary = {
        "schema_version": 4,
        "protocol": {
            "tokenizer": (
                "five simultaneous ternary healthy-quantile tracks; stateless primary"
            ),
            "tracks": list(TRACK_NAMES),
            "threshold_quantiles": [0.2, 0.8],
            "inference_inputs": (
                "MoE routing prefix only; no task id, absolute query index, physical state, or outcome"
            ),
            "sequence_models": [
                "no-history factorial unigram",
                "bag of last four complete chords",
                "ordered last four complete chords",
                "12-state full-prefix factorial categorical HMM",
            ],
            "headline_end_token": False,
            "density_sampling": "equal task count and equal episode/phase contribution",
            "state_split": "five-fold disjoint init states",
            "physical_anchor": metadata["anchor_definition"],
            "matched_fpr_note": (
                "pooled OOF Scene8 successes set the descriptive common operating point"
            ),
        },
        "counts": {
            "full_episodes": len(arrays["full_starts"]),
            "full_queries": len(arrays["full_base"]),
            "anchor_episodes": len(arrays["anchor_starts"]),
            "anchor_stasis": int(np.sum(arrays["anchor_stasis_onset"] >= 0)),
        },
        "healthy_next_chord": healthy_next,
        "healthy_excursion_return": return_evaluation,
        "physical_stasis_matched_5pct_operating_point": physical_matched,
        "physical_stasis_anchor_calibrated": physical_anchor,
        "physical_stasis_global_threshold_transfer": physical_transfer,
        "direct_motif_physical_stasis": direct_motif_metrics,
        "continuous_v2_matched_fpr_comparison": continuous_comparison,
        "descriptive_language": language,
        "fold_diagnostics": fold_diagnostics,
        "matched_thresholds": pooled_thresholds,
        "anchor_thresholds_by_fold": anchor_thresholds,
        "global_thresholds_by_fold": global_thresholds,
        "elapsed_seconds": time.time() - started,
    }
    np.savez_compressed(
        args.output_dir / "anchor_scores.npz",
        **anchor_scores,
        **{f"motif_{key}": value for key, value in motif_scores.items()},
        anchor_chords=anchor_chords,
        assigned_fold=assigned_fold,
        anchor_starts=arrays["anchor_starts"],
        anchor_lengths=arrays["anchor_lengths"],
        anchor_state=arrays["anchor_state"],
        anchor_success=arrays["anchor_success"],
        anchor_stasis_onset=arrays["anchor_stasis_onset"],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.zh.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(f"wrote {args.output_dir} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
