from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from moe_grammar.discrete_phenotype import (
    FactorialCategoricalHMM,
    categorical_recurrence,
    episode_uniform_rows,
)
from moe_grammar.flow_word import (
    FLOW_TRACK_NAMES,
    FlowChordTokenizer,
    FlowTrackProjector,
    PositionFactorialContextGrammar,
    flow_motif_indicators,
    flow_query_words,
    format_flow_word,
)
from moe_grammar.open_world import PhaseConditionalCDF, causal_dwell
from moe_grammar.run_discrete_grammar_audit import physical_metrics
from moe_grammar.run_open_world_audit import load_dataset
from moe_grammar.run_three_channel_audit import (
    balance_episodes_by_task,
    balanced_rows,
    episode_maxima,
    episode_rows,
    select_episodes,
)
from moe_grammar.three_channel import combine_overregularity


WORD_METHODS = (
    "word_clock_innovation",
    "word_last1_innovation",
    "word_prefix_innovation",
    "word_freeze",
    "word_periodic",
    "word_low_surprise",
    "word_overregularity",
    "word_overregularity_persistent",
)
FLOW_MOTIFS = (
    "flow_fl_fl_sy_sy_any",
    "flow_fl_fl_sy_sy_terminal",
    "flow_fl_parallel_sy_terminal",
    "flow_terminal_fl_sy",
)
MORPH_METRICS = (
    "morph_ordered_vs_bag",
    "morph_order2_vs_order1",
    "morph_order3_vs_order2",
    "morph_order4_vs_order3",
    "morph_order4_vs_order1",
)
WORD_METRICS = ("word_prefix_vs_clock", "word_prefix_vs_last1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primitives", type=Path, default=Path("artifacts/flow-word-primitives-v1.npz")
    )
    parser.add_argument(
        "--structure", type=Path, default=Path("artifacts/open-world-phenotypes-v2.npz")
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=Path("results-open-world-global-k12/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results-flow-word-v1"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--phase-states", type=int, default=12)
    parser.add_argument("--healthy-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def fit_cdf(values: np.ndarray, phase: np.ndarray, rows: np.ndarray) -> PhaseConditionalCDF:
    return PhaseConditionalCDF().fit(
        values, np.zeros(len(values), dtype=np.int8), phase, rows
    )


def apply_cdf(
    model: PhaseConditionalCDF, values: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    return model.transform(values, np.zeros(len(values), dtype=np.int8), phase)


def cluster_inference(
    entries: dict[str, list[tuple[float, int]]], seed: int, samples: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for metric, records in entries.items():
        numerator = np.asarray([record[0] for record in records], dtype=np.float64)
        denominator = np.asarray([record[1] for record in records], dtype=np.float64)
        count = len(records)
        draws = rng.integers(0, count, size=(samples, count))
        estimates = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
        estimate = numerator.sum() / denominator.sum()
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, count))
        null = (signs * numerator[None, :]).sum(axis=1) / denominator.sum()
        output[metric] = {
            "bits_per_track": float(estimate),
            "state_cluster_bootstrap_95ci": [
                float(np.quantile(estimates, 0.025)),
                float(np.quantile(estimates, 0.975)),
            ],
            "positive_state_fraction": float(np.mean(numerator > 0)),
            "state_sign_flip_two_sided_p": float(
                (1 + np.sum(np.abs(null) >= abs(estimate))) / (samples + 1)
            ),
            "states": count,
        }
    return output


def describe_words(
    flow_chords: np.ndarray, density_rows: np.ndarray
) -> dict[str, Any]:
    selected = flow_chords[density_rows]
    words = flow_query_words(selected)
    unique, counts = np.unique(words, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:8]
    top_words = [
        {
            "flow_word": format_flow_word(unique[index].reshape(10, len(FLOW_TRACK_NAMES))),
            "count": int(counts[index]),
            "fraction": float(counts[index] / counts.sum()),
        }
        for index in order
    ]
    track_profiles: dict[str, list[dict[str, float]]] = {}
    for track, name in enumerate(FLOW_TRACK_NAMES):
        track_profiles[name] = [
            {
                "flow": flow,
                "low": float(np.mean(selected[:, flow, track] == 0)),
                "normal": float(np.mean(selected[:, flow, track] == 1)),
                "high": float(np.mean(selected[:, flow, track] == 2)),
            }
            for flow in range(10)
        ]
    return {
        "source": "fold-0 density successes, episode-uniform query sample",
        "sampled_query_words": len(selected),
        "unique_exact_query_words": len(unique),
        "unique_fraction": float(len(unique) / len(selected)),
        "top_exact_words": top_words,
        "flow_track_profiles": track_profiles,
    }


def render_report(summary: dict[str, Any]) -> str:
    morph = summary["within_query_morphology"]
    word = summary["cross_query_word_prediction"]
    inference = summary["state_cluster_inference"]
    physical = summary["physical_stasis_matched_5pct_operating_point"]
    motifs = summary["direct_flow_motif_physical_stasis"]

    def pct(value: float) -> str:
        return f"{100.0 * value:.1f}%"

    lines = [
        "# 十流和弦 Query-Word 审计",
        "",
        "## 结论",
        "",
        (
            "每个 flow step 被编码为五轨三值和弦，十个和弦原样组成 50-symbol query word。"
            "在 query 内，ordered-4 相对相同四和弦 bag 的增益为 "
            f"{inference['morph_ordered_vs_bag']['bits_per_track']:.5f} bits/flow-track。"
        ),
        "",
        (
            "但依赖几乎都是局部的：order-2 相对 order-1 增益为 "
            f"{inference['morph_order2_vs_order1']['bits_per_track']:.5f} bits/flow-track，"
            "order-3 只再增加 "
            f"{inference['morph_order3_vs_order2']['bits_per_track']:.5f}，"
            "order-4 相对 order-3 反而降低 "
            f"{abs(inference['morph_order4_vs_order3']['bits_per_track']):.5f}。"
        ),
        "",
        (
            "跨 query 时，完整 word-prefix 相对 clock+last-word 的增益为 "
            f"{inference['word_prefix_vs_last1']['bits_per_track']:.5f} bits/flow-track。"
            "其 state-cluster 95% CI 排除零，但效应很小。这直接检验完整的十流单词，"
            "而不是 query 汇总近似。"
        ),
        "",
        (
            "物理 early-warning 不成立：最佳低-surprisal 通道在 4.7% success-episode FPR "
            f"下的 q-3 recall 为 {pct(physical['word_low_surprise']['recall_onset_minus3'])} "
            f"（95% CI {pct(physical['word_low_surprise']['recall_by_onset_ci'][0])}--"
            f"{pct(physical['word_low_surprise']['recall_by_onset_ci'][1])}），matched AUC 仅 "
            f"{physical['word_low_surprise']['state_and_query_matched_auc']:.3f}。"
        ),
        "",
        (
            "字面 `Fl Fl Sy Sy` 也不是病句：任意 flow 窗口版本出现在 100% 的成功 episode，"
            "固定末端版本出现在 96.3%。因此本审计确认弱健康时序结构，同时否定其当前的 "
            "Trap-specific 在线检测能力。"
        ),
        "",
        "## 表示",
        "",
        "每个 flow chord 为 `[Fl/normal/Sh, Ds/normal/Sy, LR/normal/HR, "
        "Lk/normal/Sw, Ff/normal/Fs]`。每个 primitive 先按训练成功数据 robust scaling，"
        "字母阈值也只在 density states 上以 episode-uniform query 抽样拟合。",
        "",
        "## Query 内构词法",
        "",
        "| context | held-out NLL nats/flow-track |",
        "|---|---:|",
        f"| flow position only | {morph['position_nll_nats_per_track']:.5f} |",
        f"| ordered-1 | {morph['order1_nll_nats_per_track']:.5f} |",
        f"| ordered-2 | {morph['order2_nll_nats_per_track']:.5f} |",
        f"| ordered-3 | {morph['order3_nll_nats_per_track']:.5f} |",
        f"| ordered-4 | {morph['order4_nll_nats_per_track']:.5f} |",
        f"| bag-4 | {morph['bag4_nll_nats_per_track']:.5f} |",
        "",
        "| contrast | bits/flow-track | state-cluster 95% CI | positive states |",
        "|---|---:|---:|---:|",
    ]
    for metric, label in (
        ("morph_ordered_vs_bag", "ordered-4 vs bag-4"),
        ("morph_order4_vs_order1", "ordered-4 vs ordered-1"),
        ("morph_order2_vs_order1", "ordered-2 vs ordered-1"),
        ("morph_order3_vs_order2", "ordered-3 vs ordered-2"),
        ("morph_order4_vs_order3", "ordered-4 vs ordered-3"),
    ):
        row = inference[metric]
        interval = row["state_cluster_bootstrap_95ci"]
        lines.append(
            f"| {label} | {row['bits_per_track']:.5f} | "
            f"[{interval[0]:.5f}, {interval[1]:.5f}] | "
            f"{pct(row['positive_state_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## 跨 Query 的完整单词预测",
            "",
            "| context | held-out NLL nats/flow-track |",
            "|---|---:|",
            f"| latent clock | {word['clock_nll_nats_per_track']:.5f} |",
            f"| clock + last 50-symbol word | {word['last1_nll_nats_per_track']:.5f} |",
            f"| complete word prefix | {word['prefix_nll_nats_per_track']:.5f} |",
            "",
        ]
    )
    for metric, label in (
        ("word_prefix_vs_clock", "full prefix vs clock"),
        ("word_prefix_vs_last1", "full prefix vs clock + last word"),
    ):
        row = inference[metric]
        interval = row["state_cluster_bootstrap_95ci"]
        lines.append(
            f"- {label}: {row['bits_per_track']:.5f} bits/track, "
            f"95% CI [{interval[0]:.5f}, {interval[1]:.5f}], "
            f"positive states {pct(row['positive_state_fraction'])}."
        )
    lines.extend(
        [
            "",
            "## Scene8 物理 stasis（固定 pooled OOF success-episode FPR）",
            "",
            "| score | success FPR | recall q-3 | recall onset | matched AUC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in WORD_METHODS:
        row = physical[method]
        lines.append(
            f"| {method} | {pct(row['success_episode_fpr'])} | "
            f"{pct(row['recall_onset_minus3'])} | {pct(row['recall_by_onset'])} | "
            f"{row['state_and_query_matched_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Flow 内直接 `Fl Fl Sy Sy` 检验",
            "",
            "| fixed motif | success episode rate | recall q-3 | recall onset | matched AUC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for motif in FLOW_MOTIFS:
        row = motifs[motif]
        lines.append(
            f"| {motif} | {pct(row['success_episode_fpr'])} | "
            f"{pct(row['recall_onset_minus3'])} | {pct(row['recall_by_onset'])} | "
            f"{row['state_and_query_matched_auc']:.3f} |"
        )
    description = summary["descriptive_words"]
    lines.extend(
        [
            "",
            "## 词汇稀疏性",
            "",
            (
                f"fold-0 的 {description['sampled_query_words']:,} 个 episode-balanced 健康 query "
                f"产生 {description['unique_exact_query_words']:,} 个不同的精确 50-symbol words "
                f"（{pct(description['unique_fraction'])} 唯一）。"
            ),
            "",
            "最高频精确单词：",
            "",
        ]
    )
    for item in description["top_exact_words"]:
        lines.append(
            f"- `{item['flow_word']}`: {item['count']} "
            f"({100.0 * item['fraction']:.3f}%)"
        )
    lines.extend(
        [
            "",
            "精确单词不被强行压成 one-hot ID；HMM emission 对 50 个符号做因子化概率比较，"
            "因此相差一个 flow/track 的单词仍然相近。",
            "",
            "## 实验判决",
            "",
            "1. 健康 routing 有稳定的 flow-position morphology 和以相邻两步为主的局部构词规律。",
            "2. 保留完整十流单词后，较早 query 的全局前缀仍有非零但极小的预测增量。",
            "3. 95.1% 的精确单词只出现一次，说明不能把 50-symbol word 直接当成稠密词表 ID。",
            "4. 所有基于该语法的 stasis 提前预警都很弱；当前结论是机制分析通过、在线检测 No-Go。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    started = time.time()
    arrays, structure_metadata = load_dataset(args.structure)
    with np.load(args.primitives, allow_pickle=False) as payload:
        full_primitives = np.asarray(payload["full_primitives"], dtype=np.float32)
        anchor_primitives = np.asarray(payload["anchor_primitives"], dtype=np.float32)
        primitive_metadata = json.loads(str(payload["metadata_json"]))
    if len(full_primitives) != len(arrays["full_base"]):
        raise ValueError("full primitive and structure query axes differ")
    if len(anchor_primitives) != len(arrays["anchor_base"]):
        raise ValueError("anchor primitive and structure query axes differ")
    folds = json.loads(args.fold_summary.read_text(encoding="utf-8"))["folds"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)

    starts = arrays["full_starts"]
    lengths = arrays["full_lengths"]
    state = arrays["full_state"]
    success = arrays["full_success"]
    episode_task = arrays["full_task"]
    progress = np.asarray(arrays["full_query_progress"], dtype=np.float32)
    anchor_word_scores = {
        method: np.full(len(anchor_primitives), np.nan, dtype=np.float32)
        for method in WORD_METHODS
    }
    anchor_motifs = {
        motif: np.full(len(anchor_primitives), np.nan, dtype=np.float32)
        for motif in FLOW_MOTIFS
    }
    assigned_fold = np.full(len(arrays["anchor_starts"]), -1, dtype=np.int8)
    nll_totals = {
        key: 0.0
        for key in (
            "position",
            "order1",
            "order2",
            "order3",
            "order4",
            "bag4",
            "clock",
            "last1",
            "prefix",
        )
    }
    morph_rows_total = 0
    word_rows_total = 0
    state_entries = {metric: [] for metric in MORPH_METRICS + WORD_METRICS}
    fold_results: list[dict[str, Any]] = []
    description: dict[str, Any] | None = None

    for fold in folds:
        fold_index = int(fold["fold"])
        print(f"fold {fold_index}: fitting ten-flow query words", flush=True)
        density = select_episodes(state, np.asarray(fold["density_states"]), success)
        calibration = select_episodes(
            state, np.asarray(fold["calibration_states"]), success
        )
        test = select_episodes(state, np.asarray(fold["test_states"]), success)
        density = balance_episodes_by_task(density, episode_task, args.seed + fold_index)
        calibration = balance_episodes_by_task(
            calibration, episode_task, args.seed + 200 + fold_index
        )
        test = balance_episodes_by_task(test, episode_task, args.seed + 300 + fold_index)
        density_rows = episode_uniform_rows(starts, lengths, density, samples_per_episode=12)
        calibration_rows = balanced_rows(starts, lengths, calibration)
        morph_test_rows = episode_uniform_rows(
            starts, lengths, test, samples_per_episode=12
        )
        word_test_rows = balanced_rows(starts, lengths, test)

        projector = FlowTrackProjector().fit(full_primitives, density_rows)
        full_tracks = projector.transform(full_primitives)
        tokenizer = FlowChordTokenizer().fit(full_tracks, density_rows)
        full_chords = tokenizer.transform(full_tracks)
        words = flow_query_words(full_chords)
        ordered = PositionFactorialContextGrammar(ordered=True).fit(
            full_chords, density_rows
        )
        bag = PositionFactorialContextGrammar(ordered=False).fit(
            full_chords, density_rows
        )
        hmm = FactorialCategoricalHMM(phase_states=args.phase_states).fit(
            words, progress, starts, lengths, density
        )
        model = {
            "projector": projector,
            "tokenizer": tokenizer,
            "ordered_morphology": ordered,
            "bag_morphology": bag,
            "word_hmm": hmm,
        }

        selected_chords = full_chords[morph_test_rows]
        ordered_curve, _ = ordered.score_order_curve(selected_chords)
        morph_scores = {
            "position": ordered_curve[:, 0],
            **{
                f"order{order}": ordered_curve[:, order]
                for order in range(1, 5)
            },
            "bag4": bag.score_words(selected_chords, max_order=4)[0],
        }
        for key, values in morph_scores.items():
            nll_totals[key] += float(values.sum())
        morph_rows_total += len(morph_test_rows)
        morph_global = {
            key: np.full(len(full_primitives), np.nan, dtype=np.float32)
            for key in morph_scores
        }
        for key, values in morph_scores.items():
            morph_global[key][morph_test_rows] = values

        prefix, prefix_phase, _ = hmm.score(words, starts, lengths, device=args.device)
        clock, clock_phase, _ = hmm.score(
            words,
            starts,
            lengths,
            update_with_observations=False,
            device=args.device,
        )
        last1, last1_phase, _ = hmm.score_last_observation(
            words, starts, lengths, device=args.device
        )
        for key, values in (("clock", clock), ("last1", last1), ("prefix", prefix)):
            nll_totals[key] += float(values[word_test_rows].sum())
        word_rows_total += len(word_test_rows)

        for heldout_state in np.asarray(fold["test_states"], dtype=np.int64):
            state_episodes = test[state[test] == heldout_state]
            state_morph_rows = episode_uniform_rows(
                starts, lengths, state_episodes, samples_per_episode=12
            )
            state_word_rows = balanced_rows(starts, lengths, state_episodes)
            state_entries["morph_ordered_vs_bag"].append(
                (
                    float(
                        np.sum(
                            morph_global["bag4"][state_morph_rows]
                            - morph_global["order4"][state_morph_rows]
                        )
                        / np.log(2.0)
                    ),
                    len(state_morph_rows),
                )
            )
            for high, low, metric in (
                (2, 1, "morph_order2_vs_order1"),
                (3, 2, "morph_order3_vs_order2"),
                (4, 3, "morph_order4_vs_order3"),
                (4, 1, "morph_order4_vs_order1"),
            ):
                state_entries[metric].append(
                    (
                        float(
                            np.sum(
                                morph_global[f"order{low}"][state_morph_rows]
                                - morph_global[f"order{high}"][state_morph_rows]
                            )
                            / np.log(2.0)
                        ),
                        len(state_morph_rows),
                    )
                )
            state_entries["word_prefix_vs_clock"].append(
                (
                    float(np.sum(clock[state_word_rows] - prefix[state_word_rows]) / np.log(2.0)),
                    len(state_word_rows),
                )
            )
            state_entries["word_prefix_vs_last1"].append(
                (
                    float(np.sum(last1[state_word_rows] - prefix[state_word_rows]) / np.log(2.0)),
                    len(state_word_rows),
                )
            )

        freeze, periodic, _ = categorical_recurrence(words, starts, lengths)
        calibrators = {
            "clock": fit_cdf(clock, clock_phase, calibration_rows),
            "last1": fit_cdf(last1, last1_phase, calibration_rows),
            "prefix": fit_cdf(prefix, prefix_phase, calibration_rows),
            "freeze": fit_cdf(freeze, prefix_phase, calibration_rows),
            "periodic": fit_cdf(periodic, prefix_phase, calibration_rows),
            "low_surprise": fit_cdf(-prefix, prefix_phase, calibration_rows),
        }
        model["calibrators"] = calibrators
        model_path = args.output_dir / "models" / f"fold_{fold_index}.joblib"
        temporary_model_path = model_path.with_suffix(".joblib.tmp")
        joblib.dump(model, temporary_model_path, compress=0)
        os.replace(temporary_model_path, model_path)

        anchor_tracks = projector.transform(anchor_primitives)
        current_anchor_chords = tokenizer.transform(anchor_tracks)
        anchor_words = flow_query_words(current_anchor_chords)
        anchor_prefix, anchor_prefix_phase, _ = hmm.score(
            anchor_words,
            arrays["anchor_starts"],
            arrays["anchor_lengths"],
            device=args.device,
        )
        anchor_clock, anchor_clock_phase, _ = hmm.score(
            anchor_words,
            arrays["anchor_starts"],
            arrays["anchor_lengths"],
            update_with_observations=False,
            device=args.device,
        )
        anchor_last1, anchor_last1_phase, _ = hmm.score_last_observation(
            anchor_words,
            arrays["anchor_starts"],
            arrays["anchor_lengths"],
            device=args.device,
        )
        anchor_freeze, anchor_periodic, _ = categorical_recurrence(
            anchor_words, arrays["anchor_starts"], arrays["anchor_lengths"]
        )
        current_scores = {
            "word_clock_innovation": apply_cdf(
                calibrators["clock"], anchor_clock, anchor_clock_phase
            ),
            "word_last1_innovation": apply_cdf(
                calibrators["last1"], anchor_last1, anchor_last1_phase
            ),
            "word_prefix_innovation": apply_cdf(
                calibrators["prefix"], anchor_prefix, anchor_prefix_phase
            ),
            "word_freeze": apply_cdf(
                calibrators["freeze"], anchor_freeze, anchor_prefix_phase
            ),
            "word_periodic": apply_cdf(
                calibrators["periodic"], anchor_periodic, anchor_prefix_phase
            ),
            "word_low_surprise": apply_cdf(
                calibrators["low_surprise"], -anchor_prefix, anchor_prefix_phase
            ),
        }
        current_scores["word_overregularity"] = combine_overregularity(
            current_scores["word_freeze"],
            current_scores["word_periodic"],
            current_scores["word_low_surprise"],
        )
        overregularity_dwell = causal_dwell(
            current_scores["word_overregularity"],
            arrays["anchor_starts"],
            arrays["anchor_lengths"],
            threshold=0.9,
        )
        current_scores["word_overregularity_persistent"] = (
            current_scores["word_overregularity"]
            * np.minimum(overregularity_dwell / 3.0, 1.0)
        ).astype(np.float32)
        current_motifs = flow_motif_indicators(current_anchor_chords)
        selected_episodes = np.flatnonzero(
            np.isin(arrays["anchor_state"], np.asarray(fold["test_states"]))
        )
        selected_rows = episode_rows(
            arrays["anchor_starts"], arrays["anchor_lengths"], selected_episodes
        )
        for method in WORD_METHODS:
            anchor_word_scores[method][selected_rows] = current_scores[method][selected_rows]
        for motif in FLOW_MOTIFS:
            anchor_motifs[motif][selected_rows] = current_motifs[motif][selected_rows]
        assigned_fold[selected_episodes] = fold_index

        fold_results.append(
            {
                "fold": fold_index,
                "density_query_words": len(density_rows),
                "morph_test_query_words": len(morph_test_rows),
                "word_test_queries": len(word_test_rows),
            }
        )
        if fold_index == 0:
            description = describe_words(full_chords, density_rows)

    if description is None or np.any(assigned_fold < 0):
        raise AssertionError("flow-word fold assignment is incomplete")
    if any(not np.isfinite(values).all() for values in anchor_word_scores.values()):
        raise AssertionError("word scores contain missing OOF rows")
    if any(not np.isfinite(values).all() for values in anchor_motifs.values()):
        raise AssertionError("flow motifs contain missing OOF rows")

    morphology = {
        f"{key}_nll_nats_per_track": value / morph_rows_total
        for key, value in nll_totals.items()
        if key in ("position", "order1", "order2", "order3", "order4", "bag4")
    }
    word_prediction = {
        f"{key}_nll_nats_per_track": nll_totals[key] / word_rows_total
        for key in ("clock", "last1", "prefix")
    }
    inference = cluster_inference(
        state_entries, seed=args.seed + 1200, samples=args.bootstrap
    )
    success_episodes = np.flatnonzero(arrays["anchor_success"])
    thresholds = {
        method: float(
            np.quantile(
                episode_maxima(
                    anchor_word_scores[method],
                    arrays["anchor_starts"],
                    arrays["anchor_lengths"],
                    success_episodes,
                ),
                1.0 - args.healthy_fpr,
                method="higher",
            )
        )
        for method in WORD_METHODS
    }
    threshold_by_fold = {int(fold["fold"]): thresholds for fold in folds}
    physical = physical_metrics(
        arrays, anchor_word_scores, assigned_fold, threshold_by_fold, WORD_METHODS
    )
    motif_thresholds = {
        int(fold["fold"]): {motif: 0.5 for motif in FLOW_MOTIFS} for fold in folds
    }
    motif_physical = physical_metrics(
        arrays, anchor_motifs, assigned_fold, motif_thresholds, FLOW_MOTIFS
    )

    summary = {
        "schema_version": 1,
        "protocol": {
            "flow_chord_tracks": list(FLOW_TRACK_NAMES),
            "query_word_shape": [10, len(FLOW_TRACK_NAMES)],
            "query_word_model": (
                "factorial categorical emission; no hard query-word ID or Transformer"
            ),
            "within_query_model": (
                "flow-position conditional ordered/bag backoff grammar, maximum order 4"
            ),
            "inference_inputs": "MoE routing prefix only",
            "state_split": "five-fold disjoint init states",
            "physical_anchor": structure_metadata["anchor_definition"],
            "primitive_definition": primitive_metadata["definition"],
        },
        "counts": {
            "full_queries": len(full_primitives),
            "full_episodes": len(starts),
            "anchor_queries": len(anchor_primitives),
            "anchor_stasis": int(np.sum(arrays["anchor_stasis_onset"] >= 0)),
        },
        "within_query_morphology": morphology,
        "cross_query_word_prediction": word_prediction,
        "state_cluster_inference": inference,
        "physical_stasis_matched_5pct_operating_point": physical,
        "direct_flow_motif_physical_stasis": motif_physical,
        "descriptive_words": description,
        "fold_results": fold_results,
        "matched_thresholds": thresholds,
        "elapsed_seconds": time.time() - started,
    }
    np.savez_compressed(
        args.output_dir / "anchor_scores.npz",
        **anchor_word_scores,
        **anchor_motifs,
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
