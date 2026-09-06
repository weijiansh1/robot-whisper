"""Does routing let the alarm fire earlier than a stopwatch, at the same false-alarm cost?

All 1,442 annotated failures run to their run's horizon and 129 of 34,656 successes do, so
episode length alone separates the outcome. Matching that away (as the earlier matched-AUC
work did) removes almost all the signal and answers a question nobody deploys. Ignoring it
inflates every number: `VLA_MUI_HUB/moe-failure-alarm` reports routing AUC 0.821 against
0.950 for a stopwatch.

The deployable question is the nested one. A stopwatch already fires on every failure --
that is guaranteed, because failures are exactly the episodes that do not finish -- but it
can only fire once the episode is already overdue, leaving about 14 queries of an average
38.8-query failure. So the test is not recall. It is:

    at a matched success-episode false-alarm budget, does adding routing move the alarm
    earlier?

`slowness` is the train-success duration ECDF for the task, so it is exactly "how overdue
is this episode". `routing` is the task- and position-conditioned upper tail of healthy
grammar surprisal. Both detectors are calibrated on the same calibration states and use
the same persistence rule, so only the score differs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from moe_grammar.corpus import Corpus, Episode, load_corpus, make_state_folds
from moe_grammar.run_full40_audit import project_in_batches, tokenize_in_batches

PERSISTENCE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40-v2"))
    parser.add_argument("--model-root", default="results-full40-v2-fold")
    parser.add_argument("--output-dir", type=Path, default=Path("results-slowness-alarm"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--healthy-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def slowness_curve(durations: np.ndarray, horizon: int) -> np.ndarray:
    """Fraction of healthy episodes that have already finished by each query."""
    grid = np.arange(horizon + 1)
    return np.searchsorted(np.sort(durations), grid, side="right") / max(len(durations), 1)


def causal_word_histograms(
    words: np.ndarray, episodes: list[Episode], n_words: int
) -> np.ndarray:
    """Normalized histogram of every word strictly before each query."""
    output = np.zeros((len(words), n_words), dtype=np.float32)
    for episode in episodes:
        running = np.zeros(n_words, dtype=np.float32)
        for offset in range(episode.length):
            row = episode.start + offset
            output[row] = running / max(running.sum(), 1.0)
            running[words[row]] += 1.0
    return output


def infer_task_posterior(
    histograms: np.ndarray,
    episodes: list[Episode],
    train_episodes: list[Episode],
    n_tasks: int,
    seed: int,
    samples_per_episode: int = 6,
) -> np.ndarray:
    """P(task | causal routing prefix, query index), with no task oracle online.

    The strict deployment cannot look up a per-task duration curve, so the task has to be
    read out of the routing prefix itself. Query index is included because it is observed.
    """
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for episode in train_episodes:
        if episode.length < 2:
            continue
        picks = np.unique(
            rng.integers(1, episode.length, size=min(samples_per_episode, episode.length - 1))
        )
        for offset in picks:
            rows.append(episode.start + int(offset))
            labels.append(episode.task_index)
    index = np.asarray(rows)
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(
        histograms[index], np.asarray(labels)
    )
    posterior = np.full((len(histograms), n_tasks), 1.0 / n_tasks, dtype=np.float32)
    all_rows = np.concatenate(
        [np.arange(e.start, e.stop) for e in episodes]
    )
    predicted = model.predict_proba(histograms[all_rows])
    posterior[all_rows[:, None], model.classes_[None, :]] = predicted
    return posterior


def routing_percentile(
    values: np.ndarray,
    episodes: list[Episode],
    reference: dict[tuple[int, int], np.ndarray],
    task_reference: dict[int, np.ndarray],
) -> np.ndarray:
    """Upper-tail percentile of surprisal within the same task and query index."""
    output = np.zeros(len(values), dtype=np.float64)
    for episode in episodes:
        for offset in range(episode.length):
            row = episode.start + offset
            sample = reference.get((episode.task_index, offset))
            if sample is None or len(sample) < 20:
                sample = task_reference.get(episode.task_index)
            if sample is None or len(sample) == 0:
                continue
            output[row] = np.searchsorted(sample, values[row], side="right") / len(sample)
    return output


def first_alarm(score: np.ndarray, episode: Episode, threshold: float) -> int | None:
    """First query where the score stays above threshold for PERSISTENCE queries."""
    run = 0
    for offset in range(episode.length):
        if score[episode.start + offset] > threshold:
            run += 1
            if run >= PERSISTENCE:
                return offset
        else:
            run = 0
    return None


def calibrate(
    score: np.ndarray, episodes: list[Episode], target_fpr: float
) -> float:
    """Highest threshold whose success false-alarm rate stays within budget."""
    maxima = np.asarray(
        [
            max(
                (
                    min(score[e.start + i], score[e.start + i + 1])
                    for i in range(e.length - 1)
                ),
                default=0.0,
            )
            for e in episodes
        ]
    )
    if len(maxima) == 0:
        return 1.0
    return float(np.quantile(maxima, 1.0 - target_fpr, method="higher"))


def evaluate(
    score: np.ndarray,
    successes: list[Episode],
    failures: list[Episode],
    horizons: dict[int, int],
    threshold: float,
) -> dict[str, Any]:
    false_alarms = sum(first_alarm(score, e, threshold) is not None for e in successes)
    alarms = [(e, first_alarm(score, e, threshold)) for e in failures]
    detected = [(e, a) for e, a in alarms if a is not None]
    leads = [horizons[e.task_index] - a for e, a in detected]
    return {
        "alarm_by_episode": {
            int(e.index): (None if a is None else int(a), int(horizons[e.task_index]))
            for e, a in alarms
        },
        "success_episodes": len(successes),
        "success_fpr": false_alarms / max(len(successes), 1),
        "failure_episodes": len(failures),
        "failure_recall": len(detected) / max(len(failures), 1),
        "median_alarm_query": float(np.median([a for _, a in detected])) if detected else None,
        "median_lead_queries": float(np.median(leads)) if leads else None,
        "mean_lead_queries": float(np.mean(leads)) if leads else None,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("Loading full-40 corpus...", flush=True)
    corpus: Corpus = load_corpus(
        args.features_dir,
        None,
        allow_task_segments=True,
        require_flow_shuffled=False,
        require_stasis_labels=False,
        feature_dtype=np.float16,
    )
    splits = make_state_folds(
        np.asarray([e.init_state_id for e in corpus.episodes]), 5, args.seed
    )
    horizons = {
        index: max(e.length for e in corpus.episodes if e.task_index == index)
        for index in range(len(corpus.tasks))
    }

    # Each combined detector is compared against the baseline it is allowed to use:
    # the oracle-task stopwatch, or the strict routing-inferred one.
    baselines = {
        "slowness_plus_routing": "slowness",
        "slowness_plus_shuffled": "slowness",
        "slowness_or_routing": "slowness",
        "slowness_or_shuffled": "slowness",
        "strict_inferred_or_routing": "slowness_inferred",
        "strict_inferred_or_shuffled": "slowness_inferred",
    }
    per_fold: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "slowness",
            "slowness_global",
            "slowness_inferred",
            *baselines,
        )
    }
    for fold in args.folds:
        split = splits[fold]
        model = joblib.load(Path(f"{args.model_root}{fold}") / "full40_model.joblib")
        projected = project_in_batches(
            corpus.features, corpus.feature_names, model["preprocessor"]
        )
        words, emissions, _ = tokenize_in_batches(projected, model["tokenizer"])
        del projected
        word_ids = np.asarray(words, dtype=np.int64)
        n_words = model["tokenizer"].n_words

        surprisal = np.empty(len(words), dtype=np.float32)
        for episode in corpus.episodes:
            piece = slice(episode.start, episode.stop)
            values, _ = model["task_models"][episode.task_index]["history"].continuous_nll(
                words[piece], emissions[piece]
            )
            surprisal[piece] = values.astype(np.float32)
        del emissions, words

        train_states = set(np.asarray(split["train"]).tolist())
        calibration_states = set(np.asarray(split["calibration"]).tolist())
        test_states = set(np.asarray(split["test"]).tolist())
        train_success = [e for e in corpus.episodes if e.init_state_id in train_states and e.success]

        # Slowness: how overdue is this episode for its task, from train successes only.
        slow = np.zeros(len(surprisal), dtype=np.float64)
        curves = {}
        for index in range(len(corpus.tasks)):
            durations = np.asarray(
                [e.length for e in train_success if e.task_index == index], dtype=np.int64
            )
            curves[index] = slowness_curve(durations, horizons[index])
        for episode in corpus.episodes:
            curve = curves[episode.task_index]
            offsets = np.arange(episode.length)
            slow[episode.start : episode.stop] = curve[np.minimum(offsets, len(curve) - 1)]

        # Strict MoE-only slowness: no task oracle. Either pool all tasks, or read the
        # task out of the causal routing prefix and mix the per-task curves by it.
        pooled_curve = slowness_curve(
            np.asarray([e.length for e in train_success], dtype=np.int64),
            max(horizons.values()),
        )
        global_slow = np.zeros(len(surprisal), dtype=np.float64)
        for episode in corpus.episodes:
            offsets = np.arange(episode.length)
            global_slow[episode.start : episode.stop] = pooled_curve[
                np.minimum(offsets, len(pooled_curve) - 1)
            ]

        histograms = causal_word_histograms(word_ids, corpus.episodes, n_words)
        train_all = [e for e in corpus.episodes if e.init_state_id in train_states]
        posterior = infer_task_posterior(
            histograms, corpus.episodes, train_all, len(corpus.tasks), args.seed + fold
        )
        del histograms
        limit = max(horizons.values())
        curve_matrix = np.stack(
            [
                np.pad(curves[i], (0, limit + 1 - len(curves[i])), mode="edge")
                for i in range(len(corpus.tasks))
            ]
        )
        inferred_slow = np.zeros(len(surprisal), dtype=np.float64)
        for episode in corpus.episodes:
            rows = np.arange(episode.start, episode.stop)
            offsets = np.minimum(np.arange(episode.length), limit)
            inferred_slow[rows] = np.einsum(
                "rt,tr->r", posterior[rows], curve_matrix[:, offsets]
            )
        del posterior

        reference: dict[tuple[int, int], np.ndarray] = {}
        task_reference: dict[int, np.ndarray] = {}
        for index in range(len(corpus.tasks)):
            pool = [e for e in train_success if e.task_index == index]
            if pool:
                task_reference[index] = np.sort(
                    np.concatenate([surprisal[e.start : e.stop] for e in pool])
                )
            for offset in range(horizons[index]):
                sample = [
                    surprisal[e.start + offset] for e in pool if offset < e.length
                ]
                if sample:
                    reference[(index, offset)] = np.sort(np.asarray(sample))
        routing = routing_percentile(surprisal, corpus.episodes, reference, task_reference)

        # Permutation control: same marginal routing distribution inside every
        # (task, query) cell, but the episode identity is destroyed. Combining any second
        # channel by max and recalibrating can move the alarm on its own, so the real
        # effect is whatever survives against this.
        generator = np.random.default_rng(args.seed + 17 * fold)
        shuffled = routing.copy()
        cells: dict[tuple[int, int], list[int]] = {}
        for episode in corpus.episodes:
            for offset in range(episode.length):
                cells.setdefault((episode.task_index, offset), []).append(
                    episode.start + offset
                )
        for rows in cells.values():
            index = np.asarray(rows)
            shuffled[index] = routing[generator.permutation(index)]

        # A single threshold over max(slow, routing) has to rise to hold the budget,
        # which delays alarms the stopwatch would already have raised. Splitting the
        # budget across two independently calibrated channels avoids that.
        scores = {
            "slowness": slow,
            "slowness_plus_routing": np.maximum(slow, routing),
            "slowness_plus_shuffled": np.maximum(slow, shuffled),
        }
        scores["slowness_global"] = global_slow
        scores["slowness_inferred"] = inferred_slow
        split_channels = {
            "slowness_or_routing": (slow, routing),
            "slowness_or_shuffled": (slow, shuffled),
            "strict_inferred_or_routing": (inferred_slow, routing),
            "strict_inferred_or_shuffled": (inferred_slow, shuffled),
        }
        calibration_success = [
            e for e in corpus.episodes if e.init_state_id in calibration_states and e.success
        ]
        test_success = [
            e for e in corpus.episodes if e.init_state_id in test_states and e.success
        ]
        test_failure = [
            e for e in corpus.episodes if e.init_state_id in test_states and not e.success
        ]
        for name, score in scores.items():
            threshold = calibrate(score, calibration_success, args.healthy_fpr)
            record = evaluate(score, test_success, test_failure, horizons, threshold)
            record["threshold"] = threshold
            record["fold"] = fold
            per_fold[name].append(record)
        for name, (primary, secondary) in split_channels.items():
            first = calibrate(primary, calibration_success, args.healthy_fpr * 0.8)
            second = calibrate(secondary, calibration_success, args.healthy_fpr * 0.2)
            union = np.maximum(
                np.where(primary > first, 1.0, 0.0), np.where(secondary > second, 1.0, 0.0)
            )
            record = evaluate(union, test_success, test_failure, horizons, 0.5)
            record["thresholds"] = [first, second]
            record["fold"] = fold
            per_fold[name].append(record)
        print(f"fold {fold}: done", flush=True)
        del model

    def pooled(name: str, key: str) -> dict[str, Any]:
        values = np.asarray(
            [r[key] for r in per_fold[name] if r[key] is not None], dtype=float
        )
        rng = np.random.default_rng(args.seed)
        draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
        return {
            "mean": float(values.mean()),
            "fold_values": values.tolist(),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        }

    summary: dict[str, Any] = {"schema_version": 1, "detectors": {}, "folds": per_fold}
    for name in per_fold:
        summary["detectors"][name] = {
            key: pooled(name, key)
            for key in ("success_fpr", "failure_recall", "median_alarm_query", "median_lead_queries")
        }
    rng = np.random.default_rng(args.seed)
    for name, baseline in baselines.items():
        earlier = np.asarray(
            [
                a["median_alarm_query"] - b["median_alarm_query"]
                for a, b in zip(per_fold[name], per_fold[baseline])
                if a["median_alarm_query"] is not None and b["median_alarm_query"] is not None
            ]
        )
        draws = earlier[rng.integers(0, len(earlier), size=(5000, len(earlier)))].mean(axis=1)
        summary[f"{name}_minus_slowness_alarm_query"] = {
            "mean": float(earlier.mean()),
            "fold_values": earlier.tolist(),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "folds_earlier": int(np.sum(earlier < 0)),
        }
    # Median-of-medians hides that a higher joint threshold can delay the alarm on
    # episodes where routing never fires. Compare the same episode under both detectors.
    for name, baseline in baselines.items():
        deltas: list[int] = []
        by_horizon: dict[int, list[tuple[int, int]]] = {}
        for combined, base in zip(per_fold[name], per_fold[baseline]):
            for key, (value, horizon) in combined["alarm_by_episode"].items():
                other = base["alarm_by_episode"].get(key)
                if value is None or other is None or other[0] is None:
                    continue
                deltas.append(value - other[0])
                by_horizon.setdefault(horizon, []).append((value, other[0]))
        groups = {}
        for horizon, pairs in sorted(by_horizon.items()):
            shift = np.asarray([a - b for a, b in pairs], dtype=float)
            picks = rng.integers(0, len(shift), size=(5000, len(shift)))
            samples = shift[picks].mean(axis=1)
            groups[str(horizon)] = {
                "failures": len(shift),
                "median_shift": float(np.median(shift)),
                "shift_quantiles": [float(np.quantile(shift, q)) for q in (0.05,0.25,0.5,0.75,0.95)],
                "shift_min": float(shift.min()),
                "shift_max": float(shift.max()),
                "mean_shift_trimmed10": float(
                    shift[(shift >= np.quantile(shift, 0.05)) & (shift <= np.quantile(shift, 0.95))].mean()
                ),
                "horizon": horizon,
                "baseline_median_alarm": float(np.median([b for _, b in pairs])),
                "detector_median_alarm": float(np.median([a for a, _ in pairs])),
                "mean_shift": float(shift.mean()),
                "ci95": [
                    float(np.quantile(samples, 0.025)),
                    float(np.quantile(samples, 0.975)),
                ],
                "shift_as_horizon_fraction": float(shift.mean() / horizon),
                "fraction_earlier": float(np.mean(shift < 0)),
                "fraction_later": float(np.mean(shift > 0)),
                # The median shift is zero everywhere, so a mean describes nobody. What
                # the channel actually does is fire much earlier on a minority.
                "fraction_earlier_by_3plus": float(np.mean(shift <= -3)),
                "median_gain_when_earlier_by_3plus": (
                    float(-np.median(shift[shift <= -3])) if np.any(shift <= -3) else 0.0
                ),
                "fraction_later_by_3plus": float(np.mean(shift >= 3)),
            }
        summary[f"{name}_by_horizon"] = groups
        array = np.asarray(deltas, dtype=float)
        draws = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
        summary[f"{name}_paired_alarm_shift"] = {
            "episodes": len(array),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "fraction_earlier": float(np.mean(array < 0)),
            "fraction_later": float(np.mean(array > 0)),
            "fraction_unchanged": float(np.mean(array == 0)),
        }
    for record in [r for rows in per_fold.values() for r in rows]:
        record.pop("alarm_by_episode", None)
    summary["elapsed_seconds"] = time.time() - started
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for name, row in summary["detectors"].items():
        print(
            f"{name:24s} fpr={row['success_fpr']['mean']:.3f} "
            f"recall={row['failure_recall']['mean']:.3f} "
            f"alarm@q={row['median_alarm_query']['mean']:.1f} "
            f"lead={row['median_lead_queries']['mean']:.1f}"
        )
    for name in baselines:
        paired = summary[f"{name}_paired_alarm_shift"]
        print(
            f"{name:24s} paired per-episode shift {paired['mean']:+.2f} "
            f"CI[{paired['ci95'][0]:+.2f}, {paired['ci95'][1]:+.2f}] "
            f"earlier {paired['fraction_earlier']:.1%} / later {paired['fraction_later']:.1%} "
            f"/ same {paired['fraction_unchanged']:.1%}"
        )
    for name in ("slowness_or_routing", "strict_inferred_or_routing"):
        shift = summary[f"{name}_minus_slowness_alarm_query"]
        print(
            f"{name:24s} moves the alarm {shift['mean']:+.2f} queries "
            f"CI[{shift['ci95'][0]:+.2f}, {shift['ci95'][1]:+.2f}] "
            f"({shift['folds_earlier']}/{len(shift['fold_values'])} folds earlier)"
        )


if __name__ == "__main__":
    main()
