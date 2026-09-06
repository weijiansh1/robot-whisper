"""Is the grammar channel worth anything next to the published `mobility` quantity?

`results-slowness-alarm` showed that a routing channel on top of a stopwatch fires
earlier on about 9% of failures. That channel was the grammar's context-conditioned
surprisal. But `moe-flow-semantics-0906` and `moe-circuit-analogy-0906` report a
survival-conditioned |AUC-0.5| of 0.1147 for `mobility` against 0.0847-0.0969 for the
next best quantities, which is stronger than anything this directory produced.

So the question is not whether routing helps. It is whether *the grammar* helps once a
much simpler cross-query quantity is already in the detector.

`mobility` is the Hellinger distance between this query's action routing and the previous
query's at the same denoise step, precomputed in
`moe-flow-semantics-0906/results/step_profiles`. Step 9 reproduces the frozen v4
definition bit for bit. Both channels are converted to the same task- and
position-conditioned percentile, so only the underlying quantity differs.

Every detector shares one calibration set, one 5% episode budget, one persistence rule,
and the same permutation control.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from moe_grammar.corpus import Corpus, load_corpus, make_state_folds
from moe_grammar.run_full40_audit import project_in_batches, tokenize_in_batches
from moe_grammar.run_slowness_alarm_audit import (
    calibrate,
    causal_word_histograms,
    evaluate,
    infer_task_posterior,
    routing_percentile,
    slowness_curve,
)

PROFILES = Path(
    "/home/jovyan/work/himoe-vla/moe-flow-semantics-0906/results/step_profiles"
)
COHORTS = ("development_main", "development_extra", "external_8b")
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
V4_STEP = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40-v2"))
    parser.add_argument("--model-root", default="results-full40-v2-fold")
    parser.add_argument("--output-dir", type=Path, default=Path("results-channel-comparison"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--layer", default="L12", choices=LAYER_NAMES)
    parser.add_argument("--healthy-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def load_mobility(corpus: Corpus, layer: str, total_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Attach the published per-query mobility to this corpus's query rows."""
    index = {}
    for episode in corpus.episodes:
        key = (corpus.tasks[episode.task_index], episode.init_state_id, episode.flow_noise_seed)
        index[key] = episode
    column = LAYER_NAMES.index(layer)
    values = np.zeros(total_rows, dtype=np.float64)
    covered = np.zeros(total_rows, dtype=bool)
    for cohort in COHORTS:
        meta = np.load(PROFILES / f"{cohort}_index.npz", allow_pickle=True)
        names = [str(x) for x in meta["task_names"]]
        array = np.load(PROFILES / f"{cohort}_mobility.npy", mmap_mode="r")
        for row in range(len(meta["task_index"])):
            key = (
                names[int(meta["task_index"][row])],
                int(meta["init_state_id"][row]),
                int(meta["flow_noise_seed"][row]),
            )
            episode = index.get(key)
            if episode is None:
                continue
            length = min(episode.length, array.shape[1])
            piece = np.asarray(array[row, :length, column, V4_STEP], dtype=np.float64)
            # Query 0 has no predecessor, so mobility is undefined there.
            piece = np.where(np.isfinite(piece), piece, 0.0)
            values[episode.start : episode.start + length] = piece
            covered[episode.start : episode.start + length] = True
    return values, covered


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
    total_rows = int(sum(e.length for e in corpus.episodes))
    mobility_raw, covered = load_mobility(corpus, args.layer, total_rows)
    # Only episodes fully covered by the published profiles enter the comparison, so
    # every detector is scored on exactly the same episodes.
    usable = [e for e in corpus.episodes if covered[e.start : e.stop].all()]
    print(
        f"mobility layer {args.layer} step {V4_STEP}: "
        f"{len(usable):,}/{len(corpus.episodes):,} episodes covered",
        flush=True,
    )

    names = [
        "slowness_inferred",
        "plus_grammar",
        "plus_mobility",
        "plus_both",
        "plus_grammar_shuffled",
        "plus_mobility_shuffled",
        "budget_cost_only",
    ]
    per_fold: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
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

        surprisal = np.zeros(len(word_ids), dtype=np.float32)
        for episode in corpus.episodes:
            piece = slice(episode.start, episode.stop)
            values, _ = model["task_models"][episode.task_index]["history"].continuous_nll(
                word_ids[piece], emissions[piece]
            )
            surprisal[piece] = values.astype(np.float32)
        del emissions, words

        train_states = set(np.asarray(split["train"]).tolist())
        train_success = [e for e in usable if e.init_state_id in train_states and e.success]

        curves = {
            index: slowness_curve(
                np.asarray(
                    [e.length for e in train_success if e.task_index == index], dtype=np.int64
                ),
                horizons[index],
            )
            for index in range(len(corpus.tasks))
        }
        histograms = causal_word_histograms(word_ids, usable, n_words)
        posterior = infer_task_posterior(
            histograms,
            usable,
            [e for e in usable if e.init_state_id in train_states],
            len(corpus.tasks),
            args.seed + fold,
        )
        del histograms
        limit = max(horizons.values())
        matrix = np.stack(
            [
                np.pad(curves[i], (0, limit + 1 - len(curves[i])), mode="edge")
                for i in range(len(corpus.tasks))
            ]
        )
        slow = np.zeros(len(surprisal), dtype=np.float64)
        for episode in usable:
            rows = np.arange(episode.start, episode.stop)
            offsets = np.minimum(np.arange(episode.length), limit)
            slow[rows] = np.einsum("rt,tr->r", posterior[rows], matrix[:, offsets])
        del posterior

        channels = {}
        # Failures sit at the 0.24-0.40 percentile of matched successes, so mobility is a
        # low-value risk signal and must be negated before an upper-tail percentile.
        for label, raw in (
            ("grammar", surprisal.astype(np.float64)),
            ("mobility", -mobility_raw),
        ):
            reference: dict[tuple[int, int], np.ndarray] = {}
            task_reference: dict[int, np.ndarray] = {}
            for index in range(len(corpus.tasks)):
                pool = [e for e in train_success if e.task_index == index]
                if pool:
                    task_reference[index] = np.sort(
                        np.concatenate([raw[e.start : e.stop] for e in pool])
                    )
                for offset in range(horizons[index]):
                    sample = [raw[e.start + offset] for e in pool if offset < e.length]
                    if sample:
                        reference[(index, offset)] = np.sort(np.asarray(sample))
            channels[label] = routing_percentile(raw, usable, reference, task_reference)

        generator = np.random.default_rng(args.seed + 31 * fold)
        cells: dict[tuple[int, int], list[int]] = {}
        for episode in usable:
            for offset in range(episode.length):
                cells.setdefault((episode.task_index, offset), []).append(episode.start + offset)
        shuffled = {}
        for label, values in channels.items():
            copy = values.copy()
            for rows in cells.values():
                index = np.asarray(rows)
                copy[index] = values[generator.permutation(index)]
            shuffled[label] = copy

        # `plus_both` must not be max() of two percentiles: a single threshold over the
        # max has to rise to hold the budget and cancels what each channel contributes.
        # Each channel is calibrated independently and the alarms are unioned.
        secondary = {
            "plus_grammar": [channels["grammar"]],
            "plus_mobility": [channels["mobility"]],
            "plus_both": [channels["grammar"], channels["mobility"]],
            "plus_grammar_shuffled": [shuffled["grammar"]],
            "plus_mobility_shuffled": [shuffled["mobility"]],
            # No second channel at all: isolates the cost of reserving 20% of the budget.
            "budget_cost_only": [],
        }

        calibration_states = set(np.asarray(split["calibration"]).tolist())
        test_states = set(np.asarray(split["test"]).tolist())
        calibration_success = [
            e for e in usable if e.init_state_id in calibration_states and e.success
        ]
        test_success = [e for e in usable if e.init_state_id in test_states and e.success]
        test_failure = [e for e in usable if e.init_state_id in test_states and not e.success]

        base_threshold = calibrate(slow, calibration_success, args.healthy_fpr)
        record = evaluate(slow, test_success, test_failure, horizons, base_threshold)
        record["fold"] = fold
        per_fold["slowness_inferred"].append(record)
        for name, group in secondary.items():
            first = calibrate(slow, calibration_success, args.healthy_fpr * 0.8)
            union = np.where(slow > first, 1.0, 0.0)
            share = 0.2 / max(len(group), 1)
            for channel in group:
                threshold = calibrate(channel, calibration_success, args.healthy_fpr * share)
                union = np.maximum(union, np.where(channel > threshold, 1.0, 0.0))
            row = evaluate(union, test_success, test_failure, horizons, 0.5)
            row["fold"] = fold
            per_fold[name].append(row)
        print(f"fold {fold}: done", flush=True)
        del model

    rng = np.random.default_rng(args.seed)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "layer": args.layer,
        "step": V4_STEP,
        "episodes_used": len(usable),
        "detectors": {},
    }
    for name in names:
        rows = per_fold[name]
        summary["detectors"][name] = {
            "success_fpr": float(np.mean([r["success_fpr"] for r in rows])),
            "failure_recall": float(np.mean([r["failure_recall"] for r in rows])),
            "median_alarm_query": float(np.mean([r["median_alarm_query"] for r in rows])),
        }
    for name in names[1:]:
        deltas: list[int] = []
        for combined, base in zip(per_fold[name], per_fold["slowness_inferred"]):
            for key, (value, _) in combined["alarm_by_episode"].items():
                other = base["alarm_by_episode"].get(key)
                if value is None or other is None or other[0] is None:
                    continue
                deltas.append(value - other[0])
        array = np.asarray(deltas, dtype=float)
        draws = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
        summary["detectors"][name]["paired_shift"] = {
            "episodes": len(array),
            "mean": float(array.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "fraction_earlier_by_3plus": float(np.mean(array <= -3)),
            "fraction_later_by_3plus": float(np.mean(array >= 3)),
        }
    for rows in per_fold.values():
        for record in rows:
            record.pop("alarm_by_episode", None)
    summary["folds"] = per_fold
    summary["elapsed_seconds"] = time.time() - started
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\n{'detector':26s} {'FPR':>6s} {'recall':>7s} {'alarm@q':>8s} {'配对移动':>22s} {'早>=3':>7s}")
    for name in names:
        row = summary["detectors"][name]
        shift = row.get("paired_shift")
        cell = (
            f"{shift['mean']:+.2f} [{shift['ci95'][0]:+.2f},{shift['ci95'][1]:+.2f}]"
            if shift
            else "—"
        )
        early = f"{shift['fraction_earlier_by_3plus']:.1%}" if shift else "—"
        print(
            f"{name:26s} {row['success_fpr']:6.3f} {row['failure_recall']:7.3f} "
            f"{row['median_alarm_query']:8.1f} {cell:>22s} {early:>7s}"
        )


if __name__ == "__main__":
    main()
