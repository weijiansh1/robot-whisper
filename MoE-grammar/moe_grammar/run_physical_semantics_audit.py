"""Does the semantics layer predict the robot, or only the model's own dynamics?

`results-semantics-5fold` showed that a grammar context predicts whether a routing
deviation returns to the healthy manifold. That target is defined by the grammar's own
surprisal percentile, so a positive result is compatible with the layer having learned
nothing about the robot.

This audit runs the identical estimator on the same episodes, the same contexts and the
same horizon against two targets:

* ``model_return``   -- the routing-defined return used by the existing semantics audit;
* ``physical_stasis`` -- physically annotated no-further-progress onset within the horizon.

If context predicts the first but not the second, the layer models the model.

Two confounds are controlled by construction. Every stasis episode in this corpus runs to
the 52-query timeout while successful episodes end at a median of 39, so episode age
alone nearly separates the classes; scoring is therefore restricted to queries where both
classes are still running. And a query already inside a stasis excursion is dropped,
because "will stasis start" is not a question there.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from moe_grammar.corpus import Corpus, Episode, load_corpus
from moe_grammar.features import build_clean_query_descriptors
from moe_grammar.grammar import PositionContextGrammar
from moe_grammar.semantics import SemanticSuffixTrie
from moe_grammar.tokenizer import GMMTokenizer, Preprocessor

BINARY = ("no", "yes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results-physical-semantics"))
    parser.add_argument("--task-substring", default="SCENE8")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--n-words", type=int, default=32)
    parser.add_argument("--pca-dim", type=int, default=24)
    parser.add_argument("--deviation-percentile", type=float, default=0.8)
    parser.add_argument("--max-query", type=int, default=30)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def log_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
    picked = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.maximum(picked, 1e-12)).mean())


def state_folds(states: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(states)
    order = np.random.default_rng(seed).permutation(unique)
    return [order[index::folds] for index in range(folds)]


def build_targets(
    episodes: list[Episode],
    words: np.ndarray,
    deviation: np.ndarray,
    horizon: int,
    threshold: float,
    max_query: int,
) -> dict[str, list[tuple[np.ndarray, str, int]]]:
    """Emit ``(history, label, init_state)`` for both targets at eligible queries."""
    model_rows: list[tuple[np.ndarray, str, int]] = []
    physical_rows: list[tuple[np.ndarray, str, int]] = []
    for episode in episodes:
        onset = episode.stasis_onset
        for offset in range(min(episode.length, max_query)):
            row = episode.start + offset
            history = np.append(
                np.asarray(words[episode.start : row], dtype=np.int64), offset
            )

            # Model-defined target: only meaningful while the sentence is off-manifold.
            if deviation[row] >= threshold and offset + horizon < episode.length:
                future = deviation[row + 1 : row + 1 + horizon]
                model_rows.append(
                    (history, "yes" if np.any(future < threshold) else "no", episode.init_state_id)
                )

            # Physical target: the episode must still be running through the whole
            # horizon, and must not already be inside a stasis excursion.
            if offset + horizon >= episode.length:
                continue
            if onset >= 0 and offset >= onset:
                continue
            hit = onset >= 0 and offset < onset <= offset + horizon
            physical_rows.append((history, "yes" if hit else "no", episode.init_state_id))
    return {"model_return": model_rows, "physical_stasis": physical_rows}


def evaluate(
    train_rows: list[tuple[np.ndarray, str, int]],
    test_rows: list[tuple[np.ndarray, str, int]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Nested comparison: root, clock only, then clock plus word context.

    Histories end with the query index, so the trie's own suffix backoff produces exactly
    this nesting. Reporting words against the *clock* rather than against the root is
    essential here: stasis onset is concentrated at queries 15-21, so any clock
    information leaking into the words would otherwise read as a semantic result.
    """
    if not train_rows or not test_rows:
        return None
    if len({label for _, label, _ in test_rows}) < 2:
        return None
    observations = [(h, y) for h, y, _ in train_rows]
    models = {
        "root": SemanticSuffixTrie(max_order=0, alpha=1.0, min_support=1, outcomes=BINARY),
        "clock": SemanticSuffixTrie(
            max_order=1, alpha=1.0, min_support=args.min_support, outcomes=BINARY
        ),
        "clock_words": SemanticSuffixTrie(
            max_order=1 + args.max_order,
            alpha=1.0,
            min_support=args.min_support,
            outcomes=BINARY,
        ),
    }
    targets = np.asarray([BINARY.index(y) for _, y, _ in test_rows])
    losses = {}
    for name, model in models.items():
        model.fit(observations)
        probabilities = np.stack([model.posterior(h)[0] for h, _, _ in test_rows])
        losses[name] = log_loss(probabilities, targets)
    orders = np.asarray([models["clock_words"].posterior(h)[1] for h, _, _ in test_rows])
    return {
        "queries": len(test_rows),
        "positive_rate": float(targets.mean()),
        "root_log_loss": losses["root"],
        "clock_log_loss": losses["clock"],
        "clock_words_log_loss": losses["clock_words"],
        "clock_minus_root": losses["clock"] - losses["root"],
        "difference": losses["clock_words"] - losses["clock"],
        "word_context_share": float(np.mean(orders >= 2)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    corpus: Corpus = load_corpus(
        args.features_dir, require_flow_shuffled=False, require_stasis_labels=False
    )
    task_index = next(
        index for index, name in enumerate(corpus.tasks) if args.task_substring in name
    )
    episodes = [e for e in corpus.episodes if e.task_index == task_index]
    states = np.asarray([e.init_state_id for e in episodes])
    print(
        f"{corpus.tasks[task_index]}: {len(episodes)} episodes, "
        f"{sum(e.stasis_onset >= 0 for e in episodes)} stasis, "
        f"{sum(e.success for e in episodes)} success",
        flush=True,
    )

    per_fold: dict[str, list[dict[str, Any]]] = {"model_return": [], "physical_stasis": []}
    for fold, test_states in enumerate(state_folds(states, args.folds, args.seed)):
        train = [e for e in episodes if e.init_state_id not in set(test_states.tolist())]
        test = [e for e in episodes if e.init_state_id in set(test_states.tolist())]
        train_success = [e for e in train if e.success]

        rows = np.concatenate([np.arange(e.start, e.stop) for e in train_success])
        descriptors = build_clean_query_descriptors(corpus.features, corpus.feature_names)
        preprocessor = Preprocessor(args.pca_dim, args.seed).fit(descriptors[rows])
        projected = preprocessor.transform(descriptors)
        del descriptors
        tokenizer = GMMTokenizer(args.n_words, seed=args.seed).fit(projected[rows])
        words, emissions, _ = tokenizer.transform(projected)

        grammar = PositionContextGrammar(args.n_words + 1, max_order=2, min_support=20).fit(
            [np.asarray(words[e.start : e.stop], dtype=np.int16) for e in train_success]
        )
        # Only this task's rows are filled; the rest must still be finite because the
        # percentile transform below runs over the whole array.
        surprisal = np.zeros(len(words), dtype=np.float32)
        for episode in episodes:
            piece = slice(episode.start, episode.stop)
            values, _ = grammar.continuous_nll(words[piece], emissions[piece])
            surprisal[piece] = values.astype(np.float32)
        reference = np.sort(
            np.concatenate([surprisal[e.start : e.stop] for e in train_success]).astype(np.float64)
        )
        deviation = np.searchsorted(reference, surprisal, side="right") / len(reference)

        train_sets = build_targets(
            train, words, deviation, args.horizon, args.deviation_percentile, args.max_query
        )
        test_sets = build_targets(
            test, words, deviation, args.horizon, args.deviation_percentile, args.max_query
        )
        for name in per_fold:
            record = evaluate(train_sets[name], test_sets[name], args)
            if record is not None:
                record["fold"] = fold
                per_fold[name].append(record)
        print(f"fold {fold}: done", flush=True)

    summary: dict[str, Any] = {"schema_version": 1, "targets": {}}
    rng = np.random.default_rng(args.seed)
    for name, records in per_fold.items():
        values = np.asarray([r["difference"] for r in records])
        clock_gain = np.asarray([r["clock_minus_root"] for r in records])
        draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
        summary["targets"][name] = {
            "folds": records,
            "mean_context_minus_root": float(values.mean()),
            "fold_bootstrap_ci95": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            "folds_improved": int(np.sum(values < 0)),
            "total_queries": int(sum(r["queries"] for r in records)),
            "mean_positive_rate": float(np.mean([r["positive_rate"] for r in records])),
            "mean_clock_minus_root": float(clock_gain.mean()),
            "mean_word_context_share": float(np.mean([r["word_context_share"] for r in records])),
        }
    summary["protocol"] = {
        "task": corpus.tasks[task_index],
        "horizon": args.horizon,
        "max_query": args.max_query,
        "n_words": args.n_words,
        "folds": args.folds,
        "note": (
            "Scoring stops at max_query because every stasis episode runs to the "
            "52-query timeout while successes end near 39, so episode age alone would "
            "separate the classes."
        ),
    }
    summary["elapsed_seconds"] = time.time() - started
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, row in summary["targets"].items():
        print(
            f"{name:16s} n={row['total_queries']:6d} pos={row['mean_positive_rate']:.4f} "
            f"clock-root={row['mean_clock_minus_root']:+.4f} "
            f"words-clock={row['mean_context_minus_root']:+.4f} "
            f"CI[{row['fold_bootstrap_ci95'][0]:+.4f}, {row['fold_bootstrap_ci95'][1]:+.4f}] "
            f"{row['folds_improved']}/{len(row['folds'])} folds"
        )


if __name__ == "__main__":
    main()
