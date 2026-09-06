"""Measure what the hard-argmax history costs the healthy grammar.

``score_raw_metrics`` marginalizes the current query over all GMM components but
commits every *past* query to its argmax word. A query near a cluster boundary
therefore rewrites the context of the next few steps. This audit reuses a fitted fold
model and rescores held-out successful episodes with the history marginalized over a
weighted beam of plausible past words, so the two can be compared directly.

It reports bits/query for both, the state-blocked paired test, and how often the
committed word is not the beam's most likely word.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from moe_grammar.corpus import load_corpus, make_state_folds
from moe_grammar.run_full40_audit import (
    project_in_batches,
    select_episodes,
    tokenize_in_batches,
)
from moe_grammar.statistics import paired_state_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40-v2"))
    parser.add_argument("--model", type=Path, default=Path("results-full40-v2-fold0/full40_model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("results-full40-v2-fold0/soft_history.json"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-beam", type=int, default=8)
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Held-out successful episodes to rescore; the beam is far slower than argmax",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)

    print("Loading full-40 corpus...", flush=True)
    corpus = load_corpus(
        args.features_dir,
        None,
        allow_task_segments=True,
        require_flow_shuffled=False,
        require_stasis_labels=False,
        feature_dtype=np.float16,
    )
    split = make_state_folds(
        np.asarray([episode.init_state_id for episode in corpus.episodes]), 5, args.seed
    )[args.fold]
    test_success = select_episodes(corpus, split["test"], True)
    rng = np.random.default_rng(args.seed)
    if len(test_success) > args.episodes:
        chosen = rng.choice(len(test_success), args.episodes, replace=False)
        test_success = [test_success[int(index)] for index in sorted(chosen)]
    print(f"Rescoring {len(test_success):,} held-out successful episodes...", flush=True)

    projected = project_in_batches(corpus.features, corpus.feature_names, model["preprocessor"])
    words, emissions, _ = tokenize_in_batches(projected, model["tokenizer"])

    hard_bits: list[float] = []
    beam_bits: list[float] = []
    states: list[int] = []
    disagreements = 0
    positions = 0
    for index, episode in enumerate(test_success):
        selected = slice(episode.start, episode.stop)
        sequence = words[selected]
        emission = emissions[selected].astype(np.float64)
        grammar = model["task_models"][episode.task_index]["history"]
        hard, _ = grammar.continuous_nll(sequence, emission)
        beam = grammar.beam_continuous_nll(
            emission, beam_width=args.beam_width, max_beam=args.max_beam
        )
        hard_bits.append(float(hard.mean() / math.log(2.0)))
        beam_bits.append(float(beam.mean() / math.log(2.0)))
        states.append(episode.init_state_id)
        # How often is the committed word not the emission's most likely word by a
        # clear margin? This is the boundary wobble the hard path silently absorbs.
        ordered = np.sort(emission, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        disagreements += int(np.sum(margin < math.log(2.0)))
        positions += len(sequence)
        if (index + 1) % 250 == 0:
            print(f"  rescored {index + 1:,}/{len(test_success):,}", flush=True)

    hard_array = np.asarray(hard_bits)
    beam_array = np.asarray(beam_bits)
    state_array = np.asarray(states, dtype=np.int16)
    summary: dict[str, Any] = {
        "fold": args.fold,
        "episodes": len(test_success),
        "beam_width": args.beam_width,
        "max_beam": args.max_beam,
        "bits_per_query": {
            "hard_argmax_history": float(hard_array.mean()),
            "beam_marginalized_history": float(beam_array.mean()),
        },
        "beam_minus_hard": paired_state_test(beam_array, hard_array, state_array, args.seed),
        "ambiguous_query_fraction": disagreements / positions,
        "scored_queries": positions,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    difference = summary["beam_minus_hard"]["mean_left_minus_right"]
    print(
        f"\nhard argmax history : {hard_array.mean():.4f} bits/query\n"
        f"beam marginalized   : {beam_array.mean():.4f} bits/query\n"
        f"beam - hard         : {difference:+.6f} "
        f"(one-sided p={summary['beam_minus_hard']['one_sided_p_left_less']:.4f})\n"
        f"queries within 1 bit of a tie: {100.0 * summary['ambiguous_query_fraction']:.2f}%\n"
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
