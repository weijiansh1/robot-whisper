"""Cache per-fold PCA projections so the vocabulary sweep needs no corpus load.

The sweep itself is GPU-bound; loading the 508k-query corpus is the memory-heavy part and
is done exactly once here rather than once per (K, fold) combination.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from moe_grammar.corpus import load_corpus, make_state_folds
from moe_grammar.run_full40_audit import balanced_query_sample, project_in_batches, select_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40-v2"))
    parser.add_argument("--model-root", default="results-full40-v2-fold")
    parser.add_argument("--output", type=Path, default=Path("artifacts/vocabulary-sweep.npz"))
    parser.add_argument("--rows-per-task", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading full-40 corpus...", flush=True)
    corpus = load_corpus(
        args.features_dir,
        None,
        allow_task_segments=True,
        require_flow_shuffled=False,
        require_stasis_labels=False,
        feature_dtype=np.float16,
    )
    states = np.asarray([episode.init_state_id for episode in corpus.episodes])
    folds = make_state_folds(states, 5, args.seed)

    payload: dict[str, np.ndarray] = {}
    for fold_index, split in enumerate(folds):
        model = joblib.load(Path(f"{args.model_root}{fold_index}") / "full40_model.joblib")
        projected = project_in_batches(
            corpus.features, corpus.feature_names, model["preprocessor"]
        )
        train_success = select_episodes(corpus, split["train"], True)
        test_success = select_episodes(corpus, split["test"], True)
        payload[f"projected_{fold_index}"] = projected.astype(np.float32)
        payload[f"train_rows_{fold_index}"] = balanced_query_sample(
            train_success, args.rows_per_task, args.seed
        )
        payload[f"test_rows_{fold_index}"] = np.concatenate(
            [np.arange(e.start, e.stop) for e in test_success]
        )
        print(f"fold {fold_index}: projected {projected.shape}", flush=True)
        del projected

    np.savez_compressed(args.output, **payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
