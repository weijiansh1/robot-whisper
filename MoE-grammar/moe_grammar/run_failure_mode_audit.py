"""Two length-free questions about physically annotated failure modes.

Every earlier failure result in this directory is dominated by one structural fact:
1,442 of 1,442 annotated failures run to their run's maximum horizon, while 129 of 34,656
successes do. Episode length therefore separates the outcome label almost perfectly, and
`VLA_MUI_HUB/moe-failure-alarm` measures exactly that -- its routing alarm reaches episode
AUC 0.821 against 0.950 for a stopwatch.

Both questions here are asked *inside the failure set*, where length is constant by
construction, so that confound cannot operate.

1. ``failure_type``  -- does routing distinguish which physical failure mode occurred,
   beyond what the task identity already implies? Failure modes are strongly task
   dependent (a drawer task cannot drop an object), so the baseline is task-conditioned.
2. ``release_onset`` -- does routing anticipate the annotated moment the gripper drops the
   object? `sim_state` is recorded once per action chunk, so a snapshot index equals a
   query index, giving an exactly aligned mid-episode physical event.

Labels come from `physical-failure-labels`, which restores every recorded MuJoCo control
state and evaluates the original BDDL predicates. They are failure-mode observations, not
proven policy causes.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from moe_grammar.corpus import Corpus, Episode, load_corpus, make_state_folds
from moe_grammar.run_full40_audit import project_in_batches, tokenize_in_batches
from moe_grammar.semantics import SemanticSuffixTrie

LABELS = Path(
    "/home/jovyan/work/himoe-vla/VLA_MUI_HUB/physical-failure-labels/results/failures.jsonl"
)
BINARY = ("no", "yes")
MIN_TYPE_SUPPORT = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40-v2"))
    parser.add_argument("--model-root", default="results-full40-v2-fold")
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--data-root", default="cache_new")
    parser.add_argument("--output-dir", type=Path, default=Path("results-failure-mode"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def load_labels(path: Path, data_root: str) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["data_root"] == data_root:
            records.append(record)
    return records


def release_query(record: dict[str, Any]) -> int | None:
    """Earliest annotated release snapshot, which indexes queries one to one."""
    snapshots = [
        physics["first_release_snapshot"]
        for physics in (record.get("goal_subject_physics") or {}).values()
        if physics.get("first_release_snapshot") is not None
    ]
    return min(snapshots) if snapshots else None


def log_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
    picked = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.maximum(picked, 1e-12)).mean())


def word_histogram(words: np.ndarray, episode: Episode, n_words: int) -> np.ndarray:
    """Order-free episode summary, plus the same over the final third."""
    piece = words[episode.start : episode.stop]
    late = piece[max(0, len(piece) - len(piece) // 3) :]
    whole = np.bincount(piece, minlength=n_words).astype(np.float64)
    tail = np.bincount(late, minlength=n_words).astype(np.float64)
    return np.concatenate([whole / max(whole.sum(), 1.0), tail / max(tail.sum(), 1.0)])


STRENGTHS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, classes: int, seed: int
) -> np.ndarray:
    """Select regularization inside the training split.

    The routing histogram has more columns than the task dummies, so a fixed penalty
    lets the richer design overfit and read as a negative result. Each design therefore
    picks its own strength on held-out training episodes.
    """
    scaler = StandardScaler().fit(train_x)
    scaled = scaler.transform(train_x)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(scaled))
    cut = max(1, int(0.75 * len(order)))
    inner_train, inner_valid = order[:cut], order[cut:]
    best, best_loss = STRENGTHS[0], np.inf
    if len(np.unique(train_y[inner_train])) == len(np.unique(train_y)):
        for strength in STRENGTHS:
            trial = LogisticRegression(max_iter=2000, C=strength, random_state=seed).fit(
                scaled[inner_train], train_y[inner_train]
            )
            probabilities = np.full((len(inner_valid), classes), 1.0 / classes)
            probabilities[:, trial.classes_] = trial.predict_proba(scaled[inner_valid])
            loss = log_loss(probabilities, train_y[inner_valid])
            if loss < best_loss:
                best, best_loss = strength, loss
    model = LogisticRegression(max_iter=2000, C=best, random_state=seed).fit(scaled, train_y)
    probabilities = np.full((len(test_x), classes), 1.0 / classes)
    probabilities[:, model.classes_] = model.predict_proba(scaler.transform(test_x))
    return probabilities


def failure_type_fold(
    train: list[tuple[np.ndarray, int, int]],
    test: list[tuple[np.ndarray, int, int]],
    n_types: int,
    n_tasks: int,
    seed: int,
) -> dict[str, Any] | None:
    """Nested: global prior, task only, then task plus routing histogram."""
    if not train or not test:
        return None
    train_y = np.asarray([y for _, y, _ in train])
    test_y = np.asarray([y for _, y, _ in test])
    if len(np.unique(train_y)) < 2:
        return None

    counts = np.bincount(train_y, minlength=n_types) + 1.0
    prior = np.repeat((counts / counts.sum())[None, :], len(test_y), axis=0)

    def onehot(rows: list[tuple[np.ndarray, int, int]]) -> np.ndarray:
        matrix = np.zeros((len(rows), n_tasks))
        matrix[np.arange(len(rows)), [t for _, _, t in rows]] = 1.0
        return matrix

    task_only = fit_predict(onehot(train), train_y, onehot(test), n_types, seed)
    combined = fit_predict(
        np.hstack([onehot(train), np.stack([x for x, _, _ in train])]),
        train_y,
        np.hstack([onehot(test), np.stack([x for x, _, _ in test])]),
        n_types,
        seed,
    )
    return {
        "episodes": len(test_y),
        "prior_log_loss": log_loss(prior, test_y),
        "task_log_loss": log_loss(task_only, test_y),
        "task_routing_log_loss": log_loss(combined, test_y),
        "routing_minus_task": log_loss(combined, test_y) - log_loss(task_only, test_y),
        "task_minus_prior": log_loss(task_only, test_y) - log_loss(prior, test_y),
        "accuracy_task": float((task_only.argmax(axis=1) == test_y).mean()),
        "accuracy_task_routing": float((combined.argmax(axis=1) == test_y).mean()),
    }


def release_rows(
    episodes: list[Episode],
    onsets: dict[int, int],
    words: np.ndarray,
    horizon: int,
    within_release_only: bool,
) -> list[tuple[np.ndarray, str]]:
    """Label whether the annotated release happens in the next ``horizon`` queries.

    Histories end with ``task`` then ``query index`` so the trie's suffix backoff yields
    root -> clock -> clock+task -> clock+task+words. Words strongly encode the task, so a
    clock-only baseline would credit task identity to the routing context.
    """
    rows: list[tuple[np.ndarray, str]] = []
    for episode in episodes:
        onset = onsets.get(episode.index)
        if within_release_only and onset is None:
            continue
        limit = episode.length if onset is None else min(onset, episode.length)
        for offset in range(limit):
            if offset + horizon >= episode.length:
                continue
            history = np.concatenate(
                [
                    np.asarray(words[episode.start : episode.start + offset], dtype=np.int64),
                    [-1 - episode.task_index, offset],
                ]
            )
            hit = onset is not None and offset < onset <= offset + horizon
            rows.append((history, "yes" if hit else "no"))
    return rows


def release_fold(
    train: list[tuple[np.ndarray, str]],
    test: list[tuple[np.ndarray, str]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not train or not test or len({y for _, y in test}) < 2:
        return None
    models = {
        "root": SemanticSuffixTrie(max_order=0, alpha=1.0, min_support=1, outcomes=BINARY),
        "clock": SemanticSuffixTrie(
            max_order=1, alpha=1.0, min_support=args.min_support, outcomes=BINARY
        ),
        "clock_task": SemanticSuffixTrie(
            max_order=2, alpha=1.0, min_support=args.min_support, outcomes=BINARY
        ),
        "clock_words": SemanticSuffixTrie(
            max_order=2 + args.max_order,
            alpha=1.0,
            min_support=args.min_support,
            outcomes=BINARY,
        ),
    }
    targets = np.asarray([BINARY.index(y) for _, y in test])
    losses = {}
    for name, model in models.items():
        model.fit(train)
        losses[name] = log_loss(
            np.stack([model.posterior(h)[0] for h, _ in test]), targets
        )
    return {
        "queries": len(test),
        "positive_rate": float(targets.mean()),
        "root_log_loss": losses["root"],
        "clock_log_loss": losses["clock"],
        "clock_task_log_loss": losses["clock_task"],
        "clock_words_log_loss": losses["clock_words"],
        "clock_minus_root": losses["clock"] - losses["root"],
        "task_minus_clock": losses["clock_task"] - losses["clock"],
        "words_minus_clock": losses["clock_words"] - losses["clock_task"],
    }


def pooled(values: list[float], seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
    return {
        "fold_values": array.tolist(),
        "mean": float(array.mean()),
        "fold_bootstrap_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "folds_improved": int(np.sum(array < 0)),
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

    index = {}
    for episode in corpus.episodes:
        suite, task = corpus.tasks[episode.task_index].split("/", 1)
        index[(suite, task, episode.init_state_id, episode.flow_noise_seed)] = episode

    records = load_labels(args.labels, args.data_root)
    reason: dict[int, str] = {}
    onsets: dict[int, int] = {}
    for record in records:
        key = (
            record["suite"],
            record["task_name"],
            record["init_state_id"],
            record["flow_noise_seed"],
        )
        episode = index.get(key)
        if episode is None:
            raise ValueError(f"unjoined failure label {key}")
        reason[episode.index] = record["primary_failure_reason"]
        onset = release_query(record)
        if onset is not None and onset < episode.length:
            onsets[episode.index] = onset

    frequent = {name for name, count in Counter(reason.values()).items() if count >= MIN_TYPE_SUPPORT}
    types = sorted(frequent) + ["other"]
    type_index = {name: types.index(name if name in frequent else "other") for name in set(reason.values())}
    print(
        f"joined {len(reason)} typed failures, {len(onsets)} with a release query; "
        f"{len(types)} classes",
        flush=True,
    )

    failures = [e for e in corpus.episodes if e.index in reason]
    per_fold: dict[str, list[dict[str, Any]]] = {
        "failure_type": [],
        "release_within_release_episodes": [],
        "release_all_failures": [],
    }
    for fold in args.folds:
        model = joblib.load(Path(f"{args.model_root}{fold}") / "full40_model.joblib")
        projected = project_in_batches(
            corpus.features, corpus.feature_names, model["preprocessor"]
        )
        words, _, _ = tokenize_in_batches(projected, model["tokenizer"])
        del projected
        n_words = model["tokenizer"].n_words
        test_states = set(np.asarray(splits[fold]["test"]).tolist())
        train_eps = [e for e in failures if e.init_state_id not in test_states]
        test_eps = [e for e in failures if e.init_state_id in test_states]

        def pack(
            eps: list[Episode], assigned: np.ndarray = words
        ) -> list[tuple[np.ndarray, int, int]]:
            return [
                (
                    word_histogram(assigned, e, n_words),
                    type_index[reason[e.index]],
                    e.task_index,
                )
                for e in eps
            ]

        record = failure_type_fold(
            pack(train_eps), pack(test_eps), len(types), len(corpus.tasks), args.seed
        )
        if record:
            record["fold"] = fold
            per_fold["failure_type"].append(record)

        for name, only in (
            ("release_within_release_episodes", True),
            ("release_all_failures", False),
        ):
            row = release_fold(
                release_rows(train_eps, onsets, words, args.horizon, only),
                release_rows(test_eps, onsets, words, args.horizon, only),
                args,
            )
            if row:
                row["fold"] = fold
                per_fold[name].append(row)
        print(f"fold {fold}: done", flush=True)
        del words, model

    summary: dict[str, Any] = {
        "schema_version": 1,
        "counts": {
            "typed_failures": len(reason),
            "release_events": len(onsets),
            "classes": types,
            "type_distribution": dict(Counter(reason.values())),
        },
        "failure_type": {
            "folds": per_fold["failure_type"],
            "routing_minus_task": pooled(
                [r["routing_minus_task"] for r in per_fold["failure_type"]], args.seed
            ),
            "task_minus_prior": pooled(
                [r["task_minus_prior"] for r in per_fold["failure_type"]], args.seed
            ),
        },
        "protocol": {
            "note": (
                "All units are failure episodes, which all run to the run horizon, so "
                "episode length is constant and cannot separate the classes."
            ),
            "horizon": args.horizon,
            "min_type_support": MIN_TYPE_SUPPORT,
        },
        "elapsed_seconds": time.time() - started,
    }
    for name in ("release_within_release_episodes", "release_all_failures"):
        rows = per_fold[name]
        summary[name] = {
            "folds": rows,
            "words_minus_clock": pooled([r["words_minus_clock"] for r in rows], args.seed),
            "clock_minus_root": pooled([r["clock_minus_root"] for r in rows], args.seed),
            "mean_positive_rate": float(np.mean([r["positive_rate"] for r in rows])),
            "total_queries": int(sum(r["queries"] for r in rows)),
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    ft = summary["failure_type"]
    print(
        f"\nfailure_type  task-prior {ft['task_minus_prior']['mean']:+.4f} | "
        f"routing-task {ft['routing_minus_task']['mean']:+.4f} "
        f"CI[{ft['routing_minus_task']['fold_bootstrap_ci95'][0]:+.4f}, "
        f"{ft['routing_minus_task']['fold_bootstrap_ci95'][1]:+.4f}] "
        f"{ft['routing_minus_task']['folds_improved']}/{len(ft['folds'])} folds"
    )
    for name in ("release_within_release_episodes", "release_all_failures"):
        row = summary[name]
        print(
            f"{name:34s} n={row['total_queries']:6d} pos={row['mean_positive_rate']:.4f} "
            f"clock-root {row['clock_minus_root']['mean']:+.4f} | "
            f"words-clock {row['words_minus_clock']['mean']:+.4f} "
            f"CI[{row['words_minus_clock']['fold_bootstrap_ci95'][0]:+.4f}, "
            f"{row['words_minus_clock']['fold_bootstrap_ci95'][1]:+.4f}] "
            f"{row['words_minus_clock']['folds_improved']}/{len(row['folds'])} folds"
        )


if __name__ == "__main__":
    main()
