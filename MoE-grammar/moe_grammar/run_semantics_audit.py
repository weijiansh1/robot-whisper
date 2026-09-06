"""Fourth layer audit: does a grammar context carry outcome semantics?

Every earlier audit took a grammar surprisal and regressed it against the final episode
outcome. `idea.md` asks for something different and more local: attach outcome counts to
suffix-tree contexts and read off what a sentence implies about the next few queries.

The decisive comparison here is context-conditioned `mu(c, h)` against the same model
with the context removed. If the trie cannot beat its own root distribution, grammar
contexts carry no outcome semantics and the layer is empty regardless of how well the
grammar predicts the next word.

Intervention semantics stops at the arm level on purpose: the fork corpus has only a
handful of independent trunks, so `P(Y | c, do(u))` conditioned on routing context is not
estimable and is not claimed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
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
from moe_grammar.semantics import (
    OUTCOMES,
    InterventionTable,
    SemanticSuffixTrie,
    label_outcomes,
)

WORKSPACE = Path("/home/jovyan/work/himoe-vla")
RECOVERY_ROOT = WORKSPACE / "trap-recovery-depth-20260904"
RECOVERY_COHORTS = ("runs", "runs_cap52", "runs_long", "runs_probe", "runs_routed")
FORMAL_RUN = re.compile(r"^i\d+_s\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40-v2"))
    parser.add_argument("--model-root", default="results-full40-v2-fold")
    parser.add_argument("--output-dir", type=Path, default=Path("results-semantics-v1"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--deviation-percentile", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def log_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
    picked = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.maximum(picked, 1e-12)).mean())


def brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(targets)), targets] = 1.0
    return float(np.square(probabilities - onehot).sum(axis=1).mean())


def build_observations(
    episodes: list[Any],
    words: np.ndarray,
    deviation: np.ndarray,
    horizon: int,
    threshold: float,
) -> list[tuple[np.ndarray, str, int]]:
    """Return ``(history, outcome, init_state)`` for every deviating query."""
    output: list[tuple[np.ndarray, str, int]] = []
    for episode in episodes:
        for offset, outcome in label_outcomes(
            deviation, episode.start, episode.length, horizon, episode.success, threshold
        ):
            # The query index is appended so the trie's own suffix backoff yields the
            # nesting root -> clock -> clock+words. Without a clock control, any phase
            # information carried by the words reads as a semantic result.
            history = np.append(
                np.asarray(words[episode.start : episode.start + offset], dtype=np.int64), offset
            )
            output.append((history, outcome, episode.init_state_id))
    return output


def scan_intervention_corpus() -> tuple[InterventionTable, dict[str, Any]]:
    table = InterventionTable()
    runs = 0
    for cohort in RECOVERY_COHORTS:
        root = RECOVERY_ROOT / cohort
        if not root.exists():
            continue
        for manifest_path in sorted(root.glob("*/manifest.json")):
            run_path = manifest_path.parent
            if not FORMAL_RUN.fullmatch(run_path.name):
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete" or manifest.get("trunk", {}).get("success"):
                continue
            config = json.loads((run_path / "experiment_config.json").read_text(encoding="utf-8"))
            trunk = f"{cohort}/{run_path.name}/state{config['init_state_id']}"
            observed = False
            for arm_path in sorted(p for p in run_path.iterdir() if p.is_dir()):
                for record_path in sorted(arm_path.glob("candidate_*.json")):
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    table.add(trunk, arm_path.name, bool(record["success"]))
                    observed = True
            runs += int(observed)
    return table, {
        "complete_failed_trunk_runs": runs,
        "independent_trunks": len(table.trunks),
        "arms": list(table.arms),
    }


def evaluate_fold(
    corpus: Any, split: dict[str, Any], model: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    train = select_episodes(corpus, split["train"], None)
    test = select_episodes(corpus, split["test"], None)

    projected = project_in_batches(corpus.features, corpus.feature_names, model["preprocessor"])
    words, emissions, _ = tokenize_in_batches(projected, model["tokenizer"])
    del projected

    print("Scoring grammar surprisal...", flush=True)
    surprisal = np.empty(len(words), dtype=np.float32)
    for episode in corpus.episodes:
        selected = slice(episode.start, episode.stop)
        values, _ = model["task_models"][episode.task_index]["history"].continuous_nll(
            words[selected], emissions[selected]
        )
        surprisal[selected] = values.astype(np.float32)
    del emissions

    # "Currently deviating" is the calibrated upper tail of the grammar's own surprisal.
    reference = np.sort(
        np.concatenate(
            [surprisal[e.start : e.stop] for e in train if e.success]
        ).astype(np.float64)
    )
    deviation = np.searchsorted(reference, surprisal, side="right") / len(reference)
    threshold = args.deviation_percentile

    train_rows = build_observations(train, words, deviation, args.horizon, threshold)
    test_rows = build_observations(test, words, deviation, args.horizon, threshold)
    print(f"deviating queries: train {len(train_rows):,}, test {len(test_rows):,}", flush=True)
    if not train_rows or not test_rows:
        raise ValueError("no deviating queries at the requested threshold")

    observations = [(history, outcome) for history, outcome, _ in train_rows]
    trie = SemanticSuffixTrie(
        max_order=1 + args.max_order, alpha=1.0, min_support=args.min_support
    ).fit(observations)
    clock = SemanticSuffixTrie(max_order=1, alpha=1.0, min_support=args.min_support).fit(
        observations
    )
    root = SemanticSuffixTrie(max_order=0, alpha=1.0, min_support=1).fit(observations)

    targets = np.asarray([OUTCOMES.index(outcome) for _, outcome, _ in test_rows])
    states = np.asarray([state for _, _, state in test_rows], dtype=np.int16)
    context_probabilities = np.stack([trie.posterior(h)[0] for h, _, _ in test_rows])
    clock_probabilities = np.stack([clock.posterior(h)[0] for h, _, _ in test_rows])
    root_probabilities = np.stack([root.posterior(h)[0] for h, _, _ in test_rows])
    orders = np.asarray([trie.posterior(h)[1] for h, _, _ in test_rows])

    # State-blocked paired difference: does context beat no context per init state?
    per_state = []
    for state in np.unique(states):
        mask = states == state
        per_state.append(
            log_loss(context_probabilities[mask], targets[mask])
            - log_loss(clock_probabilities[mask], targets[mask])
        )
    per_state = np.asarray(per_state)
    rng = np.random.default_rng(args.seed)
    draws = per_state[rng.integers(0, len(per_state), size=(5000, len(per_state)))].mean(axis=1)

    # Part of the full-alphabet gain could just be "this context means the episode is
    # about to end". Restricting to return-vs-persist removes the episode-length signal
    # and isolates the Correction->Return versus Correction->Persist distinction the
    # design actually asks for.
    binary = np.isin(targets, [OUTCOMES.index("return"), OUTCOMES.index("persist")])
    columns = [OUTCOMES.index("return"), OUTCOMES.index("persist")]
    binary_targets = (targets[binary] == OUTCOMES.index("persist")).astype(np.int64)

    def renormalized(probabilities: np.ndarray) -> np.ndarray:
        restricted = probabilities[binary][:, columns]
        return restricted / np.maximum(restricted.sum(axis=1, keepdims=True), 1e-12)

    binary_context = renormalized(context_probabilities)
    binary_root = renormalized(clock_probabilities)
    binary_states = states[binary]
    binary_per_state = np.asarray(
        [
            log_loss(binary_context[binary_states == state], binary_targets[binary_states == state])
            - log_loss(binary_root[binary_states == state], binary_targets[binary_states == state])
            for state in np.unique(binary_states)
        ]
    )
    binary_draws = binary_per_state[
        rng.integers(0, len(binary_per_state), size=(5000, len(binary_per_state)))
    ].mean(axis=1)

    return {
        "horizon": args.horizon,
        "return_vs_persist": {
            "queries": int(binary.sum()),
            "persist_base_rate": float(binary_targets.mean()),
            "context_log_loss": log_loss(binary_context, binary_targets),
            "root_log_loss": log_loss(binary_root, binary_targets),
            "context_minus_root_log_loss": float(binary_per_state.mean()),
            "context_minus_root_ci95": [
                float(np.quantile(binary_draws, 0.025)),
                float(np.quantile(binary_draws, 0.975)),
            ],
            "states_improved": int(np.sum(binary_per_state < 0)),
        },
        "deviating_test_queries": len(test_rows),
        "outcome_distribution": {
            outcome: int(np.sum(targets == index)) for index, outcome in enumerate(OUTCOMES)
        },
        "context_log_loss": log_loss(context_probabilities, targets),
        "clock_log_loss": log_loss(clock_probabilities, targets),
        "root_log_loss": log_loss(root_probabilities, targets),
        "clock_minus_root_log_loss": (
            log_loss(clock_probabilities, targets) - log_loss(root_probabilities, targets)
        ),
        "context_brier": brier(context_probabilities, targets),
        "root_brier": brier(root_probabilities, targets),
        "context_minus_root_log_loss": float(per_state.mean()),
        "context_minus_root_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "states": int(len(per_state)),
        "states_improved": int(np.sum(per_state < 0)),
        "backoff_order_share": {
            str(order): float(np.mean(orders == order)) for order in range(args.max_order + 1)
        },
    }



def aggregate(folds: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Pool per-fold effects; every init state is tested exactly once across folds."""

    def pooled(path: list[str]) -> dict[str, Any]:
        values = np.asarray([_dig(fold, path) for fold in folds], dtype=float)
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
        return {
            "fold_values": values.tolist(),
            "mean": float(values.mean()),
            "fold_bootstrap_ci95": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            "folds_improved": int(np.sum(values < 0)),
        }

    return {
        "full_alphabet_context_minus_root": pooled(["context_minus_root_log_loss"]),
        "return_vs_persist_context_minus_root": pooled(
            ["return_vs_persist", "context_minus_root_log_loss"]
        ),
        "deviating_test_queries": int(
            sum(_dig(fold, ["deviating_test_queries"]) for fold in folds)
        ),
    }


def _dig(record: dict[str, Any], path: list[str]) -> Any:
    for key in path:
        record = record[key]
    return record


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("Loading full-40 corpus...", flush=True)
    corpus = load_corpus(
        args.features_dir,
        None,
        allow_task_segments=True,
        require_flow_shuffled=False,
        require_stasis_labels=False,
        feature_dtype=np.float16,
    )
    splits = make_state_folds(
        np.asarray([episode.init_state_id for episode in corpus.episodes]), 5, args.seed
    )

    per_fold: list[dict[str, Any]] = []
    for fold in args.folds:
        print(f"=== fold {fold} ===", flush=True)
        model = joblib.load(Path(f"{args.model_root}{fold}") / "full40_model.joblib")
        record = evaluate_fold(corpus, splits[fold], model, args)
        record["fold"] = fold
        per_fold.append(record)
        del model

    table, corpus_counts = scan_intervention_corpus()
    intervention: dict[str, Any] = {"corpus": corpus_counts, "arms": {}}
    for arm in table.arms:
        interval = table.trunk_blocked_interval(arm)
        contrast = table.contrast_interval(arm, "control") if arm != "control" else None
        intervention["arms"][arm] = {
            "success_rate": table.posterior_mean(arm),
            "trunk_blocked_ci95": list(interval) if interval else None,
            "minus_control": (
                {"difference": contrast[0], "ci95": list(contrast[1])} if contrast else None
            ),
        }

    summary = {
        "schema_version": 2,
        "predictive_semantics_by_fold": per_fold,
        "predictive_semantics_pooled": aggregate(per_fold, args.seed),
        "intervention_semantics": intervention,
        "protocol": {
            "folds": args.folds,
            "horizon": args.horizon,
            "max_order": args.max_order,
            "min_support": args.min_support,
            "deviation_percentile": args.deviation_percentile,
            "note": (
                "Intervention semantics is arm-level only; the fork corpus has too few "
                "independent trunks to condition on routing context."
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pooled = summary["predictive_semantics_pooled"]
    for name in ("full_alphabet_context_minus_root", "return_vs_persist_context_minus_root"):
        row = pooled[name]
        print(
            f"{name}: {row['mean']:+.4f} "
            f"[{row['fold_bootstrap_ci95'][0]:+.4f}, {row['fold_bootstrap_ci95'][1]:+.4f}] "
            f"{row['folds_improved']}/{len(row['fold_values'])} folds"
        )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
