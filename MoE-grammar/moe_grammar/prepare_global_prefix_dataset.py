from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from moe_grammar.corpus import load_corpus
from moe_grammar.run_full40_audit import project_in_batches, tokenize_in_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=Path("artifacts/features-full40")
    )
    parser.add_argument("--grammar-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading corpus for {args.grammar_model}...", flush=True)
    corpus = load_corpus(
        args.features_dir,
        None,
        allow_task_segments=True,
        require_flow_shuffled=False,
        require_stasis_labels=False,
        feature_dtype=np.float16,
    )
    artifact = joblib.load(args.grammar_model)
    if tuple(artifact["tasks"]) != corpus.tasks:
        raise ValueError("grammar-model task order does not match corpus")

    print("Projecting and tokenizing every query with this fold's train-only model...", flush=True)
    projected = project_in_batches(
        corpus.features, corpus.feature_names, artifact["preprocessor"]
    )
    words, emissions, lexical_log_likelihood = tokenize_in_batches(
        projected, artifact["tokenizer"]
    )

    episode_count = len(corpus.episodes)
    maximum_length = max(episode.length for episode in corpus.episodes)
    shape = (episode_count, maximum_length)
    phase_soft = np.full(shape, np.nan, dtype=np.float32)
    local2_soft = np.full(shape, np.nan, dtype=np.float32)
    phase_hard = np.full(shape, np.nan, dtype=np.float32)
    local2_hard = np.full(shape, np.nan, dtype=np.float32)
    lexical = np.full(shape, np.nan, dtype=np.float32)

    starts = np.empty(episode_count, dtype=np.int32)
    lengths = np.empty(episode_count, dtype=np.int16)
    task = np.empty(episode_count, dtype=np.int16)
    state = np.empty(episode_count, dtype=np.int16)
    flow_noise_seed = np.empty(episode_count, dtype=np.int16)
    success = np.empty(episode_count, dtype=np.bool_)

    print("Scoring position and local-two-word controls...", flush=True)
    for index, episode in enumerate(corpus.episodes):
        selected = slice(episode.start, episode.stop)
        sequence = words[selected]
        component = emissions[selected]
        phase = artifact["task_models"][episode.task_index]["phase"]
        local2 = artifact["task_models"][episode.task_index]["history"]
        phase_soft[episode.index, : episode.length] = phase.continuous_nll(
            sequence, component
        )[0]
        local2_soft[episode.index, : episode.length] = local2.continuous_nll(
            sequence, component
        )[0]
        phase_hard[episode.index, : episode.length] = phase.hard_nll(
            sequence, include_end=False
        )[0]
        local2_hard[episode.index, : episode.length] = local2.hard_nll(
            sequence, include_end=False
        )[0]
        lexical[episode.index, : episode.length] = -lexical_log_likelihood[selected]
        starts[episode.index] = episode.start
        lengths[episode.index] = episode.length
        task[episode.index] = episode.task_index
        state[episode.index] = episode.init_state_id
        flow_noise_seed[episode.index] = episode.flow_noise_seed
        success[episode.index] = episode.success
        if (index + 1) % 4000 == 0:
            print(f"  scored {index + 1:,}/{episode_count:,} episodes", flush=True)

    metadata = {
        "schema_version": 1,
        "grammar_model": str(args.grammar_model),
        "descriptor_dim": int(projected.shape[1]),
        "n_words": int(artifact["tokenizer"].n_words),
        "max_episode_length": int(maximum_length),
        "tasks": list(corpus.tasks),
        "split": {name: values.tolist() for name, values in artifact["split"].items()},
        "causality": (
            "projected row q is observed at q; shifted language-model construction is "
            "performed during training"
        ),
        "controls": {
            "phase": "task + absolute query position",
            "local2": "task + absolute query position + latest two ordered words",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        projected=projected.astype(np.float16),
        words=words,
        starts=starts,
        lengths=lengths,
        task=task,
        state=state,
        flow_noise_seed=flow_noise_seed,
        success=success,
        phase_soft=phase_soft,
        local2_soft=local2_soft,
        phase_hard=phase_hard,
        local2_hard=local2_hard,
        lexical=lexical,
        component_means=np.asarray(artifact["tokenizer"].model.means_, dtype=np.float32),
        component_covariances=np.asarray(
            artifact["tokenizer"].model.covariances_, dtype=np.float32
        ),
        component_weights=np.asarray(
            artifact["tokenizer"].model.weights_, dtype=np.float32
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        f"Wrote {args.output}: {episode_count:,} episodes, {len(projected):,} queries, "
        f"{args.output.stat().st_size / 2**20:.1f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
