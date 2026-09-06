from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from moe_grammar.corpus import load_corpus
from moe_grammar.run_full40_audit import project_in_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40"))
    parser.add_argument("--grammar-model", type=Path, default=Path("results-full40/full40_model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/full40-prefix-q12.npz"))
    parser.add_argument("--max-queries", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading corpus and train-only projection...", flush=True)
    corpus = load_corpus(
        args.features_dir,
        None,
        allow_task_segments=True,
        require_flow_shuffled=False,
        require_stasis_labels=False,
        feature_dtype=np.float16,
    )
    artifact = joblib.load(args.grammar_model)
    projected = project_in_batches(
        corpus.features, corpus.feature_names, artifact["preprocessor"]
    )

    episode_count = len(corpus.episodes)
    route_dimensions = projected.shape[1] + 3 * corpus.features.shape[2]
    routing = np.zeros(
        (episode_count, args.max_queries, route_dimensions), dtype=np.float16
    )
    behavior = np.zeros(
        (episode_count, args.max_queries, corpus.behavior_features.shape[1]),
        dtype=np.float16,
    )
    lengths = np.empty(episode_count, dtype=np.int16)
    task = np.empty(episode_count, dtype=np.int16)
    state = np.empty(episode_count, dtype=np.int16)
    seed = np.empty(episode_count, dtype=np.int16)
    success = np.empty(episode_count, dtype=np.bool_)

    for episode in corpus.episodes:
        count = min(args.max_queries, episode.length)
        selected = slice(episode.start, episode.start + count)
        chords = np.asarray(corpus.features[selected], dtype=np.float32)
        level = chords.mean(axis=1)
        terminal = chords[:, -1]
        flow_trend = chords[:, 7:10].mean(axis=1) - chords[:, :3].mean(axis=1)
        routing[episode.index, :count] = np.concatenate(
            [projected[selected], level, terminal, flow_trend], axis=1
        ).astype(np.float16)
        behavior[episode.index, :count] = corpus.behavior_features[selected].astype(np.float16)
        lengths[episode.index] = episode.length
        task[episode.index] = episode.task_index
        state[episode.index] = episode.init_state_id
        seed[episode.index] = episode.flow_noise_seed
        success[episode.index] = episode.success

    metadata = {
        "schema_version": 1,
        "max_queries": args.max_queries,
        "routing_features": (
            "train-only PCA24 + flow-mean81 + terminal-flow81 + late-minus-early-flow81"
        ),
        "routing_dimensions": route_dimensions,
        "behavior_dimensions": behavior.shape[2],
        "tasks": list(corpus.tasks),
        "split": {name: values.tolist() for name, values in artifact["split"].items()},
        "causality": "row q contains only routing and behavior observed for query q",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        routing=routing,
        behavior=behavior,
        lengths=lengths,
        task=task,
        state=state,
        flow_noise_seed=seed,
        success=success,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        f"Wrote {args.output}: {episode_count:,} episodes, routing={routing.shape}, "
        f"{args.output.stat().st_size / 2**20:.1f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
