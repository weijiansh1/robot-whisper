from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moe_grammar.corpus import SCENE8_TASK, load_corpus
from moe_grammar.open_world import PHENOTYPE_NAMES, phenotypes_in_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-features", type=Path, default=Path("artifacts/features-full40"))
    parser.add_argument("--anchor-features", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/open-world-phenotypes.npz"))
    return parser.parse_args()


def episode_arrays(corpus: object) -> dict[str, np.ndarray]:
    episodes = corpus.episodes
    return {
        "starts": np.asarray([episode.start for episode in episodes], dtype=np.int32),
        "lengths": np.asarray([episode.length for episode in episodes], dtype=np.int16),
        "task": np.asarray([episode.task_index for episode in episodes], dtype=np.int16),
        "state": np.asarray([episode.init_state_id for episode in episodes], dtype=np.int16),
        "seed": np.asarray([episode.flow_noise_seed for episode in episodes], dtype=np.int16),
        "success": np.asarray([episode.success for episode in episodes], dtype=np.bool_),
    }


def query_progress(starts: np.ndarray, lengths: np.ndarray, size: int) -> np.ndarray:
    output = np.empty(size, dtype=np.float32)
    for start, length in zip(starts, lengths):
        start = int(start)
        length = int(length)
        output[start : start + length] = np.arange(length) / max(length - 1, 1)
    return output


def main() -> None:
    args = parse_args()
    print("Loading full-40 routing chords...", flush=True)
    full = load_corpus(
        args.full_features,
        None,
        allow_task_segments=True,
        require_flow_shuffled=False,
        require_stasis_labels=False,
        feature_dtype=np.float16,
    )
    print("Building interpretable phenotypes for 508k queries...", flush=True)
    full_base = phenotypes_in_batches(full.features, full.feature_names)
    full_episode = episode_arrays(full)
    full_progress = query_progress(full_episode["starts"], full_episode["lengths"], len(full_base))

    print("Loading the physically labeled Scene8 anchor corpus...", flush=True)
    anchor_corpus = load_corpus(args.anchor_features, feature_dtype=np.float16)
    if SCENE8_TASK not in full.tasks or SCENE8_TASK not in anchor_corpus.tasks:
        raise ValueError("Scene8 task is absent from one corpus")
    anchor_episodes = [episode for episode in anchor_corpus.episodes if episode.task == SCENE8_TASK]
    anchor_parts = []
    anchor_starts = []
    offset = 0
    for episode in anchor_episodes:
        anchor_parts.append(anchor_corpus.features[episode.start : episode.stop])
        anchor_starts.append(offset)
        offset += episode.length
    anchor_chords = np.concatenate(anchor_parts)
    anchor_base = phenotypes_in_batches(anchor_chords, anchor_corpus.feature_names)
    anchor_lengths = np.asarray([episode.length for episode in anchor_episodes], dtype=np.int16)
    anchor_starts_array = np.asarray(anchor_starts, dtype=np.int32)
    anchor_progress = query_progress(anchor_starts_array, anchor_lengths, len(anchor_base))
    anchor_task_index = full.tasks.index(SCENE8_TASK)

    metadata = {
        "schema_version": 2,
        "definition": "independent-layer-permutation-invariant-moe-phenotype-v2",
        "phenotype_names": list(PHENOTYPE_NAMES),
        "tasks": list(full.tasks),
        "full_source_files": list(full.source_files),
        "anchor_source_files": list(anchor_corpus.source_files),
        "anchor_task": SCENE8_TASK,
        "anchor_definition": (
            "physical no-further-progress onset; MoE routing is not used to make labels"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        full_base=full_base.astype(np.float16),
        full_query_task=full.query_task_index,
        full_query_progress=full_progress.astype(np.float16),
        full_starts=full_episode["starts"],
        full_lengths=full_episode["lengths"],
        full_task=full_episode["task"],
        full_state=full_episode["state"],
        full_seed=full_episode["seed"],
        full_success=full_episode["success"],
        anchor_base=anchor_base.astype(np.float16),
        anchor_query_task=np.full(len(anchor_base), anchor_task_index, dtype=np.int16),
        anchor_query_progress=anchor_progress.astype(np.float16),
        anchor_starts=anchor_starts_array,
        anchor_lengths=anchor_lengths,
        anchor_task=np.full(len(anchor_episodes), anchor_task_index, dtype=np.int16),
        anchor_state=np.asarray(
            [episode.init_state_id for episode in anchor_episodes], dtype=np.int16
        ),
        anchor_seed=np.asarray(
            [episode.flow_noise_seed for episode in anchor_episodes], dtype=np.int16
        ),
        anchor_success=np.asarray([episode.success for episode in anchor_episodes], dtype=np.bool_),
        anchor_stasis_onset=np.asarray(
            [episode.stasis_onset for episode in anchor_episodes], dtype=np.int16
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        f"Wrote {args.output}: full={full_base.shape}, anchor={anchor_base.shape}, "
        f"{args.output.stat().st_size / 2**20:.1f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
