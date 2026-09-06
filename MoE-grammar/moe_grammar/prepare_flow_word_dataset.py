from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from moe_grammar.corpus import load_corpus
from moe_grammar.run_open_world_audit import load_dataset


PRIMITIVE_NAMES = (
    "entropy",
    "margin",
    "top1_mass",
    "top4_mass",
    "soft_token_consensus",
    "top4_token_consensus",
    "effective_rank",
    "flow_velocity",
    "flow_top4_switch",
    "flow_acceleration",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-features", type=Path, default=Path("artifacts/features-full40-v2")
    )
    parser.add_argument(
        "--anchor-features",
        type=Path,
        default=Path(
            "artifacts/features/"
            "libero_long__KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.npz"
        ),
    )
    parser.add_argument(
        "--structure", type=Path, default=Path("artifacts/open-world-phenotypes-v2.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/flow-word-primitives-v1.npz")
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32768)
    return parser.parse_args()


@torch.inference_mode()
def extract_primitives(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    indexes: list[int] = []
    for metric in PRIMITIVE_NAMES:
        for layer in (6, 7):
            name = f"{metric}|layer_{layer}"
            if name not in feature_names:
                raise ValueError(f"missing flow primitive {name}")
            indexes.append(feature_names.index(name))
    index = torch.as_tensor(indexes, dtype=torch.long, device=device)
    output = np.empty((len(features), 10, len(PRIMITIVE_NAMES)), dtype=np.float16)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        batch = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
        selected = torch.index_select(batch, dim=2, index=index)
        selected = selected.reshape(stop - start, 10, len(PRIMITIVE_NAMES), 2).mean(dim=3)
        output[start:stop] = selected.cpu().numpy().astype(np.float16)
    return output


def episode_axes(corpus: object) -> tuple[np.ndarray, ...]:
    episodes = corpus.episodes
    return (
        np.asarray([episode.start for episode in episodes], dtype=np.int32),
        np.asarray([episode.length for episode in episodes], dtype=np.int16),
        np.asarray([episode.init_state_id for episode in episodes], dtype=np.int16),
        np.asarray([episode.flow_noise_seed for episode in episodes], dtype=np.int16),
        np.asarray([episode.success for episode in episodes], dtype=np.bool_),
        np.asarray([episode.task_index for episode in episodes], dtype=np.int16),
    )


def assert_equal(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if not np.array_equal(left, right):
        raise AssertionError(f"{name} does not align with the compact structure artifact")


def main() -> None:
    args = parse_args()
    structure, structure_metadata = load_dataset(args.structure)
    print("loading corrected full40 flow tensors", flush=True)
    full = load_corpus(
        args.full_features,
        stasis_labels=None,
        allow_task_segments=True,
        require_flow_shuffled=False,
        require_stasis_labels=False,
        feature_dtype=np.float16,
    )
    full_axes = episode_axes(full)
    for name, observed, expected in zip(
        ("starts", "lengths", "state", "seed", "success", "task"),
        full_axes,
        (
            structure["full_starts"],
            structure["full_lengths"],
            structure["full_state"],
            structure["full_seed"],
            structure["full_success"],
            structure["full_task"],
        ),
        strict=True,
    ):
        assert_equal(f"full_{name}", observed, expected)
    print(f"projecting {len(full.features):,} full40 queries on {args.device}", flush=True)
    full_primitives = extract_primitives(
        full.features,
        full.feature_names,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("loading and projecting Scene8 anchor flow tensors", flush=True)
    with np.load(args.anchor_features, allow_pickle=False) as payload:
        anchor_features = np.asarray(payload["features"], dtype=np.float16)
        anchor_names = tuple(str(item) for item in payload["feature_names"])
        anchor_episode = np.asarray(payload["episode_id"], dtype=np.int64)
        anchor_state = np.asarray(payload["init_state_id"], dtype=np.int16)
        anchor_seed = np.asarray(payload["flow_noise_seed"], dtype=np.int16)
        anchor_success = np.asarray(payload["success"], dtype=np.bool_)
    unique_episode, anchor_starts = np.unique(anchor_episode, return_index=True)
    if not np.array_equal(unique_episode, np.arange(len(unique_episode))):
        raise AssertionError("anchor episode IDs are not contiguous")
    anchor_lengths = np.diff(np.r_[anchor_starts, len(anchor_episode)])
    assert_equal("anchor_starts", anchor_starts, structure["anchor_starts"])
    assert_equal("anchor_lengths", anchor_lengths, structure["anchor_lengths"])
    assert_equal("anchor_state", anchor_state[anchor_starts], structure["anchor_state"])
    assert_equal("anchor_seed", anchor_seed[anchor_starts], structure["anchor_seed"])
    assert_equal("anchor_success", anchor_success[anchor_starts], structure["anchor_success"])
    anchor_primitives = extract_primitives(
        anchor_features,
        anchor_names,
        device=args.device,
        batch_size=args.batch_size,
    )

    metadata = {
        "schema_version": 1,
        "definition": "late-layer flow-step primitives for 10-chord query words",
        "primitive_names": list(PRIMITIVE_NAMES),
        "layers": [6, 7],
        "full_queries": len(full_primitives),
        "anchor_queries": len(anchor_primitives),
        "source_structure": str(args.structure),
        "source_definition": structure_metadata["definition"],
        "full_feature_dir": str(args.full_features),
        "anchor_feature_file": str(args.anchor_features),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        full_primitives=full_primitives,
        anchor_primitives=anchor_primitives,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        f"wrote {args.output}: full={full_primitives.shape}, anchor={anchor_primitives.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
