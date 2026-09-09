"""Extract the token co-routing structure that the scalar consensus features discard.

`extract_multitrack_features` builds the 11x11 Bhattacharyya matrix between token routing
distributions and then keeps only its mean upper triangle. That average is where the
state-versus-action distinction and the layer-to-layer correspondence are lost.

Expert indices are independently permutable in every layer, so cross-layer comparison of
expert coordinates is meaningless. The 11x11 matrix asks instead which tokens route
*together*, which is permutation-invariant by construction, so a front-layer matrix and a
back-layer matrix can be compared directly with no alignment step.

Quantities are exactly those fixed in `results-token-matching/PREREG.zh.md`; nothing is
added here. Token 0 is the state token and tokens 1..10 are the action tokens, following
the online reader in `candidate_reranking.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zarr

from moe_grammar.data import DEFAULT_HUB, discover_runs, validate_episode_axes

N_LAYERS = 8
N_FLOW = 10
N_TOKENS = 11
FRONT = (0, 1, 2, 3)
BACK = (4, 5, 6, 7)
UPPER = np.triu_indices(N_TOKENS, k=1)

PER_LAYER = ("sa_coupling", "aa_coupling", "sa_gap")
PER_FLOW = ("front_back", "within_front", "within_back")


def feature_names() -> tuple[str, ...]:
    names = [
        f"{metric}|layer_{layer}|flow_{flow}"
        for metric in PER_LAYER
        for layer in range(N_LAYERS)
        for flow in range(N_FLOW)
    ]
    names += [f"{metric}|flow_{flow}" for metric in PER_FLOW for flow in range(N_FLOW)]
    return tuple(names)


def _correlate(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation between two [n, 55] stacks of upper triangles."""
    left = left - left.mean(axis=-1, keepdims=True)
    right = right - right.mean(axis=-1, keepdims=True)
    numerator = (left * right).sum(axis=-1)
    denominator = np.sqrt((left * left).sum(axis=-1) * (right * right).sum(axis=-1))
    return np.where(denominator > 1e-12, numerator / np.maximum(denominator, 1e-12), 0.0)


def token_matching_features(probability: np.ndarray) -> np.ndarray:
    """Return ``[batch, len(feature_names())]`` from ``[batch, 8, 10, 11, 32]``."""
    values = np.asarray(probability, dtype=np.float32)
    if values.shape[1:] != (N_LAYERS, N_FLOW, N_TOKENS, values.shape[-1]):
        raise ValueError(f"unexpected router shape {values.shape}")
    values = np.clip(values, 0.0, None)
    values = values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)
    root = np.sqrt(values)
    # Bhattacharyya coefficient between every token pair, permutation-invariant.
    matrix = np.einsum("blfte,blfse->blfts", root, root)

    state = matrix[:, :, :, 0, 1:].mean(axis=-1)
    action_rows, action_cols = np.triu_indices(N_TOKENS - 1, k=1)
    action = matrix[:, :, :, 1:, 1:][:, :, :, action_rows, action_cols].mean(axis=-1)
    per_layer = np.stack([state, action, action - state], axis=0)

    flat = matrix[:, :, :, UPPER[0], UPPER[1]]
    batch = len(values)
    cross = np.empty((3, batch, N_FLOW), dtype=np.float32)
    for flow in range(N_FLOW):
        piece = flat[:, :, flow, :]

        def group_mean(first: tuple[int, ...], second: tuple[int, ...], same: bool) -> np.ndarray:
            scores = [
                _correlate(piece[:, a, :], piece[:, b, :])
                for a in first
                for b in second
                if not same or a < b
            ]
            return np.mean(np.stack(scores, axis=0), axis=0)

        cross[0, :, flow] = group_mean(FRONT, BACK, same=False)
        cross[1, :, flow] = group_mean(FRONT, FRONT, same=True)
        cross[2, :, flow] = group_mean(BACK, BACK, same=True)

    return np.concatenate(
        [per_layer.transpose(1, 0, 2, 3).reshape(batch, -1), cross.transpose(1, 0, 2).reshape(batch, -1)],
        axis=1,
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--cache-name", default="cache_new")
    parser.add_argument("--run-id", default="right-50x8*-20260903")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/token-matching"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.hub, args.run_id, args.cache_name)
    if args.num_shards:
        runs = runs[args.shard_index :: args.num_shards]
    names = feature_names()
    for run in runs:
        output = args.output_dir / f"{run.output_stem}.npz"
        if output.exists() and not args.force:
            print(f"skip {output.name}", flush=True)
            continue
        started = time.time()
        store = zarr.open_group(str(run.path / "server" / "routes.zarr"), mode="r")
        episode_id, episode_step, _ = validate_episode_axes(run, store)
        array = store["hb_router_probs"]
        count = array.shape[0]
        features = np.empty((count, len(names)), dtype=np.float32)
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            features[start:stop] = token_matching_features(np.asarray(array[start:stop]))
            if stop % (args.batch_size * 8) == 0 or stop == count:
                print(f"  {run.key}: {stop:,}/{count:,}", flush=True)
        if not np.isfinite(features).all():
            raise ValueError(f"non-finite token-matching feature in {run.key}")
        np.savez_compressed(
            output,
            features=features,
            feature_names=np.asarray(names),
            episode_id=episode_id,
            episode_step=episode_step,
            metadata_json=json.dumps(
                {
                    "run_key": run.key,
                    "task_key": run.key,
                    "run_id": run.run_id,
                    "state_token": 0,
                    "action_tokens": list(range(1, N_TOKENS)),
                    "front_layers": list(FRONT),
                    "back_layers": list(BACK),
                }
            ),
        )
        print(f"wrote {output.name} ({time.time() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
