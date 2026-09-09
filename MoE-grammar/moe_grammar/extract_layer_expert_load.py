"""Per-query layer-by-expert load, for co-activation analysis.

The existing features collapse the router distribution into per-layer scalars such as
entropy, which measures only how peaked a layer is. That is one number per layer and it
cannot express *which* experts are up together, or whether a lift in one layer co-occurs
with a lift in another.

Permutation note. Experts are independently permutable across layers, so a distance
between layer a's 32-vector and layer b's 32-vector is meaningless -- that was the
`layer_disagreement` defect. Reading the flattened 8x32 matrix across queries of one
*fixed* trained model is a different operation: expert 7 of layer 2 is the same expert in
every query, so a covariance learned across queries is well defined. The resulting
components are specific to this checkpoint and do not transfer to another model.

Stored per query: the router probability averaged over the ten denoise steps, once over
the action tokens and once for the state token, as two 8x32 matrices.
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
N_EXPERTS = 32
STATE_TOKEN = 0
ACTION_TOKENS = slice(1, 11)


def layer_expert_load(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return action-mean and state-token ``[batch, 8, 32]`` loads."""
    values = np.clip(np.asarray(probability, dtype=np.float32), 0.0, None)
    values = values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)
    action = values[:, :, :, ACTION_TOKENS, :].mean(axis=(2, 3))
    state = values[:, :, :, STATE_TOKEN, :].mean(axis=2)
    return action, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--cache-name", default="cache_new")
    parser.add_argument("--run-id", default="right-50x8*-20260903")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/layer-expert-load"))
    parser.add_argument("--batch-size", type=int, default=4096)
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
        action = np.empty((count, N_LAYERS, N_EXPERTS), dtype=np.float32)
        state = np.empty((count, N_LAYERS, N_EXPERTS), dtype=np.float32)
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            action[start:stop], state[start:stop] = layer_expert_load(
                np.asarray(array[start:stop])
            )
        if not (np.isfinite(action).all() and np.isfinite(state).all()):
            raise ValueError(f"non-finite expert load in {run.key}")
        np.savez_compressed(
            output,
            action_load=action,
            state_load=state,
            episode_id=episode_id,
            episode_step=episode_step,
            metadata_json=json.dumps({"run_key": run.key, "run_id": run.run_id}),
        )
        print(f"wrote {output.name} ({time.time() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
