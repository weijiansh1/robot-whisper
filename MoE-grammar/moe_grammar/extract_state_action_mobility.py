"""Cross-query routing change, computed separately for the state token and the actions.

Gate three of the feedback chain: a world change is only visible to a routing-only monitor
if it survives observation, is used by the model, *and* leaves a mark on the gate. This
extractor produces the quantity needed to test the last step against a physically
annotated event, namely the moment the gripper releases the object.

Token 0 is the state token and tokens 1..10 are the action tokens
(`n_suffix = n_action_steps + 1`, `STATE_TOKEN = 0` in `himoe-route-capture/within64_lib.py`).
That is the token's input origin, not proof of what it encodes after attention, which is
exactly what the release-locked comparison is meant to establish.

For every query, layer and denoise step we store the Hellinger distance to the previous
query of the same episode, once for the state token and once for the action-token mean.
Query 0 of an episode has no predecessor and is stored as NaN.
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
STATE_TOKEN = 0
ACTION_TOKENS = slice(1, 11)


def channel_roots(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sqrt of the state-token and action-mean distributions."""
    values = np.clip(np.asarray(probability, dtype=np.float32), 0.0, None)
    values = values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)
    state = values[:, :, :, STATE_TOKEN, :]
    action = values[:, :, :, ACTION_TOKENS, :].mean(axis=3)
    action = action / np.maximum(action.sum(axis=-1, keepdims=True), 1e-12)
    return np.sqrt(state), np.sqrt(action)


def hellinger(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    coefficient = np.clip((current * previous).sum(axis=-1), 0.0, 1.0)
    return np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--cache-name", default="cache_new")
    parser.add_argument("--run-id", default="right-50x8*-20260903")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/state-action-mobility"))
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
        state_move = np.full((count, N_LAYERS, N_FLOW), np.nan, dtype=np.float32)
        action_move = np.full((count, N_LAYERS, N_FLOW), np.nan, dtype=np.float32)
        # One row of overlap per batch so an episode spanning a boundary keeps its link.
        previous_state = previous_action = None
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            state, action = channel_roots(np.asarray(array[start:stop]))
            if previous_state is not None:
                state = np.concatenate([previous_state, state])
                action = np.concatenate([previous_action, action])
                offset = start - 1
            else:
                offset = start
            moves_state = hellinger(state[1:], state[:-1])
            moves_action = hellinger(action[1:], action[:-1])
            state_move[offset + 1 : stop] = moves_state
            action_move[offset + 1 : stop] = moves_action
            previous_state, previous_action = state[-1:], action[-1:]
        # A query that starts an episode has no predecessor inside that episode.
        boundary = np.flatnonzero(episode_step == 0)
        state_move[boundary] = np.nan
        action_move[boundary] = np.nan
        np.savez_compressed(
            output,
            state_move=state_move,
            action_move=action_move,
            episode_id=episode_id,
            episode_step=episode_step,
            metadata_json=json.dumps(
                {"run_key": run.key, "run_id": run.run_id, "state_token": STATE_TOKEN}
            ),
        )
        print(f"wrote {output.name} ({time.time() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
