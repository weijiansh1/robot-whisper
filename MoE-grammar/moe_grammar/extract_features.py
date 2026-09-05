from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import zarr

from moe_grammar.data import (
    DEFAULT_HUB,
    RUN_ID,
    discover_runs,
    load_behavior_features,
    query_metadata,
    validate_episode_axes,
)
from moe_grammar.features import (
    deterministic_flow_permutations,
    extract_multitrack_features,
    permute_flow_tensor,
    step_feature_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument(
        "--task-indexes",
        help="Comma-separated indexes from --list; omitted means every discovered task",
    )
    parser.add_argument("--list", action="store_true", help="List discovered task indexes and exit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--shuffle-seed", type=int, default=20260905)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_indexes(value: str | None, count: int) -> list[int]:
    if value is None:
        return list(range(count))
    indexes = sorted({int(item) for item in value.split(",") if item.strip()})
    if not indexes or indexes[0] < 0 or indexes[-1] >= count:
        raise ValueError(f"task indexes must be in [0, {count - 1}]")
    return indexes


def extract_run(
    run,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    shuffle_seed: int,
) -> None:
    started = time.time()
    store = zarr.open_group(str(run.path / "server" / "routes.zarr"), mode="r")
    episode_id, episode_step, control_step = validate_episode_axes(run, store)
    metadata = query_metadata(run, episode_id)
    behavior, behavior_names = load_behavior_features(run)
    if len(behavior) != len(episode_id):
        raise ValueError(f"{run.key}: behavior and route row counts differ")

    count = len(episode_id)
    feature_shape = (count, 10, len(step_feature_names()))
    ordered = np.empty(feature_shape, dtype=np.float32)
    shuffled = np.empty(feature_shape, dtype=np.float32)
    permutations = deterministic_flow_permutations(count, shuffle_seed)

    probability_array = store["hb_router_probs"]
    expert_array = store["hb_expert_ids"]
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        probability = torch.as_tensor(
            np.asarray(probability_array[start:stop], dtype=np.float32), device=device
        )
        expert_ids = torch.as_tensor(
            np.asarray(expert_array[start:stop], dtype=np.int64), device=device
        )
        permutation = torch.as_tensor(permutations[start:stop], device=device)

        ordered[start:stop] = extract_multitrack_features(probability, expert_ids).cpu().numpy()
        shuffled_probability = permute_flow_tensor(probability, permutation)
        shuffled_ids = permute_flow_tensor(expert_ids, permutation)
        shuffled[start:stop] = (
            extract_multitrack_features(shuffled_probability, shuffled_ids).cpu().numpy()
        )
        del probability, expert_ids, permutation, shuffled_probability, shuffled_ids
        print(
            f"[{device}] {run.key}: {stop:,}/{count:,} queries",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = {
        "schema_version": 1,
        "run_key": run.key,
        "run_path": str(run.path),
        "queries": count,
        "episodes": len(run.summaries),
        "shuffle_seed": shuffle_seed,
        "feature_definition": "multitrack-routing-chord-v1",
        "elapsed_seconds": time.time() - started,
    }
    np.savez_compressed(
        output_path,
        features=ordered,
        flow_shuffled_features=shuffled,
        flow_permutations=permutations.astype(np.uint8),
        feature_names=np.asarray(step_feature_names()),
        behavior_features=behavior,
        behavior_feature_names=np.asarray(behavior_names),
        episode_id=episode_id.astype(np.int16),
        episode_step=episode_step,
        control_step=control_step,
        success=metadata["success"],
        init_state_id=metadata["init_state_id"],
        flow_noise_seed=metadata["flow_noise_seed"],
        metadata_json=np.asarray(json.dumps(payload_metadata, sort_keys=True)),
    )
    print(
        f"wrote {output_path} ({output_path.stat().st_size / 2**20:.1f} MiB, "
        f"{time.time() - started:.1f}s)",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.hub, args.run_id)
    for index, run in enumerate(runs):
        print(f"{index}: {run.key} ({len(run.summaries)} episodes)")
    if args.list:
        return
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    for index in selected_indexes(args.task_indexes, len(runs)):
        run = runs[index]
        output_path = args.output_dir / f"{run.output_stem}.npz"
        if output_path.exists() and not args.force:
            print(f"skip existing {output_path}")
            continue
        extract_run(run, output_path, device, args.batch_size, args.shuffle_seed)


if __name__ == "__main__":
    main()
