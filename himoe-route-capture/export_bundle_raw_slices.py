"""Export lossless, provenance-rich first-inference slices for a result bundle.

The full VLA_MUI_HUB corpus is about 76 GB.  This exporter keeps every candidate
from each complete right-16x32 task, but only one explicitly named MoE cell:
the first control step, one HB layer, and one denoising step.  No values are
normalized or converted; the stored NumPy dtypes are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
from typing import Any

import numpy as np
import zarr


ALIASES = {
    "open_the_middle_drawer_of_the_cabinet": "goal-middle",
    "open_the_top_drawer_and_put_the_bowl_inside": "goal-top",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": "long-t08",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": "spatial-ramekin",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": "spatial-stove",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-root", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--run-id", default="right-16x32")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--denoise", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_episode_file(client: pathlib.Path, episode: int) -> pathlib.Path:
    for width in (2, 4):
        candidate = client / f"episode_{episode:0{width}d}.npz"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"episode {episode} is absent from {client}")


def first_rows(episode_ids: np.ndarray, expected: np.ndarray) -> np.ndarray:
    rows = []
    for episode in expected:
        found = np.flatnonzero(episode_ids == episode)
        if len(found) == 0:
            raise ValueError(f"episode {episode} has no server-side route row")
        rows.append(int(found[0]))
    return np.asarray(rows, dtype=np.int64)


def select_hb(array: Any, rows: np.ndarray, layer_axis: int, denoise: int) -> np.ndarray:
    selection = (rows, [layer_axis], [denoise], slice(None), slice(None))
    return np.asarray(array.get_orthogonal_selection(selection))[:, 0, 0]


def select_hb_scalar(
    array: Any, rows: np.ndarray, layer_axis: int, denoise: int
) -> np.ndarray:
    selection = (rows, [layer_axis], [denoise], slice(None))
    return np.asarray(array.get_orthogonal_selection(selection))[:, 0, 0]


def json_load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_metadata(run: pathlib.Path, destination: pathlib.Path) -> dict[str, str]:
    relative_paths = (
        "meta.json",
        "client/summaries.json",
        "client/server_metadata.json",
        "client/sim_layout.json",
        "server/capture_summary.json",
        "server/routes.zarr/zarr.json",
        "server/hidden.zarr/zarr.json",
    )
    copied = {}
    for relative in relative_paths:
        source = run / relative
        if not source.is_file():
            continue
        target = destination / "metadata" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[relative] = sha256_file(target)
    return copied


def export_run(
    run: pathlib.Path, destination: pathlib.Path, layer: int, denoise: int
) -> dict[str, Any]:
    meta = json_load(run / "meta.json")
    task_name = str(meta["task_name"])
    alias = ALIASES.get(task_name)
    if alias is None:
        raise ValueError(f"no stable bundle alias for task {task_name}")

    client = run / "client"
    summaries = sorted(
        json_load(client / "summaries.json"), key=lambda row: int(row["episode_index"])
    )
    episodes = np.asarray([int(row["episode_index"]) for row in summaries], dtype=np.int32)
    if not np.array_equal(episodes, np.arange(len(summaries), dtype=np.int32)):
        raise ValueError(f"{alias}: episode indices are not contiguous from zero")
    scenes = np.asarray([int(row["init_state_id"]) for row in summaries], dtype=np.int32)
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries], dtype=np.int32)
    if len(summaries) != len(np.unique(scenes)) * len(np.unique(seeds)):
        raise ValueError(f"{alias}: not a complete scene x flow-noise grid")

    server_metadata = json_load(client / "server_metadata.json")
    layers = tuple(int(value) for value in server_metadata["routing_hb_layer_indices"])
    if layer not in layers:
        raise ValueError(f"{alias}: layer {layer} not in captured layers {layers}")
    layer_axis = layers.index(layer)

    routes = zarr.open_group(str(run / "server" / "routes.zarr"), mode="r")
    hidden = zarr.open_group(str(run / "server" / "hidden.zarr"), mode="r")
    route_episode = np.asarray(routes["episode_id"][:])
    hidden_episode = np.asarray(hidden["episode_id"][:])
    route_step = np.asarray(routes["control_step"][:])
    hidden_step = np.asarray(hidden["control_step"][:])
    if not np.array_equal(route_episode, hidden_episode):
        raise ValueError(f"{alias}: route/hidden episode axes differ")
    if not np.array_equal(route_step, hidden_step):
        raise ValueError(f"{alias}: route/hidden control-step axes differ")
    rows = first_rows(route_episode, episodes)

    actions = []
    states = []
    sim_states = []
    for episode in episodes:
        with np.load(resolve_episode_file(client, int(episode))) as payload:
            actions.append(np.asarray(payload["actions"][0]))
            states.append(np.asarray(payload["state"][0]))
            sim_states.append(np.asarray(payload["sim_state"][0]))

    arrays = {
        "episode_index": episodes,
        "init_state_id": scenes,
        "flow_noise_seed": seeds,
        "success": np.asarray([bool(row["success"]) for row in summaries]),
        "action_steps": np.asarray(
            [int(row["action_steps"]) for row in summaries], dtype=np.int32
        ),
        "inference_calls": np.asarray(
            [int(row["inference_calls"]) for row in summaries], dtype=np.int32
        ),
        "source_row": rows,
        "control_step": route_step[rows],
        "actions": np.stack(actions),
        "state": np.stack(states),
        "sim_state": np.stack(sim_states),
        "hb_hidden": select_hb(hidden["hb_hidden"], rows, layer_axis, denoise),
        "hb_expert_ids": select_hb(
            routes["hb_expert_ids"], rows, layer_axis, denoise
        ),
        "hb_selected_prob": select_hb(
            routes["hb_selected_prob"], rows, layer_axis, denoise
        ),
        "hb_router_probs": select_hb(
            routes["hb_router_probs"], rows, layer_axis, denoise
        ),
        "hb_entropy": select_hb_scalar(
            routes["hb_entropy"], rows, layer_axis, denoise
        ),
        "as_expert_ids": np.asarray(
            routes["as_expert_ids"].get_orthogonal_selection((rows, slice(None)))
        ),
        "as_probs": np.asarray(
            routes["as_probs"].get_orthogonal_selection(
                (rows, slice(None), slice(None))
            )
        ),
    }
    gathered = np.take_along_axis(
        arrays["hb_router_probs"], arrays["hb_expert_ids"].astype(np.int64), axis=-1
    )
    if not np.array_equal(gathered, arrays["hb_selected_prob"]):
        raise ValueError(f"{alias}: selected probabilities do not match recorded IDs")

    task_out = destination / alias
    task_out.mkdir(parents=True, exist_ok=False)
    archive = task_out / f"first_inference_hb{layer}_d{denoise}.npz"
    np.savez_compressed(archive, **arrays)
    with np.load(archive) as roundtrip:
        if set(roundtrip.files) != set(arrays):
            raise ValueError(f"{alias}: round-trip keys differ")
        for key, expected in arrays.items():
            actual = roundtrip[key]
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise ValueError(f"{alias}: round-trip failed for {key}")

    metadata_hashes = copy_metadata(run, task_out)
    manifest = {
        "alias": alias,
        "task_name": task_name,
        "benchmark": meta["benchmark"],
        "source_run": str(run.resolve()),
        "source_checkpoint_sha256": server_metadata.get("checkpoint_sha256"),
        "selection": {
            "control_step": "first server route row for every episode",
            "hb_layer": layer,
            "hb_layer_axis": layer_axis,
            "denoise_step": denoise,
            "tokens": "all 11 suffix tokens: state token then 10 action tokens",
        },
        "candidate_count": len(episodes),
        "scene_count": int(len(np.unique(scenes))),
        "flow_noise_seed_count": int(len(np.unique(seeds))),
        "success_count": int(arrays["success"].sum()),
        "array_schema": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "metadata_sha256": metadata_hashes,
        "archive_sha256": sha256_file(archive),
        "checks": {
            "complete_scene_seed_grid": True,
            "route_hidden_identity_axes_equal": True,
            "selected_prob_matches_recorded_ids": True,
            "npz_roundtrip_values_and_dtypes_equal": True,
        },
        "warning": (
            "Client episode NPZ routing placeholders are not exported. "
            "HB/AS routing here comes from authoritative server-side Zarr stores."
        ),
    }
    manifest_path = task_out / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    runs = sorted(args.hub_root.rglob(args.run_id))
    runs = [path for path in runs if (path / "meta.json").is_file()]
    if len(runs) != len(ALIASES):
        raise ValueError(f"expected {len(ALIASES)} runs, found {len(runs)}")
    args.out_dir.mkdir(parents=True)
    manifests = [
        export_run(run, args.out_dir, args.layer, args.denoise) for run in runs
    ]
    index = {
        "schema": "himoe-vla-bundle-first-inference/1",
        "lossless_selection": True,
        "run_id": args.run_id,
        "task_count": len(manifests),
        "tasks": manifests,
    }
    (args.out_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(args.out_dir), "tasks": len(manifests)}))


if __name__ == "__main__":
    main()
