"""Validate staged v2 artifacts and assemble a phase-1-compatible capture."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from behavior_forks_v2 import (
    ASSEMBLY_SCHEMA,
    CANDIDATE_SCHEMA,
    PLAN_SCHEMA,
    SHARD_REPEATS,
    SHARD_SCHEMA,
    ArtifactError,
    assert_no_seed_overlap,
    atomic_json,
    atomic_npz,
    candidate_query_id,
    candidate_dir,
    cohort_candidate_pool,
    continuation_query_id,
    flow_noise,
    load_npz,
    seed_records_from_candidate,
    seed_records_from_shard,
    shard_dir,
    sha256_file,
    validate_candidate_arrays,
    validate_shard_arrays,
    verify_artifact,
)


PHASE1_SCHEMA = "himoe.behavior_forks.v1"


def _load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != PLAN_SCHEMA or len(value.get("states", [])) != 18:
        raise ArtifactError(f"invalid plan: {path}")
    return value


def _selected_states(plan: Mapping[str, Any], cohort: str, path: Path | None) -> list[dict[str, Any]]:
    states = [dict(row) for row in plan["states"]]
    if path is None:
        if cohort == "enriched":
            raise ArtifactError("enriched assembly requires --state-ids-file")
        return states
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw or not all(isinstance(value, str) for value in raw):
        raise ArtifactError("state-ids-file must contain a non-empty JSON string list")
    wanted = set(raw)
    if len(wanted) != len(raw):
        raise ArtifactError("state-ids-file contains duplicates")
    selected = [row for row in states if row["state_id"] in wanted]
    if {row["state_id"] for row in selected} != wanted:
        raise ArtifactError("state-ids-file names a state outside the plan")
    return selected


def _target_repeats(cohort: str, target: int) -> int:
    allowed = (8,) if cohort == "screen" else (48, 96)
    if target not in allowed:
        raise ArtifactError(f"cohort {cohort!r} requires target R in {allowed}")
    return target


def _assert_noise(actual: np.ndarray, expected_words: list[tuple[int, ...]], label: str) -> None:
    flattened = np.asarray(actual).reshape((-1, 10, 24))
    if len(flattened) != len(expected_words):
        raise ArtifactError(f"{label} seed/noise counts disagree")
    for index, (value, words) in enumerate(zip(flattened, expected_words, strict=True)):
        if not np.array_equal(value, flow_noise(words)):
            raise ArtifactError(f"{label} noise does not match SeedSequence at index {index}")


def _validate_candidate_identity(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], state: Mapping[str, Any]
) -> None:
    if int(metadata["snapshot_index"]) != int(state["snapshot_index"]):
        raise ArtifactError("candidate snapshot_index differs from plan")
    pool = str(metadata["pool"])
    expected_ids = np.asarray(
        [
            candidate_query_id(int(state["snapshot_index"]), pool, candidate)
            for candidate in range(len(arrays["candidate_ids"]))
        ],
        dtype=np.int32,
    )
    if not np.array_equal(np.asarray(arrays["query_ids"]), expected_ids):
        raise ArtifactError("candidate query ids do not match the deterministic plan mapping")
    rows = np.asarray(arrays["server_trace_rows"], dtype=np.int64)
    stores = np.asarray(arrays["store_ids"]).astype(str)
    if bool(metadata.get("route_off_smoke")):
        if np.any(rows != -1) or np.any(stores != ""):
            raise ArtifactError("route-off candidate contains recorder identity")
    else:
        if np.any(rows < 0) or not np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
            raise ArtifactError("candidate server rows are not one contiguous acknowledged block")
        if len(set(stores.tolist())) != 1 or stores[0] != metadata.get("route_store_id"):
            raise ArtifactError("candidate capture_run_id is missing or inconsistent")
        if int(metadata["server_durable_through_row"]) < int(rows[-1]):
            raise ArtifactError("candidate final server row is not durable")


def _validate_shard_identity(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    state: Mapping[str, Any],
    cohort: str,
) -> None:
    executed = np.asarray(arrays["query_executed"], dtype=np.bool_)
    ids = np.asarray(arrays["query_ids"], dtype=np.int64)
    b = executed.shape[-1]
    repeat_start = int(metadata["repeat_start"])
    for candidate, local_repeat, future_step in zip(*np.nonzero(executed), strict=True):
        expected = continuation_query_id(
            int(state["snapshot_index"]),
            cohort,
            int(candidate),
            repeat_start + int(local_repeat),
            int(future_step),
            b,
        )
        if int(ids[candidate, local_repeat, future_step]) != expected:
            raise ArtifactError("continuation query id does not match its coordinates")
    if np.any(np.asarray(arrays["server_trace_rows"]) != -1):
        raise ArtifactError("continuation unexpectedly contains captured server rows")
    row_counts = np.asarray(arrays.get("server_row_counts", np.full(ids.shape, -1)))
    durable = np.asarray(arrays.get("server_durable_through_rows", np.full(ids.shape, -1)))
    if np.any(executed):
        pairs = set(zip(row_counts[executed].tolist(), durable[executed].tolist(), strict=True))
        if len(pairs) != 1:
            raise ArtifactError("uncaptured continuation advanced its recorder boundary")
        if not bool(metadata.get("route_off_smoke")):
            if metadata.get("capture_run_id") in (None, ""):
                raise ArtifactError("continuation acknowledgement has no capture_run_id")


def _copy_layout(source: Path, target: Path) -> dict[str, Any]:
    layout = json.loads(source.read_text(encoding="utf-8"))
    atomic_json(target, layout)
    return layout


def _combined_snapshot(
    candidate: Mapping[str, np.ndarray], shards: list[Mapping[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    continuation_keys = (
        "continuation_success",
        "continuation_final_sim_states",
        "continuation_action_steps",
        "continuation_flow_noise",
        "query_ids",
        "server_trace_rows",
        "query_executed",
        "continuation_actions",
        "continuation_action_executed",
    )
    combined = {name: np.asarray(value) for name, value in candidate.items()}
    for key in continuation_keys:
        axis = 0 if key == "continuation_flow_noise" else 1
        combined_key = {
            "query_ids": "continuation_query_ids",
            "server_trace_rows": "continuation_trace_rows",
            "query_executed": "continuation_query_executed",
        }.get(key, key)
        combined[combined_key] = np.concatenate(
            [np.asarray(shard[key]) for shard in shards], axis=axis
        )
    combined["candidate_trace_rows"] = combined.pop("server_trace_rows")
    combined["candidate_query_ids"] = combined.pop("query_ids")
    combined["schema_version"] = np.asarray(1, dtype=np.int32)
    combined["source_schema_version"] = np.asarray(2, dtype=np.int32)
    combined["route_capture_enabled"] = np.asarray(
        bool(np.all(combined["candidate_trace_rows"] >= 0)), dtype=np.bool_
    )
    return combined


def assemble(args: argparse.Namespace) -> int:
    plan_path = args.plan.resolve()
    plan = _load_plan(plan_path)
    target_r = _target_repeats(args.cohort, args.target_r)
    candidate_pool = cohort_candidate_pool(args.cohort)
    states = _selected_states(plan, args.cohort, args.state_ids_file)
    capture_root = args.capture_root.resolve()
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            print(f"assembly already complete: {output}")
            return 0
    if output.exists():
        raise ArtifactError(f"assembly output exists without a complete manifest: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    audit_entries = []
    phase1_entries = []
    all_seed_records: list[tuple[int, ...]] = []
    server_metadata_reference = None
    capture_run_ids: set[str] = set()
    try:
        for state in states:
            identifier = str(state["state_id"])
            candidate_path = candidate_dir(capture_root, candidate_pool, identifier) / "artifact.json"
            candidate_meta = verify_artifact(candidate_path, CANDIDATE_SCHEMA)
            if candidate_meta.get("pool") != candidate_pool:
                raise ArtifactError("candidate artifact belongs to another pool")
            if candidate_meta.get("plan_sha256") != sha256_file(plan_path):
                raise ArtifactError("candidate artifact was generated from another plan")
            candidate_arrays = load_npz(
                candidate_path.parent / candidate_meta["files"]["data"]["file"]
            )
            validate_candidate_arrays(candidate_arrays, candidate_meta)
            _validate_candidate_identity(candidate_arrays, candidate_meta, state)
            candidate_words = seed_records_from_candidate(candidate_arrays, candidate_meta)
            _assert_noise(candidate_arrays["candidate_flow_noise"], candidate_words, "candidate")
            all_seed_records.extend(candidate_words)
            if candidate_meta.get("route_store_id"):
                capture_run_ids.add(str(candidate_meta["route_store_id"]))
            server_metadata_path = (
                candidate_path.parent / candidate_meta["files"]["server_metadata"]["file"]
            )
            current_server = json.loads(server_metadata_path.read_text(encoding="utf-8"))
            identity = candidate_meta["server_identity"]
            if server_metadata_reference is None:
                server_metadata_reference = current_server
                identity_reference = identity
            elif identity != identity_reference:
                raise ArtifactError("candidate artifacts use different policy identities")
            shard_metas = []
            shard_arrays = []
            for shard_index in range(target_r // SHARD_REPEATS):
                path = shard_dir(
                    capture_root,
                    candidate_pool,
                    identifier,
                    args.cohort,
                    shard_index,
                ) / "artifact.json"
                metadata = verify_artifact(path, SHARD_SCHEMA)
                if metadata.get("label_cohort") != args.cohort:
                    raise ArtifactError("continuation shard belongs to another cohort")
                if int(metadata["repeat_start"]) != shard_index * SHARD_REPEATS:
                    raise ArtifactError("continuation shard repeat range is not contiguous")
                if metadata.get("candidate_artifact_sha256") != sha256_file(candidate_path):
                    raise ArtifactError("continuation shard references another candidate artifact")
                arrays = load_npz(path.parent / metadata["files"]["data"]["file"])
                validate_shard_arrays(arrays, metadata, len(candidate_arrays["candidate_ids"]))
                _validate_shard_identity(arrays, metadata, state, args.cohort)
                words = seed_records_from_shard(arrays, metadata)
                _assert_noise(arrays["continuation_flow_noise"], words, "continuation")
                all_seed_records.extend(words)
                shard_metas.append(metadata)
                shard_arrays.append(arrays)
            combined = _combined_snapshot(candidate_arrays, shard_arrays)
            stem = f"snapshot_{int(state['snapshot_index']):04d}_{identifier}"
            npz_path = stage / f"{stem}.npz"
            layout_path = stage / f"{stem}.layout.json"
            atomic_npz(npz_path, combined)
            layout_source = candidate_path.parent / candidate_meta["files"]["layout"]["file"]
            _copy_layout(layout_source, layout_path)
            entry = {
                "snapshot_index": int(state["snapshot_index"]),
                "snapshot_id": identifier,
                "task_id": int(state["task_id"]),
                "episode": int(state["episode"]),
                "fork_step": int(state["fork_step"]),
                "status": "complete",
                "npz_file": npz_path.name,
                "npz_sha256": sha256_file(npz_path),
                "layout_file": layout_path.name,
                "layout_sha256": sha256_file(layout_path),
            }
            phase1_entries.append(entry)
            audit_entries.append(
                {
                    **entry,
                    "candidate_artifact": str(candidate_path),
                    "candidate_artifact_sha256": sha256_file(candidate_path),
                    "continuation_artifacts": [
                        {
                            "shard_index": int(metadata["shard_index"]),
                            "artifact": str(
                                shard_dir(
                                    capture_root,
                                    candidate_pool,
                                    identifier,
                                    args.cohort,
                                    int(metadata["shard_index"]),
                                )
                                / "artifact.json"
                            ),
                        }
                        for metadata in shard_metas
                    ],
                }
            )
        assert_no_seed_overlap(all_seed_records)
        if server_metadata_reference is None:
            raise ArtifactError("assembly contains no candidate artifacts")
        atomic_json(stage / "server_metadata.json", server_metadata_reference)
        phase1_manifest = {
            "schema": PHASE1_SCHEMA,
            "schema_name": "himoe.behavior_forks",
            "schema_version": 1,
            "status": "completed",
            "complete": True,
            "server_metadata_file": "server_metadata.json",
            "seed_scheme": (
                "SeedSequence([20260824,SHA256(domain)_low32,task,episode,fork_step,*coords])"
            ),
            "config": {
                "v2_label_cohort": args.cohort,
                "v2_candidate_pool": candidate_pool,
                "continuation_repeats": target_r,
                "formal_pool": candidate_pool == "formal",
                "screen_pool": candidate_pool == "screen",
            },
            "artifacts": phase1_entries,
        }
        atomic_json(stage / "manifest.json", phase1_manifest)
        assembly_manifest = {
            "schema": ASSEMBLY_SCHEMA,
            "status": "complete",
            "complete": True,
            "plan_file": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "label_cohort": args.cohort,
            "candidate_pool": candidate_pool,
            "target_repeats": target_r,
            "state_count": len(states),
            "formal_pool": candidate_pool == "formal",
            "screen_pool": candidate_pool == "screen",
            "capture_run_ids": sorted(capture_run_ids),
            "intentional_seed_alias": (
                "main and enriched share formal/continuation CRN for the same state/repeat; "
                "each cohort assembly is validated separately"
            ),
            "phase1_manifest_file": "manifest.json",
            "phase1_manifest_sha256": sha256_file(stage / "manifest.json"),
            "artifacts": audit_entries,
        }
        atomic_json(stage / "assembly_manifest.v2.json", assembly_manifest)
        os.replace(stage, output)
    except BaseException:
        # Preserve the staging directory for forensic inspection; it is never
        # considered resumable because only the final directory is authoritative.
        raise
    print(
        f"assembled cohort={args.cohort} pool={candidate_pool} R={target_r} "
        f"states={len(states)} at {output}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--cohort", choices=("screen", "main", "enriched"), required=True)
    parser.add_argument("--target-r", type=int, required=True)
    parser.add_argument("--state-ids-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    return assemble(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
