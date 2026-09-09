"""Route--outcome geometry tests on same-snapshot candidate pairs.

The legacy mode reproduces the endpoint-proxy feasibility result.  Formal mode
loads checksummed behavior-fork captures, freezes all geometry choices on task
0, and evaluates task 1/3 with exact event tapes and paired-CRN Q intervals.
Pairs are never inference units: formal effects are matched and reduced within
snapshot before task/episode/snapshot hierarchical inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr

from analyze_behavior_geometry import (
    _coarse_event_matrices,
    _layout_physical_components,
    _validate_formal_arrays,
    load_formal,
)
from behavior_forks_v2 import (
    ASSEMBLY_SCHEMA,
    CANDIDATE_SCHEMA,
    PLAN_SCHEMA,
    SHARD_SCHEMA,
    verify_artifact,
)
from behavior_geometry import (
    action_distance_matrix,
    action_rms_distance_matrix,
    contact_event_distance_matrix,
    exact_event_equal_matrix,
    paired_binary_difference_interval,
)
from behavior_micro_analysis import (
    FormalCalibration,
    apply_formal_calibration,
    apply_physical_scales,
    deterministic_hidden_projection,
    fit_formal_calibration,
    hierarchical_effect_summary,
    matched_snapshot_effects,
    pairwise_rms_distance,
    physical_presence_masks,
    q_label_coverage_and_topup,
    screen_snapshot_flags,
    screening_metrics,
    snapshot_uid,
)
from route_noise_selector import pairwise_hellinger


ACTION_TOKEN_SLICE = slice(1, 11)
FORMAL_SPEC_SCHEMA = "himoe.route_outcome_formal_input.v1"
SCREEN_QUEUE_SCHEMA = "himoe.behavior_screen_queue.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "formal"), default="legacy")
    parser.add_argument("--formal-spec", type=Path,
                        help="formal multi-capture input manifest; required in formal mode")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--phase1-summary", type=Path)
    parser.add_argument("--fork-records", type=Path)
    parser.add_argument("--routes", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_snapshot_mean(
    values: Iterable[float],
    *,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    effects = np.asarray(list(values), dtype=np.float64)
    if effects.ndim != 1 or not len(effects) or not np.all(np.isfinite(effects)):
        raise ValueError("snapshot effects must be one non-empty finite vector")
    if draws < 100:
        raise ValueError("bootstrap draws must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    result: dict[str, Any] = {
        "mean": float(effects.mean()),
        "snapshots": int(len(effects)),
        "confidence": float(confidence),
    }
    if len(effects) == 1:
        result.update({"lower": None, "upper": None, "draws": 0, "available": False})
        return result
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(effects), size=(draws, len(effects)))
    distribution = effects[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(distribution, [tail, 1.0 - tail])
    result.update(
        {
            "lower": float(lower),
            "upper": float(upper),
            "draws": int(draws),
            "available": True,
        }
    )
    return result


def one_sided_sign_flip(values: Iterable[float], *, seed: int) -> dict[str, Any]:
    """Test whether the equal-snapshot mean is greater than zero."""

    effects = np.asarray(list(values), dtype=np.float64)
    if effects.ndim != 1 or not len(effects) or not np.all(np.isfinite(effects)):
        raise ValueError("snapshot effects must be one non-empty finite vector")
    observed = float(effects.mean())
    if len(effects) <= 20:
        total = 1 << len(effects)
        exceed = 0
        bit = np.arange(len(effects), dtype=np.uint64)
        for start in range(0, total, 65_536):
            masks = np.arange(start, min(start + 65_536, total), dtype=np.uint64)
            signs = np.where(((masks[:, None] >> bit) & 1) == 0, 1.0, -1.0)
            null = (signs @ effects) / len(effects)
            exceed += int(np.count_nonzero(null >= observed - 1e-15))
        return {
            "alternative": "mean contrast > 0",
            "observed": observed,
            "p": float(exceed / total),
            "draws": int(total),
            "exact": True,
        }
    rng = np.random.default_rng(seed)
    draws = 100_000
    signs = rng.choice((-1.0, 1.0), size=(draws, len(effects)))
    exceed = int(np.count_nonzero((signs @ effects) / len(effects) >= observed - 1e-15))
    return {
        "alternative": "mean contrast > 0",
        "observed": observed,
        "p": float((exceed + 1) / (draws + 1)),
        "draws": draws,
        "exact": False,
    }


def label_pair(
    row: Mapping[str, Any],
    *,
    action_near: float,
    action_far: float,
    outcome_same_max: float,
    outcome_different_min: float,
) -> tuple[str, str]:
    action = float(row["d_action"])
    outcome = float(row["proxy_outcome_gap"])
    if action <= action_near:
        action_stratum = "near"
    elif action >= action_far:
        action_stratum = "far"
    else:
        action_stratum = "middle"
    if outcome <= outcome_same_max:
        outcome_relation = "same"
    elif outcome >= outcome_different_min:
        outcome_relation = "different"
    else:
        outcome_relation = "ambiguous"
    return action_stratum, outcome_relation


def conditional_route_contrast(
    rows: Iterable[Mapping[str, Any]],
    *,
    action_stratum: str,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"same": [], "different": []}
    )
    for row in rows:
        if row["action_stratum"] != action_stratum:
            continue
        relation = str(row["outcome_relation"])
        if relation in {"same", "different"}:
            grouped[str(row["snapshot"])][relation].append(float(row["d_route"]))

    per_snapshot = []
    for snapshot in sorted(grouped):
        same = np.asarray(grouped[snapshot]["same"], dtype=np.float64)
        different = np.asarray(grouped[snapshot]["different"], dtype=np.float64)
        if not len(same) or not len(different):
            continue
        same_mean = float(same.mean())
        different_mean = float(different.mean())
        per_snapshot.append(
            {
                "snapshot": snapshot,
                "same_pair_count": int(len(same)),
                "different_pair_count": int(len(different)),
                "same_mean_d_route": same_mean,
                "different_mean_d_route": different_mean,
                "different_minus_same": different_mean - same_mean,
            }
        )
    if not per_snapshot:
        return {
            "available": False,
            "reason": "no snapshot contains both outcome groups in this action stratum",
        }

    effects = [row["different_minus_same"] for row in per_snapshot]
    bootstrap = bootstrap_snapshot_mean(
        effects, draws=draws, confidence=confidence, seed=seed
    )
    sign_flip = one_sided_sign_flip(effects, seed=seed + 1)
    same_pairs = sum(len(value["same"]) for value in grouped.values())
    different_pairs = sum(len(value["different"]) for value in grouped.values())
    return {
        "available": True,
        "hypothesis": (
            f"d_R({action_stratum}-action,different-outcome) > "
            f"d_R({action_stratum}-action,same-outcome)"
        ),
        "contrast": "different-outcome minus same-outcome",
        "all_labeled_pair_counts": {
            "same": int(same_pairs),
            "different": int(different_pairs),
        },
        "snapshots_with_any_labeled_pair": int(len(grouped)),
        "eligible_snapshot_count": int(len(per_snapshot)),
        "per_snapshot": per_snapshot,
        "snapshot_equal_bootstrap": bootstrap,
        "snapshot_sign_flip_one_sided": sign_flip,
        "point_estimate_holds": bool(bootstrap["mean"] > 0.0),
        "confidence_interval_above_zero": bool(
            bootstrap["available"] and bootstrap["lower"] > 0.0
        ),
    }


def load_pair_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("pair CSV has no header")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("pair CSV is empty")
    required = {
        "snapshot",
        "candidate_i",
        "candidate_j",
        "d_action",
        "proxy_outcome_gap",
    }
    missing = required.difference(reader.fieldnames)
    if missing:
        raise ValueError(f"pair CSV is missing fields: {sorted(missing)}")
    return list(reader.fieldnames), rows


def route_distances_by_snapshot(
    pair_rows: list[dict[str, Any]], records_path: Path, route_path: Path
) -> dict[str, Any]:
    raw_records = load_json(records_path)
    if not isinstance(raw_records, list):
        raise ValueError("fork records must be a JSON list")
    record_map: dict[tuple[str, int], int] = {}
    for record in raw_records:
        snapshot = f"ep{int(record['episode'])}/t{int(record['fork_step'])}"
        key = (snapshot, int(record["candidate"]))
        if key in record_map:
            raise ValueError(f"duplicate fork record for {key}")
        record_map[key] = int(record["trace_row"])

    by_snapshot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_snapshot[str(row["snapshot"])].append(row)
    group = zarr.open(str(route_path), mode="r")
    if "hb_router_probs" not in group:
        raise ValueError("route store is missing hb_router_probs")
    route_array = group["hb_router_probs"]
    if route_array.ndim != 5 or route_array.shape[1:] != (8, 10, 11, 32):
        raise ValueError(
            f"expected route tensor [call,8,10,11,32], got {route_array.shape}"
        )

    selected_rows: set[int] = set()
    mass_min = np.inf
    mass_max = -np.inf
    for snapshot, rows in sorted(by_snapshot.items()):
        candidates = sorted(
            {
                int(row["candidate_i"])
                for row in rows
            }
            | {int(row["candidate_j"]) for row in rows}
        )
        trace_rows = []
        for candidate in candidates:
            key = (snapshot, candidate)
            if key not in record_map:
                raise ValueError(f"no fork route record for {key}")
            trace_rows.append(record_map[key])
        if len(set(trace_rows)) != len(trace_rows):
            raise ValueError(f"snapshot {snapshot} reuses a route trace row")
        if min(trace_rows) < 0 or max(trace_rows) >= route_array.shape[0]:
            raise ValueError(f"snapshot {snapshot} references a route row outside the store")
        selected_rows.update(trace_rows)
        probabilities = np.asarray(
            route_array.oindex[trace_rows, :, :, ACTION_TOKEN_SLICE, :],
            dtype=np.float64,
        )
        masses = probabilities.sum(axis=-1)
        mass_min = min(mass_min, float(masses.min()))
        mass_max = max(mass_max, float(masses.max()))
        distance = pairwise_hellinger(probabilities)
        axis = {candidate: index for index, candidate in enumerate(candidates)}
        for row in rows:
            row["d_route"] = float(
                distance[axis[int(row["candidate_i"])]][axis[int(row["candidate_j"])]]
            )
    return {
        "route_store_rows": int(route_array.shape[0]),
        "selected_unique_trace_rows": int(len(selected_rows)),
        "selected_trace_row_min": int(min(selected_rows)),
        "selected_trace_row_max": int(max(selected_rows)),
        "raw_probability_mass_min": float(mass_min),
        "raw_probability_mass_max": float(mass_max),
    }


def write_pairs(path: Path, original_fields: list[str], rows: list[dict[str, Any]]) -> None:
    added = ["d_route", "action_stratum", "outcome_relation"]
    fields = original_fields + [field for field in added if field not in original_fields]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_interval(result: Mapping[str, Any]) -> str:
    if not result.get("available"):
        return "CI unavailable"
    return f"[{result['lower']:.6f}, {result['upper']:.6f}]"


def render_report(summary: Mapping[str, Any]) -> str:
    far = summary["inequalities"]["far_action"]
    near = summary["inequalities"]["near_action"]
    lines = [
        "# Exploratory route--outcome geometry check",
        "",
        "This is a legacy-proxy feasibility check, not confirmatory evidence.",
        "",
        "| Action stratum | Eligible snapshots | Same pairs | Different pairs | "
        "Mean `different - same` d_R | 95% snapshot bootstrap CI | One-sided sign-flip p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in (("far", far), ("near", near)):
        if not result["available"]:
            lines.append(f"| {name} | 0 | - | - | - | - | - |")
            continue
        bootstrap = result["snapshot_equal_bootstrap"]
        counts = result["all_labeled_pair_counts"]
        lines.append(
            f"| {name} | {result['eligible_snapshot_count']} | {counts['same']} | "
            f"{counts['different']} | {bootstrap['mean']:.6f} | "
            f"{format_interval(bootstrap)} | "
            f"{result['snapshot_sign_flip_one_sided']['p']:.6f} |"
        )
    both_point = far.get("point_estimate_holds", False) and near.get(
        "point_estimate_holds", False
    )
    both_ci = far.get("confidence_interval_above_zero", False) and near.get(
        "confidence_interval_above_zero", False
    )
    lines.extend(
        [
            "",
            f"Both inequalities hold at the point-estimate level: **{str(both_point).lower()}**.",
            f"Both snapshot-bootstrap intervals are above zero: **{str(both_ci).lower()}**.",
            "",
            "## Metric and inference",
            "",
            "`d_R` is RMS Hellinger distance over aligned 8 HB layers x 10 denoise "
            "rounds x 10 action tokens, using each site's normalized full 32-way "
            "router distribution. Contrasts are computed within snapshot and snapshots "
            "receive equal weight.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("")
    return "\n".join(lines)


def _required_legacy_path(value: Path | None, option: str) -> Path:
    if value is None:
        raise SystemExit(f"{option} is required in legacy mode")
    return value


def _spec_path(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"formal bundle field {field!r} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _snapshot_file_map(capture: Path, pools: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    result = {}
    for pool in pools:
        name = pool.get("data_file")
        if not isinstance(name, str):
            raise ValueError("formal pool has no data_file")
        uid = f"task{int(pool['task_id'])}/{pool['snapshot_state_sha256']}"
        path = capture / name
        if uid in result or not path.is_file():
            raise ValueError(f"invalid or duplicate formal snapshot file for {uid}")
        result[uid] = path
    return result


def _candidate_noise_digests(
    capture: Path, pools: Sequence[Mapping[str, Any]]
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for uid, path in _snapshot_file_map(capture, pools).items():
        with np.load(path, allow_pickle=False) as source:
            noise = np.asarray(source["candidate_flow_noise"])
        output[uid] = {
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in noise
        }
    return output


def _v2_state_digest(task_id: int, state_id: str) -> str:
    payload = f"himoe.behavior-study-state.v2/task{task_id}/{state_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _v2_pair_rows(
    capture: Path,
    entry: Mapping[str, Any],
    action_std: np.ndarray,
    gripper_weight: float,
    confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name = entry.get("npz_file")
    if not isinstance(name, str):
        raise ValueError("v2 assembly snapshot has no NPZ file")
    path = capture / name
    if sha256_file(path) != entry.get("npz_sha256"):
        raise ValueError(f"v2 assembly snapshot checksum mismatch: {path}")
    layout_name = entry.get("layout_file")
    if not isinstance(layout_name, str):
        raise ValueError("v2 assembly snapshot has no layout file")
    layout_path = capture / layout_name
    if sha256_file(layout_path) != entry.get("layout_sha256"):
        raise ValueError(f"v2 assembly layout checksum mismatch: {layout_path}")
    layout = load_json(layout_path)
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    task_id = int(entry["task_id"])
    state_id = str(entry["snapshot_id"])
    state_digest = _v2_state_digest(task_id, state_id)
    validated = dict(arrays)
    validated["snapshot_full_state_sha256"] = np.asarray(state_digest)
    k, h, repeats = _validate_formal_arrays(validated)

    actions = np.asarray(arrays["actions"], dtype=np.float64)
    d_action = action_distance_matrix(actions, action_std, gripper_weight)
    d_action_full = action_distance_matrix(actions, action_std, 1.0)
    d_action_rms = action_rms_distance_matrix(actions, action_std)
    physical = _layout_physical_components(
        arrays["sim_states"],
        arrays["eef_positions"],
        arrays["eef_quaternions"],
        arrays["gripper_qpos"],
        layout,
    )
    d_contact = contact_event_distance_matrix(arrays["contact_active"])
    raw_equal = exact_event_equal_matrix(
        arrays["contact_active"], arrays["chunk_success"]
    )
    d_event, events_equal, event_labels, event_provenance = _coarse_event_matrices(
        arrays, layout
    )
    if "event_flags" in arrays:
        flags = np.asarray(arrays["event_flags"], dtype=np.bool_)
        if flags.ndim != 3 or flags.shape[:2] != (k, h + 1):
            raise ValueError("v2 aligned event_flags have an invalid shape")
        flat_flags = flags.reshape(k, -1)
        primitive_distance = np.mean(
            flat_flags[:, None, :] != flat_flags[None, :, :], axis=-1
        )
        primitive_equal = np.all(
            flat_flags[:, None, :] == flat_flags[None, :, :], axis=-1
        )
        d_event = 0.5 * (d_event + primitive_distance)
        events_equal = events_equal & primitive_equal
        event_provenance = {
            **event_provenance,
            "v2_primitive_event_flags_exact_in_events_equal": True,
        }
    outcomes = np.asarray(arrays["continuation_success"], dtype=np.bool_)
    q_hat = outcomes.mean(axis=1)
    candidate_ids = np.asarray(arrays["candidate_ids"], dtype=np.int64)
    interval_cache: dict[tuple[int, int], dict[str, float | int]] = {}
    rows = []
    for left in range(k):
        for right in range(left + 1, k):
            left_only = int(np.sum(outcomes[left] & ~outcomes[right]))
            right_only = int(np.sum(~outcomes[left] & outcomes[right]))
            key = (left_only, right_only)
            if key not in interval_cache:
                interval_cache[key] = paired_binary_difference_interval(
                    outcomes[left], outcomes[right], confidence
                )
            interval = interval_cache[key]
            row: dict[str, Any] = {
                "snapshot": state_id,
                "snapshot_state_sha256": state_digest,
                "snapshot_identity_kind": "preregistered-v2-plan-state-id",
                "snapshot_index": int(entry["snapshot_index"]),
                "episode": int(entry["episode"]),
                "fork_step": int(entry["fork_step"]),
                "task_id": task_id,
                "task_name": str(entry.get("task_name", "unknown")),
                "candidate_i": int(candidate_ids[left]),
                "candidate_j": int(candidate_ids[right]),
                "d_action": float(d_action[left, right]),
                "d_action_full_gripper": float(d_action_full[left, right]),
                "d_action_rms": float(d_action_rms[left, right]),
                "d_event": float(d_event[left, right]),
                "d_contact_raw": float(d_contact[left, right]),
                "raw_contacts_equal": bool(raw_equal[left, right]),
                "events_equal": bool(events_equal[left, right]),
                "q_i": float(q_hat[left]),
                "q_j": float(q_hat[right]),
                "abs_delta_q": float(abs(q_hat[left] - q_hat[right])),
                "q_delta": float(interval["estimate"]),
                "q_ci_lower": float(interval["lower"]),
                "q_ci_upper": float(interval["upper"]),
                "q_discordant": int(interval["discordant"]),
            }
            for component, matrix in physical.items():
                row[f"dX_{component}"] = float(matrix[left, right])
            rows.append(row)
    pool = {
        "tag": state_id,
        "snapshot_id": state_id,
        "snapshot_index": int(entry["snapshot_index"]),
        "snapshot_identity_kind": "preregistered-v2-plan-state-id",
        "snapshot_state_sha256": state_digest,
        "episode": int(entry["episode"]),
        "fork_step": int(entry["fork_step"]),
        "task_id": task_id,
        "task_name": str(entry.get("task_name", "unknown")),
        "candidates": k,
        "chunk_steps": h,
        "continuation_repeats": repeats,
        "event_labels": event_labels,
        "event_provenance": event_provenance,
        "physical_components": sorted(physical),
        "data_file": name,
    }
    return pool, rows


def _load_v2_assembly(
    capture: Path, gripper_weight: float, confidence: float
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    assembly_path = capture / "assembly_manifest.v2.json"
    assembly = load_json(assembly_path)
    if assembly.get("schema") != ASSEMBLY_SCHEMA or assembly.get("complete") is not True:
        raise ValueError(f"invalid or incomplete v2 assembly: {capture}")
    phase1_name = assembly.get("phase1_manifest_file")
    if not isinstance(phase1_name, str):
        raise ValueError("v2 assembly has no phase-1 manifest")
    manifest_path = capture / phase1_name
    if sha256_file(manifest_path) != assembly.get("phase1_manifest_sha256"):
        raise ValueError("v2 assembly phase-1 manifest checksum mismatch")
    manifest = load_json(manifest_path)
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, Mapping) or (
        manifest_config.get("v2_label_cohort") != assembly.get("label_cohort")
        or manifest_config.get("v2_candidate_pool") != assembly.get("candidate_pool")
        or int(manifest_config.get("continuation_repeats", -1))
        != int(assembly.get("target_repeats", -2))
    ):
        raise ValueError("v2 assembly and phase-1 manifest cohort metadata disagree")
    entries = manifest.get("artifacts")
    audits = assembly.get("artifacts")
    if not isinstance(entries, list) or not entries or not isinstance(audits, list):
        raise ValueError("v2 assembly has no artifact audit rows")
    if any(entry.get("status") != "complete" for entry in entries):
        raise ValueError("v2 assembly phase-1 manifest contains an incomplete snapshot")
    if len(entries) != len(audits) or int(assembly.get("state_count", -1)) != len(entries):
        raise ValueError("v2 assembly state counts disagree")
    audit_by_index = {int(row["snapshot_index"]): row for row in audits}
    if len(audit_by_index) != len(audits):
        raise ValueError("v2 assembly repeats a snapshot index")
    verified_candidates = 0
    verified_shards = 0
    candidate_server_metadata_digests = set()
    for entry in entries:
        index = int(entry["snapshot_index"])
        audit = audit_by_index.get(index)
        if audit is None:
            raise ValueError("v2 assembly audit omits a snapshot")
        for key in (
            "snapshot_id",
            "task_id",
            "episode",
            "fork_step",
            "npz_file",
            "npz_sha256",
            "layout_file",
            "layout_sha256",
        ):
            if audit.get(key) != entry.get(key):
                raise ValueError(f"v2 assembly audit disagrees on {key}")
        candidate_path = Path(str(audit["candidate_artifact"]))
        if sha256_file(candidate_path) != audit.get("candidate_artifact_sha256"):
            raise ValueError("v2 candidate artifact checksum mismatch")
        candidate_metadata = verify_artifact(candidate_path, CANDIDATE_SCHEMA)
        if (
            candidate_metadata.get("state_id") != entry.get("snapshot_id")
            or int(candidate_metadata.get("snapshot_index", -1)) != index
            or candidate_metadata.get("pool") != assembly.get("candidate_pool")
        ):
            raise ValueError("v2 candidate artifact identity disagrees with assembly")
        server_record = candidate_metadata.get("files", {}).get("server_metadata", {})
        if not isinstance(server_record.get("sha256"), str):
            raise ValueError("v2 candidate artifact has no server metadata checksum")
        candidate_server_metadata_digests.add(str(server_record["sha256"]))
        verified_candidates += 1
        continuation = audit.get("continuation_artifacts")
        if not isinstance(continuation, list):
            raise ValueError("v2 assembly has no continuation artifact list")
        for item in continuation:
            shard_metadata = verify_artifact(Path(str(item["artifact"])), SHARD_SCHEMA)
            if (
                shard_metadata.get("state_id") != entry.get("snapshot_id")
                or shard_metadata.get("label_cohort") != assembly.get("label_cohort")
                or shard_metadata.get("candidate_artifact_sha256")
                != audit.get("candidate_artifact_sha256")
            ):
                raise ValueError("v2 continuation artifact identity disagrees with assembly")
            verified_shards += 1
    metadata_path = capture / str(manifest.get("server_metadata_file"))
    if (
        len(candidate_server_metadata_digests) != 1
        or sha256_file(metadata_path) not in candidate_server_metadata_digests
    ):
        raise ValueError("v2 assembly server metadata differs from candidate artifacts")
    metadata = load_json(metadata_path)
    action_std = np.asarray(metadata["normalization_action_std"], dtype=np.float64)
    if action_std.shape != (7,) or not np.all(np.isfinite(action_std)) or np.any(
        action_std <= 0.0
    ):
        raise ValueError("v2 assembly action std must have seven positive values")
    pools = []
    rows = []
    for entry in entries:
        pool, pair_rows = _v2_pair_rows(
            capture, entry, action_std, gripper_weight, confidence
        )
        pools.append(pool)
        rows.extend(pair_rows)
    return (
        {
            "assembly_manifest": str(assembly_path),
            "assembly_manifest_sha256": sha256_file(assembly_path),
            "phase1_manifest": str(manifest_path),
            "phase1_manifest_sha256": sha256_file(manifest_path),
            "server_metadata": str(metadata_path),
            "server_metadata_sha256": sha256_file(metadata_path),
            "capture_schema": manifest.get("schema"),
            "source_schema": ASSEMBLY_SCHEMA,
            "label_cohort": assembly.get("label_cohort"),
            "candidate_pool": assembly.get("candidate_pool"),
            "plan_file": assembly.get("plan_file"),
            "plan_sha256": assembly.get("plan_sha256"),
            "capture_run_ids": assembly.get("capture_run_ids"),
            "candidate_artifacts_checksum_validated": verified_candidates,
            "continuation_shards_checksum_validated": verified_shards,
            "checksums_validated": True,
            "fidelity_validated": True,
            "snapshot_identity": (
                "SHA256 of preregistered task/state_id; v2 assembly does not store a "
                "full simulator-state digest"
            ),
        },
        pools,
        rows,
    )


def _load_formal_capture(
    capture: Path, gripper_weight: float, confidence: float
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if (capture / "assembly_manifest.v2.json").is_file():
        return _load_v2_assembly(capture, gripper_weight, confidence)
    provenance, pools, rows = load_formal(capture, gripper_weight, confidence)
    provenance["checksums_validated"] = True
    provenance["fidelity_validated"] = True
    return provenance, pools, rows


def _attach_candidate_features(
    rows: list[dict[str, Any]],
    pools: Sequence[Mapping[str, Any]],
    *,
    capture: Path,
    routes_path: Path,
    hidden_path: Path,
    flow_path: Path,
    projection_seed: int,
) -> dict[str, Any]:
    route = zarr.open_group(str(routes_path), mode="r")
    hidden = zarr.open_group(str(hidden_path), mode="r")
    flow = zarr.open_group(str(flow_path), mode="r")
    required_route = {"hb_router_probs", "episode_id", "control_step"}
    required_hidden = {"hb_hidden", "episode_id", "control_step"}
    if not required_route.issubset(set(route.array_keys())):
        raise ValueError("formal route store lacks hb_router_probs or episode_id")
    if not required_hidden.issubset(set(hidden.array_keys())):
        raise ValueError("formal hidden store lacks hb_hidden or episode_id")
    required_flow = {
        "x_traj",
        "capture_row",
        "query_id",
        "candidate_id",
        "local_call_ordinal",
        "observation_sha256",
        "flow_noise_sha256",
        "actions_sha256",
    }
    if not required_flow.issubset(set(flow.array_keys())):
        raise ValueError("formal flow trajectory store lacks identity or digest arrays")
    route_array = route["hb_router_probs"]
    hidden_array = hidden["hb_hidden"]
    if route_array.ndim != 5 or route_array.shape[1:] != (8, 10, 11, 32):
        raise ValueError(f"expected formal route shape [N,8,10,11,32], got {route_array.shape}")
    if hidden_array.ndim != 5 or hidden_array.shape[1:4] != (8, 10, 11):
        raise ValueError(f"expected formal hidden shape [N,8,10,11,H], got {hidden_array.shape}")
    flow_array = flow["x_traj"]
    if flow_array.ndim != 4 or flow_array.shape[1:] != (11, 10, 24):
        raise ValueError(
            f"expected normalized flow trajectory shape [N,11,10,24], got {flow_array.shape}"
        )
    if route_array.shape[0] != hidden_array.shape[0] or route_array.shape[0] != flow_array.shape[0]:
        raise ValueError("formal route, hidden, and flow stores have different row counts")
    store_rows = int(route_array.shape[0])
    run_ids = {
        str(group.attrs.get("capture_run_id", "")) for group in (route, hidden, flow)
    }
    if len(run_ids) != 1 or not next(iter(run_ids)):
        raise ValueError("formal route, hidden, and flow stores do not share capture_run_id")
    durable = [int(group.attrs.get("durable_rows", -1)) for group in (route, hidden, flow)]
    if durable != [store_rows, store_rows, store_rows]:
        raise ValueError("formal feature stores are not complete at one durable row boundary")
    route_query = np.asarray(route["episode_id"][:], dtype=np.int64)
    hidden_query = np.asarray(hidden["episode_id"][:], dtype=np.int64)
    flow_query = np.asarray(flow["query_id"][:], dtype=np.int64)
    route_call = np.asarray(route["control_step"][:], dtype=np.int64)
    hidden_call = np.asarray(hidden["control_step"][:], dtype=np.int64)
    flow_call = np.asarray(flow["local_call_ordinal"][:], dtype=np.int64)
    capture_row = np.asarray(flow["capture_row"][:], dtype=np.int64)
    if not (
        np.array_equal(route_query, hidden_query)
        and np.array_equal(route_query, flow_query)
        and np.array_equal(route_call, hidden_call)
        and np.array_equal(route_call, flow_call)
        and np.array_equal(capture_row, np.arange(len(capture_row), dtype=np.int64))
    ):
        raise ValueError("formal route, hidden, and flow identity spines disagree")

    manifest = None
    for manifest_name in ("manifest.json", "behavior_manifest.json"):
        candidate = capture / manifest_name
        if candidate.is_file():
            manifest = load_json(candidate)
            break
    if not isinstance(manifest, Mapping):
        raise ValueError(f"formal capture has no manifest for snapshot identity audit: {capture}")
    entries = manifest.get("snapshots", manifest.get("artifacts"))
    if not isinstance(entries, list):
        raise ValueError("formal capture manifest has no snapshot entries")
    snapshot_index_by_file = {}
    for entry in entries:
        name = (
            entry.get("data_file")
            or entry.get("npz")
            or entry.get("physics_file")
            or entry.get("npz_file")
        )
        if not isinstance(name, str) or name in snapshot_index_by_file:
            raise ValueError("formal capture manifest has invalid snapshot data files")
        snapshot_index_by_file[name] = int(entry.get("snapshot_index", -1))

    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[snapshot_uid(row)].append(row)
    selected_rows: set[int] = set()
    mass_min = np.inf
    mass_max = -np.inf
    store_id_rows_validated = 0
    for uid, path in _snapshot_file_map(capture, pools).items():
        material = by_uid.get(uid)
        if not material:
            raise ValueError(f"formal feature store has no pair rows for {uid}")
        with np.load(path, allow_pickle=False) as source:
            candidate_ids = np.asarray(source["candidate_ids"], dtype=np.int64)
            trace_rows = np.asarray(source["candidate_trace_rows"], dtype=np.int64)
            query_ids = np.asarray(source["candidate_query_ids"], dtype=np.int64)
            candidate_noise = np.asarray(source["candidate_flow_noise"], dtype=np.float32)
            candidate_actions = np.asarray(source["actions"], dtype=np.float32)
            store_ids = (
                np.asarray(source["store_ids"]).astype(str)
                if "store_ids" in source
                else None
            )
            if "continuation_trace_rows" in source and np.any(
                np.asarray(source["continuation_trace_rows"]) >= 0
            ):
                raise ValueError("formal capture contains continuation route rows")
        if (
            candidate_ids.ndim != 1
            or trace_rows.shape != candidate_ids.shape
            or query_ids.shape != candidate_ids.shape
            or len(np.unique(candidate_ids)) != len(candidate_ids)
        ):
            raise ValueError(f"candidate feature indices are malformed for {uid}")
        if np.any(trace_rows < 0) or np.any(trace_rows >= route_array.shape[0]):
            raise ValueError(f"candidate trace row lies outside formal stores for {uid}")
        if len(np.unique(trace_rows)) != len(trace_rows):
            raise ValueError(f"candidate trace rows repeat within {uid}")
        if len(candidate_ids) != 32:
            raise ValueError(f"formal micro-pilot requires K=32 for {uid}")
        if not np.array_equal(route_query[trace_rows], query_ids):
            raise ValueError(f"route rows do not acknowledge candidate query ids for {uid}")
        if not np.array_equal(
            np.asarray(flow["candidate_id"].oindex[trace_rows], dtype=np.int64), candidate_ids
        ):
            raise ValueError(f"flow rows do not acknowledge candidate ids for {uid}")
        expected_snapshot_index = snapshot_index_by_file.get(
            str(path.relative_to(capture)), -1
        )
        if expected_snapshot_index < 0 or not np.all(
            np.asarray(flow["snapshot_index"].oindex[trace_rows], dtype=np.int64)
            == expected_snapshot_index
        ):
            raise ValueError(f"flow rows cross the declared snapshot boundary for {uid}")
        if store_ids is not None and (
            store_ids.shape != candidate_ids.shape
            or not np.all(store_ids == next(iter(run_ids)))
        ):
            raise ValueError(f"candidate capture_run_id disagrees with shared stores for {uid}")
        if store_ids is not None:
            store_id_rows_validated += len(store_ids)
        stored_noise_digest = np.asarray(
            flow["flow_noise_sha256"].oindex[trace_rows], dtype=np.uint8
        )
        stored_action_digest = np.asarray(
            flow["actions_sha256"].oindex[trace_rows], dtype=np.uint8
        )
        expected_noise_digest = np.stack(
            [
                np.frombuffer(
                    hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest(),
                    dtype=np.uint8,
                )
                for value in candidate_noise
            ]
        )
        expected_action_digest = np.stack(
            [
                np.frombuffer(
                    hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest(),
                    dtype=np.uint8,
                )
                for value in candidate_actions
            ]
        )
        if not np.array_equal(stored_noise_digest, expected_noise_digest):
            raise ValueError(f"flow noise digest disagrees with candidate artifact for {uid}")
        if not np.array_equal(stored_action_digest, expected_action_digest):
            raise ValueError(f"action digest disagrees with candidate artifact for {uid}")
        observations = np.asarray(
            flow["observation_sha256"].oindex[trace_rows], dtype=np.uint8
        )
        if not np.all(observations == observations[:1]):
            raise ValueError(f"candidate rows do not share one observation digest for {uid}")
        selected_rows.update(map(int, trace_rows))
        probabilities = np.asarray(
            route_array.oindex[trace_rows, :, :, ACTION_TOKEN_SLICE, :], dtype=np.float64
        )
        masses = probabilities.sum(axis=-1)
        mass_min = min(mass_min, float(masses.min()))
        mass_max = max(mass_max, float(masses.max()))
        route_distance = pairwise_hellinger(probabilities)
        hidden_value = np.asarray(
            hidden_array.oindex[trace_rows, :, :, ACTION_TOKEN_SLICE, :], dtype=np.float32
        )
        projected = deterministic_hidden_projection(
            hidden_value, output_width=probabilities.shape[-1], seed=projection_seed
        )
        hidden_distance = pairwise_rms_distance(projected)
        flow_value = np.asarray(flow_array.oindex[trace_rows], dtype=np.float32)
        flow_distance = pairwise_rms_distance(flow_value)
        axis = {int(candidate): index for index, candidate in enumerate(candidate_ids)}
        for row in material:
            left = axis[int(row["candidate_i"])]
            right = axis[int(row["candidate_j"])]
            row["d_route"] = float(route_distance[left, right])
            row["d_hidden"] = float(hidden_distance[left, right])
            row["d_flow"] = float(flow_distance[left, right])
    expected_selected = sum(int(pool["candidates"]) for pool in pools)
    if len(selected_rows) != expected_selected:
        raise ValueError("candidate trace rows overlap across formal snapshot pools")
    if any(
        not all(metric in row for metric in ("d_route", "d_hidden", "d_flow"))
        for row in rows
    ):
        raise ValueError("one or more formal pair rows lack aligned candidate features")
    return {
        "route_store_rows": store_rows,
        "hidden_store_rows": store_rows,
        "flow_store_rows": store_rows,
        "selected_candidate_rows": len(selected_rows),
        "selected_row_indices": sorted(selected_rows),
        "route_probability_mass_min": float(mass_min),
        "route_probability_mass_max": float(mass_max),
        "hidden_input_width": int(hidden_array.shape[-1]),
        "hidden_projection_width": 32,
        "hidden_projection_seed": int(projection_seed),
        "hidden_projection": "deterministic Rademacher projection at each aligned HB site",
        "flow_trajectory_shape": list(flow_array.shape[1:]),
        "capture_run_id": next(iter(run_ids)),
        "flow_snapshot_index_alignment": True,
        "candidate_store_id_rows_validated": store_id_rows_validated,
        "candidate_store_ids_match_capture_run_id": True,
        "three_store_identity_and_digest_alignment": True,
    }


def _queue_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = load_json(path)
    if value.get("schema") != SCREEN_QUEUE_SCHEMA or not isinstance(
        value.get("snapshots"), list
    ):
        raise ValueError(f"screen queue manifest must use {SCREEN_QUEUE_SCHEMA}")
    result = {}
    for row in value["snapshots"]:
        state_hash = str(row.get("snapshot_state_sha256", ""))
        task_id = int(row.get("task_id", -1))
        if len(state_hash) != 64 or task_id < 0:
            raise ValueError("screen queue row has no task id or full-state hash")
        uid = f"task{task_id}/{state_hash}"
        if uid in result:
            raise ValueError(f"screen queue repeats {uid}")
        probability = float(row.get("inclusion_probability", np.nan))
        if not 0.0 < probability <= 1.0:
            raise ValueError(f"screen queue has invalid inclusion probability for {uid}")
        cohort = row.get("cohort")
        if not isinstance(cohort, str) or not cohort.strip():
            raise ValueError(f"screen queue has no sampling cohort for {uid}")
        material = dict(row)
        material["cohort"] = cohort.strip()
        result[uid] = material
    return result


def _queue_from_v2_plan(path: Path) -> dict[str, dict[str, Any]]:
    value = load_json(path)
    states = value.get("states")
    if value.get("schema") != PLAN_SCHEMA or not isinstance(states, list) or len(states) != 18:
        raise ValueError("study_plan must be one complete v2 18-state plan")
    result = {}
    for row in states:
        task = int(row["task_id"])
        state_id = str(row["state_id"])
        uid = f"task{task}/{_v2_state_digest(task, state_id)}"
        selection = str(row.get("selection", ""))
        cohort = "event_critical" if selection.startswith("event_critical") else selection
        if not cohort:
            raise ValueError(f"v2 plan state has no sampling cohort: {state_id}")
        result[uid] = {
            "task_id": task,
            "state_id": state_id,
            "snapshot_state_sha256": _v2_state_digest(task, state_id),
            "cohort": cohort,
            "selection": selection,
            "source_success": bool(row["source_success"]),
            "inclusion_probability": 1.0,
        }
    if len(result) != 18:
        raise ValueError("v2 study plan repeats a state identity")
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in result.values():
        counts[(int(row["task_id"]), str(row["cohort"]))] += 1
    expected = {
        (task, cohort): count
        for task in (0, 1, 3)
        for cohort, count in (("event_critical", 4), ("uniform_audit", 2))
    }
    if dict(counts) != expected:
        raise ValueError("v2 study plan must contain 4 event-critical and 2 audit states per task")
    return result


def _pool_uid(pool: Mapping[str, Any]) -> str:
    return f"task{int(pool['task_id'])}/{pool['snapshot_state_sha256']}"


def _annotate_capture(
    rows: list[dict[str, Any]],
    pools: list[dict[str, Any]],
    *,
    bundle_id: str,
    queue: str,
    capture: Path,
) -> None:
    for row in rows:
        row["bundle_id"] = bundle_id
        row["queue"] = queue
    for pool in pools:
        pool["bundle_id"] = bundle_id
        pool["queue"] = queue
        pool["capture_path"] = str(capture)


def _trace_audit(capture: Path, pools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_rows = 0
    continuation_rows = 0
    repeat_counts = set()
    for path in _snapshot_file_map(capture, pools).values():
        with np.load(path, allow_pickle=False) as source:
            candidate_rows += int(np.count_nonzero(source["candidate_trace_rows"] >= 0))
            continuation_rows += int(
                np.count_nonzero(source["continuation_trace_rows"] >= 0)
            )
            repeat_counts.add(int(np.asarray(source["continuation_success"]).shape[1]))
    return {
        "candidate_trace_rows_nonnegative": candidate_rows,
        "continuation_trace_rows_nonnegative": continuation_rows,
        "continuation_repeats": sorted(repeat_counts),
        "candidate_independent_crn_array": True,
    }


def _repeat_noise_digests(value: np.ndarray) -> set[str]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 4 or array.shape[2:] != (10, 24):
        raise ValueError("continuation flow noise must have shape [R,B,10,24]")
    return {
        hashlib.sha256(np.ascontiguousarray(repeat).tobytes()).hexdigest()
        for repeat in array
    }


def _screen_main_restore_audit(
    screen_capture: Path,
    screen_pools: Sequence[Mapping[str, Any]],
    main_capture: Path,
    main_pools: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    screen_files = _snapshot_file_map(screen_capture, screen_pools)
    main_files = _snapshot_file_map(main_capture, main_pools)
    if set(screen_files) != set(main_files):
        raise ValueError("screen/main restore audit covers different states")
    maximum = 0.0
    for uid in sorted(screen_files):
        with (
            np.load(screen_files[uid], allow_pickle=False) as screen,
            np.load(main_files[uid], allow_pickle=False) as main,
        ):
            for name in ("sim_states", "eef_positions", "eef_quaternions", "gripper_qpos"):
                left = np.asarray(screen[name])[0, 0]
                right = np.asarray(main[name])[0, 0]
                if left.shape != right.shape:
                    raise ValueError(f"screen/main initial {name} shapes differ for {uid}")
                difference = float(np.max(np.abs(left - right)))
                if name == "eef_quaternions":
                    difference = min(
                        difference, float(np.max(np.abs(left + right)))
                    )
                maximum = max(maximum, difference)
            if (
                "snapshot_full_state_sha256" in screen
                or "snapshot_full_state_sha256" in main
            ) and (
                "snapshot_full_state_sha256" not in screen
                or "snapshot_full_state_sha256" not in main
                or not np.array_equal(
                    screen["snapshot_full_state_sha256"],
                    main["snapshot_full_state_sha256"],
                )
            ):
                raise ValueError("screen/main full-state digests disagree")
    if maximum > tolerance:
        raise ValueError(
            f"screen/main restored initial states differ by {maximum:.3e} > {tolerance:.3e}"
        )
    return {
        "snapshots": len(screen_files),
        "fields": ["sim_states", "eef_positions", "eef_quaternions", "gripper_qpos"],
        "maximum_initial_state_abs_difference": maximum,
        "tolerance": tolerance,
        "passed": True,
    }


def _validate_reused_screen_candidates(
    screen_capture: Path,
    screen_pools: Sequence[Mapping[str, Any]],
    enriched_capture: Path,
    enriched_pools: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    screen_files = _snapshot_file_map(screen_capture, screen_pools)
    enriched_files = _snapshot_file_map(enriched_capture, enriched_pools)
    if not set(enriched_files).issubset(screen_files):
        raise ValueError("enriched queue contains a snapshot absent from screening")
    candidate_arrays = (
        "actions",
        "sim_states",
        "eef_positions",
        "eef_quaternions",
        "gripper_qpos",
        "chunk_success",
        "contact_active",
        "contact_pair_names",
        "candidate_ids",
        "candidate_flow_noise",
    )
    repeat_counts = set()
    for uid, enriched_path in enriched_files.items():
        with (
            np.load(screen_files[uid], allow_pickle=False) as screen,
            np.load(enriched_path, allow_pickle=False) as enriched,
        ):
            for name in candidate_arrays:
                if name not in screen or name not in enriched:
                    raise ValueError(f"candidate reuse audit lacks {name} for {uid}")
                if not np.array_equal(screen[name], enriched[name]):
                    raise ValueError(f"enriched queue does not exactly reuse {name} for {uid}")
            for optional_name in (
                "snapshot_full_state_sha256",
                "store_ids",
                "candidate_query_ids",
                "event_flags",
                "execution_order",
                "robot_qpos_indices",
                "object_qpos_indices",
            ):
                if (optional_name in screen) != (optional_name in enriched) or (
                    optional_name in screen
                    and not np.array_equal(screen[optional_name], enriched[optional_name])
                ):
                    raise ValueError(
                        f"enriched queue does not preserve {optional_name} for {uid}"
                    )
            screen_r = int(np.asarray(screen["continuation_success"]).shape[1])
            enriched_r = int(np.asarray(enriched["continuation_success"]).shape[1])
            if screen_r != 8 or enriched_r not in {48, 96}:
                raise ValueError("screen/formal repeat counts must be R8 and R48 or R96")
            repeat_counts.add(enriched_r)
            overlap = _repeat_noise_digests(screen["continuation_flow_noise"]) & (
                _repeat_noise_digests(enriched["continuation_flow_noise"])
            )
            if overlap:
                raise ValueError("R8 screen noise leaked into enriched formal labels")
            enriched_candidate_rows = np.asarray(enriched["candidate_trace_rows"])
            if not (
                np.all(enriched_candidate_rows < 0)
                or np.array_equal(enriched_candidate_rows, screen["candidate_trace_rows"])
            ):
                raise ValueError("enriched candidate trace rows are neither reused nor absent")
            if np.any(np.asarray(enriched["continuation_trace_rows"]) >= 0):
                raise ValueError("enriched continuations were written to candidate stores")
    return {
        "snapshots": len(enriched_files),
        "candidate_arrays_exactly_reused": True,
        "screen_R8_excluded_from_formal_labels": True,
        "screen_and_formal_continuation_noise_disjoint": True,
        "formal_repeats": sorted(repeat_counts),
    }


def _copy_candidate_features(
    source_rows: Sequence[Mapping[str, Any]], target_rows: list[dict[str, Any]]
) -> None:
    source = {
        (
            snapshot_uid(row),
            int(row["candidate_i"]),
            int(row["candidate_j"]),
        ): row
        for row in source_rows
    }
    for row in target_rows:
        key = (
            snapshot_uid(row),
            int(row["candidate_i"]),
            int(row["candidate_j"]),
        )
        if key not in source:
            raise ValueError("enriched formal pair has no screen candidate feature row")
        for metric in ("d_route", "d_hidden", "d_flow"):
            row[metric] = float(source[key][metric])


def _merge_queue_rows(
    destination: dict[str, dict[str, Any]], source: Mapping[str, dict[str, Any]]
) -> None:
    overlap = set(destination) & set(source)
    if overlap:
        raise ValueError(f"screen queue metadata repeats {sorted(overlap)[0]}")
    destination.update(source)


def _load_formal_spec(path: Path, confidence: float) -> dict[str, Any]:
    spec = load_json(path)
    if spec.get("schema") != FORMAL_SPEC_SCHEMA:
        raise ValueError(f"formal spec must use schema {FORMAL_SPEC_SCHEMA}")
    calibration_tasks = tuple(int(value) for value in spec.get("calibration_task_ids", [0]))
    evaluation_tasks = tuple(int(value) for value in spec.get("evaluation_task_ids", [1, 3]))
    if calibration_tasks != (0,) or set(evaluation_tasks) != {1, 3}:
        raise ValueError("this micro-pilot freezes task 0 and evaluates exactly tasks 1 and 3")
    bundles = spec.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("formal spec contains no bundles")
    stores = spec.get("stores")
    if not isinstance(stores, Mapping):
        raise ValueError("formal spec must declare one shared stores object")
    base = path.resolve().parent
    routes = _spec_path(base, stores.get("routes"), "stores.routes")
    hidden = _spec_path(base, stores.get("hidden"), "stores.hidden")
    flow = _spec_path(
        base, stores.get("flow_trajectory"), "stores.flow_trajectory"
    )
    projection_seed = int(spec.get("hidden_projection_seed", 20260824))
    queue_rows: dict[str, dict[str, Any]] = {}
    study_plan_path = None
    if spec.get("study_plan") is not None:
        study_plan_path = _spec_path(base, spec["study_plan"], "study_plan")
        _merge_queue_rows(queue_rows, _queue_from_v2_plan(study_plan_path))
    if spec.get("screen_queue") is not None:
        declared = _queue_manifest(
            _spec_path(base, spec["screen_queue"], "screen_queue")
        )
        if queue_rows:
            if set(declared) != set(queue_rows):
                raise ValueError("screen queue and v2 study plan cover different states")
            for uid, row in declared.items():
                if str(row["cohort"]) != str(queue_rows[uid]["cohort"]):
                    raise ValueError("screen queue cohort disagrees with v2 study plan")
                queue_rows[uid].update(row)
        else:
            _merge_queue_rows(queue_rows, declared)

    queues: dict[str, dict[str, Any]] = {
        name: {"rows": [], "pools": []} for name in ("screen", "main", "enriched")
    }
    provenance: list[dict[str, Any]] = []
    screen_selected: set[int] = set()
    main_selected: set[int] = set()
    store_rows: int | None = None
    for index, raw_bundle in enumerate(bundles):
        if not isinstance(raw_bundle, Mapping):
            raise ValueError("formal bundle must be an object")
        bundle_id = str(raw_bundle.get("id", f"bundle-{index}"))
        screen_capture = _spec_path(
            base, raw_bundle.get("screen_capture"), "screen_capture"
        )
        main_value = raw_bundle.get("main_capture", raw_bundle.get("capture"))
        main_capture = _spec_path(base, main_value, "main_capture")
        enriched_capture = (
            _spec_path(base, raw_bundle["enriched_capture"], "enriched_capture")
            if raw_bundle.get("enriched_capture") is not None
            else None
        )
        if raw_bundle.get("screen_queue") is not None:
            _merge_queue_rows(
                queue_rows,
                _queue_manifest(
                    _spec_path(base, raw_bundle["screen_queue"], "screen_queue")
                ),
            )

        screen_provenance, screen_pools, screen_rows = _load_formal_capture(
            screen_capture, 0.25, confidence
        )
        main_provenance, main_pools, main_rows = _load_formal_capture(
            main_capture, 0.25, confidence
        )
        if screen_provenance.get("source_schema") == ASSEMBLY_SCHEMA and (
            screen_provenance.get("label_cohort") != "screen"
            or screen_provenance.get("candidate_pool") != "screen"
        ):
            raise ValueError("screen_capture is not a screen/screen v2 assembly")
        if main_provenance.get("source_schema") == ASSEMBLY_SCHEMA and (
            main_provenance.get("label_cohort") != "main"
            or main_provenance.get("candidate_pool") != "formal"
        ):
            raise ValueError("main_capture is not a main/formal v2 assembly")
        if {int(pool["continuation_repeats"]) for pool in screen_pools} != {8}:
            raise ValueError("screen captures must contain exactly R=8 continuations")
        if not {int(pool["continuation_repeats"]) for pool in main_pools}.issubset(
            {48, 96}
        ):
            raise ValueError("main captures must contain R=48 or R=96 continuations")
        _annotate_capture(
            screen_rows,
            screen_pools,
            bundle_id=bundle_id,
            queue="screen",
            capture=screen_capture,
        )
        _annotate_capture(
            main_rows,
            main_pools,
            bundle_id=bundle_id,
            queue="main",
            capture=main_capture,
        )
        screen_alignment = _attach_candidate_features(
            screen_rows,
            screen_pools,
            capture=screen_capture,
            routes_path=routes,
            hidden_path=hidden,
            flow_path=flow,
            projection_seed=projection_seed,
        )
        main_alignment = _attach_candidate_features(
            main_rows,
            main_pools,
            capture=main_capture,
            routes_path=routes,
            hidden_path=hidden,
            flow_path=flow,
            projection_seed=projection_seed,
        )
        this_store_rows = int(main_alignment["route_store_rows"])
        if store_rows is None:
            store_rows = this_store_rows
        elif store_rows != this_store_rows:
            raise ValueError("shared candidate stores changed length between bundles")
        screen_indices = set(screen_alignment["selected_row_indices"])
        main_indices = set(main_alignment["selected_row_indices"])
        if (screen_selected | main_selected) & (screen_indices | main_indices):
            raise ValueError("candidate trace rows repeat across capture bundles")
        if screen_indices & main_indices:
            raise ValueError("screen and main candidate trace rows overlap")
        screen_selected.update(screen_indices)
        main_selected.update(main_indices)

        screen_noise = _candidate_noise_digests(screen_capture, screen_pools)
        main_noise = _candidate_noise_digests(main_capture, main_pools)
        if set(screen_noise) != set(main_noise):
            raise ValueError("screen and main captures do not cover the same snapshots")
        if any(screen_noise[uid] & main_noise[uid] for uid in screen_noise):
            raise ValueError("screen and main candidate noise domains overlap")
        restore_audit = _screen_main_restore_audit(
            screen_capture, screen_pools, main_capture, main_pools
        )

        queues["screen"]["rows"].extend(screen_rows)
        queues["screen"]["pools"].extend(screen_pools)
        queues["main"]["rows"].extend(main_rows)
        queues["main"]["pools"].extend(main_pools)
        bundle_provenance: dict[str, Any] = {
            "id": bundle_id,
            "screen_capture": screen_provenance,
            "main_capture": main_provenance,
            "screen_feature_alignment": screen_alignment,
            "main_feature_alignment": main_alignment,
            "screen_trace_audit": _trace_audit(screen_capture, screen_pools),
            "main_trace_audit": _trace_audit(main_capture, main_pools),
            "screen_main_candidate_seeds_disjoint": True,
            "screen_main_restore_fidelity": restore_audit,
        }
        if enriched_capture is not None:
            enriched_provenance, enriched_pools, enriched_rows = _load_formal_capture(
                enriched_capture, 0.25, confidence
            )
            if enriched_provenance.get("source_schema") == ASSEMBLY_SCHEMA and (
                enriched_provenance.get("label_cohort") != "enriched"
                or enriched_provenance.get("candidate_pool") != "screen"
            ):
                raise ValueError(
                    "enriched_capture is not an enriched/screen v2 assembly"
                )
            _annotate_capture(
                enriched_rows,
                enriched_pools,
                bundle_id=bundle_id,
                queue="enriched",
                capture=enriched_capture,
            )
            reuse = _validate_reused_screen_candidates(
                screen_capture,
                screen_pools,
                enriched_capture,
                enriched_pools,
            )
            _copy_candidate_features(screen_rows, enriched_rows)
            queues["enriched"]["rows"].extend(enriched_rows)
            queues["enriched"]["pools"].extend(enriched_pools)
            bundle_provenance.update(
                {
                    "enriched_capture": enriched_provenance,
                    "enriched_trace_audit": _trace_audit(
                        enriched_capture, enriched_pools
                    ),
                    "enriched_candidate_reuse": reuse,
                }
            )
        provenance.append(bundle_provenance)

    main_pools = queues["main"]["pools"]
    screen_pools = queues["screen"]["pools"]
    expected_tasks = set(calibration_tasks) | set(evaluation_tasks)
    observed_tasks = {int(pool["task_id"]) for pool in main_pools}
    if observed_tasks != expected_tasks:
        raise ValueError(
            f"main captures contain tasks {sorted(observed_tasks)}, expected {sorted(expected_tasks)}"
        )
    main_uids = [_pool_uid(pool) for pool in main_pools]
    screen_uids = [_pool_uid(pool) for pool in screen_pools]
    if len(main_uids) != len(set(main_uids)) or len(screen_uids) != len(set(screen_uids)):
        raise ValueError("screen or main queue repeats a simulator snapshot")
    if set(main_uids) != set(screen_uids) or len(main_uids) != 18:
        raise ValueError("screen and main queues must cover the same 18 snapshots")
    if set(queue_rows) != set(main_uids):
        raise ValueError("screen queue metadata must exactly cover all 18 main snapshots")
    enriched_uids = [_pool_uid(pool) for pool in queues["enriched"]["pools"]]
    if len(enriched_uids) != len(set(enriched_uids)) or not set(enriched_uids).issubset(
        main_uids
    ):
        raise ValueError("enriched queue repeats or introduces a simulator snapshot")
    selected = screen_selected | main_selected
    if store_rows != 1152 or len(selected) != 1152 or selected != set(range(1152)):
        raise ValueError(
            "screen+main candidate rows must exactly cover all 1152 shared store rows"
        )
    if study_plan_path is not None:
        expected_plan_sha256 = sha256_file(study_plan_path)
        for bundle in provenance:
            for key in ("screen_capture", "main_capture", "enriched_capture"):
                capture_provenance = bundle.get(key)
                if (
                    isinstance(capture_provenance, Mapping)
                    and capture_provenance.get("source_schema") == ASSEMBLY_SCHEMA
                    and capture_provenance.get("plan_sha256") != expected_plan_sha256
                ):
                    raise ValueError(f"{key} was assembled from another v2 study plan")
    return {
        "spec": spec,
        "calibration_tasks": calibration_tasks,
        "evaluation_tasks": evaluation_tasks,
        "queues": queues,
        "queue_rows": queue_rows,
        "provenance": provenance,
        "projection_seed": projection_seed,
        "study_plan": None
        if study_plan_path is None
        else {
            "path": str(study_plan_path),
            "sha256": sha256_file(study_plan_path),
            "schema": PLAN_SCHEMA,
        },
        "stores": {
            "routes": str(routes),
            "hidden": str(hidden),
            "flow_trajectory": str(flow),
            "rows": int(store_rows),
            "screen_selected_rows": len(screen_selected),
            "main_selected_rows": len(main_selected),
            "union_exactly_covers_store": True,
            "flow_snapshot_index_alignment": True,
            "candidate_store_id_rows_validated": sum(
                int(bundle[key]["candidate_store_id_rows_validated"])
                for bundle in provenance
                for key in ("screen_feature_alignment", "main_feature_alignment")
            ),
            "candidate_store_ids_match_capture_run_id": True,
            "three_store_identity_and_digest_alignment": True,
        },
    }


def _snapshot_metadata(
    pools: Sequence[Mapping[str, Any]],
    queue_rows: Mapping[str, Mapping[str, Any]],
    evaluation_tasks: set[int],
) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for pool in pools:
        if int(pool["task_id"]) not in evaluation_tasks:
            continue
        uid = _pool_uid(pool)
        if uid in seen or uid not in queue_rows:
            raise ValueError("formal queue has a duplicate snapshot or lacks queue metadata")
        seen.add(uid)
        queue = queue_rows[uid]
        probability = float(queue["inclusion_probability"])
        if not 0.0 < probability <= 1.0:
            raise ValueError(f"invalid inclusion probability for {uid}")
        result.append(
            {
                "snapshot_uid": uid,
                "task_id": int(pool["task_id"]),
                "episode": int(pool["episode"]),
                "cohort": str(queue["cohort"]),
                "inclusion_probability": probability,
            }
        )
    return sorted(result, key=lambda row: str(row["snapshot_uid"]))


def _horizon_censoring(
    pools: Sequence[Mapping[str, Any]],
    queue_rows: Mapping[str, Mapping[str, Any]],
    evaluation_tasks: set[int],
    *,
    queue_name: str,
) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    expected_cells = set()
    for pool in pools:
        task = int(pool["task_id"])
        if task not in evaluation_tasks:
            continue
        uid = _pool_uid(pool)
        cohort = str(queue_rows[uid]["cohort"])
        expected_cells.add((task, cohort))
        path = Path(str(pool["capture_path"])) / str(pool["data_file"])
        with np.load(path, allow_pickle=False) as source:
            success = np.asarray(source["continuation_success"], dtype=np.bool_)
            steps = np.asarray(source["continuation_action_steps"], dtype=np.int64)
        if success.shape != steps.shape:
            raise ValueError(f"continuation success/step tapes disagree for {uid}")
        censored = ~success & (steps == 50)
        grouped[(task, cohort)].append((int(censored.sum()), int(censored.size)))
    cells = {}
    all_within_limit = True
    for (task, cohort), values in sorted(grouped.items()):
        count = sum(value[0] for value in values)
        total = sum(value[1] for value in values)
        fraction = count / total
        all_within_limit &= fraction <= 0.20
        cells[f"task{task}/{cohort}"] = {
            "horizon_censored": count,
            "continuations": total,
            "fraction": fraction,
            "at_most_0_20": fraction <= 0.20,
        }
    return {
        "queue": queue_name,
        "definition": "no success and all 50 environment actions executed",
        "unit": "candidate x paired-CRN continuation repeat",
        "by_evaluation_task_and_sampling_cohort": cells,
        "available": bool(cells),
        "expected_cells": [
            f"task{task}/{cohort}" for task, cohort in sorted(expected_cells)
        ],
        "all_expected_cells_reported": set(cells)
        == {f"task{task}/{cohort}" for task, cohort in expected_cells},
        "all_observed_cells_at_most_0_20": all_within_limit,
    }


def _queue_analysis(
    *,
    queue_name: str,
    rows: list[dict[str, Any]],
    pools: Sequence[Mapping[str, Any]],
    queue_rows: Mapping[str, Mapping[str, Any]],
    calibration: FormalCalibration,
    evaluation_tasks: set[int],
    min_matches: int,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    apply_formal_calibration(rows, calibration)
    evaluation_rows = [
        row for row in rows if int(row["task_id"]) in evaluation_tasks
    ]
    effects, matching_ineligible = matched_snapshot_effects(
        evaluation_rows, calibration, min_matches=min_matches
    )
    snapshots = _snapshot_metadata(pools, queue_rows, evaluation_tasks)
    metadata_by_uid = {row["snapshot_uid"]: row for row in snapshots}
    for effect in effects:
        metadata = metadata_by_uid[str(effect["snapshot_uid"])]
        effect["inclusion_probability"] = metadata["inclusion_probability"]
        effect["cohort"] = metadata["cohort"]

    summaries: dict[str, Any] = {}
    metrics = (
        "route_effect",
        "route_residual_effect",
        "hidden_effect",
        "flow_effect",
        "physics_effect",
    )
    for stratum_index, stratum in enumerate(("near", "far")):
        material = [row for row in effects if row["stratum"] == stratum]
        summaries[stratum] = {
            metric: hierarchical_effect_summary(
                material,
                metric,
                draws=draws,
                confidence=confidence,
                seed=seed + 100 * stratum_index + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
    return {
        "queue": queue_name,
        "scope": (
            "population main queue"
            if queue_name == "main"
            else "screen-positive conditional enriched queue"
        ),
        "formal_Q_excludes_screen_R8": True,
        "snapshot_count_all_tasks": len(pools),
        "snapshot_count_evaluation": len(snapshots),
        "pair_count_evaluation": len(evaluation_rows),
        "snapshots": snapshots,
        "effects": summaries,
        "snapshot_effects": effects,
        "matching_ineligible_snapshots": matching_ineligible,
        "eligible_snapshot_counts": {
            stratum: sum(row["stratum"] == stratum for row in effects)
            for stratum in ("near", "far")
        },
        "horizon_censoring": _horizon_censoring(
            pools,
            queue_rows,
            evaluation_tasks,
            queue_name=queue_name,
        ),
    }


def _formal_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Formal route--outcome micro-pilot",
        "",
        "This report is deliberately non-confirmatory. Task 0 freezes the protocol; "
        "tasks 1 and 3 are the only evaluation tasks.",
        "",
        "| Queue | Stratum | Eligible snapshots | Raw route effect | "
        "95% hierarchical CI |",
        "|---|---|---:|---:|---:|",
    ]
    for queue_name in ("main", "enriched"):
        for stratum in ("far", "near"):
            result = summary["queues"][queue_name]["effects"][stratum]["route_effect"]
            interval = (
                f"[{result['lower']:.6f}, {result['upper']:.6f}]"
                if result.get("available")
                else "unavailable"
            )
            mean = result.get("mean")
            mean_text = "-" if mean is None else f"{mean:.6f}"
            lines.append(
                f"| {queue_name} | {stratum} | {result.get('snapshot_count', 0)} | "
                f"{mean_text} | {interval} |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Formal micro-pilot gate passed: **{str(summary['gates']['passed']).lower()}**",
            f"- Global R48 -> R96 top-up required: **{str(summary['global_topup']['global_topup_required']).lower()}**",
            "- Route effect direction and its interval are reported but are not gates.",
            "- `confirmatory` is hard-coded false for this micro-pilot.",
            "",
            "## Fixed protocol",
            "",
            "- Same outcome requires exact aligned event tapes and a paired-CRN Q-difference "
            "interval wholly inside +/-0.1.",
            "- Different outcome requires unequal event tapes or a paired interval wholly "
            "beyond +/-0.1; all other event-equal pairs remain unlabeled.",
            "- Far-action primary rows must also be far in frozen task-0 physical "
            "trajectory geometry.",
            "- Matching is action-only, one-to-one, maximum-cardinality/minimum-cost, with "
            "a task-0 snapshot-equal caliper equal to 0.2 times stratum action-distance SD.",
            "- Raw route distance is primary. A snapshot-equal task-0 quadratic action "
            "residual is a sensitivity analysis only.",
            "- Model-normalized flow trajectory distance is a separate secondary metric; "
            "it is not the physical trajectory criterion.",
            "- Inference reduces pairs inside snapshot, then resamples task -> episode -> snapshot.",
            "- Horizon censoring means failure after all 50 environment actions; ambiguous "
            "paired-Q intervals are tracked separately as label coverage.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_formal(args: argparse.Namespace) -> int:
    if args.formal_spec is None:
        raise SystemExit("--formal-spec is required in formal mode")
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")
    loaded = _load_formal_spec(args.formal_spec, args.confidence)
    main_rows = loaded["queues"]["main"]["rows"]
    screen_rows = loaded["queues"]["screen"]["rows"]
    enriched_rows = loaded["queues"]["enriched"]["rows"]
    calibration = fit_formal_calibration(
        main_rows,
        calibration_task_ids=loaded["calibration_tasks"],
        near_quantile=0.2,
        far_quantile=0.8,
        caliper_multiplier=0.2,
    )
    apply_physical_scales(
        screen_rows, calibration.physical_schema, calibration.physical_scales
    )
    screen_presence = physical_presence_masks(screen_rows, calibration.physical_schema)
    if screen_presence != calibration.physical_presence_masks:
        raise ValueError("screen physical presence masks differ from frozen main masks")
    if enriched_rows:
        apply_physical_scales(
            enriched_rows,
            calibration.physical_schema,
            calibration.physical_scales,
        )
        enriched_presence = physical_presence_masks(
            enriched_rows, calibration.physical_schema
        )
        if any(
            mask != calibration.physical_presence_masks[task]
            for task, mask in enriched_presence.items()
        ):
            raise ValueError("enriched physical presence masks differ from frozen main masks")
    screen_flags = screen_snapshot_flags(screen_rows, calibration)
    evaluation_tasks = set(loaded["evaluation_tasks"])
    main_pools = loaded["queues"]["main"]["pools"]
    enriched_pools = loaded["queues"]["enriched"]["pools"]
    main_uids = {_pool_uid(pool) for pool in main_pools}
    screen_positive_uids = set()
    for uid in sorted(main_uids):
        queue = loaded["queue_rows"][uid]
        positive = False
        for stratum in ("near", "far"):
            derived = bool(
                screen_flags.get(uid, {}).get(stratum, {}).get(
                    "screen_positive", False
                )
            )
            positive |= derived
            declared = queue.get(f"screen_{stratum}")
            if declared is not None and bool(declared) != derived:
                raise ValueError(
                    f"screen manifest disagrees with derived {stratum} flag for {uid}"
                )
        if positive:
            screen_positive_uids.add(uid)
    enriched_uids = {_pool_uid(pool) for pool in enriched_pools}
    if enriched_uids != screen_positive_uids:
        raise ValueError(
            "enriched queue must contain exactly the derived screen-positive snapshots"
        )

    min_matches = int(loaded["spec"].get("min_matches", 5))
    main = _queue_analysis(
        queue_name="main",
        rows=main_rows,
        pools=main_pools,
        queue_rows=loaded["queue_rows"],
        calibration=calibration,
        evaluation_tasks=evaluation_tasks,
        min_matches=min_matches,
        draws=args.bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )
    enriched = _queue_analysis(
        queue_name="enriched",
        rows=enriched_rows,
        pools=enriched_pools,
        queue_rows=loaded["queue_rows"],
        calibration=calibration,
        evaluation_tasks=evaluation_tasks,
        min_matches=min_matches,
        draws=args.bootstrap,
        confidence=args.confidence,
        seed=args.seed + 10_000,
    )
    eligible_uids = {
        stratum: {
            str(row["snapshot_uid"])
            for row in main["snapshot_effects"]
            if row["stratum"] == stratum
        }
        for stratum in ("near", "far")
    }
    screen_result = screening_metrics(main["snapshots"], eligible_uids, screen_flags)
    formal_pools = [*main_pools, *enriched_pools]
    global_topup = q_label_coverage_and_topup(
        [row for row in main_rows if int(row["task_id"]) in evaluation_tasks],
        [int(pool["continuation_repeats"]) for pool in formal_pools],
        eligible_counts=main["eligible_snapshot_counts"],
        evaluation_task_ids=loaded["evaluation_tasks"],
    )
    global_topup.update(
        {
            "apply_to": "all main and enriched formal continuation pools",
            "formal_pool_count": len(formal_pools),
            "main_pool_count": len(main_pools),
            "enriched_pool_count": len(enriched_pools),
            "screen_R8_excluded": True,
        }
    )
    task_snapshot_counts = defaultdict(int)
    screen_task_counts = defaultdict(int)
    task_episode_counts: dict[int, set[int]] = defaultdict(set)
    for pool in main_pools:
        task_snapshot_counts[int(pool["task_id"])] += 1
        task_episode_counts[int(pool["task_id"])].add(int(pool["episode"]))
    for pool in loaded["queues"]["screen"]["pools"]:
        screen_task_counts[int(pool["task_id"])] += 1
    all_trace_audits = []
    screen_main_candidate_trace_rows = 0
    for row in loaded["provenance"]:
        screen_main_candidate_trace_rows += int(
            row["screen_trace_audit"]["candidate_trace_rows_nonnegative"]
        ) + int(row["main_trace_audit"]["candidate_trace_rows_nonnegative"])
        all_trace_audits.extend(
            value
            for key, value in row.items()
            if key.endswith("_trace_audit")
        )
    v2_assembly_input = any(
        row["main_capture"].get("source_schema") == ASSEMBLY_SCHEMA
        for row in loaded["provenance"]
    )
    store_id_rows = int(loaded["stores"]["candidate_store_id_rows_validated"])
    engineering_checks = {
        "main_has_18_snapshots_six_per_task": dict(task_snapshot_counts)
        == {0: 6, 1: 6, 3: 6},
        "screen_has_18_snapshots_six_per_task": dict(screen_task_counts)
        == {0: 6, 1: 6, 3: 6},
        "main_source_episodes_distinct_within_task": all(
            len(task_episode_counts[task]) == 6 for task in (0, 1, 3)
        ),
        "all_18_main_fidelity_and_checksum_valid": len(main_pools) == 18
        and all(
            bool(row["main_capture"].get("checksums_validated"))
            and bool(row["main_capture"].get("fidelity_validated"))
            for row in loaded["provenance"]
        ),
        "all_formal_pools_use_candidate_independent_crn": all(
            audit["candidate_independent_crn_array"] for audit in all_trace_audits
        ),
        "shared_stores_have_exactly_1152_candidate_rows": loaded["stores"]["rows"]
        == 1152,
        "screen_and_main_artifacts_reference_exactly_1152_candidate_rows": (
            screen_main_candidate_trace_rows == 1152
        ),
        "screen_plus_main_exactly_cover_shared_stores": bool(
            loaded["stores"]["union_exactly_covers_store"]
        ),
        "three_store_identity_and_digest_alignment": bool(
            loaded["stores"]["three_store_identity_and_digest_alignment"]
        ),
        "flow_rows_match_declared_snapshot_indices": bool(
            loaded["stores"]["flow_snapshot_index_alignment"]
        ),
        "candidate_store_ids_match_shared_capture_run_id_when_present": bool(
            loaded["stores"]["candidate_store_ids_match_capture_run_id"]
        )
        and (
            store_id_rows == 1152
            if v2_assembly_input
            else store_id_rows in {0, 1152}
        ),
        "continuations_have_zero_recorded_feature_rows": all(
            audit["continuation_trace_rows_nonnegative"] == 0
            for audit in all_trace_audits
        ),
        "screen_and_main_candidate_seeds_disjoint": all(
            row["screen_main_candidate_seeds_disjoint"]
            for row in loaded["provenance"]
        ),
        "screen_and_main_restore_fidelity_passes": all(
            row["screen_main_restore_fidelity"]["passed"]
            for row in loaded["provenance"]
        ),
        "formal_repeats_uniform": bool(global_topup["uniform_repeats"]),
        "formal_repeats_are_R48_or_R96": set(global_topup["observed_repeats"])
        in ({48}, {96}),
        "global_topup_resolved": not bool(global_topup["global_topup_required"]),
    }
    coverage_cells = global_topup["label_coverage_by_eval_task_and_stratum"]
    science_common = {
        "main_label_coverage_at_least_0_70_in_every_eval_task_stratum": bool(
            coverage_cells
        )
        and all(float(row["coverage"]) >= 0.70 for row in coverage_cells.values()),
    }
    stratum_gates = {}
    for stratum in ("near", "far"):
        material = [
            row for row in main["snapshot_effects"] if row["stratum"] == stratum
        ]
        recall = screen_result[stratum]["recall"]
        ppv = screen_result[stratum]["ppv"]
        checks = {
            "at_least_four_eligible_snapshots": len(material) >= 4,
            "both_evaluation_tasks_represented": {
                int(row["task_id"]) for row in material
            }
            == evaluation_tasks,
            "at_least_four_source_episodes": len(
                {(int(row["task_id"]), int(row["episode"])) for row in material}
            )
            >= 4,
            "all_effects_have_five_matches_and_four_candidates_per_group": all(
                int(row["matched_count"]) >= min_matches
                and int(row["same_unique_candidate_count"]) >= 4
                and int(row["different_unique_candidate_count"]) >= 4
                for row in material
            ),
            "screen_recall_at_least_0_75": recall is not None and float(recall) >= 0.75,
            "screen_ppv_at_least_0_40": ppv is not None and float(ppv) >= 0.40,
        }
        stratum_gates[stratum] = {"checks": checks, "passed": all(checks.values())}
    censoring_checks = {
        "main_all_task_cohort_rates_at_most_0_20": bool(
            main["horizon_censoring"]["available"]
        )
        and bool(main["horizon_censoring"]["all_expected_cells_reported"])
        and bool(main["horizon_censoring"]["all_observed_cells_at_most_0_20"]),
        "enriched_all_observed_task_cohort_rates_at_most_0_20": (
            not enriched["horizon_censoring"]["available"]
            or bool(
                enriched["horizon_censoring"][
                    "all_observed_cells_at_most_0_20"
                ]
            )
        ),
    }
    gates = {
        "engineering": {
            "checks": engineering_checks,
            "passed": all(engineering_checks.values()),
        },
        "science_common": {
            "checks": science_common,
            "passed": all(science_common.values()),
        },
        "main_strata": stratum_gates,
        "censoring": {
            "checks": censoring_checks,
            "passed": all(censoring_checks.values()),
        },
        "effect_direction_used": False,
        "passed": all(engineering_checks.values())
        and all(science_common.values())
        and all(value["passed"] for value in stratum_gates.values()),
    }
    gates["passed"] = bool(gates["passed"] and gates["censoring"]["passed"])

    summary = {
        "analysis": "formal route--outcome geometry micro-pilot",
        "mode": "formal_micro_pilot",
        "confirmatory": False,
        "confirmatory_reason": (
            "micro-pilot; task 0 calibration and only two evaluation tasks; "
            "the report is hard-coded non-confirmatory"
        ),
        "formal_spec": {
            "path": str(args.formal_spec),
            "sha256": sha256_file(args.formal_spec),
            "schema": FORMAL_SPEC_SCHEMA,
            "study_plan": loaded["study_plan"],
        },
        "task_split": {
            "calibration": list(loaded["calibration_tasks"]),
            "evaluation": list(loaded["evaluation_tasks"]),
            "main_snapshot_counts": {
                str(key): value for key, value in sorted(task_snapshot_counts.items())
            },
        },
        "calibration": calibration.to_dict(),
        "route_metric": "RMS Hellinger over 8 HB x 10 denoise x 10 action tokens x 32 experts",
        "primary_effect": "within-snapshot matched mean raw d_route(different - same)",
        "residual_sensitivity": (
            "task-0 snapshot-equal weighted quadratic d_route ~ d_action; not primary"
        ),
        "hidden_baseline": {
            "projection": "deterministic same-site Rademacher 1024 -> 32",
            "seed": loaded["projection_seed"],
            "status": "secondary",
        },
        "physical_trajectory": {
            "metric": "task-0-scaled RMS dense physical trajectory distance",
            "role": "far-action primary eligibility criterion and descriptive effect",
            "cross_task_schema": "frozen task-0 schema with per-task presence masks",
        },
        "model_flow_trajectory": {
            "metric": "RMS over normalized [11 flow states,10 action tokens,24 dims]",
            "status": "independent secondary distance",
        },
        "outcome_rule": {
            "same": "events_equal exact and paired Q-difference CI inside [-0.1,0.1]",
            "different": "events unequal or paired Q-difference CI outside +/-0.1",
            "otherwise": "ambiguous and excluded from labels, not horizon censoring",
        },
        "matching": {
            "variables": ["d_action"],
            "algorithm": "one-to-one maximum-cardinality then minimum-cost",
            "caliper": "0.2 * task-0 snapshot-equal within-stratum action-distance SD",
            "min_matches_per_snapshot": min_matches,
            "minimum_unique_candidates_per_outcome_group": 4,
            "quadratic_residual": "route-only sensitivity; task-0 fit, frozen on task 1/3",
            "far_requires_physical_trajectory_far": True,
        },
        "screening": {
            "rule": {
                "same": "events equal and |Delta Qhat_8| <= 0.125",
                "different": "events unequal or |Delta Qhat_8| >= 0.25",
                "support": "each relation >=2 pairs and >=3 unique candidates",
                "route_or_hidden_used": False,
            },
            "metrics": screen_result,
        },
        "global_topup": global_topup,
        "queues": {"main": main, "enriched": enriched},
        "horizon_censoring": {
            "main": main["horizon_censoring"],
            "enriched": enriched["horizon_censoring"],
        },
        "gates": gates,
        "effect_direction_is_gate": False,
        "inference_unit": (
            "equal-task macro after episode and snapshot reduction; candidate pairs are "
            "never bootstrap or sign-flip units"
        ),
        "candidate_stores": loaded["stores"],
        "provenance": loaded["provenance"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "calibration.json").write_text(
        json.dumps(calibration.to_dict(), indent=2), encoding="utf-8"
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(_formal_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {"confirmatory": False, "gates": gates, "global_topup": global_topup},
            indent=2,
        )
    )
    print(f"wrote {args.out_dir}")
    return 0


def _run_legacy(args: argparse.Namespace) -> int:
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")
    pairs = _required_legacy_path(args.pairs, "--pairs")
    phase1_summary = _required_legacy_path(args.phase1_summary, "--phase1-summary")
    fork_records = _required_legacy_path(args.fork_records, "--fork-records")
    routes = _required_legacy_path(args.routes, "--routes")
    phase1 = load_json(phase1_summary)
    if phase1.get("mode") != "legacy":
        raise SystemExit("this small validation expects the legacy phase-1 proxy output")
    thresholds = phase1["thresholds"]
    outcome_thresholds = phase1["proxy_outcome_thresholds"]
    original_fields, rows = load_pair_rows(pairs)
    route_diagnostics = route_distances_by_snapshot(rows, fork_records, routes)

    for row in rows:
        action_stratum, outcome_relation = label_pair(
            row,
            action_near=float(thresholds["action_near"]),
            action_far=float(thresholds["action_far"]),
            outcome_same_max=float(outcome_thresholds["proxy_outcome_near_q20"]),
            outcome_different_min=float(outcome_thresholds["proxy_outcome_far_q80"]),
        )
        row["action_stratum"] = action_stratum
        row["outcome_relation"] = outcome_relation

    inequalities = {
        "far_action": conditional_route_contrast(
            rows,
            action_stratum="far",
            draws=args.bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        ),
        "near_action": conditional_route_contrast(
            rows,
            action_stratum="near",
            draws=args.bootstrap,
            confidence=args.confidence,
            seed=args.seed + 10,
        ),
    }
    both_point = all(
        result.get("point_estimate_holds", False) for result in inequalities.values()
    )
    both_ci = all(
        result.get("confidence_interval_above_zero", False)
        for result in inequalities.values()
    )
    summary = {
        "analysis": "route--outcome geometry conditional on action distance",
        "mode": "legacy_proxy_exploratory",
        "confirmatory": False,
        "route_distance": {
            "name": "RMS Hellinger",
            "sites": "8 HB layers x 10 denoise rounds x 10 action tokens",
            "expert_axis": 32,
            "probability_normalization": "per aligned routing site",
            "range": [0.0, 1.0],
        },
        "provenance": {
            "pairs": str(pairs),
            "pairs_sha256": sha256_file(pairs),
            "phase1_summary": str(phase1_summary),
            "phase1_summary_sha256": sha256_file(phase1_summary),
            "fork_records": str(fork_records),
            "fork_records_sha256": sha256_file(fork_records),
            "routes": str(routes),
        },
        "snapshot_count": int(len({row["snapshot"] for row in rows})),
        "pair_count": int(len(rows)),
        "thresholds_reused_without_refitting": {
            "action_near": float(thresholds["action_near"]),
            "action_far": float(thresholds["action_far"]),
            "outcome_same_max": float(outcome_thresholds["proxy_outcome_near_q20"]),
            "outcome_different_min": float(
                outcome_thresholds["proxy_outcome_far_q80"]
            ),
        },
        "route_alignment": route_diagnostics,
        "inequalities": inequalities,
        "decision": {
            "both_point_estimates_hold": bool(both_point),
            "both_snapshot_bootstrap_intervals_above_zero": bool(both_ci),
            "phase2_small_validation_passed": bool(both_ci),
            "rule": "pass only if both primary snapshot-bootstrap intervals are above zero",
        },
        "inference_unit": "snapshot after within-snapshot pair reduction",
        "limitations": [
            "Outcome sameness/difference is the phase-1 drawer-endpoint quantile proxy, not repeated-continuation Q equivalence.",
            "Action and outcome thresholds were fit post hoc on these same 20 snapshots.",
            "All snapshots come from one task and only five source episodes; there is no held-out task validation.",
            "Several snapshots have degenerate proxy outcomes, reducing the number eligible for within-snapshot contrasts.",
            "Pair mining and this follow-up reuse the same sample; selected effects require independent confirmation.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_pairs(args.out_dir / "pairs.csv", original_fields, rows)
    (args.out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "inequalities": inequalities}, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")
    if args.mode == "formal":
        return _run_formal(args)
    return _run_legacy(args)


if __name__ == "__main__":
    raise SystemExit(main())
