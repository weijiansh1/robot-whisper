#!/usr/bin/env python3
"""Read-only audit of the existing VLA hub; write evidence outside the hub."""

import argparse
import collections
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import zarr


EXPECTED = {
    "hb_expert_ids": ((8, 10, 11, 4), "uint8"),
    "hb_selected_prob": ((8, 10, 11, 4), "float16"),
    "hb_router_probs": ((8, 10, 11, 32), "float16"),
    "hb_entropy": ((8, 10, 11), "float16"),
    "as_expert_ids": ((4,), "uint8"),
    "as_probs": ((4, 3), "float16"),
    "episode_id": ((), "int32"),
    "control_step": ((), "int32"),
}
CHECKPOINT_SHA = {
    "libero_goal": "98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953",
    "libero_spatial": "1029d0827030a7521361d1904eeb3e7e7f2792be5c99abdb5701abb7ee87c137",
    "libero_object": "f9c5661533d271dec15d54d56fcd8c6c8811fc2b96095ac87638f7b3b2bdaafa",
    "libero_long": "cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256",
}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def issue(result, code, detail):
    result["issues"].append({"code": code, "detail": detail})


def audit_run(run, hub, batch):
    result = {"path": str(run.relative_to(hub)), "issues": []}
    meta = read_json(run / "meta.json", {})
    capture = read_json(run / "server/capture_summary.json", {})
    store_path = run / "server/routes.zarr"
    group = zarr.open_group(str(store_path), mode="r")
    arrays = dict(group.arrays())
    n = arrays["control_step"].shape[0]
    result.update(rows=n, attrs=dict(group.attrs), hidden_present=(run / "server/hidden.zarr").is_dir())
    result["hidden_declared"] = bool(meta.get("store_hidden") or capture.get("store_hidden"))
    result["hook_verified_calls"] = capture.get("hook_verified_calls")
    result["hook_verify_failures"] = capture.get("hook_verify_failures")
    if capture.get("hook_verify_failures"):
        issue(result, "hook_verification_failed", capture["hook_verify_failures"])
    if capture.get("control_steps") != n:
        issue(result, "capture_count_mismatch", capture.get("control_steps"))
    if group.attrs.get("durable_rows", n) != n:
        issue(result, "durable_count_mismatch", group.attrs.get("durable_rows"))
    if capture.get("as_collapsed", n) != n:
        issue(result, "as_collapse_count_mismatch", capture.get("as_collapsed"))
    result["arrays"] = {}
    if set(arrays) != set(EXPECTED):
        issue(result, "array_keys_mismatch", sorted(arrays))
    for name, (tail, dtype) in EXPECTED.items():
        arr = arrays[name]
        if arr.shape != (n, *tail) or str(arr.dtype) != dtype:
            issue(result, "schema_mismatch", {"name": name, "shape": arr.shape, "dtype": str(arr.dtype)})
        physical = sum(len(files) for _, _, files in os.walk(store_path / name / "c"))
        logical = math.prod(math.ceil(a / b) for a, b in zip(arr.shape, arr.chunks))
        result["arrays"][name] = {"shape": arr.shape, "dtype": str(arr.dtype), "chunks": arr.chunks,
                                  "logical_chunks": logical, "physical_chunks": physical}
        if physical != logical and name != "episode_id":
            issue(result, "unexpected_chunk_count", {"name": name, "logical": logical, "physical": physical})
    episode = arrays["episode_id"][:]
    control = arrays["control_step"][:]
    if not np.array_equal(control, np.arange(n)):
        issue(result, "control_not_contiguous", int(np.count_nonzero(control != np.arange(n))))
    result["unique_episode_ids"] = int(len(np.unique(episode)))
    result["episode_id_all_zero"] = bool(np.all(episode == 0))
    rows = read_json(run / "client/summaries.json")
    if rows is None:
        rows = [json.loads(line) for line in (run / "client/sequences.jsonl").read_text().splitlines() if line.strip()]
        summary = read_json(run / "client/summary.json", {})
        result["kind"] = "calvin"
        result["episodes"] = len(rows)
        result["selected_sequence_count"] = summary.get("selected_sequence_count")
        result["coverage_complete"] = summary.get("coverage_complete")
        result["requested_new_sequences"] = meta.get("sampling", {}).get("max_new_sequences")
        if not result["coverage_complete"]:
            issue(result, "calvin_universe_incomplete", {"observed": len(rows), "selected": result["selected_sequence_count"],
                                                        "local_plan": result["requested_new_sequences"]})
        expected_ep = np.repeat([row["sequence_index"] for row in rows], [row["inference_calls"] for row in rows])
        state = np.load(run / "server/state.npy", allow_pickle=False)
        prompts = read_json(run / "server/prompts.json")
        result["state_shape"] = state.shape
        result["prompt_count"] = len(prompts)
        if len(state) != n or not np.isfinite(state).all() or len(prompts) != n:
            issue(result, "calvin_sidecar_mismatch", {"state_rows": len(state), "prompts": len(prompts)})
        result["meta_status"] = meta.get("status")
    else:
        result["kind"] = "libero"
        result["episodes"] = len(rows)
        result["successes"] = sum(bool(row["success"]) for row in rows)
        expected_ep = np.repeat([row["episode_index"] for row in rows], [row["inference_calls"] for row in rows])
        if [row["episode_index"] for row in rows] != list(range(len(rows))):
            issue(result, "episode_index_not_contiguous", None)
        pairs = [(row["init_state_id"], row["flow_noise_seed"]) for row in rows]
        inits, seeds = sorted({p[0] for p in pairs}), sorted({p[1] for p in pairs})
        result.update(init_ids=inits, noise_seeds=seeds, duplicate_pairs=len(pairs) - len(set(pairs)))
        if len(set(pairs)) != len(pairs) or set(pairs) != set(itertools.product(inits, seeds)):
            issue(result, "sampling_grid_incomplete_or_duplicate", {"unique_pairs": len(set(pairs)), "grid": len(inits) * len(seeds)})
        sampling = meta.get("sampling", {})
        for key, actual in [("actual_episodes", len(rows)), ("designed_episodes", len(rows)),
                            ("unique_init_states", len(inits)), ("unique_flow_noise_seeds", len(seeds))]:
            if sampling.get(key) != actual:
                issue(result, "sampling_metadata_mismatch", {"key": key, "stored": sampling.get(key), "actual": actual})
        if not sampling.get("complete"):
            issue(result, "sampling_not_marked_complete", sampling)
        if run.name in {"right-50x8-20260903", "right-50x8b-20260903"}:
            seed_base = 1008 if "8b-" in run.name else 1000
            if inits != list(range(50)) or seeds != list(range(seed_base, seed_base + 8)):
                issue(result, "campaign_grid_mismatch", {"inits": inits, "seeds": seeds})
        metadata = read_json(run / "client/server_metadata.json", {})
        result["checkpoint_sha256"] = metadata.get("checkpoint_sha256")
        if result["checkpoint_sha256"] != CHECKPOINT_SHA[run.parent.parent.name]:
            issue(result, "checkpoint_metadata_mismatch", result["checkpoint_sha256"])
        if metadata.get("episode_id_key") != "episode_id" or metadata.get("libero_wrist_layout") != "paper-right":
            issue(result, "server_contract_mismatch", None)
        result["client_routing_capture"] = metadata.get("client_routing_capture")
        layout = read_json(run / "client/sim_layout.json", {})
        sim_width = 1 + layout.get("nq", 0) + layout.get("nv", 0)
        files = {int(p.stem.split("_")[-1]): p for p in (run / "client").glob("episode_*.npz")}
        result["npz_files"] = len(files)
        if set(files) != {r["episode_index"] for r in rows}:
            issue(result, "npz_episode_set_mismatch", {"files": len(files), "summaries": len(rows)})
        placeholders = no_routes = 0
        for row in rows:
            path = files.get(row["episode_index"])
            if path is None:
                continue
            try:
                with zipfile.ZipFile(path) as archive:
                    bad = archive.testzip()
                    if bad:
                        issue(result, "npz_crc_failure", {"file": path.name, "member": bad})
                with np.load(path, allow_pickle=False) as data:
                    count = row["inference_calls"]
                    for key, shape in [("state", (count, 8)), ("actions", (count, 10, 7)), ("sim_state", (count, sim_width))]:
                        value = data[key]
                        if value.shape != shape or not np.isfinite(value).all():
                            issue(result, "npz_numeric_or_shape_failure", {"file": path.name, "key": key, "shape": value.shape, "expected": shape})
                    if "expert_ids" in data.files:
                        ids, weights = data["expert_ids"], data["expert_weights"]
                        stub = ids.shape == (count, 1) and weights.shape == (count, 1) and not np.any(ids) and not np.any(weights)
                        placeholders += int(stub)
                        if not stub:
                            issue(result, "unexpected_npz_routing", path.name)
                    else:
                        no_routes += 1
                if math.ceil(row["action_steps"] / 10) != count:
                    issue(result, "action_inference_count_mismatch", row["episode_index"])
                if row["task_name"] != run.parent.name or row["prompt"] != meta.get("prompt"):
                    issue(result, "task_prompt_mismatch", row["episode_index"])
            except Exception as exc:
                issue(result, "npz_read_error", {"file": path.name, "error": repr(exc)})
        result.update(npz_placeholders=placeholders, npz_without_routes=no_routes)
    result["inference_calls"] = int(sum(row["inference_calls"] for row in rows))
    if len(expected_ep) != n:
        issue(result, "client_server_count_mismatch", {"calls": len(expected_ep), "rows": n})
    elif not np.array_equal(expected_ep, episode):
        issue(result, "episode_alignment_mismatch", int(np.count_nonzero(expected_ep != episode)))

    metrics = collections.Counter()
    maximum = collections.defaultdict(float)
    route_hash = hashlib.sha256()
    for start in range(0, n, batch):
        block = {name: arr[start:start + batch] for name, arr in arrays.items() if name not in {"episode_id", "control_step"}}
        for name in sorted(block):
            route_hash.update(block[name].tobytes())
            if block[name].dtype.kind == "f":
                metrics["nonfinite_values"] += int(np.count_nonzero(~np.isfinite(block[name])))
        ids = block["hb_expert_ids"].astype(np.int64)
        p = block["hb_router_probs"].astype(np.float32)
        raw = block["hb_selected_prob"].astype(np.float32)
        metrics["hb_sites"] += int(np.prod(ids.shape[:-1]))
        metrics["hb_id_out_of_range"] += int(np.count_nonzero(ids >= 32))
        metrics["hb_duplicate_top4"] += int(np.count_nonzero(np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0, axis=-1)))
        metrics["hb_invalid_probabilities"] += int(np.count_nonzero((p < 0) | (p > 1)))
        metrics["hb_zero_probability_rows"] += int(np.count_nonzero(p.sum(-1) == 0))
        maximum["hb_probability_sum_error"] = max(maximum["hb_probability_sum_error"], float(np.max(np.abs(p.sum(-1) - 1))))
        if not np.any(ids >= 32):
            selected = np.take_along_axis(p, ids, axis=-1)
            metrics["selected_prob_mismatches"] += int(np.count_nonzero(selected != raw))
            # No unselected expert may strictly exceed the weakest selected one.
            threshold = selected.min(-1, keepdims=True)
            higher = p > threshold
            np.put_along_axis(higher, ids, False, axis=-1)
            metrics["strict_top4_violations"] += int(np.count_nonzero(np.any(higher, axis=-1)))
        entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(-1)
        maximum["entropy_recompute_error"] = max(maximum["entropy_recompute_error"], float(np.max(np.abs(entropy - block["hb_entropy"].astype(np.float32)))))
        ap = block["as_probs"].astype(np.float32)
        ai = block["as_expert_ids"].astype(np.int64)
        metrics["as_id_out_of_range"] += int(np.count_nonzero(ai >= 3))
        metrics["as_invalid_probabilities"] += int(np.count_nonzero((ap < 0) | (ap > 1)))
        maximum["as_probability_sum_error"] = max(maximum["as_probability_sum_error"], float(np.max(np.abs(ap.sum(-1) - 1))))
        if not np.any(ai >= 3):
            metrics["as_not_maximal"] += int(np.count_nonzero(np.take_along_axis(ap, ai[..., None], -1)[..., 0] < ap.max(-1)))
    result["numeric"] = {**metrics, **maximum}
    result["route_value_sha256"] = route_hash.hexdigest()
    pinned = run.name in {"pin-on", "pin-off", "pin-smoke"}
    for key, value in metrics.items():
        if key == "hb_sites" or (key == "strict_top4_violations" and pinned):
            continue
        if value:
            issue(result, "numeric_" + key, value)
    for key, tolerance in [("hb_probability_sum_error", 0.01), ("as_probability_sum_error", 0.01), ("entropy_recompute_error", 0.02)]:
        if maximum[key] > tolerance:
            issue(result, "numeric_" + key, maximum[key])
    return result


def audit_supplement(hub):
    result = {"pin_weights": [], "calvin_boundaries": [], "manifest_coverage": []}
    pin_path = hub.parent / "pin_libero_spatial_pick_up_the_black_bowl_on_th.json"
    pin = read_json(pin_path)
    result["pin_table"] = str(pin_path)
    result["pin_table_sha256"] = hashlib.sha256(pin_path.read_bytes()).hexdigest()
    base = hub / "cache/HiMoE-VLA/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
    for name, regime in [("pin-on", "on"), ("pin-off", "off"), ("pin-smoke", "off")]:
        group = zarr.open_group(str(base / name / "server/routes.zarr"), mode="r")
        ids = group["hb_expert_ids"][:, :4, :, 0, :]
        raw = group["hb_selected_prob"][:, :4, :, 0, :].astype(np.float32)
        derived = raw / raw.sum(-1, keepdims=True)
        want_ids = np.asarray([pin["layers"][str(layer)][regime]["experts"] for layer in (2, 3, 4, 5)])[None, :, None, :]
        want_weights = np.asarray([pin["layers"][str(layer)][regime]["weights"] for layer in (2, 3, 4, 5)], np.float32)[None, :, None, :]
        difference = np.abs(derived - want_weights)
        result["pin_weights"].append({
            "path": str((base / name).relative_to(hub)),
            "ids_match_pin": bool(np.all(ids == want_ids)),
            "pinned_sites": int(np.prod(ids.shape[:-1])),
            "comparison": "normalized stored raw probabilities versus nominal pin-table weights (before runtime dtype casting)",
            "max_weight_error": float(difference.max()),
            "mean_weight_error": float(difference.mean()),
            "sites_with_error_gt_0.01": int(np.count_nonzero(np.any(difference > 0.01, -1))),
            "issue": "Executed pin weights are not stored in routes.zarr; raw-probability normalization cannot recover them.",
        })
    duplicate_count = 0
    npz_paths = sorted((base / "pin-base/client").glob("episode_*.npz"))
    for path in npz_paths:
        other = base / "right-16x32/client" / path.name
        duplicate_count += hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(other.read_bytes()).digest()
    result["pin_base_duplicate_npz"] = {"checked": len(npz_paths), "byte_identical": duplicate_count}
    for run in sorted(hub.glob("cache*/HiMoE-VLA/calvin_d/task_D_D/*")):
        rows = [json.loads(line) for line in (run / "client/sequences.jsonl").read_text().splitlines() if line.strip()]
        prompts = read_json(run / "server/prompts.json")
        cursor = 0
        count_mismatches = prompt_mismatches = 0
        for sequence in rows:
            expected = []
            for subtask in sequence["subtasks"]:
                expected.extend([subtask["annotation"]] * math.ceil(subtask["environment_steps"] / 10))
            count = sequence["inference_calls"]
            count_mismatches += count != len(expected)
            prompt_mismatches += prompts[cursor:cursor + count] != expected
            cursor += count
        result["calvin_boundaries"].append({"path": str(run.relative_to(hub)), "sequences": len(rows),
                                            "rows": cursor, "subtask_inference_count_mismatches": count_mismatches,
                                            "prompt_sequence_mismatches": prompt_mismatches})
    manifest = read_json(hub / "manifest.json")
    for run_id in ["right-50x8-20260903", "right-50x8b-20260903"]:
        counts = collections.Counter()
        mismatches = []
        for benchmark, spec in manifest["benchmarks"].items():
            if not benchmark.startswith("libero"):
                continue
            suite = spec.get("hub_dir", benchmark)
            for task in spec["tasks"]:
                run = hub / "cache_new/HiMoE-VLA" / suite / task["name"] / run_id
                rows = read_json(run / "client/summaries.json")
                if rows is None:
                    mismatches.append({"suite": suite, "task": task["name"], "reason": "missing summaries"})
                    continue
                counts[suite] += 1
                for row in rows:
                    if row["task_id"] != task["task_id"] or row["prompt"] != task["language"]:
                        mismatches.append({"suite": suite, "task": task["name"], "episode_index": row["episode_index"]})
        result["manifest_coverage"].append({"run_id": run_id, "tasks_by_suite": dict(counts), "mismatches": mismatches})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", type=Path, default=Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--supplement-only", action="store_true", help="Add pin/manifest/CALVIN checks to a completed full audit")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.hub.resolve()):
        raise ValueError("Audit output must be outside the audited hub")
    if args.supplement_only:
        result = read_json(args.output)
        if not result.get("finished_utc"):
            raise ValueError("Full audit must finish before adding supplemental checks")
        result["supplement"] = audit_supplement(args.hub)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["supplement"], indent=2))
        return
    started = time.monotonic()
    result = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "hub": str(args.hub), "numpy": np.__version__, "zarr": zarr.__version__,
              "scope": "Routing capture and client alignment; hidden activations are not required by the user.",
              "method": "Every route array fully decoded and checked; every LIBERO episode NPZ CRC and numeric arrays checked; metadata checked for all runs. No inference replay or checkpoint hashing.",
              "runs": []}
    runs = sorted(p.parent.parent for p in args.hub.glob("cache*/HiMoE-VLA/*/*/*/server/routes.zarr"))
    for index, run in enumerate(runs):
        print(f"START {index + 1}/{len(runs)} {run.relative_to(args.hub)}", flush=True)
        try:
            row = audit_run(run, args.hub, args.batch)
        except Exception as exc:
            row = {"path": str(run.relative_to(args.hub)), "issues": [{"code": "audit_exception", "detail": repr(exc)}]}
        result["runs"].append(row)
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"DONE rows={row.get('rows')} episodes={row.get('episodes')} issues={len(row['issues'])} elapsed={result['elapsed_seconds']}s", flush=True)
    counts = collections.Counter(i["code"] for r in result["runs"] for i in r["issues"])
    result["issue_counts"] = dict(counts)
    result["supplement"] = audit_supplement(args.hub)
    result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"runs": len(runs), "issue_counts": counts, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
