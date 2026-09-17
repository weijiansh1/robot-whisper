"""Small actual-model replay to inspect routed expert output and shared output."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

for path in ("/data/coding/moe-control-experiments", "/data/coding/robot-whisper-0909/himoe-route-capture",
             "/data/coding/v8-methods", "/data/srv/src"):
    sys.path.insert(0, path)

from gate_runtime import load_isolated
from gate_capture import CAPTURE_KEY, PROBS_KEY, NATIVE_IDS_KEY
from himoe_functional_recorder import HBFunctionalSnapshotRecorder, save_functional_record
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file
from moe_joint_patterns import motif_scores, MOTIF_NAMES
from analyze_moe_joint_patterns import save_csv, save_json, read_json

ROOT = Path("/data/libero-runtime/samples/moe-joint-patterns-20260915")
OUT = ROOT / "functional-probe"
SOURCE = Path("/data/libero-runtime/samples/v82-evaluation-20260914T144322Z")
SAMPLES = (
    ("plus-task05-init026", (15, 18)),
    ("plus-task05-init047", (18, 27)),
    ("plus-task04-init047", (18, 27)),
    ("plus-task04-init026", (18, 19)),
    ("plus-task07-init026", (18, 24)),
    ("plus-task07-init047", (18, 40)),
)
MEASURES = ("routed_authority", "expert_cancellation", "expert_disagreement_ratio",
            "routed_shared_cosine", "routed_output_norm", "shared_output_norm")


def infer(wrapped, request):
    import torch
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
        response = wrapped.infer(dict(request))
    torch.cuda.synchronize()
    if response.get("flow/noise_sha256") != hashlib.sha256(request["flow/noise"].tobytes()).hexdigest():
        raise RuntimeError("Noise acknowledgment differs")
    return response, time.perf_counter() - started


def summarize(record, metadata):
    rows = []
    for layer_name, ls in (("front", slice(0, 4)), ("back", slice(4, 8))):
        for flow_name, ds in (("early", slice(0, 3)), ("late", slice(7, 10))):
            for token_name, ts in (("state", slice(0, 1)), ("action", slice(1, 11))):
                values = {key: float(getattr(record, key)[0, ls, ds, ts].astype(float).mean()) for key in MEASURES}
                contribution = record.expert_contrib_norm[0, ls, ds, ts].astype(float)
                weights = record.topk_exec_weight[0, ls, ds, ts].astype(float)
                # Retain ties: a tied maximum is compatible with either expert.
                maximum_weight = weights == weights.max(-1, keepdims=True)
                maximum_norm = contribution == contribution.max(-1, keepdims=True)
                values["largest_weight_and_contribution_differ"] = float((~(maximum_weight & maximum_norm).any(-1)).mean())
                rows.append(dict(**metadata, layers=layer_name, flow=flow_name, tokens=token_name, **values))
    return rows


def main():
    import torch
    if OUT.exists():
        raise SystemExit("Choose a new directory; never overwrite a model probe")
    OUT.mkdir(parents=True)
    protocol = dict(
        started_utc=datetime.now(timezone.utc).isoformat(), status="Purposeful development case probe",
        samples=[dict(episode=name, queries=list(q)) for name, q in SAMPLES],
        planned_model_forwards=24, new_environment_actions=0, online_detector_changed=False,
        model_weights_changed=False, shared_service_used=False,
        order="12 native/captured pairs, alternating order across pairs; compare both with archived native actions and full HB",
        metrics=list(MEASURES) + ["largest_weight_and_contribution_differ"],
        capture="Existing single-dispatch functional recorder; selected experts are evaluated once per forward",
        limitations=["No causal intervention on expert contributions", "Six selected parents, not twelve independent tasks",
                     "Output norm share is not causal importance", "Cancellation is vector cancellation, not proof of semantic conflict",
                     "Measured capture overhead is for this diagnostic implementation, not an optimized online monitor"],
    )
    save_json(OUT / "protocol.json", protocol)
    paths = [Path(__file__), Path("/data/coding/moe-control-experiments/gate_runtime.py"),
             Path("/data/coding/moe-control-experiments/gate_capture.py"),
             Path("/data/coding/robot-whisper-0909/himoe-route-capture/himoe_functional_recorder.py"),
             Path("/data/coding/robot-whisper-0909/himoe-route-capture/himoe_state_recorder.py"),
             ROOT / "episode-features.npz", SOURCE / "summary.json"]
    hashes = {str(p): sha256_file(p) for p in paths}
    episodes = {e["name"]: e for e in read_json(SOURCE / "summary.json")["episodes"]}
    with np.load(ROOT / "episode-features.npz") as archive:
        scores = {name: motif_scores(archive[name]) for name, _ in SAMPLES}
    started = time.perf_counter()
    wrapped, load = load_isolated()
    save_json(OUT / "model-load.json", load)
    calls, summaries, checks = [], [], []
    sample_id = 0
    for name, queries in SAMPLES:
        e = episodes[name]
        manifest, arrays = load_episode_trace(Path(e["source_artifact_dir"]))
        hashes[str(Path(e["source_artifact_dir"]) / "episode-trace.npz")] = manifest["array_file_sha256"]
        route_path = SOURCE / name / "full-hb-routes.npz"
        hashes[str(route_path)] = sha256_file(route_path)
        assert hashes[str(route_path)] == e["integrity"]["full_hb_sha256"]
        with np.load(route_path) as archive:
            hb, expert_ids = archive["hb_router_probs"], archive["hb_expert_ids"]
        for q in queries:
            request = {
                "observation/image": arrays["images"][q], "observation/wrist_image": arrays["wrist_images"][q],
                "observation/state": arrays["states"][q], "prompt": e["prompt"],
                "flow/noise": arrays["flow_noises"][q], "episode_id": -2026091500 - sample_id,
                "routing/capture": True, CAPTURE_KEY: True,
            }
            responses, times = {}, {}
            modes = ("native", "captured") if sample_id % 2 == 0 else ("captured", "native")
            for mode in modes:
                if mode == "captured":
                    recorder = HBFunctionalSnapshotRecorder(wrapped.policy._policy.model, expected_denoise=10,
                                                            sketch_dim=16).attach()
                    try:
                        recorder.begin(request["episode_id"], q)
                        responses[mode], times[mode] = infer(wrapped, request)
                        record = recorder.end()
                    finally:
                        recorder.close()
                else:
                    responses[mode], times[mode] = infer(wrapped, request)
                if any(gate._forward_hooks for _, gate in wrapped.policy._routing_layers + wrapped.as_layers):
                    raise RuntimeError("A capture hook leaked into another request")
                calls.append(dict(sample=sample_id, episode=name, query=q, mode=mode, seconds=times[mode],
                                  noise_sha256=responses[mode]["flow/noise_sha256"]))
                save_json(OUT / "calls.json", calls)
            check = dict(sample=sample_id, episode=name, query=q,
                         native_actions_exact=np.array_equal(responses["native"]["actions"], arrays["predicted_actions"][q]),
                         captured_actions_exact=np.array_equal(responses["captured"]["actions"], arrays["predicted_actions"][q]),
                         native_hb_exact=np.array_equal(responses["native"][PROBS_KEY], hb[q]),
                         captured_hb_exact=np.array_equal(responses["captured"][PROBS_KEY], hb[q]),
                         dispatch_ids_exact=np.array_equal(record.topk_idx[0], expert_ids[q]),
                         all_record_arrays_finite=all(np.isfinite(v).all() for v in vars(record).values() if isinstance(v, np.ndarray)))
            if not all(check[k] for k in ("native_actions_exact", "captured_actions_exact", "native_hb_exact",
                                         "captured_hb_exact", "dispatch_ids_exact", "all_record_arrays_finite")):
                save_json(OUT / "failed-check.json", check)
                raise RuntimeError("Functional recorder changed inference or emitted invalid data")
            sample_path = OUT / (name + "-q%03d.npz" % q)
            save_functional_record(record, sample_path)
            np.savez_compressed(OUT / (name + "-q%03d-replay.npz" % q),
                                native_actions=responses["native"]["actions"], captured_actions=responses["captured"]["actions"],
                                native_hb=responses["native"][PROBS_KEY], captured_hb=responses["captured"][PROBS_KEY])
            metadata = dict(sample=sample_id, episode=name, query=q, success=e["source_result"]["success"],
                            joint_active=bool((scores[name][q] >= np.log(1.2)).any()))
            summaries.extend(summarize(record, metadata))
            check.update(native_seconds=times["native"], captured_seconds=times["captured"],
                         delta_seconds=times["captured"] - times["native"], snapshot_bytes=sample_path.stat().st_size,
                         snapshot_array_bytes=record.array_bytes)
            checks.append(check)
            print("Functional replay verified:", name, q, "overhead_s=%.3f" % check["delta_seconds"], flush=True)
            save_json(OUT / "checks.json", checks)
            save_csv(OUT / "functional-readouts.csv", summaries)
            sample_id += 1
    save_json(OUT / "source-hashes.json", hashes)
    result = dict(status="complete", model_forwards=len(calls), snapshots=len(checks), independent_parents=len(SAMPLES),
                  new_environment_actions=0, exact_native_and_captured_replays=all(r["captured_actions_exact"] for r in checks),
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024 ** 2,
                  elapsed_seconds=time.perf_counter() - started,
                  median_paired_capture_overhead_seconds_excluding_first=float(np.median([r["delta_seconds"] for r in checks[1:]])),
                  snapshot_bytes=sum(r["snapshot_bytes"] for r in checks),
                  all_sources_unchanged=all(sha256_file(Path(p)) == digest for p, digest in hashes.items()),
                  limitations=protocol["limitations"])
    save_json(OUT / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)
    del wrapped
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
