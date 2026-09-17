"""Bounded same-checkpoint, zero-bias readiness check. Not a control trial."""

import json
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np

from gate_runtime import BASE, infer_isolated, load_isolated
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.episode_trace import load_episode_trace
from collection_routes import CAPTURE_KEY
from scope_bias_control import BIAS_KEY


def main():
    root = Path(tempfile.mkdtemp(prefix="gate-preflight-", dir=BASE / "runs"))
    config = {
        "kind": "zero-bias memory and identity preflight", "benchmark": "pro", "task": 0,
        "episode_index": 1, "init_state_id": 47, "query": 8,
        "shared_reference_request_id": -1260920001,
        "allocator_limit_mib": 16384, "nonzero_bias_forwards": 0,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"run": str(root), "status": "loading"}), flush=True)
    report = {"passed": False, "shared_calls": 0, "isolated_completed_calls": 0,
              "new_environment_actions": 0, "started_unix": time.time()}
    try:
        manifest = json.loads(Path("/data/libero-runtime/samples/pro/batch-libero-long-20260914T115911968889Z/manifest.json").read_text())
        job, = [j for j in manifest["jobs"] if j["task_id"] == 0 and j["episode_index"] == 1]
        source, arrays = load_episode_trace(Path(job["result"]["artifact_dir"]))
        if not source["result"]["trace_complete"] or job["init_state_id"] != 47:
            raise RuntimeError("Unexpected source trace")
        q = config["query"]
        request = {"observation/image": arrays["images"][q], "observation/wrist_image": arrays["wrist_images"][q],
                   "observation/state": arrays["states"][q], "prompt": source["prompt"],
                   "flow/noise": arrays["flow_noises"][q], "routing/capture": True}
        wrapped, loaded = load_isolated()
        report["loaded"] = loaded
        report["source"] = job["result"]["artifact_dir"]
        report["trace_sha256"] = source["array_file_sha256"]
        print(json.dumps({"status": "loaded", **{k: v for k, v in loaded.items() if k != "metadata"}}), flush=True)
        with PolicyClient("127.0.0.1", 9500, inference_timeout=180) as client:
            report["shared_metadata"] = client.metadata
            for key, value in source["policy_identity"].items():
                if client.metadata.get(key) != value or loaded["metadata"].get(key) != value:
                    raise RuntimeError("Policy identity mismatch: " + key)
            live = client.infer(dict(request, episode_id=config["shared_reference_request_id"]))
            report["shared_calls"] += 1
        original = None
        for name, extra in (("bare", {}), ("full", {CAPTURE_KEY: True}),
                            ("zero_bias", {CAPTURE_KEY: True, BIAS_KEY: np.zeros((8, 10, 11, 32), np.float32)}),
                            ("post", {CAPTURE_KEY: True})):
            response, resources = infer_isolated(wrapped, dict(request, **extra))
            report["isolated_completed_calls"] += 1
            for key in ("actions", "routing/expert_ids", "routing/expert_weights"):
                if not np.array_equal(response[key], live[key]):
                    raise RuntimeError("Isolated/reference mismatch: " + name + "/" + key)
            if not np.array_equal(response["actions"], arrays["predicted_actions"][q]):
                raise RuntimeError("Source action mismatch")
            np.savez_compressed(root / (name + ".npz"), **{k: v for k, v in response.items() if isinstance(v, np.ndarray)})
            if name == "full":
                original = response
            elif name in ("zero_bias", "post"):
                for key in ("collection/hb_probs", "collection/hb_effective_ids", "collection/hb_effective_weights"):
                    if not np.array_equal(response[key], original[key]):
                        raise RuntimeError("Full route zero/post mismatch: " + key)
            report[name] = {"reference_exact": True, **resources}
            print(json.dumps({"status": name, **report[name]}), flush=True)
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        report["error_type"] = type(error).__name__
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished_unix"] = time.time()
        (root / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"run": str(root), "passed": report["passed"]}), flush=True)


if __name__ == "__main__":
    main()
