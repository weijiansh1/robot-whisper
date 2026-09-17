"""Independent CPU audit of saved gate dispatch and paired action responses."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def exact(actual, expected, label):
    if not np.array_equal(actual, expected):
        raise RuntimeError("Independent audit mismatch: " + label)


def close(actual, expected, label):
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Independent metric mismatch: " + label)


def expected_bias(spec):
    shape = (8, 10, 11, 32)
    if spec is None:
        return np.zeros(shape, np.float32)
    generator = np.random.default_rng(np.random.SeedSequence([2026091423, spec["direction"]]))
    draws = generator.random(shape)
    chosen = np.argsort(draws, axis=-1)[..., :16]
    direction = np.ones(shape, np.float32)
    np.put_along_axis(direction, chosen, -1, axis=-1)
    layer, token, time = spec["scope"].split("_")
    result = np.zeros(shape, np.float32)
    ls = slice(0, 4) if layer == "front" else slice(4, 8)
    ts = slice(0, 1) if token == "state" else slice(1, 11)
    ds = slice(0, 3) if time == "early" else slice(7, 10)
    result[ls, ds, ts] = direction[ls, ds, ts] * np.float32(spec["amplitude"] * spec["sign"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.run.resolve()
    config, collection, analysis = (load(root / name) for name in ("config.json", "collection.json", "analysis.json"))
    std = np.asarray(config["shared_metadata"]["normalization_action_std"][:6], float)
    grouped = defaultdict(list)
    action_map, baseline_map = {}, {}
    normalized_values = {}
    rows = collection["rows"]
    if [r["ordinal"] for r in rows] != list(range(config["isolated_forwards"])):
        raise RuntimeError("Isolated request accounting mismatch")
    events = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    if len(events) != 2 * len(rows):
        raise RuntimeError("Unacknowledged or missing request in log")
    for i in range(len(rows)):
        if events[2 * i]["event"] != "request" or events[2 * i + 1]["event"] != "response":
            raise RuntimeError("Request/response log order mismatch")
        if events[2 * i]["ordinal"] != i or events[2 * i + 1]["ordinal"] != i:
            raise RuntimeError("Request ordinal mismatch")
    full_count = 0
    cpu_normalization_max_error = 0.0
    for parent in config["parents"]:
        with np.load(root / "states" / parent["parent"] / "native.npz", allow_pickle=False) as saved:
            baseline_map[parent["parent"]] = saved["actions"].astype(float)
            for name in ("zero", "post"):
                with np.load(root / "states" / parent["parent"] / (name + ".npz"), allow_pickle=False) as control:
                    for key in saved.files:
                        exact(control[key], saved[key], "complete zero/post output arrays")
    for row in rows:
        path = root / row["path"]
        if sha(path) != row["sha256"]:
            raise RuntimeError("Raw response digest mismatch")
        with np.load(path, allow_pickle=False) as raw:
            bias = expected_bias(row["spec"])
            exact(bias, raw["gate_probe/request_bias"], "random bias/scope")
            if row["kind"] != "bare":
                full_count += 1
                dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}[row["audit"]["arithmetic_dtype"]]
                for phase, pkey in (("native", "v8_control/native_probs_fp32"), ("effective", "v8_control/effective_probs_fp32")):
                    p = torch.tensor(raw[pkey], dtype=dtype)
                    ids = torch.tensor(raw["collection/hb_%s_ids" % phase], dtype=torch.int64)
                    weights = raw["collection/hb_%s_weights" % phase]
                    selected = torch.gather(p, -1, ids)
                    expected_weights = selected / (selected.sum(-1, keepdim=True) + 1e-20)
                    error = float(np.max(np.abs(expected_weights.float().numpy() - weights)))
                    cpu_normalization_max_error = max(cpu_normalization_max_error, error)
                    # CPU and CUDA reduce four FP32 values in different orders.
                    tolerance = 1.2e-7 if dtype == torch.float32 else 0.0
                    if error > tolerance:
                        raise RuntimeError("CPU normalization exceeds arithmetic roundoff allowance")
                    expert_ids = ids.numpy()
                    if (np.any(expert_ids < 0) or np.any(expert_ids >= 32) or
                            np.any(np.diff(np.sort(expert_ids, axis=-1), axis=-1) == 0)):
                        raise RuntimeError("Invalid or duplicate per-site expert IDs")
                outside = np.max(np.abs(bias), axis=-1) == 0
                for field in ("ids", "weights"):
                    exact(raw["collection/hb_native_" + field][outside],
                          raw["collection/hb_effective_" + field][outside], "off-scope dispatch")
                exact(raw["collection/hb_effective_ids"][:, :, 1:].transpose(1, 0, 2, 3),
                      raw["routing/expert_ids"], "independent wire IDs")
                exact(raw["collection/hb_effective_weights"][:, :, 1:].transpose(1, 0, 2, 3),
                      raw["routing/expert_weights"], "independent wire weights")
            if row["kind"] in ("probe", "repeat"):
                spec = row["spec"]
                key = (row["parent"], spec["scope"], spec["direction"], spec["amplitude"], spec["sign"])
                if row["kind"] == "probe":
                    actions = raw["actions"].astype(float)
                    action_map[key] = actions
                    delta = (actions[:, :6] - baseline_map[row["parent"]][:, :6]) / std
                    rms = float(np.sqrt(np.sum(delta * delta) / 60))
                    normalized_values[key] = rms
                    grouped[(spec["scope"], spec["amplitude"], row["parent"])].append(rms)
                else:
                    exact(raw["actions"], action_map[key], "nonzero repeated actions")
                    original = next(r for r in rows if r["parent"] == row["parent"] and r["kind"] == "probe" and r["spec"] == spec)
                    with np.load(root / original["path"], allow_pickle=False) as reference:
                        for field in ("v8_control/effective_probs_fp32", "v8_control/native_probs_fp32", "collection/hb_effective_ids",
                                      "collection/hb_effective_weights", "collection/as_probs"):
                            exact(raw[field], reference[field], "nonzero repeated routes")
    for row in analysis["probe_metrics"]:
        key = (row["parent"], row["scope"], row["direction"], row["amplitude"], row["sign"])
        close(row["normalized_action_rms"], normalized_values[key], "per-probe normalized RMS")
        bias = expected_bias({k: row[k] for k in ("scope", "direction", "amplitude", "sign")})
        energy = float(np.sqrt(np.square(bias.astype(float)).sum()))
        close(row["bias_l2"], energy, "total bias L2")
        close(row["normalized_action_l2_per_bias_l2"], normalized_values[key] * np.sqrt(60) / energy, "energy-normalized gain")
    for row in analysis["paired_differences"]:
        key = (row["parent"], row["scope"], row["direction"], row["amplitude"])
        delta = (action_map[key + (1,)][:, :6] - action_map[key + (-1,)][:, :6]) / std
        close(row["symmetric_rms_per_amplitude"], np.sqrt(np.mean(delta ** 2)) / (2 * row["amplitude"]), "paired finite difference")
    for row in analysis["table"]:
        per_parent = [np.mean(values) for (scope, amplitude, _), values in grouped.items()
                      if scope == row["scope"] and amplitude == row["amplitude"]]
        close(row["mean_normalized_action_rms"], np.mean(per_parent), "parent-balanced table mean")
    summary = {"passed": True, "raw_responses": len(rows), "full_dispatch_responses": full_count,
               "unique_nonzero_responses": len(normalized_values), "paired_differences": len(analysis["paired_differences"]),
               "gpu_exact_dtype_audit_all": all(r["audit"].get("actual_dispatch_audit", r["kind"] == "bare") for r in rows),
               "cpu_normalization_max_error": cpu_normalization_max_error, "cpu_fp32_normalization_atol": 1.2e-7,
               "scope_and_rng_exact": True, "normalized_metrics_recomputed": True,
               "independent_audit_sha256": sha(Path(__file__)),
               "scope": "CPU recalculation from saved raw bias, probabilities, actual IDs/weights and actions; no recovery labels"}
    with (root / "independent-audit.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
