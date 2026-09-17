"""Reconstruct inputs, dispatch and coverage from raw arrays, without protocol imports."""

import argparse
from collections import Counter
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
    if actual is None or expected is None:
        if actual is not expected:
            raise RuntimeError("Missing value mismatch: " + label)
    elif not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Independent numeric mismatch: " + label)


def expected_inputs(parent, source, spec):
    bias = np.zeros((8, 10, 11, 32), np.float32)
    noise = source.copy()
    if spec is None:
        return bias, noise
    if spec["generator"] == "noise":
        namespace = {"candidate": 1, "target": 2}[spec["kind"]]
        seed = [2026091501, namespace, {"plus": 0, "pro": 1}[parent["benchmark"]],
                parent["task"], 39, 8, spec["pool"], spec["candidate"]]
        noise = np.random.default_rng(np.random.SeedSequence(seed)).standard_normal(source.shape).astype(np.float32)
    else:
        rng = np.random.default_rng(np.random.SeedSequence([2026091501, 3, spec["pool"], spec["candidate"]]))
        chosen = np.argsort(rng.random(bias.shape), axis=-1)[..., :16]
        direction = np.ones(bias.shape, np.float32)
        np.put_along_axis(direction, chosen, -1., axis=-1)
        tokens = slice(0, 1) if spec["generator"] == "state_gate" else slice(1, 11)
        amplitude = np.float32(.1 / np.sqrt(10) if spec["generator"] == "action_gate_l2" else .1)
        bias[:4, 7:, tokens] = direction[:4, 7:, tokens] * amplitude
    return bias, noise


def recompute(default, candidates, targets, std):
    vectors = [((action[:, :6].astype(float) - default[:, :6]) / std).ravel() for action in candidates]
    refs = [((action[:, :6].astype(float) - default[:, :6]) / std).ravel() for action in targets]
    radii = [float(np.linalg.norm(v) / np.sqrt(60)) for v in vectors]
    points = [np.zeros(60)] + vectors
    pairs = [np.linalg.norm(points[i] - points[j]) / np.sqrt(60) for i in range(5) for j in range(i + 1, 5)]
    gains, base_rms, nearest_rms, directions = [], [], [], []
    for target in refs:
        norm = np.linalg.norm(target)
        nearest = min(np.linalg.norm(target - candidate) for candidate in points)
        base_rms.append(float(norm / np.sqrt(60)))
        nearest_rms.append(float(nearest / np.sqrt(60)))
        gains.append(float(max(0., min(1., 1 - nearest / norm))) if norm > 1e-12 else None)
        if norm > 1e-12:
            angles = [np.dot(target, candidate) / (norm * np.linalg.norm(candidate))
                      for candidate in vectors if norm * np.linalg.norm(candidate) > 1e-24]
            directions.append(float(min(1., max([0.] + angles))))
    matrix = np.asarray(vectors)
    eigenvalues = np.maximum(np.linalg.eigvalsh(matrix @ matrix.T), 0)
    rank = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()) if eigenvalues.sum() > 1e-24 else 0.
    valid = [v for v in gains if v is not None]
    return dict(mean_radius_rms=float(np.mean(radii)), max_radius_rms=max(radii), mean_pairwise_rms=float(np.mean(pairs)),
                participation_rank=rank, coverage_gain=float(np.mean(valid)) if valid else None,
                best_positive_cosine=float(np.mean(directions)) if directions else None,
                target_coverage_gains=gains, target_default_rms=base_rms, target_nearest_rms=nearest_rms,
                valid_targets=len(valid), candidate_radii_rms=radii)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    root = parser.parse_args().run.resolve()
    torch.set_num_threads(1)
    config, collection, analysis = (load(root / name) for name in ("config.json", "collection.json", "analysis.json"))
    rows, references = collection["rows"], collection["references"]
    if len(rows) != config["isolated_forwards"] or len(references) != config["shared_forwards"]:
        raise RuntimeError("Call accounting mismatch")
    events = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    if len(events) != 2 * (len(rows) + len(references)):
        raise RuntimeError("Unacknowledged request or log count mismatch")
    indexed = {(r["engine"], r["ordinal"]): r for r in rows + references}
    for i in range(0, len(events), 2):
        request, response = events[i:i + 2]
        key = (request["engine"], request["ordinal"])
        if request["event"] != "request" or response != indexed[key] or response["event"] != "response":
            raise RuntimeError("Request/response log mismatch")
        for field, value in request.items():
            if field != "event" and response[field] != value:
                raise RuntimeError("Request identity changed")
    for engine, group in (("isolated", rows), ("shared", references)):
        exact([r["ordinal"] for r in group], np.arange(len(group)), engine + " ordinals")
    if [r["request_id"] for r in references] != config["reference_ids"]:
        raise RuntimeError("Shared IDs mismatch")
    parents = {p["parent"]: p for p in config["parents"]}
    std = np.asarray(config["shared_metadata"]["normalization_action_std"][:6], float)
    native, inputs, action_map, target_map, paths = {}, {}, {}, {}, {}
    for parent in config["parents"]:
        directory = root / "states" / parent["parent"]
        with np.load(directory / "input.npz", allow_pickle=False) as raw:
            inputs[parent["parent"]] = raw["flow/noise"].copy()
        with np.load(directory / "native.npz", allow_pickle=False) as raw:
            native[parent["parent"]] = raw["actions"].copy()
            for name in ("zero", "post", "bare", "shared-reference"):
                with np.load(directory / (name + ".npz"), allow_pickle=False) as control:
                    fields = raw.files if name in ("zero", "post") else ("actions", "routing/expert_ids", "routing/expert_weights")
                    for field in fields:
                        exact(raw[field], control[field], "baseline equivalence: " + field)
    max_weight_error = max_bias_probability_error = 0.
    full_count = 0
    for row in rows + references:
        path = root / row["path"]
        if sha(path) != row["sha256"]:
            raise RuntimeError("Raw checksum mismatch")
        if row["engine"] == "shared":
            continue
        parent = row["parent"]
        bias, noise = expected_inputs(parents[parent], inputs[parent], row["spec"])
        with np.load(path, allow_pickle=False) as raw:
            exact(raw["gate_probe/request_bias"], bias, "bias random stream and scope")
            exact(raw["gate_probe/request_noise"], noise, "noise random stream")
            for array, field in ((bias, "bias_sha256"), (noise, "noise_sha256")):
                if hashlib.sha256(array.tobytes()).hexdigest() != row[field]:
                    raise RuntimeError("Request digest mismatch")
            close(row["bias_l2"], np.linalg.norm(bias.astype(float)), "bias L2")
            if row["kind"] != "bare":
                full_count += 1
                if not row["audit"]["actual_dispatch_audit"]:
                    raise RuntimeError("Missing exact GPU audit")
                dtype = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16}[row["audit"]["arithmetic_dtype"]]
                for phase in ("native", "effective"):
                    probabilities = raw["v8_control/" + phase + "_probs_fp32"]
                    ids = raw["collection/hb_" + phase + "_ids"].astype(np.int64)
                    if probabilities.shape != (8, 10, 11, 32) or ids.shape != (8, 10, 11, 4):
                        raise RuntimeError("Route shape mismatch")
                    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.max(np.abs(probabilities.astype(float).sum(-1) - 1)) > .005:
                        raise RuntimeError("Invalid probability distribution")
                    if np.any(ids < 0) or np.any(ids >= 32) or np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
                        raise RuntimeError("Invalid expert IDs")
                    selected = torch.gather(torch.tensor(probabilities, dtype=dtype), -1, torch.tensor(ids))
                    expected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
                    error = float(np.max(np.abs(expected.float().numpy() - raw["collection/hb_" + phase + "_weights"])))
                    max_weight_error = max(max_weight_error, error)
                    if error > (1.2e-7 if dtype == torch.float32 else 0):
                        raise RuntimeError("Weight normalization mismatch beyond FP32 CPU/CUDA reduction rounding")
                    selected_values = np.take_along_axis(probabilities, ids, axis=-1)
                    cutoff = np.partition(probabilities, -4, axis=-1)[..., -4]
                    if np.any(selected_values.min(axis=-1) < cutoff):
                        raise RuntimeError("Actual experts are not top-4")
                # In FP32, exp(bias) reweights saved native softmax without needing logits.
                if dtype == torch.float32:
                    reweighted = raw["v8_control/native_probs_fp32"].astype(float) * np.exp(bias.astype(float))
                    reweighted /= reweighted.sum(-1, keepdims=True)
                    error = float(np.max(np.abs(reweighted - raw["v8_control/effective_probs_fp32"])))
                    max_bias_probability_error = max(max_bias_probability_error, error)
                    if error > 3e-7:
                        raise RuntimeError("Effective probabilities do not match the requested logit bias")
                outside = ~np.any(bias != 0, axis=-1)
                for field in ("ids", "weights"):
                    exact(raw["collection/hb_native_" + field][outside], raw["collection/hb_effective_" + field][outside], "off-scope dispatch")
                    wire = "expert_ids" if field == "ids" else "expert_weights"
                    exact(raw["collection/hb_effective_" + field][:, :, 1:].transpose(1, 0, 2, 3), raw["routing/" + wire], "wire dispatch")
            if row["kind"] == "candidate":
                spec = row["spec"]
                key = (parent, spec["pool"], spec["generator"], spec["candidate"])
                action_map[key] = raw["actions"].copy()
                paths[key] = path
            elif row["kind"] == "target":
                target_map[(parent, row["spec"]["candidate"])] = raw["actions"].copy()
            elif row["kind"] == "repeat":
                spec = row["spec"]
                key = (parent, spec["pool"], spec["generator"], spec["candidate"])
                with np.load(paths[key], allow_pickle=False) as original:
                    if set(raw.files) != set(original.files):
                        raise RuntimeError("Repeated array field mismatch")
                    for field in raw.files:
                        exact(raw[field], original[field], "repeated arrays: " + field)
    with np.load(root / references[0]["path"], allow_pickle=False) as first, np.load(root / references[-1]["path"], allow_pickle=False) as post:
        for field in ("actions", "routing/expert_ids", "routing/expert_weights"):
            exact(first[field], post[field], "shared final baseline")
    expected_counts = dict(bare=len(parents), native=len(parents), zero=len(parents), post=len(parents),
                           candidate=32 * len(parents), target=8 * len(parents), repeat=4 * len(parents))
    if dict(Counter(r["kind"] for r in rows)) != expected_counts:
        raise RuntimeError("Candidate/target/control counts mismatch")
    for parent in parents:
        for pool in (0, 1):
            for i in range(4):
                if (parent, pool, "noise", i) not in action_map:
                    raise RuntimeError("Missing IID comparison")
        hashes = [hashlib.sha256(expected_inputs(parents[parent], inputs[parent], r["spec"])[1].tobytes()).hexdigest()
                  for r in rows if r["parent"] == parent and r["kind"] in ("candidate", "target") and r["spec"]["generator"] == "noise"]
        if len(set(hashes)) != 16 or hashlib.sha256(inputs[parent].tobytes()).hexdigest() in hashes:
            raise RuntimeError("Candidate and reference noise collision")
    recomputed = {}
    for row in analysis["pools"]:
        parent, pool, method = row["parent"], row["pool"], row["generator"]
        keys = [(parent, pool, g, i) for g in ("noise", "state_gate") for i in (0, 1)] if method == "mixed_noise_state" else [(parent, pool, method, i) for i in range(4)]
        metrics = recompute(native[parent], [action_map[k] for k in keys], [target_map[(parent, i)] for i in range(8)], std)
        for field, value in metrics.items():
            if field == "target_coverage_gains":
                for actual, expected in zip(row[field], value):
                    close(actual, expected, field)
            else:
                close(row[field], value, field)
        recomputed[(parent, pool, method)] = metrics
    fields = ("mean_radius_rms", "max_radius_rms", "mean_pairwise_rms", "participation_rank", "coverage_gain", "best_positive_cosine")
    if len(recomputed) != 10 * len(parents):
        raise RuntimeError("Missing or duplicated candidate pool")
    for row in analysis["per_parent"]:
        for field in fields:
            expected = np.mean([recomputed[(row["parent"], pool, row["generator"])][field] for pool in (0, 1)])
            close(row[field], expected, "per-parent mean " + field)
    for row in analysis["table"]:
        for field in fields:
            expected = np.mean([np.mean([recomputed[(p, pool, row["generator"])][field] for pool in (0, 1)]) for p in parents])
            close(row[field], expected, "parent-balanced table " + field)
        differences = [np.mean([recomputed[(p, pool, row["generator"])]["coverage_gain"] - recomputed[(p, pool, "noise")]["coverage_gain"] for pool in (0, 1)]) for p in parents]
        close(row["coverage_difference_vs_noise"], np.mean(differences), "coverage difference")
        close(row["parents_better_than_noise"], sum(d > 1e-12 for d in differences), "parent win count")
    for row in analysis["candidate_metrics"]:
        key = (row["parent"], row["pool"], row["generator"], row["candidate"])
        delta = (action_map[key][:, :6].astype(float) - native[row["parent"]][:, :6]) / std
        close(row["normalized_action_rms"], np.linalg.norm(delta) / np.sqrt(60), "candidate action RMS")
        close(row["gripper_sign_changes"], np.count_nonzero(np.sign(action_map[key][:, 6]) != np.sign(native[row["parent"]][:, 6])), "gripper signs")
    result = dict(passed=True, raw_responses=len(rows), shared_responses=len(references), full_dispatch_responses=full_count,
                  candidate_pools=len(recomputed), unique_candidates=len(action_map), reference_actions=len(target_map),
                  repeated_candidates=expected_counts["repeat"], scope_and_rng_exact=True, noise_namespaces_disjoint=True,
                  cpu_normalization_max_error=max_weight_error, cpu_fp32_normalization_atol=1.2e-7,
                  cpu_bias_probability_max_error=max_bias_probability_error, cpu_fp32_bias_probability_atol=3e-7,
                  all_pool_metrics_independently_recomputed=True, independent_audit_sha256=sha(Path(__file__)),
                  physical_recovery_evaluated=False)
    with (root / "independent-audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
