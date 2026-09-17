"""Frozen paired gate-response experiment using a bounded isolated model."""

import argparse
from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

import numpy as np
import zarr

from gate_runtime import BASE, UPSTREAM, infer_isolated, load_isolated
from gate_protocol import (TASKS, QUERY, AMPLITUDES, DIRECTIONS, SCOPES, SHAPE,
                           make_gate_bias, normalized_rms, probes)
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file
from collection_routes import (CAPTURE_KEY, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY,
                               EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY)
from scope_bias_control import BIAS_KEY
from v8_feature_control import NATIVE_PROBS, EFFECTIVE_PROBS

BATCHES = {
    "plus": Path("/data/libero-runtime/samples/plus/batch-libero-long-20260914T115910731385Z"),
    "pro": Path("/data/libero-runtime/samples/pro/batch-libero-long-20260914T115911968889Z"),
}
PREFLIGHT = BASE / "runs/gate-preflight-glafkbjc/result.json"
REFERENCE_ID = -1260930000
ROUTES = Path("/data/libero-runtime/model-20260914T035917Z/routes.zarr")
WIRE_KEYS = ("actions", "routing/expert_ids", "routing/expert_weights")
FULL_KEYS = WIRE_KEYS + (NATIVE_PROBS, EFFECTIVE_PROBS, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY,
                          EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY, "collection/as_probs")


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def frozen_files():
    files = [BASE / name for name in ("P1B_GATE_PLAN.zh.md", "gate_runtime.py", "gate_capture.py",
             "gate_protocol.py", "run_gate_experiment.py", "test_gate_probe.py")]
    files += [UPSTREAM / name for name in ("scope_bias_control.py", "v8_feature_control.py", "collection_routes.py")]
    files += [Path("/data/srv/src/himoe_libero_bridge") / name for name in ("policies.py", "protocol.py", "episode_trace.py")]
    files += [Path("/data/srv/src/moevla") / name for name in ("models/modeling_moe.py", "policies/policy.py", "policies/policy_config.py")]
    return {str(path): sha256_file(path) for path in files}


def check_config(root):
    config = read_json(root / "config.json")
    if sha256_file(root / "config.json") != (root / "config.sha256").read_text().strip():
        raise RuntimeError("Frozen configuration changed")
    if config["source_hashes"] != frozen_files():
        raise RuntimeError("Frozen implementation changed")
    return config


def check_identity(actual, identity):
    for key, value in identity.items():
        if actual.get(key) != value:
            raise RuntimeError("Policy identity mismatch: " + key)


def source_for(parent):
    manifest, arrays = load_episode_trace(Path(parent["source"]))
    if manifest["array_file_sha256"] != parent["trace_sha256"]:
        raise RuntimeError("Source trace changed")
    if sha256_file(Path(parent["source"]) / "episode-trace.json") != parent["manifest_sha256"]:
        raise RuntimeError("Source trace manifest changed")
    q = parent["query"]
    request = {"observation/image": arrays["images"][q], "observation/wrist_image": arrays["wrist_images"][q],
               "observation/state": arrays["states"][q], "prompt": manifest["prompt"],
               "flow/noise": arrays["flow_noises"][q], "routing/capture": True}
    return manifest, arrays, request


def check_reference_ids(count):
    g = zarr.open_group(str(ROUTES), mode="r")
    durable = min(int(g.attrs["durable_rows"]), g["episode_id"].shape[0])
    existing = set(np.asarray(g["episode_id"][:durable]).tolist())
    wanted = {REFERENCE_ID - i for i in range(count)}
    if existing.intersection(wanted):
        raise RuntimeError("Shared reference request IDs already used")


def prepare():
    preflight = read_json(PREFLIGHT)
    if not preflight["passed"]:
        raise RuntimeError("Gate preflight has not passed")
    with PolicyClient("127.0.0.1", 9500, inference_timeout=30) as client:
        metadata = client.metadata
    if metadata.get("server_instance_id") != preflight["shared_metadata"].get("server_instance_id"):
        raise RuntimeError("Shared server changed after preflight")
    parents, missing, names = [], [], {}
    for benchmark, batch in BATCHES.items():
        manifest = read_json(batch / "manifest.json")
        scenario = read_json(batch / "scenario-plan.json")
        description = {(r["base_task_id"], r["episode_index"]): r for r in scenario["jobs"]}
        for task in TASKS:
            job, = [j for j in manifest["jobs"] if j["task_id"] == task and j["episode_index"] == 1]
            if job["init_state_id"] != 47:
                raise RuntimeError("Unexpected initial state")
            if job["state"] != "completed":
                missing.append({"benchmark": benchmark, "task": task, "reason": "incomplete"})
                continue
            source = Path(job["result"]["artifact_dir"])
            trace, arrays = load_episode_trace(source)
            if not trace["result"]["trace_complete"] or trace["result"]["status"] != "completed":
                raise RuntimeError("Incomplete source trace")
            check_identity(metadata, trace["policy_identity"])
            if len(arrays["images"]) <= QUERY:
                missing.append({"benchmark": benchmark, "task": task, "reason": "query_absent"})
                continue
            item = description[(task, 1)]
            if names.setdefault(task, item["base_task_name"]) != item["base_task_name"]:
                raise RuntimeError("Base task mapping mismatch")
            if item["task_name"] != job["result"]["task_name"]:
                raise RuntimeError("Source task mismatch")
            parents.append({"parent": "%s-task%02d-init047-q08" % (benchmark, task),
                            "benchmark": benchmark, "task": task, "query": QUERY,
                            "source": str(source), "trace_sha256": trace["array_file_sha256"],
                            "manifest_sha256": sha256_file(source / "episode-trace.json"),
                            "policy_identity": trace["policy_identity"], "base_task_name": item["base_task_name"]})
    if len(parents) < 4:
        raise RuntimeError("Insufficient source states")
    check_reference_ids(len(parents) + 1)
    config = {
        "schema": "moe-control-gate-probe-v1", "prepared_utc": now(), "parents": parents,
        "missing": missing, "scopes": list(SCOPES), "amplitudes": list(AMPLITUDES),
        "directions": list(DIRECTIONS), "probes": probes(), "shared_metadata": metadata,
        "source_hashes": frozen_files(), "preflight": str(PREFLIGHT), "preflight_sha256": sha256_file(PREFLIGHT),
        "isolated_forwards": 70 * len(parents), "shared_forwards": len(parents) + 1,
        "nonzero_unique_forwards": 64 * len(parents), "nonzero_repeat_forwards": 2 * len(parents),
        "new_environment_actions": 0, "success_labels_used": False, "model_training": False,
        "reference_id_base": REFERENCE_ID, "reference_ids": [REFERENCE_ID - i for i in range(len(parents) + 1)],
        "interpretation": "Developmental local gate-to-action causal response, not physical recovery",
    }
    root = Path(tempfile.mkdtemp(prefix="p1b-gate-20260914-", dir=BASE / "runs"))
    save_json(root / "config.json", config)
    (root / "config.sha256").write_text(sha256_file(root / "config.json") + "\n")
    print(json.dumps({"run": str(root), "parents": len(parents), "isolated_forwards": config["isolated_forwards"],
                      "shared_forwards": config["shared_forwards"], "missing": missing}), flush=True)


def assert_equal(actual, expected, keys, label):
    for key in keys:
        if not np.array_equal(actual[key], expected[key]):
            raise RuntimeError(label + ": " + key)


def audit_response(response, bias):
    native, effective = response[NATIVE_PROBS], response[EFFECTIVE_PROBS]
    if native.shape != SHAPE or effective.shape != SHAPE:
        raise RuntimeError("Unexpected full HB shape")
    if not response["gate_probe/exact_dtype_audit"]:
        raise RuntimeError("Missing exact arithmetic audit")
    max_mass_error = max(float(np.max(np.abs(p.astype(float).sum(-1) - 1))) for p in (native, effective))
    if max_mass_error > .005 or any(not np.isfinite(p).all() or np.any(p < 0) for p in (native, effective)):
        raise RuntimeError("Invalid probability mass")
    outside = ~np.any(bias != 0, axis=-1)
    for native_key, effective_key in ((NATIVE_IDS_KEY, EFFECTIVE_IDS_KEY), (NATIVE_WEIGHTS_KEY, EFFECTIVE_WEIGHTS_KEY)):
        if not np.array_equal(response[native_key][outside], response[effective_key][outside]):
            raise RuntimeError("Changed immediate dispatch outside requested scope")
    return {"max_probability_mass_error": max_mass_error,
            "arithmetic_dtype": response["gate_probe/arithmetic_dtype"],
            "outside_scope_dispatch_exact": True, "actual_dispatch_audit": True}


def collect(root):
    config = check_config(root)
    if (root / "started.json").exists():
        raise RuntimeError("Run already started; no silent resume")
    check_reference_ids(config["shared_forwards"])
    save_json(root / "started.json", {"utc": now()})
    rows, controls, references = [], [], []
    started = time.monotonic()
    attempts = 0
    try:
        wrapped, loaded = load_isolated()
        save_json(root / "model-load.json", loaded)
        with (root / "calls.jsonl").open("x") as log, PolicyClient("127.0.0.1", 9500, inference_timeout=180) as client:
            if client.metadata.get("server_instance_id") != config["shared_metadata"].get("server_instance_id"):
                raise RuntimeError("Shared service instance changed")

            def infer(parent, request, name, kind, spec=None, full=True):
                nonlocal attempts
                ordinal = len(rows)
                bias = np.zeros(SHAPE, np.float32) if spec is None else make_gate_bias(**spec)
                extra = {CAPTURE_KEY: True} if full else {}
                if spec is not None or kind == "zero":
                    extra[BIAS_KEY] = bias
                record = {"ordinal": ordinal, "parent": parent["parent"], "task": parent["task"],
                          "benchmark": parent["benchmark"], "kind": kind, "name": name,
                          "spec": spec, "noise_sha256": digest_array(request["flow/noise"]),
                          "bias_sha256": digest_array(bias), "bias_l2": float(np.linalg.norm(bias.astype(float))),
                          "event": "request"}
                log.write(json.dumps(record) + "\n")
                log.flush()
                attempts += 1
                response, resource = infer_isolated(wrapped, dict(request, **extra))
                check = audit_response(response, bias) if full else {}
                path = root / "states" / parent["parent"] / (name + ".npz")
                np.savez_compressed(path, **{k: v for k, v in response.items() if isinstance(v, np.ndarray)},
                                    **{"gate_probe/request_bias": bias, "gate_probe/request_noise": request["flow/noise"]})
                record.update(event="response", path=str(path.relative_to(root)), sha256=sha256_file(path),
                              resource=resource, audit=check)
                rows.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                return response

            first_request = first_live = None
            for parent_index, parent in enumerate(config["parents"]):
                manifest, arrays, request = source_for(parent)
                check_identity(client.metadata, manifest["policy_identity"])
                check_identity(loaded["metadata"], manifest["policy_identity"])
                directory = root / "states" / parent["parent"]
                directory.mkdir(parents=True)
                np.savez_compressed(directory / "input.npz", **{k: v for k, v in request.items() if isinstance(v, np.ndarray)})
                save_json(directory / "input.json", {"prompt": request["prompt"], "source": parent["source"],
                                                     "array_sha256": sha256_file(directory / "input.npz")})
                live = client.infer(dict(request, episode_id=config["reference_ids"][parent_index]))
                if live["flow/noise_sha256"] != digest_array(request["flow/noise"]):
                    raise RuntimeError("Shared noise acknowledgement mismatch")
                references.append({"parent": parent["parent"], "request_id": config["reference_ids"][parent_index]})
                np.savez_compressed(directory / "shared-reference.npz", **{k: v for k, v in live.items() if isinstance(v, np.ndarray)})
                if not np.array_equal(live["actions"], arrays["predicted_actions"][parent["query"]]):
                    raise RuntimeError("Shared service/source actions changed")
                if first_request is None:
                    first_request, first_live = request, live
                bare = infer(parent, request, "bare", "bare", full=False)
                assert_equal(bare, live, WIRE_KEYS, "Isolated/shared baseline mismatch")
                baseline = infer(parent, request, "native", "native")
                assert_equal(baseline, live, WIRE_KEYS, "Full capture/shared mismatch")
                zero = infer(parent, request, "zero", "zero")
                assert_equal(zero, baseline, FULL_KEYS, "Zero-bias changed output")
                for probe_index, spec in enumerate(config["probes"]):
                    infer(parent, request, "probe-%02d" % probe_index, "probe", spec)
                    if (probe_index + 1) % 16 == 0:
                        print(json.dumps({"parent": parent["parent"], "probes_done": probe_index + 1,
                                          "isolated_calls": len(rows), "elapsed_seconds": time.monotonic() - started}), flush=True)
                for sign in (-1, 1):
                    spec = dict(scope=SCOPES[0], direction=0, amplitude=.10, sign=sign)
                    original_index = config["probes"].index(spec)
                    repeated = infer(parent, request, "repeat-%s" % ("minus" if sign < 0 else "plus"), "repeat", spec)
                    with np.load(directory / ("probe-%02d.npz" % original_index), allow_pickle=False) as original:
                        assert_equal(repeated, original, FULL_KEYS, "Nonzero repeated probe changed output")
                after = infer(parent, request, "post", "post")
                assert_equal(after, baseline, FULL_KEYS, "Residual gate intervention after cleanup")
                controls.append({"parent": parent["parent"], "shared_bare_full_exact": True,
                                 "zero_exact": True, "nonzero_repeats_exact": True, "post_exact": True})
                print(json.dumps({"parent": parent["parent"], "finished": True, "isolated_calls": len(rows)}), flush=True)
            final = client.infer(dict(first_request, episode_id=config["reference_ids"][-1]))
            assert_equal(final, first_live, WIRE_KEYS, "Shared service changed during experiment")
            references.append({"parent": config["parents"][0]["parent"], "kind": "post_check",
                               "request_id": config["reference_ids"][-1]})
            np.savez_compressed(root / "shared-post.npz", **{k: v for k, v in final.items() if isinstance(v, np.ndarray)})
        if len(rows) != config["isolated_forwards"] or len(references) != config["shared_forwards"]:
            raise RuntimeError("Forward accounting mismatch")
        save_json(root / "collection.json", {"passed": True, "finished_utc": now(), "rows": rows,
                                             "controls": controls, "references": references,
                                             "isolated_attempts": attempts, "isolated_completed": len(rows),
                                             "shared_completed": len(references), "new_environment_actions": 0,
                                             "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"completed": True, "isolated": len(rows), "shared": len(references)}), flush=True)
    except BaseException as error:
        save_json(root / "failure.json", {"failed_utc": now(), "error": str(error), "type": type(error).__name__,
                                          "isolated_attempts": attempts, "isolated_completed": len(rows),
                                          "shared_completed": len(references), "traceback": traceback.format_exc()})
        raise


def set_changes(left, right):
    return np.any(np.sort(left, axis=-1) != np.sort(right, axis=-1), axis=-1)


def cosine(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-15 else None


def analyze(root):
    config = check_config(root)
    collection = read_json(root / "collection.json")
    std = np.asarray(config["shared_metadata"]["normalization_action_std"])
    results, paired, state_summaries = [], [], []
    for parent in config["parents"]:
        directory = root / "states" / parent["parent"]
        with np.load(directory / "native.npz", allow_pickle=False) as native_file:
            native = {key: native_file[key] for key in ("actions", EFFECTIVE_IDS_KEY, EFFECTIVE_PROBS)}
        probes_by_key = {}
        for row in collection["rows"]:
            if row["parent"] != parent["parent"] or row["kind"] != "probe":
                continue
            path = root / row["path"]
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError("Raw probe checksum changed")
            spec = row["spec"]
            expected_bias = make_gate_bias(**spec)
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(expected_bias, saved["gate_probe/request_bias"]):
                    raise RuntimeError("Stored bias differs from frozen scope/random direction")
                actions = saved["actions"].copy()
                mask = np.any(expected_bias != 0, axis=-1)
                direct = set_changes(saved[NATIVE_IDS_KEY], saved[EFFECTIVE_IDS_KEY])
                relative = set_changes(native[EFFECTIVE_IDS_KEY], saved[EFFECTIVE_IDS_KEY])
                delta = actions.astype(float) - native["actions"]
                result = {"parent": parent["parent"], "task": parent["task"], "benchmark": parent["benchmark"], **spec,
                          "normalized_action_rms": normalized_rms(actions, native["actions"], std),
                          "xyz_rms": float(np.sqrt(np.mean(delta[:, :3] ** 2))),
                          "rotation_rms": float(np.sqrt(np.mean(delta[:, 3:6] ** 2))),
                          "gripper_sign_changes": int((np.sign(actions[:, 6]) != np.sign(native["actions"][:, 6])).sum()),
                          "direct_scope_set_change": float(direct[mask].mean()),
                          "off_scope_set_change_vs_baseline": float(relative[~mask].mean()),
                          "all_sites_set_change_vs_baseline": float(relative.mean()),
                          "bias_l2": float(np.linalg.norm(expected_bias.astype(float))),
                          "actions_changed": not np.array_equal(actions, native["actions"])}
                result["normalized_action_l2_per_bias_l2"] = result["normalized_action_rms"] * np.sqrt(60) / result["bias_l2"]
                results.append(result)
                probes_by_key[(spec["scope"], spec["direction"], spec["amplitude"], spec["sign"])] = actions
        for scope in SCOPES:
            for direction in DIRECTIONS:
                differences = {}
                for amplitude in AMPLITUDES:
                    plus = probes_by_key[(scope, direction, amplitude, 1)].astype(float)
                    minus = probes_by_key[(scope, direction, amplitude, -1)].astype(float)
                    difference = (plus[:, :6] - minus[:, :6]) / std[:6]
                    differences[amplitude] = difference
                    bias = make_gate_bias(scope, direction, amplitude, 1).astype(float)
                    midpoint = ((plus[:, :6] + minus[:, :6]) / 2 - native["actions"][:, :6]) / std[:6]
                    paired.append({"parent": parent["parent"], "task": parent["task"], "scope": scope,
                                   "direction": direction, "amplitude": amplitude,
                                   "symmetric_rms_per_amplitude": float(np.sqrt(np.mean(difference ** 2)) / (2 * amplitude)),
                                   "symmetric_l2_per_bias_l2": float(np.linalg.norm(difference) / (2 * np.linalg.norm(bias))),
                                   "midpoint_offset_rms": float(np.sqrt(np.mean(midpoint ** 2)))})
                low, high = differences[.05], differences[.10]
                state_summaries.append({"parent": parent["parent"], "scope": scope, "direction": direction,
                                        "amplitude_direction_cosine": cosine(low, high),
                                        "amplitude_response_ratio": float(np.linalg.norm(high) / np.linalg.norm(low))
                                        if np.linalg.norm(low) > 1e-15 else None})
    table = []
    for amplitude in AMPLITUDES:
        for scope in SCOPES:
            subset = [r for r in results if r["scope"] == scope and r["amplitude"] == amplitude]
            per_parent = []
            for parent in config["parents"]:
                rows = [r for r in subset if r["parent"] == parent["parent"]]
                per_parent.append({"parent": parent["parent"], "mean_action_rms": float(np.mean([r["normalized_action_rms"] for r in rows]))})
            table.append({"scope": scope, "amplitude": amplitude, "parents": len(per_parent),
                          "mean_normalized_action_rms": float(np.mean([r["mean_action_rms"] for r in per_parent])),
                          "median_parent_action_rms": float(np.median([r["mean_action_rms"] for r in per_parent])),
                          "mean_action_l2_per_bias_l2": float(np.mean([r["normalized_action_l2_per_bias_l2"] for r in subset])),
                          "mean_xyz_rms": float(np.mean([r["xyz_rms"] for r in subset])),
                          "mean_rotation_rms": float(np.mean([r["rotation_rms"] for r in subset])),
                          "direct_scope_set_change": float(np.mean([r["direct_scope_set_change"] for r in subset])),
                          "gripper_sign_changes": sum(r["gripper_sign_changes"] for r in subset),
                          "changed_probe_count": sum(r["actions_changed"] for r in subset),
                          "per_parent": per_parent})
    resource = {key: max(row["resource"][key] for row in collection["rows"])
                for key in ("peak_allocated_mib", "peak_reserved_mib")}
    report = {"schema": config["schema"], "finished_utc": now(), "parents": len(config["parents"]),
              "table": table, "probe_metrics": results, "paired_differences": paired,
              "amplitude_comparisons": state_summaries, "resources": resource,
              "isolated_forwards": collection["isolated_completed"], "shared_forwards": collection["shared_completed"],
              "nonzero_unique_forwards": len(results), "all_controls_passed": True,
              "physical_recovery_evaluated": False, "new_environment_actions": 0}
    save_json(root / "analysis.json", report)
    labels = {"front": "前层", "back": "后层", "state": "状态 token", "action": "动作 token",
              "early": "早去噪", "late": "晚去噪"}
    lines = ["# P1b：gate 可控性实验结果", "", "本轮实际修改了 gate 返回给专家 dispatch 的 ID/权重，但没有执行环境动作。", "",
             "- %d 条独立父轨迹，4 个基础任务，Plus/Pro，init47，固定 q8。" % len(config["parents"]),
             "- 正式实验 %d 次独立模型前向、%d 次原服务对照；不同非零探针 %d 次。" %
             (report["isolated_forwards"], report["shared_forwards"], len(results)),
             "- 裸推理、全捕获、零偏置和清理后输出一致；每状态两个非零探针重复完全一致。",
             "- 模型、观测和初始噪声固定；无训练、无物体真值、无任务成功标签筛选。", "",
             "## 主要响应，per-site RMS 0.10", "",
             "动作差异除以 checkpoint 的 6D action std 后计算 RMS。每父状态先平均两个方向和正负，再等权汇总。", "",
             "| 干预位置 | 标准化动作 RMS | 动作 L2 / 偏置 L2 | 直接专家集合变化 | 夹爪符号变化步数 |",
             "|---|---:|---:|---:|---:|"]
    primary = sorted([r for r in table if r["amplitude"] == .10], key=lambda r: -r["mean_normalized_action_rms"])
    for row in primary:
        label = " / ".join(labels[part] for part in row["scope"].split("_"))
        lines.append("| %s | %.6f | %.6f | %.1f%% | %d |" %
                     (label, row["mean_normalized_action_rms"], row["mean_action_l2_per_bias_l2"],
                      100 * row["direct_scope_set_change"], row["gripper_sign_changes"]))
    lines += ["", "## 解读边界", "",
              "这张表只定位在所测状态和幅度下有作用的位置，不识别正确恢复方向。专家集合变化很大而动作变化很小也是有效阴性发现。",
              "动作 token scope 的路由位置数量是状态 token scope 的 10 倍，因此同时给出按总偏置能量归一化的响应。",
              "正负和两档幅度只构成有限差分；top-k 切换可能非线性，不能将结果无条件解释为光滑 Jacobian。",
              "512 个不同非零探针不是 512 条独立轨迹；基础任务只有 4 个，不宣称广泛泛化或统计显著性。",
              "响应更大不代表安全，也不代表恢复更好。本轮结束，不自动启动物理恢复或训练选择器。", "",
              "## 资源与审计", "",
              "PyTorch 分配峰值 %.1f MiB，缓存峰值 %.1f MiB，上限 16384 MiB。原 9500 未修改或重启。" %
              (resource["peak_allocated_mib"], resource["peak_reserved_mib"]),
              "成功预检另有 4 次独立前向和 1 次原服务前向；第一次预检因元数据字段检查提前退出，未做模型调用。",
              "详细两档幅度、逐父状态、正负有限差分和非线性指标见 `analysis.json`；原始数据见 `states/`；协议见 `config.json`。", ""]
    (root / "REPORT.zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"parents": report["parents"], "top_scope_at_010": primary[0], "resources": resource}), flush=True)


def verify(root):
    config = check_config(root)
    collection = read_json(root / "collection.json")
    result = read_json(root / "analysis.json")
    if len(collection["rows"]) != config["isolated_forwards"]:
        raise RuntimeError("Missing isolated calls")
    probe_rows = [r for r in collection["rows"] if r["kind"] == "probe"]
    if len(probe_rows) != config["nonzero_unique_forwards"]:
        raise RuntimeError("Missing probe calls")
    for parent in config["parents"]:
        _, arrays, request = source_for(parent)
        directory = root / "states" / parent["parent"]
        with np.load(directory / "native.npz", allow_pickle=False) as baseline:
            for filename, keys in (("bare", WIRE_KEYS), ("shared-reference", WIRE_KEYS), ("zero", FULL_KEYS), ("post", FULL_KEYS)):
                with np.load(directory / (filename + ".npz"), allow_pickle=False) as value:
                    assert_equal(value, baseline, keys, "Persisted baseline equivalence failed")
            if not np.array_equal(baseline["actions"], arrays["predicted_actions"][parent["query"]]):
                raise RuntimeError("Persisted source action mismatch")
        with np.load(directory / "input.npz", allow_pickle=False) as saved:
            for key, value in request.items():
                if isinstance(value, np.ndarray) and not np.array_equal(saved[key], value):
                    raise RuntimeError("Input trace mismatch")
    for row in collection["rows"]:
        path = root / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError("Raw response checksum mismatch")
        with np.load(path, allow_pickle=False) as saved:
            bias = np.zeros(SHAPE, np.float32) if row["spec"] is None else make_gate_bias(**row["spec"])
            if not np.array_equal(saved["gate_probe/request_bias"], bias):
                raise RuntimeError("Raw bias mismatch")
            if digest_array(saved["gate_probe/request_noise"]) != row["noise_sha256"]:
                raise RuntimeError("Raw noise mismatch")
            if row["kind"] != "bare":
                for full, wire in ((EFFECTIVE_IDS_KEY, "routing/expert_ids"), (EFFECTIVE_WEIGHTS_KEY, "routing/expert_weights")):
                    if not np.array_equal(saved[full][:, :, 1:].transpose(1, 0, 2, 3), saved[wire]):
                        raise RuntimeError("Persisted actual dispatch mismatch")
    with PolicyClient("127.0.0.1", 9500, inference_timeout=30) as client:
        if client.metadata.get("server_instance_id") != config["shared_metadata"].get("server_instance_id"):
            raise RuntimeError("Shared server instance changed")
    tests = subprocess.run(["/data/venv311/bin/python", "-m", "unittest", "discover", "-s", str(BASE), "-p", "test_*.py", "-v"],
                           text=True, capture_output=True, check=True)
    (root / "tests.txt").write_text(tests.stdout + tests.stderr)
    files = {str(path.relative_to(root)): sha256_file(path) for path in root.rglob("*") if path.is_file()}
    save_json(root / "verification.json", {"passed": True, "verified_utc": now(), "isolated_forwards": len(collection["rows"]),
                                          "unique_nonzero_probes": len(probe_rows), "parents": len(config["parents"]),
                                          "source_hashes_unchanged": True, "shared_instance_unchanged": True,
                                          "files": files, "scope": "Raw identity, dispatch and control audit; no physical outcome labels"})
    print(json.dumps({"verified": True, "raw_responses": len(collection["rows"]), "files": len(files)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.run is None:
        parser.error("--run is required")
    else:
        {"collect": collect, "analyze": analyze, "verify": verify}[args.command](args.run.resolve())


if __name__ == "__main__":
    main()
