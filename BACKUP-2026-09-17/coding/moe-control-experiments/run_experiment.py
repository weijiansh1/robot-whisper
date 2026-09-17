"""Prepare, collect and audit a bounded fixed-observation MoE response study."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import zarr

from protocol import (HOLDOUT_TASKS, PAIRS, QUERIES, RIDGE, SEED, candidate_noise,
                      distances, fit_distance, medoid, metrics, pair_rms,
                      permute_state_blocks, predict_distance, split_for)

sys.path.insert(0, "/data/srv/src")
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file

BASE = Path(__file__).resolve().parent
REPO = Path("/data/coding/robot-whisper-0909")
ROUTES = Path("/data/libero-runtime/model-20260914T035917Z/routes.zarr")
CHECKPOINT = "cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256"
BATCHES = {
    "plus": Path("/data/libero-runtime/samples/plus/batch-libero-long-20260914T115910731385Z"),
    "pro": Path("/data/libero-runtime/samples/pro/batch-libero-long-20260914T115911968889Z"),
}
LAYER_IDS = [2, 3, 4, 5, 12, 13, 14, 15]
ID_BASE = -1260915000


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def script_hashes():
    paths = [BASE / name for name in ("PLAN.zh.md", "protocol.py", "run_experiment.py", "test_protocol.py")]
    paths += [Path("/data/srv/src/himoe_libero_bridge") / name
              for name in ("client.py", "episode_trace.py", "protocol.py")]
    return {str(path): sha256_file(path) for path in paths}


def frozen_run(root):
    config = load_json(root / "config.json")
    expected = (root / "config.sha256").read_text().strip()
    if sha256_file(root / "config.json") != expected:
        raise RuntimeError("Frozen configuration changed")
    if script_hashes() != config["source_hashes"]:
        raise RuntimeError("Protocol or implementation changed after preparation")
    return config


def durable_count(g):
    keys = ("episode_id", "control_step", "hb_router_probs", "hb_expert_ids", "hb_selected_prob")
    return min(int(g.attrs["durable_rows"]), *(g[key].shape[0] for key in keys))


def check_metadata(metadata, expected=None):
    if metadata.get("checkpoint_sha256") != CHECKPOINT:
        raise RuntimeError("Unexpected checkpoint")
    if metadata.get("episode_id_key") != "episode_id" or not metadata.get("routing_capture_supported"):
        raise RuntimeError("Missing required identity or capture protocol")
    if metadata.get("routing_hb_layer_indices") != LAYER_IDS:
        raise RuntimeError("Unexpected layer order")
    if expected is not None:
        for key in ("server_instance_id", "checkpoint_sha256", "normalization_stats_sha256",
                    "himoe_patch_sha256", "himoe_working_tree_diff_sha256"):
            if metadata.get(key) != expected.get(key):
                raise RuntimeError("Server identity changed: " + key)


def checked_trace(parent, metadata):
    manifest, arrays = load_episode_trace(Path(parent["source"]))
    if manifest["array_file_sha256"] != parent["trace_sha256"]:
        raise RuntimeError("Source trace changed")
    if sha256_file(Path(parent["source"]) / "episode-trace.json") != parent["manifest_sha256"]:
        raise RuntimeError("Source trace manifest changed")
    if not manifest["result"]["trace_complete"] or manifest["result"]["status"] != "completed":
        raise RuntimeError("Incomplete source")
    for key, value in manifest["policy_identity"].items():
        if metadata.get(key) != value:
            raise RuntimeError("Source/server policy mismatch: " + key)
    return manifest, arrays


def prepare():
    with PolicyClient("127.0.0.1", 9500, inference_timeout=180) as client:
        metadata = client.metadata
        check_metadata(metadata)
    parents, missing, inventories = [], [], {}
    names_by_task = {}
    for benchmark, batch in BATCHES.items():
        manifest = load_json(batch / "manifest.json")
        scenario = load_json(batch / "scenario-plan.json")
        scenarios = {(r["base_task_id"], r["episode_index"]): r for r in scenario["jobs"]}
        inventories[benchmark] = {
            "manifest_sha256": sha256_file(batch / "manifest.json"),
            "scenario_sha256": sha256_file(batch / "scenario-plan.json"),
            "completed": sum(j["state"] == "completed" for j in manifest["jobs"]),
            "total_jobs": len(manifest["jobs"]),
        }
        for task in range(10):
            matches = [j for j in manifest["jobs"] if j["task_id"] == task and j["episode_index"] == 0]
            if len(matches) != 1:
                raise RuntimeError("Non-unique planned source")
            job = matches[0]
            if job["init_state_id"] != 26:
                raise RuntimeError("Source initial-state rule changed")
            if job["state"] != "completed":
                missing.append({"benchmark": benchmark, "task": task, "reason": "source_incomplete"})
                continue
            source = Path(job["result"]["artifact_dir"])
            trace, arrays = load_episode_trace(source)
            description = scenarios[(task, 0)]
            task_name = description["base_task_name"]
            if names_by_task.setdefault(task, task_name) != task_name:
                raise RuntimeError("Base task split mismatch across benchmarks")
            if job["result"]["task_name"] != description["task_name"]:
                raise RuntimeError("Source/scenario task mismatch")
            parent = {"parent": "%s-task%02d-init026" % (benchmark, task),
                      "benchmark": benchmark, "task": task, "base_task_name": task_name,
                      "source": str(source), "trace_sha256": trace["array_file_sha256"],
                      "manifest_sha256": sha256_file(source / "episode-trace.json"),
                      "queries": [q for q in QUERIES if q < len(arrays["images"])],
                      "split": split_for(task), "source_queries": len(arrays["images"])}
            checked_trace(parent, metadata)
            for q in QUERIES:
                if q not in parent["queries"]:
                    missing.append({"benchmark": benchmark, "task": task, "query": q,
                                    "reason": "query_absent_no_replacement"})
            if parent["queries"]:
                parents.append(parent)
    states = sum(len(p["queries"]) for p in parents)
    if not states or len({p["task"] for p in parents if p["split"] == "holdout"}) < 3:
        raise RuntimeError("Insufficient predeclared task coverage")
    g = zarr.open_group(str(ROUTES), mode="r")
    durable = durable_count(g)
    proposed_ids = set(range(ID_BASE - (states * 9 + 32) + 1, ID_BASE + 1))
    if proposed_ids.intersection(np.asarray(g["episode_id"][:durable]).tolist()):
        raise RuntimeError("Proposed request IDs have already been used")
    config = {
        "schema": "moe-control-p1a-v1", "prepared_utc": utc(), "seed": SEED,
        "queries": list(QUERIES), "holdout_tasks": list(HOLDOUT_TASKS), "candidates": 8,
        "ridge": RIDGE, "request_id_base": ID_BASE, "parents": parents,
        "missing": missing, "states": states, "primary_calls": states * 9,
        "tail_controls": 32, "max_model_calls": states * 9 + 32,
        "new_environment_actions": 0, "server": metadata, "route_store": str(ROUTES),
        "durable_rows_before": durable, "inventories": inventories,
        "source_hashes": script_hashes(),
        "upstream_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_status": subprocess.check_output(["git", "-C", str(REPO), "status", "--short"], text=True),
        "gate_intervention_supported": bool(metadata.get("scope_bias_protocol")),
        "interpretation": "noise-response diagnostic, not gate intervention or physical recovery",
    }
    (BASE / "runs").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="p1a-20260914-", dir=BASE / "runs"))
    save_json(root / "config.json", config)
    (root / "config.sha256").write_text(sha256_file(root / "config.json") + "\n")
    save_json(root / "p0-audit.json", {"passed": True, "checked_utc": utc(),
                                      "parents": len(parents), "states": states,
                                      "missing": missing, "metadata": metadata})
    print(json.dumps({"run": str(root), "parents": len(parents), "states": states,
                      "max_model_calls": config["max_model_calls"], "missing": missing}), flush=True)


def collect(root):
    config = frozen_run(root)
    if (root / "collection-started.json").exists():
        raise RuntimeError("Run already started; partial runs are not silently resumed")
    g = zarr.open_group(str(ROUTES), mode="r")
    existing = set(np.asarray(g["episode_id"][:durable_count(g)]).tolist())
    ids = set(range(ID_BASE - config["max_model_calls"] + 1, ID_BASE + 1))
    if ids.intersection(existing):
        raise RuntimeError("Request ID collision")
    save_json(root / "collection-started.json", {"utc": utc(), "pid": __import__("os").getpid()})
    wire_root = root / "wire"
    wire_root.mkdir()
    rows, repeats, controls, call_id = [], [], [], 0
    started = time.monotonic()
    with PolicyClient("127.0.0.1", 9500, inference_timeout=180) as client, (root / "calls.jsonl").open("x") as log:
        check_metadata(client.metadata, config["server"])

        def infer(parent, manifest, arrays, q, candidate, kind, capture=True):
            nonlocal call_id
            request_id = config["request_id_base"] - call_id
            noise = candidate_noise(parent["benchmark"], parent["task"], q, candidate, arrays["flow_noises"][q])
            request = {"observation/image": arrays["images"][q],
                       "observation/wrist_image": arrays["wrist_images"][q],
                       "observation/state": arrays["states"][q], "prompt": manifest["prompt"],
                       "flow/noise": noise, "episode_id": request_id, "routing/capture": capture}
            begin = time.monotonic()
            log.write(json.dumps({"event": "request", "ordinal": call_id, "request_id": request_id,
                                  "parent": parent["parent"], "query": q, "candidate": candidate,
                                  "kind": kind, "noise_sha256": digest_array(noise)}) + "\n")
            log.flush()
            response = client.infer(request)
            latency = time.monotonic() - begin
            if response.get("flow/noise_sha256") != digest_array(noise):
                raise RuntimeError("Noise acknowledgement mismatch")
            actions = np.asarray(response["actions"])
            if actions.shape != (10, 7) or actions.dtype != np.float32 or not np.isfinite(actions).all():
                raise RuntimeError("Invalid actions")
            payload = {"actions": actions, "noise": noise}
            if capture:
                if list(response["routing/layer_indices"]) != LAYER_IDS:
                    raise RuntimeError("Wire layer identity mismatch")
                payload.update(expert_ids=response["routing/expert_ids"],
                               expert_weights=response["routing/expert_weights"],
                               layer_indices=response["routing/layer_indices"])
            elif "routing/expert_ids" in response:
                raise RuntimeError("Capture-off returned route payload")
            path = wire_root / ("call-%04d.npz" % call_id)
            np.savez_compressed(path, **payload)
            row = {"ordinal": call_id, "request_id": request_id, "kind": kind,
                   "parent": parent["parent"], "benchmark": parent["benchmark"],
                   "task": parent["task"], "split": parent["split"], "query": q,
                   "candidate": candidate, "capture": capture, "latency_seconds": latency,
                   "wire": str(path.relative_to(root)), "wire_sha256": sha256_file(path),
                   "noise_sha256": digest_array(noise), "actions_sha256": digest_array(actions)}
            if candidate == 0:
                row["source_actions_exact"] = bool(np.array_equal(actions, arrays["predicted_actions"][q]))
                if not row["source_actions_exact"]:
                    raise RuntimeError("Default candidate does not reproduce source actions")
            rows.append(row)
            call_id += 1
            log.write(json.dumps({"event": "response", **row}) + "\n")
            log.flush()
            return payload

        for parent in config["parents"]:
            manifest, arrays = checked_trace(parent, client.metadata)
            for q in parent["queries"]:
                base = None
                for candidate in range(8):
                    payload = infer(parent, manifest, arrays, q, candidate, "candidate")
                    if candidate == 0:
                        base = payload
                repeat = infer(parent, manifest, arrays, q, 0, "repeat")
                equal = all(np.array_equal(base[k], repeat[k]) for k in ("actions", "expert_ids", "expert_weights"))
                repeats.append({"parent": parent["parent"], "query": q, "wire_exact": equal})
                if not equal:
                    raise RuntimeError("Same-state/noise repeat changed outputs")
                print(json.dumps({"parent": parent["parent"], "query": q, "calls": call_id,
                                  "repeat_exact": equal, "elapsed_seconds": time.monotonic() - started}), flush=True)
        parent = config["parents"][0]
        manifest, arrays = checked_trace(parent, client.metadata)
        q = parent["queries"][0]
        for pair in range(16):
            on = infer(parent, manifest, arrays, q, 0, "tail_on", True)
            off = infer(parent, manifest, arrays, q, 0, "tail_off", False)
            equal = bool(np.array_equal(on["actions"], off["actions"]))
            controls.append({"pair": pair, "actions_exact": equal})
            if not equal:
                raise RuntimeError("Capture toggle changed actions")
    if len(rows) != config["max_model_calls"]:
        raise RuntimeError("Forward accounting mismatch")
    save_json(root / "collection.json", {"finished_utc": utc(), "calls": rows, "repeats": repeats,
                                         "tail_controls": controls, "model_calls": len(rows),
                                         "new_environment_actions": 0,
                                         "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"finished": True, "model_calls": len(rows), "new_environment_actions": 0}), flush=True)


def copy_routes(root, config, collection):
    g = zarr.open_group(str(ROUTES), mode="r")
    primary = [r for r in collection["calls"] if r["kind"] in ("candidate", "repeat")]
    wanted = {r["request_id"] for r in primary}
    durable = durable_count(g)
    disk_ids = np.asarray(g["episode_id"][:durable])
    locations = {}
    for index in np.flatnonzero(np.isin(disk_ids, list(wanted))):
        identity = int(disk_ids[index])
        if identity in locations:
            raise RuntimeError("Duplicate request identity in route store")
        locations[identity] = int(index)
    if set(locations) != wanted:
        raise RuntimeError("Missing durable primary rows: %d" % len(wanted - set(locations)))
    pool_root = root / "pools"
    pool_root.mkdir(exist_ok=True)
    audit, pools = [], []
    for parent in config["parents"]:
        for q in parent["queries"]:
            state_rows = [r for r in primary if r["parent"] == parent["parent"] and r["query"] == q]
            if [r["candidate"] for r in state_rows] != list(range(8)) + [0]:
                raise RuntimeError("Candidate order or repeat missing")
            p, ids, selected, actions, noises, indices = [], [], [], [], [], []
            for row in state_rows:
                path = root / row["wire"]
                if sha256_file(path) != row["wire_sha256"]:
                    raise RuntimeError("Wire file changed")
                index = locations[row["request_id"]]
                prob = np.asarray(g["hb_router_probs"][index])
                actual_ids = np.asarray(g["hb_expert_ids"][index])
                sel = np.asarray(g["hb_selected_prob"][index])
                if prob.shape != (8, 10, 11, 32) or not np.isfinite(prob).all() or np.any(prob < 0):
                    raise RuntimeError("Invalid full HB rows")
                mass_error = float(np.max(np.abs(prob.astype(np.float32).sum(-1) - 1)))
                if mass_error > .003:
                    raise RuntimeError("Full probability normalization drift")
                if not np.array_equal(np.take_along_axis(prob, actual_ids.astype(np.int64), -1), sel):
                    raise RuntimeError("Selected probabilities mismatch actual expert IDs")
                with np.load(path, allow_pickle=False) as wire:
                    if not np.array_equal(wire["expert_ids"], actual_ids[:, :, 1:, :].transpose(1, 0, 2, 3)):
                        raise RuntimeError("Wire/disk expert identity mismatch")
                    weights = sel[:, :, 1:, :].astype(np.float32).transpose(1, 0, 2, 3)
                    weights /= weights.sum(-1, keepdims=True)
                    weight_error = float(np.max(np.abs(weights - wire["expert_weights"])))
                    if weight_error > .003:
                        raise RuntimeError("Wire/disk selected-weight mismatch")
                    actions.append(wire["actions"].copy())
                    noises.append(wire["noise"].copy())
                p.append(prob)
                ids.append(actual_ids)
                selected.append(sel)
                indices.append(index)
                audit.append({"request_id": row["request_id"], "disk_row": index,
                              "control_step": int(g["control_step"][index]),
                              "mass_error": mass_error, "weight_error": weight_error})
            if not np.array_equal(p[0], p[-1]) or not np.array_equal(ids[0], ids[-1]):
                raise RuntimeError("Repeated request changed full probabilities")
            if not np.all(np.diff(indices) > 0):
                raise RuntimeError("Non-monotonic primary request order")
            path = pool_root / (parent["parent"] + "-q%02d.npz" % q)
            if path.exists():
                raise RuntimeError("Pool output already exists")
            np.savez_compressed(path, hb_router_probs=np.stack(p[:8]), hb_expert_ids=np.stack(ids[:8]),
                                hb_selected_prob=np.stack(selected[:8]), actions=np.stack(actions[:8]),
                                noises=np.stack(noises[:8]), disk_rows=np.asarray(indices[:8]),
                                request_ids=np.asarray([r["request_id"] for r in state_rows[:8]]))
            pools.append({"parent": parent["parent"], "task": parent["task"], "benchmark": parent["benchmark"],
                          "query": q, "split": parent["split"], "path": str(path.relative_to(root)),
                          "sha256": sha256_file(path)})
    save_json(root / "route-audit.json", {"passed": True, "primary_rows": len(audit), "durable_rows": durable,
                                         "max_mass_error": max(a["mass_error"] for a in audit),
                                         "max_weight_error": max(a["weight_error"] for a in audit),
                                         "rows": audit, "pools": pools})
    return pools


def analyze(root):
    config = frozen_run(root)
    collection = load_json(root / "collection.json")
    pools = copy_routes(root, config, collection)
    matrices = {name: [] for name in ("noise", "full_moe", "old_center", "structured_moe", "noise_plus_moe")}
    y, parents, tasks, splits, state_metrics = [], [], [], [], []
    scale = np.asarray(config["server"]["normalization_action_std"][:6])
    if np.any(scale <= 0):
        raise RuntimeError("Invalid action normalization")
    for pool in pools:
        with np.load(root / pool["path"], allow_pickle=False) as data:
            feature = distances(data["hb_router_probs"], data["hb_expert_ids"])
            noise = pair_rms(data["noises"])[:, None]
            normalized = data["actions"][:, :, :6] / scale
            target = pair_rms(normalized)
            matrices["noise"].append(noise)
            matrices["full_moe"].append(feature["full"])
            matrices["old_center"].append(feature["old_center"])
            matrices["structured_moe"].append(feature["structured"])
            matrices["noise_plus_moe"].append(np.concatenate((noise, feature["structured"]), axis=1))
            y.extend(target.tolist())
            parents.extend([pool["parent"]] * 28)
            tasks.extend([pool["task"]] * 28)
            splits.extend([pool["split"]] * 28)
            default_delta = np.sqrt(np.square(normalized[1:] - normalized[0]).mean((1, 2)))
            signs = np.sign(data["actions"][:, :, 6])
            noise_center, _ = medoid(noise)
            route_center, _ = medoid(feature["old_center"])
            state_metrics.append({**pool, "noise_center": noise_center, "route_center": route_center,
                                  "centers_disagree": noise_center != route_center,
                                  "max_default_action_delta": float(default_delta.max()),
                                  "median_default_action_delta": float(np.median(default_delta)),
                                  "gripper_sign_changes": int((signs[1:] != signs[0]).sum()),
                                  "correlated_action_pairs": 28})
    matrices = {key: np.concatenate(value) for key, value in matrices.items()}
    y, parents, tasks, splits = map(np.asarray, (y, parents, tasks, splits))
    train, test = splits == "train", splits == "holdout"
    if set(parents[train]).intersection(parents[test]) or set(tasks[train]).intersection(tasks[test]):
        raise RuntimeError("Train/holdout leakage")
    models, results, predictions = {}, {}, {}
    for name, x in matrices.items():
        model = fit_distance(x[train], y[train], parents[train])
        prediction = predict_distance(model, x[test])
        models[name], predictions[name] = model, prediction
        results[name] = {"holdout": metrics(y[test], prediction, parents[test]), "per_task": {}}
        for task in HOLDOUT_TASKS:
            mask = tasks[test] == task
            results[name]["per_task"][str(task)] = metrics(y[test][mask], prediction[mask], parents[test][mask])
    permuted = []
    for repetition in range(25):
        x = permute_state_blocks(matrices["structured_moe"][test], SEED + repetition)
        permuted.append(metrics(y[test], predict_distance(models["structured_moe"], x), parents[test])["rmse"])
    baseline = results["noise"]["holdout"]["rmse"]
    actual = results["structured_moe"]["holdout"]["rmse"]
    improvement = 1 - actual / baseline if baseline else 0.0
    improved_tasks = sum(results["structured_moe"]["per_task"][str(t)]["mae"] <
                         results["noise"]["per_task"][str(t)]["mae"] for t in HOLDOUT_TASKS)
    holdout_states = [s for s in state_metrics if s["split"] == "holdout"]
    diverse = sum(s["max_default_action_delta"] > .02 for s in holdout_states) / len(holdout_states)
    gates = {
        "integrity_and_coverage": len(set(tasks[test])) >= 3 and len(set(parents[test])) >= 4,
        "rmse_improvement_at_least_10pct": improvement >= .10,
        "at_least_two_holdout_tasks_improve": improved_tasks >= 2,
        "better_than_median_shuffled_routes": actual < float(np.median(permuted)),
        "at_least_half_states_have_action_delta_over_002": diverse >= .5,
    }
    result = {
        "schema": config["schema"], "finished_utc": utc(), "states": len(pools),
        "train_parents": len(set(parents[train])), "holdout_parents": len(set(parents[test])),
        "holdout_base_tasks": sorted(set(tasks[test].tolist())), "models": models,
        "feature_names": feature["names"], "results": results,
        "permuted_route_rmse": permuted, "rmse_relative_improvement": improvement,
        "improved_holdout_tasks": improved_tasks, "holdout_diverse_state_fraction": diverse,
        "gates": gates, "p1a_continue": all(gates.values()), "state_metrics": state_metrics,
        "model_calls": collection["model_calls"], "new_environment_actions": 0,
        "gate_interventions": 0, "physical_recovery_evaluated": False,
        "next_stage": "P1b requires an isolated bias-capable service" if all(gates.values()) else
                      "Stop expansion under this protocol; do not tune on this holdout",
    }
    np.savez_compressed(root / "analysis-data.npz", y=y, parents=parents, tasks=tasks, splits=splits, **matrices)
    save_json(root / "analysis.json", result)
    lines = ["# P1a 实验结果", "", "本轮为固定观测推理诊断，未执行环境动作或 gate 干预，不能报告救回率。", "",
             "- 父轨迹：拟合 %d 条，预留验证 %d 条；验证基础任务：%s。" %
             (result["train_parents"], result["holdout_parents"], result["holdout_base_tasks"]),
             "- 状态 %d 个；模型调用 %d 次；同状态重复 %d 对；capture 对照 %d 对。" %
             (len(pools), collection["model_calls"], len(collection["repeats"]), len(collection["tail_controls"])),
             "- 训练使用动作差异标签，不使用成功标签；基座未训练、共享服务未修改。", "",
             "| 诊断表示 | 预留 RMSE | 预留 MAE |", "|---|---:|---:|"]
    for name, value in results.items():
        lines.append("| %s | %.6f | %.6f |" % (name, value["holdout"]["rmse"], value["holdout"]["mae"]))
    lines += ["", "结构化纯 MoE 相对纯噪声 RMSE 改善 %.2f%%；%d/3 预留任务 MAE 改善。" %
              (100 * improvement, improved_tasks),
              "状态块置换路由的 RMSE 中位数 %.6f。候选动作差异门槛覆盖 %.1f%% 预留状态。" %
              (np.median(permuted), 100 * diverse), "", "## 预先冻结的继续门槛", ""]
    lines.extend("- %s：%s" % (key, "通过" if value else "未通过") for key, value in gates.items())
    lines += ["", "阶段判定：" + ("P1a 通过探索性门槛；P1b 仍需独立 gate 干预接口。" if all(gates.values()) else
                                 "停止本协议的扩大实验，不在该验证集重新挑参数。"),
              "", "## 解释限制", "",
              "候选对、同父轨迹的查询及两个 benchmark 的同基础任务均有关联。这里只有 3 个预留基础任务，不声明统计显著性或广泛泛化。",
              "q20 仅包含尚未结束的源轨迹。观测被冻结，未执行候选后的新观测，因此路由与动作的相关响应不证明路由对动作的因果作用，也不证明物理可恢复性。",
              "原路由中心和噪声中心的比较仅是选中索引诊断，没有候选续跑结局。", "",
              "配置、数据来源和哈希见 `config.json`；采集完整性见 `route-audit.json`；完整逐状态结果见 `analysis.json`。", ""]
    (root / "REPORT.zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"gates": gates, "p1a_continue": result["p1a_continue"],
                      "rmse_relative_improvement": improvement, "next_stage": result["next_stage"]}), flush=True)


def verify(root):
    config = frozen_run(root)
    result = load_json(root / "analysis.json")
    collection = load_json(root / "collection.json")
    audit = load_json(root / "route-audit.json")
    request_ids = [r["request_id"] for r in collection["calls"]]
    if len(request_ids) != len(set(request_ids)) or len(request_ids) != config["max_model_calls"]:
        raise RuntimeError("Request identity or accounting mismatch")
    for parent in config["parents"]:
        checked_trace(parent, config["server"])
    for row in collection["calls"]:
        if sha256_file(root / row["wire"]) != row["wire_sha256"]:
            raise RuntimeError("Wire file checksum mismatch")
    for pool in audit["pools"]:
        if sha256_file(root / pool["path"]) != pool["sha256"]:
            raise RuntimeError("Full route pool checksum mismatch")
    recomputed = {}
    with np.load(root / "analysis-data.npz", allow_pickle=False) as data:
        train, test = data["splits"] == "train", data["splits"] == "holdout"
        for name in result["results"]:
            model = fit_distance(data[name][train], data["y"][train], data["parents"][train])
            if model != result["models"][name]:
                raise RuntimeError("Non-reproducible diagnostic model")
            value = metrics(data["y"][test], predict_distance(model, data[name][test]), data["parents"][test])
            if value != result["results"][name]["holdout"]:
                raise RuntimeError("Non-reproducible holdout metrics")
            recomputed[name] = value
    with PolicyClient("127.0.0.1", 9500, inference_timeout=30) as client:
        check_metadata(client.metadata, config["server"])
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(BASE), "-p", "test_*.py", "-v"],
                           capture_output=True, text=True, check=True)
    (root / "tests.txt").write_text(tests.stdout + tests.stderr)
    files = {str(path.relative_to(root)): sha256_file(path) for path in root.rglob("*") if path.is_file()}
    save_json(root / "verification.json", {"passed": True, "verified_utc": utc(),
                                          "source_hashes_unchanged": True, "server_identity_unchanged": True,
                                          "unique_requests": len(request_ids), "primary_durable_rows": audit["primary_rows"],
                                          "metrics_recomputed": recomputed, "file_sha256": files,
                                          "limitation": "Recomputation and identity audit, not an independent physical outcome test"})
    print(json.dumps({"verified": True, "unique_requests": len(request_ids), "files_hashed": len(files)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        if args.run is None:
            parser.error("--run is required")
        {"collect": collect, "analyze": analyze, "verify": verify}[args.command](args.run.resolve())


if __name__ == "__main__":
    main()
