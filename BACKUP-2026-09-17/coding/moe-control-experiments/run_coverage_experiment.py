"""Collect and analyze frozen equal-candidate-count MoE coverage pools."""

import argparse
from collections import defaultdict
import datetime
import json
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

import numpy as np
import zarr

from gate_runtime import BASE, infer_isolated, load_isolated
from coverage_protocol import (GENERATORS, INIT, POOLS, QUERY, TARGETS, TASKS,
                               gate_bias, pool_metrics, request_noise, specifications)
from run_gate_experiment import (BATCHES, ROUTES, WIRE_KEYS, FULL_KEYS, CAPTURE_KEY, BIAS_KEY,
                                NATIVE_IDS_KEY, EFFECTIVE_IDS_KEY, audit_response, assert_equal,
                                check_identity, digest_array, frozen_files as gate_frozen_files,
                                now, read_json, save_json, set_changes, source_for)
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file

REFERENCE_ID = -1260940000
OLD_RUNS = (BASE / "runs/p1a-20260914-my_4lkxd", BASE / "runs/p1b-gate-20260914-1nvvkzw2")
METRICS = ("mean_radius_rms", "max_radius_rms", "mean_pairwise_rms", "participation_rank",
           "coverage_gain", "best_positive_cosine")
METHODS = GENERATORS + ("mixed_noise_state",)


def frozen_files():
    result = gate_frozen_files()
    for name in ("P2A_COVERAGE_PLAN.zh.md", "coverage_protocol.py", "test_coverage_protocol.py",
                 "run_coverage_experiment.py", "audit_coverage_independently.py", "plot_coverage_results.py"):
        path = BASE / name
        result[str(path)] = sha256_file(path)
    return result


def check_config(root):
    config = read_json(root / "config.json")
    if sha256_file(root / "config.json") != (root / "config.sha256").read_text().strip():
        raise RuntimeError("Frozen configuration changed")
    if config["source_hashes"] != frozen_files():
        raise RuntimeError("Frozen implementation changed")
    return config


def check_reference_ids(ids):
    group = zarr.open_group(str(ROUTES), mode="r")
    durable = min(int(group.attrs["durable_rows"]), group["episode_id"].shape[0])
    if set(ids).intersection(np.asarray(group["episode_id"][:durable]).tolist()):
        raise RuntimeError("Shared reference IDs already used")


def prepare():
    with PolicyClient("127.0.0.1", 9500, inference_timeout=60) as client:
        metadata = client.metadata
    prior_sources = set()
    for old in OLD_RUNS:
        config = read_json(old / "config.json")
        prior_sources.update(p["source"] for p in config["parents"])
    parents, missing, names = [], [], {}
    for benchmark, batch in BATCHES.items():
        manifest = read_json(batch / "manifest.json")
        scenario = read_json(batch / "scenario-plan.json")
        jobs = {(r["base_task_id"], r["episode_index"]): r for r in scenario["jobs"]}
        for task in TASKS:
            job, = [j for j in manifest["jobs"] if j["task_id"] == task and j["episode_index"] == 2]
            if job["init_state_id"] != INIT:
                raise RuntimeError("Unexpected initial state")
            if job["state"] != "completed":
                missing.append(dict(benchmark=benchmark, task=task, reason="incomplete"))
                continue
            source = Path(job["result"]["artifact_dir"])
            if str(source) in prior_sources:
                raise RuntimeError("Parent overlaps earlier experiment")
            trace, arrays = load_episode_trace(source)
            check_identity(metadata, trace["policy_identity"])
            if not trace["result"]["trace_complete"] or trace["result"]["status"] != "completed":
                raise RuntimeError("Incomplete source trace")
            if len(arrays["images"]) <= QUERY:
                missing.append(dict(benchmark=benchmark, task=task, reason="query_absent"))
                continue
            item = jobs[(task, 2)]
            if names.setdefault(task, item["base_task_name"]) != item["base_task_name"] or item["task_name"] != job["result"]["task_name"]:
                raise RuntimeError("Task mapping mismatch")
            parents.append(dict(parent="%s-task%02d-init039-q08" % (benchmark, task), benchmark=benchmark,
                                task=task, query=QUERY, source=str(source), trace_sha256=trace["array_file_sha256"],
                                manifest_sha256=sha256_file(source / "episode-trace.json"),
                                policy_identity=trace["policy_identity"], base_task_name=item["base_task_name"]))
    if len(parents) < 4:
        raise RuntimeError("Insufficient parents")
    ids = [REFERENCE_ID - i for i in range(len(parents) + 1)]
    check_reference_ids(ids)
    config = dict(schema="moe-control-coverage-v1", prepared_utc=now(), parents=parents, missing=missing,
                  shared_metadata=metadata, source_hashes=frozen_files(), specifications=specifications(),
                  targets=[dict(kind="target", generator="noise", pool=0, candidate=i) for i in TARGETS],
                  repeats=[dict(kind="candidate", generator=g, pool=0, candidate=0) for g in GENERATORS],
                  methods=list(METHODS), primary_method="state_gate", reference_ids=ids,
                  isolated_forwards=48 * len(parents), shared_forwards=len(ids),
                  candidate_forwards=32 * len(parents), target_forwards=8 * len(parents),
                  repeat_forwards=4 * len(parents), new_environment_actions=0,
                  training=False, success_labels_used=False, prior_parent_overlap=False,
                  interpretation="Action-space geometry, not physical recovery or online selection")
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    root = Path(tempfile.mkdtemp(prefix="p2a-coverage-" + date + "-", dir=BASE / "runs"))
    save_json(root / "config.json", config)
    (root / "config.sha256").write_text(sha256_file(root / "config.json") + "\n")
    print(json.dumps(dict(run=str(root), parents=len(parents), isolated_forwards=config["isolated_forwards"],
                          shared_forwards=config["shared_forwards"], missing=missing)), flush=True)


def collect(root):
    config = check_config(root)
    if (root / "started.json").exists():
        raise RuntimeError("Run already started; no silent resume")
    check_reference_ids(config["reference_ids"])
    save_json(root / "started.json", dict(utc=now()))
    rows, references, controls = [], [], []
    attempts = dict(isolated=0, shared=0)
    started = time.monotonic()
    try:
        wrapped, loaded = load_isolated()
        save_json(root / "model-load.json", loaded)
        with (root / "calls.jsonl").open("x") as log, PolicyClient("127.0.0.1", 9500, inference_timeout=180) as client:
            if client.metadata.get("server_instance_id") != config["shared_metadata"].get("server_instance_id"):
                raise RuntimeError("Shared instance changed")

            def emit(record):
                log.write(json.dumps(record) + "\n")
                log.flush()

            def shared(parent, request, request_id, path):
                record = dict(engine="shared", ordinal=len(references), parent=parent["parent"], request_id=request_id,
                              noise_sha256=digest_array(request["flow/noise"]), event="request")
                emit(record)
                attempts["shared"] += 1
                response = client.infer(dict(request, episode_id=request_id))
                if response["flow/noise_sha256"] != record["noise_sha256"]:
                    raise RuntimeError("Shared noise acknowledgement mismatch")
                np.savez_compressed(path, **{k: v for k, v in response.items() if isinstance(v, np.ndarray)})
                record.update(event="response", path=str(path.relative_to(root)), sha256=sha256_file(path))
                references.append(record)
                emit(record)
                return response

            def infer(parent, request, name, kind, spec=None):
                full = kind != "bare"
                bias = gate_bias(spec)
                noise = request_noise(parent, request["flow/noise"], spec)
                extra = {"flow/noise": noise}
                if full:
                    extra[CAPTURE_KEY] = True
                if kind == "zero" or np.any(bias):
                    extra[BIAS_KEY] = bias
                record = dict(engine="isolated", ordinal=len(rows), parent=parent["parent"], name=name, kind=kind, spec=spec,
                              noise_sha256=digest_array(noise), bias_sha256=digest_array(bias),
                              bias_l2=float(np.linalg.norm(bias.astype(float))), event="request")
                emit(record)
                attempts["isolated"] += 1
                response, resource = infer_isolated(wrapped, dict(request, **extra))
                audit = audit_response(response, bias) if full else {}
                path = root / "states" / parent["parent"] / (name + ".npz")
                np.savez_compressed(path, **{k: v for k, v in response.items() if isinstance(v, np.ndarray)},
                                    **{"gate_probe/request_bias": bias, "gate_probe/request_noise": noise})
                record.update(event="response", path=str(path.relative_to(root)), sha256=sha256_file(path),
                              resource=resource, audit=audit)
                rows.append(record)
                emit(record)
                return response

            first_request = first_response = None
            for index, parent in enumerate(config["parents"]):
                trace, arrays, request = source_for(parent)
                check_identity(client.metadata, trace["policy_identity"])
                check_identity(loaded["metadata"], trace["policy_identity"])
                directory = root / "states" / parent["parent"]
                directory.mkdir(parents=True)
                np.savez_compressed(directory / "input.npz", **{k: v for k, v in request.items() if isinstance(v, np.ndarray)})
                save_json(directory / "input.json", dict(prompt=request["prompt"], source=parent["source"],
                                                         array_sha256=sha256_file(directory / "input.npz")))
                live = shared(parent, request, config["reference_ids"][index], directory / "shared-reference.npz")
                if not np.array_equal(live["actions"], arrays["predicted_actions"][QUERY]):
                    raise RuntimeError("Shared/source action mismatch")
                if first_request is None:
                    first_request, first_response = request, live
                bare = infer(parent, request, "bare", "bare")
                assert_equal(bare, live, WIRE_KEYS, "Bare/shared mismatch")
                native = infer(parent, request, "native", "native")
                assert_equal(native, live, WIRE_KEYS, "Native/shared mismatch")
                zero = infer(parent, request, "zero", "zero")
                assert_equal(zero, native, FULL_KEYS, "Zero changed baseline")
                for number, spec in enumerate(config["specifications"]):
                    infer(parent, request, "candidate-%02d" % number, "candidate", spec)
                    if (number + 1) % 16 == 0:
                        print(json.dumps(dict(parent=parent["parent"], candidates_done=number + 1,
                                              isolated_calls=len(rows), elapsed_seconds=time.monotonic() - started)), flush=True)
                for number, spec in enumerate(config["targets"]):
                    infer(parent, request, "target-%02d" % number, "target", spec)
                for number, spec in enumerate(config["repeats"]):
                    repeated = infer(parent, request, "repeat-%02d" % number, "repeat", spec)
                    original_index = config["specifications"].index(spec)
                    with np.load(directory / ("candidate-%02d.npz" % original_index), allow_pickle=False) as original:
                        assert_equal(repeated, original, FULL_KEYS, "Repeated candidate changed")
                post = infer(parent, request, "post", "post")
                assert_equal(post, native, FULL_KEYS, "Post-cleanup changed baseline")
                controls.append(dict(parent=parent["parent"], bare_shared_full_exact=True, zero_post_exact=True, repeats_exact=True))
                print(json.dumps(dict(parent=parent["parent"], finished=True, isolated_calls=len(rows))), flush=True)
            final = shared(config["parents"][0], first_request, config["reference_ids"][-1], root / "shared-post.npz")
            assert_equal(final, first_response, WIRE_KEYS, "Shared service changed during collection")
        if len(rows) != config["isolated_forwards"] or len(references) != config["shared_forwards"]:
            raise RuntimeError("Forward accounting mismatch")
        save_json(root / "collection.json", dict(passed=True, rows=rows, references=references, controls=controls,
                                                 attempts=attempts, elapsed_seconds=time.monotonic() - started,
                                                 finished_utc=now(), new_environment_actions=0))
        print(json.dumps(dict(completed=True, isolated=len(rows), shared=len(references))), flush=True)
    except BaseException as error:
        save_json(root / "failure.json", dict(failed_utc=now(), error=str(error), type=type(error).__name__,
                                              attempts=attempts, isolated_completed=len(rows), shared_completed=len(references),
                                              traceback=traceback.format_exc()))
        raise


def analyze(root):
    config = check_config(root)
    collection = read_json(root / "collection.json")
    std = np.asarray(config["shared_metadata"]["normalization_action_std"])
    pools, candidates = [], []
    for parent in config["parents"]:
        directory = root / "states" / parent["parent"]
        with np.load(directory / "native.npz", allow_pickle=False) as saved:
            default = saved["actions"].copy()
            default_ids = saved[EFFECTIVE_IDS_KEY].copy()
        actions, targets = {}, []
        for row in collection["rows"]:
            if row["parent"] != parent["parent"] or row["kind"] not in ("candidate", "target"):
                continue
            path = root / row["path"]
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError("Raw response changed")
            spec = row["spec"]
            with np.load(path, allow_pickle=False) as raw:
                value = raw["actions"].copy()
                if row["kind"] == "target":
                    targets.append(value)
                    continue
                actions[(spec["pool"], spec["generator"], spec["candidate"])] = value
                delta = (value[:, :6].astype(float) - default[:, :6]) / std[:6]
                bias = raw["gate_probe/request_bias"]
                mask = np.any(bias != 0, axis=-1)
                direct = set_changes(raw[NATIVE_IDS_KEY], raw[EFFECTIVE_IDS_KEY])
                candidates.append(dict(parent=parent["parent"], **spec, normalized_action_rms=float(np.sqrt(np.mean(delta ** 2))),
                                       bias_l2=float(np.linalg.norm(bias.astype(float))),
                                       direct_scope_set_change=float(direct[mask].mean()) if mask.any() else None,
                                       all_sites_set_change_vs_baseline=float(set_changes(default_ids, raw[EFFECTIVE_IDS_KEY]).mean()),
                                       gripper_sign_changes=int(np.count_nonzero(np.sign(value[:, 6]) != np.sign(default[:, 6])))))
        for pool in POOLS:
            for generator in METHODS:
                if generator == "mixed_noise_state":
                    points = [actions[(pool, g, i)] for g in ("noise", "state_gate") for i in (0, 1)]
                else:
                    points = [actions[(pool, generator, i)] for i in range(4)]
                metrics = pool_metrics(default, points, targets, std)
                pools.append(dict(parent=parent["parent"], task=parent["task"], benchmark=parent["benchmark"],
                                  pool=pool, generator=generator, **metrics))
    per_parent = []
    for parent in config["parents"]:
        for generator in METHODS:
            values = [p for p in pools if p["parent"] == parent["parent"] and p["generator"] == generator]
            per_parent.append(dict(parent=parent["parent"], task=parent["task"], benchmark=parent["benchmark"], generator=generator,
                                   **{key: float(np.mean([v[key] for v in values])) for key in METRICS}))
    table = []
    baseline = {p["parent"]: p["coverage_gain"] for p in per_parent if p["generator"] == "noise"}
    for generator in METHODS:
        values = [p for p in per_parent if p["generator"] == generator]
        differences = [p["coverage_gain"] - baseline[p["parent"]] for p in values]
        table.append(dict(generator=generator, parents=len(values), **{key: float(np.mean([v[key] for v in values])) for key in METRICS},
                          coverage_difference_vs_noise=float(np.mean(differences)),
                          parents_better_than_noise=sum(d > 1e-12 for d in differences),
                          gripper_sign_changes=sum(c["gripper_sign_changes"] for c in candidates if c["generator"] == generator)
                          if generator in GENERATORS else None))
    resources = {key: max(row["resource"][key] for row in collection["rows"])
                 for key in ("peak_allocated_mib", "peak_reserved_mib")}
    result = dict(schema=config["schema"], parents=len(config["parents"]), table=table, per_parent=per_parent,
                  pools=pools, candidate_metrics=candidates, resources=resources, all_controls_passed=True,
                  isolated_forwards=len(collection["rows"]), shared_forwards=len(collection["references"]),
                  physical_recovery_evaluated=False, new_environment_actions=0, finished_utc=now())
    save_json(root / "analysis.json", result)
    labels = dict(noise="IID 噪声", state_gate="状态 gate 0.10", action_gate="动作 gate 0.10",
                  action_gate_l2="动作 gate 等 L2", mixed_noise_state="固定 2 噪声 + 2 状态 gate")
    lines = ["# P2a：候选动作覆盖结果", "", "本轮是动作空间预检，不是物理续跑或恢复率测试。", "",
             "- %d 条 init39 父轨迹，Plus / Pro，基础任务 0/3/6/9，固定 q8；与前两轮父轨迹不重叠。" % result["parents"],
             "- %d 次独立前向 + %d 次原服务对照，环境动作 0。" % (result["isolated_forwards"], result["shared_forwards"]),
             "- 每方法每池四个新候选，加共同默认动作；两个随机池，八个独立噪声参考动作。", "",
             "## 父轨迹等权结果", "", "动作 RMS 使用 checkpoint 前六维 action std 标准化；覆盖增益是到参考动作的最近距离相对默认减少的比例。", "",
             "| 方法 | 平均动作 RMS | 候选两两 RMS | 参与率维数 | 参考覆盖增益 | 优于噪声的父轨迹 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| %s | %.6f | %.6f | %.3f | %.2f%% | %d/%d |" %
                     (labels[row["generator"]], row["mean_radius_rms"], row["mean_pairwise_rms"], row["participation_rank"],
                      100 * row["coverage_gain"], row["parents_better_than_noise"], row["parents"]))
    primary = next(row for row in table if row["generator"] == "state_gate")
    lines += ["", "预定主方法为状态 gate，其参考覆盖增益相对 IID 噪声差 %.2f 个百分点。其他方法属于预定次要对照，不事后替换主方法。" %
              (100 * primary["coverage_difference_vs_noise"]), "", "## 解读边界", "",
              "参考分布来自独立 IID 噪声，衡量的是对该策略动作区域的几何覆盖，不是正确动作或恢复标签。默认动作在每池中，所以覆盖增益非负也不表示物理上无害。",
              "状态 gate 与动作 gate 等 L2 组匹配总偏置；动作 gate 0.10 组只匹配单位置幅度，其总偏置 L2 为状态组的 sqrt(10) 倍。",
              "固定混合池始终取两种方法的前两个候选，没有按动作或参考挑选。离线最近邻是评估指标，不是已经可部署的 MoE 选择器。",
              "两个池和八参考均先在父轨迹内平均；只有八父轨迹、四基础任务，不宣称统计显著性。32 次规定重复完全一致，不代表所有候选均重复。",
              "P2 物理 C0、报警触发与完整后缀尚未执行。原 P2 恢复门槛不能由本报告替代，P3/P4 没有启动。", "",
              "## 审计与资源", "", "裸推理、native、零偏置、重复、清理后输出与规定对照一致。完整 dispatch 和噪声审计在线通过；独立复算与最终文件校验另见对应 JSON。",
              "PyTorch 峰值分配 %.1f MiB，缓存 %.1f MiB；原 9500 未修改或重启。" %
              (resources["peak_allocated_mib"], resources["peak_reserved_mib"]), "",
              "详见 `analysis.json`、`states/`、`coverage.png`、`independent-audit.json` 和 `verification.json`。", ""]
    with (root / "REPORT.zh.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps(dict(parents=result["parents"], table=table, resources=resources)), flush=True)


def verify(root):
    config = check_config(root)
    collection = read_json(root / "collection.json")
    audit = read_json(root / "independent-audit.json")
    if not audit["passed"] or len(collection["rows"]) != config["isolated_forwards"]:
        raise RuntimeError("Missing collection or independent audit")
    for parent in config["parents"]:
        _, arrays, request = source_for(parent)
        directory = root / "states" / parent["parent"]
        with np.load(directory / "native.npz", allow_pickle=False) as native:
            if not np.array_equal(native["actions"], arrays["predicted_actions"][QUERY]):
                raise RuntimeError("Persisted source actions changed")
        with np.load(directory / "input.npz", allow_pickle=False) as saved:
            for key, value in request.items():
                if isinstance(value, np.ndarray) and not np.array_equal(value, saved[key]):
                    raise RuntimeError("Source input mismatch")
    with PolicyClient("127.0.0.1", 9500, inference_timeout=30) as client:
        if client.metadata.get("server_instance_id") != config["shared_metadata"].get("server_instance_id"):
            raise RuntimeError("Shared instance changed")
    test = subprocess.run(["/data/venv311/bin/python", "-m", "unittest", "discover", "-s", str(BASE), "-p", "test_*.py", "-v"],
                          capture_output=True, text=True, check=True)
    with (root / "tests.txt").open("x") as stream:
        stream.write(test.stdout + test.stderr)
    files = {str(path.relative_to(root)): sha256_file(path) for path in root.rglob("*") if path.is_file()}
    save_json(root / "verification.json", dict(passed=True, files=files, verified_utc=now(),
                                               source_hashes_unchanged=True, shared_instance_unchanged=True,
                                               isolated_forwards=len(collection["rows"]), shared_forwards=len(collection["references"]),
                                               independent_audit_passed=True, physical_recovery_evaluated=False))
    print(json.dumps(dict(verified=True, files=len(files), isolated_forwards=len(collection["rows"]))))


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
