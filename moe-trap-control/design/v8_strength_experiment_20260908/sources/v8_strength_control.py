"""Amplitude/duration sweep of the frozen v8 feature direction."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import numpy as np

from collection_protocol import HERE, stable_id, verify_frozen_alarm
from collection_storage import atomic_json, digest
from collection_routes import (FullHBCapture, FullHBPolicy, FullASCapture, HB_LAYERS, CAPTURE_KEY,
    EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY)
from v8_feature_control import FeatureCapture, make_bias, noise_for, seed_for, V8Monitor, flow_features

PROTOCOL = "moe_control.v8_strength.v1"
BIAS_KEY = "v8_strength/logit_bias"
LOGIT_FIELDS = ("v8_strength/native_logits", "v8_strength/effective_logits", "v8_strength/rounded_bias")
ARMS = {
    "combined1_5": dict(operator="combined", multiplier=1, duration=5),
    "combined4_5": dict(operator="combined", multiplier=4, duration=5),
    "combined16_5": dict(operator="combined", multiplier=16, duration=5),
    "combined1_20": dict(operator="combined", multiplier=1, duration=20),
    "combined4_20": dict(operator="combined", multiplier=4, duration=20),
    "combined16_20": dict(operator="combined", multiplier=16, duration=20),
    "random4_20": dict(operator="random", multiplier=4, duration=20),
    "random16_20": dict(operator="random", multiplier=16, duration=20),
}


def strength_bias(shadow, previous, spec, seed):
    if spec not in ARMS.values():
        raise ValueError("Unregistered strength intervention")
    unit = make_bias(shadow, previous, spec["operator"], 1., seed)
    return np.ascontiguousarray(unit * spec["multiplier"], dtype=np.float32)


class StrengthCapture(FeatureCapture):
    def __init__(self, layers, bias):
        FullHBCapture.__init__(self, layers)
        self.bias = np.asarray(bias, np.float32)
        if (self.bias.shape != (8, 10, 11, 32) or not np.isfinite(self.bias).all() or
                np.abs(self.bias).max() > 16.00001 or np.any(self.bias[:, :, 0] != 0) or np.any(self.bias[:4] != 0)):
            raise ValueError("Strength bias violates registered limits")
        self.dtypes = set()

    def _hook(self, layer):
        def capture(gate, inputs, output):
            import torch
            from torch.nn import functional as F
            value = inputs[0]
            if tuple(value.shape[:2]) != (1, 11):
                raise ValueError("Strength intervention requires batch=1")
            step, slot = divmod(len(self.records), 8)
            if step >= 10 or HB_LAYERS[slot] != layer:
                raise ValueError("Reordered gate calls")
            ids, weights, auxiliary = output
            with torch.no_grad():
                logits = F.linear(value.reshape(-1, value.shape[-1]), gate.weight, None)
                native = logits.softmax(-1)
                bias = torch.as_tensor(self.bias[slot, step], dtype=logits.dtype, device=logits.device)
                effective_logits = logits + bias
                effective = effective_logits.softmax(-1)
                proposed_weights, proposed_ids = effective.topk(4, dim=-1, sorted=False)
                proposed_weights /= proposed_weights.sum(-1, keepdim=True) + 1e-20
                touched = bias.abs().sum(-1, keepdim=True) != 0
                actual_ids = torch.where(touched, proposed_ids, ids)
                actual_weights = torch.where(touched, proposed_weights, weights)
                self.records.append((layer, native.detach().clone(), ids.detach().clone(), weights.detach().clone(),
                    native.topk(4, sorted=False).indices, actual_ids.detach().clone(), actual_weights.detach().clone(),
                    effective.detach().clone(), logits.detach().clone(), effective_logits.detach().clone(), bias.detach().clone()))
                self.dtypes.add((str(logits.dtype), str(native.dtype), str(actual_weights.dtype)))
            return actual_ids, actual_weights, auxiliary
        return capture

    def response(self):
        import torch
        result = super().response()
        for key, column in zip(LOGIT_FIELDS, (8, 9, 10)):
            result[key] = torch.stack([row[column] for row in self.records]).reshape(10, 8, 11, 32).permute(
                1, 0, 2, 3).to(device="cpu", dtype=torch.float32).numpy().copy()
        if len(self.dtypes) != 1:
            raise ValueError("Mixed gate arithmetic dtypes")
        result["v8_strength/dtypes"] = list(next(iter(self.dtypes)))
        return result


class StrengthPolicy(FullHBPolicy):
    def __init__(self, policy):
        super().__init__(policy)
        self.metadata.update(v8_strength_protocol=PROTOCOL, v8_strength_max_bias=16.,
                             v8_strength_logit_capture=True)

    def infer(self, observation):
        if BIAS_KEY not in observation:
            return super().infer(observation)
        request = dict(observation)
        bias = request.pop(BIAS_KEY)
        if not request.pop(CAPTURE_KEY, False) or not request.get("routing/capture"):
            raise ValueError("Strength control requires actual dispatch capture")
        with StrengthCapture(self.policy._routing_layers, bias) as capture, FullASCapture(self.as_layers) as as_capture:
            response = self.policy.infer(request)
        response.update(capture.response())
        response.update(as_capture.response())
        for full, independent in ((EFFECTIVE_IDS_KEY, "routing/expert_ids"), (EFFECTIVE_WEIGHTS_KEY, "routing/expert_weights")):
            np.testing.assert_array_equal(response[full][:, :, 1:].transpose(1, 0, 2, 3), response[independent])
        return response


def prepare(args):
    verify_frozen_alarm()
    if args.parents < 1:
        raise ValueError("A positive parent count is required")
    audit_path = HERE / "design/v8_feature_audit_20260908.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed":
        raise ValueError("Passed original feature experiment audit required")
    run = Path(audit["run"])
    old_plan = json.loads((run / "plan.json").read_text())
    old_tasks = {t["main_id"]: t for t in old_plan["tasks"]}
    groups = defaultdict(list)
    for row in audit["tasks"]:
        groups[row["benchmark"], tuple(row["active_heads"]), row["native_success"]].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: stable_id(PROTOCOL, "parent_selection", r["main_id"]))
    chosen = []
    while len(chosen) < min(args.parents, len(audit["tasks"])):
        for key in sorted(groups):
            if groups[key] and len(chosen) < args.parents:
                chosen.append(groups[key].pop(0))
    tasks, jobs, bound = [], [], 0
    for row in chosen:
        task = dict(old_tasks[row["main_id"]])
        directory = run / "tasks" / row["main_id"]
        result = json.loads((directory / "result.json").read_text())
        if result["status"] != "completed" or result["c0"]["status"] != "passed":
            raise ValueError("Original C0 not complete")
        event = result["events"][0]
        task.update(replay_directory=str(directory), replay_result_sha256=digest(directory / "result.json"),
            replay_c0_sha256=digest(directory / "c0/branch.json"), event=event, native_success=row["native_success"])
        tasks.append(task)
        remaining = (520 - event["action_steps_before"] + 9) // 10
        job_bound = sum(remaining * 160000 + min(spec["duration"], remaining) * 950000 for spec in ARMS.values()) + 2 * 1024**2
        for replicate in range(2):
            jobs.append(dict(job_id=stable_id(PROTOCOL, task["main_id"], replicate),
                main_id=task["main_id"], replicate=replicate, max_output_bytes=job_bound,
                maximum_queries=sum(remaining + min(spec["duration"], remaining) for spec in ARMS.values())))
            bound += job_bound
    jobs.sort(key=lambda row: (-row["maximum_queries"], row["job_id"]))
    if shutil.disk_usage(HERE).free - bound < 8 * 1024**3:
        raise ValueError("Conservative strength plan exceeds disk headroom: %.2f GiB" % (bound / 1024**3))
    sources = [Path(__file__), HERE / "v8_feature_control.py", audit_path, run / "summary.json", run / "plan.json"]
    plan = dict(protocol=PROTOCOL, stage="amplitude_duration_exploration", model="long", arms=list(ARMS),
        arms_registry=ARMS, tasks=tasks, jobs=jobs, replicates=2, allowed_gpus=[0, 1, 2, 3],
        replicas_per_gpu=8, workers_per_gpu=8, render_gpus=[0, 3], parent_audit=str(audit_path),
        source_sha256={str(p.resolve()): digest(p) for p in sources},
        selection="fixed hash round-robin by benchmark, active head and original outcome; retains original-success controls; no new intervention outcomes used",
        timing=old_plan["timing"], random_stream="exact same policy/environment/direction seeds as original feature experiment",
        references="original native and combined1_5; rerun combined1_5 must match the complete old combined suffix exactly",
        control="multiply the frozen unit combined/random bias by 1,4,16; duration5 or20; real gate top4 and normalized weights",
        logit_evidence="save native/effective logits, rounded bias, probabilities, dispatched ids/weights and arithmetic dtypes",
        horizon=520, chunk=10, hidden_capture=False, batch_size=1, threshold_fitting=False,
        new_main_coverage=0, maximum_output_bytes=bound, storage_quota_gib=9, disk_floor_gib=8)
    if args.output.exists():
        raise ValueError("Refusing to replace a frozen strength plan")
    atomic_json(args.output, plan)
    print(json.dumps(dict(plan=str(args.output), parents=len(tasks), jobs=len(jobs),
        new_suffixes=len(jobs)*len(ARMS), maximum_gib=bound/1024**3,
        successes=sum(t["native_success"] for t in tasks), benchmarks={b:sum(t["benchmark"]==b for t in tasks) for b in ("pro", "plus")})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
