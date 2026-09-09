"""One physical withdrawal module, evaluated under frozen causal trigger policies."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import shutil

import numpy as np

from adaptive_control import RouteRisk, PARAMETERS, REFERENCE
from collection_protocol import HERE, PARAMETERS_SHA256, stable_id, verify_frozen_alarm
from collection_storage import atomic_json, digest, records
from collection_routes import PROBS_KEY
from v8_feature_control import V8Monitor

PROTOCOL = "moe_control.fixed_physical_recovery.v1"
METHODS = ("v7_frozen", "v8_frozen", "knn20", "knn_euclidean_OR_cosine", "behavior_stall", "random_time")
ARMS = ("native", "hold", "withdraw")
MODULE = dict(lift_m=.04, retreat_max_m=.04, target_history_queries=3, phase_steps=8,
              max_translation_step_m=.01, rotation_delta=0., gripper="last executed command sign",
              horizon=520, chunk=10)
STALL = dict(window_queries=6, maximum_position_diameter_m=.005, first_query=7)


def stalled(positions):
    if len(positions) < STALL["window_queries"]:
        return False
    values = np.asarray(positions[-STALL["window_queries"]:], dtype=np.float64)
    diameter = np.linalg.norm(values[:, None] - values[None, :], axis=-1).max()
    return bool(diameter <= STALL["maximum_position_diameter_m"])


class TriggerMonitor:
    def __init__(self):
        self.risk, self.v8 = RouteRisk(), V8Monitor()
        self.positions, self.first = [], {name: -1 for name in METHODS[:-1]}
        self.bank = self.risk.reference["success_dynamic"]
        self.bank_norm = np.linalg.norm(self.bank, axis=1)
        self.cosine_threshold = json.loads(PARAMETERS.read_text())["geometry"]["thresholds"]["cosine_knn20"]["threshold"]

    def update(self, probabilities, position):
        status = self.v8.update(probabilities)
        vector, euclidean = self.risk.current(self.v8.v7)
        cosine = float("nan")
        if np.isfinite(vector).all():
            scores = np.clip(1 - (self.bank @ vector) / np.maximum(self.bank_norm * np.linalg.norm(vector), 1e-12), 0, 2)
            cosine = float(np.float32(np.partition(scores, 19)[:20].mean()))
        self.positions.append(np.asarray(position, dtype=np.float64).copy())
        q = self.v8.v7.query
        flags = dict(v7_frozen=bool(status["alarm"]), v8_frozen=bool(status["v8_alarm"]),
            knn20=euclidean > self.risk.threshold,
            knn_euclidean_OR_cosine=euclidean > self.risk.threshold or cosine > self.cosine_threshold,
            behavior_stall=q >= STALL["first_query"] and stalled(self.positions))
        for name, value in flags.items():
            if self.first[name] < 0 and value:
                self.first[name] = q
        status.update(knn_score=euclidean, cosine_score=cosine, behavior_stall=flags["behavior_stall"])
        return status


def targets(current, recent):
    current, recent = np.asarray(current, float), np.asarray(recent, float)
    xy = recent[:2] - current[:2]
    xy *= min(1., MODULE["retreat_max_m"] / max(np.linalg.norm(xy), 1e-12))
    lift = current + np.array([0., 0., MODULE["lift_m"]])
    retreat = lift + np.r_[xy, 0.]
    return np.stack((lift, retreat))


def recovery_action(arm, position, target, gripper, output_scale):
    if arm not in ("hold", "withdraw"):
        raise ValueError("No physical action in the native arm")
    action = np.zeros(7, dtype=np.float64)
    if arm == "withdraw":
        delta = np.asarray(target, float) - np.asarray(position, float)
        delta *= min(1., MODULE["max_translation_step_m"] / max(np.linalg.norm(delta), 1e-12))
        action[:3] = delta / np.asarray(output_scale, float)
    action[6] = 1. if gripper > 0 else -1.
    if not np.isfinite(action).all() or np.abs(action).max() > 1.000001:
        raise ValueError("Physical recovery action out of bounds")
    return action


def rng_record(step):
    n, p = np.random.get_state(), random.getstate()
    if n[0] != "MT19937" or p[0] != 3:
        raise ValueError("Unexpected environment RNG codec")
    return dict(query=np.int32(step), numpy_keys=n[1].copy(), numpy_position=np.int32(n[2]),
        numpy_has_gauss=np.int8(n[3]), numpy_gauss=np.float64(n[4]),
        python_keys=np.asarray(p[1], np.uint32), python_has_gauss=np.bool_(p[2] is not None),
        python_gauss=np.float64(0. if p[2] is None else p[2]))


def restore_rng(row):
    np.random.set_state(("MT19937", row["numpy_keys"].copy(), int(row["numpy_position"]),
                         int(row["numpy_has_gauss"]), float(row["numpy_gauss"])))
    random.setstate((3, tuple(int(x) for x in row["python_keys"]),
                     float(row["python_gauss"]) if row["python_has_gauss"] else None))


def set_environment_rng(tape, main_id, step):
    if step < len(tape):
        restore_rng(tape[step])
    else:
        seed = int(stable_id(PROTOCOL, main_id, "post_native_terminal_environment", step)[:8], 16)
        np.random.seed(seed)
        random.seed(seed)


def prepare(args):
    verify_frozen_alarm()
    audit_path = HERE / "design/experiment_long_batch1_audit_20260908.json"
    alarm_dir = HERE / "design/experiment_long_batch1_alarm_comparison_20260908"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed":
        raise ValueError("Audited native cohort required")
    with (alarm_dir / "first_alarms.csv").open(newline="") as stream:
        alarms = {r["main_id"]: r for r in csv.DictReader(stream)}
    groups = defaultdict(list)
    for parent in audit["tasks"]:
        groups[parent["benchmark"], parent["category"]].append(parent)
    selected = []
    for key in sorted(groups):
        selected.extend(sorted(groups[key], key=lambda r: stable_id(PROTOCOL, "cohort", r["main_id"]))[:args.per_category])
    tasks, bound = [], 0
    for parent in selected:
        directory = Path(parent["directory"])
        original = json.loads((directory / "result.json").read_text())
        monitor = TriggerMonitor()
        for row in records(directory / "main"):
            monitor.update(row[PROBS_KEY], row["proprio"][:3])
        first = dict(monitor.first)
        for name in METHODS[:4]:
            if first[name] != int(alarms[parent["main_id"]][name]):
                raise ValueError("Frozen trigger mismatch: " + name)
        first["random_time"] = int(np.random.default_rng(int(stable_id(PROTOCOL, parent["main_id"], "random_time")[:8], 16)).integers(8, 40))
        positions = defaultdict(list)
        for method, q in first.items():
            if 0 <= q + 1 < parent["main_queries"] and q >= 0:
                positions[q + 1].append(method)
        events = [dict(event_id=stable_id(PROTOCOL, parent["main_id"], q), start_query=q,
            alarm_query=q-1, methods=positions[q], deployable=True) for q in sorted(positions)]
        task = {k: parent[k] for k in ("main_id", "variant_id", "benchmark", "category", "analysis_role")}
        maximum_queries = parent["main_queries"] + sum(3 * (52-e["start_query"]) for e in events)
        task.update(parent_directory=str(directory), parent_queries=parent["main_queries"],
            parent_commit_sha256=digest(directory / "main_complete.json"), parent_manifest_sha256=digest(directory / "main/manifest.json"),
            noise_seed=original["seed"], init_index=original["init_index"], first_alarm=first["knn20"], first_alarms=first,
            base_task=original["variant"]["base_task"], native_success=original["success"], events=events,
            maximum_queries=maximum_queries, max_output_bytes=maximum_queries*180*1024+(len(events)+2)*1024**2+520*6*1024)
        tasks.append(task)
        bound += task["max_output_bytes"]
    plan = dict(protocol=PROTOCOL, model="long", methods=list(METHODS), arms=list(ARMS), module=MODULE,
        behavior_trigger=STALL, tasks=tasks, allowed_gpus=[0,1,2,3], render_gpus=[0,3],
        replicas_per_gpu=8, workers_per_gpu=8, hidden_capture=False, threshold_fitting=False, batch_size=1,
        selection="fixed hash, equal count per benchmark/category from all 120 audited Long parents; no alarm/outcome filtering",
        timing="first trigger q, execute native q, intervene before q+1; no effective trigger means retain complete native outcome",
        random_trigger="one query drawn uniformly from [8,39] per main, before outcomes; may occur after native termination",
        random_stream="original policy RNG indexed by query; environment RNG tape from exact C0 indexed by absolute action step; common deterministic extension beyond native termination",
        recovery="once per policy, 8 lift then 8 retreat steps, hold last gripper sign; identical 16-step hold control; remaining total520 budget unchanged",
        new_main_coverage=0, parent_audit=str(audit_path), frozen_parameters_sha256=PARAMETERS_SHA256,
        source_sha256={str(p): digest(p) for p in (Path(__file__).resolve(), audit_path, alarm_dir/"first_alarms.csv",
            alarm_dir/"verification.json", alarm_dir/"contract.json", PARAMETERS, REFERENCE)},
        maximum_output_bytes=bound, storage_quota_gib=6, disk_floor_gib=8)
    if args.output.exists() or shutil.disk_usage(HERE).free-bound < 8*1024**3:
        raise ValueError("Plan exists or insufficient disk headroom")
    atomic_json(args.output, plan)
    print(json.dumps(dict(parents=len(tasks), events=sum(len(t["events"]) for t in tasks),
        suffixes=3*sum(len(t["events"]) for t in tasks), native_successes=sum(t["native_success"] for t in tasks),
        maximum_gib=bound/1024**3, maximum_queries=sum(t["maximum_queries"] for t in tasks))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
