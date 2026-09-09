"""Frozen route-guided candidate selection and bounded recovery bursts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np

from collection_protocol import HERE, PARAMETERS_SHA256, stable_id, verify_frozen_alarm
from collection_storage import digest

sys.path.insert(0, str(HERE.parent / "moe-v7-0905/method"))
sys.path.insert(0, str(HERE.parent / "himoe-route-capture"))
from intrinsic_guard_monitor import GlobalIntrinsicProfile, IntrinsicGuardMonitor, intrinsic_score_arrays
from route_noise_selector import route_centrality_scores, stable_argmin

PROTOCOL = "moe_control.route_recovery.v1"
GPUS = (0, 1, 2, 3)
RENDER_GPUS = (0, 3)
CANDIDATES = 4
REFERENCE = HERE / "design/frozen_alarm_comparison_20260908/profiles/global_reference.npz"
REFERENCE_SHA256 = "e8076c132610be0288c0d7435902c1238751c17bf1fcc1dfb6991aa97350b581"
PARAMETERS = REFERENCE.with_name("parameters.json")
ARMS = {
    **{"candidate%d" % i: dict(selector="fixed", candidate=i, chunk=10, duration=1) for i in range(4)},
    "short2_once": dict(selector="fixed", candidate=0, chunk=2, duration=1),
    "short2_5": dict(selector="fixed", candidate=0, chunk=2, duration=5),
    "short5_5": dict(selector="fixed", candidate=0, chunk=5, duration=5),
    "short2_20": dict(selector="fixed", candidate=0, chunk=2, duration=20),
    "center5_5": dict(selector="center", candidate=0, chunk=5, duration=5),
    "edge5_5": dict(selector="edge", candidate=0, chunk=5, duration=5),
    "knn5_5": dict(selector="knn", candidate=0, chunk=5, duration=5),
    "guarded_knn5_5": dict(selector="guarded_knn", candidate=0, chunk=5, duration=5),
}
CONTRACT = dict(
    alarm="unchanged global Euclidean kNN-20; first q>=7 with distance > frozen threshold",
    causal_timing="q+1, after executing the native alarm query",
    diagnostic_timing="max(0,q-5); retrospectively placed, never a deployable trigger",
    candidate_pool="four iid N(0,1) [10,24] arrays, batch=1, all full forwards counted",
    ranking="same-observation candidates only; no suffix, outcome, task identity or simulator progress",
    knn="minimum frozen 10D successful-reference distance; relative ranking after shortened chunks is not calibrated failure probability",
    guard="candidate 0 is always eligible; same first gripper sign and live-action RMS <= pool median relative to candidate 0",
    center_edge="mean pairwise Hellinger on HB back four layers, first three flow steps, action tokens",
    recovery="fixed finite query burst, then native chunk=10 and one random candidate",
    random_stream="policy paired by branch-relative query across times/arms; environment paired by absolute action step",
    restore="complete exact C0 before any branch; main committed before replay",
    labels="all four first candidates receive complete suffix labels; oracle is diagnostic only",
    hidden_capture=False, batch_size=1, threshold_fitting=False,
)


def seed_for(main_id, replicate, index, stream, candidate=0):
    if stream not in ("policy", "environment") or replicate < 0 or index < 0:
        raise ValueError("Invalid recovery random stream")
    if not 0 <= candidate < CANDIDATES or (stream == "environment" and candidate):
        raise ValueError("Invalid candidate index")
    return int(stable_id(PROTOCOL, main_id, int(replicate), int(index), stream, int(candidate))[:8], 16)


def noise_for(main_id, replicate, index, candidate=0):
    return np.random.default_rng(seed_for(main_id, replicate, index, "policy", candidate)).standard_normal(
        (10, 24)).astype(np.float32)


def events_for(main_id, first, length, diagnostic):
    q = int(first)
    if q < 0:
        return []
    positions = [("after_alarm", q + 1)]
    if diagnostic:
        positions.append(("before_alarm_diagnostic", max(0, q - 5)))
    return [dict(event_id=stable_id(PROTOCOL, main_id, timing, start), timing=timing,
                 start_query=start, alarm_query=q, deployable=timing == "after_alarm")
            for timing, start in positions if start < length]


def chunk_limit(arm, index, remaining):
    spec = ARMS[arm]
    return min(spec["chunk"] if index < spec["duration"] else 10, remaining)


class RouteRisk:
    def __init__(self):
        verify_frozen_alarm()
        if digest(REFERENCE) != REFERENCE_SHA256:
            raise ValueError("Frozen reference bank changed")
        with np.load(REFERENCE, allow_pickle=False) as archive:
            self.reference = {key: archive[key] for key in archive.files}
        config = json.loads(PARAMETERS.read_text())
        self.profile = GlobalIntrinsicProfile(**config["legacy"]["v7"])
        self.threshold = config["geometry"]["thresholds"]["knn20"]["threshold"]

    def monitor(self):
        return IntrinsicGuardMonitor(self.profile)

    def current(self, monitor):
        if monitor.query < 7:
            return np.full(10, np.nan), float("nan")
        mobility = np.asarray(monitor._mobility_history, np.float32)[None]
        acceleration = np.asarray(monitor._acceleration_history, np.float32)[None]
        periodicity = np.asarray(monitor._periodicity_history, np.float32)[None]
        base = mobility[:, 1:5].mean(1)
        relative = -np.log(np.maximum(mobility, 1e-6) / np.maximum(base[:, None], 1e-6))
        layers = relative[:, -6:].mean(1, dtype=np.float32)[0]
        heads = intrinsic_score_arrays(mobility, acceleration, periodicity,
                                       float(self.reference["periodicity_scale"]))
        raw = np.concatenate((layers, [heads["acceleration"][0, -1], heads["periodicity"][0, -1]])).astype(np.float32)
        vector = (raw.astype(np.float64) - self.reference["dynamic_center"]) / self.reference["dynamic_scale"]
        if not np.isfinite(vector).all():
            return vector, float("nan")
        distance = np.linalg.norm(self.reference["success_dynamic"] - vector, axis=1)
        score = np.float32(np.partition(distance, 19)[:20].mean())
        return vector, float(score)

    def preview(self, monitor, probabilities):
        vectors, scores = [], []
        for probability in probabilities:
            trial = copy.deepcopy(monitor)
            trial.update(probability)
            vector, score = self.current(trial)
            vectors.append(vector)
            scores.append(score)
        return np.asarray(vectors), np.asarray(scores)


def candidate_scores(probabilities, actions, monitor, risk):
    p, a = np.asarray(probabilities), np.asarray(actions)
    if p.shape != (4, 8, 10, 11, 32) or a.ndim != 3 or a.shape[:2] != (4, 10) or a.shape[-1] < 7:
        raise ValueError("Misaligned recovery candidates")
    if not np.isfinite(a).all():
        raise ValueError("Non-finite candidate actions")
    centrality = route_centrality_scores(p[:, 4:, :3, 1:])
    vectors, knn = risk.preview(monitor, p)
    delta = np.sqrt(np.square(a[:, :, :7].astype(np.float64) - a[0, :, :7]).mean(axis=(1, 2)))
    eligible = (delta <= np.median(delta)) & (np.sign(a[:, 0, 6]) == np.sign(a[0, 0, 6]))
    eligible[0] = True
    return dict(centrality=centrality, vectors=vectors, knn=knn, action_rms=delta, eligible=eligible)


def select_candidate(selector, scores):
    if selector == "center":
        return stable_argmin(scores["centrality"])
    if selector == "edge":
        return stable_argmin(-scores["centrality"])
    if selector not in ("knn", "guarded_knn"):
        raise ValueError("Unsupported route selector")
    valid = np.isfinite(scores["knn"])
    if selector == "guarded_knn":
        valid &= scores["eligible"]
    ids = np.flatnonzero(valid)
    return stable_argmin(scores["knn"][ids], ids) if len(ids) else 0


def branch_bound(event, arms):
    remaining = 520 - event["start_query"] * 10
    queries = CANDIDATES
    for arm in arms:
        spec = ARMS[arm]
        controlled = min(spec["duration"], (remaining + spec["chunk"] - 1) // spec["chunk"])
        queries += controlled + max(0, (remaining - controlled * spec["chunk"] + 9) // 10)
        if spec["selector"] != "fixed":
            queries += 3 * max(0, controlled - 1)
    return queries * 190 * 1024 + 2 * 1024**2


def scheduled_jobs(plan, inventory, output):
    jobs = []
    for task in plan["tasks"]:
        parent_id = task["main_id"]
        base = dict(inventory[task["variant_id"]], main_id=parent_id,
                    noise_seed=int(task["noise_seed"]), init_index=int(task["init_index"]))
        replay_id = parent_id + "/replay"
        jobs.append(dict(base, job_id=replay_id, depends_on=None,
                         sampling=dict(kind="replay", parent=task, max_output_bytes=task["max_output_bytes"])))
        for event in task["events"]:
            for replicate in range(plan["replicates"]):
                job_id = "%s/events/%s/repeat%d" % (parent_id, event["event_id"], replicate)
                jobs.append(dict(base, job_id=job_id, depends_on=replay_id,
                    sampling=dict(kind="branches", parent=task, event=event, replicate=replicate,
                        arms=plan["arms"], replay_directory=str(Path(output) / "tasks" / replay_id),
                        max_output_bytes=branch_bound(event, plan["arms"]))))
    jobs.sort(key=lambda row: (row["depends_on"] is not None, -row["sampling"]["max_output_bytes"], row["job_id"]))
    return jobs


def load_plan(path, model="long"):
    plan = json.loads(Path(path).read_text())
    verify_frozen_alarm()
    if (plan["protocol"] != PROTOCOL or plan["contract"] != CONTRACT or model != "long" or
            plan["allowed_gpus"] != list(GPUS) or plan["render_gpus"] != list(RENDER_GPUS) or
            plan["frozen_parameters_sha256"] != PARAMETERS_SHA256 or plan["arms_registry"] != ARMS):
        raise ValueError("Recovery plan contract changed")
    if not set(plan["arms"]) <= set(ARMS) or not set("candidate%d" % i for i in range(4)) <= set(plan["arms"]):
        raise ValueError("All four diagnostic candidates must have complete labels")
    for source, expected in plan["source_sha256"].items():
        if digest(source) != expected:
            raise ValueError("Recovery input changed: " + source)
    import csv
    audit = json.loads(Path(plan["parent_audit"]).read_text())
    verification = json.loads(Path(plan["alarm_verification"]).read_text())
    contract = json.loads(Path(plan["alarm_contract"]).read_text())
    if audit["status"] != "passed" or verification["status"] != "passed":
        raise ValueError("Unverified parents or triggers")
    if (contract["collection_audit_sha256"] != digest(plan["parent_audit"]) or
            contract["parameters_sha256"] != PARAMETERS_SHA256 or
            verification["artifacts"]["first_alarms.csv"] != digest(plan["alarm_table"])):
        raise ValueError("Alarm replay provenance")
    parents = {row["main_id"]: row for row in audit["tasks"]}
    with Path(plan["alarm_table"]).open(newline="") as stream:
        alarms = {row["main_id"]: row for row in csv.DictReader(stream)}
    if not plan["tasks"] or len({t["main_id"] for t in plan["tasks"]}) != len(plan["tasks"]):
        raise ValueError("Empty/duplicated recovery parents")
    for task in plan["tasks"]:
        parent = parents[task["main_id"]]
        for key in ("directory", "main_queries", "variant_id", "benchmark", "category", "analysis_role"):
            alias = dict(directory="parent_directory", main_queries="parent_queries").get(key, key)
            if task[alias] != parent[key]:
                raise ValueError("Parent identity: " + key)
        directory = Path(task["parent_directory"])
        original = json.loads((directory / "result.json").read_text())
        if task["noise_seed"] != original["seed"] or task["init_index"] != original["init_index"]:
            raise ValueError("Native random seed/init changed")
        for filename, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
            if digest(directory / filename) != task[key]:
                raise ValueError("Parent commit changed")
        first = int(alarms[task["main_id"]]["knn20"])
        if task["first_alarm"] != first or task["events"] != events_for(task["main_id"], first, task["parent_queries"], plan["diagnostic"]):
            raise ValueError("Recovery trigger positions changed")
    return plan
