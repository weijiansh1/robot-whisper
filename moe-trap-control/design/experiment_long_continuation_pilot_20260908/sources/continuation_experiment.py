"""Frozen next-query interventions on committed native Long trajectories."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

from collection_protocol import HERE, PARAMETERS_SHA256, stable_id, verify_frozen_alarm
from collection_storage import digest

sys.path.insert(0, str(HERE.parent / "himoe-route-capture"))
from route_noise_selector import route_centrality_scores, stable_argmin

PROTOCOL = "moe_control.long_next_query.v1"
METHODS = ("v7_frozen", "v8_frozen", "knn20", "knn_euclidean_OR_cosine")
ARMS = ("random", "short_chunk", "edge_noise", "swap1")
REPLICATES = 4
CANDIDATES = 4
INTERVENTIONS = dict(
    timing="first query after executing the native alarm query; no intervention if already terminated",
    short_chunk="uniform integer 1..9, first branch query only; discard tail and replan; then chunk=10",
    edge_noise="maximum mean pairwise Hellinger distance; HB back four layers, first three flow steps, action tokens 1..10",
    noise_pool="four independent N(0,1) arrays [10,24] per state/replicate; random control uses candidate 0",
    swap1="existing swap1_near on first branch query only, paired with candidate 0",
    subsequent_noise="same per-query-index stream across arms; physical times differ after shortening",
    environment="same per-query-index seed across arms; actual executed steps bounded by original horizon",
    compute="all four candidate forwards counted once as shared preparation; selected first forward reused",
    no_outcome_selection=True, hidden_capture=False, batch_size=1,
)


def event_id(main_id, query):
    return stable_id(PROTOCOL, main_id, "next_query", int(query))


def seed_for(main_id, event, replicate, query, stream, candidate=0):
    if stream not in ("policy", "environment", "short_chunk") or not 0 <= replicate < REPLICATES:
        raise ValueError("Invalid experiment random stream")
    if not 0 <= candidate < CANDIDATES or (stream != "policy" and candidate):
        raise ValueError("Invalid candidate stream")
    return int(stable_id(PROTOCOL, main_id, event, replicate, query, stream, candidate)[:8], 16)


def noise_for(main_id, event, replicate, query, candidate=0):
    return np.random.default_rng(seed_for(main_id, event, replicate, query, "policy", candidate)).standard_normal(
        (10, 24)).astype(np.float32)


def short_count(main_id, event, replicate, query):
    return int(np.random.default_rng(seed_for(main_id, event, replicate, query, "short_chunk")).integers(1, 10))


def select_edge(probabilities):
    values = np.asarray(probabilities)
    if values.shape != (CANDIDATES, 8, 10, 11, 32):
        raise ValueError("Candidate routes must align by layer, flow step and token")
    scores = route_centrality_scores(values[:, 4:, :3, 1:])
    return stable_argmin(-scores), scores


def task_bound(task):
    queries = task["parent_queries"]
    for event in task["events"]:
        remaining = 52 - event["start_query"]
        queries += REPLICATES * (len(ARMS) * (remaining + 1) + CANDIDATES)
    return queries * 180 * 1024 + (len(task["events"]) + 2) * 1024**2


def load_plan(path, model="long"):
    plan = json.loads(Path(path).read_text())
    if (plan["protocol"] != PROTOCOL or plan["model"] != model or model != "long" or
            plan["interventions"] != INTERVENTIONS or plan["methods"] != list(METHODS) or
            plan["arms"] != list(ARMS) or plan["replicates"] != REPLICATES or plan["candidates"] != CANDIDATES):
        raise ValueError("Continuation protocol differs from the frozen plan")
    verify_frozen_alarm()
    if plan["frozen_parameters_sha256"] != PARAMETERS_SHA256:
        raise ValueError("Alarm configuration changed")
    for source, expected in plan["source_sha256"].items():
        if digest(source) != expected:
            raise ValueError("Frozen continuation input changed: " + source)
    audit = json.loads(Path(plan["parent_audit"]).read_text())
    verification = json.loads(Path(plan["alarm_verification"]).read_text())
    contract = json.loads(Path(plan["alarm_contract"]).read_text())
    if audit["status"] != "passed" or verification["status"] != "passed":
        raise ValueError("Parents and alarm replay must be audited")
    if contract["collection_audit_sha256"] != digest(plan["parent_audit"]) or contract["parameters_sha256"] != PARAMETERS_SHA256:
        raise ValueError("Alarm replay and native parents do not share a contract")
    if verification["artifacts"]["first_alarms.csv"] != digest(plan["alarm_table"]):
        raise ValueError("Alarm table differs from verified replay")
    parents = {row["main_id"]: row for row in audit["tasks"]}
    with Path(plan["alarm_table"]).open(newline="") as stream:
        alarms = {row["main_id"]: row for row in csv.DictReader(stream)}
    tasks = plan["tasks"]
    if not tasks or len({t["main_id"] for t in tasks}) != len(tasks):
        raise ValueError("Empty or duplicate parent IDs")
    for task in tasks:
        parent = parents[task["main_id"]]
        if parent["suite"] != "libero_10" or task["parent_directory"] != parent["directory"]:
            raise ValueError("Continuation parent identity")
        for key in ("variant_id", "benchmark", "category", "analysis_role"):
            if task[key] != parent[key]:
                raise ValueError("Continuation parent field: " + key)
        result = json.loads((Path(parent["directory"]) / "result.json").read_text())
        if task["noise_seed"] != result["seed"] or task["init_index"] != result["init_index"]:
            raise ValueError("Parent seed/init mismatch")
        if task["parent_queries"] != parent["main_queries"]:
            raise ValueError("Parent query count")
        for filename, key in (("main_complete.json", "parent_commit_sha256"),
                              ("main/manifest.json", "parent_manifest_sha256")):
            if task[key] != digest(Path(parent["directory"]) / filename):
                raise ValueError("Native parent commit changed")
        first = {name: int(alarms[task["main_id"]][name]) for name in METHODS}
        expected_events, expected_terminal = events_for(task["main_id"], first, task["parent_queries"])
        if (task["first_alarms"] != first or task["events"] != expected_events or
                task["terminal_alarms"] != expected_terminal or task["max_output_bytes"] != task_bound(task)):
            raise ValueError("Detector-specific continuation positions or bounds differ")
    return plan


def events_for(main_id, first, length):
    positions, terminal = {}, []
    for method in METHODS:
        q = first[method]
        if q < 0:
            continue
        if q + 1 >= length:
            terminal.append(method)
        else:
            positions.setdefault(q + 1, []).append(method)
    events = [dict(event_id=event_id(main_id, q), start_query=q, alarm_query=q - 1, methods=positions[q])
              for q in sorted(positions)]
    return events, terminal
