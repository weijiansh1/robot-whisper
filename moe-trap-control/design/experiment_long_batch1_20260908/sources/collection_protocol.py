"""Frozen experiment identities, per-query random streams, and task selection."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFEST_SHA256 = "4ebe63a918a48202d628fb428f0fe5998fdfef1a832faa40db6597796ca57767"
PARAMETERS_SHA256 = "f4e65d359411309756b008423d14d8f3e5d6bf604ee71b63ae8a7e510c5f8216"
PROTOCOL = "moe_control.paired_swap1.v1"
REPLICATES = 4


def stable_id(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:24]


def event_id(main_id, query):
    return stable_id(PROTOCOL, main_id, "first_alarm", int(query))


def stream_seed(main_id, event, replicate, query, stream):
    if stream not in ("policy", "environment") or not 0 <= replicate < REPLICATES:
        raise ValueError("Invalid paired random stream")
    return int(stable_id(PROTOCOL, main_id, event, replicate, query, stream)[:8], 16)


def branch_noise(main_id, event, replicate, query):
    seed = stream_seed(main_id, event, replicate, query, "policy")
    return np.random.default_rng(seed).standard_normal((10, 24)).astype(np.float32)


def load_plan(path, model):
    from collection_storage import digest

    plan = json.loads(Path(path).read_text())
    manifest = HERE / "design/collection_manifest.csv"
    if plan["source_manifest_sha256"] != MANIFEST_SHA256 or digest(manifest) != MANIFEST_SHA256:
        raise ValueError("Frozen source task manifest changed")
    if plan["protocol"] != PROTOCOL or plan["model"] != model or plan["replicates"] != REPLICATES:
        raise ValueError("Unsupported experiment protocol")
    if plan["frozen_parameters_sha256"] != PARAMETERS_SHA256:
        raise ValueError("Alarm parameter identity mismatch")
    with manifest.open(newline="") as stream:
        inventory = {row["main_id"]: row for row in csv.DictReader(stream)}
    tasks = plan["tasks"]
    if not tasks or len({task["main_id"] for task in tasks}) != len(tasks):
        raise ValueError("Empty or duplicate main identities")
    for task in tasks:
        source = inventory[task["main_id"]]
        if source["screen"] != "1":
            raise ValueError("Experiment must use the frozen screening sample")
        for key, value in source.items():
            if key != "status" and str(task[key]) != value:
                raise ValueError("Task differs from frozen manifest: " + key)
    return plan
