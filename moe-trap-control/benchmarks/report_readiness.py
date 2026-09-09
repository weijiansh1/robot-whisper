#!/usr/bin/env python3
"""Summarize benchmark installation evidence and verify the pinned source code."""

import datetime
import json
from pathlib import Path
import subprocess

from prepare_assets import digest
from run_benchmarks import ALLOWED_GPUS, CACHE, HERE, ROOTS, write_json


def read(path):
    return json.loads(path.read_text())


def source_integrity(name):
    tree = read(CACHE / ("provenance/moe-control-%s-git-tree.json" % name))
    if tree.get("truncated"):
        raise RuntimeError("Incomplete GitHub source tree")
    entries = [entry for entry in tree["tree"] if entry["type"] == "blob"
               and entry["path"].endswith((".py", ".yaml", ".yml", ".json", ".txt"))]
    checked = []
    for entry in entries:
        # Runtime-generated data directories are covered by the data manifests.
        if any(part in entry["path"].split("/") for part in ("bddl_files", "init_files", "assets")):
            continue
        path = ROOTS[name] / entry["path"]
        if not path.is_file() or digest(path, "sha1", git_blob=True) != entry["sha"]:
            raise RuntimeError("Upstream source differs: " + str(path))
        checked.append({"path": entry["path"], "git_blob_sha1": entry["sha"]})
    return {"commit": tree["sha"], "verified_source_files": checked}


def main():
    report = {
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "allowed_physical_gpus": ALLOWED_GPUS, "excluded_physical_gpu": 6,
        "model_services_restarted": False, "hidden_capture": False,
        "formal_moe_control_collection_started": False,
        "simulation_workers_left_running": False, "benchmarks": {},
    }
    generated = [read(path) for path in (HERE / "runs/pro-env-generation/pro").glob("*/result.json")]
    if len(generated) != 40 or any(row["status"] != "passed" for row in generated):
        raise RuntimeError("Missing generated Pro environment data")
    report["pro_environment_generation"] = {
        "tasks": 40, "initial_states_per_task": 50,
        "seed": 7, "official_perturbator_target": "living_room_table",
        "bddl_unchanged_from_base": [row["variant"]["variant_id"] for row in generated
                                     if row["same_bddl_as_base"]],
        "provenance": str(HERE / "runs/pro-env-generation/pro"),
    }
    for name, expected in (("pro", 200), ("plus", 10030)):
        inventory = read(HERE / ("runs/inventory-%s/%s/result.json" % (name, name)))
        smoke_root = HERE / ("runs/smoke-%s" % name)
        smoke = read(smoke_root / "summary.json")
        episodes = [read(path) for path in (smoke_root / name).glob("*/result.json")]
        if inventory["status"] != "passed" or inventory["checked_tasks"] != expected:
            raise RuntimeError("Benchmark inventory is incomplete")
        if smoke["passed"] != smoke["total"] or any(row["status"] != "passed" for row in episodes):
            raise RuntimeError("Benchmark smoke checks did not pass")
        if sorted({row["physical_gpu"] for row in episodes}) != list(ALLOWED_GPUS):
            raise RuntimeError("Smoke checks did not cover the GPU allowlist")
        report["benchmarks"][name] = {
            "root": str(ROOTS[name]), "source": source_integrity(name),
            "registered_tasks_verified": expected,
            "minimum_initial_states": inventory["minimum_init_states"],
            "distinct_init_files": inventory["distinct_init_files"],
            "smoke_cases_passed": len(episodes),
            "real_inference_queries": sum(row["queries"] for row in episodes),
            "policy_action_steps": sum(row["action_steps"] for row in episodes),
            "full_episodes_evaluated": 0,
            "inventory_report": str(HERE / ("runs/inventory-%s/%s/result.json" % (name, name))),
            "smoke_report": str(smoke_root / "summary.json"),
        }
    report["plus_assets"] = read(CACHE / "plus-assets-install.json")
    pro = read(CACHE / "pro-dataset-install.json")
    report["pro_download"] = {"revision": pro["revision"], "verified_files": len(pro["verified_files"]),
                              "bytes": pro["bytes"], "manifest": str(CACHE / "pro-dataset-install.json")}
    telemetry = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True).stdout
    report["gpu_snapshot"] = [dict(zip(("physical_gpu", "memory_mib", "utilization_percent"),
                                       map(int, line.split(",")))) for line in telemetry.splitlines()]
    usage = subprocess.run(["du", "-s", "-B1", str(CACHE)], check=True, capture_output=True, text=True).stdout
    report["installed_disk_bytes"] = int(usage.split()[0])
    report["status"] = "installed_and_smoke_verified"
    write_json(HERE / "READINESS.json", report)
    print(json.dumps({"status": report["status"], "installed_disk_bytes": report["installed_disk_bytes"],
                      "smoke_cases": sum(row["smoke_cases_passed"] for row in report["benchmarks"].values())}))


if __name__ == "__main__":
    main()
