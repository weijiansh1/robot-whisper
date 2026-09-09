#!/usr/bin/env python3
"""Build an offline sampling manifest and conditional MoE-control budgets.

This script does not load a policy, initialize a simulator, or collect rollouts.
Inventory preparation accepts downloaded official JSON / Python task maps;
Python task maps are parsed with ast.literal_eval, never executed.
"""

from __future__ import annotations

import argparse
import ast
import base64
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

KIB = 1024
MIB = 1024**2
GIB = 1024**3
HERE = Path(__file__).resolve().parent
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = dict(zip(SUITES, (220, 280, 300, 520)))
PRO_FAMILIES = {
    "object": "Object", "swap": "Position", "lan": "Semantic",
    "task": "Task", "env": "Environment",
}
PRO_COMMIT = "eafdb809426b13153aa1e4c42d6601844217dfec"
PLUS_COMMIT = "4976dc30028e805ff8094b55501d532c48fec182"
PRO_MAP_BLOB = "6eb74a4855fcdb72df3cb054221a74814c79e8b9"
PLUS_CLASSIFICATION_SHA = "faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6"
PLUS_MAP_SHA = "41640d2f542169b400be7e94249e3b1f862e7500745dc318bab49b19576b31ac"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def task_map(source):
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("libero_task_map literal was not found")


def prepare_inventory(args):
    pro_api = read_json(args.pro_map_api)
    pro_bytes = base64.b64decode(pro_api["content"], validate=False)
    blob = hashlib.sha1(b"blob " + str(len(pro_bytes)).encode() + b"\0" + pro_bytes).hexdigest()
    if blob != pro_api["sha"] or blob != PRO_MAP_BLOB:
        raise ValueError("PRO task map blob checksum mismatch")
    pro = task_map(pro_bytes.decode())
    plus_bytes = Path(args.plus_classification).read_bytes()
    plus = json.loads(plus_bytes)
    plus_map_bytes = Path(args.plus_map).read_bytes()
    if hashlib.sha256(plus_bytes).hexdigest() != PLUS_CLASSIFICATION_SHA:
        raise ValueError("PLUS classification differs from the pinned revision")
    if hashlib.sha256(plus_map_bytes).hexdigest() != PLUS_MAP_SHA:
        raise ValueError("PLUS task map differs from the pinned revision")
    plus_map = task_map(plus_map_bytes.decode())
    rows = []
    for suite in SUITES:
        bases = pro[suite]
        if len(bases) != 10 or len(set(bases)) != 10:
            raise ValueError(f"Unexpected base-task inventory: {suite}")
        folds = {
            name: rank % 5 for rank, name in enumerate(
                sorted(bases, key=lambda name: digest(["fold-20260908", suite, name]))
            )
        }
        for suffix, category in PRO_FAMILIES.items():
            registry = f"{suite}_{suffix}"
            if set(pro[registry]) != set(bases):
                raise ValueError(f"PRO base-task mapping changed: {registry}")
            for index, name in enumerate(pro[registry]):
                rows.append({
                    "benchmark": "pro", "suite": suite, "registry": registry,
                    "registry_index": index, "classification_id": None,
                    "task_name": name, "base_task": name, "category": category,
                    "difficulty": None, "base_task_fold": folds[name],
                    "horizon_steps": HORIZONS[suite],
                })
        if len(plus[suite]) != len(plus_map[suite]):
            raise ValueError(f"PLUS registry/classification count mismatch: {suite}")
        for item in plus[suite]:
            index = item["id"] - 1
            if plus_map[suite][index] != item["name"]:
                raise ValueError(f"PLUS registry/classification order mismatch: {item}")
            candidates = [b for b in bases if item["name"].lower().startswith(b.lower())]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous PLUS base task: {item['name']}")
            name = candidates[0]
            rows.append({
                "benchmark": "plus", "suite": suite, "registry": suite,
                "registry_index": index, "classification_id": item["id"],
                "task_name": item["name"], "base_task": name,
                "category": item["category"], "difficulty": item["difficulty_level"],
                "base_task_fold": folds[name], "horizon_steps": HORIZONS[suite],
            })
    for row in rows:
        row["variant_id"] = digest([row["benchmark"], row["registry"], row["task_name"]])[:24]
    if len({r["variant_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate variant identity")
    result = {
        "schema": "moe_control.inventory.v1", "retrieved_utc_date": "2026-09-08",
        "sources": {
            "pro_commit": PRO_COMMIT, "plus_commit": PLUS_COMMIT,
            "pro_map_blob_sha1": blob,
            "plus_classification_sha256": hashlib.sha256(plus_bytes).hexdigest(),
            "plus_map_sha256": hashlib.sha256(plus_map_bytes).hexdigest(),
            "pro_map_url": f"https://github.com/Zxy-MLlab/LIBERO-PRO/blob/{PRO_COMMIT}/libero/libero/benchmark/libero_suite_task_map.py",
            "plus_classification_url": f"https://github.com/sylvestf/LIBERO-plus/blob/{PLUS_COMMIT}/libero/libero/benchmark/task_classification.json",
        },
        "status": "planned; task assets, init counts, horizons and simulator compatibility require preflight",
        "variants": rows,
    }
    write_json(args.inventory, result)
    return result


def plus_screen(rows):
    """Select 100 variants per suite/category, with known stratum probabilities."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["category"])].append(row)
    selected = {}
    for group_key, members in sorted(groups.items()):
        cells = collections.defaultdict(list)
        for row in members:
            cells[row["base_task"]].append(row)
        quotas = {key: min(10, len(value)) for key, value in cells.items()}
        # Nine base/category cells have fewer than ten variants. Redistribute
        # their deficits within the same suite/category without replacement.
        while sum(quotas.values()) < 100:
            candidates = [key for key in cells if quotas[key] < len(cells[key])]
            key = min(candidates, key=lambda key: (quotas[key], digest([group_key, key])))
            quotas[key] += 1
        for base, cell in cells.items():
            difficulties = collections.defaultdict(list)
            for row in cell:
                difficulties[str(row["difficulty"])].append(row)
            quota = quotas[base]
            allocation = {key: 1 for key in difficulties}
            if sum(allocation.values()) > quota:
                raise ValueError("Insufficient quota for difficulty strata")
            while sum(allocation.values()) < quota:
                candidates = [key for key in difficulties if allocation[key] < len(difficulties[key])]
                key = min(candidates, key=lambda key: (
                    allocation[key] / len(difficulties[key]), digest([group_key, base, key])
                ))
                allocation[key] += 1
            for difficulty, pool in difficulties.items():
                n = allocation[difficulty]
                ranked = sorted(pool, key=lambda row: digest(["sample-20260908", row["variant_id"]]))
                for row in ranked[:n]:
                    selected[row["variant_id"]] = n / len(pool)
    if len(selected) != 2800:
        raise ValueError(f"Expected 2800 PLUS screen variants, got {len(selected)}")
    return selected


def build_manifests(inventory, out_dir):
    variants = inventory["variants"]
    plus = [r for r in variants if r["benchmark"] == "plus"]
    pro = [r for r in variants if r["benchmark"] == "pro"]
    if len(plus) != 10030 or len(pro) != 200:
        raise ValueError("Inventory changed; revise the preregistered sample sizes")
    selected = plus_screen(plus)
    manifests = []
    for row in variants:
        pairs = [(0, 0)] if row["benchmark"] == "plus" else [
            (init, seed) for init in range(10) for seed in range(2)
        ]
        for init, seed in pairs:
            screen = (row["variant_id"] in selected) if row["benchmark"] == "plus" else (init < 5 and seed == 0)
            main_id = digest(["main-20260908", row["variant_id"], init, seed])[:24]
            manifests.append({
                **row, "main_id": main_id, "init_index": init,
                "policy_seed_index": seed,
                "noise_seed": int(digest(["noise", main_id])[:8], 16),
                "screen": int(screen),
                "screen_variant_inclusion_probability": selected.get(row["variant_id"], "") if row["benchmark"] == "plus" and screen else (1.0 if screen else ""),
                "status": "planned_pending_preflight",
            })
    if len(manifests) != 14030 or sum(r["screen"] for r in manifests) != 3800:
        raise ValueError("Incorrect cumulative sample counts")
    if len({r["main_id"] for r in manifests}) != len(manifests):
        raise ValueError("Duplicate main trajectory")
    with (out_dir / "collection_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifests[0]))
        writer.writeheader()
        writer.writerows(manifests)
    counts = {
        "pro_variants": len(pro), "plus_variants": len(plus),
        "screen_main": 3800, "expansion_main_cumulative": 14030,
        "pro_screen_main": 1000, "plus_screen_main": 2800,
        "pro_expansion_main": 4000, "plus_expansion_main": 10030,
        "plus_by_suite": dict(collections.Counter(r["suite"] for r in plus)),
        "plus_by_category": dict(collections.Counter(r["category"] for r in plus)),
        "plus_by_difficulty": dict(collections.Counter(str(r["difficulty"]) for r in plus)),
        "plus_screen_by_difficulty": dict(collections.Counter(str(r["difficulty"]) for r in plus if r["variant_id"] in selected)),
        "plus_category_suite": {
            suite: dict(collections.Counter(r["category"] for r in plus if r["suite"] == suite))
            for suite in SUITES
        },
        "full_horizon_main_queries": {
            "pro_expansion": sum(r["horizon_steps"] // 10 for r in manifests if r["benchmark"] == "pro"),
            "plus_expansion": sum(r["horizon_steps"] // 10 for r in manifests if r["benchmark"] == "plus"),
        },
        "source_inventory_sha256": digest(inventory),
    }
    write_json(out_dir / "sampling_counts.json", counts)
    return counts


def cost(n, alarm_fraction, main_q, branch_q, bytes_per_query, cap, args):
    alarm_states = n * alarm_fraction
    expanded_states = alarm_states if cap is None else min(alarm_states, cap)
    branches = expanded_states * (1 + 2 * args.replicates)
    queries = n * main_q + branches * branch_q
    size = args.storage_padding * (
        queries * bytes_per_query + alarm_states * args.snapshot_kib * KIB
        + n * 16 * KIB + branches * 8 * KIB
    )
    seconds = args.time_padding * (
        queries / args.qps
        + branches * args.restore_seconds / args.restore_workers
    )
    return {
        "main_trajectories": n, "alarm_states": alarm_states,
        "expanded_states": expanded_states, "branch_suffixes": branches,
        "total_queries": queries, "data_gib": size / GIB,
        "hours": seconds / 3600,
    }


def make_budgets(args, counts):
    rows = []
    for mode, bytes_q in (("compressed_planning", 48 * KIB), ("uncompressed_planning", 80 * KIB)):
        for alarm in (0.0, 0.1, 0.3, 0.6, 1.0):
            unit = cost(1, alarm, args.main_queries, args.branch_queries, bytes_q, None, args)
            disk_max = math.floor(args.data_gib / unit["data_gib"])
            time_max = math.floor(args.hours / unit["hours"])
            rows.append({
                "mode": mode, "alarm_fraction": alarm,
                "bytes_per_query": bytes_q, "disk_max_main": disk_max,
                "time_max_main": time_max, "joint_max_main": min(disk_max, time_max),
                "unit": unit,
            })
    phases = []
    for name, n, cap in (
        ("screen_all_alarms", counts["screen_main"], None),
        ("expansion_all_alarms", counts["expansion_main_cumulative"], None),
        ("expansion_300_per_benchmark", counts["expansion_main_cumulative"], 600),
    ):
        for mode, bytes_q in (("compressed_planning", 48 * KIB), ("uncompressed_planning", 80 * KIB)):
            for alarm in (0.1, 0.3, 0.6, 1.0):
                # Both benchmark pools exceed 300 at these alarm rates. The
                # shared 600 cap therefore equals the two benchmark caps.
                phases.append({
                    "phase": name, "mode": mode, "alarm_fraction": alarm,
                    **cost(n, alarm, args.main_queries, args.branch_queries, bytes_q, cap, args),
                })
    stress = cost(
        1, 1.0, 52, 52, 80 * KIB, None, args
    )
    per_benchmark = []
    for phase, benchmark, n, cap in (
        ("screen_all_alarms", "pro", 1000, None),
        ("screen_all_alarms", "plus", 2800, None),
        ("expansion_300_per_benchmark", "pro", 4000, 300),
        ("expansion_300_per_benchmark", "plus", 10030, 300),
    ):
        per_benchmark.append({
            "phase": phase, "benchmark": benchmark, "alarm_fraction": 0.3,
            **cost(n, 0.3, args.main_queries, args.branch_queries, 48 * KIB, cap, args),
        })
    horizon_main_q = sum(counts["full_horizon_main_queries"].values()) / 14030
    result = {
        "schema": "moe_control.budget.v1", "status": "estimates_not_rollout_measurements",
        "assumptions": {
            "data_gib": args.data_gib, "hours": args.hours, "aggregate_query_per_second": args.qps,
            "mean_main_queries": args.main_queries, "mean_branch_queries": args.branch_queries,
            "branch_replicates_per_treatment": args.replicates,
            "suffixes_per_expanded_state": 1 + 2 * args.replicates,
            "snapshot_kib_per_alarm": args.snapshot_kib,
            "storage_padding": args.storage_padding, "time_padding": args.time_padding,
            "restore_seconds_per_branch": args.restore_seconds, "restore_workers": args.restore_workers,
            "includes": "full routes plus effective weights, numeric state/action/noise, one alarm snapshot, metadata",
            "excludes": "hidden activations, per-step RGB/video, benchmark assets, setup/pilot, optional ablations",
        },
        "capacity": rows, "phases": phases, "per_benchmark_at_30_percent_alarm": per_benchmark,
        "full_horizon_main_capped_expansion": {
            "assumptions": "all mains reach their suite horizon; 30% alarm; 600 expanded states; suffix mean unchanged",
            **cost(14030, 0.3, horizon_main_q, args.branch_queries, 48 * KIB, 600, args),
        },
        "upper_cost_stress": {
            "assumptions": "all alarms; 52 queries for every main and every suffix; 80 KiB/query",
            "unit": stress, "disk_max_main": math.floor(args.data_gib / stress["data_gib"]),
            "time_max_main": math.floor(args.hours / stress["hours"]),
        },
    }
    write_json(args.out_dir / "budget_results.json", result)
    with (args.out_dir / "capacity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "unit"])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "unit"} for row in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=HERE / "design/benchmark_inventory.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "design")
    parser.add_argument("--prepare-inventory", action="store_true")
    parser.add_argument("--pro-map-api", type=Path)
    parser.add_argument("--plus-classification", type=Path)
    parser.add_argument("--plus-map", type=Path)
    parser.add_argument("--data-gib", type=float, default=25)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--qps", type=float, default=4)
    parser.add_argument("--main-queries", type=float, default=25)
    parser.add_argument("--branch-queries", type=float, default=15)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--snapshot-kib", type=float, default=512)
    parser.add_argument("--storage-padding", type=float, default=1.10)
    parser.add_argument("--time-padding", type=float, default=1.15)
    parser.add_argument("--restore-seconds", type=float, default=2)
    parser.add_argument("--restore-workers", type=int, default=8)
    args = parser.parse_args()
    positive = (args.data_gib, args.hours, args.qps, args.main_queries, args.branch_queries,
                args.snapshot_kib, args.storage_padding, args.time_padding, args.restore_workers)
    if (not all(math.isfinite(value) and value > 0 for value in positive)
            or args.replicates < 1 or not math.isfinite(args.restore_seconds)
            or args.restore_seconds < 0):
        parser.error("Budgets/rates must be positive; replicates >= 1; restore time >= 0")
    if args.prepare_inventory and not all((args.pro_map_api, args.plus_classification, args.plus_map)):
        parser.error("Inventory preparation requires all three downloaded source paths")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory = prepare_inventory(args) if args.prepare_inventory else read_json(args.inventory)
    counts = build_manifests(inventory, args.out_dir)
    budgets = make_budgets(args, counts)
    print(json.dumps({
        "screen_main": counts["screen_main"],
        "expansion_main_cumulative": counts["expansion_main_cumulative"],
        "capacity_at_30_percent_alarm": [
            row for row in budgets["capacity"] if row["alarm_fraction"] == 0.3
        ],
        "phases_at_30_percent_alarm": [
            row for row in budgets["phases"]
            if row["alarm_fraction"] == 0.3 and row["mode"] == "compressed_planning"
        ],
        "per_benchmark": budgets["per_benchmark_at_30_percent_alarm"],
        "full_horizon_main_capped_expansion": budgets["full_horizon_main_capped_expansion"],
    }, indent=2))


if __name__ == "__main__":
    main()
