#!/usr/bin/env python3
"""Register v8 intervention cases by native alarms, without outcome selection."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

from collection_protocol import HERE, PARAMETERS_SHA256, stable_id
from collection_storage import records, atomic_json, digest
from collection_routes import PROBS_KEY
from v8_feature_control import PROTOCOL, ARMS, V8Monitor


def run(args):
    audit_path = HERE / ("design/%s_audit_20260908.json" % args.source)
    alarms_path = HERE / ("design/%s_alarm_comparison_20260908/first_alarms.csv" % args.source)
    audited = json.loads(audit_path.read_text())
    if audited["status"] != "passed":
        raise ValueError("Native corpus needs a passed independent audit")
    with alarms_path.open() as stream:
        table = {r["main_id"]: r for r in csv.DictReader(stream)}
    candidates, inventory = [], []
    for item in audited["tasks"]:
        entry = table[item["main_id"]]
        first = int(entry["v8_frozen"])
        if first < 0 or first + 1 >= item["main_queries"]:
            continue
        directory = Path(item["directory"])
        monitor, alarm = V8Monitor(), None
        for row in records(directory / "main"):
            current = monitor.update(row[PROBS_KEY])
            if current["query"] == first:
                alarm = current
        if monitor.first_alarm != first or monitor.v7.first_alarm_query != int(entry["v7_frozen"]):
            raise ValueError("Online v8 differs from the frozen native replay")
        heads = [name for name, hit in (("freeze", alarm["freeze_alarm"]),
            ("turbulence", alarm["turbulence_alarm"]),
            ("balance", alarm["v8_head_first"][0] >= 0), ("curvature", alarm["v8_head_first"][1] >= 0)) if hit]
        original = json.loads((directory / "result.json").read_text())
        task = {key: item[key] for key in ("main_id", "variant_id", "benchmark", "category", "analysis_role")}
        task.update(parent_directory=str(directory), parent_queries=item["main_queries"],
            parent_commit_sha256=digest(directory / "main_complete.json"),
            parent_manifest_sha256=digest(directory / "main/manifest.json"),
            noise_seed=original["seed"], init_index=original["init_index"],
            first_alarm=int(entry["knn20"]), first_v8_alarm=first, active_heads=heads)
        task["events"] = [dict(event_id=stable_id(PROTOCOL, task["main_id"], first + 1),
            start_query=first + 1, alarm_query=first, timing="after_v8_alarm", deployable=True)]
        candidates.append(task)
        inventory.append(dict(main_id=task["main_id"], benchmark=task["benchmark"], active_heads=heads,
                              first_v8_alarm=first, all_new_head_first=monitor.first))
    # Select rare head strata first, then a stable hash, never native success.
    counts = Counter((t["benchmark"], tuple(t["active_heads"])) for t in candidates)
    candidates.sort(key=lambda t: (counts[t["benchmark"], tuple(t["active_heads"])],
                                  stable_id(PROTOCOL, "selection", t["main_id"])))
    tasks = candidates[:args.limit] if args.limit else candidates
    arms = list(ARMS)
    bound = sum(t["parent_queries"] * 200000 + args.replicates * len(arms) * (
        (52 - t["first_v8_alarm"] - 1) * 200000 + 5 * 600000) for t in tasks)
    if shutil.disk_usage(HERE).free - bound < 8 * 1024**3:
        raise ValueError("Conservative plan bound exceeds disk floor")
    source_files = [audit_path, alarms_path, HERE / "v8_feature_control.py", Path(__file__),
                    HERE / "design/frozen_alarm_comparison_20260908/profiles/parameters.json"]
    plan = dict(protocol=PROTOCOL, stage="exploratory_mechanism_test", model="long", tasks=tasks,
        arms=arms, arms_registry=ARMS, replicates=args.replicates, allowed_gpus=[0, 1, 2, 3],
        render_gpus=[0, 3], replicas_per_gpu=args.replicas, workers_per_gpu=args.replicas,
        inventory=inventory, frozen_parameters_sha256=PARAMETERS_SHA256,
        source_sha256={str(p.resolve()): digest(p) for p in source_files},
        selection="eligible native v8 alarm; rare benchmark/head strata first, then fixed hash; no outcome selection",
        timing="first frozen v8 union alarm q+1 after executing native q; completed main and complete exact C0 first",
        control="same-observation/noise native shadow, bounded additive gate logits, recomputed actual top4/weights",
        random_null="expert permutation of combined bias with the same per-row norm on that arm's own observation",
        recovery="5 queries then native; chunk10 throughout; full remaining 520-step budget",
        measurements="shadow/native/effective routing, independent dispatched top4, actions, unforced post-release features, task success",
        paired_random_streams="policy by relative query; environment by absolute action step; arm independent",
        hidden_capture=False, threshold_fitting=False, batch_size=1, new_main_coverage=0,
        maximum_output_bytes=bound, storage_quota_gib=12, disk_floor_gib=8)
    if args.output.exists():
        raise ValueError("Never replace a frozen plan")
    atomic_json(args.output, plan)
    print(json.dumps(dict(plan=str(args.output), parents=len(tasks), branches=len(tasks)*len(arms)*args.replicates,
        head_strata={str(k): v for k, v in counts.items()}, maximum_gib=bound/1024**3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="experiment_long_batch1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--replicas", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
