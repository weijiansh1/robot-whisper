#!/usr/bin/env python3
"""Freeze fresh official Long initializations before collecting any outcomes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from collection_protocol import HERE, stable_id, verify_frozen_alarm
from collection_storage import atomic_json, digest
from control_bank import PROTOCOL as BANK_PROTOCOL, SETTINGS as BANK_SETTINGS, ARMS as BANK_ARMS
from native_long_runtime import BASE, ROOT, environment, inventory, verify_source
from prepare_native_long_v8 import RUNTIME
from v8_closed_loop import PROTOCOL, SETTINGS, limits


def prepare(args):
    import numpy as np
    from benchmarks.run_benchmarks import load_suite
    verify_frozen_alarm()
    if args.output.exists():
        raise ValueError("Cannot overwrite preregistration")
    old = json.loads((HERE / "design/native_long_modes_plan_20260909.json").read_text())
    for path, expected in old["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Previous frozen source changed: "+path)
    suite, tasks, rows = load_suite("libero_10", {}), [], inventory()
    for row in rows:
        initial = np.asarray(suite.get_task_init_states(row["registry_index"]))
        for index in range(10, 15):
            seed = int(stable_id(BANK_PROTOCOL, row["variant_id"], index, "main_noise")[:8], 16)
            tasks.append(dict(main_id=stable_id(BANK_PROTOCOL, row["variant_id"], index, seed),
                variant_id=row["variant_id"], variant=row, benchmark="native_long", category="Original",
                analysis_role="native", base_task=row["base_task"], noise_seed=seed, init_index=index,
                initial_state_sha256=hashlib.sha256(np.ascontiguousarray(initial[index]).tobytes()).hexdigest()))
    if {t["main_id"] for t in tasks} & {t["main_id"] for t in old["cohort"]}:
        raise ValueError("Fresh cohort overlaps previous mains")
    runtime = list(dict.fromkeys(list(RUNTIME)+["control_bank.py", "test_control_bank.py",
        "prepare_control_bank_mains.py", "run_control_bank_mains.py"]))
    sources = dict(old["source_sha256"])
    sources.update({str(HERE / name): digest(HERE / name) for name in runtime})
    tau, margins = limits()
    maximum = 2*52*len(tasks)
    plan = dict(protocol=PROTOCOL, stage="control_bank_fresh_mains", experiment="original_libero_long",
        settings=SETTINGS, arms=[], tasks=tasks, model="long", benchmark="native_long",
        original_source=verify_source(), allowed_gpus=[0,1,2,3], render_gpus=[0,3],
        replicas_per_gpu=8, workers_per_gpu=8, batch_size=1, hidden_capture=False, threshold_fitting=False,
        bank_protocol=BANK_PROTOCOL, bank_settings=BANK_SETTINGS, bank_arms=list(BANK_ARMS),
        selection="all 10 original Long tasks, official initializations 10..14, fixed hash-derived seeds",
        purpose="fresh outcome-blind validation cohort; complete all mains and C0 before new interventions",
        prior_init_indices=list(range(10)), fresh_init_indices=list(range(10,15)),
        runtime_sources=runtime, source_sha256=sources, thresholds=tau.tolist(), margins=margins.tolist(),
        new_main_coverage=50, maximum_model_queries=maximum,
        maximum_output_bytes=maximum*200*1024+50*20*1024**2, storage_quota_gib=4, disk_floor_gib=8)
    atomic_json(args.output, plan)
    print(json.dumps(dict(status="frozen", new_mains=len(tasks), jobs=100,
        plan_sha256=digest(args.output), bank_arms=len(BANK_ARMS))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        prepare(args)
    else:
        subprocess.run([str(BASE / "envs/libero/bin/python"), str(Path(__file__).resolve()), "--worker",
            "--output", str(args.output.resolve())], env=environment(0, initialize=True), cwd=ROOT, check=True)
