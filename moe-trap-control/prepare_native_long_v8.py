#!/usr/bin/env python3
"""Predeclare original Long coverage before observing any rollout outcome."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from adaptive_control import PARAMETERS, REFERENCE
from collection_protocol import HERE, stable_id, verify_frozen_alarm
from collection_storage import atomic_json, digest
from native_long_runtime import BASE, ROOT, environment, inventory, verify_source
from prepare_v8_closed_loop import RUNTIME as SELECTOR_RUNTIME
from v8_closed_loop import PROTOCOL, ARMS, SETTINGS, limits

RUNTIME = tuple(dict.fromkeys(SELECTOR_RUNTIME + ("native_long_runtime.py", "prepare_native_long_v8.py",
    "collect_native_long_v8.py", "run_native_long_v8.py")))


def prepare(args):
    import numpy as np
    from benchmarks.run_benchmarks import load_suite
    verify_frozen_alarm()
    source = verify_source()
    rows = inventory()
    suite, tasks = load_suite("libero_10", {}), []
    for row in rows:
        initial = np.asarray(suite.get_task_init_states(row["registry_index"]))
        for init in range(args.states_per_task):
            seed = int(stable_id("native_long_v8_20260909", row["variant_id"], init, "policy_seed")[:8], 16)
            main_id = stable_id("native_long_v8_20260909", row["variant_id"], init, seed)
            tasks.append(dict(main_id=main_id, variant_id=row["variant_id"], variant=row,
                benchmark="native_long", category="Original", analysis_role="native", base_task=row["base_task"],
                noise_seed=seed, init_index=init,
                initial_state_sha256=hashlib.sha256(np.ascontiguousarray(initial[init]).tobytes()).hexdigest()))
    thresholds, margins = limits()
    sources = [HERE / name for name in RUNTIME]+[PARAMETERS, REFERENCE,
        HERE.parent / "moe-v7-0905/method/intrinsic_guard_monitor.py",
        HERE.parent / "himoe-route-capture/route_noise_selector.py"]
    sources += [Path(row[key]) for row in rows for key in ("bddl_path", "init_file")]
    max_queries = len(tasks)*(2*52+5*52+4*15*12)
    plan = dict(protocol=PROTOCOL, experiment="original_libero_long", stage="native_long_pilot",
        model="long", benchmark="native_long", original_source=source, tasks=tasks, settings=SETTINGS,
        thresholds=thresholds.tolist(), margins=margins.tolist(), recovery_targets=(thresholds-margins).tolist(),
        arms=list(ARMS), methods=["v8_frozen"], allowed_gpus=[0, 1, 2, 3], render_gpus=[0, 3],
        replicas_per_gpu=8, workers_per_gpu=8, batch_size=1, hidden_capture=False, threshold_fitting=False,
        selection="All 10 original tasks, official init indices 0 through %d; hash-derived policy seeds; no outcome selection" % (args.states_per_task-1),
        evaluation_scope="All predeclared mains; no usable alarm retains native outcome for every arm; also report conditional alarm cohort",
        timing="Complete all native mains; exact full C0; intervene at first frozen v8 alarm q+1; chunk10; total520; settle10",
        execution="Frozen bounded loop: 16 candidates per active query, max12 queries, three complete populations below target before exit",
        source_sha256={str(path): digest(path) for path in sources}, runtime_sources=list(RUNTIME),
        maximum_model_queries=max_queries, maximum_output_bytes=max_queries*200*1024+len(tasks)*20*1024**2,
        storage_quota_gib=26, disk_floor_gib=8, new_main_coverage=len(tasks))
    if args.output.exists():
        raise ValueError("Refusing to overwrite predeclared cohort")
    atomic_json(args.output, plan)
    print(json.dumps(dict(plan=str(args.output), mains=len(tasks), tasks=10,
        maximum_queries=max_queries, maximum_gib=plan["maximum_output_bytes"]/1024**3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-per-task", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        prepare(args)
    else:
        env = environment(0, initialize=True)
        subprocess.run([str(BASE / "envs/libero/bin/python"), str(Path(__file__).resolve()), "--worker",
            "--states-per-task", str(args.states_per_task), "--output", str(args.output.resolve())],
            env=env, cwd=ROOT, check=True)
