#!/usr/bin/env python3
"""Run the fixed outcome-model comparison on private batch-one GPU replicas."""

import argparse
import json
from pathlib import Path
import shutil

import run_v8_closed_loop as scheduler
from moe_outcome_models import PROTOCOL, SETTINGS, ARMS
from native_long_runtime import ROOT, environment, verify_source


class NativeWorker(scheduler.PersistentWorker):
    def __init__(self, command, env, root, log):
        command = list(command)
        original = str(scheduler.HERE/"collect_v8_closed_loop.py")
        if command.count(original) != 1 or root != str(ROOT):
            raise ValueError("Unexpected original simulator worker")
        command[command.index(original)] = str(scheduler.HERE/"collect_moe_outcomes.py")
        command.remove("--exact-noise-fastpath")
        super().__init__(command,env,root,log)


def native_environment(benchmark, render, backend):
    if benchmark != "native_long" or backend != "egl":
        raise ValueError("Original Long EGL required")
    return environment(render)


def run(args):
    plan = json.loads(args.plan.read_text())
    if verify_source() != plan["original_source"]:
        raise ValueError("Original simulator changed")
    environment(0,initialize=True)
    scheduler.PROTOCOL,scheduler.SETTINGS,scheduler.ARMS = PROTOCOL,SETTINGS,ARMS
    scheduler.ROOTS = {"native_long":ROOT}
    scheduler.environment,scheduler.PersistentWorker = native_environment,NativeWorker
    try:
        scheduler.run(args)
    finally:
        if args.output.exists():
            shutil.copy2(plan["outcome_model_path"],args.output/"model.json")
        if verify_source() != plan["original_source"]:
            raise ValueError("Original simulator changed during run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--port-base",type=int,default=15800)
    parser.add_argument("--job-timeout",type=float,default=3600)
    run(parser.parse_args())
