#!/usr/bin/env python3
"""Use the validated bounded batch-one scheduler for native mode experiments."""

import argparse
import json
from pathlib import Path

import run_v8_closed_loop as scheduler
from mode_control import ARMS, PROTOCOL, SETTINGS
from native_long_runtime import ROOT, environment, verify_source


class NativeWorker(scheduler.PersistentWorker):
    def __init__(self, command, env, root, log):
        command = list(command)
        original = str(scheduler.HERE / "collect_v8_closed_loop.py")
        if command.count(original) != 1 or root != str(ROOT):
            raise ValueError("Unexpected native worker command")
        command[command.index(original)] = str(scheduler.HERE / "collect_native_long_modes.py")
        command.remove("--exact-noise-fastpath")
        super().__init__(command, env, root, log)


def native_environment(benchmark, render, backend):
    if benchmark != "native_long" or backend != "egl":
        raise ValueError("Only original Long is allowed")
    return environment(render)


def run(args):
    plan = json.loads(args.plan.read_text())
    if verify_source() != plan["original_source"]:
        raise ValueError("Original LIBERO identity changed")
    environment(0, initialize=True)
    scheduler.PROTOCOL, scheduler.ARMS, scheduler.SETTINGS = PROTOCOL, ARMS, SETTINGS
    scheduler.ROOTS = {"native_long": ROOT}
    scheduler.environment, scheduler.PersistentWorker = native_environment, NativeWorker
    try:
        scheduler.run(args)
    finally:
        if verify_source() != plan["original_source"]:
            raise ValueError("Original LIBERO changed during collection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=14200)
    parser.add_argument("--job-timeout", type=float, default=3600)
    run(parser.parse_args())
