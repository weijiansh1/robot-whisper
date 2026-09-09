#!/usr/bin/env python3
"""Collect fresh native mains and full C0 using the unchanged original runtime."""

import argparse
from pathlib import Path

import run_native_long_v8 as scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=14600)
    parser.add_argument("--job-timeout", type=float, default=3600)
    scheduler.ARMS = ()
    scheduler.run(parser.parse_args())
