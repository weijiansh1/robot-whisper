#!/usr/bin/env python3
"""Run resumable timeout continuations suite-by-suite on one GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
BRIDGE = Path("/home/jovyan/.cache/himoe-libero-bridge")
BRIDGE_SRC = WORKSPACE / "himoe-libero-wrist-fix/src"
MODEL_PYTHON = BRIDGE / "envs/model/bin/python"
LIBERO_PYTHON = BRIDGE / "envs/libero/bin/python"
LIBERO_ROOT = BRIDGE / "upstream/LIBERO"
HIMOE_ROOT = BRIDGE / "upstream/HiMoE-VLA"
DEFAULT_OUTPUT = HERE / "results/timeout_extension_plus10"
CHECKPOINTS = {
    "goal": BRIDGE / "checkpoints/HiMoE-VLA-Libero-Goal",
    "spatial": BRIDGE / "checkpoints/HiMoE-VLA-Libero-Spatial",
    "object": BRIDGE / "checkpoints/HiMoE-VLA-Libero-Object",
    "long": BRIDGE / "checkpoints/HiMoE-VLA-Libero-10",
}


def server_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(BRIDGE_SRC)
    environment["MOEVLA_DATA_HOME"] = str(BRIDGE / "moevla-data")
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    return environment


def client_environment(egl_device: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["MUJOCO_GL"] = "egl"
    environment["PYOPENGL_PLATFORM"] = "egl"
    environment["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
    environment["LD_LIBRARY_PATH"] = "%s:/usr/lib/x86_64-linux-gnu" % (
        BRIDGE / "system-libs/usr/lib/x86_64-linux-gnu"
    )
    environment["PYTHONPATH"] = ":".join(
        [
            str(BRIDGE_SRC),
            "/home/jovyan/work/.rs141-audit",
            "/home/jovyan/work/.paper-eval-overlay",
            str(LIBERO_ROOT),
            str(HIMOE_ROOT / "packages/openpi-client/src"),
        ]
    )
    return environment


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def wait_for_server(process: subprocess.Popen, log: Path, timeout: int = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}; see {log}")
        content = log.read_text(errors="replace") if log.exists() else ""
        if "Policy server ready" in content:
            return
        if "Traceback" in content or "CUDA out of memory" in content:
            raise RuntimeError(f"server failed during startup; see {log}")
        time.sleep(3)
    raise TimeoutError(f"server was not ready in {timeout}s; see {log}")


def stop_owned_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def counts(manifest: Path) -> dict[str, int]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        suite: sum(row["server_suite"] == suite for row in rows)
        for suite in CHECKPOINTS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--egl-device", type=int, default=5)
    parser.add_argument("--port", type=int, default=8995)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--suites", default="long,spatial,goal,object", help="ordered suite list"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = output / "source_failures.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"prepare the source manifest first: {manifest}")
    if not port_is_free(args.port):
        raise RuntimeError(f"port {args.port} is already in use")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    planned = counts(manifest)
    (output / "logs").mkdir(parents=True, exist_ok=True)

    for suite in [item.strip() for item in args.suites.split(",") if item.strip()]:
        if suite not in CHECKPOINTS:
            raise ValueError(f"unknown suite {suite}")
        if planned[suite] == 0:
            continue
        server_log = output / "logs" / f"server_{suite}.log"
        with server_log.open("a", encoding="utf-8") as log_handle:
            server = subprocess.Popen(
                [
                    str(MODEL_PYTHON),
                    "-u",
                    "-m",
                    "himoe_libero_bridge.cli",
                    "serve",
                    "--backend",
                    "himoe",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--gpu",
                    args.gpu,
                    "--suite",
                    suite,
                    "--checkpoint-dir",
                    str(CHECKPOINTS[suite]),
                    "--upstream-root",
                    str(HIMOE_ROOT),
                    "--libero-wrist-layout",
                    "paper-right",
                ],
                cwd=str(WORKSPACE),
                env=server_environment(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        clients: list[tuple[subprocess.Popen, object]] = []
        try:
            wait_for_server(server, server_log)
            print(
                json.dumps(
                    {
                        "event": "server_ready",
                        "suite": suite,
                        "gpu": args.gpu,
                        "cases": planned[suite],
                    }
                ),
                flush=True,
            )
            for shard in range(args.workers):
                client_log = output / "logs" / f"client_{suite}_shard{shard}.log"
                handle = client_log.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    [
                        str(LIBERO_PYTHON),
                        "-u",
                        str(HERE / "continue_timeout_failures.py"),
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(output),
                        "--suite",
                        suite,
                        "--port",
                        str(args.port),
                        "--libero-root",
                        str(LIBERO_ROOT),
                        "--extra-queries",
                        "10",
                        "--shard-index",
                        str(shard),
                        "--shard-count",
                        str(args.workers),
                    ],
                    cwd=str(WORKSPACE),
                    env=client_environment(args.egl_device),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                clients.append((process, handle))
            failures = []
            for process, handle in clients:
                return_code = process.wait()
                handle.close()
                if return_code != 0:
                    failures.append(return_code)
            if failures:
                raise RuntimeError(
                    f"{suite} client shards failed with return codes {failures}"
                )
            print(
                json.dumps({"event": "suite_complete", "suite": suite}), flush=True
            )
        finally:
            for process, handle in clients:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                if not handle.closed:
                    handle.close()
            stop_owned_server(server)
    print(json.dumps({"event": "pipeline_complete"}), flush=True)


if __name__ == "__main__":
    main()
