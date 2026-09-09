"""Collect a state-by-noise grid for the runtime HB5/d0 vector probe.

The runner is deliberately conservative: a suite store is either absent or
already complete.  It never deletes or overwrites a partial capture.  Each
checkpoint is loaded once, then all task/state clients for that suite append to
one v3 activation-flow store with globally unique query IDs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import signal
import subprocess
import time
from typing import Iterable

import numpy as np

from himoe_activation_store import FORMAT, ZarrActivationFlowReader


ROOT = Path(__file__).resolve().parent
BRIDGE = Path("/home/jovyan/.cache/himoe-libero-bridge")
WRIST_FIX = "/home/jovyan/work/himoe-libero-wrist-fix/src"
DEFAULT_STATES = (1, 5, 11, 18, 25, 32, 40, 48)
DEFAULT_OUT = ROOT / "runs" / "runtime-vector-grid-v3"
QUERY_TASK_STRIDE = 10_000
QUERY_STATE_STRIDE = 100
SERVER_READY_TIMEOUT_S = 600
SERVER_STOP_TIMEOUT_S = 180


@dataclass(frozen=True)
class TaskSpec:
    label: str
    suite: str
    benchmark: str
    task_id: int


TASKS = (
    TaskSpec("goal-middle", "goal", "libero_goal", 0),
    TaskSpec("goal-top", "goal", "libero_goal", 3),
    TaskSpec("long-t08", "long", "libero_10", 8),
    TaskSpec("spatial-ramekin", "spatial", "libero_spatial", 5),
    TaskSpec("spatial-stove", "spatial", "libero_spatial", 7),
)

CHECKPOINT = {
    "goal": BRIDGE / "checkpoints" / "HiMoE-VLA-Libero-Goal",
    "long": BRIDGE / "checkpoints" / "HiMoE-VLA-Libero-10",
    "spatial": BRIDGE / "checkpoints" / "HiMoE-VLA-Libero-Spatial",
}


def parse_states(value: str) -> tuple[int, ...]:
    states = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not states:
        raise argparse.ArgumentTypeError("at least one init state is required")
    if len(set(states)) != len(states):
        raise argparse.ArgumentTypeError("init states must be unique")
    if min(states) < 0 or max(states) >= 50:
        raise argparse.ArgumentTypeError("LIBERO init states must be in [0, 49]")
    return states


def parse_suites(value: str) -> tuple[str, ...]:
    suites = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = tuple(CHECKPOINT)
    if not suites or len(set(suites)) != len(suites):
        raise argparse.ArgumentTypeError("suites must be a non-empty unique list")
    invalid = sorted(set(suites) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported suites: {invalid}")
    return suites


def query_base(task_axis: int, state_axis: int) -> int:
    if task_axis < 0 or state_axis < 0:
        raise ValueError("task and state axes must be non-negative")
    return task_axis * QUERY_TASK_STRIDE + state_axis * QUERY_STATE_STRIDE


def expected_identity(
    task_axes: Iterable[int], n_states: int, candidates: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_states <= 0 or candidates <= 0:
        raise ValueError("n_states and candidates must be positive")
    queries = []
    candidate_ids = []
    for task_axis in task_axes:
        for state_axis in range(n_states):
            queries.extend([query_base(task_axis, state_axis)] * candidates)
            candidate_ids.extend(range(candidates))
    return np.asarray(queries, dtype=np.int64), np.asarray(
        candidate_ids, dtype=np.int64
    )


def server_env() -> dict[str, str]:
    env = dict(os.environ)
    env["MOEVLA_DATA_HOME"] = str(BRIDGE / "moevla-data")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def client_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "LD_LIBRARY_PATH": str(
                BRIDGE / "system-libs" / "usr" / "lib" / "x86_64-linux-gnu"
            ),
            "PYTHONPATH": ":".join(
                (
                    WRIST_FIX,
                    "/home/jovyan/work/.rs141-audit",
                    "/home/jovyan/work/.paper-eval-overlay",
                    str(BRIDGE / "upstream" / "LIBERO"),
                    str(
                        BRIDGE
                        / "upstream"
                        / "HiMoE-VLA"
                        / "packages"
                        / "openpi-client"
                        / "src"
                    ),
                )
            ),
        }
    )
    return env


def _is_nonempty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def _wait_for_server(process: subprocess.Popen, log: Path) -> None:
    deadline = time.time() + SERVER_READY_TIMEOUT_S
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with rc={process.returncode}; see {log}")
        output = log.read_text(errors="replace") if log.is_file() else ""
        if "serving on ws" in output:
            return
        if "Traceback" in output:
            raise RuntimeError(f"server startup failed; see {log}")
        time.sleep(2)
    raise TimeoutError(f"server did not become ready; see {log}")


def _stop_server(process: subprocess.Popen, server_dir: Path) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    deadline = time.time() + SERVER_STOP_TIMEOUT_S
    summary = server_dir / "capture_summary.json"
    while time.time() < deadline:
        if process.poll() is not None and summary.is_file():
            return
        time.sleep(1)
    if process.poll() is None:
        process.kill()
    raise RuntimeError(f"server did not flush a capture summary in {server_dir}")


def validate_suite_store(
    server_dir: Path,
    task_axes: tuple[int, ...],
    n_states: int,
    candidates: int,
) -> dict:
    summary_path = server_dir / "capture_summary.json"
    store_path = server_dir / "activation_flow.zarr"
    if not summary_path.is_file() or not store_path.is_dir():
        raise ValueError(f"suite capture is incomplete: {server_dir}")
    summary = json.loads(summary_path.read_text())
    reader = ZarrActivationFlowReader(str(store_path))
    expected_query, expected_candidate = expected_identity(
        task_axes, n_states, candidates
    )
    actual_query = np.asarray(reader["query_id"][:], dtype=np.int64)
    actual_candidate = np.asarray(reader["candidate_id"][:], dtype=np.int64)
    problems = []
    if reader.meta.get("format") != FORMAT:
        problems.append(f"format={reader.meta.get('format')!r}")
    if int(summary.get("queries", -1)) != len(expected_query):
        problems.append(f"summary queries={summary.get('queries')!r}")
    if not np.array_equal(actual_query, expected_query):
        problems.append("query_id axis mismatch")
    if not np.array_equal(actual_candidate, expected_candidate):
        problems.append("candidate_id axis mismatch")
    reconstruction_error = float(
        np.max(reader["hb_probe_routed_reconstruction_max_abs_error"][:])
    )
    if reconstruction_error > 1e-5:
        problems.append(f"routed reconstruction error={reconstruction_error:.3e}")
    if problems:
        raise ValueError(f"invalid suite capture {server_dir}: {problems}")
    return {
        "format": FORMAT,
        "queries": len(expected_query),
        "task_axes": list(task_axes),
        "states": n_states,
        "candidates": candidates,
        "max_probe_routed_reconstruction_error": reconstruction_error,
    }


def _server_command(suite: str, port: int, server_dir: Path, threads: int) -> list[str]:
    return [
        str(BRIDGE / "envs" / "model" / "bin" / "python"),
        "-u",
        str(ROOT / "serve_activation_flow_trace.py"),
        "--port",
        str(port),
        "--gpu",
        "cpu",
        "--suite",
        suite,
        "--checkpoint-dir",
        str(CHECKPOINT[suite]),
        "--upstream-root",
        str(BRIDGE / "upstream" / "HiMoE-VLA"),
        "--libero-wrist-layout",
        "checkpoint-right",
        "--out",
        str(server_dir),
        "--threads",
        str(threads),
    ]


def _client_command(
    task: TaskSpec,
    state: int,
    task_axis: int,
    state_axis: int,
    candidates: int,
    noise_seed: int,
    port: int,
    client_dir: Path,
) -> list[str]:
    return [
        str(BRIDGE / "envs" / "libero" / "bin" / "python"),
        "-u",
        str(ROOT / "rollout_flow_lead.py"),
        "--port",
        str(port),
        "--benchmark",
        task.benchmark,
        "--task-id",
        str(task.task_id),
        "--init-state-id",
        str(state),
        "--n-episodes",
        "1",
        "--n-candidates",
        str(candidates),
        "--noise-seed-base",
        str(noise_seed),
        "--query-base",
        str(query_base(task_axis, state_axis)),
        "--max-steps",
        "1",
        "--replan-steps",
        "1",
        "--libero-root",
        str(BRIDGE / "upstream" / "LIBERO"),
        "--label",
        f"runtime-vector-v3-{task.label}-s{state:02d}",
        "--out",
        str(client_dir),
        "--inference-timeout",
        "600",
    ]


def capture_suite(
    suite: str,
    out: Path,
    states: tuple[int, ...],
    candidates: int,
    noise_seed: int,
    port: int,
    threads: int,
) -> dict:
    selected = tuple(
        (axis, task) for axis, task in enumerate(TASKS) if task.suite == suite
    )
    task_axes = tuple(axis for axis, _task in selected)
    server_dir = out / "server" / suite
    if _is_nonempty(server_dir):
        result = validate_suite_store(server_dir, task_axes, len(states), candidates)
        result["status"] = "skipped_complete"
        return result

    stale_clients = [
        out / "client" / task.label
        for _axis, task in selected
        if _is_nonempty(out / "client" / task.label)
    ]
    if stale_clients:
        raise FileExistsError(
            f"client outputs exist without a complete suite store: {stale_clients}"
        )

    server_dir.mkdir(parents=True, exist_ok=False)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / f"server-{suite}.log"
    with server_log.open("w") as stream:
        server = subprocess.Popen(
            _server_command(suite, port, server_dir, threads),
            cwd=ROOT,
            env=server_env(),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    served = False
    try:
        _wait_for_server(server, server_log)
        served = True
        for task_axis, task in selected:
            for state_axis, state in enumerate(states):
                client_dir = out / "client" / task.label / f"state-{state:02d}"
                client_dir.mkdir(parents=True, exist_ok=False)
                client_log = log_dir / f"client-{task.label}-s{state:02d}.log"
                with client_log.open("w") as stream:
                    completed = subprocess.run(
                        _client_command(
                            task,
                            state,
                            task_axis,
                            state_axis,
                            candidates,
                            noise_seed,
                            port,
                            client_dir,
                        ),
                        cwd=ROOT,
                        env=client_env(),
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"client failed rc={completed.returncode}; see {client_log}"
                    )
    finally:
        if served:
            _stop_server(server, server_dir)
        elif server.poll() is None:
            server.terminate()
            server.wait(timeout=30)

    result = validate_suite_store(server_dir, task_axes, len(states), candidates)
    result["status"] = "captured"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--states", type=parse_states, default=DEFAULT_STATES)
    parser.add_argument(
        "--suites", type=parse_suites, default=("goal", "spatial", "long")
    )
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--noise-seed", type=int, default=8000)
    parser.add_argument("--port-base", type=int, default=8810)
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidates < 2:
        raise ValueError("--candidates must be at least two")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for offset, suite in enumerate(args.suites):
        print(f"[{suite}] starting", flush=True)
        results[suite] = capture_suite(
            suite,
            args.out,
            args.states,
            args.candidates,
            args.noise_seed,
            args.port_base + offset,
            args.threads,
        )
        print(f"[{suite}] {results[suite]['status']}", flush=True)
    manifest = {
        "format": "himoe_runtime_vector_grid_v1",
        "activation_store_format": FORMAT,
        "states": list(args.states),
        "candidates": args.candidates,
        "noise_seed": args.noise_seed,
        "suites": list(args.suites),
        "tasks": [task.__dict__ for task in TASKS if task.suite in args.suites],
        "results": results,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
