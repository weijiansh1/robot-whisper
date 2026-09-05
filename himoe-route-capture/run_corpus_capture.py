"""Drive the LIBERO-30 routing-corpus capture, one task at a time.

One server per task, not one per suite: ``serve_with_recorder.py`` takes ``--out``
once at startup and writes a single flat zarr, so a per-task directory requires a
per-task server.  That costs a 90-120 s checkpoint load per task, which is the
price of not having to slice 10 tasks back out of one store afterwards.

Run one instance per MIG slice with disjoint ``--tasks``; they never share a port
or an output directory.  Resumable: any task whose meta.json says ``complete`` is
skipped, so a killed run is restarted with the same command line.

Runs in any python with the bridge importable -- it only orchestrates
subprocesses, the model env serves and the LIBERO env rolls out.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import time

import corpus_layout as cl

BRIDGE = pathlib.Path("/home/jovyan/.cache/himoe-libero-bridge")
WRIST_FIX = "/home/jovyan/work/himoe-libero-wrist-fix/src"
ROOT = pathlib.Path(__file__).resolve().parent

CHECKPOINT_DIR = {
    "goal": BRIDGE / "checkpoints/HiMoE-VLA-Libero-Goal",
    "spatial": BRIDGE / "checkpoints/HiMoE-VLA-Libero-Spatial",
    "object": BRIDGE / "checkpoints/HiMoE-VLA-Libero-Object",
    "long": BRIDGE / "checkpoints/HiMoE-VLA-Libero-10",
}
#: 2026-08-15: was "checkpoint-right".  The paper (example.tex:591, unchanged in
#: v1 and v2) says the single arm maps to the right-arm channel while the left is
#: "zero-padded **with masks**" -- i.e. left mask false, which is `paper-right`.
#: `checkpoint-right` (both masks true) was picked on 2026-08-03 from a 20-episode
#: comparison, too small to separate saturated tasks.  Re-running all four suites
#: at 500 episodes under `paper-right`: Goal 98.0%, Object 99.0% (both now
#: statistically compatible with the paper), Spatial 95.4%, Long 92.4% -- every
#: suite improved, Long by +27 episodes.  See himoe-vla-libero-alignment-2026-08-15.md.
WRIST_LAYOUT = "paper-right"
SERVER_READY_TIMEOUT_S = 600
FLUSH_TIMEOUT_S = 180


def mig_profile(uuid: str) -> str:
    """Resolve a MIG UUID to its profile name; the two slices are not bit-identical."""
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                 check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    for line in listing.splitlines():
        if uuid in line and "MIG" in line:
            return line.strip().split()[1]
    return "unknown"


def server_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = WRIST_FIX
    env["MOEVLA_DATA_HOME"] = str(BRIDGE / "moevla-data")
    env.pop("CUDA_VISIBLE_DEVICES", None)  # --gpu sets it inside; pre-restricting hangs CUDA init
    return env


def client_env(libero_config_root=None) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    # CORPUS_MUJOCO_GL=osmesa avoids the sporadic EGL client aborts seen with
    # many concurrent clients (2026-09-03); default stays egl.
    gl = os.environ.get("CORPUS_MUJOCO_GL", "egl")
    env["MUJOCO_GL"] = gl
    env["PYOPENGL_PLATFORM"] = gl
    env["MUJOCO_EGL_DEVICE_ID"] = "0"
    if libero_config_root is not None:
        # LIBERO rewrites config.yaml every time it constructs an environment.
        # A shared path lets concurrent clients observe the file between its
        # truncate and write calls, so isolate it per captured task.
        env["LIBERO_CONFIG_PATH"] = str(libero_config_root)
    env["LD_LIBRARY_PATH"] = "%s:/usr/lib/x86_64-linux-gnu" % (
        BRIDGE / "system-libs/usr/lib/x86_64-linux-gnu")
    env["PYTHONPATH"] = ":".join([
        WRIST_FIX,
        "/home/jovyan/work/.rs141-audit",
        "/home/jovyan/work/.paper-eval-overlay",
        str(BRIDGE / "upstream/LIBERO"),
        str(BRIDGE / "upstream/HiMoE-VLA/packages/openpi-client/src"),
    ])
    return env


def wait_for_server(log_path: pathlib.Path, process, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server exited early (rc=%s); see %s" % (process.returncode, log_path))
        text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        if "serving on ws" in text:
            return
        if "Traceback" in text:
            raise RuntimeError("server failed to start; see %s" % log_path)
        time.sleep(3)
    raise RuntimeError("server not ready after %d s; see %s" % (timeout_s, log_path))


def stop_server(process, out_dir: pathlib.Path, served: bool = True) -> None:
    """SIGTERM, never a plain kill: the writer flushes in the signal handler.

    ``served=False`` means the server never reached the serving state -- it died
    during startup, so there is no capture to flush and nothing to wait for.
    Waiting anyway burns FLUSH_TIMEOUT_S and then raises a bogus "did not flush"
    that hides the real startup error (an OOM on a busy MIG slice reports itself
    as an NVML assert, which is confusing enough already).
    """
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    summary = out_dir / "capture_summary.json"
    deadline = time.time() + FLUSH_TIMEOUT_S
    while time.time() < deadline:
        if process.poll() is not None and (summary.is_file() or not served):
            return
        time.sleep(2)
    process.kill()
    if served:
        raise RuntimeError("server did not flush within %d s: %s" % (FLUSH_TIMEOUT_S, out_dir))


def task_paths(args, task):
    """Where this task's run lives, plus its two logs.

    Hub mode writes straight into ``cache/<model>/<benchmark>/<task>/<run_id>/`` so
    there is no migration step afterwards.  The hub has no cross-run level, so the
    logs go inside the run directory rather than into a shared ``_logs/``.
    """
    if args.run_id:
        path = cl.hub_task_dir(task["name"], task["benchmark"], args.run_id,
                               args.hub_root, args.model)
        return path, path / "logs" / "server.log", path / "logs" / "client.log"
    path = cl.task_dir(args.root, task["benchmark"], task["task_id"], task["name"])
    srv, cli = cl.log_paths(args.root, task["benchmark"], task["task_id"])
    return path, srv, cli


def capture_task(args, task, port: int) -> dict:
    benchmark, suite = task["benchmark"], task["suite"]
    task_path, srv_log, cli_log = task_paths(args, task)
    label = "%s/%s" % (benchmark, task["name"]) if args.run_id else task_path.name
    if cl.is_complete(task_path) and not args.force:
        print("[skip] %s already complete" % label, flush=True)
        return {"task": label, "skipped": True}

    # There is no resume: the client always starts at episode 0 and the writers
    # open their stores with overwrite=True.  Restarting a long run therefore
    # discards whatever it had already written, and for a 50x32 capture that is
    # hours of GPU.  Say so rather than doing it silently.
    try:
        prior = cl.load_summaries(task_path)
    except (OSError, ValueError):
        prior = []   # nothing written yet, or half-written; either way not a restart
    if prior:
        print("[warn] %s already holds %d episode(s); this run restarts from 0 "
              "and overwrites them (no resume support)" % (label, len(prior)),
              flush=True)

    srv_log.parent.mkdir(parents=True, exist_ok=True)
    out_server, out_client = cl.server_dir(task_path), cl.client_dir(task_path)
    out_server.mkdir(parents=True, exist_ok=True)
    out_client.mkdir(parents=True, exist_ok=True)

    started = time.time()
    cl.write_task_meta(
        task_path, status=cl.STATUS_RUNNING, suite=suite, benchmark=benchmark,
        task_id=task["task_id"], task_name=task["name"], prompt=task.get("language", ""),
        bddl_file=task.get("bddl_file", ""), mig_uuid=args.gpu,
        mig_profile=mig_profile(args.gpu), port=port,
        wrist_layout=WRIST_LAYOUT, episodes_requested=args.episodes,
        scenes=args.scenes, draws_per_scene=args.draws,
        scene_ids=args.scene_ids,
        store_hidden=bool(args.store_hidden),
        max_steps=cl.MAX_STEPS[suite],
        run_id=args.run_id, hub_model=(args.model if args.run_id else None),
    )

    server = subprocess.Popen(
        [str(BRIDGE / "envs/model/bin/python"), "-u", str(ROOT / "serve_with_recorder.py"),
         "--port", str(port), "--gpu", args.gpu, "--suite", suite,
         "--checkpoint-dir", str(CHECKPOINT_DIR[suite]),
         "--upstream-root", str(BRIDGE / "upstream/HiMoE-VLA"),
         "--libero-wrist-layout", WRIST_LAYOUT, "--store-full-probs",
         *(["--pin", args.pin] if args.pin else []),
         *(["--pin-regime", args.pin_regime] if args.pin_regime else []),
         *(["--trunc-rounds", str(args.trunc_rounds), "--no-route-capture"]
           if args.trunc_rounds is not None else []),
         *(["--store-hidden"] if args.store_hidden else []),
         "--out", str(out_server)],
        stdout=srv_log.open("w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        cwd=str(ROOT), env=server_env(), start_new_session=True,
    )
    served = False
    try:
        wait_for_server(srv_log, server, SERVER_READY_TIMEOUT_S)
        served = True
        print("[%s] server ready on %d (%s)" % (label, port, mig_profile(args.gpu)),
              flush=True)
        init_states = (args.scene_ids if args.scene_ids
                       else ",".join(str(i) for i in range(args.scenes)))
        client = subprocess.Popen(
            [str(BRIDGE / "envs/libero/bin/python"), "-u", str(ROOT / "rollout_with_routes.py"),
             "--port", str(port), "--task-id", str(task["task_id"]), "--benchmark", benchmark,
             "--init-state-ids", init_states, "--repeats", str(args.draws),
             "--noise-seed-base", str(args.noise_seed_base),
             "--max-steps", str(cl.MAX_STEPS[suite]), "--no-routing-capture",
             "--inference-timeout", str(args.inference_timeout),
             "--label", (args.run_id or task_path.name), "--out", str(out_client),
             "--libero-root", str(BRIDGE / "upstream/LIBERO")],
            stdout=cli_log.open("w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(ROOT),
            env=client_env(task_path / "runtime" / "libero-config"),
            start_new_session=True,
        )
        rc = client.wait()
        if rc != 0:
            raise RuntimeError("client rc=%d; see %s" % (rc, cli_log))
    except BaseException as error:
        cl.write_task_meta(task_path, status=cl.STATUS_FAILED, error=str(error))
        stop_server(server, out_server, served=served)
        raise
    stop_server(server, out_server, served=True)

    cl.write_task_meta(
        task_path, status=cl.STATUS_COMPLETE, wall_s=round(time.time() - started, 1),
        sampling=cl.sampling_block(
            design="%d init states x %d flow-noise draws" % (args.scenes, args.draws),
            designed_episodes=args.episodes,
            summaries=cl.load_summaries(task_path),
            held_fixed=(["task", "wrist_layout", "flow_noise_seed"] if args.draws == 1
                        else ["task", "wrist_layout"]),
            varies=(["init_state"] if args.draws == 1
                    else ["init_state", "flow_noise_seed"])),
    )
    report = cl.verify_task(task_path, expect_episodes=args.episodes,
                            expect_routes=(args.trunc_rounds is None))
    if not report["ok"]:
        cl.write_task_meta(task_path, status=cl.STATUS_FAILED,
                           error="; ".join(report["problems"]))
        print("[FAIL] %s: %s" % (label, report["problems"]), flush=True)
    else:
        print("[ok] %s  %d/%d success  %d control steps  %.0f s"
              % (label, report["successes"], report["episodes"],
                 report["control_steps"], time.time() - started), flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="capture-layout root (omit when using --run-id)")
    ap.add_argument("--run-id", help="write straight into the hub under this run id")
    ap.add_argument("--hub-root", default=str(cl.HUB_ROOT))
    ap.add_argument("--model", default=cl.HUB_MODEL)
    ap.add_argument("--gpu", required=True, help="MIG UUID")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--benchmarks", default=",".join(cl.BENCHMARKS))
    ap.add_argument("--tasks", help="comma-separated task ids; default all")
    ap.add_argument("--episodes", type=int, default=50,
                    help="shorthand for --scenes N --draws 1: init states 0..N-1, "
                         "one flow-noise draw each")
    ap.add_argument("--scenes", type=int,
                    help="how many init states (0..N-1); defaults to --episodes")
    ap.add_argument("--scene-ids",
                    help="explicit comma-separated init states, overriding --scenes. "
                         "Use this to spread the sample over the full 0..49 range "
                         "instead of taking a prefix")
    ap.add_argument("--draws", type=int, default=1,
                    help="flow-noise draws per init state.  scenes x draws episodes "
                         "in total; each draw index gets its own noise seed, shared "
                         "across scenes so the draws are paired")
    ap.add_argument("--noise-seed-base", type=int, default=1000)
    ap.add_argument("--store-hidden", action="store_true",
                    help="also capture the 1024-dim router inputs into hidden.zarr")
    ap.add_argument("--pin", default=None,
                    help="pin table from build_switch_pin.py: freeze the state "
                         "token's HB routing for a causal arm")
    ap.add_argument("--pin-regime", default=None, choices=["on", "off"])
    ap.add_argument("--trunc-rounds", type=int, default=None,
                    help="early-stop the flow loop after r rounds with a one-jump "
                         "extrapolation; implies --no-route-capture on the server")
    ap.add_argument("--inference-timeout", type=float, default=300.0,
                    help="client-side per-inference websocket timeout; raise on "
                         "CPU serving or a loaded box")
    ap.add_argument("--force", action="store_true", help="recapture completed tasks")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.run_id and not args.root:
        ap.error("give --run-id (hub mode) or --root (capture-layout mode)")
    if args.scene_ids:
        ids = [int(x) for x in args.scene_ids.split(",") if x.strip() != ""]
        if len(set(ids)) != len(ids):
            ap.error("--scene-ids has duplicates")
        args.scene_ids = ",".join(str(i) for i in ids)
        args.scenes = len(ids)
    elif args.scenes is None:
        args.scenes = args.episodes
    # From here on `episodes` is the derived total, so every count written to
    # meta.json and every completeness check refers to the same number.
    args.episodes = args.scenes * args.draws
    benchmarks = args.benchmarks.split(",")

    unsupported = [b for b in benchmarks if b not in cl.CAPTURABLE]
    if unsupported:
        # the hub has directories for libero_10 and calvin_d_d, but there is no
        # local checkpoint for the first and no capture path at all for the second
        ap.error("cannot capture %s; this driver supports %s"
                 % (unsupported, list(cl.CAPTURABLE)))

    if args.run_id:
        cl.validate_run_id(args.run_id)
        source = cl.iter_hub_tasks(args.hub_root, benchmarks)
    else:
        source = cl.iter_planned_tasks(args.root)

    wanted = {int(v) for v in args.tasks.split(",")} if args.tasks else None
    plan = [task for task in source
            if task["benchmark"] in benchmarks
            and (wanted is None or task["task_id"] in wanted)]

    # LIBERO ships exactly 50 initial states per task.  The bound is on *scenes*:
    # more draws per scene is a legitimate design (it makes flow noise the only
    # source of variation), more scenes than exist is not.
    for task in plan:
        available = int(task.get("n_init_states", 50))
        if args.scenes > available:
            ap.error("--scenes %d exceeds the %d initial states of %s/%s"
                     % (args.scenes, available, task["benchmark"], task["name"]))
        if args.scene_ids:
            over = [i for i in (int(x) for x in args.scene_ids.split(","))
                    if i >= available]
            if over:
                ap.error("--scene-ids %s are beyond the %d initial states of %s/%s"
                         % (over, available, task["benchmark"], task["name"]))

    where = ("hub %s/cache/%s/<benchmark>/<task>/%s"
             % (args.hub_root, args.model, args.run_id)) if args.run_id else args.root
    print("plan: %d tasks x %d episodes on %s\n  -> %s" % (len(plan), args.episodes, args.gpu, where))
    for task in plan:
        print("  %s/%s" % (task["benchmark"], task["dir_name"]))

    if args.run_id and not args.force:
        clashes = [t["name"] for t in plan if task_paths(args, t)[0].exists()]
        if clashes:
            ap.error("run_id %r already exists for %d task(s), e.g. %s; "
                     "pick another run_id or pass --force"
                     % (args.run_id, len(clashes), clashes[:3]))
    if args.dry_run:
        return 0

    if not args.run_id:
        for benchmark in benchmarks:
            cl.write_suite_meta(
                args.root, benchmark,
                checkpoint_dir=str(CHECKPOINT_DIR[cl.SUITE_KEY[benchmark]]),
                wrist_layout=WRIST_LAYOUT,
            )

    failures = []
    for task in plan:
        try:
            report = capture_task(args, task, args.port)
        except Exception as error:  # keep going; one bad task must not kill the sweep
            print("[ERROR] %s: %s" % (task["dir_name"], error), flush=True)
            failures.append(task["dir_name"])
            continue
        if not report.get("skipped") and not report.get("ok"):
            failures.append(task["dir_name"])
    print("\ndone; %d failure(s): %s" % (len(failures), failures or "none"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
