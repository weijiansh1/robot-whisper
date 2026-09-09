#!/usr/bin/env python3
"""End-to-end MoE routing capture for VLA_MUI_HUB: preflight -> capture -> validate.

Captures the complete routing state of every HiMoE gate for all four LIBERO
suites (40 tasks x 50 initial states) under the paper's input contract, writing
straight into ``VLA_MUI_HUB/cache/<model>/<benchmark>/<task>/<run_id>/``.

WHAT "COMPLETE ROUTING STATE" MEANS HERE
    12 gates fire per denoising step: 4 AS-MoE (layers 0/1/16/17, 3 experts,
    top-1, routed on the 24-dim data_mask) and 8 HB-MoE (layers 2-5/12-15, 32
    experts, top-4, routed on the 1024-dim hidden state).  Layers 6-11 are dense
    and have no router.  For every gate this stores

        the full softmax over all experts   (HB: 32-way, AS: 3-way)
        the top-k expert ids                (HB: 4, AS: 1)
        the selected experts' probabilities
        the router entropy                  (HB)

    which is everything the routing decision consists of.  The mixing weights are
    deliberately not stored: they are exactly recoverable
    (``himoe_router_recorder.combine_weight_from_raw``) -- HB renormalises the
    top-4 raw probabilities, AS uses the raw top-1 value.  Storing them would add
    bytes and a second source of truth.

    With ``--hidden`` it also writes ``hidden.zarr``: the 1024-dim tensors the
    routers actually read, all 12 gates.  That is the one thing not recoverable
    afterwards -- every array above is a function of it, not the reverse -- and it
    is what makes counterfactual routing (re-run the gate with perturbed weights)
    and representation probes possible.  It costs 2.70 MB per control step against
    41 KB for the routing, i.e. ~65x, so it lives in its own store and routing-only
    analysis never has to open it.

TASK / EVAL CONFIG
    Taken from the authors' own evaluation script.  The ICLR supplementary
    (OpenReview TX3oGD99CJ, 2025-09-25) and the released repo carry an identical
    ``examples/libero/main.py``: replan 10, 50 trials, settle 10, resize 224,
    seed 7, horizons 220/280/300/520.  The wrist layout is ``paper-right``
    (see run_corpus_capture.WRIST_LAYOUT for the evidence).

GPU PLACEMENT
    Model servers run on the two 32 GB MIG slices.  MIG mode disables a card's
    graphics engine, so the LIBERO clients render through EGL on GPU 0 with an
    empty CUDA_VISIBLE_DEVICES -- graphics pipeline only, no CUDA there.

Usage
    ./capture_routes_pipeline.py --run-id right-50x1 --preflight
    ./capture_routes_pipeline.py --run-id smoke-01 --smoke
    ./capture_routes_pipeline.py --run-id right-50x1
    ./capture_routes_pipeline.py --run-id right-50x1 --validate-only
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
CAPTURE = HERE / "himoe-route-capture"
HUB = HERE / "VLA_MUI_HUB"
BRIDGE = pathlib.Path("/home/jovyan/.cache/himoe-libero-bridge")

# Both 32 GB slices.  Never GPU 0 (another tenant) and never 4g.71gb.
LANES = (
    {"name": "A", "gpu": "MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a", "profile": "2g.35gb", "port": 8410},
    {"name": "B", "gpu": "MIG-60ef5cbe-ca36-5f3f-9931-35f2ce1878d3", "profile": "1g.35gb", "port": 8411},
)

SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10")

CHECKPOINT_SHA = {
    "libero_goal": "98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953",
    "libero_spatial": "1029d0827030a7521361d1904eeb3e7e7f2792be5c99abdb5701abb7ee87c137",
    "libero_object": "f9c5661533d271dec15d54d56fcd8c6c8811fc2b96095ac87638f7b3b2bdaafa",
    "libero_10": "cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256",
}
CHECKPOINT_DIR = {
    "libero_goal": "HiMoE-VLA-Libero-Goal",
    "libero_spatial": "HiMoE-VLA-Libero-Spatial",
    "libero_object": "HiMoE-VLA-Libero-Object",
    "libero_10": "HiMoE-VLA-Libero-10",
}
#: our own 500-episode paper-right results, as the sanity band for this capture.
#: These are *not* the paper's numbers -- Spatial and Long do not reproduce it.
REFERENCE_PCT = {
    "libero_goal": 98.0, "libero_spatial": 95.4,
    "libero_object": 99.0, "libero_10": 92.4,
}
#: a suite landing outside +-this many points of the reference means the capture
#: ran a different configuration, not that the policy changed.
SANITY_BAND_PP = 4.0

#: measured: 41 KB of routes.zarr per control step.
KB_PER_CONTROL_STEP = 41.0
#: control steps per episode, from the 2-episode smoke of each suite.  libero_10
#: is 3x the others -- its horizon is 520 against 220/280/300 -- so a single
#: average would put the disk estimate out by more than a factor of two.
STEPS_PER_EPISODE = {"libero_goal": 12.5, "libero_spatial": 8.0,
                     "libero_object": 14.0, "libero_10": 39.0}
#: hb_hidden 8x10x11x1024 + as_hidden 4x10x11x24, float16.  The AS routers read
#: the 24-dim data_mask, not the hidden state, so they are 1% of this.
MB_HIDDEN_PER_CONTROL_STEP = (8 * 10 * 11 * 1024 + 4 * 10 * 11 * 24) * 2 / 1e6
#: wall clock per episode, measured on the 500-episode paper-right runs.  This is
#: NOT proportional to control steps: a spatial episode is 8 control steps but
#: 13 s, a long one 39 steps and 25 s, because per-episode reset and settling
#: dominate the short suites.  Disk scales with control steps, time does not, so
#: the two need separate tables -- balancing lanes on control steps put one lane
#: at 3.6 h against the other's 7.7 h.
SECONDS_PER_EPISODE = {"libero_goal": 14.0, "libero_spatial": 13.0,
                       "libero_object": 12.0, "libero_10": 25.0}
#: every LIBERO task ships exactly 50 initial states
N_INIT_STATES = 50
#: activations are broadband.  Measured on the smoke capture: 78 control steps of
#: libero_10 wrote 118 MB of hidden.zarr, i.e. 1.52 MB/step against 1.82 MB raw --
#: 1.2x, not the 1.4x first assumed.
HIDDEN_COMPRESSION = 1.2


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def free_gb(path: pathlib.Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
def preflight(args) -> list[str]:
    """Everything that can be checked without touching a GPU or the network."""
    problems: list[str] = []
    say = lambda ok, msg: print(("  ok   " if ok else "  FAIL ") + msg)

    print("preflight")

    # -- code / config -----------------------------------------------------
    sys.path.insert(0, str(CAPTURE))
    try:
        import corpus_layout as cl  # noqa: F401
        import run_corpus_capture as rcc
    except Exception as error:
        problems.append("cannot import the capture driver: %s" % error)
        say(False, "capture driver importable")
        return problems
    say(True, "capture driver importable")

    if rcc.WRIST_LAYOUT != "paper-right":
        problems.append("wrist layout is %r, expected 'paper-right'" % rcc.WRIST_LAYOUT)
    say(rcc.WRIST_LAYOUT == "paper-right", "wrist layout = %s" % rcc.WRIST_LAYOUT)

    missing = [s for s in args.suites if s not in cl.CAPTURABLE]
    say(not missing, "suites capturable: %s" % ", ".join(args.suites))
    if missing:
        problems.append("not capturable: %s" % missing)

    # -- checkpoints -------------------------------------------------------
    for suite in args.suites:
        path = BRIDGE / "checkpoints" / CHECKPOINT_DIR[suite] / "pytorch_model.pth"
        if not path.is_file():
            problems.append("missing checkpoint %s" % path)
            say(False, "%s checkpoint" % suite)
            continue
        size = path.stat().st_size
        ok_size = size == 8_138_322_389
        if not ok_size:
            problems.append("%s checkpoint is %d bytes" % (suite, size))
        if args.verify_sha:
            digest = sh(["sha256sum", str(path)]).stdout.split()[0]
            ok = digest == CHECKPOINT_SHA[suite]
            if not ok:
                problems.append("%s sha256 mismatch" % suite)
            say(ok and ok_size, "%s checkpoint sha256" % suite)
        else:
            say(ok_size, "%s checkpoint size (sha skipped; --verify-sha to check)" % suite)

        stats = list((path.parent).glob("*/meta/stats.json"))
        if not stats:
            problems.append("%s has no normalization stats.json" % suite)
        say(bool(stats), "%s normalization stats" % suite)

    # -- environments ------------------------------------------------------
    for label, exe in (("model", BRIDGE / "envs/model/bin/python"),
                       ("libero", BRIDGE / "envs/libero/bin/python")):
        ok = exe.is_file() and os.access(exe, os.X_OK)
        if not ok:
            problems.append("missing %s env: %s" % (label, exe))
        say(ok, "%s env interpreter" % label)

    probe = sh([str(BRIDGE / "envs/model/bin/python"), "-c",
                "import zarr, torch; print(zarr.__version__, torch.__version__)"])
    ok = probe.returncode == 0
    if not ok:
        problems.append("model env cannot import zarr/torch: %s" % probe.stderr.strip()[:200])
    say(ok, "model env zarr/torch %s" % probe.stdout.strip())

    # -- hub skeleton ------------------------------------------------------
    for suite in args.suites:
        d = HUB / "cache" / cl.HUB_MODEL / cl.HUB_DIR[suite]
        n = len(list(d.iterdir())) if d.is_dir() else 0
        ok = n == 10
        if not ok:
            problems.append("hub dir %s has %d task dirs, expected 10" % (d, n))
        say(ok, "hub %s: %d task dirs" % (cl.HUB_DIR[suite], n))

    # -- run id ------------------------------------------------------------
    try:
        cl.validate_run_id(args.run_id)
        say(True, "run_id %r valid" % args.run_id)
    except Exception as error:
        problems.append(str(error))
        say(False, "run_id %r" % args.run_id)

    clashes = []
    for suite in args.suites:
        base = HUB / "cache" / cl.HUB_MODEL / cl.HUB_DIR[suite]
        if base.is_dir():
            clashes += [p for p in base.glob("*/" + args.run_id) if p.is_dir()]
    if clashes and not args.force:
        problems.append("run_id %r already exists in %d task dir(s); pick another or --force"
                        % (args.run_id, len(clashes)))
    say(not clashes or args.force, "run_id unused (%d existing)" % len(clashes))

    # -- disk --------------------------------------------------------------
    steps = sum(STEPS_PER_EPISODE[s] * len(args.select[s]) * args.episodes
                for s in args.suites)
    est_gb = steps * KB_PER_CONTROL_STEP / 1e6
    if args.hidden:
        est_gb += steps * MB_HIDDEN_PER_CONTROL_STEP / 1e3 / HIDDEN_COMPRESSION
    have = free_gb(HUB)
    # Report, do not block.  The estimate is an extrapolation from one smoke
    # capture, so a headroom multiple is guesswork dressed up as a rule; only
    # running out of space is a real failure, and that shows up as a write error
    # with the partial capture still on disk.  Hard-fail only if the estimate
    # exceeds what is actually free.
    ok = have > est_gb
    if not ok:
        problems.append("estimate %.1f GB exceeds the %.1f GB free" % (est_gb, have))
    say(ok, "disk: %.1f GB free, estimate ~%.1f GB, %.0f GB left after%s"
        % (have, est_gb, have - est_gb,
           " (incl. hidden.zarr)" if args.hidden else ""))
    if ok and have - est_gb < 20:
        print("       note: under 20 GB would remain; nothing else can be captured after")

    # -- gpus --------------------------------------------------------------
    listing = sh(["nvidia-smi", "-L"]).stdout
    for lane in LANES[: args.lanes]:
        ok = lane["gpu"] in listing
        if not ok:
            problems.append("MIG slice %s (%s) not visible" % (lane["profile"], lane["name"]))
        say(ok, "lane %s: %s" % (lane["name"], lane["profile"]))

    # Match every server that can hold a MIG slice, not just this pipeline's.  The
    # first smoke run died with `NVML_SUCCESS == r INTERNAL ASSERT FAILED` -- the
    # allocator's way of saying out-of-memory -- because a CALVIN
    # `himoe_calvin_alignment.policy_server` still owned 2g.35gb and the earlier
    # pattern here only looked for `serve_with_recorder|cli serve`.
    busy = [ln for ln in sh(["pgrep", "-af",
                             "serve_with_recorder|cli serve|policy_server|"
                             "flower_policy_server|himoe-calvin-eval"]).stdout.splitlines()
            if "pgrep" not in ln and "capture_routes_pipeline" not in ln]
    say(not busy, "no policy server holding a slice (%d)" % len(busy))
    for line in busy[:3]:
        print("       %s" % line.strip()[:110])
    if busy:
        problems.append("%d policy server(s) still running; free the slices first" % len(busy))

    # Free memory on the slices themselves cannot be read (MIG queries are
    # permission-blocked here), so the process check above is the only guard.
    # Keep it strict: an OOM shows up 2 minutes into a checkpoint load, per task.

    # -- egl ---------------------------------------------------------------
    egl = BRIDGE / "system-libs/usr/lib/x86_64-linux-gnu/libEGL.so.1"
    say(egl.is_file(), "EGL library present")
    if not egl.is_file():
        problems.append("missing %s" % egl)

    return problems


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------
def lane_env() -> dict:
    """websockets >= 14 sends even 127.0.0.1 through HTTP_PROXY; everything here
    is loopback, so drop the proxy for the whole pipeline."""
    env = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    return env


def balance_units(units: list[tuple[str, int]], lanes: int) -> list[list[tuple[str, int]]]:
    """Greedy longest-first bin packing over (suite, task) units.

    Round-robin was wrong here: a libero_10 task takes ~2x the wall clock of a
    libero_spatial one, so alternating can leave one lane running hours after the
    other is idle.  Cost is seconds per episode, not control steps -- see
    SECONDS_PER_EPISODE for why those two disagree.
    """
    order = sorted(units, key=lambda u: -SECONDS_PER_EPISODE[u[0]])
    bins: list[list[tuple[str, int]]] = [[] for _ in range(lanes)]
    load = [0.0] * lanes
    for suite, task in order:
        k = load.index(min(load))
        bins[k].append((suite, task))
        load[k] += SECONDS_PER_EPISODE[suite]
    return bins


def run_capture(args) -> int:
    """One driver invocation per (lane, suite): the driver takes --benchmarks and
    --tasks as a cross product, so a lane holding tasks from two suites has to
    call it twice rather than pass both at once."""
    import threading

    logs = HUB / "_pipeline" / args.run_id
    logs.mkdir(parents=True, exist_ok=True)
    # Persist the plan before anything runs.  Progress tooling otherwise has to
    # infer the denominator from the tasks that happen to have started, which
    # reads as "1536 total" when the plan is 2560; and this process's stdout is
    # block-buffered under nohup, so it is not a reliable record either.
    (logs / "plan.json").write_text(json.dumps({
        "run_id": args.run_id, "suites": args.suites, "select": args.select,
        "scenes": args.scenes, "draws": args.draws,
        "episodes_per_task": args.episodes,
        "n_tasks": sum(len(v) for v in args.select.values()),
        "scene_ids": args.scene_ids, "hidden": bool(args.hidden),
        "lanes": args.lanes,
    }, indent=2) + "\n")
    units = [(s, t) for s in args.suites for t in args.select[s]]
    lane_units = balance_units(units, args.lanes)

    results: dict[str, int] = {}

    def run_lane(lane: dict, mine: list[tuple[str, int]]) -> None:
        log = logs / ("lane-%s.log" % lane["name"])
        rc_total = 0
        by_suite: dict[str, list[int]] = {}
        for suite, task in mine:
            by_suite.setdefault(suite, []).append(task)
        with log.open("w") as fh:
            for suite, tasks in by_suite.items():
                cmd = [sys.executable, "-u", str(CAPTURE / "run_corpus_capture.py"),
                       "--run-id", args.run_id,
                       "--hub-root", str(HUB),
                       "--gpu", lane["gpu"],
                       "--port", str(lane["port"]),
                       "--benchmarks", suite,
                       "--tasks", ",".join(str(x) for x in sorted(tasks)),
                       "--draws", str(args.draws),
                       "--noise-seed-base", str(args.noise_seed_base)]
                if args.scene_ids:
                    cmd += ["--scene-ids", args.scene_ids]
                else:
                    cmd += ["--scenes", str(args.scenes)]
                if args.hidden:
                    cmd.append("--store-hidden")
                if args.force:
                    cmd.append("--force")
                fh.write("\n=== %s tasks %s ===\n" % (suite, sorted(tasks)))
                fh.flush()
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, cwd=str(CAPTURE),
                                     env=lane_env())
                rc_total |= rc
        results[lane["name"]] = rc_total

    threads = []
    for lane, mine in zip(LANES[: args.lanes], lane_units):
        if not mine:
            continue
        hours = sum(SECONDS_PER_EPISODE[s] for s, _ in mine) * args.episodes / 3600
        print("lane %s (%s): %s   ~%.1f h"
              % (lane["name"], lane["profile"],
                 ", ".join("%s/t%02d" % (s, t) for s, t in mine), hours))
        th = threading.Thread(target=run_lane, args=(lane, mine), daemon=False)
        th.start()
        threads.append(th)

    started = time.time()
    for th in threads:
        th.join()
    failed = [k for k, v in results.items() if v != 0]
    for name, rc in sorted(results.items()):
        print("lane %s finished rc=%d" % (name, rc))
    print("capture wall clock %.1f h" % ((time.time() - started) / 3600))
    return 1 if failed else 0


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
def task_names(cl, suite: str, task_ids: list[int]) -> set[str]:
    """Directory names for the given task ids, read from the hub manifest.

    The manifest lists tasks in ``benchmark.get_task(i)`` order, which is NOT
    alphabetical in any of the four suites -- indexing a sorted ``ls`` would map
    task 1 of libero_goal to the wrong directory.
    """
    manifest = json.loads((HUB / "manifest.json").read_text())
    entries = manifest["benchmarks"][suite]["tasks"]
    by_id = {t["task_id"]: t["name"] for t in entries}
    missing = [i for i in task_ids if i not in by_id]
    if missing:
        raise KeyError("manifest has no task_id %s for %s" % (missing, suite))
    return {by_id[i] for i in task_ids}


def run_validate(args) -> int:
    sys.path.insert(0, str(CAPTURE))
    import corpus_layout as cl

    print("\nvalidate")
    rows, problems = [], []
    for suite in args.suites:
        base = HUB / "cache" / cl.HUB_MODEL / cl.HUB_DIR[suite]
        succ = eps = steps = 0
        incomplete = []
        # Only the tasks this run was asked to capture.  Scanning all ten would
        # report the other nine as "not complete" after a smoke run, which is how
        # the first green smoke still printed four PROBLEM lines.
        wanted = task_names(cl, suite, args.select[suite])
        for task_dir in sorted(base.iterdir()):
            if task_dir.name not in wanted:
                continue
            run = task_dir / args.run_id
            if not run.is_dir():
                incomplete.append(task_dir.name)
                continue
            meta = cl.read_task_meta(run) or {}
            if meta.get("status") != cl.STATUS_COMPLETE:
                incomplete.append(task_dir.name)
                continue
            summaries = cl.load_summaries(run)
            eps += len(summaries)
            succ += sum(1 for s in summaries if s.get("success"))
            steps += cl.captured_control_steps(run) or 0
        pct = 100.0 * succ / eps if eps else 0.0
        ref = REFERENCE_PCT[suite]
        drift = pct - ref
        # The band compares against a 500-episode reference, so it only means
        # anything at full scale.  A smoke run is 1 task x 2 episodes: its rate is
        # 0/50/100% by construction and would always "fail".
        banded = not args.smoke and eps >= 100
        ok = bool(eps) and not incomplete and (not banded or abs(drift) <= SANITY_BAND_PP)
        rows.append((suite, succ, eps, pct, ref, drift, steps, len(incomplete), ok, banded))
        if incomplete:
            problems.append("%s: %d task(s) not complete: %s"
                            % (suite, len(incomplete), incomplete[:3]))
        elif not eps:
            problems.append("%s: no episodes captured" % suite)
        elif banded and abs(drift) > SANITY_BAND_PP:
            problems.append("%s: %.1f%% is %+.1f pp off the %.1f%% reference"
                            % (suite, pct, drift, ref))

    print("  %-16s %11s %8s %8s %8s %10s" % ("suite", "success", "rate", "ref", "drift", "ctrl-steps"))
    for suite, succ, eps, pct, ref, drift, steps, n_inc, ok, banded in rows:
        print("  %-16s %5d/%-5d %7.1f%% %7.1f%% %+8.1f %10d %s"
              % (suite, succ, eps, pct, ref, drift, steps,
                 ("ok" if ok else "CHECK") + ("" if banded else "  (band n/a)")))

    report = HUB / "_pipeline" / args.run_id / "REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# routing capture `%s`" % args.run_id,
        "",
        "wrist layout `paper-right`; replan 10 / settle 10 / resize 224 / seed 7,",
        "per the authors' `examples/libero/main.py` (identical in the ICLR",
        "supplementary and the released repo).",
        "",
        "| suite | success | rate | our 500-ep reference | drift |",
        "|---|---:|---:|---:|---:|",
    ]
    for suite, succ, eps, pct, ref, drift, steps, n_inc, ok, banded in rows:
        lines.append("| %s | %d/%d | %.1f%% | %.1f%% | %s |"
                     % (suite, succ, eps, pct, ref,
                        ("%+.1f pp" % drift) if banded else "n/a (too few episodes)"))
    lines += ["", "Control steps captured: %d." % sum(r[6] for r in rows), ""]
    if problems:
        lines += ["## problems", ""] + ["- %s" % p for p in problems]
    else:
        lines += ["No problems: every task complete and every suite inside the",
                  "+-%.1f pp sanity band." % SANITY_BAND_PP]
    report.write_text("\n".join(lines) + "\n")
    print("\n  report -> %s" % report)

    for p in problems:
        print("  PROBLEM: %s" % p)
    return 1 if problems else 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--episodes", type=int, default=50,
                    help="shorthand for --scenes N --draws 1")
    ap.add_argument("--scenes", type=int, help="init states per task")
    ap.add_argument("--draws", type=int, default=1,
                    help="flow-noise draws per init state; scenes x draws episodes")
    ap.add_argument("--tasks", help="comma-separated task ids, applied to every "
                                    "suite; default all ten")
    ap.add_argument("--select", action="append", metavar="SUITE:IDS", dest="select_spec",
                    help="per-suite task ids, e.g. --select libero_10:8 "
                         "--select libero_spatial:5,7 .  Repeatable.  Use this "
                         "when the suites need different tasks; --tasks cannot "
                         "express that because it is a cross product.")
    ap.add_argument("--scene-ids", help="explicit init states; overrides --scenes")
    ap.add_argument("--scene-select", choices=("prefix", "spread"), default="prefix",
                    help="prefix takes 0..scenes-1; spread samples the whole 0..49 "
                         "range at equal intervals, which represents the task better "
                         "when only a subset is captured")
    ap.add_argument("--lanes", type=int, default=2, choices=(1, 2))
    ap.add_argument("--noise-seed-base", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--hidden", action="store_true",
                    help="also capture the 1024-dim router inputs (hidden.zarr). "
                         "~65x the routing bytes and the only part that cannot be "
                         "recomputed afterwards; ~71 GB for the full 40-task sweep")
    ap.add_argument("--verify-sha", action="store_true",
                    help="sha256 all four checkpoints (~2 min for 32 GB)")
    ap.add_argument("--preflight", action="store_true", help="checks only, then stop")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="one task per suite, 2 episodes, single lane")
    args = ap.parse_args()

    args.suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    # A smoke run must exercise every checkpoint and both env paths, so it keeps
    # all four suites and cuts the other two axes: task 0 only, 2 episodes, 1 lane.
    # scenes: explicit ids > --scene-select spread > --scenes > --episodes
    if args.scene_ids:
        args.scene_ids = ",".join(str(int(x)) for x in args.scene_ids.split(","))
        args.scenes = len(args.scene_ids.split(","))
    else:
        if args.scenes is None:
            args.scenes = args.episodes
        if args.scene_select == "spread" and args.scenes < N_INIT_STATES:
            ids = sorted({int(round(i * (N_INIT_STATES - 1) / (args.scenes - 1)))
                          for i in range(args.scenes)}) if args.scenes > 1 else [0]
            args.scene_ids = ",".join(str(i) for i in ids)
            args.scenes = len(ids)
    args.episodes = args.scenes * args.draws

    # per-suite task selection
    default_ids = ([int(x) for x in args.tasks.split(",")] if args.tasks
                   else list(range(10)))
    args.select = {s: list(default_ids) for s in args.suites}
    for spec in (args.select_spec or []):
        if ":" not in spec:
            ap.error("--select needs SUITE:IDS, got %r" % spec)
        suite, ids = spec.split(":", 1)
        if suite not in args.suites:
            ap.error("--select names %r, which is not in --suites" % suite)
        args.select[suite] = [int(x) for x in ids.split(",") if x.strip() != ""]
    if args.smoke:
        args.scenes, args.draws, args.episodes = 2, 1, 2
        args.scene_ids, args.lanes = None, 1
        args.select = {s: [0] for s in args.suites}
    args.task_ids = sorted({i for v in args.select.values() for i in v})

    if args.validate_only:
        return run_validate(args)

    problems = preflight(args)
    if problems:
        print("\npreflight failed:")
        for p in problems:
            print("  - %s" % p)
        return 2
    print("preflight ok")
    if args.preflight:
        return 0

    n = sum(len(v) for v in args.select.values())
    print("\ncapture: %d task(s) x %d episodes (%d scene(s) x %d draw(s)) "
          "on %d lane(s), run_id=%s"
          % (n, args.episodes, args.scenes, args.draws, args.lanes, args.run_id))
    for suite in args.suites:
        print("   %-16s tasks %s" % (suite, args.select[suite]))
    if args.scene_ids:
        print("   scenes %s" % args.scene_ids)
    rc = run_capture(args)
    return run_validate(args) or rc


if __name__ == "__main__":
    raise SystemExit(main())
