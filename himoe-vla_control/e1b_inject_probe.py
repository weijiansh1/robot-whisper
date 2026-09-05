#!/usr/bin/env python
"""E1-B: mid-flow latent injection probe (self-testing).

Three assertions make the mechanism auditable before any science is read off:

  T1 transparency  a zero-delta injection at f* reproduces the uninjected run
                   bit-for-bit (the hook itself perturbs nothing);
  T2 causality     with a real delta at f*, routing at steps f < f* is
                   bit-identical and steps f >= f* differ (the perturbation
                   enters exactly where we say it does);
  T3 calibration   sweeping ||delta||/||x_{f*}|| locates the regime above the
                   bf16 floor documented in FINDING_01, so epsilon is chosen
                   from data instead of guessed.

Science output: for each injection step f* and epsilon, the decay/growth of the
routing perturbation over the REMAINING flow steps -- D_f for f >= f*, the
finite-time contraction index lambda, and peak/terminal gains.  This is the
contraction question E1-A could not ask (it perturbed only the f=0 entry point).

Run in the LIBERO env against serve_latent_inject.py --return-full-probs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src")
sys.path.insert(0, "/home/jovyan/work/himoe-vla/himoe-route-capture")
sys.path.insert(0, "/home/jovyan/work/himoe-vla/himoe-vla_control")

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.protocol import FLOW_NOISE_KEY, FLOW_NOISE_SHAPE
from bestofn_protocol import flow_noise, seed_words
import control_metrics as cm
from e1e2_collect import load_snapshot_observation

PROBS_KEY = "recorder/hb_router_probs"
INJECT_STEP_KEY = "inject/step"
INJECT_DELTA_KEY = "inject/delta"
NORM_KEY = "inject/x_norm_before"
PRIMARY_LAYERS = slice(4, 8)
ACTION_TOKENS = slice(1, 11)


def psi_of(probs, subset="primary"):
    p = np.asarray(probs, np.float32)
    if subset == "primary":
        p = p[PRIMARY_LAYERS, :, ACTION_TOKENS, :]
    return cm.flatten_query(np.transpose(p, (1, 0, 2, 3)))


def ask(client, obs, noise, step=None, delta=None):
    req = dict(obs)
    req[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, np.float32)
    if step is not None:
        req[INJECT_STEP_KEY] = int(step)
        req[INJECT_DELTA_KEY] = np.ascontiguousarray(delta, np.float32)
    resp = client.infer(req)
    if PROBS_KEY not in resp:
        raise RuntimeError("server must run --return-full-probs")
    return (np.asarray(resp[PROBS_KEY], np.float32),
            np.asarray(resp["actions"], np.float32),
            float(resp.get(NORM_KEY, float("nan"))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--snapshots", help="label=dir,... (or use --plan)")
    ap.add_argument("--plan", help="e1e2 plan JSON: uses trunk_uid/q_offset/arm")
    ap.add_argument("--steps", default="2,5,8", help="injection steps f*")
    ap.add_argument("--epsilons", default="0.01,0.03,0.1,0.3",
                    help="||delta|| relative to ||x_{f*}||")
    ap.add_argument("--n-dirs", type=int, default=8)
    ap.add_argument("--master-seed", type=int, default=20260904)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest-only", action="store_true")
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    epsilons = [float(e) for e in args.epsilons.split(",")]
    if args.plan:
        jobs = [(f"t{j['trunk_uid']}.q{j['q_offset']:+d}.{j['arm']}",
                 pathlib.Path(j["snapshot_dir"]))
                for j in json.loads(pathlib.Path(args.plan).read_text())]
    elif args.snapshots:
        jobs = [(t.split("=", 1)[0], pathlib.Path(t.split("=", 1)[1]))
                for t in args.snapshots.split(",")]
    else:
        ap.error("give --plan or --snapshots")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.master_seed)
    records, checks = [], []

    with PolicyClient(host=args.host, port=args.port, connect_timeout=600.0,
                      inference_timeout=300.0) as client:
        for label, snap in jobs:
            obs = load_snapshot_observation(snap)
            xi = flow_noise(seed_words(args.master_seed, "e1b/base", 0, 0, 0))
            P0, A0, _ = ask(client, obs, xi)
            F = P0.shape[1]

            # ---- T1 transparency -------------------------------------------
            zero = np.zeros(FLOW_NOISE_SHAPE, np.float32)
            Pz, Az, xn = ask(client, obs, xi, step=steps[0], delta=zero)
            t1 = bool(np.array_equal(P0, Pz) and np.array_equal(A0, Az))
            assert t1, f"T1 transparency failed on {label}"
            checks.append({"snapshot": label, "check": "T1_transparency",
                           "passed": t1, "x_norm_at_step": xn})
            print(f"[T1] {label}: zero-delta injection identical = {t1} "
                  f"(||x_f*||={xn:.3f})")

            # ---- T2 causality ----------------------------------------------
            u = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
            u /= np.linalg.norm(u)
            fstar = steps[len(steps) // 2]
            _, _, xn2 = ask(client, obs, xi, step=fstar, delta=zero)
            delta = np.float32(0.1 * xn2) * u
            Pc, Ac, _ = ask(client, obs, xi, step=fstar, delta=delta)
            pre_same = bool(np.array_equal(P0[:, :fstar], Pc[:, :fstar]))
            post_diff = bool(not np.array_equal(P0[:, fstar:], Pc[:, fstar:]))
            checks.append({"snapshot": label, "check": "T2_causality",
                           "f_star": fstar, "passed": pre_same and post_diff,
                           "pre_identical": pre_same, "post_differs": post_diff})
            print(f"[T2] {label}: steps<{fstar} identical = {pre_same}, "
                  f"steps>={fstar} differ = {post_diff}")
            if not (t1 and pre_same and post_diff):
                raise SystemExit("E1-B self-test FAILED; not collecting science")
            if args.selftest_only:
                continue

            # ---- T3 calibration + science ----------------------------------
            psi0 = psi_of(P0)
            for fstar in steps:
                _, _, xnorm = ask(client, obs, xi, step=fstar, delta=zero)
                for eps in epsilons:
                    for d in range(args.n_dirs):
                        v = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
                        v /= np.linalg.norm(v)
                        delta = np.float32(eps * xnorm) * v
                        P, A, _ = ask(client, obs, xi, step=fstar, delta=delta)
                        D = cm.route_distance(psi0, psi_of(P))
                        pre = float(np.abs(D[:fstar]).max()) if fstar else 0.0
                        tail = D[fstar:]
                        parts = label.split(".")
                        rec = {
                            "snapshot": label,
                            "trunk_uid": int(parts[0][1:]) if parts[0][1:].isdigit() else -1,
                            "q_offset": int(parts[1][1:]) if len(parts) > 1 and parts[1][1:].lstrip("+-").isdigit() else 0,
                            "arm": parts[2] if len(parts) > 2 else label,
                            "f_star": fstar, "eps": eps,
                            "direction": d, "x_norm": xnorm,
                            "delta_norm": float(np.linalg.norm(delta)),
                            "D_pre_max": pre,          # must be 0.0
                            "D_at_inject": float(tail[0]),
                            "D_terminal": float(tail[-1]),
                            "D_peak": float(tail.max()),
                            "act_max_abs_diff": float(np.abs(A - A0).max()),
                            "D_f": [float(x) for x in D],
                        }
                        if len(tail) >= 3:
                            rec["lambda"] = cm.finite_time_contraction(tail, 0)
                        records.append(rec)
                print(f"[e1b] {label} f*={fstar}: {len(records)} records", flush=True)

    (out / "records.json").write_text(json.dumps(records, indent=1))
    (out / "selftest.json").write_text(json.dumps(checks, indent=1))
    if not records:
        print("self-test only; no science records"); return 0

    # ---- summary ----------------------------------------------------------
    import collections
    grp = collections.defaultdict(list)
    for r in records:
        grp[(r["snapshot"], r["f_star"], r["eps"])].append(r)
    bad_pre = [r for r in records if r["D_pre_max"] > 0]
    print(f"\npre-injection leakage (must be 0): {len(bad_pre)} record(s) violate")
    print("%-10s %4s %7s %10s %11s %11s %10s %9s"
          % ("snapshot", "f*", "eps", "||delta||", "D_at_inj", "D_peak",
             "D_term", "lambda"))
    for (label, fstar, eps), rs in sorted(grp.items()):
        m = lambda k: float(np.median([r[k] for r in rs if k in r]))
        print("%-10s %4d %7.3f %10.4f %11.6f %11.6f %10.6f %9.3f"
              % (label, fstar, eps, m("delta_norm"), m("D_at_inject"),
                 m("D_peak"), m("D_terminal"),
                 m("lambda") if "lambda" in rs[0] else float("nan")))
    print(f"\nwrote {out/'records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
