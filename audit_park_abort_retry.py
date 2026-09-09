#!/usr/bin/env python3
"""Can a parking detector early-stop failing rollouts and buy success-per-
compute via seed re-roll?

Facts this builds on: failures end in a fixed parking state (arm stopped,
gripper open) and run to the 52-inference timeout while successes end at
~35-45; seed re-roll is outcome-independent (phi ~ 0.1).  Policy simulated
offline on the t08 16x32 grid: run a seed; if the detector fires, abort and
draw the next seed; stop at first success or budget exhaustion.

Detector (pre-declared form): pooled-standardized proprio speed; fire at the
first t >= MIN_T with mean speed over the last W steps < theta AND gripper
open.  theta calibrated on 8 scenes by policy value, evaluated on the other 8.
A routing twin (state-token prob Hellinger speed) runs alongside for the
"can the routing light do it" question.  Baselines: naive retry (no abort),
oracle abort (failures aborted at MIN_T).  Writes audit_park_abort_retry.json.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
NPZ = (HERE / "himoe-routing-rules-20260819/data"
       / "libero_long__KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.npz")
W = 5
MIN_T = 12
BUDGETS = (60, 100, 150, 208)
N_ORD = 2000
RNG = np.random.default_rng(0)


def fire_time(speed, open_, theta):
    """First index with trailing-W mean speed < theta and gripper open."""
    cs = np.concatenate([[0.0], np.cumsum(speed)])
    for t in range(MIN_T, len(speed)):
        if open_[t] and (cs[t + 1] - cs[max(0, t + 1 - W)]) / W < theta:
            return t + 1                     # cost if aborted here
    return None


def simulate(scene_eps, budget, use_abort, oracle=False):
    """scene_eps: list of (cost_full, success, fire_or_None). One bootstrap."""
    order = RNG.permutation(len(scene_eps))
    spent = 0
    for i in order:
        cost, suc, fire = scene_eps[i]
        if oracle:
            attempt = cost if suc else MIN_T
            hit = suc
        elif use_abort and fire is not None:
            attempt = fire                   # aborted (may be a would-succeed)
            hit = False
        else:
            attempt = cost
            hit = suc
        if spent + attempt > budget:
            return 0, budget
        spent += attempt
        if hit:
            return 1, spent
    return 0, spent


def main() -> int:
    z = np.load(NPZ, allow_pickle=True)
    n_rows = z["n_rows"]
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = z["success"].astype(bool)
    scene = z["scene"]
    P = z["proprio"]
    probs = z["state_token_probs"].astype(np.float32)
    mu, sd = P.mean(0), P.std(0) + 1e-9

    eps = []
    for i in range(len(y)):
        sl = slice(off[i], off[i] + n_rows[i])
        p = (P[sl] - mu) / sd
        speed = np.linalg.norm(np.diff(p, axis=0), axis=1)
        speed = np.concatenate([[speed[0]], speed])
        opening = P[sl][:, 6] - P[sl][:, 7]
        r = np.sqrt(probs[sl].reshape(n_rows[i], -1))
        rspeed = np.linalg.norm(np.diff(r, axis=0), axis=1)
        rspeed = np.concatenate([[rspeed[0]], rspeed])
        eps.append({"cost": int(n_rows[i]), "suc": bool(y[i]),
                    "scene": int(scene[i]), "speed": speed,
                    "open": opening > 0.04, "rspeed": rspeed})

    scenes = np.unique(scene)
    cal, ev = scenes[::2], scenes[1::2]

    def policy_value(theta, which, scene_set, budget, use_abort=True,
                     oracle=False, n_ord=400):
        vals = []
        for s in scene_set:
            se = [e for e in eps if e["scene"] == s]
            tab = [(e["cost"], e["suc"],
                    fire_time(e[which], e["open"], theta)) for e in se]
            hits = [simulate(tab, budget, use_abort, oracle)[0]
                    for _ in range(n_ord)]
            vals.append(np.mean(hits))
        return float(np.mean(vals))

    out = {"min_t": MIN_T, "window": W}
    for which, label in (("speed", "proprio"), ("rspeed", "routing")):
        grid = ([0.05, 0.1, 0.15, 0.2, 0.3, 0.4] if which == "speed"
                else [0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
        best = max(grid, key=lambda t: policy_value(t, which, cal, 100))
        # detector stats on eval scenes
        fires_f, fires_s, leads = 0, 0, []
        nf = ns = 0
        for e in eps:
            if e["scene"] not in ev:
                continue
            f = fire_time(e[which], e["open"], best)
            if e["suc"]:
                ns += 1; fires_s += f is not None
            else:
                nf += 1
                fires_f += f is not None
                if f is not None:
                    leads.append(e["cost"] - f)
        row = {"theta": best,
               "recall_on_failures": round(fires_f / nf, 3),
               "false_fire_on_successes": round(fires_s / ns, 3),
               "median_inferences_saved_per_caught_failure":
                   int(np.median(leads)) if leads else 0}
        for B in BUDGETS:
            row["succ@%d" % B] = {
                "naive_retry": round(policy_value(0, which, ev, B,
                                                  use_abort=False), 3),
                "abort_retry": round(policy_value(best, which, ev, B), 3),
                "oracle": round(policy_value(0, which, ev, B, oracle=True), 3),
            }
        out[label] = row
        print("%s detector: theta=%.2f  recall(fail)=%.2f  false(succ)=%.2f  "
              "saved/caught=%d" % (label, best, row["recall_on_failures"],
                                   row["false_fire_on_successes"],
                                   row["median_inferences_saved_per_caught_failure"]))
        for B in BUDGETS:
            r = row["succ@%d" % B]
            print("   budget %-4d  naive %.3f  abort %.3f  oracle %.3f"
                  % (B, r["naive_retry"], r["abort_retry"], r["oracle"]))

    (HERE / "audit_park_abort_retry.json").write_text(json.dumps(out, indent=1))
    print("\nwrote audit_park_abort_retry.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
