#!/usr/bin/env python3
"""Final analysis of the state-token pin arms.

Three arms of 512 episodes on libero_spatial t05, paired on (init_state_id,
flow_noise_seed): base, and the state token's HB routing at layers 2-5 frozen to
the off regime and to the on regime.  A p-value alone would not settle this --
a null with a wide interval is an underpowered experiment, a null with a tight
interval is a result -- so this reports:

  1. the paired effect with an exact-bootstrap interval, not just McNemar's p
  2. proof that the intervention reached the rollout at all: if the pinned arms
     produced the same trajectories as base, the null would be about nothing
  3. the reproducibility floor, measured here rather than assumed, by pairing
     the fresh base arm against the earlier right-16x32 capture of the same task
     with the same scenes and the same noise seeds.  An effect smaller than the
     floor is not detectable by this design at any n.
  4. what this design could have detected, so the null has a size attached
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

HUB = pathlib.Path(__file__).resolve().parent / "VLA_MUI_HUB"
TASK = ("libero_spatial/pick_up_the_black_bowl_on_the_ramekin"
        "_and_place_it_on_the_plate")
ARMS = ["pin-base", "pin-off", "pin-on"]
RNG = np.random.default_rng(20260819)


def load(arm):
    run = HUB / "cache/HiMoE-VLA" / TASK / arm
    s = json.loads((run / "client/summaries.json").read_text())
    by = {(e["init_state_id"], e["flow_noise_seed"]): e for e in s}
    summ = None
    p = run / "server/capture_summary.json"
    if p.exists():
        summ = json.loads(p.read_text())
    return by, summ


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_diff(base, arm, keys, n=20000):
    """Bootstrap the paired difference in success rate, resampling pairs."""
    y0 = np.array([base[k]["success"] for k in keys], bool)
    y1 = np.array([arm[k]["success"] for k in keys], bool)
    idx = RNG.integers(0, len(keys), (n, len(keys)))
    d = y1[idx].mean(1) - y0[idx].mean(1)
    return float(y1.mean() - y0.mean()), np.percentile(d, [2.5, 97.5])


def main() -> int:
    arms = {}
    for a in ARMS:
        arms[a] = load(a)
    keys = sorted(set(arms[ARMS[0]][0]) & set(arms[ARMS[1]][0]) & set(arms[ARMS[2]][0]))
    base = arms["pin-base"][0]
    print("=== 1. outcome, %d paired (init state, noise seed) draws ===" % len(keys))
    print("arm        success        paired diff vs base    95%% CI            "
          "b / c     McNemar p")
    for a in ARMS:
        d = arms[a][0]
        ok = sum(d[k]["success"] for k in keys)
        if a == "pin-base":
            print("%-10s %3d/%d = %5.2f%%   -                      -                 "
                  "  -         -" % (a, ok, len(keys), 100 * ok / len(keys)))
            continue
        b = sum(1 for k in keys if base[k]["success"] and not d[k]["success"])
        c = sum(1 for k in keys if not base[k]["success"] and d[k]["success"])
        pt, ci = boot_diff(base, d, keys)
        print("%-10s %3d/%d = %5.2f%%   %+6.2f pp              [%+.2f, %+.2f] pp   "
              "%2d / %-2d    %.4f"
              % (a, ok, len(keys), 100 * ok / len(keys), 100 * pt,
                 100 * ci[0], 100 * ci[1], b, c, exact_mcnemar(b, c)))

    print("\n=== 2. did the intervention reach the rollout? ===")
    for a in ARMS:
        d, summ = arms[a]
        steps = np.array([d[k]["action_steps"] for k in keys])
        s0 = np.array([base[k]["action_steps"] for k in keys])
        diff = steps != s0
        pin = (summ or {}).get("pin")
        note = ("no pin" if not pin else
                "%s: %d replacements, mean |original ∩ pin| = %.2f / 4"
                % (pin["regime"], pin["n_replacements"],
                   pin["mean_overlap_with_original"]))
        print("  %-10s trajectories differing from base: %3d/%d = %5.1f%%   "
              "median |Δ steps| %3.0f" % (a, diff.sum(), len(keys),
                                          100 * diff.mean(),
                                          np.median(np.abs(steps - s0)[diff])
                                          if diff.any() else 0))
        print("             %s" % note)

    print("\n=== 3. the reproducibility floor, measured on this very task ===")
    old, _ = load("right-16x32")
    shared = sorted(set(base) & set(old))
    b = sum(1 for k in shared if old[k]["success"] and not base[k]["success"])
    c = sum(1 for k in shared if not old[k]["success"] and base[k]["success"])
    agree = sum(1 for k in shared if old[k]["success"] == base[k]["success"])
    st = np.array([base[k]["action_steps"] != old[k]["action_steps"] for k in shared])
    print("  base vs the earlier right-16x32 capture, identical config, %d pairs:"
          % len(shared))
    print("     outcome agreement %d/%d = %.1f%%   discordant b/c = %d/%d   "
          "McNemar p = %.4f" % (agree, len(shared), 100 * agree / len(shared),
                                b, c, exact_mcnemar(b, c)))
    print("     trajectories differing: %d/%d = %.1f%%"
          % (st.sum(), len(shared), 100 * st.mean()))
    print("     -> two runs of the *same* thing already disagree on %d outcomes;"
          % (b + c))
    print("        an intervention effect has to clear that to be visible at all")

    print("\n=== 5. marginal rate vs which episodes ===")
    # The marginal is only half the question.  pin-off's c is 10 while base has
    # 12 failures in total, so the intervention rescued 10 of the 12 -- a table
    # this close to what independence predicts means the intervention did not
    # shift how often the policy succeeds, it re-rolled *which* episodes do.
    from scipy.stats import fisher_exact
    for a in ARMS[1:] + ["right-16x32"]:
        d = load(a)[0]
        kk = [k for k in keys if k in d]
        y0 = np.array([base[k]["success"] for k in kk], bool)
        y1 = np.array([d[k]["success"] for k in kk], bool)
        t = np.array([[int((y0 & y1).sum()), int((y0 & ~y1).sum())],
                      [int((~y0 & y1).sum()), int((~y0 & ~y1).sum())]])
        n_ = t.sum()
        exp = np.outer(t.sum(1), t.sum(0)) / n_
        den = math.sqrt(t.sum(1).prod() * t.sum(0).prod())
        phi = (t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]) / den if den else float("nan")
        _, pf = fisher_exact(t)
        print("  %-12s  observed [[%3d %3d] [%3d %3d]]   independence predicts "
              "[[%5.1f %4.1f] [%4.1f %4.1f]]"
              % (a, t[0, 0], t[0, 1], t[1, 0], t[1, 1],
                 exp[0, 0], exp[0, 1], exp[1, 0], exp[1, 1]))
        print("                phi = %+.3f   Fisher p = %.4f   "
              "(phi 0 = outcome independent of base, 1 = unchanged)"
              % (phi, pf))

    print("\n=== 4. what this design could have detected ===")
    n = len(keys)
    y0 = np.array([base[k]["success"] for k in keys], bool)
    p0 = y0.mean()
    # Power against a purely one-directional alternative -- an intervention
    # that only ever breaks a success.  That is the best case for McNemar, so
    # read it as an upper bound on sensitivity; the assumption-free statement is
    # the bootstrap interval in section 1, which bounds the effect within a few
    # points either way.
    for delta in (0.02, 0.03, 0.05, 0.10):
        hits = 0
        for _ in range(2000):
            y1 = y0 & ~(RNG.random(n) < delta)
            hits += exact_mcnemar(int((y0 & ~y1).sum()),
                                  int((~y0 & y1).sum())) < 0.05
        print("     a true %4.1f pp one-directional drop: detected %5.1f%% of the time"
              % (100 * delta, 100 * hits / 2000))
    print("     (base success on these pairs = %.1f%%)" % (100 * p0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
