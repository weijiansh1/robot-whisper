"""Step 10: the controls that decide whether steps 5-9 found anything.

Step 9's circular-shift null scored *better* than the real arms.  That is not
noise, it is diagnostic: a circular shift moves an episode's excursion to a
random chunk but keeps the episode's own values, so it preserves the whole
episode-to-outcome association and only destroys the timing.  Scoring higher
under it means the arms' discriminative power is an episode-level property,
not a temporally localised event -- and that relocating the excursion earlier
increases the lead.

So two harder controls are needed, and they are the ones that matter:

  TRIVIAL      fire at a fixed chunk c on every episode still running, union
               with v8.3.  This has no routing input at all.  Because risk
               episodes run to the cap and successes do not, a mid-episode
               alarm is nearly free in the short-capped suites: libero_spatial
               has 3,874 successes alive at chunk 6 but only 47 at chunk 14.
               Any head must beat this or it has found nothing.

  RATE-MATCHED fire on a random subset of episodes, matched to the head's own
               alarm count and alarm-chunk distribution, independent of the
               outcome.  This destroys only *which* episodes are selected.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C
from step4_ceiling_raw import build_bank
from step9_freeze_and_seal import ARM_A, ARM_B, apply_arms, fit_on_development

SEED = 20260906
N_NULL = 5


def trivial_first(d, chunk):
    """Alarm at `chunk` on every episode still running there."""
    return np.where(d["length"] > chunk, chunk, -1).astype(int)


def rate_matched(first, rng):
    """Same number of alarms, same chunk distribution, random episodes."""
    fired = np.flatnonzero(first >= 0)
    out = np.full(len(first), -1)
    pick = rng.choice(len(first), size=len(fired), replace=False)
    out[pick] = rng.permutation(first[fired])
    return out


def main() -> None:
    data = C.load_all()
    series = {c: build_bank(c, d) for c, d in data.items()}
    fitted = fit_on_development(series["development_main"])
    arms, firsts = {}, {}
    for c, d in data.items():
        a, b = apply_arms(series[c], fitted)
        firsts[c] = {"A": a, "B": b}
        arms.setdefault("v8.3", {})[c] = d["v83"]
        arms.setdefault("v8.3+A+B", {})[c] = C.union(d["v83"], a, b)

    base = C.profile(arms["v8.3"], data)
    real = C.profile(arms["v8.3+A+B"], data)
    print("v8.3        %s" % C.fmt_profile(base))
    print("v8.3+A+B    %s" % C.fmt_profile(real))

    # ---- where do the arms fire? ----------------------------------------
    print("\n=== alarm chunk of each arm, per suite (whole corpus) ===")
    print("%-16s %-4s %8s %10s %28s" % ("suite", "arm", "fires", "of", "chunk"))
    for suite in C.SUITES:
        for key in ("A", "B"):
            f, n = [], 0
            for c, d in data.items():
                m = d["suite"] == suite
                if not m.any():
                    continue
                n += int(m.sum())
                f.append(firsts[c][key][m])
            f = np.concatenate(f)
            hit = f[f >= 0]
            if not len(hit):
                print("%-16s %-4s %8d %10d %28s" % (suite, key, 0, n, "-"))
                continue
            print("%-16s %-4s %8d %10d   p10 q%-3d median q%-3d p90 q%-3d"
                  % (suite, key, len(hit), n, np.percentile(hit, 10),
                     np.median(hit), np.percentile(hit, 90)))

    # ---- TRIVIAL control -------------------------------------------------
    print("\n=== TRIVIAL control: alarm at a fixed chunk on every survivor,"
          " unioned with v8.3 (NO routing input) ===")
    print("%-10s %s" % ("chunk", "  ".join("lead>=%-2d" % l for l in C.LEADS)))
    rows = []
    for chunk in range(4, 22, 2):
        al = {c: C.union(d["v83"], trivial_first(d, chunk))
              for c, d in data.items()}
        p = C.profile(al, data)
        rows.append({"chunk": chunk} | {f"tp{l}": p[l][0] for l in C.LEADS}
                    | {f"fp{l}": p[l][1] for l in C.LEADS})
        print("q%-9d %s" % (chunk, "  ".join("%4d/%-5d" % p[l]
                                             for l in C.LEADS)))
    print("%-10s %s" % ("v8.3+A+B", "  ".join("%4d/%-5d" % real[l]
                                              for l in C.LEADS)))
    print("%-10s %s" % ("v8.3", "  ".join("%4d/%-5d" % base[l]
                                          for l in C.LEADS)))
    pd.DataFrame(rows).to_csv(C.RESULTS / "trivial_control.csv", index=False)

    print("\n  the honest comparison is at a matched FALSE-ALARM budget."
          "  No chunk constant comes close: its cheapest operating point is")
    triv = pd.DataFrame(rows)
    for lead in C.LEADS:
        cheap = triv.loc[triv[f"fp{lead}"].idxmin()]
        afford = triv[triv[f"fp{lead}"] <= real[lead][1]]
        verdict = ("no chunk constant is affordable at this budget"
                   if afford.empty else
                   "trivial q%d %d/%d" % (int(afford[f"tp{lead}"].idxmax()),
                                          int(afford[f"tp{lead}"].max()), 0))
        print("    lead>=%-2d  head %4d/%-4d | cheapest chunk constant q%-2d"
              " %4d/%-5d (%.0fx the head's FP) | %s"
              % (lead, real[lead][0], real[lead][1], int(cheap.chunk),
                 int(cheap[f"tp{lead}"]), int(cheap[f"fp{lead}"]),
                 cheap[f"fp{lead}"] / max(real[lead][1], 1), verdict))

    print("\n  per suite at lead>=4: head against the best trivial chunk with"
          " no more false alarms")
    hs = C.per_suite(arms["v8.3+A+B"], data, 4)
    bs = C.per_suite(arms["v8.3"], data, 4)
    for i, suite in enumerate(C.SUITES):
        best = None
        for chunk in range(4, 22, 2):
            al = {c: C.union(d["v83"], trivial_first(d, chunk))
                  for c, d in data.items()}
            t = C.per_suite(al, data, 4).iloc[i]
            if t.fp <= hs.iloc[i].fp and (best is None or t.tp > best[1]):
                best = (chunk, int(t.tp), int(t.fp))
        print("    %-16s v8.3 %4d/%-4d %3dFP | head %4d/%-4d %3dFP | trivial"
              " %s" % (suite, bs.iloc[i].tp, bs.iloc[i].n_risk, bs.iloc[i].fp,
                       hs.iloc[i].tp, hs.iloc[i].n_risk, hs.iloc[i].fp,
                       "q%d %d/%d %dFP" % (best[0], best[1],
                                           hs.iloc[i].n_risk, best[2])
                       if best else "none inside the budget"))

    # ---- RATE-MATCHED null ----------------------------------------------
    print("\n=== RATE-MATCHED null: same alarm count and chunk distribution,"
          " random episodes ===")
    rng = np.random.default_rng(SEED)
    nulls = []
    for k in range(N_NULL):
        al = {}
        for c, d in data.items():
            a = rate_matched(firsts[c]["A"], rng)
            b = rate_matched(firsts[c]["B"], rng)
            al[c] = C.union(d["v83"], a, b)
        p = C.profile(al, data)
        nulls.append(p)
        print("  draw %d: %s" % (k + 1, C.fmt_profile(p)))
    print("  real  : %s" % C.fmt_profile(real))
    print("\n  margin over the rate-matched null (TP, at matched alarm rate):")
    for lead in C.LEADS:
        mtp = max(p[lead][0] for p in nulls)
        mfp = min(p[lead][1] for p in nulls)
        print("    lead>=%-2d  real %4d/%-4d   null best %4d/%-4d   margin"
              " %+d TP" % (lead, real[lead][0], real[lead][1], mtp, mfp,
                           real[lead][0] - mtp))

    (C.RESULTS / "controls.json").write_text(json.dumps({
        "v83": {str(l): list(base[l]) for l in C.LEADS},
        "head": {str(l): list(real[l]) for l in C.LEADS},
        "trivial": rows,
        "rate_matched": [{str(l): list(p[l]) for l in C.LEADS} for p in nulls],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
