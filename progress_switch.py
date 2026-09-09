#!/usr/bin/env python3
"""Progress and running result for the three state-token-pin arms.

Pairing is by (init_state_id, flow_noise_seed), which is exactly what the three
arms hold in common; a pair only counts once every arm has run it.  McNemar is
on the discordant pairs against the base arm, because that is the test that
survives the simulator's own irreproducibility -- a byte-identical recapture
already disagrees on ~9% of outcomes, and only a paired test cancels it.
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
import sys

HUB = pathlib.Path(__file__).resolve().parent / "VLA_MUI_HUB"
TASK = ("libero_spatial/pick_up_the_black_bowl_on_the_ramekin"
        "_and_place_it_on_the_plate")
ARMS = ["pin-base", "pin-off", "pin-on"]
WANT = 512


def load(arm: str):
    run = HUB / "cache/HiMoE-VLA" / TASK / arm
    try:
        s = json.loads((run / "client/summaries.json").read_text())
    except (OSError, ValueError):
        return {}, (run / "meta.json").exists()
    return ({(e["init_state_id"], e["flow_noise_seed"]): bool(e["success"])
             for e in s}, True)


def main() -> int:
    data, started = {}, {}
    for a in ARMS:
        data[a], started[a] = load(a)
    print("arm        done/512   success      note")
    for a in ARMS:
        d = data[a]
        if not d:
            print("%-10s      -/512      -         %s"
                  % (a, "running, no episodes yet" if started[a] else "not started"))
            continue
        ok = sum(d.values())
        print("%-10s %6d/512   %5.1f%%      %s"
              % (a, len(d), 100.0 * ok / len(d), "COMPLETE" if len(d) >= WANT else ""))

    common = set(data[ARMS[0]])
    for a in ARMS[1:]:
        common &= set(data[a])
    print("\npairs complete in all three arms: %d" % len(common))
    if len(common) < 30:
        print("(too few for a verdict yet)")
        return 0

    base = data["pin-base"]
    print("\narm        succ on the shared pairs   b (base ok, arm fail)   "
          "c (base fail, arm ok)   McNemar p")
    for a in ARMS:
        d = data[a]
        ok = sum(d[k] for k in common)
        if a == "pin-base":
            print("%-10s %5d/%d = %5.1f%%              -                       -"
                  "                       -" % (a, ok, len(common),
                                                 100.0 * ok / len(common)))
            continue
        b = sum(1 for k in common if base[k] and not d[k])
        c = sum(1 for k in common if not base[k] and d[k])
        n = b + c
        # exact binomial two-sided on the discordant pairs
        if n == 0:
            p = 1.0
        else:
            k = min(b, c)
            tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
            p = min(1.0, 2 * tail)
        print("%-10s %5d/%d = %5.1f%%              %-24d%-24d%.4f"
              % (a, ok, len(common), 100.0 * ok / len(common), b, c, p))
    print("\n(b and c are the discordant pairs; the sign of b-c is the direction "
          "of the effect)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
