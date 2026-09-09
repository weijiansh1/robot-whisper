#!/usr/bin/env python3
"""Derive the pin table for the state-token switch intervention.

The gate at HB layers 2-5 turns the state token's routing on and off with the
gripper.  To ask whether that gate does anything functionally, the routing has to
be replaced by a constant -- one constant taken from the on regime and one from
the off regime -- and the rollout rerun.  This builds those constants from the
existing capture so the intervention lands on routing the model actually
produces, not on something invented.

For each front layer, within each regime, the "typical" routing is the top-4 of
the regime's mean router distribution, with weights the renormalised mean
probabilities of those four -- which is exactly the form the gate emits, since
`norm_topk_prob` divides the selected raw probabilities by their sum.

Writes fig/../pin_<suite>_<task>.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN_ID = "right-16x32"
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
FRONT = [0, 1, 2, 3]          # axis-1 indices of layers 2,3,4,5
UNIF, ON, OFF, CH = 1 / 32, 0.5, 2 / 32, 2048


def main() -> int:
    suite, task = sys.argv[1], sys.argv[2]
    sys.path.insert(0, str(HERE / "himoe-route-capture"))
    import corpus_layout as cl
    run = HUB / "cache" / cl.HUB_MODEL / cl.HUB_DIR[suite] / task / RUN_ID
    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    n = z["hb_router_probs"].shape[0]

    # state-token probabilities; constant across denoise, so take iteration 0
    P = np.empty((n, 8, 32), np.float32)
    for a in range(0, n, CH):
        b = min(a + CH, n)
        P[a:b] = np.asarray(z["hb_router_probs"][a:b, :, 0, 0], np.float32)
    star = P.mean(0).argmax(-1)

    out = {"suite": suite, "task": task, "run_id": RUN_ID, "n_control_steps": int(n),
           "layers": {}}
    print("layer  favourite  on-rate   regime   top-4 experts        weights")
    for i in FRONT:
        L = HB_LAYER[i]
        p = P[:, i, star[i]]
        m_on, m_off = p > ON, p < OFF
        rec = {"axis_index": i, "favourite": int(star[i]),
               "on_rate": float(m_on.mean()), "off_rate": float(m_off.mean())}
        for tag, m in (("on", m_on), ("off", m_off)):
            if m.sum() < 50:
                print("  %2d      e%-2d     %5.1f%%   %-4s  too few steps (%d), skipped"
                      % (L, star[i], 100 * m_on.mean(), tag, m.sum()))
                continue
            mu = P[m, i].mean(0)
            top = np.sort(np.argsort(-mu)[:4])
            w = mu[top] / mu[top].sum()
            rec[tag] = {"experts": [int(v) for v in top],
                        "weights": [float(v) for v in w],
                        "n_steps": int(m.sum())}
            print("  %2d      e%-2d     %5.1f%%   %-4s  %-20s %s"
                  % (L, star[i], 100 * m_on.mean(), tag, list(top),
                     " ".join("%.3f" % v for v in w)))
        out["layers"][str(L)] = rec

    # how often does the real routing already equal each pin?
    print("\n  how far each pin is from the routing actually produced:")
    for i in FRONT:
        L = HB_LAYER[i]
        rec = out["layers"][str(L)]
        real = np.sort(np.argsort(-P[:, i], -1)[:, :4], -1)
        for tag in ("on", "off"):
            if tag not in rec:
                continue
            pin = np.array(rec[tag]["experts"])
            ov = np.array([len(np.intersect1d(r, pin)) for r in real[::17]])
            print("     L%-2d %-4s  overlap with the real top-4: %.2f / 4"
                  % (L, tag, ov.mean()))

    path = HERE / ("pin_%s_%s.json" % (suite, task[:28]))
    path.write_text(json.dumps(out, indent=2))
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
