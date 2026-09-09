#!/usr/bin/env python3
"""Emit demo.json — every branch of the rolling-star run, plus one sibling pair's videos.

What the page shows
-------------------
All 352 branches (235 fail, 117 succeed), so the aggregate quoted on the page is exactly
the fan that is drawn.  Two of them are highlighted and carry rendered video: siblings
restored from a **bit-identical simulator state** (the run's own
`design.branch_noise_unique_q0_and_full_stream_within_every_snapshot`), differing only in
the recorded flow-noise stream — one succeeds, one does not.

The score
---------
Per control step, take the router probability vector at each (HB layer 12-15, suffix token)
site and compare it with the previous step's by top-8 Jaccard — set overlap only, so the
value is unchanged if every expert is renumbered.  The detector never reads *which* expert
is active, only whether it is the same one as before; the static identity turns out to be a
scene fingerprint (it recovers the init state at 99.6-100%) and carries almost no outcome
information once the group is held out strictly.

    r(t) = mean(last W distances) / mean(own first W0 distances)

Dividing by the branch's own opening makes r dimensionless, which is what lets a single
fixed threshold work with no per-group calibration — the same theta produces false-alarm
rates within 2.6 points on two independent corpora.

Alarm: K consecutive steps with (r_st + r_ac)/2 below THETA.

Alignment: one video frame is one action step, one control step is ten frames, so at 20 fps
a control step is 0.5 s of video.  Videos are remuxed with +faststart because the source
clips carry `moov` after `mdat`, which makes them unseekable over HTTP.

Usage:  python3 demo/build_demo_data.py
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
RS = (HERE.parent / "himoe-route-capture" / "runs"
      / "rolling-star-a100-long-t08-k16-20260828")

LAYERS, DENOISE, TOPK = slice(4, 8), 9, 8      # HB 12-15, denoise 9, top-8
W, W0, K, THETA = 4, 8, 3, 0.925
STOP = 37          # 等机会窗口上界(绝对 query 下标),= min(snapshot + T)
FPS, FRAMES_PER_STEP = 20, 10

# The fan is the whole run — 352 branches, 235 of which fail — so the aggregate quoted on
# the page describes exactly what is drawn.  A single snapshot will not do: w3/s000 has one
# failure in sixteen, w0/s000 has nine but neither of its two clips declines cleanly, and
# w1/s000 and w2/s000 are all-failure so they carry no contrast at all.
#
# The two highlighted branches are siblings from w3/s000, restored from a bit-identical
# simulator state and differing only in the flow-noise stream.  c02 steps down at q32 and
# holds a flat floor near 0.35 for the rest of the run — the deep-layer horizontal line
# this project started from.
HILITE = dict(worker=3, snapshot=0, success=7, failure=2)


def top_jaccard(P: np.ndarray) -> np.ndarray:
    """(T, sites, 32) probabilities -> (T-1,) top-8 set-overlap distance per step.

    Invariant under any fixed renumbering of the 32 experts: identity is used only to
    match a site against itself one step earlier, never as a label.
    """
    p, q = P[:-1], P[1:]
    mp = np.zeros(p.shape, bool)
    mq = np.zeros(q.shape, bool)
    np.put_along_axis(mp, np.argsort(-p, -1)[..., :TOPK], True, -1)
    np.put_along_axis(mq, np.argsort(-q, -1)[..., :TOPK], True, -1)
    inter = (mp & mq).sum(-1)
    union = np.maximum((mp | mq).sum(-1), 1)
    return (1.0 - inter / union).mean(-1)


def ratio(d: np.ndarray) -> list:
    """Trailing-W mean over the branch's own opening.  None before the window fills.

    Three decimals: the page only ever resolves this to a pixel, and 352 branches at four
    decimals is a needlessly large download.
    """
    base = d[:W0].mean()
    out = []
    for t in range(len(d) + 1):
        out.append(round(float(d[max(0, t - W):t].mean() / base), 3)
                   if t >= W and base > 0 else None)
    return out


def alarm(st: list, ac: list, offset: int) -> int | None:
    """First step with K consecutive readings below THETA, inside the decision window.

    The window closes at STOP in *absolute* query index — a branch spawned at trunk query
    `offset` covers [offset, offset + T).  STOP is the smallest such endpoint over the run,
    so every branch is still alive for the whole window and each gets the same number of
    chances to trip.  Letting the scan run past it hands failures more opportunities than
    successes (all 235 failures reach the horizon, successes end earlier) and detection
    climbs with the opportunity ratio whether or not the score carries signal: unbounded,
    this same detector reads 89.4% recall at 27.4% false alarms instead of 78.3% at 16.2%.
    """
    run = 0
    for t in range(2 * W0, len(st)):
        if offset + t >= STOP:
            break
        both = st[t] is not None and ac[t] is not None
        run = run + 1 if both and (st[t] + ac[t]) / 2 < THETA else 0
        if run >= K:
            return t
    return None


def videos(out: Path) -> dict:
    """Copy the two clips with the moov atom moved to the front so they can be seeked."""
    out.mkdir(exist_ok=True)
    found = {}
    for kind in ("success", "failure"):
        cand = HILITE[kind]
        src = next((RS / "formal" / f"worker{HILITE['worker']}" / "videos").glob(
            f"s{HILITE['snapshot']:03d}_c{cand:02d}_*.mp4"), None)
        if src is None:
            print(f"  ! no clip for c{cand:02d}")
            continue
        dst = out / f"{kind}.mp4"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                            "-c", "copy", "-movflags", "+faststart", str(dst)],
                           capture_output=True)
        if r.returncode:
            shutil.copy(src, dst)
            print(f"  ! remux failed for {src.name}, copied verbatim")
        found[kind] = f"video/{kind}.mp4"
        print(f"  {kind:8s} <- {src.name}")
    return found


def main() -> None:
    z = zarr.open(str(RS / "formal" / "server" / "routes.zarr"), mode="r")
    eid = np.asarray(z["episode_id"][:])
    row = np.asarray(z["control_step"][:])          # global row counter, not a query index

    labels = {int(r["episode_id"]): r for r in
              csv.DictReader(open(RS / "analysis" / "candidate_physical_labels.csv"))}
    print(f"{len(labels)} branches")

    flag = lambda r, k: str(r[k]).strip().lower() in ("true", "1")
    branches = []
    for e, r in sorted(labels.items()):
        m = np.where(eid == e)[0]
        if len(m) < 12:
            continue
        X = np.asarray(z["hb_router_probs"][m[np.argsort(row[m])], LAYERS, DENOISE, :, :],
                       dtype=np.float32)
        T = X.shape[0]
        st = ratio(top_jaccard(X[:, :, 0:1, :].reshape(T, -1, 32)))
        ac = ratio(top_jaccard(X[:, :, 1:, :].reshape(T, -1, 32)))
        w, s, cand = int(r["worker"]), int(r["snapshot"]), int(r["candidate"])
        ok = r["success"] == "True"
        here = (w, s) == (HILITE["worker"], HILITE["snapshot"])
        branches.append(dict(
            worker=w, snapshot=s, candidate=cand, T=T, success=ok,
            kind=("success" if ok else
                  "stagnation" if flag(r, "label_stagnation") else
                  "loop" if flag(r, "label_loop_or_cycling") else "other"),
            mean=[None if (a is None or b is None) else round((a + b) / 2, 3)
                  for a, b in zip(st, ac)],
            alarm=alarm(st, ac, s),
            video=("success" if here and cand == HILITE["success"] else
                   "failure" if here and cand == HILITE["failure"] else None)))

    fired = [b for b in branches if b["alarm"] is not None]
    fail = [b for b in branches if not b["success"]]
    ok_n = len(branches) - len(fail)
    hit = sum(1 for b in fired if not b["success"])
    fa = sum(1 for b in fired if b["success"])
    print(f"  {len(fail)} fail / {ok_n} succeed · {hit}/{len(fail)} caught "
          f"({hit / len(fail):.1%}) · {fa}/{ok_n} false alarms ({fa / ok_n:.1%}) · "
          f"{(hit + ok_n - fa) / len(branches):.1%} correct")

    print("videos")
    vid = videos(HERE / "video")

    doc = dict(
        config=dict(layers="HB 12-15", denoise=DENOISE, topk=TOPK, W=W, W0=W0, K=K,
                    theta=THETA, stop=STOP, fps=FPS, frames_per_step=FRAMES_PER_STEP),
        source=dict(run=RS.name, highlight=HILITE),
        corpus=dict(
            A=dict(n=352, accuracy=0.801, baseline=0.668, fa=0.162, recall=0.783),
            B=dict(n=512, accuracy=0.820, baseline=0.578, fa=0.176, recall=0.815)),
        videos=vid, branches=branches)
    (HERE / "demo.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"demo.json  {(HERE / 'demo.json').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
