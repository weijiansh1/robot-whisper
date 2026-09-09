#!/usr/bin/env python3
"""Post-capture review: prove the routing in the hub was recorded correctly.

Four checks per task, in increasing order of strength:

  1. shapes / row counts agree between routes.zarr, hidden.zarr and
     capture_summary.json, and control_step is contiguous
  2. the recorder's own online hook check left no failures
  3. AS routing really is loop-invariant (the schema collapses it, so if that
     ever stopped holding the stored array would be silently lossy)
  4. offline recompute: softmax(W_gate @ hidden) against the stored
     hb_router_probs, using the router weights read from the checkpoint

Check 4 is the one that cannot be faked.  Every array in routes.zarr is a
function of the router input, so if the hook had grabbed the wrong tensor the
probabilities would differ wholesale.  Requires --hidden captures; without them
the first three still run.

What the tolerance is, and why it is not f16
    The stored probabilities are float16, but they were *computed* in bfloat16:
    inference runs inside ``torch.autocast(dtype=torch.bfloat16)`` (policy.py),
    so the hook's softmax carries bf16 rounding before anything is stored.  bf16
    keeps 8 mantissa bits against f16's 11, so it, not the storage, sets the
    floor.  Measured on a real capture: recomputing in float32 differs from the
    stored probabilities by 0.0030, and re-rounding the hidden state by one f16
    ulp moves them only 0.0002 -- the gap is the model's own bf16 arithmetic.
    An earlier version of this script used 2x f16 eps and failed a good capture.

    The threshold is therefore bf16 eps.  It is three orders of magnitude below
    what a wrong tensor would produce: if the hook had captured something other
    than the router input, the probabilities would be unrelated, not off in the
    fourth decimal.

On top-4 disagreement
    Expect a few percent of routing sites to disagree on the top-4 *set* even
    when the probabilities match, for the same reason: where the 4th and 5th
    experts sit within rounding distance the order flips.  The check requires
    every disagreement to have a probability gap inside that band -- a
    disagreement with a real gap would be a genuine fault.

Usage
    ./verify_capture.py --run-id right-50x1
    ./verify_capture.py --run-id right-50x1 --suites libero_10 --steps 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
CAPTURE = HERE / "himoe-route-capture"
BRIDGE = pathlib.Path("/home/jovyan/.cache/himoe-libero-bridge")

CHECKPOINT_DIR = {
    "libero_goal": "HiMoE-VLA-Libero-Goal",
    "libero_spatial": "HiMoE-VLA-Libero-Spatial",
    "libero_object": "HiMoE-VLA-Libero-Object",
    "libero_10": "HiMoE-VLA-Libero-10",
}
F16_EPS = float(np.finfo(np.float16).eps)
#: bfloat16 has 8 mantissa bits: 2**-8.  numpy has no bfloat16, so it is spelled
#: out rather than read off finfo.
BF16_EPS = 2.0 ** -8


def hb_gate_weights(suite: str) -> np.ndarray:
    """The 8 HB router matrices from the suite's checkpoint -> [8, 32, 1024]."""
    import torch

    path = BRIDGE / "checkpoints" / CHECKPOINT_DIR[suite] / "pytorch_model.pth"
    sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    keys = [k for k in sd if k.endswith("mlp.gate.weight") and sd[k].shape[0] == 32]
    keys.sort(key=lambda k: int(re.search(r"layers\.(\d+)\.", k).group(1)))
    if len(keys) != 8:
        raise RuntimeError("expected 8 HB gates in %s, found %d" % (suite, len(keys)))
    return np.stack([sd[k].float().numpy() for k in keys])


def check_task(run: pathlib.Path, W: np.ndarray | None, n_steps: int) -> list[str]:
    import zarr

    bad: list[str] = []
    server = run / "server"

    # capture_summary.json is written by the server's shutdown handler, so its
    # absence means the server never flushed -- a SIGKILL rather than the SIGTERM
    # the driver sends.  That is exactly the case worth reporting, so report it
    # instead of dying on the read.
    try:
        summary = json.loads((server / "capture_summary.json").read_text())
    except (OSError, ValueError):
        return ["capture_summary.json missing or unreadable: the server never "
                "flushed, so this capture is truncated at an unknown point"]
    try:
        routes = zarr.open(str(server / "routes.zarr"), mode="r")
        n = routes["hb_expert_ids"].shape[0]
    except Exception as error:
        return ["routes.zarr unreadable: %s" % error]

    # 1 -- counts and contiguity
    if n != summary["control_steps"]:
        bad.append("routes has %d rows, capture_summary says %d"
                   % (n, summary["control_steps"]))
    step = np.asarray(routes["control_step"][:])
    if n and not np.array_equal(step, np.arange(step[0], step[0] + n)):
        bad.append("control_step is not contiguous -- a chunk was dropped")
    for name in ("hb_router_probs", "hb_expert_ids", "hb_selected_prob",
                 "hb_entropy", "as_expert_ids", "as_probs", "episode_id"):
        if name not in routes:
            bad.append("routes.zarr is missing %s" % name)
        elif routes[name].shape[0] != n:
            bad.append("%s has %d rows, expected %d" % (name, routes[name].shape[0], n))

    # 2 -- the recorder's online hook check
    if summary.get("hook_verify_failures"):
        bad.append("hook verification failed: %s" % summary["hook_verify_failures"][:2])
    if not summary.get("hook_verified_calls"):
        bad.append("no hook verification ran (old capture?)")

    # 3 -- AS collapse held for every control step
    if summary.get("as_collapsed") != summary["control_steps"]:
        bad.append("AS routing collapsed on only %s of %s control steps"
                   % (summary.get("as_collapsed"), summary["control_steps"]))

    # 4 -- offline recompute
    hidden_path = server / "hidden.zarr"
    if W is None or not hidden_path.exists():
        return bad
    hidden = zarr.open(str(hidden_path), mode="r")
    if hidden["hb_hidden"].shape[0] != n:
        bad.append("hidden.zarr has %d rows, routes has %d"
                   % (hidden["hb_hidden"].shape[0], n))
        return bad
    if not np.array_equal(np.asarray(hidden["control_step"][:]), step):
        bad.append("hidden.zarr and routes.zarr disagree on control_step")

    idx = np.unique(np.linspace(0, n - 1, min(n_steps, n)).astype(int))
    worst_prob, sites, mismatched, worst_gap = 0.0, 0, 0, 0.0
    for i in idx:
        h = np.asarray(hidden["hb_hidden"][int(i)], dtype=np.float32)
        logits = np.einsum("ldsh,leh->ldse", h, W)
        logits -= logits.max(-1, keepdims=True)
        e = np.exp(logits)
        mine = e / e.sum(-1, keepdims=True)
        theirs = np.asarray(routes["hb_router_probs"][int(i)], dtype=np.float32)
        worst_prob = max(worst_prob, float(np.abs(mine - theirs).max()))

        got = np.sort(np.argsort(-mine, -1)[..., :4], -1)
        want = np.sort(np.asarray(routes["hb_expert_ids"][int(i)]).astype(np.int64), -1)
        differs = (got != want).any(-1)
        sites += differs.size
        mismatched += int(differs.sum())
        if differs.any():
            srt = np.sort(mine, -1)[..., ::-1]
            worst_gap = max(worst_gap, float((srt[..., 3] - srt[..., 4])[differs].max()))

    if worst_prob > BF16_EPS:
        bad.append("recomputed probs differ by %.5f, above the %.5f bf16 band -- "
                   "the hook may not have captured the router's input"
                   % (worst_prob, BF16_EPS))
    if worst_gap > BF16_EPS:
        bad.append("a top-4 disagreement had a %.6f probability gap, outside bf16 "
                   "resolution -- not a tie" % worst_gap)
    return bad + ["_stats:%.5f:%d:%d:%.6f" % (worst_prob, mismatched, sites, worst_gap)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--suites", default=",".join(CHECKPOINT_DIR))
    ap.add_argument("--steps", type=int, default=4,
                    help="control steps per task to recompute (each is a 1.8 MB read)")
    ap.add_argument("--no-recompute", action="store_true",
                    help="skip check 4; no checkpoint load, seconds instead of minutes")
    args = ap.parse_args()
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]

    sys.path.insert(0, str(CAPTURE))
    import corpus_layout as cl

    failed = 0
    for suite in suites:
        base = HUB / "cache" / cl.HUB_MODEL / cl.HUB_DIR[suite]
        runs = sorted(p / args.run_id for p in base.iterdir()
                      if (p / args.run_id / "server" / "routes.zarr").exists())
        if not runs:
            print("%-16s no captures for run_id %r" % (suite, args.run_id))
            continue

        W = None
        if not args.no_recompute and any((r / "server" / "hidden.zarr").exists() for r in runs):
            print("%-16s loading router weights..." % suite, flush=True)
            W = hb_gate_weights(suite)

        print("%s  (%d task(s))" % (suite, len(runs)))
        for run in runs:
            issues = check_task(run, W, args.steps)
            stats = [i for i in issues if i.startswith("_stats:")]
            issues = [i for i in issues if not i.startswith("_stats:")]
            note = ""
            if stats:
                _, dp, mm, st, gap = stats[0].split(":")
                note = ("  max|dp|=%s  top4 differs %s/%s (max gap %s)"
                        % (dp, mm, st, gap))
            if issues:
                failed += 1
                print("  FAIL %s%s" % (run.parent.name[:52], note))
                for i in issues:
                    print("       %s" % i)
            else:
                print("  ok   %s%s" % (run.parent.name[:52], note))

    print("\n%d task(s) failed verification" % failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
