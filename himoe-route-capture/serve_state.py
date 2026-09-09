"""Serve the real checkpoint with the four-tier HiMoEStateRecorder attached."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--suite", default="goal")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--libero-wrist-layout", default="checkpoint-right")
    ap.add_argument("--tier", default="expert")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-control-steps", type=int, default=40)
    args = ap.parse_args()

    from himoe_libero_bridge.server import PolicyServer, create_policy

    policy = create_policy("himoe", args.checkpoint_dir, args.upstream_root,
                           args.gpu, args.suite, args.libero_wrist_layout)
    from himoe_state_recorder import HiMoEStateRecorder, discover_blocks

    core = policy._policy.model
    blocks = discover_blocks(core)
    print("blocks:", [(b.layer_idx, b.kind, b.n_experts, b.top_k, b.has_shared)
                      for b in blocks], flush=True)
    rec = HiMoEStateRecorder(core, tier=args.tier).attach()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    acc, stats = [], {"n": 0, "infer": 0.0, "cap": 0.0, "done": False}
    inner = policy.infer

    def infer(observation):
        if stats["done"]:
            return inner(observation)
        rec.begin_control_step(0, stats["n"])
        t0 = time.perf_counter()
        response = inner(observation)
        t1 = time.perf_counter()
        r = rec.end_control_step()
        t2 = time.perf_counter()
        acc.append(r)
        stats["n"] += 1
        stats["infer"] += t1 - t0
        stats["cap"] += t2 - t1
        if stats["n"] >= args.max_control_steps:
            stats["done"] = True
            flat = {}
            for group in ("router", "block", "expert"):
                for k in getattr(acc[0], group):
                    flat["%s/%s" % (group, k)] = np.stack(
                        [getattr(a, group)[k] for a in acc])
            np.savez_compressed(out / "state.npz", **flat)
            json.dump({"control_steps": stats["n"],
                       "mean_infer_ms": 1000 * stats["infer"] / stats["n"],
                       "mean_capture_ms": 1000 * stats["cap"] / stats["n"],
                       "arrays": {k: list(v.shape) for k, v in flat.items()},
                       "bytes": int(sum(v.nbytes for v in flat.values()))},
                      open(out / "state_summary.json", "w"), indent=2)
            print("wrote %d control steps, %d arrays, %.1f KiB/step"
                  % (stats["n"], len(flat),
                     sum(v.nbytes for v in flat.values()) / stats["n"] / 1024),
                  flush=True)
        return response

    policy.infer = infer
    print("serving on ws://127.0.0.1:%d  tier=%s" % (args.port, args.tier), flush=True)
    PolicyServer(policy, "127.0.0.1", args.port, "himoe").serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
