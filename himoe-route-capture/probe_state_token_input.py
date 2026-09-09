#!/usr/bin/env python3
"""What does the state token's routing actually read?

Structurally it can read two things and only two.  `embed_suffix` gives the
state token `att_masks = 1` and the ten action tokens `1, 0 x 9`, and
`make_att_2d_masks` lets a token attend only where the cumulative mask is <= its
own -- prefix 0, state 1, actions 2.  So the state token sees the image/language
prefix and itself, and is structurally blind to the action tokens.  That is why
its routing is bit-identical across flow-noise seeds: the noise enters through
`noisy_actions`, which lives downstream of it.

That leaves the question of which of the two remaining inputs moves it.  Across
the 16 initial states of a capture both the images and the proprioceptive vector
differ, so the observation cannot separate them.  This does, by holding one fixed
and varying the other:

    A  real proprio from each of the 16 scenes, one fixed image  -> proprio only
    B  one fixed proprio, several different images               -> image only

against the spread actually observed in the capture, where both varied.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from himoe_libero_bridge.protocol import (  # noqa: E402
    FLOW_NOISE_KEY, FLOW_NOISE_SHAPE, IMAGE_KEY, IMAGE_SHAPE, PROMPT_KEY,
    STATE_DIM, STATE_KEY, WRIST_IMAGE_KEY,
)

HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--proprio", required=True,
                    help="npy of shape [n_scene, 8]: the real step-0 states")
    ap.add_argument("--images", type=int, default=6)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(args.threads)
    from probe_flow_lead_cpu import install_cpu_autocast
    install_cpu_autocast(torch)

    from himoe_libero_bridge.policies import HiMoEPolicy
    from himoe_router_recorder import HiMoERouteRecorder, discover_gates

    policy = HiMoEPolicy(checkpoint_dir=args.checkpoint_dir, suite=args.suite,
                         upstream_root=args.upstream_root, require_cuda=False,
                         libero_wrist_layout="paper-right")
    core = policy._policy.model
    gates = discover_gates(core)
    prop = np.load(args.proprio).astype(np.float32)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
    imgs = [(rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
             rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8))
            for _ in range(args.images)]

    def route(image, wrist, state):
        rec = HiMoERouteRecorder(core, store_full_probs=True, verify_steps=0).attach()
        rec.begin_control_step(episode_id=0, control_step=0)
        policy.infer({IMAGE_KEY: image, WRIST_IMAGE_KEY: wrist,
                      STATE_KEY: state, PROMPT_KEY: args.prompt,
                      FLOW_NOISE_KEY: noise.copy()})
        r = rec.end_control_step()
        rec.close()
        p = np.asarray(r.hb_router_probs, np.float32)
        if p.ndim == 5:
            p = p[0]
        return p[:, 0, 0]                       # [8, 32] state token, denoise 0

    print("A. one fixed image, the %d real per-scene proprio vectors" % len(prop))
    A = np.stack([route(imgs[0][0], imgs[0][1], prop[i]) for i in range(len(prop))])
    print("B. one fixed proprio, %d different images" % len(imgs))
    B = np.stack([route(im, wr, prop[0]) for im, wr in imgs])

    print("\nlargest change in the state token's router probability, per layer")
    print("layer   proprio varies (image fixed)   image varies (proprio fixed)")
    for i, L in enumerate(HB_LAYER):
        da = np.abs(A[:, i] - A[0, i]).max()
        db = np.abs(B[:, i] - B[0, i]).max()
        print("  %2d           %.5f                        %.5f" % (L, da, db))
    print("\n  for reference, the spread actually seen across the 16 captured")
    print("  scenes (where both varied) was 0.020-0.061 depending on the layer")
    print("\n  proprio dims that vary here: %s"
          % np.array2string(prop.max(0) - prop.min(0), precision=6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
