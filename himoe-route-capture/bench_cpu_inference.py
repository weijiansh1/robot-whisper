"""How slow is one HiMoE-VLA control step on CPU?

The GPU path costs 717 ms per control step.  This measures the same call with
no CUDA device visible, so `policy_config.py:69` falls back to `torch.device("cpu")`.

Two things change besides speed, and both matter more than the timing:

  * `policy.py:71` opens `torch.autocast(device_type='cuda', ...)`, which is a
    no-op without CUDA, so the model runs in fp32 instead of bf16.  The AS-MoE
    layer-1 expert choice is decided by bf16 rounding (three logits collapse to
    0.25 and the top two differ by 1.22e-4), so CPU routing is not the same
    routing.
  * Nothing captured on CPU is comparable per-episode with any existing GPU
    capture.  Changing the MIG profile alone already flips 6/50 outcomes.

Run in the model env.  Reports per-call wall time and the projected cost of the
384-rollout prefix-commitment grid.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")

from himoe_libero_bridge.protocol import (  # noqa: E402
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
    IMAGE_KEY,
    IMAGE_SHAPE,
    PROMPT_KEY,
    STATE_DIM,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)

# Median control steps per episode over the 64-draw captures, and the grid size
# of run_prefix_commitment_k6.sh.
CONTROL_STEPS_PER_EPISODE = 21
GRID_ROLLOUTS = 384
GPU_MS_PER_CONTROL_STEP = 717.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--threads", type=int, default=0, help="0 keeps torch's default")
    ap.add_argument("--calls", type=int, default=3)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA is still visible; the fallback would not be exercised")
    if args.threads:
        torch.set_num_threads(args.threads)
    print("torch threads =", torch.get_num_threads(), flush=True)

    # policy.py:71 hardcodes device_type='cuda'.  Without CUDA torch disables
    # autocast entirely, and the AS gate then hits float32 weights with a bf16
    # data_mask: "expected m1 and m2 to have the same dtype".  Redirect the
    # context to the CPU autocast backend so the run gets far enough to time.
    # This is a measurement shim, not a proposed fix -- CPU bf16 autocast is a
    # different numerical path from CUDA bf16 autocast.
    _autocast = torch.autocast

    def _cpu_autocast(device_type: str, *a, **kw):
        if device_type == "cuda":
            device_type = "cpu"
        return _autocast(device_type, *a, **kw)

    torch.autocast = _cpu_autocast

    from himoe_libero_bridge.policies import HiMoEPolicy

    t0 = time.perf_counter()
    policy = HiMoEPolicy(
        checkpoint_dir=args.checkpoint_dir,
        suite="goal",
        upstream_root=args.upstream_root,
        require_cuda=False,
        libero_wrist_layout="released-left",
    )
    print("load: %.1f s" % (time.perf_counter() - t0), flush=True)

    param = next(policy._policy.model.parameters())
    print("model device =", param.device, " dtype =", param.dtype, flush=True)

    rng = np.random.default_rng(0)
    observation = {
        IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
        WRIST_IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
        STATE_KEY: rng.standard_normal(STATE_DIM).astype(np.float32) * 0.1,
        PROMPT_KEY: "open the middle drawer of the cabinet",
        FLOW_NOISE_KEY: rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32),
    }

    times = []
    for call in range(args.calls):
        t0 = time.perf_counter()
        policy.infer(dict(observation))
        dt = time.perf_counter() - t0
        times.append(dt)
        print("call %d: %.2f s" % (call, dt), flush=True)

    # The first call pays lazy-init costs that a long run amortises away.
    steady = float(np.median(times[1:])) if len(times) > 1 else times[0]
    print("\nsteady-state: %.2f s per control step (%.0fx the GPU's %.0f ms)"
          % (steady, steady * 1000 / GPU_MS_PER_CONTROL_STEP, GPU_MS_PER_CONTROL_STEP))
    episode_s = steady * CONTROL_STEPS_PER_EPISODE
    print("one episode (%d control steps): %.1f min" % (CONTROL_STEPS_PER_EPISODE, episode_s / 60))
    print("%d-rollout grid, single process: %.1f h"
          % (GRID_ROLLOUTS, GRID_ROLLOUTS * episode_s / 3600))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
