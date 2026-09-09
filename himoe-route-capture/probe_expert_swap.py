"""Are the HB experts functionally interchangeable, or is the routing merely fragile?

Two mechanisms are still consistent with everything measured so far and they carry
very different names:

    sparse random-feature ensemble    the experts are interchangeable in this
                                      operating regime, so which four are selected
                                      changes the realisation and not the function
    near-tie fragile assignment       the experts compute genuinely different things
                                      and a quantisation-scale gate decides which of
                                      them runs -- routing is unreadable AND consequential

The earlier near-tie probe swapped ONE of the four selected experts and looked only
at the final action.  That cannot separate the two: a 4% action change is small
either because the experts agree or because one quarter of a 9%-of-residual branch
is small.  This replaces the whole selected set and measures the effect at three
points along the causal chain:

    dy_MoE    the MoE block output, where the swap acts directly
    dv        the flow velocity that block feeds
    dA        the final action chunk, after 10 Euler steps and 8 blocks

Arms, all with the four renormalised weights left untouched so only identity moves:

    none          control; must reproduce the baseline exactly
    swap1_near    the weakest selected expert -> the 5th ranked  (a swap the deployed
                  model could have made itself: at 24% of sites their bf16 scores are
                  bit-identical)
    swap1_random  the weakest selected -> a uniformly drawn non-selected expert
    swap4_next    all four -> ranks 5-8
    swap4_random  all four -> four uniformly drawn non-selected experts

The decisive contrast is swap4_random.  If replacing every selected expert with a
random one leaves y_MoE largely intact, the experts are interchangeable and the
strong wording is earned.  If y_MoE moves a lot while A barely does, the experts
differ and the branch is simply too small to matter -- a different claim.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")

from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
    IMAGE_KEY,
    IMAGE_SHAPE,
    PROMPT_KEY,
    STATE_DIM,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)

MODES = ("none", "swap1_near", "swap1_random", "swap4_next", "swap4_random")


class SwapProbe:
    """Gate mirror with a swap, plus taps on the MoE output and the flow velocity."""

    def __init__(self, core, blocks, torch, mode="none", seed=0):
        self.torch = torch
        self.mode = mode
        self.core = core
        self.blocks = blocks
        self.generator = torch.Generator().manual_seed(seed)
        self.moe_out: list[np.ndarray] = []
        self.velocity: list[np.ndarray] = []
        self.gate_orig, self.block_orig = {}, {}
        for name, block in blocks:
            self.gate_orig[name] = block.gate.forward
            block.gate.forward = self._gate(block.gate)
            self.block_orig[name] = block.forward
            block.forward = self._block(name, block)
        self.denoise_orig = core.denoise_step

        def traced(*a, **k):
            v = self.denoise_orig(*a, **k)
            self.velocity.append(v.detach().to("cpu", copy=True).float().numpy())
            return v

        core.denoise_step = traced

    def _gate(self, gate):
        torch = self.torch
        functional = torch.nn.functional

        def forward(hidden_states):
            _, _, h = hidden_states.shape
            flat = hidden_states.view(-1, h)
            scores = functional.linear(flat, gate.weight, None).softmax(dim=-1)
            topk_weight, topk_idx = torch.topk(scores, k=gate.top_k, dim=-1, sorted=False)
            if self.mode != "none":
                topk_idx = self._swap(scores, topk_idx)
            if gate.top_k > 1 and gate.norm_topk_prob:
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            return topk_idx, topk_weight, None

        return forward

    def _swap(self, scores, topk_idx):
        torch = self.torch
        masked = scores.clone().float()
        masked.scatter_(-1, topk_idx.long(), -float("inf"))
        out = topk_idx.clone()
        if self.mode.startswith("swap1"):
            selected = torch.take_along_dim(scores, topk_idx, dim=-1)
            slot = selected.argmin(dim=-1, keepdim=True)
            if self.mode == "swap1_near":
                replacement = masked.argmax(dim=-1, keepdim=True)
            else:
                replacement = torch.multinomial(
                    (masked > -float("inf")).float(), 1, generator=self.generator)
            out.scatter_(-1, slot, replacement.to(out.dtype))
            return out
        # replace the whole selected set
        if self.mode == "swap4_next":
            replacement = masked.topk(topk_idx.shape[-1], dim=-1).indices
        else:
            replacement = torch.multinomial(
                (masked > -float("inf")).float(), topk_idx.shape[-1],
                replacement=False, generator=self.generator)
        return replacement.to(out.dtype)

    def _block(self, name, block):
        torch = self.torch

        def forward(hidden_states, data_mask):
            identity = hidden_states
            shape = hidden_states.shape
            topk_idx, topk_weight, _ = block.gate(hidden_states)
            flat = hidden_states.view(-1, shape[-1])
            routed = block.moe_infer(
                flat, topk_idx.view(-1), topk_weight.view(-1, 1)).view(*shape)
            y = routed + block.shared_experts(identity)
            with torch.no_grad():
                self.moe_out.append(y.detach().to("cpu", copy=True).float().numpy())
            return y

        return forward

    def restore(self):
        for name, block in self.blocks:
            block.gate.forward = self.gate_orig[name]
            block.forward = self.block_orig[name]
        self.core.denoise_step = self.denoise_orig


def relative(a, b):
    """||a-b|| / ||b||, over concatenated captures."""
    x, y = np.concatenate([v.ravel() for v in a]), np.concatenate([v.ravel() for v in b])
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--suite", default="goal",
                    help="must match the checkpoint; the bridge validates it")
    ap.add_argument("--observations", type=int, default=4)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/expert-swap/summary.json")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(args.threads)
    from probe_flow_lead_cpu import install_cpu_autocast

    install_cpu_autocast(torch)
    from himoe_libero_bridge.policies import HiMoEPolicy

    policy = HiMoEPolicy(
        checkpoint_dir=args.checkpoint_dir, suite=args.suite,
        upstream_root=args.upstream_root, require_cuda=False,
        libero_wrist_layout="checkpoint-right",
    )
    core = policy._policy.model
    blocks = [(f"layer{i}", layer.mlp)
              for i, layer in enumerate(core.paligemma_with_expert.gemma_expert.layers)
              if type(layer.mlp).__name__ == "HBMoE"]
    print("HB blocks: %d" % len(blocks), flush=True)

    rng = np.random.default_rng(args.seed)
    collected = {mode: {"y": [], "v": [], "a": []} for mode in MODES}
    for index in range(args.observations):
        request = {
            IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
            WRIST_IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
            STATE_KEY: (rng.standard_normal(STATE_DIM) * 0.1).astype(np.float32),
            PROMPT_KEY: "open the middle drawer of the cabinet",
            FLOW_NOISE_KEY: rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32),
        }
        for mode in MODES:
            probe = SwapProbe(core, blocks, torch, mode=mode, seed=args.seed + index)
            action = np.asarray(policy.infer(dict(request))[ACTION_KEY], np.float32)
            probe.restore()
            collected[mode]["y"].append(probe.moe_out)
            collected[mode]["v"].append(probe.velocity)
            collected[mode]["a"].append(action)
        print("  observation %d done" % index, flush=True)

    print("\n=== relative change against the unswapped model ===")
    print("  arm             dy_MoE      dv        dA        (||.||2 relative)")
    summary = {}
    for mode in MODES:
        y = float(np.mean([relative(collected[mode]["y"][i], collected["none"]["y"][i])
                           for i in range(args.observations)]))
        v = float(np.mean([relative(collected[mode]["v"][i], collected["none"]["v"][i])
                           for i in range(args.observations)]))
        a = float(np.mean([relative([collected[mode]["a"][i]], [collected["none"]["a"][i]])
                           for i in range(args.observations)]))
        summary[mode] = {"dy_moe": y, "dv": v, "dA": a}
        print("  %-13s  %8.4f  %8.4f  %8.4f" % (mode, y, v, a), flush=True)

    if summary["none"]["dA"] != 0.0:
        print("\nWARNING: the 'none' arm is not bit-identical; every other row includes drift")

    strong = summary["swap4_random"]["dy_moe"]
    print("\nverdict input: replacing ALL FOUR selected experts with random ones moves")
    print("  the MoE block output by %.1f%% and the final action by %.1f%%"
          % (100 * strong, 100 * summary["swap4_random"]["dA"]))
    print("  small dy_MoE  => experts are interchangeable: 'sparse random-feature ensemble'")
    print("  large dy_MoE with small dA => experts differ, the branch is just small:")
    print("     'near-degenerate routing over a consequential-but-small branch'")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
