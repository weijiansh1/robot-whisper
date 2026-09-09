"""Layer 3 of the MoE decomposition: what do the experts actually compute?

The first two layers are already dead on this checkpoint -- which experts are
selected is near-arbitrary (24% of sites have a bit-identical 4th/5th bf16 score,
and a random substitution costs 4% of the action), and their weights are within
1.5 percentage points of uniform.  Neither says anything about E_e(h) itself.

Two very different worlds are still consistent with everything measured so far,
and they call for opposite conclusions about the architecture:

    routed branch is tiny        routing cannot matter regardless of how different
                                 the experts are; MoE is decoration on the shared
                                 expert
    experts are interchangeable  the branch may be large, but every expert computes
                                 nearly the same function, so selecting among them
                                 is sampling noise around one expectation

`HBMoE.forward` computes `y = moe_infer(...)` and then `y = y + shared_experts(identity)`,
so a per-instance mirror separates the two exactly rather than by estimation.  The
mirror is checked against the original module output before anything is recorded.

The decisive statistic is the last one: if the mean of the four SELECTED experts is
no closer to their own output than the mean of four RANDOM experts is, then the
gate is choosing among functionally equivalent options and the routed branch is an
ensemble average, not a decision.
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
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
    IMAGE_KEY,
    IMAGE_SHAPE,
    PROMPT_KEY,
    STATE_DIM,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)


class ContributionProbe:
    """Mirrors HBMoE.forward, keeping the routed and shared branches apart."""

    def __init__(self, blocks, torch, sample_tokens=4, seed=0):
        self.torch = torch
        self.blocks = blocks
        self.records: list[dict] = []
        self.diversity: list[dict] = []
        self.sample_tokens = sample_tokens
        self.rng = np.random.default_rng(seed)
        self.originals = {}
        for name, block in blocks:
            self.originals[name] = block.forward
            block.forward = self._make(name, block)

    def _make(self, name, block):
        torch = self.torch

        def forward(hidden_states, data_mask):
            identity = hidden_states
            orig_shape = hidden_states.shape
            topk_idx, topk_weight, _aux = block.gate(hidden_states)
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            routed = block.moe_infer(
                flat, topk_idx.view(-1), topk_weight.view(-1, 1)).view(*orig_shape)
            shared = block.shared_experts(identity)

            with torch.no_grad():
                self.records.append({
                    "block": name,
                    "routed": routed.detach().float().reshape(-1, orig_shape[-1]).norm(dim=-1).cpu().numpy(),
                    "shared": shared.detach().float().reshape(-1, orig_shape[-1]).norm(dim=-1).cpu().numpy(),
                    "identity": identity.detach().float().reshape(-1, orig_shape[-1]).norm(dim=-1).cpu().numpy(),
                })
                self._diversity(name, block, flat, topk_idx, topk_weight)

            return routed + shared

        return forward

    def _diversity(self, name, block, flat, topk_idx, topk_weight):
        """Run ALL experts on a few tokens to see how much they differ at all."""
        torch = self.torch
        n_tokens = flat.shape[0]
        picks = self.rng.choice(n_tokens, min(self.sample_tokens, n_tokens), replace=False)
        for token in picks:
            h = flat[token: token + 1]
            outputs = torch.stack([expert(h).float().squeeze(0) for expert in block.experts])
            selected = topk_idx.view(-1, topk_idx.shape[-1])[token].detach().cpu().numpy()
            weights = topk_weight.view(-1, topk_weight.shape[-1])[token].detach().float().cpu().numpy()
            self.diversity.append({
                "block": name,
                "outputs": outputs.cpu().numpy(),
                "selected": selected.astype(int),
                "weights": weights,
            })

    def restore(self):
        for name, block in self.blocks:
            block.forward = self.originals[name]


def summarise_diversity(entries):
    """Cosine spread between experts, and selected-vs-random subset behaviour."""
    cosines, selected_gap, random_gap, weighted_gap = [], [], [], []
    rng = np.random.default_rng(0)
    for entry in entries:
        outputs = entry["outputs"]                      # [32, d]
        norms = np.linalg.norm(outputs, axis=-1, keepdims=True)
        unit = outputs / np.maximum(norms, 1e-12)
        similarity = unit @ unit.T
        upper = np.triu_indices(len(outputs), 1)
        cosines.append(similarity[upper])

        pool_mean = outputs.mean(0)
        scale = np.linalg.norm(pool_mean) + 1e-12
        chosen = outputs[entry["selected"]]
        selected_gap.append(np.linalg.norm(chosen.mean(0) - pool_mean) / scale)
        # the deployed combination is weighted, not a plain mean
        weighted_gap.append(
            np.linalg.norm((chosen * entry["weights"][:, None]).sum(0) - pool_mean) / scale)
        others = rng.choice(len(outputs), len(entry["selected"]), replace=False)
        random_gap.append(np.linalg.norm(outputs[others].mean(0) - pool_mean) / scale)

    cosines = np.concatenate(cosines)
    return {
        "pairwise_cosine_mean": float(cosines.mean()),
        "pairwise_cosine_p5": float(np.percentile(cosines, 5)),
        "pairwise_cosine_p95": float(np.percentile(cosines, 95)),
        "selected4_vs_pool_mean": float(np.mean(selected_gap)),
        "random4_vs_pool_mean": float(np.mean(random_gap)),
        "weighted_selected_vs_pool_mean": float(np.mean(weighted_gap)),
        "selected_over_random": float(np.mean(selected_gap) / max(np.mean(random_gap), 1e-12)),
        "n_tokens": len(entries),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--queries", type=int, default=2)
    ap.add_argument("--sample-tokens", type=int, default=4)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/expert-contribution/summary.json")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(args.threads)
    from probe_flow_lead_cpu import install_cpu_autocast

    install_cpu_autocast(torch)

    from himoe_libero_bridge.policies import HiMoEPolicy
    from himoe_libero_bridge.protocol import ACTION_KEY

    policy = HiMoEPolicy(
        checkpoint_dir=args.checkpoint_dir, suite="goal",
        upstream_root=args.upstream_root, require_cuda=False,
        libero_wrist_layout="checkpoint-right",
    )
    core = policy._policy.model
    blocks = [(f"layer{i}", layer.mlp)
              for i, layer in enumerate(core.paligemma_with_expert.gemma_expert.layers)
              if type(layer.mlp).__name__ == "HBMoE"]
    print("HB blocks: %d" % len(blocks), flush=True)

    rng = np.random.default_rng(args.seed)

    def observation():
        return {
            IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
            WRIST_IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
            STATE_KEY: (rng.standard_normal(STATE_DIM) * 0.1).astype(np.float32),
            PROMPT_KEY: "open the middle drawer of the cabinet",
            FLOW_NOISE_KEY: rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32),
        }

    reference = observation()
    baseline = np.asarray(policy.infer(dict(reference))[ACTION_KEY], np.float32)
    probe = ContributionProbe(blocks, torch, args.sample_tokens, args.seed)
    mirrored = np.asarray(policy.infer(dict(reference))[ACTION_KEY], np.float32)
    drift = float(np.abs(mirrored - baseline).max())
    if drift != 0.0:
        probe.restore()
        raise RuntimeError("mirrored HBMoE changed the action by %.3e; not faithful" % drift)
    print("mirror check: action bit-identical to the original module", flush=True)

    probe.records, probe.diversity = [], []
    for _ in range(args.queries):
        policy.infer(observation())
    probe.restore()

    print("\n=== branch magnitudes (norm per token) ===")
    print("  block     routed/shared   routed/identity   routed/(routed+shared)")
    per_block = {}
    for name, _ in blocks:
        rows = [r for r in probe.records if r["block"] == name]
        routed = np.concatenate([r["routed"] for r in rows])
        shared = np.concatenate([r["shared"] for r in rows])
        identity = np.concatenate([r["identity"] for r in rows])
        stat = {
            "routed_over_shared": float(np.median(routed / np.maximum(shared, 1e-12))),
            "routed_over_identity": float(np.median(routed / np.maximum(identity, 1e-12))),
            "routed_share": float(np.median(routed / np.maximum(routed + shared, 1e-12))),
        }
        per_block[name] = stat
        print("  %-8s  %12.4f  %16.4f  %20.4f"
              % (name, stat["routed_over_shared"], stat["routed_over_identity"],
                 stat["routed_share"]), flush=True)

    print("\n=== expert diversity (all 32 experts run on sampled tokens) ===")
    diversity = {}
    for name, _ in blocks:
        entries = [d for d in probe.diversity if d["block"] == name]
        if not entries:
            continue
        stat = summarise_diversity(entries)
        diversity[name] = stat
        print("  %-8s cos(E_i,E_j) mean %+.3f [p5 %+.3f, p95 %+.3f]   "
              "selected4 %.4f vs random4 %.4f  ratio %.3f"
              % (name, stat["pairwise_cosine_mean"], stat["pairwise_cosine_p5"],
                 stat["pairwise_cosine_p95"], stat["selected4_vs_pool_mean"],
                 stat["random4_vs_pool_mean"], stat["selected_over_random"]), flush=True)

    ratios = [d["selected_over_random"] for d in diversity.values()]
    print("\nselected/random deviation from the 32-expert mean: %.3f averaged over blocks"
          % float(np.mean(ratios)))
    print("  ~1.0 => the gate picks a subset no more special than a random one")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"branches": per_block, "diversity": diversity}, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
