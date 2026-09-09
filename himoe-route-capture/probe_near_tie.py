"""Near-tie audit of the HB gate: is the Top-4 boundary decided by preference or by rounding?

`router_audit.py` showed the action-token router is degenerate in aggregate --
M_4 never exceeds 0.185 over 240k sites and the 4th-vs-5th gap D_45 sits within 5
bf16 ulp at 86% of sites.  That makes one question decisive before any further
routing work: when the model picks four experts out of thirty-two, is it
expressing a preference, or breaking a tie?

The three things that separate those, all measured on the exact tensor `topk`
consumes rather than on a recomputation:

    n_unique      distinct representable score values among the 32.  If this is
                  well below 32, many experts are literally indistinguishable at
                  the gate's dtype and the selection among them is arbitrary.
    argsort set   a deterministic ordering of the SAME scores.  Disagreement with
                  the model's set is tie-breaking, not precision -- both see
                  identical numbers.
    fp32 shadow   the same hidden state and weights at fp32.  Disagreement here is
                  preference that the deployed dtype quantised away.

The gate's forward is replaced by a line-for-line mirror that stashes `scores`
before the topk call, so the captured tensor is the one selected from, not an
approximation of it.  The mirror is checked against the original gate on the same
input before anything is recorded.

Device caveat: dtype behaviour is what this measures, and dtype is set by autocast
rather than by the device, so the tie structure transfers.  What does NOT transfer
is kernel-level tie-breaking order -- "same card repeated / different MIG /
different device" needs GPU and is deliberately out of scope here.
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


class TieProbe:
    """Mirrors MoEGate_load_bal.forward, stashing the tensor topk consumes.

    ``intervene`` swaps the lowest-scoring SELECTED expert for another one while
    leaving the four renormalised weights untouched, which isolates expert
    *identity* from expert *weight*:

        nearest   the 5th-ranked expert -- at 24% of sites its score is bit-identical
                  to the 4th's, so this is a swap the deployed model could equally
                  well have made itself
        random    a uniformly drawn non-selected expert, as the scale control.  If
                  nearest and random move the output by the same amount, rank order
                  below the cut carries no function.
    """

    def __init__(self, gates, torch, intervene="none", seed=0):
        self.torch = torch
        self.records: list[dict] = []
        self.originals = {}
        self.intervene = intervene
        self.generator = torch.Generator().manual_seed(seed)
        self.swapped = 0
        for name, gate in gates:
            self.originals[name] = gate.forward
            gate.forward = self._make(name, gate)

    def _make(self, name, gate):
        torch = self.torch
        functional = torch.nn.functional

        def forward(hidden_states):
            bsz, seq_len, h = hidden_states.shape
            flat = hidden_states.view(-1, h)
            logits = functional.linear(flat, gate.weight, None)
            scores = logits.softmax(dim=-1)
            topk_weight, topk_idx = torch.topk(scores, k=gate.top_k, dim=-1, sorted=False)

            with torch.no_grad():
                # fp32 shadow: same hidden state, same weights, genuinely at fp32.
                # `enabled=False` is mandatory -- autocast intercepts F.linear and
                # runs it in bf16 no matter what dtype the inputs are cast to, so
                # without this the "shadow" is just the deployed computation again
                # and trivially agrees with it.
                with torch.autocast("cpu", enabled=False):
                    shadow_logits = functional.linear(flat.float(), gate.weight.float(), None)
                    shadow = shadow_logits.softmax(dim=-1)
                if shadow.dtype != torch.float32:
                    raise RuntimeError("shadow gate ran at %s, not fp32" % shadow.dtype)
                self.records.append({
                    "gate": name,
                    "dtype": str(scores.dtype),
                    "shadow_dtype": str(shadow.dtype),
                    "scores": scores.detach().cpu().float().numpy(),
                    "shadow": shadow.detach().cpu().float().numpy(),
                    "model_idx": topk_idx.detach().cpu().numpy(),
                    # count distinct values in the score dtype, before any upcast
                    "n_unique": np.array([
                        len(np.unique(row)) for row in
                        scores.detach().cpu().view(torch.int16 if scores.dtype
                                                   in (torch.bfloat16, torch.float16)
                                                   else torch.int32).numpy()]),
                })

            if self.intervene != "none":
                topk_idx = self._swap(scores, topk_idx, gate)
            if gate.top_k > 1 and gate.norm_topk_prob:
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            return topk_idx, topk_weight, None

        return forward

    def _swap(self, scores, topk_idx, gate):
        """Replace the weakest selected expert; weights are deliberately untouched."""
        torch = self.torch
        selected = torch.take_along_dim(scores, topk_idx, dim=-1)
        weakest = selected.argmin(dim=-1, keepdim=True)          # slot to overwrite
        masked = scores.clone().float()
        masked.scatter_(-1, topk_idx.long(), -float("inf"))       # exclude the chosen four
        if self.intervene == "nearest":
            replacement = masked.argmax(dim=-1, keepdim=True)     # the 5th-ranked expert
        else:
            weights = (masked > -float("inf")).float()
            replacement = torch.multinomial(weights, 1, generator=self.generator)
        out = topk_idx.clone()
        out.scatter_(-1, weakest, replacement.to(out.dtype))
        self.swapped += int(out.shape[0])
        return out

    def restore(self, gates):
        for name, gate in gates:
            gate.forward = self.originals[name]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--suite", default="goal",
                    help="must match the checkpoint; the bridge validates it")
    ap.add_argument("--queries", type=int, default=3)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/near-tie/summary.json")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(args.threads)
    from probe_flow_lead_cpu import install_cpu_autocast

    install_cpu_autocast(torch)

    from himoe_libero_bridge.policies import HiMoEPolicy
    from himoe_router_recorder import discover_gates

    policy = HiMoEPolicy(
        checkpoint_dir=args.checkpoint_dir, suite=args.suite,
        upstream_root=args.upstream_root, require_cuda=False,
        libero_wrist_layout="checkpoint-right",
    )
    core = policy._policy.model
    hb = [(g.name, _resolve(core, g.name)) for g in discover_gates(core) if g.kind == "HB"]
    print("HB gates: %d" % len(hb), flush=True)

    rng = np.random.default_rng(args.seed)

    def observation():
        return {
            IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
            WRIST_IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
            STATE_KEY: (rng.standard_normal(STATE_DIM) * 0.1).astype(np.float32),
            PROMPT_KEY: "open the middle drawer of the cabinet",
            FLOW_NOISE_KEY: rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32),
        }

    # faithfulness check: the mirror must reproduce the original gate's selection
    reference = observation()
    baseline = _collect_idx(policy, dict(reference), core)
    probe = TieProbe(hb, torch)
    mirrored = _collect_idx(policy, dict(reference), core)
    if not _sets_equal(baseline, mirrored):
        probe.restore(hb)
        raise RuntimeError("the mirrored gate does not reproduce the original selection")
    print("mirror check: identical Top-4 sets on the reference query", flush=True)

    probe.records = []
    for _ in range(args.queries):
        policy.infer(observation())
    probe.restore(hb)

    scores = np.concatenate([r["scores"] for r in probe.records])
    shadow = np.concatenate([r["shadow"] for r in probe.records])
    model_idx = np.concatenate([r["model_idx"] for r in probe.records])
    n_unique = np.concatenate([r["n_unique"] for r in probe.records])
    dtype = probe.records[0]["dtype"]

    argsort_set = np.sort(np.argsort(-scores, axis=-1)[:, :4], axis=-1)
    shadow_set = np.sort(np.argsort(-shadow, axis=-1)[:, :4], axis=-1)
    model_set = np.sort(model_idx.astype(np.int64), axis=-1)

    ordered = np.sort(scores, axis=-1)[:, ::-1]
    gap = ordered[:, 3] - ordered[:, 4]
    # how many experts sit exactly at the 4th-place value: >1 means the boundary
    # is a tie and the selection among them is whatever the kernel happened to do
    boundary_ties = (scores == ordered[:, 3:4]).sum(-1)

    summary = {
        "score_dtype": dtype,
        "n_sites": int(scores.shape[0]),
        "n_unique_of_32": {"mean": float(n_unique.mean()), "p5": float(np.percentile(n_unique, 5)),
                           "median": float(np.median(n_unique)), "max": int(n_unique.max())},
        "model_vs_argsort_disagree": float(np.mean(np.any(model_set != argsort_set, axis=-1))),
        "model_vs_fp32shadow_disagree": float(np.mean(np.any(model_set != shadow_set, axis=-1))),
        "argsort_vs_fp32shadow_disagree": float(np.mean(np.any(argsort_set != shadow_set, axis=-1))),
        "boundary_tie_rate": float(np.mean(boundary_ties > 1)),
        "boundary_tie_multiplicity_mean": float(boundary_ties.mean()),
        "gap_exactly_zero_rate": float(np.mean(gap == 0)),
        "gap_median": float(np.median(gap)),
    }

    print("\n=== near-tie audit (%d sites, scores are %s) ===" % (summary["n_sites"], dtype))
    print("distinct score values out of 32: mean %.1f  median %.0f  p5 %.0f  max %d"
          % (summary["n_unique_of_32"]["mean"], summary["n_unique_of_32"]["median"],
             summary["n_unique_of_32"]["p5"], summary["n_unique_of_32"]["max"]))
    print("4th-place value shared by >1 expert (an exact tie at the boundary): %.2f%%"
          % (100 * summary["boundary_tie_rate"]))
    print("  mean multiplicity at the 4th place: %.3f" % summary["boundary_tie_multiplicity_mean"])
    print("  D_45 exactly zero: %.2f%%" % (100 * summary["gap_exactly_zero_rate"]))
    print("\nTop-4 SET disagreement:")
    print("  model topk   vs deterministic argsort of the same scores : %.2f%%   <- tie-breaking"
          % (100 * summary["model_vs_argsort_disagree"]))
    print("  model topk   vs fp32 shadow gate                         : %.2f%%   <- quantisation"
          % (100 * summary["model_vs_fp32shadow_disagree"]))
    print("  argsort      vs fp32 shadow gate                         : %.2f%%"
          % (100 * summary["argsort_vs_fp32shadow_disagree"]))

    # --- functional half: does swapping a tied expert change anything? ---------
    print("\n=== intervention: swap the weakest selected expert (weights kept) ===")
    reference_obs = observation()
    baseline_action = _final_action(policy, dict(reference_obs))
    rows = {}
    for mode in ("none", "nearest", "random"):
        swap = TieProbe(hb, torch, intervene=mode, seed=args.seed)
        swap.records = []
        action = _final_action(policy, dict(reference_obs))
        swap.restore(hb)
        # relative L2, NOT max|dA| / mean|A|.  The max is a single worst element of
        # the chunk and is wildly sensitive to outliers: on the spatial checkpoint it
        # read 36.2% for an intervention whose relative L2 is 0.69%, which briefly
        # looked like a real cross-checkpoint difference.  probe_expert_swap.py uses
        # the same relative L2, so the two are directly comparable.
        delta = float(np.linalg.norm(action - baseline_action)
                      / max(np.linalg.norm(baseline_action), 1e-9))
        rows[mode] = {"relative_l2": delta, "sites_swapped": swap.swapped}
        print("  %-8s  relative L2 %.4f   (%d sites touched)"
              % (mode, delta, swap.swapped), flush=True)
    summary["intervention"] = rows
    if rows["none"]["relative_l2"] != 0.0:
        print("  WARNING: the 'none' arm did not reproduce the baseline, so the other "
              "two deltas include that much run-to-run drift")
    ratio = rows["nearest"]["relative_l2"] / max(rows["random"]["relative_l2"], 1e-12)
    print("  nearest / random = %.3f   (~1 => rank order below the cut carries no "
          "function; <<1 => the ranking is functional)" % ratio)
    summary["nearest_over_random"] = ratio

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print("\nwrote %s" % out)
    return 0


def _final_action(policy, observation):
    from himoe_libero_bridge.protocol import ACTION_KEY

    return np.asarray(policy.infer(observation)[ACTION_KEY], np.float32)


def _resolve(root, dotted):
    node = root
    for part in dotted.split("."):
        node = getattr(node, part) if not part.isdigit() else node[int(part)]
    return node


def _collect_idx(policy, observation, core):
    from himoe_router_recorder import HiMoERouteRecorder

    recorder = HiMoERouteRecorder(core, store_full_probs=False).attach()
    recorder.begin_control_step(episode_id=0, control_step=0)
    policy.infer(observation)
    record = recorder.end_control_step()
    recorder.close()
    return np.asarray(record.hb_expert_ids)


def _sets_equal(a, b):
    return np.array_equal(np.sort(a.astype(np.int64), -1), np.sort(b.astype(np.int64), -1))


if __name__ == "__main__":
    raise SystemExit(main())
