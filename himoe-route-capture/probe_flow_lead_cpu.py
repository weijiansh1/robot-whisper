"""Routing Lead Test, CPU feasibility run: does routing resolve the basin early?

Runs K sibling candidates on ONE fixed observation -- same images, same state, same
prompt, independent flow noise -- capturing at every Euler step tau both halves of
the comparison:

    action signal    x^(tau), the provisional action chunk
    routing signal   R^(tau), the HB top-4 field that step produced

and asks whether the pairwise structure at step tau predicts the pairwise structure
of the final chunk.  The claim worth testing is not "routing wins at the end" but
"routing wins EARLY" -- if AUC_route(tau) > AUC_action(tau) at small tau, routing
leads action formation and can prune before the flow finishes.

Two built-in negative controls, both with a known-zero answer, run alongside:

  * the state token (suffix 0) is routed identically for every candidate, because
    it attends only to the prefix and the noise never reaches it -- measured at
    99.4% identical across 64 independent draws in the routing-cloud batch;
  * AS routing is constant for this whole deployment.

If either scores above chance, the pipeline is wrong, not the model.

WHY THIS IS A FEASIBILITY RUN AND NOT A RESULT: with no CUDA, `policy.py:71`'s
`torch.autocast(device_type='cuda')` is a no-op and the model runs fp32, while the
deployed path is bf16.  The stored corpus puts the 4th-vs-5th expert gap at
D_45 ~ 3.3e-4, which is on the order of ONE bf16 ulp near 1/32 -- so the top-4
boundary is decided within rounding on GPU, and fp32 will not select the same
experts. Plumbing, timing and effect direction transfer; the numbers do not.

Run in the model env (python 3.11).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

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


def install_cpu_autocast(torch):
    """policy.py:71 hardcodes device_type='cuda'; without CUDA that silently
    disables autocast and the AS gate then hits fp32 weights with a bf16
    data_mask.  Redirect to the CPU backend so the call completes."""
    original = torch.autocast

    def shim(device_type: str, *a, **kw):
        return original("cpu" if device_type == "cuda" else device_type, *a, **kw)

    torch.autocast = shim


def weighted_jaccard_distance(idx_a, weight_a, idx_b, weight_b, n_experts=32):
    """1 - sum_e min(p_e,q_e)/sum_e max(p_e,q_e), averaged over sites.

    Uses the raw gathered probabilities rather than the gate's renormalised top-4
    weights: renormalisation forces every site's four weights to sum to 1 and
    throws away exactly the mass difference that distinguishes a confident site
    from a coin-flip one.
    """
    shape = idx_a.shape[:-1]
    dense_a = np.zeros((*shape, n_experts), np.float64)
    dense_b = np.zeros((*shape, n_experts), np.float64)
    np.put_along_axis(dense_a, idx_a.astype(np.int64), weight_a.astype(np.float64), axis=-1)
    np.put_along_axis(dense_b, idx_b.astype(np.int64), weight_b.astype(np.float64), axis=-1)
    lower = np.minimum(dense_a, dense_b).sum(-1)
    upper = np.maximum(dense_a, dense_b).sum(-1)
    return float(np.mean(1.0 - lower / np.maximum(upper, 1e-12)))


def pairwise(values, metric):
    n = len(values)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = metric(values[i], values[j])
    return out


def auc_and_rho(predictor, final, same_basin):
    """AUC for 'same basin', and Spearman rho against the final-chunk distance."""
    upper = np.triu_indices(len(final), 1)
    x, y, label = predictor[upper], final[upper], same_basin[upper]
    positive, negative = x[label == 1], x[label == 0]
    if len(positive) == 0 or len(negative) == 0:
        auc = float("nan")
    else:
        # small predictor distance should mean same basin, so orient accordingly
        wins = (positive[:, None] < negative[None, :]).sum()
        ties = (positive[:, None] == negative[None, :]).sum()
        auc = float((wins + 0.5 * ties) / (len(positive) * len(negative)))

    def rank(v):
        order = np.argsort(np.argsort(v))
        return order.astype(float)

    rho = float(np.corrcoef(rank(x), rank(y))[0, 1])
    return auc, rho


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--observations", type=int, default=1)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/flow-lead-cpu")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA is still visible; this is the CPU feasibility path")
    torch.set_num_threads(args.threads)
    install_cpu_autocast(torch)
    print("torch threads = %d" % torch.get_num_threads(), flush=True)

    # how big is the top-4 boundary gap relative to the deployed dtype's resolution?
    ulp_bf16 = float(np.float64(
        torch.tensor(1 / 32, dtype=torch.bfloat16).item()
        - torch.nextafter(torch.tensor(1 / 32, dtype=torch.bfloat16),
                          torch.tensor(0.0, dtype=torch.bfloat16)).item()))
    print("bf16 ulp near 1/32 = %.2e  (corpus D_45 ~ 3.3e-4 -> %.1f ulp)"
          % (ulp_bf16, 3.3e-4 / ulp_bf16), flush=True)

    from himoe_libero_bridge.policies import HiMoEPolicy
    from himoe_router_recorder import HiMoERouteRecorder, discover_gates
    from serve_flow_trace import FlowTracer

    start = time.perf_counter()
    policy = HiMoEPolicy(
        checkpoint_dir=args.checkpoint_dir, suite="goal",
        upstream_root=args.upstream_root, require_cuda=False,
        libero_wrist_layout="checkpoint-right",
    )
    print("load: %.1f s" % (time.perf_counter() - start), flush=True)

    core = policy._policy.model
    gates = discover_gates(core)
    recorder = HiMoERouteRecorder(core, store_full_probs=False).attach()
    tracer = FlowTracer(core)
    n_denoise = int(core.config.num_steps)
    n_action = int(core.config.n_action_steps)
    print("gates=%d  n_action_steps=%d  num_steps=%d" % (len(gates), n_action, n_denoise),
          flush=True)

    rng = np.random.default_rng(args.seed)
    results = []
    for observation_index in range(args.observations):
        base = {
            IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
            WRIST_IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
            STATE_KEY: (rng.standard_normal(STATE_DIM) * 0.1).astype(np.float32),
            PROMPT_KEY: "open the middle drawer of the cabinet",
        }
        flows, routes, timings, residuals = [], [], [], []
        for candidate in range(args.candidates):
            request = dict(base)
            request[FLOW_NOISE_KEY] = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
            recorder.begin_control_step(episode_id=observation_index, control_step=candidate)
            tracer.begin()
            t0 = time.perf_counter()
            policy.infer(request)
            timings.append(time.perf_counter() - t0)
            record = recorder.end_control_step()
            residuals.append(tracer.euler_residual())
            flows.append(tracer.trajectory())                # [T+1, n_action, 24]
            # the recorder emits [B, L, D, S, K] and B is always 1 here
            routes.append((np.asarray(record.hb_expert_ids)[0],
                           np.asarray(record.hb_selected_prob)[0]))
            print("  obs %d cand %2d  %.1f s" % (observation_index, candidate, timings[-1]),
                  flush=True)

        flows = np.stack(flows)                              # [K, T, n_action, 24]
        idx = np.stack([r[0] for r in routes])               # [K, L, T, S, 4]
        weight = np.stack([r[1] for r in routes])
        if idx.shape[2] != n_denoise:
            raise RuntimeError("routing denoise axis %d != num_steps %d" % (idx.shape[2], n_denoise))

        # x^(T), reconstructed from the final velocity.  Using x^(T-1) here would
        # make the tau = T-1 column a comparison of a vector with itself, which is
        # not a finding about action maturity but an identity.
        print("  Euler residual max %.2e over %d candidates (0 = hook is on the "
              "integrated tensors)" % (max(residuals), args.candidates), flush=True)
        final = flows[:, -1]
        if flows.shape[1] != n_denoise + 1:
            raise RuntimeError("expected T+1=%d states, got %d" % (n_denoise + 1, flows.shape[1]))
        final_distance = pairwise(list(final), lambda a, b: float(np.linalg.norm(a - b)))
        threshold = np.median(final_distance[np.triu_indices(args.candidates, 1)])
        same_basin = (final_distance < threshold).astype(int)

        print("\n  tau   action AUC   route AUC | action rho  route rho | state-token AUC (control)")
        rows = []
        for tau in range(n_denoise):
            action_distance = pairwise(list(flows[:, tau]),
                                       lambda a, b: float(np.linalg.norm(a - b)))
            route_distance = pairwise(
                list(range(args.candidates)),
                lambda i, j: weighted_jaccard_distance(
                    idx[i][:, tau, 1:], weight[i][:, tau, 1:],
                    idx[j][:, tau, 1:], weight[j][:, tau, 1:]))
            state_distance = pairwise(
                list(range(args.candidates)),
                lambda i, j: weighted_jaccard_distance(
                    idx[i][:, tau, :1], weight[i][:, tau, :1],
                    idx[j][:, tau, :1], weight[j][:, tau, :1]))
            action_auc, action_rho = auc_and_rho(action_distance, final_distance, same_basin)
            route_auc, route_rho = auc_and_rho(route_distance, final_distance, same_basin)
            state_auc, _ = auc_and_rho(state_distance, final_distance, same_basin)
            rows.append({"tau": tau, "action_auc": action_auc, "route_auc": route_auc,
                         "action_rho": action_rho, "route_rho": route_rho,
                         "state_token_auc": state_auc})
            print("   %2d     %.3f        %.3f   |   %+.3f      %+.3f   |   %.3f"
                  % (tau, action_auc, route_auc, action_rho, route_rho, state_auc), flush=True)

        results.append({"observation": observation_index, "rows": rows,
                        "mean_infer_s": float(np.mean(timings)),
                        "final_distance_median": float(threshold)})

    recorder.close()
    tracer.close()

    lead = [r["tau"] for res in results for r in res["rows"] if r["route_auc"] > r["action_auc"]]
    print("\nflow steps where routing beats the provisional action: %s"
          % (sorted(set(lead)) if lead else "none"))
    print("mean inference %.1f s/candidate on CPU"
          % float(np.mean([r["mean_infer_s"] for r in results])))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lead_curve.json").write_text(json.dumps(
        {"results": results, "candidates": args.candidates,
         "note": "CPU fp32 via autocast shim; not numerically the deployed bf16 path"},
        indent=1))
    print("wrote %s" % (out / "lead_curve.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
