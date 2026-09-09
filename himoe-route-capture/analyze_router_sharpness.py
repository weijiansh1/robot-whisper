"""How peaked is the HB router?  From the full 32-way softmax, 312 control steps."""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")
from himoe_route_store import ZarrRouteReader  # noqa: E402

HB_LAYERS = [2, 3, 4, 5, 12, 13, 14, 15]
STORE = "/home/jovyan/work/himoe-route-capture/runs/fullprob-right/routes.zarr"


def main():
    r = ZarrRouteReader(STORE)
    P = np.asarray(r["hb_router_probs"][:]).astype(np.float32)  # [N,L,D,S,E]
    ids = np.asarray(r["hb_expert_ids"][:])
    N, L, D, S, E = P.shape
    print("full router probs %s  (%d control steps x %d layers x %d denoise x %d tokens)"
          % (P.shape, N, L, D, S))
    print("uniform reference:  H = ln(%d) = %.4f,  top1 = %.4f,  margin = 0\n"
          % (E, np.log(E), 1 / E))

    srt = np.sort(P, axis=-1)[..., ::-1]
    top1, top2 = srt[..., 0], srt[..., 1]
    H = -(P * np.log(np.clip(P, 1e-12, None))).sum(-1)
    top4mass = srt[..., :4].sum(-1)

    print("%-6s %-9s %-9s %-9s %-10s %-10s %s" %
          ("layer", "熵", "熵/ln32", "top1", "top1/均匀", "top1-top2", "top4 质量"))
    out = {}
    for li, layer in enumerate(HB_LAYERS):
        h, t1, t2, m4 = H[:, li], top1[:, li], top2[:, li], top4mass[:, li]
        out[layer] = {
            "entropy": float(h.mean()), "entropy_frac": float(h.mean() / np.log(E)),
            "top1": float(t1.mean()), "top1_over_uniform": float(t1.mean() * E),
            "margin": float((t1 - t2).mean()), "top4_mass": float(m4.mean()),
            "eff_experts": float(np.exp(h.mean())),
        }
        print("%-6d %-9.4f %-9.4f %-9.4f %-10.2f %-10.5f %.4f" %
              (layer, h.mean(), h.mean() / np.log(E), t1.mean(),
               t1.mean() * E, (t1 - t2).mean(), m4.mean()))
    print("\n有效专家数 exp(H)  (32 = 完全均匀):")
    print("  " + "  ".join("L%d:%.1f" % (l, out[l]["eff_experts"]) for l in HB_LAYERS))
    print("  top-4 质量 4/32 = %.4f 是均匀参考值" % (4 / E))

    # does the router respond to the input at all?
    print("\n=== 固定偏好 vs 输入驱动 ===")
    print("把 log p 拆成「每个专家的常数偏好」+「随输入变化的残差」")
    print("%-6s %-16s %-16s %s" % ("layer", "常数分量方差", "输入驱动方差", "输入驱动占比"))
    for li, layer in enumerate(HB_LAYERS):
        lp = np.log(np.clip(P[:, li], 1e-12, None)).reshape(-1, E)
        per_expert_mean = lp.mean(0)
        between = per_expert_mean.var()
        within = (lp - per_expert_mean).var()
        out[layer]["var_between_experts"] = float(between)
        out[layer]["var_input_driven"] = float(within)
        out[layer]["frac_input_driven"] = float(within / (within + between))
        print("%-6d %-16.6f %-16.6f %.3f"
              % (layer, between, within, within / (within + between)))

    # how much does the actual top-4 set move?
    print("\n=== top-4 集合在多大程度上真的在变 ===")
    print("%-6s %-16s %-18s %s" % ("layer", "不同集合数", "与前一步 Jaccard", "站点总数"))
    for li, layer in enumerate(HB_LAYERS):
        flat = ids[:, li].reshape(-1, 4)
        keys = {tuple(sorted(x.tolist())) for x in flat}
        a = ids[:-1, li].reshape(-1, 4)
        b = ids[1:, li].reshape(-1, 4)
        j = np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)])
        out[layer]["distinct_top4_sets"] = len(keys)
        out[layer]["consecutive_jaccard"] = float(j)
        print("%-6d %-16d %-18.4f %d" % (layer, len(keys), j, len(flat)))
    print("  C(32,4) = 35960 种可能集合")

    json.dump(out, open("/home/jovyan/work/himoe-route-capture/router_sharpness.json", "w"),
              indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
