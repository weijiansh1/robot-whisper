"""Compare HB-MoE routing between wrist layouts, and success vs failure.

Route trace per episode: [n_infer, flow=10, layer=8, action=10, topk=4].
Reuses the audited dense/normalised route reconstruction from the bridge.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
from himoe_libero_bridge.routing import sparse_routes_to_dense  # noqa: E402

HB_LAYERS = [2, 3, 4, 5, 12, 13, 14, 15]
N_EXPERTS = 32


def load(group_dir):
    d = pathlib.Path(group_dir)
    summaries = json.loads((d / "summaries.json").read_text())
    episodes = []
    for s in summaries:
        idx = s.get("episode_index", s["init_state_id"])
        z = np.load(d / ("episode_%02d.npz" % idx))
        episodes.append(
            {
                "summary": s,
                "ids": z["expert_ids"],  # [T,10,8,10,4]
                "w": z["expert_weights"].astype(np.float64),
            }
        )
    return episodes


def load_per_layer(ids, w):
    """Expert usage mass per layer -> [8, 32], normalised to sum 1 per layer."""
    dense = sparse_routes_to_dense(ids, w)  # [...,32]
    # axes: infer, flow, layer, action, expert
    load = dense.sum(axis=(0, 1, 3))
    return load / load.sum(axis=-1, keepdims=True)


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(axis=-1)


def topk_set_jaccard(a, b):
    """Mean Jaccard of the top-4 expert *sets* at aligned sites."""
    out = []
    flat_a = a.reshape(-1, a.shape[-1])
    flat_b = b.reshape(-1, b.shape[-1])
    for x, y in zip(flat_a, flat_b):
        sx, sy = set(x.tolist()), set(y.tolist())
        out.append(len(sx & sy) / len(sx | sy))
    return float(np.mean(out))


def route_drift(ids, w):
    """TV between consecutive control steps, averaged over sites."""
    dense = sparse_routes_to_dense(ids, w)
    dense = dense / dense.sum(-1, keepdims=True)
    if dense.shape[0] < 2:
        return float("nan")
    return float(tv(dense[1:], dense[:-1]).mean())


def main():
    right = load("/tmp/moe-rollout/right")
    left = load("/tmp/moe-rollout/left")
    print("=" * 74)
    print("group            n   success   mean steps   mean infers")
    for name, g in (("checkpoint-right", right), ("released-left", left)):
        ok = [e for e in g if e["summary"]["success"]]
        print(
            "%-16s %2d   %d/%-6d %8.1f %12.1f"
            % (
                name,
                len(g),
                len(ok),
                len(g),
                np.mean([e["summary"]["action_steps"] for e in g]),
                np.mean([e["summary"]["inference_calls"] for e in g]),
            )
        )

    # ---------- 1. paired first-inference contrast ----------
    print("\n" + "=" * 74)
    print("1. 第一次推理的配对对比（同 init state / 同 settle / 同 flow noise，")
    print("   唯一差异是 wrist 槽位）—— top-4 专家集合的 Jaccard，按层")
    print("\n%-6s %s" % ("layer", "  ".join("ep%d" % i for i in range(len(right)))))
    per_layer_j = np.zeros((8, min(len(right), len(left))))
    for li, layer in enumerate(HB_LAYERS):
        row = []
        for ep in range(per_layer_j.shape[1]):
            a = right[ep]["ids"][0, :, li]  # [flow, action, topk]
            b = left[ep]["ids"][0, :, li]
            j = topk_set_jaccard(a, b)
            per_layer_j[li, ep] = j
            row.append("%.2f" % j)
        print("%-6d %s" % (layer, "  ".join(row)))
    print("%-6s %s" % ("mean", "  ".join("%.2f" % v for v in per_layer_j.mean(0))))
    print("\n全局平均 Jaccard = %.3f  (1.0 = 路由完全相同, 0.0 = 完全不相交)"
          % per_layer_j.mean())

    # ---------- 2. aggregate expert load ----------
    print("\n" + "=" * 74)
    print("2. 专家负载分布（整组所有 control step 聚合），按层的 TV 距离")
    lr = np.mean([load_per_layer(e["ids"], e["w"]) for e in right], axis=0)
    ll = np.mean([load_per_layer(e["ids"], e["w"]) for e in left], axis=0)
    print("\n%-6s %8s   %-28s %-28s" % ("layer", "TV", "right top-3 专家", "left top-3 专家"))
    for li, layer in enumerate(HB_LAYERS):
        tr = np.argsort(-lr[li])[:3]
        tl = np.argsort(-ll[li])[:3]
        print(
            "%-6d %8.4f   %-28s %-28s"
            % (
                layer,
                tv(lr[li], ll[li]),
                " ".join("e%d:%.3f" % (e, lr[li][e]) for e in tr),
                " ".join("e%d:%.3f" % (e, ll[li][e]) for e in tl),
            )
        )
    print("\n平均 TV = %.4f" % tv(lr, ll).mean())

    print("\n每层实际被用到的专家数（负载 > 0.1%%，共 32 个）")
    print("%-6s %8s %8s %10s" % ("layer", "right", "left", "共同"))
    for li, layer in enumerate(HB_LAYERS):
        sr = set(np.where(lr[li] > 1e-3)[0].tolist())
        sl = set(np.where(ll[li] > 1e-3)[0].tolist())
        print("%-6d %8d %8d %10d" % (layer, len(sr), len(sl), len(sr & sl)))

    # ---------- 3. success vs failure inside released-left ----------
    print("\n" + "=" * 74)
    print("3. released-left 组内：成功 vs 失败")
    ok = [e for e in left if e["summary"]["success"]]
    bad = [e for e in left if not e["summary"]["success"]]
    print("   成功 %d 集, 失败 %d 集" % (len(ok), len(bad)))
    if ok and bad:
        lo = np.mean([load_per_layer(e["ids"], e["w"]) for e in ok], axis=0)
        lb = np.mean([load_per_layer(e["ids"], e["w"]) for e in bad], axis=0)
        print("\n%-6s %10s" % ("layer", "TV(成功,失败)"))
        for li, layer in enumerate(HB_LAYERS):
            print("%-6d %10.4f" % (layer, tv(lo[li], lb[li])))
        print("平均 TV = %.4f" % tv(lo, lb).mean())

    print("\n   路由随控制步的漂移量（相邻 control step 的平均 TV）")
    print("   —— 卡死的策略应当漂移很小")
    print("\n%-18s %-8s %-8s %-10s %s" % ("group", "ep", "success", "steps", "drift"))
    for name, g in (("checkpoint-right", right), ("released-left", left)):
        for e in g:
            s = e["summary"]
            print(
                "%-18s %-8d %-8s %-10d %.5f"
                % (name, s["init_state_id"], s["success"], s["action_steps"],
                   route_drift(e["ids"], e["w"]))
            )
    for label, sel in (
        ("right 全部", right),
        ("left 成功", [e for e in left if e["summary"]["success"]]),
        ("left 失败", [e for e in left if not e["summary"]["success"]]),
    ):
        if sel:
            print("  %-12s mean drift = %.5f"
                  % (label, np.mean([route_drift(e["ids"], e["w"]) for e in sel])))

    # ---------- 4. late-episode routing in failures ----------
    print("\n" + "=" * 74)
    print("4. 失败集后半段是否锁死：前 1/3 vs 后 1/3 control step 的路由 TV")
    print("\n%-18s %-6s %-8s %s" % ("group", "ep", "success", "TV(前1/3, 后1/3)"))
    for name, g in (("checkpoint-right", right), ("released-left", left)):
        for e in g:
            d = sparse_routes_to_dense(e["ids"], e["w"])
            d = d / d.sum(-1, keepdims=True)
            n = d.shape[0]
            third = max(1, n // 3)
            early = d[:third].mean(axis=(0, 1, 3))
            late = d[-third:].mean(axis=(0, 1, 3))
            print("%-18s %-6d %-8s %.5f"
                  % (name, e["summary"]["init_state_id"], e["summary"]["success"],
                     tv(early, late).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
