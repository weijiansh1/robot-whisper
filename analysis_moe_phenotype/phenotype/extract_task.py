#!/usr/bin/env python3
"""单任务表型提取：routes.zarr + summaries.json → rows.npz / reps.npy / meta.json。

  python3 extract_task.py --task-dir <.../<task>/<run_id>> --out <features/.../<task>>

对账契约：按 episode_id 分组的 zarr 行数必须等于该集 client inference_calls；
不符的集整集剔除并记入 meta.json（不猜、不补）。语料目录只读。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import zarr

import features as F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    task = pathlib.Path(args.task_dir)
    out = pathlib.Path(args.out)
    if (out / "rows.npz").exists():
        print(f"[skip] {out} 已存在")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    g = zarr.open_group(str(task / "server" / "routes.zarr"), mode="r")
    if "hb_router_probs" not in set(g.array_keys()):
        raise SystemExit(f"{task}: 无 hb_router_probs（未开 store-full-probs），跳过")
    P = np.asarray(g["hb_router_probs"])                    # (N,8,10,11,32) f16
    ep = np.asarray(g["episode_id"]).astype(np.int32)
    cs = np.asarray(g["control_step"]).astype(np.int32)
    summ = json.load(open(task / "client" / "summaries.json"))

    # 行序规范化：按 (episode, control_step) 排
    order = np.lexsort((cs, ep))
    P, ep, cs = P[order], ep[order], cs[order]

    # 对账：episode_id ↔ summaries[episode_index].inference_calls
    by_ep = {int(e["episode_index"]): e for e in summ}
    counts = {int(k): int(v) for k, v in zip(*np.unique(ep, return_counts=True))}
    dropped = []
    keep_eps = []
    for e_id, cnt in sorted(counts.items()):
        s = by_ep.get(e_id)
        if s is None or int(s["inference_calls"]) != cnt:
            dropped.append({"episode_id": e_id, "zarr_rows": cnt,
                            "client_calls": None if s is None else int(s["inference_calls"])})
        else:
            keep_eps.append(e_id)
    extra_client = [k for k in by_ep if k not in counts]
    keep = np.isin(ep, keep_eps)
    P, ep, cs = P[keep], ep[keep], cs[keep]
    n = len(P)
    if n == 0:
        raise SystemExit(f"{task}: 对账后无剩余行")

    # state token 跨 flow 不变性抽检（已知性质，偏差应≈0）
    sample = P[: min(200, n), F.DEEP, :, F.STATE, :].astype(np.float32)
    state_dev = float(np.abs(sample - sample[:, :, :1]).max())

    scal, profile, reps = F.per_row(P)
    del P
    mob, w8 = F.episode_recurrence(reps, ep, cs)

    meta_cols = {
        "episode_id": ep, "control_step": cs,
        "scene": np.array([by_ep[e]["init_state_id"] for e in ep], np.int16),
        "repeat": np.array([by_ep[e].get("repeat", -1) for e in ep], np.int16),
        "success": np.array([bool(by_ep[e]["success"]) for e in ep], np.int8),
        "flow_noise_seed": np.array([by_ep[e].get("flow_noise_seed", -1) for e in ep], np.int32),
    }
    np.savez_compressed(out / "rows.npz", **scal, **meta_cols,
                        flow_profile=profile, mob_k=mob, mob1_w8=w8)
    np.save(out / "reps.npy", reps.reshape(n, -1).astype(np.float16))

    eps_meta = [by_ep[e] for e in keep_eps]
    meta = {
        "task_dir": str(task), "rows": int(n),
        "episodes": len(keep_eps),
        "success_episodes": int(sum(bool(m["success"]) for m in eps_meta)),
        "dropped_episodes": dropped, "client_only_episodes": extra_client,
        "state_token_flow_dev_max": state_dev,
        "queries_per_episode_median": float(np.median([counts[e] for e in keep_eps])),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[ok] {task.parent.name}: rows={n} eps={len(keep_eps)} "
          f"succ={meta['success_episodes']} drop={len(dropped)} "
          f"state_dev={state_dev:.2e} {meta['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
