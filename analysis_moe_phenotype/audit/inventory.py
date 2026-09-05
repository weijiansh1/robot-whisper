#!/usr/bin/env python3
"""语料清点 → audit/data_map.json（任务级：集数/成功/zarr 行数/对账/queries 中位）。

语料只读；zarr 只读 episode_id/control_step（不碰 hb_router_probs 数据，只记 shape）。
CALVIN 只记规模（目录大小、zarr 行数、集数），不分析。
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import zarr

HUB = pathlib.Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB")
OUT = pathlib.Path(__file__).resolve().parent
SUITES = ["libero_goal", "libero_long", "libero_object", "libero_spatial"]

CORPORA = {
    "grid50x8": {"root": HUB / "cache_new/HiMoE-VLA", "run_id": "right-50x8-20260903"},
    "main16x32": {"root": HUB / "cache/HiMoE-VLA", "run_id": "right-16x32"},
}


def audit_task(run: pathlib.Path) -> dict:
    rec: dict = {"run_dir": str(run)}
    summ_p = run / "client" / "summaries.json"
    if not summ_p.exists():
        rec["status"] = "no_summaries"
        return rec
    summ = json.load(open(summ_p))
    calls = {int(e["episode_index"]): int(e["inference_calls"]) for e in summ}
    succ = {int(e["episode_index"]): bool(e["success"]) for e in summ}
    rec["episodes"] = len(summ)
    rec["success"] = int(sum(succ.values()))
    rec["failure"] = rec["episodes"] - rec["success"]
    rec["sum_inference_calls"] = int(sum(calls.values()))
    rec["queries_median"] = float(np.median(list(calls.values())))
    rec["queries_min"] = int(min(calls.values()))
    rec["queries_max"] = int(max(calls.values()))
    rec["npz_files"] = len(list((run / "client").glob("episode_*.npz")))

    # sim_state 维度 + layout 存在性（episode_00 抽一眼）
    ep0 = run / "client" / "episode_00.npz"
    if ep0.exists():
        with np.load(ep0) as d:
            rec["sim_state_dim"] = int(d["sim_state"].shape[1])
            rec["state_dim"] = int(d["state"].shape[1])
    rec["has_sim_layout"] = (run / "client" / "sim_layout.json").exists()

    zp = run / "server" / "routes.zarr"
    if not zp.exists():
        rec["status"] = "no_zarr"
        return rec
    g = zarr.open_group(str(zp), mode="r")
    keys = set(g.array_keys())
    rec["has_hb_router_probs"] = "hb_router_probs" in keys
    if rec["has_hb_router_probs"]:
        arr = g["hb_router_probs"]
        rec["probs_shape"] = list(arr.shape)
        rec["probs_dtype"] = str(arr.dtype)
    ep = np.asarray(g["episode_id"]).astype(np.int64)
    rec["zarr_rows"] = int(len(ep))
    rec["zarr_rows_eq_sum_calls"] = rec["zarr_rows"] == rec["sum_inference_calls"]
    uniq, cnt = np.unique(ep, return_counts=True)
    rec["zarr_episodes"] = int(len(uniq))
    mism = []
    for e, c in zip(uniq.tolist(), cnt.tolist()):
        if calls.get(e) != c:
            mism.append({"episode_id": e, "zarr_rows": c, "client_calls": calls.get(e)})
    client_only = sorted(set(calls) - set(uniq.tolist()))
    for e in client_only:
        mism.append({"episode_id": e, "zarr_rows": 0, "client_calls": calls[e]})
    rec["reconcile_mismatch_episodes"] = len(mism)
    if mism:
        rec["reconcile_detail"] = mism[:20]
    rec["status"] = "ok"
    return rec


def du_mb(p: pathlib.Path) -> float:
    try:
        out = subprocess.run(["du", "-sm", str(p)], capture_output=True, text=True, timeout=300)
        return float(out.stdout.split()[0])
    except Exception:
        return -1.0


def audit_calvin() -> list[dict]:
    recs = []
    for root_tag, root in [("cache", HUB / "cache/HiMoE-VLA"), ("cache_new", HUB / "cache_new/HiMoE-VLA")]:
        for cal in sorted(root.glob("calvin_d*")):
            for task in sorted(cal.iterdir()):
                if not task.is_dir():
                    continue
                runs = sorted([r for r in task.iterdir() if r.is_dir()])
                if not runs:
                    recs.append({"cache": root_tag, "suite": cal.name, "task": task.name,
                                 "runs": [], "note": "empty"})
                    continue
                for run in runs:
                    r: dict = {"cache": root_tag, "suite": cal.name, "task": task.name,
                               "run": run.name, "size_mb": du_mb(run)}
                    sp = run / "client" / "summaries.json"
                    if sp.exists():
                        s = json.load(open(sp))
                        r["episodes"] = len(s)
                        if s and "success" in s[0]:
                            r["success"] = int(sum(bool(e["success"]) for e in s))
                    zp = run / "server" / "routes.zarr"
                    if zp.exists():
                        try:
                            g = zarr.open_group(str(zp), mode="r")
                            r["zarr_keys"] = sorted(g.array_keys())
                            if "episode_id" in r["zarr_keys"]:
                                r["zarr_rows"] = int(g["episode_id"].shape[0])
                            r["has_hb_router_probs"] = "hb_router_probs" in r["zarr_keys"]
                        except Exception as ex:
                            r["zarr_error"] = str(ex)[:200]
                    recs.append(r)
    return recs


def main() -> None:
    data: dict = {"generated": "2026-09-04", "corpora": {}}
    for cname, cfg in CORPORA.items():
        croot: dict = {"root": str(cfg["root"]), "run_id": cfg["run_id"], "suites": {}}
        for suite in SUITES:
            sdir = cfg["root"] / suite
            if not sdir.exists():
                continue
            tasks = {}
            for task in sorted(sdir.iterdir()):
                run = task / cfg["run_id"]
                if not run.exists():
                    continue
                tasks[task.name] = audit_task(run)
                print(f"[{cname}/{suite}] {task.name}: {tasks[task.name].get('status')}")
            if tasks:
                agg = {
                    "n_tasks": len(tasks),
                    "episodes": sum(t.get("episodes", 0) for t in tasks.values()),
                    "success": sum(t.get("success", 0) for t in tasks.values()),
                    "failure": sum(t.get("failure", 0) for t in tasks.values()),
                    "reconcile_mismatch_episodes": sum(t.get("reconcile_mismatch_episodes", 0) for t in tasks.values()),
                }
                croot["suites"][suite] = {"summary": agg, "tasks": tasks}
        croot["totals"] = {
            "episodes": sum(s["summary"]["episodes"] for s in croot["suites"].values()),
            "success": sum(s["summary"]["success"] for s in croot["suites"].values()),
            "failure": sum(s["summary"]["failure"] for s in croot["suites"].values()),
        }
        data["corpora"][cname] = croot
    data["calvin_inventory"] = audit_calvin()
    (OUT / "data_map.json").write_text(json.dumps(data, indent=1))
    print("written", OUT / "data_map.json")


if __name__ == "__main__":
    main()
