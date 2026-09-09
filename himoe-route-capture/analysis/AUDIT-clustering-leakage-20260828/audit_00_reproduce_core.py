#!/usr/bin/env python3
"""Reproduce the claimed consensus failure core from published assignments CSVs.

Outputs a JSON with the exact membership sets so downstream sentinel scripts
use the identical evaluation convention.
"""
import json, pathlib
import numpy as np
import pandas as pd

BASE = pathlib.Path("/home/jovyan/work/himoe-vla/himoe-route-capture/analysis")
OUT = BASE / "AUDIT-clustering-leakage-20260828"
OUT.mkdir(parents=True, exist_ok=True)

ak = pd.read_csv(BASE / "aligned-route-kernel/assignments.csv")
ev = pd.read_csv(BASE / "route-change-events/assignments.csv")
al = pd.read_csv(BASE / "alternative-routing-organizations/assignments.csv")

key = ["task", "episode"]
for df in (ak, ev, al):
    df["uid"] = df["task"] + "#" + df["episode"].astype(str)

assert set(ak.uid) == set(ev.uid) == set(al.uid), "episode sets differ"
# Keep the aligned-kernel CSV row order: it is byte-identical to the feature-cache
# row order (verified: meta_task/meta_episode match position by position).
ev = ev.set_index("uid").loc[ak.uid.values].reset_index()
al = al.set_index("uid").loc[ak.uid.values].reset_index()
assert (ak.uid.values == ev.uid.values).all() and (ak.uid.values == al.uid.values).all()

failure = (ak.outcome.values == "failure")
assert (ev.failure.values == failure).all()
assert (al.failure.values == failure).all()
length = ak.episode_length.values.astype(int)
task = ak.task.values
init = ak.init_state_id.values.astype(int)
seed = ak.flow_noise_seed.values.astype(int)

raw_c1 = (ak.raw_cluster.values == 1)
event_c0 = (ev.event_cluster.values == 0)
lag_c7 = (al.lag_spectrum_louvain.values == "C7")

def stats(mask, name):
    n = int(mask.sum())
    f = int((mask & failure).sum())
    s = n - f
    prec = f / n if n else float("nan")
    rec = f / int(failure.sum())
    return dict(name=name, n=n, failure=f, success=s,
                precision=round(prec, 4), failure_recall=round(rec, 4))

rows = [stats(raw_c1, "aligned raw C1"),
        stats(event_c0, "event C0"),
        stats(lag_c7, "lag Louvain C7")]

votes = raw_c1.astype(int) + event_c0.astype(int) + lag_c7.astype(int)
maj = votes >= 2
inter = votes == 3
union = votes >= 1
rows += [stats(maj, "consensus >=2 votes"),
         stats(inter, "intersection (3 votes)"),
         stats(union, "union (>=1 vote)")]

def jac(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))

pair = {"rawC1~eventC0": round(jac(raw_c1, event_c0), 4),
        "rawC1~lagC7": round(jac(raw_c1, lag_c7), 4),
        "eventC0~lagC7": round(jac(event_c0, lag_c7), 4)}

# union misses, by task
miss = failure & ~union
miss_by_task = pd.Series(task[miss]).value_counts().to_dict()

res = dict(rows=rows, pairwise_jaccard=pair,
           union_missed_failures=int(miss.sum()),
           union_missed_by_task=miss_by_task,
           n_episodes=int(len(ak)), n_failure=int(failure.sum()))
print(json.dumps(res, indent=2, ensure_ascii=False))

np.savez_compressed(OUT / "core_membership.npz",
                    uid=ak.uid.values.astype(str), task=task.astype(str),
                    episode=ak.episode.values.astype(int),
                    init_state_id=init, flow_noise_seed=seed,
                    episode_length=length, failure=failure,
                    raw_c1=raw_c1, event_c0=event_c0, lag_c7=lag_c7,
                    votes=votes, consensus=maj, union=union, intersection=inter)
(OUT / "reproduce_core.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
