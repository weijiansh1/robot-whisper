#!/usr/bin/env python3
"""Per-query internal features for every Long screen episode: kNN-20 risk, v7/v8 scores, state-token and action-token routing."""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from adaptive_control import RouteRisk
from analyze_repair_experiment import arc_metrics
from collection_routes import PROBS_KEY
from collection_storage import records
from v8_feature_control import flow_features

AUDITS = ("experiment_long_scale", "experiment_long_batch1")
_RISK = None


def directories():
    out = {}
    for stem in AUDITS:
        for task in json.loads((Path("design") / ("%s_audit_20260908.json" % stem)).read_text())["tasks"]:
            out[task["main_id"]] = task["directory"]
    return out


def episode_features(args):
    main_id, directory = args
    global _RISK
    if _RISK is None:
        _RISK = RouteRisk()
    monitor = _RISK.monitor()
    knn, vectors, v7, v8, state_top1, state_entropy, state_probs, action_probs, arcs, successes = [], [], [], [], [], [], [], [], [], []
    for row in records(Path(directory) / "main"):
        probs = np.asarray(row[PROBS_KEY], np.float32)
        status = monitor.update(probs)
        vector, score = _RISK.current(monitor)
        knn.append(score)
        vectors.append(np.asarray(vector, np.float32))
        v7.append([status["freeze_score"], status["acceleration_score"], status["periodicity_score"]])
        v8.append(flow_features(probs)[0])
        p = probs[:, -1].astype(np.float64)
        p = p / np.maximum(p.sum(-1, keepdims=True), 1e-12)
        state = p[:, 0]
        state_top1.append(state.max(-1))
        state_entropy.append(-(state * np.log(np.maximum(state, 1e-12))).sum(-1))
        state_probs.append(state.astype(np.float32))
        action_probs.append(p[:, 1:].mean(1).astype(np.float32))
        arc = arc_metrics(probs)
        arcs.append([arc["state_top1_front"], arc["arc_bandedness_back"], arc["arc_size_back"]])
        successes.append(bool(row["success"]))
    return main_id, dict(knn=np.asarray(knn, np.float32), knn_vector=np.asarray(vectors, np.float32), v7=np.asarray(v7, np.float32),
                         v8=np.asarray(v8, np.float32), state_top1=np.asarray(state_top1, np.float32),
                         state_entropy=np.asarray(state_entropy, np.float32), state_probs=np.asarray(state_probs, np.float32),
                         action_probs=np.asarray(action_probs, np.float32), arc=np.asarray(arcs, np.float32),
                         success=np.asarray(successes, bool))


def run(args):
    alarms = list(csv.DictReader(args.alarms.open(newline="")))
    dirs = directories()
    jobs = [(r["main_id"], dirs[r["main_id"]]) for r in alarms if r["main_id"] in dirs]
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {}
    with Pool(args.workers) as pool:
        for i, (main_id, feats) in enumerate(pool.imap_unordered(episode_features, jobs, chunksize=4)):
            np.savez_compressed(args.out / ("%s.npz" % main_id), **feats)
            if (i + 1) % 100 == 0:
                print("extracted", i + 1, "/", len(jobs), flush=True)
    for r in alarms:
        if r["main_id"] in dirs:
            meta[r["main_id"]] = dict(benchmark=r["benchmark"], task_name=r["task_name"], analysis_role=r["analysis_role"],
                                      failure=r["failure"] == "True", length=int(r["length"]), knn20=int(r["knn20"]),
                                      v7_frozen=int(r["v7_frozen"]), v8_frozen=int(r["v8_frozen"]), category=r["category"])
    (args.out / "meta.json").write_text(json.dumps(meta))
    print(json.dumps(dict(episodes=len(meta), out=str(args.out))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alarms", type=Path, default=Path("design/experiment_long_screen_summary_20260908/first_alarms.csv"))
    parser.add_argument("--out", type=Path, default=Path("design/post_alarm_routing_20260909"))
    parser.add_argument("--workers", type=int, default=48)
    run(parser.parse_args())
