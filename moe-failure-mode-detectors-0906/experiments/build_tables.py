#!/usr/bin/env python3
"""Build the frozen operating-point tables.

Development defines everything: the per-chunk reference, the persistence gates
and the threshold grid.  External is scored two ways, both reported:

  external_8b_ops_frozenvalue  development's numeric thresholds, applied
                               unchanged - the strictest form of transfer, in
                               which the alarm *rate* is free to move;
  external_8b_ops_ratematched  the same grid index k, re-derived from
                               external's own *unlabelled* routing
                               distribution - the alarm rate is held fixed,
                               which is the knob an operator actually has.

Neither uses an external label for anything.

Output per cohort is one npz:
  ops        (n_ops,) index: quantity, layer, sign, stat, family, arm, threshold
  counts     (n_ops, n_task, 4 group, 5 lead) int32 alarm counts
  tasks      (n_task,) task names
  group order: 0 non-risk, 1 drop, 2 grasp, 3 other risk
  lead order: (0, 2, 4, 8, 12)
"""

from __future__ import annotations

import argparse
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import arms as A
import modes_common as M

G: dict = {}


def _init(cohort: str, calib_path: Path | None, grid_path: Path | None) -> None:
    G["cohort"] = cohort
    frame = M.load(cohort)
    G["frame"] = frame
    names, code = A.task_codes(frame)
    G["tasks"] = names
    G["task_code"] = code
    G["n_task"] = len(names)
    G["group"] = A.episode_groups(frame)
    G["channels"] = M.channel_names(frame, include_controls=True)
    G["calib"] = pickle.loads(calib_path.read_bytes()) if calib_path else None
    G["grid"] = pickle.loads(grid_path.read_bytes()) if grid_path else None


def _one(index: int) -> tuple:
    quantity, layer = G["channels"][index]
    frame = G["frame"]
    x = M.channel(frame, quantity, layer)
    calib = (G["calib"][(quantity, layer)] if G["calib"] is not None
             else A.calibrate(x, frame))
    rows, counts, grids = [], [], {}
    for sign in (+1, -1):
        source = None if G["grid"] is None else G["grid"][(quantity, layer, sign)]
        table = A.channel_table(x, frame, calib, sign, G["task_code"],
                                G["n_task"], G["group"], source)
        grids[(quantity, layer, sign)] = {k: v["thresholds"] for k, v in table.items()}
        for stat in sorted(table):
            blob = table[stat]
            for j, th in enumerate(blob["thresholds"]):
                rows.append((quantity, layer, sign, stat, A.stat_family(stat),
                             A.STAT_ARM[A.stat_family(stat)], float(th),
                             float(blob["median_chunk"][j]), A.GRID_K[j],
                             float(calib["within_episode_constant"])))
                counts.append(blob["counts"][j])
    return index, rows, np.asarray(counts, np.int32), calib, grids


def build(cohort: str, out: Path, calib_path: Path | None,
          grid_path: Path | None, workers: int, save_calib: bool) -> None:
    t0 = time.time()
    _init(cohort, calib_path, grid_path)
    n = len(G["channels"])
    print(f"{cohort} -> {out.name}: {n} channels, {G['n_task']} tasks, "
          f"{len(G['frame']['risk'])} episodes")
    if workers > 1:
        with Pool(workers) as pool:
            got = pool.map(_one, range(n), chunksize=4)
    else:
        got = [_one(i) for i in range(n)]
    got.sort(key=lambda r: r[0])

    rows = [r for _, rs, _, _, _ in got for r in rs]
    counts = np.concatenate([c for _, _, c, _, _ in got], axis=0)
    calib = {G["channels"][i]: c for i, _, _, c, _ in got}
    grids = {}
    for _, _, _, _, g in got:
        grids.update(g)

    ops = np.rec.fromrecords(
        rows, names=("quantity,layer,sign,stat,family,arm,threshold,"
                     "median_chunk,grid_k,within_episode_constant"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ops=ops, counts=counts,
                        tasks=np.asarray(G["tasks"], dtype=object),
                        leads=np.asarray(M.LEADS),
                        n_risk=int(G["frame"]["risk"].sum()),
                        n_episode=len(G["frame"]["risk"]))
    if save_calib:
        (out.parent / f"{cohort}_calibration.pkl").write_bytes(pickle.dumps(calib))
        (out.parent / f"{cohort}_grid.pkl").write_bytes(pickle.dumps(grids))
    print(f"  {len(ops)} operating points, counts {counts.shape}, "
          f"{time.time() - t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path, default=M.CACHE)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    build("development_main", args.out / "development_main_ops.npz",
          None, None, args.workers, save_calib=True)
    build("external_8b", args.out / "external_8b_ops_frozenvalue.npz",
          args.out / "development_main_calibration.pkl",
          args.out / "development_main_grid.pkl", args.workers, save_calib=False)
    build("external_8b", args.out / "external_8b_ops_ratematched.npz",
          args.out / "development_main_calibration.pkl",
          None, args.workers, save_calib=False)


if __name__ == "__main__":
    main()
