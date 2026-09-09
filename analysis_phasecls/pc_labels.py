"""Derive corpus-B failure-type labels with corpus A's frozen predicate.

Corpus A stores DENSE physics (control_sim_state, 521 rows); corpus B stores only
QUERY-resolution physics (sim_state, 52 rows).  The loop predicate is already a
query-level predicate and transfers exactly.  The stagnation predicate is dense
(20-dense-step rolling window), so a query-resolution surrogate is defined here
and VALIDATED against corpus A's dense ground truth before it is used on B.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jovyan/work/himoe-vla")
OUT = ROOT / "analysis_phasecls"
A_ROOT = ROOT / "himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
B_ROOT = (ROOT / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
          "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")

# ---- frozen constants, copied verbatim from analyze_rolling_star_experiment.py
STATIC_WINDOW = 20
STATIC_EEF_PATH_M = 0.020
STATIC_OBJECT_PATH_M = 0.005
STATIC_GRIPPER_PATH_M = 0.001
LOOP_MIN_QUERY_LAG = 3
LOOP_EEF_RETURN_M = 0.045
LOOP_OBJECT_RETURN_M = 0.030
LOOP_GRIPPER_RETURN_M = 0.012
LOOP_EEF_PATH_M = 0.120
LOOP_PROGRESS_M = 0.035
POT_SLOTS = (10, 17)          # moka_pot_1_joint0, moka_pot_2_joint0
CHUNK = 10                    # dense control steps per inference call


def longest_true_run(flags: np.ndarray) -> int:
    best = cur = 0
    for v in flags:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def dist_to_refs(pos: np.ndarray, refs: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pos[:, None, :].astype(np.float32)
                          - refs[None, :, :].astype(np.float32), axis=-1).min(axis=1)


def query_loop_metrics(eef, objects, gripper, goal):
    """Verbatim port of analyze_rolling_star_experiment.query_loop_metrics."""
    n = len(eef)
    if n <= LOOP_MIN_QUERY_LAG:
        return dict(loop_return_count=0.0, loop_return_fraction=0.0)
    step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(step)]
    count = 0
    returned = set()
    for right in range(LOOP_MIN_QUERY_LAG, n):
        for left in range(0, right - LOOP_MIN_QUERY_LAG + 1):
            if np.linalg.norm(eef[right] - eef[left]) > LOOP_EEF_RETURN_M:
                continue
            if np.linalg.norm(objects[right] - objects[left], axis=1).max() > LOOP_OBJECT_RETURN_M:
                continue
            if abs(gripper[right] - gripper[left]) > LOOP_GRIPPER_RETURN_M:
                continue
            if cum[right] - cum[left] < LOOP_EEF_PATH_M:
                continue
            if goal[left] - goal[right] > LOOP_PROGRESS_M:
                continue
            count += 1
            returned.add(right)
    return dict(loop_return_count=float(count),
                loop_return_fraction=float(len(returned) / n))


def static_metrics_dense(eef, objects, gripper):
    """Verbatim dense predicate (corpus A only)."""
    if len(eef) - 1 < STATIC_WINDOW:
        return dict(late_static_window_fraction=0.0, terminal_static_window_steps=0.0)
    es = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    os_ = np.linalg.norm(np.diff(objects, axis=0), axis=2).max(axis=1)
    gs = np.abs(np.diff(gripper))
    k = np.ones(STATIC_WINDOW)
    static = ((np.convolve(es, k, "valid") <= STATIC_EEF_PATH_M)
              & (np.convolve(os_, k, "valid") <= STATIC_OBJECT_PATH_M)
              & (np.convolve(gs, k, "valid") <= STATIC_GRIPPER_PATH_M))
    term = 0
    for v in static[::-1]:
        if not v:
            break
        term += 1
    return dict(late_static_window_fraction=float(static[len(static) // 2:].mean()),
                terminal_static_window_steps=float(term + STATIC_WINDOW - 1 if term else 0.0))


def static_metrics_query(eef, objects, gripper):
    """Query-resolution surrogate.  One query interval = CHUNK dense steps, so a
    STATIC_WINDOW=20 dense window is 2 query intervals.  Thresholds unchanged."""
    n_int = len(eef) - 1
    w = STATIC_WINDOW // CHUNK              # = 2 query intervals
    if n_int < w:
        return dict(late_static_window_fraction=0.0, terminal_static_window_steps=0.0)
    es = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    os_ = np.linalg.norm(np.diff(objects, axis=0), axis=2).max(axis=1)
    gs = np.abs(np.diff(gripper))
    k = np.ones(w)
    static = ((np.convolve(es, k, "valid") <= STATIC_EEF_PATH_M)
              & (np.convolve(os_, k, "valid") <= STATIC_OBJECT_PATH_M)
              & (np.convolve(gs, k, "valid") <= STATIC_GRIPPER_PATH_M))
    term = 0
    for v in static[::-1]:
        if not v:
            break
        term += 1
    # a terminal run of `term` query-windows spans term*CHUNK dense windows
    return dict(late_static_window_fraction=float(static[len(static) // 2:].mean()),
                terminal_static_window_steps=float(term * CHUNK + STATIC_WINDOW - 1 if term else 0.0))


def classify(stag_terminal, stag_late, loop_count, loop_frac):
    stag = bool(stag_terminal >= 80 or stag_late >= 0.45)
    loop = bool(loop_count >= 2 or loop_frac >= 0.10)
    return stag, loop


# ---------------------------------------------------------------- corpus A
def corpus_a():
    lab = pd.read_csv(A_ROOT / "analysis/candidate_physical_labels.csv")
    paths = {}
    for w in range(4):
        for p in sorted((A_ROOT / f"formal/worker{w}").glob("snapshot_*/candidate_*.npz")):
            paths[(w, int(p.parent.name.split("_")[1]), int(p.stem.split("_")[1]))] = p
    # shared goal reference cloud from dense terminal states of successes
    succ = lab[lab["success"]]
    pooled = []
    for r in succ.itertuples():
        d = np.load(paths[(r.worker, r.snapshot, r.candidate)])
        for lo in POT_SLOTS:
            pooled.append(d["control_sim_state"][-1, lo:lo + 3])
    refs = np.stack(pooled).astype(np.float32)

    rows = []
    for r in lab.itertuples():
        d = np.load(paths[(r.worker, r.snapshot, r.candidate)])
        cs = d["control_sim_state"]
        eef_d = d["control_eef_position"].astype(np.float32)
        grip_d = d["control_gripper_qpos"].astype(np.float32).mean(axis=1)
        obj_d = np.stack([cs[:, lo:lo + 3] for lo in POT_SLOTS], axis=1).astype(np.float32)
        goal_d = np.stack([dist_to_refs(obj_d[:, i], refs) for i in range(2)], 1).mean(1)
        qs = np.r_[0, np.flatnonzero(np.diff(d["control_query_index"]) != 0) + 1][:r.inference_calls]
        md = static_metrics_dense(eef_d, obj_d, grip_d)
        mq = static_metrics_query(eef_d[qs], obj_d[qs], grip_d[qs])
        lp = query_loop_metrics(eef_d[qs], obj_d[qs], grip_d[qs], goal_d[qs])
        sd, ld = classify(md["terminal_static_window_steps"], md["late_static_window_fraction"],
                          lp["loop_return_count"], lp["loop_return_fraction"])
        sq, _ = classify(mq["terminal_static_window_steps"], mq["late_static_window_fraction"],
                         lp["loop_return_count"], lp["loop_return_fraction"])
        rows.append(dict(episode_id=r.episode_id, worker=r.worker, success=bool(r.success),
                         n_q=int(r.inference_calls),
                         rederived_stag_dense=sd, rederived_stag_query=sq, rederived_loop=ld,
                         ref_stag=bool(r.label_stagnation), ref_loop=bool(r.label_loop_or_cycling),
                         **{f"d_{k}": v for k, v in md.items()},
                         **{f"q_{k}": v for k, v in mq.items()}, **lp))
    df = pd.DataFrame(rows)
    np.save(OUT / "refs_A.npy", refs)
    return df


# ---------------------------------------------------------------- corpus B
def corpus_b():
    summ = json.load(open(B_ROOT / "client/summaries.json"))
    def arrays(i):
        d = np.load(B_ROOT / f"client/episode_{i:02d}.npz", allow_pickle=True)
        st = d["state"].astype(np.float32)
        sim = d["sim_state"].astype(np.float32)
        eef = st[:, :3]
        grip = st[:, 6:8].mean(axis=1)
        obj = np.stack([sim[:, lo:lo + 3] for lo in POT_SLOTS], axis=1)
        return eef, obj, grip

    pooled = []
    for i, s in enumerate(summ):
        if s["success"]:
            _, obj, _ = arrays(i)
            pooled.append(obj[-1])
    refs = np.concatenate(pooled, axis=0).astype(np.float32)
    np.save(OUT / "refs_B.npy", refs)

    rows = []
    for i, s in enumerate(summ):
        eef, obj, grip = arrays(i)
        goal = np.stack([dist_to_refs(obj[:, j], refs) for j in range(2)], 1).mean(1)
        mq = static_metrics_query(eef, obj, grip)
        lp = query_loop_metrics(eef, obj, grip, goal)
        sq, lq = classify(mq["terminal_static_window_steps"], mq["late_static_window_fraction"],
                          lp["loop_return_count"], lp["loop_return_fraction"])
        rows.append(dict(episode_id=i, init_state_id=s["init_state_id"], seed=s["seed"],
                         success=bool(s["success"]), n_q=int(s["inference_calls"]),
                         stag=sq, loop=lq, **{f"q_{k}": v for k, v in mq.items()}, **lp))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    a = corpus_a()
    a.to_csv(OUT / "labels_A_rederived.csv", index=False)
    fail = a[~a["success"]]
    print("=== corpus A: re-derived vs shipped labels (failures only) ===")
    for nm, col in [("stagnation dense", "rederived_stag_dense"),
                    ("stagnation query-surrogate", "rederived_stag_query"),
                    ("loop", "rederived_loop")]:
        ref = fail["ref_stag"] if "stag" in col else fail["ref_loop"]
        agree = (fail[col] == ref).mean()
        print(f"  {nm:28s} n_pos={int(fail[col].sum()):4d} ref={int(ref.sum()):4d} "
              f"agree={agree:.4f}  (TP={int((fail[col]&ref).sum())} "
              f"FP={int((fail[col]&~ref).sum())} FN={int((~fail[col]&ref).sum())})")
    b = corpus_b()
    b.to_csv(OUT / "labels_B_derived.csv", index=False)
    print("\n=== corpus B ===")
    print(f"  episodes={len(b)}  successes={int(b['success'].sum())}  "
          f"failures={int((~b['success']).sum())}  init_states={b['init_state_id'].nunique()}")
    bf = b[~b["success"]].copy()
    bf["cls"] = np.where(bf["stag"], "stagnation",
                         np.where(bf["loop"], "loop", "other"))
    print(bf["cls"].value_counts().to_string())
    ra, rb = np.load(OUT / "refs_A.npy"), np.load(OUT / "refs_B.npy")
    print(f"\n  refs A n={len(ra)} mean={ra.mean(0).round(4)} std={ra.std(0).round(4)}")
    print(f"  refs B n={len(rb)} mean={rb.mean(0).round(4)} std={rb.std(0).round(4)}")
    print(f"  min dist A-cloud -> B-cloud mean = "
          f"{np.linalg.norm(ra[:,None]-rb[None],axis=-1).min(1).mean():.4f} m")
