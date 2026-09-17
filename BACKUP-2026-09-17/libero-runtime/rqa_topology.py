"""Recurrence quantification (RQA) of MoE trajectories vs physical loops, stagnation and outcome.

Uses the 60 causal distance matrices per episode stored by the topology grid
(moe-capture/topo-20260916/_grid/episodes/<tag>.npz), the episode traces (end-effector
position and gripper at every query boundary) and the frozen v8.2 traces.  No model calls.

Per causal window of W queries ending at q:
  RR   recurrence rate: pairs (i<j, j-i>=theiler) with d<=eps
  RET  true-return rate: fraction of window points j that recur (d<=eps) with an earlier i
       (j-i>=theiler) after the path first left the 2*eps ball around i  (loop, not staying)
  LAM  laminarity: fraction of consecutive steps with d<=eps               (staying put)
  LMAX longest diagonal line with offset>=theiler                         (repeated sub-trajectory)
  DIAM window diameter (magnitude baseline)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

GRID = Path("/home/swj/data/moe-capture/topo-20260916/_grid/episodes")
SIM = Path("/home/swj/data/libero-runtime/simulations")
OUT = Path("/home/swj/data/libero-runtime/samples/moe-recurrence-20260916")
BEND = Path("/home/swj/data/libero-runtime/samples/flow-path-straightness-20260916/trap-chunk-rows.npy")
W = 14
ANCHORS = (16, 20, 24, 28, 32, 36, 40)
LOOP_R, LOOP_EXC, LOOP_LOOKBACK = 0.010, 0.025, 12     # m: return within 1 cm after an excursion >= 2.5 cm
TAU, GTAU = 0.005, 0.005
MAIN = dict(alpha=0.10, theiler=3, eps="global")
VARIANTS = [dict(alpha=a, theiler=t, eps=e) for a in (0.05, 0.10, 0.20) for t in (2, 3) for e in ("global", "prefix")]
SUBSET = ["route|back_path", "input|back_path", "shared|back_path", "total|back_path", "input_total|back_path",
          "route|full_path", "input|full_path", "total|full_path", "route|back_last", "input|back_last", "total|layer3", "shared|layer5"]


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def task_auroc(vals, labels, tasks):
    out = []
    for t in sorted(set(tasks)):
        m = tasks == t
        a = auroc(vals[m & labels], vals[m & ~labels])
        if np.isfinite(a):
            out.append(a)
    return float(np.mean(out)) if out else np.nan, len(out)


def load_all():
    eps = []
    for f in sorted(GRID.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        tag, ds = meta["tag"], meta["dataset"]
        sim = list((SIM / ds / tag).glob("episode-*"))
        if len(sim) != 1:
            raise RuntimeError(tag)
        tr = np.load(sim[0] / "episode-trace.npz")
        pos = np.concatenate([tr["states"][:, :3], tr["final_state"][None, :3]])
        grip = np.concatenate([tr["states"][:, 6:8], tr["final_state"][None, 6:8]])
        Q = int(z["matrices"].shape[1])
        if tr["states"].shape[0] != Q:
            raise RuntimeError("query mismatch " + tag)
        eps.append(dict(tag=tag, ds=ds, key=ds + "__" + tag, base=int(tag[4:6]), success=bool(meta["success"]), Q=Q,
                        D=z["matrices"].astype(np.float64), names=list(z["names"]), v82=z["v82"], pos=pos, ap=grip[:, 0] - grip[:, 1]))
    return eps


def physical(e):
    pos, ap, Q = e["pos"], e["ap"], e["Q"]
    disp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    stag = (disp < TAU) & (np.abs(np.diff(ap)) < GTAU)
    loop_b = np.zeros(Q + 1, bool)
    for q in range(3, Q + 1):
        for p in range(max(0, q - LOOP_LOOKBACK), q - 2):
            if np.linalg.norm(pos[q] - pos[p]) < LOOP_R and abs(ap[q] - ap[p]) < 0.01:
                if max(np.linalg.norm(pos[k] - pos[p]) for k in range(p + 1, q)) >= LOOP_EXC:
                    loop_b[q] = True
                    break
    loop = loop_b[:Q].copy()
    loop[Q - 1] |= loop_b[Q]
    return disp, stag, loop


def rqa(S, eps, wt):
    w = len(S)
    R = S <= eps
    J = np.arange(w)
    far = (J[None, :] - J[:, None]) >= wt
    upper = far & (J[None, :] > J[:, None])
    rr = float(R[upper].mean()) if upper.any() else np.nan
    cm = np.full((w, w), -np.inf)
    for i in range(w - 1):
        cm[i, i + 1:] = np.maximum.accumulate(S[i, i + 1:])
    exc = np.zeros((w, w), bool)
    exc[:, 1:] = cm[:, :-1] > 2 * eps
    ret_pairs = R & exc & upper
    ret = float(ret_pairs.any(axis=0)[wt:].mean())
    lam = float(np.mean(np.diag(S, 1) <= eps))
    lmax = 0
    for k in range(wt, w):
        run = 0
        for v in np.diag(R, k):
            run = run + 1 if v else 0
            lmax = max(lmax, run)
    return rr, ret, lam, lmax


def global_eps(eps_list, names, alpha, wt):
    out = {}
    for mi, name in enumerate(names):
        vals = []
        for e in eps_list:
            S = e["D"][mi][:W, :W]
            J = np.arange(len(S))
            vals.append(S[(J[None, :] - J[:, None]) >= wt])
        out[name] = float(np.quantile(np.concatenate(vals), alpha))
    return out


def prefix_eps(D, q, alpha, wt):
    S = D[:q + 1, :q + 1]
    J = np.arange(len(S))
    v = S[(J[None, :] - J[:, None]) >= wt]
    return float(np.quantile(v, alpha)) if v.size else np.nan


def per_query_features(eps_list, names, setting):
    """returns dict name -> array (episode, q) of [rr, ret, lam, lmax, diam]"""
    alpha, wt, mode = setting["alpha"], setting["theiler"], setting["eps"]
    g = global_eps(eps_list, names, alpha, wt) if mode == "global" else None
    feats = {}
    for mi, name in enumerate(names):
        rows = []
        for e in eps_list:
            D = e["D"][mi]
            arr = np.full((e["Q"], 5), np.nan)
            for q in range(W - 1, e["Q"]):
                S = D[q - W + 1:q + 1, q - W + 1:q + 1]
                ep = g[name] if mode == "global" else prefix_eps(D, q, alpha, wt)
                if not np.isfinite(ep) or ep <= 0:
                    continue
                rr, ret, lam, lmax = rqa(S, ep, wt)
                arr[q] = [rr, ret, lam, lmax, S.max()]
            rows.append(arr)
        feats[name] = rows
    return feats


def whole_episode(eps_list, names, alpha, wt):
    g = global_eps(eps_list, names, alpha, wt)
    out = {}
    for mi, name in enumerate(names):
        vals = []
        for e in eps_list:
            rr, ret, lam, lmax = rqa(e["D"][mi], g[name], wt)
            vals.append([rr, ret, lam, lmax])
        out[name] = np.array(vals)
    return out


def main():
    eps_list = load_all()
    names = eps_list[0]["names"]
    n_succ = sum(e["success"] for e in eps_list)
    print("episodes %d (success %d), matrices %d" % (len(eps_list), n_succ, len(names)))
    phys = [physical(e) for e in eps_list]
    for e, (disp, stag, loop) in zip(eps_list, phys):
        e.update(disp=disp, stag=stag, loop=loop)
    bend = {}
    for r in np.load(BEND, allow_pickle=True):
        bend[(r["episode"], r["q"])] = r["ld"]
    succ = np.array([e["success"] for e in eps_list])
    tasks = np.array([e["base"] for e in eps_list])
    ds = np.array([e["ds"] for e in eps_list])
    res = {"n_episodes": len(eps_list), "n_success": int(n_succ)}

    # ---- physical loop labels: how common, in which episodes
    loop_count = np.array([int(e["loop"].sum()) for e in eps_list])
    stag_frac = np.array([float(e["stag"].mean()) for e in eps_list])
    loop_ep = loop_count >= 2
    print("physical loops: episodes with >=1 event %d, >=2 events %d (failures %d, successes %d); total events %d" % (
        (loop_count >= 1).sum(), loop_ep.sum(), (loop_ep & ~succ).sum(), (loop_ep & succ).sum(), loop_count.sum()))
    print("failure types: loop(>=2 events) %d, stagnant>=30%% without loops %d, other %d of %d failures" % (
        (loop_ep & ~succ).sum(), ((~loop_ep) & (stag_frac >= 0.3) & ~succ).sum(), ((~loop_ep) & (stag_frac < 0.3) & ~succ).sum(), (~succ).sum()))
    res["physical"] = dict(loop_events_total=int(loop_count.sum()), loop_episodes=int(loop_ep.sum()), loop_failures=int((loop_ep & ~succ).sum()),
                           loop_successes=int((loop_ep & succ).sum()), stagnant_failures_without_loops=int(((~loop_ep) & (stag_frac >= 0.3) & ~succ).sum()))

    # ---- E1: whole-episode RQA (non-causal, mechanism) vs physical loops / stagnation / outcome
    we = whole_episode(eps_list, names, MAIN["alpha"], MAIN["theiler"])
    e1 = []
    for name in names:
        v = we[name]
        e1.append(dict(matrix=name,
                       ret_vs_loop_episode=auroc(v[loop_ep, 1], v[~loop_ep, 1]),
                       ret_vs_loop_failures_only=auroc(v[loop_ep & ~succ, 1], v[~loop_ep & ~succ, 1]),
                       lmax_vs_loop_episode=auroc(v[loop_ep, 3], v[~loop_ep, 3]),
                       ret_spearman_loop_count=float(spearmanr(v[:, 1], loop_count).statistic),
                       lam_spearman_stag_frac=float(spearmanr(v[:, 2], stag_frac).statistic),
                       ret_vs_failure=auroc(v[~succ, 1], v[succ, 1]), rr_vs_failure=auroc(v[~succ, 0], v[succ, 0]),
                       lam_vs_failure=auroc(v[~succ, 2], v[succ, 2]), lmax_vs_failure=auroc(v[~succ, 3], v[succ, 3])))
    res["E1_whole_episode"] = e1
    def med(key):
        return float(np.nanmedian([r[key] for r in e1]))
    def best(key):
        r = max(e1, key=lambda r: abs(r[key] - 0.5) if np.isfinite(r[key]) else -1)
        return r["matrix"], r[key]
    print("E1 whole-episode (alpha=%.2f, theiler=%d, global eps): median over 60 matrices / best" % (MAIN["alpha"], MAIN["theiler"]))
    for key in ("ret_vs_loop_episode", "ret_vs_loop_failures_only", "lmax_vs_loop_episode", "lam_spearman_stag_frac",
                "ret_vs_failure", "rr_vs_failure", "lam_vs_failure", "lmax_vs_failure"):
        print("   %-28s median %.3f   best %s = %.3f" % (key, med(key), *best(key)))

    # ---- per-query causal features (main setting) for E2 / E3
    feats = per_query_features(eps_list, names, MAIN)
    # E2: detect physical loops in moving windows
    rows_e2 = []
    for ei, e in enumerate(eps_list):
        for q in range(W - 1, e["Q"]):
            lo = max(0, q - 5)
            label = bool(e["loop"][lo:q + 1].any())
            moving = float(e["stag"][lo:q + 1].mean()) <= 0.5
            rows_e2.append((ei, q, label, moving))
    rows_e2 = np.array(rows_e2, dtype=object)
    sel = np.array([r[3] for r in rows_e2], bool)
    lab = np.array([r[2] for r in rows_e2], bool)
    ei_arr = np.array([r[0] for r in rows_e2]); q_arr = np.array([r[1] for r in rows_e2])
    print("E2 windows (q>=13): %d, moving %d, of which with a physical loop event in [q-5,q]: %d" % (len(rows_e2), sel.sum(), (lab & sel).sum()))
    e2 = []
    for name in names:
        F = feats[name]
        vals = np.array([F[ei][q] for ei, q in zip(ei_arr, q_arr)])
        e2.append(dict(matrix=name, **{k: auroc(vals[sel & lab, c], vals[sel & ~lab, c]) for c, k in enumerate(("rr", "ret", "lam", "lmax", "diam"))}))
    freeze = np.array([eps_list[ei]["v82"][q, 1] for ei, q in zip(ei_arr, q_arr)])
    ldv = np.array([bend.get((eps_list[ei]["key"], q), np.nan) for ei, q in zip(ei_arr, q_arr)])
    res["E2_loop_detection"] = dict(rows=e2, baseline_freeze=auroc(freeze[sel & lab], freeze[sel & ~lab]), baseline_bending=auroc(ldv[sel & lab], ldv[sel & ~lab]),
                                    n_windows=int(sel.sum()), n_loop=int((lab & sel).sum()))
    print("E2 physical-loop detection in moving windows: baselines freeze %.3f, bending L/D %.3f" % (res["E2_loop_detection"]["baseline_freeze"], res["E2_loop_detection"]["baseline_bending"]))
    for k in ("ret", "lmax", "rr", "lam", "diam"):
        arr = np.array([r[k] for r in e2])
        b = e2[int(np.nanargmax(arr))]
        print("   %-5s median over matrices %.3f, best %s = %.3f, #matrices >= 0.6: %d" % (k, np.nanmedian(arr), b["matrix"], b[k], int((arr >= 0.6).sum())))

    # E3: outcome at anchors, causal, incl. late anchors
    e3 = {}
    print("E3 outcome AUROC (failure higher) at anchors, pooled | task-mean; n fail/succ")
    for a in ANCHORS:
        ok = np.array([e["Q"] > a for e in eps_list])
        nf, ns = int((ok & ~succ).sum()), int((ok & succ).sum())
        if ns < 3:
            print("  q%-2d (%3d/%3d)  skipped: too few successes" % (a, nf, ns)); continue
        line = {"n_fail": nf, "n_succ": ns, "features": {}}
        fr = np.array([e["v82"][a, 1] if e["Q"] > a else np.nan for e in eps_list])
        ld = np.array([bend.get((e["key"], a), np.nan) for e in eps_list])
        line["features"]["v82_freeze"] = dict(pooled=auroc(fr[ok & ~succ], fr[ok & succ]), task=task_auroc(fr[ok], ~succ[ok], tasks[ok])[0])
        line["features"]["bending_ld"] = dict(pooled=auroc(ld[ok & ~succ], ld[ok & succ]), task=task_auroc(ld[ok], ~succ[ok], tasks[ok])[0])
        for c, k in enumerate(("rr", "ret", "lam", "lmax", "diam")):
            per = []
            for name in names:
                v = np.array([feats[name][ei][a, c] if eps_list[ei]["Q"] > a else np.nan for ei in range(len(eps_list))])
                per.append((name, auroc(v[ok & ~succ], v[ok & succ]), task_auroc(v[ok], ~succ[ok], tasks[ok])[0]))
            pooled = np.array([p[1] for p in per]); tk = np.array([p[2] for p in per])
            bi = int(np.nanargmax(np.abs(pooled - 0.5)))
            line["features"][k] = dict(median_pooled=float(np.nanmedian(pooled)), median_task=float(np.nanmedian(tk)), best_matrix=per[bi][0], best_pooled=float(pooled[bi]), best_task=float(tk[bi]),
                                       n_matrices_pooled_ge_0_65=int((pooled >= 0.65).sum()), n_matrices_pooled_le_0_35=int((pooled <= 0.35).sum()))
        e3["q%d" % a] = line
        f = line["features"]
        print("  q%-2d (%3d/%3d)  freeze %.3f|%.3f  bend %.3f|%.3f  RET med %.3f|%.3f best %.3f (%s)  LMAX med %.3f best %.3f  LAM med %.3f best %.3f  RR med %.3f best %.3f  DIAM med %.3f best %.3f" % (
            a, nf, ns, f["v82_freeze"]["pooled"], f["v82_freeze"]["task"], f["bending_ld"]["pooled"], f["bending_ld"]["task"],
            f["ret"]["median_pooled"], f["ret"]["median_task"], f["ret"]["best_pooled"], f["ret"]["best_matrix"],
            f["lmax"]["median_pooled"], f["lmax"]["best_pooled"], f["lam"]["median_pooled"], f["lam"]["best_pooled"], f["rr"]["median_pooled"], f["rr"]["best_pooled"], f["diam"]["median_pooled"], f["diam"]["best_pooled"]))
    res["E3_outcome_anchors"] = e3

    # E5: does recurrence add information beyond magnitude?  diameter-matched AUROC, failure-type split, combination, per benchmark
    e5 = {}
    stag_at = lambda e, a: float(e["stag"][max(0, a - 5):a + 1].mean())
    print("E5 beyond magnitude (median over 60 matrices [best]); oriented so that failure scores higher")
    for a in (16, 20, 24):
        ok = np.array([e["Q"] > a for e in eps_list])
        moving_fail = np.array([e["Q"] > a and not e["success"] and stag_at(e, a) < 0.5 for e in eps_list])
        stag_fail = np.array([e["Q"] > a and not e["success"] and stag_at(e, a) >= 0.5 for e in eps_list])
        succ_ok = ok & succ
        line = {"n_moving_fail": int(moving_fail.sum()), "n_stag_fail": int(stag_fail.sum()), "n_succ": int(succ_ok.sum())}
        for k, c, orient in (("rr", 0, 1), ("lmax", 3, 1), ("lam", 2, 1), ("diam", 4, -1)):
            matched, movef, stagf, comb, alone, lib, plus = [], [], [], [], [], [], []
            for name in names:
                v = np.array([feats[name][ei][a, c] if eps_list[ei]["Q"] > a else np.nan for ei in range(len(eps_list))]) * orient
                dm = np.array([feats[name][ei][a, 4] if eps_list[ei]["Q"] > a else np.nan for ei in range(len(eps_list))])
                # diameter-matched: AUROC within diameter tertiles of the eligible population, weighted by min class size
                qs = np.nanquantile(dm[ok], [1 / 3, 2 / 3]); acc, wsum = 0.0, 0.0
                for lo, hi in ((-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf)):
                    m = ok & (dm > lo) & (dm <= hi)
                    au = auroc(v[m & ~succ], v[m & succ]); wgt = min((m & ~succ).sum(), (m & succ).sum())
                    if np.isfinite(au) and wgt >= 3:
                        acc += au * wgt; wsum += wgt
                matched.append(acc / wsum if wsum else np.nan)
                movef.append(auroc(v[moving_fail], v[succ_ok])); stagf.append(auroc(v[stag_fail], v[succ_ok]))
                lib.append(auroc(v[ok & ~succ & (ds == "topo-libero10")], v[succ_ok & (ds == "topo-libero10")]))
                plus.append(auroc(v[ok & ~succ & (ds == "topo-plus")], v[succ_ok & (ds == "topo-plus")]))
                if k != "diam":
                    from scipy.stats import rankdata
                    rv, rd = np.full(len(v), np.nan), np.full(len(v), np.nan)
                    rv[ok] = rankdata(v[ok]); rd[ok] = rankdata(-dm[ok])
                    comb.append(auroc((rv + rd)[ok & ~succ], (rv + rd)[ok & succ])); alone.append(auroc(rd[ok & ~succ], rd[ok & succ]))
            line[k] = dict(diam_matched_median=float(np.nanmedian(matched)), diam_matched_best=float(np.nanmax(matched)),
                           moving_fail_vs_succ_median=float(np.nanmedian(movef)), moving_fail_vs_succ_best=float(np.nanmax(movef)),
                           stag_fail_vs_succ_median=float(np.nanmedian(stagf)), libero10_median=float(np.nanmedian(lib)), plus_median=float(np.nanmedian(plus)))
            if k != "diam":
                line[k].update(combined_with_diam_median=float(np.nanmedian(comb)), diam_alone_median=float(np.nanmedian(alone)), combined_gain_median=float(np.nanmedian(np.array(comb) - np.array(alone))))
            print("  q%d %-4s diam-matched %.3f [%.3f] | moving-fail vs succ %.3f [%.3f] (n %d) | stag-fail vs succ %.3f (n %d) | libero10 %.3f plus %.3f%s" % (
                a, k, line[k]["diam_matched_median"], line[k]["diam_matched_best"], line[k]["moving_fail_vs_succ_median"], line[k]["moving_fail_vs_succ_best"], line["n_moving_fail"],
                line[k]["stag_fail_vs_succ_median"], line["n_stag_fail"], line[k]["libero10_median"], line[k]["plus_median"],
                (" | rank(RQA)+rank(-diam) %.3f vs diam alone %.3f (gain %+.3f)" % (line[k]["combined_with_diam_median"], line[k]["diam_alone_median"], line[k]["combined_gain_median"])) if k != "diam" else ""))
        # baselines for the moving-failure split
        fr = np.array([e["v82"][a, 1] if e["Q"] > a else np.nan for e in eps_list]); ld = np.array([bend.get((e["key"], a), np.nan) for e in eps_list])
        line["baselines_moving_fail_vs_succ"] = dict(freeze=auroc(fr[moving_fail], fr[succ_ok]), bending=auroc(ld[moving_fail], ld[succ_ok]))
        print("  q%d baselines moving-fail vs succ: freeze %.3f, bending %.3f" % (a, line["baselines_moving_fail_vs_succ"]["freeze"], line["baselines_moving_fail_vs_succ"]["bending"]))
        e5["q%d" % a] = line
    res["E5_beyond_magnitude"] = e5

    # E4: robustness of RET/LMAX across thresholds and Theiler windows on a subset of matrices (E2 loop detection and q24 outcome)
    e4 = []
    sub_idx = [names.index(n) for n in SUBSET]
    for st in VARIANTS:
        f2 = per_query_features([dict(e, D=e["D"][sub_idx], names=SUBSET) for e in eps_list], SUBSET, st)
        row = dict(setting=st)
        for k, c in (("ret", 1), ("lmax", 3), ("rr", 0)):
            loop_aucs, out_aucs = [], []
            for name in SUBSET:
                vals = np.array([f2[name][ei][q][c] for ei, q in zip(ei_arr, q_arr)])
                loop_aucs.append(auroc(vals[sel & lab], vals[sel & ~lab]))
                a = 24
                v = np.array([f2[name][ei][a][c] if eps_list[ei]["Q"] > a else np.nan for ei in range(len(eps_list))])
                ok = np.array([e["Q"] > a for e in eps_list])
                out_aucs.append(auroc(v[ok & ~succ], v[ok & succ]))
            row[k] = dict(loop_median=float(np.nanmedian(loop_aucs)), loop_max=float(np.nanmax(loop_aucs)), q24_median=float(np.nanmedian(out_aucs)), q24_min=float(np.nanmin(out_aucs)), q24_max=float(np.nanmax(out_aucs)))
        e4.append(row)
        print("E4 alpha=%.2f theiler=%d eps=%-6s  RET loop med/max %.3f/%.3f q24 med [%.3f..%.3f] | LMAX loop %.3f/%.3f q24 med %.3f | RR loop %.3f q24 med %.3f" % (
            st["alpha"], st["theiler"], st["eps"], row["ret"]["loop_median"], row["ret"]["loop_max"], row["ret"]["q24_min"], row["ret"]["q24_max"],
            row["lmax"]["loop_median"], row["lmax"]["loop_max"], row["lmax"]["q24_median"], row["rr"]["loop_median"], row["rr"]["q24_median"]))
    res["E4_robustness"] = e4

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rqa-results.json").write_text(json.dumps(clean(res), indent=2))
    np.savez_compressed(OUT / "rqa-per-query.npz", names=np.array(names), keys=np.array([e["key"] for e in eps_list]), success=succ, tasks=tasks, dataset=ds,
                        loop_count=loop_count, stag_frac=stag_frac,
                        **{"feat__" + n: np.array([np.pad(f, ((0, 52 - len(f)), (0, 0)), constant_values=np.nan) for f in feats[n]]) for n in names},
                        **{"loop__" + e["key"]: e["loop"] for e in eps_list}, **{"stag__" + e["key"]: e["stag"] for e in eps_list})
    print("saved", OUT / "rqa-results.json")


if __name__ == "__main__":
    main()
