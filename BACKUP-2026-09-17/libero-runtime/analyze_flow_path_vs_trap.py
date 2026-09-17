"""Does denoising-path bending relate to physical stagnation and to escaping from it?

Inputs: per-query denoising paths replayed by probe_flow_path_straightness.py (topo-workers/w*/*.npz),
episode traces (end-effector position and gripper at every query boundary), and the frozen v8.2
per-query traces computed by the topology grid run. No model calls, no environment actions.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

RT = Path("/home/swj/data/libero-runtime")
sys.path.insert(0, str(RT))
from probe_flow_path_straightness import straightness  # noqa: E402

GRID = Path("/home/swj/data/moe-capture/topo-20260916/_grid/episodes")
OUT = RT / "samples/flow-path-straightness-20260916"
WORK = OUT / "topo-workers"
TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005   # m of end-effector travel per 10-step chunk
GTAU = 0.005                                                # gripper aperture change per chunk
MIN_RUN = 2


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u.statistic / (len(pos) * len(neg))), float(u.pvalue)


def episode_bootstrap_auroc(records, key, label, n=2000, seed=0):
    """AUROC of records[key] for label True vs False, CI by resampling episodes."""
    eps = sorted({r["episode"] for r in records})
    rng = np.random.default_rng(seed)
    by_ep = {e: [r for r in records if r["episode"] == e] for e in eps}
    def stat(sample):
        rows = [r for e in sample for r in by_ep[e]]
        p = [r[key] for r in rows if r[label]]
        q = [r[key] for r in rows if not r[label]]
        return auroc(p, q)[0] if p and q else np.nan
    point = stat(eps)
    boots = [stat(list(rng.choice(eps, size=len(eps), replace=True))) for _ in range(n)]
    boots = np.array([b for b in boots if np.isfinite(b)])
    return point, (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots.size else (np.nan, np.nan)


def load_episodes():
    out = []
    seen = set()
    for npz in sorted(WORK.glob("*/*.npz")):
        name = npz.stem
        if "__" in name:  # <dataset>__<tag> naming from the rerun of duplicated episode names
            dataset, tag = name.split("__", 1)
            hits = list((RT / "simulations" / dataset / tag).glob("episode-*"))
        else:
            hits = list((RT / "simulations").glob("topo-*/*/" + name))
        if len(hits) != 1:
            continue  # ambiguous episode name: covered by the tag-named rerun
        sim = hits[0]
        if str(sim) in seen:
            continue
        seen.add(str(sim))
        name = sim.parent.parent.name + "__" + sim.parent.name
        tag, dataset = sim.parent.name, sim.parent.parent.name
        z = np.load(npz)
        dims = np.flatnonzero(z["data_mask"])
        Q = z["X"].shape[0]
        rows = [straightness(z["X"][q], z["V"][q], dims) for q in range(Q)]
        tr = np.load(sim / "episode-trace.npz")
        summ = json.loads((sim / "summary.json").read_text())
        if tr["predicted_actions"].shape[0] != Q:
            raise RuntimeError("query count mismatch " + name)
        pos = np.concatenate([tr["states"][:, :3], tr["final_state"][None, :3]])
        grip = np.concatenate([tr["states"][:, 6:8], tr["final_state"][None, 6:8]])
        disp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        dgrip = np.abs(np.diff(grip[:, 0] - grip[:, 1]))
        g = GRID / (tag + ".npz")
        v82 = np.load(g, allow_pickle=True)["v82"] if g.exists() else np.full((Q, 4), np.nan)
        if v82.shape[0] != Q:
            raise RuntimeError("v82 length mismatch " + name)
        act = tr["predicted_actions"]
        dact = np.r_[np.nan, np.linalg.norm((act[1:, :, :6] - act[:-1, :, :6]).reshape(Q - 1, -1), axis=1)]
        out.append(dict(
            episode=name, tag=tag, dataset=dataset, success=bool(summ["success"]), Q=Q,
            ld=np.array([r["path_over_chord"] for r in rows]),
            angle=np.array([r["angle_first_last_deg"] for r in rows]),
            perp=np.array([r["max_perp_over_chord"] for r in rows]),
            one_step=np.array([r["one_step_endpoint_error"] for r in rows]),
            disp=disp, dgrip=dgrip, v82=v82, dact=dact,
        ))
    return out


def runs_of(flags):
    runs, s = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            runs.append((s, i - 1)); s = None
    return runs


def main():
    eps = load_episodes()
    n_succ = sum(e["success"] for e in eps)
    print("episodes %d (success %d), queries %d" % (len(eps), n_succ, sum(e["Q"] for e in eps)))
    all_disp = np.concatenate([e["disp"] for e in eps])
    print("eef displacement per chunk (m): p10 %.4f p25 %.4f p50 %.4f p75 %.4f p90 %.4f; TAU=%.4f" % (
        *np.percentile(all_disp, [10, 25, 50, 75, 90]), TAU))
    res = {"n_episodes": len(eps), "n_success": n_succ, "tau_m": TAU, "gtau": GTAU}

    # ---- T1: bending during stagnant vs moving chunks
    chunk_rows = []
    for e in eps:
        stag = (e["disp"] < TAU) & (e["dgrip"] < GTAU)
        moving = (e["disp"] >= 2 * TAU) | (e["dgrip"] >= 2 * GTAU)
        for q in range(e["Q"]):
            chunk_rows.append(dict(episode=e["episode"], q=q, success=e["success"], stag=bool(stag[q]),
                                   moving=bool(moving[q]), ld=e["ld"][q], angle=e["angle"][q], perp=e["perp"][q],
                                   one_step=e["one_step"][q], freeze=e["v82"][q, 1], alarm=e["v82"][q, 0],
                                   disp_next=e["disp"][q + 1] if q + 1 < e["Q"] else np.nan,
                                   stag_next=bool(stag[q + 1]) if q + 1 < e["Q"] else None, dact=e["dact"][q]))
    ld_stag = [r["ld"] for r in chunk_rows if r["stag"]]
    ld_mov = [r["ld"] for r in chunk_rows if r["moving"]]
    a1 = auroc(ld_stag, ld_mov)
    res["T1_stagnant_vs_moving"] = dict(n_stag=len(ld_stag), n_moving=len(ld_mov), median_ld_stag=float(np.median(ld_stag)),
                                        median_ld_moving=float(np.median(ld_mov)), auroc=a1[0], p=a1[1])
    print("T1 bending stagnant(n=%d, median L/D %.4f) vs moving(n=%d, %.4f): AUROC %.3f p=%.2g" % (
        len(ld_stag), np.median(ld_stag), len(ld_mov), np.median(ld_mov), a1[0], a1[1]))
    # success vs failure per chunk
    ld_s = [r["ld"] for r in chunk_rows if r["success"]]
    ld_f = [r["ld"] for r in chunk_rows if not r["success"]]
    res["episode_outcome_per_chunk"] = dict(median_ld_success=float(np.median(ld_s)), median_ld_failure=float(np.median(ld_f)),
                                            auroc_failure_vs_success=auroc(ld_f, ld_s)[0])
    ep_med_s = [np.median(e["ld"]) for e in eps if e["success"]]
    ep_med_f = [np.median(e["ld"]) for e in eps if not e["success"]]
    res["episode_outcome_per_episode"] = dict(n_success=len(ep_med_s), n_failure=len(ep_med_f),
                                              median_of_medians_success=float(np.median(ep_med_s)),
                                              median_of_medians_failure=float(np.median(ep_med_f)),
                                              auroc_failure_vs_success=auroc(ep_med_f, ep_med_s)[0], p=auroc(ep_med_f, ep_med_s)[1])
    print("outcome: per-episode median L/D failure %.4f vs success %.4f, AUROC %.3f (p=%.2g, %d vs %d episodes)" % (
        np.median(ep_med_f), np.median(ep_med_s), auroc(ep_med_f, ep_med_s)[0], auroc(ep_med_f, ep_med_s)[1], len(ep_med_f), len(ep_med_s)))

    # ---- T2: stagnation runs that end with resumed motion vs runs that persist to a failed end
    run_rows = []
    for e in eps:
        stag = (e["disp"] < TAU) & (e["dgrip"] < GTAU)
        for s, t in runs_of(stag):
            if t - s + 1 < MIN_RUN:
                continue
            ends_by_motion = t < e["Q"] - 1
            sustained = ends_by_motion and (t + 2 < e["Q"]) and (not stag[t + 1]) and (not stag[t + 2])
            zld = (e["ld"] - np.median(e["ld"])) / (np.std(e["ld"]) + 1e-9)
            run_rows.append(dict(episode=e["episode"], success=e["success"], start=s, end=t, length=t - s + 1,
                                 ends_by_motion=ends_by_motion, ends_at_end=not ends_by_motion,
                                 escape_then_success=ends_by_motion and e["success"], escape_sustained=sustained,
                                 zld_last=float(np.mean(zld[max(s, t - 1):t + 1])), q_end_bin=int(min(t // 15, 2)),
                                 ld_first=float(np.mean(e["ld"][s:s + 2])), ld_last=float(np.mean(e["ld"][max(s, t - 1):t + 1])),
                                 ld_max=float(e["ld"][s:t + 1].max()), ld_trend=float(np.mean(e["ld"][max(s, t - 1):t + 1]) - np.mean(e["ld"][s:s + 2])),
                                 ld_after=float(e["ld"][t + 1]) if ends_by_motion else np.nan,
                                 alarm_in_run=bool(np.nanmax(e["v82"][s:t + 1, 0]) > 0) if np.isfinite(e["v82"][s:t + 1, 0]).any() else None))
    esc = [r for r in run_rows if r["ends_by_motion"]]
    persist_fail = [r for r in run_rows if r["ends_at_end"] and not r["success"]]
    persist_succ = [r for r in run_rows if r["ends_at_end"] and r["success"]]
    print("stagnation runs (>=%d chunks): %d; ended by motion %d (in %d episodes), persisted to failed end %d, persisted to successful end %d" % (
        MIN_RUN, len(run_rows), len(esc), len({r["episode"] for r in esc}), len(persist_fail), len(persist_succ)))
    t2 = {}
    for key in ("ld_last", "ld_trend", "ld_max", "ld_first"):
        a = auroc([r[key] for r in esc], [r[key] for r in persist_fail])
        t2[key] = dict(median_escape=float(np.median([r[key] for r in esc])) if esc else np.nan,
                       median_persist_fail=float(np.median([r[key] for r in persist_fail])) if persist_fail else np.nan,
                       auroc_escape_vs_persist=a[0], p=a[1])
        print("T2 %-9s escape median %.4f vs persist-to-failure median %.4f: AUROC %.3f p=%.2g" % (
            key, t2[key]["median_escape"], t2[key]["median_persist_fail"], a[0], a[1]))
    # same, only inside failed episodes (escape runs from failed episodes vs their terminal runs)
    esc_f = [r for r in esc if not r["success"]]
    a = auroc([r["ld_last"] for r in esc_f], [r["ld_last"] for r in persist_fail])
    t2["ld_last_within_failed_episodes"] = dict(n_escape=len(esc_f), n_persist=len(persist_fail), auroc=a[0], p=a[1])
    print("T2 within failed episodes only: escape runs %d vs terminal runs %d, ld_last AUROC %.3f p=%.2g" % (len(esc_f), len(persist_fail), a[0], a[1]))
    # matched by run length (>=3) and long runs
    long_esc = [r for r in esc if r["length"] >= 3]; long_per = [r for r in persist_fail if r["length"] >= 3]
    a = auroc([r["ld_last"] for r in long_esc], [r["ld_last"] for r in long_per])
    t2["ld_last_runs_ge3"] = dict(n_escape=len(long_esc), n_persist=len(long_per), auroc=a[0], p=a[1])
    print("T2 runs >=3 chunks: escape %d vs persist %d, ld_last AUROC %.3f p=%.2g" % (len(long_esc), len(long_per), a[0], a[1]))
    res["T2_runs"] = t2
    res["T2_counts"] = dict(runs=len(run_rows), escape=len(esc), persist_fail=len(persist_fail), persist_succ=len(persist_succ),
                            escape_run_lengths=[r["length"] for r in esc], persist_fail_lengths=[r["length"] for r in persist_fail])

    # ---- T3: among stagnant chunks, does bending now predict motion in the next chunk?
    st = [r for r in chunk_rows if r["stag"] and r["stag_next"] is not None]
    for r in st:
        r["next_moving"] = not r["stag_next"]
    t3 = {}
    for key in ("ld", "angle", "perp", "one_step"):
        point, ci = episode_bootstrap_auroc(st, key, "next_moving")
        t3[key] = dict(auroc=point, ci95=ci, n_next_moving=sum(r["next_moving"] for r in st), n_stay=sum(not r["next_moving"] for r in st))
        print("T3 stagnant chunk %-8s -> next chunk moves: AUROC %.3f [%.3f, %.3f] (n moves %d, stays %d)" % (
            key, point, ci[0], ci[1], t3[key]["n_next_moving"], t3[key]["n_stay"]))
    # within-episode normalized bending (subtract the episode's median over moving chunks)
    med_mov = {}
    for e in eps:
        moving = (e["disp"] >= 2 * TAU) | (e["dgrip"] >= 2 * GTAU)
        med_mov[e["episode"]] = float(np.median(e["ld"][moving])) if moving.any() else np.nan
    for r in st:
        r["ld_rel"] = r["ld"] - med_mov[r["episode"]]
    st2 = [r for r in st if np.isfinite(r["ld_rel"])]
    point, ci = episode_bootstrap_auroc(st2, "ld_rel", "next_moving")
    t3["ld_relative_to_episode_moving_median"] = dict(auroc=point, ci95=ci, n=len(st2))
    print("T3 within-episode-normalized L/D -> next chunk moves: AUROC %.3f [%.3f, %.3f]" % (point, ci[0], ci[1]))
    # continuous: Spearman between bending now and displacement in the next chunk, among stagnant chunks and among all chunks
    for name, rows in (("stagnant", st), ("all", [r for r in chunk_rows if np.isfinite(r["disp_next"])])):
        rho = spearmanr([r["ld"] for r in rows], [r["disp_next"] for r in rows])
        t3["spearman_ld_vs_next_disp_" + name] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(rows))
        print("T3 Spearman(L/D now, eef displacement next chunk) over %s chunks: rho %.3f p=%.2g n=%d" % (name, rho.statistic, rho.pvalue, len(rows)))
    res["T3_next_chunk"] = t3

    # ---- T4: relation to v8.2 (routing-based) trap signals
    fr = [r for r in chunk_rows if np.isfinite(r["freeze"])]
    rho = spearmanr([r["ld"] for r in fr], [r["freeze"] for r in fr])
    t4 = dict(spearman_ld_vs_freeze=dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(fr)))
    print("T4 Spearman(L/D, v8.2 freeze score) rho %.3f p=%.2g n=%d" % (rho.statistic, rho.pvalue, len(fr)))
    post = []
    for e in eps:
        al = np.flatnonzero(e["v82"][:, 0] > 0)
        if al.size == 0:
            continue
        a0 = int(al[0])
        post.append(dict(episode=e["episode"], success=e["success"], first_alarm=a0,
                         ld_post=float(np.mean(e["ld"][a0:a0 + 5])), ld_pre=float(np.mean(e["ld"][max(0, a0 - 5):a0])) if a0 > 0 else np.nan))
    ps = [r["ld_post"] for r in post if r["success"]]; pf = [r["ld_post"] for r in post if not r["success"]]
    a = auroc(ps, pf)
    t4["after_first_alarm"] = dict(n_alarmed_success=len(ps), n_alarmed_failure=len(pf), median_ld_post_success=float(np.median(ps)) if ps else np.nan,
                                   median_ld_post_failure=float(np.median(pf)) if pf else np.nan, auroc_success_vs_failure=a[0], p=a[1])
    print("T4 alarmed episodes: success %d (post-alarm L/D median %.4f) vs failure %d (%.4f): AUROC(success higher) %.3f p=%.2g" % (
        len(ps), np.median(ps) if ps else np.nan, len(pf), np.median(pf) if pf else np.nan, a[0], a[1]))
    res["T4_v82"] = t4

    # ---- T5: bending vs change of the commanded action between consecutive queries
    da = [r for r in chunk_rows if np.isfinite(r["dact"])]
    rho = spearmanr([r["ld"] for r in da], [r["dact"] for r in da])
    res["T5_bending_vs_action_change"] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(da))
    print("T5 Spearman(L/D, ||a_q - a_{q-1}||) rho %.3f p=%.2g n=%d" % (rho.statistic, rho.pvalue, len(da)))
    # time confound: bending vs query index
    rho = spearmanr([r["ld"] for r in chunk_rows], [r["q"] for r in chunk_rows])
    res["confound_ld_vs_query_index"] = dict(rho=float(rho.statistic), p=float(rho.pvalue))
    print("confound: Spearman(L/D, query index) rho %.3f p=%.2g" % (rho.statistic, rho.pvalue))

    # ---- T6: confound control (query index, within-episode scale) and stricter escape definitions
    t6 = {}
    def partial_spearman(x, y, z):
        rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
        def resid(u):
            A = np.c_[rz, np.ones_like(rz)]
            return u - A @ np.linalg.lstsq(A, u, rcond=None)[0]
        ex, ey = resid(rx), resid(ry)
        return float(np.corrcoef(ex, ey)[0, 1])
    rows_next = [r for r in chunk_rows if np.isfinite(r["disp_next"])]
    t6["partial_spearman_ld_vs_next_disp_given_q_all"] = partial_spearman([r["ld"] for r in rows_next], [r["disp_next"] for r in rows_next], [r["q"] for r in rows_next])
    t6["partial_spearman_ld_vs_next_disp_given_q_stagnant"] = partial_spearman([r["ld"] for r in st], [r["disp_next"] for r in st], [r["q"] for r in st])
    print("T6 partial Spearman(L/D, next displacement | q): all %.3f, stagnant %.3f" % (
        t6["partial_spearman_ld_vs_next_disp_given_q_all"], t6["partial_spearman_ld_vs_next_disp_given_q_stagnant"]))
    # T3 stratified by query-index bin (0-14, 15-29, 30+)
    strat = {}
    for b, (lo, hi) in enumerate(((0, 15), (15, 30), (30, 999))):
        sub = [r for r in st if lo <= r["q"] < hi]
        if sum(r["next_moving"] for r in sub) >= 3 and sum(not r["next_moving"] for r in sub) >= 3:
            point, ci = episode_bootstrap_auroc(sub, "ld", "next_moving", n=1000)
            strat["q_%d_%d" % (lo, hi)] = dict(auroc=point, ci95=ci, n_moves=sum(r["next_moving"] for r in sub), n_stay=sum(not r["next_moving"] for r in sub))
            print("T6 T3 stratified q in [%d,%d): AUROC %.3f [%.3f, %.3f] (moves %d, stays %d)" % (lo, hi, point, ci[0], ci[1], strat["q_%d_%d" % (lo, hi)]["n_moves"], strat["q_%d_%d" % (lo, hi)]["n_stay"]))
    t6["T3_by_query_bin"] = strat
    # T1 stratified by query bin
    strat1 = {}
    for b, (lo, hi) in enumerate(((0, 15), (15, 30), (30, 999))):
        p = [r["ld"] for r in chunk_rows if r["stag"] and lo <= r["q"] < hi]
        n = [r["ld"] for r in chunk_rows if r["moving"] and lo <= r["q"] < hi]
        if len(p) >= 3 and len(n) >= 3:
            strat1["q_%d_%d" % (lo, hi)] = dict(auroc=auroc(p, n)[0], n_stag=len(p), n_moving=len(n))
    t6["T1_by_query_bin"] = strat1
    print("T6 T1 stratified:", {k: round(v["auroc"], 3) for k, v in strat1.items()})
    # T2 with episode bootstrap and stricter escape labels; persisting-to-failure runs are the negatives
    t2b = {}
    for label in ("ends_by_motion", "escape_then_success", "escape_sustained"):
        rows = [r for r in run_rows if r[label] or (r["ends_at_end"] and not r["success"])]
        for key in ("ld_last", "zld_last"):
            if sum(r[label] for r in rows) >= 3:
                point, ci = episode_bootstrap_auroc(rows, key, label, n=1000)
                t2b[label + "__" + key] = dict(auroc=point, ci95=ci, n_pos=sum(r[label] for r in rows), n_neg=sum(not r[label] for r in rows))
                print("T6 T2 %-20s %-8s AUROC %.3f [%.3f, %.3f] (pos %d, neg %d)" % (label, key, point, ci[0], ci[1], t2b[label + "__" + key]["n_pos"], t2b[label + "__" + key]["n_neg"]))
    # T2 matched on end-time bin
    m = {}
    for b in range(3):
        p = [r["ld_last"] for r in run_rows if r["ends_by_motion"] and r["q_end_bin"] == b]
        n = [r["ld_last"] for r in run_rows if r["ends_at_end"] and not r["success"] and r["q_end_bin"] == b]
        if len(p) >= 3 and len(n) >= 3:
            m["end_bin_%d" % b] = dict(auroc=auroc(p, n)[0], n_escape=len(p), n_persist=len(n))
    t2b["ld_last_matched_end_bin"] = m
    print("T6 T2 matched by end-time bin:", {k: (round(v["auroc"], 3), v["n_escape"], v["n_persist"]) for k, v in m.items()})
    # bending vs action change within stagnant chunks only
    da2 = [r for r in st if np.isfinite(r["dact"])]
    rho = spearmanr([r["ld"] for r in da2], [r["dact"] for r in da2])
    t6["spearman_ld_vs_action_change_stagnant_only"] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(da2))
    print("T6 Spearman(L/D, action change) within stagnant chunks: rho %.3f p=%.2g n=%d" % (rho.statistic, rho.pvalue, len(da2)))
    res["T6_confound_control"] = t6
    res["T2_bootstrap"] = t2b

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
    (OUT / "trap-analysis.json").write_text(json.dumps(clean(res), indent=2))
    np.save(OUT / "trap-run-rows.npy", np.array(run_rows, dtype=object), allow_pickle=True)
    np.save(OUT / "trap-chunk-rows.npy", np.array(chunk_rows, dtype=object), allow_pickle=True)
    print("saved", OUT / "trap-analysis.json")


if __name__ == "__main__":
    main()
