"""校准集评估（SCENE8 两单位，各自独立报）。产出 summary.json + 控制台表格。

只读：features/{main16x32,grid50x8}/libero_long/KITCHEN_SCENE8_.../rows.npz
      events/同路径/events.csv（**只在评估侧使用**，检测器从不读它）
写入：仅本目录。
"""
import numpy as np, csv, json
import detector as DET

ROOT = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
UNITS = ["main16x32", "grid50x8"]
ALPHAS = [0.02, 0.05, 0.10, 0.20, 0.35]
ARMS = ["main", "P", "C", "T", "C_sticky", "main_nomob", "main_median"]
Q_PREFIX = [32, 36, 38]        # 前缀口径：只认 step<=Q0 的报警（剥掉"钟"）
Q_GRID = [30, 32, 34, 36, 38, 40]
FROZEN = DET.CONFIG["alpha"]


def load_events(corp):
    e = {}
    with open(f"{ROOT}/events/{corp}/{TASK}/events.csv") as f:
        for r in csv.DictReader(f):
            e[int(r["episode_id"])] = dict(loop=int(r["loop_onset_q"]), static=int(r["static_onset_q"]),
                                           succ=int(r["success"]), nq=int(r["n_queries"]),
                                           scene=int(r["scene"]), grade=r["proxy_grade"])
    return e


def populations(ev):
    P = {}
    P["clean_success"] = [e for e, m in ev.items() if m["succ"] == 1 and m["loop"] < 0 and m["static"] < 0]
    P["recovery_loop"] = [e for e, m in ev.items() if m["succ"] == 1 and m["loop"] >= 0]
    P["success_static"] = [e for e, m in ev.items() if m["succ"] == 1 and m["static"] >= 0 and m["loop"] < 0]
    P["fail_loop_only"] = [e for e, m in ev.items() if m["succ"] == 0 and m["loop"] >= 0 and m["static"] < 0]
    P["fail_static_only"] = [e for e, m in ev.items() if m["succ"] == 0 and m["static"] >= 0 and m["loop"] < 0]
    P["fail_both"] = [e for e, m in ev.items() if m["succ"] == 0 and m["loop"] >= 0 and m["static"] >= 0]
    P["fail_no_event"] = [e for e, m in ev.items() if m["succ"] == 0 and m["loop"] < 0 and m["static"] < 0]
    P["fail_loop_any"] = [e for e, m in ev.items() if m["succ"] == 0 and m["loop"] >= 0]
    P["fail_static_any"] = [e for e, m in ev.items() if m["succ"] == 0 and m["static"] >= 0]
    P["fail_all"] = [e for e, m in ev.items() if m["succ"] == 0]
    return P


def rate(R, ids, key="alarm"):
    if not ids:
        return None
    k = sum(1 for e in ids if (R[e]["alarm"] if key == "alarm" else R[e][key + "_q"] is not None))
    return dict(k=int(k), n=len(ids), rate=round(k / len(ids), 4))


def lead_stats(R, ev, ids, onset_key, ch):
    L = [(R[e]["alarm_q"] if ch is None else R[e][ch + "_q"]) for e in ids]
    L = [t - ev[e][onset_key] for e, t in zip(ids, L) if t is not None]
    if not L:
        return None
    L = np.array(L, float)
    return dict(n=int(L.size), q25=float(np.percentile(L, 25)), med=float(np.median(L)),
                q75=float(np.percentile(L, 75)), min=float(L.min()), max=float(L.max()),
                frac_before_onset=round(float(np.mean(L < 0)), 4))


def paired_auc(R, ev, pos, neg, key):
    bg = {}
    for e in pos:
        bg.setdefault(ev[e]["scene"], [[], []])[0].append(e)
    for e in neg:
        bg.setdefault(ev[e]["scene"], [[], []])[1].append(e)
    num = den = 0.0; ng = 0
    for g, (Pp, Nn) in bg.items():
        if not Pp or not Nn:
            continue
        ng += 1
        for a in Pp:
            for b in Nn:
                va = R[a][key] if np.isfinite(R[a][key]) else -1e18
                vb = R[b][key] if np.isfinite(R[b][key]) else -1e18
                num += 1.0 if va > vb else (0.5 if va == vb else 0.0); den += 1
    return None if den == 0 else dict(auc=round(num / den, 4), pairs=int(den), groups=ng)


def mcnemar(Ra, Rb, ids):
    """配对差：a 报而 b 不报 / b 报而 a 不报（同一次运行、同一 alpha、同一集）。"""
    n10 = sum(1 for e in ids if Ra[e]["alarm"] and not Rb[e]["alarm"])
    n01 = sum(1 for e in ids if Rb[e]["alarm"] and not Ra[e]["alarm"])
    n11 = sum(1 for e in ids if Ra[e]["alarm"] and Rb[e]["alarm"])
    n = len(ids)
    d = (n10 - n01) / n if n else None
    # 精确二项（对称零假设）双侧 p
    p = None
    if n10 + n01:
        from math import comb
        k, m = min(n10, n01), n10 + n01
        p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)
    return dict(n=n, a_only=n10, b_only=n01, both=n11, delta=None if d is None else round(d, 4),
                p_exact=None if p is None else round(p, 5))


def alarm_by(rec, Q):
    return any(a["step"] <= Q for a in rec["alarms"])


def prefix_rate(R, ids, Q):
    if not ids:
        return None
    k = sum(1 for e in ids if alarm_by(R[e], Q))
    return dict(k=int(k), n=len(ids), rate=round(k / len(ids), 4))


def mcnemar_prefix(Ra, Rb, ids, Q):
    n10 = sum(1 for e in ids if alarm_by(Ra[e], Q) and not alarm_by(Rb[e], Q))
    n01 = sum(1 for e in ids if alarm_by(Rb[e], Q) and not alarm_by(Ra[e], Q))
    n = len(ids)
    p = None
    if n10 + n01:
        from math import comb
        k, m = min(n10, n01), n10 + n01
        p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)
    return dict(n=n, a_only=n10, b_only=n01, delta=None if not n else round((n10 - n01) / n, 4),
                p_exact=None if p is None else round(p, 5))


def horizon_margin(R, ev, P, q_grid, min_clean=20, min_fail=10):
    """匹配视界 margin(q) = P(已报警 | 失败-有事件, 存活到 q) - P(已报警 | 干净成功, 存活到 q)。
    任何**纯 q 的报警器**在该口径下构造性地 margin == 0。"""
    fail = P["fail_loop_any"] + [e for e in P["fail_static_any"] if e not in P["fail_loop_any"]]
    out = {}
    for q in q_grid:
        cl = [e for e in P["clean_success"] if ev[e]["nq"] > q]
        fl = [e for e in fail if ev[e]["nq"] > q]
        if len(cl) < min_clean or len(fl) < min_fail:
            out[str(q)] = None
            continue
        rf = sum(1 for e in fl if alarm_by(R[e], q)) / len(fl)
        rc = sum(1 for e in cl if alarm_by(R[e], q)) / len(cl)
        out[str(q)] = dict(q=q, n_clean_alive=len(cl), n_fail_alive=len(fl),
                           rate_fail=round(rf, 4), rate_clean=round(rc, 4), margin=round(rf - rc, 4))
    vals = [v["margin"] for v in out.values() if v]
    out["max_margin"] = max(vals) if vals else None
    return out


def type_stats(R, ids, want):
    det = [e for e in ids if R[e]["alarm"]]
    if not det:
        return None
    c = sum(1 for e in det if R[e]["alarm_type"] == want)
    return dict(n_detected=len(det), correct=int(c), acc=round(c / len(det), 4))


def metrics(R, ev, P):
    M = {}
    M["FA_clean_success"] = rate(R, P["clean_success"])
    M["FA_clean_success_loopch"] = rate(R, P["clean_success"], "loop")
    M["FA_clean_success_staticch"] = rate(R, P["clean_success"], "static")
    for pop in ["fail_loop_only", "fail_static_only", "fail_both", "fail_no_event",
                "fail_loop_any", "fail_static_any", "fail_all", "recovery_loop", "success_static"]:
        M["DET_" + pop] = rate(R, P[pop])
        M["DET_" + pop + "_loopch"] = rate(R, P[pop], "loop")
        M["DET_" + pop + "_staticch"] = rate(R, P[pop], "static")
    M["TYPE_fail_loop_only"] = type_stats(R, P["fail_loop_only"], "loop")
    M["TYPE_fail_static_only"] = type_stats(R, P["fail_static_only"], "static")
    tl, ts = M["TYPE_fail_loop_only"], M["TYPE_fail_static_only"]
    if tl and ts:
        M["TYPE_macro"] = round((tl["acc"] + ts["acc"]) / 2, 4)
        M["TYPE_micro"] = round((tl["correct"] + ts["correct"]) / (tl["n_detected"] + ts["n_detected"]), 4)
        M["TYPE_micro_const_static"] = round(ts["n_detected"] / (tl["n_detected"] + ts["n_detected"]), 4)
    else:
        M["TYPE_macro"] = M["TYPE_micro"] = M["TYPE_micro_const_static"] = None
    M["TYPE_confusion"] = {pop: {"loop": sum(1 for e in P[pop] if R[e]["alarm_type"] == "loop"),
                                 "static": sum(1 for e in P[pop] if R[e]["alarm_type"] == "static")}
                           for pop in ["fail_loop_only", "fail_static_only", "fail_both",
                                       "fail_no_event", "clean_success"]}
    # 类型正确性的第二口径：正确通道是否**曾**报警（不看谁先）
    M["TYPE_ch_hit_loop_only"] = rate(R, P["fail_loop_only"], "loop")
    M["TYPE_ch_hit_static_only"] = rate(R, P["fail_static_only"], "static")
    M["LEAD_loop_only_firstalarm"] = lead_stats(R, ev, P["fail_loop_only"], "loop", None)
    M["LEAD_loop_only_loopch"] = lead_stats(R, ev, P["fail_loop_only"], "loop", "loop")
    M["LEAD_static_only_firstalarm"] = lead_stats(R, ev, P["fail_static_only"], "static", None)
    M["LEAD_static_only_staticch"] = lead_stats(R, ev, P["fail_static_only"], "static", "static")
    M["LEAD_loop_any_firstalarm"] = lead_stats(R, ev, P["fail_loop_any"], "loop", None)
    M["LEAD_static_any_firstalarm"] = lead_stats(R, ev, P["fail_static_any"], "static", None)
    M["PREFIX"] = {str(Q): {pop: prefix_rate(R, P[pop], Q)
                            for pop in ["clean_success", "fail_loop_only", "fail_static_only",
                                        "fail_no_event", "fail_both"]}
                   for Q in Q_PREFIX}
    for pop in ["fail_loop_any", "recovery_loop", "fail_static_any", "clean_success", "fail_no_event"]:
        v = [R[e]["persist"] for e in P[pop] if R[e]["alarm"]]
        dec = [x for x in v if x is not None]
        M["PERSIST_" + pop] = dict(n_alarm=len(v), n_decidable=len(dec),
                                   persist=int(sum(dec)) if dec else 0,
                                   frac=round(sum(dec) / len(dec), 4) if dec else None)
    return M


def eval_unit(corp, cfg_over=None, arms=ARMS, alphas=ALPHAS):
    p = f"{ROOT}/features/{corp}/{TASK}/rows.npz"
    ev = load_events(corp)
    P = populations(ev)
    out = dict(corpus=corp, populations={k: len(v) for k, v in P.items()})
    store = {}
    for arm in arms:
        cfg = dict(cfg_over or {})
        a_arm = "main" if arm in ("main_nomob", "main_median") else arm
        if arm == "main_nomob":
            cfg["drop_period"] = True
        if arm == "main_median":
            cfg["agg"] = "median"
        S = DET.score(p, "scene", config=cfg, arm=a_arm)
        res = {}
        for al in alphas:
            R = DET.alarms_from_score(S, alpha=al)
            store[(arm, al)] = R
            M = metrics(R, ev, P)
            M["alpha"] = al
            M["q_hi"] = {str(g): S["per_group"][g]["q_hi"] for g in S["groups"]}
            M["HORIZON_MARGIN"] = horizon_margin(R, ev, P, Q_GRID)
            if abs(al - FROZEN) < 1e-9:
                M["AUC_type_loop_vs_static"] = None
                for e in R:
                    a_, b_ = R[e]["maxGL"], R[e]["maxGS"]
                    R[e]["dtype"] = ((a_ if np.isfinite(a_) else -20.0) - (b_ if np.isfinite(b_) else -20.0))
                M["AUC_type_loop_vs_static"] = paired_auc(R, ev, P["fail_loop_only"], P["fail_static_only"], "dtype")
                M["AUC_maxGS_staticev_vs_clean"] = paired_auc(R, ev, P["fail_static_any"], P["clean_success"], "maxGS")
                M["AUC_maxGL_looponly_vs_clean"] = paired_auc(R, ev, P["fail_loop_only"], P["clean_success"], "maxGL")
                M["thr_loop_median"] = float(np.median([R[e]["thr_loop"] for e in R]))
                M["thr_static_median"] = float(np.median([R[e]["thr_static"] for e in R]))
                M["_per_episode_special"] = {str(e): dict(alarm_q=R[e]["alarm_q"], type=R[e]["alarm_type"],
                                                          persist=R[e]["persist"],
                                                          first=R[e]["first_by_channel"])
                                             for e in P["recovery_loop"] + P["success_static"]}
            res[f"alpha={al}"] = M
        out[arm] = res
    # ---- 配对差（同一次运行、同一 alpha、同一集）：主臂 vs 消融臂 C ----
    pair = {}
    for who in ("main", "P"):
      if who not in arms: continue
      for al in alphas:
        Ra, Rb = store[(who, al)], store[("C", al)]
        d = {}
        for pop in ["clean_success", "fail_loop_only", "fail_static_only", "fail_both",
                    "fail_no_event", "fail_loop_any", "fail_static_any", "recovery_loop"]:
            d[pop] = mcnemar(Ra, Rb, P[pop])
        # 分型的配对差（只在两臂都报警的集上比）
        for pop, want in [("fail_loop_only", "loop"), ("fail_static_only", "static")]:
            both = [e for e in P[pop] if Ra[e]["alarm"] and Rb[e]["alarm"]]
            d["TYPE_" + pop] = dict(n_both=len(both),
                                    main_correct=sum(1 for e in both if Ra[e]["alarm_type"] == want),
                                    C_correct=sum(1 for e in both if Rb[e]["alarm_type"] == want))
        # 提前量的配对差（两臂都报警的集）
        for pop, ok in [("fail_static_only", "static"), ("fail_loop_only", "loop")]:
            both = [e for e in P[pop] if Ra[e]["alarm"] and Rb[e]["alarm"]]
            if both:
                dl = np.array([Ra[e]["alarm_q"] - Rb[e]["alarm_q"] for e in both], float)
                d["LEADDIFF_" + pop] = dict(n=len(both), med=float(np.median(dl)),
                                            q25=float(np.percentile(dl, 25)), q75=float(np.percentile(dl, 75)),
                                            frac_main_earlier=round(float(np.mean(dl < 0)), 4))
        pair[f"{who}|alpha={al}"] = d
    out["paired_vs_C"] = pair
    # 同一"实测 FA"下的对齐比较（FA 在干净成功集上实测；alpha->FA 映射是逐臂的）
    fa = {a: {al: metrics(store[(a, al)], ev, P)["FA_clean_success"]["rate"] for al in alphas}
          for a in arms}
    matched = {}
    for tgt in (0.05, 0.10, 0.20):
        entry = {}
        for a in arms:
            al = min(alphas, key=lambda x: abs(fa[a][x] - tgt))
            R = store[(a, al)]
            M = metrics(R, ev, P)
            entry[a] = dict(alpha=al, FA=M["FA_clean_success"]["rate"],
                            loop_only=M["DET_fail_loop_only"]["rate"],
                            static_only=M["DET_fail_static_only"]["rate"],
                            both=M["DET_fail_both"]["rate"],
                            no_event=M["DET_fail_no_event"]["rate"],
                            type_loop=(M.get("TYPE_fail_loop_only") or {}).get("acc"),
                            type_static=(M.get("TYPE_fail_static_only") or {}).get("acc"),
                            lead_loop=(M["LEAD_loop_only_firstalarm"] or {}).get("med"),
                            lead_static=(M["LEAD_static_only_firstalarm"] or {}).get("med"))
        for a in arms:
            for ref in ("C", "T"):
                if a == ref or ref not in arms:
                    continue
                entry[f"{a}_vs_{ref}_paired"] = {
                    pop: mcnemar(store[(a, entry[a]["alpha"])], store[(ref, entry[ref]["alpha"])], P[pop])
                    for pop in ["clean_success", "fail_loop_only", "fail_static_only", "fail_no_event"]}
                entry[f"{a}_vs_{ref}_paired_prefix36"] = {
                    pop: mcnemar_prefix(store[(a, entry[a]["alpha"])], store[(ref, entry[ref]["alpha"])],
                                        P[pop], 36)
                    for pop in ["clean_success", "fail_loop_only", "fail_static_only", "fail_no_event"]}
        matched[f"targetFA={tgt}"] = entry
    out["matched_FA"] = matched
    return out


if __name__ == "__main__":
    res = {u: eval_unit(u) for u in UNITS}
    KNOBS = [("W_static", [4, 5, 6]), ("W_loop_sig", [3, 4, 5]), ("B_ref", [1, 2, 3]),
             ("n_ref_eps", [15, 25, 40]), ("polarity_gate", [True, False]),
             ("filt_loop", ["DC", "CS"]), ("q_cen_hi", [25, 30, 35]), ("center", [True, False])]
    sens = {}
    for u in UNITS:
        sens[u] = {}
        for knob, vals in KNOBS:
            sens[u][knob] = {}
            for v in vals:
                r = eval_unit(u, cfg_over={knob: v}, arms=["main", "C"], alphas=[FROZEN])
                M = r["main"][f"alpha={FROZEN}"]
                MC = r["C"][f"alpha={FROZEN}"]
                pd_ = r["paired_vs_C"][f"main|alpha={FROZEN}"]
                sens[u][knob][str(v)] = dict(
                    FA=M["FA_clean_success"]["rate"], FA_C=MC["FA_clean_success"]["rate"],
                    DET_loop_only=M["DET_fail_loop_only"]["rate"], DET_loop_only_C=MC["DET_fail_loop_only"]["rate"],
                    DET_static_only=M["DET_fail_static_only"]["rate"], DET_static_only_C=MC["DET_fail_static_only"]["rate"],
                    TYPE_loop=(M.get("TYPE_fail_loop_only") or {}).get("acc"),
                    TYPE_loop_C=(MC.get("TYPE_fail_loop_only") or {}).get("acc"),
                    delta_static=pd_["fail_static_only"]["delta"], delta_loop=pd_["fail_loop_only"]["delta"])
    summary = dict(design="D_freeform / Polarity Matched Filter (PMF)",
                   frozen_config=DET.CONFIG, param_sweep=DET.PARAM_SWEEP,
                   templates=dict(T_LOOP=DET.T_LOOP, T_STATIC=DET.T_STATIC,
                                  arm_C_loop=DET.C_T_LOOP, arm_C_static=DET.C_T_STATIC),
                   units=res, knob_sensitivity=sens)
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=float)

    def f(d):
        return "n/a" if d is None else f"{d['k']}/{d['n']}={d['rate']:.3f}"
    for u in UNITS:
        r = res[u]
        print("=" * 118)
        print(u, "populations:", r["populations"])
        for arm in ARMS:
            print(f"  --- {arm} ---")
            for al in ALPHAS:
                M = r[arm][f"alpha={al}"]
                tl = (M.get("TYPE_fail_loop_only") or {}).get("acc")
                ts = (M.get("TYPE_fail_static_only") or {}).get("acc")
                ld = M["LEAD_loop_only_firstalarm"]; sd = M["LEAD_static_only_firstalarm"]
                print(f"   a={al:<5} FA={f(M['FA_clean_success'])} | loop_only={f(M['DET_fail_loop_only'])}"
                      f" (type {tl}, ch-hit {M['TYPE_ch_hit_loop_only']['rate']})"
                      f" | static_only={f(M['DET_fail_static_only'])} (type {ts})"
                      f" | both={f(M['DET_fail_both'])} | noev={f(M['DET_fail_no_event'])}"
                      f" | recov={f(M['DET_recovery_loop'])}"
                      f" | lead L/S {None if not ld else ld['med']}/{None if not sd else sd['med']}")
        for who in ("main", "P"):
            print(f"  --- paired {who} vs C (same run, same alpha, same episodes) ---")
            for al in ALPHAS:
                d = r["paired_vs_C"][f"{who}|alpha={al}"]
                print(f"   a={al:<5} " + "  ".join(
                    f"{k}: d={d[k]['delta']:+.3f}(A{d[k]['a_only']}/C{d[k]['b_only']},p={d[k]['p_exact']})"
                    for k in ["clean_success", "fail_loop_only", "fail_static_only"]))
        print("  --- prefix (only alarms with step<=Q) @ alpha frozen ---")
        for arm in ARMS:
            M = r[arm][f"alpha={FROZEN}"]
            for Q in Q_PREFIX:
                pr = M["PREFIX"][str(Q)]
                print(f"   {arm:12s} Q<={Q}: FA={f(pr['clean_success'])} loop={f(pr['fail_loop_only'])}"
                      f" static={f(pr['fail_static_only'])} noev={f(pr['fail_no_event'])}")
        print("  --- matched-horizon margin (same absolute q) @ alpha frozen ---")
        for arm in ARMS:
            hm = r[arm][f"alpha={FROZEN}"]["HORIZON_MARGIN"]
            print(f"   {arm:12s} " + "  ".join(
                f"q{q}:{(hm[str(q)]['margin'] if hm.get(str(q)) else None)}" for q in Q_GRID)
                + f"   max={hm['max_margin']}")
        print("  --- typing: macro vs micro @ alpha frozen ---")
        for arm in ARMS:
            M = r[arm][f"alpha={FROZEN}"]
            print(f"   {arm:12s} macro={M['TYPE_macro']} micro={M['TYPE_micro']}"
                  f" const-static-micro={M['TYPE_micro_const_static']}")
        print("  --- matched realised FA ---")
        for tgt, e in r["matched_FA"].items():
            print(f"   {tgt}")
            for a in ARMS:
                x = e[a]
                for ref in ("C", "T"):
                    k = f"{a}_vs_{ref}_paired"
                    if k in e:
                        d = e[k]; dp = e[k + "_prefix36"]
                        print(f"       [{a} vs {ref}] anytime loop d={d['fail_loop_only']['delta']:+.3f}"
                              f"(p={d['fail_loop_only']['p_exact']}) static d={d['fail_static_only']['delta']:+.3f}"
                              f"(p={d['fail_static_only']['p_exact']}) | prefix36 loop d={dp['fail_loop_only']['delta']:+.3f}"
                              f"(p={dp['fail_loop_only']['p_exact']}) static d={dp['fail_static_only']['delta']:+.3f}"
                              f"(p={dp['fail_static_only']['p_exact']}) FA d={dp['clean_success']['delta']:+.3f}")
                print(f"     {a:12s} a={x['alpha']:<5} FA={x['FA']:.3f} loop={x['loop_only']:.3f}"
                      f" static={x['static_only']:.3f} noev={x['no_event']:.3f}"
                      f" typeL={x['type_loop']} typeS={x['type_static']}"
                      f" leadL/S {x['lead_loop']}/{x['lead_static']}")
    print("\nwrote summary.json")
