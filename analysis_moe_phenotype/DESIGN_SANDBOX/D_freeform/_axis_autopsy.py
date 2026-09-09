"""为什么 loop-only 失败集会先被 static 通道报出来？报警窗内的逐轴符号解剖（冻结实现）。"""
import numpy as np, csv, detector as DET
ROOT="/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
TASK="libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
AX=["inst","desync","flat","sharp","period"]
for corp in ["main16x32","grid50x8"]:
    p=f"{ROOT}/features/{corp}/{TASK}/rows.npz"
    S=DET.score(p,"scene"); R=DET.alarms_from_score(S)
    D=DET._load(p,"scene"); eps={r["episode_id"]:r for r in D["eps"]}
    ev={}
    for r in csv.DictReader(open(f"{ROOT}/events/{corp}/{TASK}/events.csv")):
        ev[int(r["episode_id"])]=(int(r["loop_onset_q"]),int(r["static_onset_q"]),int(r["success"]))
    # 用与该组一致的 LOGO 参照重算轴（逐组）
    per={}
    for gi,g in enumerate(S["groups"]):
        ref=np.zeros(D["n"],bool)
        for r in D["eps"]:
            if r["success"]==1 and r["group"]!=g: ref[r["a"]:r["b"]]=True
        tr=DET._score_pass  # not needed; recompute Z here
        Zt={}
        cfg=DET.CONFIG
        # 复用 _score_pass 的前半段：直接调用它拿不到 Z，故本地重算标准化
        from detector import ALL_SIGS
        Z={s:np.full(D["n"],np.nan) for s in ALL_SIGS}
        for qq in range(cfg["q_lo"],int(D["q"].max())+1):
            m=ref&(np.abs(D["q"]-qq)<=cfg["B_ref"])
            if m.sum()<cfg["n_ref_rows"]: continue
            t=D["q"]==qq
            for s in ALL_SIGS:
                v=D["X"][s][m]; v=v[np.isfinite(v)]
                if v.size<cfg["n_ref_rows"]: continue
                md=np.median(v); sd=1.4826*np.median(np.abs(v-md))
                if sd<=0: continue
                Z[s][t]=(D["X"][s][t]-md)/sd
        A=DET._axes(Z,"main")
        per[g]={k:A[k] for k in AX}
    pops={"clean-succ (FA)":lambda m: m[2]==1 and m[0]<0 and m[1]<0,
          "fail loop-only":lambda m: m[2]==0 and m[0]>=0 and m[1]<0,
          "fail static-only":lambda m: m[2]==0 and m[1]>=0 and m[0]<0}
    print("="*96); print(corp,"  STATIC 通道报警窗(5拍)内逐轴 signed 值（正=支持 static）")
    print(f"  {'population':<18s}{'n':>5s}  "+"".join(f"{k:>9s}" for k in AX)+"   #pos/5")
    for nm,f in pops.items():
        rows=[]
        for e,rec in R.items():
            if not f(ev[e]) or rec["static_q"] is None: continue
            g=rec["group"]; ee=eps[e]; t=rec["static_q"]
            sl=slice(ee["a"]+t-DET.CONFIG["W_static"]+1, ee["a"]+t+1)
            rows.append([np.nanmean(DET.T_STATIC[k]*per[g][k][sl]) for k in AX])
        if not rows: print(f"  {nm:<18s}{0:>5d}"); continue
        M=np.array(rows)
        print(f"  {nm:<18s}{len(rows):>5d}  "+"".join(f"{np.nanmean(M[:,j]):9.2f}" for j in range(5))
              +f"     {(M>0).sum(1).mean():.2f}")
