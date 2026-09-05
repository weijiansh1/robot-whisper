"""重跑全部 PPT 配图。python3 make_figs.py"""
import sys,os,json,glob; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from _style import *
import numpy as np
from matplotlib.patches import Patch,Ellipse
HERE=os.path.dirname(os.path.abspath(__file__)); TMP="/home/jovyan/.claude/jobs/4a175cdd/tmp"
def save(fig,name,rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(os.path.join(HERE,name),bbox_inches="tight"); plt.close(fig)
    print("%-34s %4d KB"%(name,os.path.getsize(os.path.join(HERE,name))//1024))

# ---- 01 语料规模 ----
rows=[("long / SCENE8",512,216),("goal / 开抽屉放碗",512,42),("spatial / 炉上碗",512,37),
      ("spatial / ramekin",512,12),("goal / 开中层抽屉",512,0),
      ("libero30 · 30 任务",1500,55),("rolling-star 分叉",352,235)]
fig,ax=plt.subplots(figsize=(7.6,4.0)); y=np.arange(len(rows))[::-1]
for i,(nm,n,nf) in enumerate(rows):
    yy=y[i]
    ax.barh(yy,n-nf,color=C["sky"],height=.6,zorder=2)
    if nf: ax.barh(yy,nf,left=n-nf,color=C["orange"],height=.6,zorder=3)
    ax.text(n+35,yy,"%d"%n+("  /  失败 %d"%nf if nf else "  /  无失败"),
            color=C["ink"],va="center",fontsize=14,fontweight="bold")
    ax.text(-35,yy,nm,color=C["ink"],va="center",ha="right",fontsize=15,fontweight="bold")
ax.set_xlim(0,2100); ax.set_ylim(-.7,len(rows)-.3); ax.set_yticks([]); ax.set_xticks([])
clean(ax,spines=())
ax.legend(handles=[Patch(color=C["sky"],label="成功"),Patch(color=C["orange"],label="失败")],
          loc="lower right",frameon=False,fontsize=14,labelcolor=C["ink"],handlelength=1.2)
title(fig,"3 364 条 rollout,失败仅 597 条")
save(fig,"fig01_corpus_scale.png",[0,0,1,0.93])

# ---- 02 可用性表 ----
fig,ax=plt.subplots(figsize=(6.4,3.0)); ax.axis("off"); xs=[0.0,0.52,0.72,0.90]
for x,h in zip(xs,["任务组","失败数","长度","状态"]):
    ax.text(x,0.95,h,color=C["grey"],fontsize=13,fontweight="bold",transform=ax.transAxes)
ax.plot([0,1],[0.885,0.885],color=C["ink"],lw=1.6,transform=ax.transAxes)
for i,(nm,nf,L,ok) in enumerate([("long / SCENE8",216,40,1),("goal / 开抽屉放碗",42,19,1),
        ("spatial / 炉上碗",37,13,1),("spatial / ramekin",12,10,0),("goal / 开中层抽屉",0,13,0)]):
    yy=0.74-i*0.175
    if ok: ax.add_patch(plt.Rectangle((-0.015,yy-0.05),1.03,0.13,color=C["mist"],
                                      transform=ax.transAxes,zorder=0,lw=0))
    c=C["ink"] if ok else C["grey"]
    ax.text(xs[0],yy,nm,color=c,fontsize=15,fontweight="bold",transform=ax.transAxes,zorder=2)
    ax.text(xs[1],yy,"%d"%nf,color=C["rust"] if ok else c,fontsize=15,fontweight="bold",transform=ax.transAxes,zorder=2)
    ax.text(xs[2],yy,"%d"%L,color=c,fontsize=15,fontweight="bold",transform=ax.transAxes,zorder=2)
    ax.text(xs[3],yy,"可用" if ok else "不足",color=C["blue"] if ok else C["grey"],
            fontsize=14,fontweight="bold",transform=ax.transAxes,zorder=2)
ax.text(0,-0.08,"判据:失败 ≥ 30 条,且长度容得下基线窗口",color=C["grey"],
        fontsize=11,transform=ax.transAxes)
save(fig,"fig02_usable_groups.png")

# ---- 03 初态极化 ----
fig,ax=plt.subplots(figsize=(5.4,3.4)); sr=[0.26,0.00,0.06,0.91]
ax.axhspan(0.10,0.90,color=C["mist"],zorder=0)
ax.bar(range(4),sr,color=[C["blue"],C["orange"],C["orange"],C["orange"]],width=.55,zorder=2)
for i,s in enumerate(sr):
    ax.text(i,s+0.035,"%.0f%%"%(s*100),color=C["ink"],ha="center",fontsize=16,fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(["init 0","init 1","init 2","init 3"],fontsize=14)
ax.set_ylim(0,1.08); ax.set_yticks([0,.5,1.0]); ax.set_yticklabels(["0","50%","100%"])
clean(ax,grid="y"); title(fig,"4 个初态高度极化","蓝 = 可做组内比较")
save(fig,"fig03_init_polarization.png",[0,0,1,0.88])

# ---- 04 相图 ----
br=json.load(open("/home/jovyan/work/himoe-vla/demo/demo.json"))["branches"]
def ser(b):
    r=np.array([np.nan if v is None else v for v in b["mean"]],float); return r[~np.isnan(r)]
def sm(r,k=5): return None if len(r)<k+2 else np.convolve(r,np.ones(k)/k,mode="valid")
grp={"success":[],"loop":[],"stagnation":[],"other":[]}
for b in br:
    rs=sm(ser(b))
    if rs is not None: grp[b["kind"]].append((rs[-1],np.diff(rs)[-3:].mean()))
al=[]
for b in br:
    if b["alarm"] is None: continue
    r=np.array([np.nan if v is None else v for v in b["mean"]],float)
    h=int(np.argmax(~np.isnan(r))); rs=sm(r[~np.isnan(r)])
    if rs is None: continue
    j=b["alarm"]-h-2
    if 0<j<len(rs)-1: al.append((rs[j],rs[j+1]-rs[j]))
al=np.array(al); S=np.array(grp["success"]); L=np.array(grp["loop"]); G=np.array(grp["stagnation"])
fig,ax=plt.subplots(figsize=(6.8,4.4)); ax.axhline(0,color=C["line"],lw=1,zorder=1)
ax.scatter(al[:,0],al[:,1],s=10,color=C["sand"],alpha=.9,edgecolors="none",zorder=2,label="报警时刻")
ax.scatter(L[:,0],L[:,1],s=34,facecolors="none",edgecolors=C["orange"],lw=1.3,alpha=.8,zorder=3,label="打转")
ax.scatter(S[:,0],S[:,1],s=30,color=C["blue"],alpha=.9,edgecolors="none",zorder=4,label="成功")
ax.scatter(G[:,0],G[:,1],s=34,color=C["rust"],alpha=.95,edgecolors="none",zorder=5,label="停滞")
for cx,cy,rx,ry,col,txt,dy in [(np.median(S[:,0]),np.median(S[:,1]),0.21,0.078,C["blue"],"巡航",0.085),
                               (np.median(G[:,0]),np.median(G[:,1]),0.18,0.065,C["rust"],"不动点",-0.085)]:
    ax.add_patch(Ellipse((cx,cy),rx,ry,fill=False,ls=(0,(4,3)),lw=1.6,ec=col,zorder=6))
    ax.text(cx,cy+dy,txt,color=C["ink"],fontsize=15,fontweight="bold",ha="center",va="center",zorder=7)
ax.set_xlabel("r(t)  路由变化率",fontsize=15); ax.set_ylabel("Δr  每步变化",fontsize=15)
ax.set_xlim(0.28,1.38); ax.set_ylim(-0.15,0.15); clean(ax,grid="both")
ax.legend(frameon=False,fontsize=13,labelcolor=C["ink"],loc="upper left",handletextpad=.4,
          borderpad=.15,labelspacing=.3)
title(fig,"三种命运,三个位置")
save(fig,"fig04_phase_portrait.png",[0,0,1,0.93])

# ---- 05 / 06 尖锐度 ----
ROOT="/home/jovyan/work/himoe-vla/himoe-route-capture/corpus/libero30-right-v1"
R=json.load(open(os.path.join(HERE,"sharp_rep.json")))
SU={"goal":"libero_goal","object":"libero_object","spatial":"libero_spatial"}
def spear(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<8 or np.std(x)==0 or np.std(y)==0: return None
    return np.corrcoef(np.argsort(np.argsort(x)),np.argsort(np.argsort(y)))[0,1]
def pts_of(tag):
    su,t2=tag.split("/"); td=glob.glob(os.path.join(ROOT,SU[su],t2+"*"))[0]
    P=[];RH=[]
    for r in R[tag]:
        f=os.path.join(td,"client","episode_%02d.npz"%r["init"])
        if not os.path.isfile(f): continue
        st=np.asarray(np.load(f,allow_pickle=True)["state"],float)
        T=min(len(st)-1,len(r["top4"]))
        if T<8: continue
        d=np.linalg.norm(np.diff(st[:T+1,:3],axis=0),axis=1); sh=np.asarray(r["top4"][:T],float)
        c=spear(sh,d)
        if c is not None: RH.append(c)
        P.append((d-d.mean(),sh-sh.mean()))
    return P,RH
P,RH=pts_of("object/t05")
X=np.concatenate([p[0] for p in P]); Y=np.concatenate([p[1] for p in P])
fig,ax=plt.subplots(figsize=(6.4,4.2))
ax.scatter(X,Y,s=9,color=C["blue"],alpha=.30,edgecolors="none",zorder=2)
k,b0=np.polyfit(X,Y,1); xs=np.linspace(X.min(),X.max(),50)
ax.plot(xs,k*xs+b0,color=C["rust"],lw=3,zorder=4)
ax.axhline(0,color=C["line"],lw=1,zorder=1); ax.axvline(0,color=C["line"],lw=1,zorder=1)
ax.set_xlabel("末端位移(相对自身均值)",fontsize=15)
ax.set_ylabel("路由 top-4 质量\n↑ 分布更尖锐",fontsize=15)
ax.set_xlim(np.percentile(X,0.5),np.percentile(X,99.5)); ax.set_ylim(np.percentile(Y,0.5),np.percentile(Y,99.5))
clean(ax,grid="both")
ax.text(0.97,0.95,"ρ = %+.2f"%np.mean(RH),transform=ax.transAxes,ha="right",va="top",
        fontsize=26,fontweight="bold",color=C["rust"])
ax.text(0.97,0.79,"rollout 内部相关",transform=ax.transAxes,ha="right",va="top",
        fontsize=12,color=C["grey"])
title(fig,"「路由尖锐」= 「手臂动得慢」","object/t05 · 50 条 rollout · 各自中心化")
save(fig,"fig05_sharpness_vs_motion.png",[0,0,1,0.90])
tags=["goal/t03","object/t05","spatial/t04","spatial/t07"]
vals=[np.mean(pts_of(t)[1]) for t in tags]
fig,ax=plt.subplots(figsize=(5.0,2.9)); y=np.arange(len(tags))[::-1]
ax.barh(y,vals,color=[C["blue"] if v<-0.4 else C["sky"] for v in vals],height=.55,zorder=2)
for i,v in enumerate(vals):
    ax.text(v-0.025,y[i],"%.2f"%v,color=C["ink"],va="center",ha="right",fontsize=15,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(tags,fontsize=14)
ax.set_xlim(-0.78,0.03); ax.set_xticks([-0.6,-0.3,0])
clean(ax,grid="x",spines=("bottom",))
title(fig,"四个任务上一致","ρ(尖锐度, 末端位移)")
save(fig,"fig06_sharpness_rho_by_task.png",[0,0,1,0.86])

# ---- 07 各失败类型的 r(t) 轨迹 ----
KIND=[("stagnation","停滞",C["rust"],49),("loop","打转",C["orange"],150),
      ("other","其他",C["sand"],36),("success","成功",C["blue"],117)]
fig,ax=plt.subplots(figsize=(7.0,4.4))
for key,lab,col,n in KIND:
    sub=[b for b in br if b["kind"]==key]; m=[];lo=[];hi=[];ts=[]
    for t in range(8,50):
        v=[b["mean"][t] for b in sub if t<len(b["mean"]) and b["mean"][t] is not None]
        if len(v)<8: continue
        ts.append(t); m.append(np.mean(v)); lo.append(np.percentile(v,25)); hi.append(np.percentile(v,75))
    ax.fill_between(ts,lo,hi,color=col,alpha=.14,lw=0,zorder=2)
    ax.plot(ts,m,color=col,lw=3,zorder=3,label="%s(%d)"%(lab,n))
ax.axhline(0.95,color=C["grey"],ls=(0,(5,4)),lw=1.6,zorder=1)
ax.text(9.2,0.90,"θ = 0.95",color=C["ink"],fontsize=13,ha="left",fontweight="bold")
ax.set_xlabel("控制步",fontsize=15); ax.set_ylabel("r(t)  路由变化率",fontsize=15)
ax.set_xlim(8,49); ax.set_ylim(0.35,1.28); clean(ax,grid="y")
ax.legend(frameon=False,fontsize=13,labelcolor=C["ink"],loc="lower left",
          bbox_to_anchor=(0.0,0.06),labelspacing=.35)
save(fig,"fig07_rt_by_failure_type.png")

# ---- 08 失败类型构成 ----
fig,ax=plt.subplots(figsize=(6.2,1.9)); x0=0
for key,lab,col,n in KIND:
    ax.barh(0,n,left=x0,color=col,height=.55,zorder=2)
    ax.text(x0+n/2,0,"%d"%n,color="white" if key!="other" else C["ink"],
            ha="center",va="center",fontsize=15,fontweight="bold",zorder=3)
    ax.text(x0+n/2,-0.45,lab,color=C["ink"],ha="center",va="center",fontsize=14,fontweight="bold")
    x0+=n
ax.set_xlim(0,352); ax.set_ylim(-0.75,0.4); ax.axis("off")
save(fig,"fig08_failure_composition.png")

# ---- 09 各类型的报警触发率 ----
W0,W,K,TH,STOP=8,4,3,0.95,37
def fire(seq):
    s=np.array([np.nan if v is None else v for v in seq],float); s=s[~np.isnan(s)]
    if len(s)<W0+W: return None
    b=s[:W0].mean(); run=0
    for t in range(W0+W-1,min(len(s),STOP)):
        run=run+1 if s[t-W+1:t+1].mean()/b<TH else 0
        if run>=K: return True
    return False
rates=[(lab,np.mean([x for x in (fire(b["mean"]) for b in br if b["kind"]==key) if x is not None]),col)
       for key,lab,col,_ in KIND]
fig,ax=plt.subplots(figsize=(5.8,3.4)); xs=np.arange(4)
ax.bar(xs,[r[1] for r in rates],color=[r[2] for r in rates],width=.58,zorder=2)
for i,(lab,v,col) in enumerate(rates):
    ax.text(i,v+0.035,"%.0f%%"%(v*100),ha="center",fontsize=17,fontweight="bold",color=C["ink"])
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rates],fontsize=15)
ax.set_ylim(0,1.18); ax.set_yticks([0,.5,1.0]); ax.set_yticklabels(["0","50%","100%"])
ax.set_ylabel("报警触发率",fontsize=15); clean(ax,grid="y")
save(fig,"fig09_recall_by_type.png")

# ---- 10 聚合度 AUC:HUB 三个任务 ----
GA=[("spatial / 炉上碗",0.936,"37 / 347"),
    ("goal / 开抽屉放碗",0.758,"42 / 182"),
    ("long / SCENE8",0.653,"184 / 232")]
fig,ax=plt.subplots(figsize=(6.2,3.2)); y=np.arange(len(GA))[::-1]
ax.barh(y,[g[1] for g in GA],height=.5,color=C["blue"],zorder=2)
for i,g in enumerate(GA):
    ax.text(g[1]+0.012,y[i]+0.13,"%.3f"%g[1],va="center",fontsize=16,fontweight="bold",color=C["ink"])
    ax.text(g[1]+0.014,y[i]-0.18,"n = %s"%g[2],va="center",fontsize=11,color=C["grey"])
ax.axvline(0.5,color=C["grey"],ls=(0,(4,3)),lw=1.4,zorder=1)
ax.set_yticks(y); ax.set_yticklabels([g[0] for g in GA],fontsize=14)
ax.set_xlim(0.45,1.10); ax.set_xticks([0.5,0.6,0.7,0.8,0.9,1.0])
ax.set_xlabel("初态内配对 AUC",fontsize=15)
clean(ax,grid="x",spines=("bottom",))
save(fig,"fig10_sharpness_auc.png")

# ---- 11 聚合度 AUC:独立复现集 ----
REP=[("spatial / t04",0.944,"14 / 36",1),("object / t05",0.920,"7 / 43",1),
     ("spatial / t07",0.920,"5 / 45",1),("goal / t03",0.613,"5 / 45",0)]
fig,ax=plt.subplots(figsize=(6.0,3.2)); y=np.arange(len(REP))[::-1]
ax.barh(y,[r[1] for r in REP],height=.5,zorder=2,
        color=[C["blue"] if r[3] else C["sky"] for r in REP])
for i,r in enumerate(REP):
    ax.text(r[1]+0.012,y[i]+0.13,"%.3f"%r[1],va="center",fontsize=16,fontweight="bold",color=C["ink"])
    ax.text(r[1]+0.014,y[i]-0.18,"n = %s"%r[2],va="center",fontsize=11,color=C["grey"])
ax.axvline(0.5,color=C["grey"],ls=(0,(4,3)),lw=1.4,zorder=1)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in REP],fontsize=14)
ax.set_xlim(0.45,1.10); ax.set_xticks([0.5,0.7,0.9])
ax.set_xlabel("AUC(路由聚合度)",fontsize=15)
clean(ax,grid="x",spines=("bottom",))
save(fig,"fig11_sharpness_replication.png")

# ---- 12 各初态的失败构成(堆叠占比) ----
from collections import Counter
F=[b for b in br if not b["success"]]
fig,ax=plt.subplots(figsize=(6.8,3.4))
order=[("stagnation","停滞",C["rust"]),("loop","打转",C["orange"]),("other","其他",C["sand"])]
ws=[0,2,1,3]  # 按成功率排序展示:26%,6%,0%,91%
labels=["init 0\n71 条失败","init 2\n75 条","init 1\n80 条","init 3\n9 条"]
bottom=np.zeros(4)
for key,lab,col in order:
    vals=[]
    for w in ws:
        sub=[b for b in F if b["worker"]==w]
        vals.append(100*sum(1 for b in sub if b["kind"]==key)/max(len(sub),1))
    vals=np.array(vals)
    ax.bar(range(4),vals,bottom=bottom,color=col,width=.6,zorder=2,label=lab)
    for i,v in enumerate(vals):
        if v>=7: ax.text(i,bottom[i]+v/2,"%.0f%%"%v,ha="center",va="center",
                         fontsize=13,fontweight="bold",
                         color="white" if key!="other" else C["ink"])
    bottom+=vals
ax.set_xticks(range(4)); ax.set_xticklabels(labels,fontsize=12.5)
ax.set_ylim(0,100); ax.set_yticks([0,50,100]); ax.set_yticklabels(["0","50%","100%"])
ax.set_ylabel("失败类型占比",fontsize=15)
clean(ax,grid="y")
ax.legend(frameon=False,fontsize=13,labelcolor=C["ink"],ncol=3,loc="lower center",
          bbox_to_anchor=(0.5,1.01),columnspacing=1.8,handlelength=1.2)
save(fig,"fig12_type_by_init.png")

# ---- 13 三类的 r(t) 末段分布(箱线) ----
fig,ax=plt.subplots(figsize=(6.2,3.4))
data=[];cols=[];names=[]
for key,lab,col in [("stagnation","停滞",C["rust"]),("other","其他",C["sand"]),
                    ("loop","打转",C["orange"]),("success","成功",C["blue"])]:
    v=[]
    for b in br:
        if b["kind"]!=key: continue
        s=[x for x in b["mean"] if x is not None]
        if s: v.append(np.mean(s[-4:]))
    data.append(v); cols.append(col); names.append("%s\n(%d)"%(lab,len(v)))
bp=ax.boxplot(data,vert=True,patch_artist=True,widths=.55,showfliers=False,
              medianprops=dict(color=C["ink"],lw=2.2),
              whiskerprops=dict(color=C["grey"],lw=1.4),
              capprops=dict(color=C["grey"],lw=1.4))
for patch,col in zip(bp["boxes"],cols):
    patch.set_facecolor(col); patch.set_edgecolor(C["grey"]); patch.set_linewidth(1.1)
for i,v in enumerate(data):
    ax.scatter(np.random.default_rng(i).normal(i+1,0.055,len(v)),v,s=7,
               color=C["ink"],alpha=.20,zorder=3)
ax.axhline(0.95,color=C["grey"],ls=(0,(4,3)),lw=1.4,zorder=1)
ax.set_xticklabels(names,fontsize=13)
ax.set_ylabel("r(t) 末段值",fontsize=15); ax.set_ylim(0.25,1.35)
clean(ax,grid="y")
save(fig,"fig13_rt_end_distribution.png")

# ---- 14 单条曲线:打转 vs 成功 vs 停滞 ----
def seqf(b):
    s=np.array([np.nan if v is None else v for v in b["mean"]],float); return s
rng2=np.random.default_rng(11)
fig,axes=plt.subplots(1,3,figsize=(11.4,3.2),sharey=True)
for ax,(key,lab,col) in zip(axes,[("success","成功",C["blue"]),
                                  ("loop","打转",C["orange"]),
                                  ("stagnation","停滞",C["rust"])]):
    sub=[b for b in br if b["kind"]==key]
    pick=rng2.choice(len(sub),min(8,len(sub)),replace=False)
    for i in pick:
        s=seqf(sub[i]); ts=np.arange(len(s))
        m=~np.isnan(s)
        ax.plot(ts[m],s[m],color=col,lw=1.6,alpha=.55,zorder=2)
    ax.axhline(0.95,color=C["grey"],ls=(0,(4,3)),lw=1.3,zorder=1)
    ax.set_xlim(8,52); ax.set_ylim(0.25,1.62)
    ax.set_xlabel("控制步",fontsize=13)
    ax.text(0.5,1.03,lab,transform=ax.transAxes,fontsize=16,fontweight="bold",
            color=col,va="bottom",ha="center")
    clean(ax,grid="y")
axes[0].set_ylabel("r(t)",fontsize=15)
save(fig,"fig14_individual_curves.png")

# ---- 15 自相关:打转有没有周期 ----
def acf(rs,K=12):
    A=np.zeros(K); n=0
    for b in rs:
        s=seqf(b)[8:]; s=s[~np.isnan(s)]
        if len(s)<K+6: continue
        s=s-s.mean()
        A+=np.array([np.sum(s[k:]*s[:-k or None])/np.sum(s*s) for k in range(1,K+1)]); n+=1
    return A/n
fig,ax=plt.subplots(figsize=(6.2,3.4))
ks=np.arange(1,13)
for key,lab,col in [("stagnation","停滞",C["rust"]),("loop","打转",C["orange"]),
                    ("success","成功",C["blue"])]:
    ax.plot(ks,acf([b for b in br if b["kind"]==key]),color=col,lw=3,
            marker="o",ms=6,label=lab,zorder=3)
ax.axhline(0,color=C["grey"],lw=1.2,zorder=1)
ax.set_xlabel("滞后 k(控制步)",fontsize=15); ax.set_ylabel("r(t) 自相关",fontsize=15)
ax.set_xticks([1,3,5,7,9,11]); ax.set_ylim(-0.35,1.0)
clean(ax,grid="y")
ax.legend(frameon=False,fontsize=13,labelcolor=C["ink"],loc="upper right",labelspacing=.3)
save(fig,"fig15_autocorrelation.png")
