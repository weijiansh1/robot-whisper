"""Label-free 时间滤波择优（在冻结实现下重跑；结果写入 SPEC §4.1）。

判据（先于看检出率声明，**只用参照成功集**）：
  在 E2/E3 声明的 lead 剖面形状下，两个候选滤波的"信号幅度"是可算的：
    loop   剖面 = 0(lead<=-5) -> 1(lead -4..-1)  => DC(W=4) 幅度 1 ; CS(4,6) 幅度 1-0 = 1
    static 剖面 = 全程 1（E3：onset 前已建立、贯穿始终） => DC(W=5) 幅度 1 ; CS(5,6) 幅度 1-1 = 0
  幅度为 0 的候选直接淘汰；幅度相同的候选取"LOGO 阈值 theta（=参照成功集每集 max 的
  1-alpha/2 分位）"更小者 —— 等信号、低阈 ⇒ 严格更灵敏。
"""
import numpy as np, detector as DET
ROOT = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
print("declared amplitude:  loop DC=1.0 CS=1.0 | static DC=1.0 CS=0.0")
print("=> static 只有 DC 可用；loop 两者等幅，按阈值择低。\n")
for corp in ["main16x32", "grid50x8"]:
    p = f"{ROOT}/features/{corp}/{TASK}/rows.npz"
    for fl in ["DC", "CS"]:
        R = DET.detect(p, "scene", config={"filt_loop": fl})
        th = float(np.median([r["thr_loop"] for r in R.values()]))
        v = np.array([r["maxGL"] for r in R.values() if np.isfinite(r["maxGL"])])
        print(f"  {corp:10s} loop filt={fl}: LOGO theta(median over groups)={th:7.4f}"
              f"   all-episode maxGL med={np.median(v):6.3f}")
    for fs in ["DC", "CS"]:
        R = DET.detect(p, "scene", config={"filt_static": fs})
        th = float(np.median([r["thr_static"] for r in R.values()]))
        print(f"  {corp:10s} stat filt={fs}: LOGO theta(median over groups)={th:7.4f}")
    print()
