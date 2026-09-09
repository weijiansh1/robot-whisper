"""Stage 2 - per-channel CUSUM screen, DEVELOPMENT ONLY.

The single-chunk floor is a poor proxy for how a channel behaves under
temporal integration (a 4-sigma single-chunk rule catches almost nothing at
v7's false-alarm budget), so channels are ranked by their own CUSUM instead.

External is never opened here.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank  # noqa: E402
import capfree_common as C  # noqa: E402
import detectors as D  # noqa: E402

COHORT = "development_main"
K_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
SIGNS = (+1, -1)
ALARM_COUNTS = np.unique(np.round(np.geomspace(25, 5000, 44)).astype(int))
LEAD = C.HEADLINE_LEAD
# v7_guard's own development false-alarm count at lead>=4, and the external
# anchor's count rescaled to the development negative pool.
V7_DEV_FP = 38
V7_EXT_FP_SCALED = 54

_G = {}


def _init(Z, valid, risk, length):
    _G["Z"], _G["valid"], _G["risk"], _G["length"] = Z, valid, risk, length
    _G["base_at"] = C.baseline_frontier(
        C.fixed_chunk_baseline(risk, length), LEAD)


def _one(args):
    ci, arm_i = args
    Z = _G["Z"][arm_i]
    valid, risk, length = _G["valid"], _G["risk"], _G["length"]
    base_at = _G["base_at"]
    rows = []
    for sign in SIGNS:
        z = (sign * Z[ci]).astype(np.float32)
        if not np.isfinite(z).all() or z.std() < 1e-9:
            continue
        for k in K_GRID:
            S = D.cusum_path(z, valid, k)
            cm = D.cummax_valid(S, valid)
            thrs = D.alarm_count_grid(cm, ALARM_COUNTS)
            for r in D.sweep(cm, thrs, risk, length, base_at, LEAD):
                rows.append((sign, k, r["thr"], r[f"tp_lead{LEAD}"],
                             r[f"fp_lead{LEAD}"], r["excess_tp"],
                             r["median_lead"]))
    if not rows:
        return ci, arm_i, None
    a = np.array(rows, float)
    sign, k, thr, tp, fp, ex, ml = a.T
    out = {"ci": ci, "arm_i": arm_i}
    win = (fp >= C.FP_LO) & (fp <= C.FP_HI)
    for tag, m in (("win", win), ("v7dev", fp <= V7_DEV_FP),
                   ("v7ext", fp <= V7_EXT_FP_SCALED), ("fp80", fp <= 80)):
        if not m.any():
            out[f"{tag}_tp"] = 0
            continue
        sub = np.where(m)[0]
        j = sub[np.argmax(ex[sub])] if tag == "win" else sub[np.argmax(tp[sub])]
        out[f"{tag}_tp"] = int(tp[j])
        out[f"{tag}_fp"] = int(fp[j])
        out[f"{tag}_excess"] = int(ex[j])
        out[f"{tag}_sign"] = int(sign[j])
        out[f"{tag}_k"] = float(k[j])
        out[f"{tag}_thr"] = float(thr[j])
        out[f"{tag}_lead"] = float(ml[j])
    return ci, arm_i, out


def main() -> None:
    t0 = time.time()
    f = C.frame(COHORT)
    valid, risk, length = f["valid"], f["risk"], f["length"]
    names, X = bank.build(COHORT)
    print(f"通道 {len(names)} 个，{time.time() - t0:.0f}s")
    stats = bank.fit_norm(X, valid)
    Z = [bank.apply_norm(X, valid, stats, a) for a in bank.ARMS]
    del X
    print(f"归一化完成 {time.time() - t0:.0f}s")

    jobs = [(ci, ai) for ai in range(len(bank.ARMS)) for ci in range(len(names))]
    with mp.Pool(48, initializer=_init,
                 initargs=(Z, valid, risk, length)) as pool:
        res = pool.map(_one, jobs, chunksize=4)
    rows = []
    for ci, ai, out in res:
        if out is None:
            continue
        out["channel"] = names[ci]
        out["arm"] = bank.ARMS[ai]
        rows.append(out)
    df = pd.DataFrame(rows).sort_values("win_excess", ascending=False)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS / "cusum_channel_screen_development.csv", index=False)
    print(f"完成 {time.time() - t0:.0f}s")

    cols = ["channel", "arm", "win_sign", "win_k", "win_tp", "win_fp",
            "win_excess", "win_lead"]
    print("\n=== 单通道 CUSUM，按 FP∈[80,320] 内超出基线的 TP 排序（提前量>=4）===")
    print(df.head(25)[cols].to_string(index=False))
    print("\n=== 单通道 CUSUM，按 FP<=54（v7 外部预算折算）下的 TP 排序 ===")
    d2 = df.sort_values("v7ext_tp", ascending=False)
    print(d2.head(20)[["channel", "arm", "v7ext_sign", "v7ext_k", "v7ext_tp",
                       "v7ext_fp", "v7ext_lead"]].to_string(index=False))
    np.save(C.RESULTS / "_bank_names.npy", np.array(names))
    (C.RESULTS / "screen_cusum_meta.json").write_text(json.dumps({
        "cohort": COHORT, "n_channels": len(names), "k_grid": list(K_GRID),
        "arms": list(bank.ARMS), "alarm_counts": ALARM_COUNTS.tolist(),
        "lead": LEAD, "seconds": round(time.time() - t0, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
