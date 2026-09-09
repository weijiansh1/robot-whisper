"""机械自检：P1（组内 success 置换不变性）与 P2（在线因果 / 截断不变性）。

P1  在组 g 内随机置换 success 标签后，组 g 内每个 episode 的报警序列必须逐位不变。
P2  把某集截断到其首次报警步 s，仍必须在 s 报警；截断到 s-1，必须完全不报。
"""
import numpy as np, tempfile, os, sys, json
import detector as DET

ROOT = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
RNG = np.random.default_rng(20260903)


def write_npz(path, d, keep=None, success=None):
    out = {}
    for k in d.files:
        a = d[k]
        if keep is not None and a.shape[0] == len(d["episode_id"]):
            a = a[keep]
        out[k] = a
    if success is not None:
        out["success"] = success if keep is None else success[keep]
    np.savez(path, **out)


def sig(rec):
    return [(a["step"], a["channel"], round(a["score"], 9)) for a in rec["alarms"]]


def run(corp, arm="main", n_p2=12):
    src = f"{ROOT}/features/{corp}/{TASK}/rows.npz"
    d = np.load(src, allow_pickle=True)
    base = DET.detect(src, "scene", arm=arm)
    tmp = tempfile.mkdtemp(dir=".")
    ok = True
    REPS = np.load(f"{ROOT}/features/{corp}/{TASK}/reps.npy") if arm == "P" else None

    def put_reps(path, keep=None):
        if REPS is None:
            return
        np.save(os.path.join(os.path.dirname(path), "reps.npy"),
                REPS if keep is None else REPS[keep])

    # ---------------- P1 ----------------
    ep = d["episode_id"].astype(np.int64)
    scn = d["scene"].astype(np.int64)
    suc = d["success"].astype(np.int64).copy()
    groups = sorted(set(scn.tolist()))
    tested = 0
    for g in groups[:4]:
        m = scn == g
        eids = sorted(set(ep[m].tolist()))
        lab = {e: suc[ep == e][0] for e in eids}
        vals = list(lab.values())
        for _ in range(20):
            RNG.shuffle(vals)
            if vals != list(lab.values()):
                break
        new = suc.copy()
        for e, v in zip(eids, vals):
            new[ep == e] = v
        p = os.path.join(tmp, "p1.npz")
        write_npz(p, d, success=new); put_reps(p)
        r2 = DET.detect(p, "scene", arm=arm)
        for e in eids:
            if sig(base[e]) != sig(r2[e]):
                print(f"  P1 FAIL {corp} arm={arm} group={g} ep={e}")
                print("    base:", sig(base[e])[:4], " perm:", sig(r2[e])[:4])
                ok = False
        tested += len(eids)
    print(f"  P1 {'PASS' if ok else 'FAIL'} ({corp}/{arm}): {tested} episodes over 4 groups, "
          f"success labels permuted within group")

    # ---------------- P2 ----------------
    cands = [e for e, r in base.items() if r["alarm"]][:n_p2]
    ok2 = True
    for e in cands:
        s = base[e]["alarm_q"]
        for cut, expect_alarm_at_s in ((s, True), (s - 1, False)):
            keep = ~((ep == e) & (np.arange(len(ep)) - np.flatnonzero(ep == e)[0] > cut))
            keep = np.ones(len(ep), bool)
            idx = np.flatnonzero(ep == e)
            keep[idx[cut + 1:]] = False
            p = os.path.join(tmp, "p2.npz")
            write_npz(p, d, keep=keep); put_reps(p, keep)
            r2 = DET.detect(p, "scene", arm=arm)
            got = r2[e]
            if expect_alarm_at_s:
                if not got["alarm"] or got["alarm_q"] != s:
                    print(f"  P2 FAIL {corp}/{arm} ep={e}: truncate to {cut} -> alarm_q={got['alarm_q']} (want {s})")
                    ok2 = False
            else:
                if got["alarm"]:
                    print(f"  P2 FAIL {corp}/{arm} ep={e}: truncate to {cut} -> alarm at {got['alarm_q']} (want none)")
                    ok2 = False
    print(f"  P2 {'PASS' if ok2 else 'FAIL'} ({corp}/{arm}): {len(cands)} alarming episodes truncated to s and s-1")
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return ok and ok2


if __name__ == "__main__":
    allok = True
    for corp in ["main16x32", "grid50x8"]:
        for arm in ["main", "C"]:
            print(f"== {corp} / arm {arm}")
            allok &= run(corp, arm)
    print("\nSELFTEST", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)
