"""Frozen MoE-routing detector transferred to CALVIN D->D (2026-09-02 session).

Consolidates the inline analyses that had no file on disk:
  1. sequence boundaries from ceil(env_steps/10) (episode_id is all-zero in this run)
  2. frozen rule r(t)=mean(last W)/mean(first W0), K consecutive < theta; theta sweep
  3. online length sentinel ("subtask has run >= k calls") at matched runway
  4. subtask-level risk-set AUC (the only length-clean comparison)
  5. executed-motion baselines (joint / eef displacement), raw and self-normalised
  6. label-free theta: theta = 1 - z * CV * sqrt(1/W + 1/W0), CV from first W0 steps
  7. success-only quantile calibration (the project's one-class protocol)
Writes calvin_transfer_results.csv.  Run with OMP_NUM_THREADS=1.
"""
import json, math, numpy as np, pandas as pd, zarr
R = "/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/calvin_d/task_D_D/routes-v1"
W, W0, K = 4, 8, 3

def load():
    S = [json.loads(l) for l in open(f"{R}/client/sequences.jsonl")]
    P = zarr.open_group(f"{R}/server/routes.zarr", mode="r")["hb_router_probs"][:, 4:8, 9, 1:11, :].astype(np.float32)
    P = np.clip(P, 1e-12, None); P /= P.sum(-1, keepdims=True)
    idx = np.argpartition(P, -4, axis=-1)[..., -4:]; M = np.zeros(P.shape, bool); np.put_along_axis(M, idx, True, axis=-1)
    ST = np.load(f"{R}/server/state.npy")
    seqs, subs, off = [], [], 0
    for r in S:
        m = r["inference_calls"]; sl = slice(off, off + m); off += m
        mm = M[sl]; inter = (mm[:-1] & mm[1:]).sum(-1); d = (1 - inter / (8 - inter)).mean((1, 2))
        calls = [math.ceil(s["environment_steps"] / 10) for s in r["subtasks"]]
        assert sum(calls) == m, "boundary reconstruction failed"
        cum = np.cumsum([0] + calls); cs = r["completed_subtasks"]
        st = ST[sl]
        seqs.append(dict(idx=r["sequence_index"], succ=cs == 5, m=m, d=d,
                         jnt=np.linalg.norm(np.diff(st[:, 7:14], axis=0), axis=1) + 1e-6,
                         eef=np.linalg.norm(np.diff(st[:, 0:3], axis=0), axis=1) + 1e-6,
                         calls=calls, end=(cum[cs + 1] if cs < 5 else m)))
        c = 0
        for s, n in zip(r["subtasks"], calls):
            subs.append(dict(ok=s["success"], n=n, task=s["task"], seg=d[max(c - 1, 0):c + n - 1], base=d[:W0].mean())); c += n
    return seqs, subs

def fire(d, th, t0=W0, stop=10**9):
    base = d[:W0].mean(); c = 0
    for t in range(t0, min(len(d), stop)):
        c = c + 1 if d[t - W + 1:t + 1].mean() / base < th else 0
        if c >= K: return t
    return None

def auc(sc, pos):
    sc = np.asarray(sc, float); pos = np.asarray(pos, bool)
    if pos.sum() < 3 or (~pos).sum() < 3: return np.nan
    r = np.argsort(np.argsort(sc)) + 1
    a = (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()); return max(a, 1 - a)

def gate_table(seqs, sig, ths, nm):
    F = [x for x in seqs if not x["succ"]]; G = [x for x in seqs if x["succ"]]; rows = []
    for th in ths:
        tp = [x["end"] - f for x in F for f in [fire(x[sig], th)] if f is not None]
        fp = sum(fire(x[sig], th) is not None for x in G); na = len(tp) + fp
        rows.append(dict(section=nm, signal=sig, theta=th, alarms=na, hits=len(tp),
                         precision=len(tp) / na if na else np.nan, detection=len(tp) / len(F),
                         false_alarm=fp / len(G), runway_median=np.median(tp) if tp else np.nan))
    return rows

if __name__ == "__main__":
    seqs, subs = load(); out = []
    nF = sum(not x["succ"] for x in seqs); print(f"CALVIN: {len(seqs)} sequences, {nF} failed, {len(subs)} subtask instances")
    # 2. frozen rule, theta sweep (routing)
    out += gate_table(seqs, "d", [0.95, 0.92, 0.90, 0.88, 0.85], "routing_theta_sweep")
    # 5. motion, self-normalised (same rule) and raw-threshold
    out += gate_table(seqs, "jnt", [0.95, 0.85, 0.70], "motion_selfnorm")
    # 3. online length sentinel
    for k in [8, 10, 12, 14, 16, 20]:
        hit = tp = 0; runs = []
        for x in seqs:
            c = np.cumsum([0] + x["calls"])
            for j, n in enumerate(x["calls"]):
                if n >= k:
                    hit += 1; isf = (not x["succ"]) and j == len(x["calls"]) - 1; tp += isf
                    if isf: runs.append(x["end"] - (c[j] + k))
                    break
        out.append(dict(section="length_sentinel", signal="calls>=k", theta=k, alarms=hit, hits=tp,
                        precision=tp / hit if hit else np.nan, detection=tp / nF, false_alarm=np.nan,
                        runway_median=np.median(runs) if runs else np.nan))
    # 4. subtask-level risk set
    for k in [3, 4, 5, 6, 8, 10, 12]:
        alive = [s for s in subs if s["n"] >= k and len(s["seg"]) >= k]; pos = np.array([not s["ok"] for s in alive])
        out.append(dict(section="subtask_riskset", signal="routing_selfnorm", theta=k, alarms=len(alive), hits=int(pos.sum()),
                        precision=auc([-s["seg"][max(k - 4, 0):k].mean() / s["base"] for s in alive], pos), detection=np.nan, false_alarm=np.nan, runway_median=np.nan))
    # 6. label-free theta from CV
    normal = np.concatenate([x["d"][:W0] for x in seqs]); CV = normal.std() / normal.mean(); sdr = CV * np.sqrt(1 / W + 1 / W0)
    print(f"CV(first {W0} steps, label-free) = {CV:.4f}   sd(r) = {sdr:.4f}")
    for z in [1.5, 2.0, 2.5, 3.0]:
        for row in gate_table(seqs, "d", [1 - z * sdr], "theta_from_CV"): row["z"] = z; out.append(row)
    # 7. success-only quantile calibration (leave-one-out)
    S = [x for x in seqs if x["succ"]]
    def runmin(d): base = d[:W0].mean(); return min(d[t - W + 1:t + 1].mean() / base for t in range(W0, len(d)))
    mins = np.array([runmin(x["d"]) for x in S])
    for q in [0.05, 0.10, 0.15, 0.20]:
        tp = []; fp = 0
        for x in seqs:
            ref = mins if not x["succ"] else np.delete(mins, [i for i, s in enumerate(S) if s is x])
            f = fire(x["d"], float(np.quantile(ref, q)))
            if f is None: continue
            if x["succ"]: fp += 1
            else: tp.append(x["end"] - f)
        na = len(tp) + fp
        out.append(dict(section="success_quantile_LOO", signal="d", theta=q, alarms=na, hits=len(tp), precision=len(tp) / na if na else np.nan,
                        detection=len(tp) / nF, false_alarm=fp / len(S), runway_median=np.median(tp) if tp else np.nan))
    df = pd.DataFrame(out); df.to_csv("/home/jovyan/work/himoe-vla/calvin_transfer_results.csv", index=False)
    pd.set_option("display.width", 200); print(df.to_string(index=False)); print("wrote calvin_transfer_results.csv")
