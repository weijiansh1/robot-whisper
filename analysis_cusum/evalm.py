"""Metrics for an alarm vector: rates, phase, index distribution, delay, ARL."""
import numpy as np

import seq


def _q(v, qs=(10, 25, 50, 75, 90)):
    if len(v) == 0:
        return {f"p{q}": np.nan for q in qs}
    return {f"p{q}": float(np.percentile(v, q)) for q in qs}


def bimodality(tidx, lo=seq.ALO, hi=seq.TMAX, bw=1.6):
    """Kernel-smoothed alarm-index density -> (#modes, relative dip depth).

    dip depth = 1 - (deepest valley between the two highest modes) / (lower of
    the two mode heights).  0 means no gap, 1 means the density touches zero
    between the peaks.  NaN if fewer than two modes.
    """
    tidx = np.asarray(tidx, float)
    if len(tidx) < 8:
        return 0, np.nan
    g = np.arange(lo, hi, 0.25)
    d = np.exp(-0.5 * ((g[:, None] - tidx[None, :]) / bw) ** 2).sum(1)
    d = d / d.sum()
    m = [i for i in range(1, len(d) - 1) if d[i] > d[i - 1] and d[i] >= d[i + 1]]
    if len(m) < 2:
        return len(m), np.nan
    top = sorted(m, key=lambda i: -d[i])[:2]
    a, b = sorted(top)
    valley = d[a:b + 1].min()
    return len(m), float(1.0 - valley / min(d[a], d[b]))


def metrics(alarm, meta, cls=None, onset=None, sel=None, lo=seq.ALO, hi=seq.TMAX):
    """sel restricts the *population* (e.g. a failure class); FA is always
    computed over all successes in the corpus."""
    Tb = meta["T"].values.astype(float)
    su = meta["success"].values.astype(bool)
    fired = alarm >= 0
    out = {}

    # --- false alarm over successes (whole corpus)
    out["n_succ"] = int(su.sum())
    out["fa"] = float(fired[su].mean()) if su.any() else np.nan
    out["n_fa"] = int(fired[su].sum())

    pop = (~su) if sel is None else sel
    out["n_pop"] = int(pop.sum())
    out["det"] = float(fired[pop].mean()) if pop.any() else np.nan
    out["n_det"] = int((fired & pop).sum())

    f = fired & pop
    ph = alarm[f] / Tb[f]
    ti = alarm[f].astype(float)
    out["phase_mean"] = float(ph.mean()) if len(ph) else np.nan
    for k, v in _q(ph).items():
        out["phase_" + k] = v
    out["t_mean"] = float(ti.mean()) if len(ti) else np.nan
    for k, v in _q(ti).items():
        out["t_" + k] = v
    nm, dip = bimodality(ti, lo, hi)
    out["t_nmodes"], out["t_dip"] = nm, dip
    # coarse histogram of the alarm index
    edges = [8, 12, 16, 20, 24, 28, 32, 35]
    hcnt, _ = np.histogram(ti, bins=edges)
    for i in range(len(edges) - 1):
        out[f"hist_{edges[i]}_{edges[i+1]}"] = int(hcnt[i])

    # --- delay relative to the frozen physical loop onset
    if onset is not None:
        ok = f & np.isfinite(onset)
        d = alarm[ok] - onset[ok]
        out["n_delay"] = int(ok.sum())
        out["delay_mean"] = float(d.mean()) if len(d) else np.nan
        out["frac_early"] = float((d < 0).mean()) if len(d) else np.nan
        out["frac_late"] = float((d > 0).mean()) if len(d) else np.nan
        for k, v in _q(d).items():
            out["delay_" + k] = v

    # --- ARL.  Censoring-aware hazard estimate: total exposed steps / alarms.
    #     Exposure for a branch = steps observed in [lo, min(hi, T)) up to and
    #     including its alarm (or the whole window if it never alarms).
    def arl(mask):
        expo = 0.0
        na = 0
        for b in np.where(mask)[0]:
            top = min(hi, int(Tb[b]))
            if top <= lo:
                continue
            if alarm[b] >= 0:
                expo += alarm[b] - lo + 1
                na += 1
            else:
                expo += top - lo
        return (expo / na if na else np.inf), expo, na

    a0, e0, n0 = arl(su)
    a1, e1, n1 = arl(pop)
    out["arl0"] = float(a0)
    out["arl0_expo"] = float(e0)
    out["arl1"] = float(a1)
    out["arl1_expo"] = float(e1)
    out["arl0_mean_alarm_t"] = float(alarm[fired & su].mean()) if (fired & su).any() else np.nan
    out["arl1_mean_alarm_t"] = float(alarm[f].mean()) if f.any() else np.nan
    return out
