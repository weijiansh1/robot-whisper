"""Action-input ablation: does the control term B A_k earn its place?

Compares the held-out one-step predictive negative log-likelihood and the detection AUCs
of the main latent config fitted WITH the executed action chunk as control input against
the same model with the control input zeroed (everything else identical).
"""
import sys

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_ssm")
import ssm_lib as L
from ssm_run import MAIN, KEYS

TS = (20, 25, 30)


def gather(corpus, variant):
    C = np.load(f"{L.CACHE}/prep/{corpus}_real_common.npz")
    valid, grp = C["valid"], C["grp"]
    out = {k: np.zeros(valid.shape) for k in KEYS}
    for g in np.unique(grp):
        z = np.load(f"{L.CACHE}/em/{corpus}_{variant}_f{g}.npz", allow_pickle=True)
        ho = z["ho"]
        for k in KEYS:
            out[k][ho] = z[f"{k}__{MAIN[0]}_{MAIN[1]}"]
    return out, C


def main():
    L.note(f"\n## Action-input ablation (main config r={MAIN[0]}, d={MAIN[1]})")
    L.note("| corpus | held-out mean predictive nll/chunk (with action) | (action zeroed) | "
           "delta | stat | det with action | det without |")
    L.note("|---|---|---|---|---|---|---|")
    for corpus in ("A", "B"):
        Wi, C = gather(corpus, "real")
        Wo, _ = gather(corpus, "noact")
        valid, y, grp, base = C["valid"], C["y"], C["grp"], C["base"]
        kk = np.arange(valid.shape[1])[None, :]
        m = valid & (kk <= 30)
        n1 = float(Wi["nll"][m].mean())
        n0 = float(Wo["nll"][m].mean())
        first = True
        for t in TS:
            for st, W in (("nis_w8", 8), ("innov_w8", 8), ("nis_w12", 12)):
                key = "nis" if st.startswith("nis") else "innov"
                v1 = Wi[key][:, t - W + 1:t + 1].mean(1)
                v0 = Wo[key][:, t - W + 1:t + 1].mean(1)
                a1, _ = L.within_group_auc(v1, y, grp)
                a0, _ = L.within_group_auc(v0, y, grp)
                lead = (f"| {corpus} | {n1:.3f} | {n0:.3f} | {n1-n0:+.4f} "
                        if first else "|  |  |  |  ")
                first = False
                L.note(lead + f"| t={t} {st} | {max(a1,1-a1):.3f} | {max(a0,1-a0):.3f} |")
                L.emit(corpus=corpus, t=t, stat=st, metric="ablate_action",
                       nll_with=n1, nll_without=n0, det_with=max(a1, 1 - a1),
                       det_without=max(a0, 1 - a0),
                       auc_with=a1, auc_without=a0)
    L.flush("moe_ssm_probe.csv")


if __name__ == "__main__":
    main()
