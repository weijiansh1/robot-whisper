"""Merge the per-fold LGSSM outputs into one series file per (corpus, variant)."""
import io
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_ssm")
import ssm_lib as L
from ssm_run import CONFIGS, KEYS, MAIN


def main():
    corpus, variant = sys.argv[1], sys.argv[2]
    C = np.load(f"{L.CACHE}/prep/{corpus}_{variant}_common.npz")
    valid, grp = C["valid"], C["grp"]
    Bn, Tmax = valid.shape
    out = {f"{k}__{r}_{d}": np.zeros((Bn, Tmax)) for (r, d) in CONFIGS for k in KEYS}
    zfm = np.zeros((Bn, Tmax, MAIN[1]), np.float32)
    diags = []
    for g in np.unique(grp):
        z = np.load(f"{L.CACHE}/em/{corpus}_{variant}_f{g}.npz", allow_pickle=True)
        ho = z["ho"]
        for k in out:
            out[k][ho] = z[k]
        zfm[ho] = z["zf_main"]
        diags.append(pd.read_csv(io.StringIO(str(z["diag"]))))
    np.savez_compressed(f"{L.CACHE}/series_{corpus}_{variant}.npz", zf_main=zfm, **out)
    pd.concat(diags).to_csv(f"{L.OUT}/emdiag_{corpus}_{variant}.csv", index=False)
    print(f"assembled {corpus}/{variant}: {len(diags)} folds")


if __name__ == "__main__":
    main()
