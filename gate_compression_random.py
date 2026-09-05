"""Does the gate's rank-31 slice have any privilege over an arbitrary one?

`gate_compression.py` established that the 993 dims the router analytically
cannot see know the outcome just as well as the 31 it can.  That leaves the
question PPT_MOE_MECHANISM Slide 7 flags as the missing cell: is the gate's
particular 31-dim slice special at all, or would any slice do?

Two random arms, because one is not enough to answer it:

  rand        an unconstrained random orthonormal 31-dim slice.  This is the
              literal question -- but it is not a fair contest, since the
              gate's row space carries only 0.6-1.8% of h's variance while a
              random slice of a spectrally concentrated h carries much more.
              If rand wins on that, it won on budget, not on direction.

  rand_var    a 31-PC band chosen so its variance fraction matches the gate's
              at the same site.  Same family size, same variance budget, only
              the direction differs -- the comparison the project's own
              methodology battery ("give the control equal representational
              capacity") actually asks for.

Protocol, permutations and the `row` / `nullpc` arms are imported unchanged
from gate_compression so the two scripts cannot drift apart.  All four arms
share one permutation stream per site, so arm-to-arm differences are paired.
"""
import json

import numpy as np
import pandas as pd
import zarr

import gate_compression as gc

SEED = 20260903
OUT = f"{gc.ROOT}/gate_compression_random.csv"


def random_basis(rng, dim=1024, k=gc.NDIM):
    """Uniform random orthonormal k-frame in R^dim."""
    Q, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return Q


def matched_band(V, s2, target, k=gc.NDIM):
    """The k consecutive PCs whose variance fraction is closest to `target`."""
    frac = s2 / s2.sum()
    csum = np.concatenate([[0.0], np.cumsum(frac)])
    width = csum[k:] - csum[:-k]                       # variance of band [i, i+k)
    i = int(np.argmin(np.abs(width - target)))
    return V[:, i:i + k], float(width[i])


def main() -> int:
    bases, r2s = gc.fit_W()
    print("gate recovery R^2 per deep layer:", [f"{v:.5f}" for v in r2s], flush=True)

    rows, g, fail = gc.eval_rows()
    print(f"eval: {len(rows)} episodes at t={gc.T_EVAL}, "
          f"{len(np.unique(g))} init states, {fail.sum()} fail\n", flush=True)

    h = zarr.open_group(f"{gc.B_DIR}/server/hidden.zarr", mode="r")["hb_hidden"]
    order = np.argsort(rows)
    H = np.empty((len(rows), 4, 11, 1024), np.float32)
    for j, oi in enumerate(order):
        H[oi] = h[int(rows[oi]), gc.DEEP, gc.DENOISE, :, :]
        if j % 100 == 0:
            print(f"  read {j}/{len(rows)}", flush=True)

    out = []
    for i, L in enumerate(gc.LAYERS):
        U = bases[i]
        for tok in range(11):
            X = H[:, i, tok, :]
            Xc = X - X.mean(0)
            row = Xc @ U
            null = Xc - row @ U.T
            v_row = (row ** 2).sum()
            v_null = (null ** 2).sum()
            var_frac = v_row / (v_row + v_null)

            Un, sn, _ = np.linalg.svd(null, full_matrices=False)
            nullpc = Un[:, :gc.NDIM] * sn[:gc.NDIM]

            # full-space spectrum, for the variance-matched random band
            Uf, sf, Vft = np.linalg.svd(Xc, full_matrices=False)
            band, band_frac = matched_band(Vft.T, sf ** 2, var_frac)
            rand_var = Xc @ band

            rng = np.random.default_rng(SEED + 1000 * L + tok)
            rand = Xc @ random_basis(rng)
            v_rand = (rand ** 2).sum() / (v_row + v_null)

            rec = dict(layer=L, token=tok, var_frac_row=var_frac,
                       var_frac_rand=v_rand, var_frac_rand_var=band_frac)
            for name, M in (("row", row), ("nullpc", nullpc),
                            ("rand", rand), ("rand_var", rand_var)):
                gc.RNG = np.random.default_rng(SEED)      # paired null per site
                d, p, med = gc.det_family(M, g, fail)
                rec[f"det_{name}"] = d.max()
                rec[f"p_{name}"] = p.min()
                rec[f"nsig_{name}"] = int((p < 0.05).sum())
                rec[f"nullmed_{name}"] = med
            out.append(rec)
        print(f"  layer {L} done", flush=True)

    df = pd.DataFrame(out)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}\n", flush=True)

    print("variance fraction (median over 44 sites):")
    for c in ("var_frac_row", "var_frac_rand", "var_frac_rand_var"):
        print(f"  {c:20s} {df[c].median():.4f}")
    print("\nmax det (median over 44 sites), and sites where the arm beats `row`:")
    for a in ("row", "nullpc", "rand", "rand_var"):
        win = int((df[f"det_{a}"] > df["det_row"]).sum())
        print(f"  {a:9s} det {df[f'det_{a}'].median():.4f}   "
              f"sig sites {int((df[f'p_{a}'] < 0.05).sum())}/44   beats row {win}/44")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
