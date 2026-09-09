"""Stage 1: compute every divergence series d[b,t] for both corpora and cache to /tmp."""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

CACHE = "/tmp/moe_agg_series_{tag}.pkl"


def build(tag):
    p = CACHE.format(tag=tag)
    if os.path.exists(p):
        with open(p, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    P, valid, meta = lib.build_dense(tag)
    print(f"[{tag}] dense {P.shape} {P.nbytes/1e6:.0f} MB  {time.time()-t0:.1f}s", flush=True)
    D = {}
    for name in lib.DIVERGENCES:
        t1 = time.time()
        D[name] = lib.divergence_series(P, valid, name)
        print(f"[{tag}] {name:16s} {time.time()-t1:5.1f}s "
              f"finite={np.isfinite(D[name][:, 1:]).mean():.3f}", flush=True)
    obj = dict(D=D, valid=valid,
               groups=meta["group"].values, y=meta["success"].values.astype(bool),
               T=meta["T"].values, episode_id=meta["episode_id"].values)
    with open(p, "wb") as fh:
        pickle.dump(obj, fh, 4)
    print(f"[{tag}] cached -> {p}  total {time.time()-t0:.1f}s", flush=True)
    return obj


if __name__ == "__main__":
    for tag in ("A", "B"):
        o = build(tag)
        print(tag, "branches", len(o["y"]), "succ", int(o["y"].sum()),
              "groups", len(np.unique(o["groups"])))
