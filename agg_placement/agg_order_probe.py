"""Order sensitivity AMONG the 'before' reductions.

The main sweep fixes the order layer -> denoise -> token inside the Before group.
Because the before-ops act on probability vectors and are re-normalised after each
step, they do not commute either.  This probe measures how big that residual
free parameter is: all 6 orders x several operator triples x 2 scope sets,
placement = BBB (everything reduced before the divergence).
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import agg_core as A

AXES = ["layer", "denoise", "token"]
SCOPESETS = {
    "back/all/act": dict(layer="back", denoise="all", token="act"),
    "all/all/all11": dict(layer="all", denoise="all", token="all11"),
}
OPTRIPLES = [("mean", "mean", "mean"), ("max", "mean", "mean"), ("mean", "max", "mean"),
             ("mean", "mean", "max"), ("median", "median", "median"),
             ("min", "min", "min"), ("std", "mean", "mean"), ("q75", "q75", "q75"),
             ("max", "min", "median")]
SC = dict(layer=A.LAYER_SCOPES, denoise=A.DEN_SCOPES, token=A.TOK_SCOPES)


def feature(P, order, scopes, ops, first_row):
    """Reduce all three axes before the divergence, in the given order."""
    Q = P
    axpos = {"layer": 1, "denoise": 2, "token": 3}
    for ax in order:
        a = axpos[ax]
        sl = (slice(None),) * a + (SC[ax][scopes[ax]],)
        Q = A.red_prob(Q[sl], a, ops[ax])
        for other in axpos:
            if axpos[other] > a:
                axpos[other] -= 1
        axpos[ax] = -1
    return A.hell_consec(Q, first_row)


def gauc(s, y, g):
    num = den = 0.0
    for gg in np.unique(g):
        m = g == gg
        sf, ss = s[m][y[m] == 0], s[m][y[m] == 1]
        if len(sf) == 0 or len(ss) == 0:
            continue
        r = rankdata(np.concatenate([sf, ss]))
        a = (r[:len(sf)].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sf) * len(ss))
        num += a * len(sf) * len(ss)
        den += len(sf) * len(ss)
    return num / den


def run(tag):
    M = A.corpus_meta(tag)
    nbr = len(M["eids"])
    rows = []
    for i0 in range(0, nbr, 128):
        pass
    # single pass, blocked
    store = {}
    for i0 in range(0, nbr, 128):
        i1 = min(i0 + 128, nbr)
        P = A.read_block(M["arr"], M["rows"][i0:i1])
        nb = i1 - i0
        fr = np.zeros(nb * A.SLAB, bool)
        fr[::A.SLAB] = True
        W = np.empty((nb, len(A.TS), A.WIN), np.int64)
        for i in range(nb):
            for j, t in enumerate(A.TS):
                W[i, j] = i * A.SLAB + np.arange(t - A.T0 - A.WIN + 1, t - A.T0 + 1)
        for sname, scopes in SCOPESETS.items():
            for tri in OPTRIPLES:
                ops = dict(zip(AXES, tri))
                for order in itertools.permutations(AXES):
                    s = feature(P, order, scopes, ops, fr)
                    key = (sname, "/".join(tri), "->".join(o[0] for o in order))
                    store.setdefault(key, []).append(s[W].mean(-1))
        del P
    y, g = M["y"], M["g"]
    for key, parts in store.items():
        v = np.concatenate(parts, 0)
        for j, t in enumerate(A.TS):
            a = gauc(v[:, j], y, g)
            rows.append(dict(corpus=tag, scopes=key[0], ops=key[1], order=key[2], t=t,
                             auc=round(float(a), 6), det=round(float(max(a, 1 - a)), 6)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    D = pd.concat([run("A"), run("B")], ignore_index=True)
    D.to_csv("/tmp/moe_agg_order_probe.csv", index=False)
    print("wrote /tmp/moe_agg_order_probe.csv", len(D), "rows")
    piv = D[D.t == 30].pivot_table(index=["corpus", "scopes", "ops"], columns="order",
                                   values="det")
    piv["spread"] = piv.max(1) - piv.min(1)
    print(piv.round(4).to_string())
