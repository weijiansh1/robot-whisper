"""Shared protocol for the online-detector study (corpus B).

Every number in ../tables/ comes through these five functions, so the protocol
is stated once.  Five knobs must be declared with any figure quoted from this
study — omitting any of them makes the number irreproducible:

    (channel, distance, W, q, decision-window upper bound)

plus K, which is fixed at 3 throughout.

Score
-----
Each control step contributes one scalar: the distance between chunk t-1 and
chunk t's router probability vector, computed per (layer, token) site over HB
layers 12-15 at denoise 9, then averaged over sites.  `st_deep` = suffix token 0
(4 sites).  `ac_deep` = suffix tokens 1-10 (40 sites).

The running score at step t is the mean of the last W such distances.  Nothing
downstream of step t is used; nothing about total episode length enters.

Threshold
---------
Leave-one-init-state-out.  For the held-out state g, take every branch from the
other 15 states that *succeeded*, and record the lowest running score it ever
reached inside the decision window.  The threshold is the q-quantile of those
minima.  So q is a nominal false-alarm rate on known-successful runs, not an
observed one — requiring K=3 consecutive steps below threshold turns a nominal
q=0.25 into an observed FA near 6%.

Decision window
---------------
`STOP=35` is the only uncontaminated setting: corpus B's shortest success runs
35 chunks, so every one of the 512 branches is alive for the whole window and
each gets the same number of chances to trip the alarm.  Extending STOP hands
failures more chances than successes (all 216 failures run to the 52-chunk
horizon), and detection climbs with the opportunity ratio whether or not the
score carries signal — see ../tables/window_inflation.csv.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
STOP_CLEAN = 35
K = 3


def load(name: str):
    return pickle.load(open(DATA / f"{name}.pkl", "rb"))


def classes(y):
    """success + the three ported physical failure predicates (128/41/47)."""
    cls = {int(r["episode_id"]): r["cls"]
           for r in csv.DictReader(open(DATA / "labels_B_derived.csv"))}
    return np.array([("success" if y[i] == 0 else cls.get(i, "?"))
                     for i in range(len(y))])


def threshold(series, y, grp, q, stop=STOP_CLEAN, W=6):
    """q-quantile of other init states' successful branches' running minima."""
    out = {}
    for g in set(grp):
        ref = [min(series[i][max(0, t - W):t].mean()
                   for t in range(W, min(len(series[i]), stop)))
               for i in range(len(y)) if grp[i] != g and y[i] == 0]
        out[g] = float(np.quantile(ref, q))
    return out


def detect(series, y, grp, W, q, stop=STOP_CLEAN, k=K):
    """K consecutive steps below threshold, latching.  Returns a bool array."""
    th = threshold(series, y, grp, q, stop, W)
    fired = np.zeros(len(y), bool)
    for i in range(len(y)):
        c, run = series[i], 0
        for t in range(W, min(len(c), stop)):
            run = run + 1 if c[max(0, t - W):t].mean() < th[grp[i]] else 0
            if run >= k:
                fired[i] = True
                break
    return fired


def summarise(fired, y, kd):
    """(observed FA, recall, per-class recall) — the row every table reports."""
    return dict(fa=float(fired[y == 0].mean()),
                recall=float(fired[y == 1].mean()),
                n_detected=int(fired[y == 1].sum()),
                **{c: float(fired[kd == c].mean())
                   for c in ("stagnation", "other", "loop")})
