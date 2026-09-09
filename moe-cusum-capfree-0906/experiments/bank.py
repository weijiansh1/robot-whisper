"""Build and cache the per-cohort channel bank, plus the two normalisation
arms.  Normalisation constants are always fitted on `development_main` and
applied unchanged to `external_8b`.
"""

from __future__ import annotations

import numpy as np

import capfree_common as C

DEV = "development_main"
EXT = "external_8b"
ARMS = ("raw", "self")


def derived_specs():
    """Log-mobility channels.  v7's `freeze` head is
    `-log(mobility_q / mean(mobility_{1..4}))` on the back layers, so a
    self-baselined log-mobility channel is the natural CUSUM analogue.
    """
    out = []
    for lay in C.LAYERS:
        for st in ("s9", "sm"):
            out.append(("deriv", "log_mobility", lay, st))
            out.append(("deriv", "log_state_mobility", lay, st))
    out.append(("deriv", "log_mobility", "back", "s9"))
    out.append(("deriv", "log_mobility", "front", "s9"))
    out.append(("deriv", "log_state_mobility", "back", "s9"))
    out.append(("deriv", "log_state_mobility", "front", "s9"))
    return out


def build(cohort: str):
    """-> (names, X[n_ch, n, 52] float32)."""
    specs = C.channel_specs()
    names, blocks = [], []
    mob = {}
    for name, x in C.iter_channels(cohort, specs):
        names.append(name)
        blocks.append(x)
        parts = name.split(":")
        if parts[1] in ("mobility", "state_mobility"):
            mob[(parts[1], parts[2], parts[3])] = x
    for src, base, lay, st in derived_specs():
        q = base.replace("log_", "")
        if lay in ("back", "front"):
            group = ("L12", "L13", "L14", "L15") if lay == "back" else \
                    ("L2", "L3", "L4", "L5")
            stack = np.stack([np.log(np.maximum(mob[(q, L, st)], 1e-6))
                              for L in group])
            x = np.median(stack, axis=0).astype(np.float32)
        else:
            x = np.log(np.maximum(mob[(q, lay, st)], 1e-6)).astype(np.float32)
        names.append(f"{src}:{base}:{lay}:{st}")
        blocks.append(x)
    return names, np.stack(blocks)


def fit_norm(X: np.ndarray, valid: np.ndarray):
    """Per-chunk mean/sd for each channel and each arm, fitted on development."""
    stats = {}
    for arm in ARMS:
        mus, sds = [], []
        for c in range(X.shape[0]):
            x = X[c]
            if arm == "self":
                x = _selfbase(x, valid)
            mu, sd = C.chunk_stats(x, valid)
            mus.append(mu)
            sds.append(sd)
        stats[arm] = (np.stack(mus), np.stack(sds))
    return stats


def _selfbase(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    base = np.where(valid[:, :4], x[:, :4], np.nan)
    with np.errstate(invalid="ignore", all="ignore"):
        b = np.nanmean(base, axis=1)
    b = np.where(np.isfinite(b), b, 0.0).astype(np.float32)
    return x - b[:, None]


def apply_norm(X: np.ndarray, valid: np.ndarray, stats, arm: str) -> np.ndarray:
    mus, sds = stats[arm]
    out = np.empty_like(X)
    for c in range(X.shape[0]):
        x = X[c]
        if arm == "self":
            x = _selfbase(x, valid)
        z = (x - mus[c][None, :]) / sds[c][None, :]
        z = np.where(np.isfinite(z), z, 0.0)
        out[c] = np.where(valid, z, 0.0)
    return out
