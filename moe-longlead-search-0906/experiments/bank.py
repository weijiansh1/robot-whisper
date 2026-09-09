"""Series bank: named [n, 52] per-chunk series, all causal, all routing-only.

Two families:
  * `from_cached`  -- built from the two published arrays (step mobility
    [n,52,8,10] and flow_speed [n,52,8,9]).  No new extraction.
  * `from_raw`     -- built from `results/{cohort}_rawq.npz`, which
    `extract_raw_quantities.py` writes in one pass over routes.zarr.

Every series is defined so that chunk q uses only chunks <= q, and is NaN
wherever the episode has already ended (so an alarm can never be scored after
the fact).  Nothing here touches labels, horizons, phases or task identity.
"""

from __future__ import annotations

import numpy as np

FRONT = slice(0, 4)
BACK = slice(4, 8)


def _safe_log_ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.maximum(a, 1e-12) / np.maximum(b, 1e-12))
    out[~np.isfinite(out)] = np.nan
    return out


def _nan(fn, *args, **kw):
    with np.errstate(invalid="ignore", divide="ignore"):
        return fn(*args, **kw)


def open_baseline(series: np.ndarray, lo: int = 1, hi: int = 4) -> np.ndarray:
    """The episode's own opening regime: mean over chunks lo..hi-1.

    Fixed window, so unlike a running/expanding baseline it cannot adapt to a
    later anomaly.  Free to use: the detector observes it before it decides.
    """
    with np.errstate(invalid="ignore"):
        return np.nanmean(series[:, lo:hi], axis=1, keepdims=True)


def from_cached(mob: np.ndarray, speed: np.ndarray) -> dict[str, np.ndarray]:
    """mob [n,52,8,10] adjacent-chunk Hellinger per (layer, denoise step);
    speed [n,52,8,9] within-chunk flow speed per (layer, step transition)."""
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore"):
        # ---- amplitude, the family v7/v8 already mine -------------------
        out["mob_back_s9"] = np.nanmean(mob[:, :, BACK, 9], axis=2)
        out["mob_front_s9"] = np.nanmean(mob[:, :, FRONT, 9], axis=2)
        out["mob_all_smean"] = np.nanmean(mob, axis=(2, 3))
        # ---- dispersion across the two internal axes -------------------
        # how much the eight layers DISAGREE at one chunk (not how many are
        # extreme -- that is the consensus construction, deliberately avoided)
        out["mob_layer_spread"] = np.nanstd(mob[:, :, :, 9], axis=2)
        # how much the ten denoising steps disagree within one chunk
        out["mob_step_spread"] = np.nanstd(np.nanmean(mob, axis=2), axis=2)
        # ---- shape along the denoising axis ----------------------------
        late = np.nanmean(mob[:, :, :, 5:], axis=(2, 3))
        early = np.nanmean(mob[:, :, :, :5], axis=(2, 3))
        out["mob_late_early"] = _safe_log_ratio(late, early)
        out["mob_front_back"] = _safe_log_ratio(
            np.nanmean(mob[:, :, FRONT], axis=(2, 3)),
            np.nanmean(mob[:, :, BACK], axis=(2, 3)))
        # ---- flow speed family -----------------------------------------
        path = np.nansum(speed, axis=3)
        path[~np.isfinite(speed).any(axis=3)] = np.nan
        out["flow_path_back"] = np.nanmean(path[:, :, BACK], axis=2)
        out["flow_path_front"] = np.nanmean(path[:, :, FRONT], axis=2)
        out["flow_peak_back"] = np.nanmax(speed[:, :, BACK], axis=(2, 3))
        out["flow_tail_back"] = np.nanmean(speed[:, :, BACK, 6:], axis=(2, 3))
        out["flow_head_back"] = np.nanmean(speed[:, :, BACK, :3], axis=(2, 3))
        out["flow_tail_head"] = _safe_log_ratio(out["flow_tail_back"],
                                                out["flow_head_back"])
    return out


RAW_NAMES = ("set_jacc_adj", "set_inflow", "set_outflow", "set_asym",
             "set_novel_open", "rank_footrule_adj", "rank_footrule_open",
             "tok_disp", "tok_near_far", "tok_effrank", "state_action",
             "load_top4mass", "load_pr", "step_endpoint",
             "state_mob", "state_top4mass", "state_set_jacc")


def from_raw(raw) -> dict[str, np.ndarray]:
    """`raw[name]` is [n, 52, 8]; collapse the layer axis three ways.

    `_back` (L4..L7) is the depth band every frozen head reads, `_all` pools
    all eight, `_fbr` is the front/back log ratio -- a *shape* across depth
    rather than a magnitude, so it survives any per-episode gain change.
    """
    out: dict[str, np.ndarray] = {}
    for name in RAW_NAMES:
        arr = np.asarray(raw[name], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            back = np.nanmean(arr[:, :, BACK], axis=2)
            front = np.nanmean(arr[:, :, FRONT], axis=2)
            out[f"{name}_back"] = back
            out[f"{name}_all"] = np.nanmean(arr, axis=2)
            out[f"{name}_fbr"] = _safe_log_ratio(front, back)
    return out


def with_open_ratio(series: dict[str, np.ndarray],
                    keys=None) -> dict[str, np.ndarray]:
    """Add `<name>@open`: log(series_q / episode's own q1..q3 mean).

    This is the *fixed-window* self-baseline, not a running one; a running
    baseline was already shown to adapt to the anomaly it should flag.
    """
    out = {}
    for name, s in series.items():
        if keys is not None and name not in keys:
            continue
        base = open_baseline(s)
        out[f"{name}@open"] = _safe_log_ratio(s, base)
    return out


def mask_after_end(series: dict[str, np.ndarray],
                   length: np.ndarray) -> dict[str, np.ndarray]:
    """NaN out chunks >= length.  Online this is automatic (the episode is
    over); offline the padded arrays can leak a spurious constant, which is
    exactly how the frozen heads end up 'firing' on 31,598 successes with a
    negative lead."""
    q = np.arange(next(iter(series.values())).shape[1])[None, :]
    alive = q < np.asarray(length)[:, None]
    out = {}
    for name, s in series.items():
        t = s.copy()
        t[~alive] = np.nan
        out[name] = t
    return out
