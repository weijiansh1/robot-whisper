"""Shared plumbing for the cap-free CUSUM study.

Nothing here may read the horizon cap, the phase label or the task identity.
The only episode-level quantity a detector is allowed to see is the *chunk
index* it is currently at; everything else must come from routing.

The evaluation metric, the fixed-chunk baseline and the rate-matched null are
imported from ``moe-capfree-0906/experiments/capfree_protocol.py`` rather than
reimplemented, so the anchors published against that harness are directly
comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
ROOT = BUNDLE.parent
RESULTS = BUNDLE / "results"

sys.path.insert(0, str(ROOT / "moe-capfree-0906" / "experiments"))

import capfree_protocol as cp  # noqa: E402

score = cp.score
fixed_chunk_baseline = cp.fixed_chunk_baseline
baseline_frontier = cp.baseline_frontier
rate_matched_null = cp.rate_matched_null
cohort_frame = cp.cohort_frame
LEADS = cp.LEADS
HEADLINE_LEAD = cp.HEADLINE_LEAD

N_CHUNKS = 52
LAYERS = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
# Target operating region for false alarms on an external-sized cohort.
FP_LO, FP_HI = 80, 320
SEED = 20260906

STEP_AGGS = {"s0": 0, "s3": 3, "s6": 6, "s9": 9, "sm": None}  # None = mean over steps

FLOW_DIR = ROOT / "moe-flow-semantics-0906" / "results" / "step_profiles"
CHAN_DIR = ROOT / "moe-unused-channels-0906" / "results" / "channels"
HB_DIR = ROOT / "moe-hb-front-back-0905" / "results" / "layer_graphs"

# `expert_load_effective_rank` == exp(token_entropy + token_differentiation)/32.
# The two halves carry opposite-signed information and cancel (L2: A=.252,
# B=.823, A+B=.514).  Banned as a channel; A and B are offered separately.
# `load_entropy` is *bit-exactly* that sum (verified: load_entropy - A == B to
# float tolerance on every valid cell), so it is banned on the same grounds.
BANNED = {"expert_load_effective_rank", "load_entropy"}


# --------------------------------------------------------------------------
# cohorts


def frame(cohort: str) -> dict:
    f = cohort_frame(cohort)
    length = f["length"].astype(int)
    f["length"] = length
    f["valid"] = np.arange(N_CHUNKS)[None, :] < length[:, None]
    f["n"] = len(length)
    return f


# --------------------------------------------------------------------------
# channel bank


def channel_specs() -> list[tuple[str, str, str, str]]:
    """(source, quantity, layer, step_agg).  ~540 candidates."""
    specs = []
    flow_metrics = ["token_entropy", "load_entropy", "token_differentiation",
                    "action_consensus", "state_action_alignment",
                    "conditional_energy", "conditional_effective_rank",
                    "flow_speed"]
    for lay in LAYERS:
        for m in flow_metrics:
            for sa in STEP_AGGS:
                specs.append(("flow", m, lay, sa))
        for m in ("mobility", "state_mobility"):
            for sa in STEP_AGGS:
                specs.append(("flow", m, lay, sa))
        for m in ("query_hard_churn", "query_hard_churn_d9", "query_top1_churn",
                  "denoise_hard_churn", "set_dwell", "tie_margin",
                  "selected_mass", "hb_entropy_action"):
            specs.append(("chan", m, lay, "-"))
        for m in ("flow_path", "flow_settling_log_ratio", "flow_endpoint",
                  "action_consensus", "state_action_alignment",
                  "conditional_energy", "conditional_effective_rank",
                  "partial_edge_std", "conditional_query_d1",
                  "partial_query_d1"):
            specs.append(("hb", m, lay, "-"))
    return [s for s in specs if s[1] not in BANNED]


def name_of(spec) -> str:
    return "{}:{}:{}:{}".format(*spec)


def _agg(block: np.ndarray, sa: str) -> np.ndarray:
    """block is [n, 52, 10] over denoising steps."""
    if sa == "sm":
        return block.mean(axis=2)
    return block[:, :, STEP_AGGS[sa]]


def iter_channels(cohort: str, specs):
    """Yield (spec_name, array[n, 52] float32), streaming one layer at a time."""
    wanted = {}
    for s in specs:
        wanted.setdefault((s[0], s[2]), []).append(s)

    idx = np.load(FLOW_DIR / f"{cohort}_index.npz", allow_pickle=True)
    flow_names = list(idx["metric_names"].astype(str))
    lay_names = list(idx["layer_names"].astype(str))
    q_idx = np.load(CHAN_DIR / f"{cohort}_index.npz", allow_pickle=True)
    q_names = list(q_idx["quantity_names"].astype(str))
    hb_names = None

    metrics = np.load(FLOW_DIR / f"{cohort}_metrics.npy", mmap_mode="r")
    mob = np.load(FLOW_DIR / f"{cohort}_mobility.npy", mmap_mode="r")
    smob = np.load(FLOW_DIR / f"{cohort}_state_mobility.npy", mmap_mode="r")
    quant = np.load(CHAN_DIR / f"{cohort}_quantities.npy", mmap_mode="r")
    hb = None

    for lay in LAYERS:
        li = lay_names.index(lay)
        fl = wanted.get(("flow", lay), [])
        if fl:
            blk = np.asarray(metrics[:, :, li], dtype=np.float32)      # [n,52,10,8]
            mblk = np.asarray(mob[:, :, li], dtype=np.float32)         # [n,52,10]
            sblk = np.asarray(smob[:, :, li], dtype=np.float32)
            for s in fl:
                _, m, _, sa = s
                if m == "mobility":
                    yield name_of(s), _agg(mblk, sa)
                elif m == "state_mobility":
                    yield name_of(s), _agg(sblk, sa)
                else:
                    yield name_of(s), _agg(blk[:, :, :, flow_names.index(m)], sa)
            del blk, mblk, sblk
        ch = wanted.get(("chan", lay), [])
        if ch:
            qblk = np.asarray(quant[:, :, li], dtype=np.float32)       # [n,52,8]
            for s in ch:
                yield name_of(s), qblk[:, :, q_names.index(s[1])].copy()
            del qblk
        hbs = wanted.get(("hb", lay), [])
        if hbs:
            if hb is None:
                z = np.load(HB_DIR / f"{cohort}.npz", allow_pickle=True)
                hb_names = list(z["metric_names"].astype(str))
                hb = z["metrics"]                                      # [n,52,8,11]
            hblk = np.asarray(hb[:, :, li], dtype=np.float32)
            for s in hbs:
                yield name_of(s), hblk[:, :, hb_names.index(s[1])].copy()
            del hblk


# --------------------------------------------------------------------------
# normalisation arms.  "raw" is the default; the prior ablation found raw-only
# best at 6 of 8 budgets and proved raw and a population rank are the same
# order within a fixed chunk.


def chunk_stats(x: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-chunk-index mean/sd over the episodes still running at that chunk.

    Uses the chunk index only, which the protocol allows; it never uses the
    cap, the phase or the task.  Affine within a chunk, so it preserves the
    within-chunk ordering of `raw` exactly.
    """
    mu = np.zeros(N_CHUNKS, np.float64)
    sd = np.ones(N_CHUNKS, np.float64)
    for q in range(N_CHUNKS):
        v = valid[:, q]
        if v.sum() < 8:
            continue
        col = x[v, q].astype(np.float64)
        col = col[np.isfinite(col)]
        if col.size < 8:
            continue
        mu[q] = col.mean()
        s = col.std()
        sd[q] = s if s > 1e-9 else 1.0
    return mu, sd


def normalise(x: np.ndarray, valid: np.ndarray, mu: np.ndarray, sd: np.ndarray,
              arm: str = "raw", edges: np.ndarray | None = None) -> np.ndarray:
    """Return a z-array; invalid cells are set to 0 (never read)."""
    if arm == "self":
        base = np.where(valid[:, :4], x[:, :4], np.nan)
        with np.errstate(invalid="ignore"):
            b = np.nanmean(base, axis=1)
        b = np.where(np.isfinite(b), b, 0.0)
        x = x - b[:, None]
    z = (x - mu[None, :]) / sd[None, :]
    if arm == "pop":
        # within-chunk population rank -> normal quantile, using development
        # break points.  Same within-chunk order as raw by construction.
        out = np.zeros_like(z)
        for q in range(N_CHUNKS):
            e = edges[q]
            r = np.searchsorted(e, z[:, q], side="left") / max(1, len(e))
            out[:, q] = np.clip(r, 1e-4, 1 - 1e-4)
        from scipy.special import ndtri
        z = ndtri(out).astype(np.float32)
    z = np.where(np.isfinite(z), z, 0.0)
    z = np.where(valid, z, 0.0)
    return z.astype(np.float32)


def pop_edges(x: np.ndarray, valid: np.ndarray, mu, sd, n_edge: int = 256):
    z = (x - mu[None, :]) / sd[None, :]
    edges = []
    for q in range(N_CHUNKS):
        v = valid[:, q]
        col = z[v, q]
        col = col[np.isfinite(col)]
        if col.size < 8:
            edges.append(np.array([0.0]))
        else:
            edges.append(np.quantile(col, np.linspace(0, 1, n_edge)))
    return edges


# --------------------------------------------------------------------------
# detectors.  Ties always use `>=`: `select_early_lock.py:176,207` uses strict
# `>` and drops the whole tie group (89.3% loss on set_dwell).


def first_alarm_single(z: np.ndarray, valid: np.ndarray, thr: float,
                       q_min: int = 0) -> np.ndarray:
    fire = (z >= thr) & valid
    if q_min:
        fire[:, :q_min] = False
    any_fire = fire.any(axis=1)
    return np.where(any_fire, fire.argmax(axis=1), -1)


def first_alarm_kofm(z: np.ndarray, valid: np.ndarray, thr: float, K: int,
                     M: int, q_min: int = 0) -> np.ndarray:
    ind = ((z >= thr) & valid).astype(np.int16)
    cs = np.cumsum(ind, axis=1)
    pad = np.zeros((ind.shape[0], M), np.int16)
    cnt = cs - np.concatenate([pad, cs[:, :-M]], axis=1)[:, :N_CHUNKS]
    fire = (cnt >= K) & valid
    if q_min:
        fire[:, :q_min] = False
    any_fire = fire.any(axis=1)
    return np.where(any_fire, fire.argmax(axis=1), -1)


def cusum_path(z: np.ndarray, valid: np.ndarray, k: float) -> np.ndarray:
    """S_q = max(0, S_{q-1} + (z_q - k)), frozen on invalid chunks."""
    n, Q = z.shape
    S = np.zeros(n, np.float64)
    out = np.zeros((n, Q), np.float64)
    for q in range(Q):
        v = valid[:, q]
        S = np.where(v, np.maximum(0.0, S + (z[:, q] - k)), S)
        out[:, q] = S
    return out


def first_alarm_cusum(S: np.ndarray, valid: np.ndarray, h: float,
                      q_min: int = 0) -> np.ndarray:
    fire = (S >= h) & valid
    if q_min:
        fire[:, :q_min] = False
    any_fire = fire.any(axis=1)
    return np.where(any_fire, fire.argmax(axis=1), -1)


# --------------------------------------------------------------------------
# frontier helpers


def frontier(rows, lead: int = HEADLINE_LEAD):
    """Pareto-front (fp, tp) at a given lead, monotone in fp."""
    pts = sorted((r[f"fp_lead{lead}"], r[f"tp_lead{lead}"], i)
                 for i, r in enumerate(rows))
    best, out = -1, []
    for fp, tp, i in pts:
        if tp > best:
            best = tp
            out.append((fp, tp, i))
    return out


def excess_over_baseline(row, base_at, lead: int = HEADLINE_LEAD) -> int:
    return int(row[f"tp_lead{lead}"] - base_at(row[f"fp_lead{lead}"]))


def hull_frontier(base, lead: int = HEADLINE_LEAD):
    """Randomised cap-free baseline: the upper concave hull of the fixed-chunk
    points, through the origin.

    `capfree_protocol.baseline_frontier` takes the *step* envelope
    `max{tp : fp <= budget}`.  Two things make that envelope an understatement
    of what a zero-information rule can reach:

      * below the smallest baseline false-alarm count (28 on external) no q0 is
        admissible at all, so the envelope returns 0 and *every* detector's
        "excess" there equals its own TP - which is why a white-noise null
        scores +208 in that region;
      * flipping a coin between q0=a and q0=b gives the chord between their
        (fp, tp) points, so the whole concave hull is reachable with no
        information whatsoever.

    This reference removes both.  The step envelope is still reported, because
    it is the frozen protocol; the hull is reported alongside it.
    """
    pts = base[[f"fp_lead{lead}", f"tp_lead{lead}"]].to_numpy(float)
    pts = pts[np.argsort(pts[:, 0])]
    xs = np.concatenate([[0.0], pts[:, 0]])
    ys = np.concatenate([[0.0], pts[:, 1]])
    hull: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (x - x1) <= (y - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append((float(x), float(y)))
    hx = np.array([p[0] for p in hull])
    hy = np.maximum.accumulate(np.array([p[1] for p in hull]))

    def at(fp_budget: float) -> float:
        return float(np.interp(min(fp_budget, hx[-1]), hx, hy))

    return at
