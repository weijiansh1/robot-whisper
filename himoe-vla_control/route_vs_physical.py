#!/usr/bin/env python3
"""Decisive control: at one and the same pre-loop moment, does MoE ROUTING
predict the future loop better than the PHYSICAL state does?

If the physical features do just as well, the routing signal is redundant and
the "routing is an internal observer" claim collapses.  The experiment is a
head-to-head of three models on identical matched comparisons:

  (a) route only      (b) physical only      (c) route + physical

and the load-bearing statistic is whether (c) beats (b) -- the only evidence
that routing carries *incremental* information.

Design (frozen before looking at any AUC)
-----------------------------------------
* Time point: every loop event contributes exactly one positive, at
  ``q = loop_onset - 2``.
* Controls: canonical ``all_no_event`` -- same group (corpus A: snapshot_key;
  harvest: init_state), SAME ABSOLUTE query q, trajectory without that event
  (loop_onset < 0).  Reused verbatim from
  analyze_trainfree_signal_matrix.matched_group_statistics (asserted equal).
* Independent unit: the group (22 snapshots on A, 6 init states on harvest),
  never the candidate/trunk.  All CIs bootstrap groups; the sign-flip
  permutation flips group-level effects.
* Causality: every feature at q uses only data with index <= q.  Physical
  features come from the query-resolution series produced by
  ``harvest_label.candidate_query_series`` / ``harvest_query_series`` (the
  extraction validated 352/352 against the authoritative onset table) -- they
  are imported, never re-implemented.  Route features come from the canonical
  ``analyze_trainfree_signal_matrix.extract_row_features`` (imported), reduced
  to trailing means over w in {2,4} queries.

Label-leakage discipline
------------------------
Two physical features are continuous relaxations of the loop onset rule itself
(``eef_tortuosity_w*`` = the canonical ``physical_loop_ratio``, path vs net
displacement; ``eef_return_dist`` = the rule's non-local-return clause with the
same lag >= 3 and no threshold).  They are marked ``leak_flagged`` in every
table, and every model is reported twice: with them (``phys16``/``phys3_canon``)
and without them (``phys13_clean``/``phys2_canon_clean``).

Outputs (all under himoe-vla_control/)
--------------------------------------
runs/route_vs_physical/summary.csv   headline: three AUCs + three deltas per
                                     corpus x control variant x model tier
runs/route_vs_physical/univariate.csv  per-feature grouped AUC + permutation p
runs/route_vs_physical/models.csv      per-model LOGO-CV AUC + bootstrap CI
runs/route_vs_physical/deltas.csv      model-vs-model deltas + sign-flip maxT
runs/route_vs_physical/meta.json       provenance, counts, canonical-match audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

CONTROL_ROOT = pathlib.Path(__file__).resolve().parent
ROUTE_CAPTURE = CONTROL_ROOT.parent / "himoe-route-capture"
TRAP_CODE = CONTROL_ROOT.parent / "himoe-vla_trap/code"

sys.dont_write_bytecode = True  # never drop __pycache__ into the read-only trees
for _p in (str(CONTROL_ROOT), str(ROUTE_CAPTURE), str(TRAP_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import harvest_label as hl  # noqa: E402  validated physical extraction (352/352)
import analyze_trainfree_signal_matrix as tf  # noqa: E402  canonical route features

from sklearn.linear_model import LogisticRegression  # noqa: E402

A_RUN = hl.A_RUN
A_EVENTS = A_RUN / "analysis_trap_event_moe_20260829/event_onsets.csv"
SSM_CACHE = tf.SSM_CACHE
HARVEST = CONTROL_ROOT / "runs/harvest-pilot"
OUT_DIR = CONTROL_ROOT / "runs/route_vs_physical"
CACHE_DIR = OUT_DIR / "cache"

LEAD = -2
WINDOWS = (2, 4)
SEED = 20260904
HB_LAYERS_EXPECTED = [12, 13, 14, 15]  # tf.LAYERS = slice(4, 8) of the stored HB stack

ROUTE_FEATURES = [f"route_{base}_w{w}" for base in ("lfv", "acc") for w in WINDOWS]
PHYS_BASES = (
    "eef_speed_mean",
    "eef_speed_var",
    "eef_path_len",
    "eef_tortuosity",
    "grip_change",
    "obj_disp",
    "goal_rate",
)
PHYS_FEATURES = [f"phys_{b}_w{w}" for b in PHYS_BASES for w in WINDOWS] + [
    "phys_goal_dist",
    "phys_eef_return_dist",
]
LEAK_FLAGGED = {"phys_eef_tortuosity_w2", "phys_eef_tortuosity_w4",
                "phys_eef_return_dist"}
PHYS_CLEAN = [f for f in PHYS_FEATURES if f not in LEAK_FLAGGED]

# Canonical comparator triple already used by the frozen train-free pipeline
# (eef_motion_w4 / physical_progress_w4 / physical_loop_ratio_w4) and the two
# pre-registered route signals -- a low-dimensional, pre-specified model that
# small corpora can actually fit.
ROUTE_CANON = ["route_lfv_w4", "route_acc_w4"]
PHYS_CANON = ["phys_eef_speed_mean_w4", "phys_goal_rate_w4", "phys_eef_tortuosity_w4"]
PHYS_CANON_CLEAN = ["phys_eef_speed_mean_w4", "phys_goal_rate_w4"]

MODELS = {
    "route4": ROUTE_FEATURES,
    "phys16": PHYS_FEATURES,
    "phys13_clean": PHYS_CLEAN,
    "route4+phys16": ROUTE_FEATURES + PHYS_FEATURES,
    "route4+phys13_clean": ROUTE_FEATURES + PHYS_CLEAN,
    "route2_canon": ROUTE_CANON,
    "phys3_canon": PHYS_CANON,
    "phys2_canon_clean": PHYS_CANON_CLEAN,
    "route2+phys3_canon": ROUTE_CANON + PHYS_CANON,
    "route2+phys2_canon_clean": ROUTE_CANON + PHYS_CANON_CLEAN,
}
# (model_a, model_b, tag, family).  ``increment`` = (c) vs (b): does ROUTING add
# on top of physics -- the confirmatory, load-bearing family.  ``physinc`` =
# (c) vs (a), the mirror question.  ``duel`` = (a) vs (b) head to head.  maxT is
# corrected inside each family AND, most conservatively, across all twelve.
COMPARISONS = [
    ("route4+phys16", "phys16", "increment_full", "increment"),
    ("route4+phys13_clean", "phys13_clean", "increment_full_clean", "increment"),
    ("route2+phys3_canon", "phys3_canon", "increment_canon", "increment"),
    ("route2+phys2_canon_clean", "phys2_canon_clean", "increment_canon_clean", "increment"),
    ("route4+phys16", "route4", "physinc_full", "physinc"),
    ("route4+phys13_clean", "route4", "physinc_full_clean", "physinc"),
    ("route2+phys3_canon", "route2_canon", "physinc_canon", "physinc"),
    ("route2+phys2_canon_clean", "route2_canon", "physinc_canon_clean", "physinc"),
    ("route4", "phys16", "route_vs_phys_full", "duel"),
    ("route4", "phys13_clean", "route_vs_phys_full_clean", "duel"),
    ("route2_canon", "phys3_canon", "route_vs_phys_canon", "duel"),
    ("route2_canon", "phys2_canon_clean", "route_vs_phys_canon_clean", "duel"),
]


# --------------------------------------------------------------- provenance

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hb_layout(capture_summary: pathlib.Path) -> list[int]:
    """tf.LAYERS indexes stored HB gates; assert those are model layers 12-15."""
    gates = json.loads(capture_summary.read_text(encoding="utf-8"))["gates"]
    hb = [int(g["layer"]) for g in gates if g["kind"] == "HB"]
    stored = hb[tf.LAYERS]
    if stored != HB_LAYERS_EXPECTED:
        raise RuntimeError(f"{capture_summary}: HB layers {stored} != {HB_LAYERS_EXPECTED}")
    return stored


# ------------------------------------------------------------------ features

def scalar_stats(row_arrays: dict) -> dict:
    """Raw per-query route scalars, so the two captures can be compared."""
    return {name: {"mean": float(np.mean(row_arrays[name])),
                   "sd": float(np.std(row_arrays[name])),
                   "min": float(np.min(row_arrays[name])),
                   "max": float(np.max(row_arrays[name]))}
            for name in ("late_flow_volatility", "route_acceleration")}


def route_features(row_arrays: dict, rowidx: np.ndarray) -> dict[str, np.ndarray]:
    """Trailing means of the two canonical pre-loop route signals (causal)."""
    base = {
        "lfv": tf.scatter_rows(
            np.asarray(row_arrays["late_flow_volatility"], np.float64), rowidx),
        "acc": tf.scatter_rows(
            np.asarray(row_arrays["route_acceleration"], np.float64), rowidx),
    }
    out = {}
    for name, series in base.items():
        for w in WINDOWS:
            out[f"route_{name}_w{w}"] = tf.rolling_mean(series, w)
    return out


def physical_features(eef: np.ndarray, objects: np.ndarray, gripper: np.ndarray,
                      valid: np.ndarray, references: np.ndarray) -> dict[str, np.ndarray]:
    """Physical descriptors at query q from the query series up to q only.

    ``goal_rate`` reuses tf.trailing_physical (= the canonical
    ``physical_progress``); ``eef_speed_mean_w4`` reproduces the canonical
    ``eef_motion``; ``eef_tortuosity_w4`` reproduces the canonical
    ``physical_loop_ratio`` (leak-flagged, see module docstring).
    """
    n, tmax = valid.shape
    goal = np.full((n, tmax), np.nan)
    for b in range(n):
        length = int(valid[b].sum())
        goal[b, :length] = tf.goal_distance(objects[b, :length], references)

    step = np.full((n, tmax), np.nan)
    step[:, 1:] = np.linalg.norm(np.diff(eef, axis=1), axis=2)
    grip_step = np.full((n, tmax), np.nan)
    grip_step[:, 1:] = np.abs(np.diff(gripper, axis=1))

    out: dict[str, np.ndarray] = {}
    for w in WINDOWS:
        mean_step = tf.rolling_mean(step, w)
        out[f"phys_eef_speed_mean_w{w}"] = mean_step
        path = mean_step * w
        out[f"phys_eef_path_len_w{w}"] = path
        out[f"phys_grip_change_w{w}"] = tf.rolling_mean(grip_step, w) * w

        variance = np.full((n, tmax), np.nan)
        net = np.full((n, tmax), np.nan)
        obj_disp = np.full((n, tmax), np.nan)
        for t in range(w, tmax):
            segment = step[:, t - w + 1: t + 1]
            ok = np.isfinite(segment).all(axis=1) & valid[:, t]
            variance[ok, t] = segment[ok].var(axis=1)
            pair = valid[:, t] & valid[:, t - w]
            net[pair, t] = np.linalg.norm(eef[pair, t] - eef[pair, t - w], axis=1)
            obj_disp[pair, t] = np.linalg.norm(
                objects[pair, t] - objects[pair, t - w], axis=2).max(axis=1)
        out[f"phys_eef_speed_var_w{w}"] = variance
        out[f"phys_obj_disp_w{w}"] = obj_disp
        out[f"phys_eef_tortuosity_w{w}"] = 1.0 - net / np.maximum(path, 1e-12)
        out[f"phys_goal_rate_w{w}"] = tf.trailing_physical(goal, valid, w)

    out["phys_goal_dist"] = np.where(valid, goal, np.nan)
    # Nearest non-local past visit (lag >= 3, exactly the loop rule's window,
    # but threshold-free).  Leak-flagged.
    ret = np.full((n, tmax), np.nan)
    for b in range(n):
        length = int(valid[b].sum())
        for t in range(3, length):
            ret[b, t] = float(np.linalg.norm(eef[b, :t - 2] - eef[b, t], axis=1).min())
    out["phys_eef_return_dist"] = ret
    return out


# ------------------------------------------------------------------ corpora

class Corpus:
    def __init__(self, tag, unit_id, group, T, loop_onset, success, features, audit):
        self.tag = tag
        self.unit_id = np.asarray(unit_id)
        self.group = np.asarray(group, dtype=object)
        self.T = np.asarray(T, int)
        self.loop_onset = np.asarray(loop_onset, int)
        self.success = np.asarray(success, bool)
        self.features = features
        self.audit = audit
        names = list(features)
        stack = np.stack([features[n] for n in names])
        self.finite = np.isfinite(stack).all(axis=0)
        self.valid = np.zeros(self.finite.shape, bool)
        for b, length in enumerate(self.T):
            self.valid[b, :length] = True
        self.finite &= self.valid

    def as_tf_corpus(self) -> tf.Corpus:
        """Minimal tf.Corpus so the canonical matched_group_statistics runs."""
        meta = pd.DataFrame({
            "episode_id": np.arange(len(self.T)),
            "success": self.success,
            "T": self.T,
        })
        empty = np.zeros(0)
        return tf.Corpus(
            tag=self.tag, meta=meta, valid=self.valid, rowidx=empty, actions=empty,
            match_group=self.group, trap_onset=self.loop_onset,
            loop_onset=self.loop_onset, static_onset=self.loop_onset,
            trap_proxy=self.loop_onset, goal_distance=empty,
            physical_progress=empty, eef_motion=empty, loop_ratio=empty)


def load_corpus_a() -> Corpus:
    events = pd.read_csv(A_EVENTS)
    meta = pd.read_csv(SSM_CACHE / "A_meta.csv")
    rowidx = np.load(SSM_CACHE / "A_rowidx.npy")
    table = events.set_index("episode_id").reindex(meta.episode_id.to_numpy())
    keys = ["worker", "snapshot", "candidate", "snapshot_key", "loop_onset", "n_query"]
    if table[keys].isna().any().any():
        raise RuntimeError("corpus A: event table does not cover the route cache")
    if not np.array_equal(table.n_query.to_numpy(int), meta["T"].to_numpy(int)):
        raise RuntimeError("corpus A: n_query disagrees with the route-cache T")

    n, tmax = rowidx.shape
    valid = rowidx >= 0
    eef = np.full((n, tmax, 3), np.nan)
    objects = np.full((n, tmax, 2, 3), np.nan)
    gripper = np.full((n, tmax), np.nan)
    for b, row in enumerate(table.itertuples()):
        path = (A_RUN / f"formal/worker{int(row.worker)}/snapshot_{int(row.snapshot):03d}/"
                f"candidate_{int(row.candidate):02d}.npz")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in
                      ("control_sim_state", "control_eef_position",
                       "control_gripper_qpos", "control_query_index")}
        e, o, g = hl.candidate_query_series(arrays)  # validated 352/352 extraction
        length = len(e)
        if length != int(row.n_query) or int(valid[b].sum()) != length:
            raise RuntimeError(f"corpus A episode {row.Index}: query bookkeeping mismatch")
        eef[b, :length] = e
        objects[b, :length] = o
        gripper[b, :length] = g

    references = hl.load_references(pathlib.Path(hl.DEFAULT_REFERENCES))
    # Re-derive the loop onset from this very extraction and require it to match
    # the authoritative table row for row (guards against silent drift).
    rederived = np.array([
        tf.physical_onsets(eef[b, :int(t)], objects[b, :int(t)],
                           gripper[b, :int(t)], references)[0]
        for b, t in enumerate(meta["T"].to_numpy(int))])
    if not np.array_equal(rederived, table.loop_onset.to_numpy(int)):
        raise RuntimeError("corpus A: re-derived loop onsets differ from event_onsets.csv")

    row_arrays, route_audit = tf.extract_row_features("A", False)
    features = route_features(row_arrays, rowidx)
    features.update(physical_features(eef, objects, gripper, valid, references))
    audit = {
        "route_cache": str(tf.INTERMEDIATE_DIR / "corpus_A_route_features.npz"),
        "route_cache_sha256": sha256(tf.INTERMEDIATE_DIR / "corpus_A_route_features.npz"),
        "route_cache_reused": bool(route_audit["cache_reused"]),
        "hb_layers": check_hb_layout(A_RUN / "formal/server/capture_summary.json"),
        "loop_onset_rederived_exact": True,
        "route_scalar_stats": scalar_stats(row_arrays),
        "state_flow_max_deviation": float(route_audit["state_flow_max_deviation"]),
        "n_units": int(n),
        "n_groups": int(pd.unique(table.snapshot_key.to_numpy()).size),
    }
    return Corpus("A", table.index.to_numpy(), table.snapshot_key.to_numpy(),
                  meta["T"].to_numpy(int), table.loop_onset.to_numpy(int),
                  meta["success"].to_numpy(bool), features, audit)


def harvest_route_rows() -> tuple[dict, dict]:
    """Canonical extract_row_features on the harvest store, cache kept local."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    store = HARVEST / "server/routes.zarr"
    saved_dir, saved_store = tf.INTERMEDIATE_DIR, tf.route_store
    tf.INTERMEDIATE_DIR = CACHE_DIR          # never write into himoe-vla_trap/
    tf.route_store = lambda tag: store
    try:
        return tf.extract_row_features("H", False)
    finally:
        tf.INTERMEDIATE_DIR, tf.route_store = saved_dir, saved_store


def load_corpus_h() -> Corpus:
    import zarr

    labels = pd.read_csv(HARVEST / "labels.csv")
    dirs = {}
    for tdir in sorted((HARVEST / "trunks").glob("trunk_*")):
        if tdir.is_dir() and (tdir / "dense.npz").exists():
            dirs[int(json.loads((tdir / "summary.json").read_text())["trunk_uid"])] = tdir
    if set(dirs) != set(labels.trunk_uid.astype(int)):
        raise RuntimeError("harvest: labels.csv and trunk directories disagree")
    hl.check_scene8_layout(json.loads(
        (HARVEST / "trunks/sim_layout.json").read_text(encoding="utf-8")))

    group = zarr.open_group(str(HARVEST / "server/routes.zarr"), mode="r")
    zarr_eid = np.asarray(group["episode_id"][:])
    row_of = {int(e): i for i, e in enumerate(zarr_eid)}

    n = len(labels)
    tmax = int(labels.n_queries.max())
    rowidx = np.full((n, tmax), -1, np.int64)
    eef = np.full((n, tmax, 3), np.nan)
    objects = np.full((n, tmax, 2, 3), np.nan)
    gripper = np.full((n, tmax), np.nan)
    for b, row in enumerate(labels.itertuples()):
        tdir = dirs[int(row.trunk_uid)]
        with np.load(tdir / "dense.npz", allow_pickle=False) as archive:
            dense = {name: np.asarray(archive[name]) for name in archive.files}
        length = int(row.n_queries)
        e, o, g = hl.harvest_query_series(tdir, dense, length)  # same conventions as A
        eef[b, :length], objects[b, :length], gripper[b, :length] = e, o, g
        base = int(json.loads((tdir / "summary.json").read_text())["episode_id_base"])
        if base != int(row.trunk_uid) * 100:
            raise RuntimeError(f"{tdir}: episode_id_base {base} != trunk_uid*100")
        for q in range(length):
            rowidx[b, q] = row_of[base + q]

    references = hl.load_references(pathlib.Path(hl.DEFAULT_REFERENCES))
    rederived = np.array([
        tf.physical_onsets(eef[b, :int(t)], objects[b, :int(t)],
                           gripper[b, :int(t)], references)[0]
        for b, t in enumerate(labels.n_queries.to_numpy(int))])
    if not np.array_equal(rederived, labels.loop_onset.to_numpy(int)):
        raise RuntimeError("harvest: re-derived loop onsets differ from labels.csv")

    row_arrays, route_audit = harvest_route_rows()
    valid = rowidx >= 0
    features = route_features(row_arrays, rowidx)
    features.update(physical_features(eef, objects, gripper, valid, references))
    audit = {
        "route_cache": str(CACHE_DIR / "corpus_H_route_features.npz"),
        "route_cache_reused": bool(route_audit["cache_reused"]),
        "route_rows": int(len(zarr_eid)),
        "hb_layers": check_hb_layout(HARVEST / "server/capture_summary.json"),
        "loop_onset_rederived_exact": True,
        "route_scalar_stats": scalar_stats(row_arrays),
        "state_flow_max_deviation": float(route_audit["state_flow_max_deviation"]),
        "n_units": int(n),
        "n_groups": int(labels.init_state.nunique()),
    }
    return Corpus("H", labels.trunk_uid.to_numpy(int),
                  labels.init_state.astype(str).to_numpy(),
                  labels.n_queries.to_numpy(int), labels.loop_onset.to_numpy(int),
                  labels.success.to_numpy(bool), features, audit)


# -------------------------------------------------------- comparison design

def comparison_sets(corpus: Corpus, variant: str, lead: int = LEAD) -> list[dict]:
    """One set per loop event: the event at q=onset+lead vs its controls at q.

    ``matched``  -- controls from the SAME group (canonical all_no_event).
    ``pooled``   -- controls from ANY group (secondary; breaks init-state
                    matching, reported separately and never as primary).
    Eligibility uses one common finite mask over ALL features so that every
    feature and every model is scored on exactly the same comparisons.
    """
    event = corpus.loop_onset >= 0
    control = corpus.loop_onset < 0
    sets = []
    for group in sorted(set(corpus.group)):
        in_group = corpus.group == group
        event_idx = np.flatnonzero(event & in_group)
        pool = np.flatnonzero(control & in_group) if variant == "matched" \
            else np.flatnonzero(control)
        for i in event_idx:
            q = int(corpus.loop_onset[i]) + lead
            if q < 0 or q >= corpus.finite.shape[1] or not corpus.finite[i, q]:
                continue
            eligible = pool[(corpus.T[pool] > q) & corpus.finite[pool, q]]
            if not len(eligible):
                continue
            sets.append({"group": str(group), "pos": int(i), "q": q,
                         "ctrl": [int(c) for c in eligible]})
    return sets


def set_values(corpus: Corpus, sets: list[dict], name: str) -> list[np.ndarray]:
    series = corpus.features[name]
    return [np.asarray([series[s["pos"], s["q"]]]
                       + [series[c, s["q"]] for c in s["ctrl"]], float)
            for s in sets]


def group_concordance(values: list[np.ndarray], sets: list[dict]) -> pd.DataFrame:
    rows = {}
    for v, s in zip(values, sets):
        rest = v[1:]
        wins = float((v[0] > rest).sum() + 0.5 * (v[0] == rest).sum())
        acc = rows.setdefault(s["group"], [0.0, 0, 0])
        acc[0] += wins
        acc[1] += len(rest)
        acc[2] += 1
    return pd.DataFrame([{"group": g, "concordant": c, "pairs": p, "n_events": k}
                         for g, (c, p, k) in sorted(rows.items())])


def bootstrap_groups(frame: pd.DataFrame, n_boot: int, rng) -> tuple[float, float]:
    if frame.empty:
        return np.nan, np.nan
    concordant = frame.concordant.to_numpy(float)
    pairs = frame.pairs.to_numpy(float)
    draws = rng.integers(0, len(frame), size=(n_boot, len(frame)))
    auc = concordant[draws].sum(axis=1) / np.maximum(pairs[draws].sum(axis=1), 1)
    return float(np.quantile(auc, 0.025)), float(np.quantile(auc, 0.975))


def bootstrap_delta(frame_a: pd.DataFrame, frame_b: pd.DataFrame, n_boot: int,
                    rng) -> tuple[float, float]:
    merged = frame_a.merge(frame_b, on="group", suffixes=("_a", "_b"))
    merged = merged[merged.pairs_a == merged.pairs_b]  # only identically-scored groups
    if merged.empty:
        return np.nan, np.nan
    ca = merged.concordant_a.to_numpy(float)
    cb = merged.concordant_b.to_numpy(float)
    pairs = merged.pairs_a.to_numpy(float)
    draws = rng.integers(0, len(merged), size=(n_boot, len(merged)))
    delta = (ca[draws].sum(axis=1) - cb[draws].sum(axis=1)) / \
        np.maximum(pairs[draws].sum(axis=1), 1)
    return float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))


# ------------------------------------------------------------------- tests

def within_set_permutation(corpus: Corpus, sets: list[dict], names: list[str],
                           n_perm: int, seed: int) -> dict:
    """Permute the OUTCOME inside every comparison set (which member is the
    event is exchangeable given group and query); maxT over the feature family."""
    n_sets, n_feat = len(sets), len(names)
    sizes = np.array([1 + len(s["ctrl"]) for s in sets])
    m_max = int(sizes.max())
    wins = np.zeros((n_feat, n_sets, m_max))
    for si, s in enumerate(sets):
        vals = np.stack([set_values(corpus, [s], n)[0] for n in names])  # [F, m]
        greater = (vals[:, :, None] > vals[:, None, :]).sum(axis=2)
        equal = (vals[:, :, None] == vals[:, None, :]).sum(axis=2) - 1
        wins[:, si, :sizes[si]] = greater + 0.5 * equal
    total = float((sizes - 1).sum())
    observed = wins[:, np.arange(n_sets), 0].sum(axis=1) / total
    observed_det = np.maximum(observed, 1.0 - observed)

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, n_feat))
    index = np.arange(n_sets)
    chunk = 400
    for start in range(0, n_perm, chunk):
        stop = min(n_perm, start + chunk)
        picks = rng.integers(0, sizes[None, :], size=(stop - start, n_sets))
        auc = wins[:, index[None, :], picks].sum(axis=2) / total   # [F, B]
        null[start:stop] = np.maximum(auc, 1.0 - auc).T
    null_max = null.max(axis=1)
    return {
        "auc": observed,
        "det_auc": observed_det,
        "p_point": (1 + (null >= observed_det).sum(axis=0)) / (n_perm + 1),
        "p_maxT": (1 + (null_max[:, None] >= observed_det).sum(axis=0)) / (n_perm + 1),
        "null_max_q95": float(np.quantile(null_max, 0.95)),
    }


def group_signflip(effects: np.ndarray, weights: np.ndarray, n_perm: int,
                   seed: int) -> dict:
    """Canonical sign-flip over GROUPS (the independent unit), maxT over columns.

    Two maxT variants are returned.  ``p_maxT`` is the house version on raw
    effects; it is only fair when every column lives on the same scale.  When a
    family mixes comparisons with very different effect sizes (e.g. a +0.02 and
    a +0.40 delta), the raw null_max is dominated by the largest column, so
    ``p_maxT_stud`` studentizes each column by its own exact sign-flip SD
    (sqrt(sum (e_g w_g)^2) / sum w_g) before taking the maximum (Westfall-Young
    on t-statistics).  Per-column p-values are identical either way.
    """
    contrib = np.nan_to_num(effects * weights)
    denominator = np.maximum(weights.sum(axis=0), 1e-12)
    observed = np.abs(contrib.sum(axis=0) / denominator)
    sd = np.sqrt((contrib ** 2).sum(axis=0)) / denominator
    sd = np.where(sd > 1e-12, sd, np.inf)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, effects.shape[1]))
    for draw in range(n_perm):
        signs = rng.choice((-1.0, 1.0), size=effects.shape[0])
        null[draw] = np.abs((contrib * signs[:, None]).sum(axis=0) / denominator)
    null_max = null.max(axis=1)
    null_max_stud = (null / sd).max(axis=1)
    return {
        "p_point": (1 + (null >= observed).sum(axis=0)) / (n_perm + 1),
        "p_maxT": (1 + (null_max[:, None] >= observed).sum(axis=0)) / (n_perm + 1),
        "p_maxT_stud": (1 + (null_max_stud[:, None] >= observed / sd).sum(axis=0))
        / (n_perm + 1),
        "null_max_q95": float(np.quantile(null_max, 0.95)),
    }


# -------------------------------------------------------------- model layer

def design_rows(corpus: Corpus, sets: list[dict], names: list[str]):
    """Unique (unit, query) rows with labels + the set -> row-index mapping."""
    keys, labels = {}, []
    membership = []
    for s in sets:
        entry = []
        for unit, label in [(s["pos"], 1)] + [(c, 0) for c in s["ctrl"]]:
            key = (unit, s["q"])
            if key not in keys:
                keys[key] = len(labels)
                labels.append(label)
            elif labels[keys[key]] != label:
                raise RuntimeError("a unit is both event and control at one query")
            entry.append(keys[key])
        membership.append(entry)
    order = sorted(keys, key=keys.get)
    X = np.stack([[corpus.features[n][u, q] for n in names] for u, q in order])
    return X, np.asarray(labels, int), membership


def logo_cv_predictions(corpus: Corpus, sets: list[dict], names: list[str]):
    """Leave-one-group-out CV; a fold's training rows exclude every row that the
    held-out group's comparisons touch (no feature leakage across folds)."""
    X, y, membership = design_rows(corpus, sets, names)
    groups = sorted({s["group"] for s in sets})
    scores = np.full(len(y), np.nan)
    train_sizes = []
    for held in groups:
        test_sets = [i for i, s in enumerate(sets) if s["group"] == held]
        test_rows = sorted({r for i in test_sets for r in membership[i]})
        mask = np.ones(len(y), bool)
        mask[test_rows] = False
        if mask.sum() < 4 or len(set(y[mask].tolist())) < 2:
            train_sizes.append(0)
            continue
        train_sizes.append(int(mask.sum()))
        mu = X[mask].mean(axis=0)
        sd = X[mask].std(axis=0)
        sd[sd < 1e-12] = 1.0
        # sklearn default penalty is L2; C=1.0 on standardized features keeps the
        # 20-feature model estimable on ~18 groups.
        model = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")
        model.fit((X[mask] - mu) / sd, y[mask])
        scores[test_rows] = model.decision_function((X[test_rows] - mu) / sd)
    values = []
    keep = []
    for i, entry in enumerate(membership):
        vals = scores[entry]
        if not np.isfinite(vals).all():
            continue
        values.append(vals)
        keep.append(i)
    return values, [sets[i] for i in keep], train_sizes


# ------------------------------------------------------------------ runners

def within_set_rank_correlation(a, b) -> float:
    """Mean Spearman between two score vectors inside each comparison set."""
    from scipy.stats import spearmanr

    if not a or not b:
        return float("nan")
    values_a, kept_a = a
    values_b, kept_b = b
    index_b = {(s["group"], s["pos"], s["q"]): v for s, v in zip(kept_b, values_b)}
    rhos, weights = [], []
    for s, va in zip(kept_a, values_a):
        vb = index_b.get((s["group"], s["pos"], s["q"]))
        if vb is None or len(va) < 4:
            continue
        if np.ptp(va) == 0 or np.ptp(vb) == 0:
            continue
        rhos.append(float(spearmanr(va, vb).statistic))
        weights.append(len(va))
    if not rhos:
        return float("nan")
    return float(np.average(rhos, weights=weights))


def verify_canonical_sets(corpus: Corpus, sets: list[dict]) -> dict:
    """The comparison sets must reproduce the canonical matched_group_statistics."""
    name = ROUTE_FEATURES[0]
    masked = np.where(corpus.finite, corpus.features[name], np.nan)
    canonical, _ = tf.matched_group_statistics(
        masked, corpus.as_tf_corpus(), LEAD, "all_no_event",
        onset_values=corpus.loop_onset)
    mine = group_concordance(set_values(corpus, sets, name), sets)
    ref = pd.DataFrame(canonical)[["group", "concordant", "pairs", "n_events"]] \
        .sort_values("group").reset_index(drop=True)
    ref["group"] = ref.group.astype(str)
    got = mine.sort_values("group").reset_index(drop=True)
    ok = (len(ref) == len(got) and np.allclose(ref.concordant, got.concordant)
          and np.array_equal(ref.pairs.to_numpy(int), got.pairs.to_numpy(int))
          and np.array_equal(ref.n_events.to_numpy(int), got.n_events.to_numpy(int)))
    if not ok:
        raise RuntimeError("comparison sets differ from canonical all_no_event")
    return {"canonical_all_no_event_match": True,
            "n_groups": int(len(ref)), "n_pairs": int(ref.pairs.sum())}


def analyse(corpus: Corpus, variant: str, n_perm: int, n_boot: int):
    sets = comparison_sets(corpus, variant)
    if not sets:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"n_sets": 0}
    names = ROUTE_FEATURES + PHYS_FEATURES
    rng = np.random.default_rng(SEED + ord(corpus.tag) + (0 if variant == "matched" else 7))

    per_group = {n: group_concordance(set_values(corpus, sets, n), sets) for n in names}
    groups = sorted({s["group"] for s in sets})
    effects = np.full((len(groups), len(names)), np.nan)
    weights = np.zeros((len(groups), len(names)))
    for j, n in enumerate(names):
        frame = per_group[n].set_index("group")
        for i, g in enumerate(groups):
            if g in frame.index:
                effects[i, j] = frame.loc[g, "concordant"] / frame.loc[g, "pairs"] - 0.5
                weights[i, j] = frame.loc[g, "pairs"]
    perm = within_set_permutation(corpus, sets, names, n_perm, SEED + 11)
    flip = group_signflip(effects, weights, n_perm, SEED + 23)

    uni_rows = []
    for j, n in enumerate(names):
        frame = per_group[n]
        auc = float(frame.concordant.sum() / frame.pairs.sum())
        lo, hi = bootstrap_groups(frame, n_boot, rng)
        if abs(auc - perm["auc"][j]) > 1e-9:
            raise RuntimeError(f"{n}: AUC disagreement between code paths")
        uni_rows.append(dict(
            corpus=corpus.tag, control_variant=variant, block=n.split("_")[0],
            feature=n, leak_flagged=int(n in LEAK_FLAGGED),
            auc_loop_high=auc, det_auc=float(perm["det_auc"][j]),
            ci95_lo=lo, ci95_hi=hi,
            n_groups=int(len(frame)), n_events=int(frame.n_events.sum()),
            n_pairs=int(frame.pairs.sum()),
            p_point_permset=float(perm["p_point"][j]),
            p_maxT_permset=float(perm["p_maxT"][j]),
            p_point_signflip=float(flip["p_point"][j]),
            p_maxT_signflip=float(flip["p_maxT"][j]),
            p_maxT_signflip_stud=float(flip["p_maxT_stud"][j]),
            family_size=len(names)))

    model_rows, model_frames, model_scores = [], {}, {}
    for model, feats in MODELS.items():
        values, kept, train_sizes = logo_cv_predictions(corpus, sets, feats)
        model_scores[model] = (values, kept)
        if not values:
            model_rows.append(dict(
                corpus=corpus.tag, control_variant=variant, model=model,
                n_features=len(feats), cv_auc=np.nan, ci95_lo=np.nan, ci95_hi=np.nan,
                n_groups=0, n_events=0, n_pairs=0, min_train_rows=0,
                degenerate_folds=len(set(s["group"] for s in sets)),
                features=";".join(feats)))
            continue
        frame = group_concordance(values, kept)
        model_frames[model] = frame
        auc = float(frame.concordant.sum() / frame.pairs.sum())
        lo, hi = bootstrap_groups(frame, n_boot, rng)
        model_rows.append(dict(
            corpus=corpus.tag, control_variant=variant, model=model,
            n_features=len(feats), cv_auc=auc, ci95_lo=lo, ci95_hi=hi,
            n_groups=int(len(frame)), n_events=int(frame.n_events.sum()),
            n_pairs=int(frame.pairs.sum()),
            min_train_rows=int(min(train_sizes)),
            degenerate_folds=int(sum(1 for s in train_sizes if s == 0)),
            features=";".join(feats)))

    delta_rows = []
    usable = [c for c in COMPARISONS if c[0] in model_frames and c[1] in model_frames]
    if usable:
        d_effects = np.full((len(groups), len(usable)), np.nan)
        d_weights = np.zeros((len(groups), len(usable)))
        for j, (a, b, _tag, _fam) in enumerate(usable):
            fa = model_frames[a].set_index("group")
            fb = model_frames[b].set_index("group")
            for i, g in enumerate(groups):
                if g in fa.index and g in fb.index and fa.loc[g, "pairs"] == fb.loc[g, "pairs"]:
                    d_effects[i, j] = (fa.loc[g, "concordant"] - fb.loc[g, "concordant"]) \
                        / fa.loc[g, "pairs"]
                    d_weights[i, j] = fa.loc[g, "pairs"]
        wide = group_signflip(d_effects, d_weights, n_perm, SEED + 31)
        narrow = {}
        for family in sorted({c[3] for c in usable}):
            cols = [j for j, c in enumerate(usable) if c[3] == family]
            result = group_signflip(d_effects[:, cols], d_weights[:, cols],
                                    n_perm, SEED + 41)
            for k, j in enumerate(cols):
                narrow[j] = (result["p_maxT"][k], result["p_maxT_stud"][k], len(cols))
        for j, (a, b, tag, family) in enumerate(usable):
            fa, fb = model_frames[a], model_frames[b]
            auc_a = float(fa.concordant.sum() / fa.pairs.sum())
            auc_b = float(fb.concordant.sum() / fb.pairs.sum())
            lo, hi = bootstrap_delta(fa, fb, n_boot, rng)
            p_fam, p_fam_stud, fam_size = narrow[j]
            delta_rows.append(dict(
                corpus=corpus.tag, control_variant=variant, comparison=tag,
                family=family, model_a=a, model_b=b, auc_a=auc_a, auc_b=auc_b,
                delta=auc_a - auc_b, delta_ci95_lo=lo, delta_ci95_hi=hi,
                n_groups=int(np.count_nonzero(d_weights[:, j])),
                n_pairs=int(d_weights[:, j].sum()),
                p_point_signflip=float(wide["p_point"][j]),
                p_maxT_family=float(p_fam), p_maxT_family_stud=float(p_fam_stud),
                p_maxT_all12=float(wide["p_maxT"][j]),
                p_maxT_all12_stud=float(wide["p_maxT_stud"][j]),
                family_size=fam_size, family_size_all=len(usable)))

    audit = {
        # How redundant are the two blocks?  Mean within-comparison-set Spearman
        # of the two single-block out-of-fold scores (>=4 members per set).
        "within_set_spearman_route4_vs_phys16":
            within_set_rank_correlation(model_scores.get("route4"),
                                        model_scores.get("phys16")),
        "n_sets": len(sets), "n_groups": len(groups),
        "n_pairs": int(sum(len(s["ctrl"]) for s in sets)),
        "median_controls_per_event": float(np.median([len(s["ctrl"]) for s in sets])),
        "null_max_q95_permset": perm["null_max_q95"],
        "null_max_q95_signflip": flip["null_max_q95"],
        "n_groups_in_corpus": int(len(set(corpus.group))),
        "n_loop_events_in_corpus": int((corpus.loop_onset >= 0).sum()),
        "n_loop_events_used": len(sets),
        "groups": groups,
    }
    if variant == "matched":
        audit.update(verify_canonical_sets(corpus, sets))
    return (pd.DataFrame(uni_rows), pd.DataFrame(model_rows),
            pd.DataFrame(delta_rows), audit)


HEADLINE_TIERS = {
    "full": ("route4", "phys16", "route4+phys16",
             "increment_full", "physinc_full", "route_vs_phys_full"),
    "full_clean": ("route4", "phys13_clean", "route4+phys13_clean",
                   "increment_full_clean", "physinc_full_clean",
                   "route_vs_phys_full_clean"),
    "canon": ("route2_canon", "phys3_canon", "route2+phys3_canon",
              "increment_canon", "physinc_canon", "route_vs_phys_canon"),
    "canon_clean": ("route2_canon", "phys2_canon_clean", "route2+phys2_canon_clean",
                    "increment_canon_clean", "physinc_canon_clean",
                    "route_vs_phys_canon_clean"),
}


def headline(models: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    """One row per corpus x control variant x model tier: the three AUCs and the
    three deltas that the finding actually rests on."""
    rows = []
    for (corpus, variant), part in models.groupby(["corpus", "control_variant"]):
        by_model = part.set_index("model")
        dpart = deltas[(deltas.corpus == corpus)
                       & (deltas.control_variant == variant)].set_index("comparison")
        for tier, (a, b, c, inc, phinc, duel) in HEADLINE_TIERS.items():
            row = dict(corpus=corpus, control_variant=variant, tier=tier,
                       n_groups=int(by_model.loc[c, "n_groups"]),
                       n_events=int(by_model.loc[c, "n_events"]),
                       n_pairs=int(by_model.loc[c, "n_pairs"]))
            for key, model in (("route_only", a), ("phys_only", b), ("both", c)):
                row[f"auc_{key}"] = float(by_model.loc[model, "cv_auc"])
                row[f"ci_lo_{key}"] = float(by_model.loc[model, "ci95_lo"])
                row[f"ci_hi_{key}"] = float(by_model.loc[model, "ci95_hi"])
            for key, comparison in (("route_increment", inc),
                                    ("phys_increment", phinc), ("duel", duel)):
                if comparison not in dpart.index:
                    continue
                d = dpart.loc[comparison]
                row[f"delta_{key}"] = float(d.delta)
                row[f"delta_{key}_ci_lo"] = float(d.delta_ci95_lo)
                row[f"delta_{key}_ci_hi"] = float(d.delta_ci95_hi)
                row[f"p_{key}_point"] = float(d.p_point_signflip)
                row[f"p_{key}_maxT_family"] = float(d.p_maxT_family_stud)
                row[f"p_{key}_maxT_all12"] = float(d.p_maxT_all12_stud)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--permutations", type=int, default=5000)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--corpora", default="A,H")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loaders = {"A": load_corpus_a, "H": load_corpus_h}
    uni, models, deltas = [], [], []
    meta = {"lead": LEAD, "seed": SEED, "permutations": args.permutations,
            "bootstrap": args.bootstrap, "windows": list(WINDOWS),
            "route_features": ROUTE_FEATURES, "physical_features": PHYS_FEATURES,
            "leak_flagged": sorted(LEAK_FLAGGED), "models": MODELS,
            "comparisons": COMPARISONS, "corpora": {}}
    for tag in args.corpora.split(","):
        corpus = loaders[tag]()
        print(f"[{tag}] units={len(corpus.T)} groups={len(set(corpus.group))} "
              f"loop_events={int((corpus.loop_onset >= 0).sum())}", flush=True)
        meta["corpora"][tag] = {"load": corpus.audit, "variants": {}}
        for variant in ("matched", "pooled"):
            u, m, d, audit = analyse(corpus, variant, args.permutations, args.bootstrap)
            meta["corpora"][tag]["variants"][variant] = audit
            uni.append(u)
            models.append(m)
            deltas.append(d)
            print(f"  [{tag}/{variant}] sets={audit['n_sets']} groups={audit.get('n_groups')} "
                  f"pairs={audit.get('n_pairs')}", flush=True)

    uni = pd.concat([f for f in uni if len(f)], ignore_index=True)
    models = pd.concat([f for f in models if len(f)], ignore_index=True)
    deltas = pd.concat([f for f in deltas if len(f)], ignore_index=True)
    uni.to_csv(OUT_DIR / "univariate.csv", index=False)
    models.to_csv(OUT_DIR / "models.csv", index=False)
    deltas.to_csv(OUT_DIR / "deltas.csv", index=False)
    headline(models, deltas).to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n",
                                       encoding="utf-8")

    pd.set_option("display.width", 220)
    for tag in uni.corpus.unique():
        for variant in ("matched", "pooled"):
            part = uni[(uni.corpus == tag) & (uni.control_variant == variant)]
            if part.empty:
                continue
            head = part.iloc[0]
            print(f"\n=== corpus {tag} :: univariate ({variant}, lead {LEAD}) :: "
                  f"groups={head.n_groups} events={head.n_events} pairs={head.n_pairs} ===")
            print(part.sort_values("det_auc", ascending=False)[
                ["feature", "block", "leak_flagged", "auc_loop_high", "det_auc",
                 "ci95_lo", "ci95_hi", "p_maxT_permset", "p_maxT_signflip"]
            ].to_string(index=False))
            print(f"\n=== corpus {tag} :: models ({variant}) ===")
            print(models[(models.corpus == tag) & (models.control_variant == variant)][
                ["model", "n_features", "cv_auc", "ci95_lo", "ci95_hi", "n_groups",
                 "n_events", "n_pairs", "min_train_rows", "degenerate_folds"]
            ].to_string(index=False))
            print(f"\n=== corpus {tag} :: deltas ({variant}) ===")
            print(deltas[(deltas.corpus == tag) & (deltas.control_variant == variant)][
                ["comparison", "family", "auc_a", "auc_b", "delta", "delta_ci95_lo",
                 "delta_ci95_hi", "p_point_signflip", "p_maxT_family_stud",
                 "p_maxT_all12_stud", "n_groups"]
            ].to_string(index=False))
    print(f"\nwrote -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
