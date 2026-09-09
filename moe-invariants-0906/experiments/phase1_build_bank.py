"""Phase 1 (LABEL-BLIND): build the routing-functional variable bank.

This script may read ONLY cached routing functionals and the structural index
fields needed to know which (episode, chunk) cells exist.  It must never touch
an outcome label.  Explicitly forbidden and asserted below:

  * `length`            - episode length; the risk label is defined against it
  * `risk`, `original_failure`, physical failure modes, first-alarm vectors

`valid` IS used, because without it we cannot tell a real chunk from padding.
`valid` is a structural mask; it is never a variable, never a target and never
enters any tightness score.  `task_index` is used only for stratification,
because the protocol asks for coefficient stability *across* tasks.

Output: results/bank/{cohort}_bank.npz
  X            [rows, V]  float32, z-scored later; NaN-free by construction
  row_query    [rows]     int32   episode row index into the cohort
  row_chunk    [rows]     int16   chunk index (>= 1; chunk 0 has NaN diffs)
  row_task     [rows]     int16
  row_suite    [rows]     int8
  var_names    [V]        str
  var_source / var_metric / var_layer / var_reduction
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
ROOT = BUNDLE.parent
OUT = BUNDLE / "results" / "bank"

SP = ROOT / "moe-flow-semantics-0906" / "results" / "step_profiles"
CH = ROOT / "moe-unused-channels-0906" / "results" / "channels"
LG = ROOT / "moe-hb-front-back-0905" / "results" / "layer_graphs"

COHORTS = ("development_main", "development_extra", "external_8b")
LAYERS = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
FRONT = ("L2", "L3", "L4", "L5")
BACK = ("L12", "L13", "L14", "L15")
REDUCTIONS = ("mean", "s0", "s9")

# Fields that would leak an outcome.  Asserted never to be pulled out of the
# index files by this module.
FORBIDDEN_INDEX_FIELDS = ("length",)


def _suite(task_name: str) -> str:
    return task_name.split("/")[0]


SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")


def load_index(path: Path) -> dict:
    f = np.load(path, allow_pickle=True)
    keys = set(f.files)
    for bad in FORBIDDEN_INDEX_FIELDS:
        assert bad in keys, f"{bad} expected present but must be skipped"
    return {
        "task_names": [str(t) for t in f["task_names"]],
        "task_index": np.asarray(f["task_index"]).astype(np.int16),
        "valid": np.asarray(f["valid"]),
        "layer_names": [str(t) for t in f["layer_names"]],
        "names": [str(t) for t in (f["metric_names"] if "metric_names" in keys
                                   else f["quantity_names"])],
    }


def build(cohort: str) -> dict:
    sp_ix = load_index(SP / f"{cohort}_index.npz")
    ch_ix = load_index(CH / f"{cohort}_index.npz")
    lg_f = np.load(LG / f"{cohort}.npz", allow_pickle=True)

    valid = sp_ix["valid"]
    assert np.array_equal(valid, ch_ix["valid"]), "cache misalignment"
    assert np.array_equal(valid, np.asarray(lg_f["valid"])), "cache misalignment"
    assert sp_ix["layer_names"] == list(LAYERS)
    n, n_chunk = valid.shape

    # chunk 0 has NaN in every query-difference quantity; drop it wholesale so
    # every variable shares one sample set.
    use = valid.copy()
    use[:, 0] = False
    qi, ci = np.nonzero(use)
    rows = qi.size

    cols: list[np.ndarray] = []
    names: list[str] = []
    src: list[str] = []
    metric: list[str] = []
    layer: list[str] = []
    red: list[str] = []

    def add(vec, s, m, ly, r):
        cols.append(np.asarray(vec, dtype=np.float32))
        names.append(f"{s}.{m}@{ly}[{r}]")
        src.append(s)
        metric.append(m)
        layer.append(ly)
        red.append(r)

    # ---- step_profiles: [n, 52, 8 layer, 10 step, 8 metric] -----------------
    sp = np.load(SP / f"{cohort}_metrics.npy", mmap_mode="r")
    assert sp.shape == (n, n_chunk, 8, 10, 8), sp.shape
    for li, ly in enumerate(LAYERS):
        block = np.asarray(sp[:, :, li, :, :])          # [n, 52, 10, 8]
        block = block[qi, ci]                            # [rows, 10, 8]
        for mi, mname in enumerate(sp_ix["names"]):
            series = block[:, :, mi]                     # [rows, 10]
            for r in REDUCTIONS:
                if r == "mean":
                    v = np.nanmean(series, axis=1)
                elif r == "s0":
                    v = series[:, 0]
                else:
                    v = series[:, 9]
                if not np.isfinite(v).all():
                    continue                             # flow_speed[s0]
                add(v, "sp", mname, ly, r)
        del block
    del sp

    # ---- mobility / state_mobility: [n, 52, 8 layer, 10 step] --------------
    for tag, fname in (("mobility", f"{cohort}_mobility.npy"),
                       ("state_mobility", f"{cohort}_state_mobility.npy")):
        arr = np.load(SP / fname, mmap_mode="r")
        assert arr.shape == (n, n_chunk, 8, 10), arr.shape
        for li, ly in enumerate(LAYERS):
            series = np.asarray(arr[:, :, li, :])[qi, ci]      # [rows, 10]
            for r in REDUCTIONS:
                v = (series.mean(axis=1) if r == "mean"
                     else series[:, 0] if r == "s0" else series[:, 9])
                if not np.isfinite(v).all():
                    continue
                add(v, "mob", tag, ly, r)
        del arr

    # ---- unused-channels: [n, 52, 8 layer, 8 quantity] ---------------------
    q = np.load(CH / f"{cohort}_quantities.npy", mmap_mode="r")
    assert q.shape == (n, n_chunk, 8, 8), q.shape
    for li, ly in enumerate(LAYERS):
        block = np.asarray(q[:, :, li, :])[qi, ci]
        for mi, mname in enumerate(ch_ix["names"]):
            v = block[:, mi]
            if not np.isfinite(v).all():
                continue
            add(v, "ch", mname, ly, "agg")
        del block
    del q

    # ---- layer graphs: [n, 52, 8 layer, 11 metric] -------------------------
    gm = lg_f["metrics"]
    gnames = [str(t) for t in lg_f["metric_names"]]
    assert gm.shape == (n, n_chunk, 8, 11), gm.shape
    for li, ly in enumerate(LAYERS):
        block = np.asarray(gm[:, :, li, :])[qi, ci]
        for mi, mname in enumerate(gnames):
            v = block[:, mi]
            if not np.isfinite(v).all():
                continue
            add(v, "lg", mname, ly, "agg")
        del block

    X = np.stack(cols, axis=1)
    assert np.isfinite(X).all()

    task_names = sp_ix["task_names"]
    suite_of_task = np.array([SUITES.index(_suite(t)) for t in task_names],
                             dtype=np.int8)
    row_task = sp_ix["task_index"][qi]
    return {
        "X": X,
        "row_query": qi.astype(np.int32),
        "row_chunk": ci.astype(np.int16),
        "row_task": row_task.astype(np.int16),
        "row_suite": suite_of_task[row_task].astype(np.int8),
        "var_names": np.array(names),
        "var_source": np.array(src),
        "var_metric": np.array(metric),
        "var_layer": np.array(layer),
        "var_reduction": np.array(red),
        "task_names": np.array(task_names),
        "n_episodes": np.int32(n),
    }


def within_episode_info(bank: dict) -> list[dict]:
    """Protocol requirement: report within-episode information of every
    quantity.  eta2_between = fraction of total variance that is between
    episodes; a quantity with eta2 ~ 1 is an episode-level constant."""
    X = bank["X"].astype(np.float64)
    q = bank["row_query"]
    nq = int(q.max()) + 1
    cnt = np.bincount(q, minlength=nq).astype(np.float64)
    keep = cnt > 0
    out = []
    tot = X.var(axis=0)
    sums = np.zeros((nq, X.shape[1]))
    np.add.at(sums, q, X)
    means = np.zeros_like(sums)
    means[keep] = sums[keep] / cnt[keep, None]
    grand = X.mean(axis=0)
    between = ((cnt[keep, None] * (means[keep] - grand) ** 2).sum(axis=0)
               / cnt.sum())
    for i, name in enumerate(bank["var_names"]):
        col = X[:, i]
        uniq = np.unique(col).size
        # fraction of episodes on which the value never changes
        out.append({
            "var": str(name),
            "mean": float(col.mean()),
            "std": float(np.sqrt(tot[i])),
            "n_unique": int(uniq),
            "eta2_between_episode": float(between[i] / tot[i]) if tot[i] > 0 else 1.0,
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cohort in COHORTS:
        bank = build(cohort)
        np.savez_compressed(OUT / f"{cohort}_bank.npz", **bank)
        info = within_episode_info(bank)
        (OUT / f"{cohort}_within_episode_info.json").write_text(
            json.dumps(info, indent=1))
        summary[cohort] = {
            "rows": int(bank["X"].shape[0]),
            "n_vars": int(bank["X"].shape[1]),
            "n_episodes": int(bank["n_episodes"]),
            "n_tasks": int(len(bank["task_names"])),
            "constant_like_vars": [d["var"] for d in info
                                   if d["eta2_between_episode"] > 0.95],
            "low_cardinality_vars": [d["var"] for d in info if d["n_unique"] < 64],
        }
        print(cohort, summary[cohort]["rows"], "rows x",
              summary[cohort]["n_vars"], "vars", file=sys.stderr)
    (BUNDLE / "results" / "bank_summary.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
