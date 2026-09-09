"""Part 5 - is there any unexploited complementarity left.

The question is not whether two detectors fire on different episodes; at a fixed
false alarm budget almost any two do.  The question is whether combining them
buys true alarms that could not be bought more cheaply by simply loosening one
detector's own threshold.  So every combination is scored against the *matched
false alarm* single-detector alternative:

    at a development false alarm budget F, does the best OR-pair (or AND-pair,
    or greedy OR-of-k) beat the best single detector that also fits in F?

Selection happens on development; the development argmax is then scored once on
external.  The null arm replaces the whole detector bank with rate-matched
random alarm vectors (per-suite rates preserved) and repeats the comparison: a
combination gain that the null also shows is a base-rate artefact.

The negative control (length at the cap) recalls every risk by construction and
is excluded from all of this.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ledger_core as lc
import synth_core as sc

SEED = 20260906
BUDGET_GRID = (25, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000)
NULL_DRAWS = 20


def _parts(c: sc.Cohort, order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    iw = lc.in_window(c, order)
    return iw[:, c.risk].astype(np.float32), iw[:, ~c.risk].astype(np.float32)


def pair_tables(r: np.ndarray, s: np.ndarray) -> dict:
    tp, fp = r.sum(1), s.sum(1)
    ir = r @ r.T
    isf = s @ s.T
    return {
        "tp": tp, "fp": fp,
        "or_tp": tp[:, None] + tp[None, :] - ir,
        "or_fp": fp[:, None] + fp[None, :] - isf,
        "and_tp": ir, "and_fp": isf,
    }


def greedy_or(r: np.ndarray, s: np.ndarray, cap: float, kmax: int) -> tuple[list[int], float, float]:
    n = r.shape[0]
    cov_r = np.zeros(r.shape[1], bool)
    cov_s = np.zeros(s.shape[1], bool)
    chosen: list[int] = []
    for _ in range(kmax):
        best, best_gain = -1, 0
        for i in range(n):
            if i in chosen:
                continue
            nf = int((cov_s | (s[i] > 0)).sum())
            if nf > cap:
                continue
            g = int((cov_r | (r[i] > 0)).sum() - cov_r.sum())
            if g > best_gain:
                best, best_gain = i, g
        if best < 0:
            break
        chosen.append(best)
        cov_r |= r[best] > 0
        cov_s |= s[best] > 0
    return chosen, float(cov_r.sum()), float(cov_s.sum())


def frontier(dev_r, dev_s, ext_r, ext_s, names, tag) -> pd.DataFrame:
    d = pair_tables(dev_r, dev_s)
    n = len(names)
    iu = np.triu_indices(n, 1)
    rows = []
    for F in BUDGET_GRID:
        rec = {"arm": tag, "dev_fp_budget": F}
        ok = d["fp"] <= F
        if ok.any():
            i = int(np.flatnonzero(ok)[np.argmax(d["tp"][ok])])
            rec |= {"single_dev_tp": float(d["tp"][i]), "single_dev_fp": float(d["fp"][i]),
                    "single_ext_tp": float(ext_r[i].sum()), "single_ext_fp": float(ext_s[i].sum()),
                    "single_detector": names[i]}
        m = d["or_fp"][iu] <= F
        if m.any():
            k = int(np.flatnonzero(m)[np.argmax(d["or_tp"][iu][m])])
            a, b = int(iu[0][k]), int(iu[1][k])
            cov_r = (ext_r[a] > 0) | (ext_r[b] > 0)
            cov_s = (ext_s[a] > 0) | (ext_s[b] > 0)
            rec |= {"or_pair_dev_tp": float(d["or_tp"][a, b]), "or_pair_dev_fp": float(d["or_fp"][a, b]),
                    "or_pair_ext_tp": float(cov_r.sum()), "or_pair_ext_fp": float(cov_s.sum()),
                    "or_pair": f"{names[a]} OR {names[b]}"}
        m = d["and_fp"][iu] <= F
        if m.any():
            k = int(np.flatnonzero(m)[np.argmax(d["and_tp"][iu][m])])
            a, b = int(iu[0][k]), int(iu[1][k])
            cov_r = (ext_r[a] > 0) & (ext_r[b] > 0)
            cov_s = (ext_s[a] > 0) & (ext_s[b] > 0)
            rec |= {"and_pair_dev_tp": float(d["and_tp"][a, b]), "and_pair_dev_fp": float(d["and_fp"][a, b]),
                    "and_pair_ext_tp": float(cov_r.sum()), "and_pair_ext_fp": float(cov_s.sum()),
                    "and_pair": f"{names[a]} AND {names[b]}"}
        ch, gtp, gfp = greedy_or(dev_r, dev_s, F, 8)
        if ch:
            cov_r = np.zeros(ext_r.shape[1], bool)
            cov_s = np.zeros(ext_s.shape[1], bool)
            for i in ch:
                cov_r |= ext_r[i] > 0
                cov_s |= ext_s[i] > 0
            rec |= {"greedy_k": len(ch), "greedy_dev_tp": gtp, "greedy_dev_fp": gfp,
                    "greedy_ext_tp": float(cov_r.sum()), "greedy_ext_fp": float(cov_s.sum()),
                    "greedy_members": " | ".join(names[i] for i in ch)}
        rows.append(rec)
    return pd.DataFrame(rows)


def null_frontier(dev: sc.Cohort, order, rng, draws=NULL_DRAWS) -> pd.DataFrame:
    """Same frontier on rate-matched random alarm vectors, development only."""
    iw = lc.in_window(dev, order)
    rows = []
    for d in range(draws):
        nf = sc.rate_matched_null(iw, rng, strata=dev.suite)
        r = nf[:, dev.risk].astype(np.float32)
        s = nf[:, ~dev.risk].astype(np.float32)
        t = pair_tables(r, s)
        iu = np.triu_indices(len(order), 1)
        for F in BUDGET_GRID:
            ok = t["fp"] <= F
            best_single = float(t["tp"][ok].max()) if ok.any() else np.nan
            m = t["or_fp"][iu] <= F
            best_or = float(t["or_tp"][iu][m].max()) if m.any() else np.nan
            rows.append({"draw": d, "dev_fp_budget": F,
                         "null_single_dev_tp": best_single,
                         "null_or_pair_dev_tp": best_or,
                         "null_or_gain": best_or - best_single})
    return (pd.DataFrame(rows).groupby("dev_fp_budget").agg(["mean", "std"])
            .reset_index())


def family_pairs(dev: sc.Cohort, ext: sc.Cohort, reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = reps.detector.tolist()
    fams = reps.family.tolist()
    for c in (dev, ext):
        iw = lc.in_window(c, order)
        for scope, sm in [("pooled", np.ones(c.n, bool))] + [
            (s, c.suite == s) for s in sorted(set(c.suite))
        ]:
            rmask = c.risk & sm
            smask = (~c.risk) & sm
            for i in range(len(order)):
                for j in range(i + 1, len(order)):
                    a_r, b_r = iw[i, rmask], iw[j, rmask]
                    a_s, b_s = iw[i, smask], iw[j, smask]
                    ur, us = (a_r | b_r), (a_s | b_s)
                    tp_a, tp_b = int(a_r.sum()), int(b_r.sum())
                    fp_a, fp_b = int(a_s.sum()), int(b_s.sum())
                    better = 0 if tp_a >= tp_b else 1
                    base_tp, base_fp = (tp_a, fp_a) if better == 0 else (tp_b, fp_b)
                    d_tp, d_fp = int(ur.sum()) - base_tp, int(us.sum()) - base_fp
                    rows.append({
                        "cohort": c.name, "suite": scope,
                        "family_a": fams[i], "family_b": fams[j],
                        "tp_jaccard": float((a_r & b_r).sum() / max((a_r | b_r).sum(), 1)),
                        "fp_jaccard": float((a_s & b_s).sum() / max((a_s | b_s).sum(), 1)),
                        "tp_a": tp_a, "tp_b": tp_b, "fp_a": fp_a, "fp_b": fp_b,
                        "union_tp": int(ur.sum()), "union_fp": int(us.sum()),
                        "and_tp": int((a_r & b_r).sum()), "and_fp": int((a_s & b_s).sum()),
                        "marginal_tp_over_better": d_tp,
                        "marginal_fp_over_better": d_fp,
                        "exchange_rate_tp_per_fp": float(d_tp / d_fp) if d_fp else np.inf,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    dev = sc.load_cohort("development_main")
    ext = sc.load_cohort("external_8b")
    names, _ = lc.load_family_labels()

    dev_r, dev_s = _parts(dev, names)
    ext_r, ext_s = _parts(ext, names)
    fr = [frontier(dev_r, dev_s, ext_r, ext_s, names, "shared419_dev_selected")]

    # per suite, same protocol
    for s in sorted(set(dev.suite)):
        dm, em = dev.suite == s, ext.suite == s
        iwd, iwe = lc.in_window(dev, names), lc.in_window(ext, names)
        fr.append(frontier(
            iwd[:, dm & dev.risk].astype(np.float32),
            iwd[:, dm & ~dev.risk].astype(np.float32),
            iwe[:, em & ext.risk].astype(np.float32),
            iwe[:, em & ~ext.risk].astype(np.float32),
            names, f"shared419_dev_selected|{s}"))

    # in-sample arm on external only: the four bundles that never published a
    # development vector cannot be selected honestly, so this is an upper bound
    # on what they could add, not an estimate of it.
    ext_only = [d for d in ext.detectors if d not in set(names)]
    if ext_only:
        allnames = names + ext_only
        er, es = _parts(ext, allnames)
        fr.append(frontier(er, es, er, es, allnames, "external_all622_IN_SAMPLE"))

    pd.concat(fr, ignore_index=True).to_csv(sc.RESULTS / "combination_frontier.csv",
                                            index=False)
    null_frontier(dev, names, rng).to_csv(sc.RESULTS / "combination_null.csv", index=False)

    reps = lc.representatives(dev, names, _labels(), lc.REFERENCE_BUDGET)
    family_pairs(dev, ext, reps).to_csv(sc.RESULTS / "family_pair_complementarity.csv",
                                        index=False)

    head = pd.concat(fr, ignore_index=True)
    head = head[head.arm == "shared419_dev_selected"]
    meta = {
        "schema": "himoe.method_synthesis.complementarity.v1",
        "budget_grid": list(BUDGET_GRID),
        "protocol": ("combination selected on development at a matched development "
                     "false alarm budget, external scored once"),
    }
    (sc.RESULTS / "complementarity.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 260)
    print(head[["dev_fp_budget", "single_dev_tp", "or_pair_dev_tp", "and_pair_dev_tp",
                "greedy_k", "greedy_dev_tp", "single_ext_tp", "single_ext_fp",
                "or_pair_ext_tp", "or_pair_ext_fp", "greedy_ext_tp", "greedy_ext_fp"]]
          .to_string(index=False))


def _labels():
    z = np.load(sc.RESULTS / "family_labels.npz", allow_pickle=False)
    return z["labels_dev"]


if __name__ == "__main__":
    main()
