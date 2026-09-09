"""Part 1 - the coverage ledger: for every risk, which methods caught it.

Three questions, answered at a ladder of false alarm budgets because the answer
to all three is a function of the budget and is meaningless without one:

  * how many risks does no method catch in-window
  * how many are caught by exactly one, and is it always the same one
  * how does union recall grow as methods are added greedily, and where does it
    flatten

Selection is on development, scoring on external, and the two are reported
separately throughout.  The routing comparison for the uncaught risks reads the
already-computed layerwise mobility cache descriptively; it fits nothing.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import ledger_core as lc
import synth_core as sc


def uncaught_profile(c: sc.Cohort, caught: np.ndarray, tag: str) -> pd.DataFrame:
    """Suite / task / physical-mode / length profile of the risks nobody caught."""
    risk = c.risk
    unc = risk & ~caught
    cau = risk & caught
    rows = []
    for key, values in (("suite", c.suite), ("physical_mode", c.mode),
                        ("task", c.task)):
        for v in sorted(set(values[risk])):
            m = values == v
            rows.append({
                "cohort": c.name, "arm": tag, "field": key, "value": v,
                "n_risk": int((risk & m).sum()),
                "n_uncaught": int((unc & m).sum()),
                "uncaught_share": float((unc & m).sum() / max((risk & m).sum(), 1)),
                "share_of_all_uncaught": float((unc & m).sum() / max(unc.sum(), 1)),
            })
    # Length within suite, because the suites have different caps and a pooled
    # length comparison is a composition artefact, not a property of the risks.
    for s in sorted(set(c.suite[risk])):
        m = c.suite == s
        rows.append({
            "cohort": c.name, "arm": tag, "field": "mean_length_within_suite",
            "value": s, "n_risk": int((risk & m).sum()),
            "n_uncaught": int((unc & m).sum()),
            "uncaught_share": float(c.length[unc & m].mean()) if (unc & m).any() else np.nan,
            "share_of_all_uncaught": float(c.length[cau & m].mean()) if (cau & m).any() else np.nan,
        })
    rows.append({
        "cohort": c.name, "arm": tag, "field": "mean_length_pooled", "value": "all",
        "n_risk": int(risk.sum()), "n_uncaught": int(unc.sum()),
        "uncaught_share": float(c.length[unc].mean()) if unc.any() else np.nan,
        "share_of_all_uncaught": float(c.length[cau].mean()) if cau.any() else np.nan,
    })
    return pd.DataFrame(rows)


def routing_contrast(c: sc.Cohort, caught: np.ndarray, rng: np.random.Generator,
                     draws: int = 2000) -> pd.DataFrame:
    """Does the routing of an uncaught risk look different at all?

    Reads the published layerwise mobility cache and averages it over the
    in-window chunks.  This is description, not detection: no threshold is set
    and nothing is fitted.  The null shuffles the caught/uncaught label within
    (suite, physical mode) so a difference cannot come from composition.
    """
    sys.path.insert(0, str(sc.ROOT / "moe-v7-0905" / "experiments"))
    from evaluate_intrinsic_guard_v7 import EXTERNAL_LAYER, MAIN_LAYER, load_npz

    cache = load_npz(MAIN_LAYER if c.name == "development_main" else EXTERNAL_LAYER)
    mob = cache["mobility"].astype(np.float64)          # (n, chunks, layers)
    layers = [str(x) for x in cache["layer_names"]]
    chunk = np.arange(mob.shape[1])[None, :]
    win = (chunk < c.deadline[:, None]) & (chunk < c.length[:, None])
    with np.errstate(invalid="ignore"):
        prof = np.where(win[:, :, None], mob, np.nan)
        mean_mob = np.nanmean(prof, axis=1)             # (n, layers)

    unc = c.risk & ~caught
    cau = c.risk & caught
    safe = ~c.risk
    by_mode = pd.Series([f"{s}|{m}" for s, m in zip(c.suite, c.mode)]).to_numpy()
    by_task = pd.Series(c.task).to_numpy()

    contrasts = [
        ("uncaught_vs_caught_risk", unc, cau, by_mode),
        ("uncaught_risk_vs_safe", unc, safe, by_task),
        ("caught_risk_vs_safe", cau, safe, by_task),
    ]
    rows = []
    for li, lay in enumerate(layers):
        x = mean_mob[:, li]
        ok = np.isfinite(x)
        for label, ga, gb, strat in contrasts:
            a, b = ga & ok, gb & ok
            if a.sum() < 5 or b.sum() < 5:
                continue
            obs = float(np.mean(x[a]) - np.mean(x[b]))
            pool = np.flatnonzero(a | b)
            lab = a[pool]
            groups = strat[pool]
            uniq = np.unique(groups)
            null = np.empty(draws)
            for d in range(draws):
                perm = lab.copy()
                for g in uniq:
                    sel = groups == g
                    perm[sel] = rng.permutation(perm[sel])
                null[d] = np.mean(x[pool][perm]) - np.mean(x[pool][~perm])
            rows.append({
                "cohort": c.name, "layer": lay, "contrast": label,
                "n_a": int(a.sum()), "n_b": int(b.sum()),
                "mean_a": float(np.mean(x[a])), "mean_b": float(np.mean(x[b])),
                "difference": obs,
                "null_mean": float(null.mean()), "null_sd": float(null.std()),
                "z_vs_stratified_null": float((obs - null.mean()) / null.std())
                if null.std() > 0 else np.nan,
                "p_two_sided": float(
                    (np.abs(null - null.mean()) >= abs(obs - null.mean())).mean()
                ),
            })
    return pd.DataFrame(rows)


def saturation(dev: sc.Cohort, ext: sc.Cohort, order_names: list[str],
               budget: float, tag: str) -> pd.DataFrame:
    """Greedy addition, order fixed on development, scored on both cohorts."""
    iw_d = lc.in_window(dev, order_names)
    iw_e = lc.in_window(ext, order_names)
    remaining = list(range(len(order_names)))
    cov_d = np.zeros(dev.n, bool)
    cov_e = np.zeros(ext.n, bool)
    rows = []
    while remaining:
        gains = [(int((iw_d[i] & dev.risk & ~cov_d).sum()), -int((iw_d[i] & ~dev.risk & ~cov_d).sum()), i)
                 for i in remaining]
        gains.sort(reverse=True)
        best = gains[0][2]
        if gains[0][0] == 0:
            # nothing left adds a development true alarm; stop growing
            pass
        remaining.remove(best)
        cov_d |= iw_d[best]
        cov_e |= iw_e[best]
        rows.append({
            "arm": tag, "budget": budget, "step": len(rows) + 1,
            "added": order_names[best],
            "dev_recall": float((cov_d & dev.risk).sum() / dev.risk.sum()),
            "dev_fp": int((cov_d & ~dev.risk).sum()),
            "ext_recall": float((cov_e & ext.risk).sum() / ext.risk.sum()),
            "ext_fp": int((cov_e & ~ext.risk).sum()),
            "ext_marginal_tp": int((iw_e[best] & ext.risk & ~(cov_e & ~iw_e[best])).sum()),
        })
        if gains[0][0] == 0 and len(rows) > 1:
            break
    # recompute honest marginal gain on external following the dev order
    cov = np.zeros(ext.n, bool)
    for r in rows:
        i = order_names.index(r["added"])
        r["ext_marginal_tp"] = int((iw_e[i] & ext.risk & ~cov).sum())
        r["ext_marginal_fp"] = int((iw_e[i] & ~ext.risk & ~cov).sum())
        cov |= iw_e[i]
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(20260906)
    dev = sc.load_cohort("development_main")
    ext = sc.load_cohort("external_8b")
    names, labels = lc.load_family_labels()

    meta = {"schema": "himoe.method_synthesis.coverage.v1",
            "budgets": list(lc.BUDGETS), "reference_budget": lc.REFERENCE_BUDGET,
            "n_families": int(len(np.unique(labels)))}

    # ---- the literal question: caught by *no* method at *any* threshold ----
    for c, tag in ((dev, "development_main"), (ext, "external_8b")):
        iw_all = (c.alarms >= 0) & (c.alarms < c.deadline[None, :])
        cov = iw_all.any(0)
        meta[f"{tag}_unconstrained_union"] = {
            "n_detectors": int(c.alarms.shape[0]),
            "recall": float((cov & c.risk).sum() / c.risk.sum()),
            "n_uncaught_risks": int((c.risk & ~cov).sum()),
            "fp": int((cov & ~c.risk).sum()),
            "fpr": float((cov & ~c.risk).sum() / (~c.risk).sum()),
        }

    rep_rows, ledger_rows, prof_rows, sat_rows, one_rows = [], [], [], [], []
    for budget in lc.BUDGETS:
        reps = lc.representatives(dev, names, labels, budget)
        if reps.empty:
            continue
        rep_rows.append(reps)
        fam_d, fams = lc.family_matrix(dev, reps)
        fam_e, _ = lc.family_matrix(ext, reps)
        # ceiling: union over *every* development-admissible detector, not just
        # the nine representatives, so the family reduction can be costed
        adm_names = [names[i] for i in range(len(names))]
        iw_dev_all = lc.in_window(dev, adm_names)
        adm = [adm_names[i] for i in range(len(adm_names))
               if (iw_dev_all[i] & ~dev.risk).sum() / (~dev.risk).sum() <= budget]

        for c, mat, tag in ((dev, fam_d, "development_main"), (ext, fam_e, "external_8b")):
            ceil = lc.in_window(c, adm).any(0) if adm else np.zeros(c.n, bool)
            votes = mat.sum(0)
            cov = votes >= 1
            risk = c.risk
            ledger_rows.append({
                "cohort": tag, "budget": budget, "n_families_available": len(fams),
                "union_recall": float((cov & risk).sum() / risk.sum()),
                "n_uncaught": int((risk & ~cov).sum()),
                "union_fp": int((cov & ~risk).sum()),
                "union_fpr": float((cov & ~risk).sum() / (~risk).sum()),
                "union_precision": float((cov & risk).sum() / max(cov.sum(), 1)),
                "n_caught_by_exactly_one": int(((votes == 1) & risk).sum()),
                "n_caught_by_all": int(((votes == len(fams)) & risk).sum()),
                "mean_votes_on_risk": float(votes[risk].mean()),
                "mean_votes_on_safe": float(votes[~risk].mean()),
                "mean_votes_on_fired_safe": float(votes[~risk & (votes > 0)].mean())
                if (~risk & (votes > 0)).any() else np.nan,
                "n_admissible_detectors": len(adm),
                "ceiling_recall_all_admissible": float((ceil & risk).sum() / risk.sum()),
                "ceiling_fp_all_admissible": int((ceil & ~risk).sum()),
                "ceiling_n_uncaught": int((risk & ~ceil).sum()),
            })
            for s in sorted(set(c.suite)):
                m = c.suite == s
                v, r = votes[m], risk[m]
                ledger_rows.append({
                    "cohort": f"{tag}|{s}", "budget": budget,
                    "n_families_available": len(fams),
                    "union_recall": float(((v >= 1) & r).sum() / max(r.sum(), 1)),
                    "n_uncaught": int((r & (v == 0)).sum()),
                    "union_fp": int(((v >= 1) & ~r).sum()),
                    "union_fpr": float(((v >= 1) & ~r).sum() / max((~r).sum(), 1)),
                    "union_precision": float(((v >= 1) & r).sum() / max((v >= 1).sum(), 1)),
                    "n_caught_by_exactly_one": int(((v == 1) & r).sum()),
                    "n_caught_by_all": int(((v == len(fams)) & r).sum()),
                    "mean_votes_on_risk": float(v[r].mean()) if r.any() else np.nan,
                    "mean_votes_on_safe": float(v[~r].mean()),
                    "mean_votes_on_fired_safe": float(v[~r & (v > 0)].mean())
                    if (~r & (v > 0)).any() else np.nan,
                    "n_admissible_detectors": len(adm),
                    "ceiling_recall_all_admissible": float((ceil[m] & r).sum() / max(r.sum(), 1)),
                    "ceiling_fp_all_admissible": int((ceil[m] & ~r).sum()),
                    "ceiling_n_uncaught": int((r & ~ceil[m]).sum()),
                })
            # which family is the sole catcher
            sole = (votes == 1) & risk
            if sole.any():
                who = np.array(fams)[mat[:, sole].argmax(0)]
                for f, n in pd.Series(who).value_counts().items():
                    one_rows.append({"cohort": tag, "budget": budget, "family": int(f),
                                     "n_sole_catches": int(n),
                                     "share_of_sole": float(n / sole.sum()),
                                     "detector": reps.set_index("family").detector[f]})
            prof_rows.append(uncaught_profile(c, cov, f"families@{budget}"))

        sat_rows.append(saturation(dev, ext, reps.detector.tolist(), budget,
                                   "family_representatives"))
        if adm:
            sat_rows.append(saturation(dev, ext, adm, budget, "all_admissible_detectors"))

    reps_all = pd.concat(rep_rows, ignore_index=True)
    reps_all.to_csv(sc.RESULTS / "family_representatives.csv", index=False)
    pd.DataFrame(ledger_rows).to_csv(sc.RESULTS / "coverage_ledger.csv", index=False)
    pd.concat(prof_rows, ignore_index=True).to_csv(
        sc.RESULTS / "uncaught_profile.csv", index=False)
    pd.concat(sat_rows, ignore_index=True).to_csv(
        sc.RESULTS / "saturation_curve.csv", index=False)
    pd.DataFrame(one_rows).to_csv(sc.RESULTS / "sole_catcher.csv", index=False)

    # per-episode ledger at the reference budget
    reps = lc.representatives(dev, names, labels, lc.REFERENCE_BUDGET)
    routing = []
    for c, tag in ((dev, "development_main"), (ext, "external_8b")):
        mat, fams = lc.family_matrix(c, reps)
        cov = mat.any(0)
        ep = pd.DataFrame({
            "cohort": tag, "row": np.arange(c.n), "task": c.task, "suite": c.suite,
            "episode": c.episode, "length": c.length, "risk": c.risk,
            "physical_mode": c.mode, "n_families_firing": mat.sum(0),
            "caught_in_window": cov,
        })
        for f, det in zip(fams, reps.detector):
            ep[f"fam{f}"] = mat[fams.index(f)]
        ep.to_csv(sc.RESULTS / f"episode_ledger_{tag}.csv.gz", index=False,
                  compression="gzip")
        routing.append(routing_contrast(c, cov, rng))
    pd.concat(routing, ignore_index=True).to_csv(
        sc.RESULTS / "uncaught_routing_contrast.csv", index=False)

    (sc.RESULTS / "coverage.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(pd.DataFrame(ledger_rows).query("cohort in ['external_8b','development_main']")
          .to_string(index=False))


if __name__ == "__main__":
    main()
