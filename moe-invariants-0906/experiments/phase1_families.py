"""Phase 1 (LABEL-BLIND): the four structured relation families.

  F1 cross-layer sum rules      is a weighted combination over the 8 layers of
                                one functional near-constant?
  F2 front-back transport       does mixing a front and a back layer beat both
                                parts?  Tightness is the instrument, NOT AUC.
  F3 per-layer sum rules        a combination over functionals at one layer
  F4 step-axis relations        a combination over the 10 denoising steps,
                                tested against a Toeplitz (matched-lag-
                                autocorrelation) surrogate
  F5 spectrum                   effective dimension of the whole bank versus
                                the two surrogates

No outcome label anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase1_definitional import family, parse  # noqa: E402
from phase1_lib import cov_decomposition, null_corr, to_corr  # noqa: E402

BUNDLE = HERE.parent
ROOT = BUNDLE.parent
BANK = BUNDLE / "results" / "bank"
OUT = BUNDLE / "results"
SP = ROOT / "moe-flow-semantics-0906" / "results" / "step_profiles"
LAYERS = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
FRONT, BACK = LAYERS[:4], LAYERS[4:]
COHORTS = ("development_main", "external_8b")


def spectrum(R):
    w = np.linalg.eigvalsh(R)
    w = np.clip(w, 0, None)
    pr = (w.sum() ** 2) / max((w ** 2).sum(), 1e-30)
    return {"lam_min": float(w[0]), "combo_std": float(np.sqrt(w[0])),
            "participation_ratio": float(pr),
            "n_pc_90pct_var": int(np.searchsorted(
                np.cumsum(w[::-1]) / w.sum(), 0.90) + 1)}


def subset_report(R_obs, R_nc, R_ne, idx, members, R_wc=None, R_we=None):
    S = R_obs[np.ix_(idx, idx)]
    w, V = np.linalg.eigh(S)
    rec = {"members": members, "combo_std": float(np.sqrt(max(w[0], 0.0))),
           "weights": [float(x) for x in V[:, 0]],
           "combo_std_null_cell": float(np.sqrt(max(
               np.linalg.eigvalsh(R_nc[np.ix_(idx, idx)])[0], 0.0))),
           "combo_std_null_epi": float(np.sqrt(max(
               np.linalg.eigvalsh(R_ne[np.ix_(idx, idx)])[0], 0.0))),
           "mean_abs_offdiag_r": float(np.abs(S[np.triu_indices(len(idx), 1)]).mean())}
    rec["tightness_gain_vs_null_cell"] = (
        rec["combo_std"] / max(rec["combo_std_null_cell"], 1e-12))
    rec["tightness_gain_vs_null_epi"] = (
        rec["combo_std"] / max(rec["combo_std_null_epi"], 1e-12))
    if R_wc is not None:
        rec["combo_std_within_cell"] = float(np.sqrt(max(
            np.linalg.eigvalsh(R_wc[np.ix_(idx, idx)])[0], 0.0)))
    if R_we is not None:
        rec["combo_std_within_episode"] = float(np.sqrt(max(
            np.linalg.eigvalsh(R_we[np.ix_(idx, idx)])[0], 0.0)))
    return rec


# --------------------------------------------------------------------------
def main() -> None:
    c = np.load(OUT / "phase1_corr.npz", allow_pickle=True)
    names = [str(s) for s in c["var_names"]]
    R_obs, R_nc, R_ne = c["R_obs"].astype(np.float64), \
        c["R_Ncell"].astype(np.float64), c["R_Nepi"].astype(np.float64)
    R_wc, R_we = c["R_wcell"].astype(np.float64), c["R_wepi"].astype(np.float64)
    idx_of = {n: i for i, n in enumerate(names)}
    fam = {n: family(n) for n in names}

    res: dict = {}

    # ---- F5 spectrum -------------------------------------------------------
    seen, dedup = set(), []
    for i, n in enumerate(names):
        if fam[n] in seen:
            continue
        seen.add(fam[n])
        dedup.append(i)
    res["F5_spectrum"] = {
        "observed": spectrum(R_obs),
        "null_cell": spectrum(R_nc),
        "null_epi": spectrum(R_ne),
        "n_family_deduplicated_columns": len(dedup),
        "observed_family_deduplicated": spectrum(R_obs[np.ix_(dedup, dedup)]),
        "null_cell_family_deduplicated": spectrum(R_nc[np.ix_(dedup, dedup)]),
        "null_epi_family_deduplicated": spectrum(R_ne[np.ix_(dedup, dedup)]),
        "note": "384 bank columns; the observed matrix is degenerate because "
                "of the definitional duplicates catalogued in "
                "phase1_definitional.py, so this is a description of the "
                "cache, not a finding.",
    }

    # ---- F1 cross-layer, one functional at a time --------------------------
    f1 = []
    groups: dict[tuple[str, str], list[str]] = {}
    for n in names:
        cache, metric, layer, red = parse(n)
        groups.setdefault((f"{cache}.{metric}", red), []).append(n)
    for (met, red), members in sorted(groups.items()):
        members = [m for m in members if parse(m)[2] in LAYERS]
        if len(members) != 8:
            continue
        members.sort(key=lambda m: LAYERS.index(parse(m)[2]))
        idx = [idx_of[m] for m in members]
        rec = subset_report(R_obs, R_nc, R_ne, idx, members, R_wc, R_we)
        rec.update({"metric": met, "reduction": red, "axis": "layer"})
        rec["front_back_sign_split"] = bool(
            np.sign(np.array(rec["weights"][:4])).sum() *
            np.sign(np.array(rec["weights"][4:])).sum() < 0)
        f1.append(rec)
    res["F1_cross_layer"] = sorted(f1, key=lambda d: d["combo_std"])

    # ---- F2 front-back transport ------------------------------------------
    f2 = []
    for (met, red), members in sorted(groups.items()):
        members = [m for m in members if parse(m)[2] in LAYERS]
        if len(members) != 8:
            continue
        by_layer = {parse(m)[2]: m for m in members}
        for lf in FRONT:
            for lb in BACK:
                idx = [idx_of[by_layer[lf]], idx_of[by_layer[lb]]]
                rec = subset_report(R_obs, R_nc, R_ne, idx,
                                    [by_layer[lf], by_layer[lb]], R_wc, R_we)
                rec.update({"metric": met, "reduction": red,
                            "front": lf, "back": lb,
                            "r": float(R_obs[idx[0], idx[1]]),
                            "rel_resid": float(np.sqrt(max(
                                0, 1 - R_obs[idx[0], idx[1]] ** 2)))})
                f2.append(rec)
    res["F2_front_back"] = sorted(f2, key=lambda d: d["combo_std"])[:200]
    res["F2_front_back_token_entropy"] = sorted(
        [r for r in f2 if r["metric"] == "sp.token_entropy"],
        key=lambda d: d["combo_std"])

    # ---- F3 per-layer, all functionals ------------------------------------
    f3 = []
    for layer in LAYERS:
        for red in ("mean", "s0", "s9", "agg"):
            members = [n for n in names
                       if parse(n)[2] == layer and parse(n)[3] == red]
            # one column per definitional family, else the answer is forced
            seen, keep = set(), []
            for m in members:
                if fam[m] in seen:
                    continue
                seen.add(fam[m])
                keep.append(m)
            if len(keep) < 3:
                continue
            idx = [idx_of[m] for m in keep]
            rec = subset_report(R_obs, R_nc, R_ne, idx, keep, R_wc, R_we)
            rec.update({"layer": layer, "reduction": red, "axis": "functional"})
            f3.append(rec)
            # again with token_differentiation dropped: with the full entropy
            # trio present the answer is forced to 0 by the known identity
            keep2 = [m for m in keep if "token_differentiation" not in m
                     and "action_consensus" not in m]
            if len(keep2) >= 3:
                r2 = subset_report(R_obs, R_nc, R_ne,
                                   [idx_of[m] for m in keep2], keep2, R_wc, R_we)
                r2.update({"layer": layer, "reduction": red,
                           "axis": "functional_no_entropy_trio"})
                f3.append(r2)
            continue
            f3.append(rec)
    res["F3_per_layer"] = sorted(f3, key=lambda d: d["combo_std"])

    # ---- F4 step axis ------------------------------------------------------
    res["F4_step_axis"] = step_axis()

    (OUT / "phase1_families.json").write_text(json.dumps(res, indent=1))
    print("F1 tightest:", res["F1_cross_layer"][0]["metric"],
          res["F1_cross_layer"][0]["reduction"],
          "%.4f vs null %.4f/%.4f" % (res["F1_cross_layer"][0]["combo_std"],
                                      res["F1_cross_layer"][0]["combo_std_null_cell"],
                                      res["F1_cross_layer"][0]["combo_std_null_epi"]),
          file=sys.stderr)
    print("F4 tightest:", res["F4_step_axis"]["ranked"][0], file=sys.stderr)


def step_axis() -> dict:
    """Correlation across the 10 denoising steps, samples = (episode, chunk).

    Surrogate: the Toeplitz matrix with the same average correlation at each
    lag.  Any stationary series with that autocorrelation gives that lam_min,
    so a step relation only counts if it is tighter than its own Toeplitz
    projection."""
    cohort = "development_main"
    ix = np.load(SP / f"{cohort}_index.npz", allow_pickle=True)
    valid = np.asarray(ix["valid"]).copy()
    valid[:, 0] = False
    qi, ci = np.nonzero(valid)
    metric_names = [str(s) for s in ix["metric_names"]]
    task = np.asarray(ix["task_index"]).astype(np.int64)[qi]
    cell = task * (valid.shape[1]) + ci
    _, cell = np.unique(cell, return_inverse=True)

    out = []
    sp = np.load(SP / f"{cohort}_metrics.npy", mmap_mode="r")
    mob = np.load(SP / f"{cohort}_mobility.npy", mmap_mode="r")
    smob = np.load(SP / f"{cohort}_state_mobility.npy", mmap_mode="r")
    for li, layer in enumerate(LAYERS):
        block = np.asarray(sp[:, :, li, :, :])[qi, ci]           # [rows,10,8]
        series = {metric_names[m]: block[:, :, m] for m in range(8)}
        series["mobility"] = np.asarray(mob[:, :, li, :])[qi, ci]
        series["state_mobility"] = np.asarray(smob[:, :, li, :])[qi, ci]
        for mname, S in series.items():
            keep = np.isfinite(S).all(axis=0)
            Ssub = S[:, keep].astype(np.float64)
            if Ssub.shape[1] < 4:
                continue
            sd = Ssub.std(axis=0)
            if (sd <= 0).any():
                out.append({"metric": mname, "layer": layer,
                            "steps": [int(k) for k in np.nonzero(keep)[0]],
                            "degenerate_zero_variance_step": True})
                continue
            Zs = (Ssub - Ssub.mean(axis=0)) / sd
            dec = cov_decomposition(Zs, cell[: Zs.shape[0]])
            R = to_corr(dec["tot"])
            Rn = null_corr(dec["tot"], dec["bet"])
            k = R.shape[0]
            lags = np.array([np.mean(np.diag(R, d)) for d in range(k)])
            T = lags[np.abs(np.subtract.outer(np.arange(k), np.arange(k)))]
            Rw = to_corr(dec["wit"])
            tmin = float(np.linalg.eigvalsh(T)[0])
            w, V = np.linalg.eigh(R)
            out.append({
                "combo_std_within_cell": float(np.sqrt(max(
                    np.linalg.eigvalsh(Rw)[0], 0.0))),
                "r_step0_step9_within_cell": float(Rw[0, -1]),
                "toeplitz_psd": bool(tmin > 1e-8),
                "metric": mname, "layer": layer,
                "steps": [int(x) for x in np.nonzero(keep)[0]],
                "combo_std": float(np.sqrt(max(w[0], 0.0))),
                "combo_std_toeplitz_matched": float(np.sqrt(max(tmin, 0.0))),
                "combo_std_null_cell": float(np.sqrt(max(
                    np.linalg.eigvalsh(Rn)[0], 0.0))),
                "weights": [float(x) for x in V[:, 0]],
                "r_step0_step9": float(R[0, -1]),
                "lag1_mean_r": float(lags[1]),
                "participation_ratio": spectrum(R)["participation_ratio"],
            })
        del block
    ranked = sorted([o for o in out if "combo_std" in o],
                    key=lambda d: d["combo_std"])
    return {"cohort": cohort, "all": out,
            "ranked": [{"metric": r["metric"], "layer": r["layer"],
                        "combo_std": r["combo_std"],
                        "toeplitz": r["combo_std_toeplitz_matched"],
                        "toeplitz_psd": r["toeplitz_psd"],
                        "ratio_to_toeplitz": (
                            r["combo_std"] / r["combo_std_toeplitz_matched"]
                            if r["toeplitz_psd"] else None),
                        "pr": r["participation_ratio"],
                        "r_s0_s9": r["r_step0_step9"],
                        "r_s0_s9_within_cell": r["r_step0_step9_within_cell"],
                        "combo_std_within_cell": r["combo_std_within_cell"]}
                       for r in ranked]}


if __name__ == "__main__":
    main()
