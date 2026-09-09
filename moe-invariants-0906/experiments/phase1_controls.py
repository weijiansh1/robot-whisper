"""Phase 1 (LABEL-BLIND): verify every definitional identity at raw precision.

Two jobs:
  1. calibrate the tightness instrument - the identities the extractor code
     makes exact must come out at ~0, otherwise the metric is broken;
  2. check the ONE falsifiable prediction the definitions make that nobody has
     tested: the second-order expansion says
         token_differentiation = (9/5) (1 - action_consensus)
     with slope 9/5 = 1.8 exactly.  A measured slope away from 1.8 would mean
     the near-uniform expansion does not control this cache.

No outcome label anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
ROOT = BUNDLE.parent
OUT = BUNDLE / "results"
SP = ROOT / "moe-flow-semantics-0906" / "results" / "step_profiles"
CH = ROOT / "moe-unused-channels-0906" / "results" / "channels"
LG = ROOT / "moe-hb-front-back-0905" / "results" / "layer_graphs"
LAYERS = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")


def rel(res: np.ndarray, tgt: np.ndarray) -> float:
    return float(res.std() / max(tgt.std(), 1e-30))


def main() -> None:
    cohort = sys.argv[1] if len(sys.argv) > 1 else "development_main"
    ix = np.load(SP / f"{cohort}_index.npz", allow_pickle=True)
    valid = np.asarray(ix["valid"]).copy()
    valid[:, 0] = False
    qi, ci = np.nonzero(valid)
    mn = [str(s) for s in ix["metric_names"]]
    sp = np.load(SP / f"{cohort}_metrics.npy", mmap_mode="r")
    lgf = np.load(LG / f"{cohort}.npz", allow_pickle=True)
    lgm, lgn = lgf["metrics"], [str(s) for s in lgf["metric_names"]]
    chq = np.load(CH / f"{cohort}_quantities.npy", mmap_mode="r")
    chn = [str(s) for s in np.load(CH / f"{cohort}_index.npz",
                                   allow_pickle=True)["quantity_names"]]
    smob = np.load(SP / f"{cohort}_state_mobility.npy", mmap_mode="r")

    out: dict = {"cohort": cohort, "n_rows": int(qi.size)}
    per_layer = {}
    for li, layer in enumerate(LAYERS):
        b = np.asarray(sp[:, :, li, :, :])[qi, ci].astype(np.float64)  # [rows,10,8]
        g = np.asarray(lgm[:, :, li, :])[qi, ci].astype(np.float64)
        q = np.asarray(chq[:, :, li, :])[qi, ci].astype(np.float64)
        sm = np.asarray(smob[:, :, li, :])[qi, ci].astype(np.float64)
        col = {n: b[:, :, k] for k, n in enumerate(mn)}
        gcol = {n: g[:, k] for k, n in enumerate(lgn)}
        qcol = {n: q[:, k] for k, n in enumerate(chn)}
        rec = {}

        # C1 entropy identity, per step and step-averaged
        r = col["load_entropy"] - col["token_entropy"] - col["token_differentiation"]
        rec["C1_entropy_identity"] = {
            "max_abs_residual": float(np.abs(r).max()),
            "rel_resid_per_step_max": max(
                rel(r[:, s], col["load_entropy"][:, s]) for s in range(10)),
            "rel_resid_step_mean": rel(r.mean(1), col["load_entropy"].mean(1)),
            "load_entropy_std": float(col["load_entropy"].mean(1).std()),
        }
        # C2 layer_graphs geometry == step_profiles at step 9
        rec["C2_layergraph_is_step9"] = {
            m: {"max_abs_delta": float(np.abs(gcol[m] - col[m][:, 9]).max()),
                "rel_resid": rel(gcol[m] - col[m][:, 9], col[m][:, 9])}
            for m in ("action_consensus", "state_action_alignment",
                      "conditional_energy", "conditional_effective_rank")}
        # C3 flow_path == 9 * mean_s flow_speed (steps 1..9)
        fp = gcol["flow_path"]
        fs = np.nanmean(col["flow_speed"], axis=1)
        rec["C3_flow_path"] = {"max_abs_delta": float(np.abs(fp - 9 * fs).max()),
                               "rel_resid": rel(fp - 9 * fs, fp),
                               "fitted_slope": float(np.polyfit(fs, fp, 1)[0])}
        # C4 expert_load_effective_rank == exp(load_entropy@s9)/32
        er = gcol["expert_load_effective_rank"]
        pred = np.exp(col["load_entropy"][:, 9]) / 32.0
        rec["C4_eff_rank_is_exp_entropy"] = {
            "max_abs_delta": float(np.abs(er - pred).max()),
            "rel_resid": rel(er - pred, er)}
        # C5 hb_entropy_action == mean_s token_entropy
        he = qcol["hb_entropy_action"]
        rec["C5_hb_entropy"] = {
            "max_abs_delta_vs_token_entropy":
                float(np.abs(he - col["token_entropy"].mean(1)).max()),
            "rel_resid_vs_token_entropy":
                rel(he - col["token_entropy"].mean(1), he),
            "max_abs_delta_vs_load_entropy":
                float(np.abs(he - col["load_entropy"].mean(1)).max())}
        # C6 conditional_energy = 1 - sa^2 - Var_t(s_t): envelope, not identity
        ce, sa = col["conditional_energy"], col["state_action_alignment"]
        gap = (1 - sa ** 2) - ce
        rec["C6_cond_energy_envelope"] = {
            "min_gap": float(gap.min()), "max_gap": float(gap.max()),
            "mean_gap": float(gap.mean()),
            "rel_resid_of_1_minus_sa2": rel(gap, ce),
            "envelope_never_violated": bool((gap >= -1e-6).all())}
        # C7 the falsifiable Taylor slope: td = (9/5)(1 - ac)
        td, ac = col["token_differentiation"], col["action_consensus"]
        x, y = (1 - ac).ravel(), td.ravel()
        slope, icpt = np.polyfit(x, y, 1)
        rec["C7_taylor_slope"] = {
            "predicted_slope": 1.8, "fitted_slope": float(slope),
            "fitted_intercept": float(icpt),
            "rel_resid_at_predicted_slope": rel(y - 1.8 * x, y),
            "rel_resid_at_fitted_slope": rel(y - slope * x - icpt, y),
            "r": float(np.corrcoef(x, y)[0, 1])}
        # C8 state_mobility is step invariant
        spread = sm.max(axis=1) - sm.min(axis=1)
        rec["C8_state_mobility_step_invariant"] = {
            "max_spread": float(spread.max()),
            "mean_spread": float(spread.mean()),
            "mean_level": float(sm.mean()),
            "spread_over_level": float(spread.mean() / sm.mean()),
            "r_step0_step9": float(np.corrcoef(sm[:, 0], sm[:, 9])[0, 1])}
        # reference: mobility (action tokens) is NOT step invariant
        per_layer[layer] = rec
        del b, g, q, sm
    out["per_layer"] = per_layer
    out["summary"] = {
        "C1_max_abs_residual_any_layer":
            max(v["C1_entropy_identity"]["max_abs_residual"] for v in per_layer.values()),
        "C1_worst_rel_resid_step_mean":
            max(v["C1_entropy_identity"]["rel_resid_step_mean"] for v in per_layer.values()),
        "C3_worst_rel_resid":
            max(v["C3_flow_path"]["rel_resid"] for v in per_layer.values()),
        "C7_fitted_slope_range":
            [min(v["C7_taylor_slope"]["fitted_slope"] for v in per_layer.values()),
             max(v["C7_taylor_slope"]["fitted_slope"] for v in per_layer.values())],
        "C7_worst_rel_resid_at_fitted":
            max(v["C7_taylor_slope"]["rel_resid_at_fitted_slope"] for v in per_layer.values()),
        "C8_worst_spread_over_level":
            max(v["C8_state_mobility_step_invariant"]["spread_over_level"]
                for v in per_layer.values()),
    }
    (OUT / f"phase1_controls_{cohort}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
