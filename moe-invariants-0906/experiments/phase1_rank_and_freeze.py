"""Phase 1 (LABEL-BLIND): assemble the ranked candidate list, measure
coefficient stability, and FREEZE.

Coefficient stability is the third instrument the protocol asks for: a relation
whose fitted coefficients move between tasks, suites or cohorts is not a law,
however small its pooled residual.  Coefficients are reported in RAW units so
that they are comparable across strata (standardised coefficients absorb the
stratum's own variance and would hide instability).

The output `phase1_frozen_candidates.json` plus its sha256 in
`phase1_freeze.json` is the pre-registration for phase 2.  Nothing downstream
may add a candidate.

No outcome label anywhere.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase1_definitional import classify  # noqa: E402

BUNDLE = HERE.parent
BANK = BUNDLE / "results" / "bank"
OUT = BUNDLE / "results"
COHORTS = ("development_main", "development_extra", "external_8b")
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
MIN_ROWS = 400
EPS = 1e-12


def load_bank(cohort):
    f = np.load(BANK / f"{cohort}_bank.npz", allow_pickle=True)
    return {k: f[k] for k in f.files}


def ols(Xd, y):
    A = np.c_[Xd, np.ones(len(y))]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ b
    return b[:-1], float(b[-1]), float(resid.std() / max(y.std(), EPS))


def apply_coefs(Xd, y, b):
    """Residual with transported slopes; the intercept is refit because a level
    offset is not a coefficient."""
    r = y - Xd @ b
    return float((r - r.mean()).std() / max(y.std(), EPS))


def main() -> None:
    rel = json.loads((OUT / "phase1_relations.json").read_text())
    fam = json.loads((OUT / "phase1_families.json").read_text())
    nlin = json.loads((OUT / "phase1_nonlinear.json").read_text())

    cands: list[dict] = []
    seen = set()

    def add(kind, target, preds, source, extra=None):
        key = (target, tuple(sorted(preds)))
        if key in seen:
            return
        seen.add(key)
        cands.append({"id": f"C{len(cands):03d}", "kind": kind,
                      "target": target, "preds": list(preds),
                      "category": classify([target] + list(preds)),
                      "source": source, **(extra or {})})

    for key, n, src in (("pairs_empirical_cross_metric", 40, "pair_cross"),
                        ("pairs_empirical_same_metric", 40, "pair_same"),
                        ("triples_empirical_cross_metric", 40, "triple_cross"),
                        ("triples_empirical_same_metric", 40, "triple_same")):
        for r in rel.get(key, [])[:n]:
            add(r["kind"], r["target"], r["preds"], src,
                {"rel_resid_pooled": r["rel_resid"],
                 "null_cell": r["null_cell"], "null_epi": r["null_epi"],
                 "rel_resid_within_cell": r.get("rel_resid_within_cell"),
                 "rel_resid_within_epi": r.get("rel_resid_within_epi")})
    # positive controls: the known identity at every layer
    for ly in ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15"):
        for red in ("s0", "s9"):
            add("triple", f"sp.load_entropy@{ly}[{red}]",
                [f"sp.token_entropy@{ly}[{red}]",
                 f"sp.token_differentiation@{ly}[{red}]"], "positive_control")
    # tightest structured combinations, expressed as OLS relations
    f3 = [r for r in fam["F3_per_layer"]
          if r["axis"] == "functional_no_entropy_trio"]
    for rec in fam["F1_cross_layer"][:20] + fam["F2_front_back"][:20] + f3[:10]:
        m, w = rec["members"], np.array(rec["weights"])
        t = int(np.argmax(np.abs(w)))
        add("combo", m[t], [x for k, x in enumerate(m) if k != t],
            "family", {"combo_std": rec["combo_std"],
                       "combo_std_null_cell": rec["combo_std_null_cell"],
                       "combo_std_null_epi": rec["combo_std_null_epi"]})
    for r in nlin.get("top_nondefinitional", [])[:20]:
        add("pair", r["target"], [r["predictor"]], "nonlinear",
            {"rel_resid_nl": r["rel_resid_nl"],
             "null_cell_nl": r["null_cell"], "null_epi_nl": r["null_epi"]})

    used = sorted({v for c in cands for v in [c["target"]] + c["preds"]})
    print(f"{len(cands)} candidates over {len(used)} variables", file=sys.stderr)

    # ---- per-stratum fits --------------------------------------------------
    data = {}
    for coh in COHORTS:
        b = load_bank(coh)
        names = [str(s) for s in b["var_names"]]
        col = {n: i for i, n in enumerate(names)}
        X = b["X"][:, [col[u] for u in used]].astype(np.float64)
        tn = [str(s) for s in b["task_names"]]
        data[coh] = {"X": X, "task": b["row_task"].astype(int),
                     "suite": b["row_suite"].astype(int), "task_names": tn}
    uidx = {u: i for i, u in enumerate(used)}

    for c in cands:
        ti, pi = uidx[c["target"]], [uidx[p] for p in c["preds"]]
        Xd = data["development_main"]["X"]
        b0, i0, rr0 = ols(Xd[:, pi], Xd[:, ti])
        c["coefs_raw_dev_main"] = [float(x) for x in b0]
        c["intercept_dev_main"] = i0
        c["rel_resid_dev_main"] = rr0
        strata = {}
        for coh in COHORTS:
            D = data[coh]
            Xc = D["X"]
            strata[f"{coh}|all"] = ols(Xc[:, pi], Xc[:, ti])[0::2]
            strata[f"{coh}|transported"] = (b0, apply_coefs(Xc[:, pi], Xc[:, ti], b0))
            for s, sname in enumerate(SUITES):
                m = D["suite"] == s
                if m.sum() < MIN_ROWS:
                    continue
                strata[f"{coh}|suite:{sname}"] = ols(Xc[m][:, pi], Xc[m][:, ti])[0::2]
            for t in np.unique(D["task"]):
                m = D["task"] == t
                if m.sum() < MIN_ROWS:
                    continue
                strata[f"{coh}|task:{D['task_names'][t]}"] = \
                    ols(Xc[m][:, pi], Xc[m][:, ti])[0::2]
        c["strata"] = {k: {"coefs": [float(x) for x in v[0]],
                           "rel_resid": float(v[1])} for k, v in strata.items()}
        task_keys = [k for k in strata if "|task:" in k]
        suite_keys = [k for k in strata if "|suite:" in k]
        coh_keys = [f"{x}|all" for x in COHORTS]
        c["stability"] = {
            "coef_cv_across_tasks": cv([strata[k][0] for k in task_keys]),
            "coef_cv_across_suites": cv([strata[k][0] for k in suite_keys]),
            "coef_cv_across_cohorts": cv([strata[k][0] for k in coh_keys]),
            "max_angle_deg_across_tasks": angle([strata[k][0] for k in task_keys]),
            "max_angle_deg_across_cohorts": angle([strata[k][0] for k in coh_keys]),
            "rel_resid_transported_external_8b":
                float(strata["external_8b|transported"][1]),
            "rel_resid_transported_development_extra":
                float(strata["development_extra|transported"][1]),
            "worst_rel_resid_any_task":
                float(max(strata[k][1] for k in task_keys)) if task_keys else None,
            "n_task_strata": len(task_keys),
        }

    cands.sort(key=lambda c: (c["stability"]["rel_resid_transported_external_8b"],
                              c["rel_resid_dev_main"]))
    payload = {
        "schema": "himoe.invariants.phase1_frozen.v1",
        "note": "LABEL-BLIND. Ranked by out-of-cohort transported relative "
                "residual, then by pooled residual. No outcome label was "
                "loaded to produce this list.",
        "n_candidates": len(cands),
        "variables_used": used,
        "candidates": cands,
    }
    p = OUT / "phase1_frozen_candidates.json"
    p.write_text(json.dumps(payload, indent=1))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "phase1_freeze.json").write_text(json.dumps({
        "frozen_file": p.name, "sha256": h, "n_candidates": len(cands),
        "phase1_inputs": {
            n: hashlib.sha256((OUT / n).read_bytes()).hexdigest()
            for n in ("phase1_relations.json", "phase1_families.json",
                      "phase1_nonlinear.json")},
    }, indent=1))
    print("frozen sha256", h, file=sys.stderr)
    for c in cands[:15]:
        print("  %-28s trans_ext %.4f  pooled %.4f  cv_task %.3f  %s ~ %s"
              % (c["category"], c["stability"]["rel_resid_transported_external_8b"],
                 c["rel_resid_dev_main"],
                 c["stability"]["coef_cv_across_tasks"],
                 c["target"], "+".join(c["preds"])), file=sys.stderr)


def cv(vecs):
    A = np.array(vecs, dtype=float)
    if A.size == 0 or A.shape[0] < 2:
        return None
    m = np.abs(A.mean(axis=0))
    s = A.std(axis=0)
    return float(np.max(s / np.maximum(m, EPS)))


def angle(vecs):
    A = np.array(vecs, dtype=float)
    if A.shape[0] < 2:
        return None
    U = np.c_[np.ones(len(A)), -A]
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    G = np.clip(U @ U.T, -1, 1)
    return float(np.degrees(np.arccos(G.min())))


if __name__ == "__main__":
    main()
