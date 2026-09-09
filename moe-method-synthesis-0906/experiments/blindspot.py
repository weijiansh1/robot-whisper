"""Part 4 - the blind-spot matrix: family x physical failure mode.

The physical failure mode comes from the 36,098-trajectory replay in
``VLA_MUI_HUB/physical-failure-labels``; it names what went wrong in the world,
independently of any routing signal.  The join is exact on
(run_id, suite, task, episode) and covers every risk episode in both cohorts.

Mode is confounded with suite - libero_long fails by dropping, libero_spatial by
never establishing a grasp - so every cell is reported within suite as well as
pooled.  "Systematically missed" is defined against the family's own recall on
the same suite, tested by shuffling the mode label among that suite's risks, so
a mode is only called a blind spot when the family does worse on it than on the
rest of the suite it lives in.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ledger_core as lc
import synth_core as sc

SEED = 20260906
DRAWS = 4000


def cells(mat: np.ndarray, fams: list[int], reps: pd.DataFrame, c: sc.Cohort,
          rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    risk = c.risk
    union = mat.any(0)
    members = [(f"family_{f}", mat[i]) for i, f in enumerate(fams)]
    members.append(("union_of_families", union))
    for name, vec in members:
        for scope, smask in [("pooled", np.ones(c.n, bool))] + [
            (s, c.suite == s) for s in sorted(set(c.suite))
        ]:
            rmask = risk & smask
            if rmask.sum() == 0:
                continue
            base = float(vec[rmask].mean())
            for mode in sorted(set(c.mode[rmask])):
                mm = rmask & (c.mode == mode)
                n = int(mm.sum())
                if n == 0:
                    continue
                obs = float(vec[mm].mean())
                # shuffle the mode label among this suite's risks
                pool = np.flatnonzero(rmask)
                null = np.array([
                    vec[rng.choice(pool, size=n, replace=False)].mean()
                    for _ in range(DRAWS)
                ])
                rows.append({
                    "cohort": c.name, "method": name, "suite": scope, "mode": mode,
                    "n_risk_in_cell": n,
                    "recall_in_cell": obs,
                    "recall_on_suite": base,
                    "delta_vs_suite": obs - base,
                    "null_mean": float(null.mean()), "null_sd": float(null.std()),
                    "p_lower": float((null <= obs).mean()),
                    "p_upper": float((null >= obs).mean()),
                    "detector": reps.set_index("family").detector.get(
                        int(name.split("_")[-1]), "") if name.startswith("family_") else "",
                })
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    dev = sc.load_cohort("development_main")
    ext = sc.load_cohort("external_8b")
    names, labels = lc.load_family_labels()
    reps = lc.representatives(dev, names, labels, lc.REFERENCE_BUDGET)

    out = []
    for c in (dev, ext):
        mat, fams = lc.family_matrix(c, reps)
        out.append(cells(mat, fams, reps, c, rng))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(sc.RESULTS / "blindspot_matrix.csv", index=False)

    # Which modes does *nothing* cover, at the reference budget and at the
    # unconstrained ceiling over every admissible detector?
    meta = {"schema": "himoe.method_synthesis.blindspot.v1",
            "reference_budget": lc.REFERENCE_BUDGET, "modes": {}}
    iw_dev_all = lc.in_window(dev, names)
    fpr = (iw_dev_all & ~dev.risk).sum(1) / (~dev.risk).sum()
    adm = [names[i] for i in range(len(names)) if fpr[i] <= lc.REFERENCE_BUDGET]
    for c, tag in ((dev, "development_main"), (ext, "external_8b")):
        mat, _ = lc.family_matrix(c, reps)
        union = mat.any(0)
        ceil = lc.in_window(c, adm).any(0)
        allu = ((c.alarms >= 0) & (c.alarms < c.deadline[None, :])).any(0)
        entry = {}
        for mode in sorted(set(c.mode[c.risk])):
            mm = c.risk & (c.mode == mode)
            entry[mode] = {
                "n": int(mm.sum()),
                "recall_9_families": float(union[mm].mean()),
                "recall_ceiling_admissible": float(ceil[mm].mean()),
                "recall_unconstrained": float(allu[mm].mean()),
                "suites": {s: int(((c.suite == s) & mm).sum())
                           for s in sorted(set(c.suite[mm]))},
            }
        meta["modes"][tag] = entry
    (sc.RESULTS / "blindspot.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["modes"]["external_8b"], indent=2))


if __name__ == "__main__":
    main()
