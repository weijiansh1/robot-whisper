"""Two properly stratified tests on the false alarms that survive voting.

The pooled profile makes the multi-vote false alarms look long and look
concentrated on particular tasks and init states, but both could be composition
artefacts: libero_long has a cap of 52 and long episodes are common there, and
any small subset of episodes spread over 50 init states looks concentrated.  So

  * the "near miss" claim is tested against a null that resamples safe episodes
    from the *same tasks* in the same proportions,
  * the "specific init state" claim is tested the same way, on the probability
    that two randomly chosen members of the group share an init state.

Only the tests are here; the descriptive profile is in ``fp_ledger.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ledger_core as lc
import synth_core as sc

SEED = 20260906
DRAWS = 5000


def _same_init_rate(init: np.ndarray, task: np.ndarray) -> float:
    """P(two distinct members share an init state), within the same task."""
    num = den = 0
    for t in np.unique(task):
        m = task == t
        v = init[m]
        counts = np.bincount(v - v.min()) if len(v) else np.array([])
        num += int((counts * (counts - 1)).sum())
        den += int(len(v) * (len(v) - 1))
    return float(num / den) if den else float("nan")


def test_group(c: sc.Cohort, sel: np.ndarray, rng: np.random.Generator,
               label: str, pool_mask: np.ndarray | None = None) -> dict:
    """``pool_mask`` is the population the group is compared against, restricted
    to the same tasks in the same counts.  For false alarms it is the safe
    episodes; for the uncaught risks it must be the *caught* risks, since risks
    reach the cap by definition and comparing them to safe episodes on length
    would only restate that definition."""
    safe = ~c.risk if pool_mask is None else pool_mask
    tasks = c.task[sel]
    cap = np.array([sc.CAP[s] for s in c.suite])
    frac = c.length / cap
    obs_len = float(frac[sel].mean())
    obs_init = _same_init_rate(c.init_state[sel], tasks)

    pools = {t: np.flatnonzero(safe & (c.task == t)) for t in set(tasks)}
    counts = {t: min(k, len(pools[t])) for t, k in
              pd.Series(tasks).value_counts().to_dict().items()}
    counts = {t: k for t, k in counts.items() if k > 0}
    null_len = np.empty(DRAWS)
    null_init = np.empty(DRAWS)
    for d in range(DRAWS):
        take = np.concatenate([rng.choice(pools[t], size=k, replace=False)
                               for t, k in counts.items()])
        null_len[d] = frac[take].mean()
        null_init[d] = _same_init_rate(c.init_state[take], c.task[take])
    return {
        "cohort": c.name, "group": label, "n": int(sel.sum()),
        "null_pool": "safe" if pool_mask is None else "caught_risk",
        "n_resampled": int(sum(counts.values())),
        "mean_length_fraction_of_cap": obs_len,
        "null_mean_length_fraction": float(null_len.mean()),
        "null_sd_length_fraction": float(null_len.std()),
        "z_length": float((obs_len - null_len.mean()) / null_len.std())
        if null_len.std() > 0 else np.nan,
        "p_length_upper": float((null_len >= obs_len).mean()),
        "same_init_state_rate": obs_init,
        "null_same_init_state_rate": float(np.nanmean(null_init)),
        "p_init_upper": float(np.nanmean(null_init >= obs_init)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    dev = sc.load_cohort("development_main")
    ext = sc.load_cohort("external_8b")
    names, labels = lc.load_family_labels()
    reps = lc.representatives(dev, names, labels, lc.REFERENCE_BUDGET)

    rows = []
    for c in (dev, ext):
        mat, _ = lc.family_matrix(c, reps)
        votes = mat.sum(0)
        safe = ~c.risk
        for label, sel in (
            ("fp_votes1_idiosyncratic", safe & (votes == 1)),
            ("fp_votes2", safe & (votes == 2)),
            ("fp_votes3plus", safe & (votes >= 3)),
            ("any_fp", safe & (votes >= 1)),
        ):
            if sel.sum() >= 5:
                rows.append(test_group(c, sel, rng, label))
        # the same test on the risks nobody caught, for symmetry
        unc = c.risk & ~mat.any(0)
        cau = c.risk & mat.any(0)
        if unc.sum() >= 5:
            rows.append(test_group(c, unc, rng, "uncaught_risk_vs_caught_risk", cau))
    df = pd.DataFrame(rows)
    df.to_csv(sc.RESULTS / "irreducible_fp_tests.csv", index=False)
    (sc.RESULTS / "irreducible_fp.json").write_text(json.dumps(
        {"schema": "himoe.method_synthesis.irreducible_fp.v1",
         "draws": DRAWS, "reference_budget": lc.REFERENCE_BUDGET,
         "null": "safe episodes resampled from the same tasks in the same counts"},
        indent=2))
    pd.set_option("display.width", 260)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
