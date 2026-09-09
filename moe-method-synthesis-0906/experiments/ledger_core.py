"""Family representatives selected on development, scored on external.

Selection never touches an external outcome.  For each family and each false
alarm budget B, the representative is the member with the most development
in-window true alarms whose development in-window FPR is <= B (ties broken to
the smaller development FP count, then by name so the choice is deterministic).
A family with no member inside the budget is simply unavailable at that budget.

The budget ladder is fixed in advance and includes 0.0543, the realised FPR of
the honest arm that is currently standing, so the accounting can be read at the
operating point that matters as well as around it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import synth_core as sc

BUDGETS = (0.002, 0.005, 0.01, 0.02, 0.0543, 0.10, 0.20, 1.00)

# Per-detector budget.  At 0.01 the union of the nine family representatives
# lands on the operating point of the standing honest arm (external union FPR
# 0.0579 against its 0.0543, union recall 0.785 against its 0.798), so the
# ledgers are read there.  0.0543 is the honest arm's own *union* FPR and is
# kept in the ladder for reference, but as a per-detector budget it produces a
# much looser union.
REFERENCE_BUDGET = 0.01
HONEST_ARM = {"in_window_recall": 0.798, "fp": 817, "fpr": 0.0543, "precision": 0.355}


def load_family_labels() -> tuple[list[str], np.ndarray]:
    z = np.load(sc.RESULTS / "family_labels.npz", allow_pickle=False)
    return [str(x) for x in z["names_dev"]], z["labels_dev"]


def in_window(c: sc.Cohort, order: list[str]) -> np.ndarray:
    idx = [c.detectors.index(k) for k in order]
    a = c.alarms[idx]
    return (a >= 0) & (a < c.deadline[None, :])


def representatives(dev: sc.Cohort, names: list[str], labels: np.ndarray,
                    budget: float) -> pd.DataFrame:
    iw = in_window(dev, names)
    tp = (iw & dev.risk).sum(1)
    fp = (iw & ~dev.risk).sum(1)
    fpr = fp / (~dev.risk).sum()
    rows = []
    for fam in np.unique(labels):
        idx = np.flatnonzero((labels == fam) & (fpr <= budget))
        if len(idx) == 0:
            continue
        order = sorted(idx, key=lambda i: (-tp[i], fp[i], names[i]))
        best = order[0]
        rows.append({
            "budget": budget, "family": int(fam), "detector": names[best],
            "dev_tp": int(tp[best]), "dev_fp": int(fp[best]),
            "dev_recall": float(tp[best] / dev.risk.sum()),
            "dev_fpr": float(fpr[best]),
            "n_members_in_budget": int(len(idx)),
        })
    return pd.DataFrame(rows)


def family_matrix(c: sc.Cohort, reps: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """(n_families, n_episodes) in-window fire matrix for one cohort."""
    order = reps.detector.tolist()
    return in_window(c, order), reps.family.tolist()
