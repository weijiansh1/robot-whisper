"""Is the load-entropy signal a failure precursor, or the absence of a
completion signature - and does it survive honest selection?

Two things must be ruled out before the fixed-chunk AUCs mean anything.

1. SELECTION.  Reporting max |AUC-0.5| over 8 layers x 4 denoising steps is a
   choice among 32.  Here the (layer, step) is chosen on development and the
   number reported is the one measured on external, which never saw the choice.

2. THE COMPLETION CONFOUND.  Among episodes still running at chunk q, the
   non-risk arm contains episodes that are about to finish.  Routing that
   differs because a task is *completing* is not an early warning.  The guard
   is to require the non-risk arm to keep running at least R more chunks: if
   the separation survives comparing "will time out" against "will finish, but
   not for a while", it is not the completion signature.

The decomposition is evaluated at a single fixed (layer, step) so that
A + B = A + B actually holds cell by cell.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

from recompute_task_matched_lift import cohort_frame  # noqa: E402

STEPS = ROOT / "moe-flow-semantics-0906/results/step_profiles"
METRICS = ("token_entropy", "load_entropy", "token_differentiation",
           "action_consensus", "state_action_alignment", "conditional_energy",
           "conditional_effective_rank", "flow_speed")
LAYERS = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
CAPS = {"libero_goal": 30, "libero_long": 52, "libero_object": 28, "libero_spatial": 22}
TRIO = ("load_entropy", "token_entropy", "token_differentiation")
MIN_PER_ARM = 3
LEADS = (0, 3, 5)


def stratified_auc(score, keep, task, risk, extra_neg=None):
    """Within-task Mann-Whitney AUC, pooled with n+ * n- weights.

    ``extra_neg`` further restricts the negative arm without touching the
    positive arm, which is how the completion confound is held out.
    """
    num = den = 0.0
    ntask = 0
    keep = keep & np.isfinite(score)
    pos_all = keep & risk
    neg_all = keep & ~risk if extra_neg is None else keep & ~risk & extra_neg
    for name in np.unique(task[keep]):
        at = task == name
        pos, neg = score[pos_all & at], score[neg_all & at]
        if len(pos) < MIN_PER_ARM or len(neg) < MIN_PER_ARM:
            continue
        ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
        num += ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
        den += len(pos) * len(neg)
        ntask += 1
    return (num / den if den else np.nan), int(den), ntask


def load(cohort_name):
    coh = cohort_frame(cohort_name)
    coh["metrics"] = np.load(STEPS / f"{cohort_name}_metrics.npy", mmap_mode="r")
    coh["mobility"] = np.load(STEPS / f"{cohort_name}_mobility.npy", mmap_mode="r")
    return coh


def series(coh, q, li, step, mi):
    if mi is None:
        return np.asarray(coh["mobility"][:, q, li, step]).astype(float)
    return np.asarray(coh["metrics"][:, q, li, step, mi]).astype(float)


def main() -> None:
    dev, ext = load("development_main"), load("external_8b")
    rows, chosen = [], {}

    for suite in sorted(CAPS):
        window = int(0.65 * CAPS[suite])
        # The last chunk fully inside the operational window.
        q = window - 1
        for quantity in TRIO + ("mobility",):
            mi = None if quantity == "mobility" else METRICS.index(quantity)

            # --- selection, on development only ---
            best, best_sig = None, -1.0
            keep_dev = (dev["suite"] == suite) & (dev["length"] > q)
            for li, layer in enumerate(LAYERS):
                for step in (0, 3, 6, 9):
                    auc, den, nt = stratified_auc(
                        series(dev, q, li, step, mi), keep_dev,
                        dev["task"], dev["risk"])
                    if np.isfinite(auc) and abs(auc - 0.5) > best_sig:
                        best, best_sig = (li, layer, step, auc, den, nt), abs(auc - 0.5)
            if best is None:
                continue
            li, layer, step, dev_auc, dev_den, dev_nt = best
            chosen[(suite, quantity)] = (layer, step)

            # --- evaluation, on external, at every lead ---
            keep_ext = (ext["suite"] == suite) & (ext["length"] > q)
            for lead in LEADS:
                extra = ext["length"] > q + lead if lead else None
                auc, den, nt = stratified_auc(
                    series(ext, q, li, step, mi), keep_ext,
                    ext["task"], ext["risk"], extra_neg=extra)
                n_neg = int(
                    (keep_ext & ~ext["risk"] & (extra if extra is not None else True)).sum()
                )
                rows.append({
                    "suite": suite, "chunk": q, "quantity": quantity,
                    "layer": layer, "step": step, "lead": lead,
                    "dev_auc": dev_auc, "ext_auc": auc,
                    "ext_pairs": den, "ext_tasks": nt, "ext_neg": n_neg,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "essence_heldout.csv", index=False)

    print("(layer, step) chosen on development; AUC reported on external.")
    print("AUC < 0.5 means LOW values predict eventual timeout.\n")
    print("=== held-out AUC, and what survives removing 'about to finish' ===")
    for suite in sorted(CAPS):
        sub = df[df.suite == suite]
        if not len(sub):
            continue
        print(f"\n-- {suite}, chunk {int(sub.chunk.iloc[0])} "
              f"(last chunk inside the 65% window) --")
        table = []
        for quantity in TRIO + ("mobility",):
            g = sub[sub.quantity == quantity]
            if not len(g):
                continue
            row = {"quantity": quantity,
                   "layer/step": f"{g.layer.iloc[0]}/s{g.step.iloc[0]}",
                   "dev": "%.3f" % g.dev_auc.iloc[0]}
            for lead in LEADS:
                h = g[g.lead == lead]
                row[f"ext lead+{lead}"] = (
                    "%.3f (n-=%d)" % (h.ext_auc.iloc[0], h.ext_neg.iloc[0])
                    if len(h) and np.isfinite(h.ext_auc.iloc[0]) else "-"
                )
            table.append(row)
        print(pd.DataFrame(table).to_string(index=False))

    print("\n=== does A + B = load_entropy hold, at one fixed layer/step? ===")
    print("(chosen independently per quantity, so the triple only coincides")
    print(" when the same cell wins - listed to show where it does)")
    for suite in sorted(CAPS):
        cells = {q: chosen.get((suite, q)) for q in TRIO}
        same = len(set(cells.values())) == 1
        print("  %-15s load=%s  token=%s  diff=%s  %s"
              % (suite, cells["load_entropy"], cells["token_entropy"],
                 cells["token_differentiation"],
                 "SAME CELL" if same else ""))

    keep = df[df.lead == 5]
    survived = keep[(keep.ext_auc - 0.5).abs() > 0.10]
    print(f"\n{len(survived)}/{len(keep)} suite x quantity cells keep "
          f"|AUC-0.5| > 0.10 after removing episodes finishing within 5 chunks")
    summary = {
        "chosen": {f"{k[0]}|{k[1]}": f"{v[0]}/s{v[1]}" for k, v in chosen.items()},
        "n_survived_lead5": int(len(survived)),
        "n_cells": int(len(keep)),
    }
    (OUT / "essence_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
