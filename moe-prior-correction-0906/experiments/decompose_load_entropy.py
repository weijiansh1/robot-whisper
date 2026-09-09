"""What is expert_load_effective_rank actually measuring?

    expert_load_effective_rank = exp(load_entropy at the final denoising step) / 32

and the cached per-step profiles carry the exact additive decomposition:

    load_entropy          = H( mean_t p_t )              aggregate load concentration
    token_entropy         = mean_t H( p_t )              per-token router indecision
    token_differentiation = load_entropy - token_entropy = I(token; expert)

So the detector's quantity is A + B where A is "each action token's router is
undecided" and B is "the ten action tokens route to different experts".  These
are different mechanisms and only one of them needs to be carrying the signal.

Estimator throughout: within task, at a fixed chunk, over episodes still
running at that chunk.  Every episode in a stratum shares a task and an elapsed
length, so neither task identity nor elapsed time can contribute - which is why
length scores exactly 0.5 here, asserted as a check.
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
# 65% of each suite's horizon cap - the operational target window.
CAPS = {"libero_goal": 30, "libero_long": 52, "libero_object": 28, "libero_spatial": 22}
MIN_PER_ARM = 3


def stratified_auc(score: np.ndarray, keep: np.ndarray, task: np.ndarray,
                   risk: np.ndarray) -> tuple[float, int, int]:
    """Mann-Whitney AUC pooled over tasks with n+ * n- weights."""
    num = den = 0.0
    ntask = 0
    keep = keep & np.isfinite(score)
    for name in np.unique(task[keep]):
        sel = keep & (task == name)
        pos = score[sel & risk]
        neg = score[sel & ~risk]
        if len(pos) < MIN_PER_ARM or len(neg) < MIN_PER_ARM:
            continue
        ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
        num += ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
        den += len(pos) * len(neg)
        ntask += 1
    return (num / den if den else np.nan), int(den), ntask


def main() -> None:
    rows = []
    for cohort_name in ("external_8b", "development_main"):
        coh = cohort_frame(cohort_name)
        risk, task, suite, length = (coh["risk"], coh["task"], coh["suite"],
                                     coh["length"])
        metrics = np.load(STEPS / f"{cohort_name}_metrics.npy", mmap_mode="r")
        mobility = np.load(STEPS / f"{cohort_name}_mobility.npy", mmap_mode="r")

        for s in sorted(CAPS):
            in_suite = suite == s
            window = int(0.65 * CAPS[s])
            # Probe the window at quarter points plus the last in-window chunk.
            chunks = sorted({max(4, window // 4), window // 2,
                             3 * window // 4, window - 1})
            for q in chunks:
                keep = in_suite & (length > q)
                if (keep & risk).sum() < MIN_PER_ARM:
                    continue

                # Elapsed time is constant inside the stratum, so a constant
                # must score exactly 0.5 - the estimator's null.
                auc_null, _, _ = stratified_auc(
                    np.ones(len(risk)), keep, task, risk)
                assert not np.isfinite(auc_null) or abs(auc_null - 0.5) < 1e-12, auc_null

                # *Total* length is not constant here: survivors at chunk q
                # still differ in when they eventually stop, and risk is
                # exactly "stopped at the cap".  So length is a hindsight
                # restatement of the label, scoring near 1.  Recorded as the
                # ceiling a non-baseline can reach, never as a baseline.
                auc_len, den_len, nt_len = stratified_auc(
                    length.astype(float), keep, task, risk)
                rows.append({
                    "cohort": cohort_name, "suite": s, "chunk": q,
                    "window_end": window, "layer": "-", "step": -1,
                    "quantity": "_hindsight_length", "auc": auc_len,
                    "pairs": den_len, "tasks": nt_len,
                })

                plane = np.asarray(metrics[:, q])          # [n, 8 layers, 10 steps, 8]
                mob = np.asarray(mobility[:, q])           # [n, 8 layers, 10 steps]
                for li, layer in enumerate(LAYERS):
                    for step in (0, 3, 6, 9):
                        for mi, metric in enumerate(METRICS):
                            auc, den, nt = stratified_auc(
                                plane[:, li, step, mi].astype(float), keep, task, risk)
                            rows.append({
                                "cohort": cohort_name, "suite": s, "chunk": q,
                                "window_end": window, "layer": layer, "step": step,
                                "quantity": metric, "auc": auc,
                                "pairs": den, "tasks": nt,
                            })
                        auc, den, nt = stratified_auc(
                            mob[:, li, step].astype(float), keep, task, risk)
                        rows.append({
                            "cohort": cohort_name, "suite": s, "chunk": q,
                            "window_end": window, "layer": layer, "step": step,
                            "quantity": "mobility", "auc": auc,
                            "pairs": den, "tasks": nt,
                        })

    df = pd.DataFrame(rows).dropna(subset=["auc"])
    df["signal"] = (df.auc - 0.5).abs()
    df.to_csv(OUT / "load_entropy_decomposition.csv", index=False)

    df = df[~df.quantity.str.startswith("_")].copy() if False else df
    trio = ("load_entropy", "token_entropy", "token_differentiation")
    print("Within task, at a fixed chunk, over survivors.  AUC 0.5 = nothing")
    print("beyond 'still running'.  AUC < 0.5 means LOW values predict risk.\n")
    print("=== the decomposition: A+B vs A vs B, best layer/step per cell ===")
    for cohort_name in ("external_8b", "development_main"):
        print(f"\n-- {cohort_name} --")
        table = []
        sub = df[(df.cohort == cohort_name) & df.quantity.isin(trio + ("mobility",))]
        for s in sorted(CAPS):
            for q in sorted(sub[sub.suite == s].chunk.unique()):
                cell = sub[(sub.suite == s) & (sub.chunk == q)]
                row = {"suite": s.replace("libero_", ""), "chunk": q,
                       "of": int(cell.window_end.iloc[0]) - 1}
                for name in trio + ("mobility",):
                    g = cell[cell.quantity == name]
                    if not len(g):
                        row[name] = "-"
                        continue
                    best = g.loc[g.signal.idxmax()]
                    tag = "mob" if name == "mobility" else ""
                    row[name] = "%.3f %s/s%d" % (best.auc, best.layer, best.step)
                table.append(row)
        print(pd.DataFrame(table).to_string(index=False))

    print("\n=== which half of the decomposition carries it? ===")
    piv = (df[df.quantity.isin(trio)]
           .groupby(["cohort", "suite", "quantity"]).signal.max().unstack())
    piv["B beats A"] = piv.token_differentiation > piv.token_entropy
    print(piv.round(4).to_string())

    print("\n=== strongest cells overall (any quantity) ===")
    cols = ["cohort", "suite", "chunk", "quantity", "layer", "step", "auc",
            "pairs", "tasks"]
    print(df.nlargest(20, "signal")[cols].to_string(index=False))

    summary = {
        "n_cells": int(len(df)),
        "max_signal": float(df.signal.max()),
        "median_signal": float(df.signal.median()),
        "best_by_quantity": {
            q: float(df[df.quantity == q].signal.max()) for q in df.quantity.unique()
        },
    }
    (OUT / "decomposition_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nbest |AUC-0.5| by quantity:")
    for q, v in sorted(summary["best_by_quantity"].items(), key=lambda kv: -kv[1]):
        print("   %-28s %.4f" % (q, v))


if __name__ == "__main__":
    main()
