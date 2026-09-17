"""Honest checks on the topology grid: per-scale stability, permutation null, cross-dataset transfer,
leave-one-task-out selection.  Reads _grid/features.pkl produced by topology_grid.py."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import topology_grid as tg  # noqa: E402

MIN_CLASS = 15
TOPK = 20
PERMUTATIONS = 30


def oriented(a):
    return 0.5 + np.abs(a - 0.5)


_PERM_CONTEXT = None


def _one_permutation(seed):
    X, labels, task, family, fams = _PERM_CONTEXT
    rng = np.random.default_rng(seed)
    perm = labels.copy()
    for t in set(task):
        idx = np.flatnonzero(task == t)
        perm[idx] = rng.permutation(perm[idx])
    a, n1, n0 = tg.auroc_columns(X, perm)
    sa, _ = tg.stratified_columns(X, perm, list(task))
    ok = (n1 >= MIN_CLASS) & (n0 >= MIN_CLASS)
    sa = np.where(ok, sa, np.nan)
    return {fam: float(np.nanmax(oriented(sa[family == fam]))) for fam in fams}


def main():
    grid = Path(sys.argv[1]) if len(sys.argv) > 1 else tg.ROOT.parent / "moe-capture" / "topo-20260916" / "_grid"
    results = pickle.loads((grid / "features.pkl").read_bytes())
    X, meta = tg.flatten(results)
    labels = np.array([r["meta"]["success"] for r in results])
    dataset = np.array([r["meta"]["dataset"] for r in results])
    task = np.array(["%s/%s" % (r["meta"]["dataset"], r["meta"]["tag"][:6]) for r in results])
    base_task = np.array([r["meta"]["tag"][:6] for r in results])
    anchor = np.array([m[6] for m in meta])
    feature = np.array([m[7] for m in meta])
    scale = np.array([m[4] if m[4] != "" else np.nan for m in meta], float)
    family = np.where(np.isin(feature, tg.REGION_FEATURES), "region",
                      np.where(np.isin(feature, tg.PERSIST_FEATURES), "persistence",
                               np.where(feature == "v82_alarm", "v82", np.where(np.isin(feature, tg.V82_FEATURES), "v82", "simple"))))
    out = {}

    def scores(rows):
        a, n1, n0 = tg.auroc_columns(X[rows], labels[rows])
        sa, ng = tg.stratified_columns(X[rows], labels[rows], list(task[rows]))
        ok = (n1 >= MIN_CLASS) & (n0 >= MIN_CLASS)
        return np.where(ok, a, np.nan), np.where(ok, sa, np.nan)

    rows_all = np.arange(len(results))
    a_all, t_all = scores(rows_all)
    rows_l = np.flatnonzero(dataset == "topo-libero10")
    rows_p = np.flatnonzero(dataset == "topo-plus")
    a_l, t_l = scores(rows_l)
    a_p, t_p = scores(rows_p)

    # 1. per-scale stability of the best region settings (selected by task AUROC at scale 1.0, pooled data)
    region = family == "region"
    index = {}
    for j in np.flatnonzero(region):
        m, h, ref, q, s, c, anc, f = meta[j]
        index[(m, h, ref, q, c, anc, f, s)] = j
    keyed = {k[:7]: j for k, j in index.items() if k[7] == 1.0 and np.isfinite(t_all[j])}
    best = sorted(keyed.items(), key=lambda kv: -oriented(t_all[kv[1]]))[:10]
    per_scale = []
    for key, j in best:
        row = dict(setting=list(key), task_auroc_scale1=float(t_all[j]))
        for s in tg.SCALES:
            jj = index.get(key + (s,))
            row["scale_%s" % s] = float(t_all[jj]) if jj is not None and np.isfinite(t_all[jj]) else None
        per_scale.append(row)
    out["region_per_scale"] = per_scale
    # fraction of region settings informative (task |AUROC| >= 0.65) at >= 4 of 6 scales in a consistent direction
    consistent = total = 0
    for key in keyed:
        vals = np.array([t_all[index[key + (s,)]] for s in tg.SCALES if key + (s,) in index])
        vals = vals[np.isfinite(vals)]
        if len(vals) == len(tg.SCALES):
            total += 1
            sign = np.sign(vals[np.argmax(np.abs(vals - 0.5))] - 0.5)
            if np.sum((np.abs(vals - 0.5) >= 0.15) & (np.sign(vals - 0.5) == sign)) >= 4:
                consistent += 1
    out["region_consistent_settings"] = dict(total=total, informative_at_4_of_6_scales=consistent)

    # 2. permutation null: shuffle outcomes within base task, recompute max oriented task AUROC per family
    observed = {fam: float(np.nanmax(oriented(t_all[family == fam]))) for fam in ("simple", "persistence", "region", "v82")}
    seeds = [20260916 + i for i in range(PERMUTATIONS)]
    global _PERM_CONTEXT
    _PERM_CONTEXT = (X, labels, task, family, tuple(observed))
    from multiprocessing import Pool
    with Pool(min(PERMUTATIONS, 30)) as pool:
        per_perm = pool.map(_one_permutation, seeds)
    null = {fam: [row[fam] for row in per_perm] for fam in observed}
    out["permutation"] = {fam: dict(observed_max=observed[fam], null_max_mean=float(np.mean(null[fam])),
                                    null_max_95=float(np.quantile(null[fam], 0.95)), null_max_max=float(np.max(null[fam])),
                                    exceed_fraction=float(np.mean(np.array(null[fam]) >= observed[fam])))
                          for fam in observed}

    # 3. cross-dataset transfer: select top-K columns on one dataset, evaluate on the other
    def transfer(t_sel, t_eval, a_eval, name):
        rows = []
        for fam in ("simple", "persistence", "region", "v82"):
            cols = np.flatnonzero((family == fam) & np.isfinite(t_sel) & np.isfinite(t_eval))
            if not len(cols):
                continue
            top = cols[np.argsort(-oriented(t_sel[cols]))[:TOPK]]
            direction = np.sign(t_sel[top] - 0.5)
            same = 0.5 + direction * (t_eval[top] - 0.5)      # evaluation AUROC oriented by the selection direction
            rows.append(dict(family=fam, selected_mean=float(oriented(t_sel[top]).mean()),
                             transferred_mean=float(same.mean()), transferred_min=float(same.min()),
                             transferred_pooled_mean=float((0.5 + direction * (a_eval[top] - 0.5)).mean()),
                             top=[dict(column=list(meta[j]), selected=float(t_sel[j]), transferred=float(t_eval[j])) for j in top[:5]]))
        out[name] = rows
    transfer(t_l, t_p, a_p, "select_libero10_eval_plus")
    transfer(t_p, t_l, a_l, "select_plus_eval_libero10")

    # 4. leave-one-base-task-out selection on pooled data (both benchmarks), per family
    loto = {}
    for fam in ("simple", "persistence", "region", "v82"):
        cols = np.flatnonzero(family == fam)
        held = []
        for t in sorted(set(base_task)):
            train = np.flatnonzero(base_task != t)
            test = np.flatnonzero(base_task == t)
            if labels[test].sum() == 0 or (~labels[test]).sum() == 0:
                continue
            a_tr, n1, n0 = tg.auroc_columns(X[np.ix_(train, cols)], labels[train])
            sa_tr, _ = tg.stratified_columns(X[np.ix_(train, cols)], labels[train], list(task[train]))
            ok = (n1 >= MIN_CLASS) & (n0 >= MIN_CLASS) & np.isfinite(sa_tr)
            if not ok.any():
                continue
            j = cols[np.flatnonzero(ok)[np.argmax(oriented(sa_tr[ok]))]]
            direction = np.sign(sa_tr[ok][np.argmax(oriented(sa_tr[ok]))] - 0.5)
            a_te, _, _ = tg.auroc_columns(X[test][:, [j]], labels[test])
            held.append(dict(task=t, chosen=list(meta[j]), train_task_auroc=float(oriented(sa_tr[ok]).max()),
                             test_auroc=float(0.5 + direction * (a_te[0] - 0.5)) if np.isfinite(a_te[0]) else None,
                             n_success=int(labels[test].sum()), n_failure=int((~labels[test]).sum())))
        vals = [h["test_auroc"] for h in held if h["test_auroc"] is not None]
        loto[fam] = dict(folds=held, held_out_mean=float(np.mean(vals)) if vals else None, held_out_min=float(np.min(vals)) if vals else None)
    out["leave_one_task_out"] = loto

    (grid / "validation.json").write_text(json.dumps(out, indent=1))
    lines = ["# Validation of the grid", ""]
    lines += ["## Region settings: task AUROC at each radius scale (top 10 selected at scale 1.0)", "",
              "| setting | " + " | ".join("s=%s" % s for s in tg.SCALES) + " |", "|---|" + "---:|" * len(tg.SCALES)]
    for r in per_scale:
        lines.append("| %s | " % " ".join(str(x) for x in r["setting"]) + " | ".join("%.3f" % r["scale_%s" % s] if r["scale_%s" % s] is not None else "-" for s in tg.SCALES) + " |")
    c = out["region_consistent_settings"]
    lines += ["", "Region settings with |task AUROC| >= 0.65 in a consistent direction at >= 4 of 6 scales: %d / %d." % (c["informative_at_4_of_6_scales"], c["total"]), ""]
    lines += ["## Permutation null (%d within-task label shuffles): max oriented task AUROC per feature family" % PERMUTATIONS, "",
              "| family | observed max | null mean of max | null 95% | null max | P(null >= observed) |", "|---|---:|---:|---:|---:|---:|"]
    for fam, v in out["permutation"].items():
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.2f |" % (fam, v["observed_max"], v["null_max_mean"], v["null_max_95"], v["null_max_max"], v["exceed_fraction"]))
    for name in ("select_libero10_eval_plus", "select_plus_eval_libero10"):
        lines += ["", "## %s (top %d per family by task AUROC)" % (name, TOPK), "",
                  "| family | selected mean | transferred mean (task) | transferred min | transferred pooled |", "|---|---:|---:|---:|---:|"]
        for r in out[name]:
            lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (r["family"], r["selected_mean"], r["transferred_mean"], r["transferred_min"], r["transferred_pooled_mean"]))
        lines.append("")
        for r in out[name]:
            lines.append("- %s: " % r["family"] + "; ".join("%s -> %.3f/%.3f" % (" ".join(str(x) for x in t["column"]), t["selected"], t["transferred"]) for t in r["top"][:3]))
    lines += ["", "## Leave-one-base-task-out: best feature chosen on 9 tasks, AUROC on the held-out task", "",
              "| family | held-out mean | held-out min | folds |", "|---|---:|---:|---:|"]
    for fam, v in loto.items():
        lines.append("| %s | %s | %s | %d |" % (fam, "%.3f" % v["held_out_mean"] if v["held_out_mean"] else "-", "%.3f" % v["held_out_min"] if v["held_out_min"] else "-", len(v["folds"])))
    lines.append("")
    for fam, v in loto.items():
        lines.append("- %s: " % fam + "; ".join("%s: %s -> %s" % (h["task"], " ".join(str(x) for x in h["chosen"][:3] + h["chosen"][6:]), "%.2f" % h["test_auroc"] if h["test_auroc"] is not None else "-") for h in v["folds"]))
    (grid / "VALIDATION.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
