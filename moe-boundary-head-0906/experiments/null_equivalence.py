"""How many false alarms would a null head need to buy the same true positives?

The brief asks for the figure that was reported for v8's own two heads: shuffled
across episodes within each chunk, they cost 4,418 FP to reach a comparable TP
(1,210/1,358 against v8's 932 at 126 FP).

Here the frozen boundary head is replaced by a surrogate of the same series, its
quantile is loosened until the union reaches the real head's true-positive
count, and the false alarms at that point are reported.  Three surrogates of
increasing strength are used; `flat_shuffle` is the one directly comparable to
the published v8 figure, `episode_swap` is the hardest.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import ALL_CELLS, COHORTS, HEADLINE_LEAD, OUT, load_all, score, union
from freeze_and_score import build_first
from heads import constructions
from pairwise_scan import SURROGATES

SEED = 20260906
DRAWS = 6
GRID = (0.99, 0.98, 0.97, 0.95, 0.92, 0.88, 0.84, 0.80, 0.70, 0.60, 0.50)
DEV = "development_main"


def total(spec, series, dev_scores, n_chunk, data):
    firsts, *_ = build_first(spec, series, dev_scores, n_chunk)
    tp = fp = 0
    for cohort in COHORTS:
        d = data[cohort]
        s = score(union(d["v8"], firsts[cohort]), d["risk"], d["length"],
                  HEADLINE_LEAD)
        tp += s["tp"]
        fp += s["fp"]
    return tp, fp


def main() -> None:
    spec = json.loads((OUT / "frozen_spec.json").read_text())
    real = json.loads((OUT / "frozen_summary.json").read_text())["headline"]
    data = load_all()
    dev = data[DEV]
    n_chunk = {c: data[c]["front_state"].shape[1] for c in COHORTS}
    dev_scores = constructions(dev, dev)

    print("真实：v8 %d/%d TP %d FP  ->  v8+头 %d/%d TP %d FP  （净 %+d/%+d）"
          % (real["v8_tp"], real["risk"], real["v8_fp"], real["new_tp"],
             real["risk"], real["new_fp"], real["d_tp"], real["d_fp"]))
    print("问题：把这个头换成代理序列，要多少 FP 才能买到同样的 %d TP？\n"
          % real["new_tp"])

    rows = []
    for kind, fn in SURROGATES.items():
        needed = []
        for draw in range(DRAWS):
            rng = np.random.default_rng(SEED + 7919 * draw + abs(hash(kind)) % 977)
            series = {}
            for cohort in COHORTS:
                d = data[cohort]
                fake = dict(d)
                for cell in ALL_CELLS:
                    fake[cell] = fn(d[cell], d, rng)
                series[cohort] = constructions(fake, dev)
            hit = None
            for q in GRID:
                probe = dict(spec, quantile=q)
                tp, fp = total(probe, series, dev_scores, n_chunk, data)
                rows.append({"surrogate": kind, "draw": draw, "quantile": q,
                             "tp": tp, "fp": fp})
                if tp >= real["new_tp"]:
                    hit = (q, tp, fp)
                    break
            needed.append(hit)
            print("  %-15s draw %d: %s" % (kind, draw,
                  "分位放到 %.2f 时达到 %d TP，代价 %d FP" % hit if hit
                  else "放到分位 %.2f 仍未达到 %d TP（最好 %d TP / %d FP）"
                       % (GRID[-1], real["new_tp"], tp, fp)))
        got = [h for h in needed if h]
        if got:
            fps = [h[2] for h in got]
            print("  %-15s -> 中位 %d FP（真实头只要 %d FP），比值 %.1fx\n"
                  % (kind, int(np.median(fps)), real["new_fp"],
                     np.median(fps) / real["new_fp"]))
        else:
            print("  %-15s -> 在整个分位网格上都达不到，比值为无穷\n" % kind)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "null_equivalence.csv", index=False)

    summary = []
    for kind in SURROGATES:
        sub = table[table.surrogate == kind]
        reach = sub[sub.tp >= real["new_tp"]]
        summary.append({
            "surrogate": kind,
            "reached_target": int(len(reach)),
            "median_fp_at_target": float(reach.fp.median()) if len(reach) else None,
            "real_fp": real["new_fp"],
            "cost_ratio": (float(reach.fp.median()) / real["new_fp"]
                           if len(reach) else None),
            "best_tp_seen": int(sub.tp.max()),
            "fp_at_best_tp": int(sub.loc[sub.tp.idxmax()].fp),
        })
    (OUT / "null_equivalence.json").write_text(json.dumps(
        {"real": real, "surrogates": summary, "grid": list(GRID),
         "draws": DRAWS}, indent=2, default=float))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
