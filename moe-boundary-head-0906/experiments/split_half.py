"""Step 2c: how much of the development winner is search noise?

The sweep evaluates 138,003 configurations on development.  Its maximum is
therefore a 1-in-138,003 order statistic, and reporting it as if it were a
measurement would be the same mistake as reporting the argmax of an 84-cell
scan without a null.

This measures the shrinkage directly and *without spending the test cohorts*:
development's 37 tasks are split in half at random, the identical selection rule
is applied to half A, and the winner is scored on half B.  Tasks, not episodes,
are the split unit - episodes within a task share an initial-state grid and are
not independent, and the per-task recall on this corpus is known to be bimodal.

The selection rule is frozen here, before any test cohort is touched:

    among configurations with an exchange rate >= 2.0 TP per FP at lead >= 0,
    take the largest net TP over v8; break ties by fewer net FP, then by more
    net TP at lead >= 4, then lexicographically by the configuration key.

`rawdiff` configurations are excluded from selection: they are the labelled
negative control (subtracting raw values whose scales differ 3.9x, so the
difference is essentially minus the state cell).
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from common import HEADLINE_LEAD, LEADS, OUT, load, score, trailing_mean, union
from heads import (
    NEGATIVE_CONTROL_FORMS,
    PAIRS,
    constructions,
    fire_from,
    first_true_from,
    held_runs,
    tag,
    threshold_of,
    window_of,
)
from sweep_head import CONFIRMS, DEV, EARLIESTS, GATES, QUANTILES, WIDTHS

MIN_RATE = 2.0
N_SPLIT = 12
SEED = 20260906


def enumerate_configs(d, scores, n_chunk):
    """Yield (key, first_alarm_vector) for every configuration in the sweep."""
    tail = list(itertools.product(EARLIESTS, GATES))
    windows: dict[int, list] = {}
    for earliest, gate in tail:
        start, hi = window_of(earliest, gate)
        windows.setdefault(start, []).append((earliest, gate, hi))
    for name, series in scores.items():
        form = name.split("|")[0]
        if form in NEGATIVE_CONTROL_FORMS:
            continue
        for w in WIDTHS:
            sm = trailing_mean(series, w)
            for direction in ("high", "low"):
                for quantile in QUANTILES:
                    thr = threshold_of(sm, quantile, direction)
                    for start, members in windows.items():
                        run = held_runs(sm, thr, direction, start)
                        for confirm in CONFIRMS:
                            ft = first_true_from(run >= confirm)
                            for earliest, gate, hi in members:
                                first = fire_from(ft, start, hi, n_chunk)
                                if (first >= 0).sum() < 5:
                                    continue
                                yield ((name, direction, w, confirm, quantile,
                                        earliest, "-" if gate is None
                                        else "%d-%d" % gate), first)
    for family in ("bnd", "wc"):
        for act, st in PAIRS:
            for form in ("raw", "sb", "rank"):
                for w in WIDTHS:
                    sa = trailing_mean(scores[f"{form}|{family}:{act}"], w)
                    ss = trailing_mean(scores[f"{form}|{family}:{st}"], w)
                    for quantile in QUANTILES:
                      ta = threshold_of(sa, quantile, "high")
                      ts = threshold_of(ss, quantile, "low")
                      for start, members in windows.items():
                        ra = held_runs(sa, ta, "high", start)
                        rs = held_runs(ss, ts, "low", start)
                        for confirm in CONFIRMS:
                            fta = first_true_from(ra >= confirm)
                            fts = first_true_from(rs >= confirm)
                            for earliest, gate, hi in members:
                                fa = fire_from(fta, start, hi, n_chunk)
                                fs = fire_from(fts, start, hi, n_chunk)
                                both = np.where((fa >= 0) & (fs >= 0),
                                                np.maximum(fa, fs), -1)
                                g = "-" if gate is None else "%d-%d" % gate
                                for combo, first in (("OR", union(fa, fs)),
                                                     ("AND", both)):
                                    if (first >= 0).sum() < 5:
                                        continue
                                    yield (("%s|%s|%s" % (combo, form,
                                                          tag(family, act, st)),
                                            combo, w, confirm, quantile,
                                            earliest, g), first)


def pick(frame: pd.DataFrame, suffix: str) -> pd.Series | None:
    """The frozen selection rule, applied to one scored table."""
    tp, fp = frame[f"d_tp0{suffix}"], frame[f"d_fp0{suffix}"]
    rate = np.where(fp > 0, tp / np.maximum(fp, 1), np.where(tp > 0, np.inf, 0.0))
    ok = frame[(rate >= MIN_RATE) & (tp > 0)]
    if not len(ok):
        return None
    ok = ok.sort_values([f"d_tp0{suffix}", f"d_fp0{suffix}", f"d_tp4{suffix}", "key"],
                        ascending=[False, True, False, True])
    return ok.iloc[0]


def main() -> None:
    rng = np.random.default_rng(SEED)
    d = load(DEV)
    risk, length, v8 = d["risk"], d["length"], d["v8"]
    n_chunk = d["front_state"].shape[1]
    tasks = np.unique(d["task"])
    n = len(risk)

    halves = []
    for r in range(N_SPLIT):
        order = rng.permutation(len(tasks))
        a = np.isin(d["task"], tasks[order[: len(tasks) // 2]])
        halves.append((a, ~a))
    masks = [m for pair in halves for m in pair]          # 2 * N_SPLIT
    m_risk = np.stack([(m & risk).astype(np.float32) for m in masks])
    m_safe = np.stack([(m & ~risk).astype(np.float32) for m in masks])

    def counts(first, lead):
        fired = first >= 0
        t = (fired & ((length - first) >= lead)).astype(np.float32)
        return m_risk @ t, m_safe @ t

    base = {lead: counts(v8, lead) for lead in (0, HEADLINE_LEAD)}
    scores = constructions(d, d)
    all_risk = risk.astype(np.float32)
    all_safe = (~risk).astype(np.float32)
    whole = {lead: (all_risk @ ((v8 >= 0) & ((length - v8) >= lead)).astype(np.float32),
                    all_safe @ ((v8 >= 0) & ((length - v8) >= lead)).astype(np.float32))
             for lead in (0, HEADLINE_LEAD)}

    rows, full_rows = [], []
    for key, first in enumerate_configs(d, scores, n_chunk):
        name = "|".join(map(str, key))
        row = {"key": name}
        fr = {"key": name}
        u = union(v8, first)
        for lead in (0, HEADLINE_LEAD):
            tp, fp = counts(u, lead)
            row |= {f"d_tp{lead}_h{i}": float(tp[i] - base[lead][0][i])
                    for i in range(len(masks))}
            row |= {f"d_fp{lead}_h{i}": float(fp[i] - base[lead][1][i])
                    for i in range(len(masks))}
            t = ((u >= 0) & ((length - u) >= lead)).astype(np.float32)
            fr[f"d_tp{lead}"] = float(all_risk @ t - whole[lead][0])
            fr[f"d_fp{lead}"] = float(all_safe @ t - whole[lead][1])
        rows.append(row)
        full_rows.append(fr)
    table = pd.DataFrame(rows)
    ftab = pd.DataFrame(full_rows)
    print("枚举 %d 个配置（已排除 rawdiff 负对照）" % len(table))

    out = []
    for r in range(N_SPLIT):
        ia, ib = 2 * r, 2 * r + 1
        for train, test in ((ia, ib), (ib, ia)):
            win = pick(table, f"_h{train}")
            if win is None:
                continue
            out.append({
                "split": r, "train_half": train,
                "key": win.key,
                "train_d_tp0": win[f"d_tp0_h{train}"],
                "train_d_fp0": win[f"d_fp0_h{train}"],
                "test_d_tp0": win[f"d_tp0_h{test}"],
                "test_d_fp0": win[f"d_fp0_h{test}"],
                "train_d_tp4": win[f"d_tp4_h{train}"],
                "test_d_tp4": win[f"d_tp4_h{test}"],
                "test_d_fp4": win[f"d_fp4_h{test}"],
            })
    res = pd.DataFrame(out)
    res.to_csv(OUT / "split_half.csv", index=False)

    print("\n===== 任务对半分：在 A 半上按冻结规则选，在 B 半上打分 =====")
    print("%d 次选择（%d 个划分 x 2 个方向）" % (len(res), N_SPLIT))
    print("%-12s %14s %14s" % ("", "选择半（A）", "留出半（B）"))
    print("%-12s %6.1f TP %5.1f FP %6.1f TP %5.1f FP"
          % ("中位", res.train_d_tp0.median(), res.train_d_fp0.median(),
             res.test_d_tp0.median(), res.test_d_fp0.median()))
    print("%-12s %6.1f TP %5.1f FP %6.1f TP %5.1f FP"
          % ("均值", res.train_d_tp0.mean(), res.train_d_fp0.mean(),
             res.test_d_tp0.mean(), res.test_d_fp0.mean()))
    tr = res.train_d_tp0.sum() / max(res.train_d_fp0.sum(), 1e-9)
    te = res.test_d_tp0.sum() / max(res.test_d_fp0.sum(), 1e-9)
    print("合计兑换率  选择半 %.2f TP/FP   留出半 %.2f TP/FP" % (tr, te))
    print("留出半净增 TP > 0 的比例: %d/%d" % (int((res.test_d_tp0 > 0).sum()), len(res)))
    clears = ((res.test_d_tp0 >= MIN_RATE * res.test_d_fp0)
              & (res.test_d_tp0 > 0))
    print("留出半兑换率 >= %.1f 的比例: %d/%d"
          % (MIN_RATE, int(clears.sum()), len(res)))
    print("\n被选中的配置（前 8 次）:")
    print(res.head(8)[["split", "key", "train_d_tp0", "train_d_fp0",
                       "test_d_tp0", "test_d_fp0"]].to_string(index=False))
    print("\n不同划分选出同一个配置的次数: %d 个不同配置 / %d 次选择"
          % (res.key.nunique(), len(res)))

    # ---- freeze on the whole of development ------------------------------
    ftab.to_csv(OUT / "full_development_configs.csv", index=False)
    winner = pick(ftab, "")
    print("\n===== 在全部 development 上按同一规则冻结 =====")
    print("  %s" % winner.key)
    print("  净增 lead>=0: %+d TP / %+d FP     lead>=4: %+d TP / %+d FP"
          % (winner.d_tp0, winner.d_fp0, winner.d_tp4, winner.d_fp4))

    parts = winner.key.split("|")
    spec = {"score": "|".join(parts[:-6]) if len(parts) > 7 else parts[0],
            "raw_key": winner.key}
    fields = ["direction", "width", "confirm", "quantile", "earliest", "gate"]
    for name, value in zip(fields, parts[-6:]):
        spec[name] = value
    spec["score"] = "|".join(parts[: len(parts) - 6])
    spec["width"] = int(spec["width"])
    spec["confirm"] = int(spec["confirm"])
    spec["quantile"] = float(spec["quantile"])
    spec["earliest"] = int(spec["earliest"])
    spec["development"] = {"d_tp0": int(winner.d_tp0), "d_fp0": int(winner.d_fp0),
                           "d_tp4": int(winner.d_tp4), "d_fp4": int(winner.d_fp4)}
    spec["selection_rule"] = ("max d_tp0 subject to d_tp0/d_fp0 >= %.1f at "
                              "lead >= 0; ties by fewer d_fp0, then more d_tp4,"
                              " then key order" % MIN_RATE)
    spec["n_configs_enumerated"] = int(len(ftab))
    spec["split_half"] = {
        "n_selections": int(len(res)),
        "median_train_d_tp0": float(res.train_d_tp0.median()),
        "median_test_d_tp0": float(res.test_d_tp0.median()),
        "median_train_d_fp0": float(res.train_d_fp0.median()),
        "median_test_d_fp0": float(res.test_d_fp0.median()),
        "pooled_train_rate": float(tr), "pooled_test_rate": float(te),
        "frac_test_positive": float((res.test_d_tp0 > 0).mean()),
    }
    (OUT / "frozen_spec.json").write_text(json.dumps(spec, indent=2))
    print("\n写出 %s" % (OUT / "frozen_spec.json"))


if __name__ == "__main__":
    main()
