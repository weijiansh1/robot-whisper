#!/usr/bin/env python3
"""Protocol tests.  Run: python tests/test_protocol.py"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))

import protocol as P  # noqa: E402
import detector as D  # noqa: E402
import auc_table as A  # noqa: E402
import anchors  # noqa: E402
import controls as CTL  # noqa: E402

FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not condition:
        FAILED.append(name)


def test_window_constants() -> None:
    print("window constants")
    check("anchor window ends",
          P.IN_WINDOW_END == {"libero_goal": 18, "libero_long": 33,
                              "libero_object": 17, "libero_spatial": 13},
          str(P.IN_WINDOW_END))
    check("strict phase window differs only for long",
          P.IN_WINDOW_END_STRICT["libero_long"] == 32
          and all(P.IN_WINDOW_END_STRICT[s] == P.IN_WINDOW_END[s]
                  for s in ("libero_goal", "libero_object", "libero_spatial")))


def test_self_normalisation() -> None:
    print("self normalisation")
    x = np.arange(24, dtype=np.float32).reshape(2, 6, 2)
    out = P.apply_self(x)
    base = x[:, :4].mean(axis=1)
    check("chunks 0..3 undefined", bool(np.all(np.isnan(out[:, :4]))))
    check("chunk 4 == value - own head mean",
          np.allclose(out[:, 4], x[:, 4] - base, atol=1e-5))
    y = x.copy()
    y[0, 0, 0] = np.nan
    y[0, 1, 0] = np.nan
    out2 = P.apply_self(y)
    check("fewer than 3 finite head chunks -> NaN", bool(np.isnan(out2[0, 4, 0])))
    check("3 finite head chunks are enough", bool(np.isfinite(P.apply_self(
        np.where(np.arange(6)[None, :, None] == 0, np.nan, x))[0, 4, 0])))


def test_pop_normalisation() -> None:
    print("pop normalisation")
    values = np.array([[[1.0], [2.0]], [[1.0], [2.0]], [[3.0], [9.0]],
                       [[5.0], [9.0]]], np.float32)
    alive = np.ones((4, 2), bool)
    ref = P.pop_reference(values, alive)
    got = P.apply_pop(values, ref)
    # chunk 0 sample is [1,1,3,5]: the tie group {1,1} occupies ranks 1..2 so its
    # mid-rank fraction is (0+2)/(2*4) = 0.25
    check("tie group receives the centre of its block",
          np.allclose(got[0, 0, 0], 0.25) and np.allclose(got[1, 0, 0], 0.25),
          f"{got[:, 0, 0]}")
    # mid-rank of the single largest of n=4 is (3+4)/(2*4) = 0.875, not 1.0
    check("largest value gets its mid-rank, not 1.0",
          np.allclose(got[3, 0, 0], 0.875))
    check("pop is monotone in raw",
          bool(np.all(np.argsort(got[:, 0, 0]) == np.argsort(values[:, 0, 0]))))


def test_ties() -> None:
    print("ties: >= must fire on the whole tie group")
    sample = np.array([0.0] * 50 + [1.0] * 40 + [2.0] * 10)
    tau = float(np.quantile(sample, 1 - 0.20, method="lower"))
    n_ge, n_gt = int((sample >= tau).sum()), int((sample > tau).sum())
    check("threshold is a sample value", tau in set(sample.tolist()))
    check(">= keeps the tie group", n_ge == 50, f"n_ge={n_ge} n_gt={n_gt}")
    check("> would drop it", n_gt == 10)
    got = CTL.tie_group_fires(sample[None, :], np.ones((1, len(sample)), bool), 0.20)
    check("tie_group_fires assertion holds", got["tie_group_fires"])


def test_first_alarm() -> None:
    print("first alarm")
    fire = np.array([[False, False, True, True], [False, False, False, False],
                     [True, False, False, False]])
    check("argmax semantics", list(P.first_alarm(fire)) == [2, -1, 0])


def test_cohorts_and_estimator() -> None:
    print("cohorts, risk definition and the fixed-chunk estimator")
    frames = {c: P.load_cohort(c, include_step=False) for c in P.COHORTS}
    counts = {c: int(f["risk"].sum()) for c, f in frames.items()}
    check("risk counts",
          counts == {"development_main": 487, "external_8b": 564,
                     "legacy_main16x32": 307}, str(counts))
    check("total risks == 1358", sum(counts.values()) == 1358)
    for c, f in frames.items():
        check(f"{c}: every risk episode is exactly at its suite cap",
              bool(np.all(f["length"][f["risk"]] == f["cap"][f["risk"]])))
        check(f"{c}: every risk episode is alive at the window end",
              bool(np.all(f["alive"][np.arange(len(f["risk"])), f["window_end"]][f["risk"]])))

    anchor = anchors.check(frames)
    check("v7 anchors reproduced exactly", anchor["reproduced"], str(anchor["mismatch"]))

    dev = frames[P.FIT_COHORT]
    cols = dev["columns"]
    ctrl_idx = [j for j, c in enumerate(cols) if c.startswith("ctrl_const_elapsed|")]
    j = ctrl_idx[0]
    take = (dev["suite"] == "libero_spatial") & (dev["length"] > 8)
    x = dev["values"][take, 8, :][:, [j]]
    got = A.per_task_auc(x, dev["risk"][take], dev["task"][take])
    check("constant-elapsed scores exactly 0.5",
          float(abs(got["auc"][0] - 0.5)) == 0.0, f"auc={got['auc'][0]!r}")

    # raw and pop share one within-chunk order, so their within-task AUC is equal
    sub_cols = [c for c in cols if c.startswith(("mobility|", "flow_path|", "set_dwell|"))]
    idx = [cols.index(c) for c in sub_cols]
    raw = dev["values"][:, :, idx]
    ref = P.pop_reference(raw, dev["alive"])
    pop = P.apply_pop(raw, ref)
    a_raw = A.per_task_auc(raw[take, 8, :], dev["risk"][take], dev["task"][take])["auc"]
    a_pop = A.per_task_auc(pop[take, 8, :], dev["risk"][take], dev["task"][take])["auc"]
    check("AUC(raw) == AUC(pop) within task at a fixed chunk",
          bool(np.allclose(a_raw, a_pop, atol=1e-12, equal_nan=True)),
          f"max diff {np.nanmax(np.abs(a_raw - a_pop)):.3e}")
    return frames


def test_sealed_scoring(frames) -> None:
    print("sealed scoring")
    path = P.RESULTS / "artefacts.pkl"
    if not path.exists():
        print("  [SKIP] artefacts.pkl not built yet")
        return
    with open(path, "rb") as fh:
        art = pickle.load(fh)
    det = art["detectors"]["unified"]
    ext = frames["external_8b"]
    cols = list(dict.fromkeys(det.columns))
    idx = [ext["columns"].index(c) for c in det.columns]
    full = D.score(det, ext, ext["values"][:, :, idx], det.columns)
    take = np.arange(0, ext["n_ep"], 7)
    sub_frame = dict(ext)
    sub_frame["alive"] = ext["alive"][take]
    part = D.score(det, sub_frame, ext["values"][take][:, :, idx], det.columns)
    same = np.allclose(full[take], part, atol=1e-6, equal_nan=True)
    check("external score of an episode does not depend on the other external "
          "episodes (nothing is estimated on the sealed cohort)", bool(same))
    check("detector uses no control channel",
          not any(c.split("|")[0] in P.CONTROL_QUANTITIES for c in det.columns))
    check("detector is portable to all three cohorts",
          all(set(cols) <= set(frames[c]["columns"]) for c in P.COHORTS))
    check("term count is small enough to state", len(det.terms) <= 12,
          f"{len(det.terms)} terms")


def main() -> None:
    test_window_constants()
    test_self_normalisation()
    test_pop_normalisation()
    test_ties()
    test_first_alarm()
    frames = test_cohorts_and_estimator()
    test_sealed_scoring(frames)
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        raise SystemExit(1)
    print("all tests passed")


if __name__ == "__main__":
    main()
