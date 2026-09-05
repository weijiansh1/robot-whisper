#!/usr/bin/env python3
"""Offline onset labels for trunk_harvest.py output (+ corpus-A validation).

Labels every trunk directory (dense.npz + summary.json) with the frozen
canonical rule ``physical_onsets`` IMPORTED from
himoe-vla_trap/code/analyze_trainfree_signal_matrix.py (L347-393) -- never
copied, thresholds never touched.  Its query-resolution inputs are built with
the exact corpus-A conventions:

* query state = the state the policy saw at that query (state BEFORE the
  query's first action).  ``per_query_arrays`` (analyze_trap_onset_sweep.py)
  picks the same rows from the candidate dense streams; verified bit-equal to
  the candidates' per-query policy_state/sim_state arrays.
* eef = state[:, :3]; gripper = gripper_qpos[:, :2].mean(axis=1); float32
  casts exactly as per_query_arrays.
* objects = sim_state[:, 10:13] and [:, 17:20] (KITCHEN_SCENE8 moka pots);
  slots asserted against sim_layout.json exactly like
  analysis_moe_phenotype/events/build_events.py::check_scene8_layout.
* references = success_terminal_references["moka_pot_1_joint0"] from the
  rolling-star t08 analysis (same key as build_events.py and
  collect_snapshot_fork_recovery.py::load_references).

Array-format adapters (candidate npz vs harvest dense.npz):
* corpus-A candidate npz: control_sim_state / control_eef_position /
  control_gripper_qpos carry T+1 rows (row 0 = state BEFORE the first branch
  action) while control_action / control_query_index carry T rows, so
  per_query_arrays' ``starts`` (first action row of each query) lands directly
  on query-start states.
* harvest dense.npz (trunk_harvest.py): every control_* stream has exactly T
  rows recorded AFTER each env step and no pre-action row, so query-start rows
  are ``starts - 1`` for q >= 1; q0 eef/gripper come from dense.npz:query_state
  (the stored policy observation, the exact analogue of the candidates'
  policy_state) and the q0 object poses from query_000/full_state.npz.

--validate-corpus-a checks this extraction against the authoritative table
analysis_trap_event_moe_20260829/event_onsets.csv:

* loop_onset: extraction + imported physical_onsets must match row for row
  (the table's loop column came from the logic-identical loop_onset_query on
  the same query-start series).
* static_onset: the table's static column is the DENSE 80-action rule
  (analyze_trap_event_moe.py::static_onset_query), NOT the query-resolution
  stasis proxy inside physical_onsets.  The like-for-like extraction check
  therefore IMPORTS that very function and must also match row for row.
* The canonical proxy-vs-dense static gap is reported alongside and, on the
  full 352-candidate set, asserted equal to the frozen record of the trainfree
  pipeline (results/trainfree_signal_matrix/tables/onset_inventory.csv:
  proxy fires on 71, sensitivity 1.0, 17 false positives, onset delta median
  +1).  That gap is a documented property of the frozen rule itself
  (METHODS_AND_REPRODUCIBILITY_ZH.md: corpus A ground truth is dense; the
  query proxy was validated, not defined, on it) -- it is not an extraction
  error, and harvest labels deliberately stay on the canonical proxy rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

TRAP_CODE = pathlib.Path("/home/jovyan/work/himoe-vla/himoe-vla_trap/code")
ROUTE_CAPTURE = pathlib.Path("/home/jovyan/work/himoe-vla/himoe-route-capture")
A_RUN = ROUTE_CAPTURE / "runs/rolling-star-a100-long-t08-k16-20260828"
DEFAULT_REFERENCES = A_RUN / "analysis/physical_label_definitions.json"

sys.dont_write_bytecode = True  # never drop __pycache__ into the read-only trees
sys.path.insert(0, str(ROUTE_CAPTURE))
sys.path.insert(0, str(TRAP_CODE))

from analyze_trainfree_signal_matrix import physical_onsets  # noqa: E402  canonical L347-393
from branch_snapshot import decode_full_state  # noqa: E402  safe snapshot codec

# Frozen corpus-A record of the query-proxy vs dense-static gap
# (himoe-vla_trap/results/trainfree_signal_matrix/tables/onset_inventory.csv).
CANON_STATIC_PROXY = {"fires": 71, "false_negatives": 0, "false_positives": 17,
                      "onset_delta_median": 1.0}


def load_references(path: pathlib.Path) -> np.ndarray:
    """Mirrors collect_snapshot_fork_recovery.load_references (t08 terminals)."""
    definitions = json.loads(path.read_text(encoding="utf-8"))
    references = np.asarray(
        definitions["success_terminal_references"]["moka_pot_1_joint0"],
        dtype=np.float64,
    )
    if references.ndim != 2 or references.shape[1] != 3:
        raise RuntimeError("terminal references have the wrong shape")
    return references


def check_scene8_layout(layout: dict) -> None:
    """Same frozen-slot assertions as build_events.check_scene8_layout."""
    slots = {j["joint"]: (j["state_lo"], j["state_hi"]) for j in layout["joints"]}
    assert slots.get("moka_pot_1_joint0") == (10, 17), slots
    assert slots.get("moka_pot_2_joint0") == (17, 24), slots
    assert layout["state_dim"] == 47, layout["state_dim"]


def query_starts(query_index: np.ndarray) -> np.ndarray:
    """Verbatim per_query_arrays: first dense row of every query."""
    return np.r_[0, np.flatnonzero(np.diff(query_index) != 0) + 1]


def candidate_query_series(arrays: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query-start (eef, objects, gripper) from one corpus-A candidate npz.

    Adapter: candidate state streams have T+1 rows (row 0 = pre-action state)
    against T-row control_query_index, so ``starts`` indexes query-start rows
    directly.  Casts and the gripper scalar follow per_query_arrays verbatim.
    """
    if len(arrays["control_sim_state"]) != len(arrays["control_query_index"]) + 1:
        raise RuntimeError("candidate npz lost its pre-action state row")
    sim = np.asarray(arrays["control_sim_state"], np.float32)
    eef = np.asarray(arrays["control_eef_position"], np.float32)
    gripper = np.asarray(arrays["control_gripper_qpos"], np.float32).mean(axis=1)
    starts = query_starts(np.asarray(arrays["control_query_index"]))
    objects = np.stack((sim[:, 10:13], sim[:, 17:20]), axis=1)
    return eef[starts], objects[starts], gripper[starts]


def load_q0_sim(trunk_dir: pathlib.Path) -> np.ndarray:
    """Sim vector saved by trunk_harvest at query 0 (state before any action)."""
    qdir = trunk_dir / "query_000"
    metadata = json.loads((qdir / "full_state.json").read_text(encoding="utf-8"))
    with np.load(qdir / "full_state.npz", allow_pickle=False) as archive:
        snapshot = decode_full_state(metadata, {k: archive[k] for k in archive.files})
    return np.asarray(snapshot["sim"], np.float64)


def harvest_query_series(trunk_dir: pathlib.Path, dense: dict,
                         n_queries: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query-start (eef, objects, gripper) from one harvest trunk.

    Adapter: harvest control_* rows are post-action and there is no pre-action
    row (unlike candidate npz), so the query-start state of query q >= 1 is the
    dense row ``starts[q] - 1`` (after the last action of q-1).  eef/gripper for
    ALL queries (incl. q0) come from query_state -- the recorded policy
    observation, whose [:3] / [6:8] fields are the exact analogue of the
    candidates' policy_state (verified bit-equal on corpus A).  q0 object poses
    come from the query_000 snapshot.
    """
    qi = np.asarray(dense["control_query_index"])
    steps = len(qi)
    for name in ("control_sim_state", "control_eef_position",
                 "control_gripper_qpos", "control_action", "control_success"):
        if len(dense[name]) != steps:  # harvest streams are all T rows
            raise RuntimeError(f"{trunk_dir}: {name} misaligned with query_index")
    starts = query_starts(qi)
    if len(starts) != n_queries or int(qi[-1]) != n_queries - 1:
        raise RuntimeError(f"{trunk_dir}: query bookkeeping mismatch")
    qstate = np.asarray(dense["query_state"], np.float32)
    if qstate.shape != (n_queries, 8):
        raise RuntimeError(f"{trunk_dir}: query_state shape {qstate.shape}")
    eef = qstate[:, :3]
    gripper = qstate[:, 6:8].mean(axis=1)
    sim = np.asarray(dense["control_sim_state"], np.float32)
    objects = np.empty((n_queries, 2, 3), np.float32)
    objects[1:, 0] = sim[starts[1:] - 1, 10:13]
    objects[1:, 1] = sim[starts[1:] - 1, 17:20]
    sim0 = load_q0_sim(trunk_dir).astype(np.float32)
    objects[0, 0] = sim0[10:13]
    objects[0, 1] = sim0[17:20]
    # Alignment audit: the policy observation at query q must sit on the dense
    # row after the last action of q-1 (same obs, float32 vs float64 storage).
    dense_eef = np.asarray(dense["control_eef_position"], np.float32)
    if not np.allclose(dense_eef[starts[1:] - 1], eef[1:], atol=1e-5):
        raise RuntimeError(f"{trunk_dir}: query_state / dense stream misalignment")
    return eef, objects, gripper


def label_harvest(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.harvest_dir)
    references = load_references(pathlib.Path(args.references))
    check_scene8_layout(json.loads((root / "sim_layout.json").read_text(encoding="utf-8")))
    rows = []
    for tdir in sorted(root.glob("trunk_*")):
        if not tdir.is_dir():
            continue
        if not (tdir / "summary.json").exists() or not (tdir / "dense.npz").exists():
            print(f"[skip] {tdir.name}: incomplete trunk", flush=True)
            continue
        summary = json.loads((tdir / "summary.json").read_text(encoding="utf-8"))
        with np.load(tdir / "dense.npz", allow_pickle=False) as archive:
            dense = {name: np.asarray(archive[name]) for name in archive.files}
        n_queries = int(summary["queries"])
        eef, objects, gripper = harvest_query_series(tdir, dense, n_queries)
        loop, static, trap = (int(v) for v in
                              physical_onsets(eef, objects, gripper, references))
        rows.append([int(summary["trunk_uid"]), int(summary["init_state"]),
                     int(summary["stream"]), int(bool(summary["success"])),
                     n_queries, loop, static, trap])
        print(f"[label] {tdir.name}: q={n_queries} loop={loop} static={static} "
              f"trap={trap} success={bool(summary['success'])}", flush=True)
    if not rows:
        raise SystemExit(f"{root}: no complete trunk_* directories")
    out = pathlib.Path(args.out) if args.out else root / "labels.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trunk_uid", "init_state", "stream", "success",
                         "n_queries", "loop_onset", "static_onset", "trap_onset"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}", flush=True)
    return 0


def pick_sample(table, size: int, seed: int) -> list[int]:
    """>=10 (or all) of each class {loop, static, no-event}, rest random."""
    rng = np.random.default_rng(seed)
    loop_idx = np.flatnonzero(table.loop_onset.to_numpy() >= 0)
    static_idx = np.flatnonzero(table.static_onset.to_numpy() >= 0)
    none_idx = np.flatnonzero((table.loop_onset.to_numpy() < 0)
                              & (table.static_onset.to_numpy() < 0))
    chosen: set[int] = set()
    for idx in (loop_idx, static_idx, none_idx):
        take = min(10, len(idx))
        chosen.update(rng.choice(idx, size=take, replace=False).tolist())
    rest = np.setdiff1d(np.arange(len(table)), sorted(chosen))
    extra = max(0, size - len(chosen))
    if extra:
        chosen.update(rng.choice(rest, size=min(extra, len(rest)),
                                 replace=False).tolist())
    return sorted(chosen)


def validate_corpus_a(args: argparse.Namespace) -> int:
    import pandas as pd
    import analyze_rolling_star_experiment as rolling
    import analyze_trap_event_moe as trap_event  # authority's dense static rule

    run_root = pathlib.Path(args.run_root)
    references = load_references(pathlib.Path(args.references))
    table = pd.read_csv(run_root / "analysis_trap_event_moe_20260829/event_onsets.csv")
    targets, _layout = rolling.load_layout(run_root)
    if [(t["joint"], int(t["state_lo"])) for t in targets] != [
            ("moka_pot_1_joint0", 10), ("moka_pot_2_joint0", 17)]:
        raise RuntimeError(f"unexpected SCENE8 layout targets: {targets}")

    picked = pick_sample(table, args.sample, args.seed) if args.sample else list(range(len(table)))
    if len(picked) < 30:
        raise SystemExit("validation needs at least 30 candidates")
    sub = table.iloc[picked]
    n_loop = int((sub.loop_onset >= 0).sum())
    n_static = int((sub.static_onset >= 0).sum())
    n_none = int(((sub.loop_onset < 0) & (sub.static_onset < 0)).sum())
    if min(n_loop, n_static, n_none) == 0:
        raise SystemExit("sample must contain loop, static and no-event candidates")

    records, mismatches = [], []
    for row in sub.itertuples():
        path = (run_root / f"formal/worker{row.worker}/snapshot_{row.snapshot:03d}/"
                f"candidate_{row.candidate:02d}.npz")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in
                      ("control_sim_state", "control_eef_position",
                       "control_gripper_qpos", "control_query_index")}
        eef, objects, gripper = candidate_query_series(arrays)
        loop_mine, static_proxy, _trap = physical_onsets(eef, objects, gripper, references)
        static_dense = trap_event.static_onset_query(arrays, targets)
        rec = dict(episode_id=int(row.episode_id),
                   loop_ok=bool(loop_mine == row.loop_onset),
                   static_ok=bool(static_dense == row.static_onset),
                   nq_ok=bool(len(eef) == row.n_query),
                   loop_ref=int(row.loop_onset), loop_mine=int(loop_mine),
                   static_ref=int(row.static_onset), static_dense=int(static_dense),
                   static_proxy=int(static_proxy))
        records.append(rec)
        if not (rec["loop_ok"] and rec["static_ok"] and rec["nq_ok"]):
            mismatches.append(rec)

    n = len(records)
    loop_exact = sum(r["loop_ok"] for r in records)
    static_exact = sum(r["static_ok"] for r in records)
    nq_exact = sum(r["nq_ok"] for r in records)
    print(f"corpus-A validation on {n} candidates "
          f"(classes: loop={n_loop} static={n_static} no-event={n_none})")
    print(f"  loop_onset   exact-match: {loop_exact}/{n} "
          f"(extraction + imported physical_onsets vs event_onsets.csv)")
    print(f"  static_onset exact-match: {static_exact}/{n} "
          f"(extraction + imported static_onset_query vs event_onsets.csv)")
    print(f"  n_query      exact-match: {nq_exact}/{n}")

    # Canonical context: the frozen physical_onsets stasis proxy vs the dense
    # table -- a property of the rule, reported for transparency and pinned to
    # the trainfree onset_inventory record when the full set is validated.
    ref = np.array([r["static_ref"] for r in records])
    proxy = np.array([r["static_proxy"] for r in records])
    both = (ref >= 0) & (proxy >= 0)
    proxy_stats = {
        "fires": int((proxy >= 0).sum()),
        "false_negatives": int(((proxy < 0) & (ref >= 0)).sum()),
        "false_positives": int(((proxy >= 0) & (ref < 0)).sum()),
        "onset_delta_median": float(np.median(proxy[both] - ref[both])) if both.any() else float("nan"),
    }
    print(f"  [canon] physical_onsets static proxy vs dense table: "
          f"exact {int((proxy == ref).sum())}/{n}, {proxy_stats}")
    if not args.sample and len(table) == n:
        assert proxy_stats == CANON_STATIC_PROXY, (proxy_stats, CANON_STATIC_PROXY)
        print("  [canon] proxy gap identical to frozen onset_inventory.csv record")

    if args.report:
        report = {"n": n, "classes": {"loop": n_loop, "static": n_static, "none": n_none},
                  "loop_exact": loop_exact, "static_exact": static_exact,
                  "nq_exact": nq_exact, "static_proxy_stats": proxy_stats,
                  "mismatches": mismatches}
        path = pathlib.Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {path}")

    if mismatches:
        print(f"FAILED: {len(mismatches)} mismatching candidates, e.g. {mismatches[:3]}")
        return 1
    print("validation PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--harvest-dir",
                      help="trunk_harvest output root (trunk_*/ + sim_layout.json)")
    mode.add_argument("--validate-corpus-a", action="store_true",
                      help="check extraction against the rolling-star authority table")
    ap.add_argument("--out", help="labels.csv path (default: <harvest-dir>/labels.csv)")
    ap.add_argument("--references", default=str(DEFAULT_REFERENCES),
                    help="physical_label_definitions.json with t08 terminal references")
    ap.add_argument("--run-root", default=str(A_RUN),
                    help="corpus-A run root (validation mode)")
    ap.add_argument("--sample", type=int, default=0,
                    help="validate on a stratified sample (0 = all 352)")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--report", help="optional JSON report path (validation mode)")
    args = ap.parse_args()
    if args.harvest_dir:
        return label_harvest(args)
    return validate_corpus_a(args)


if __name__ == "__main__":
    raise SystemExit(main())
