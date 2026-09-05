"""Unit tests for e1e2_analyze.py against a synthetic routes.zarr + manifest
with hand-computable answers.

Construction (F=10, L=8, U=11, E=32; distributions are exact one-hots so
Hellinger terms are exact):

  base row: every gate one-hot on expert 0.
  E1 pert  (loop):    gate (l=4,u=1) flipped to expert 1 at ALL f ("carrier",
                      k=1 everywhere); at f=3, 3 extra primary gates flipped
                      (k_pri(3)=4; 15 extras => k=16 for scale 1e-2 dir 2) plus
                      one secondary-only gate (l=0,u=0) (k_sec(3)=k_pri(3)+1).
  E1 pert  (control): carrier only (k=1 at every f).
  E2 seeds (loop):    seed i flips gate (l=5,u=3) at f=5 to expert i
                      -> mid-flow fan-out, terminal reconvergence.
  E2 seeds (control): seed i flips gate (l=6,u=2) at f=9 to expert i
                      -> terminal divergence.

k orthogonal one-hot flips give route_distance = sqrt(k)/sqrt(n_entries), so
with b = 1/sqrt(4*10*32) (primary) and c = 1/sqrt(8*11*32) (secondary):
  loop   G_f = b/s except G_3 = 2b/s  => G_peak=2b/s, argmax=3, funnel=0.5,
                                          lambda=0
  control G_f = b/s constant          => funnel=0, argmax=0, lambda=0
  loop   D_seed: 1/sqrt(n) at f=5 only => FunnelIndex=1, argmax=5
  control D_seed: 1/sqrt(n) at f=9    => FunnelIndex~0, argmax=9

2 trunks x 2 q_offsets x 2 arms (same trunk_uid in both arms => paired CRN
design), 2 scales x 3 directions + 6 seeds each; rows are written to the zarr
store in SHUFFLED order plus one orphan row to exercise the episode_id join.
"""
import json
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e1e2_analyze as ana  # noqa: E402

L, F, U, E = 8, 10, 11, 32
B_PRI = 1.0 / np.sqrt(4 * 10 * 32)      # one flipped gate, primary subset
B_SEC = 1.0 / np.sqrt(8 * 11 * 32)      # one flipped gate, secondary subset
TRUNKS = [101, 202]
QOFFS = [-2, 0]
ARMS = ["loop", "control"]
SCALES = [1e-3, 1e-2]
N_DIRS = 3
N_SEEDS = 6
CARRIER = (4, 1)                                     # (layer, token), primary
EXTRA3 = [(4, 2), (4, 3), (4, 4)]
EXTRA15 = [(4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10),
           (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
           (5, 8), (5, 9)]
SEC_ONLY = (0, 0)                                    # outside primary subset


def base_row() -> np.ndarray:
    r = np.zeros((L, F, U, E), np.float16)
    r[..., 0] = 1.0
    return r


def flip(row: np.ndarray, f: int, l: int, u: int, expert: int = 1) -> None:
    row[l, f, u, :] = 0.0
    row[l, f, u, expert] = 1.0


def e1_pert_row(arm: str, scale: float, d: int) -> np.ndarray:
    row = base_row()
    for f in range(F):
        flip(row, f, *CARRIER)
    if arm == "loop":
        extras = EXTRA15 if (scale == 1e-2 and d == 2) else EXTRA3
        for l, u in extras:
            flip(row, 3, l, u)
        flip(row, 3, *SEC_ONLY)
    return row


def e2_seed_row(arm: str, i: int) -> np.ndarray:
    row = base_row()
    if arm == "loop":
        flip(row, 5, 5, 3, expert=i)
    else:
        flip(row, 9, 6, 2, expert=i)
    return row


def build_dataset(root: pathlib.Path):
    """Write routes.zarr (shuffled + 1 orphan row) and manifest.jsonl."""
    rows, manifest = [], []
    eid = 1_000_000
    for trunk in TRUNKS:
        for q in QOFFS:
            for arm in ARMS:
                common = {"snapshot_dir": f"/fake/{trunk}_{q}_{arm}",
                          "trunk_uid": trunk, "q_offset": q, "arm": arm,
                          "noise_sha256": "0" * 64}
                manifest.append({**common, "episode_id": eid, "kind": "e1_base"})
                rows.append(base_row())
                eid += 1
                for scale in SCALES:
                    for d in range(N_DIRS):
                        manifest.append({**common, "episode_id": eid,
                                         "kind": "e1_pert", "scale": scale,
                                         "direction": d, "delta_norm": scale})
                        rows.append(e1_pert_row(arm, scale, d))
                        eid += 1
                for i in range(N_SEEDS):
                    manifest.append({**common, "episode_id": eid,
                                     "kind": "e2_seed", "seed_index": i})
                    rows.append(e2_seed_row(arm, i))
                    eid += 1
    eids = [m["episode_id"] for m in manifest]
    rows.append(base_row())                       # orphan: no manifest line
    eids.append(999_999)

    n = len(rows)
    perm = np.random.default_rng(42).permutation(n)
    probs = np.zeros((n, L, F, U, E), np.float16)
    eid_arr = np.zeros(n, np.int64)
    for src, dst in enumerate(perm):
        probs[dst] = rows[src]
        eid_arr[dst] = eids[src]

    store = root / "routes.zarr"
    g = zarr.open_group(str(store), mode="w")
    a = g.create_array("hb_router_probs", shape=probs.shape, dtype="float16",
                       chunks=(8, L, F, U, E))
    a[:] = probs
    e = g.create_array("episode_id", shape=(n,), dtype="int64")
    e[:] = eid_arr
    c = g.create_array("control_step", shape=(n,), dtype="int64")
    c[:] = 0

    man_dir = root / "collect"
    man_dir.mkdir()
    with open(man_dir / "manifest.jsonl", "w") as fh:
        for m in manifest:
            fh.write(json.dumps(m) + "\n")
    return store, man_dir, manifest


def run_analyzer(store, man_dir, out_dir):
    rc = ana.main(["--server-store", str(store), "--manifest", str(man_dir),
                   "--out", str(out_dir), "--n-boot", "300",
                   "--boot-seed", "7"])
    assert rc == 0
    return types.SimpleNamespace(
        det=pd.read_csv(out_dir / "e1_per_direction.csv"),
        agg=pd.read_csv(out_dir / "e1_aggregate.csv"),
        e2=pd.read_csv(out_dir / "e2_seed_dispersion.csv"),
        summary=json.loads((out_dir / "summary.json").read_text()),
    )


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    root = tmp_path_factory.mktemp("e1e2")
    store, man_dir, manifest = build_dataset(root)
    res = run_analyzer(store, man_dir, root / "out")

    # degraded dataset: only one trunk in each arm -> insufficient_trunks
    man1 = root / "collect_1trunk"
    man1.mkdir()
    with open(man1 / "manifest.jsonl", "w") as fh:
        for m in manifest:
            if m["trunk_uid"] == 101:
                fh.write(json.dumps(m) + "\n")
    res1 = run_analyzer(store, man1, root / "out_1trunk")

    # unpaired dataset: control arm relabeled to disjoint trunk_uids
    man2 = root / "collect_unpaired"
    man2.mkdir()
    with open(man2 / "manifest.jsonl", "w") as fh:
        for m in manifest:
            m2 = dict(m)
            if m2["arm"] == "control":
                m2["trunk_uid"] = m2["trunk_uid"] + 200
            fh.write(json.dumps(m2) + "\n")
    res2 = run_analyzer(store, man2, root / "out_unpaired")

    return types.SimpleNamespace(full=res, one_trunk=res1, unpaired=res2)


def one(df, **kv):
    m = pd.Series(True, index=df.index)
    for k, v in kv.items():
        m &= df[k] == v
    sel = df[m]
    assert len(sel) == 1, f"expected 1 row for {kv}, got {len(sel)}"
    return sel.iloc[0]


def find(records, **kv):
    sel = [r for r in records if all(r[k] == v for k, v in kv.items())]
    assert len(sel) == 1, f"expected 1 record for {kv}, got {len(sel)}"
    return sel[0]


# ------------------------------------------------------------------ the join

def test_join_accounting(ws):
    d = ws.full.summary["data"]
    assert d["n_manifest_lines"] == 104          # 8 groups x (1+6+6) rows
    assert d["n_zarr_rows"] == 105               # + 1 orphan
    assert d["n_joined_episodes"] == 104
    assert d["n_manifest_missing_in_zarr"] == 0
    assert d["n_zarr_rows_unmatched"] == 1
    assert d["F"] == F and d["L"] == L and d["U"] == U and d["E"] == E
    assert d["kinds"] == {"e1_base": 8, "e1_pert": 48, "e2_seed": 48}
    assert d["n_trunks_by_arm"] == {"loop": 2, "control": 2}
    # detail rows: 48 perts x 2 gate subsets
    assert ws.full.summary["e1"]["n_detail_rows"] == 96
    assert len(ws.full.e2) == 16                 # 8 groups x 2 subsets


# ------------------------------------------------------------------------ E1

def test_e1_primary_loop_direction_metrics(ws):
    r = one(ws.full.det, gate_subset="primary", trunk_uid=101, q_offset=-2,
            arm="loop", scale=1e-3, direction=0)
    s = 1e-3
    assert r["G_peak"] == pytest.approx(2 * B_PRI / s, rel=1e-9)
    assert r["G_terminal"] == pytest.approx(B_PRI / s, rel=1e-9)
    assert r["argmax_f"] == 3
    assert r["funnel"] == pytest.approx(0.5, abs=1e-9)
    assert r["lambda"] == pytest.approx(0.0, abs=1e-12)
    # G_f wide columns: peak at f=3, carrier level elsewhere
    assert r["G_f_03"] == pytest.approx(2 * B_PRI / s, rel=1e-9)
    assert r["G_f_00"] == pytest.approx(B_PRI / s, rel=1e-9)
    assert r["G_f_09"] == pytest.approx(B_PRI / s, rel=1e-9)
    # big direction at the big scale: k=16 -> 4x carrier
    r2 = one(ws.full.det, gate_subset="primary", trunk_uid=101, q_offset=-2,
             arm="loop", scale=1e-2, direction=2)
    assert r2["G_peak"] == pytest.approx(4 * B_PRI / 1e-2, rel=1e-9)
    assert r2["argmax_f"] == 3


def test_e1_control_direction_metrics(ws):
    r = one(ws.full.det, gate_subset="primary", trunk_uid=202, q_offset=0,
            arm="control", scale=1e-3, direction=1)
    assert r["G_peak"] == pytest.approx(B_PRI / 1e-3, rel=1e-9)
    assert r["G_peak"] == pytest.approx(r["G_terminal"], rel=1e-12)
    assert r["argmax_f"] == 0                    # flat gains, first index wins
    assert r["funnel"] == pytest.approx(0.0, abs=1e-12)
    assert r["lambda"] == pytest.approx(0.0, abs=1e-12)


def test_e1_secondary_subset_sees_the_extra_gate(ws):
    # secondary includes the (l=0,u=0) flip: k_sec(3)=5 while k_pri(3)=4
    r = one(ws.full.det, gate_subset="secondary", trunk_uid=101, q_offset=-2,
            arm="loop", scale=1e-3, direction=0)
    assert r["G_peak"] == pytest.approx(np.sqrt(5) * B_SEC / 1e-3, rel=1e-9)
    assert r["G_terminal"] == pytest.approx(B_SEC / 1e-3, rel=1e-9)
    assert r["argmax_f"] == 3
    assert r["funnel"] == pytest.approx(1 - 1 / np.sqrt(5), rel=1e-9)


def test_e1_aggregate_median_iqr(ws):
    # scale 1e-2, loop: G_peak over dirs = [200b, 200b, 400b]
    a = one(ws.full.agg, gate_subset="primary", trunk_uid=101, q_offset=-2,
            arm="loop", scale=1e-2)
    assert a["n_directions"] == 3
    assert a["G_peak_median"] == pytest.approx(2 * B_PRI / 1e-2, rel=1e-9)
    assert a["G_peak_iqr"] == pytest.approx(B_PRI / 1e-2, rel=1e-9)  # 300b-200b
    # scale 1e-3: identical dirs -> IQR exactly 0
    b = one(ws.full.agg, gate_subset="primary", trunk_uid=101, q_offset=-2,
            arm="loop", scale=1e-3)
    assert b["G_peak_median"] == pytest.approx(2 * B_PRI / 1e-3, rel=1e-9)
    assert b["G_peak_iqr"] == pytest.approx(0.0, abs=1e-9)
    assert b["funnel_median"] == pytest.approx(0.5, abs=1e-9)
    assert b["lambda_median"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------------ E2

def test_e2_loop_mid_flow_fanout_terminal_convergence(ws):
    r = one(ws.full.e2, gate_subset="primary", trunk_uid=101, q_offset=-2,
            arm="loop")
    assert r["n_seeds"] == N_SEEDS
    assert r["fan_out"] == pytest.approx(B_PRI, rel=1e-9)
    assert r["terminal_diversity"] == pytest.approx(0.0, abs=1e-12)
    assert r["funnel_index"] == pytest.approx(1.0, abs=1e-9)
    assert r["argmax_f"] == 5
    assert r["D_seed_f_05"] == pytest.approx(B_PRI, rel=1e-9)
    assert r["D_seed_f_00"] == pytest.approx(0.0, abs=1e-12)
    assert r["D_seed_f_09"] == pytest.approx(0.0, abs=1e-12)


def test_e2_control_terminal_divergence(ws):
    r = one(ws.full.e2, gate_subset="primary", trunk_uid=202, q_offset=0,
            arm="control")
    assert r["fan_out"] == pytest.approx(B_PRI, rel=1e-9)
    assert r["terminal_diversity"] == pytest.approx(B_PRI, rel=1e-9)
    assert r["funnel_index"] == pytest.approx(0.0, abs=1e-9)
    assert r["argmax_f"] == 9


def test_e2_secondary_normalization(ws):
    r = one(ws.full.e2, gate_subset="secondary", trunk_uid=101, q_offset=-2,
            arm="loop")
    assert r["fan_out"] == pytest.approx(B_SEC, rel=1e-9)
    assert r["funnel_index"] == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------------- summaries

def test_summary_e1_paired_bootstrap(ws):
    comps = ws.full.summary["comparisons"]["primary"]["e1"]
    g = find(comps, q_offset=-2, scale=1e-3, metric="G_peak")
    assert g["method"] == "paired_trunk_bootstrap" and g["status"] == "ok"
    assert g["n_trunks"] == 2
    # loop-control per-direction diff = (2b-b)/1e-3, identical trunks -> exact
    assert g["estimate"] == pytest.approx(B_PRI / 1e-3, rel=1e-9)
    assert g["ci_lo"] == pytest.approx(g["estimate"], rel=1e-9)
    assert g["ci_hi"] == pytest.approx(g["estimate"], rel=1e-9)
    assert g["loop"]["n_trunks"] == 2 and g["control"]["n_trunks"] == 2
    # big scale: per-trunk mean diff = (100b+100b+300b)/3
    g2 = find(comps, q_offset=-2, scale=1e-2, metric="G_peak")
    assert g2["estimate"] == pytest.approx(500 * B_PRI / 3, rel=1e-9)
    f = find(comps, q_offset=-2, scale=1e-3, metric="funnel")
    assert f["estimate"] == pytest.approx(0.5, abs=1e-9)
    lam = find(comps, q_offset=-2, scale=1e-3, metric="lambda")
    assert lam["estimate"] == pytest.approx(0.0, abs=1e-12)


def test_summary_e2_paired_bootstrap(ws):
    comps = ws.full.summary["comparisons"]["primary"]["e2"]
    fan = find(comps, q_offset=-2, metric="fan_out")
    assert fan["method"] == "paired_trunk_bootstrap"
    assert fan["estimate"] == pytest.approx(0.0, abs=1e-12)   # both arms b
    fi = find(comps, q_offset=-2, metric="funnel_index")
    assert fi["estimate"] == pytest.approx(1.0, abs=1e-9)     # 1.0 vs ~0
    assert fi["ci_lo"] == pytest.approx(1.0, abs=1e-9)
    assert fi["n_trunks"] == 2
    assert ws.full.summary["insufficient_trunks"] is False
    assert ws.full.summary["n_comparisons_insufficient"] == 0


def test_summary_secondary_subset_comparison(ws):
    comps = ws.full.summary["comparisons"]["secondary"]["e1"]
    g = find(comps, q_offset=0, scale=1e-3, metric="G_peak")
    assert g["estimate"] == pytest.approx((np.sqrt(5) - 1) * B_SEC / 1e-3,
                                          rel=1e-9)


def test_insufficient_trunks_degradation(ws):
    s = ws.one_trunk.summary
    assert s["insufficient_trunks"] is True
    assert s["n_comparisons_insufficient"] == s["n_comparisons"] > 0
    g = find(s["comparisons"]["primary"]["e1"], q_offset=-2, scale=1e-3,
             metric="G_peak")
    assert g["status"] == "insufficient_trunks"
    assert g["method"] == "descriptive_only"
    assert g["ci_lo"] is None and g["ci_hi"] is None
    assert g["n_trunks"] == {"loop": 1, "control": 1}
    # descriptive difference of arm means is still reported
    assert g["estimate"] == pytest.approx(B_PRI / 1e-3, rel=1e-9)
    # per-direction / dispersion tables are still fully produced
    assert len(ws.one_trunk.det) == 48           # 24 perts x 2 subsets
    assert len(ws.one_trunk.e2) == 8


def test_unpaired_conservative_path(ws):
    s = ws.unpaired.summary
    assert s["data"]["n_trunks_by_arm"] == {"loop": 2, "control": 2}
    g = find(s["comparisons"]["primary"]["e1"], q_offset=-2, scale=1e-3,
             metric="G_peak")
    assert g["method"] == "unpaired_conservative" and g["status"] == "ok"
    assert g["estimate"] == pytest.approx(B_PRI / 1e-3, rel=1e-9)
    # identical trunks per arm -> degenerate per-arm CIs -> tight interval
    assert g["ci_lo"] == pytest.approx(g["estimate"], rel=1e-9)
    assert g["ci_hi"] == pytest.approx(g["estimate"], rel=1e-9)
    assert g["n_trunks"] == {"loop": 2, "control": 2}
    assert g["loop_bootstrap"]["n_trunks"] == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
