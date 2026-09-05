#!/usr/bin/env python
"""E1/E2 analyzer: join a serve_with_recorder legacy routes.zarr with the
e1e2_collect.py client manifest and compute the EXPERIMENT_PROGRAM metrics.

Alignment contracts
-------------------
routes.zarr (legacy --store-full-probs mode)
    hb_router_probs [N, L=8, D=F, U=11, E=32] float16   (one row = one inference)
    episode_id      [N] int                              (client-owned id space)
    control_step    [N] int                              (unused here)
manifest.jsonl (one line per inference, written by e1e2_collect.py)
    episode_id, kind in {e1_base, e1_pert, e2_seed}, trunk_uid, q_offset, arm,
    and per kind: scale/direction/delta_norm (e1_pert), seed_index (e2_seed).

Per row we transpose [L, D, U, E] -> [F=D, L, U, E] and evaluate every metric
on two gate subsets:
    primary   = storage layers 4..7 (model layers 12-15) x tokens 1..10
                (action tokens; token 0 is the state token)
    secondary = all 8 layers x all 11 tokens
All numerics come from control_metrics (flatten_query / perturbation_response /
finite_time_contraction / seed_dispersion / fanout_funnel / trunk_bootstrap);
nothing is re-implemented here.

Outputs (under --out)
    e1_per_direction.csv    one row per (gate_subset, trunk, q_offset, arm,
                            scale, direction): G_peak/G_terminal/funnel/
                            argmax_f/lambda + the G_f curve (wide columns)
    e1_aggregate.csv        median & IQR over directions per (gate_subset,
                            trunk, q_offset, arm, scale)
    e2_seed_dispersion.csv  one row per (gate_subset, trunk, q_offset, arm):
                            fan_out/terminal_diversity/funnel_index/argmax_f +
                            the D_seed(f) curve (wide columns)
    summary.json            data-join accounting + loop-vs-control comparisons
                            per q_offset (E1 additionally per scale) via
                            control_metrics.trunk_bootstrap with trunk_uid as
                            the independent unit.

Comparison methods (recorded per cell in summary.json):
    paired_trunk_bootstrap        both arms share trunk_uids (common-random-
                                  numbers design): per-trunk paired differences
                                  (per shared direction for E1) are bootstrapped
                                  over trunks.
    unpaired_conservative         disjoint trunk sets, >=2 trunks per arm: each
                                  arm bootstrapped separately; the difference CI
                                  is the conservative interval
                                  [lo_loop - hi_ctrl, hi_loop - lo_ctrl].
    descriptive_only              <2 usable trunks in an arm: no CI, status is
                                  marked "insufficient_trunks".
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import time

import numpy as np
import pandas as pd
import zarr

import control_metrics as cm

# gate subsets on the [F, L, U, E] tensor (slice first, then flatten_query)
PRIMARY_LAYERS = slice(4, 8)    # storage layers 4..7 == model layers 12..15
PRIMARY_TOKENS = slice(1, 11)   # action tokens (token 0 = state token)
GATE_SUBSETS = ("primary", "secondary")

E1_METRICS = ("G_peak", "funnel", "lambda")
E2_METRICS = ("fan_out", "funnel_index")

EXPECTED_L = 8
EXPECTED_U = 11


# ---------------------------------------------------------------- I/O helpers

def load_manifest(path: str | pathlib.Path, warnings: list[str]) -> tuple[list[dict], dict]:
    """Read manifest.jsonl (accepts the collect dir or the file itself);
    de-duplicate by episode_id keeping the last record."""
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / "manifest.jsonl"
    if not p.exists():
        raise SystemExit(f"manifest not found: {p}")
    by_eid: dict[int, dict] = {}
    n_lines = 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        n_lines += 1
        rec = json.loads(line)
        by_eid[int(rec["episode_id"])] = rec
    n_dup = n_lines - len(by_eid)
    if n_dup:
        warnings.append(f"manifest: {n_dup} duplicate episode_id lines (kept last)")
    stats = {"manifest_path": str(p), "n_manifest_lines": n_lines,
             "n_manifest_unique_episodes": len(by_eid),
             "n_manifest_duplicate_lines": n_dup}
    return list(by_eid.values()), stats


def open_routes(path: str | pathlib.Path, warnings: list[str]):
    """Open legacy routes.zarr; return (probs_array, eid->row map, stats)."""
    root = zarr.open(str(path), mode="r")
    try:
        probs = root["hb_router_probs"]
        eids = np.asarray(root["episode_id"][:], dtype=np.int64)
    except (KeyError, TypeError) as ex:
        raise SystemExit(f"{path}: not a legacy routes.zarr store "
                         f"(need hb_router_probs + episode_id): {ex}")
    if probs.ndim != 5:
        raise SystemExit(f"hb_router_probs must be [N,L,D,U,E], got {probs.shape}")
    if probs.shape[0] != len(eids):
        raise SystemExit(f"row mismatch: hb_router_probs N={probs.shape[0]} "
                         f"vs episode_id N={len(eids)}")
    n, L, F, U, E = probs.shape
    if L != EXPECTED_L or U != EXPECTED_U:
        raise SystemExit(f"unexpected row layout [L={L},D={F},U={U},E={E}]; "
                         f"contract is [L={EXPECTED_L},D=F,U={EXPECTED_U},E]")
    if "control_step" not in root:
        warnings.append("routes.zarr: control_step array missing (join uses "
                        "episode_id only; continuing)")
    row_of: dict[int, int] = {}
    for i, e in enumerate(eids):
        row_of[int(e)] = i          # keep-last on duplicates (retry-after-crash)
    n_dup = n - len(row_of)
    if n_dup:
        warnings.append(f"routes.zarr: {n_dup} duplicate episode_id rows "
                        "(kept last occurrence)")
    stats = {"server_store": str(path), "n_zarr_rows": int(n),
             "n_zarr_unique_episodes": len(row_of),
             "n_zarr_duplicate_episodes": int(n_dup),
             "F": int(F), "L": int(L), "U": int(U), "E": int(E)}
    return probs, row_of, stats


def fetch_rows(probs, row_of: dict[int, int], eids: list[int],
               warnings: list[str], ctx: str) -> dict[int, np.ndarray]:
    """Batch-read the zarr rows for the given episode_ids; missing ids are
    reported once and omitted from the result."""
    want, rows = [], []
    missing = []
    for e in eids:
        e = int(e)
        r = row_of.get(e)
        if r is None:
            missing.append(e)
        else:
            want.append(e)
            rows.append(r)
    if missing:
        warnings.append(f"{ctx}: {len(missing)} manifest episode(s) missing "
                        f"from routes.zarr (e.g. {missing[:5]}); skipped")
    if not want:
        return {}
    order = np.argsort(rows, kind="stable")
    sorted_rows = np.asarray(rows, dtype=np.int64)[order]
    try:
        data = probs.oindex[sorted_rows]
    except Exception:                                    # fallback: per-row read
        data = np.stack([np.asarray(probs[int(i)]) for i in sorted_rows])
    out: dict[int, np.ndarray] = {}
    for pos, o in enumerate(order):
        out[want[int(o)]] = np.asarray(data[pos])
    return out


def row_to_flue(row: np.ndarray) -> np.ndarray:
    """One zarr row [L, D, U, E] -> [F=D, L, U, E] float32."""
    arr = np.asarray(row, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] != EXPECTED_L or arr.shape[2] != EXPECTED_U:
        raise ValueError(f"expected [L={EXPECTED_L},D,U={EXPECTED_U},E] row, "
                         f"got {arr.shape}")
    return arr.transpose(1, 0, 2, 3)


def subset_psis(flue: np.ndarray) -> dict[str, np.ndarray]:
    """Slice each gate subset out of [F, L, U, E], then flatten to Psi[F, .]."""
    return {
        "primary": cm.flatten_query(flue[:, PRIMARY_LAYERS, PRIMARY_TOKENS, :]),
        "secondary": cm.flatten_query(flue),
    }


# ------------------------------------------------------------------- E1 / E2

def analyze_e1(groups: dict, probs, row_of, warnings: list[str], F: int) -> list[dict]:
    detail: list[dict] = []
    for (trunk, q, arm), grp in sorted(groups.items()):
        base, perts = grp["base"], grp["perts"]
        ctx = f"e1 trunk={trunk} q={q:+d} arm={arm}"
        if base is None:
            warnings.append(f"{ctx}: no e1_base row; skipped {len(perts)} perts")
            continue
        if not perts:
            warnings.append(f"{ctx}: e1_base without e1_pert rows; skipped")
            continue
        base_eid = int(base["episode_id"])
        eids = [base_eid] + [int(p["episode_id"]) for p in perts]
        rows = fetch_rows(probs, row_of, eids, warnings, ctx)
        if base_eid not in rows:
            warnings.append(f"{ctx}: base episode {base_eid} missing from "
                            "routes.zarr; group skipped")
            continue
        psi_base = subset_psis(row_to_flue(rows[base_eid]))
        for p in sorted(perts, key=lambda r: (float(r.get("scale", np.nan)),
                                              int(r.get("direction", -1)))):
            peid = int(p["episode_id"])
            if peid not in rows:
                continue                       # already warned by fetch_rows
            dn = p.get("delta_norm")
            if dn is None or not math.isfinite(float(dn)) or float(dn) <= 0:
                warnings.append(f"{ctx}: episode {peid} has invalid "
                                f"delta_norm={dn!r}; skipped")
                continue
            psi_pert = subset_psis(row_to_flue(rows[peid]))
            for sname in GATE_SUBSETS:
                r = cm.perturbation_response(psi_base[sname], psi_pert[sname],
                                             float(dn))
                try:
                    lam = cm.finite_time_contraction(r["D_f"])
                except ValueError:
                    lam = float("nan")
                rec = {
                    "gate_subset": sname, "trunk_uid": int(trunk),
                    "q_offset": int(q), "arm": str(arm),
                    "scale": float(p["scale"]), "direction": int(p["direction"]),
                    "delta_norm": float(dn), "episode_id": peid,
                    "base_episode_id": base_eid,
                    "G_peak": float(r["G_peak"]),
                    "G_terminal": float(r["G_terminal"]),
                    "funnel": float(r["funnel"]),
                    "argmax_f": int(r["argmax_f"]),
                    "lambda": float(lam),
                }
                for f in range(F):
                    rec[f"G_f_{f:02d}"] = float(r["G_f"][f])
                detail.append(rec)
    return detail


def aggregate_e1(det: pd.DataFrame) -> list[dict]:
    """Per (gate_subset, trunk, q_offset, arm, scale): median & IQR over the
    perturbation directions."""
    out: list[dict] = []
    keys = ["gate_subset", "trunk_uid", "q_offset", "arm", "scale"]
    for key, g in det.groupby(keys, sort=True):
        rec = dict(zip(keys, key))
        rec["n_directions"] = int(g["direction"].nunique())
        for m in ("G_peak", "G_terminal", "funnel", "lambda"):
            v = g[m].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                q25, q75 = np.quantile(v, [0.25, 0.75])
                rec[f"{m}_median"] = float(np.median(v))
                rec[f"{m}_iqr"] = float(q75 - q25)
            else:
                rec[f"{m}_median"] = float("nan")
                rec[f"{m}_iqr"] = float("nan")
        rec["argmax_f_median"] = float(np.median(g["argmax_f"].to_numpy(dtype=float)))
        out.append(rec)
    return out


def analyze_e2(groups: dict, probs, row_of, warnings: list[str], F: int) -> list[dict]:
    out: list[dict] = []
    for (trunk, q, arm), seeds in sorted(groups.items()):
        ctx = f"e2 trunk={trunk} q={q:+d} arm={arm}"
        seeds = sorted(seeds, key=lambda r: int(r.get("seed_index", -1)))
        eids = [int(s["episode_id"]) for s in seeds]
        rows = fetch_rows(probs, row_of, eids, warnings, ctx)
        present = [e for e in eids if e in rows]
        if len(present) < 2:
            warnings.append(f"{ctx}: {len(present)} seed row(s) available "
                            "(<2); group skipped")
            continue
        psis: dict[str, list[np.ndarray]] = {s: [] for s in GATE_SUBSETS}
        for e in present:
            ps = subset_psis(row_to_flue(rows[e]))
            for sname in GATE_SUBSETS:
                psis[sname].append(ps[sname])
        for sname in GATE_SUBSETS:
            d = cm.seed_dispersion(psis[sname])
            ff = cm.fanout_funnel(d)
            rec = {
                "gate_subset": sname, "trunk_uid": int(trunk),
                "q_offset": int(q), "arm": str(arm),
                "n_seeds": len(present),
                "fan_out": float(ff["fan_out"]),
                "terminal_diversity": float(ff["terminal_diversity"]),
                "funnel_index": float(ff["funnel_index"]),
                "argmax_f": int(ff["argmax_f"]),
            }
            for f in range(F):
                rec[f"D_seed_f_{f:02d}"] = float(d[f])
            out.append(rec)
    return out


# ---------------------------------------------------------------- comparisons

def _describe_arm(m: dict[int, dict]) -> dict:
    """m: {trunk_uid: {unit_key: value}} -> per-arm descriptive stats."""
    if not m:
        return {"n_trunks": 0, "mean_of_trunk_means": None}
    tm = [float(np.mean(list(units.values()))) for units in m.values()]
    return {"n_trunks": len(m), "mean_of_trunk_means": float(np.mean(tm))}


def compare_arms(loop_map: dict[int, dict], ctrl_map: dict[int, dict],
                 n_boot: int, seed: int) -> dict:
    """loop-minus-control comparison with trunk_uid as the independent unit.
    Inputs: {trunk_uid: {unit_key: finite value}} per arm."""
    desc = {"loop": _describe_arm(loop_map), "control": _describe_arm(ctrl_map)}
    # paired path: shared trunk_uids (collector CRN design shares the noise
    # namespace between arms of one matched pair)
    paired: dict[int, list[float]] = {}
    for t in sorted(set(loop_map) & set(ctrl_map)):
        ks = sorted(set(loop_map[t]) & set(ctrl_map[t]), key=str)
        diffs = [loop_map[t][k] - ctrl_map[t][k] for k in ks]
        if diffs:
            paired[t] = diffs
    if len(paired) >= 2:
        bs = cm.trunk_bootstrap(paired, n_boot=n_boot, seed=seed)
        return {"method": "paired_trunk_bootstrap", "status": "ok",
                "estimate": bs["estimate"], "ci_lo": bs["ci_lo"],
                "ci_hi": bs["ci_hi"], "n_trunks": bs["n_trunks"], **desc}
    # unpaired path: disjoint trunk sets, each arm bootstrapped on its own
    if len(loop_map) >= 2 and len(ctrl_map) >= 2:
        bl = cm.trunk_bootstrap({t: list(v.values()) for t, v in loop_map.items()},
                                n_boot=n_boot, seed=seed)
        bc = cm.trunk_bootstrap({t: list(v.values()) for t, v in ctrl_map.items()},
                                n_boot=n_boot, seed=seed + 1)
        return {"method": "unpaired_conservative", "status": "ok",
                "estimate": bl["estimate"] - bc["estimate"],
                "ci_lo": bl["ci_lo"] - bc["ci_hi"],
                "ci_hi": bl["ci_hi"] - bc["ci_lo"],
                "n_trunks": {"loop": bl["n_trunks"], "control": bc["n_trunks"]},
                "loop_bootstrap": bl, "control_bootstrap": bc, **desc}
    # graceful degradation: descriptive statistics only
    lm = desc["loop"]["mean_of_trunk_means"]
    cmn = desc["control"]["mean_of_trunk_means"]
    est = (lm - cmn) if (lm is not None and cmn is not None) else None
    return {"method": "descriptive_only", "status": "insufficient_trunks",
            "estimate": est, "ci_lo": None, "ci_hi": None,
            "n_trunks": {"loop": len(loop_map), "control": len(ctrl_map)},
            **desc}


def _e1_arm_map(det: pd.DataFrame, subset: str, q: int, scale: float,
                metric: str, arm: str) -> dict[int, dict]:
    sub = det[(det["gate_subset"] == subset) & (det["q_offset"] == q)
              & (det["scale"] == scale) & (det["arm"] == arm)]
    out: dict[int, dict] = {}
    for t, g in sub.groupby("trunk_uid"):
        vals = {int(d): float(v) for d, v in zip(g["direction"], g[metric])
                if np.isfinite(float(v))}
        if vals:
            out[int(t)] = vals
    return out


def _e2_arm_map(e2: pd.DataFrame, subset: str, q: int, metric: str,
                arm: str) -> dict[int, dict]:
    sub = e2[(e2["gate_subset"] == subset) & (e2["q_offset"] == q)
             & (e2["arm"] == arm)]
    out: dict[int, dict] = {}
    for t, g in sub.groupby("trunk_uid"):
        v = float(g[metric].iloc[-1])
        if np.isfinite(v):
            out[int(t)] = {"all": v}
    return out


def build_comparisons(det: pd.DataFrame | None, e2: pd.DataFrame | None,
                      loop_arm: str, control_arm: str, n_boot: int,
                      seed: int) -> dict:
    comps: dict = {s: {"e1": [], "e2": []} for s in GATE_SUBSETS}
    if det is not None and len(det):
        cells = sorted({(int(q), float(s))
                        for q, s in zip(det["q_offset"], det["scale"])})
        for sname in GATE_SUBSETS:
            for q, sc in cells:
                for metric in E1_METRICS:
                    lm = _e1_arm_map(det, sname, q, sc, metric, loop_arm)
                    cmap = _e1_arm_map(det, sname, q, sc, metric, control_arm)
                    rec = compare_arms(lm, cmap, n_boot, seed)
                    comps[sname]["e1"].append(
                        {"q_offset": q, "scale": sc, "metric": metric, **rec})
    if e2 is not None and len(e2):
        qs = sorted({int(q) for q in e2["q_offset"]})
        for sname in GATE_SUBSETS:
            for q in qs:
                for metric in E2_METRICS:
                    lm = _e2_arm_map(e2, sname, q, metric, loop_arm)
                    cmap = _e2_arm_map(e2, sname, q, metric, control_arm)
                    rec = compare_arms(lm, cmap, n_boot, seed)
                    comps[sname]["e2"].append(
                        {"q_offset": q, "metric": metric, **rec})
    return comps


# ------------------------------------------------------------------- driver

def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (np.integer, int)) and not isinstance(o, bool):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def _write_csv(records: list[dict], path: pathlib.Path, lead_cols: list[str],
               sort_cols: list[str]) -> pd.DataFrame:
    if not records:
        df = pd.DataFrame(columns=lead_cols)
        df.to_csv(path, index=False)
        return df
    df = pd.DataFrame(records)
    cols = lead_cols + [c for c in df.columns if c not in lead_cols]
    df = df[cols].sort_values(sort_cols, kind="stable").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    warnings: list[str] = []
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, man_stats = load_manifest(args.manifest, warnings)
    probs, row_of, store_stats = open_routes(args.server_store, warnings)
    F = store_stats["F"]

    man_eids = {int(r["episode_id"]) for r in records}
    matched_eids = man_eids & set(row_of)
    n_missing = len(man_eids) - len(matched_eids)
    n_unmatched_zarr = len(set(row_of) - man_eids)
    if n_missing:
        sample = sorted(man_eids - set(row_of))[:10]
        warnings.append(f"join: {n_missing} manifest episode(s) have no "
                        f"routes.zarr row (e.g. {sample})")
    if n_unmatched_zarr:
        warnings.append(f"join: {n_unmatched_zarr} routes.zarr row(s) not in "
                        "the manifest (warmup/other clients); ignored")

    kinds = collections.Counter(str(r.get("kind")) for r in records)
    for k in kinds:
        if k not in ("e1_base", "e1_pert", "e2_seed"):
            warnings.append(f"manifest: {kinds[k]} row(s) of unknown kind "
                            f"{k!r}; ignored")
    trunks_by_arm: dict[str, set] = collections.defaultdict(set)
    for r in records:
        trunks_by_arm[str(r.get("arm", "?"))].add(int(r["trunk_uid"]))

    # group manifest rows -------------------------------------------------
    e1_groups: dict[tuple, dict] = {}
    e2_groups: dict[tuple, list] = collections.defaultdict(list)
    for r in records:
        kind = str(r.get("kind"))
        if kind not in ("e1_base", "e1_pert", "e2_seed"):
            continue
        key = (int(r["trunk_uid"]), int(r["q_offset"]), str(r.get("arm", "?")))
        if kind == "e2_seed":
            e2_groups[key].append(r)
            continue
        grp = e1_groups.setdefault(key, {"base": None, "perts": []})
        if kind == "e1_base":
            if grp["base"] is not None:
                warnings.append(f"e1 trunk={key[0]} q={key[1]:+d} arm={key[2]}: "
                                "multiple e1_base rows; kept last")
            grp["base"] = r
        else:
            grp["perts"].append(r)

    # E1 -------------------------------------------------------------------
    e1_detail = analyze_e1(e1_groups, probs, row_of, warnings, F)
    lead = ["gate_subset", "trunk_uid", "q_offset", "arm", "scale", "direction",
            "delta_norm", "episode_id", "base_episode_id", "G_peak",
            "G_terminal", "funnel", "argmax_f", "lambda"]
    det_df = _write_csv(e1_detail, out_dir / "e1_per_direction.csv", lead,
                        ["gate_subset", "trunk_uid", "q_offset", "arm",
                         "scale", "direction"])
    e1_agg = aggregate_e1(det_df) if len(det_df) else []
    agg_lead = ["gate_subset", "trunk_uid", "q_offset", "arm", "scale",
                "n_directions"]
    agg_df = _write_csv(e1_agg, out_dir / "e1_aggregate.csv", agg_lead,
                        ["gate_subset", "trunk_uid", "q_offset", "arm", "scale"])

    # E2 -------------------------------------------------------------------
    e2_recs = analyze_e2(e2_groups, probs, row_of, warnings, F)
    e2_lead = ["gate_subset", "trunk_uid", "q_offset", "arm", "n_seeds",
               "fan_out", "terminal_diversity", "funnel_index", "argmax_f"]
    e2_df = _write_csv(e2_recs, out_dir / "e2_seed_dispersion.csv", e2_lead,
                       ["gate_subset", "trunk_uid", "q_offset", "arm"])

    # loop-vs-control statistics ------------------------------------------
    comps = build_comparisons(det_df if len(det_df) else None,
                              e2_df if len(e2_df) else None,
                              args.loop_arm, args.control_arm,
                              args.n_boot, args.boot_seed)
    all_comp = [c for s in GATE_SUBSETS for kind in ("e1", "e2")
                for c in comps[s][kind]]
    n_insufficient = sum(c["status"] == "insufficient_trunks" for c in all_comp)

    summary = {
        "generated": time.strftime("%FT%TZ", time.gmtime()),
        "config": {
            "server_store": str(args.server_store),
            "manifest": man_stats.pop("manifest_path"),
            "out": str(out_dir),
            "loop_arm": args.loop_arm, "control_arm": args.control_arm,
            "n_boot": args.n_boot, "boot_seed": args.boot_seed,
            "gate_subsets": {
                "primary": {"storage_layers": [4, 5, 6, 7],
                            "model_layers": [12, 13, 14, 15],
                            "tokens": list(range(1, 11)),
                            "note": "action tokens only (token 0 = state)"},
                "secondary": {"storage_layers": list(range(8)),
                              "tokens": list(range(11))},
            },
        },
        "data": {
            **man_stats, **store_stats,
            "kinds": dict(kinds),
            "n_joined_episodes": len(matched_eids),
            "n_manifest_missing_in_zarr": n_missing,
            "n_zarr_rows_unmatched": n_unmatched_zarr,
            "n_trunks_by_arm": {a: len(t) for a, t in
                                sorted(trunks_by_arm.items())},
            "q_offsets": sorted({int(r["q_offset"]) for r in records}),
            "scales": sorted({float(r["scale"]) for r in records
                              if r.get("scale") is not None}),
            "arms": sorted(trunks_by_arm),
        },
        "e1": {"n_detail_rows": len(det_df), "n_groups": len(e1_groups),
               "n_aggregate_rows": len(agg_df)},
        "e2": {"n_groups": len(e2_groups), "n_rows": len(e2_df)},
        "comparisons": comps,
        "insufficient_trunks": bool(all_comp) and n_insufficient == len(all_comp),
        "n_comparisons": len(all_comp),
        "n_comparisons_insufficient": n_insufficient,
        "warnings": warnings,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=1) + "\n")

    print(f"[e1e2_analyze] joined {len(matched_eids)}/{len(man_eids)} manifest "
          f"episodes (F={F}); e1 detail rows {len(det_df)}, e2 rows "
          f"{len(e2_df)}; comparisons {len(all_comp)} "
          f"({n_insufficient} insufficient_trunks); "
          f"{len(warnings)} warning(s); wrote {out_dir}")
    for w in warnings:
        print(f"  [warn] {w}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--server-store", required=True,
                    help="routes.zarr from serve_with_recorder legacy mode")
    ap.add_argument("--manifest", required=True,
                    help="e1e2_collect output dir (or manifest.jsonl path)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="bootstrap resamples for trunk_bootstrap")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--loop-arm", default="loop",
                    help="manifest arm label of the loop condition")
    ap.add_argument("--control-arm", default="control",
                    help="manifest arm label of the matched-control condition")
    args = ap.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
