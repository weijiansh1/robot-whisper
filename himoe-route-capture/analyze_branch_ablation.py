"""Paired comparison of the four HB-MoE branch-ablation arms.

Every arm runs the same 50 init states with the same flow-noise seeds, so each
episode has a matched partner in every other arm and the right test is McNemar's
on the discordant pairs, not a two-sample proportion test.  Both are reported:
the unpaired Fisher p is the number you would get by throwing the pairing away,
and it is always the weaker of the two.

Arms:
  none        both branches           block(x) = shared(x) + routed(x)
  shared_off  routed branch only      block(x) = routed(x)
  routed_off  shared branch only      block(x) = shared(x)
  block_off   neither                 block(x) = 0
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy import stats

ARMS = ["none", "shared_off", "routed_off", "block_off"]
LABEL = {
    "none": "未改动（基线）",
    "shared_off": "shared 置零（只剩 routed）",
    "routed_off": "routed 置零（只剩 shared）",
    "block_off": "整个 block 置零",
}


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(base: np.ndarray, other: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact McNemar.  b = base ok -> other fail, c = base fail -> other ok."""
    b = int(np.sum(base & ~other))
    c = int(np.sum(~base & other))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = float(stats.binomtest(min(b, c), n, 0.5).pvalue)
    return b, c, min(1.0, p)


def load(run_dir: pathlib.Path) -> tuple[np.ndarray, list[dict]]:
    rows = json.loads((run_dir / "summaries.json").read_text())
    rows.sort(key=lambda r: (r["init_state_id"], r["flow_noise_seed"]))
    return np.array([bool(r["success"]) for r in rows]), rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--prefix", default="abl2-")
    ap.add_argument("--out", default="analysis/branch-ablation")
    args = ap.parse_args()

    root = pathlib.Path(args.runs_root)
    data, meta = {}, {}
    for arm in ARMS:
        d = root / (args.prefix + arm)
        if not (d / "summaries.json").exists():
            print("missing:", d)
            continue
        data[arm], meta[arm] = load(d)

    if "none" not in data:
        raise SystemExit("baseline arm 'none' is required for the paired tests")

    base, base_rows = data["none"], meta["none"]
    n = len(base)
    keys = [(r["init_state_id"], r["flow_noise_seed"]) for r in base_rows]

    report = {"n": n, "arms": {}}
    print("=== 分支消融（released-left，配对 %d 集）===\n" % n)
    print("%-28s %-9s %-18s %-22s %s" % ("条件", "成功", "Wilson 95%", "McNemar (降/升)", "Fisher p"))
    for arm in ARMS:
        if arm not in data:
            continue
        y = data[arm]
        if [(r["init_state_id"], r["flow_noise_seed"]) for r in meta[arm]] != keys:
            raise SystemExit(f"arm {arm} does not share the baseline's episode keys; not paired")
        k = int(y.sum())
        lo, hi = wilson(k, n)
        entry = {"success": k, "n": n, "rate": k / n, "wilson95": [lo, hi]}
        if arm == "none":
            print("%-28s %2d/%-6d [%.1f%%, %.1f%%]%s %-22s %s"
                  % (LABEL[arm], k, n, 100 * lo, 100 * hi, "", "—", "—"))
        else:
            b, c, p = mcnemar_exact(base, y)
            table = [[k, n - k], [int(base.sum()), n - int(base.sum())]]
            fisher = float(stats.fisher_exact(table)[1])
            entry.update({"mcnemar_down": b, "mcnemar_up": c, "mcnemar_p": p,
                          "fisher_p": fisher,
                          "delta_points": 100 * (k - int(base.sum())) / n})
            print("%-28s %2d/%-6d [%.1f%%, %.1f%%]  p=%-6.3f (%2d↓/%2d↑)   %.3f"
                  % (LABEL[arm], k, n, 100 * lo, 100 * hi, p, b, c, fisher))
        report["arms"][arm] = entry

    # decomposition read-out
    if all(a in data for a in ARMS):
        r = {a: data[a].sum() / n for a in ARMS}
        print("\n=== 分解 ===")
        print("both on (none)        %.0f%%" % (100 * r["none"]))
        print("routed only (sh_off)  %.0f%%   相对基线 %+.0f 点" % (100 * r["shared_off"], 100 * (r["shared_off"] - r["none"])))
        print("shared only (rt_off)  %.0f%%   相对基线 %+.0f 点" % (100 * r["routed_off"], 100 * (r["routed_off"] - r["none"])))
        print("neither (block_off)   %.0f%%   相对基线 %+.0f 点" % (100 * r["block_off"], 100 * (r["block_off"] - r["none"])))
        report["decomposition"] = {a: float(r[a]) for a in ARMS}

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "branch_ablation.json").write_text(json.dumps(report, indent=2))
    print("\nwrote", out / "branch_ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
