"""Aggregate all controls/exp-*/report.json into one markdown table (rescue rates per strategy and setting)."""
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
lines = ["# Control experiments summary", "", "| experiment | branch | trigger | control | noise | strategy | rescued / failure branches | rate | success parents kept | total success rate |", "|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
for rep in sorted(glob.glob(str(ROOT / "exp-*/report.json"))):
    name = Path(rep).parent.name
    r = json.load(open(rep))
    m = r.pop("_meta", {})
    trigger = "online" if m.get("branch") == "online" else "hindsight"
    for s, v in sorted(r.items(), key=lambda x: -(x[1]["rate"] or 0)):
        pf = v.get("parent_failure_rescued", [v["rescued"], v["branches"]])
        ps = v.get("parent_success_kept", [0, 0])
        total = (pf[0] + ps[0], pf[1] + ps[1])
        lines.append("| %s | %s | %s | %s | %s | %s | %d / %d | %.1f%% | %s | %s |" % (
            name, m.get("branch"), trigger, m.get("control_queries"), m.get("noise_scale"), s, pf[0], pf[1], 100.0 * pf[0] / max(1, pf[1]),
            ("%d / %d" % tuple(ps)) if ps[1] else "-", ("%.1f%%" % (100.0 * total[0] / total[1])) if ps[1] else "-"))
(ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
