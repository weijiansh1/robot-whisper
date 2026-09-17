"""Cross-arm overlap of rescues: are the rescued failures the same brittle episodes regardless of the intervention?"""
import json, sys, itertools
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT = Path("/home/swj/data/libero-runtime")
sys.path.insert(0, str(ROOT))
from run_control_batch import jobs_from_topology
jobs = {j["tag"]: j for j in jobs_from_topology()}
fails = sorted(t for t, j in jobs.items() if not j["native_success"])
outs = [ROOT / "simulations" / d for d in ("control-r1", "control-r2", "control-r3")]
arms = {}
for out in outs:
    if not out.exists(): continue
    for d in sorted(out.iterdir()):
        if not d.is_dir(): continue
        res = {}
        for e in d.iterdir():
            f = e / "summary.json"
            if f.exists():
                s = json.loads(f.read_text())
                if s.get("status") == "completed":
                    res[e.name] = bool(s["success"])
        arms[d.name] = res
names = sorted(arms)
print("arm            completed-failures rescued  rate")
M = {}
for a in names:
    r = arms[a]; done = [t for t in fails if t in r]
    resc = [t for t in done if r[t]]
    M[a] = set(resc)
    print("%-15s %3d               %2d      %.3f" % (a, len(done), len(resc), len(resc) / max(len(done), 1)))
union = set().union(*M.values())
print("union of rescued failures across arms: %d/%d = %.3f; rescued by >=2 arms: %d; by >=3: %d" % (
    len(union), len(fails), len(union) / len(fails), sum(1 for t in union if sum(t in M[a] for a in names) >= 2), sum(1 for t in union if sum(t in M[a] for a in names) >= 3)))
print("per-episode counts:", {t: sum(t in M[a] for a in names) for t in sorted(union)})
print("pairwise McNemar-style (exact binomial on discordant pairs), rows = arm A, cols = arm B; cell = A-only/B-only rescues (p)")
for a, b in itertools.combinations(names, 2):
    common = [t for t in fails if t in arms[a] and t in arms[b]]
    ao = sum(1 for t in common if arms[a][t] and not arms[b][t]); bo = sum(1 for t in common if arms[b][t] and not arms[a][t])
    p = binomtest(ao, ao + bo, 0.5).pvalue if ao + bo else 1.0
    print("  %-15s vs %-15s n=%3d  %d/%d  p=%.2f" % (a, b, len(common), ao, bo, p))
json.dump(dict(rescued={a: sorted(M[a]) for a in names}, union=sorted(union)), open(ROOT / "simulations/control-r1/overlap.json", "w"), indent=2)
