"""Where in the action cloud do the high-value candidates actually sit?

The budget decomposition showed the opportunity is real (O_32 = 5 per-state SDs) and
that the action medoid realises a NEGATIVE share of it.  Before inventing another
selector, this asks the diagnostic question that explains why:

    is quality aligned with consensus, or against it?

Per snapshot, every candidate gets geometric features computed only from its own
action chunk and the robot's recent motion -- everything a deployed selector could
see -- and each is correlated with the counterfactual value Q_i within the snapshot:

    r_i     distance to the medoid          consensus radius
    rho_i   sum of exp(-d/sigma) to others  local density
    L_i     sum ||A_{h+1} - A_h||           path length / how much it commits
    |D_i|   || sum_h A_h[:6] ||             net displacement magnitude
    P_i     <D_i, v_recent>                 alignment with what the arm just did
    J_i     sum ||A_{h+2} - 2A_{h+1} + A_h|| jerk
    g_i     first index where the gripper channel crosses zero

Correlations are computed WITHIN each snapshot and then aggregated across snapshots,
because the value scale differs per state; snapshots are the independent unit and
the permutation null shuffles candidates within a snapshot.

v_recent is taken as the net pose displacement of the previously executed chunk from
the source capture.  That is a command-space proxy: the arm need not have tracked it
exactly, and the honest alternative -- joint velocities from the snapshot -- lives in
a different space from the 6-dim pose deltas the candidates are written in, with no
Jacobian available here to bridge them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

GRIPPER = 6


def normalise(chunks, gripper_weight=0.25):
    pose = chunks[..., :GRIPPER]
    grip = chunks[..., GRIPPER:]
    pose = pose / np.maximum(pose.reshape(-1, GRIPPER).std(0), 1e-8)
    grip = grip / np.maximum(grip.reshape(-1, 1).std(0), 1e-8)
    return np.concatenate([pose.reshape(len(chunks), -1),
                           gripper_weight * grip.reshape(len(chunks), -1)], axis=1)


def features(chunks, v_recent):
    """Per-candidate geometry, all computable at deployment time."""
    flat = normalise(chunks)
    distance = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1)
    medoid = int(distance.sum(1).argmin())
    sigma = np.median(distance[np.triu_indices(len(chunks), 1)])
    pose = chunks[..., :GRIPPER]
    step = np.diff(pose, axis=1)
    displacement = pose.sum(1)
    unit = v_recent / max(np.linalg.norm(v_recent), 1e-9)
    gripper = chunks[..., GRIPPER]
    crossing = []
    for row in gripper:
        sign = np.sign(row - row[0])
        hit = np.flatnonzero(sign != 0)
        crossing.append(float(hit[0]) if len(hit) else float(len(row)))
    return {
        "medoid_radius": distance[medoid],
        "density": np.exp(-distance / max(sigma, 1e-9)).sum(1) - 1.0,
        "path_length": np.linalg.norm(step, axis=-1).sum(1),
        "displacement": np.linalg.norm(displacement, axis=-1),
        "momentum_align": displacement @ unit,
        "jerk": np.linalg.norm(np.diff(step, axis=1), axis=-1).sum(1),
        "gripper_time": np.asarray(crossing),
    }, medoid


def spearman(a, b):
    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="runs/fork-pilot-n32-client/fork_records.json")
    ap.add_argument("--chunks-dir", default="runs/fork-pilot-n32-client")
    ap.add_argument("--source", default="runs/objstate-t0s24-client",
                    help="capture whose recorded action tape the fork replayed")
    ap.add_argument("--analysis", default="analysis/fork-pilot-n32/fork_analysis.json")
    ap.add_argument("--value", default="drawer_delta")
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/value-location/summary.json")
    args = ap.parse_args()

    records = json.loads(pathlib.Path(args.records).read_text())
    degenerate = set()
    if pathlib.Path(args.analysis).exists():
        degenerate = set(json.loads(pathlib.Path(args.analysis).read_text()).get(
            "snapshots_with_degenerate_label", []))
    grouped = defaultdict(list)
    for row in records:
        grouped[(int(row["episode"]), int(row["fork_step"]))].append(row)

    per_snapshot, skipped = [], []
    for (episode, step), rows in sorted(grouped.items()):
        tag = "ep%d/t%d" % (episode, step)
        rows = sorted(rows, key=lambda r: int(r["candidate"]))
        values = np.asarray([float(r[args.value]) for r in rows])
        path = pathlib.Path(args.chunks_dir) / ("chunks_ep%02d_t%02d.npy" % (episode, step))
        if tag in degenerate or values.std() == 0 or not path.exists():
            skipped.append(tag)
            continue
        chunks = np.load(path)[: len(rows)]
        source = pathlib.Path(args.source) / ("episode_%02d.npz" % episode)
        v_recent = np.zeros(GRIPPER)
        if source.exists() and step > 0:
            tape = np.load(source, allow_pickle=True)["actions"]
            if step - 1 < len(tape):
                v_recent = np.asarray(tape[step - 1])[:, :GRIPPER].sum(0)
        feature, medoid = features(chunks, v_recent)
        per_snapshot.append({"tag": tag, "values": values, "features": feature,
                             "medoid": medoid, "n": len(rows)})

    print("snapshots used %d, skipped %d" % (len(per_snapshot), len(skipped)))
    rng = np.random.default_rng(args.seed)
    names = list(per_snapshot[0]["features"])
    summary = {"value": args.value, "n_snapshots": len(per_snapshot), "skipped": skipped,
               "features": {}}

    print("\n=== within-snapshot Spearman(Q_i, feature), averaged over snapshots ===")
    print("  feature            rho      perm p     snapshots with rho>0")
    for name in names:
        rhos = np.array([spearman(s["values"], s["features"][name]) for s in per_snapshot])
        rhos = rhos[np.isfinite(rhos)]
        null = []
        for _ in range(args.permutations):
            draw = [spearman(s["values"][rng.permutation(s["n"])], s["features"][name])
                    for s in per_snapshot]
            null.append(np.nanmean(draw))
        null = np.asarray(null)
        p = float((np.sum(np.abs(null) >= abs(rhos.mean())) + 1) / (len(null) + 1))
        summary["features"][name] = {"rho": float(rhos.mean()), "p": p,
                                     "positive": int((rhos > 0).sum()), "n": len(rhos)}
        print("  %-16s  %+.3f    %.4f      %d/%d"
              % (name, rhos.mean(), p, (rhos > 0).sum(), len(rhos)), flush=True)

    print("\n=== where do the extremes sit? (percentile within each snapshot) ===")
    oracle_density, oracle_radius, medoid_value = [], [], []
    for s in per_snapshot:
        best = int(np.argmax(s["values"]))
        rank = lambda v, i: float((v < v[i]).mean())
        oracle_density.append(rank(s["features"]["density"], best))
        oracle_radius.append(rank(s["features"]["medoid_radius"], best))
        medoid_value.append(rank(s["values"], s["medoid"]))
    for label, values in (("oracle candidate's density percentile", oracle_density),
                          ("oracle candidate's medoid-radius percentile", oracle_radius),
                          ("medoid candidate's VALUE percentile", medoid_value)):
        values = np.asarray(values)
        print("  %-42s mean %.3f   median %.3f" % (label, values.mean(), np.median(values)))
        summary[label] = {"mean": float(values.mean()), "median": float(np.median(values))}

    print("\n  0.5 = no preference. medoid value percentile well below 0.5 means the")
    print("  consensus pick is systematically worse than a random candidate.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
