"""Does sampling more rollouts actually buy anything, once a real selector picks?

Splits the value of extra samples into the two things that get conflated:

    O_K(s)      = E_C[ max_{i in C} Q_i ] - mean_i Q_i      opportunity: is a better
                                                            candidate even in the pool?
    R_K^med(s)  = E_C[ max_{i in C} Q_i - Q_{medoid(C)} ]   regret: does the deployable
                                                            selector miss it?
    G_K^med(s)  = O_K - R_K^med                             what is actually realised

Only G is a reason to spend compute. O large with G near zero means the bottleneck is
the selector, not the budget, and no amount of adaptive-K prediction will help.

Runs on the existing fork pilot, so it costs no simulator time: 20 snapshots x 32
candidates, each candidate executed open-loop for its whole chunk and then continued
under a shared CRN stream, reached by REPLAYING a recorded action tape into a fresh
env rather than restoring state -- the mode measured at zero drift, because restore
fidelity turned out to be a property of the individual state.

Subsets are drawn at random rather than taken as candidates 1..K, so no state's
result can be dominated by whether candidate 1 happened to be good.

The selector is the 10-step action medoid, normalised per dimension, with the gripper
kept out of the continuous distance so one near-binary channel cannot dominate the
pose geometry.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

from analyze_value_location import features as geometry

GRIPPER_DIM = 6  # LIBERO action layout: 0-5 pose delta, 6 gripper


def medoid(chunks, gripper_weight=0.25):
    """Index of the candidate minimising total distance to the others."""
    pose = chunks[..., :GRIPPER_DIM]
    grip = chunks[..., GRIPPER_DIM:]
    scale = pose.reshape(-1, pose.shape[-1]).std(0)
    pose = pose / np.maximum(scale, 1e-8)
    grip_scale = grip.reshape(-1, grip.shape[-1]).std(0)
    grip = grip / np.maximum(grip_scale, 1e-8)
    flat = np.concatenate([pose.reshape(len(chunks), -1),
                           gripper_weight * grip.reshape(len(chunks), -1)], axis=1)
    distance = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1)
    return int(distance.sum(1).argmin())


def select(chunks, take, kind, feature, quantile=0.5):
    """Index (within `take`) chosen by the named deployable selector."""
    if kind == "medoid":
        return medoid(chunks[take])
    if kind == "anti_medoid":
        return int(np.argmax(feature["medoid_radius"][take]))
    if kind == "commit":
        # momentum-aligned commitment, penalised for oscillation
        def z(v):
            return (v - v.mean()) / max(v.std(), 1e-9)
        score = z(feature["momentum_align"][take]) - 0.5 * z(feature["jerk"][take])
        return int(np.argmax(score))
    if kind == "pcm":
        # progress-conditioned medoid: medoid WITHIN the committed sub-group, so the
        # pick is representative of the candidates that actually advance rather than
        # of the whole cloud
        def z(v):
            return (v - v.mean()) / max(v.std(), 1e-9)
        score = z(feature["momentum_align"][take]) - 0.5 * z(feature["jerk"][take])
        keep = np.flatnonzero(score >= np.quantile(score, quantile))
        if len(keep) < 2:
            return medoid(chunks[take])
        inner = medoid(chunks[take][keep])
        return int(keep[inner])
    raise ValueError(kind)


def decompose(values, chunks, subsets, rng, kind="medoid", feature=None, quantile=0.5):
    """O_K, R_K^med, G_K^med at each budget K."""
    n = len(values)
    baseline = float(values.mean())
    out = {}
    for budget in (1, 2, 4, 8, 16, min(32, n)):
        if budget > n:
            continue
        best, picked = [], []
        for _ in range(subsets):
            take = rng.choice(n, budget, replace=False)
            best.append(values[take].max())
            picked.append(values[take[select(chunks, take, kind, feature, quantile)]])
        best, picked = np.asarray(best), np.asarray(picked)
        oracle = float(best.mean() - baseline)
        gain = float(picked.mean() - baseline)
        if budget == 1:
            # O_1 = R_1 = G_1 = 0 by definition: a one-element subset's max and its
            # medoid are both the element itself, so the expectation over random
            # draws is exactly the baseline.  Anything nonzero here is Monte-Carlo
            # error from finite subset sampling, and is reported as such rather than
            # left in the table looking like an effect.
            out[budget] = {"oracle": 0.0, "medoid": 0.0, "regret": 0.0,
                           "mc_error": max(abs(oracle), abs(gain))}
            continue
        out[budget] = {
            "oracle": oracle,
            "medoid": gain,
            "regret": float((best - picked).mean()),
        }
    return out, baseline


def _recent(args, episode, step):
    """Net pose displacement of the previously executed chunk, as a motion proxy."""
    source = pathlib.Path(args.source) / ("episode_%02d.npz" % episode)
    if not source.exists() or step <= 0:
        return np.zeros(GRIPPER_DIM)
    tape = np.load(source, allow_pickle=True)["actions"]
    if step - 1 >= len(tape):
        return np.zeros(GRIPPER_DIM)
    return np.asarray(tape[step - 1])[:, :GRIPPER_DIM].sum(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="runs/fork-pilot-n32-client/fork_records.json")
    ap.add_argument("--chunks-dir", default="runs/fork-pilot-n32-client")
    ap.add_argument("--analysis", default="analysis/fork-pilot-n32/fork_analysis.json",
                    help="used only to drop the snapshots it flagged as degenerate")
    ap.add_argument("--source", default="runs/objstate-t0s24-client")
    ap.add_argument("--value", default="drawer_delta",
                    choices=("drawer_delta", "drawer_delta_chunk", "success_in_window"))
    ap.add_argument("--subsets", type=int, default=256)
    ap.add_argument("--selector", default="medoid",
                    choices=("medoid", "anti_medoid", "commit", "pcm"))
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="progress quantile for pcm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/budget-gain/summary.json")
    args = ap.parse_args()

    records = json.loads(pathlib.Path(args.records).read_text())
    degenerate = set()
    analysis = pathlib.Path(args.analysis)
    if analysis.exists():
        degenerate = set(json.loads(analysis.read_text()).get(
            "snapshots_with_degenerate_label", []))

    grouped = defaultdict(list)
    for row in records:
        grouped[(int(row["episode"]), int(row["fork_step"]))].append(row)

    rng = np.random.default_rng(args.seed)
    per_state, skipped = {}, []
    for (episode, step), rows in sorted(grouped.items()):
        tag = "ep%d/t%d" % (episode, step)
        if tag in degenerate:
            skipped.append(tag)
            continue
        rows = sorted(rows, key=lambda r: int(r["candidate"]))
        values = np.asarray([float(r[args.value]) for r in rows])
        if values.std() == 0:
            skipped.append(tag)
            continue
        path = pathlib.Path(args.chunks_dir) / ("chunks_ep%02d_t%02d.npy" % (episode, step))
        if not path.exists():
            skipped.append(tag)
            continue
        chunks = np.load(path)[: len(rows)]
        feature, _ = geometry(chunks, np.zeros(6) if args.selector in ("medoid", "anti_medoid")
                              else _recent(args, episode, step))
        table, baseline = decompose(values, chunks, args.subsets, rng,
                                    args.selector, feature, args.quantile)
        per_state[tag] = {"baseline": baseline, "spread": float(values.std()), **{
            str(k): v for k, v in table.items()}}

    print("states used: %d   skipped (degenerate or missing): %d %s"
          % (len(per_state), len(skipped), skipped))
    print("value = %s   selector = %s\n" % (args.value, args.selector))
    print("   K    O_K (oracle)   G_K (medoid)   R_K (regret)   G/O    states with G>0")
    budgets = sorted({int(k) for s in per_state.values() for k in s if k.isdigit()})
    summary = {"value": args.value, "n_states": len(per_state), "skipped": skipped,
               "budgets": {}}
    for budget in budgets:
        oracle = np.array([s[str(budget)]["oracle"] for s in per_state.values()])
        gain = np.array([s[str(budget)]["medoid"] for s in per_state.values()])
        regret = np.array([s[str(budget)]["regret"] for s in per_state.values()])
        # bootstrap over states: they are the independent unit
        draws = np.array([gain[rng.integers(0, len(gain), len(gain))].mean()
                          for _ in range(2000)])
        low, high = np.percentile(draws, [2.5, 97.5])
        share = float(gain.mean() / oracle.mean()) if oracle.mean() > 0 else float("nan")
        summary["budgets"][budget] = {
            "oracle": float(oracle.mean()), "medoid_gain": float(gain.mean()),
            "regret": float(regret.mean()), "gain_ci": [float(low), float(high)],
            "share_realised": share,
            "states_positive": int((gain > 0).sum())}
        print("  %2d      %+.4f        %+.4f [%+.4f,%+.4f]   %+.4f      %+.2f    %d/%d"
              % (budget, oracle.mean(), gain.mean(), low, high, regret.mean(),
                 share, (gain > 0).sum(), len(gain)))

    top = summary["budgets"][budgets[-1]]
    print("\nat K=%d: the pool holds %+.4f of extra value, the medoid realises %+.4f (%.0f%%)"
          % (budgets[-1], top["oracle"], top["medoid_gain"], 100 * top["share_realised"]))
    if top["gain_ci"][0] <= 0:
        print("  medoid gain is not distinguishable from zero -> adaptive K cannot pay off")
        print("  with this selector; change the selector before tuning the budget.")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_state": per_state}, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
