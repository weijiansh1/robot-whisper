#!/usr/bin/env python3
"""Gate 1: how much power does the event-detection pipeline actually have?

A null result only means something if the pipeline could have found a signal.
This injects known perturbations into the router distribution itself - not into
the derived churn scalar - and re-runs the real feature extraction, the real 224
statistics, the real within-initial-state AUC and the real max-statistic
permutation. Output is a power curve and the minimum detectable effect, so the
negative result on real events can be stated as "this excludes effects of at
least size X" rather than "p > 0.05".

Design mirrors the real test exactly: 19 synthetic events against the remaining
successes, same initial-state clustering, same horizon.

Injection stays on the routing manifold: the perturbed distribution is an
interpolation toward this rollout's own routing at a different task phase, not
Gaussian noise, which could never occur.

Shapes cover what a real event might look like:
    impulse   one chunk
    plateau   d consecutive chunks
    step      persists to the end of the window
    jitter    plateau whose onset moves +-1 chunk per rollout
    hetero    plateau whose sign flips for half the rollouts

A natural positive control - the gripper open/close transition, where routing is
known to move - is run through the identical pipeline. Synthetic sensitivity is
only trustworthy if the pipeline also finds that.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

import analyze_early_structure as core


HERE = pathlib.Path(__file__).resolve().parent
HORIZON = 34
WINDOW = core.HISTORY                 # 7 chunks
NEED = WINDOW + 1                     # one extra for the first difference
N_EVENT = 19                          # exactly the real cohort size
PERMS = 2000
STAT_NAMES = ("hard_max", "hard_min", "hard_range", "hard_jolt",
              "soft_max", "soft_min", "soft_range", "soft_jolt",
              "ent_max", "ent_min", "ent_range", "ent_jolt",
              "innov_max", "innov_last")


def load_window(run, episodes):
    """[episode, NEED, layer, token, expert] router probabilities and Top-4."""
    store = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    lo = HORIZON - WINDOW              # first step whose diff we need
    P, I, Q = [], [], []
    for ep in episodes:
        a = ep.offset + lo
        p = np.asarray(store["hb_router_probs"][a:a + NEED, :, -1], np.float32)
        p = np.maximum(p, 0.0)
        p /= np.maximum(p.sum(-1, keepdims=True), 1e-12)
        P.append(p)
        I.append(np.asarray(store["hb_expert_ids"][a:a + NEED, :, -1], np.int64))
        # a real routing state from an earlier phase of the same rollout
        b = max(ep.offset, a - 15)
        q = np.asarray(store["hb_router_probs"][b:b + 1, :, -1], np.float32)
        q = np.maximum(q, 0.0)
        Q.append(q / np.maximum(q.sum(-1, keepdims=True), 1e-12))
    return np.stack(P), np.stack(I), np.concatenate(Q)


def features(p):
    """The 224 transient statistics, recomputed exactly as in audit_transient."""
    top = np.argpartition(p, -4, axis=-1)[..., -4:]
    hot = np.zeros(p.shape, bool)
    np.put_along_axis(hot, top, True, -1)
    ent = -np.sum(p * np.log(np.maximum(p, 1e-12)), -1)
    inter = np.logical_and(hot[:, 1:], hot[:, :-1]).sum(-1)
    union = np.logical_or(hot[:, 1:], hot[:, :-1]).sum(-1)
    hard = 1.0 - inter / np.maximum(union, 1)
    root = np.sqrt(p)
    soft = np.sqrt(np.maximum(0.5 * np.square(root[:, 1:] - root[:, :-1]).sum(-1), 0.0))
    ema = np.empty_like(p)
    ema[:, 0] = p[:, 0]
    for t in range(1, p.shape[1]):
        ema[:, t] = 0.5 * ema[:, t - 1] + 0.5 * p[:, t - 1]
    innov = np.sqrt(np.maximum(0.5 * np.square(np.sqrt(p) - np.sqrt(ema)).sum(-1), 0.0))
    out = []
    for X in (hard, soft, ent[:, 1:]):
        jolt = np.abs(np.diff(X, axis=1)).max(1)
        out += [X.max(1), X.min(1), X.max(1) - X.min(1), jolt]
    out += [innov[:, 1:].max(1), innov[:, -1]]
    stack = np.stack(out, 1)                       # [ep, 14, layer, token]
    return np.concatenate([stack.mean(-1).reshape(len(p), -1),
                           stack.max(-1).reshape(len(p), -1)], 1)


def inject(P, Q, idx, shape, lam, dur, rng):
    """Move the chosen rollouts' routing toward a real off-phase distribution."""
    out = P.copy()
    for e in idx:
        onset = WINDOW - dur                      # plateau ends at the window edge
        if shape == "jitter":
            onset = int(np.clip(onset + rng.integers(-1, 2), 0, NEED - 1))
        a = lam
        if shape == "hetero" and rng.random() < 0.5:
            a = -lam                              # half the rollouts move the other way
        span = range(onset, NEED) if shape == "step" else \
            range(onset, min(onset + max(dur, 1), NEED))
        for t in span:
            m = (1 - a) * out[e, t] + a * Q[e]
            m = np.maximum(m, 1e-12)
            out[e, t] = m / m.sum(-1, keepdims=True)
    return out


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return np.nan
    return (float((p[:, None] > n[None, :]).sum())
            + 0.5 * float((p[:, None] == n[None, :]).sum())) / (len(p) * len(n))


def weighted(lab, col, groups):
    num = den = 0.0
    for i in groups:
        y = lab[i]
        if len(np.unique(y)) < 2:
            continue
        w = int(y.sum()) * int((1 - y).sum())
        num += auc(y, col[i]) * w
        den += w
    return num / den if den else np.nan


def detect(F, lab, groups, rng, perms=PERMS):
    """Max-statistic permutation over all 224 features. Returns family-wise p."""
    obs = np.abs(np.array([weighted(lab, F[:, j], groups) for j in range(F.shape[1])]) - .5)
    peak = obs.max()
    hits = 0
    for _ in range(perms):
        p = lab.copy()
        for g in groups:
            p[g] = rng.permutation(p[g])
        m = np.abs(np.array([weighted(p, F[:, j], groups)
                             for j in range(F.shape[1])]) - .5).max()
        hits += int(m >= peak)
    return peak, (1 + hits) / (perms + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=core.RUN)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--reps", type=int, default=12, help="repeats per sweep cell")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    episodes = core.load_episodes(args.run)
    sims = core.load_sim(args.run, episodes)
    stasis, _, _, _ = core.build_targets(episodes, sims)
    fail = np.asarray([e.failure for e in episodes], np.int8)
    succ = np.flatnonzero(fail == 0)
    state = np.asarray([e.state for e in episodes], np.int16)
    print("成功 rollout %d 条，用其中 %d 条做合成事件" % (len(succ), N_EVENT), flush=True)

    cache = HERE / "gate1_window.npz"
    if cache.exists():
        with np.load(cache) as z:
            P, I, Q = z["P"], z["I"], z["Q"]
    else:
        P, I, Q = load_window(args.run, episodes)
        np.savez_compressed(cache, P=P, I=I, Q=Q)
    print("routing 窗口 %s" % (P.shape,), flush=True)

    base = features(P)
    sigma = np.array([np.std(base[succ, j]) for j in range(base.shape[1])])
    sigma[sigma < 1e-9] = 1e-9

    rng = np.random.default_rng(args.seed)
    rows = []

    # ---- natural positive control: the gripper open/close transition ----
    grip = []
    for ep in episodes:
        with np.load(core.episode_path(args.run, ep), allow_pickle=False) as z:
            g = np.asarray(z["state"][:, 6] - z["state"][:, 7], np.float32)
        w = g[HORIZON - WINDOW: HORIZON + 1]
        grip.append(float(np.abs(np.diff(w)).max()))
    grip = np.array(grip)
    move = succ[np.argsort(-grip[succ])[:N_EVENT]]        # biggest gripper change
    lab = np.zeros(len(episodes), int)
    lab[move] = 1
    keep = np.zeros(len(episodes), bool)
    keep[succ] = True
    gs = [np.flatnonzero(keep & (state == v)) for v in np.unique(state)]
    gs = [g for g in gs if len(np.unique(lab[g])) == 2]
    peak, p = detect(base, lab, gs, np.random.default_rng(1))
    print("\n自然阳性对照（夹爪开合最剧烈的 %d 条 vs 其余成功）"
          "：可用 state %d  最强 |AUC−0.5|=%.3f  maxT p=%.4f  %s"
          % (N_EVENT, len(gs), peak, p, "✅ 管线能检出" if p < .05 else "❌ 管线检不出"),
          flush=True)
    natural = {"peak": float(peak), "p": float(p), "states": len(gs)}

    # ---- synthetic sweep ----
    print("\n合成注入扫描（每格 %d 次重复，注入到路由分布后重算全部 224 个特征）" % args.reps,
          flush=True)
    print("%-9s %5s %5s   %8s %8s %7s" % ("形态", "λ", "时长", "诱发效应σ", "检出率", "中位p"))
    for shape in ("impulse", "plateau", "step", "jitter", "hetero"):
        for lam in (0.10, 0.20, 0.35, 0.55):
            dur = 1 if shape == "impulse" else 3
            det, ps, eff = 0, [], []
            for r in range(args.reps):
                idx = rng.choice(succ, N_EVENT, replace=False)
                Pi = inject(P, Q, idx, shape, lam, dur, rng)
                F = features(Pi)
                d = np.abs(F[idx].mean(0) - base[idx].mean(0)) / sigma
                eff.append(float(d.max()))
                lab = np.zeros(len(episodes), int)
                lab[idx] = 1
                g2 = [np.flatnonzero(keep & (state == v)) for v in np.unique(state)]
                g2 = [g for g in g2 if len(np.unique(lab[g])) == 2]
                _, pv = detect(F, lab, g2, rng, perms=600)
                ps.append(pv)
                det += int(pv < .05)
            rows.append({"shape": shape, "lam": lam, "dur": dur,
                         "effect_sigma": float(np.median(eff)),
                         "power": det / args.reps, "p_median": float(np.median(ps))})
            print("%-9s %5.2f %5d   %8.2f %8.0f%% %7.3f"
                  % (shape, lam, dur, np.median(eff), 100 * det / args.reps, np.median(ps)),
                  flush=True)

    args.out.write_text(json.dumps({"n_event": N_EVENT, "reps": args.reps,
                                    "natural_control": natural, "sweep": rows},
                                   indent=2, ensure_ascii=False) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
