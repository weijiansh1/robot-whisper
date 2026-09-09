"""Analyse the 64-repeat single-scene capture and emit data for the viewer.

One init state, 64 flow-noise draws, everything else held fixed.  That removes
the scene-difficulty confound the report's section 6 had to subtract out: here
the observation at control step 0 is byte-identical across all 64 episodes and
the flow noise is the only thing that differs.

Two length traps this script avoids:

  * Episodes that succeed terminate early, so episode length is almost a perfect
    label.  Any decode over whole-episode features would read that off instead of
    the routing.  All decoding below is restricted to a fixed prefix of control
    steps, and control step 0 -- where the observation is identical -- is
    reported separately.
  * Drift statistics are averaged per episode before pooling, so long (failing)
    episodes do not dominate the pooled mean.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from within64_lib import (
    HB_LAYERS,
    N_EXPERTS,
    TOP_K,
    Run,
    action_token_probs,
    jaccard_from_intersection,
    load_run,
    onehot_sets,
    state_token_probs,
)

RNG = np.random.default_rng(0)
LN32 = float(np.log(N_EXPERTS))


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------
def reduce_dims(x: np.ndarray, n_comp: int = 30) -> np.ndarray:
    """Unsupervised PCA down to ``n_comp`` dims.

    Fit on all rows including the held-out one.  That is not label leakage --
    PCA never sees ``y`` -- and the permutation test below inherits exactly the
    same projection, so whatever optimism the shared basis buys is present in the
    null distribution too.  Without it, 2560 correlated features on 64 samples
    make the logistic fit unidentifiable.
    """
    n_comp = min(n_comp, x.shape[0] - 2, x.shape[1])
    xc = x - x.mean(0)
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:n_comp].T


def _hat(x: np.ndarray, lam: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Hat matrix of a ridge probe on standardised ``x`` with an intercept.

    A ridge probe is used rather than logistic regression because its
    leave-one-out predictions have a closed form,
    ``yhat_loo = (H y - diag(H) * y) / (1 - diag(H))``, and ``H`` does not depend
    on ``y``.  The permutation test therefore reduces to one matrix product for
    all permutations instead of refitting 64 x 1000 models.
    """
    xs = (x - x.mean(0)) / (x.std(0) + 1e-8)
    design = np.hstack([xs, np.ones((len(xs), 1))])
    penalty = lam * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0  # never penalise the intercept
    hat = design @ np.linalg.solve(design.T @ design + penalty, design.T)
    return hat, np.clip(np.diag(hat), -np.inf, 1 - 1e-6)


def _loo_scores(hat: np.ndarray, h_diag: np.ndarray, y_pm: np.ndarray) -> np.ndarray:
    """LOO fitted values for one or many label vectors (columns of ``y_pm``)."""
    fitted = hat @ y_pm
    return (fitted - h_diag[:, None] * y_pm) / (1.0 - h_diag)[:, None]


def _balanced_accuracy(pred_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Balanced accuracy per column of ``pred_pos`` (bool: predicted success)."""
    pos, neg = y == 1, y == 0
    tpr = pred_pos[pos].mean(0) if pos.any() else 0.0
    tnr = (~pred_pos[neg]).mean(0) if neg.any() else 0.0
    return (tpr + tnr) / 2.0


def decode(x: np.ndarray, y: np.ndarray, n_perm: int = 1000) -> dict:
    """LOO balanced accuracy of a linear probe, with a label-permutation null."""
    hat, h_diag = _hat(x)
    y_pm = np.where(y == 1, 1.0, -1.0)
    observed = float(_balanced_accuracy(_loo_scores(hat, h_diag, y_pm[:, None]) > 0, y)[0])
    perms = np.stack([RNG.permutation(y_pm) for _ in range(n_perm)], axis=1)
    null = _balanced_accuracy(_loo_scores(hat, h_diag, perms) > 0, y)
    return {
        "balanced_accuracy": observed,
        "permutation_p": float((1 + (null >= observed).sum()) / (1 + n_perm)),
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
    }


def balanced_accuracy_loo(x: np.ndarray, y: np.ndarray) -> float:
    hat, h_diag = _hat(x)
    y_pm = np.where(y == 1, 1.0, -1.0)[:, None]
    return float(_balanced_accuracy(_loo_scores(hat, h_diag, y_pm) > 0, y)[0])


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def outcome_table(run: Run) -> dict:
    rows = [
        {
            "episode": e.index,
            "seed": e.flow_noise_seed,
            "success": e.success,
            "action_steps": e.action_steps,
            "n_control": e.n_control,
        }
        for e in run.episodes
    ]
    n_ok = sum(r["success"] for r in rows)
    n = len(rows)
    # Wilson 95%
    p = n_ok / n
    z = 1.959963985
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {
        "n": n,
        "n_success": n_ok,
        "success_rate": p,
        "wilson95": [max(0.0, centre - half), min(1.0, centre + half)],
        "rows": rows,
    }


def sharpness(run: Run) -> list[dict]:
    """Per-layer router sharpness, averaged within episode then across episodes."""
    out = []
    for li, layer in enumerate(HB_LAYERS):
        ent, top1, margin, top4 = [], [], [], []
        for e in run.episodes:
            p = e.probs[:, li]  # [T, 10, 11, 32]
            srt = np.sort(p, axis=-1)[..., ::-1]
            ent.append(float(e.entropy[:, li].mean()))
            top1.append(float(srt[..., 0].mean()))
            margin.append(float((srt[..., 0] - srt[..., 1]).mean()))
            top4.append(float(srt[..., :4].sum(-1).mean()))
        out.append(
            {
                "layer": layer,
                "entropy": float(np.mean(ent)),
                "entropy_frac_ln32": float(np.mean(ent)) / LN32,
                "effective_experts": float(np.exp(np.mean(ent))),
                "top1_prob": float(np.mean(top1)),
                "top1_over_uniform": float(np.mean(top1)) * N_EXPERTS,
                "top1_minus_top2": float(np.mean(margin)),
                "top4_mass": float(np.mean(top4)),
            }
        )
    return out


def churn(run: Run) -> list[dict]:
    """Top-4 set overlap per routing site: across time, and across noise draws.

    Two baselines matter here.  ``adjacent_step`` is how much the selection moves
    between consecutive control steps within one episode.  ``cross_noise`` is how
    much it differs between two episodes at the *same* control step -- and at
    step 0 those two episodes saw a byte-identical observation, so any gap is
    caused by the flow noise alone.  Random 4-of-32 sets give 0.5/7.5 = 0.067.
    """
    onehots = [onehot_sets(e) for e in run.episodes]
    out = []
    for li, layer in enumerate(HB_LAYERS):
        adj = []
        for oh in onehots:
            if oh.shape[0] < 2:
                continue
            a = oh[:-1, li]
            b = oh[1:, li]
            inter = (a & b).sum(-1)
            adj.append(float(jaccard_from_intersection(inter.astype(np.float64)).mean()))

        cross = {}
        for t in (0, 1, 2):
            sel = np.stack([oh[t, li] for oh in onehots if oh.shape[0] > t]).astype(np.float32)
            n_ep = sel.shape[0]
            flat = sel.reshape(n_ep, -1, N_EXPERTS)
            inter = np.einsum("ipk,jpk->ijp", flat, flat)
            jac = jaccard_from_intersection(inter.astype(np.float64))
            iu = np.triu_indices(n_ep, k=1)
            cross[str(t)] = float(jac[iu].mean())

        out.append(
            {
                "layer": layer,
                "adjacent_step_jaccard": float(np.mean(adj)),
                "cross_noise_jaccard_at_step": cross,
                "random_baseline": float(TOP_K * TOP_K / N_EXPERTS)
                / (2 * TOP_K - TOP_K * TOP_K / N_EXPERTS),
            }
        )
    return out


def pca_trajectories(run: Run, n_comp: int = 3) -> dict:
    """Project every (episode, control step) routing state onto a shared 2-3D basis."""
    feats, owner = [], []
    for e in run.episodes:
        f = action_token_probs(e).reshape(e.n_control, -1)  # [T, 8*10*32]
        feats.append(f)
        owner.extend([(e.index, t) for t in range(e.n_control)])
    x = np.concatenate(feats, axis=0)
    mu = x.mean(0)
    xc = x - mu
    # scale-free: the constant per-expert preference is already removed by mu
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    comp = vt[:n_comp]
    coords = xc @ comp.T
    var = (s**2) / (s**2).sum()
    per_ep: dict[int, list] = {}
    for (ep, _t), c in zip(owner, coords):
        per_ep.setdefault(ep, []).append([round(float(v), 4) for v in c])
    return {
        "explained_variance": [float(v) for v in var[:n_comp]],
        "coords": per_ep,
    }


def step0_decode(run: Run) -> dict:
    """Predict the outcome from routing at control step 0.

    At step 0 the observation is identical across episodes -- same init state,
    same 10 settle steps, deterministic env -- so the flow noise draw is the only
    input that differs, and there is no length confound to subtract.
    """
    y = np.array([int(e.success) for e in run.episodes])
    results = {}

    variants = {
        "action_tokens_all_layers": lambda e: action_token_probs(e)[0].ravel(),
        "state_token_all_layers": lambda e: state_token_probs(e)[0].ravel(),
        "entropy_only": lambda e: e.entropy[0].ravel(),
    }
    for name, fn in variants.items():
        raw = np.stack([fn(e) for e in run.episodes])
        x = reduce_dims(raw)
        results[name] = {
            "raw_dim": int(raw.shape[1]),
            "pca_dim": int(x.shape[1]),
            **decode(x, y),
        }

    per_layer = []
    for li, layer in enumerate(HB_LAYERS):
        raw = np.stack([action_token_probs(e)[0, li].ravel() for e in run.episodes])
        per_layer.append(
            {"layer": layer, "balanced_accuracy": balanced_accuracy_loo(reduce_dims(raw), y)}
        )
    results["per_layer_action_tokens"] = per_layer
    return results


def prefix_decode(run: Run, n_steps: int) -> dict:
    """Same decode over the first ``n_steps`` control steps (length-matched).

    Features are averaged over the prefix rather than concatenated: concatenation
    multiplies the dimension by ``n_steps`` for no extra sample count, and the
    per-step detail is already covered by the step-0 decode.
    """
    usable = [e for e in run.episodes if e.n_control >= n_steps]
    y = np.array([int(e.success) for e in usable])
    raw = np.stack([action_token_probs(e)[:n_steps].mean(0).ravel() for e in usable])
    x = reduce_dims(raw)
    return {
        "n_steps": n_steps,
        "n_episodes": len(usable),
        "n_success": int(y.sum()),
        **decode(x, y),
    }


def action_ensemble(client_dir: pathlib.Path, run: Run) -> dict:
    """Spread of the 64 step-0 action chunks -- the free flow-model ensemble.

    Identical observation, 64 noise draws, so this is exactly the K-sample
    disagreement signal, measured here at K=64.
    """
    chunks = []
    for e in run.episodes:
        f = client_dir / ("episode_%02d.npz" % e.index)
        if not f.exists():
            return {"available": False}
        with np.load(f) as z:
            chunks.append(z["actions"][0])  # [replan, action_dim]
    a = np.stack(chunks)  # [64, replan, dim]
    mean = a.mean(0)
    dev = np.linalg.norm((a - mean).reshape(len(a), -1), axis=1)
    y = np.array([int(e.success) for e in run.episodes])
    per_dim_std = a.std(0).mean(0)
    return {
        "available": True,
        "chunk_shape": list(a.shape[1:]),
        "per_dim_std": [float(v) for v in per_dim_std],
        "mean_deviation_from_ensemble_mean": float(dev.mean()),
        "deviation_success": float(dev[y == 1].mean()) if (y == 1).any() else None,
        "deviation_failure": float(dev[y == 0].mean()) if (y == 0).any() else None,
        "deviation_auroc": _auroc(dev, y),
    }


def _auroc(score: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None  # not NaN: json.dump would emit invalid JSON
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-within", default="runs/within/summaries.json")
    ap.add_argument("--compare-init-state", type=int, default=24)
    args = ap.parse_args()

    run = load_run(args.server_dir, args.client_dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("run %s  layout=%s  episodes=%d  success=%d/%d"
          % (run.label, run.layout, len(run.episodes), run.n_success, len(run.episodes)))

    report: dict = {
        "label": run.label,
        "layout": run.layout,
        "checkpoint_sha256": run.checkpoint_sha256,
        "outcomes": outcome_table(run),
        "sharpness": sharpness(run),
        "churn": churn(run),
        "step0_decode": step0_decode(run),
    }

    n_min = min(e.n_control for e in run.episodes)
    report["prefix_decode"] = [prefix_decode(run, k) for k in (1, 3, min(5, n_min), n_min)]
    report["action_ensemble"] = action_ensemble(pathlib.Path(args.client_dir), run)

    # reproducibility check against the earlier 10-repeat batch on the same state
    cmp_path = pathlib.Path(args.compare_within)
    if cmp_path.exists():
        old = [
            s for s in json.loads(cmp_path.read_text())
            if s["init_state_id"] == args.compare_init_state
        ]
        by_seed = {s["flow_noise_seed"]: s for s in old}
        overlap = []
        for e in run.episodes:
            o = by_seed.get(e.flow_noise_seed)
            if o is None:
                continue
            overlap.append(
                {
                    "seed": e.flow_noise_seed,
                    "old_success": bool(o["success"]),
                    "new_success": e.success,
                    "old_steps": int(o["action_steps"]),
                    "new_steps": e.action_steps,
                }
            )
        report["reproducibility"] = {
            "n_overlapping_seeds": len(overlap),
            "outcome_agreement": sum(r["old_success"] == r["new_success"] for r in overlap),
            "step_identical": sum(r["old_steps"] == r["new_steps"] for r in overlap),
            "rows": overlap,
        }

    (out / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("outcomes", "step0_decode", "reproducibility")},
                     indent=2)[:3000])

    pca = pca_trajectories(run)
    (out / "pca.json").write_text(json.dumps(pca))
    print("PCA explained variance:", [round(v, 3) for v in pca["explained_variance"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
