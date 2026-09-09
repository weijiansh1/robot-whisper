"""Post-hoc control for expert-activation v2 (outside the frozen family).

The frozen B1 baseline contains only pair-distance features, while the winning
channels (c2/c3/c7) contain pair-mean (level) features.  Internal levels are
functions of the noise, so they may simply re-extract noise-magnitude
information the baseline was blind to by construction.  This script re-scores
the same 200-cell family against B1+ = B1 + per-member noise-level features,
adds a generic |h|-level control channel, and splits c3_dcq into its
distance-only and mean-only halves.  Everything here is exploratory and is
labelled post-hoc; the frozen results in summary.json are not modified.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from joblib import Parallel, delayed

import analyze_expert_activation_v2 as v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", default="2,5,12,15")
    parser.add_argument("--perms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--jobs", type=int, default=48)
    return parser.parse_args()


def build_b1plus(data: dict, frame: dict) -> np.ndarray:
    """B1 (12 distance features) + 44 pair mean/absdiff noise-level features."""
    noise = data["noise"]                       # [rows, 10, 24]
    live = noise[:, :, : v2.LIVE_ACTION_DIMS]
    member = np.column_stack(
        (
            np.sqrt(np.mean(np.square(live.reshape(len(noise), -1)), axis=1)),
            np.sqrt(np.mean(np.square(live), axis=2)),
            np.sqrt(np.mean(np.square(noise.reshape(len(noise), -1)), axis=1)),
            np.sqrt(np.mean(np.square(noise), axis=2)),
        )
    ).astype(np.float32)                        # [rows, 22]
    iu = frame["iu"]
    n_local = frame["n_local"]
    level = np.empty((len(frame["labels"]), member.shape[1] * 2), np.float32)
    for s, scene in enumerate(frame["scene_ids"]):
        index = frame["order"][scene]
        vi, vj = member[index][iu[0]], member[index][iu[1]]
        sl = slice(s * n_local, (s + 1) * n_local)
        level[sl, : member.shape[1]] = np.abs(vi - vj)
        level[sl, member.shape[1] :] = 0.5 * (vi + vj)
    return np.column_stack((frame["b1"], level))


def main() -> int:
    args = parse_args()
    run = pathlib.Path(args.run).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    layer_numbers = tuple(int(v) for v in args.layers.split(",") if v)

    data = v2.load_dataset(run, layer_numbers)
    features = v2.load_or_compute_features(
        out_dir / "features_v2.npz", data, pathlib.Path(args.checkpoint), args.threads, False
    )
    frame = v2.build_pair_frame(data)
    cell_feats = v2.build_cell_features(data, features, frame)
    labels = frame["labels"]
    b1plus = build_b1plus(data, frame)
    print("B1+ has %d features" % b1plus.shape[1], flush=True)

    b1plus_matrix = v2._fit_fold_scene_auc(b1plus, labels, frame, args.logistic_c)
    b1plus_value = float(np.nanmean(b1plus_matrix))
    b1_value = float(np.nanmean(v2._fit_fold_scene_auc(frame["b1"], labels, frame, args.logistic_c)))
    print("B1 %.4f -> B1+ %.4f" % (b1_value, b1plus_value), flush=True)

    n_layers = len(layer_numbers)
    n_denoise = features["scalars"].shape[1]

    # extra channels: generic |h| level, and the two halves of c3_dcq
    hlevel_idx = v2.SCALAR_NAMES.index("input_rms")
    dist_cols = [0, 2, 4, 6, 8, 10]
    mean_cols = [1, 3, 5, 7, 9, 11]
    for (la, tau), channels in cell_feats["cells"].items():
        scal = None  # placeholder to keep loop simple
        channels["c8_h_level"] = np.empty((len(labels), 4), np.float32)
        for s, scene in enumerate(frame["scene_ids"]):
            index = frame["order"][scene]
            sl = slice(s * frame["n_local"], (s + 1) * frame["n_local"])
            channels["c8_h_level"][sl] = v2._pair_scalar_block(
                features["scalars"][index, tau, la, :, hlevel_idx], frame["iu"]
            )
        channels["c9_dcq_dist"] = channels["c3_dcq"][:, dist_cols]
        channels["c10_dcq_mean"] = channels["c3_dcq"][:, mean_cols]

    observed = {}
    extra = ("c8_h_level", "c9_dcq_dist", "c10_dcq_mean")
    for (la, tau), channels in sorted(cell_feats["cells"].items()):
        for name in v2.ALL_CHANNELS + extra:
            x = np.column_stack((b1plus, channels[name]))
            matrix = v2._fit_fold_scene_auc(x, labels, frame, args.logistic_c)
            observed[(name, la, tau)] = float(np.nanmean(matrix)) - b1plus_value
    print("observed fits vs B1+ done", flush=True)

    rng = np.random.default_rng(args.seed)
    n_scenes = len(frame["scene_ids"])
    perms = np.stack(
        [
            np.stack([rng.permutation(frame["n_slots"]) for _ in range(n_scenes)])
            for _ in range(args.perms)
        ]
    )
    family_feats = {
        (name, la, tau): cell_feats["cells"][(la, tau)][name]
        for name in v2.FAMILY_CHANNELS
        for la in range(n_layers)
        for tau in range(n_denoise)
    }
    frame_small = {
        key: frame[key]
        for key in ("iu", "pairid", "n_local", "fold_i", "fold_j", "scene_of_pair",
                     "scene_ids", "labels", "n_slots")
    }
    batches = np.array_split(np.arange(args.perms), min(args.jobs, args.perms))
    started = time.perf_counter()
    results = Parallel(n_jobs=args.jobs, verbose=1)(
        delayed(v2._perm_worker)(
            perms[batch], family_feats, b1plus, labels, frame_small,
            args.logistic_c, b1plus_value,
        )
        for batch in batches
        if len(batch)
    )
    perm_delta = np.concatenate(results, axis=0)
    print("B1+ permutations done in %.1fs" % (time.perf_counter() - started), flush=True)
    family_keys = sorted(family_feats)
    null_max = perm_delta.max(axis=1)
    fwer_p = {
        "%s_l%d_t%d" % key: float((1 + np.sum(null_max >= observed[key])) / (1 + args.perms))
        for key in family_keys
    }

    payload = {
        "note": "post-hoc control, outside the frozen prereg family",
        "b1_value": b1_value,
        "b1plus_value": b1plus_value,
        "b1plus_features": int(b1plus.shape[1]),
        "observed_delta_vs_b1plus": {
            "%s_l%d_t%d" % key: value for key, value in observed.items()
        },
        "fwer_p_vs_b1plus": fwer_p,
        "null_max_quantiles": {
            "q50": float(np.quantile(null_max, 0.50)),
            "q95": float(np.quantile(null_max, 0.95)),
            "q99": float(np.quantile(null_max, 0.99)),
        },
        "n_pass": int(sum(1 for v in fwer_p.values() if v < 0.05)),
    }
    (out_dir / "posthoc_b1plus.json").write_text(json.dumps(payload, indent=2))
    print("passes vs B1+: %d / 200" % payload["n_pass"], flush=True)
    for key in sorted(fwer_p, key=lambda k: fwer_p[k])[:12]:
        print(
            "%-22s delta %+0.4f  fwer_p %.3f"
            % (key, payload["observed_delta_vs_b1plus"][key], fwer_p[key]),
            flush=True,
        )
    print("wrote %s" % (out_dir / "posthoc_b1plus.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
