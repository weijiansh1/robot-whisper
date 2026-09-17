"""Read-only formula audit against raw pools, without importing protocol.py."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/data/coding/robot-whisper-0909/himoe-route-capture")
from route_noise_selector import pairwise_hellinger, pairwise_rms


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-12):
        raise RuntimeError("Independent mismatch: " + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    config, result = load(root / "config.json"), load(root / "analysis.json")
    audit, collection = load(root / "route-audit.json"), load(root / "collection.json")
    pairs = list(itertools.combinations(range(8), 2))
    i, j = np.asarray(pairs).T
    std = np.asarray(config["server"]["normalization_action_std"][:6], dtype=float)
    errors, outputs = [], []
    with np.load(root / "analysis-data.npz", allow_pickle=False) as analyzed:
        for index, pool in enumerate(audit["pools"]):
            path = root / pool["path"]
            if sha(path) != pool["sha256"]:
                raise RuntimeError("Raw pool hash changed")
            block = slice(index * 28, (index + 1) * 28)
            with np.load(path, allow_pickle=False) as raw:
                p = raw["hb_router_probs"].astype(float)
                ids = raw["hb_expert_ids"]
                p /= p.sum(-1, keepdims=True)
                target = []
                for a, b in pairs:
                    difference = (raw["actions"][a, :, :6].astype(float) - raw["actions"][b, :, :6]) / std
                    target.append(float(np.sqrt(np.mean(difference ** 2))))
                close(analyzed["y"][block], target, "raw action target")
                close(analyzed["noise"][block, 0], pairwise_rms(raw["noises"])[i, j], "noise RMS")
                close(analyzed["full_moe"][block, 0], pairwise_hellinger(p)[i, j], "full Hellinger")
                close(analyzed["old_center"][block, 0], pairwise_hellinger(p[:, 4:, :3, 1:])[i, j], "old geometry")
                h, top4 = [], []
                for layer in range(8):
                    for tokens in (slice(0, 1), slice(1, 11)):
                        for denoise in (slice(0, 3), slice(7, 10)):
                            region = np.sqrt(p[:, layer, denoise, tokens])
                            h.append([np.sqrt(np.sum((region[a] - region[b]) ** 2, axis=-1).mean() / 2)
                                      for a, b in pairs])
                            selected = ids[:, layer, denoise, tokens].reshape(8, -1, 4)
                            values = []
                            for a, b in pairs:
                                shared = [len(set(left.tolist()).intersection(right.tolist()))
                                          for left, right in zip(selected[a], selected[b])]
                                values.append(1 - np.mean(shared) / 4)
                            top4.append(values)
                close(analyzed["structured_moe"][block], np.asarray(h + top4).T, "64 structured features")
                expected_split = "holdout" if pool["task"] in (2, 5, 8) else "train"
                if not np.all(analyzed["splits"][block] == expected_split):
                    raise RuntimeError("Task split mismatch")
                rows = [r for r in collection["calls"] if r["kind"] == "candidate" and
                        r["parent"] == pool["parent"] and r["query"] == pool["query"]]
                for candidate, row in enumerate(rows):
                    with np.load(root / row["wire"], allow_pickle=False) as wire:
                        if not np.array_equal(wire["actions"], raw["actions"][candidate]):
                            raise RuntimeError("Raw action/wire mismatch")
                        if not np.array_equal(wire["noise"], raw["noises"][candidate]):
                            raise RuntimeError("Raw noise/wire mismatch")
                        if candidate:
                            seed = np.random.SeedSequence([2026091421, int(pool["benchmark"] == "pro"),
                                                           pool["task"], pool["query"], candidate])
                            expected = np.random.default_rng(seed).standard_normal((10, 24)).astype(np.float32)
                            if not np.array_equal(expected, wire["noise"]):
                                raise RuntimeError("Candidate random stream mismatch")
                errors.append(float(np.max(np.abs(analyzed["y"][block] - target))))
        train, test = analyzed["splits"] == "train", analyzed["splits"] == "holdout"
        train_parents = analyzed["parents"][train]
        unique = sorted(set(train_parents))
        weights = np.zeros(len(train_parents))
        for parent in unique:
            mask = train_parents == parent
            weights[mask] = 1 / (len(unique) * mask.sum())
        for name, model in result["models"].items():
            x = analyzed[name][train]
            scale = np.maximum(np.sqrt(np.sum(weights[:, None] * x ** 2, axis=0)), 1e-12)
            close(scale, model["scale"], name + " train-only scaling")
            design = x / scale
            beta = np.asarray(model["coefficient"])
            gradient = design.T @ (weights * (design @ beta - analyzed["y"][train])) + .01 * beta
            if np.min(beta) < 0 or np.min(gradient) < -1e-8 or np.any(np.abs(gradient[beta > 1e-9]) > 1e-8):
                raise RuntimeError("Nonnegative ridge optimality failed")
            prediction = analyzed[name][test] / scale @ beta
            squared, absolute = [], []
            for parent in sorted(set(analyzed["parents"][test])):
                mask = analyzed["parents"][test] == parent
                residual = prediction[mask] - analyzed["y"][test][mask]
                squared.append(np.mean(residual ** 2))
                absolute.append(np.mean(np.abs(residual)))
            rmse, mae = float(np.sqrt(np.mean(squared))), float(np.mean(absolute))
            close(rmse, result["results"][name]["holdout"]["rmse"], name + " RMSE")
            close(mae, result["results"][name]["holdout"]["mae"], name + " MAE")
            outputs.append({"model": name, "rmse": rmse, "mae": mae})
    summary = {"passed": True, "raw_pools_checked": len(errors), "models_checked": len(outputs),
               "max_target_error": max(errors), "metrics": outputs,
               "audit_script_sha256": sha(Path(__file__)),
               "upstream_geometry_sha256": sha(Path("/data/coding/robot-whisper-0909/himoe-route-capture/route_noise_selector.py")),
               "scope": "Independent raw-feature formulas, source RNG, task split, KKT and parent-balanced errors"}
    with (root / "independent-audit.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
