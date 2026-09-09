"""Check genuinely different initializations after deterministic PCA repeats."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from analyze import ROOT, SEED, digest, nearest_ids, write_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round4_geometry")
    args = parser.parse_args()
    root = args.input.resolve()
    destination = root / "random_init"
    destination.mkdir(exist_ok=False)
    records, identical = [], []
    for suite in sorted(pd.read_csv(root / "projection_metrics.csv").suite.unique()):
        with np.load(root / "projections" / f"{suite}_matched_vectors.npz", allow_pickle=False) as z:
            values = {k: z[k] for k in z.files}
        for name in ("routing", "dynamics"):
            with np.load(root / "projections" / f"{suite}_{name}_matched_seed{SEED}.npz", allow_pickle=False) as z:
                original = z["xy"]
            x = values[name]
            high = nearest_ids(x)
            for seed in (SEED + 1, SEED + 2):
                with np.load(root / "projections" / f"{suite}_{name}_matched_seed{seed}.npz", allow_pickle=False) as z:
                    identical.append({"suite": suite, "representation": name, "seed": seed,
                                      "pca_initialization_identical_to_primary": bool(np.array_equal(z["xy"], original))})
                start = time.perf_counter()
                model = TSNE(n_components=2, perplexity=30, init="random", max_iter=1000,
                             learning_rate="auto", random_state=seed, n_jobs=1)
                xy = model.fit_transform(x)
                low = nearest_ids(xy)
                overlap = np.asarray([len(set(h) & set(l)) / len(h) for h, l in zip(high, low)])
                np.savez_compressed(destination / f"{suite}_{name}_matched_seed{seed}.npz", xy=xy, neighbor_overlap=overlap)
                records.append({"suite": suite, "representation": name, "seed": seed, "init": "random",
                                "neighbor_overlap_15": float(overlap.mean()), "kl_divergence": float(model.kl_divergence_),
                                "seconds": time.perf_counter() - start})
                print(f"RANDOM INIT {suite} {name} {seed}: {records[-1]['seconds']:.1f}s", flush=True)
    pd.DataFrame(records).to_csv(destination / "metrics.csv", index=False)
    write_json(destination / "audit.json", {"pca_seed_repeats": identical,
        "source_sha256": digest(Path(__file__)), "protocol_sha256": digest(HERE / "RANDOM_INIT_ZH.md"),
        "inputs": {str(p.relative_to(root)): digest(p) for p in sorted((root / "projections").glob("*_matched_vectors.npz"))},
        "artifacts": {p.name: digest(p) for p in sorted(destination.iterdir()) if p.is_file()}})


if __name__ == "__main__":
    main()
