"""Why do the action tokens look "nearly uniform"?  Two different claims, split.

"Normalised router entropy 0.998" and "the aggregated expert histogram is flat"
sound like the same statement and are not.  They can even have opposite causes:

  per-site softmax flat        the router barely separates experts at all;
                               selection is decided by a tiny logit margin
  aggregate histogram flat     each site may choose sharply, but different sites
                               choose differently and the mixture washes out

The decomposition that separates them is the standard one for a mixture:

    H(p-bar)  =  E_i[H(p_i)]  +  I

where p-bar is the site-averaged router distribution and I >= 0 is the mutual
information between site and expert -- exactly "how much does the router change
from site to site".  Flat softmax means E[H] is near log2(32); a washed-out
mixture means I is large relative to it.

For the *selection* (top-4) rather than the softmax, the same split is done on the
top-1 choice: its marginal entropy against its entropy conditioned on the position
(denoise round, suffix index).  If position explains the spread, the aggregate is
flat because different positions prefer different experts, not because the choice
is arbitrary.

Also reports the top-4 probability mass, since a router whose selected mass is
barely above the uniform 4/32 is contributing almost nothing to the weighted
combination downstream regardless of which experts it names.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr

N_EXPERTS = 32
LOG2_E = float(np.log2(N_EXPERTS))
UNIFORM_TOP4 = 4.0 / N_EXPERTS


def entropy(p, axis=-1):
    p = np.clip(p, 1e-12, None)
    return -np.sum(p * np.log2(p), axis=axis)


def discrete_entropy(labels, n=N_EXPERTS):
    counts = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    return float(entropy(counts / total))


def softmax_split(probs):
    """probs [n_sites, 32] -> per-site entropy, mixture entropy, and their gap."""
    per_site = float(np.mean(entropy(probs)))
    mixture = float(entropy(probs.mean(0)))
    return {
        "per_site_entropy_bits": per_site,
        "per_site_entropy_normalised": per_site / LOG2_E,
        "mixture_entropy_bits": mixture,
        "mixture_entropy_normalised": mixture / LOG2_E,
        # I = H(mixture) - E[H(site)]: how much the router moves between sites.
        # Bounded above by log2(32) - E[H], so a flat softmax caps it near zero.
        "site_expert_information_bits": mixture - per_site,
    }


def selection_split(top1, position):
    """Marginal vs position-conditioned entropy of the top-1 choice."""
    marginal = discrete_entropy(top1)
    conditional, weight = 0.0, 0
    for key in np.unique(position):
        take = top1[position == key]
        conditional += take.size * discrete_entropy(take)
        weight += take.size
    conditional /= max(weight, 1)
    return {
        "top1_marginal_entropy_bits": marginal,
        "top1_marginal_normalised": marginal / LOG2_E,
        "top1_entropy_given_position_bits": conditional,
        # what fraction of the flat-looking aggregate is explained by "different
        # positions prefer different experts" rather than by an arbitrary choice
        "position_explains_bits": marginal - conditional,
        "position_explains_fraction": (marginal - conditional) / max(marginal, 1e-12),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", required=True)
    ap.add_argument("--max-steps", type=int, default=200,
                    help="control steps to read; the full-probs array is the big one")
    ap.add_argument("--out", default="analysis/routing-graph/uniformity.json")
    args = ap.parse_args()

    report = {}
    for task in args.task:
        path = pathlib.Path(task)
        group = zarr.open_group(str(path / "server" / "routes.zarr"), mode="r")
        n_steps = min(args.max_steps, group["hb_router_probs"].shape[0])
        probs = np.asarray(group["hb_router_probs"][:n_steps], dtype=np.float32)
        ids = np.asarray(group["hb_expert_ids"][:n_steps])
        _, n_layers, n_denoise, n_suffix, _ = ids.shape
        print("\n=== %s  (%d control steps) ===" % (path.name[:58], n_steps), flush=True)

        entry = {}
        for kind, token in (("action", slice(1, n_suffix)), ("state", slice(0, 1))):
            print("-- %s tokens" % kind)
            print("   layer  softmax H  mixture H   I(site;e)  top4 mass  "
                  "top1 H  H|position  pos.explains")
            per_layer = []
            for layer in range(n_layers):
                block = probs[:, layer, :, token, :]
                flat = block.reshape(-1, N_EXPERTS)
                # renormalise: stored as float16, and the rows must be distributions
                flat = flat / np.maximum(flat.sum(-1, keepdims=True), 1e-12)
                soft = softmax_split(flat)

                chosen = ids[:, layer, :, token, :]
                top1 = chosen[..., 0].ravel()
                rounds, tokens = np.meshgrid(
                    np.arange(n_denoise), np.arange(chosen.shape[2]), indexing="ij")
                position = np.tile((rounds * 100 + tokens).ravel(), n_steps)
                select = selection_split(top1, position)

                top4_mass = float(np.mean(np.take_along_axis(
                    flat, chosen.reshape(-1, 4).astype(np.int64), axis=1).sum(-1)))
                stat = dict(soft, **select)
                stat["top4_mass"] = top4_mass
                stat["top4_mass_over_uniform"] = top4_mass / UNIFORM_TOP4
                per_layer.append(stat)
                print("     %d     %.4f     %.4f     %.4f     %.4f    %.3f   %.3f      %4.1f%%"
                      % (layer, soft["per_site_entropy_normalised"],
                         soft["mixture_entropy_normalised"],
                         soft["site_expert_information_bits"], top4_mass,
                         select["top1_marginal_entropy_bits"],
                         select["top1_entropy_given_position_bits"],
                         100 * select["position_explains_fraction"]), flush=True)
            entry[kind] = per_layer
        report[path.name] = entry

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
