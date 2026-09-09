"""Pairwise discrimination within physical-stage strata, with cluster intervals."""

import numpy as np
import pandas as pd


def matched_auc(frame, models, strata, population, repeats=1000):
    """Bootstrap whole task/init groups using precomputed concordant-pair counts.

    Each matrix entry counts positive/negative query pairs from two init groups
    in the same task/query[/stage]. Resampling a group w times multiplies such
    pair counts by w_i*w_j, including diagonal entries for seed siblings.
    """
    tasks = sorted(population.task.unique())
    clusters = {t: sorted(population.loc[population.task == t, "cluster"].unique()) for t in tasks}
    width = max(map(len, clusters.values()))
    count = np.zeros((len(tasks), width, width))
    concordant = {name: np.zeros_like(count) for name in models}
    coverage, task_rows = [], []
    rng = np.random.default_rng(20260907)
    weights = np.zeros((repeats, len(tasks), width))
    for ti, task in enumerate(tasks):
        ids = {cluster: i for i, cluster in enumerate(clusters[task])}
        n = len(ids)
        weights[:, ti, :n] = rng.multinomial(n, np.full(n, 1/n), size=repeats)
        task_frame = frame[frame.task == task]
        for key, block in task_frame.groupby(strata[1:], sort=True):
            pos, neg = block[block.progress == 1], block[block.progress == 0]
            pairs = len(pos) * len(neg)
            if not isinstance(key, tuple):
                key = (key,)
            coverage.append(dict(task=task, **dict(zip(strata[1:], key, strict=True)),
                                 rows=len(block), positives=len(pos), negatives=len(neg), pairs=pairs))
            if not pairs:
                continue
            pi, ni = pos.cluster.map(ids).to_numpy(), neg.cluster.map(ids).to_numpy()
            positions = (pi[:, None] * width + ni[None, :]).ravel()
            count[ti] += np.bincount(positions, minlength=width*width).reshape(width, width)
            for name in models:
                left, right = pos[name].to_numpy()[:, None], neg[name].to_numpy()[None, :]
                scores = (left > right).astype(float) + 0.5 * (left == right)
                concordant[name][ti] += np.bincount(positions, weights=scores.ravel(), minlength=width*width).reshape(width, width)
        denominator = count[ti].sum()
        for name in models:
            task_rows.append(dict(task=task, model=name, pairs=int(denominator),
                                  matched_auroc=concordant[name][ti].sum()/denominator if denominator else None))
    sampled_count = np.einsum("rti,tij,rtj->r", weights, count, weights, optimize=True)
    good = sampled_count > 0
    total = count.sum()
    support = pd.DataFrame(coverage)
    output = []
    for name in models:
        values = np.einsum("rti,tij,rtj->r", weights, concordant[name], weights, optimize=True)
        low, high = np.quantile(values[good]/sampled_count[good], [0.025, 0.975]) if good.any() else (None, None)
        output.append(dict(model=name, matched_auroc=float(concordant[name].sum()/total) if total else None,
                           low=low, high=high, pairs=int(total), strata=int((support.pairs > 0).sum()),
                           covered_rows=int(support.loc[support.pairs > 0, "rows"].sum()),
                           all_rows=len(frame), bootstrap_valid=int(good.sum())))
    return pd.DataFrame(output), support, pd.DataFrame(task_rows)
