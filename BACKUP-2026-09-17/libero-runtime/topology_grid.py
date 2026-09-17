"""Exhaustive topology-definition grid on captured MoE trajectories with known outcomes.

Stage 1 (`distances`): for every captured episode build causal distance matrices for
    kinds   route (Hellinger, as P3i) / input / shared / total / input_total (prefix-scaled RMS, as P3j)
    scopes  back_path, full_path, back_last, state_back, layer0..layer7
  and the frozen v8.2 monitor trace, and attach the episode outcome from the simulation summary.

Stage 2 (`grid`): for every matrix x history x reference length x epsilon quantile x radius scale
  x exit-confirmation rule x anchor query, compute causal recurrence-region features (P3i rules,
  parametrised), sliding-window persistence (H1) and the P3j simple distance features, then score
  how well each feature separates eventual success from failure (AUROC, pooled and task-stratified).

Nothing here is a success label at run time: outcomes are only used offline for scoring.
"""
import argparse
import csv
import json
import os
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")   # workers are process-parallel; BLAS threads would only oversubscribe
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
CODING = ROOT.parent / "coding"
sys.path.insert(0, str(CODING / "moe-control-experiments" / ".topology-deps-iUKcff"))
sys.path.insert(0, str(CODING / "robot-whisper-0909" / "moe-trap-control"))

KINDS = ("route", "input", "shared", "total", "input_total")
SCOPES = ("back_path", "full_path", "back_last", "state_back") + tuple("layer%d" % i for i in range(8))
PREFIX = 12                      # causal scaling prefix for port distances (queries)
HISTORIES = (2, 3, 5)
REFERENCES = (8, 12, 16)
QUANTILES = (0.90, 0.95, 0.99)
SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
CONFIRMS = (2, 3, 4)
ANCHORS = (12, 16, 20, 24, 28, 32)
HORIZON = 8                      # queries observed after the anchor (80 physical steps)
REGION_FEATURES = ("eligible", "outside_fraction", "exit_confirmed", "final_outside", "returned")
PERSIST_FEATURES = ("h1_lifetime", "h1_normalized", "h0_largest")
SIMPLE_FEATURES = ("lag1", "lag2", "lag4", "lag8", "lag13", "step_mean", "step_std", "step_max",
                   "net_path", "nonadjacent_nearest", "diameter", "relative_median")
V82_FEATURES = ("v82_alarm", "freeze_score", "acceleration_score", "periodicity_score")


# ----------------------------------------------------------------------------- distances
def scope_slice(a, scope):
    if scope == "back_path":
        return a[:, 4:, :, 1:]
    if scope == "full_path":
        return a[:, :, :, 1:]
    if scope == "back_last":
        return a[:, 4:, -1:, 1:]
    if scope == "state_back":
        return a[:, 4:, :, :1]
    layer = int(scope[5:])
    return a[:, layer:layer + 1, :, 1:]


def route_distance(probs, scope):
    p = scope_slice(probs, scope).astype(np.float64)
    p /= p.sum(-1, keepdims=True)
    cells = int(np.prod(p.shape[1:-1]))
    points = np.sqrt(p).reshape(len(p), -1) / np.sqrt(2 * cells)
    return squareform(pdist(points))


def port_distance(values, scope, prefix=PREFIX):
    a = scope_slice(values, scope).astype(np.float64)
    prefix = min(prefix, len(a))
    scales = np.maximum(np.sqrt(np.mean(a[:prefix] ** 2, axis=(0, 2, 3, 4))), 1e-12)
    squared = np.zeros((len(a), len(a)))
    for layer in range(a.shape[1]):
        x = a[:, layer].reshape(len(a), -1)
        squared += squareform(pdist(x, metric="sqeuclidean")) / x.shape[1] / (a.shape[1] * scales[layer] ** 2)
    return np.sqrt(squared)


def load_episode(tag_dir):
    files = sorted(tag_dir.glob("q*.npz"), key=lambda p: int(p.stem[1:]))
    steps = [int(p.stem[1:]) for p in files]
    if steps != list(range(len(files))):
        raise ValueError("Non-contiguous capture steps in %s" % tag_dir)
    routes, ports = [], {"input": [], "shared": [], "total": []}
    for path in files:
        with np.load(path) as z:
            routes.append(z["hb_router_probs"].astype(np.float32))
            for field in ports:
                ports[field].append(z["mechanism/" + field])
    return np.stack(routes), {k: np.stack(v) for k, v in ports.items()}


def v82_trace(routes):
    from v82_closed_loop import V82Monitor
    monitor, rows = V82Monitor(), []
    for probability in routes:
        status = monitor.update(probability)
        alarm = status.get("v82_alarm", status.get("alarm"))
        rows.append([float(bool(alarm)), status.get("freeze_score", np.nan),
                     status.get("acceleration_score", np.nan), status.get("periodicity_score", np.nan)])
    return np.asarray(rows, np.float64)


def find_outcome(tag, sim_roots):
    for sim in sim_roots:
        matches = list(Path(sim).glob(tag + "/episode-*/summary.json"))
        if matches:
            s = json.loads(matches[0].read_text())
            return dict(dataset=Path(sim).name, success=bool(s.get("success")), action_steps=int(s.get("action_steps", 0)),
                        task_id=int(s.get("selected_task_id", -1)), task_name=s.get("selected_task_name"),
                        benchmark=s.get("benchmark_variant"), status=s.get("status"))
    return None


def stage1_one(job):
    tag_dir, sim_roots, out_dir = job
    tag = tag_dir.name
    outcome = find_outcome(tag, sim_roots)
    if outcome is None or outcome["status"] != "completed":
        return dict(tag=tag, skipped="no completed outcome")
    try:
        routes, ports = load_episode(tag_dir)
    except Exception as error:  # noqa: BLE001
        return dict(tag=tag, skipped=repr(error))
    names, matrices = [], []
    for scope in SCOPES:
        d = {"route": route_distance(routes, scope)}
        for field in ("input", "shared", "total"):
            d[field] = port_distance(ports[field], scope)
        d["input_total"] = np.sqrt((d["input"] ** 2 + d["total"] ** 2) / 2)
        for kind in KINDS:
            names.append("%s|%s" % (kind, scope))
            matrices.append(d[kind].astype(np.float32))
    v82 = v82_trace(routes)
    out = out_dir / (tag + ".npz")
    np.savez_compressed(out, matrices=np.stack(matrices), names=np.array(names), v82=v82,
                        meta=np.array(json.dumps(dict(outcome, tag=tag, n_queries=len(routes)))))
    return dict(tag=tag, n_queries=len(routes), **outcome)


# ----------------------------------------------------------------------------- grid primitives
def delayed_distance(raw, history):
    times = np.arange(history - 1, len(raw))
    squared = sum(raw[np.ix_(times - lag, times - lag)] ** 2 for lag in range(history)) / history
    return np.sqrt(squared)


def fit_reference(distance, anchor, reference, quantile, scale, history):
    indices = np.arange(anchor - reference + 1, anchor + 1)
    local = distance[np.ix_(indices, indices)]
    separated = np.abs(indices[:, None] - indices[None, :]) >= history
    nearest = np.where(separated, local, np.inf).min(1)
    nearest = nearest[np.isfinite(nearest)]
    epsilon = max(1e-12, float(np.quantile(nearest, quantile)) * scale) if len(nearest) else 1e-12
    labels = fcluster(linkage(squareform(local, checks=False), method="complete"),
                      t=epsilon, criterion="distance").astype(int) - 1
    counts = np.zeros((int(labels.max()) + 1,) * 2, np.int64)
    np.add.at(counts, (labels[:-1], labels[1:]), 1)
    _, components = connected_components(csr_matrix(counts), directed=True, connection="strong")
    component = int(components[labels[-1]])
    selected = np.flatnonzero(components[labels] == component)
    nodes = np.flatnonzero(components == component)
    path = float(sum(local[a, b] for a, b in zip(selected[:-1], selected[1:])))
    net_ratio = 0.0 if path <= 1e-12 else float(local[selected[0], selected[-1]] / path)
    visited = len(selected) >= 4 and int(selected[-1] - selected[0]) >= 3
    cyclic = len(nodes) > 1 or bool(counts[nodes[0], nodes[0]])
    return epsilon, indices[selected], bool(visited and cyclic and net_ratio <= 0.35)


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def region_features(inside, confirm):
    exits = [(a, b) for a, b in runs(~inside) if b - a + 1 >= confirm]
    returns = [(a, b) for a, b in runs(inside) if b - a + 1 >= 2]
    came_back = any(a > end for _, end in exits for a, _ in returns)
    final_outside = bool(exits) and exits[-1][1] == len(inside) - 1 and exits[-1][1] - exits[-1][0] + 1 >= confirm + 1
    return [float(np.mean(~inside)), float(bool(exits)), float(final_outside), float(came_back)]


def persistence(distance):
    import gudhi
    tree = gudhi.RipsComplex(distance_matrix=distance).create_simplex_tree(max_dimension=2)
    tree.compute_persistence(homology_coeff_field=2, min_persistence=0)
    h0, h1 = (tree.persistence_intervals_in_dimension(d) for d in (0, 1))
    lifetime = float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.0
    diameter = float(distance.max())
    h0_largest = float(np.max(h0[np.isfinite(h0[:, 1]), 1])) if np.isfinite(h0[:, 1]).any() else 0.0
    return [lifetime, 0.0 if diameter <= 1e-12 else lifetime / diameter, h0_largest]


def simple_features(raw):
    n = len(raw)
    steps, diameter = np.diag(raw, 1), float(raw.max())
    separated = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) >= 3
    path = float(steps.sum())
    return [raw[-1, -1 - lag] if lag < n else np.nan for lag in (1, 2, 4, 8, 13)] + [
        steps.mean(), steps.std(), steps.max(), 0.0 if path <= 1e-12 else raw[0, -1] / path,
        raw[-1, :-3].min(), diameter, 0.0 if diameter <= 1e-12 else np.median(raw[separated]) / diameter]


def region_settings():
    return [(h, r, q, s) for h in HISTORIES for r in REFERENCES for q in QUANTILES for s in SCALES]


def stage2_one(path):
    with np.load(path) as z:
        matrices, names, v82 = z["matrices"].astype(np.float64), list(z["names"]), z["v82"]
        meta = json.loads(str(z["meta"]))
    Q = matrices.shape[1]
    settings = region_settings()
    region = np.full((len(names), len(settings), len(ANCHORS), len(CONFIRMS), len(REGION_FEATURES)), np.nan, np.float32)
    persist = np.full((len(names), len(HISTORIES), len(ANCHORS), len(PERSIST_FEATURES)), np.nan, np.float32)
    simple = np.full((len(names), len(ANCHORS), len(SIMPLE_FEATURES)), np.nan, np.float32)
    for m, raw in enumerate(matrices):
        delayed = {h: delayed_distance(raw, h) for h in HISTORIES}
        for ai, anchor in enumerate(ANCHORS):
            if anchor >= Q:
                continue
            window = raw[max(0, anchor - 13):anchor + 1, max(0, anchor - 13):anchor + 1]
            if len(window) >= 4:
                simple[m, ai] = simple_features(window)
            for hi, h in enumerate(HISTORIES):
                ia = anchor - (h - 1)
                d = delayed[h]
                if ia - 11 >= 0:
                    persist[m, hi, ai] = persistence(d[ia - 11:ia + 1, ia - 11:ia + 1])
                for si, (hh, r, q, s) in enumerate(settings):
                    if hh != h or ia - r + 1 < 0:
                        continue
                    if ia + HORIZON > len(d) - 1:
                        continue  # full follow-up horizon required: a truncated horizon would leak the episode end
                    epsilon, region_idx, eligible = fit_reference(d, ia, r, q, s, h)
                    follow = d[ia:ia + HORIZON + 1][:, region_idx].min(1)
                    inside = follow <= epsilon
                    for ci, confirm in enumerate(CONFIRMS):
                        region[m, si, ai, ci] = [float(eligible)] + region_features(inside, confirm)
    v82_at = np.full((len(ANCHORS), len(V82_FEATURES)), np.nan, np.float32)
    for ai, anchor in enumerate(ANCHORS):
        if anchor < len(v82):
            v82_at[ai] = [v82[:anchor + 1, 0].max(), *v82[anchor, 1:]]
    return dict(meta=meta, names=names, region=region, persist=persist, simple=simple, v82=v82_at)


# ----------------------------------------------------------------------------- scoring
def average_ranks(x):
    """Average (tie-aware) 1-based ranks along axis 0; NaN rows sort last and must be masked by the caller."""
    E = x.shape[0]
    order = np.argsort(x, axis=0, kind="stable")
    xs = np.take_along_axis(x, order, axis=0)
    new = np.ones(xs.shape, bool)
    new[1:] = xs[1:] != xs[:-1]
    idx = np.broadcast_to(np.arange(E)[:, None], xs.shape)
    first = np.maximum.accumulate(np.where(new, idx, 0), axis=0)
    ends = np.where(np.concatenate([new[1:], np.ones((1, xs.shape[1]), bool)]), idx, E - 1)
    last = np.minimum.accumulate(ends[::-1], axis=0)[::-1]
    avg = (first + last) / 2.0 + 1.0
    ranks = np.empty(xs.shape, np.float64)
    np.put_along_axis(ranks, order, avg, axis=0)
    return ranks


def auroc_columns(x, labels):
    """AUROC of every column of x [E, N] for binary labels [E]; NaN entries are excluded per column."""
    labels = np.asarray(labels, bool)
    valid = np.isfinite(x)
    n1 = (valid & labels[:, None]).sum(0)
    n0 = (valid & ~labels[:, None]).sum(0)
    ranks = average_ranks(np.where(valid, x, np.inf))
    pos_sum = np.where(valid & labels[:, None], ranks, 0.0).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (pos_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    a[(n1 == 0) | (n0 == 0)] = np.nan
    return a, n1, n0


def stratified_columns(x, labels, groups):
    values, count = np.zeros(x.shape[1]), np.zeros(x.shape[1], int)
    for g in sorted(set(groups)):
        rows = np.array([i for i, v in enumerate(groups) if v == g])
        a, _, _ = auroc_columns(x[rows], np.asarray(labels)[rows])
        ok = np.isfinite(a)
        values[ok] += a[ok]
        count += ok
    with np.errstate(invalid="ignore"):
        mean = np.where(count > 0, values / np.maximum(count, 1), np.nan)
    return mean, count


def flatten(results):
    """Stack per-episode features into X [E, N] plus one metadata tuple per column."""
    names = results[0]["names"]
    settings = region_settings()
    blocks, meta = [], []
    blocks.append(np.stack([r["v82"].reshape(-1) for r in results]))
    meta += [("v82", "", "", "", "", "", a, f) for a in ANCHORS for f in V82_FEATURES]
    blocks.append(np.stack([r["simple"].reshape(-1) for r in results]))
    meta += [(m, "", "", "", "", "", a, f) for m in names for a in ANCHORS for f in SIMPLE_FEATURES]
    blocks.append(np.stack([r["persist"].reshape(-1) for r in results]))
    meta += [(m, h, "", "", "", "", a, f) for m in names for h in HISTORIES for a in ANCHORS for f in PERSIST_FEATURES]
    region = np.stack([r["region"].reshape(-1) for r in results])
    rmeta = [(m, h, ref, q, s, c, a, f) for m in names for (h, ref, q, s) in settings for a in ANCHORS
             for c in CONFIRMS for f in REGION_FEATURES]
    keep = np.array([not (f == "eligible" and c != CONFIRMS[0]) for (_, _, _, _, _, c, _, f) in rmeta])
    blocks.append(region[:, keep])
    meta += [(m, h, ref, q, s, "" if f == "eligible" else c, a, f) for (m, h, ref, q, s, c, a, f), k in zip(rmeta, keep) if k]
    X = np.concatenate(blocks, axis=1).astype(np.float64)
    assert X.shape[1] == len(meta)
    return X, meta


def score_table(results, out_dir):
    X, meta = flatten(results)
    datasets = sorted({r["meta"]["dataset"] for r in results})
    keys = ["dataset", "matrix", "history", "reference", "quantile", "scale", "confirm", "anchor", "feature",
            "auroc", "abs_auroc", "task_auroc", "task_groups", "n_success", "n_failure"]
    rows = []
    for dataset in datasets + ["all"]:
        idx = np.array([i for i, r in enumerate(results) if dataset == "all" or r["meta"]["dataset"] == dataset])
        labels = np.array([results[i]["meta"]["success"] for i in idx])
        groups = ["%s/%s" % (results[i]["meta"]["dataset"], results[i]["meta"]["tag"][:6]) for i in idx]
        a, n1, n0 = auroc_columns(X[idx], labels)
        sa, ng = stratified_columns(X[idx], labels, groups)
        for j in np.flatnonzero(np.isfinite(a)):
            m, h, ref, q, s, c, anchor, f = meta[j]
            rows.append(dict(dataset=dataset, matrix=m, history=h, reference=ref, quantile=q, scale=s, confirm=c,
                             anchor=anchor, feature=f, auroc=round(float(a[j]), 4),
                             abs_auroc=round(abs(float(a[j]) - 0.5) + 0.5, 4),
                             task_auroc=round(float(sa[j]), 4) if np.isfinite(sa[j]) else "",
                             task_groups=int(ng[j]), n_success=int(n1[j]), n_failure=int(n0[j])))
    with (out_dir / "auroc-table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return rows


MIN_CLASS = 15


def oriented(a):
    return 0.5 + abs(a - 0.5)


def summarize(rows, results, out_dir):
    """Rank settings by outcome discrimination at anchors with enough episodes of both outcomes."""
    lines = ["# Topology definition grid: outcome discrimination", ""]
    counts = {}
    for r in results:
        counts.setdefault(r["meta"]["dataset"], [0, 0])[0 if r["meta"]["success"] else 1] += 1
    lines += ["Episodes: " + ", ".join("%s %d success / %d failure" % (k, v[0], v[1]) for k, v in sorted(counts.items())),
              "AUROC positive class = eventual success (0.5 chance; <0.5 means higher on failures). |AUROC| = 0.5 + |AUROC - 0.5|.",
              "Task AUROC = mean AUROC within base task (10 tasks), removing per-task base-rate differences.",
              "Only anchors with at least %d episodes of each outcome are ranked; late anchors keep few successes because successes end early." % MIN_CLASS, ""]
    headline = {}
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    for dataset, drows in sorted(by_dataset.items()):
        lines += ["## %s" % dataset, ""]
        usable = [r for r in drows if r["n_success"] >= MIN_CLASS and r["n_failure"] >= MIN_CLASS]
        anchors = sorted({r["anchor"] for r in usable})
        lines += ["Anchors with >= %d of each outcome: %s" % (MIN_CLASS, ", ".join("q%d" % a for a in anchors)), ""]
        lines += ["### Baselines (frozen v8.2 monitor)", "", "| anchor | feature | AUROC | task AUROC | n succ / fail |", "|---|---|---:|---:|---:|"]
        for row in usable:
            if row["matrix"] == "v82":
                lines.append("| q%s | %s | %.3f | %s | %d / %d |" % (row["anchor"], row["feature"], row["auroc"], row["task_auroc"], row["n_success"], row["n_failure"]))
        lines.append("")
        groups = {}
        for row in usable:
            if row["feature"] in REGION_FEATURES and row["feature"] != "eligible":
                key = (row["matrix"], row["history"], row["reference"], row["quantile"], row["confirm"], row["anchor"], row["feature"])
                groups.setdefault(key, []).append(row)
        ranked = []
        for key, g in groups.items():
            if len(g) < len(SCALES):
                continue
            at1 = next((r for r in g if r["scale"] == 1.0), g[0])
            task = at1["task_auroc"] if at1["task_auroc"] != "" else np.nan
            direction = np.sign(np.mean([r["auroc"] for r in g]) - 0.5) or 1.0
            worst = min(0.5 + direction * (r["auroc"] - 0.5) for r in g)
            ranked.append((oriented(task) if np.isfinite(task) else 0.0, worst, key, at1))
        ranked.sort(key=lambda x: (-x[0], -x[1]))
        lines += ["### Recurrence-region features, best 20 by task AUROC at scale 1.0 (with worst case over six radius scales)", "",
                  "| matrix | hist | ref | quantile | confirm | anchor | feature | AUROC | task AUROC | min |AUROC| over scales | n succ / fail |",
                  "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|"]
        for task, worst, key, at1 in ranked[:20]:
            lines.append("| %s | %s | %s | %s | %s | q%s | %s | %.3f | %s | %.3f | %d / %d |" % (key[0], key[1], key[2], key[3], key[4], key[5], key[6], at1["auroc"], at1["task_auroc"], worst, at1["n_success"], at1["n_failure"]))
        stable = sum(1 for _, worst, _, _ in ranked if worst > 0.5)
        strong = sum(1 for task, worst, _, _ in ranked if task >= 0.65 and worst >= 0.55)
        lines += ["", "Region settings ranked: %d; same side of 0.5 at all six scales: %d (%.1f%%); task AUROC >= 0.65 and every scale >= 0.55: %d." % (len(ranked), stable, 100.0 * stable / max(1, len(ranked)), strong), ""]
        other = [r for r in usable if r["feature"] in PERSIST_FEATURES + SIMPLE_FEATURES and r["task_auroc"] != ""]
        other.sort(key=lambda r: -oriented(r["task_auroc"]))
        lines += ["### Persistence and simple distance features, best 20 by task AUROC", "",
                  "| matrix | history | anchor | feature | AUROC | task AUROC | n succ / fail |", "|---|---:|---|---|---:|---:|---:|"]
        for row in other[:20]:
            lines.append("| %s | %s | q%s | %s | %.3f | %s | %d / %d |" % (row["matrix"], row["history"], row["anchor"], row["feature"], row["auroc"], row["task_auroc"], row["n_success"], row["n_failure"]))
        lines.append("")
        persist = [r for r in other if r["feature"] in PERSIST_FEATURES]
        lines += ["Persistence (H1/H0) features alone: best task AUROC %.3f, best pooled |AUROC| %.3f over %d rows." % (
            max((oriented(r["task_auroc"]) for r in persist), default=0.5), max((r["abs_auroc"] for r in persist), default=0.5), len(persist)), ""]
        headline[dataset] = dict(anchors=anchors, region_ranked=len(ranked), region_stable=stable, region_strong=strong,
                                 best_region=[dict(key=list(k), auroc=a["auroc"], task_auroc=a["task_auroc"], worst=w) for t, w, k, a in ranked[:5]],
                                 best_other=[dict(matrix=r["matrix"], history=r["history"], anchor=r["anchor"], feature=r["feature"], auroc=r["auroc"], task_auroc=r["task_auroc"]) for r in other[:5]],
                                 v82=[dict(anchor=r["anchor"], feature=r["feature"], auroc=r["auroc"], task_auroc=r["task_auroc"]) for r in usable if r["matrix"] == "v82"])
    (out_dir / "REPORT.md").write_text("\n".join(lines))
    (out_dir / "headline.json").write_text(json.dumps(headline, indent=1))
    return lines


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("distances", "grid"))
    p.add_argument("--capture-root", type=Path, default=ROOT.parent / "moe-capture" / "topo-20260916")
    p.add_argument("--sims", nargs="+", default=[str(ROOT / "simulations/topo-libero10"), str(ROOT / "simulations/topo-plus")])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--exclude-prefix", default="smoke")
    a = p.parse_args()
    out = a.out or (a.capture_root / "_grid")
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "distances":
        ep_dir = out / "episodes"
        ep_dir.mkdir(exist_ok=True)
        tags = sorted(d for s in a.capture_root.glob("server-p*") if s.is_dir() for d in s.iterdir()
                      if d.is_dir() and not d.name.startswith(a.exclude_prefix))
        jobs = [(d, a.sims, ep_dir) for d in tags]
        with Pool(a.workers) as pool:
            rows = pool.map(stage1_one, jobs, chunksize=1)
        (out / "episodes.json").write_text(json.dumps(rows, indent=1))
        done = [r for r in rows if "skipped" not in r]
        print(json.dumps(dict(episodes=len(rows), usable=len(done), skipped=len(rows) - len(done),
                              success=sum(r["success"] for r in done), matrices=len(KINDS) * len(SCOPES))), flush=True)
    else:
        import pickle
        cache = out / "features.pkl"
        if cache.exists():
            results = pickle.loads(cache.read_bytes())
        else:
            paths = sorted((out / "episodes").glob("*.npz"))
            with Pool(a.workers) as pool:
                results = pool.map(stage2_one, paths, chunksize=1)
            cache.write_bytes(pickle.dumps(results))
        rows = score_table(results, out)
        lines = summarize(rows, results, out)
        print("\n".join(lines[:60]), flush=True)
        print(json.dumps(dict(episodes=len(results), rows=len(rows), settings=len(region_settings()) * len(results[0]["names"]))), flush=True)


if __name__ == "__main__":
    main()
