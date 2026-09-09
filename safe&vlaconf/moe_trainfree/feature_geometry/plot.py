"""Draw SAFE-style recolorings and synchronized recorded-state case studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from analyze import SEED, REPRESENTATIONS, ROOT, digest, write_json

HERE = Path(__file__).resolve().parent
SUCCESS = "#14796d"
FAILURE = "#cf493b"
LABELS = {"routing_full": "Expert distribution, all queries",
          "routing_matched": "Expert distribution, paired q >= 7",
          "dynamics_matched": "v7 layer dynamics, paired q >= 7"}
MODE_NAMES = ("Success", "Release/drop", "Grasp/contact", "Goal unmet/regressed", "Holding timeout", "Other")
MODE_COLORS = ("#3264ae", "#d74332", "#ba8a10", "#8a5ea9", "#288e78", "#808080")


def save(figure, path):
    figure.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def load_projection(root, suite, representation, seed=SEED):
    kind = representation.split("_", 1)[1]
    meta = pd.read_csv(root / "projections" / f"{suite}_{kind}_points.csv")
    with np.load(root / "projections" / f"{suite}_{representation}_seed{seed}.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(meta.global_row, data["point_global_rows"])
    np.testing.assert_array_equal(meta["query"], data["point_queries"])
    return meta, data


def mode_ids(meta):
    def category(row):
        reason = str(row.primary_failure_reason)
        if not row.failure:
            return 0
        if "released" in reason or "dropped" in reason:
            return 1
        if "grasp" in reason or "contact" in reason:
            return 2
        if "holding" in reason:
            return 4
        if "goal" in reason or "mechanism" in reason or "progress" in reason:
            return 3
        return 5
    return np.asarray([category(r) for r in meta.itertuples()])


def decorate(axis, xy):
    lo, hi = xy.min(0), xy.max(0)
    padding = np.maximum((hi - lo) * .04, 1e-3)
    axis.set_xlim(lo[0] - padding[0], hi[0] + padding[0])
    axis.set_ylim(lo[1] - padding[1], hi[1] + padding[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#d0d0d0")


def scatter(axis, xy, meta, view, tasks):
    order = np.random.default_rng(SEED).permutation(len(meta))
    z, data = xy[order], meta.iloc[order]
    kwargs = dict(s=4, alpha=.65, linewidths=0, rasterized=True)
    if view == "progress":
        artist = axis.scatter(*z.T, c=data.safe_color, cmap="coolwarm", vmin=0, vmax=1, **kwargs)
    elif view == "task":
        artist = axis.scatter(*z.T, c=[plt.get_cmap("tab10")(tasks.index(t)) for t in data.task], **kwargs)
    elif view == "mode":
        artist = axis.scatter(*z.T, c=[MODE_COLORS[i] for i in mode_ids(data)], **kwargs)
    elif view == "motion":
        artist = axis.scatter(*z.T, c=np.maximum(data.eef_motion_m, 1e-5), cmap="viridis",
                              norm=LogNorm(1e-5, .15), **kwargs)
    else:
        raise ValueError(view)
    decorate(axis, xy)
    return artist


def atlas(root, suite):
    figure = plt.figure(figsize=(17, 13), layout="constrained")
    grid = figure.add_gridspec(4, 4, height_ratios=[1, 1, 1, .14])
    axes = np.asarray([[figure.add_subplot(grid[row, col]) for col in range(4)] for row in range(3)])
    legends = [figure.add_subplot(grid[3, col]) for col in range(4)]
    for axis in legends:
        axis.axis("off")
    titles = ("Outcome / failed-rollout progress", "Task identity", "Final failure type (all prefixes)", "Past EEF motion (m / query)")
    tasks = None
    for row, representation in enumerate(REPRESENTATIONS):
        meta, data = load_projection(root, suite, representation)
        tasks = sorted(meta.task.unique())
        for col, view in enumerate(("progress", "task", "mode", "motion")):
            artist = scatter(axes[row, col], data["xy"], meta, view, tasks)
            center = (data["xy"].max(0) + data["xy"].min(0)) / 2
            radius = float(np.ptp(data["xy"], axis=0).max()) * .54
            axes[row, col].set_xlim(center[0] - radius, center[0] + radius)
            axes[row, col].set_ylim(center[1] - radius, center[1] + radius)
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=11)
            if col == 0:
                axes[row, col].set_ylabel(LABELS[representation] + f"\n{len(meta):,} points", fontsize=11)
            if row == 2 and view in ("progress", "motion"):
                cax = legends[col].inset_axes([.18, .5, .64, .28])
                figure.colorbar(artist, cax=cax, orientation="horizontal")
    task_handles = [Line2D([], [], color=plt.get_cmap("tab10")(i), marker="o", linestyle="none", markersize=4,
                           label=f"T{i+1}") for i in range(len(tasks))]
    mode_handles = [Line2D([], [], color=c, marker="o", linestyle="none", markersize=4, label=n)
                    for n, c in zip(MODE_NAMES, MODE_COLORS)]
    legends[1].legend(handles=task_handles, loc="center", ncol=5, fontsize=8, frameon=False)
    legends[2].legend(handles=mode_handles, loc="center", ncol=2, fontsize=7, frameon=False)
    figure.suptitle(f"{suite}: same coordinates, different annotations\n"
                    "Success stays blue; failed rollouts change from blue (early) to red (late). "
                    "Sampling is enriched for failures.", fontsize=13)
    save(figure, root / "figures" / f"{suite}_atlas")
    return [{"suite": suite, "label": f"T{i+1}", "task": task} for i, task in enumerate(tasks)]


def sensitivity(root, suite):
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    for row, representation in enumerate(REPRESENTATIONS[1:]):
        for col, seed in enumerate((SEED, SEED + 1, SEED + 2)):
            meta, data = load_projection(root, suite, representation, seed)
            init_label = "PCA init"
            alternate = root / "random_init" / f"{suite}_{representation}_seed{seed}.npz"
            if seed != SEED and alternate.exists():
                with np.load(alternate, allow_pickle=False) as z:
                    data.update(xy=z["xy"], neighbor_overlap=z["neighbor_overlap"])
                init_label = "Random init"
            scatter(axes[row, col], data["xy"], meta, "progress", sorted(meta.task.unique()))
            overlap = float(data["neighbor_overlap"].mean())
            axes[row, col].set_title(f"{init_label}, seed {seed}\n15-NN preservation {overlap:.1%}", fontsize=10)
        scatter(axes[row, 3], data["pca"], meta, "progress", sorted(meta.task.unique()))
        axes[row, 3].set_title(f"PCA (same points)\nVariance retained {data['pca_explained_variance_ratio'].sum():.1%}", fontsize=10)
        axes[row, 0].set_ylabel(LABELS[representation])
    figure.suptitle(f"{suite}: initialization sensitivity and a linear projection", fontsize=13)
    save(figure, root / "figures" / f"{suite}_sensitivity")


def snapshot_queries(case):
    end = case["length"] - 1
    if case["release_query"] is None:
        return np.linspace(0, end, 6).astype(int).tolist()
    event = int(case["release_query"])
    chosen = {0, max(0, event - 3), event, min(end, event + 3), int(.75 * end), end}
    for q in np.linspace(0, end, 12).astype(int):
        if len(chosen) == 6:
            break
        chosen.add(int(q))
    return sorted(chosen)


def path_plot(axis, xy, meta, cases):
    axis.scatter(*xy.T, c="#8b9397", s=3, alpha=.13, linewidths=0, rasterized=True)
    for case in cases:
        mask = meta.global_row.to_numpy() == case["global_row"]
        path = xy[mask]
        queries = meta.loc[mask, "query"].to_numpy()
        color = FAILURE if case["role"] == "failure" else SUCCESS
        axis.plot(*path.T, color=color, lw=1.2, marker=".", markersize=3, label=case["role"].capitalize())
        for i in range(0, len(path) - 1, max(1, len(path) // 5)):
            axis.annotate("", xy=path[i+1], xytext=path[i], arrowprops=dict(arrowstyle="->", color=color, lw=1))
        marked = {len(path) - 1}
        axis.scatter(*path[0], s=28, marker="o", facecolor="white", edgecolor=color, zorder=4)
        if case["release_query"] is not None and int(case["release_query"]) in queries:
            marked.add(int(np.flatnonzero(queries == case["release_query"])[0]))
        for i in sorted(marked):
            offset = (4, 8) if case["role"] == "failure" else (4, -13)
            axis.annotate(f"q{queries[i]}", path[i], xytext=offset, textcoords="offset points", fontsize=8,
                          color=color, bbox=dict(fc="white", ec="none", alpha=.75, pad=.7))
    decorate(axis, xy)
    axis.legend(fontsize=8, loc="best")


def case_study(root, suite, cases):
    selected = sorted([c for c in cases if c["suite"] == suite], key=lambda c: c["role"], reverse=True)
    figure = plt.figure(figsize=(17, 10), layout="constrained")
    subfigs = figure.subfigures(3, 1, height_ratios=[1.8, 1.3, 1.3])
    axes = subfigs[0].subplots(1, 4)
    full_meta, full = load_projection(root, suite, "routing_full")
    dyn_meta, dyn = load_projection(root, suite, "dynamics_matched")
    path_plot(axes[0], full["xy"], full_meta, selected)
    axes[0].set_title("Expert distribution: full path", fontsize=11)
    path_plot(axes[1], dyn["xy"], dyn_meta, selected)
    axes[1].set_title("v7 dynamics: path from q7", fontsize=11)
    for case in selected:
        data = full_meta.loc[full_meta.global_row == case["global_row"]]
        color = FAILURE if case["role"] == "failure" else SUCCESS
        axes[2].plot(data["query"], data.freeze_score, color=color, label=case["role"])
        axes[3].plot(data["query"], data.eef_motion_m * 100, color=color, label=case["role"])
        if case["release_query"] is not None:
            for axis in axes[2:]:
                axis.axvline(case["release_query"], ls="--", color="#333333", lw=1, label="Matched release")
    axes[2].set(title="Original v7 freeze signal", xlabel="Decision query", ylabel="Relative freeze (log ratio)")
    axes[3].set(title="Recorded past EEF motion", xlabel="Decision query", ylabel="cm / query")
    axes[2].legend(fontsize=8)
    for row, case in enumerate(selected, 1):
        images = subfigs[row].subplots(1, 6)
        subfigs[row].suptitle(f"{case['role'].capitalize()} | same init {case['init_state_id']} | "
                             f"episode {case['episode']}, noise seed {case['noise_seed']}", fontsize=11)
        for axis, query in zip(images, snapshot_queries(case)):
            axis.imshow(Image.open(root / "frames" / str(case["global_row"]) / f"q{query:03d}.png"))
            suffix = " | release" if query == case["release_query"] else ""
            axis.set_title(f"q{query}{suffix}", fontsize=10)
            axis.axis("off")
    task = selected[0]["task"].split("/", 1)[1].replace("_", " ")
    figure.suptitle(textwrap.fill(f"{suite}: {task}", 125) + "\nRecorded-state redraws; final tile is the last recorded query, not necessarily the terminal state.", fontsize=12)
    save(figure, root / "figures" / f"{suite}_case")


def case_animation(root, case):
    suite = case["suite"]
    full_meta, full = load_projection(root, suite, "routing_full")
    dyn_meta, dyn = load_projection(root, suite, "dynamics_matched")
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.7), dpi=95, layout="constrained")
    image = axes[0].imshow(Image.open(root / "frames" / str(case["global_row"]) / "q000.png"))
    axes[0].axis("off")
    trajectories = []
    for axis, meta, data, title in ((axes[1], full_meta, full, "Expert distribution"),
                                    (axes[2], dyn_meta, dyn, "v7 dynamics (from q7)")):
        axis.scatter(*data["xy"].T, color="#8b9397", s=2, alpha=.2, linewidths=0)
        decorate(axis, data["xy"])
        axis.set_title(title, fontsize=11)
        line, = axis.plot([], [], color=FAILURE, lw=1, marker=".", markersize=2)
        marker, = axis.plot([], [], color=FAILURE, marker="*", markersize=11)
        mask = meta.global_row.to_numpy() == case["global_row"]
        trajectories.append((line, marker, data["xy"][mask], meta.loc[mask, "query"].to_numpy()))
    title = figure.suptitle("", fontsize=11)
    frames = []
    for query in range(case["length"]):
        image.set_data(Image.open(root / "frames" / str(case["global_row"]) / f"q{query:03d}.png"))
        for line, marker, xy, queries in trajectories:
            visible = xy[queries <= query]
            if len(visible):
                line.set_data(visible[:, 0], visible[:, 1])
                marker.set_data(visible[-1:, 0], visible[-1:, 1])
        event = " | matched release" if query == case["release_query"] else ""
        title.set_text(f"{suite} | failed episode {case['episode']} | recorded query {query}{event}")
        figure.canvas.draw()
        frames.append(Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy()).convert("RGB"))
    plt.close(figure)
    frames[0].save(root / "figures" / f"{suite}_recorded_path.gif", save_all=True, append_images=frames[1:],
                   duration=300, loop=0, optimize=False)


def diagnostics(root):
    points = pd.read_csv(root / "neighbor_points.csv")
    keys = ["suite", "representation", "query", "failure"]
    columns = ["neighbor_failure_rate", "candidate_failure_rate", "excess_failure_neighbor_rate"]
    task_means = points.groupby(keys + ["task"])[columns].mean()
    summary = task_means.groupby(level=keys).mean().reset_index()
    summary = summary.merge(points.groupby(keys).size().rename("episodes").reset_index(), on=keys)
    summary = summary.merge(task_means.groupby(level=keys).size().rename("tasks").reset_index(), on=keys)
    summary.to_csv(root / "neighborhood_summary.csv", index=False)
    figure, axes = plt.subplots(1, 4, figsize=(16, 3.5), layout="constrained", sharey=True)
    for axis, suite in zip(axes, sorted(points.suite.unique())):
        part = summary.loc[(summary.suite == suite) & summary.failure]
        for name, color in (("routing", "#456eb0"), ("dynamics", "#b0518e")):
            rows = part.loc[part.representation == name].sort_values("query")
            axis.plot(rows["query"], rows.excess_failure_neighbor_rate * 100, "o-", color=color, label=name)
        axis.axhline(0, color="#333333", ls="--", lw=.8)
        axis.set(title=suite.replace("libero_", ""), xlabel="Fixed absolute query", xticks=[7, 14, 21])
    axes[0].set_ylabel("Excess failure neighbors (percentage points)")
    axes[0].legend(fontsize=9)
    figure.suptitle("Original high-dimensional space: failure-query neighbors from OTHER tasks\n"
                    "20 neighbors; subtract eligible-candidate failure rate; task-macro average", fontsize=12)
    save(figure, root / "figures" / "cross_task_neighborhood")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round4_geometry")
    args = parser.parse_args()
    root = args.input.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        if digest(root / name) != expected:
            raise AssertionError(f"geometry artifact changed: {name}")
    (root / "figures").mkdir(exist_ok=True)
    cases = json.loads((root / "cases.json").read_text())
    mapping = []
    for suite in sorted({c["suite"] for c in cases}):
        mapping.extend(atlas(root, suite))
        sensitivity(root, suite)
        case_study(root, suite, cases)
        case_animation(root, next(c for c in cases if c["suite"] == suite and c["role"] == "failure"))
        print(f"PLOTTED {suite}", flush=True)
    pd.DataFrame(mapping).to_csv(root / "task_key.csv", index=False)
    diagnostics(root)
    write_json(root / "figure_manifest.json", {"plotter_sha256": digest(Path(__file__)),
        "files": {str(p.relative_to(root)): digest(p) for p in sorted((root / "figures").iterdir())}})
    print(root / "figures", flush=True)


if __name__ == "__main__":
    main()
