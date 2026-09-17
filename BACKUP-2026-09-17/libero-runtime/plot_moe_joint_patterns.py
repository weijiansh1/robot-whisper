"""Standalone research figures and actual observation contact sheets."""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/data/coding/v8-methods")
from moe_joint_patterns import INDEX, MOTIF_NAMES, motif_scores

OUT = Path("/data/libero-runtime/samples/moe-joint-patterns-20260915")
SOURCE = Path("/data/libero-runtime/samples/v82-evaluation-20260914T144322Z")
COLORS = ["#2678a8", "#db6b46", "#32916c", "#b44671", "#9a8b29", "#777777"]
LABELS = ["Frozen + active", "Frozen + diversity loss", "State/action split",
          "Cycling + churn", "Front/back split", "Churn to cooling"]


def load_csv(name):
    with (OUT / name).open() as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=180, facecolor="white")
    fig.savefig(OUT / (name + ".pdf"), facecolor="white")
    plt.close(fig)


def overview(episodes, features):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for e in episodes:
        x = features[e["name"]]
        success = e["source_result"]["success"]
        color = "#32916c" if success else "#bf4c53"
        axes[0].scatter(x[18, INDEX["mobility_log_ratio"]], x[18, INDEX["acceleration_log_ratio"]],
                        c=color, marker="o" if e["benchmark"] == "plus" else "x", s=38, alpha=.8)
    axes[0].axvline(-np.log(1.2), color=".6", ls="--", lw=1)
    axes[0].axhline(np.log(1.2), color=".6", ls="--", lw=1)
    axes[0].set(xlabel="Cross-chunk mobility log ratio", ylabel="Within-flow acceleration log ratio",
                title="Same observed prefix: query 18")
    axes[0].scatter([], [], c="#32916c", label="Plus success (9)")
    axes[0].scatter([], [], c="#bf4c53", label="Plus failure (21)")
    axes[0].scatter([], [], c="#bf4c53", marker="x", label="Pro failure (30)")
    axes[0].legend(fontsize=8, loc="upper right")
    counts = load_csv("event-counts.csv")
    methods = ["freeze_control", "frozen_active", "frozen_collapsed", "cycling_churn", "joint_union", "v82_coverage"]
    labels = ["Freeze control", "Frozen + active", "Frozen + diversity loss", "Cycling + churn", "Any joint motif", "v8.2 coverage"]
    y = np.arange(len(methods))
    for j, (limit, color) in enumerate([(18, "#88afbb"), (36, "#2678a8"), (51, "#22494c")]):
        chosen = [next(r for r in counts if r["factor"] == "1.2" and r["benchmark"] == "all"
                       and int(r["last_query"]) == limit and r["method"] == m) for m in methods]
        values = [int(r["detected_failures"]) for r in chosen]
        axes[1].barh(y + .24 * (j - 1), values, .22, color=color,
                     label="180 actions" if limit == 18 else "360 actions" if limit == 36 else "Full episode")
    axes[1].set(yticks=y, yticklabels=labels, xlim=(0, 53), xlabel="Failed episodes with an event / 51",
                title="Primary factor 1.2; 3-query confirmation")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.1), ncol=3)
    axes[2].axis("off")
    table_rows = [[label, "%s / 51" % next(r["detected_failures"] for r in counts if r["factor"] == "1.2"
                    and r["benchmark"] == "all" and r["last_query"] == "51" and r["method"] == m),
                   "%s / 9" % next(r["flagged_successes"] for r in counts if r["factor"] == "1.2"
                    and r["benchmark"] == "all" and r["last_query"] == "51" and r["method"] == m)]
                  for label, m in zip(labels, methods)]
    table = axes[2].table(cellText=table_rows, colLabels=["Full episode", "Failure", "Success"],
                         colWidths=[.55, .23, .22], cellLoc="center", bbox=[0, .32, 1, .59])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    axes[2].text(0, .2, "All 9 successes end before 360 actions.\nLate counts have unequal observation time.\nThese are development data, not blind tests.",
                 fontsize=9, va="top", linespacing=1.6)
    fig.suptitle("Joint MoE motifs: observed associations and limits", fontsize=14)
    save_figure(fig, "joint-pattern-overview")


def cases(episodes, features, external):
    names = ["plus-task05-init026", "plus-task05-init047", "plus-task04-init026", "plus-task04-init047",
             "plus-task07-init047", "plus-task07-init026"]
    by_name = {e["name"]: e for e in episodes}
    fig, axes = plt.subplots(len(names), 3, figsize=(14, 15), layout="constrained")
    for row, name in enumerate(names):
        x = features[name]
        steps = np.arange(len(x)) * 10
        score = motif_scores(x)
        success = by_name[name]["source_result"]["success"]
        for col, label, color in [("mobility_log_ratio", "Mobility", COLORS[0]),
                                  ("acceleration_log_ratio", "Flow acceleration", COLORS[1]),
                                  ("diversity_log_ratio", "Token diversity", COLORS[2])]:
            axes[row, 0].plot(steps, x[:, INDEX[col]], color=color, label=label, lw=1.5)
        axes[row, 0].axhline(0, color=".75", lw=.8)
        axes[row, 0].set(title=name + (" | SUCCESS" if success else " | FAILURE"), ylabel="Log ratio to q1-4")
        for k in (0, 1, 3):
            axes[row, 1].plot(steps, score[:, k], color=COLORS[k], label=LABELS[k], lw=1.5)
        axes[row, 1].axhline(np.log(1.2), color="black", ls="--", lw=.8)
        axes[row, 1].set(ylabel="Confirmed joint score", title="Current pattern, not a latched alarm")
        axes[row, 2].plot(steps, external[name][:, 0] * 1000, color="#333333", label="Observed EEF displacement")
        axes[row, 2].set(ylabel="EEF displacement / chunk (mm)", title="External check only")
        for ax in axes[row]:
            ax.set_xlim(0, 520)
            ax.grid(alpha=.14)
            if row == len(names) - 1:
                ax.set_xlabel("Executed actions before query")
        if row == 0:
            axes[row, 0].legend(fontsize=7, loc="upper right")
            axes[row, 1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Same-task examples and uncovered failure; case selection is post-analysis", fontsize=14)
    save_figure(fig, "joint-pattern-cases")
    frames(names, by_name)


def frames(names, by_name):
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 16)
    small = ImageFont.truetype(font_path, 14)
    tile, gutter, top, row_h = 224, 12, 54, 278
    sheet = Image.new("RGB", (5 * tile + 6 * gutter, top + len(names) * row_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), "Recorded observations. No inferred trap labels. Query time is before the displayed action chunk.",
              fill="#222222", font=font)
    metadata = []
    for row, name in enumerate(names):
        e = by_name[name]
        path = Path(e["source_artifact_dir"]) / "episode-trace.npz"
        with np.load(path) as archive:
            images, final_image = archive["images"], archive["final_image"]
        queries = [7, 18, 27, 51]
        y = top + row * row_h
        title = name + (" | SUCCESS" if e["source_result"]["success"] else " | FAILURE")
        draw.text((12, y), title, fill="#222222", font=font)
        for col, q in enumerate(queries):
            px = gutter + col * (tile + gutter)
            if q < len(images):
                sheet.paste(Image.fromarray(images[q]), (px, y + 24))
                draw.text((px, y + 251), "q%d | %d actions" % (q, q * 10), fill="#333333", font=small)
            else:
                draw.rectangle((px, y + 24, px + tile - 1, y + 247), fill="#f2f3f4")
                draw.text((px + 24, y + 115), "Episode already ended", fill="#666666", font=small)
                draw.text((px, y + 251), "q%d | unobserved" % q, fill="#666666", font=small)
        px = gutter + 4 * (tile + gutter)
        sheet.paste(Image.fromarray(final_image), (px, y + 24))
        draw.text((px, y + 251), "Endpoint | %d actions" % e["action_steps"], fill="#333333", font=small)
        metadata.append(dict(episode=name, displayed_queries=queries, source_trace=str(path)))
    sheet.save(OUT / "recorded-case-frames.jpg", quality=94)
    (OUT / "case-frame-index.json").write_text(json.dumps(metadata, indent=2) + "\n")


def interventions():
    rows = load_csv("branch-query-scores.csv")
    names = list(dict.fromkeys(r["parent"] for r in rows))
    with np.load(OUT / "branch-features.npz") as archive:
        features = {k: archive[k] for k in archive.files}
    fig, axes = plt.subplots(5, 2, figsize=(12, 12), layout="constrained")
    for i, name in enumerate(names):
        for candidate in range(4):
            selected = [r for r in rows if r["parent"] == name and int(r["candidate"]) == candidate]
            q = np.asarray([int(r["query"]) for r in selected])
            target = np.asarray([float(r["target_severity"]) for r in selected])
            x = features[name + "__candidate%d" % candidate]
            joint = motif_scores(x)
            max_joint = np.max(joint[q], axis=1)
            axes[i, 0].plot((q - q[0]) * 10, target, label="c%d" % candidate, color=COLORS[candidate])
            axes[i, 1].plot((q - q[0]) * 10, max_joint, label="c%d" % candidate, color=COLORS[candidate])
        axes[i, 0].axhline(0, color="black", ls="--", lw=.8)
        axes[i, 1].axhline(np.log(1.2), color="black", ls="--", lw=.8)
        axes[i, 0].set(title=name + " | " + selected[0]["head"], ylabel="Target severity (margin units)")
        axes[i, 1].set(title="All candidates " + ("SUCCEEDED" if name == "plus-task03-init039" else "FAILED"),
                       ylabel="Maximum confirmed joint score")
        for ax in axes[i]:
            ax.set_xlim(0, 70)
            ax.grid(alpha=.15)
            if i == 0:
                ax.legend(fontsize=8, ncol=4)
            if i == 4:
                ax.set_xlabel("Actions since candidate intervention")
    fig.suptitle("Reanalysis of 20 existing simulation branches, 5 independent parents", fontsize=14)
    save_figure(fig, "joint-pattern-interventions")


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    episodes = json.loads((SOURCE / "summary.json").read_text())["episodes"]
    with np.load(OUT / "episode-features.npz") as archive:
        features = {k: archive[k] for k in archive.files}
    with np.load(OUT / "offline-external.npz") as archive:
        external = {k: archive[k] for k in archive.files}
    overview(episodes, features)
    cases(episodes, features, external)
    interventions()
    print("Wrote three PNG/PDF figures and actual-observation contact sheet", flush=True)


if __name__ == "__main__":
    main()
