"""E4 onset 对齐图：三人群 V+A z 对齐 loop onset（lead −4..+8），2 panel（grid/main）。

系列：recovering loop（蓝 #2a78d6）、fatal loop（橙 #eb6834）、
matched clean success（灰参照带）。均值 ± 1.96·SEM；n<8 的 lead 不画（图注声明）。
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype/E4_recurrence_lockin"
z = np.load(f"{OUT}/curves_onset_aligned.npz")
leads = z["leads"]

INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
SERIES = [
    ("recovering_loop", "Recovering loop (success)", "#2a78d6"),
    ("fatal_loop", "Fatal loop (failure)", "#eb6834"),
    ("clean_success_matched", "Clean success (matched query)", "#898781"),
]
MIN_N = 8

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), dpi=200, sharey=True)
fig.patch.set_facecolor(SURF)
handles = []
for ax, corpus, title in zip(
        axes, ("grid50x8", "main16x32"),
        ("grid50x8 (4 suites pooled, red-flag tasks excluded)", "main16x32 (5 tasks)")):
    ax.set_facecolor(SURF)
    ns = []
    for key, label, col in SERIES:
        M = z[f"{corpus}__{key}"]
        ns.append(M.shape[0])
        n = np.isfinite(M).sum(0)
        with np.errstate(all="ignore"):
            mu = np.nanmean(M, 0)
            se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.maximum(n, 1))
        ok = n >= MIN_N
        mu, se = np.where(ok, mu, np.nan), np.where(ok, se, np.nan)
        band = 1.96 * se
        if key == "clean_success_matched":
            ax.fill_between(leads, mu - band, mu + band, color=col, alpha=0.22, lw=0)
            h, = ax.plot(leads, mu, color=col, lw=1.6, ls="--", label=label)
        else:
            ax.fill_between(leads, mu - band, mu + band, color=col, alpha=0.16, lw=0)
            h, = ax.plot(leads, mu, color=col, lw=2.0, marker="o", ms=4.0, label=label)
        if ax is axes[0]:
            handles.append(h)
    ax.axvline(0, color=AXIS, lw=1.2, ls=":", zorder=0)
    ax.text(-0.12, 0.02, "loop\nonset", transform=ax.get_xaxis_transform(),
            color=MUTED, fontsize=7.5, ha="right", va="bottom")
    ax.axhline(0, color=AXIS, lw=1.0, zorder=0)
    ax.set_title(title, fontsize=9.5, color=INK, pad=8)
    ax.set_xlabel("Lead relative to loop onset (queries)", fontsize=9, color=INK)
    ax.set_xticks(leads[::2])
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.text(0.985, 0.02, f"n = {ns[0]} rec / {ns[1]} fatal / {ns[2]} clean",
            transform=ax.transAxes, color=MUTED, fontsize=7.5, ha="right", va="bottom")
axes[0].set_ylabel("Pulse score  z(V+A), group-robust", fontsize=9, color=INK)
fig.suptitle("V+A pulse score aligned to loop onset — mean ± 1.96 SEM "
             f"(leads with n<{MIN_N} suppressed)", fontsize=10, color=INK, y=0.995)
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8.5,
           frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, 0.965))
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(f"{OUT}/fig_onset_aligned.png", facecolor=SURF, bbox_inches="tight")
print("saved fig_onset_aligned.png")
