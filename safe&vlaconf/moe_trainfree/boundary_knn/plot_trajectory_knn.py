"""A complete recorded rollout: exact kNN neighbors, state frames, and timing."""

import argparse
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

from knn import HERE, ROOT, PRIMARY, ReferenceScorer, dynamics
from core import digest, write_json

GOOD = "#247e79"
BAD = "#bd425b"
NEIGHBOR = "#d39825"
HISTORY = "#426ca1"
GRAY = "#92979d"
INK = "#22292d"


def configure_font():
    source = ROOT / "himoe-route-capture/assets/NotoSansSC.ttf"
    checksum = digest(source)
    regular = Path(tempfile.gettempdir()) / f"himoe-knn-noto-regular-{checksum[:12]}.ttf"
    if not regular.exists():
        # The bundled variable font defaults to Thin; bake Regular for exports.
        font = instantiateVariableFont(TTFont(source), {"wght": 400}, inplace=True)
        for platform, encoding, language in ((3, 1, 1033), (1, 0, 0)):
            for name_id, value in ((1, "Himoe Noto Sans SC"), (2, "Regular"),
                                   (16, "Himoe Noto Sans SC"), (17, "Regular"),
                                   (6, "HimoeNotoSansSC-Regular")):
                font["name"].setName(value, name_id, platform, encoding, language)
        font.save(regular)
        font.close()
    font_manager.fontManager.addfont(regular)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=regular).get_name(),
                         "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "savefig.facecolor": "white", "text.color": INK})
    return checksum


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def prepare(output, row, fold):
    original = HERE.parent / "results/round5_knn"
    geometry = HERE.parent / "results/round4_geometry"
    frame = pd.read_csv(original / "outcome_alignment.csv")
    episode = frame.iloc[row]
    cases = json.loads((geometry / "cases.json").read_text())
    case = next(case for case in cases if case["global_row"] == row)
    manifest = json.loads((original / "sealed_manifest.json").read_text())
    inputs = {}
    for path in (original / "outcome_alignment.csv", geometry / "cases.json",
                 geometry / "render_verification.json", original / "sealed_manifest.json"):
        inputs[str(path.relative_to(ROOT))] = digest(path)
    loaded = {}
    for kind in ("profiles", "predictions"):
        path = original / kind / f"{fold}.npz"
        checksum = digest(path)
        assert checksum == manifest["artifacts"][str(path.relative_to(original))]
        inputs[str(path.relative_to(ROOT))] = checksum
        loaded[kind] = load_npz(path)
    profile, prediction = loaded["profiles"], loaded["predictions"]
    position = int(np.flatnonzero(prediction["test_rows"] == row)[0])
    method = list(prediction["methods"]).index(PRIMARY)
    alpha = int(np.flatnonzero(np.isclose(prediction["alphas"], .05))[0])
    tau = float(prediction["thresholds"][1, alpha, method])
    length = int(episode.length)
    score = prediction["scores"][method, position, :length].astype(float)
    first = int(prediction["first"][1, alpha, method, position])
    valid = np.isfinite(score)
    assert not valid[:7].any() and valid[7:].all()
    hits = np.flatnonzero(score > tau)
    assert first == (int(hits[0]) if len(hits) else -1)
    cache_path = HERE.parent / "results/round3_safe/v7/v7_inputs.npz"
    inputs[str(cache_path.relative_to(ROOT))] = digest(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        dynamic, _ = dynamics(cache["mobility"][[row]], cache["acceleration"][[row]],
                              cache["periodicity"][[row]], float(profile["periodicity_scale"]))
        assert cache["valid"][row, :length].all() and not cache["valid"][row, length:].any()
    normalized = (dynamic[0, :length].astype(float) - profile["dynamic_center"]) / profile["dynamic_scale"]
    xy = (normalized - profile["pca_mean"]) @ profile["pca_components"].T
    distances, identities = ReferenceScorer(profile).neighbors("success_dynamic", normalized[valid])
    np.testing.assert_allclose(distances.mean(1), score[valid], rtol=2e-6, atol=2e-6)
    for current, expected in zip(normalized[valid], distances):
        direct = np.sort(np.linalg.norm(profile["success_dynamic"] - current, axis=1))[:20]
        np.testing.assert_allclose(direct, expected, rtol=1e-10, atol=1e-10)
    bank_xy = (profile["success_dynamic"] - profile["pca_mean"]) @ profile["pca_components"].T
    np.testing.assert_allclose(bank_xy, profile["success_pca2"], atol=1e-12)
    neighbor_ids = np.full((length, 20), -1, dtype=int)
    neighbor_distances = np.full((length, 20), np.nan)
    neighbor_ids[valid], neighbor_distances[valid] = identities, distances

    source = ROOT / "VLA_MUI_HUB" / episode.source
    summary_path = source / "client/summaries.json"
    summary = next(v for v in json.loads(summary_path.read_text()) if int(v["episode_index"]) == int(episode.episode))
    raw_path = source / f"client/episode_{int(episode.episode):02d}.npz"
    inputs[str(summary_path.relative_to(ROOT))] = digest(summary_path)
    inputs[str(raw_path.relative_to(ROOT))] = digest(raw_path)
    assert summary["inference_calls"] == length
    assert bool(summary["success"]) == (not bool(episode.failure))
    assert int(summary["init_state_id"]) == episode.init_state_id
    assert int(summary["flow_noise_seed"]) == episode.noise_seed
    steps = int(summary["action_steps"])
    assert 10*(length-1) < steps <= 10*length
    with np.load(raw_path) as raw:
        assert len(raw["state"]) == length
        simulation_seconds = raw["sim_state"][:, 0].astype(float)
    simulation_seconds -= simulation_seconds[0]
    audit = json.loads((geometry / "render_verification.json").read_text())
    rendered = next(item for item in audit["cases"] if item["global_row"] == row)
    assert digest(raw_path) == rendered["source_sha256"]
    images = []
    for query in range(length):
        path = geometry / "frames" / str(row) / f"q{query:03d}.png"
        checksum = digest(path)
        assert checksum == rendered["image_sha256"][str(path.relative_to(geometry))]
        inputs[str(path.relative_to(ROOT))] = checksum
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB")))
        assert images[-1].shape == (320, 320, 3) and images[-1].std() > 5

    queries = np.arange(length)
    data = pd.DataFrame(dict(global_row=row, fold=fold, query=queries,
        executed_actions_before_query=10*queries,
        executed_actions_after_chunk=np.minimum(10*(queries+1), steps),
        simulation_seconds_before_query=simulation_seconds, knn_score=score,
        threshold=tau, score_over_threshold=score/tau, scorable=valid,
        current_exceeds=score > tau, alarm_latched=(first >= 0) & (queries >= first),
        first_alarm_query=first, pc1=xy[:, 0], pc2=xy[:, 1],
        nearest_distance=neighbor_distances[:, 0], twentieth_distance=neighbor_distances[:, -1]))
    data.to_csv(output / "trajectory_queries.csv", index=False)
    records = []
    for query in queries[valid]:
        for rank, (bank_id, distance) in enumerate(zip(neighbor_ids[query], neighbor_distances[query]), 1):
            reference_row = int(profile["success_global_rows"][bank_id])
            reference_query = int(profile["success_queries"][bank_id])
            reference = frame.iloc[reference_row]
            assert not reference.failure and reference.run_id == "right-50x8-20260903"
            records.append(dict(query=int(query), rank=rank, bank_index=int(bank_id),
                reference_global_row=reference_row, reference_query=reference_query,
                reference_task=reference.task, reference_init_state_id=int(reference.init_state_id),
                reference_noise_seed=int(reference.noise_seed), distance_10d=float(distance),
                projected_distance_2d=float(np.linalg.norm(bank_xy[bank_id]-xy[query])),
                neighbor_pc1=bank_xy[bank_id, 0], neighbor_pc2=bank_xy[bank_id, 1]))
    pd.DataFrame(records).to_csv(output / "trajectory_neighbors.csv", index=False)
    np.savez_compressed(output / "trajectory_geometry.npz", query=queries, normalized_10d=normalized,
        score=score, threshold=tau, xy=xy, bank_xy=bank_xy, bank_10d=profile["success_dynamic"],
        neighbor_ids=neighbor_ids, neighbor_distances_10d=neighbor_distances,
        bank_global_rows=profile["success_global_rows"], bank_queries=profile["success_queries"],
        pca_mean=profile["pca_mean"], pca_components=profile["pca_components"])
    combined = np.concatenate((bank_xy, xy[valid]))
    lower, upper = combined.min(0), combined.max(0)
    pad = (upper-lower)*.06
    case.update(fold=fold, first_alarm_query=first, threshold=tau, actual_action_steps=steps,
        unseen=bool(prediction["test_unseen"][position]), primary_failure_reason=str(episode.primary_failure_reason),
        pca_explained_variance=float(profile["pca_explained_variance_ratio"].sum()),
        task_title="白杯放到盘子上，巧克力布丁放到盘子右侧" if row == 15023 else episode.task.split("/")[-1],
        new_rollouts=False, changed_noise=False, frame_source="Verified redraws of recorded pre-action simulator states")
    write_json(output / "case.json", case)
    return dict(case=case, data=data, bank_xy=bank_xy, xy=xy, normalized=normalized,
        neighbors=neighbor_ids, images=images, limits=(lower-pad, upper+pad), inputs=inputs,
        score_reproduction_max_abs_error=float(np.abs(distances.mean(1)-score[valid]).max()))


def status(bundle, query):
    row = bundle["data"].iloc[query]
    if not row.scorable:
        return GRAY, "基线积累，尚无 kNN 分数"
    label = "当前越界" if row.current_exceeds else "当前阈下"
    if query == bundle["case"]["first_alarm_query"]:
        label = "首次报警"
    elif row.alarm_latched and not row.current_exceeds:
        label = "阈下，已有报警"
    return BAD if row.current_exceeds else GOOD, f"d={row.knn_score:.3f}   d/阈值={row.score_over_threshold:.2f}   {label}"


def geometry_axis(ax, bundle):
    xy = bundle["bank_xy"]
    ax.scatter(xy[:, 0], xy[:, 1], s=2, color=GRAY, alpha=.24, linewidths=0, rasterized=True)
    low, high = bundle["limits"]
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#d7dbde")
        spine.set_linewidth(.55)


def geometry_at(ax, bundle, query):
    geometry_axis(ax, bundle)
    xy, bank = bundle["xy"], bundle["bank_xy"]
    if query < 7:
        ax.text(.5, .52, "基线尚未就绪", ha="center", va="center", transform=ax.transAxes,
                color="#656a70", fontsize=9, bbox=dict(fc="white", ec="none", alpha=.9, pad=3))
        return
    selected = bank[bundle["neighbors"][query]]
    segments = np.stack((np.repeat(xy[query:query+1], 20, axis=0), selected), axis=1)
    ax.add_collection(LineCollection(segments, colors=NEIGHBOR, linewidths=.4, alpha=.65, zorder=2))
    ax.scatter(selected[:, 0], selected[:, 1], s=11, color=NEIGHBOR, alpha=.9, linewidths=0, zorder=3)
    ax.plot(xy[7:query+1, 0], xy[7:query+1, 1], color=HISTORY, lw=.8, marker=".", markersize=2, alpha=.9, zorder=4)
    color, _ = status(bundle, query)
    ax.scatter(*xy[query], s=48, color=color, marker="*", edgecolors="white", linewidths=.5, zorder=6)


def curve_axis(ax, bundle, annotations=True):
    case, data = bundle["case"], bundle["data"]
    ax.axvspan(-.3, 6.8, color="#eceff1", zorder=0)
    ax.axhline(1, color=BAD, ls="--", lw=1.2, label="报警阈值")
    ax.set(xlim=(-.3, len(data)), ylim=(0, max(1.4, data.score_over_threshold.max()*1.37)))
    ax.set_xticks(sorted(set([*range(0, len(data), 5), len(data)])))
    ax.set_xlabel("推理序号 q", fontsize=10)
    ax.set_ylabel("kNN 距离 / 报警阈值", fontsize=10)
    ax.grid(axis="y", alpha=.18)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(labelsize=9)
    if annotations:
        ax.text(3.1, .10, "基线期", ha="center", fontsize=10, color="#606970")
        top = ax.secondary_xaxis("top", functions=(lambda q: q*10, lambda action: action/10))
        top.set_xlabel("报警计算前已执行的环境动作数", fontsize=10, labelpad=6)
        top.set_xticks([*range(0, case["actual_action_steps"], 100), case["actual_action_steps"]])
        top.tick_params(labelsize=9)
        release = case["release_query"]
        if release is not None:
            ax.axvline(release, color="#78889a", ls=":", lw=1)
            ax.text(release+.4, 1.42, f"q{int(release)} 物体释放标记\n事后标注，非确证失败起点", fontsize=9, color="#627286")
        first = case["first_alarm_query"]
        if first >= 0:
            ax.axvline(first, color=BAD, ls=":", lw=1.1)
            ax.annotate(f"q{first} 首次报警", xy=(first, data.iloc[first].score_over_threshold),
                xytext=(first-8, 1.48), color=BAD, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1))
        if case["global_row"] == 15023:
            ax.axvspan(35.5, 40.5, color=GOOD, alpha=.065)
            ax.text(38, .16, "q36–40 回到阈下\n未实施干预", ha="center", fontsize=9, color=GOOD)
            ax.text(46, 1.48, "q41–51 再次持续越界", ha="center", fontsize=10, color=BAD)
        ax.axvline(len(data), color=INK, lw=1)


def legend_handles():
    return [Line2D([], [], marker=".", color=GRAY, ls="none", markersize=6, label="A 成功参考"),
            Line2D([], [], marker="o", color=NEIGHBOR, ls="-", linewidth=.7, markersize=4, label="真实 10D 的 20 个近邻"),
            Line2D([], [], color=HISTORY, lw=1.2, marker=".", markersize=4, label="截至当前的投影轨迹"),
            Line2D([], [], marker="*", color=GOOD, ls="none", markersize=9, label="当前阈下"),
            Line2D([], [], marker="*", color=BAD, ls="none", markersize=9, label="当前越界")]


def poster(bundle, output):
    case, data = bundle["case"], bundle["data"]
    columns = 6
    rows = int(np.ceil(len(data)/columns))
    fig = plt.figure(figsize=(26, 5.7+rows*2.5), facecolor="white")
    grid = fig.add_gridspec(rows+2, columns, height_ratios=[1.65, .28]+[1.15]*rows,
                           left=.025, right=.99, bottom=.024, top=.931, wspace=.16, hspace=.37)
    fig.suptitle(f"逐步 kNN：{case['task_title']}", fontsize=23, x=.025, ha="left", y=.990)
    fig.text(.025, .974, f"失败轨迹  |  LIBERO-Long  |  episode {case['episode']}  |  初始状态 {case['init_state_id']}  |  噪声种子 {case['noise_seed']}  |  {len(data)} 次推理 / {case['actual_action_steps']} 个实际动作", fontsize=13, color=INK)
    fig.text(.025, .959, f"原始 10D / 20NN  |  未见任务折 {case['fold']}  |  阈值 {case['threshold']:.4f}  |  无噪声干预，最终未完成任务", fontsize=11, color="#5c646a")
    curve = fig.add_subplot(grid[0, :])
    curve_axis(curve, bundle)
    q = data["query"].to_numpy()
    ratio = data.score_over_threshold.to_numpy()
    curve.plot(q, ratio, color=HISTORY, lw=1.8, zorder=3)
    curve.scatter(q, ratio, c=np.where(data.current_exceeds, BAD, GOOD), s=26, zorder=4)
    curve.fill_between(q, 1, ratio, where=ratio > 1, color=BAD, alpha=.10)
    legend = fig.add_subplot(grid[1, :])
    legend.axis("off")
    legend.legend(handles=legend_handles(), ncol=5, loc="upper left", fontsize=11, frameon=False)
    legend.text(1, .6, f"同一固定 PCA 坐标；保留 {case['pca_explained_variance']:.1%} 的参考方差", ha="right", fontsize=10, color="#606970", transform=legend.transAxes)
    for query in range(len(data)):
        outer = grid[2+query//columns, query % columns]
        inner = outer.subgridspec(3, 2, height_ratios=[.20, 1, .26], width_ratios=[1, 1.5], hspace=.03, wspace=.06)
        title = fig.add_subplot(inner[0, :])
        title.axis("off")
        title.text(0, .4, f"q{query:02d}  |  已执行 {query*10} 个动作", fontsize=11, color=INK, va="center")
        image_ax = fig.add_subplot(inner[1, 0])
        image_ax.imshow(bundle["images"][query])
        image_ax.axis("off")
        geometry_at(fig.add_subplot(inner[1, 1]), bundle, query)
        label = fig.add_subplot(inner[2, :])
        label.axis("off")
        color, value = status(bundle, query)
        label.text(0, .7, value, color=color, fontsize=9.3, va="center")
    if len(data) % columns:
        note = fig.add_subplot(grid[-1, len(data) % columns:])
        note.axis("off")
        note.text(.02, .92, "回到阈下 ≠ 已救回", fontsize=18, color=INK, va="top")
        note.text(.02, .68, "q33 已触发一次报警；q36–40 只是当前距离回落。\n"
                  "q41 起再次越界，直到最后一次推理。\n\n"
                  "以后改变噪声的分支应从同一记录状态出发，\n"
                  "用相同执行预算比较最终结果与 MoE 演化。\n"
                  "本图没有新增 rollout，也没有救回实验结果。",
                  fontsize=12, color="#53616a", linespacing=1.65, va="top")
    fig.text(.025, .012, "画面是已校验的记录状态重绘，每格均为执行当前 chunk 之前；q51 不是终止后的画面。近邻和分数在 10D 计算，二维连线长度不能替代原距离。", fontsize=11, color="#606970")
    for extension in ("png", "pdf"):
        path = output / f"trajectory_knn_all_queries.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(f"SAVED {path.name}", flush=True)
    plt.close(fig)


def animation(bundle, output):
    case, data = bundle["case"], bundle["data"]
    fig = plt.figure(figsize=(13.8, 8), dpi=105, facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1, 1.9],
                           left=.065, right=.98, bottom=.10, top=.86, hspace=.35, wspace=.1)
    state = fig.add_subplot(grid[0, 0])
    image = state.imshow(bundle["images"][0])
    state.axis("off")
    state.set_title("记录状态重绘", fontsize=11)
    geometry = fig.add_subplot(grid[0, 1])
    geometry_axis(geometry, bundle)
    geometry.set_title("固定 PCA 投影：实际 20 个近邻在原 10D 选择", fontsize=11)
    lines = LineCollection([], colors=NEIGHBOR, linewidths=.7, alpha=.7)
    geometry.add_collection(lines)
    neighbors = geometry.scatter([], [], s=24, color=NEIGHBOR, zorder=3)
    path, = geometry.plot([], [], color=HISTORY, lw=1.4, marker=".", markersize=4, zorder=4)
    current = geometry.scatter([], [], s=150, color=GOOD, marker="*", edgecolors="white", linewidths=.8, zorder=5)
    warmup = geometry.text(.5, .52, "基线积累，尚无 kNN 分数", ha="center", transform=geometry.transAxes,
                           fontsize=12, color="#626b72", bbox=dict(fc="white", ec="none", alpha=.9, pad=4))
    curve = fig.add_subplot(grid[1, :])
    curve_axis(curve, bundle, annotations=False)
    score_line, = curve.plot([], [], color=HISTORY, lw=1.7)
    score_points = curve.scatter([], [], s=22, zorder=4)
    cursor = curve.axvline(0, color=INK, lw=1, alpha=.7)
    first_line = curve.axvline(case["first_alarm_query"], color=BAD, lw=1, ls=":")
    first_line.set_visible(False)
    title = fig.suptitle("", x=.065, ha="left", fontsize=17, y=.974)
    metric = fig.text(.065, .923, "", fontsize=13)
    fig.text(.065, .038, "每帧对应一次推理，演示节奏非真实耗时。无噪声干预；原轨迹最终失败。", fontsize=10, color="#606970")
    frames = []
    for query in range(len(data)):
        color, label = status(bundle, query)
        title.set_text(f"{case['task_title']}  |  q{query:02d} / {len(data)-1}")
        metric.set_text(f"已执行 {query*10} 个动作   |   {label}")
        metric.set_color(color)
        image.set_data(bundle["images"][query])
        cursor.set_xdata([query, query])
        first_line.set_visible(query >= case["first_alarm_query"] >= 0)
        part = data.iloc[:query+1]
        finite = part.scorable.to_numpy()
        visible = np.column_stack((part["query"].to_numpy()[finite], part.score_over_threshold.to_numpy()[finite]))
        score_line.set_data(visible[:, 0], visible[:, 1])
        score_points.set_offsets(visible)
        score_points.set_color(np.where(part.current_exceeds.to_numpy()[finite], BAD, GOOD).tolist())
        if query >= 7:
            warmup.set_visible(False)
            xy = bundle["xy"]
            selected = bundle["bank_xy"][bundle["neighbors"][query]]
            lines.set_segments(np.stack((np.repeat(xy[query:query+1], 20, axis=0), selected), axis=1))
            neighbors.set_offsets(selected)
            path.set_data(xy[7:query+1, 0], xy[7:query+1, 1])
            current.set_offsets(xy[query:query+1])
            current.set_facecolor(color)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB"))
    plt.close(fig)
    frames[0].save(output / "trajectory_knn_replay.gif", save_all=True, append_images=frames[1:],
                   duration=450, loop=0, optimize=False, disposal=2)
    print("SAVED trajectory_knn_replay.gif", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-row", type=int, default=15023)
    parser.add_argument("--fold", default="libero_long_20260907")
    parser.add_argument("--output", type=Path, default=HERE.parent / "results/round5_knn/trajectory_dynamics/long_episode_223")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    font_checksum = configure_font()
    bundle = prepare(output, args.global_row, args.fold)
    print(f"VERIFIED {len(bundle['data'])} queries and {int(bundle['data'].scorable.sum())*20} neighbor identities", flush=True)
    poster(bundle, output)
    animation(bundle, output)
    write_json(output / "verification.json", dict(source_sha256=digest(Path(__file__)), inputs=bundle["inputs"],
        global_row=args.global_row, fold=args.fold, queries=len(bundle["data"]),
        independent_full_bank_distance_checks=int(bundle["data"].scorable.sum()),
        score_reproduction_max_abs_error=bundle["score_reproduction_max_abs_error"],
        first_alarm_reproduced=True, recorded_frame_hashes_verified=len(bundle["images"]),
        projection="Frozen A mixture PCA; neighbors and scores use original 10D distances",
        candidate_selection="Reuse the previous Round 4 fixed Long failure case; illustrative, not population evidence",
        font_source_sha256=font_checksum, exported_font_weight=400,
        new_rollouts=False, noise_intervention=False,
        artifacts={path.name: digest(path) for path in sorted(output.iterdir())
                   if path.is_file() and path.name != "verification.json"}))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
