"""Retrospective, outcome-stratified raw MoE inspection for spatial task S05."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.stats import rankdata

from monitor import ROOT
from run_analysis import BASE, digest, write_json


TASK = "libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
RUNS = {"A": "right-50x8-20260903", "B": "right-50x8b-20260903"}
LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
AS_LAYERS = (0, 1, 16, 17)
QMAX = 22
METRICS = (
    "entropy", "top1_probability", "top4_mass", "token_js", "token_top1_agreement",
    "mobility", "top1_switch", "top4_turnover", "lag1_similarity", "recurrence_excess",
    "flow_path", "flow_acceleration", "flow_efficiency", "flow_top4_turnover",
    "flow_entropy_change", "flow_curvature", "relative_mobility", "relative_acceleration",
)
REGIONS = {"front": slice(0, 4), "back": slice(4, 8)}
SNAPSHOTS = (0, 4, 6, 8, 9, 10, 12)
WINDOWS = (("q1_4", 1, 4), ("q5_8", 5, 8), ("q9_12", 9, 12))


def auc_high_failure(success, failure):
    if not len(success) or not len(failure):
        return np.nan
    ranks = rankdata(np.concatenate((success, failure)))
    return float((ranks[len(success):].sum() - len(failure) * (len(failure) + 1) / 2)
                 / (len(success) * len(failure)))


def describe(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=np.nan, median=np.nan, p25=np.nan, p75=np.nan)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p25=float(np.quantile(values, .25)), p75=float(np.quantile(values, .75)))


def comparison(values, frame, seed=20260908):
    finite = np.isfinite(values)
    failure = frame.failure.to_numpy(bool)
    success = values[finite & ~failure]
    failed = values[finite & failure]
    result = {f"{label}_{key}": value for label, part in (("success", success), ("failure", failed))
              for key, value in describe(part).items()}
    result["auc_high_failure"] = auc_high_failure(success, failed)
    paired_success, paired_failure, differences, state_auc = [], [], [], []
    for _, group in frame.loc[finite].groupby("init_state_id"):
        s = values[group.index[~group.failure]]
        f = values[group.index[group.failure]]
        if len(s) and len(f):
            paired_success.append(float(s.mean()))
            paired_failure.append(float(f.mean()))
            differences.append(float(f.mean() - s.mean()))
            state_auc.append(auc_high_failure(s, f))
    result.update(paired_states=len(differences), paired_success_mean=np.nan,
                  paired_failure_mean=np.nan, paired_delta=np.nan,
                  paired_delta_ci_low=np.nan, paired_delta_ci_high=np.nan,
                  paired_auc=np.nan, positive_delta_states=0)
    if differences:
        d = np.asarray(differences)
        boots = d[np.random.default_rng(seed).integers(len(d), size=(2000, len(d)))].mean(1)
        result.update(paired_success_mean=float(np.mean(paired_success)),
                      paired_failure_mean=float(np.mean(paired_failure)), paired_delta=float(d.mean()),
                      paired_delta_ci_low=float(np.quantile(boots, .025)),
                      paired_delta_ci_high=float(np.quantile(boots, .975)),
                      paired_auc=float(np.mean(state_auc)), positive_delta_states=int((d > 0).sum()))
    return result


def scalar_features(hb):
    names = {name: i for i, name in enumerate(METRICS)}
    scalar = {f"{region}_{name}": hb["features"][:, :, layers, j].mean(2)
              for region, layers in REGIONS.items() for j, name in enumerate(METRICS)}
    scalar["front_back_path_ratio"] = scalar["front_flow_path"] / scalar["back_flow_path"]
    scalar["v7_relative_acceleration"] = hb["v7_relative_acceleration"]
    scalar["v7_relative_freeze"] = hb["v7_relative_freeze"]
    scalar["v7_recurrence_excess"] = hb["v7_recurrence_excess"]
    assert len(names) == len(METRICS)
    return scalar


def extract(frame, cohort, output, extraction_audits, cached_v7, cached_v8):
    frame = frame.reset_index(drop=True)
    source = ROOT / "VLA_MUI_HUB" / frame.source.iloc[0]
    store = zarr.open_group(str(source / "server/routes.zarr"), mode="r")
    raw = store["hb_router_probs"]
    summaries = sorted(json.loads((source / "client/summaries.json").read_text()), key=lambda x: x["episode_index"])
    lengths = frame.length.to_numpy(int)
    np.testing.assert_array_equal([s["inference_calls"] for s in summaries], lengths)
    np.testing.assert_array_equal([s["episode_index"] for s in summaries], frame.episode)
    np.testing.assert_array_equal(store["episode_id"][:], np.repeat(frame.episode, lengths))
    assert np.all(np.diff(store["control_step"][:]) == 1)
    assert raw.shape == (int(lengths.sum()), 8, 10, 11, 32)
    assert int(store.attrs["top_k"]) == 4
    n = len(frame)
    features = np.full((n, QMAX, 8, len(METRICS)), np.nan, np.float32)
    load = np.full((n, QMAX, 8, 32), np.nan, np.float32)
    hard_load = np.full_like(load, np.nan)
    speed = np.full((n, QMAX, 8, 9), np.nan, np.float32)
    flow_entropy = np.full((n, QMAX, 8, 10), np.nan, np.float32)
    as_probs = np.full((n, QMAX, 4, 3), np.nan, np.float32)
    as_ids = np.full((n, QMAX, 4), -1, np.int16)
    v7_period = np.full((n, QMAX), np.nan, np.float32)
    valid = np.arange(QMAX)[None] < lengths[:, None]
    raw_digest, final_digest, selected_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    checks, max_topk_rounding, offset = 0, 0., 0
    for i, row in enumerate(frame.itertuples()):
        length = int(row.length)
        original = np.asarray(raw[offset:offset + length])
        raw_digest.update(original.tobytes())
        final_digest.update(original[:, :, 9].tobytes())
        p = original.astype(np.float32)
        assert np.isfinite(p).all() and (p >= 0).all()
        p /= p.sum(-1, keepdims=True)
        action = p[:, :, :, 1:]
        root = np.sqrt(action)
        final = action[:, :, 9]
        final_root = root[:, :, 9]
        selected = np.asarray(store["hb_expert_ids"][offset:offset + length, :, :, 1:])
        selected_digest.update(selected.tobytes())
        assert selected.shape == (length, 8, 10, 10, 4)
        assert np.all(np.diff(np.sort(selected, axis=-1), axis=-1) > 0)
        selected_p = np.take_along_axis(action, selected.astype(int), axis=-1)
        top = np.partition(action, -4, axis=-1)[..., -4:]
        discrepancy = np.abs(np.sort(top, axis=-1) - np.sort(selected_p, axis=-1)).max()
        max_topk_rounding = max(max_topk_rounding, float(discrepancy))
        assert discrepancy < 2e-4
        final_ids = selected[:, :, 9]
        top1 = np.take_along_axis(final_ids, selected_p[:, :, 9].argmax(-1)[..., None], axis=-1)[..., 0]
        ent = -(action * np.log(np.maximum(action, 1e-30))).sum(-1).mean(-1)
        flow_entropy[i, :length] = ent / np.log(32)
        mean_load = final.mean(2)
        load[i, :length] = mean_load
        load_entropy = -(mean_load * np.log(np.maximum(mean_load, 1e-30))).sum(-1)
        for expert in range(32):
            hard_load[i, :length, :, expert] = (final_ids == expert).sum(axis=(-2, -1)) / 40.
        per = dict(entropy=ent[:, :, 9] / np.log(32), top1_probability=final.max(-1).mean(-1),
                   top4_mass=selected_p[:, :, 9].sum(-1).mean(-1),
                   token_js=np.maximum(load_entropy - ent[:, :, 9], 0) / np.log(32),
                   token_top1_agreement=np.stack([(top1 == e).mean(-1) for e in range(32)]).max(0))
        per["mobility"] = np.full((length, 8), np.nan)
        per["mobility"][1:] = (np.linalg.norm(np.diff(final_root, axis=0), axis=-1) / np.sqrt(2)).mean(-1)
        per["top1_switch"] = np.full((length, 8), np.nan)
        per["top1_switch"][1:] = (top1[1:] != top1[:-1]).mean(-1)
        per["top4_turnover"] = np.full((length, 8), np.nan)
        per["top4_turnover"][1:] = 1 - (final_ids[1:, ..., None] == final_ids[:-1, ..., None, :]).any(-1).mean(axis=(-2, -1))
        lag = np.full((length, 8, 4), np.nan)
        for k in range(1, 5):
            sim = np.minimum(final[k:], final[:-k]).sum(-1) / np.maximum(final[k:], final[:-k]).sum(-1)
            lag[k:, :, k - 1] = sim.mean(-1)
        per["lag1_similarity"] = lag[:, :, 0]
        per["recurrence_excess"] = np.full((length, 8), np.nan)
        per["recurrence_excess"][2:] = np.nanmax(lag[2:, :, 1:], axis=-1) - lag[2:, :, 0]
        back_lag = np.asarray(lag[:, 4:], np.float32).mean(1)
        v7_period[i, 2:length] = np.nanmax(back_lag[2:, 1:], axis=-1) - back_lag[2:, 0]
        per_speed = (np.linalg.norm(np.diff(root, axis=2), axis=-1) / np.sqrt(2)).mean(-1)
        speed[i, :length] = per_speed
        per["flow_path"] = per_speed.sum(-1)
        second = root[:, :, 2:] - 2 * root[:, :, 1:-1] + root[:, :, :-2]
        per["flow_acceleration"] = (np.linalg.norm(second, axis=-1) / np.sqrt(2)).mean(axis=(-2, -1))
        endpoint = (np.linalg.norm(root[:, :, -1] - root[:, :, 0], axis=-1) / np.sqrt(2)).mean(-1)
        per["flow_efficiency"] = endpoint / np.maximum(per["flow_path"], 1e-12)
        per["flow_top4_turnover"] = 1 - (selected[:, :, 1:, ..., None] == selected[:, :, :-1, ..., None, :]).any(-1).mean(axis=(-3, -2, -1))
        per["flow_entropy_change"] = (ent[:, :, 9] - ent[:, :, 0]) / np.log(32)
        per["flow_curvature"] = np.abs(per_speed[:, :, 0] - 2 * per_speed[:, :, 4] + per_speed[:, :, 8])
        per["relative_mobility"] = np.full((length, 8), np.nan)
        per["relative_mobility"][5:] = np.log(np.maximum(per["mobility"][5:], 1e-12) / np.maximum(per["mobility"][1:5].mean(0), 1e-12))
        per["relative_acceleration"] = np.full((length, 8), np.nan)
        if length >= 8:
            baseline = np.median(per["flow_acceleration"][2:8], axis=0)
            per["relative_acceleration"][7:] = np.log(np.maximum(per["flow_acceleration"][7:], 1e-12) / np.maximum(baseline, 1e-12))
        for j, name in enumerate(METRICS):
            features[i, :length, :, j] = per[name]
        asp = np.asarray(store["as_probs"][offset:offset + length], np.float32)
        asp /= asp.sum(-1, keepdims=True)
        as_probs[i, :length] = asp
        as_ids[i, :length] = store["as_expert_ids"][offset:offset + length]
        np.testing.assert_allclose(asp.max(-1), np.take_along_axis(asp, as_ids[i, :length, :, None], axis=-1)[..., 0], atol=2e-4, rtol=0)
        checks += length * 8 * 10 * 10 + length * 4
        offset += length
        if (i + 1) % 100 == 0:
            print(f"RAW {cohort}: {i + 1}/400 episodes", flush=True)
    ids = {name: i for i, name in enumerate(METRICS)}
    mobility = features[..., ids["mobility"]]
    acceleration = features[:, :, 4:, ids["flow_acceleration"]].mean(-1)
    relative_acc = np.log(np.maximum(acceleration, 1e-12) / np.maximum(np.median(acceleration[:, 2:8], axis=1)[:, None], 1e-12))
    relative_acc[:, :7] = np.nan
    freeze = -np.median(features[:, :, 4:, ids["relative_mobility"]], axis=-1)
    path = speed.sum(-1)
    v8 = np.stack((-np.log(path[:, :, :4].mean(-1) / path[:, :, 4:].mean(-1)),
                   features[:, :, 4:, ids["flow_curvature"]].mean(-1)), axis=-1)
    rows = frame.global_row.to_numpy(int)
    np.testing.assert_allclose(mobility, cached_v7["mobility"][rows, :QMAX], rtol=2e-4, atol=3e-6, equal_nan=True)
    np.testing.assert_allclose(acceleration, cached_v7["acceleration"][rows, :QMAX], rtol=2e-4, atol=3e-6, equal_nan=True)
    np.testing.assert_allclose(v7_period, cached_v7["periodicity"][rows, :QMAX], rtol=2e-4, atol=3e-6, equal_nan=True)
    np.testing.assert_allclose(v8, cached_v8[rows, :QMAX], rtol=2e-4, atol=3e-6, equal_nan=True)
    audit = next(a for a in extraction_audits if a["source"] == frame.source.iloc[0])
    assert final_digest.hexdigest() == audit["route_final_sha256"]
    old_path = BASE / "round3_safe/features" / audit["output"]
    with np.load(old_path, allow_pickle=False) as z:
        np.testing.assert_allclose(load, z["load"][:, :QMAX].reshape(n, QMAX, 8, 32) ** 2, atol=2e-7, rtol=2e-5, equal_nan=True)
        stats = z["stats"][:, :QMAX].reshape(n, QMAX, 8, 4)
        for key, j in (("entropy", 0), ("token_js", 2)):
            np.testing.assert_allclose(features[..., ids[key]], stats[..., j], atol=3e-7, rtol=2e-4, equal_nan=True)
    data = dict(features=features, load=load, hard_load=hard_load, speed=speed,
                flow_entropy=flow_entropy, as_probs=as_probs, as_ids=as_ids, valid=valid,
                v7_relative_acceleration=relative_acc, v7_relative_freeze=freeze,
                v7_recurrence_excess=v7_period, metrics=np.asarray(METRICS), layers=np.asarray(LAYERS),
                global_rows=rows, episode=frame.episode.to_numpy(), failure=frame.failure.to_numpy())
    np.savez_compressed(output / f"raw_features_{cohort}.npz", **data)
    result = dict(cohort=cohort, episodes=n, failures=int(frame.failure.sum()), queries=int(lengths.sum()),
                  source=str(source), raw_shape=raw.shape, raw_sha256=raw_digest.hexdigest(),
                  final_sha256=final_digest.hexdigest(), selected_sha256=selected_digest.hexdigest(),
                  topk_and_as_checks=checks, topk_max_probability_rounding_error=max_topk_rounding,
                  cached_v7_v8_and_static_features_match=True,
                  inputs={str(p): digest(p) for p in (source / "client/summaries.json", source / "client/server_metadata.json", old_path)})
    return data, result


def compare_all(frame, data, cohort):
    scalar = scalar_features(data)
    records = []
    for metric, values in scalar.items():
        for q in range(QMAX):
            records.append(dict(cohort=cohort, kind="query", query=q, metric=metric, region="aggregate",
                                **comparison(values[:, q], frame)))
        for name, lo, hi in WINDOWS:
            values_window = values[:, lo:hi + 1].mean(1)
            records.append(dict(cohort=cohort, kind=name, query=hi, metric=metric, region="aggregate",
                                **comparison(values_window, frame)))
    for j, metric in enumerate(METRICS):
        for li, layer in enumerate(LAYERS):
            for q in SNAPSHOTS:
                records.append(dict(cohort=cohort, kind="query", query=q, metric=metric, region=f"L{layer}",
                                    **comparison(data["features"][:, q, li, j], frame)))
    for li, layer in enumerate(AS_LAYERS):
        asp = data["as_probs"][:, :, li]
        entropy = -(asp * np.log(np.maximum(asp, 1e-30))).sum(-1) / np.log(3)
        for name, values in (("as_entropy", entropy), ("as_top1", asp.max(-1))):
            for q in SNAPSHOTS:
                records.append(dict(cohort=cohort, kind="query", query=q, metric=name, region=f"L{layer}",
                                    **comparison(values[:, q], frame)))
    return records, scalar


def expert_comparisons(frame, data, cohort):
    records, distances, as_counts = [], [], []
    for q in SNAPSHOTS:
        active = data["valid"][:, q]
        for representation in ("load", "hard_load"):
            for li, layer in enumerate(LAYERS):
                distributions = data[representation][:, q, li]
                paired_success, paired_failure = [], []
                for _, group in frame.loc[active].groupby("init_state_id"):
                    s = distributions[group.index[~group.failure]]
                    f = distributions[group.index[group.failure]]
                    if len(s) and len(f):
                        paired_success.append(s.mean(0))
                        paired_failure.append(f.mean(0))
                if not paired_success:
                    continue
                s, f = np.mean(paired_success, axis=0), np.mean(paired_failure, axis=0)
                for e in range(32):
                    records.append(dict(cohort=cohort, query=q, representation=representation, layer=layer,
                                        expert=e, paired_states=len(paired_success), success_mean=float(s[e]),
                                        failure_mean=float(f[e]), delta=float(f[e] - s[e])))
                distances.append(dict(cohort=cohort, query=q, representation=representation, layer=layer,
                                      paired_states=len(paired_success), total_variation=float(np.abs(f - s).sum() / 2),
                                      success_top4=" ".join(map(str, np.argsort(s)[-4:][::-1])),
                                      failure_top4=" ".join(map(str, np.argsort(f)[-4:][::-1]))))
        for li, layer in enumerate(AS_LAYERS):
            for failure in (False, True):
                mask = active & frame.failure.eq(failure).to_numpy()
                for e in range(3):
                    as_counts.append(dict(cohort=cohort, query=q, layer=layer, failure=failure,
                                          expert=e, n=int(mask.sum()), selected=int((data["as_ids"][mask, q, li] == e).sum()),
                                          probability_mean=float(data["as_probs"][mask, q, li, e].mean())))
    return records, distances, as_counts


def event_comparisons(frame, scalar, output):
    selected = frame.loc[frame.release_query >= 0].reset_index().rename(columns={"index": "position"})
    records, aligned = [], []
    for offset in range(-2, 6):
        q = selected.release_query.to_numpy(int) + offset
        active = (q >= 0) & (q < selected.length.to_numpy(int))
        for metric, values in scalar.items():
            observations = np.full(len(selected), np.nan)
            observations[active] = values[selected.position.to_numpy(int)[active], q[active]]
            result = comparison(observations, selected)
            # Release-query strata remove changes caused only by different event positions.
            by_release = []
            for _, part in selected.groupby("release_query"):
                s = observations[part.index[~part.failure]]
                f = observations[part.index[part.failure]]
                s, f = s[np.isfinite(s)], f[np.isfinite(f)]
                if len(s) and len(f):
                    by_release.append((f.mean() - s.mean(), auc_high_failure(s, f)))
            result["release_query_strata"] = len(by_release)
            result["release_query_matched_delta"] = float(np.mean([x[0] for x in by_release])) if by_release else np.nan
            result["release_query_matched_auc"] = float(np.mean([x[1] for x in by_release])) if by_release else np.nan
            records.append(dict(offset=offset, metric=metric, **result))
            for i, row in selected.iterrows():
                if np.isfinite(observations[i]):
                    aligned.append(dict(episode=row.episode, init_state_id=row.init_state_id, failure=row.failure,
                                        release_query=row.release_query, offset=offset, query=int(q[i]), metric=metric,
                                        value=float(observations[i])))
    pd.DataFrame(records).to_csv(output / "release_comparison.csv", index=False)
    pd.DataFrame(aligned).to_csv(output / "release_observations.csv", index=False)
    return pd.DataFrame(records)


def plots(frame, data, scalar, event, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {False: "#167e76", True: "#bd4e49"}
    selected = (("back_entropy", "Back-layer entropy / log(32)"),
                ("back_top4_mass", "Back-layer top-4 probability mass"),
                ("back_token_js", "Back-layer token diversity (JS)"),
                ("back_mobility", "Between-query routing change"),
                ("back_flow_acceleration", "Within-query flow acceleration"),
                ("v7_relative_acceleration", "Flow acceleration: log / opening baseline"),
                ("v7_relative_freeze", "Relative freeze: -log mobility / baseline"),
                ("front_back_path_ratio", "Front / back flow path ratio"))
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (name, title) in zip(axes.flat, selected):
        for failure in (False, True):
            values = scalar[name][frame.failure.eq(failure).to_numpy()]
            rows = pd.DataFrame([dict(query=q, **describe(values[:, q])) for q in range(14)])
            ax.plot(rows["query"], rows["median"], color=colors[failure], label="Failure" if failure else "Success")
            ax.fill_between(rows["query"], rows.p25, rows.p75, color=colors[failure], alpha=.14)
        ax.axvline(8, color="#777777", linestyle=":", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 13)
        ax.set_xlabel("Query q")
        ax.grid(alpha=.15)
    for failure in (False, True):
        values = data["valid"][frame.failure.eq(failure).to_numpy(), :14].sum(0)
        axes.flat[-1].plot(range(14), values, color=colors[failure], marker="o", label="Failure" if failure else "Success")
    axes.flat[-1].set_title("Observed episodes at each query")
    axes.flat[-1].set_xlabel("Query q")
    axes.flat[-1].set_ylabel("Count (descriptive support only)")
    axes.flat[-1].legend()
    axes.flat[0].legend()
    fig.suptitle("S05, B: raw MoE medians and middle 50%; all 400 observed through q8", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save_figure(fig, output, "routing_overview")

    event_metrics = (selected[3], selected[4], selected[5], selected[6], selected[7])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (name, title) in zip(axes.flat, event_metrics):
        rows = event.loc[event.metric.eq(name)].sort_values("offset")
        for failure in (False, True):
            prefix = "failure" if failure else "success"
            ax.plot(rows.offset, rows[prefix + "_median"], color=colors[failure], marker="o", label=prefix)
            ax.fill_between(rows.offset, rows[prefix + "_p25"], rows[prefix + "_p75"], color=colors[failure], alpha=.14)
        ax.axvline(0, color="#555555", linestyle=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Queries from observed release outside goal")
        ax.grid(alpha=.15)
    rows = event.loc[event.metric.eq("back_mobility")].sort_values("offset")
    for failure in (False, True):
        prefix = "failure" if failure else "success"
        axes.flat[-1].plot(rows.offset, rows[prefix + "_n"], color=colors[failure], marker="o", label=prefix)
    axes.flat[-1].set_title("Observed episodes in event comparison")
    axes.flat[-1].set_xlabel("Queries from release")
    axes.flat[-1].legend()
    axes.flat[0].legend()
    fig.suptitle("S05, B: 22 eventual failures and 22 eventual successes with the same neutral event", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save_figure(fig, output, "release_aligned")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    matched = frame.init_state_id.isin(frame.loc[frame.failure, "init_state_id"])
    for li, (layer, ax) in enumerate(zip(LAYERS, axes.flat)):
        for failure in (False, True):
            group_means = []
            for _, part in frame.loc[matched].groupby("init_state_id"):
                positions = part.index[part.failure.eq(failure)]
                group_means.append(data["speed"][positions, 8, li].mean(0))
            ax.plot(range(1, 10), np.mean(group_means, axis=0), color=colors[failure], marker="o", markersize=3,
                    label="Failure" if failure else "Success")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Denoising transition")
        ax.grid(alpha=.15)
    axes[0, 0].legend()
    axes[0, 0].set_ylabel("Routing Hellinger change")
    axes[1, 0].set_ylabel("Routing Hellinger change")
    fig.suptitle("S05, B, q8: same-initial-state flow profiles (11 states, equal state weights)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save_figure(fig, output, "layer_flow_q8")


def save_figure(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "v82_s05_moe_scan_20260908")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "execution_contract.json", dict(
        task=TASK, purpose="descriptive raw MoE comparison; no detector training, selection, or threshold tuning",
        primary_cohort="B: all 400 episodes, 24 failures and 376 successes",
        secondary_cohort="A: historical direction check, not a blind replication",
        hb_layers=LAYERS, as_layers=AS_LAYERS, final_flow=9, action_tokens=list(range(1, 11)),
        raw_features=METRICS, primary_complete_snapshot=8, snapshots=SNAPSHOTS, complete_windows=WINDOWS,
        missing_queries="NaN, never zero-filled; each reported comparison includes both group sizes",
        matching="same query, then same initial state with equal state weights",
        uncertainty="2,000 paired initial-state bootstrap resamples; pointwise intervals, no multiplicity claim",
        auc="descriptive raw-value orientation: higher values indicate failure; never flip after observing labels",
        physical_event="first observed grasp loss outside goal; not an irreversible failure or trap label",
        runtime_changes=False, trajectory_duration_used_as_score=False, time_only_baseline=False,
        feature_future_inputs=False, author_script_sha256=digest(__file__),
    ))
    index_path = BASE / "v82_validation_20260908/index.csv"
    diagnosis_path = BASE / "v82_s05_diagnosis_20260908/episodes.csv"
    v7_path, v8_path = BASE / "round3_safe/v7/v7_inputs.npz", BASE / "round7_temporal_fusion/v8_inputs.npz"
    expected = json.loads((BASE / "v82_validation_20260908/input_verification.json").read_text())["inputs"]
    inputs = {str(p): digest(p) for p in (index_path, diagnosis_path, v7_path, v8_path)}
    for p in (v7_path, v8_path):
        assert inputs[str(p)] == expected[str(p.relative_to(ROOT))]
    old = json.loads((BASE / "v82_s05_diagnosis_20260908/verification.json").read_text())
    assert inputs[str(diagnosis_path)] == old["artifacts"]["episodes.csv"]
    frame = pd.read_csv(index_path)
    audits = json.loads((BASE / "round3_safe/extraction_audit.json").read_text())
    with np.load(v7_path, allow_pickle=False) as z:
        v7 = {key: z[key] for key in ("mobility", "acceleration", "periodicity")}
    with np.load(v8_path, allow_pickle=False) as z:
        v8 = z["raw"]
    comparisons, experts, distances, as_counts, extraction = [], [], [], [], []
    for cohort in ("B", "A"):
        task = frame.loc[frame.task.eq(TASK) & frame.run_id.eq(RUNS[cohort])].reset_index(drop=True)
        assert len(task) == 400
        data, audit = extract(task, cohort, output, audits, v7, v8)
        extraction.append(audit)
        compared, scalar = compare_all(task, data, cohort)
        comparisons.extend(compared)
        e, d, a = expert_comparisons(task, data, cohort)
        experts.extend(e)
        distances.extend(d)
        as_counts.extend(a)
        if cohort == "B":
            diagnosis = pd.read_csv(diagnosis_path)
            np.testing.assert_array_equal(task.global_row, diagnosis.global_row)
            task["release_query"] = diagnosis.release_query
            task["v82_first"] = diagnosis.v82_first
            task["alarm_group"] = np.select((task.failure & task.v82_first.ge(0), task.failure,
                                               ~task.failure & task.v82_first.ge(0)),
                                              ("failure_detected", "failure_missed", "success_alarm"), default="success_no_alarm")
            episodes = pd.concat((task, pd.DataFrame({f"{name}_q{q}": values[:, q]
                                  for name, values in scalar.items() for q in SNAPSHOTS})), axis=1)
            episodes.to_csv(output / "episodes_B.csv", index=False)
            event = event_comparisons(task, scalar, output)
            plots(task, data, scalar, event, output)
            streams = []
            for i, row in task.iterrows():
                for q in np.flatnonzero(data["valid"][i]):
                    streams.append(dict(episode=row.episode, global_row=row.global_row, failure=row.failure,
                                        init_state_id=row.init_state_id, query=q, alarm_group=row.alarm_group,
                                        **{name: float(values[i, q]) for name, values in scalar.items()}))
            pd.DataFrame(streams).to_csv(output / "streams_B.csv", index=False)
        print(f"COMPARED {cohort}", flush=True)
    pd.DataFrame(comparisons).to_csv(output / "comparisons.csv", index=False)
    pd.DataFrame(experts).to_csv(output / "expert_load_comparison.csv", index=False)
    pd.DataFrame(distances).to_csv(output / "expert_distribution_distances.csv", index=False)
    pd.DataFrame(as_counts).to_csv(output / "as_expert_counts.csv", index=False)
    write_json(output / "verification.json", dict(all_checks_passed=True, extraction=extraction,
               inputs=inputs, source_sha256=digest(__file__),
               artifacts={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    table = pd.DataFrame(comparisons)
    names = ("back_entropy", "back_mobility", "back_flow_acceleration", "v7_relative_acceleration",
             "v7_relative_freeze", "front_back_path_ratio", "back_top4_turnover", "back_token_js")
    print(table.loc[table.kind.eq("query") & table["query"].eq(8) & table.metric.isin(names),
                    ["cohort", "metric", "success_n", "failure_n", "success_median", "failure_median",
                     "paired_states", "paired_delta", "paired_delta_ci_low", "paired_delta_ci_high", "paired_auc"]].to_string(index=False))
    print(f"SAVED {output}")


if __name__ == "__main__":
    main()
