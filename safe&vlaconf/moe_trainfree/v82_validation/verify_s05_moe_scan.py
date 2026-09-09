"""Independent numeric checks of the S05 raw-routing inspection artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu
import zarr

from monitor import ROOT
from run_analysis import BASE, digest, write_json

sys.path.insert(0, str(ROOT / "moe-v7-0905/method"))
from intrinsic_guard_monitor import IntrinsicGuardMonitor


def main():
    output = BASE / "v82_s05_moe_scan_20260908"
    manifest = json.loads((output / "verification.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
    index = pd.read_csv(BASE / "v82_validation_20260908/index.csv").set_index("global_row")
    table = pd.read_csv(output / "comparisons.csv")
    cache, scalar, frames = {}, {}, {}
    replay_checks, snapshot_checks = 0, 0
    for cohort in ("A", "B"):
        with np.load(output / f"raw_features_{cohort}.npz", allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        cache[cohort] = data
        frame = index.loc[data["global_rows"]].reset_index()
        frames[cohort] = frame
        y, valid = frame.failure.to_numpy(bool), data["valid"]
        np.testing.assert_array_equal(y, data["failure"])
        np.testing.assert_array_equal(valid, np.arange(22)[None] < frame.length.to_numpy()[:, None])
        np.testing.assert_array_equal(data["as_ids"][valid], np.tile([2, 0, 0, 1], (valid.sum(), 1)))
        assert np.all(np.ptp(data["as_probs"][valid], axis=0) == 0)
        for key in ("load", "hard_load"):
            np.testing.assert_allclose(data[key][valid].sum(-1), 1, atol=2e-7, rtol=0)
        metrics = list(data["metrics"])
        source = ROOT / "VLA_MUI_HUB" / frame.source.iloc[0]
        store = zarr.open_group(str(source / "server/routes.zarr"), mode="r")
        offsets = np.r_[0, np.cumsum(frame.length.to_numpy())]
        selected = sorted(set([0, 399, *np.flatnonzero(y)[:3].tolist(), *([77, 96, 174, 190] if cohort == "B" else [])]))
        for i in selected:
            raw = np.asarray(store["hb_router_probs"][offsets[i]:offsets[i + 1]])
            previous, history = None, []
            for q, query in enumerate(raw):
                mobility, acceleration, periodicity, final = IntrinsicGuardMonitor._query_features(query, previous, history)
                history.append(final[4:].reshape(40, 32))
                previous = final
                actual = data["features"][i, q]
                np.testing.assert_allclose(actual[:, metrics.index("mobility")], mobility, atol=3e-7, rtol=2e-5, equal_nan=True)
                np.testing.assert_allclose(actual[4:, metrics.index("flow_acceleration")].mean(), acceleration, atol=3e-8, rtol=2e-5)
                np.testing.assert_allclose(data["v7_recurrence_excess"][i, q], periodicity, atol=3e-7, rtol=2e-5, equal_nan=True)
                replay_checks += 1
        values = {f"{region}_{name}": data["features"][:, :, layers, j].mean(2)
                  for region, layers in (("front", slice(0, 4)), ("back", slice(4, 8)))
                  for j, name in enumerate(metrics)}
        values.update(v7_relative_acceleration=data["v7_relative_acceleration"],
                      v7_relative_freeze=data["v7_relative_freeze"],
                      v7_recurrence_excess=data["v7_recurrence_excess"])
        values["front_back_path_ratio"] = values["front_flow_path"] / values["back_flow_path"]
        scalar[cohort] = values
        selected_rows = table.loc[table.cohort.eq(cohort) & table.region.eq("aggregate") & table.kind.eq("query")]
        for row in selected_rows.itertuples():
            v = values[row.metric][:, row.query]
            s, f = v[~y & np.isfinite(v)], v[y & np.isfinite(v)]
            assert len(s) == row.success_n and len(f) == row.failure_n
            if len(s) and len(f):
                np.testing.assert_allclose([np.median(s), np.median(f)], [row.success_median, row.failure_median], atol=1e-12, rtol=1e-12)
                u = mannwhitneyu(f.astype(np.float64), s.astype(np.float64), alternative="two-sided").statistic / (len(s) * len(f))
                np.testing.assert_allclose(u, row.auc_high_failure, atol=1e-12, rtol=0)
                differences = []
                for state in np.unique(frame.init_state_id):
                    mask = frame.init_state_id.eq(state).to_numpy() & np.isfinite(v)
                    ss, ff = v[mask & ~y], v[mask & y]
                    if len(ss) and len(ff):
                        differences.append(float(ff.mean() - ss.mean()))
                assert len(differences) == row.paired_states
                if differences:
                    np.testing.assert_allclose(np.mean(differences), row.paired_delta, atol=1e-12, rtol=1e-12)
            snapshot_checks += 1
    streams = pd.read_csv(output / "streams_B.csv")
    episodes = pd.read_csv(output / "episodes_B.csv")
    assert len(streams) == 4332 and len(episodes) == 400
    assert episodes.alarm_group.value_counts().to_dict() == {
        "success_no_alarm": 374, "failure_detected": 18, "failure_missed": 6, "success_alarm": 2}
    records, patterns = [], []
    for group, part in streams.groupby("alarm_group"):
        start = part.loc[part["query"].eq(8)].set_index("episode")
        end = part.loc[part["query"].eq(12)].set_index("episode")
        ids = start.index.intersection(end.index)
        for metric in ("back_mobility", "back_flow_acceleration"):
            ratio = end.loc[ids, metric] / start.loc[ids, metric]
            records.append(dict(group=group, metric=metric, start_query=8, end_query=12,
                                episodes=len(ids), median_ratio=float(ratio.median())))
        lower = end.loc[ids, "back_mobility"] < start.loc[ids, "back_mobility"]
        higher = end.loc[ids, "back_flow_acceleration"] > start.loc[ids, "back_flow_acceleration"]
        patterns.append(dict(group=group, start_query=8, end_query=12, episodes=len(ids),
                             mobility_decreased=int(lower.sum()), flow_acceleration_increased=int(higher.sum()),
                             both=int((lower & higher).sum())))
    pd.DataFrame(records).to_csv(output / "fixed_episode_changes.csv", index=False)
    pd.DataFrame(patterns).to_csv(output / "fixed_episode_pattern_counts.csv", index=False)
    events = pd.read_csv(output / "release_observations.csv")
    event_table = pd.read_csv(output / "release_comparison.csv")
    for row in event_table.itertuples():
        sample = events.loc[events.offset.eq(row.offset) & events.metric.eq(row.metric)]
        for failure in (False, True):
            x = sample.loc[sample.failure.eq(failure), "value"]
            label = "failure" if failure else "success"
            assert len(x) == getattr(row, label + "_n")
            if len(x):
                np.testing.assert_allclose(x.median(), getattr(row, label + "_median"), atol=1e-12, rtol=1e-12)
    first = events.loc[events.offset.eq(0) & events.metric.eq("back_mobility")]
    assert first.groupby("failure").size().to_dict() == {False: 22, True: 22}
    by_episode = episodes.set_index("episode")
    by_query = streams.set_index(["episode", "query"])
    for row in events.itertuples():
        record = by_episode.loc[row.episode]
        assert row.query == record.release_query + row.offset
        assert 0 <= row.query < record.length
        original = by_query.loc[(row.episode, row.query), row.metric]
        np.testing.assert_allclose(row.value, original, atol=1e-12, rtol=1e-12)
    images = []
    for path in output.glob("*.png"):
        a = np.asarray(Image.open(path).convert("RGB"))
        nonwhite = (a.min(-1) < 245).mean()
        assert a.shape[0] > 800 and a.shape[1] > 1500 and nonwhite > .04
        images.append(dict(name=path.name, shape=a.shape, nonwhite_share=float(nonwhite)))
    write_json(output / "independent_verification.json", dict(
        all_checks_passed=True, cohorts=2, episodes=800, raw_queries=8502,
        online_query_feature_replays=replay_checks, independently_checked_aggregate_rows=snapshot_checks,
        event_observations_checked=len(events), as_routes_exactly_constant=True,
        figures=images, verifier_sha256=digest(__file__),
        artifact_sha256={p.name: digest(p) for p in sorted(output.iterdir())
                         if p.is_file() and p.name != "independent_verification.json"}))
    print(f"VERIFIED: {replay_checks} original-monitor query replays, {snapshot_checks} aggregate comparisons, {len(events)} event observations")


if __name__ == "__main__":
    main()
