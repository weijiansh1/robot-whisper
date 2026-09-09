"""Independent raw-routing replay and metric checks for ST/AC confirmation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import zarr

from guard import ALPHAS, KINDS, PeakBank
from state_action import GATE_ALPHA, StateActionGuardMonitor, frozen_triggers
from run_state_action_experiment import BASE, FEATURE_DIR, HERE, LEGACY_DIR, METHODS, ROOT, S05
from run_analysis import digest, write_json


def selected_rows(b, profiles, scores, first, frozen):
    selected, boundary = set(), set()
    for _, part in b.groupby(["task", "failure"]):
        selected.add(int(part.index[0]))
    selected.update(b.index[b.task.eq(S05) & b.episode.isin([77, 96, 102, 115, 174, 190, 286, 363, 366, 367])])
    for ki in range(len(KINDS)):
        combined = first[ki, ALPHAS.index(.01), METHODS.index("v82_plus_confirmed")]
        selected.update(np.flatnonzero((combined >= 0) & (frozen < 0)))
    for profile in profiles:
        ix = np.flatnonzero(b.init_state_id.to_numpy() % 5 == profile["fold"])
        for kind in KINDS:
            for name, alpha in (("gap", GATE_ALPHA), ("acceleration", .005), ("decoupling", .005)):
                threshold, _ = PeakBank.from_list(profile["banks"][kind][name]).threshold(alpha)
                delta = scores[name][ix] - threshold
                for side in (delta <= 0, delta > 0):
                    distance = np.where(np.isfinite(delta) & side, np.abs(delta), np.inf)
                    if np.isfinite(distance).any():
                        local, _ = np.unravel_index(distance.argmin(), distance.shape)
                        boundary.add(int(ix[local]))
    selected.update(boundary)
    return sorted(selected), boundary


def verify_frozen_cache(frame, profiles):
    with np.load(BASE / "round3_safe/v7/v7_inputs.npz", allow_pickle=False) as z:
        cache = {name: z[name] for name in ("mobility", "acceleration", "periodicity", "valid")}
    with np.load(BASE / "round7_temporal_fusion/v8_inputs.npz", allow_pickle=False) as z:
        extra = z["raw"]
    hits = frozen_triggers(cache, extra, profiles[0]["frozen"])
    first = np.where(hits.any(1), hits.argmax(1), -1)
    expected = pd.read_csv(LEGACY_DIR / "frozen_first_alarms.csv")
    np.testing.assert_array_equal(expected.global_row, frame.global_row)
    np.testing.assert_array_equal(first, expected.v82_frozen)
    return int(cache["valid"].sum())


def replay_raw(frame, b, profiles, scores, first, frozen, state, output):
    selected, boundary = selected_rows(b, profiles, scores, first, frozen)
    maximum = dict.fromkeys(scores, 0.)
    state_error, queries, mismatches, rank_differences = 0., 0, 0, 0
    records, groups, offsets = [], {}, {}
    for number, ix in enumerate(selected):
        row = b.iloc[ix]
        profile = profiles[int(row.init_state_id % 5)]
        if row.source not in groups:
            run = ROOT / "VLA_MUI_HUB" / row.source
            groups[row.source] = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
            entries = sorted(json.loads((run / "client/summaries.json").read_text()), key=lambda x: x["episode_index"])
            offsets[row.source] = np.r_[0, np.cumsum([x["inference_calls"] for x in entries])]
        lo, hi = offsets[row.source][row.episode:row.episode + 2]
        group = groups[row.source]
        assert hi - lo == row.length
        np.testing.assert_array_equal(group["episode_id"][lo:hi], row.episode)
        raw = group["hb_router_probs"][lo:hi]
        monitor = StateActionGuardMonitor(output / f"fold_{profile['fold']}_profile.json", row.checkpoint)
        observed = [monitor.update(x) for x in raw]
        actual_state = np.asarray([r["state_mobility"] for r in observed])
        expected_state = state[row.global_row, :row.length]
        np.testing.assert_array_equal(actual_state, expected_state)
        state_error = max(state_error, float(np.nanmax(np.abs(actual_state - expected_state))))
        current = {name: np.asarray([r["scores"][name] for r in observed]) for name in scores}
        for name, values in current.items():
            expected = scores[name][ix, :row.length]
            np.testing.assert_array_equal(np.isfinite(values), np.isfinite(expected))
            finite = np.isfinite(values)
            if finite.any():
                maximum[name] = max(maximum[name], float(np.max(np.abs(values[finite] - expected[finite]))))
                np.testing.assert_allclose(values, expected, atol=2e-3, rtol=2e-3, equal_nan=True)
        for before, after in zip(observed, observed[1:]):
            if before["ever_alarm"]:
                assert after["ever_alarm"] and before["first_alarm_query"] == after["first_alarm_query"]
        assert monitor.first["v82"] == frozen[ix]
        assert monitor.first["combined"] == first[KINDS.index("task_init"), ALPHAS.index(.01), METHODS.index("v82_plus_confirmed"), ix]
        for ki, kind in enumerate(KINDS):
            banks = {name: PeakBank.from_list(profile["banks"][kind][name]) for name in scores}
            tails = {name: np.asarray([(1 + (banks[name].peaks >= x).sum()) / (len(banks[name].peaks) + 1)
                                      if np.isfinite(x) else np.nan for x in values]) for name, values in current.items()}
            for name, values in tails.items():
                expected = banks[name].tail(scores[name][ix, :row.length])
                rank_differences += int((np.isfinite(values) & (values != expected)).sum())
            ac_tail = np.fmin(tails["acceleration"], tails["decoupling"]) * 2
            accepted = (current["gap"] > 0) & (tails["gap"] <= GATE_ALPHA)
            for ai, alpha in enumerate(ALPHAS):
                trigger_sets = (ac_tail <= alpha, (current["gap"] > 0) & (tails["gap"] <= alpha),
                                (ac_tail <= alpha) & accepted)
                alarms = [int(np.flatnonzero(hits)[0]) if hits.any() else -1 for hits in trigger_sets]
                for value in (alarms[0], alarms[2]):
                    observed_first = [x for x in (value, int(frozen[ix])) if x >= 0]
                    alarms.append(min(observed_first) if observed_first else -1)
                for mi, actual in enumerate(alarms):
                    expected = int(first[ki, ai, mi, ix])
                    mismatches += actual != expected
                    records.append(dict(global_row=row.global_row, task=row.task, episode=row.episode,
                                        failure=row.failure, boundary=ix in boundary, calibration=kind,
                                        alpha=alpha, method=METHODS[mi], cached_first=expected,
                                        raw_first=actual, exact=actual == expected))
        queries += row.length
        if (number + 1) % 20 == 0:
            print(f"RAW ST/AC REPLAY {number+1}/{len(selected)} episodes, {queries} queries", flush=True)
    pd.DataFrame(records).to_csv(output / "raw_replay.csv", index=False)
    assert mismatches == 0, f"{mismatches} first alarms differ under raw replay"
    return dict(raw_episodes=len(selected), raw_queries=int(queries), boundary_episodes=len(boundary),
                selected_global_rows=b.iloc[selected].global_row.tolist(), maximum_score_error=maximum,
                maximum_state_error=state_error, tail_rank_differences=rank_differences,
                first_alarm_checks=len(records), first_alarm_mismatches=mismatches)


def verify_metrics(b, first, frozen, scores, tails, old_score, output):
    data = pd.read_csv(output / "alarm_metrics.csv")
    y = b.failure.to_numpy(bool)
    for row in data.itertuples():
        alarms = frozen if row.method == "v82_frozen" else first[KINDS.index(row.calibration), ALPHAS.index(row.alpha), METHODS.index(row.method)]
        mask = np.ones(len(b), bool) if row.scope == "all" else (b.task.eq(row.scope) | b.suite.eq(row.scope)).to_numpy()
        assert row.tp == int((mask & y & (alarms >= 0)).sum())
        assert row.fp == int((mask & ~y & (alarms >= 0)).sum())
    paired = pd.read_csv(output / "paired_changes.csv")
    for row in paired.itertuples():
        ki, ai = KINDS.index(row.calibration), ALPHAS.index(row.alpha)
        new = first[ki, ai, METHODS.index("v82_plus_confirmed")]
        old = frozen if row.baseline == "v82_frozen" else first[ki, ai, METHODS.index(row.baseline)]
        mask = np.ones(len(b), bool) if row.scope == "all" else (b.task.eq(row.scope) | b.suite.eq(row.scope)).to_numpy()
        assert row.lost_tp == int((mask & y & (old >= 0) & (new < 0)).sum())
        assert row.gained_tp == int((mask & y & (old < 0) & (new >= 0)).sum())
        assert row.removed_fp == int((mask & ~y & (old >= 0) & (new < 0)).sum())
        assert row.added_fp == int((mask & ~y & (old < 0) & (new >= 0)).sum())
    checks = 0
    names = tuple(scores)
    for row in pd.read_csv(output / "query_auroc.csv").iloc[::37].itertuples():
        ki, q = KINDS.index(row.calibration), row.query
        ac_p = 2 * np.fmin(tails[ki, names.index("acceleration"), :, q], tails[ki, names.index("decoupling"), :, q])
        p_st = np.where(scores["gap"][:, q] > 0, tails[ki, names.index("gap"), :, q], 1.)
        rank = -np.log(ac_p) if row.method == "ac" else scores[row.method][:, q] if row.method in scores else np.minimum(np.log(.01 / ac_p), np.log(GATE_ALPHA / p_st))
        mask = b.task.eq(row.task).to_numpy() & np.isfinite(rank) & np.isfinite(old_score[:, q]) & np.isfinite(scores["gap"][:, q])
        positive, negative = rank[mask & y], rank[mask & ~y]
        actual = mannwhitneyu(positive, negative).statistic / (len(positive) * len(negative))
        np.testing.assert_allclose(actual, row.auc, atol=1e-12)
        assert len(positive) == row.failures and len(negative) == row.successes
        checks += 1
    return dict(metric_rows=len(data), paired_rows=len(paired), independent_auc_checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "state_action_confirmation_20260909")
    output = parser.parse_args().output.resolve()
    manifest = json.loads((output / "experiment_verification.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
    for name, expected in json.loads((output / "execution_contract.json").read_text())["sources"].items():
        assert digest(HERE / name) == expected, name
    frame = pd.read_csv(FEATURE_DIR / "index.csv")
    b = pd.read_csv(output / "test_index.csv")
    profiles = [json.loads((output / f"fold_{f}_profile.json").read_text()) for f in range(5)]
    with np.load(output / "predictions.npz", allow_pickle=False) as z:
        first, frozen, tails, old_score = z["first"], z["frozen_first"], z["tails"], z["raw_v82_scores"]
        scores = {str(name): z[f"score_{name}"] for name in z["branches"]}
    with np.load(output / "state_readouts.npz", allow_pickle=False) as z:
        state = z["state"]
    cache_checks = verify_frozen_cache(frame, profiles)
    print(f"FROZEN CACHE exact: 32000 episodes, {cache_checks} queries", flush=True)
    raw = replay_raw(frame, b, profiles, scores, first, frozen, state, output)
    metrics = verify_metrics(b, first, frozen, scores, tails, old_score, output)
    result = dict(all_checks_passed=True, original_frozen_cache_queries=cache_checks, **raw, **metrics,
                  verifier_sha256=digest(__file__), raw_replay_sha256=digest(output / "raw_replay.csv"))
    write_json(output / "independent_verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "selected_global_rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
