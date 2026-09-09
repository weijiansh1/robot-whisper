"""Fixed retrospective ST/AC confirmation experiment on the existing A/B corpus."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from encoder import EncoderConfig
from guard import ALPHAS, KINDS, PeakBank, combine_tails, dynamics_scores, first_trigger
from state_action import (GATE_ALPHA, RELATION_NAMES, SCHEMA, confirmation_tail,
                          confirmed_relations, encode_relations)
from run_guard_experiment import (ACTION_CAPS, BASE, FEATURE_DIR, HERE, LEGACY_DIR, ROOT,
                                 RUN_B, S05, verified_inputs)
from run_analysis import bootstrap_indices, digest, rate_interval, state_splits, write_json
from monitor import auc, union

OLD_GUARD = BASE / "routing_guard_20260908"
STATE_CACHE = ROOT / "MoE-grammar/artifacts/state-action-mobility"
METHODS = ("ac_only", "st_gap_only", "confirmed_only", "v82_plus_ac", "v82_plus_confirmed")


def load_states(frame, valid, output):
    state = np.full((*valid.shape, 8), np.nan, np.float32)
    hashes, audits = {}, []
    for source, part in frame.groupby("source", sort=True):
        part = part.sort_values("episode")
        task, run_id = part.task.iloc[0], part.run_id.iloc[0]
        path = STATE_CACHE / f"{task.replace('/', '__')}__run__{run_id}.npz"
        hashes[str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"]))
            assert meta == dict(run_key=task, run_id=run_id, state_token=0)
            episode, query = z["episode_id"], z["episode_step"]
            expected_episode = np.repeat(part.episode.to_numpy(), part.length.to_numpy())
            expected_query = np.concatenate([np.arange(n) for n in part.length])
            np.testing.assert_array_equal(episode, expected_episode)
            np.testing.assert_array_equal(query, expected_query)
            values = z["state_move"][:, :, -1]
            assert np.isnan(values[query == 0]).all()
            assert np.isfinite(values[query > 0]).all() and (values[query > 0] >= 0).all()
            rows = np.repeat(part.global_row.to_numpy(), part.length.to_numpy())
            state[rows, query] = values
        summary_path = ROOT / "VLA_MUI_HUB" / source / "client/summaries.json"
        summaries = sorted(json.loads(summary_path.read_text()), key=lambda x: x["episode_index"])
        np.testing.assert_array_equal([s["episode_index"] for s in summaries], part.episode)
        np.testing.assert_array_equal([s["inference_calls"] for s in summaries], part.length)
        hashes[str(summary_path)] = digest(summary_path)
        audits.append(dict(source=source, episodes=len(part), queries=len(rows), cache_sha256=hashes[str(path)]))
    assert np.isfinite(state[:, 1:][valid[:, 1:]]).all()
    assert np.isnan(state[~valid]).all()
    np.savez_compressed(output / "state_readouts.npz", state=state, valid=valid,
                        global_rows=frame.global_row.to_numpy())
    pd.DataFrame(audits).to_csv(output / "state_cache_audit.csv", index=False)
    print(f"ST CACHE aligned: {len(audits)} sources, {len(frame)} episodes, {valid.sum()} queries", flush=True)
    return state, hashes


def calibrate(frame, relations, output):
    manifest = json.loads((OLD_GUARD / "experiment_verification.json").read_text())["artifacts"]
    source_profile = ROOT / "moe-v7-0905/results/intrinsic_guard_v7/global_profile.npz"
    with np.load(source_profile, allow_pickle=False) as z:
        frozen = {name: float(z[name]) for name in
                  ("freeze_threshold", "acceleration_threshold", "periodicity_threshold", "periodicity_scale")}
    frozen.update(frontback_threshold=0., curvature_threshold=.3890000581741333, slope=.0015)
    profiles, thresholds = [], []
    for fold, ref, cal, test in state_splits(frame):
        path = OLD_GUARD / f"fold_{fold}_profile.json"
        assert digest(path) == manifest[path.name]
        previous = json.loads(path.read_text())
        for name, values in (("reference_rows", ref), ("calibration_rows", cal), ("test_rows", test)):
            np.testing.assert_array_equal(previous[name], values)
        success = ~frame.iloc[cal].failure.to_numpy(bool)
        cal_frame = frame.iloc[cal].loc[success]
        group_index, groups = pd.factorize(pd.MultiIndex.from_frame(cal_frame[["task", "init_state_id"]]), sort=True)
        banks = {kind: {name: previous["banks"][kind][name] for name in ("acceleration", "decoupling")}
                 for kind in KINDS}
        for name, scores in relations.items():
            current = scores[cal][success]
            peaks = np.where(np.isfinite(current), current, -np.inf).max(axis=1)
            grouped = np.full(len(groups), -np.inf)
            np.maximum.at(grouped, group_index, peaks)
            for kind, units in zip(KINDS, (peaks, grouped)):
                bank = PeakBank(units)
                banks[kind][name] = bank.to_list()
                for alpha in (*ALPHAS, GATE_ALPHA):
                    threshold, rank = bank.threshold(alpha)
                    thresholds.append(dict(fold=fold, calibration=kind, feature=name, alpha=alpha,
                                           threshold=threshold, rank=rank, units=len(units),
                                           no_exposure_units=int(np.isneginf(units).sum()),
                                           attainable=rank <= len(units)))
        profile = dict(schema=SCHEMA, fold=fold, encoder_config=asdict(EncoderConfig()),
                       gate_alpha=GATE_ALPHA, default_calibration="task_init", default_alpha=.01,
                       checkpoints=previous["checkpoints"], banks=banks, frozen=frozen,
                       reference_rows=ref, calibration_rows=cal, test_rows=test,
                       success_calibration_rows=cal[success], old_ac_profile_sha256=digest(path),
                       frozen_profile_sha256=digest(source_profile), retrospective=True,
                       no_classifier_training=True)
        write_json(output / path.name, profile)
        profiles.append(json.loads((output / path.name).read_text()))
        print(f"CALIBRATED fold={fold}: {success.sum()} successful episodes, {len(groups)} groups", flush=True)
    pd.DataFrame(thresholds).to_csv(output / "calibration_thresholds.csv", index=False)
    names = [f"fold_{f}_profile.json" for f in range(5)] + ["calibration_thresholds.csv", "execution_contract.json"]
    write_json(output / "calibration_seal.json", dict(phase="before_B_metrics", artifacts={name: digest(output / name) for name in names}))
    return profiles


def predict(frame, valid, branches, profiles, output):
    b_rows = np.flatnonzero(frame.run_id.eq(RUN_B))
    b = frame.iloc[b_rows].reset_index(drop=True)
    assert len(b) == 16000 and b.failure.sum() == 564
    inverse = np.full(len(frame), -1, int)
    inverse[b_rows] = np.arange(len(b))
    branch_names = tuple(branches)
    tails = np.full((len(KINDS), len(branch_names), len(b), valid.shape[1]), np.nan)
    first = np.full((len(KINDS), len(ALPHAS), len(METHODS), len(b)), -2, np.int16)
    path = OLD_GUARD / "predictions.npz"
    assert digest(path) == json.loads((OLD_GUARD / "experiment_verification.json").read_text())["artifacts"][path.name]
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["global_rows"], b_rows)
        np.testing.assert_array_equal(z["valid"], valid[b_rows])
        old_branch_tails = z["branch_tails"]
        old_ac_first = z["first"][:, :, list(z["methods"]).index("dynamics_only")]
        old_branch_names = list(z["branches"])
        frozen, old_score = z["frozen_first"], z["raw_v82_scores"]
    b_scores = {name: value[b_rows] for name, value in branches.items()}
    assigned = np.zeros(len(b), bool)
    for profile in profiles:
        rows = np.asarray(profile["test_rows"])
        ix = inverse[rows]
        assert not assigned[ix].any()
        assigned[ix] = True
        for ki, kind in enumerate(KINDS):
            current = {name: PeakBank.from_list(profile["banks"][kind][name]).tail(value[rows])
                       for name, value in branches.items()}
            for bi, name in enumerate(branch_names):
                tails[ki, bi, ix] = current[name]
                if name in ("acceleration", "decoupling"):
                    np.testing.assert_array_equal(current[name], old_branch_tails[ki, old_branch_names.index(name), ix])
            ac = combine_tails(current, "dynamics_only")
            confirmed = confirmation_tail(current, current["gap"], branches["gap"][rows])
            state = np.where(branches["gap"][rows] > 0, current["gap"], 1.)
            for ai, alpha in enumerate(ALPHAS):
                original = frozen[ix]
                ac_first = first_trigger(ac, valid[rows], alpha)
                st_first = first_trigger(state, valid[rows], alpha)
                gated_first = first_trigger(confirmed, valid[rows], alpha)
                assert not ((confirmed <= alpha) & ~(ac <= alpha)).any()
                current_first = (ac_first, st_first, gated_first,
                                 union(original, ac_first), union(original, gated_first))
                for mi, alarm in enumerate(current_first):
                    first[ki, ai, mi, ix] = alarm
                combined = current_first[-1]
                assert ((combined[original >= 0] >= 0) & (combined[original >= 0] <= original[original >= 0])).all()
    assert assigned.all() and (first >= -1).all()
    np.testing.assert_array_equal(first[:, :, METHODS.index("ac_only")], old_ac_first)
    np.savez_compressed(output / "predictions.npz", tails=tails, first=first, frozen_first=frozen,
                        raw_v82_scores=old_score, global_rows=b_rows, valid=valid[b_rows],
                        branches=np.asarray(branch_names), methods=np.asarray(METHODS),
                        alphas=np.asarray(ALPHAS), kinds=np.asarray(KINDS),
                        **{f"score_{name}": values for name, values in b_scores.items()})
    b.to_csv(output / "test_index.csv", index=False)
    print("PREDICTED: existing AC alarms exact in all 8 settings; frozen v8.2 alarms preserved", flush=True)
    return b, tails, first, frozen, old_score, b_scores


def alarm_metrics(b, first, frozen, output):
    scopes = [("all", np.arange(len(b)))]
    scopes += [(str(name), part.index.to_numpy()) for name, part in b.groupby("suite")]
    scopes += [(str(name), part.index.to_numpy()) for name, part in b.groupby("task")]
    y = b.failure.to_numpy(bool)
    caps = b.suite.map(ACTION_CAPS).to_numpy()
    metrics, paired, episodes = [], [], []
    settings = [("original", np.nan, "v82_frozen", frozen)]
    settings += [(kind, alpha, method, first[ki, ai, mi]) for ki, kind in enumerate(KINDS)
                 for ai, alpha in enumerate(ALPHAS) for mi, method in enumerate(METHODS)]
    for kind, alpha, method, alarms in settings:
        alarms = alarms.astype(np.int64)
        fired = alarms >= 0
        for scope, ix in scopes:
            labels, hits = y[ix], fired[ix]
            tp, fp = int((hits & labels).sum()), int((hits & ~labels).sum())
            count_f, count_s = int(labels.sum()), int((~labels).sum())
            hit_q = alarms[ix][hits & labels]
            row = dict(calibration=kind, alpha=alpha, method=method, scope=scope,
                       tp=tp, fp=fp, failures=count_f, successes=count_s,
                       recall=tp / count_f if count_f else np.nan, fpr=fp / count_s if count_s else np.nan,
                       median_failure_q=float(np.median(hit_q)) if tp else np.nan,
                       median_failure_cap_percent=float(np.median(1000 * hit_q / caps[ix][hits & labels])) if tp else np.nan)
            for fraction in (.5, .6):
                early = hits & (10 * alarms[ix] <= fraction * caps[ix])
                row[f"tp_by_cap_{int(100*fraction)}"] = int((early & labels).sum())
                row[f"fp_by_cap_{int(100*fraction)}"] = int((early & ~labels).sum())
            if scope == "all":
                row.update(rate_interval(b, fired, y))
            metrics.append(row)
        if method == "v82_plus_confirmed":
            ki, ai = KINDS.index(kind), ALPHAS.index(alpha)
            for baseline, old in (("v82_frozen", frozen), ("v82_plus_ac", first[ki, ai, METHODS.index("v82_plus_ac")])):
                for scope, ix in scopes:
                    a, o, labels = fired[ix], old[ix] >= 0, y[ix]
                    both = a & o & labels
                    delta = alarms[ix][both] - old[ix][both]
                    paired.append(dict(calibration=kind, alpha=alpha, baseline=baseline, scope=scope,
                                       gained_tp=int((a & ~o & labels).sum()), lost_tp=int((~a & o & labels).sum()),
                                       added_fp=int((a & ~o & ~labels).sum()), removed_fp=int((~a & o & ~labels).sum()),
                                       earlier=int((delta < 0).sum()), same=int((delta == 0).sum()),
                                       later=int((delta > 0).sum())))
        if alpha == .01 or kind == "original":
            for i, row in b.iterrows():
                episodes.append(dict(global_row=row.global_row, task=row.task, suite=row.suite, episode=row.episode,
                                     init_state_id=row.init_state_id, failure=row.failure, calibration=kind,
                                     alpha=alpha, method=method, first_alarm=int(alarms[i]),
                                     cap_percent=1000 * int(alarms[i]) / caps[i] if fired[i] else np.nan))
    metrics, paired, episodes = pd.DataFrame(metrics), pd.DataFrame(paired), pd.DataFrame(episodes)
    metrics.to_csv(output / "alarm_metrics.csv", index=False)
    paired.to_csv(output / "paired_changes.csv", index=False)
    episodes.to_csv(output / "episode_alarms.csv", index=False)
    episodes.loc[episodes.task.eq(S05)].to_csv(output / "s05_alarms.csv", index=False)
    return metrics, paired


def auroc_tables(b, tails, old_score, scores, output):
    names = tuple(scores)
    records = []
    for ki, kind in enumerate(KINDS):
        lookup = {name: tails[ki, names.index(name)] for name in names}
        ac_p = 2 * np.fmin(lookup["acceleration"], lookup["decoupling"])
        gap_p = np.where(scores["gap"] > 0, lookup["gap"], 1.)
        rankings = dict(ac=-np.log(ac_p), ac_st_margin=np.minimum(np.log(.01 / ac_p), np.log(GATE_ALPHA / gap_p)),
                        **{name: scores[name] for name in RELATION_NAMES})
        common = np.isfinite(old_score) & np.logical_and.reduce([np.isfinite(v) for v in rankings.values()])
        for (suite, task), part in b.groupby(["suite", "task"]):
            ix, y = part.index.to_numpy(), part.failure.to_numpy(bool)
            for q in range(8, old_score.shape[1]):
                available = common[ix, q]
                labels = y[available]
                if not labels.any() or labels.all():
                    continue
                rows = ix[available]
                old_auc, ac_auc = auc(labels, old_score[rows, q]), auc(labels, rankings["ac"][rows, q])
                for name, values in rankings.items():
                    value = auc(labels, values[rows, q])
                    records.append(dict(calibration=kind, method=name, suite=suite, task=task, query=q,
                                        auc=value, ac_auc=ac_auc, v82_auc=old_auc,
                                        delta_ac=value - ac_auc, delta_v82=value - old_auc,
                                        failures=int(labels.sum()), successes=int((~labels).sum())))
    table = pd.DataFrame(records)
    table.to_csv(output / "query_auroc.csv", index=False)
    tasks, summaries = [], []
    fields = ("auc", "ac_auc", "v82_auc", "delta_ac", "delta_v82")
    for window, selected in (("q8_13", table.loc[table["query"].between(8, 13)]), ("all_comparable_queries", table)):
        grouped = selected.groupby(["calibration", "method", "suite", "task"], as_index=False).agg(
            **{name: (name, "mean") for name in fields}, queries=("query", "nunique"))
        grouped["window"] = window
        tasks.append(grouped)
        for (kind, method), part in grouped.groupby(["calibration", "method"]):
            for scope, rows in (("all", part), ("excluding_S05", part.loc[part.task.ne(S05)])):
                draws = bootstrap_indices(rows.task, rows.suite)
                for field in fields:
                    values = rows[field].to_numpy()
                    lo, hi = np.quantile(values[draws].mean(1), [.025, .975])
                    summaries.append(dict(calibration=kind, method=method, window=window, scope=scope,
                                          metric=field, estimate=float(values.mean()), lo=float(lo), hi=float(hi), tasks=len(rows)))
    pd.concat(tasks, ignore_index=True).to_csv(output / "task_auroc.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "auroc_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "state_action_confirmation_20260909")
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame, features, valid, hashes = verified_inputs()
    source_names = ("encoder.py", "guard.py", "state_action.py", "run_state_action_experiment.py", "STATE_ACTION_PROTOCOL_ZH.md")
    write_json(output / "execution_contract.json", dict(
        retrospective=True, no_classifier_training=True, B_parameter_selection=False,
        primary=dict(method="v82_plus_confirmed", calibration="task_init", alpha=.01, gate_alpha=GATE_ALPHA),
        alphas=ALPHAS, kinds=KINDS, raw_routing_only=True, duration_baseline_computed=False,
        inherited_v82_query_slope=.0015, old_ac_banks_unchanged=True,
        input_hashes=hashes, sources={name: digest(HERE / name) for name in source_names}))
    state, hashes = load_states(frame, valid, output)
    with np.load(FEATURE_DIR / "readouts.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["valid"], valid)
        action = z["values"][..., :8]
    expected = json.loads((FEATURE_DIR / "input_verification.json").read_text())["artifacts"]["readouts.npz"]
    assert digest(FEATURE_DIR / "readouts.npz") == expected
    hashes[str(FEATURE_DIR / "readouts.npz")] = expected
    relation = encode_relations(action, state, valid)
    np.savez_compressed(output / "relations.npz", features=relation, names=np.asarray(RELATION_NAMES),
                        global_rows=frame.global_row.to_numpy(), valid=valid)
    write_json(output / "input_verification.json", dict(all_checks_passed=True, episodes=len(frame),
               queries=int(valid.sum()), inputs=hashes, artifacts={name: digest(output / name)
               for name in ("state_readouts.npz", "relations.npz", "state_cache_audit.csv")}))
    relations = confirmed_relations(relation, valid)
    branches = dict(**dynamics_scores(features, valid), **relations)
    del features, state, action, relation
    profiles = calibrate(frame, relations, output)
    b, tails, first, frozen, old_score, b_scores = predict(frame, valid, branches, profiles, output)
    metrics, paired = alarm_metrics(b, first, frozen, output)
    auroc_tables(b, tails, old_score, b_scores, output)
    for name, expected in json.loads((output / "calibration_seal.json").read_text())["artifacts"].items():
        assert digest(output / name) == expected
    write_json(output / "experiment_verification.json", dict(all_checks_passed=True,
        every_B_episode_once=True, B_episodes=len(b), B_failures=int(b.failure.sum()),
        original_ac_first_alarms_exact=2 * 4 * len(b), frozen_alarms_never_lost_or_delayed=True,
        confirmation_is_query_level_subset_of_ac=True, calibration_seal_unchanged=True,
        artifacts={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
    main_rows = metrics.loc[metrics.scope.eq("all") & (metrics.alpha.eq(.01) | metrics.calibration.eq("original"))]
    print(main_rows[["calibration", "method", "tp", "fp", "recall", "fpr"]].to_string(index=False), flush=True)
    print(paired.loc[paired.scope.eq("all") & paired.alpha.eq(.01)].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
