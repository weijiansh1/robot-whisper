"""Exploratory equal-prefix validation of the unchanged v8.2 detector."""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import rankdata

from evaluate_v82_sample import frozen_sources, sha256_file, wilson_interval
from v82_closed_loop import V82Monitor, thresholds_at
from v8_closed_loop import SIGNALS, limits, risk, score_status


SOURCE = Path("/data/libero-runtime/samples/v82-evaluation-20260914T144322Z")
OUTPUT = Path("/data/libero-runtime/samples/v82-mechanism-validation-20260915")
PREFIXES = (10, 15, 18)
PRIMARY_QUERY = 15
SEED = 20260915
MONTE_CARLO_DRAWS = 20000
BRANCHES = ("freeze", "turbulence", "inversion", "curvature")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strata_for(episodes, mode):
    groups = {}
    for i, episode in enumerate(episodes):
        key = "all" if mode == "global" else episode["benchmark"]
        if mode == "benchmark_task":
            key += "/task%02d" % episode["base_task_id"]
        groups.setdefault(key, []).append(i)
    return [np.asarray(indices, int) for indices in groups.values()]


def exact_labelings(labels, groups):
    """All distinct assignments with the observed class counts in every group."""
    labels = np.asarray(labels, bool)
    np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(len(labels)))
    count = math.prod(math.comb(len(g), int(labels[g].sum())) for g in groups)
    if count > 200000:
        raise ValueError("Exact enumeration exceeds the declared resource bound")
    choices = [list(itertools.combinations(g, int(labels[g].sum()))) for g in groups]
    result = np.zeros((count, len(labels)), bool)
    for row, assignment in enumerate(itertools.product(*choices)):
        result[row, list(itertools.chain.from_iterable(assignment))] = True
    return result


def sampled_labelings(labels, groups, draws, rng):
    result = np.tile(np.asarray(labels, bool), (draws, 1))
    for group in groups:
        result[:, group] = rng.permuted(result[:, group], axis=1)
    return result


def auc_many(labelings, scores, groups=None):
    """Rank-sum AUC, pooling only positive/negative pairs inside each group."""
    labelings = np.atleast_2d(np.asarray(labelings, bool))
    scores = np.asarray(scores, float)
    if not np.isfinite(scores).all():
        raise ValueError("AUC requires finite scores; warmup is not a risk value")
    groups = [np.arange(len(scores))] if groups is None else groups
    numerator = np.zeros(len(labelings), float)
    pairs = 0
    for group in groups:
        positives = labelings[:, group].sum(axis=1)
        if not np.all(positives == positives[0]):
            raise ValueError("Conditional AUC requires preserved group class counts")
        k = int(positives[0])
        if k == 0 or k == len(group):
            continue
        numerator += labelings[:, group] @ rankdata(scores[group]) - k * (k + 1) / 2
        pairs += k * (len(group) - k)
    return (numerator / pairs if pairs else None), pairs


def youden_many(labelings, alarms):
    labels = np.atleast_2d(np.asarray(labelings, bool))
    alarms = np.asarray(alarms, float)
    positive, negative = labels.sum(axis=1), (~labels).sum(axis=1)
    if np.any(positive == 0) or np.any(negative == 0):
        return None
    return (labels @ alarms) / positive - ((~labels) @ alarms) / negative


def permutation_result(observed, null, exact):
    null = np.asarray(null, float)
    extreme = int(np.count_nonzero(null >= observed - 1e-12))
    p = extreme / len(null) if exact else (extreme + 1) / (len(null) + 1)
    return dict(observed=float(observed), null_mean=float(null.mean()),
                null_central_95=np.quantile(null, [.025, .975]).tolist(),
                null_draws=len(null), at_least_as_large=extreme,
                p_one_sided=p, exact=exact,
                tail="At least as much positive failure association as observed")


def confusion(labels, alarms):
    labels, alarms = np.asarray(labels, bool), np.asarray(alarms, bool)
    tp, fn = int((labels & alarms).sum()), int((labels & ~alarms).sum())
    fp, tn = int((~labels & alarms).sum()), int((~labels & ~alarms).sum())
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    j = recall - fpr if recall is not None and fpr is not None else None
    return dict(episodes=len(labels), tp=tp, fn=fn, fp=fp, tn=tn,
                recall=recall, false_positive_rate=fpr, youden_j=j,
                balanced_accuracy=(1 + j) / 2 if j is not None else None,
                recall_wilson_95=wilson_interval(tp, tp + fn),
                fpr_wilson_95=wilson_interval(fp, fp + tn))


def extract_status(status):
    return dict(scores=score_status(status),
                versions=np.asarray([status["alarm"], status["v8_alarm"], status["v82_alarm"]], bool),
                first=np.asarray([status["first_alarm_query"], status["v8_first"], status["v82_first"]], int),
                branches=np.asarray([status["freeze_alarm"], status["turbulence_alarm"],
                                     *(np.asarray(status["v82_head_first"]) >= 0)], bool),
                fixed_heads=np.asarray(status["v8_head_first"]) >= 0)


def variants(snapshot):
    branches = snapshot["branches"]
    result = dict(zip(("v7", "v8", "v82"), snapshot["versions"].tolist()))
    for i, branch in enumerate(BRANCHES):
        result["v82_without_" + branch] = bool(np.delete(branches, i).any())
        result["v82_only_" + branch] = bool(branches[i])
    return result


def plan(source, output):
    summary = json.loads((source / "summary.json").read_text())
    if not summary["complete"] or summary["evaluated_episodes"] != 60:
        raise ValueError("Expected the complete original 60-episode cohort")
    hashes = frozen_sources()
    if hashes != summary["source_sha256"]:
        raise ValueError("Frozen detector identity differs from the original evaluation")
    specification = dict(
        schema="local.v82.equal_prefix_validation.v1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(source), script_sha256=sha256_file(Path(__file__)),
        source_sha256={name: sha256_file(source / name) for name in (
            "plan.json", "summary.json", "signals.csv", "verification.json",
            "original-route-verification.json")},
        detector_sha256=hashes,
        episodes=[dict(name=e["name"], queries=e["queries"],
                       route_sha256=e["integrity"]["full_hb_sha256"])
                  for e in summary["episodes"]],
        primary_query_zero_based=PRIMARY_QUERY,
        secondary_queries_zero_based=[q for q in PREFIXES if q != PRIMARY_QUERY],
        observed_chunks={str(q): q + 1 for q in PREFIXES},
        action_steps_before_current_chunk={str(q): q * 10 for q in PREFIXES},
        all_episodes_required_at_every_prefix=True, future_padding=False,
        primary_statistic="v82 latched alarm Youden J at q15",
        primary_null="Exact whole-prefix/outcome reassignment within benchmark x base task",
        secondary_analyses=[
            "Frozen v7, fixed v8 and v8.2 at q10, q15 and q18",
            "Leave-one-branch-out and branch-only v8.2 alarms",
            "Pooled and within-task AUC of the existing continuous selector risk at each fixed query",
            "Global and within-benchmark Monte Carlo permutation controls",
            "Whole-episode ablations, descriptive only because observation counts differ",
        ],
        shuffle_unit="Entire equal-length routing prefix, with all axes, head correlations and temporal order intact",
        permutation_implementation=("For a deterministic monitor reset per episode, permuting outcome labels "
            "among fixed prefix predictions is equivalent to permuting whole routing prefixes against fixed outcomes. "
            "Exact enumeration removes duplicate assignments of indistinguishable labels."),
        permutation_seed=SEED, monte_carlo_draws=MONTE_CARLO_DRAWS,
        p_value_rule="One-sided; exact fraction including identity, or (extreme+1)/(draws+1) for Monte Carlo",
        fitted_parameters=0, new_policy_inferences=0, new_environment_actions=0,
        known_before_this_plan={"cohort_seen": True, "full_episode_metrics_seen": True,
                                "q15_alarm_counts_seen": True},
        study_status="Exploratory validation on previously inspected samples; not preregistered or independent holdout",
        assumptions=[
            "Under the conditional null, route prefixes and outcomes are exchangeable within benchmark/task",
            "Related tasks and shared initial states limit independent-sample interpretations",
            "No physical failure-onset or recovery labels exist",
            "The continuous selector risk is not a probability or the confirmed latched alarm",
            "Query index aligns observations and the original threshold schedule; wall-clock runtime is never a predictor",
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", specification)
    print(json.dumps(dict(event="protocol_frozen", output=str(output))), flush=True)


def replay(source, output, specification):
    summary = json.loads((source / "summary.json").read_text())
    episodes = summary["episodes"]
    with (source / "signals.csv").open(newline="") as stream:
        prior = {(r["episode"], int(r["query"])): r for r in csv.DictReader(stream)}
    expected = {e["name"]: e for e in specification["episodes"]}
    predictions, verification, rows = {}, [], []
    for index, episode in enumerate(episodes):
        name = episode["name"]
        archive = source / name / "full-hb-routes.npz"
        if sha256_file(archive) != expected[name]["route_sha256"]:
            raise ValueError("Route archive identity changed: " + name)
        with np.load(archive, allow_pickle=False) as data:
            probabilities = data["hb_router_probs"]
        if probabilities.shape != (episode["queries"], 8, 10, 11, 32):
            raise ValueError("Incomplete routing history: " + name)
        if len(probabilities) <= max(PREFIXES):
            raise ValueError("Episode ended before the common observation window: " + name)
        monitor, history = V82Monitor(), []
        for q, probability in enumerate(probabilities):
            snapshot = extract_status(monitor.update(probability))
            original = prior[name, q]
            reference = np.asarray([float(original[s + "_score"]) if original[s + "_score"] else np.nan
                                    for s in SIGNALS])
            np.testing.assert_allclose(snapshot["scores"], reference, rtol=0, atol=0, equal_nan=True)
            np.testing.assert_array_equal(snapshot["versions"],
                [original[v + "_alarm"] == "True" for v in ("v7", "v8", "v82")])
            np.testing.assert_array_equal(snapshot["first"],
                [int(original[v + "_first_query"]) for v in ("v7", "v8", "v82")])
            b = snapshot["branches"]
            np.testing.assert_array_equal(snapshot["versions"],
                [b[:2].any(), b[:2].any() or snapshot["fixed_heads"].any(), b.any()])
            if q <= max(PREFIXES) and int(original["action_step_before"]) != q * 10:
                raise ValueError("Action/chunk alignment differs across the common prefix")
            history.append(snapshot)
            row = dict(episode=name, query=q, action_step_before=int(original["action_step_before"]),
                       **variants(snapshot))
            row.update({s + "_score": float(value) if np.isfinite(value) else None
                        for s, value in zip(SIGNALS, snapshot["scores"])})
            rows.append(row)
        # Fresh monitors receive only each truncated input, never a future suffix or outcome.
        for q in PREFIXES:
            prefix_monitor = V82Monitor()
            prefix_status = None
            for probability in probabilities[:q + 1]:
                prefix_status = extract_status(prefix_monitor.update(probability))
            for key in history[q]:
                np.testing.assert_allclose(prefix_status[key], history[q][key], rtol=0, atol=0, equal_nan=True)
            if not np.isfinite(history[q]["scores"]).all():
                raise ValueError("A declared continuous-score prefix still contains warmup")
        np.testing.assert_array_equal(history[-1]["first"],
            [episode["first_alarm_query_zero_based"][v] for v in ("v7", "v8", "v82")])
        predictions[name] = history
        verification.append(dict(episode=name, queries=len(history), archive_hash_valid=True,
            all_original_scores_and_alarms_exact=True, all_branch_formulas_exact=True,
            isolated_prefixes_exact=list(PREFIXES)))
        if (index + 1) % 10 == 0:
            print(json.dumps(dict(event="replay_verified", episodes=index + 1)), flush=True)
    if len(rows) != len(prior):
        raise ValueError("Replay does not reproduce the complete source query set")
    save_csv(output / "predictions.csv", rows)
    save_json(output / "replay-verification.json", dict(episodes=verification, queries=len(rows),
              predictor_inputs="One complete routing tensor per query; monitor reset per episode",
              source_labels_used_by_monitor=False, future_suffix_used_by_monitor=False))
    return episodes, predictions, summary


def analyze(source, output):
    specification = json.loads((output / "protocol.json").read_text())
    implementation_hash = sha256_file(Path(__file__))
    amendment = None
    if specification["script_sha256"] != implementation_hash:
        amendment = json.loads((output / "implementation-amendment.json").read_text())
        if (amendment["original_script_sha256"] != specification["script_sha256"] or
                amendment["corrected_script_sha256"] != implementation_hash or
                amendment["analysis_specification_unchanged"] is not True):
            raise ValueError("Unrecorded validation code change after protocol freeze")
    for name, expected in specification["source_sha256"].items():
        if sha256_file(source / name) != expected:
            raise ValueError("Source evaluation changed: " + name)
    if frozen_sources() != specification["detector_sha256"]:
        raise ValueError("Frozen detector changed")
    episodes, predictions, original = replay(source, output, specification)
    labels = np.asarray([not e["source_result"]["success"] for e in episodes], bool)
    if (int(labels.sum()), int((~labels).sum())) != (51, 9):
        raise ValueError("Unexpected original cohort outcomes")
    thresholds, margins = limits()
    metric_rows, prefix_rows, metrics = [], [], {}
    for q in (*PREFIXES, -1):
        window = "full_episode" if q == -1 else "q%d" % q
        snapshots = [predictions[e["name"]][q] for e in episodes]
        alarms = [variants(s) for s in snapshots]
        metrics[window] = {}
        for variant in alarms[0]:
            values = np.asarray([a[variant] for a in alarms], bool)
            metrics[window][variant] = {}
            for benchmark in ("all", "plus", "pro"):
                subset = np.asarray([benchmark == "all" or e["benchmark"] == benchmark for e in episodes])
                m = confusion(labels[subset], values[subset])
                metrics[window][variant][benchmark] = m
                metric_rows.append(dict(window=window, variant=variant, benchmark=benchmark,
                    **{key: value for key, value in m.items() if not key.endswith("wilson_95")}))
                if q == -1 and variant in ("v7", "v8", "v82"):
                    for key in ("tp", "fn", "fp", "tn"):
                        if m[key] != original["metrics"][variant][benchmark][key]:
                            raise ValueError("Original full-episode metrics do not reproduce")
        if q != -1:
            current_risk = risk(np.stack([s["scores"] for s in snapshots]),
                                thresholds_at("iid_v82", q, thresholds), margins)
            for i, episode in enumerate(episodes):
                prefix_rows.append(dict(episode=episode["name"], benchmark=episode["benchmark"],
                    base_task_id=episode["base_task_id"], init_state_id=episode["init_state_id"],
                    failed=bool(labels[i]), query=q, chunks_observed=q + 1,
                    action_steps_before_current_chunk=q * 10, **alarms[i],
                    continuous_selector_risk=float(current_risk[i])))
    save_csv(output / "metrics.csv", metric_rows)
    save_csv(output / "prefix-episodes.csv", prefix_rows)

    groups = strata_for(episodes, "benchmark_task")
    exact = exact_labelings(labels, groups)
    if len(exact) != 6561 or len(np.unique(exact, axis=0)) != len(exact):
        raise ValueError("Unexpected exact conditional permutation space")
    if not np.any(np.all(exact == labels, axis=1)):
        raise ValueError("Exact null does not include the observed assignment")
    permutations = {"benchmark_task_exact": exact}
    rng = np.random.default_rng(SEED)
    for mode in ("global", "benchmark"):
        permutations[mode + "_monte_carlo"] = sampled_labelings(
            labels, strata_for(episodes, mode), MONTE_CARLO_DRAWS, rng)
    null_arrays, permutation_tests = {}, {}
    for q in PREFIXES:
        window = "q%d" % q
        snapshots = [predictions[e["name"]][q] for e in episodes]
        alarms = np.asarray([s["versions"][2] for s in snapshots], bool)
        scores = risk(np.stack([s["scores"] for s in snapshots]),
                      thresholds_at("iid_v82", q, thresholds), margins)
        observed_j = float(youden_many(labels, alarms)[0])
        observed_auc = float(auc_many(labels, scores)[0][0])
        observed_conditional, informative_pairs = auc_many(labels, scores, groups)
        permutation_tests[window] = {}
        for mode, reassigned in permutations.items():
            is_exact = mode.endswith("_exact")
            null_j = youden_many(reassigned, alarms)
            null_auc = auc_many(reassigned, scores)[0]
            result = dict(alarm_youden_j=permutation_result(observed_j, null_j, is_exact),
                          pooled_continuous_auc=permutation_result(observed_auc, null_auc, is_exact))
            null_arrays[window + "__" + mode + "__alarm_youden_j"] = null_j
            null_arrays[window + "__" + mode + "__pooled_continuous_auc"] = null_auc
            if is_exact:
                conditional_auc = auc_many(reassigned, scores, groups)[0]
                result["conditional_continuous_auc"] = dict(
                    permutation_result(float(observed_conditional[0]), conditional_auc, True),
                    informative_failure_success_pairs=informative_pairs,
                    informative_groups=int(sum(0 < labels[g].sum() < len(g) for g in groups)))
                null_arrays[window + "__" + mode + "__conditional_continuous_auc"] = conditional_auc
            permutation_tests[window][mode] = result
    np.savez_compressed(output / "permutation-distributions.npz", **null_arrays)
    stratum_rows = []
    for group in groups:
        e = episodes[int(group[0])]
        stratum_rows.append(dict(benchmark=e["benchmark"], base_task_id=e["base_task_id"],
            failures=int(labels[group].sum()), successes=int((~labels[group]).sum()),
            exact_distinct_assignments=math.comb(len(group), int(labels[group].sum()))))
    save_csv(output / "permutation-strata.csv", stratum_rows)
    if frozen_sources() != specification["detector_sha256"]:
        raise ValueError("Detector changed during verification")
    result = dict(complete=True, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        study_status=specification["study_status"], source=str(source),
        protocol_sha256=sha256_file(output / "protocol.json"),
        implementation_sha256=implementation_hash, implementation_amendment=amendment,
        failures=int(labels.sum()), successes=int((~labels).sum()),
        minimum_episode_action_steps=min(e["action_steps"] for e in episodes),
        common_prefixes=list(PREFIXES), primary_query=PRIMARY_QUERY,
        metrics=metrics, permutation_tests=permutation_tests,
        checks=dict(frozen_detector_unchanged=True, full_episode_metrics_reproduced=True,
            all_2891_source_queries_reproduced=True, fresh_prefixes_exact=True,
            branch_or_formulas_exact=True, exact_permutations_unique=6561,
            observed_labeling_in_exact_null=True, statistic_self_tests_passed=True),
        limitations=specification["assumptions"] + [
            "Only nine successes, all from Plus; Pro has no evidence about false alarms",
            "Only eight mixed-label benchmark/task groups contribute to conditional discrimination",
            "Secondary comparisons are exploratory and their p values are not multiplicity adjusted",
            "Label reassignment tests association; it is not a routing intervention or recovery experiment",
        ])
    save_json(output / "results.json", result)
    write_report(output, result)
    print(json.dumps(dict(event="validation_complete", output=str(output),
                          primary=permutation_tests["q15"]["benchmark_task_exact"])), flush=True)


def write_report(output, result):
    lines = ["# Frozen v8.2: equal-prefix validation", "",
        result["study_status"] + ".", "",
        "60 existing episodes: 51 failures and 9 successes. No training, tuning, new policy inference, "
        "or simulator intervention. Every episode is still active at all tested prefixes. "
        "The shortest episode finishes after 182 actions. Query index is an alignment coordinate; "
        "wall-clock runtime is not used.", "",
        "## Fixed observation budgets", "",
        "All numbers use the original frozen, confirmed, latched alarm. The primary comparison is q15. "
        "At q15, 16 routing chunks have been observed and 150 actions have already been executed. "
        "The newly predicted chunk has not yet been executed.", "",
        "| Prefix | Version | Detected failures / 51 | False alarms / 9 | Recall |",
        "|---|---|---:|---:|---:|"]
    for q in PREFIXES:
        for version in ("v7", "v8", "v82"):
            m = result["metrics"]["q%d" % q][version]["all"]
            lines.append(f"| {q + 1} chunks / {q * 10} actions | {version} | {m['tp']} | {m['fp']} | {m['recall']:.1%} |")
    lines += ["", "Zero observed false alarms among nine successes is not a zero population false-alarm rate. "
              "Its approximate 95% Wilson interval extends to 29.9%. Pro contributes 30 failures and no successes.",
              "", "## Conditional permutation control", "",
              "The primary null reassigns entire equal-length routing histories against outcomes within each "
              "benchmark and base task. It preserves temporal order, denoising order, layer/token axes, "
              "head correlations, query index, threshold schedule, group class counts, and observation length. "
              "Because scoring is deterministic and resets per episode, reassigning labels to frozen predictions "
              "is equivalent to reassigning those complete histories.", "",
              "There are 6,561 distinct conditional assignments. Eight groups contain both outcomes, "
              "providing 16 within-group failure/success pairs. All other groups cannot distinguish failure "
              "from success conditionally. P values are one-sided under within-group exchangeability.", "",
              "| Prefix | Alarm Youden J | Null mean J | Exact p | Conditional continuous AUC | Exact p (secondary) |",
              "|---|---:|---:|---:|---:|---:|"]
    for q in PREFIXES:
        test = result["permutation_tests"]["q%d" % q]["benchmark_task_exact"]
        j, auc = test["alarm_youden_j"], test["conditional_continuous_auc"]
        lines.append(f"| q{q} | {j['observed']:.4f} | {j['null_mean']:.4f} | {j['p_one_sided']:.4f} | "
                     f"{auc['observed']:.4f} | {auc['p_one_sided']:.4f} |")
    primary = result["permutation_tests"]["q15"]["benchmark_task_exact"]["alarm_youden_j"]
    if primary["p_one_sided"] >= .05:
        lines += ["", "The primary equal-prefix test does not establish failure-specific discrimination beyond "
                  "benchmark/task grouping at the conventional 0.05 level. This is limited evidence, "
                  "not proof that the detector is ineffective."]
    else:
        lines += ["", "The primary equal-prefix test finds positive conditional association in this inspected "
                  "sample. Independent validation is still needed; this is not causal evidence."]
    lines += ["", "The continuous quantity is the existing v8.2 selector risk at the stated query, not a "
              "new fitted score, calibrated probability, or the confirmed latched alarm. Warmup values "
              "are excluded by requiring every declared prefix score to be finite. Global and "
              "benchmark-only controls are recorded in results.json; they control less confounding.",
              "", "## Branch ablations", "",
              "These remove Boolean alarm branches after preserving their original scores, confirmation, "
              "and latching. Turbulence is the conjunction of the latched acceleration and recurrence-loss "
              "flags. Removing a branch does not modify the policy or simulate a repaired trajectory.", "",
              "| Variant | q15 TP / FP | q18 TP / FP | Full episode TP / FP |",
              "|---|---:|---:|---:|"]
    for variant in ("v7", "v8", "v82", *("v82_without_" + b for b in BRANCHES)):
        cells = []
        for window in ("q15", "q18", "full_episode"):
            m = result["metrics"][window][variant]["all"]
            cells.append(f"{m['tp']} / {m['fp']}")
        lines.append("| " + variant + " | " + " | ".join(cells) + " |")
    lines += ["", "The full-episode column is descriptive: failures have longer observation histories. "
              "Its 84.3% recall must not be interpreted as early equal-prefix recall.",
              "", "## Verification and limits", "",
              "All 2,891 original query scores and alarms reproduce exactly from the hashed complete HB "
              "probability archives. Fresh monitors given only each truncated prefix produce exactly "
              "the same state as the corresponding prefix of full replay. Branch formulas and all "
              "original confusion matrices reproduce. Frozen detector hashes remain unchanged.", "",
              "The statistical helpers are checked against hand-enumerated assignments and direct pairwise "
              "AUC calculations, including ties and groups without both labels. The exact null includes "
              "the observed assignment and contains no duplicates.", "",
              "This supports correct causal-prefix implementation, not a causal theory of failure or proof "
              "of recovery benefit. The next independent validation needs new trajectories with adequate "
              "successful cases, predefined prefixes, and unchanged thresholds. Testing whether acting "
              "on alarms improves success requires a separate matched intervention experiment.", "",
              "## Artifacts", "",
              "- protocol.json: analysis specification and source hashes, written before this run.",
              "- results.json / metrics.csv: complete aggregate results and confidence intervals.",
              "- prefix-episodes.csv: episode-level fixed-prefix scores, predictions, and outcomes.",
              "- predictions.csv / replay-verification.json: full replay and prefix checks.",
              "- permutation-strata.csv / permutation-distributions.npz: auditable null construction.",
              "- validation.png / validation.pdf: fixed-prefix results and primary null distribution.", ""]
    if result["implementation_amendment"]:
        lines += ["The original protocol is preserved. implementation-amendment.json records a JSON "
                  "export type correction after the first run. No analysis choices or parameters changed.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


class StatisticalTests(unittest.TestCase):
    def test_exact_assignments_and_one_sided_tail(self):
        labels = np.array([True, False, False, True, False])
        groups = [np.array([0, 1, 2]), np.array([3, 4])]
        assignments = exact_labelings(labels, groups)
        expected = {(a, b) for a in range(3) for b in range(3, 5)}
        self.assertEqual({tuple(np.flatnonzero(row)) for row in assignments}, expected)
        scores = np.array([3., 2., 1., 2., 1.])
        observed, pairs = auc_many(labels, scores, groups)
        null, _ = auc_many(assignments, scores, groups)
        self.assertEqual(pairs, 3)
        self.assertEqual(float(observed[0]), 1.)
        self.assertAlmostEqual(permutation_result(observed[0], null, True)["p_one_sided"], 1 / 6)

    def test_auc_matches_direct_pairs_including_ties(self):
        labels = np.array([True, False, True, False, True, True])
        scores = np.array([2., 2., 1., 0., 99., -99.])
        groups = [np.array([0, 1, 2, 3]), np.array([4, 5])]
        assignments = exact_labelings(labels, groups)
        actual, count = auc_many(assignments, scores, groups)
        for i, assignment in enumerate(assignments):
            direct = []
            for group in groups:
                for p in group[assignment[group]]:
                    for n in group[~assignment[group]]:
                        direct.append(float(scores[p] > scores[n]) + .5 * (scores[p] == scores[n]))
            self.assertEqual(count, len(direct))
            self.assertAlmostEqual(actual[i], float(np.mean(direct)))

    def test_sampled_assignments_conserve_groups_and_mc_correction(self):
        labels = np.array([True, False, False, True, True])
        groups = [np.array([0, 1, 2]), np.array([3, 4])]
        sampled = sampled_labelings(labels, groups, 100, np.random.default_rng(5))
        self.assertTrue(np.all(sampled[:, :3].sum(1) == 1))
        self.assertTrue(sampled[:, 3:].all())
        self.assertAlmostEqual(permutation_result(1., np.zeros(9), False)["p_one_sided"], .1)

    def test_undefined_fpr_and_known_youden(self):
        self.assertIsNone(confusion([True, True], [True, False])["false_positive_rate"])
        self.assertIsNone(auc_many([True, True], [0., 1.])[0])
        self.assertAlmostEqual(youden_many([True, True, False, False], [True, False, True, False])[0], 0.)


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StatisticalTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit("Statistical implementation tests failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "test"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "test":
        run_tests()
    elif args.command == "plan":
        plan(args.source, args.output)
    else:
        run_tests()
        analyze(args.source, args.output)
