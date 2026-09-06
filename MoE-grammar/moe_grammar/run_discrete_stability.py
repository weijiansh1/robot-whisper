from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from moe_grammar.discrete_phenotype import phenotype_track_scores
from moe_grammar.run_discrete_grammar_audit import physical_metrics, render_report
from moe_grammar.open_world import PhaseConditionalCDF
from moe_grammar.run_open_world_audit import load_dataset
from moe_grammar.run_three_channel_audit import (
    balance_episodes_by_task,
    balanced_rows,
    episode_maxima,
    episode_rows,
    select_episodes,
)


METRICS = (
    "ordered_vs_bag",
    "prefix_vs_clock",
    "prefix_vs_last1",
    "ordered_vs_unigram",
    "order2_vs_order1",
    "order3_vs_order2",
    "order4_vs_order3",
    "order4_vs_order1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/open-world-phenotypes-v2.npz")
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=Path("results-open-world-global-k12/summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results-discrete-grammar-v1")
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    return parser.parse_args()


def clustered_inference(
    numerators: np.ndarray,
    denominators: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, dict[str, float | list[float]]]:
    rng = np.random.default_rng(seed)
    states = len(denominators)
    draws = rng.integers(0, states, size=(samples, states))
    denominator_draws = denominators[draws].sum(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, states))
    output: dict[str, dict[str, float | list[float]]] = {}
    for column, metric in enumerate(METRICS):
        numerator = numerators[:, column]
        estimate = float(numerator.sum() / denominators.sum())
        bootstrap = numerator[draws].sum(axis=1) / denominator_draws
        null = (signs * numerator[None, :]).sum(axis=1) / denominators.sum()
        output[metric] = {
            "bits_per_track": estimate,
            "state_cluster_bootstrap_95ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "state_sign_flip_two_sided_p": float(
                (1 + np.sum(np.abs(null) >= abs(estimate))) / (samples + 1)
            ),
            "positive_state_fraction": float(np.mean(numerator > 0)),
        }
    return output


def main() -> None:
    args = parse_args()
    arrays, _ = load_dataset(args.dataset)
    folds = json.loads(args.fold_summary.read_text(encoding="utf-8"))["folds"]
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    base = np.asarray(arrays["full_base"], dtype=np.float32)
    starts = arrays["full_starts"]
    lengths = arrays["full_lengths"]
    states = arrays["full_state"]
    success = arrays["full_success"]
    episode_task = arrays["full_task"]

    state_numerators: list[np.ndarray] = []
    state_denominators: list[int] = []
    state_ids: list[int] = []
    fold_results: list[dict[str, float | int]] = []
    nll_totals = defaultdict_float()
    total_rows = 0
    anchor_last1 = np.full(len(arrays["anchor_base"]), np.nan, dtype=np.float32)
    anchor_assigned_fold = np.full(len(arrays["anchor_starts"]), -1, dtype=np.int8)

    for fold in folds:
        fold_index = int(fold["fold"])
        print(f"fold {fold_index}: rescoring held-out state clusters", flush=True)
        model = joblib.load(args.output_dir / "models" / f"fold_{fold_index}.joblib")
        scaled = model["scaler"].transform(base, np.zeros(len(base), dtype=np.int8))
        tracks = phenotype_track_scores(scaled)
        chords = model["tokenizer"].transform(tracks)
        prefix = model["hmm"].score(chords, starts, lengths, device=args.device)[0]
        clock = model["hmm"].score(
            chords,
            starts,
            lengths,
            update_with_observations=False,
            device=args.device,
        )[0]
        last1, last1_phase, _ = model["hmm"].score_last_observation(
            chords, starts, lengths, device=args.device
        )
        unigram = model["hmm"].global_nll(chords)
        test_success = select_episodes(
            states, np.asarray(fold["test_states"]), success
        )
        test_success = balance_episodes_by_task(
            test_success, episode_task, args.seed + 300 + fold_index
        )
        ordered_by_order = {
            order: model["ordered"].score(
                chords,
                starts,
                lengths,
                episodes=test_success,
                max_order=order,
            )[0]
            for order in range(1, 5)
        }
        ordered = ordered_by_order[4]
        bag = model["bag"].score(chords, starts, lengths, episodes=test_success)[0]
        rows = balanced_rows(starts, lengths, test_success)
        values = {
            "unigram": unigram,
            "clock": clock,
            "last1": last1,
            "prefix": prefix,
            "ordered": ordered,
            "bag": bag,
            **{f"order{order}": value for order, value in ordered_by_order.items()},
        }
        for key, value in values.items():
            nll_totals[key] += float(value[rows].sum())
        total_rows += len(rows)
        fold_gain = {
            "fold": fold_index,
            "rows": len(rows),
            "ordered_vs_bag_bits_per_track": float(
                np.sum(bag[rows] - ordered[rows]) / len(rows) / np.log(2.0)
            ),
            "prefix_vs_clock_bits_per_track": float(
                np.sum(clock[rows] - prefix[rows]) / len(rows) / np.log(2.0)
            ),
            "prefix_vs_last1_bits_per_track": float(
                np.sum(last1[rows] - prefix[rows]) / len(rows) / np.log(2.0)
            ),
            "ordered_vs_unigram_bits_per_track": float(
                np.sum(unigram[rows] - ordered[rows]) / len(rows) / np.log(2.0)
            ),
            "order2_vs_order1_bits_per_track": float(
                np.sum(ordered_by_order[1][rows] - ordered_by_order[2][rows])
                / len(rows)
                / np.log(2.0)
            ),
            "order3_vs_order2_bits_per_track": float(
                np.sum(ordered_by_order[2][rows] - ordered_by_order[3][rows])
                / len(rows)
                / np.log(2.0)
            ),
            "order4_vs_order3_bits_per_track": float(
                np.sum(ordered_by_order[3][rows] - ordered_by_order[4][rows])
                / len(rows)
                / np.log(2.0)
            ),
            "order4_vs_order1_bits_per_track": float(
                np.sum(ordered_by_order[1][rows] - ordered_by_order[4][rows])
                / len(rows)
                / np.log(2.0)
            ),
        }
        fold_results.append(fold_gain)

        for state in np.asarray(fold["test_states"], dtype=np.int64):
            episodes = test_success[states[test_success] == state]
            state_rows = balanced_rows(starts, lengths, episodes)
            numerators = np.asarray(
                [
                    np.sum(bag[state_rows] - ordered[state_rows]),
                    np.sum(clock[state_rows] - prefix[state_rows]),
                    np.sum(last1[state_rows] - prefix[state_rows]),
                    np.sum(unigram[state_rows] - ordered[state_rows]),
                    np.sum(
                        ordered_by_order[1][state_rows]
                        - ordered_by_order[2][state_rows]
                    ),
                    np.sum(
                        ordered_by_order[2][state_rows]
                        - ordered_by_order[3][state_rows]
                    ),
                    np.sum(
                        ordered_by_order[3][state_rows]
                        - ordered_by_order[4][state_rows]
                    ),
                    np.sum(
                        ordered_by_order[1][state_rows]
                        - ordered_by_order[4][state_rows]
                    ),
                ],
                dtype=np.float64,
            ) / np.log(2.0)
            state_numerators.append(numerators)
            state_denominators.append(len(state_rows))
            state_ids.append(int(state))

        calibration = select_episodes(
            states, np.asarray(fold["calibration_states"]), success
        )
        calibration = balance_episodes_by_task(
            calibration, episode_task, args.seed + 200 + fold_index
        )
        calibration_rows = balanced_rows(starts, lengths, calibration)
        calibrator = PhaseConditionalCDF().fit(
            last1,
            np.zeros(len(last1), dtype=np.int8),
            last1_phase,
            calibration_rows,
        )
        anchor_base = np.asarray(arrays["anchor_base"], dtype=np.float32)
        anchor_scaled = model["scaler"].transform(
            anchor_base, np.zeros(len(anchor_base), dtype=np.int8)
        )
        anchor_chords = model["tokenizer"].transform(
            phenotype_track_scores(anchor_scaled)
        )
        anchor_last, anchor_phase, _ = model["hmm"].score_last_observation(
            anchor_chords,
            arrays["anchor_starts"],
            arrays["anchor_lengths"],
            device=args.device,
        )
        anchor_percentile = calibrator.transform(
            anchor_last,
            np.zeros(len(anchor_last), dtype=np.int8),
            anchor_phase,
        )
        selected_episodes = np.flatnonzero(
            np.isin(arrays["anchor_state"], np.asarray(fold["test_states"]))
        )
        selected_rows = episode_rows(
            arrays["anchor_starts"], arrays["anchor_lengths"], selected_episodes
        )
        anchor_last1[selected_rows] = anchor_percentile[selected_rows]
        anchor_assigned_fold[selected_episodes] = fold_index

    numerators_array = np.stack(state_numerators)
    denominators_array = np.asarray(state_denominators, dtype=np.float64)
    inference = clustered_inference(
        numerators_array,
        denominators_array,
        seed=args.seed + 900,
        samples=args.bootstrap,
    )
    aggregate_nll = {
        f"{key}_nll_nats_per_track": value / total_rows
        for key, value in nll_totals.items()
    }
    expected = summary["healthy_next_chord"]
    checks = {
        key: float(aggregate_nll[key] - expected[key])
        for key in expected
        if key.endswith("_nll_nats_per_track") and key in aggregate_nll
    }
    if max(abs(value) for value in checks.values()) > 1e-6:
        raise AssertionError(f"stability rescoring does not reproduce headline NLL: {checks}")

    summary["sequence_stability"] = {
        "cluster_unit": "held-out init state",
        "state_clusters": len(state_ids),
        "bootstrap_samples": args.bootstrap,
        "fold_results": fold_results,
        "clustered_inference": inference,
        "reproduction_error_nats_per_track": checks,
        "context_order_curve_nats_per_track": {
            f"order_{order}": nll_totals[f"order{order}"] / total_rows
            for order in range(1, 5)
        },
        "rescored_nll_nats_per_track": aggregate_nll,
    }
    if np.any(~np.isfinite(anchor_last1)) or np.any(anchor_assigned_fold < 0):
        raise AssertionError("last-observation anchor scoring is incomplete")
    anchor_success = np.flatnonzero(arrays["anchor_success"])
    threshold = float(
        np.quantile(
            episode_maxima(
                anchor_last1,
                arrays["anchor_starts"],
                arrays["anchor_lengths"],
                anchor_success,
            ),
            0.95,
            method="higher",
        )
    )
    thresholds = {
        int(fold["fold"]): {"last1_innovation": threshold} for fold in folds
    }
    last1_physical = physical_metrics(
        arrays,
        {"last1_innovation": anchor_last1},
        anchor_assigned_fold,
        thresholds,
        ("last1_innovation",),
    )["last1_innovation"]
    summary["prefix_history_ablation"] = {
        "last1_matched_threshold": threshold,
        "last1_physical_stasis": last1_physical,
        "full_prefix_physical_stasis": summary[
            "physical_stasis_matched_5pct_operating_point"
        ]["prefix_innovation"],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.zh.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(f"updated {summary_path} with state-cluster inference", flush=True)


def defaultdict_float() -> dict[str, float]:
    return {
        key: 0.0
        for key in (
            "unigram",
            "clock",
            "last1",
            "prefix",
            "ordered",
            "bag",
            "order1",
            "order2",
            "order3",
            "order4",
        )
    }


if __name__ == "__main__":
    main()
