"""Label-free release detection from the per-query layer-by-expert router load.

The question is whether the 512-dim load -- ``action_load`` and ``state_load``, each an
8x32 grid of router probabilities -- supports an alarm for the object-release event that
is built without ever seeing a failure label, a release annotation, or a failure-derived
threshold. The configuration is pre-registered in ``results-release-unsupervised/PREREG.zh.md``
before any label is read, because the previous detector in this project calibrated its
threshold outcome-blind and then picked its rule out of 51,200 candidates by ranking them
on recall -- supervised model selection in a training-free costume.

Two detectors, neither of which is trained on failures.

1. Self-referential drift. Each episode's own queries ``[0, window)`` define its baseline
   mean and its baseline dispersion, per (channel, layer). Later queries are scored by
   their within-layer distance from that baseline in the episode's own dispersion units,
   and a CUSUM accumulates the excess so that the alarm needs a sustained departure rather
   than one spike. The statistic reads nothing but the episode's own past; only the two
   constants -- the CUSUM slack and the alarm height -- come from success episodes.

2. Success band. SAFE-style. Per task, the calibration successes' early window fixes a
   location and a scale for every one of the 512 coordinates, and every query of every
   episode is standardised by them. The time-varying band is then pooled across tasks at
   each query index, which is only legitimate *after* the per-task scale has been removed.
   Per-task success counts collapse after q15, so past the last query index with enough
   pooled support the band abstains outright rather than silently falling back to a pooled
   distribution that no longer represents the task mix. Coverage is reported next to every
   number.

Compositional handling. The 32 numbers of one (channel, layer) are a probability vector,
so every distance is taken after a CLR or square-root transform. Experts are independently
permutable between layers, so no distance is ever taken between layer a's 32-vector and
layer b's; the 16 within-layer distances are combined only after each has been
standardised into its own units.

Leakage. Episode length separates outcome almost perfectly on this corpus -- annotated
failures run to the horizon, almost no success does -- so nothing episode-level enters the
detector. Inference reads routing, the current query index and the task name, nothing else.

Nulls. (a) The expert coordinates are permuted within each (query, channel, layer), which
preserves how peaked a layer is and destroys which expert is up; the whole pipeline is
recalibrated on the permuted corpus. (b) A clock-only alarm that fires at a fixed query
index chosen to spend the same success false-alarm budget. The clock null is the one that
matters: it is what the length confound alone can buy.

Release annotations are opened only after every detector parameter and every threshold of
every pre-registered configuration is frozen; the freeze point is marked in ``main``.
``--dry-run`` stops before it and prints calibration-side diagnostics only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LABELS = Path(
    "/home/jovyan/work/himoe-vla/VLA_MUI_HUB/physical-failure-labels/results/failures.jsonl"
)
CHANNELS = ("action_load", "state_load")
# Layer index 0-3 are the front blocks L2-L5, 4-7 the back blocks L12-L15.
LAYER_LABELS = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
PERSISTENCE = 2
FPR_TARGETS = (0.01, 0.05, 0.10)
HEADLINE_FPR = 0.05
MIN_POOLED_ROWS = 50
MIN_POOLED_TASKS = 8
MIN_TASK_ROWS = 5
CLR_FLOOR = 1e-6
SCALE_FLOOR = 1e-4
SPREAD_FLOOR_QUANTILE = 0.05

# Pre-registered configurations. A is primary because it is written first here, not
# because of anything measured. All three are reported; none is selected.
CONFIGS: dict[str, dict[str, Any]] = {
    "A": {"transform": "clr", "window": 5, "slack_quantile": 0.50},
    "B": {"transform": "sqrt", "window": 5, "slack_quantile": 0.50},
    "C": {"transform": "clr", "window": 3, "slack_quantile": 0.75},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-dir", type=Path, default=Path("artifacts/layer-expert-load"))
    parser.add_argument("--meta-dir", type=Path, default=Path("artifacts/features-full40-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results-release-unsupervised"))
    parser.add_argument("--configs", default="A,B,C")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stop at the freeze point; print success-side diagnostics and read no label",
    )
    return parser.parse_args()


@dataclass
class Corpus:
    """Per-query router load plus an episode table. No outcome enters the load array."""

    load: np.ndarray  # [rows, 16, 32] float32 probabilities, grouped as channel x layer
    start: np.ndarray
    length: np.ndarray
    task: np.ndarray
    state: np.ndarray  # init_state_id, the blocking unit
    success: np.ndarray
    key: list[tuple[str, int, int]]
    task_names: list[str]
    groups: list[str]
    episode_of_row: np.ndarray
    query_of_row: np.ndarray

    @property
    def episodes(self) -> int:
        return len(self.start)

    @property
    def t_max(self) -> int:
        return int(self.length.max())


def transform(values: np.ndarray, kind: str) -> np.ndarray:
    """Map a probability vector to the space in which distances are taken."""
    if kind == "clr":
        logs = np.log(np.maximum(values, CLR_FLOOR))
        return (logs - logs.mean(axis=-1, keepdims=True)).astype(np.float32)
    if kind == "sqrt":
        return np.sqrt(values).astype(np.float32)
    raise ValueError(f"unknown transform {kind}")


def load_corpus(args: argparse.Namespace) -> Corpus:
    blocks, start, length, task, state, success, key = [], [], [], [], [], [], []
    task_names: list[str] = []
    cursor = 0
    for path in sorted(glob.glob(str(args.load_dir / "*.npz"))):
        stem = os.path.basename(path)
        load = np.load(path, allow_pickle=False)
        meta = np.load(args.meta_dir / stem, allow_pickle=False)
        episode_id = load["episode_id"]
        step = load["episode_step"]
        if not (
            np.array_equal(meta["episode_id"], episode_id)
            and np.array_equal(meta["episode_step"], step)
        ):
            raise ValueError(f"metadata rows do not align with the expert load in {stem}")
        run_key = json.loads(str(load["metadata_json"].item()))["run_key"]
        if run_key not in task_names:
            task_names.append(run_key)
        grid = np.stack([load[channel] for channel in CHANNELS], axis=1)
        if not np.isfinite(grid).all():
            raise ValueError(f"non-finite expert load in {stem}")
        blocks.append(grid.reshape(len(episode_id), -1, grid.shape[-1]).astype(np.float32))
        _, first, counts = np.unique(episode_id, return_index=True, return_counts=True)
        for index, count in zip(first, counts):
            if not np.array_equal(step[index : index + count], np.arange(count)):
                raise ValueError(f"query index is not 0..L-1 in {stem}")
            start.append(cursor + index)
            length.append(count)
            task.append(task_names.index(run_key))
            state.append(int(meta["init_state_id"][index]))
            success.append(bool(meta["success"][index]))
            key.append((run_key.split("/")[-1], state[-1], int(meta["flow_noise_seed"][index])))
        cursor += len(episode_id)
    lengths = np.asarray(length, dtype=np.int32)
    return Corpus(
        load=np.concatenate(blocks, axis=0),
        start=np.asarray(start, dtype=np.int64),
        length=lengths,
        task=np.asarray(task, dtype=np.int32),
        state=np.asarray(state, dtype=np.int32),
        success=np.asarray(success, dtype=bool),
        key=key,
        task_names=task_names,
        groups=[f"{channel}|{label}" for channel in CHANNELS for label in LAYER_LABELS],
        episode_of_row=np.repeat(np.arange(len(lengths)), lengths),
        query_of_row=np.concatenate([np.arange(n) for n in lengths]).astype(np.int32),
    )


def to_grid(values: np.ndarray, corpus: Corpus) -> np.ndarray:
    """Scatter per-row values into ``[episodes, t_max, groups]``, NaN where absent."""
    grid = np.full((corpus.episodes, corpus.t_max, values.shape[1]), np.nan, dtype=np.float32)
    grid[corpus.episode_of_row, corpus.query_of_row] = values
    return grid


def shuffle_experts(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Null (a): permute expert coordinates inside every (query, channel, layer).

    A single fixed permutation would leave every within-layer distance untouched, so the
    permutation is drawn per query. What survives is how peaked a layer is; what dies is
    which expert is up, and whether it is the same expert as one query ago.
    """
    out = np.empty_like(values)
    chunk = 32768
    for start in range(0, len(values), chunk):
        block = values[start : start + chunk]
        order = np.argsort(rng.random(block.shape, dtype=np.float32), axis=-1)
        out[start : start + chunk] = np.take_along_axis(block, order, axis=-1)
    return out


def drift_distance(
    values: np.ndarray, corpus: Corpus, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Within-layer distance from the episode's own early baseline, plus its spread.

    Nothing here crosses an episode boundary or a layer boundary.
    """
    groups, experts = values.shape[1], values.shape[2]
    baseline = np.empty((corpus.episodes, groups, experts), dtype=np.float32)
    spread = np.empty((corpus.episodes, groups), dtype=np.float32)
    for index, (start, length) in enumerate(zip(corpus.start, corpus.length)):
        block = values[start : start + min(window, length)]
        mean = block.mean(axis=0)
        baseline[index] = mean
        spread[index] = np.sqrt(np.square(block - mean).sum(axis=-1).mean(axis=0))
    distance = np.sqrt(
        np.square(values - baseline[corpus.episode_of_row]).sum(axis=-1, dtype=np.float64)
    )
    return distance.astype(np.float32), spread


def band_magnitude(
    values: np.ndarray, corpus: Corpus, calibration: np.ndarray, window: int
) -> np.ndarray:
    """Per-task standardised within-layer deviation magnitude.

    Location and scale are the calibration successes' early window, per task and per
    coordinate, so the per-task scale is gone before anything is pooled across tasks.
    """
    episode = corpus.episode_of_row
    early = corpus.query_of_row < window
    magnitude = np.empty((len(values), values.shape[1]), dtype=np.float32)
    task_of_row = corpus.task[episode]
    for task in range(len(corpus.task_names)):
        rows = task_of_row == task
        block = values[rows & early & calibration[episode]]
        mean = block.mean(axis=0)
        scale = np.maximum(block.std(axis=0), SCALE_FLOOR)
        z = (values[rows] - mean) / scale
        magnitude[rows] = np.sqrt(np.square(z).mean(axis=-1, dtype=np.float64))
    return magnitude


def pooled_band(
    grid: np.ndarray, calibration: np.ndarray, corpus: Corpus
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Median and scaled MAD of the deviation magnitude at each query index.

    Support is counted in calibration success episodes and in distinct tasks contributing
    at that query index. A query index failing either test is not covered, and the band
    refuses to score it rather than falling back to a pooled distribution that no longer
    represents the task mix.
    """
    t_max, groups = grid.shape[1], grid.shape[2]
    location = np.full((t_max, groups), np.nan, dtype=np.float32)
    width = np.full((t_max, groups), np.nan, dtype=np.float32)
    covered = np.zeros(t_max, dtype=bool)
    rows_per_query, tasks_per_query = [], []
    for query in range(t_max):
        present = calibration & np.isfinite(grid[:, query, 0])
        tasks = int(len(np.unique(corpus.task[present])))
        rows_per_query.append(int(present.sum()))
        tasks_per_query.append(tasks)
        if present.sum() < MIN_POOLED_ROWS or tasks < MIN_POOLED_TASKS:
            continue
        block = grid[present, query]
        median = np.median(block, axis=0)
        location[query] = median
        width[query] = np.maximum(1.4826 * np.median(np.abs(block - median), axis=0), 1e-3)
        covered[query] = True
    return (
        location,
        width,
        covered,
        {"calibration_rows_per_query": rows_per_query, "tasks_per_query": tasks_per_query},
    )


def cusum_alarm(
    score: np.ndarray, window: int, slack: float, height: float
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided CUSUM per episode; return the peak statistic and the first crossing."""
    accumulator = np.zeros(len(score), dtype=np.float32)
    peak = np.zeros(len(score), dtype=np.float32)
    alarm = np.full(len(score), -1, dtype=np.int32)
    for query in range(window, score.shape[1]):
        column = score[:, query]
        present = np.isfinite(column)
        step = np.nan_to_num(column - slack, nan=0.0, posinf=0.0, neginf=0.0)
        accumulator = np.where(present, np.maximum(0.0, accumulator + step), accumulator)
        peak = np.maximum(peak, accumulator)
        fresh = (alarm < 0) & present & (accumulator > height)
        alarm[fresh] = query
    return peak, alarm


def persistence_alarm(
    score: np.ndarray, covered: np.ndarray, window: int, height: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fire when the band is exceeded at ``PERSISTENCE`` consecutive covered query indices."""
    usable = np.where(np.isfinite(score) & covered[None, :], score, -np.inf)
    held = usable.copy()
    for offset in range(1, PERSISTENCE):
        shifted = np.full_like(usable, -np.inf)
        shifted[:, offset:] = usable[:, :-offset]
        held = np.minimum(held, shifted)
    held[:, :window] = -np.inf
    peak = held.max(axis=1)
    fired = held > height
    alarm = np.where(fired.any(axis=1), np.argmax(fired, axis=1), -1).astype(np.int32)
    return peak.astype(np.float32), alarm


def height_for_budget(peak: np.ndarray, target: float) -> float:
    """Alarm height spending ``target`` of *all* calibration success episodes.

    Episodes too short to produce a statistic stay in the denominator as non-firing, so the
    nominal budget is the realised calibration false-alarm rate.
    """
    finite = peak[np.isfinite(peak)]
    if finite.size == 0:
        return float("inf")
    filled = np.where(np.isfinite(peak), peak, finite.min() - 1.0)
    return float(np.quantile(filled, 1.0 - target))


def clock_null(corpus: Corpus, calibration: np.ndarray, window: int, target: float) -> int:
    """Earliest fixed query index whose reach rate on calibration successes fits the budget.

    This is the length confound turned into a detector, and it is the number every routing
    alarm has to beat.
    """
    lengths = corpus.length[calibration]
    for query in range(window, corpus.t_max):
        if float((lengths > query).mean()) <= target:
            return query
    return int(corpus.t_max)


def collapse(components: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Mean of the selected within-layer components; NaN marks an absent query."""
    return components[:, :, groups].mean(axis=2)


def group_sets(corpus: Corpus) -> dict[str, np.ndarray]:
    sets: dict[str, np.ndarray] = {"all": np.arange(len(corpus.groups))}
    for position, channel in enumerate(CHANNELS):
        base = position * 8
        sets[channel] = np.arange(base, base + 8)
        sets[f"{channel}|front"] = np.arange(base, base + 4)
        sets[f"{channel}|back"] = np.arange(base + 4, base + 8)
    sets["front"] = np.asarray([0, 1, 2, 3, 8, 9, 10, 11])
    sets["back"] = np.asarray([4, 5, 6, 7, 12, 13, 14, 15])
    for index, name in enumerate(corpus.groups):
        sets[name] = np.asarray([index])
    return sets


def offset_stats(offsets: np.ndarray) -> dict[str, Any]:
    if offsets.size == 0:
        return {"n": 0}
    edges = np.arange(-25, 30, 5)
    return {
        "n": int(offsets.size),
        "median": float(np.median(offsets)),
        "mean": float(offsets.mean()),
        "median_abs": float(np.median(np.abs(offsets))),
        "quantiles": {str(q): float(np.percentile(offsets, q)) for q in (5, 10, 25, 50, 75, 90, 95)},
        "within_2": float((np.abs(offsets) <= 2).mean()),
        "within_5": float((np.abs(offsets) <= 5).mean()),
        "early_fraction": float((offsets < 0).mean()),
        "histogram_edges": edges.tolist(),
        "histogram": np.histogram(np.clip(offsets, -25, 25), bins=edges)[0].tolist(),
    }


def run_config(
    corpus: Corpus, args: argparse.Namespace, fold_of: np.ndarray, name: str
) -> dict[str, Any]:
    """Everything that happens before the freeze point, for one pre-registered config."""
    setting = CONFIGS[name]
    window, quantile = int(setting["window"]), float(setting["slack_quantile"])
    sets = group_sets(corpus)
    rng = np.random.default_rng(args.seed)
    variants = {
        "real": transform(corpus.load, setting["transform"]),
    }
    variants["null_shuffled_experts"] = shuffle_experts(variants["real"], rng)
    alarms: dict[str, np.ndarray] = {}
    coverage: dict[str, Any] = {}
    clock_index: dict[str, list[int]] = {f"{t:.2f}": [] for t in FPR_TARGETS}
    at_alarm = np.full((corpus.episodes, len(corpus.groups)), np.nan, dtype=np.float32)
    calibration_fpr: dict[str, list[float]] = {}
    for variant, values in variants.items():
        distance, spread = drift_distance(values, corpus, window)
        distance_grid = to_grid(distance, corpus)
        del distance
        per_fold_meta: list[dict[str, Any]] = []
        floor_hits = 0
        for fold in range(args.folds):
            calibration = corpus.success & (fold_of != fold)
            evaluation = fold_of == fold
            # Numerical floor for the self-normalisation, from calibration successes only,
            # so a near-static baseline window cannot divide the distance to infinity.
            reference = np.quantile(spread[calibration], SPREAD_FLOOR_QUANTILE, axis=0)
            floor_hits += int((spread < reference[None, :]).sum())
            drift = distance_grid / np.maximum(spread, reference[None, :])[:, None, :]
            magnitude = to_grid(band_magnitude(values, corpus, calibration, window), corpus)
            location, width, covered, support = pooled_band(magnitude, calibration, corpus)
            per_fold_meta.append(
                {
                    "fold": fold,
                    "calibration_success": int(calibration.sum()),
                    "evaluation_success": int((evaluation & corpus.success).sum()),
                    "evaluation_failure": int((evaluation & ~corpus.success).sum()),
                    "covered_queries": int(covered.sum()),
                    "last_covered_query": int(np.flatnonzero(covered).max())
                    if covered.any()
                    else -1,
                    **support,
                }
            )
            standardised = (magnitude - location[None]) / width[None]
            standardised[:, ~covered] = np.nan
            for group_name, groups in sets.items():
                drift_score = collapse(drift, groups)
                band_score = collapse(standardised, groups)
                slack = float(np.nanquantile(drift_score[calibration, window:], quantile))
                peak_drift, _ = cusum_alarm(drift_score[calibration], window, slack, float("inf"))
                peak_band, _ = persistence_alarm(
                    band_score[calibration], covered, window, float("inf")
                )
                for target in FPR_TARGETS:
                    tag = f"{target:.2f}"
                    height = height_for_budget(peak_drift, target)
                    _, fired = cusum_alarm(drift_score[evaluation], window, slack, height)
                    alarms.setdefault(
                        f"{variant}|drift|{group_name}|{tag}",
                        np.full(corpus.episodes, -1, dtype=np.int32),
                    )[evaluation] = fired
                    if group_name == "all":
                        calibration_fpr.setdefault(f"{variant}|drift|{tag}", []).append(
                            float((peak_drift > height).mean())
                        )
                    if variant == "real" and group_name == "all" and target == HEADLINE_FPR:
                        rows = np.flatnonzero(evaluation)[fired >= 0]
                        at_alarm[rows] = drift[rows, fired[fired >= 0]]
                    height = height_for_budget(peak_band, target)
                    _, fired = persistence_alarm(band_score[evaluation], covered, window, height)
                    alarms.setdefault(
                        f"{variant}|band|{group_name}|{tag}",
                        np.full(corpus.episodes, -1, dtype=np.int32),
                    )[evaluation] = fired
                    if group_name == "all":
                        calibration_fpr.setdefault(f"{variant}|band|{tag}", []).append(
                            float((peak_band > height).mean())
                        )
                        if variant == "real":
                            clock_index[tag].append(clock_null(corpus, calibration, window, target))
            print(f"  [{name}] {variant} fold {fold} done", flush=True)
        coverage[variant] = {"folds": per_fold_meta, "spread_floor_hits": floor_hits}
        del distance_grid
    return {
        "setting": setting,
        "alarms": alarms,
        "coverage": coverage,
        "clock_index": clock_index,
        "at_alarm": at_alarm,
        "calibration_fpr": calibration_fpr,
    }


def same_task_coverage(
    corpus: Corpus, fold_of: np.ndarray, folds: int, subject: np.ndarray
) -> list[float]:
    """Fraction of the subject episodes' queries with a same-task same-query success reference."""
    out = []
    for fold in range(folds):
        counts = np.zeros((len(corpus.task_names), corpus.t_max), dtype=np.int32)
        for index in np.flatnonzero(corpus.success & (fold_of != fold)):
            counts[corpus.task[index], : corpus.length[index]] += 1
        covered_rows = total_rows = 0
        for index in np.flatnonzero(subject & (fold_of == fold)):
            span = counts[corpus.task[index], : corpus.length[index]]
            total_rows += len(span)
            covered_rows += int((span >= MIN_TASK_ROWS).sum())
        out.append(covered_rows / max(total_rows, 1))
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("loading corpus...", flush=True)
    corpus = load_corpus(args)
    print(
        f"{corpus.episodes:,} episode, {len(corpus.load):,} query 行, "
        f"成功 {int(corpus.success.sum()):,} / 失败 {int((~corpus.success).sum()):,}, "
        f"{len(corpus.task_names)} 个任务, t_max={corpus.t_max}",
        flush=True,
    )

    order = np.random.default_rng(args.seed).permutation(np.unique(corpus.state))
    fold_states = [order[index :: args.folds] for index in range(args.folds)]
    fold_of = np.full(corpus.episodes, -1, dtype=np.int16)
    for index, states in enumerate(fold_states):
        fold_of[np.isin(corpus.state, states)] = index
    if (fold_of < 0).any():
        raise ValueError("every episode must belong to exactly one state-blocked fold")

    names = [name.strip() for name in args.configs.split(",")]
    runs = {name: run_config(corpus, args, fold_of, name) for name in names}
    print(f"all configurations frozen at {time.time() - started:.0f}s", flush=True)

    if args.dry_run:
        for name, run in runs.items():
            meta = run["coverage"]["real"]["folds"]
            print(f"\n[{name}] setting {run['setting']}")
            print(
                "  pooled band last covered query per fold: "
                + ", ".join(str(row["last_covered_query"]) for row in meta)
            )
            for key, values in sorted(run["calibration_fpr"].items()):
                print(f"  calibration FPR {key}: " + ", ".join(f"{v:.3f}" for v in values))
            fired = np.isfinite(run["at_alarm"][:, 0])
            print(f"  episodes with a headline drift alarm: {int(fired.sum())}")
        print("\ndry run: no label was read")
        return

    # ------------------------------------------------------------------
    # FREEZE POINT. Every detector parameter -- transform, baseline window, CUSUM slack,
    # band location and width, coverage mask -- and every alarm height of every
    # pre-registered configuration is now fixed, each computed from success episodes of the
    # calibration folds alone. The release annotations are opened here, for the first time,
    # and only to score alarms that already exist.
    # ------------------------------------------------------------------
    release_map: dict[tuple[str, int, int], int] = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["data_root"] != "cache_new":
            continue
        for physics in (record.get("goal_subject_physics") or {}).values():
            snapshot = physics.get("first_release_snapshot")
            if snapshot is None:
                continue
            key = (record["task_name"], record["init_state_id"], record["flow_noise_seed"])
            release_map[key] = int(snapshot)
            break
    release = np.full(corpus.episodes, -1, dtype=np.int32)
    for index, key in enumerate(corpus.key):
        if not corpus.success[index]:
            release[index] = release_map.get(key, -1)
    annotated = release >= 0
    print(
        f"标注脱手的失败 episode: {int(annotated.sum())}; 脱手时刻 "
        f"min {int(release[annotated].min())} median {int(np.median(release[annotated]))} "
        f"max {int(release[annotated].max())}",
        flush=True,
    )

    def evaluate(alarm: np.ndarray, detail: bool = False) -> dict[str, Any]:
        hit = annotated & (alarm >= 0)
        per_fold = []
        for fold in range(args.folds):
            mask = fold_of == fold
            sub = annotated & mask
            local = sub & (alarm >= 0)
            per_fold.append(
                {
                    "fold": fold,
                    "n_failures": int(sub.sum()),
                    "detection_rate": float((alarm[sub] >= 0).mean()),
                    "success_fpr": float((alarm[mask & corpus.success] >= 0).mean()),
                    "median_offset": float(np.median(alarm[local] - release[local]))
                    if local.any()
                    else float("nan"),
                    "median_abs_offset": float(np.median(np.abs(alarm[local] - release[local])))
                    if local.any()
                    else float("nan"),
                }
            )
        per_task = []
        for task in range(len(corpus.task_names)):
            mask = annotated & (corpus.task == task)
            if not mask.any():
                continue
            local = mask & (alarm >= 0)
            per_task.append(
                {
                    "task": corpus.task_names[task],
                    "n": int(mask.sum()),
                    "detection_rate": float((alarm[mask] >= 0).mean()),
                    "median_abs_offset": float(np.median(np.abs(alarm[local] - release[local])))
                    if local.any()
                    else float("nan"),
                }
            )
        rates = [row["detection_rate"] for row in per_task]
        absolute = [
            row["median_abs_offset"] for row in per_task if np.isfinite(row["median_abs_offset"])
        ]
        out: dict[str, Any] = {
            "detection_rate": float((alarm[annotated] >= 0).mean()),
            "success_fpr": float((alarm[corpus.success] >= 0).mean()),
            "offsets": offset_stats((alarm[hit] - release[hit]).astype(np.float64)),
            "per_fold": per_fold,
            "per_task_detection": {
                "min": float(np.min(rates)),
                "median": float(np.median(rates)),
                "max": float(np.max(rates)),
            },
            "per_task_median_abs_offset": {
                "min": float(np.min(absolute)) if absolute else float("nan"),
                "median": float(np.median(absolute)) if absolute else float("nan"),
                "max": float(np.max(absolute)) if absolute else float("nan"),
            },
        }
        if detail:
            out["per_task"] = per_task
        return out

    headline = f"{HEADLINE_FPR:.2f}"
    summary: dict[str, Any] = {
        "episodes": corpus.episodes,
        "queries": int(len(corpus.load)),
        "success_episodes": int(corpus.success.sum()),
        "failure_episodes": int((~corpus.success).sum()),
        "annotated_release_failures": int(annotated.sum()),
        "release_quantiles": {
            str(q): float(np.percentile(release[annotated], q)) for q in (5, 25, 50, 75, 95)
        },
        "tasks": len(corpus.task_names),
        "folds": args.folds,
        "seed": args.seed,
        "persistence": PERSISTENCE,
        "fpr_targets": list(FPR_TARGETS),
        "min_pooled_rows": MIN_POOLED_ROWS,
        "min_pooled_tasks": MIN_POOLED_TASKS,
        "min_task_rows": MIN_TASK_ROWS,
        "same_task_same_query_coverage_per_fold": same_task_coverage(
            corpus, fold_of, args.folds, annotated
        ),
        "configs": {},
    }
    summary["same_task_same_query_coverage"] = float(
        np.mean(summary["same_task_same_query_coverage_per_fold"])
    )
    for name, run in runs.items():
        scored = {
            key: evaluate(alarm, detail=key.endswith(f"|all|{headline}"))
            for key, alarm in run["alarms"].items()
        }
        for tag, queries in run["clock_index"].items():
            alarm = np.full(corpus.episodes, -1, dtype=np.int32)
            for fold in range(args.folds):
                alarm[(fold_of == fold) & (corpus.length > queries[fold])] = queries[fold]
            scored[f"null_clock|clock|all|{tag}"] = evaluate(alarm, detail=tag == headline)
            scored[f"null_clock|clock|all|{tag}"]["fixed_query_per_fold"] = queries
        fired = np.isfinite(run["at_alarm"][:, 0])
        share: dict[str, Any] = {}
        for label, mask in (("failures", annotated & fired), ("successes", corpus.success & fired)):
            if not mask.any():
                continue
            block = run["at_alarm"][mask]
            block = block / block.sum(axis=1, keepdims=True)
            share[label] = {
                key: float(value) for key, value in zip(corpus.groups, block.mean(axis=0))
            }
        summary["configs"][name] = {
            "setting": run["setting"],
            "coverage": run["coverage"],
            "calibration_fpr": run["calibration_fpr"],
            "clock_null_query": run["clock_index"],
            "drift_component_share_at_alarm": share,
            "results": scored,
        }
    summary["elapsed_seconds"] = time.time() - started
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\n同任务同时刻成功参考覆盖率: {summary['same_task_same_query_coverage']:.1%}")
    for name in names:
        block = summary["configs"][name]
        last = block["coverage"]["real"]["folds"][0]["last_covered_query"]
        print(f"\n=== 配置 {name} {block['setting']}  pooled band 覆盖到 q{last} ===")
        print(
            f"{'检测器':40s} {'检出率':>7s} {'成功FPR':>8s} {'中位偏移':>8s} "
            f"{'|偏移|中位':>9s} {'±2内':>6s} {'±5内':>6s}"
        )
        for key in (
            f"real|drift|all|{headline}",
            f"null_shuffled_experts|drift|all|{headline}",
            f"real|band|all|{headline}",
            f"null_shuffled_experts|band|all|{headline}",
            f"null_clock|clock|all|{headline}",
        ):
            row = block["results"][key]
            stats = row["offsets"]
            print(
                f"{key:40s} {row['detection_rate']:7.1%} {row['success_fpr']:8.1%} "
                f"{stats.get('median', float('nan')):8.1f} "
                f"{stats.get('median_abs', float('nan')):9.1f} "
                f"{stats.get('within_2', float('nan')):6.1%} "
                f"{stats.get('within_5', float('nan')):6.1%}"
            )
    print(f"\nwrote {args.output_dir / 'summary.json'} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
