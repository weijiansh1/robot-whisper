from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from moe_grammar.features import step_feature_names

SCENE8_TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
DEFAULT_STASIS_LABELS = Path(
    "/home/jovyan/work/himoe-vla/VLA_MUI_HUB/moe-token-dynamics/results/early_structure_scores.npz"
)


@dataclass(frozen=True)
class Episode:
    index: int
    task_index: int
    task: str
    local_episode_id: int
    start: int
    stop: int
    init_state_id: int
    flow_noise_seed: int
    success: bool
    stasis_onset: int = -1

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class Corpus:
    features: np.ndarray
    flow_shuffled_features: np.ndarray | None
    behavior_features: np.ndarray
    feature_names: tuple[str, ...]
    behavior_feature_names: tuple[str, ...]
    query_task_index: np.ndarray
    query_episode_index: np.ndarray
    query_episode_step: np.ndarray
    episodes: list[Episode]
    tasks: tuple[str, ...]
    source_files: tuple[str, ...]

    def episode_rows(self, episode: Episode) -> slice:
        return slice(episode.start, episode.stop)


def _load_stasis_labels(path: Path | None) -> dict[tuple[int, int], tuple[bool, int]]:
    if path is None or not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as payload:
        state = np.asarray(payload["state"], dtype=np.int16)
        seed = np.asarray(payload["noise_seed"], dtype=np.int16)
        included = np.asarray(payload["included"], dtype=np.bool_)
        onset = np.asarray(payload["onset"], dtype=np.int16)
    if not (len(state) == len(seed) == len(included) == len(onset)):
        raise ValueError(f"invalid stasis label axes in {path}")
    return {
        (int(item_state), int(item_seed)): (bool(item_included), int(item_onset))
        for item_state, item_seed, item_included, item_onset in zip(state, seed, included, onset)
    }


def load_corpus(
    feature_dir: Path,
    stasis_labels: Path | None = DEFAULT_STASIS_LABELS,
    *,
    allow_task_segments: bool = False,
    require_flow_shuffled: bool = True,
    require_stasis_labels: bool = True,
    feature_dtype: type[np.floating] = np.float32,
) -> Corpus:
    paths = sorted(feature_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no extracted feature files in {feature_dir}")
    stasis = _load_stasis_labels(stasis_labels)

    feature_parts: list[np.ndarray] = []
    shuffled_parts: list[np.ndarray] = []
    behavior_parts: list[np.ndarray] = []
    task_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    episodes: list[Episode] = []
    tasks: list[str] = []
    expected_feature_names = step_feature_names()
    legacy_feature_names = (*expected_feature_names[:-1], "layer_disagreement")
    loaded_feature_names: tuple[str, ...] | None = None
    behavior_names: tuple[str, ...] | None = None
    task_to_index: dict[str, int] = {}
    seen_episode_keys: set[tuple[str, int, int]] = set()
    has_shuffled: bool | None = None
    query_offset = 0

    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            task = str(metadata.get("task_key", metadata["run_key"]))
            features = np.asarray(payload["features"], dtype=feature_dtype)
            current_has_shuffled = "flow_shuffled_features" in payload
            shuffled = (
                np.asarray(payload["flow_shuffled_features"], dtype=feature_dtype)
                if current_has_shuffled
                else None
            )
            behavior = np.asarray(payload["behavior_features"], dtype=np.float32)
            feature_names = tuple(str(item) for item in payload["feature_names"])
            local_episode = np.asarray(payload["episode_id"], dtype=np.int16)
            episode_step = np.asarray(payload["episode_step"], dtype=np.int16)
            success = np.asarray(payload["success"], dtype=np.bool_)
            state = np.asarray(payload["init_state_id"], dtype=np.int16)
            noise_seed = np.asarray(payload["flow_noise_seed"], dtype=np.int16)
            current_behavior_names = tuple(str(item) for item in payload["behavior_feature_names"])
        if feature_names not in (expected_feature_names, legacy_feature_names):
            raise ValueError(f"feature schema mismatch in {path}")
        if loaded_feature_names is None:
            loaded_feature_names = feature_names
        elif feature_names != loaded_feature_names:
            raise ValueError("cannot mix legacy and corrected routing feature schemas")
        if len(features) != len(local_episode):
            raise ValueError(f"query-axis mismatch in {path}")
        if shuffled is not None and features.shape != shuffled.shape:
            raise ValueError(f"flow-shuffled shape mismatch in {path}")
        if require_flow_shuffled and shuffled is None:
            raise ValueError(f"flow-shuffled control missing in {path}")
        if has_shuffled is None:
            has_shuffled = current_has_shuffled
        elif has_shuffled != current_has_shuffled:
            raise ValueError("cannot mix feature files with and without flow-shuffled controls")
        if behavior.shape[0] != len(features):
            raise ValueError(f"behavior-axis mismatch in {path}")
        if behavior_names is None:
            behavior_names = current_behavior_names
        elif behavior_names != current_behavior_names:
            raise ValueError(f"behavior schema mismatch in {path}")
        if task in task_to_index and not allow_task_segments:
            raise ValueError(f"duplicate task feature file for {task}")
        if task not in task_to_index:
            task_to_index[task] = len(tasks)
            tasks.append(task)
        task_index = task_to_index[task]

        unique_episode, starts = np.unique(local_episode, return_index=True)
        if not np.array_equal(unique_episode, np.arange(len(unique_episode))):
            raise ValueError(f"non-contiguous episode IDs in {path}")
        stops = np.r_[starts[1:], len(local_episode)]
        for local_id, local_start, local_stop in zip(unique_episode, starts, stops):
            selected = slice(local_start, local_stop)
            if not np.array_equal(
                episode_step[selected], np.arange(local_stop - local_start, dtype=np.int16)
            ):
                raise ValueError(f"episode-step mismatch for {task} episode {local_id}")
            for values, label in (
                (success[selected], "success"),
                (state[selected], "init_state_id"),
                (noise_seed[selected], "flow_noise_seed"),
            ):
                if np.any(values != values[0]):
                    raise ValueError(f"{label} changes within {task} episode {local_id}")
            onset = -1
            if task == SCENE8_TASK:
                key = (int(state[local_start]), int(noise_seed[local_start]))
                if require_stasis_labels and key not in stasis:
                    raise ValueError(f"missing scene8 stasis label for state/seed {key}")
                if key in stasis:
                    included, candidate_onset = stasis[key]
                    if included and not bool(success[local_start]):
                        onset = candidate_onset
            episode_key = (
                task,
                int(state[local_start]),
                int(noise_seed[local_start]),
            )
            if episode_key in seen_episode_keys:
                raise ValueError(f"duplicate task/state/seed episode {episode_key}")
            seen_episode_keys.add(episode_key)
            episodes.append(
                Episode(
                    index=len(episodes),
                    task_index=task_index,
                    task=task,
                    local_episode_id=int(local_id),
                    start=query_offset + int(local_start),
                    stop=query_offset + int(local_stop),
                    init_state_id=int(state[local_start]),
                    flow_noise_seed=int(noise_seed[local_start]),
                    success=bool(success[local_start]),
                    stasis_onset=onset,
                )
            )

        feature_parts.append(features)
        if shuffled is not None:
            shuffled_parts.append(shuffled)
        behavior_parts.append(behavior)
        task_parts.append(np.full(len(features), task_index, dtype=np.int16))
        episode_parts.append(
            np.concatenate(
                [
                    np.full(stop - start, len(episodes) - len(starts) + index, dtype=np.int16)
                    for index, (start, stop) in enumerate(zip(starts, stops))
                ]
            )
        )
        step_parts.append(episode_step)
        query_offset += len(features)

    corpus = Corpus(
        features=np.concatenate(feature_parts, axis=0),
        flow_shuffled_features=(
            np.concatenate(shuffled_parts, axis=0) if shuffled_parts else None
        ),
        behavior_features=np.concatenate(behavior_parts, axis=0),
        feature_names=loaded_feature_names or expected_feature_names,
        behavior_feature_names=behavior_names or (),
        query_task_index=np.concatenate(task_parts),
        query_episode_index=np.concatenate(episode_parts),
        query_episode_step=np.concatenate(step_parts),
        episodes=episodes,
        tasks=tuple(tasks),
        source_files=tuple(str(path) for path in paths),
    )
    if len(corpus.features) != query_offset:
        raise AssertionError("corpus query count mismatch")
    for episode in corpus.episodes:
        if np.any(corpus.query_episode_index[episode.start : episode.stop] != episode.index):
            raise AssertionError(f"global episode mapping failed for {episode.index}")
    return corpus


def make_state_folds(states: np.ndarray, folds: int, seed: int) -> list[dict[str, np.ndarray]]:
    unique = np.unique(np.asarray(states, dtype=np.int16))
    if folds < 3 or folds > len(unique) // 2:
        raise ValueError("fold count must leave disjoint train/calibration/test state groups")
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    groups = [np.sort(group) for group in np.array_split(shuffled, folds)]
    result = []
    for fold in range(folds):
        test = groups[fold]
        calibration = groups[(fold + 1) % folds]
        train = np.asarray(
            sorted(set(unique.tolist()) - set(test.tolist()) - set(calibration.tolist())),
            dtype=np.int16,
        )
        result.append({"train": train, "calibration": calibration, "test": test})
    return result


def corpus_summary(corpus: Corpus) -> dict[str, object]:
    return {
        "queries": int(len(corpus.features)),
        "episodes": int(len(corpus.episodes)),
        "successes": int(sum(episode.success for episode in corpus.episodes)),
        "failures": int(sum(not episode.success for episode in corpus.episodes)),
        "stasis_labeled_failures": int(
            sum(episode.stasis_onset >= 0 for episode in corpus.episodes)
        ),
        "states": sorted({episode.init_state_id for episode in corpus.episodes}),
        "tasks": {
            task: {
                "episodes": int(sum(episode.task == task for episode in corpus.episodes)),
                "successes": int(
                    sum(episode.task == task and episode.success for episode in corpus.episodes)
                ),
                "queries": int(
                    sum(episode.length for episode in corpus.episodes if episode.task == task)
                ),
            }
            for task in corpus.tasks
        },
        "feature_shape": list(corpus.features.shape[1:]),
        "feature_names": list(corpus.feature_names),
        "source_files": list(corpus.source_files),
    }
