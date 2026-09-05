"""Data joins and matched event evaluation for assurance experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from profile_schema import AssuranceProfile


BUNDLE = Path(__file__).resolve().parent.parent
PROJECT = BUNDLE.parent
PROFILE_ROOT = BUNDLE / "results/profiles"
EVENT_ROOT = PROJECT / "analysis_moe_phenotype/events"
LOOP_BLIND_TASKS = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "open_the_middle_drawer_of_the_cabinet",
}


@dataclass(frozen=True)
class TaskData:
    corpus: str
    suite: str
    task: str
    profile_path: Path
    event_path: Path
    profile: AssuranceProfile
    events: tuple[dict[str, object], ...]

    @property
    def event_by_episode(self) -> dict[int, dict[str, object]]:
        return {int(row["episode_id"]): row for row in self.events}


def profile_inventory() -> Iterator[tuple[str, str, str, Path, Path]]:
    for path in sorted(PROFILE_ROOT.glob("*/*/*.npz")):
        corpus, suite = path.parts[-3:-1]
        task = path.stem
        event = EVENT_ROOT / corpus / suite / task / "events.csv"
        if not event.exists():
            raise FileNotFoundError(event)
        yield corpus, suite, task, path, event


def _parse_event(row: dict[str, str]) -> dict[str, object]:
    integer = {
        "episode_id",
        "scene",
        "repeat",
        "success",
        "n_queries",
        "loop_onset_q",
        "static_onset_q",
        "trap_onset_q",
    }
    return {key: int(value) if key in integer else value for key, value in row.items()}


def load_task(path_tuple: tuple[str, str, str, Path, Path]) -> TaskData:
    corpus, suite, task, profile_path, event_path = path_tuple
    with event_path.open(newline="", encoding="utf-8") as handle:
        events = tuple(_parse_event(row) for row in csv.DictReader(handle))
    profile = AssuranceProfile.load(profile_path)
    profile_episodes = set(map(int, np.unique(profile.episode_id)))
    event_episodes = {int(row["episode_id"]) for row in events}
    if profile_episodes != event_episodes:
        raise ValueError(f"profile/event episode mismatch for {profile_path}")
    return TaskData(corpus, suite, task, profile_path, event_path, profile, events)


def row_lookup(profile: AssuranceProfile) -> dict[tuple[int, int], int]:
    return {
        (int(episode), int(query)): index
        for index, (episode, query) in enumerate(zip(profile.episode_id, profile.query))
    }


def matched_onset_auc(
    task_data: list[TaskData],
    task_scores: dict[tuple[str, str, str], np.ndarray],
    event_type: str,
    lead: int,
) -> dict[str, float | int]:
    """Compare every event query only with same-task/scene/query no-event controls."""

    if event_type not in {"loop", "static"}:
        raise ValueError(event_type)
    onset_key = f"{event_type}_onset_q"
    wins = 0.0
    pairs = 0
    event_count = 0
    matched_controls: set[tuple[str, str, str, int]] = set()
    groups: set[tuple[str, str, str, int, int]] = set()
    for data in task_data:
        if event_type == "loop" and data.task in LOOP_BLIND_TASKS:
            continue
        key = (data.corpus, data.suite, data.task)
        score = np.asarray(task_scores[key], dtype=np.float64)
        lookup = row_lookup(data.profile)
        controls: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for event in data.events:
            if int(event["loop_onset_q"]) >= 0 or int(event["static_onset_q"]) >= 0:
                continue
            episode = int(event["episode_id"])
            scene = int(event["scene"])
            for query in data.profile.query[data.profile.episode_id == episode]:
                index = lookup[(episode, int(query))]
                value = score[index]
                if np.isfinite(value):
                    controls.setdefault((scene, int(query)), []).append((episode, float(value)))

        for event in data.events:
            onset = int(event[onset_key])
            query = onset + lead
            episode = int(event["episode_id"])
            scene = int(event["scene"])
            index = lookup.get((episode, query))
            negative = controls.get((scene, query), ())
            if index is None or not negative or not np.isfinite(score[index]):
                continue
            positive = float(score[index])
            values = np.asarray([value for _, value in negative], dtype=np.float64)
            wins += float(np.count_nonzero(positive > values))
            wins += 0.5 * float(np.count_nonzero(positive == values))
            pairs += len(values)
            event_count += 1
            groups.add((data.corpus, data.suite, data.task, scene, query))
            matched_controls.update((data.corpus, data.suite, data.task, item[0]) for item in negative)
    auc = wins / pairs if pairs else float("nan")
    return {
        "auc_event_high": auc,
        "auc_detection": max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan"),
        "events": event_count,
        "control_episodes": len(matched_controls),
        "pairs": pairs,
        "matched_groups": len(groups),
    }


def episode_rows(profile: AssuranceProfile, episode: int) -> np.ndarray:
    return np.flatnonzero(profile.episode_id == episode)
