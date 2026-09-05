from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import zarr

from moe_grammar.features import extract_multitrack_features
from moe_grammar.open_world import HealthyPrefixGrammar, build_query_phenotypes


FORMAL_RUN = re.compile(r"^i\d+_s\d+$")


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _arm_sort_key(name: str) -> tuple[int, int]:
    if name == "triggered":
        return 0, 0
    if name.startswith("delayed_+"):
        return 1, int(name.split("+")[-1])
    if name == "control":
        return 2, 0
    return 3, 0


@dataclass(frozen=True)
class CandidateTrace:
    arm: str
    candidate: int
    path: Path
    episode_id: int
    success: bool
    fork_query: int
    budget: int
    route_ids: np.ndarray


@dataclass(frozen=True)
class RunTrace:
    cohort: str
    run_id: str
    path: Path
    init_state_id: int
    trunk_queries: int
    traces: tuple[CandidateTrace, ...]

    @property
    def arm_names(self) -> tuple[str, ...]:
        return tuple(sorted({trace.arm for trace in self.traces}, key=_arm_sort_key))

    def arm(self, name: str) -> tuple[CandidateTrace, ...]:
        return tuple(
            sorted(
                (trace for trace in self.traces if trace.arm == name),
                key=lambda trace: trace.candidate,
            )
        )


@dataclass
class RouteStore:
    path: Path
    group: Any
    episode_id: np.ndarray
    expert_ids: np.ndarray

    @classmethod
    def open(cls, path: Path) -> "RouteStore":
        group = zarr.open_group(str(path), mode="r")
        required = {"episode_id", "hb_expert_ids", "hb_router_probs"}
        missing = required.difference(group.array_keys())
        if missing:
            raise ValueError(f"{path}: missing route arrays {sorted(missing)}")
        return cls(
            path=path,
            group=group,
            episode_id=np.asarray(group["episode_id"], dtype=np.int32),
            expert_ids=np.asarray(group["hb_expert_ids"], dtype=np.uint8),
        )

    def response_ids(self, start: int, stop: int) -> np.ndarray:
        # The online response retained ten action tokens and put flow before layer.
        return self.expert_ids[start:stop, :, :, 1:, :].transpose(0, 2, 1, 3, 4)

    def response_ids_rows(self, rows: Sequence[int]) -> np.ndarray:
        selected = self.expert_ids[np.asarray(rows, dtype=np.int64), :, :, 1:, :]
        return selected.transpose(0, 2, 1, 3, 4)

    def probabilities(self, rows: slice | Sequence[int]) -> np.ndarray:
        array = self.group["hb_router_probs"]
        if isinstance(rows, slice):
            return np.asarray(array[rows], dtype=np.float32)
        return np.stack([np.asarray(array[int(row)], dtype=np.float32) for row in rows])


@dataclass(frozen=True)
class LocatedRun:
    run: RunTrace
    store: RouteStore
    trunk_start: int
    branch_rows: dict[tuple[str, int], int]
    duplicate_matches: int


@dataclass(frozen=True)
class CandidateSnapshot:
    cohort: str
    run_id: str
    init_state_id: int
    arm: str
    fork_query: int
    budget: int
    candidate_ids: np.ndarray
    success: np.ndarray
    actions: np.ndarray
    prefix_probability: np.ndarray
    prefix_expert_ids: np.ndarray
    candidate_probability: np.ndarray
    candidate_expert_ids: np.ndarray
    snapshot_state_sha256: str
    policy_state_sha256: str
    duplicate_route_matches: int

    @property
    def cluster(self) -> str:
        return f"{self.cohort}/{self.run_id}"


def discover_run_traces(cohort: str, root: Path) -> list[RunTrace]:
    output: list[RunTrace] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        run_path = manifest_path.parent
        if not FORMAL_RUN.fullmatch(run_path.name):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("trunk", {}).get("success"):
            continue
        config = json.loads((run_path / "experiment_config.json").read_text(encoding="utf-8"))
        traces: list[CandidateTrace] = []
        arm_paths = [path for path in run_path.iterdir() if path.is_dir()]
        for arm_path in sorted(arm_paths, key=lambda path: _arm_sort_key(path.name)):
            files = sorted(arm_path.glob("candidate_*.npz"))
            if not files:
                continue
            arm_traces: list[CandidateTrace] = []
            for path in files:
                record_path = path.with_suffix(".json")
                if not record_path.exists():
                    arm_traces = []
                    break
                record = json.loads(record_path.read_text(encoding="utf-8"))
                with np.load(path, allow_pickle=False) as payload:
                    if "routing_expert_ids" not in payload.files:
                        arm_traces = []
                        break
                    route_ids = np.asarray(payload["routing_expert_ids"], dtype=np.uint8)
                if route_ids.ndim != 5 or route_ids.shape[1:] != (10, 8, 10, 4):
                    raise ValueError(f"{path}: invalid online route shape {route_ids.shape}")
                arm_traces.append(
                    CandidateTrace(
                        arm=arm_path.name,
                        candidate=int(record["candidate"]),
                        path=path,
                        episode_id=int(record["episode_id"]),
                        success=bool(record["success"]),
                        fork_query=int(record["fork_query"]),
                        budget=int(record["fork_budget_queries"]),
                        route_ids=route_ids,
                    )
                )
            if arm_traces:
                expected = list(range(len(arm_traces)))
                observed = [trace.candidate for trace in arm_traces]
                if observed != expected:
                    raise ValueError(f"{arm_path}: non-contiguous candidate ids {observed}")
                traces.extend(arm_traces)
        if not traces:
            continue
        output.append(
            RunTrace(
                cohort=cohort,
                run_id=run_path.name,
                path=run_path,
                init_state_id=int(config["init_state_id"]),
                trunk_queries=int(manifest["trunk"]["queries"]),
                traces=tuple(traces),
            )
        )
    return output


class RouteArchive:
    def __init__(self, stores: Iterable[Path]) -> None:
        paths = sorted({Path(path).resolve() for path in stores})
        if not paths:
            raise ValueError("no route stores were found")
        self.stores = [RouteStore.open(path) for path in paths]

    @staticmethod
    def _trace_matches(
        store: RouteStore, trace: CandidateTrace
    ) -> list[tuple[int, int]]:
        """Return exact matches while allowing rows from concurrent episodes in between."""

        episode_rows = np.flatnonzero(store.episode_id == trace.episode_id)
        width = len(trace.route_ids)
        matches: list[tuple[int, int]] = []
        for offset in range(max(0, len(episode_rows) - width + 1)):
            rows = episode_rows[offset : offset + width]
            if np.array_equal(store.response_ids_rows(rows), trace.route_ids):
                matches.append((int(rows[0]), int(rows[-1])))
        return matches

    @classmethod
    def _match_collections(
        cls, store: RouteStore, run: RunTrace
    ) -> list[tuple[int, dict[tuple[str, int], int]]]:
        trace_matches = [cls._trace_matches(store, trace) for trace in run.traces]
        if any(not matches for matches in trace_matches):
            return []
        collections: list[tuple[int, dict[tuple[str, int], int], int]] = []
        first = run.traces[0]
        for first_start, first_end in trace_matches[0]:
            trunk_start = first_start - run.trunk_queries
            trunk_episode_id = first.episode_id - 1000
            if trunk_start < 0 or not np.all(
                store.episode_id[trunk_start:first_start] == trunk_episode_id
            ):
                continue
            rows = {(first.arm, first.candidate): first_start}
            previous_end = first_end
            complete = True
            for trace, candidates in zip(run.traces[1:], trace_matches[1:], strict=True):
                following = [item for item in candidates if item[0] > previous_end]
                if not following:
                    complete = False
                    break
                start, previous_end = min(following)
                rows[(trace.arm, trace.candidate)] = start
            if complete:
                collections.append((trunk_start, rows, previous_end - first_start))
        if not collections:
            return []
        minimum_span = min(item[2] for item in collections)
        return [(start, rows) for start, rows, span in collections if span == minimum_span]

    def locate(self, run: RunTrace) -> LocatedRun:
        first = run.traces[0]
        if first.arm != "triggered" or first.candidate != 0:
            raise ValueError(f"{run.path}: the collection does not start with triggered/candidate 0")
        matches: list[tuple[RouteStore, int, dict[tuple[str, int], int]]] = []
        for store in self.stores:
            for trunk_start, branch_rows in self._match_collections(store, run):
                matches.append((store, trunk_start, branch_rows))
        if not matches:
            raise ValueError(f"{run.path}: no exact candidate block found in the route archive")
        if len(matches) > 1:
            signatures = []
            sample_rows = [0, max(run.trunk_queries - 1, 0)]
            for store, trunk_start, branch_rows in matches:
                rows = [trunk_start + offset for offset in sample_rows]
                rows.extend(branch_rows.values())
                signatures.append(_sha256(store.probabilities(rows)))
            if len(set(signatures)) != 1:
                locations = [(str(item[0].path), item[1]) for item in matches]
                raise ValueError(f"{run.path}: ambiguous non-identical route blocks {locations}")
        store, trunk_start, branch_rows = matches[0]
        return LocatedRun(run, store, trunk_start, branch_rows, len(matches))


def _load_candidate_arrays(trace: CandidateTrace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(trace.path, allow_pickle=False) as payload:
        actions = np.asarray(payload["action_chunks"][0], dtype=np.float32)
        sim_state = np.asarray(payload["sim_state"][0])
        policy_state = np.asarray(payload["policy_state"][0])
    return actions, sim_state, policy_state


def materialize_snapshots(located: LocatedRun) -> list[CandidateSnapshot]:
    run = located.run
    output: list[CandidateSnapshot] = []
    for arm in run.arm_names:
        traces = run.arm(arm)
        if not traces:
            continue
        fork_queries = {trace.fork_query for trace in traces}
        budgets = {trace.budget for trace in traces}
        if len(fork_queries) != 1 or len(budgets) != 1:
            raise ValueError(f"{run.path / arm}: candidates do not share fork query and budget")
        fork_query = fork_queries.pop()
        if not 0 <= fork_query < run.trunk_queries:
            raise ValueError(f"{run.path / arm}: fork query is outside the captured trunk")
        actions, states, policy_states = [], [], []
        for trace in traces:
            action, state, policy_state = _load_candidate_arrays(trace)
            actions.append(action)
            states.append(state)
            policy_states.append(policy_state)
        state_array = np.stack(states)
        policy_array = np.stack(policy_states)
        if not np.array_equal(state_array, np.broadcast_to(state_array[0], state_array.shape)):
            raise ValueError(f"{run.path / arm}: candidates do not start at the same sim state")
        if not np.array_equal(policy_array, np.broadcast_to(policy_array[0], policy_array.shape)):
            raise ValueError(f"{run.path / arm}: candidates do not share the policy state")
        branch_rows = [located.branch_rows[(arm, trace.candidate)] for trace in traces]
        prefix = slice(located.trunk_start, located.trunk_start + fork_query)
        output.append(
            CandidateSnapshot(
                cohort=run.cohort,
                run_id=run.run_id,
                init_state_id=run.init_state_id,
                arm=arm,
                fork_query=fork_query,
                budget=budgets.pop(),
                candidate_ids=np.asarray([trace.candidate for trace in traces], dtype=np.int16),
                success=np.asarray([trace.success for trace in traces], dtype=np.bool_),
                actions=np.stack(actions),
                prefix_probability=located.store.probabilities(prefix),
                prefix_expert_ids=located.store.expert_ids[prefix],
                candidate_probability=located.store.probabilities(branch_rows),
                candidate_expert_ids=located.store.expert_ids[branch_rows],
                snapshot_state_sha256=_sha256(state_array[0]),
                policy_state_sha256=_sha256(policy_array[0]),
                duplicate_route_matches=located.duplicate_matches,
            )
        )
    return output


def extract_query_words(
    probability: np.ndarray,
    expert_ids: np.ndarray,
    feature_names: tuple[str, ...],
    device: str,
) -> np.ndarray:
    if len(probability) == 0:
        return np.empty((0, 22), dtype=np.float32)
    torch_device = torch.device(device)
    with torch.inference_mode():
        features = extract_multitrack_features(
            torch.as_tensor(probability, dtype=torch.float32, device=torch_device),
            torch.as_tensor(expert_ids, dtype=torch.int64, device=torch_device),
        ).cpu().numpy()
    return build_query_phenotypes(features, feature_names)


def filter_prefix(grammar: HealthyPrefixGrammar, values: np.ndarray) -> np.ndarray | None:
    belief = None
    for value in np.asarray(values):
        _, belief, _, _ = grammar.step(value, belief)
    return belief


def clock_prefix_belief(grammar: HealthyPrefixGrammar, length: int) -> np.ndarray | None:
    if length <= 0:
        return None
    belief = grammar.initial_.copy()
    for _ in range(1, length):
        belief = belief @ grammar.transition_
    return belief.astype(np.float32)


def pairwise_rms(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return np.sqrt(np.square(flat[:, None] - flat[None, :]).mean(axis=-1))


def standardize_within(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = values.std()
    return np.zeros_like(values) if scale < 1e-12 else (values - values.mean()) / scale


def selector_indexes(
    grammar_surprise: np.ndarray,
    actions: np.ndarray,
    routing_words: np.ndarray,
    *,
    diversity_beta: float = 0.25,
) -> dict[str, int]:
    surprise = np.asarray(grammar_surprise, dtype=np.float64)
    if surprise.ndim != 1 or len(surprise) != len(actions) or len(actions) != len(routing_words):
        raise ValueError("candidate score arrays do not align")
    action_distance = pairwise_rms(actions)
    route_distance = pairwise_rms(routing_words)
    action_novelty = action_distance.mean(axis=1)
    route_novelty = route_distance.mean(axis=1)
    healthy = -standardize_within(surprise)
    return {
        "action_medoid": int(np.argmin(action_distance.sum(axis=1))),
        "grammar": int(np.argmin(surprise)),
        "anti_grammar": int(np.argmax(surprise)),
        "grammar_action_diverse": int(
            np.argmax(healthy + diversity_beta * standardize_within(action_novelty))
        ),
        "grammar_route_diverse": int(
            np.argmax(healthy + diversity_beta * standardize_within(route_novelty))
        ),
    }


def stratified_pair_auc(score: np.ndarray, success: np.ndarray, group: np.ndarray) -> float:
    """Probability that a success is healthier than a failure in the same snapshot."""

    wins = 0.0
    comparisons = 0
    for value in np.unique(group):
        selected = group == value
        positive = np.asarray(score)[selected & success]
        negative = np.asarray(score)[selected & ~success]
        if not len(positive) or not len(negative):
            continue
        difference = negative[:, None] - positive[None, :]
        wins += float((difference > 0).sum() + 0.5 * (difference == 0).sum())
        comparisons += difference.size
    return float(wins / comparisons) if comparisons else float("nan")


def cluster_bootstrap_mean(
    values: np.ndarray,
    cluster: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    cluster = np.asarray(cluster)
    clusters = np.unique(cluster)
    if not len(clusters):
        return float("nan"), float("nan")
    grouped = [values[cluster == item] for item in clusters]
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        estimates[draw] = np.concatenate([grouped[index] for index in sampled]).mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)
