from __future__ import annotations

import json

import numpy as np

from capture_behavior_forks import (
    EPISODE_ID_KEY,
    QueryLedger,
    _CONTINUATION_NOISE_ROLE,
    _noise_from_words,
    _noise_sha256,
    _physical_point,
    _recover_snapshot_commits,
    _seed_words,
    _sha256_file,
)
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
)


class FakeClient:
    def __init__(self, route_capture: bool) -> None:
        self.metadata = {"episode_id_key": EPISODE_ID_KEY} if route_capture else {}
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        return {
            ACTION_KEY: np.zeros((10, 7), dtype=np.float32),
            FLOW_NOISE_SHA256_KEY: _noise_sha256(request[FLOW_NOISE_KEY]),
        }


class _FakeRobotModel:
    eef_name = "eef_body"


class _FakeRobot:
    eef_site_id = 1
    robot_model = _FakeRobotModel()
    _ref_gripper_joint_pos_indexes = np.asarray([2, 4], dtype=np.int64)


class _FakeData:
    ncon = 0
    site_xpos = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    qpos = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0])

    @staticmethod
    def get_body_xquat(name):
        assert name == "eef_body"
        return np.asarray([0.5, 0.1, 0.2, 0.3])  # MuJoCo wxyz


class _FakeEnvironment:
    def __init__(self) -> None:
        sim = type("FakeSim", (), {"data": _FakeData(), "model": object()})()
        self.env = type("FakeBase", (), {"sim": sim, "robots": [_FakeRobot()]})()

    @staticmethod
    def get_sim_state():
        return np.asarray([7.0, 8.0, 9.0])

    @staticmethod
    def check_success():
        return True


def test_phase1_query_ledger_does_not_require_or_send_route_annotation() -> None:
    client = FakeClient(route_capture=False)
    ledger = QueryLedger(
        client,
        trace_row_offset=100,
        query_id_base=7,
        capture_routes=False,
    )
    words = _seed_words(1, _CONTINUATION_NOISE_ROLE, 2, 3, 4, [5, 6])
    response, trace_row, query_id = ledger.infer(
        {"observation/state": np.zeros(8)},
        _noise_from_words(words),
        "continuation",
        {"candidate": 9},
        words,
    )

    assert response[ACTION_KEY].shape == (10, 7)
    assert trace_row == -1
    assert query_id == 7
    assert EPISODE_ID_KEY not in client.requests[0]
    assert ledger.records[0]["server_episode_id"] is None


def test_route_query_ledger_sends_id_and_assigns_contiguous_trace_row() -> None:
    client = FakeClient(route_capture=True)
    ledger = QueryLedger(
        client,
        trace_row_offset=100,
        query_id_base=7,
        capture_routes=True,
    )
    words = _seed_words(1, _CONTINUATION_NOISE_ROLE, 2, 3, 4, [5, 6])
    _, trace_row, query_id = ledger.infer(
        {"observation/state": np.zeros(8)},
        _noise_from_words(words),
        "candidate",
        {"candidate": 0},
        words,
    )

    assert trace_row == 100
    assert query_id == 7
    assert client.requests[0][EPISODE_ID_KEY] == 7
    assert ledger.records[0]["server_episode_id"] == 7


def test_continuation_seed_key_has_no_candidate_coordinate() -> None:
    first = _seed_words(11, _CONTINUATION_NOISE_ROLE, 2, 3, 4, [5, 6])
    repeated = _seed_words(11, _CONTINUATION_NOISE_ROLE, 2, 3, 4, [5, 6])
    another_repeat = _seed_words(11, _CONTINUATION_NOISE_ROLE, 2, 3, 4, [7, 6])

    np.testing.assert_array_equal(_noise_from_words(first), _noise_from_words(repeated))
    assert not np.array_equal(_noise_from_words(first), _noise_from_words(another_repeat))


def test_physical_point_reads_synchronous_simulator_signals() -> None:
    environment = _FakeEnvironment()
    stale_observation = {
        "robot0_eef_pos": np.asarray([-1.0, -1.0, -1.0]),
        "robot0_eef_quat": np.asarray([-1.0, -1.0, -1.0, -1.0]),
        "robot0_gripper_qpos": np.asarray([-1.0, -1.0]),
    }

    sim, position, quaternion, gripper, success, contacts = _physical_point(
        environment, stale_observation
    )

    np.testing.assert_array_equal(sim, [7.0, 8.0, 9.0])
    np.testing.assert_array_equal(position, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(quaternion, [0.1, 0.2, 0.3, 0.5])
    np.testing.assert_array_equal(gripper, [12.0, 14.0])
    assert success
    assert contacts == set()


def test_resume_adopts_snapshot_journal_before_recollection(tmp_path) -> None:
    npz_path = tmp_path / "snapshot_0000_ep0000_t0004.npz"
    layout_path = tmp_path / "snapshot_0000_ep0000_t0004.layout.json"
    npz_path.write_bytes(b"complete atomic npz placeholder")
    layout_path.write_text(json.dumps({"arrays": {"actions": {"shape": [2, 10, 7]}}}))
    record = {
        "snapshot_index": 0,
        "episode": 0,
        "fork_step": 4,
        "task_id": 0,
        "init_state_id": 24,
        "task_name": "synthetic",
        "status": "complete",
        "npz_file": npz_path.name,
        "npz_sha256": _sha256_file(npz_path),
        "layout_file": layout_path.name,
        "layout_sha256": _sha256_file(layout_path),
    }
    records = {"snapshots": [record], "queries": []}
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records))
    manifest = {"artifacts": [], "query_count": 0}

    changed = _recover_snapshot_commits(
        tmp_path,
        manifest,
        records,
        [{"snapshot_index": 0, "episode": 0, "fork_step": 4}],
    )

    assert changed
    assert manifest["artifacts"][0]["snapshot_index"] == 0
    assert manifest["artifacts"][0]["arrays"]["actions"]["shape"] == [2, 10, 7]
    assert npz_path.exists() and layout_path.exists()
