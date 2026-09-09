"""Regression tests for per-episode labelling of server-side routing captures.

The server writes one flat zarr with no episode boundaries.  Before this, the
only way back to per-episode slices was accumulating ``inference_calls`` from
summaries.json, which cannot detect a dropped chunk -- it just shifts every
later episode.  The client now stamps ``episode_id`` on each request, but only
when the server advertises that it consumes the key: any other server hands the
observation straight to the policy, which does not expect an extra field.
"""

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

rwr = pytest.importorskip("rollout_with_routes", reason="needs the LIBERO bridge env")

from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY,
    ROUTING_CAPTURE_KEY,
    ROUTING_EXPERT_IDS_KEY,
    ROUTING_EXPERT_WEIGHTS_KEY,
    ROUTING_LAYER_INDICES_KEY,
)


# --------------------------------------------------------------------------- #
# advertisement gate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("metadata,expected", [
    ({"episode_id_key": "episode_id"}, True),
    ({"episode_id_key": "ep"}, False),
    ({"route_recorder": "himoe_router_recorder"}, False),  # recorder but no advert
    ({}, False),
    (None, False),
])
def test_should_send_episode_id(metadata, expected):
    assert rwr.should_send_episode_id(metadata) is expected


# --------------------------------------------------------------------------- #
# request stamping
# --------------------------------------------------------------------------- #

class _Env:
    """Just enough LIBERO environment to drive run_one."""

    def __init__(self, succeed_after):
        self.succeed_after = succeed_after
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return {"obs": self.steps}, 0.0, False, {}

    def check_success(self):
        return self.steps >= self.succeed_after

    def get_sim_state(self):
        return np.zeros(79, np.float32)

    def close(self):
        pass


class _Client:
    def __init__(self, chunk_steps=10):
        self.requests = []
        self.chunk_steps = chunk_steps

    def infer(self, request):
        self.requests.append(dict(request))
        response = {ACTION_KEY: np.zeros((self.chunk_steps, 7), np.float32)}
        if request.get(ROUTING_CAPTURE_KEY):
            response[ROUTING_EXPERT_IDS_KEY] = np.zeros((8, 10, 11, 4), np.uint8)
            response[ROUTING_EXPERT_WEIGHTS_KEY] = np.zeros((8, 10, 11, 4), np.float16)
            response[ROUTING_LAYER_INDICES_KEY] = np.arange(8, dtype=np.int16)
        return response


class _Config:
    task_suite = "libero_goal"
    task_id = 0
    init_state_id = 0
    seed = 7
    settle_steps = 2
    max_steps = 30
    replan_steps = 10


@pytest.fixture
def patched(monkeypatch):
    env = _Env(succeed_after=25)
    monkeypatch.setattr(rwr, "_load_task",
                        lambda config: (env, {"obs": 0}, type("T", (), {"name": "t"})(), "do it"))
    monkeypatch.setattr(rwr, "sim_joint_layout", lambda environment: {"state_dim": 79})
    monkeypatch.setattr(rwr, "build_policy_observation",
                        lambda observation, prompt: {"observation/state": np.zeros(8, np.float32)})
    monkeypatch.setattr(rwr, "validate_action_response", lambda response: response)
    return env


def test_episode_id_is_stamped_on_every_request(patched):
    client = _Client()
    summary, *_ = rwr.run_one(_Config(), client, flow_noise_seed=1000,
                              no_capture=True, episode_id=7)
    assert client.requests, "no inference happened"
    assert all(request[rwr.EPISODE_ID_KEY] == 7 for request in client.requests)
    assert summary["inference_calls"] == len(client.requests)


def test_episode_id_absent_when_not_requested(patched):
    client = _Client()
    rwr.run_one(_Config(), client, flow_noise_seed=1000, no_capture=True, episode_id=None)
    assert client.requests
    assert all(rwr.EPISODE_ID_KEY not in request for request in client.requests), \
        "an unadvertised key would reach the policy on a non-recorder server"


def test_episode_id_zero_is_still_sent(patched):
    """Episode 0 is falsy; a truthiness check here would drop the first episode."""
    client = _Client()
    rwr.run_one(_Config(), client, flow_noise_seed=1000, no_capture=True, episode_id=0)
    assert all(request[rwr.EPISODE_ID_KEY] == 0 for request in client.requests)


def test_stamping_does_not_disturb_the_rest_of_the_request(patched):
    client = _Client()
    rwr.run_one(_Config(), client, flow_noise_seed=1000, no_capture=False, episode_id=3)
    first = client.requests[0]
    assert first[ROUTING_CAPTURE_KEY] is True
    assert "observation/state" in first
    assert first[rwr.FLOW_NOISE_KEY].shape == rwr.FLOW_NOISE_SHAPE
