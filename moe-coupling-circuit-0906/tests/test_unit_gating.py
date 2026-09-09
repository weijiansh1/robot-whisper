"""Regression tests for which conditions the physical gates apply to.

The first full run admitted 0 `post` anchors. The cause was that `light`
renders the *unedited* state, so at q_post it inherits the gripper's contacts
with the carried object, and the gate rejected it as "edited target touches the
robot". These tests pin the distinction with a fake scene, so the rule cannot
regress without a red test.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from build_coupling_counterfactuals import build_unit  # noqa: E402


LO = 10
VELOCITY = (58, 64)
STATE_DIM = 92


class FakeScene:
    """Contacts follow the target's height; images follow the target and the light.

    In-gripper (the carried object) reports robot contacts and no support, which
    is the situation that exposed the bug. On the table it reports support and
    no robot contact.
    """

    RESTING_HEIGHT = 0.90
    CARRIED_HEIGHT = 1.10

    def __init__(self, policy_state: np.ndarray | None = None) -> None:
        self.light_on = False
        self.current: np.ndarray | None = None
        self.light_calls = 0
        # The restoration gates compare this against the trace's saved row, so a
        # fake that always returned zeros would fail them for the wrong reason.
        self.policy_state = (
            np.zeros(8, np.float32) if policy_state is None else np.asarray(policy_state, np.float32)
        )

    def probe(self, sim_state):
        self.current = np.asarray(sim_state, np.float64)
        return self.contact_report()

    def restore(self, sim_state, prompt):
        self.current = np.asarray(sim_state, np.float64)
        target = self.current[LO : LO + 3]
        image = np.zeros((224, 224, 3), np.uint8)
        # Paint a patch whose position encodes the target and whose value
        # encodes the light, so both kinds of edit clear the visibility floor.
        column = int(abs(target[0]) * 1000) % 100
        image[:, column : column + 60, 0] = 200 if not self.light_on else 60
        return {
            "observation/image": image,
            "observation/wrist_image": image.copy(),
            "observation/state": self.policy_state.copy(),
        }

    def restored_state(self):
        return np.array(self.current, np.float64, copy=True)

    def contact_report(self):
        carried = abs(float(self.current[LO + 2]) - self.CARRIED_HEIGHT) < 1e-6
        return {
            "robot_contacts": 10 if carried else 0,
            "robot_min_dist": -5e-4 if carried else None,
            "support_contacts": 0 if carried else 4,
        }

    def perturb_light(self):
        self.light_on = True
        self.light_calls += 1

    def reset_light(self):
        self.light_on = False


def _episode(carried: bool):
    """A two-query trace: query 0 resting, the anchor query as asked."""
    sim = np.zeros((8, STATE_DIM), np.float64)
    sim[:, LO] = 0.30
    sim[:, LO + 2] = FakeScene.RESTING_HEIGHT
    sim[:, LO + 6] = 1.0  # unit quaternion
    if carried:
        sim[4:, LO] = 0.45
        sim[4:, LO + 2] = FakeScene.CARRIED_HEIGHT
    state = np.zeros((8, 8), np.float32)
    state[:, 0] = 0.20
    state[:, 2] = 0.95
    actions = np.zeros((8, 10, 7), np.float32)
    return state, sim, actions


def _summary():
    return {"episode_index": 3, "flow_noise_seed": 1000, "init_state_id": 0}


def test_post_anchor_is_admitted_even_though_held_touches_the_robot(tmp_path):
    """The bug: `light` was gated as if it had moved the object."""
    state, sim, actions = _episode(carried=True)
    episode_path = tmp_path / "episode_03.npz"
    np.savez(episode_path, state=state, sim_state=sim, actions=actions)
    scene = FakeScene(state[5])
    unit = build_unit(
        scene=scene,
        prompt="pick it up",
        anchor="post",
        query=5,
        state=state,
        sim=sim,
        actions=actions,
        lo=LO,
        velocity=VELOCITY,
        summary=_summary(),
        episode_path=episode_path,
    )
    assert unit["ok"], unit.get("reason")
    assert {row["condition"] for row in unit["conditions"]} == {"held", "uncoupled", "light"}
    assert scene.light_calls == 1


def test_light_is_rendered_with_the_light_moved_and_then_restored(tmp_path):
    state, sim, actions = _episode(carried=True)
    episode_path = tmp_path / "episode_03.npz"
    np.savez(episode_path, state=state, sim_state=sim, actions=actions)
    scene = FakeScene(state[5])
    build_unit(
        scene=scene,
        prompt="pick it up",
        anchor="post",
        query=5,
        state=state,
        sim=sim,
        actions=actions,
        lo=LO,
        velocity=VELOCITY,
        summary=_summary(),
        episode_path=episode_path,
    )
    assert scene.light_on is False, "the light must be put back after rendering"


def test_an_uncoupled_target_that_lands_on_the_robot_is_still_rejected(tmp_path):
    """The gate must still fire for a condition that really does move the object."""
    state, sim, actions = _episode(carried=True)
    # Query 0's "resting" pose is itself in the gripper, so writing it back
    # reproduces the carried state: robot contact, no support.
    sim[0, LO + 2] = FakeScene.CARRIED_HEIGHT
    episode_path = tmp_path / "episode_03.npz"
    np.savez(episode_path, state=state, sim_state=sim, actions=actions)
    unit = build_unit(
        scene=FakeScene(state[5]),
        prompt="pick it up",
        anchor="post",
        query=5,
        state=state,
        sim=sim,
        actions=actions,
        lo=LO,
        velocity=VELOCITY,
        summary=_summary(),
        episode_path=episode_path,
    )
    assert not unit["ok"]
    assert unit["reason"] == "edited target touches the robot"
    assert unit["detail_condition"] == "uncoupled"
