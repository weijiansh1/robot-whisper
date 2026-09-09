"""Exact save/restore of a LIBERO episode, including controller state.

``env.get_sim_state()`` returns only MuJoCo's flattened state (time, qpos, qvel,
act) and ``env.set_init_state()`` restores only that.  robosuite's
OperationalSpaceController keeps its own state -- the goal pose it is servoing
to, the orientation reference, and the two interpolators -- and none of it is in
the MuJoCo vector.  Restoring a snapshot therefore leaves the controller holding
whatever goal the *previous* rollout left behind, and the first step after the
restore differs accordingly.

Measured on LIBERO-Goal task 0 / init 24, branching after 70 env steps:

    same snapshot, different preceding action   ->  max|dstate| = 3.1e-03
    same snapshot, same preceding action        ->  max|dstate| = 0.0

which is why a naive branch test can look perfectly deterministic by accident:
if the replay tape happens to re-issue the same action that preceded the
snapshot, the controller state coincides and the divergence vanishes.

This module snapshots both halves so that every candidate branches from a
byte-identical starting point regardless of what ran before it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# OSC fields that carry across a set_init_state() and change the next command.
_CONTROLLER_FIELDS = (
    "goal_pos", "goal_ori", "ori_ref", "relative_ori",
    "current_lin_vel", "current_ang_vel",
)
_INTERPOLATORS = ("interpolator_pos", "interpolator_ori")
# Interpolator internals differ by class; copy whatever numeric state exists.
_INTERP_FIELDS = ("start", "goal", "step", "total_steps", "prev_goal",
                  "set_states_called", "ori_ref")


def _controllers(env) -> list[Any]:
    base = getattr(env, "env", env)
    return [r.controller for r in getattr(base, "robots", []) if getattr(r, "controller", None)]


def _grab(obj, fields) -> dict:
    out = {}
    for f in fields:
        if not hasattr(obj, f):
            continue
        v = getattr(obj, f)
        out[f] = np.array(v, copy=True) if isinstance(v, np.ndarray) else v
    return out


def _put(obj, state: dict) -> None:
    for f, v in state.items():
        setattr(obj, f, np.array(v, copy=True) if isinstance(v, np.ndarray) else v)


# robosuite keeps the episode clock and terminal flag as plain Python attributes
# (base.py:307-308, set in _post_action at base.py:432).  They survive a
# set_init_state(), so branching N candidates off one snapshot silently burns the
# horizon N times over and eventually raises
# "executing action in terminated episode".  Each branch must start from the same
# clock, so these belong in the snapshot too.
_ENV_FIELDS = ("timestep", "done", "cur_time")

# MuJoCo's constraint solver warm-starts from the previous step's acceleration.
# qacc_warmstart lives in mjData, not in mjSimState, so get_sim_state() does not
# carry it and a restored branch starts the solver from whatever the previous
# rollout left.  With no contacts the solver converges to the same answer either
# way; once the gripper touches something the warm start changes the solution and
# the branch diverges.  Measured across 40 probe points: 23 restored to <1e-8,
# but 12 were off by 1e-3..0.56, and those cluster on the contact phases.
_MJDATA_FIELDS = ("qacc_warmstart",)


def _mjdata(env):
    return _base_env(env).sim.data


def _base_env(env):
    return getattr(env, "env", env)


def save_full_state(env) -> dict:
    """MuJoCo state, every controller's internal state, and the episode clock."""
    ctrl = []
    for c in _controllers(env):
        entry = _grab(c, _CONTROLLER_FIELDS)
        for name in _INTERPOLATORS:
            interp = getattr(c, name, None)
            if interp is not None:
                entry[name] = _grab(interp, _INTERP_FIELDS)
        ctrl.append(entry)
    return {
        "sim": np.asarray(env.get_sim_state(), dtype=np.float64).copy(),
        "controllers": ctrl,
        "env": _grab(_base_env(env), _ENV_FIELDS),
        "mjdata": _grab(_mjdata(env), _MJDATA_FIELDS),
    }


def restore_full_state(env, snap: dict):
    """Restore MuJoCo state, then overwrite controller state.  Returns the obs.

    Order matters: set_init_state() re-runs the observable pipeline, and some
    controller fields are re-derived from the simulator during that call, so the
    controller must be written *after* it.
    """
    obs = env.set_init_state(snap["sim"])
    for c, entry in zip(_controllers(env), snap["controllers"]):
        for name in _INTERPOLATORS:
            if name in entry:
                interp = getattr(c, name, None)
                if interp is not None:
                    _put(interp, entry[name])
        _put(c, {k: v for k, v in entry.items() if k not in _INTERPOLATORS})
    _put(_base_env(env), snap.get("env", {}))
    for f, v in snap.get("mjdata", {}).items():
        getattr(_mjdata(env), f)[:] = v
    return obs


def state_delta(env, snap: dict) -> float:
    return float(np.abs(np.asarray(env.get_sim_state(), dtype=np.float64) - snap["sim"]).max())


SAFE_SNAPSHOT_CODEC = "himoe.full_state.safe.v1"


class SnapshotCodecError(ValueError):
    """The full-state value cannot be represented by the safe codec."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def encode_full_state(snapshot: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Encode a snapshot as JSON metadata plus non-object NumPy arrays.

    The returned payload is suitable for ``json.dump`` and ``np.savez`` with
    ``allow_pickle=False`` on read. Unknown values fail closed rather than being
    coerced with ``repr`` or Python pickle.
    """

    arrays: dict[str, np.ndarray] = {}

    def encode(value: Any, path: str) -> dict[str, Any]:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise SnapshotCodecError(f"object array is forbidden at {path}")
            if value.dtype.kind not in "biufcSU":
                raise SnapshotCodecError(
                    f"unsupported array dtype {value.dtype} at {path}"
                )
            key = f"array_{len(arrays):06d}"
            array = np.ascontiguousarray(value)
            arrays[key] = array
            return {
                "kind": "ndarray",
                "key": key,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": _array_sha256(array),
            }
        if isinstance(value, np.generic):
            if value.dtype.kind not in "biufU":
                raise SnapshotCodecError(
                    f"unsupported NumPy scalar dtype {value.dtype} at {path}"
                )
            return {
                "kind": "numpy_scalar",
                "dtype": str(value.dtype),
                "value": value.item(),
            }
        if value is None:
            return {"kind": "none"}
        if isinstance(value, bool):
            return {"kind": "bool", "value": value}
        if isinstance(value, int):
            return {"kind": "int", "value": value}
        if isinstance(value, float):
            if not np.isfinite(value):
                raise SnapshotCodecError(f"non-finite float at {path}")
            return {"kind": "float", "value": value}
        if isinstance(value, str):
            return {"kind": "str", "value": value}
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise SnapshotCodecError(f"mapping keys must be strings at {path}")
            return {
                "kind": "mapping",
                "items": [
                    [key, encode(value[key], f"{path}.{key}")]
                    for key in sorted(value)
                ],
            }
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "items": [encode(item, f"{path}[{index}]") for index, item in enumerate(value)],
            }
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return {
                "kind": "list",
                "items": [encode(item, f"{path}[{index}]") for index, item in enumerate(value)],
            }
        raise SnapshotCodecError(
            f"unsupported full-state value {type(value).__name__} at {path}"
        )

    root = encode(snapshot, "snapshot")
    metadata = {
        "codec": SAFE_SNAPSHOT_CODEC,
        "root": root,
        "array_keys": sorted(arrays),
    }
    return metadata, arrays


def decode_full_state(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Decode and fully validate a payload produced by :func:`encode_full_state`."""

    if metadata.get("codec") != SAFE_SNAPSHOT_CODEC:
        raise SnapshotCodecError(f"unsupported snapshot codec {metadata.get('codec')!r}")
    declared = metadata.get("array_keys")
    if not isinstance(declared, list) or not all(isinstance(key, str) for key in declared):
        raise SnapshotCodecError("snapshot array_keys must be a string list")
    if set(declared) != set(arrays):
        raise SnapshotCodecError("snapshot metadata and NPZ array keys disagree")
    consumed: set[str] = set()

    def decode(node: Any, path: str) -> Any:
        if not isinstance(node, Mapping) or not isinstance(node.get("kind"), str):
            raise SnapshotCodecError(f"malformed codec node at {path}")
        kind = node["kind"]
        if kind == "ndarray":
            key = node.get("key")
            if not isinstance(key, str) or key not in arrays or key in consumed:
                raise SnapshotCodecError(f"invalid or repeated array key at {path}")
            value = np.asarray(arrays[key])
            if value.dtype.hasobject:
                raise SnapshotCodecError(f"object array is forbidden at {path}")
            if str(value.dtype) != node.get("dtype") or list(value.shape) != node.get("shape"):
                raise SnapshotCodecError(f"array dtype/shape mismatch at {path}")
            if _array_sha256(value) != node.get("sha256"):
                raise SnapshotCodecError(f"array checksum mismatch at {path}")
            consumed.add(key)
            return value.copy()
        if kind == "numpy_scalar":
            try:
                dtype = np.dtype(node["dtype"])
                return dtype.type(node["value"])
            except (KeyError, TypeError, ValueError) as error:
                raise SnapshotCodecError(f"malformed NumPy scalar at {path}") from error
        if kind == "none":
            return None
        if kind == "bool" and isinstance(node.get("value"), bool):
            return node["value"]
        if kind == "int" and isinstance(node.get("value"), int) and not isinstance(node.get("value"), bool):
            return node["value"]
        if kind == "float" and isinstance(node.get("value"), (int, float)):
            value = float(node["value"])
            if np.isfinite(value):
                return value
        if kind == "str" and isinstance(node.get("value"), str):
            return node["value"]
        if kind in {"list", "tuple"} and isinstance(node.get("items"), list):
            values = [decode(item, f"{path}[{index}]") for index, item in enumerate(node["items"])]
            return tuple(values) if kind == "tuple" else values
        if kind == "mapping" and isinstance(node.get("items"), list):
            result: dict[str, Any] = {}
            for index, item in enumerate(node["items"]):
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or not isinstance(item[0], str)
                    or item[0] in result
                ):
                    raise SnapshotCodecError(f"malformed mapping item at {path}[{index}]")
                result[item[0]] = decode(item[1], f"{path}.{item[0]}")
            return result
        raise SnapshotCodecError(f"malformed or unsupported node kind {kind!r} at {path}")

    result = decode(metadata.get("root"), "snapshot")
    if not isinstance(result, dict):
        raise SnapshotCodecError("full-state root must decode to a mapping")
    if consumed != set(declared):
        raise SnapshotCodecError("snapshot payload contains unreferenced arrays")
    return result
