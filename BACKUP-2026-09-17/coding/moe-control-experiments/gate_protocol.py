"""Frozen balanced gate directions and intervention scopes for P1b."""

import numpy as np

SEED = 2026091423
TASKS = (0, 3, 6, 9)
QUERY = 8
AMPLITUDES = (.05, .10)
DIRECTIONS = (0, 1)
SHAPE = (8, 10, 11, 32)
SCOPES = tuple("%s_%s_%s" % (layer, token, phase)
               for layer in ("front", "back")
               for token in ("state", "action")
               for phase in ("early", "late"))


def scope_mask(scope):
    if scope not in SCOPES:
        raise ValueError("Unknown gate scope")
    layer, token, phase = scope.split("_")
    mask = np.zeros(SHAPE[:-1], bool)
    layers = slice(0, 4) if layer == "front" else slice(4, 8)
    tokens = slice(0, 1) if token == "state" else slice(1, 11)
    steps = slice(0, 3) if phase == "early" else slice(7, 10)
    mask[layers, steps, tokens] = True
    return mask


def make_gate_bias(scope, direction, amplitude, sign):
    if direction not in DIRECTIONS or amplitude not in AMPLITUDES or sign not in (-1, 1):
        raise ValueError("Unregistered gate probe")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, direction]))
    rank = np.argsort(np.argsort(rng.random(SHAPE), axis=-1), axis=-1)
    balanced = np.where(rank < 16, -1., 1.).astype(np.float32)
    return balanced * np.float32(amplitude * sign) * scope_mask(scope)[..., None]


def probes():
    return [dict(scope=scope, direction=direction, amplitude=amplitude, sign=sign)
            for scope in SCOPES for direction in DIRECTIONS
            for amplitude in AMPLITUDES for sign in (-1, 1)]


def normalized_rms(actions, baseline, action_std):
    delta = (np.asarray(actions, float)[:, :6] - np.asarray(baseline, float)[:, :6]) / np.asarray(action_std)[:6]
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite action difference")
    return float(np.sqrt(np.mean(delta ** 2)))
