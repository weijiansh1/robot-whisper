"""Physical-time control and noise pairing; no changes inside the policy."""

from dataclasses import dataclass

import numpy as np

FLOW_STEPS = GENERATED = NORMAL = 10
SHORT = 5
WINDOW = 20
HORIZON = 520
REPLICATES = 4
SEED = 2026091508
ARMS = ('native10', 'window5')
ANCHORS = (
    ('plus-task00-init026', 18, 'pre_alarm'),
    ('plus-task06-init039', 13, 'pre_alarm'),
    ('plus-task03-init047', 21, 'early_alarm'),
    ('plus-task03-init026', 33, 'early_alarm'),
    ('pro-task03-init039', 17, 'static_proxy'),
    ('plus-task06-init026', 29, 'static_proxy'),
    ('plus-task03-init039', 22, 'source_success_slow'),
    ('plus-task00-init047', 12, 'source_success_progress'),
)
# Reserve whole task/init groups. No outcomes of new branches select this panel.
TIMING_PARENTS = ('plus-task01-init047', 'pro-task01-init039',
                  'plus-task02-init039', 'plus-task04-init047',
                  'pro-task07-init026', 'plus-task09-init047')


def physical_noise(parent, flow_seed, replicate, step):
    if replicate not in range(REPLICATES) or step not in range(0, HORIZON, SHORT):
        raise ValueError('Invalid paired physical-time index')
    if replicate == 0 and step % NORMAL == 0:
        rng = np.random.default_rng(flow_seed)
        for _ in range(step // NORMAL + 1):
            value = rng.standard_normal((GENERATED, 24)).astype(np.float32)
        return value
    identity = [SEED, 0 if parent['benchmark'] == 'plus' else 1,
                parent['base_task_id'], parent['init_state_id'], replicate, step]
    return np.random.default_rng(np.random.SeedSequence(identity)).standard_normal(
        (GENERATED, 24)).astype(np.float32)


@dataclass
class ExecutionWindow:
    """One bounded window; a latched alarm cannot silently retrigger it."""

    arm: str
    start: int

    def __post_init__(self):
        if self.arm not in ARMS or self.start not in range(0, HORIZON - WINDOW + 1, NORMAL):
            raise ValueError('Invalid arm or recovery start')

    def length(self, step):
        if step not in range(self.start, HORIZON, SHORT):
            raise ValueError('Invalid execution boundary')
        active = self.arm == 'window5' and step < self.start + WINDOW
        return min(SHORT if active else NORMAL, HORIZON - step)

    def prefix(self, actions, step):
        value = np.asarray(actions)
        if value.shape != (GENERATED, 7) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Expected unchanged ten-action float32 generation')
        # The returned copy is the entire queue. No old suffix survives the call.
        return value[:self.length(step)].copy()


def maximum_calls():
    return REPLICATES * sum(2 * (HORIZON // NORMAL - q) + 2 for _, q, _ in ANCHORS) + len(ANCHORS)


def group_key(parent):
    return (parent['base_task_id'], parent['init_state_id'])
