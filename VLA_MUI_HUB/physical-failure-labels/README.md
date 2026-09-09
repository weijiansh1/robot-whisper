# Physical failure labels

This analysis attaches simulator-backed outcome and failure-mode labels to the
completed LIBERO runs in `VLA_MUI_HUB/cache` and `VLA_MUI_HUB/cache_new`.
Source rollout files are never modified.

For every failed episode, `annotate.py` restores all recorded MuJoCo control
states in the pinned LIBERO/robosuite environment and evaluates:

- the original atomic BDDL goal predicates;
- target motion and height change;
- end-effector distance, gripper contact, and two-finger stable grasp;
- release after grasp and height loss;
- articulated fixture joint motion;
- motion of non-goal movable objects.

The inferred `primary_failure_reason` is kept separate from the physical
evidence and carries a confidence level. It is a failure-mode label, not proof
of the policy's internal causal mechanism.

Run the full analysis with:

```bash
./VLA_MUI_HUB/physical-failure-labels/run.sh
```

The wrapper selects the pinned Python 3.8 environment and the same LIBERO,
robosuite 1.4.1 overlay, MuJoCo, and bridge source used during collection. It
uses CPU OSMesa and does not load a VLA checkpoint. Use
`--dense-audit-per-task 0` to skip the default one-episode-per-task exact action
replay. Each task's dense replay resets the task simulator after failure-state
inspection and before executing the recorded action sequence. Its audit record
separates full-state, position (`qpos`), velocity (`qvel`), auxiliary-state, and
recorded robot-state errors. The pass criterion is matching action count,
outcome, `qpos`, and robot state; velocity error is retained as a diagnostic
because MuJoCo contact trajectories need not remain bit-reproducible from
float32 actions. Use `--strict-dense-audit` when replay divergence should make
the command exit nonzero.

Outputs are written atomically under `results/`:

- `episodes.csv`: compact index for all episodes;
- `episodes.jsonl`: all outcomes, with full evidence on failed episodes;
- `failures.jsonl`: failed episodes only;
- `dense_replay_audit.jsonl`: deterministic action-replay checks;
- `summary.json`: counts, thresholds, validation results, and limitations;
- `report.zh.md`: Chinese human-readable report.

The source `sim_state` is recorded every 10 actions before each action chunk.
Consequently, the final chunk of a failed episode is not present as a saved
terminal state. This is represented explicitly by
`physics_validation.unobserved_terminal_tail_actions`. Short contact or grasp
events between checkpoints can be missed, so negative contact/grasp evidence is
never assigned high confidence. The dense replay audit attempts an action-only
reproduction for a task-stratified subset and records any configuration,
velocity, robot-state, or outcome divergence. Failure-mode labels always use
the original saved trajectory states, not a divergent replay.

CALVIN is deliberately excluded: the existing partial probe does not have the
same per-episode state contract and cannot be evaluated by LIBERO predicates.
