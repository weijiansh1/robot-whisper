# Failure timeout-extension protocol

Status: frozen before any continuation rollout is executed.

## Question

Some offline failures may be successful executions censored by the original
suite horizon. Continue every failure used by the current online-alarm study and
identify trajectories that succeed with ten additional VLA control queries.

The experiment never overwrites source trajectories or labels. It creates new
labels with explicit `original_failure`, `late_success_plus10_queries`, and
`failure` fields.

## Cohorts

- `development_main`: the exact 14,800-row, 37-task cohort in
  `results/hub_binary_audit/episode_physical_labels.csv`; 487 failures from
  `right-50x8-20260903`.
- `external_8b`: the exact 15,600-row, 39-task sealed cohort; 564 failures from
  `right-50x8b-20260903`.
- Total continuations: 1,051. Every source failure must have reached its suite
  horizon and have a complete trajectory NPZ.

## Meaning of ten control steps

In the routing corpus, one `control_step` is one VLA query and one action chunk.
Each chunk contains ten low-level LIBERO actions. The primary extension is ten
additional VLA queries, or at most 100 low-level actions. Success is checked
after every low-level action, so the output also reports the stricter result for
only one extra query / ten low-level actions.

## Continuation construction

The original model need not be rerun over the saved prefix. For each failure:

1. Recreate the same task, initial state, environment seed, settle steps, render
   size, and wrist layout.
2. Replay every saved action from the source NPZ.
3. Before every source query, require the replayed float32 simulator state and
   policy state to match the saved arrays exactly. Require the source prefix to
   remain unsuccessful through the original horizon.
4. Recreate the per-episode NumPy RNG from `flow_noise_seed`, consume exactly
   the number of source-query draws, and use subsequent draws for continuation.
5. Query the unchanged HiMoE checkpoint and execute up to ten additional
   action chunks, stopping at the first successful low-level action.

Any source hash mismatch, prefix replay mismatch, premature success, server
identity mismatch, or runtime error is an incomplete case and is never relabeled.

## Outputs

For every case save the source identity and hash, checkpoint/server identity,
exact-prefix audit, continuation noise/action hashes, first successful extra
query and action, and compact continuation arrays. The runner is resumable and
skips only cases with a completed result whose source identity still matches.

After all cases complete, produce:

- `late_successes.csv`;
- `remaining_failures.csv`;
- `development_main_clean_labels.csv`;
- `external_8b_clean_labels.csv`;
- success curves for extra queries 1 through 10 and summaries by suite/task.

A trajectory is removed from the failure set iff it was an original failure and
the exact-prefix continuation succeeds within the frozen ten-query extension.

## Interpretation boundary

This identifies horizon-censored late successes under continued closed-loop
execution on GPU 5. It does not prove that every remaining failure is a semantic
Trap, and it does not retrospectively make information after the original
horizon available to an online detector.
