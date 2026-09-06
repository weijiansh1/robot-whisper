# Online Progress Ratio Guard v12

## Contract

The runtime monitor receives only the current `hb_router_probs` tensor with
shape `[8, 10, 11, 32]`. It has no task or suite identity, task prototype,
per-task threshold, outcome, horizon, simulator state, or future query.

One global profile is calibrated from an unlabeled routing reference corpus.
The profile holds exactly five scalars: `window`, `confirmations`,
`ratio_threshold`, `eps_length`, `layer_group`. It contains no task-indexed
array.

## Internal mechanism

At each query the monitor appends the final-flow action route and, once a
previous route exists, the adjacent Hellinger distance between it and the
route one query back. Once `window` queries have accumulated, it forms

```text
R(q, window) = d(z_q, z_{q-window}) / sum_{k=1..window} d(z_{q-k+1}, z_{q-k})
```

restricted to `layer_group`, using only routes observed at queries `0..q`.
Before the window fills, `R` is undefined (`NaN`) and no alarm can fire.

An alarm latches the first time `R` stays below `ratio_threshold` for
`confirmations` consecutive queries, and remains latched for the rest of the
rollout. The earliest possible alarm is at `q = window + confirmations - 1`.

## Excluded by design

The monitor and its profile never read or store: task or suite ID, task
prototypes, per-task thresholds, episode outcome, episode horizon, simulator
state, or any query beyond the one just observed. Nothing about a future
query can change a score already emitted for an earlier query.

## Online/batch agreement

The streaming ring-buffer computation and the vectorized batch computation
over a full trajectory (`lag_distances` + `path_length` + `progress_ratio` +
`group_ratio`) must produce identical `R` sequences, since operating points
are selected against the batch path and deployed through the online path.
