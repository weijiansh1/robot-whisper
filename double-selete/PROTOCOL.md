# Double-selector protocol (frozen before evaluation)

This experiment asks whether a type-aware two-head selector is better at
selecting physical Trap trajectories than one binary head.

## Clarified estimand (current)

After the original constant-total-budget run, the user clarified that "two
heads" means processing each subtype independently: the loop head and static
head each receive a complete K budget. The current primary interpretation is
therefore:

- at query 34, each typed head independently selects 8 of 32 trajectories;
- each head's own-type precision/recall is compared with the union-label single
  head at the same K=8;
- the two independent selections are then unioned (natural overlap retained);
- because that union contains more than eight trajectories, aggregate union
  performance is compared with the single head selecting exactly the same
  number of trajectories in each held-out pool;
- uncertainty is a paired 20,000-draw bootstrap over 16 initial states.

This clarification corrects the estimand rather than selecting a favorable
metric after seeing results. The earlier 4+4 experiment remains below as a
constant-compute control.

## Scope

- Corpus: VLA_MUI_HUB `right-16x32` moka-pot run, 512 trajectories.
- Labels: the frozen query-resolution physical predicates in
  `himoe-vla_trap/code/analyze_trainfree_signal_matrix.py`.
- Positive labels are multi-label: `loop`, `static`, and their union `trap`.
- Split: leave one complete `init_state_id` out. No branch from the held-out
  state is used to fit a selector.
- Inputs: routing data available at or before the scored query. Episode length,
  success, physical state, action, and onset-relative time are excluded.
- Clean decision window: queries 12 through 34 inclusive. Query 34 is the last
  query available to every trajectory, so later unequal-opportunity windows are
  not used for the primary result.

## Selectors

All logistic heads use standardized features, fixed `C=0.1`, balanced class
weights, and no hyperparameter search.

1. `single`: one head predicts `loop OR static` from the union of all features.
2. `double_shared_max`: loop and static heads both see that same union; their
   within-pool percentile scores are combined by `max`.
3. `double_typed_max`: two heads use mechanism-specific subsets; their
   within-pool percentile scores are combined by `max`.
4. `double_shared_quota` and `double_typed_quota`: half of an exact top-K budget
   is assigned to each head, duplicates are merged, and remaining slots are
   filled by the maximum within-pool head percentile.

The single head sees every feature available to either typed head, including
the hard Top-4 support features. Thus a gain cannot be attributed to extra
inputs available only to the double selector.

## Original constant-total-budget control

- Snapshot: query 34.
- Selection budget: 8 of 32 trajectories per held-out initial state.
- Contrast: `double_typed_quota - single`; retained as a fixed-total-K control,
  not as the clarified independent-head primary comparison.
- Primary metrics: Trap precision, Trap recall, and macro-average of loop recall
  and static recall.
- Uncertainty: paired bootstrap over the 16 held-out initial states (20,000
  resamples). Query 30 and budgets 4/16 are secondary sensitivity analyses.

An additional sequential analysis calibrates each outer fold from inner
leave-one-state-out predictions to a nominal 8% trajectory-level false alarm
rate. It is secondary because the type labels describe eventual events, not
only already-observed onsets.

## Post-protocol diagnostics

After the first frozen-primary run, two diagnostics were added without changing
the primary contrast above:

- canonical multi-label noisy-OR fusion,
  `1 - (1 - p_loop) * (1 - p_static)`, to separate head quality from the
  originally frozen max/quota fusion;
- `soft` variants that remove every hard Top-4 identity/support feature from
  both the single and typed heads, to price fourth/fifth-boundary instability.

These rows are descriptive sensitivity analyses, not additional primary tests.

## Interpretation limits

- Corpus B static onset is a validated query-level proxy, not dense-contact
  ground truth.
- Actual Top-4 IDs are valid recorder outputs but are sensitive to fp16 ties at
  the fourth/fifth boundary. Results involving support-set identity are
  reported separately from probability-only features.
- The study evaluates Trap-trajectory selection. It does not evaluate choosing
  an action from a same-state candidate-action pool and does not establish a
  recovery intervention benefit.
- There is one task and 16 independent initial-state groups. Generalization to
  another task requires new held-out-task data.
