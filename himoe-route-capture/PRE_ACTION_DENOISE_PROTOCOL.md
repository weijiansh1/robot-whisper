# Pre-action MoE denoising signal protocol

> **Status (audited 2026-08-12): NOT RUN. There is no result behind this protocol.**
>
> The grid below calls for 512 rollouts (64 first-step seeds x 8 future streams).
> `runs/preaction64x8-s24-mig2g-client` holds **28 cells, 5.5% of it**: only 4 of the
> 64 first-step routes were reached, and 2 of those 4 rows have 6 of 8 future streams.
> The declared primary endpoint (`Z_6`) was never computed —
> `analysis/preaction64x8-s24-mig2g-pilot/REPORT.md` says "the grid is incomplete;
> commitment statistics were not computed", and that is the whole of it.
>
> Everything below this line is a *pre-registration*, not a finding. It stays frozen and
> unedited so that it can still be used as one if the grid is ever collected. Do not cite
> it as evidence, and do not tune it against the 28 cells that exist — that would spend
> the pre-registration for nothing.

Status: frozen before the new `64 x 8` outcome grid is collected.

## Question

Before the first action is executed, does the internal MoE routing trajectory
estimate how robustly that initial policy state will succeed under independent
future flow-noise continuations?

This is not a test of whether one rollout is already deterministic. Identical
first-step internal states can have different outcomes under future noise. The
target is therefore the continuation success probability of an initial route.

## Fixed conditions

- benchmark: `libero_goal`
- task: `0`
- initial state: `24`
- environment seed: `7`
- checkpoint SHA-256:
  `98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953`
- wrist layout: `released-left`
- hardware for every model inference: `MIG 2g.35gb`, UUID
  `MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a`
- rollout horizon: 300 action steps
- action chunk/replanning interval: 10

The run uses 64 independent first-step flow-noise seeds `3000..3063` and eight
common future-noise seeds `10000..10007`, for exactly 512 rollouts. Collection
does not stop based on intermediate results.

## Paired outcome target

For initial-route row `i` and future continuation column `c`, let `Y_ic` be the
binary rollout outcome. The target for row `i` is

```text
p_i = (1/8) * sum_c Y_ic.
```

Every cell in a row must have exactly equal first flow noise, complete AS/HB
routing state, and first action chunk. Every cell in a column must receive the
same future-noise tensor sequence from control step 1 onward. These conditions
are hard integrity checks, not assumptions.

## Pre-action feature axis

The first inference contains HB routing probabilities

```text
P_i[layer, denoise d, suffix token, expert]
```

with 8 HB layers, 10 internal denoising rounds, 1 state token, 10 action tokens,
and 32 experts. No feature may use control step 1 or later.

For each internal round `j=0..9`, `Z_j` is computed only from action tokens and
rounds `d<=j`. It concatenates these fixed, predeclared components per layer:

1. mean 32-expert probability across `d<=j` and action tokens;
2. normalized router entropy, averaged across `d<=j` and action tokens;
3. top-1 minus top-2 probability margin, averaged the same way;
4. action-token top-1 consensus, averaged across `d<=j`;
5. top-1 churn between successive denoising rounds up to `j`;
6. total-variation displacement between rounds `0` and `j`.

AS routing has no denoising-round axis. Its cross-row variance is reported as an
integrity/control statistic but it is not included in `Z_j`.

## Predictor and endpoints

The predictor is distance-weighted 5-nearest-neighbor regression with strict
leave-one-initial-route-out evaluation. Feature centering/scaling for a held-out
row uses only the other 63 rows. Neighbor count and feature definition are not
tuned on outcomes.

Primary endpoint:

- out-of-sample correlation between the `Z_6` prediction and `p_i`;
- one-sided column-preserving permutation p-value.

Round 6 is fixed because it was the strongest exploratory candidate in the
earlier 16-route grid. The new seeds and fixed MIG make this an independent
confirmation attempt.

Secondary endpoints:

- MSE skill relative to the leave-one-route-out mean target;
- the full `j=0..9` correlation curve;
- family-wise p-values from the maximum correlation over all ten rounds.

## Null construction

Each permutation independently shuffles row identities inside every common
future-noise column, then recomputes `p_i`. This preserves the success count and
difficulty of every future continuation while breaking its association with the
initial route. The feature geometry and all neighbor weights stay fixed.

The final analysis uses 20,000 permutations. The primary `Z_6` p-value is
reported without ten-round correction because `j=6` is declared here. The scan
over all rounds is reported only with max-statistic FWER correction.

## Controls and label reliability

The same label-free 5-NN procedure is applied to two pre-action controls:

- raw first flow noise `epsilon_0`;
- the first `[10, 7]` action chunk.

Target reliability is estimated over all 35 unique 4-vs-4 splits of the eight
future columns. Half-grid correlations and Spearman-Brown corrected reliability
are reported. A weak reliability ceiling limits any route predictor and must not
be interpreted as evidence that routing lacks information.

## Interpretation boundary

- A significant result means the pre-action internal route estimates robustness
  to future perturbations under this task/init/checkpoint/hardware condition.
- It does not mean the exact future rollout is already determined.
- A null result means this declared representation and sample did not detect a
  reliable signal; it does not prove that no pre-action representation can.
