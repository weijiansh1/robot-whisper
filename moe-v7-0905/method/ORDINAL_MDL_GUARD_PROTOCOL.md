# Ordinal MDL Guard: Calibration-Free v7 Alternative

## Scope and freeze

This is one exploratory, fixed alternative to v7, not a replacement selected
using outcome metrics. Freeze this protocol and implementation before computing
new outcome metrics. The author has already seen v7 and other experiments on
these cohorts: this is NOT blind discovery or a fresh holdout experiment.

There is no offline fit, outcome-guided grid, reference corpus, calibrated
threshold, significance level, hazard rate, smoothing width, persistence count,
or task-specific parameter. The method DOES have a fixed code/model convention
and an implicit decision boundary. It is not assumption-free or a proof that
threshold-free binary failure detection is possible. Online symbol counts are
adaptive statistics, not a trained failure classifier; under a definition that
forbids all statistical estimation, this method would not qualify either.

## Inputs and ordinal representation

Reuse v7's raw MoE feature extraction without its profile or score transform.
The input is only the completed current `[8, 10, 11, 32]` HB routing tensor.
The three oriented scalar features are:

1. Negative median final-flow mobility across the four back layers.
2. Back-layer denoising-flow route acceleration.
3. Negative lag-periodicity contrast.

Larger values are in the failure-associated direction hypothesized by v7.
Only jointly finite features from q2 onward are used; q2 is the first query with
lag-2 recurrence available, not a tuned warm-up. q2 seeds ordinal history.
For every later query compare each feature with the LOWER MEDIAN of all its
previous jointly available values in this episode. Encode smaller / equal /
larger as 0 / 1 / 2. Never recompute earlier symbols using later observations.
Ties are a separate symbol; constant routes are not automatically failures.

This removes numerical feature scale but discards magnitude. The historical
median is an ordinal reference, not a learned alarm threshold. Layer choice,
feature directions, and recurrence lags remain inherited design assumptions.

## Fixed universal code

For a ternary sequence with counts c0, c1, c2 and length n, use the
Dirichlet-half (Krichevsky-Trofimov) integrated code in bits:

```text
L(c) = [lgamma(n + 3/2) - lgamma(3/2)
        - sum_j(lgamma(c_j + 1/2) - lgamma(1/2))] / log(2)
```

No probability parameters are optimized. Compare these fixed explanations:

```text
stable:      1 + sum_h L(all symbols of head h)
freeze(k):   2 + log2(k*(k+1)) + L(freeze[:k]) + L(freeze[k:])
               + L(all acceleration) + L(all recurrence)
turbulent(k):2 + log2(k*(k+1)) + L(all freeze)
               + L(acceleration[:k]) + L(acceleration[k:])
               + L(recurrence[:k]) + L(recurrence[k:])
```

Here k is the number of encoded observations before the split, from 1 through
n-1. The model identifiers `0`, `10`, `11` cost 1, 2, 2 bits. The split prior
`1/(k*(k+1))` sums to one over all positive integers and needs no final horizon.
These are ideal code lengths for the SYMBOLS, not lossless codes of raw routes.

For each changed head, require the suffix signed-symbol mean to be positive
and greater than its prefix mean. These are direction comparisons, not fitted
sensitivity parameters. The turbulent branch requires both heads to change
at the SAME split. Their evidence is combined under the stated product-code
convention; route heads are not claimed to be independent in real rollouts.

Enumerate every available split, select the shortest eligible explanation,
and prefer stable on a tie. The continuous output `gain_bits` is stable code
length minus the best eligible alternative length, even when negative. It is
NOT a calibrated failure probability, p-value, or anytime-valid e-value.
The fixed convenience decision is that the alternative is strictly shorter.
There is no configurable score cutoff and no false-positive-rate guarantee.

## Timing and state

Report the current decision query separately from the estimated earlier change
query. Never backdate the first alarm to the estimated split. Current evidence
may recover; the first-alarm fields latch once for compatibility with v7.
An eligible split can be reported with negative gain; it is not an alarm.
All histories reset when constructing a new monitor. No future observations,
episode length, outcome, or task identity enter a decision.

## Evaluation and acceptance

Replay development_main and external_8b in their original aligned order. Write
and hash all new score streams, decisions, and first-alarm arrays BEFORE
opening outcome CSVs. Only then compare to the existing sealed v7 alarms.
Do not run a search over code priors, feature subsets, signs, or alarm rules
after seeing results. Report both cohorts, per-suite metrics, false alarms,
precision, recall, endpoint lead, and recall before half the recorded episode.
Endpoint-relative timing is retrospective evaluation, not evidence of the
first irreversible physical failure. External_8b is already inspected data.

Verify the code probability formula, constant and synthetic changed sequences,
stream/batch equality, prefix causality, padding invariance, invalid inputs,
and raw-router/cache replay on label-blind samples. Preserve v7 unchanged.
Improved constraint compliance is not improved detection accuracy: reject this
as a drop-in replacement if the replay does not support the latter claim.

## References

- Barron, Rissanen, Yu (1998), minimum description length and mixture coding:
  https://web.mit.edu/6.433/www/handouts/minimumdescriptionlength.pdf
- Veness et al. (2011), Section 5, the KT integrated coding formula:
  https://users.cecs.anu.edu.au/~kee/jair-aixi-ctw.pdf

These references motivate the coding principle. They do not validate this
particular ordinal MoE representation or its failure-detection performance.
