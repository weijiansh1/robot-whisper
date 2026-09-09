# Amendment 2: remove router weights and CountSketch from the primary test

Time: `2026-08-23T13:24:37Z` (UTC).

At this time no reconstructed `m_e` feature value, cache metadata value, raw
summary, or predictive outcome had been opened or inspected. No outcome model
had been run.

The primary pre-down feature is now the four sorted absolute
`L2(m_e)` values at each token, yielding an exact 40-dimensional block. It does
not multiply or divide by router weights and is invariant to top-k slot order
and a global expert-ID permutation. The matched control is the exact 40 sorted
`L2(E_e(h))` values. Router-weighted and companion-statistic aggregates are
secondary/descriptive only and cannot enter the primary claim gate.

The B1 baseline is represented by the exact, train-fold-standardized additive
linear kernel over all eight original B1 families. No CountSketch or other
random projection is used. This amendment follows the authoritative frozen
protocol at
[`../raw-internal-activation-correction/PREREGISTRATION.md`](../raw-internal-activation-correction/PREREGISTRATION.md).
