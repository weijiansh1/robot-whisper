# Alternative routing organizations

## Cohort

- 2,560 episodes: 2,253 success and 307 failure.
- Five formal `right-16x32` task runs; every task contains 16 initial states by 32 flow-noise seeds.
- The episode is the statistical unit.
- Primary window: ten relative-phase anchors over `[0.50, 1.00]`.
- Terminal sensitivity: ten anchors over `[0.50, 0.90]`.
- Absolute control-step index and episode length are never clustering features.

## Label blindness

Outcome is not used to build a representation, scale a feature, fit PCA, select a graph, or fit a clusterer. Task and initial-state identifiers are used only by the explicitly named sibling-residual quotient. Outcome labels are opened for the final association and composition audit.

## Representations

### Peer rank

For every `task x initial-state` cell, subtract the median route path of its 32 flow-noise siblings. At each phase, rank route displacement, velocity displacement, and bend displacement within that cell. Concatenate the three rank curves and their DCT coefficients. This asks whether an episode is persistently ordinary or unusual relative to counterfactual siblings, while discarding task and initial-state route location.

### Lag spectrum

Group the 45 phase-pair route distances by relative lag 1 through 9. For every soft/hard routing channel, retain the mean and dispersion at each lag and the lag DCT, normalized by that episode's mean distance. This is a translation-invariant routing variogram: it preserves the temporal scale of recurrence while forgetting where in the episode it occurred and how large the absolute route motion was.

### Path signature

Subtract the `task x initial-state` sibling median, robust-scale, and project query states to eight label-blind PCs. Divide path increments by total arc length, then retain displacement, coordinate-wise variation, second-order signed Levy areas, sorted turn cosines, sorted step lengths, chord ratio, radius, reversal fraction, and step concentration. The representation is invariant to translation and positive scale and is approximately invariant to monotone time reparameterization.

### Route topology

Reconstruct the ten-by-ten within-episode soft and hard route-distance matrices. Normalize each by its median pair distance. Summarize zero- and one-dimensional Vietoris-Rips persistence, Betti curves, multiscale nonlocal recurrence graphs, terminal back-links, cycle rank, and normalized-Laplacian eigenvalues. This tests whether a trap is a topological loop rather than a location in route space.

### Layer wave

Parse each query descriptor back into eight layers by 40 expert-permutation-invariant statistics. After sibling-median residualization, measure the normalized distribution of route-change energy over layers, its center, spread, entropy, cross-layer directional synchrony, adjacent-layer synchrony, and phase DCT. This tests whether failures share a propagation pattern through the MoE depth.

## Clusterers

Two deliberately different cluster assumptions are audited:

- HDBSCAN: `min_cluster_size=32`, `min_samples=16`, EOM selection. Noise remains noise and is not forced into a cluster. Sensitivity covers `(min_cluster_size, min_samples)` equal to `(16,8)`, `(32,8)`, `(32,16)`, `(64,16)`, and `(64,32)`.
- Louvain: a Gaussian-weighted 25-nearest-neighbor graph at resolution 1.0. Sensitivity covers 15, 25, and 40 neighbors crossed with resolution 0.8, 1.0, and 1.2.

Every representation is median/IQR scaled and reduced to at most 24 PCs, retaining up to 90% variance subject to the cap. Louvain communities below 32 episodes are reported as fragmentation and are not interpreted as reliable classes.

## Outcome audit

Association is measured as normalized mutual information in excess of 2,000 permutations within `task x initial-state`. Task NMI and exact episode-length NMI are reported as confound diagnostics. A nominal outcome effect of `0.05` is retained from the earlier formal analysis as a descriptive threshold; this exploratory multi-view screen is not confirmatory.

## Interpretation

Relative phase, lag normalization, and arc-length normalization remove numeric duration from the feature vector. They cannot make successful completion and failure timeout physically equivalent. HDBSCAN noise means absence of a dense mode at the frozen scale, not a positive failure class.
