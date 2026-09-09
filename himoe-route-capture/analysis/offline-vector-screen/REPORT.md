# Fixed HB5/d0 offline vector screen

## Result

The pre-runtime screen is **negative** under the frozen decision rule. Adding the train-fold base-residualized routed vector changes macro held8 Spearman by `-0.014` ([-0.021, -0.007]) and relative K8 coverage by `-0.1%` ([-0.5%, +0.2%]).

This target is the normalized final `actions[0]` 10x7 geometry. It is an offline screening target and is not the runtime remaining-correction target.

The thresholds below are conservative engineering effect-size gates fixed before the corrected run. They are not null-hypothesis significance tests; paired bootstrap intervals are descriptive.

## Primary nested increment

`base` is the equal-kernel combination of noise24, hidden, shared, and router. Within every outer train fold, routed is ridge-residualized against that base. The augmented kernel is exactly `K_base + K_routed_residual`; `K_base` is unchanged.

| task | base rho | base+residual routed rho | rho gain (CI95) | coverage gain (CI95) | rho pool wins | coverage pool wins |
|---|---:|---:|---:|---:|---:|---:|
| goal-middle | 0.400 | 0.415 | +0.015 [-0.001, 0.033] | -0.5% [-1.0%, +0.1%] | 0.625 | 0.312 |
| goal-top | 0.520 | 0.468 | -0.052 [-0.065, -0.038] | -0.6% [-0.9%, -0.3%] | 0.000 | 0.188 |
| long-t08 | 0.605 | 0.629 | +0.024 [-0.002, 0.046] | +0.2% [-1.4%, +1.5%] | 0.750 | 0.688 |
| spatial-ramekin | 0.560 | 0.544 | -0.016 [-0.027, -0.005] | -0.0% [-0.6%, +0.6%] | 0.250 | 0.375 |
| spatial-stove | 0.477 | 0.434 | -0.043 [-0.054, -0.032] | +0.3% [-0.0%, +0.7%] | 0.000 | 0.562 |

| engineering gate | pass |
|---|---:|
| every task rho gain >= +0.02 | false |
| every task relative coverage gain >= +1% | false |
| paired rho pool win rate >= 60% | false |
| paired coverage pool win rate >= 60% | false |
| macro coverage over exact random improves >= 1% | true |

## Macro over 80 state pools

Spearman is computed only within each held-eight seed block. Coverage ratios below one are better than the exact two-per-held8 random expectation.

| method | within-held8 Spearman | K8 coverage / exact random | improved pools |
|---|---:|---:|---:|
| noise24 | 0.018 | 1.015 | 16/80 |
| HB5 hidden token-mean | 0.537 | 0.978 | 76/80 |
| shared token-mean | 0.466 | 0.988 | 65/80 |
| routed expert token-mean | 0.482 | 0.977 | 78/80 |
| router probability token-mean | 0.297 | 0.994 | 48/80 |
| expert scalars (s1,D,C,Q) | 0.097 | 1.000 | 37/80 |
| unweighted selected raw-vector mean | 0.483 | 0.977 | 78/80 |
| weighted expert channel std | 0.256 | 0.989 | 61/80 |
| per-channel cancellation gap | 0.098 | 0.997 | 55/80 |
| fixed base (4 equal blocks) | 0.512 | 0.978 | 76/80 |
| base + residual routed | 0.498 | 0.979 | 73/80 |

## Standalone routed diagnostic

Standalone routed is compared with the best observed standalone nonexpert baseline among noise24, hidden, shared, and router. That baseline choice is descriptive and data-dependent, so this table cannot make the primary screen positive.

| task | best baseline | baseline rho | routed rho | routed-best | routed-hidden | routed-shared |
|---|---|---:|---:|---:|---:|---:|
| goal-middle | hidden | 0.450 | 0.380 | -0.070 | -0.070 | -0.011 |
| goal-top | hidden | 0.504 | 0.437 | -0.067 | -0.067 | -0.060 |
| long-t08 | hidden | 0.588 | 0.589 | +0.001 | +0.001 | +0.162 |
| spatial-ramekin | hidden | 0.572 | 0.558 | -0.014 | -0.014 | +0.026 |
| spatial-stove | hidden | 0.569 | 0.445 | -0.123 | -0.123 | -0.037 |

Standalone routed exceeds hidden in 1/5 tasks and shared in 2/5 tasks. It therefore does not satisfy the original five-task control-consistency question either.

## Descriptive raw decomposition controls

These three controls were added to check whether the weighted routed merge erases a useful raw-expert signal. They are not part of the primary gate or fixed base; their own K8 metrics are descriptive only.

| task | raw equal mean | weighted channel std | cancellation gap |
|---|---:|---:|---:|
| goal-middle | 0.377 | 0.156 | -0.000 |
| goal-top | 0.443 | 0.252 | 0.104 |
| long-t08 | 0.589 | 0.152 | 0.013 |
| spatial-ramekin | 0.556 | 0.402 | 0.227 |
| spatial-stove | 0.450 | 0.319 | 0.144 |

## Per-task method Spearman

| task | noise24 | hidden | shared | routed | router | scalars |
|---|---:|---:|---:|---:|---:|---:|
| goal-middle | 0.083 | 0.450 | 0.391 | 0.380 | 0.253 | 0.184 |
| goal-top | -0.003 | 0.504 | 0.497 | 0.437 | 0.164 | 0.177 |
| long-t08 | 0.010 | 0.588 | 0.427 | 0.589 | 0.423 | 0.050 |
| spatial-ramekin | -0.040 | 0.572 | 0.532 | 0.558 | 0.362 | -0.023 |
| spatial-stove | 0.039 | 0.569 | 0.482 | 0.445 | 0.285 | 0.098 |

## Per-pool paired results

Each pool value is the mean of its four held8 Spearman correlations.

| task | state | base rho | base+residual rho | rho gain | relative coverage gain | hidden | shared | routed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goal-middle | 0 | 0.463 | 0.553 | +0.090 | -0.6% | 0.521 | 0.490 | 0.476 |
| goal-middle | 3 | 0.385 | 0.400 | +0.015 | -2.3% | 0.439 | 0.358 | 0.335 |
| goal-middle | 7 | 0.329 | 0.364 | +0.035 | +0.7% | 0.392 | 0.373 | 0.371 |
| goal-middle | 10 | 0.439 | 0.475 | +0.037 | -0.0% | 0.491 | 0.462 | 0.359 |
| goal-middle | 13 | 0.339 | 0.388 | +0.050 | +1.8% | 0.398 | 0.374 | 0.323 |
| goal-middle | 16 | 0.389 | 0.407 | +0.018 | -2.6% | 0.409 | 0.413 | 0.335 |
| goal-middle | 20 | 0.440 | 0.440 | -0.000 | +0.0% | 0.487 | 0.465 | 0.405 |
| goal-middle | 23 | 0.452 | 0.507 | +0.054 | -2.0% | 0.519 | 0.468 | 0.440 |
| goal-middle | 26 | 0.386 | 0.358 | -0.028 | -0.2% | 0.401 | 0.353 | 0.267 |
| goal-middle | 29 | 0.444 | 0.411 | -0.033 | -0.8% | 0.495 | 0.367 | 0.443 |
| goal-middle | 33 | 0.425 | 0.416 | -0.009 | +0.2% | 0.496 | 0.400 | 0.438 |
| goal-middle | 36 | 0.458 | 0.473 | +0.015 | +0.1% | 0.495 | 0.382 | 0.455 |
| goal-middle | 39 | 0.356 | 0.384 | +0.027 | +0.9% | 0.395 | 0.381 | 0.326 |
| goal-middle | 42 | 0.418 | 0.396 | -0.022 | -1.6% | 0.454 | 0.320 | 0.381 |
| goal-middle | 46 | 0.344 | 0.305 | -0.039 | -0.9% | 0.375 | 0.339 | 0.318 |
| goal-middle | 49 | 0.336 | 0.370 | +0.034 | -0.2% | 0.439 | 0.312 | 0.412 |
| goal-top | 0 | 0.545 | 0.511 | -0.034 | -1.0% | 0.523 | 0.481 | 0.481 |
| goal-top | 3 | 0.551 | 0.469 | -0.082 | -0.2% | 0.525 | 0.500 | 0.391 |
| goal-top | 7 | 0.476 | 0.432 | -0.044 | +0.0% | 0.488 | 0.435 | 0.371 |
| goal-top | 10 | 0.524 | 0.449 | -0.076 | +0.0% | 0.486 | 0.535 | 0.371 |
| goal-top | 13 | 0.568 | 0.555 | -0.013 | -0.2% | 0.527 | 0.538 | 0.577 |
| goal-top | 16 | 0.539 | 0.448 | -0.090 | -0.3% | 0.481 | 0.522 | 0.358 |
| goal-top | 20 | 0.533 | 0.467 | -0.066 | -1.0% | 0.502 | 0.510 | 0.492 |
| goal-top | 23 | 0.409 | 0.357 | -0.051 | -0.5% | 0.425 | 0.388 | 0.389 |
| goal-top | 26 | 0.562 | 0.494 | -0.069 | -0.4% | 0.579 | 0.563 | 0.502 |
| goal-top | 29 | 0.514 | 0.472 | -0.042 | -2.0% | 0.548 | 0.484 | 0.392 |
| goal-top | 33 | 0.584 | 0.553 | -0.030 | -0.9% | 0.542 | 0.533 | 0.473 |
| goal-top | 36 | 0.551 | 0.459 | -0.093 | -0.7% | 0.519 | 0.513 | 0.416 |
| goal-top | 39 | 0.453 | 0.425 | -0.028 | -1.1% | 0.434 | 0.445 | 0.433 |
| goal-top | 42 | 0.535 | 0.449 | -0.087 | -0.3% | 0.521 | 0.519 | 0.436 |
| goal-top | 46 | 0.496 | 0.481 | -0.014 | -1.6% | 0.481 | 0.472 | 0.466 |
| goal-top | 49 | 0.479 | 0.469 | -0.010 | +0.3% | 0.484 | 0.519 | 0.448 |
| long-t08 | 0 | 0.717 | 0.686 | -0.031 | +1.3% | 0.689 | 0.487 | 0.690 |
| long-t08 | 3 | 0.546 | 0.603 | +0.057 | +1.5% | 0.532 | 0.420 | 0.568 |
| long-t08 | 7 | 0.600 | 0.562 | -0.037 | -1.6% | 0.578 | 0.458 | 0.525 |
| long-t08 | 10 | 0.659 | 0.718 | +0.059 | +1.1% | 0.643 | 0.447 | 0.683 |
| long-t08 | 13 | 0.601 | 0.540 | -0.061 | +2.9% | 0.592 | 0.418 | 0.531 |
| long-t08 | 16 | 0.646 | 0.717 | +0.071 | +2.2% | 0.623 | 0.460 | 0.686 |
| long-t08 | 20 | 0.630 | 0.682 | +0.052 | +2.3% | 0.617 | 0.375 | 0.608 |
| long-t08 | 23 | 0.619 | 0.661 | +0.042 | +1.6% | 0.553 | 0.418 | 0.608 |
| long-t08 | 26 | 0.380 | 0.303 | -0.077 | -8.4% | 0.430 | 0.331 | 0.268 |
| long-t08 | 29 | 0.630 | 0.661 | +0.031 | +3.8% | 0.601 | 0.443 | 0.617 |
| long-t08 | 33 | 0.550 | 0.610 | +0.060 | -4.0% | 0.491 | 0.377 | 0.503 |
| long-t08 | 36 | 0.671 | 0.738 | +0.067 | +3.0% | 0.633 | 0.437 | 0.703 |
| long-t08 | 39 | 0.583 | 0.654 | +0.071 | +0.7% | 0.546 | 0.407 | 0.629 |
| long-t08 | 42 | 0.641 | 0.651 | +0.010 | -1.3% | 0.646 | 0.444 | 0.578 |
| long-t08 | 46 | 0.539 | 0.607 | +0.068 | -2.5% | 0.524 | 0.338 | 0.559 |
| long-t08 | 49 | 0.669 | 0.672 | +0.004 | +0.7% | 0.702 | 0.568 | 0.661 |
| spatial-ramekin | 0 | 0.529 | 0.521 | -0.007 | -0.8% | 0.500 | 0.451 | 0.546 |
| spatial-ramekin | 3 | 0.550 | 0.532 | -0.018 | +1.0% | 0.492 | 0.484 | 0.558 |
| spatial-ramekin | 7 | 0.612 | 0.627 | +0.015 | -1.0% | 0.630 | 0.552 | 0.643 |
| spatial-ramekin | 10 | 0.555 | 0.553 | -0.002 | +1.3% | 0.541 | 0.502 | 0.512 |
| spatial-ramekin | 13 | 0.496 | 0.498 | +0.002 | -0.4% | 0.557 | 0.496 | 0.508 |
| spatial-ramekin | 16 | 0.647 | 0.604 | -0.043 | -0.1% | 0.686 | 0.624 | 0.608 |
| spatial-ramekin | 20 | 0.524 | 0.523 | -0.001 | +2.6% | 0.606 | 0.524 | 0.568 |
| spatial-ramekin | 23 | 0.623 | 0.593 | -0.030 | -2.2% | 0.635 | 0.553 | 0.662 |
| spatial-ramekin | 26 | 0.527 | 0.498 | -0.029 | -0.5% | 0.486 | 0.501 | 0.436 |
| spatial-ramekin | 29 | 0.544 | 0.501 | -0.044 | -1.9% | 0.546 | 0.584 | 0.543 |
| spatial-ramekin | 33 | 0.661 | 0.643 | -0.018 | +1.8% | 0.721 | 0.609 | 0.646 |
| spatial-ramekin | 36 | 0.497 | 0.450 | -0.047 | +0.5% | 0.486 | 0.478 | 0.427 |
| spatial-ramekin | 39 | 0.500 | 0.516 | +0.016 | +0.2% | 0.536 | 0.518 | 0.625 |
| spatial-ramekin | 42 | 0.432 | 0.377 | -0.055 | -0.2% | 0.462 | 0.451 | 0.414 |
| spatial-ramekin | 46 | 0.628 | 0.635 | +0.007 | -0.2% | 0.641 | 0.601 | 0.673 |
| spatial-ramekin | 49 | 0.631 | 0.630 | -0.001 | -0.7% | 0.628 | 0.584 | 0.558 |
| spatial-stove | 0 | 0.426 | 0.403 | -0.023 | -0.2% | 0.507 | 0.422 | 0.423 |
| spatial-stove | 3 | 0.419 | 0.386 | -0.033 | +0.9% | 0.531 | 0.419 | 0.425 |
| spatial-stove | 7 | 0.570 | 0.520 | -0.049 | +1.5% | 0.614 | 0.528 | 0.520 |
| spatial-stove | 10 | 0.513 | 0.438 | -0.075 | +0.5% | 0.612 | 0.515 | 0.465 |
| spatial-stove | 13 | 0.456 | 0.456 | -0.001 | -0.3% | 0.545 | 0.450 | 0.461 |
| spatial-stove | 16 | 0.493 | 0.453 | -0.040 | -0.3% | 0.585 | 0.514 | 0.500 |
| spatial-stove | 20 | 0.421 | 0.397 | -0.025 | -0.0% | 0.495 | 0.427 | 0.469 |
| spatial-stove | 23 | 0.556 | 0.491 | -0.065 | +0.3% | 0.615 | 0.536 | 0.478 |
| spatial-stove | 26 | 0.383 | 0.336 | -0.047 | +0.2% | 0.539 | 0.442 | 0.340 |
| spatial-stove | 29 | 0.479 | 0.472 | -0.006 | -0.9% | 0.582 | 0.543 | 0.483 |
| spatial-stove | 33 | 0.491 | 0.450 | -0.040 | +0.4% | 0.597 | 0.524 | 0.481 |
| spatial-stove | 36 | 0.403 | 0.382 | -0.021 | +0.5% | 0.450 | 0.361 | 0.386 |
| spatial-stove | 39 | 0.428 | 0.368 | -0.060 | -0.0% | 0.569 | 0.462 | 0.346 |
| spatial-stove | 42 | 0.505 | 0.419 | -0.087 | +0.2% | 0.600 | 0.492 | 0.413 |
| spatial-stove | 46 | 0.601 | 0.532 | -0.069 | +0.0% | 0.669 | 0.582 | 0.488 |
| spatial-stove | 49 | 0.493 | 0.447 | -0.046 | +2.4% | 0.589 | 0.499 | 0.449 |

## Protocol and boundary

- Cell selection was frozen at HB5/d0. No layer or denoise round was searched.
- HB5/d0 is evaluated on the initial x0 forward. The endpoint remains the normalized final action chunk; it is not relabeled as a remaining-correction target.
- The five tasks are the complete right-16x32 sources already referenced by the expert-activation proxy analysis. Every task has 16 state pools and the same 32 seeds.
- For every held state and held-eight seed block, training uses only the other 15 states and other 24 seeds. Feature centering/RMS is fitted only on that outer-train fold and applied to held8. Target centering is used only for outer-train targets; held8 targets are untouched and pair distances are invariant to a common offset.
- The base kernel is the arithmetic mean of four outer-train trace-normalized block kernels. Routed residualization uses fixed ridge 1.0 fitted on the same outer train fold; the augmented kernel preserves the full base kernel and appends only the residual.
- Final-action geometry is the full client `actions[0]` 10x7 chunk divided by the official checkpoint action standard deviations.
- Routed/shared vectors are offline fp32 checkpoint reconstructions from stored fp16 hidden states. Recorded selected expert IDs and probabilities remain authoritative; this is not a runtime-exact or causal intervention.
- Expert IDs are used only within their own checkpoint suite. No expert identity is aligned or pooled across the Goal, Spatial, and Libero-10 checkpoints.
- The vector features are ten-token means. This screen does not test per-token raw Gram structure or S_LOO, so the negative result does not exclude those interfaces.
- K8 selection is PAM K2 independently within each held8. The coverage baseline is the exact combinatorial expectation for two uniform candidates from every held8.
- Paired bootstrap intervals resample the 16 state pools within each fixed task; the macro interval averages the five task-stratified draws. They are descriptive and are not used as significance gates.
