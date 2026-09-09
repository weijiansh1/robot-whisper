# Early MoE Route Commitment Grid

## Question

Fix the complete control-step-0 routing trajectory, vary only future flow noise, and ask whether the final outcome changes.

## Integrity

- Cells captured: `128`
- Initial observation hashes: `1`
- Same AS-route maximum absolute difference: `0`
- Same HB-route maximum absolute difference: `0`
- Same first-action maximum absolute difference: `0`
- Future-noise mismatches: `0 / 2570`
- Integrity valid: `True`

## Outcome Matrix

Rows fix the first route; columns fix the common future-noise stream. `S` is success and `F` is failure.

```text
route 00: S S F S F S F F  p=0.500
route 01: F F F S S F S F  p=0.375
route 02: F F F S F F S F  p=0.250
route 03: F S F S F F S F  p=0.375
route 04: S F S S F S S S  p=0.750
route 05: F S F F S S S F  p=0.500
route 06: F S F F F S F S  p=0.375
route 07: S F F S F S F S  p=0.500
route 08: F S F F S F S S  p=0.500
route 09: F F F F S S F S  p=0.375
route 10: F S F S F F S F  p=0.375
route 11: F S F F S S S F  p=0.500
route 12: S F S F S S F F  p=0.500
route 13: F F F F F S F S  p=0.250
route 14: F S F F S F S F  p=0.375
route 15: F F F F F S S F  p=0.250
```

## Primary Result

- Mixed rows: `16 / 16`
- All-failure rows: `0`
- All-success rows: `0`
- Conditional entropy H(outcome | first route): `0.9357 bits`
- Plug-in mutual information: `0.0466 bits`
- Permutation p (stable rows): `1.000000`
- Permutation p (conditional entropy): `0.921004`
- Permutation p (common-future column variance): `0.082596`

At least one identical early route produced both outcomes under different future noise. The early route alone therefore did not determine the final outcome in this grid.
