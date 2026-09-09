# Early-prefix commitment curve

The first k complete control steps are replayed exactly; only policy noise from control step k onward changes.

## Integrity

- Cells: `384`
- Initial observation hashes: `1`
- Same-prefix AS route max difference: `0`
- Same-prefix HB route max difference: `0`
- Same-prefix state max difference: `0`
- Same-prefix action max difference: `0`
- Future-noise mismatches: `0 / 3245`
- Nested-prefix route/state/action differences: `0 / 0 / 0 / 0`
- Successes before branch: `0`
- Valid: `True`

## Curve

| branch k | fixed action steps | mixed rows | stable rows | H(Y|prefix) | route-row p | success |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 40 | 8/8 | 0/8 | 0.9064 | 0.2107 | 0.562 |
| 8 | 80 | 6/8 | 2/8 | 0.6009 | 0.0001 | 0.500 |
| 10 | 100 | 5/8 | 3/8 | 0.5957 | 0.0000 | 0.641 |
| 11 | 110 | 2/8 | 6/8 | 0.2028 | 0.0000 | 0.625 |
| 12 | 120 | 1/8 | 7/8 | 0.1014 | 0.0000 | 0.656 |
| 14 | 140 | 2/8 | 6/8 | 0.1929 | 0.0000 | 0.672 |

## Outcome matrices

### k=4

```text
prefix 00: S F S S F F S S  p=0.625
prefix 01: F F S F S S S S  p=0.625
prefix 02: F F S S F F F F  p=0.250
prefix 03: F F S S S F F F  p=0.375
prefix 04: S F S S S S S F  p=0.750
prefix 05: F F S S S F S S  p=0.625
prefix 06: F F S S S F S F  p=0.500
prefix 07: S F S S S S S F  p=0.750
```

### k=8

```text
prefix 00: S S F S S S S S  p=0.875
prefix 01: F F F F F F F F  p=0.000
prefix 02: F S S S F F F S  p=0.500
prefix 03: F F S F F F S S  p=0.375
prefix 04: S S F S S S S S  p=0.875
prefix 05: F F F F F F F F  p=0.000
prefix 06: F S F S S F S S  p=0.625
prefix 07: S S F S S S S F  p=0.750
```

### k=10

```text
prefix 00: S S S S S S S S  p=1.000
prefix 01: F S S F S F F S  p=0.500
prefix 02: S F F F S F S F  p=0.375
prefix 03: F F F S F F S F  p=0.250
prefix 04: S S S S S S S S  p=1.000
prefix 05: S F F S S F S F  p=0.500
prefix 06: S F F S S F S F  p=0.500
prefix 07: S S S S S S S S  p=1.000
```

### k=11

```text
prefix 00: S S S F S S S F  p=0.750
prefix 01: S F F F F S F F  p=0.250
prefix 02: F F F F F F F F  p=0.000
prefix 03: S S S S S S S S  p=1.000
prefix 04: S S S S S S S S  p=1.000
prefix 05: F F F F F F F F  p=0.000
prefix 06: S S S S S S S S  p=1.000
prefix 07: S S S S S S S S  p=1.000
```

### k=12

```text
prefix 00: S S S S S S S S  p=1.000
prefix 01: S F F F F S F F  p=0.250
prefix 02: F F F F F F F F  p=0.000
prefix 03: S S S S S S S S  p=1.000
prefix 04: S S S S S S S S  p=1.000
prefix 05: F F F F F F F F  p=0.000
prefix 06: S S S S S S S S  p=1.000
prefix 07: S S S S S S S S  p=1.000
```

### k=14

```text
prefix 00: S S S S S S F S  p=0.875
prefix 01: S F F S S F F S  p=0.500
prefix 02: F F F F F F F F  p=0.000
prefix 03: S S S S S S S S  p=1.000
prefix 04: S S S S S S S S  p=1.000
prefix 05: F F F F F F F F  p=0.000
prefix 06: S S S S S S S S  p=1.000
prefix 07: S S S S S S S S  p=1.000
```
