# routing capture `right-16x32`

wrist layout `paper-right`; replan 10 / settle 10 / resize 224 / seed 7,
per the authors' `examples/libero/main.py` (identical in the ICLR
supplementary and the released repo).

| suite | success | rate | our 500-ep reference | drift |
|---|---:|---:|---:|---:|
| libero_10 | 296/512 | 57.8% | 92.4% | -34.6 pp |
| libero_spatial | 975/1024 | 95.2% | 95.4% | -0.2 pp |
| libero_goal | 982/1024 | 95.9% | 98.0% | -2.1 pp |

Control steps captured: 51308.

## problems

- libero_10: 57.8% is -34.6 pp off the 92.4% reference
