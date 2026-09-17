# Control experiments summary

| experiment | branch | trigger | control | noise | strategy | rescued / failure branches | rate | success parents kept | total success rate |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | flow_straight | 4 / 136 | 2.9% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | kick | 4 / 136 | 2.9% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | small_cluster | 3 / 136 | 2.2% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | motion_max | 3 / 136 | 2.2% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | route_stay | 3 / 136 | 2.2% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | flow_curved | 3 / 136 | 2.2% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | v82_min | 3 / 136 | 2.2% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | random | 2 / 136 | 1.5% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | boundary | 2 / 135 | 1.5% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | route_exit | 2 / 136 | 1.5% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | native | 1 / 135 | 0.7% | - | - |
| exp-01-alarm-c1 | alarm | hindsight | 1 | None | big_cluster | 1 / 136 | 0.7% | - | - |
| exp-02a-alarm-8-n25-c3 | alarm-8 | hindsight | 3 | 2.5 | kick | 19 / 136 | 14.0% | - | - |
| exp-02a-alarm-8-n25-c3 | alarm-8 | hindsight | 3 | 2.5 | small_cluster | 10 / 136 | 7.4% | - | - |
| exp-02a-alarm-8-n25-c3 | alarm-8 | hindsight | 3 | 2.5 | random | 9 / 136 | 6.6% | - | - |
| exp-02a-alarm-8-n25-c3 | alarm-8 | hindsight | 3 | 2.5 | route_exit | 9 / 136 | 6.6% | - | - |
| exp-02a-alarm-8-n25-c3 | alarm-8 | hindsight | 3 | 2.5 | boundary | 8 / 136 | 5.9% | - | - |
| exp-02a-alarm-8-n25-c3 | alarm-8 | hindsight | 3 | 2.5 | motion_max | 8 / 136 | 5.9% | - | - |
| exp-02a-alarm-8-n25-c3 | alarm-8 | hindsight | 3 | 2.5 | v82_min | 7 / 136 | 5.1% | - | - |
| exp-02a-alarm-8-n25-c3 | alarm-8 | hindsight | 3 | 2.5 | flow_straight | 7 / 136 | 5.1% | - | - |
| exp-02a-alarm-8-n25-c3 | alarm-8 | hindsight | 3 | 2.5 | native | 4 / 136 | 2.9% | - | - |
| exp-02b-alarm-n25-c3 | alarm | hindsight | 3 | 2.5 | kick | 6 / 136 | 4.4% | - | - |
| exp-02b-alarm-n25-c3 | alarm | hindsight | 3 | 2.5 | random | 5 / 136 | 3.7% | - | - |
| exp-02b-alarm-n25-c3 | alarm | hindsight | 3 | 2.5 | v82_min | 4 / 136 | 2.9% | - | - |
| exp-02b-alarm-n25-c3 | alarm | hindsight | 3 | 2.5 | boundary | 3 / 136 | 2.2% | - | - |
| exp-02b-alarm-n25-c3 | alarm | hindsight | 3 | 2.5 | small_cluster | 3 / 136 | 2.2% | - | - |
| exp-02b-alarm-n25-c3 | alarm | hindsight | 3 | 2.5 | motion_max | 3 / 136 | 2.2% | - | - |
| exp-02b-alarm-n25-c3 | alarm | hindsight | 3 | 2.5 | route_exit | 3 / 136 | 2.2% | - | - |
| exp-02b-alarm-n25-c3 | alarm | hindsight | 3 | 2.5 | flow_straight | 3 / 136 | 2.2% | - | - |
| exp-02b-alarm-n25-c3 | alarm | hindsight | 3 | 2.5 | native | 2 / 136 | 1.5% | - | - |
| exp-02c-oracle-random8 | alarm-8 | hindsight | 3 | 2.5 | random | 35 / 544 | 6.4% | - | - |
| exp-03a-kickvariants-alarm-8-c3 | alarm-8 | hindsight | 3 | 2.5 | kick_wiggle | 20 / 136 | 14.7% | - | - |
| exp-03a-kickvariants-alarm-8-c3 | alarm-8 | hindsight | 3 | 2.5 | kick_then_motion | 20 / 136 | 14.7% | - | - |
| exp-03a-kickvariants-alarm-8-c3 | alarm-8 | hindsight | 3 | 2.5 | kick_then_random | 20 / 136 | 14.7% | - | - |
| exp-03a-kickvariants-alarm-8-c3 | alarm-8 | hindsight | 3 | 2.5 | kick_high | 14 / 136 | 10.3% | - | - |
| exp-03a-kickvariants-alarm-8-c3 | alarm-8 | hindsight | 3 | 2.5 | kick_open | 11 / 136 | 8.1% | - | - |
| exp-03a-kickvariants-alarm-8-c3 | alarm-8 | hindsight | 3 | 2.5 | kick_random | 10 / 136 | 7.4% | - | - |
| exp-03a-kickvariants-alarm-8-c3 | alarm-8 | hindsight | 3 | 2.5 | kick_retreat | 7 / 136 | 5.1% | - | - |
| exp-03a-kickvariants-alarm-8-c3 | alarm-8 | hindsight | 3 | 2.5 | kick_lift_closed | 5 / 136 | 3.7% | - | - |
| exp-03b-kick-duration-alarm-8 | alarm-8 | hindsight | 1 | 2.5 | kick | 18 / 136 | 13.2% | - | - |
| exp-03b2-kick-duration6-alarm-8 | alarm-8 | hindsight | 6 | 2.5 | kick | 18 / 136 | 13.2% | - | - |
| exp-03c-kick-timing-alarm-16 | alarm-16 | hindsight | 3 | 2.5 | kick | 24 / 136 | 17.6% | - | - |
| exp-03c-kick-timing-alarm-16 | alarm-16 | hindsight | 3 | 2.5 | native | 4 / 136 | 2.9% | - | - |
| exp-03c2-kick-timing-alarm-4 | alarm-4 | hindsight | 3 | 2.5 | kick | 15 / 136 | 11.0% | - | - |
| exp-03c2-kick-timing-alarm-4 | alarm-4 | hindsight | 3 | 2.5 | native | 4 / 136 | 2.9% | - | - |
| exp-03d-online-v82-kick | online | online | 3 | 2.5 | kick | 37 / 172 | 21.5% | 152 / 168 | 55.6% |
| exp-03d-online-v82-kick | online | online | 3 | 2.5 | native | 22 / 172 | 12.8% | 155 / 168 | 52.1% |
| exp-03e-proswap-alarm-8 | alarm-8 | hindsight | 3 | 2.5 | native | 0 / 74 | 0.0% | - | - |
| exp-03e-proswap-alarm-8 | alarm-8 | hindsight | 3 | 2.5 | random | 0 / 74 | 0.0% | - | - |
| exp-03e-proswap-alarm-8 | alarm-8 | hindsight | 3 | 2.5 | kick | 0 / 74 | 0.0% | - | - |
| exp-03e-proswap-alarm-8 | alarm-8 | hindsight | 3 | 2.5 | kick_then_motion | 0 / 74 | 0.0% | - | - |
| exp-04a-fixed-q16-all | 16 | hindsight | 3 | 2.5 | native | 2 / 172 | 1.2% | 161 / 168 | 47.9% |
| exp-04a-fixed-q16-all | 16 | hindsight | 3 | 2.5 | kick | 29 / 172 | 16.9% | 130 / 168 | 46.8% |
| exp-04b-fixed-q12-all | 12 | hindsight | 3 | 2.5 | native | 5 / 172 | 2.9% | 162 / 168 | 49.1% |
| exp-04b-fixed-q12-all | 12 | hindsight | 3 | 2.5 | kick | 24 / 172 | 14.0% | 88 / 168 | 32.9% |
| exp-04c-fixed-q24-all | 24 | hindsight | 3 | 2.5 | native | 0 / 172 | 0.0% | 98 / 100 | 36.0% |
| exp-04c-fixed-q24-all | 24 | hindsight | 3 | 2.5 | kick | 11 / 172 | 6.4% | 82 / 100 | 34.2% |
| exp-05a-online-routestep-kick | online | online | 1 | 2.5 | kick | 29 / 172 | 16.9% | 148 / 168 | 52.1% |
| exp-05c-online-v82-kick-retrigger | online | online | 1 | 2.5 | kick | 29 / 172 | 16.9% | 150 / 168 | 52.6% |
| exp-06a-online-diam110-kick | online | online | 1 | 2.5 | kick | 34 / 172 | 19.8% | 149 / 168 | 53.8% |
| exp-06b-online-diam115-kick | online | online | 1 | 2.5 | kick | 31 / 172 | 18.0% | 147 / 168 | 52.4% |
| exp-06c-online-diam110-or-v82-kick | online | online | 1 | 2.5 | kick | 40 / 172 | 23.3% | 145 / 168 | 54.4% |
| exp-06d-online-diam110-kick-retrigger | online | online | 1 | 2.5 | kick | 29 / 172 | 16.9% | 143 / 168 | 50.6% |
| exp-07a-online-diam110-kickifopen | online | online | 1 | 2.5 | kick_if_open | 28 / 172 | 16.3% | 152 / 168 | 52.9% |
| exp-07b-online-diam115-kickifopen | online | online | 1 | 2.5 | kick_if_open | 26 / 172 | 15.1% | 151 / 168 | 52.1% |
| exp-07c-online-v82-kickifopen | online | online | 1 | 2.5 | kick_if_open | 32 / 172 | 18.6% | 151 / 168 | 53.8% |
| exp-07d-online-diam115-kickifopen-retrigger | online | online | 1 | 2.5 | kick_if_open | 29 / 172 | 16.9% | 152 / 168 | 53.2% |
| exp-08a-confirm-v82-kick-c3-seeds2to5 | online | online | 3 | 2.5 | kick | 72 / 344 | 20.9% | 288 / 336 | 52.9% |
| exp-08a-confirm-v82-kick-c3-seeds2to5 | online | online | 3 | 2.5 | native | 36 / 344 | 10.5% | 311 / 336 | 51.0% |
| exp-08b-confirm-diamorv82-kick-c3-seeds2to5 | online | online | 3 | 2.5 | kick | 73 / 344 | 21.2% | 288 / 336 | 53.1% |
| exp-09a-online-diamorv82-kickifstuck3 | online | online | 3 | 2.5 | kick_if_stuck | 27 / 172 | 15.7% | 149 / 168 | 51.8% |
| exp-09b-online-v82-kickifstuck3 | online | online | 3 | 2.5 | kick_if_stuck | 32 / 172 | 18.6% | 153 / 168 | 54.4% |
| exp-09c-online-diam115-kickifstuck3 | online | online | 3 | 2.5 | kick_if_stuck | 27 / 172 | 15.7% | 151 / 168 | 52.4% |
| exp-10a-plus-confirm-v82-kick-seeds6to11 | online | online | 3 | 2.5 | kick | 8 / 75 | 10.7% | - | - |
| exp-10a-plus-confirm-v82-kick-seeds6to11 | online | online | 3 | 2.5 | native | 2 / 72 | 2.8% | - | - |
| exp-10b-plus-confirm-diamorv82-kick-seeds6to11 | online | online | 3 | 2.5 | kick | 59 / 288 | 20.5% | 115 / 132 | 41.4% |
| exp-11a-hindsight-alarm-8-gentle | alarm-8 | hindsight | 3 | 2.5 | obs_perturb | 11 / 136 | 8.1% | - | - |
| exp-11a-hindsight-alarm-8-gentle | alarm-8 | hindsight | 3 | 2.5 | as_swap_random | 11 / 136 | 8.1% | - | - |
| exp-11a-hindsight-alarm-8-gentle | alarm-8 | hindsight | 3 | 2.5 | as_swap_motion | 10 / 136 | 7.4% | - | - |
| exp-11a-hindsight-alarm-8-gentle | alarm-8 | hindsight | 3 | 2.5 | gain2 | 9 / 136 | 6.6% | - | - |
| exp-11a-hindsight-alarm-8-gentle | alarm-8 | hindsight | 3 | 2.5 | gain3 | 8 / 136 | 5.9% | - | - |
| exp-11a-hindsight-alarm-8-gentle | alarm-8 | hindsight | 3 | 2.5 | native | 7 / 136 | 5.1% | - | - |
| exp-11a-hindsight-alarm-8-gentle | alarm-8 | hindsight | 3 | 2.5 | partial8 | 6 / 136 | 4.4% | - | - |
| exp-11a-hindsight-alarm-8-gentle | alarm-8 | hindsight | 3 | 2.5 | partial9 | 4 / 126 | 3.2% | - | - |
| exp-11b-hindsight-alarm-16-gentle | alarm-16 | hindsight | 3 | 2.5 | kick | 27 / 136 | 19.9% | - | - |
| exp-11b-hindsight-alarm-16-gentle | alarm-16 | hindsight | 3 | 2.5 | gain2 | 16 / 136 | 11.8% | - | - |
| exp-11b-hindsight-alarm-16-gentle | alarm-16 | hindsight | 3 | 2.5 | obs_perturb | 9 / 136 | 6.6% | - | - |
| exp-11b-hindsight-alarm-16-gentle | alarm-16 | hindsight | 3 | 2.5 | partial8 | 8 / 136 | 5.9% | - | - |
| exp-11b-hindsight-alarm-16-gentle | alarm-16 | hindsight | 3 | 2.5 | native | 7 / 136 | 5.1% | - | - |
| exp-12a-online-knnorv82-kick | online | online | 1 | 2.5 | kick | 39 / 172 | 22.7% | 149 / 168 | 55.3% |
| exp-12a-online-knnorv82-kick | online | online | 1 | 2.5 | native | 19 / 172 | 11.0% | 158 / 168 | 52.1% |
| exp-12b-online-knn-kick | online | online | 1 | 2.5 | kick | 29 / 172 | 16.9% | 154 / 168 | 53.8% |
| exp-12c-online-v82-gain2 | online | online | 3 | 2.5 | gain2 | 31 / 172 | 18.0% | 150 / 168 | 53.2% |
| exp-12d-online-v82-partial8 | online | online | 3 | 2.5 | partial8 | 33 / 172 | 19.2% | 150 / 168 | 53.8% |
| exp-12e-online-v82-obsperturb | online | online | 3 | 2.5 | obs_perturb | 30 / 172 | 17.4% | 150 / 168 | 52.9% |
| exp-12f-online-knnorv82-gain2 | online | online | 3 | 2.5 | gain2 | 28 / 172 | 16.3% | 150 / 168 | 52.4% |
| exp-12g-online-knnorv82-obsperturb | online | online | 3 | 2.5 | obs_perturb | 28 / 172 | 16.3% | 149 / 168 | 52.1% |
