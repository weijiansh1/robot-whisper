# Topology definition grid: outcome discrimination

Episodes: topo-libero10 61 success / 39 failure, topo-plus 25 success / 45 failure
AUROC positive class = eventual success (0.5 chance; <0.5 means higher on failures). |AUROC| = 0.5 + |AUROC - 0.5|.
Task AUROC = mean AUROC within base task (10 tasks), removing per-task base-rate differences.
Only anchors with at least 15 episodes of each outcome are ranked; late anchors keep few successes because successes end early.

## all

Anchors with >= 15 of each outcome: q12, q16, q20, q24

### Baselines (frozen v8.2 monitor)

| anchor | feature | AUROC | task AUROC | n succ / fail |
|---|---|---:|---:|---:|
| q12 | v82_alarm | 0.476 | 0.4784 | 86 / 84 |
| q12 | freeze_score | 0.336 | 0.3405 | 86 / 84 |
| q12 | acceleration_score | 0.394 | 0.3965 | 86 / 84 |
| q12 | periodicity_score | 0.528 | 0.455 | 86 / 84 |
| q16 | v82_alarm | 0.470 | 0.4799 | 86 / 84 |
| q16 | freeze_score | 0.357 | 0.3391 | 86 / 84 |
| q16 | acceleration_score | 0.249 | 0.2718 | 86 / 84 |
| q16 | periodicity_score | 0.581 | 0.5572 | 86 / 84 |
| q20 | v82_alarm | 0.429 | 0.4446 | 79 / 84 |
| q20 | freeze_score | 0.345 | 0.3456 | 79 / 84 |
| q20 | acceleration_score | 0.317 | 0.336 | 79 / 84 |
| q20 | periodicity_score | 0.615 | 0.5315 | 79 / 84 |
| q24 | v82_alarm | 0.415 | 0.4012 | 51 / 84 |
| q24 | freeze_score | 0.227 | 0.3165 | 51 / 84 |
| q24 | acceleration_score | 0.348 | 0.2855 | 51 / 84 |
| q24 | periodicity_score | 0.621 | 0.5917 | 51 / 84 |

### Recurrence-region features, best 20 by task AUROC at scale 1.0 (with worst case over six radius scales)

| matrix | hist | ref | quantile | confirm | anchor | feature | AUROC | task AUROC | min |AUROC| over scales | n succ / fail |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| total|layer3 | 3 | 8 | 0.9 | 2 | q16 | returned | 0.730 | 0.747 | 0.494 | 51 / 84 |
| total|layer3 | 3 | 8 | 0.95 | 2 | q16 | returned | 0.715 | 0.7351 | 0.494 | 51 / 84 |
| total|layer3 | 3 | 8 | 0.99 | 2 | q16 | returned | 0.727 | 0.7232 | 0.494 | 51 / 84 |
| route|layer1 | 2 | 8 | 0.95 | 2 | q16 | outside_fraction | 0.341 | 0.277 | 0.500 | 51 / 84 |
| route|layer1 | 2 | 8 | 0.95 | 3 | q16 | outside_fraction | 0.341 | 0.277 | 0.500 | 51 / 84 |
| route|layer1 | 2 | 8 | 0.95 | 4 | q16 | outside_fraction | 0.341 | 0.277 | 0.500 | 51 / 84 |
| route|layer1 | 2 | 8 | 0.99 | 2 | q16 | outside_fraction | 0.349 | 0.278 | 0.500 | 51 / 84 |
| route|layer1 | 2 | 8 | 0.99 | 3 | q16 | outside_fraction | 0.349 | 0.278 | 0.500 | 51 / 84 |
| route|layer1 | 2 | 8 | 0.99 | 4 | q16 | outside_fraction | 0.349 | 0.278 | 0.500 | 51 / 84 |
| input_total|layer6 | 2 | 12 | 0.95 | 2 | q12 | outside_fraction | 0.697 | 0.711 | 0.417 | 79 / 84 |
| input_total|layer6 | 2 | 12 | 0.95 | 3 | q12 | outside_fraction | 0.697 | 0.711 | 0.417 | 79 / 84 |
| input_total|layer6 | 2 | 12 | 0.95 | 4 | q12 | outside_fraction | 0.697 | 0.711 | 0.417 | 79 / 84 |
| route|layer3 | 3 | 8 | 0.9 | 2 | q12 | outside_fraction | 0.575 | 0.7075 | 0.494 | 79 / 84 |
| route|layer3 | 3 | 8 | 0.9 | 3 | q12 | outside_fraction | 0.575 | 0.7075 | 0.494 | 79 / 84 |
| route|layer3 | 3 | 8 | 0.9 | 4 | q12 | outside_fraction | 0.575 | 0.7075 | 0.494 | 79 / 84 |
| total|back_path | 2 | 12 | 0.99 | 2 | q16 | outside_fraction | 0.555 | 0.7067 | 0.445 | 51 / 84 |
| total|back_path | 2 | 12 | 0.99 | 3 | q16 | outside_fraction | 0.555 | 0.7067 | 0.445 | 51 / 84 |
| total|back_path | 2 | 12 | 0.99 | 4 | q16 | outside_fraction | 0.555 | 0.7067 | 0.445 | 51 / 84 |
| route|layer1 | 2 | 8 | 0.9 | 2 | q16 | outside_fraction | 0.332 | 0.2946 | 0.500 | 51 / 84 |
| route|layer1 | 2 | 8 | 0.9 | 3 | q16 | outside_fraction | 0.332 | 0.2946 | 0.500 | 51 / 84 |

Region settings ranked: 23760; same side of 0.5 at all six scales: 0 (0.0%); task AUROC >= 0.65 and every scale >= 0.55: 0.

### Persistence and simple distance features, best 20 by task AUROC

| matrix | history | anchor | feature | AUROC | task AUROC | n succ / fail |
|---|---:|---|---|---:|---:|---:|
| input|layer6 |  | q16 | net_path | 0.233 | 0.1354 | 86 / 84 |
| input|back_last |  | q20 | diameter | 0.848 | 0.8639 | 79 / 84 |
| input|back_path |  | q16 | net_path | 0.235 | 0.1391 | 86 / 84 |
| shared|layer5 |  | q20 | diameter | 0.826 | 0.8599 | 79 / 84 |
| input|layer5 |  | q16 | net_path | 0.236 | 0.1416 | 86 / 84 |
| shared|layer3 | 3 | q24 | h1_lifetime | 0.774 | 0.8477 | 51 / 84 |
| input|layer6 |  | q20 | diameter | 0.823 | 0.8476 | 79 / 84 |
| input|layer7 |  | q16 | net_path | 0.232 | 0.1539 | 86 / 84 |
| shared|layer4 |  | q20 | diameter | 0.794 | 0.8438 | 79 / 84 |
| input|back_path |  | q20 | diameter | 0.827 | 0.8428 | 79 / 84 |
| total|full_path |  | q24 | diameter | 0.793 | 0.8393 | 51 / 84 |
| input|layer5 |  | q20 | diameter | 0.823 | 0.8392 | 79 / 84 |
| input|layer4 |  | q16 | net_path | 0.240 | 0.161 | 86 / 84 |
| input_total|back_path |  | q16 | net_path | 0.234 | 0.1617 | 86 / 84 |
| shared|back_last |  | q24 | net_path | 0.240 | 0.1637 | 51 / 84 |
| input|back_last |  | q16 | net_path | 0.214 | 0.164 | 86 / 84 |
| shared|full_path | 3 | q16 | h1_lifetime | 0.729 | 0.8358 | 86 / 84 |
| input|layer4 |  | q20 | diameter | 0.832 | 0.8353 | 79 / 84 |
| input|back_last | 5 | q24 | h0_largest | 0.814 | 0.8343 | 51 / 84 |
| shared|layer3 |  | q20 | lag4 | 0.832 | 0.8331 | 79 / 84 |

Persistence (H1/H0) features alone: best task AUROC 0.848, best pooled |AUROC| 0.848 over 1800 rows.

## topo-libero10

Anchors with >= 15 of each outcome: q12, q16, q20, q24

### Baselines (frozen v8.2 monitor)

| anchor | feature | AUROC | task AUROC | n succ / fail |
|---|---|---:|---:|---:|
| q12 | v82_alarm | 0.474 | 0.4815 | 61 / 39 |
| q12 | freeze_score | 0.361 | 0.3857 | 61 / 39 |
| q12 | acceleration_score | 0.369 | 0.3174 | 61 / 39 |
| q12 | periodicity_score | 0.587 | 0.5188 | 61 / 39 |
| q16 | v82_alarm | 0.483 | 0.4954 | 61 / 39 |
| q16 | freeze_score | 0.403 | 0.4054 | 61 / 39 |
| q16 | acceleration_score | 0.201 | 0.206 | 61 / 39 |
| q16 | periodicity_score | 0.576 | 0.5062 | 61 / 39 |
| q20 | v82_alarm | 0.445 | 0.4611 | 57 / 39 |
| q20 | freeze_score | 0.411 | 0.4066 | 57 / 39 |
| q20 | acceleration_score | 0.252 | 0.3236 | 57 / 39 |
| q20 | periodicity_score | 0.606 | 0.5225 | 57 / 39 |
| q24 | v82_alarm | 0.437 | 0.4115 | 35 / 39 |
| q24 | freeze_score | 0.245 | 0.3351 | 35 / 39 |
| q24 | acceleration_score | 0.349 | 0.2517 | 35 / 39 |
| q24 | periodicity_score | 0.580 | 0.5208 | 35 / 39 |

### Recurrence-region features, best 20 by task AUROC at scale 1.0 (with worst case over six radius scales)

| matrix | hist | ref | quantile | confirm | anchor | feature | AUROC | task AUROC | min |AUROC| over scales | n succ / fail |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| total|state_back | 3 | 12 | 0.95 | 2 | q16 | outside_fraction | 0.637 | 0.7552 | 0.500 | 35 / 39 |
| total|state_back | 3 | 12 | 0.95 | 3 | q16 | outside_fraction | 0.637 | 0.7552 | 0.500 | 35 / 39 |
| total|state_back | 3 | 12 | 0.95 | 4 | q16 | outside_fraction | 0.637 | 0.7552 | 0.500 | 35 / 39 |
| shared|state_back | 2 | 16 | 0.9 | 2 | q16 | outside_fraction | 0.605 | 0.7448 | 0.437 | 35 / 39 |
| shared|state_back | 2 | 16 | 0.9 | 3 | q16 | outside_fraction | 0.605 | 0.7448 | 0.437 | 35 / 39 |
| shared|state_back | 2 | 16 | 0.9 | 4 | q16 | outside_fraction | 0.605 | 0.7448 | 0.437 | 35 / 39 |
| total|state_back | 3 | 12 | 0.99 | 2 | q16 | outside_fraction | 0.657 | 0.7387 | 0.500 | 35 / 39 |
| total|state_back | 3 | 12 | 0.99 | 3 | q16 | outside_fraction | 0.657 | 0.7387 | 0.500 | 35 / 39 |
| total|state_back | 3 | 12 | 0.99 | 4 | q16 | outside_fraction | 0.657 | 0.7387 | 0.500 | 35 / 39 |
| route|layer0 | 5 | 8 | 0.9 | 3 | q16 | exit_confirmed | 0.262 | 0.2656 | 0.500 | 35 / 39 |
| total|state_back | 2 | 16 | 0.9 | 2 | q16 | outside_fraction | 0.559 | 0.7309 | 0.492 | 35 / 39 |
| total|state_back | 2 | 16 | 0.9 | 3 | q16 | outside_fraction | 0.559 | 0.7309 | 0.492 | 35 / 39 |
| total|state_back | 2 | 16 | 0.9 | 4 | q16 | outside_fraction | 0.559 | 0.7309 | 0.492 | 35 / 39 |
| input_total|state_back | 2 | 12 | 0.9 | 4 | q12 | final_outside | 0.679 | 0.7301 | 0.449 | 57 / 39 |
| total|state_back | 3 | 12 | 0.9 | 2 | q16 | outside_fraction | 0.648 | 0.73 | 0.500 | 35 / 39 |
| total|state_back | 3 | 12 | 0.9 | 3 | q16 | outside_fraction | 0.648 | 0.73 | 0.500 | 35 / 39 |
| total|state_back | 3 | 12 | 0.9 | 4 | q16 | outside_fraction | 0.648 | 0.73 | 0.500 | 35 / 39 |
| route|full_path | 2 | 16 | 0.99 | 2 | q16 | outside_fraction | 0.224 | 0.276 | 0.500 | 35 / 39 |
| route|full_path | 2 | 16 | 0.99 | 3 | q16 | outside_fraction | 0.224 | 0.276 | 0.500 | 35 / 39 |
| route|full_path | 2 | 16 | 0.99 | 4 | q16 | outside_fraction | 0.224 | 0.276 | 0.500 | 35 / 39 |

Region settings ranked: 23760; same side of 0.5 at all six scales: 0 (0.0%); task AUROC >= 0.65 and every scale >= 0.55: 0.

### Persistence and simple distance features, best 20 by task AUROC

| matrix | history | anchor | feature | AUROC | task AUROC | n succ / fail |
|---|---:|---|---|---:|---:|---:|
| shared|back_last |  | q24 | net_path | 0.246 | 0.1406 | 35 / 39 |
| shared|full_path | 3 | q16 | h1_lifetime | 0.747 | 0.8418 | 61 / 39 |
| input|back_last | 3 | q24 | h0_largest | 0.724 | 0.8385 | 35 / 39 |
| route|layer5 |  | q16 | net_path | 0.256 | 0.1644 | 61 / 39 |
| shared|full_path | 3 | q16 | h1_normalized | 0.736 | 0.8325 | 61 / 39 |
| shared|layer2 |  | q20 | lag4 | 0.787 | 0.8322 | 57 / 39 |
| shared|layer3 | 3 | q24 | h1_lifetime | 0.750 | 0.8299 | 35 / 39 |
| shared|layer5 |  | q20 | step_std | 0.824 | 0.8292 | 57 / 39 |
| input|layer7 | 2 | q24 | h0_largest | 0.744 | 0.8247 | 35 / 39 |
| shared|back_last |  | q20 | lag2 | 0.712 | 0.8202 | 57 / 39 |
| shared|layer4 |  | q16 | diameter | 0.794 | 0.8179 | 61 / 39 |
| shared|layer2 | 3 | q24 | h1_lifetime | 0.781 | 0.8177 | 35 / 39 |
| shared|layer7 | 3 | q24 | h0_largest | 0.804 | 0.8177 | 35 / 39 |
| shared|layer4 | 3 | q16 | h1_normalized | 0.747 | 0.8174 | 61 / 39 |
| shared|full_path |  | q20 | lag4 | 0.786 | 0.8165 | 57 / 39 |
| shared|state_back |  | q20 | lag4 | 0.702 | 0.8165 | 57 / 39 |
| shared|layer2 | 3 | q24 | h1_normalized | 0.774 | 0.8142 | 35 / 39 |
| input|layer4 |  | q20 | diameter | 0.819 | 0.813 | 57 / 39 |
| total|full_path |  | q24 | diameter | 0.764 | 0.8125 | 35 / 39 |
| input|full_path | 3 | q24 | h0_largest | 0.725 | 0.8125 | 35 / 39 |

Persistence (H1/H0) features alone: best task AUROC 0.842, best pooled |AUROC| 0.845 over 1800 rows.

## topo-plus

Anchors with >= 15 of each outcome: q12, q16, q20, q24

### Baselines (frozen v8.2 monitor)

| anchor | feature | AUROC | task AUROC | n succ / fail |
|---|---|---:|---:|---:|
| q12 | v82_alarm | 0.478 | 0.475 | 25 / 45 |
| q12 | freeze_score | 0.291 | 0.2896 | 25 / 45 |
| q12 | acceleration_score | 0.420 | 0.4854 | 25 / 45 |
| q12 | periodicity_score | 0.484 | 0.3833 | 25 / 45 |
| q16 | v82_alarm | 0.456 | 0.4625 | 25 / 45 |
| q16 | freeze_score | 0.293 | 0.2646 | 25 / 45 |
| q16 | acceleration_score | 0.291 | 0.3458 | 25 / 45 |
| q16 | periodicity_score | 0.602 | 0.6146 | 25 / 45 |
| q20 | v82_alarm | 0.411 | 0.426 | 22 / 45 |
| q20 | freeze_score | 0.282 | 0.2771 | 22 / 45 |
| q20 | acceleration_score | 0.378 | 0.35 | 22 / 45 |
| q20 | periodicity_score | 0.631 | 0.5417 | 22 / 45 |
| q24 | v82_alarm | 0.389 | 0.3875 | 16 / 45 |
| q24 | freeze_score | 0.214 | 0.2917 | 16 / 45 |
| q24 | acceleration_score | 0.353 | 0.3306 | 16 / 45 |
| q24 | periodicity_score | 0.664 | 0.6861 | 16 / 45 |

### Recurrence-region features, best 20 by task AUROC at scale 1.0 (with worst case over six radius scales)

| matrix | hist | ref | quantile | confirm | anchor | feature | AUROC | task AUROC | min |AUROC| over scales | n succ / fail |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| route|layer1 | 2 | 8 | 0.99 | 2 | q16 | outside_fraction | 0.301 | 0.1069 | 0.500 | 16 / 45 |
| route|layer1 | 2 | 8 | 0.99 | 3 | q16 | outside_fraction | 0.301 | 0.1069 | 0.500 | 16 / 45 |
| route|layer1 | 2 | 8 | 0.99 | 4 | q16 | outside_fraction | 0.301 | 0.1069 | 0.500 | 16 / 45 |
| route|layer1 | 2 | 8 | 0.95 | 2 | q16 | outside_fraction | 0.328 | 0.1208 | 0.500 | 16 / 45 |
| route|layer1 | 2 | 8 | 0.95 | 3 | q16 | outside_fraction | 0.328 | 0.1208 | 0.500 | 16 / 45 |
| route|layer1 | 2 | 8 | 0.95 | 4 | q16 | outside_fraction | 0.328 | 0.1208 | 0.500 | 16 / 45 |
| route|layer1 | 2 | 8 | 0.9 | 2 | q16 | outside_fraction | 0.301 | 0.1389 | 0.500 | 16 / 45 |
| route|layer1 | 2 | 8 | 0.9 | 3 | q16 | outside_fraction | 0.301 | 0.1389 | 0.500 | 16 / 45 |
| route|layer1 | 2 | 8 | 0.9 | 4 | q16 | outside_fraction | 0.301 | 0.1389 | 0.500 | 16 / 45 |
| route|layer2 | 2 | 8 | 0.9 | 2 | q16 | outside_fraction | 0.369 | 0.1403 | 0.500 | 16 / 45 |
| route|layer2 | 2 | 8 | 0.9 | 3 | q16 | outside_fraction | 0.369 | 0.1403 | 0.500 | 16 / 45 |
| route|layer2 | 2 | 8 | 0.9 | 4 | q16 | outside_fraction | 0.369 | 0.1403 | 0.500 | 16 / 45 |
| route|layer2 | 2 | 8 | 0.99 | 2 | q16 | outside_fraction | 0.315 | 0.1458 | 0.500 | 16 / 45 |
| route|layer2 | 2 | 8 | 0.99 | 3 | q16 | outside_fraction | 0.315 | 0.1458 | 0.500 | 16 / 45 |
| route|layer2 | 2 | 8 | 0.99 | 4 | q16 | outside_fraction | 0.315 | 0.1458 | 0.500 | 16 / 45 |
| route|layer2 | 2 | 8 | 0.95 | 2 | q16 | outside_fraction | 0.347 | 0.1542 | 0.500 | 16 / 45 |
| route|layer2 | 2 | 8 | 0.95 | 3 | q16 | outside_fraction | 0.347 | 0.1542 | 0.500 | 16 / 45 |
| route|layer2 | 2 | 8 | 0.95 | 4 | q16 | outside_fraction | 0.347 | 0.1542 | 0.500 | 16 / 45 |
| route|back_last | 3 | 8 | 0.95 | 3 | q16 | exit_confirmed | 0.684 | 0.8097 | 0.500 | 16 / 45 |
| route|back_last | 3 | 8 | 0.99 | 2 | q16 | outside_fraction | 0.689 | 0.8083 | 0.500 | 16 / 45 |

Region settings ranked: 23760; same side of 0.5 at all six scales: 0 (0.0%); task AUROC >= 0.65 and every scale >= 0.55: 0.

### Persistence and simple distance features, best 20 by task AUROC

| matrix | history | anchor | feature | AUROC | task AUROC | n succ / fail |
|---|---:|---|---|---:|---:|---:|
| shared|layer0 |  | q20 | lag1 | 0.878 | 0.9479 | 22 / 45 |
| input|layer7 |  | q16 | net_path | 0.186 | 0.0563 | 25 / 45 |
| input|back_path |  | q16 | net_path | 0.197 | 0.0583 | 25 / 45 |
| input|layer5 |  | q16 | net_path | 0.196 | 0.0583 | 25 / 45 |
| input|layer6 |  | q16 | net_path | 0.195 | 0.0583 | 25 / 45 |
| total|layer0 |  | q20 | lag1 | 0.882 | 0.9354 | 22 / 45 |
| shared|full_path |  | q16 | net_path | 0.212 | 0.0729 | 25 / 45 |
| input|back_last |  | q20 | diameter | 0.866 | 0.925 | 22 / 45 |
| shared|layer2 |  | q20 | lag1 | 0.867 | 0.925 | 22 / 45 |
| shared|layer4 | 5 | q16 | h1_normalized | 0.789 | 0.9229 | 25 / 45 |
| input_total|back_path |  | q16 | net_path | 0.198 | 0.0833 | 25 / 45 |
| input_total|back_path |  | q12 | diameter | 0.811 | 0.9146 | 25 / 45 |
| shared|layer5 |  | q20 | diameter | 0.855 | 0.9146 | 22 / 45 |
| input|back_last |  | q16 | net_path | 0.170 | 0.0875 | 25 / 45 |
| input_total|layer7 |  | q16 | net_path | 0.180 | 0.0875 | 25 / 45 |
| input|back_path | 5 | q16 | h1_normalized | 0.817 | 0.9104 | 25 / 45 |
| shared|layer4 | 5 | q16 | h1_lifetime | 0.790 | 0.9104 | 25 / 45 |
| input|back_last | 5 | q24 | h0_largest | 0.821 | 0.9097 | 16 / 45 |
| input|layer4 |  | q16 | net_path | 0.193 | 0.0917 | 25 / 45 |
| input|layer4 | 5 | q16 | h1_lifetime | 0.834 | 0.9083 | 25 / 45 |

Persistence (H1/H0) features alone: best task AUROC 0.923, best pooled |AUROC| 0.902 over 1800 rows.
