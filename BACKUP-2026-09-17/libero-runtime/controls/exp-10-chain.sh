#!/bin/bash
source /home/swj/data/env.sh >/dev/null 2>&1; cd /home/swj/data/libero-runtime
run() { name=$1; shift; mkdir -p controls/$name; date +%s > controls/$name/t0; python3 run_control_experiment.py --ports 9570,9571,9572,9573,9574,9575,9576,9577 --clients-per-server 16 --noise-scale 2.5 --candidates 8 "$@" --out controls/$name > controls/$name/driver.log 2>&1; date +%s > controls/$name/t1; }
run exp-10a-plus-confirm-v82-kick-seeds6to11 --parents controls/parents-topo-all-i.json --benchmarks plus --branch online --trigger v82 --include-successes --control-queries 3 --seeds 6,7,8,9,10,11 --strategies native,kick
run exp-10b-plus-confirm-diamorv82-kick-seeds6to11 --parents controls/parents-topo-all-i.json --benchmarks plus --branch online --trigger diam_or_v82 --include-successes --control-queries 3 --seeds 6,7,8,9,10,11 --strategies kick --extra-args "--diam-threshold 1.10"
