#!/bin/bash
source /home/swj/data/env.sh >/dev/null 2>&1; cd /home/swj/data/libero-runtime
while [ ! -f controls/exp-03e-proswap-alarm-8/t1 ]; do sleep 20; done
run() { name=$1; shift; mkdir -p controls/$name; date +%s > controls/$name/t0; python3 run_control_experiment.py --ports 9570,9571,9572,9573,9574,9575,9576,9577 --clients-per-server 16 --noise-scale 2.5 --candidates 8 "$@" --out controls/$name > controls/$name/driver.log 2>&1; date +%s > controls/$name/t1; }
run exp-04a-fixed-q16-all --parents controls/parents-topo-all-i.json --branch 16 --include-successes --control-queries 3 --seeds 0,1 --strategies native,kick
run exp-04b-fixed-q12-all --parents controls/parents-topo-all-i.json --branch 12 --include-successes --control-queries 3 --seeds 0,1 --strategies native,kick
run exp-04c-fixed-q24-all --parents controls/parents-topo-all-i.json --branch 24 --include-successes --control-queries 3 --seeds 0,1 --strategies native,kick
