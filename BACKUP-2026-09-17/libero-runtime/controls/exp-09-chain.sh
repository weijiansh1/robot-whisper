#!/bin/bash
source /home/swj/data/env.sh >/dev/null 2>&1; cd /home/swj/data/libero-runtime
run() { name=$1; shift; mkdir -p controls/$name; date +%s > controls/$name/t0; python3 run_control_experiment.py --ports 9570,9571,9572,9573,9574,9575,9576,9577 --clients-per-server 12 --noise-scale 2.5 --candidates 8 "$@" --out controls/$name > controls/$name/driver.log 2>&1; date +%s > controls/$name/t1; }
run exp-09a-online-diamorv82-kickifstuck3 --parents controls/parents-topo-all-i.json --branch online --trigger diam_or_v82 --include-successes --control-queries 3 --seeds 0,1 --strategies kick_if_stuck --extra-args "--diam-threshold 1.10 --stuck-threshold 3.0"
run exp-09b-online-v82-kickifstuck3 --parents controls/parents-topo-all-i.json --branch online --trigger v82 --include-successes --control-queries 3 --seeds 0,1 --strategies kick_if_stuck --extra-args "--stuck-threshold 3.0"
run exp-09c-online-diam115-kickifstuck3 --parents controls/parents-topo-all-i.json --branch online --trigger diam --include-successes --control-queries 3 --seeds 0,1 --strategies kick_if_stuck --extra-args "--diam-threshold 1.15 --stuck-threshold 3.0"
