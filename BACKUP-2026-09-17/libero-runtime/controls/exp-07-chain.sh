#!/bin/bash
source /home/swj/data/env.sh >/dev/null 2>&1; cd /home/swj/data/libero-runtime
run() { name=$1; shift; mkdir -p controls/$name; date +%s > controls/$name/t0; python3 run_control_experiment.py --ports 9570,9571,9572,9573,9574,9575,9576,9577 --clients-per-server 12 --noise-scale 2.5 --candidates 8 "$@" --out controls/$name > controls/$name/driver.log 2>&1; date +%s > controls/$name/t1; }
run exp-07a-online-diam110-kickifopen --parents controls/parents-topo-all-i.json --branch online --trigger diam --include-successes --control-queries 1 --seeds 0,1 --strategies kick_if_open --extra-args "--diam-threshold 1.10"
run exp-07b-online-diam115-kickifopen --parents controls/parents-topo-all-i.json --branch online --trigger diam --include-successes --control-queries 1 --seeds 0,1 --strategies kick_if_open --extra-args "--diam-threshold 1.15"
run exp-07c-online-v82-kickifopen --parents controls/parents-topo-all-i.json --branch online --trigger v82 --include-successes --control-queries 1 --seeds 0,1 --strategies kick_if_open
run exp-07d-online-diam115-kickifopen-retrigger --parents controls/parents-topo-all-i.json --branch online --trigger diam --include-successes --control-queries 1 --seeds 0,1 --strategies kick_if_open --extra-args "--diam-threshold 1.15 --retrigger-cooldown 8"
