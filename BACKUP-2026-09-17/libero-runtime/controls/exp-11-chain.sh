#!/bin/bash
source /home/swj/data/env.sh >/dev/null 2>&1; cd /home/swj/data/libero-runtime
run() { name=$1; shift; mkdir -p controls/$name; date +%s > controls/$name/t0; python3 run_control_experiment.py --ports 9570,9571,9572,9573,9574,9575,9576,9577 --clients-per-server 8 --noise-scale 2.5 --candidates 8 "$@" --out controls/$name > controls/$name/driver.log 2>&1; date +%s > controls/$name/t1; }
run exp-11a-hindsight-alarm-8-gentle --parents controls/parents-topo-all-i.json --branch alarm-8 --control-queries 3 --seeds 0,1 --strategies native,gain2,gain3,partial8,partial9,obs_perturb,as_swap_motion,as_swap_random
run exp-11b-hindsight-alarm-16-gentle --parents controls/parents-topo-all-i.json --branch alarm-16 --control-queries 3 --seeds 0,1 --strategies native,gain2,partial8,obs_perturb,kick
