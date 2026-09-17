#!/bin/bash
source /home/swj/data/env.sh >/dev/null 2>&1; cd /home/swj/data/libero-runtime
run() { name=$1; shift; mkdir -p controls/$name; date +%s > controls/$name/t0; python3 run_control_experiment.py --ports 9570,9571,9572,9573,9574,9575,9576,9577 --clients-per-server 10 --noise-scale 2.5 --candidates 8 "$@" --out controls/$name > controls/$name/driver.log 2>&1; date +%s > controls/$name/t1; }
run exp-12a-online-knnorv82-kick --parents controls/parents-topo-all-i.json --branch online --trigger knn_or_v82 --include-successes --control-queries 1 --seeds 0,1 --strategies native,kick
run exp-12b-online-knn-kick --parents controls/parents-topo-all-i.json --branch online --trigger knn --include-successes --control-queries 1 --seeds 0,1 --strategies kick
run exp-12c-online-v82-gain2 --parents controls/parents-topo-all-i.json --branch online --trigger v82 --include-successes --control-queries 3 --seeds 0,1 --strategies gain2
run exp-12d-online-v82-partial8 --parents controls/parents-topo-all-i.json --branch online --trigger v82 --include-successes --control-queries 3 --seeds 0,1 --strategies partial8
run exp-12e-online-v82-obsperturb --parents controls/parents-topo-all-i.json --branch online --trigger v82 --include-successes --control-queries 3 --seeds 0,1 --strategies obs_perturb
run exp-12f-online-knnorv82-gain2 --parents controls/parents-topo-all-i.json --branch online --trigger knn_or_v82 --include-successes --control-queries 3 --seeds 0,1 --strategies gain2
run exp-12g-online-knnorv82-obsperturb --parents controls/parents-topo-all-i.json --branch online --trigger knn_or_v82 --include-successes --control-queries 3 --seeds 0,1 --strategies obs_perturb
