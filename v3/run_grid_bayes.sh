#!/usr/bin/env bash
# Main search grid with the Bayesian player-value scorer (bayes.py), plus baselines
# and sensitivity runs. CPU only. Needs `python v3/bayes.py fit` caches (built on demand).
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=-1
run() { echo "== $*"; python "$@" 2>&1 | grep -v Warning | tail -n 3; }
B="--scorer bayes"
IFS=, read -ra CASES <<< "${CASES:-paris USA,paris Canada,4nf CAN}"
for ct in "${CASES[@]}"; do
  set -- $ct; c=$1; t=$2
  run v3/run_case.py --case $c --team $t $B --seeds 3 --audit --out bayes_main
  run v3/baselines.py --case $c --team $t $B --search-out bayes_main
  run v3/run_case.py --case $c --team $t $B --seeds 3 --no-locks --out bayes_unlocked
  run v3/run_case.py --case $c --team $t $B --seeds 1 --order club --out bayes_s_club
  run v3/run_case.py --case $c --team $t $B --seeds 1 --no-quota --out bayes_s_noquota
  for k in 0 1 2; do
    run v3/run_case.py --case $c --team $t $B --seeds 1 --drop-opp $k --out bayes_s_dropopp$k
  done
done
