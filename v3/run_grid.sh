#!/usr/bin/env bash
# Main search grid, baselines and sensitivity runs with the additive value scorer.
# Everything runs on CPU in a few minutes per case (no GPU queue needed).
# The graph-scorer grid (appendix comparison) is submit_grid.sh.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=-1
run() { echo "== $*"; python "$@" 2>&1 | grep -v Warning | tail -n 3; }
for ct in "paris USA" "paris Canada" "4nf CAN"; do
  set -- $ct; c=$1; t=$2
  run v3/run_case.py --case $c --team $t --seeds 3 --audit --out main
  run v3/baselines.py --case $c --team $t --search-out main
  run v3/run_case.py --case $c --team $t --seeds 3 --no-locks --out unlocked
  run v3/run_case.py --case $c --team $t --seeds 1 --order club --out s_club
  run v3/run_case.py --case $c --team $t --seeds 1 --no-quota --out s_noquota
  run v3/run_case.py --case $c --team $t --seeds 1 --decay 0.5 --out s_decay05
  run v3/run_case.py --case $c --team $t --seeds 1 --decay 1.0 --out s_decay10
  for k in 0 1 2; do
    run v3/run_case.py --case $c --team $t --seeds 1 --drop-opp $k --out s_dropopp$k
  done
done
