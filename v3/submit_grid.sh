#!/usr/bin/env bash
# Main search grid + baselines + sensitivity (FIFO on the GPU queue).
set -e
cd "$(dirname "$0")/.."
sub() { pcslurm submit --shared --name "$1" -- "${@:2}" | grep -E "Submitted"; }
for ct in "paris USA" "paris Canada" "4nf CAN"; do
  set -- $ct; c=$1; t=$2
  sub "g-$t-main"  python v3/run_case.py --case $c --team $t --tag main --seeds 3 --audit --out main
  sub "g-$t-base"  python v3/baselines.py --case $c --team $t --tag main --search-out main
  sub "g-$t-unl"   python v3/run_case.py --case $c --team $t --tag main --seeds 3 --no-locks --out unlocked
done
for ct in "paris USA" "paris Canada" "4nf CAN"; do
  set -- $ct; c=$1; t=$2
  sub "s-$t-prior" python v3/run_case.py --case $c --team $t --tag main --seeds 1 --mode prior --out s_prior
  sub "s-$t-pre"   python v3/run_case.py --case $c --team $t --tag main_pre --seeds 1 --out s_pretrained
  sub "s-$t-noq"   python v3/run_case.py --case $c --team $t --tag main --seeds 1 --no-quota --out s_noquota
  sub "s-$t-lam"   python v3/run_case.py --case $c --team $t --tag main --seeds 1 --lam 0.2 --out s_lam02
done
