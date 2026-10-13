#!/usr/bin/env bash
# Build NHL data (CPU) then queue the NHL pipeline on the GPU (jobs run FIFO).
set -e
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=-1 python v3/nhl_build.py > v3/data/nhl/build.log 2>&1
sub() { pcslurm submit --shared --name "$1" -- "${@:2}" | grep -E "Submitted|Log"; }
sub nhl-main      python v3/train.py --sport nhl --tag main
sub nhl-hold      python v3/train.py --sport nhl --tag holdout --test-season 2025
sub nhl-hold-lag1 python v3/train.py --sport nhl --tag holdout_lag1 --test-season 2025 --lag 1
sub nhl-hold-shuf python v3/train.py --sport nhl --tag holdout_shuf --test-season 2025 --shuffle-graph
sub nhl-E1        python v3/eval_scorer.py --sport nhl --test-season 2025 --tags holdout,holdout_lag1,holdout_shuf
