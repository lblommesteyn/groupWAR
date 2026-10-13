#!/usr/bin/env bash
# Queue the NBA pipeline on the GPU (jobs run FIFO).
set -e
cd "$(dirname "$0")/.."
python v3/nba_features.py > v3/data/nba/features.log 2>&1
sub() { pcslurm submit --shared --name "$1" -- "${@:2}" | grep -E "Submitted|Log"; }
sub nba-main      python v3/train.py --sport nba --tag main
sub nba-hold      python v3/train.py --sport nba --tag holdout --test-season 2023
sub nba-hold-lag1 python v3/train.py --sport nba --tag holdout_lag1 --test-season 2023 --lag 1
sub nba-hold-shuf python v3/train.py --sport nba --tag holdout_shuf --test-season 2023 --shuffle-graph
sub nba-E1        python v3/eval_scorer.py --sport nba --test-season 2023 --tags holdout,holdout_lag1,holdout_shuf
sub nba-usa-time  python v3/run_case.py --case paris --team USA --tag main --seeds 1 --out timing
