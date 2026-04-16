#!/bin/bash
#SBATCH --job-name=nba_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4

module load gcc arrow python/3.12
source ~/venv_war/bin/activate

cd /scratch/lblommes/nba_war
mkdir -p logs models
python nba_train.py
