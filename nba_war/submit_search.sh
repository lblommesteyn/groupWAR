#!/bin/bash
#SBATCH --job-name=nba_search
#SBATCH --output=logs/search_%j.out
#SBATCH --error=logs/search_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4

module load gcc arrow python/3.12
source ~/venv_war/bin/activate

cd /scratch/lblommes/nba_war
python nba_search.py
