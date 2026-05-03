#!/bin/bash
#SBATCH --job-name=hockey_war
#SBATCH --output=logs/search_%j.out
#SBATCH --error=logs/search_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4

module load gcc arrow python/3.12
source ~/venv_war/bin/activate

cd /scratch/lblommes/groupWAR
mkdir -p logs

python -u run_search.py
