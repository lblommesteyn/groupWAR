#!/bin/bash
#SBATCH --job-name=nba_pull
#SBATCH --output=logs/pull_%j.out
#SBATCH --error=logs/pull_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

module load gcc arrow python/3.12
source ~/venv_war/bin/activate

cd /scratch/lblommes/nba_war
python nba_data_pull.py
