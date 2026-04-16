#!/bin/bash
#SBATCH --job-name=nba_process
#SBATCH --output=logs/process_%j.out
#SBATCH --error=logs/process_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

module load gcc arrow python/3.12
source ~/venv_war/bin/activate

cd /scratch/lblommes/nba_war
python nba_process.py
