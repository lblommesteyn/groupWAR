@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > logs\chain.log
python -u fetch_pbp.py >> logs\chain.log 2>&1
python -u ms_stints.py results\latent\pbp_2325.parquet results\latent\games_2325.parquet ms_stints_2325.pkl >> logs\chain.log 2>&1
set SEASONS=2025-26:2026
set TAG=2526
python -u fetch_pbp.py >> logs\chain.log 2>&1
python -u ms_stints.py results\latent\pbp_2526.parquet results\latent\games_2526.parquet ms_stints_2526.pkl >> logs\chain.log 2>&1
set SEASONS=
set TAG=
python -u ms_latent_2526.py >> logs\chain.log 2>&1
echo finished %date% %time% >> logs\chain.log
