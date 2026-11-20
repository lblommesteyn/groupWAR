@echo off
rem 2024-25: pull player totals, rebuild player-seasons, refit the Bayesian prior with 2023-24 and 2024-25 lineup seasons,
rem then rerun the rolling evaluation of the rating backbones with the new prior (separate output folder).
cd /d C:\Users\LukeB\dev\groupWAR\v3
set PYTHONIOENCODING=utf-8
if not exist data\nba\player_seasons_backup_20261120.parquet copy /y data\nba\player_seasons.parquet data\nba\player_seasons_backup_20261120.parquet
echo started %date% %time% > C:\Users\LukeB\dev\court-model\logs\prior.log
python -u -c "import nba_pull_totals as t; t.SEASONS=['2024-25']; t.main()" >> C:\Users\LukeB\dev\court-model\logs\prior.log 2>&1
python -u nba_features.py >> C:\Users\LukeB\dev\court-model\logs\prior.log 2>&1
python -u export_prior_v2.py >> C:\Users\LukeB\dev\court-model\logs\prior.log 2>&1
copy /y results\prior_2026_v2.json C:\Users\LukeB\dev\court-model\results\groupwar_t\prior_2026_v2.json >> C:\Users\LukeB\dev\court-model\logs\prior.log
cd /d C:\Users\LukeB\dev\court-model
set PRIOR_FILE=prior_2026_v2.json
set ROLLING_OUT=rolling_prior_v2
python -u rolling_prior.py >> logs\prior.log 2>&1
echo finished %date% %time% >> logs\prior.log
