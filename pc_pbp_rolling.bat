@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set EXTRA_PBP=1
set ROLLING_OUT=rolling_pbp
echo started %date% %time% > logs\pbp_rolling.log
python -u rolling_prior.py >> logs\pbp_rolling.log 2>&1
echo finished %date% %time% >> logs\pbp_rolling.log
