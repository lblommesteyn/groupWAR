@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set REB=1
set ROLLING_OUT=rolling_reb
echo started %date% %time% > logs\reb_rolling.log
python -u rolling_prior.py >> logs\reb_rolling.log 2>&1
echo finished %date% %time% >> logs\reb_rolling.log
