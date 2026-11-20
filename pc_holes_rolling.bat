@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HOLES=1
set ROLLING_OUT=rolling_holes
echo started %date% %time% > logs\holes_rolling.log
python -u rolling_prior.py >> logs\holes_rolling.log 2>&1
echo finished %date% %time% >> logs\holes_rolling.log
