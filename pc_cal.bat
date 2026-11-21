@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set ROLLING_OUT=rolling_cal
echo started %date% %time% > logs\cal.log
python -u rolling_prior.py >> logs\cal.log 2>&1
echo finished %date% %time% >> logs\cal.log
