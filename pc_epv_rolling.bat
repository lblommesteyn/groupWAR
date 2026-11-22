@echo off
rem usage: pc_epv_rolling.bat <EPV groups> <out name>
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set EXTRA_PBP=1
set EPV=%1
set ROLLING_OUT=%2
echo started %date% %time% > logs\%2.log
python -u rolling_prior.py >> logs\%2.log 2>&1
echo finished %date% %time% >> logs\%2.log
