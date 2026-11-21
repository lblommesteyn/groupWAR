@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > logs\fetch.log
python -u fetch_pbp.py >> logs\fetch.log 2>&1
echo finished %date% %time% >> logs\fetch.log
