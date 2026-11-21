@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > logs\roster.log
python -u roster_search.py >> logs\roster.log 2>&1
echo finished %date% %time% >> logs\roster.log
