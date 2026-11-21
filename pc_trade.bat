@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > logs\trade.log
python -u trade_test.py >> logs\trade.log 2>&1
echo finished %date% %time% >> logs\trade.log
