@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > logs\benchmark.log
python -u benchmark_public.py >> logs\benchmark.log 2>&1
echo finished %date% %time% >> logs\benchmark.log
