@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set ROLLING_BASE=rolling_cal
set BENCH_OUT=benchmark_cal
echo started %date% %time% > logs\bench_cal.log
python -u benchmark_public.py >> logs\bench_cal.log 2>&1
echo finished %date% %time% >> logs\bench_cal.log
