@echo off
cd /d C:\Users\LukeB\dev\court-epv
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
del /q data\._* labels2\._* 2>nul
echo started %date% %time% > epv2.log
python -u epv2_train.py --mode split --out out2 >> epv2.log 2>&1
python -u epv2_train.py --mode cf --out out2 >> epv2.log 2>&1
echo finished %date% %time% >> epv2.log
