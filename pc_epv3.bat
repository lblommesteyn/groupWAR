@echo off
cd /d C:\Users\LukeB\dev\court-epv
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > epv3.log
python -u epv3_train.py --mode split >> epv3.log 2>&1
python -u epv3_train.py --mode cf --epochs 3 >> epv3.log 2>&1
echo finished %date% %time% >> epv3.log
