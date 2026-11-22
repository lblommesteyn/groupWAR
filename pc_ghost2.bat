@echo off
cd /d C:\Users\LukeB\dev\court-epv
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > ghost2.log
python -u epv_ghost2.py 2 >> ghost2.log 2>&1
echo finished %date% %time% >> ghost2.log
