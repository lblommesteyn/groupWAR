@echo off
cd /d C:\Users\LukeB\dev\court-epv
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > ghost.log
python -u epv_ghost.py 2 >> ghost.log 2>&1
echo finished %date% %time% >> ghost.log
