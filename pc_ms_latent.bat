@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > logs\ms_latent.log
python -u ms_latent.py >> logs\ms_latent.log 2>&1
echo finished %date% %time% >> logs\ms_latent.log
