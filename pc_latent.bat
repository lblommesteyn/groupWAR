@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > logs\latent.log
python -u latent_skill.py >> logs\latent.log 2>&1
echo finished %date% %time% >> logs\latent.log
