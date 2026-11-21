@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set EPOCHS=40
set TRAIN_BEFORE=2026-04-01
echo started %date% %time% > logs\mae_id.log
python -u jepa.py mae_id >> logs\mae_id.log 2>&1
echo finished %date% %time% >> logs\mae_id.log
