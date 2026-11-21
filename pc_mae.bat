@echo off
cd /d C:\Users\LukeB\dev\court-model
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo started %date% %time% > logs\mae.log
set EPOCHS=40
set TRAIN_BEFORE=2026-04-01
python -u jepa.py mae_long >> logs\mae.log 2>&1
python -u mae_apply.py >> logs\mae.log 2>&1
echo applied %date% %time% >> logs\mae.log
set MAE=1
set ROLLING_OUT=rolling_mae
python -u rolling_prior.py >> logs\mae.log 2>&1
echo finished %date% %time% >> logs\mae.log
