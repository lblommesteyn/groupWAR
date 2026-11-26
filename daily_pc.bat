@echo off
rem Daily live forecast: new play-by-play -> stints -> ratings -> tonights forecasts -> push to court-live.
cd /d C:\Users\LukeB\dev\court-model
set SEASONS=2026-27:2027
set TAG=2627
echo ==== %date% %time% >> logs\daily.log
python fetch_pbp.py >> logs\daily.log 2>&1
python ms_stints.py results\latent\pbp_2627.parquet results\latent\games_2627.parquet ms2_stints_2627.pkl >> logs\daily.log 2>&1
cd /d C:\Users\LukeB\dev\court-live
git pull -q >> ..\court-model\logs\daily.log 2>&1
cd /d C:\Users\LukeB\dev\court-model
python live_forecast.py --live C:\Users\LukeB\dev\court-live >> logs\daily.log 2>&1
cd /d C:\Users\LukeB\dev\court-live
git add forecasts data/darko data/injury
git -c user.name=lblommesteyn -c user.email=lblommesteyn@icloud.com commit -q -m "forecast %date%" >> ..\court-model\logs\daily.log 2>&1
git push -q >> ..\court-model\logs\daily.log 2>&1
