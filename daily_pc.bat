@echo off
rem Daily live forecast (11:00 ET): new play-by-play -> stints -> ratings -> tonight's forecasts -> push to court-live.
rem Also publishes what the cloud backup needs (ratings, calibration, schedule) and a heartbeat when the forecast
rem step succeeded. Exits 1 on failure so Task Scheduler retries (3 x every 15 min).
cd /d C:\Users\LukeB\dev\court-model
set SEASONS=2026-27:2027
set TAG=2627
set PYTHONIOENCODING=utf-8
set FAIL=0
set LIVE=C:\Users\LukeB\dev\court-live
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set D=%%d
for /f %%d in ('powershell -NoProfile -Command "(Get-Date).AddDays(-1).ToString('yyyy-MM-dd')"') do set Y=%%d
echo ==== %date% %time% >> logs\daily.log
cd /d %LIVE%
git pull -q >> C:\Users\LukeB\dev\court-model\logs\daily.log 2>&1
cd /d C:\Users\LukeB\dev\court-model
rem last night's live positions + companion history -> normalized tracking (outside the repo; summary committed)
python %LIVE%\convert_live.py --date %Y% --root %LIVE% --out C:\Users\LukeB\dev\tracking-live --summary %LIVE%\data\tracking_summary\%Y%.json >> logs\daily.log 2>&1
python fetch_pbp.py >> logs\daily.log 2>&1
python fetch_gamelogs.py 2027 >> logs\daily.log 2>&1
python ms_stints.py results\latent\pbp_2627.parquet results\latent\games_2627.parquet ms2_stints_2627.pkl >> logs\daily.log 2>&1
python publish_schedule.py --live %LIVE% >> logs\daily.log 2>&1
python live_forecast.py --live %LIVE% >> logs\daily.log 2>&1
if errorlevel 1 set FAIL=1
python duncan_score.py --live %LIVE% >> logs\daily.log 2>&1
if not exist %LIVE%\data\ratings mkdir %LIVE%\data\ratings
if exist results\latent\live_ratings_%D%.npz copy /y results\latent\live_ratings_%D%.npz %LIVE%\data\ratings\%D%.npz >nul
copy /y results\latent\live_calib_v2.json %LIVE%\data\ratings\live_calib_v2.json >nul
if not exist %LIVE%\data\heartbeat mkdir %LIVE%\data\heartbeat
if "%FAIL%"=="0" echo am %date% %time% > %LIVE%\data\heartbeat\%D%-am.txt
cd /d %LIVE%
git add forecasts data/darko data/injury data/duncan data/ratings data/schedule data/heartbeat data/tracking_summary
git -c user.name=lblommesteyn -c user.email=lblommesteyn@icloud.com commit -q -m "forecast %D%" >> C:\Users\LukeB\dev\court-model\logs\daily.log 2>&1
git pull --rebase -X theirs -q >> C:\Users\LukeB\dev\court-model\logs\daily.log 2>&1
git push -q >> C:\Users\LukeB\dev\court-model\logs\daily.log 2>&1
if errorlevel 1 set FAIL=1
echo ---- exit %FAIL% %time% >> C:\Users\LukeB\dev\court-model\logs\daily.log
exit /b %FAIL%
