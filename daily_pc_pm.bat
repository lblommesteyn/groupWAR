@echo off
rem Evening run (17:45 ET): latest injury report before the 7pm tips, same-day ratings reused. Heartbeat on success;
rem exits 1 on failure so Task Scheduler retries.
set PYTHONIOENCODING=utf-8
set FAIL=0
set LIVE=C:\Users\LukeB\dev\court-live
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set D=%%d
cd /d %LIVE%
git pull -q >> C:\Users\LukeB\dev\court-model\logs\daily.log 2>&1
cd /d C:\Users\LukeB\dev\court-model
echo ==== pm %date% %time% >> logs\daily.log
python live_forecast.py --run pm --live %LIVE% >> logs\daily.log 2>&1
if errorlevel 1 set FAIL=1
if not exist %LIVE%\data\heartbeat mkdir %LIVE%\data\heartbeat
if "%FAIL%"=="0" echo pm %date% %time% > %LIVE%\data\heartbeat\%D%-pm.txt
cd /d %LIVE%
git add forecasts data/darko data/injury data/heartbeat
git -c user.name=lblommesteyn -c user.email=lblommesteyn@icloud.com commit -q -m "forecast pm %D%" >> C:\Users\LukeB\dev\court-model\logs\daily.log 2>&1
git pull --rebase -X theirs -q >> C:\Users\LukeB\dev\court-model\logs\daily.log 2>&1
git push -q >> C:\Users\LukeB\dev\court-model\logs\daily.log 2>&1
if errorlevel 1 set FAIL=1
rem alerts: trigger the cloud health check now (GitHub's own schedule is unreliable); a failed run emails the owner
"C:\Program Files\GitHub CLI\gh.exe" workflow run health.yml -R lblommesteyn/court-live -f mode=pm >> C:\Users\LukeB\dev\court-model\logs\daily.log 2>&1
echo ---- pm exit %FAIL% %time% >> C:\Users\LukeB\dev\court-model\logs\daily.log
exit /b %FAIL%
