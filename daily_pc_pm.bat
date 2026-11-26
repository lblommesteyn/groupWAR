@echo off
rem Evening run: latest injury report before the 7pm tips, same-day ratings reused.
cd /d C:\Users\LukeB\dev\court-live
git pull -q >> ..\court-model\logs\daily.log 2>&1
cd /d C:\Users\LukeB\dev\court-model
echo ==== pm %date% %time% >> logs\daily.log
python live_forecast.py --run pm --live C:\Users\LukeB\dev\court-live >> logs\daily.log 2>&1
cd /d C:\Users\LukeB\dev\court-live
git add forecasts data/darko data/injury
git -c user.name=lblommesteyn -c user.email=lblommesteyn@icloud.com commit -q -m "forecast pm %date%" >> ..\court-model\logs\daily.log 2>&1
git pull --rebase -q >> ..\court-model\logs\daily.log 2>&1
git push -q >> ..\court-model\logs\daily.log 2>&1
