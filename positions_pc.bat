@echo off
rem PRIMARY live position capture (every 10 min, 11:00-03:00 ET). GitHub's scheduled capture runs too rarely to rely on
rem (2026-11-27: two runs in eight hours), so the PC captures into data/positions_pc in the court-live clone and
rem pushes whenever something new arrived. GitHub's copy (data/positions) is a second source; the converter merges both.
set PYTHONIOENCODING=utf-8
set LOG=C:\Users\LukeB\dev\court-model\logs\positions_pc.log
echo ---- run %date% %time% >> %LOG%
cd /d C:\Users\LukeB\dev\court-live
python collect_positions.py --out C:\Users\LukeB\dev\court-live\data\positions_pc >> %LOG% 2>&1
git add data/positions_pc >nul 2>&1
git diff --cached --quiet && exit /b 0
git -c user.name=lblommesteyn -c user.email=lblommesteyn@icloud.com commit -q -m "positions (PC) %date% %time%" >> %LOG% 2>&1
git pull --rebase -X theirs -q >> %LOG% 2>&1
git push -q >> %LOG% 2>&1
