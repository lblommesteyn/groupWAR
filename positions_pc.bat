@echo off
rem PC backup of the live position capture (every 10 min, 11:00-03:00 ET). Writes outside the repo, so it never
rem conflicts with the GitHub Actions capture; if GitHub misses a game, this copy covers it.
set PYTHONIOENCODING=utf-8
cd /d C:\Users\LukeB\dev\court-live
python collect_positions.py --out C:\Users\LukeB\dev\positions-backup >> C:\Users\LukeB\dev\court-model\logs\positions_pc.log 2>&1
