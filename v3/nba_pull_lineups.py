"""Pull per-team five-man lineup totals (regular season) from api.pbpstats.com.

Lineup rows carry seconds and points for/against, so they extend the stint
data (which ends with 2022-23) to later seasons. The opponent unit is not
known, so these rows enter the ridge regression against an average opponent.
Seasons that overlap the stint data are pulled too, to check that the two
sources give the same player values.

Usage: python v3/nba_pull_lineups.py
"""
import json
import time
from pathlib import Path

from nba_pull_totals import TEAM_IDS, get

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v3" / "data" / "nba" / "lineups"
OUT.mkdir(parents=True, exist_ok=True)
# case-study seasons first; the overlap seasons are only for cross-checking
SEASONS = ["2023-24", "2024-25", "2022-23", "2021-22", "2020-21", "2019-20"]
KEEP = ["EntityId", "TeamId", "SecondsPlayed", "Points", "OpponentPoints", "OffPoss", "DefPoss"]


def query(s, t, extra=""):
    d = get("https://api.pbpstats.com/get-totals/nba?"
            f"Season={s}&SeasonType=Regular%2BSeason&Type=Lineup&TeamId={t}{extra}")
    return [{k: r.get(k) for k in KEEP} for r in d["multi_row_table_data"]]


def pull(s, t):
    """Whole season in one query unless it hits the 500-row cap; then month by
    month (ISO FromDate/ToDate), summing rows of the same lineup."""
    rows = query(s, t)
    if len(rows) < 500:
        return rows
    y0 = int(s[:4])
    months = [(y0, m) for m in range(9, 13)] + [(y0 + 1, m) for m in range(1, 10)]
    acc = {}
    for y, m in months:
        last = 30 if m in (4, 6, 9, 11) else (29 if m == 2 and y % 4 == 0 else 28 if m == 2 else 31)
        part = query(s, t, f"&FromDate={y}-{m:02d}-01&ToDate={y}-{m:02d}-{last}")
        if len(part) >= 500:
            print("  still capped", s, t, y, m, flush=True)
        for r in part:
            a = acc.setdefault(r["EntityId"], {"EntityId": r["EntityId"], "TeamId": r["TeamId"]})
            for k in KEEP[2:]:
                a[k] = (a.get(k) or 0) + (r.get(k) or 0)
        time.sleep(1.0)
    return list(acc.values())


def main():
    for s in SEASONS:
        for t in TEAM_IDS:
            f = OUT / f"{s}_{t}.json"
            if f.exists():
                continue
            rows = pull(s, t)
            f.write_text(json.dumps(rows))
            print(s, t, len(rows), flush=True)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
