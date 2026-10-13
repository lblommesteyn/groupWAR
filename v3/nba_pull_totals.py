"""Pull per-player season totals (regular season) from api.pbpstats.com for
2019-20 .. 2023-24. Queries are made per team because each query truncates at
500 rows; rows for traded players are summed downstream.

Usage: python v3/nba_pull_totals.py
"""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v3" / "data" / "nba" / "totals"
OUT.mkdir(parents=True, exist_ok=True)

SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
TEAM_IDS = list(range(1610612737, 1610612767))


def get(url, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print("retry", k, e, flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError(url)


def main():
    for s in SEASONS:
        for t in TEAM_IDS:
            f = OUT / f"{s}_{t}.json"
            if f.exists():
                continue
            d = get("https://api.pbpstats.com/get-totals/nba?"
                    f"Season={s}&SeasonType=Regular%2BSeason&Type=Player&TeamId={t}")
            f.write_text(json.dumps(d["multi_row_table_data"]))
            print(s, t, len(d["multi_row_table_data"]), flush=True)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
