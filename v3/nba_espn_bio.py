"""Fill missing NBA bio data (height, weight, position, birth country) from ESPN's
public athlete API. Used only for players absent from the Kaggle player table.

Usage: python v3/nba_espn_bio.py
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v3" / "data" / "nba" / "espn_bio.parquet"
BASE = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/{s}/athletes"


def get(url):
    for _ in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            pass
    return None


def athlete(url):
    x = get(url)
    if not x:
        return None
    bp = x.get("birthPlace") or {}
    pos = x.get("position") or {}
    return {"espn_id": x.get("id"), "name": x.get("fullName"), "height": x.get("height"),
            "weight": x.get("weight"), "birth_country": bp.get("country"),
            "position_espn": pos.get("displayName"), "dob": x.get("dateOfBirth")}


def main():
    refs = set()
    for s in range(2019, 2025):
        page = 1
        while True:
            d = get(BASE.format(s=s) + f"?limit=200&page={page}")
            if not d:
                break
            refs.update(i["$ref"].split("?")[0] for i in d["items"])
            if page >= d["pageCount"]:
                break
            page += 1
    by_id = {}
    for r in refs:
        by_id.setdefault(r.rsplit("/", 1)[1], r)
    print("athletes", len(by_id), flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [r for r in ex.map(athlete, by_id.values()) if r]
    pd.DataFrame(rows).to_parquet(OUT, index=False)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
