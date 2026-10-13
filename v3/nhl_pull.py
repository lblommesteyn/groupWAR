"""Pull NHL play-by-play, shift charts and team rosters (2021-22 .. 2024-25,
regular season + playoffs) from the public NHL endpoints into compact per-game
pickles. Resumable: games already on disk are skipped.

Usage: python v3/nhl_pull.py
"""
import json
import pickle
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v3" / "data" / "nhl"
GAMES = OUT / "games"
GAMES.mkdir(parents=True, exist_ok=True)

SEASONS = [2021, 2022, 2023, 2024]  # start year
REG_GAMES = {2021: 1312, 2022: 1312, 2023: 1312, 2024: 1312}
TEAMS = ["ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
         "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", "PHI",
         "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"]

PLAY_DETAIL_KEYS = [
    "xCoord", "yCoord", "zoneCode", "shotType", "eventOwnerTeamId", "shootingPlayerId",
    "scoringPlayerId", "assist1PlayerId", "assist2PlayerId", "goalieInNetId",
    "hittingPlayerId", "hitteePlayerId", "blockingPlayerId", "winningPlayerId",
    "losingPlayerId", "playerId", "committedByPlayerId", "drawnByPlayerId", "reason",
]


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** k)
        except Exception:
            time.sleep(2 ** k)
    return None


def compact_pbp(d):
    plays = []
    for p in d.get("plays", []):
        det = p.get("details", {}) or {}
        row = {
            "eventId": p.get("eventId"),
            "period": p.get("periodDescriptor", {}).get("number"),
            "periodType": p.get("periodDescriptor", {}).get("periodType"),
            "time": p.get("timeInPeriod"),
            "type": p.get("typeDescKey"),
            "situation": p.get("situationCode"),
            "homeSide": p.get("homeTeamDefendingSide"),
        }
        for k in PLAY_DETAIL_KEYS:
            if k in det:
                row[k] = det[k]
        plays.append(row)
    return {
        "id": d["id"], "season": d.get("season"), "gameType": d.get("gameType"),
        "date": d.get("gameDate"),
        "home": d["homeTeam"]["id"], "homeAbbrev": d["homeTeam"].get("abbrev"),
        "away": d["awayTeam"]["id"], "awayAbbrev": d["awayTeam"].get("abbrev"),
        "homeScore": d["homeTeam"].get("score"), "awayScore": d["awayTeam"].get("score"),
        "lastPeriodType": (d.get("gameOutcome") or {}).get("lastPeriodType"),
        "roster": [{"playerId": r["playerId"], "teamId": r["teamId"],
                    "pos": r.get("positionCode")} for r in d.get("rosterSpots", [])],
        "plays": plays,
    }


def compact_shifts(d):
    return [{"playerId": s["playerId"], "teamId": s["teamId"], "period": s["period"],
             "start": s["startTime"], "end": s["endTime"], "type": s["typeCode"]}
            for s in (d or {}).get("data", []) if s.get("typeCode") == 517]


def pull_game(gid):
    f = GAMES / f"{gid}.pkl"
    if f.exists():
        return "skip"
    pbp = get(f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play")
    if pbp is None or pbp.get("gameState") not in ("OFF", "FINAL"):
        return "missing"
    sh = get(f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}")
    g = compact_pbp(pbp)
    g["shifts"] = compact_shifts(sh)
    with open(f, "wb") as fh:
        pickle.dump(g, fh)
    return "ok"


def game_ids():
    ids = []
    for s in SEASONS:
        ids += [int(f"{s}02{n:04d}") for n in range(1, REG_GAMES[s] + 1)]
        for rnd, nser in [(1, 8), (2, 4), (3, 2), (4, 1)]:
            for ser in range(1, nser + 1):
                for g in range(1, 8):
                    ids.append(int(f"{s}030{rnd}{ser}{g}"))
    return ids


def pull_rosters():
    f = OUT / "rosters.json"
    if f.exists():
        return
    rows = []
    for s in SEASONS:
        for t in TEAMS:
            d = get(f"https://api-web.nhle.com/v1/roster/{t}/{s}{s + 1}")
            if not d:
                continue
            for grp in ("forwards", "defensemen", "goalies"):
                for p in d.get(grp, []):
                    rows.append({
                        "season": s, "team": t, "id": p["id"],
                        "first": p["firstName"]["default"], "last": p["lastName"]["default"],
                        "pos": p.get("positionCode"), "shoots": p.get("shootsCatches"),
                        "height": p.get("heightInInches"), "weight": p.get("weightInPounds"),
                        "birthDate": p.get("birthDate"), "birthCountry": p.get("birthCountry"),
                    })
    f.write_text(json.dumps(rows))
    print("rosters", len(rows), flush=True)


def main():
    pull_rosters()
    ids = game_ids()
    counts = {"ok": 0, "skip": 0, "missing": 0}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, res in enumerate(ex.map(pull_game, ids)):
            counts[res] += 1
            if i % 200 == 0:
                print(i, len(ids), counts, flush=True)
    print("done", counts, flush=True)


if __name__ == "__main__":
    main()
