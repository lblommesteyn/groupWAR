"""Build NHL game graphs, RAPM stints and player-season features from the
compact per-game pickles written by nhl_pull.py.

Leakage rule: the 4 Nations Face-Off began 2025-02-12, so every 2024-25 game on
or after that date is dropped. Season labels use the end year (2024-25 -> 2025).

Usage: python v3/nhl_build.py
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "v3" / "data" / "nhl"
CUTOFF = "2025-02-12"

FEATURES = ["toi_pg", "gp_share", "sog60", "att60", "g60", "a1_60", "a2_60", "ixg60",
            "shot_dist", "hits60", "hitsrec60", "blk60", "give60", "take60",
            "pen60", "pendrawn60", "fo_pct", "fo60", "oz_share",
            "height", "weight", "isC", "isL", "isR", "isD", "shootsL"]

SHOT_TYPES = ("shot-on-goal", "missed-shot", "goal")


def mmss(s):
    m, x = s.split(":")
    return int(m) * 60 + int(x)


def period_len(g, period):
    if period <= 3:
        return 1200
    return 1200 if g["gameType"] == 3 else 300


def load_games():
    out = []
    for f in sorted((D / "games").glob("*.pkl")):
        with open(f, "rb") as fh:
            g = pickle.load(fh)
        if g["date"] >= CUTOFF and g["season"] == 20242025:
            continue
        if not g["shifts"]:
            continue
        out.append(g)
    return out


def on_ice(g, skaters):
    """Per-period boolean matrices players x seconds, skaters only."""
    per = defaultdict(lambda: defaultdict(list))
    for s in g["shifts"]:
        if s["playerId"] in skaters and s["start"] and s["end"]:
            per[s["period"]][s["playerId"]].append((mmss(s["start"]), mmss(s["end"])))
    mats = {}
    for p, d in per.items():
        L = period_len(g, p)
        ids = sorted(d)
        M = np.zeros((len(ids), L), dtype=bool)
        for i, pid in enumerate(ids):
            for a, b in d[pid]:
                M[i, max(0, a):min(L, b)] = True
        mats[p] = (ids, M)
    return mats


def xg_model(games):
    rows = []
    for g in games:
        last = None
        for e in g["plays"]:
            if e["type"] in SHOT_TYPES and e.get("xCoord") is not None:
                t = (e["period"], mmss(e["time"]))
                x, y = abs(e["xCoord"]), e["yCoord"]
                dist = np.hypot(89 - x, y)
                ang = np.degrees(np.arctan2(abs(y), max(89 - x, 0.1)))
                reb = last is not None and last[0] == t[0] and 0 <= t[1] - last[1] <= 3
                rows.append((dist, ang, reb, e.get("shotType") or "other", e["type"] == "goal"))
                last = t
    df = pd.DataFrame(rows, columns=["dist", "ang", "reb", "stype", "goal"])
    X = pd.get_dummies(df[["dist", "ang", "reb", "stype"]], columns=["stype"], dtype=float)
    X["dist2"] = X["dist"] ** 2
    m = LogisticRegression(max_iter=2000).fit(X, df["goal"])
    print("xG model on", len(df), "unblocked shots; mean xG", m.predict_proba(X)[:, 1].mean())
    return m, list(X.columns)


def main():
    games = load_games()
    print("games", len(games))
    bios = pd.DataFrame(json.loads((D / "rosters.json").read_text()))
    bios = bios.sort_values("season").groupby("id").last().reset_index()
    pos_bio = dict(zip(bios["id"], bios["pos"]))
    xgm, xcols = xg_model(games)

    graphs, stints = [], []
    stat = defaultdict(lambda: defaultdict(float))  # (pid, season) -> stat -> value
    for n, g in enumerate(games):
        season = int(str(g["season"])[4:])
        pos = {r["playerId"]: r["pos"] for r in g["roster"]}
        team = {r["playerId"]: r["teamId"] for r in g["roster"]}
        skaters = {p for p, c in pos.items() if c != "G"}
        mats = on_ice(g, skaters)
        secs, shared = defaultdict(float), defaultdict(float)
        length = 0
        for p, (ids, M) in mats.items():
            length += M.shape[1]
            Mf = M.astype(np.float32)
            S = Mf @ Mf.T
            for i, a in enumerate(ids):
                secs[a] += S[i, i]
                for j in range(i + 1, len(ids)):
                    if S[i, j] > 0:
                        b = ids[j]
                        shared[(a, b) if a < b else (b, a)] += S[i, j]
            # stints: maximal runs with a constant on-ice set
            change = np.any(M[:, 1:] != M[:, :-1], axis=0)
            bounds = [0] + list(np.where(change)[0] + 1) + [M.shape[1]]
            goals = [(mmss(e["time"]), e.get("eventOwnerTeamId")) for e in g["plays"]
                     if e["type"] == "goal" and e["period"] == p]
            for a, b in zip(bounds[:-1], bounds[1:]):
                onset = [ids[i] for i in np.where(M[:, a])[0]]
                h = tuple(sorted(x for x in onset if team.get(x) == g["home"]))
                w = tuple(sorted(x for x in onset if team.get(x) == g["away"]))
                gf = sum(1 for t, o in goals if a < t <= b and o == g["home"])
                ga = sum(1 for t, o in goals if a < t <= b and o == g["away"])
                if b > a and h and w:
                    stints.append((g["id"], h, w, b - a, gf, ga))
        if length < 3500:
            continue
        margin = g["homeScore"] - g["awayScore"]
        if g["lastPeriodType"] == "SO":
            margin = 0
        graphs.append({"game_id": g["id"], "season": season, "playoff": g["gameType"] == 3,
                       "date": g["date"], "home": g["home"], "away": g["away"],
                       "margin": float(margin), "length": float(length),
                       "secs": dict(secs), "shared": dict(shared),
                       "team_of": {p: team[p] for p in secs}, "pos": {p: pos[p] for p in secs}})
        if g["gameType"] != 2:
            continue
        # individual counting stats (regular season only)
        for p in secs:
            stat[(p, season)]["toi"] += secs[p]
            stat[(p, season)]["gp"] += 1
        # on-ice faceoff zone shares
        on_at = {}
        for pnum, (ids, M) in mats.items():
            on_at[pnum] = (ids, M)
        last = None
        for e in g["plays"]:
            k = e["type"]
            if k in SHOT_TYPES or k == "blocked-shot":
                sh = e.get("shootingPlayerId") or e.get("scoringPlayerId")
                if sh:
                    stat[(sh, season)]["att"] += 1
                    if k in SHOT_TYPES and e.get("xCoord") is not None:
                        x, y = abs(e["xCoord"]), e["yCoord"]
                        dist = np.hypot(89 - x, y)
                        ang = np.degrees(np.arctan2(abs(y), max(89 - x, 0.1)))
                        t = (e["period"], mmss(e["time"]))
                        reb = last is not None and last[0] == t[0] and 0 <= t[1] - last[1] <= 3
                        last = t
                        row = {c: 0.0 for c in xcols}
                        row.update({"dist": dist, "ang": ang, "reb": float(reb), "dist2": dist ** 2})
                        st = f"stype_{e.get('shotType') or 'other'}"
                        if st in row:
                            row[st] = 1.0
                        xg = xgm.predict_proba(pd.DataFrame([row])[xcols])[0, 1]
                        stat[(sh, season)]["ixg"] += xg
                        stat[(sh, season)]["dist"] += dist
                        stat[(sh, season)]["nloc"] += 1
                    if k in ("shot-on-goal", "goal"):
                        stat[(sh, season)]["sog"] += 1
                    if k == "goal":
                        stat[(sh, season)]["g"] += 1
                if k == "goal":
                    if e.get("assist1PlayerId"):
                        stat[(e["assist1PlayerId"], season)]["a1"] += 1
                    if e.get("assist2PlayerId"):
                        stat[(e["assist2PlayerId"], season)]["a2"] += 1
                if k == "blocked-shot" and e.get("blockingPlayerId"):
                    stat[(e["blockingPlayerId"], season)]["blk"] += 1
            elif k == "hit":
                if e.get("hittingPlayerId"):
                    stat[(e["hittingPlayerId"], season)]["hits"] += 1
                if e.get("hitteePlayerId"):
                    stat[(e["hitteePlayerId"], season)]["hitsrec"] += 1
            elif k == "giveaway" and e.get("playerId"):
                stat[(e["playerId"], season)]["give"] += 1
            elif k == "takeaway" and e.get("playerId"):
                stat[(e["playerId"], season)]["take"] += 1
            elif k == "penalty":
                if e.get("committedByPlayerId"):
                    stat[(e["committedByPlayerId"], season)]["pen"] += 1
                if e.get("drawnByPlayerId"):
                    stat[(e["drawnByPlayerId"], season)]["pendrawn"] += 1
            elif k == "faceoff":
                w, l = e.get("winningPlayerId"), e.get("losingPlayerId")
                if w:
                    stat[(w, season)]["fow"] += 1
                if l:
                    stat[(l, season)]["fol"] += 1
                zc, owner = e.get("zoneCode"), e.get("eventOwnerTeamId")
                if zc in ("O", "D") and e["period"] in on_at:
                    ids, M = on_at[e["period"]]
                    t = min(mmss(e["time"]), M.shape[1] - 1)
                    for i in np.where(M[:, t])[0]:
                        pid = ids[i]
                        z = zc if team.get(pid) == owner else ("D" if zc == "O" else "O")
                        stat[(pid, season)]["oz" if z == "O" else "dz"] += 1
        if n % 500 == 0:
            print(n, len(graphs), flush=True)

    with open(D / "game_graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)
    pd.DataFrame(stints, columns=["game_id", "home5", "away5", "secs", "pts_home", "pts_away"]
                 ).to_pickle(D / "stints.pkl")

    rows = []
    for (p, s), v in stat.items():
        h = v["toi"] / 3600.0
        if h <= 0:
            continue
        rows.append({
            "player_id": p, "season": s, "minutes_total": v["toi"] / 60, "gp": v["gp"],
            "toi_pg": v["toi"] / 60 / max(v["gp"], 1),
            "gp_share": v["gp"] / (82 if s != 2025 else 56),
            "sog60": v["sog"] / h, "att60": v["att"] / h, "g60": v["g"] / h,
            "a1_60": v["a1"] / h, "a2_60": v["a2"] / h, "ixg60": v["ixg"] / h,
            "shot_dist": v["dist"] / v["nloc"] if v["nloc"] else np.nan,
            "hits60": v["hits"] / h, "hitsrec60": v["hitsrec"] / h, "blk60": v["blk"] / h,
            "give60": v["give"] / h, "take60": v["take"] / h,
            "pen60": v["pen"] / h, "pendrawn60": v["pendrawn"] / h,
            "fo_pct": (v["fow"] + 25) / (v["fow"] + v["fol"] + 50), "fo60": (v["fow"] + v["fol"]) / h,
            "oz_share": (v["oz"] + 5) / (v["oz"] + v["dz"] + 10),
        })
    ps = pd.DataFrame(rows)
    ps["shot_dist"] = ps["shot_dist"].fillna(ps["shot_dist"].median())
    b = bios.rename(columns={"id": "player_id"})
    ps = ps.merge(b[["player_id", "first", "last", "pos", "shoots", "height", "weight",
                     "birthCountry", "birthDate"]], on="player_id", how="left")
    ps["pos"] = ps["pos"].fillna(ps["player_id"].map(pos_bio))
    for c, code in [("isC", "C"), ("isL", "L"), ("isR", "R"), ("isD", "D")]:
        ps[c] = (ps["pos"] == code).astype(float)
    ps["shootsL"] = (ps["shoots"] == "L").astype(float)
    ps["height"] = ps["height"].fillna(ps["height"].median())
    ps["weight"] = ps["weight"].fillna(ps["weight"].median())
    ps["name"] = ps["first"].fillna("") + " " + ps["last"].fillna("")
    ps.to_parquet(D / "player_seasons.parquet", index=False)
    print("graphs", len(graphs), "stints", len(stints), "player-seasons", len(ps))
    print(ps.groupby("season").size())


if __name__ == "__main__":
    main()
