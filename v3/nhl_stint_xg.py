"""Add expected goals to the NHL stint table.

Goals are rare, so a stint-level regression on goal differential is very noisy.
This recomputes the stints exactly as nhl_build.py does and attaches the xG of
every unblocked shot (same xG model) to the stint in which it was taken.
Writes data/nhl/stints_xg.pkl with columns pts_home/pts_away replaced by xG
for/against, plus the original goal columns as g_home/g_away.

Usage: python v3/nhl_stint_xg.py
"""
import pickle

import numpy as np
import pandas as pd

from nhl_build import D, SHOT_TYPES, load_games, mmss, on_ice, xg_model


def main():
    games = load_games()
    xgm, xcols = xg_model(games)
    out = []
    for n, g in enumerate(games):
        pos = {r["playerId"]: r["pos"] for r in g["roster"]}
        team = {r["playerId"]: r["teamId"] for r in g["roster"]}
        skaters = {p for p, c in pos.items() if c != "G"}
        mats = on_ice(g, skaters)
        # xG for every unblocked located shot, in one batch per game
        shots, last = [], None
        for e in g["plays"]:
            if e["type"] in SHOT_TYPES and e.get("xCoord") is not None:
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
                shots.append((e["period"], t[1], e.get("eventOwnerTeamId"), row))
        xg = xgm.predict_proba(pd.DataFrame([s[3] for s in shots])[xcols])[:, 1] if shots else []
        for p, (ids, M) in mats.items():
            change = np.any(M[:, 1:] != M[:, :-1], axis=0)
            bounds = [0] + list(np.where(change)[0] + 1) + [M.shape[1]]
            goals = [(mmss(e["time"]), e.get("eventOwnerTeamId")) for e in g["plays"]
                     if e["type"] == "goal" and e["period"] == p]
            ps = [(t, o, v) for (pp, t, o, _), v in zip(shots, xg) if pp == p]
            for a, b in zip(bounds[:-1], bounds[1:]):
                onset = [ids[i] for i in np.where(M[:, a])[0]]
                h = tuple(sorted(x for x in onset if team.get(x) == g["home"]))
                w = tuple(sorted(x for x in onset if team.get(x) == g["away"]))
                if not (b > a and h and w):
                    continue
                gf = sum(1 for t, o in goals if a < t <= b and o == g["home"])
                ga = sum(1 for t, o in goals if a < t <= b and o == g["away"])
                xf = sum(v for t, o, v in ps if a < t <= b and o == g["home"])
                xa = sum(v for t, o, v in ps if a < t <= b and o == g["away"])
                out.append((g["id"], h, w, b - a, xf, xa, gf, ga))
        if n % 500 == 0:
            print(n, len(out), flush=True)
    st = pd.DataFrame(out, columns=["game_id", "home5", "away5", "secs", "pts_home", "pts_away",
                                    "g_home", "g_away"])
    st.to_pickle(D / "stints_xg.pkl")
    print("stints", len(st), "xG per 60:", st[["pts_home", "pts_away"]].sum().sum() / st["secs"].sum() * 3600 / 2)


if __name__ == "__main__":
    main()
