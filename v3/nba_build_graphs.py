"""Reconstruct on-court lineups from Kaggle NBA play-by-play and build, per game:
  * per-player seconds on court
  * pairwise shared seconds (teammates and opponents)
  * final margin
and a stint table (10 players, seconds, points for/against) for RAPM.

Usage: python v3/nba_build_graphs.py
"""
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "v3" / "data" / "nba"

SUB, TIMEOUT, EJECT, REPLAY = 8, 9, 11, 18
START_PERIOD, END_PERIOD = 12, 13
TECH_ACTIONS = {11, 12, 13, 16, 17, 18, 19, 25, 30}  # technical / delay fouls can be bench


def clock_to_elapsed(s, period):
    m, sec = s.split(":")
    length = 720 if period <= 4 else 300
    return length - (int(m) * 60 + int(float(sec)))


def parse_score(s):
    if not isinstance(s, str) or "-" not in s:
        return None
    a, h = s.split("-")
    return int(a.strip()), int(h.strip())  # (visitor, home)


def period_starters(ev, team_ids, prev_end):
    """Infer the five players per team on court at the start of a period."""
    starters = {t: [] for t in team_ids}
    subbed_in = set()
    for r in ev:
        if r["eventmsgtype"] == SUB:
            out_p, in_p, t = r["player1_id"], r["player2_id"], r["player1_team_id"]
            if t in starters and out_p and out_p not in subbed_in and out_p not in starters[t]:
                starters[t].append(out_p)
            subbed_in.add(in_p)
            continue
        if r["eventmsgtype"] in (TIMEOUT, EJECT, REPLAY, START_PERIOD, END_PERIOD):
            continue
        if r["eventmsgtype"] == 6 and r["eventmsgactiontype"] in TECH_ACTIONS:
            continue
        for k in (1, 2, 3):
            p, t = r[f"player{k}_id"], r[f"player{k}_team_id"]
            if p and t in starters and p not in subbed_in and p not in starters[t]:
                starters[t].append(p)
    for t in team_ids:
        st = starters[t][:5]
        if len(st) < 5:  # player on court all period without an event: carry over
            for p in prev_end.get(t, []):
                if len(st) >= 5:
                    break
                if p not in st and p not in subbed_in:
                    st.append(p)
        starters[t] = st
    return starters


def process_game(gid, ev_all, home_id, away_id):
    secs = defaultdict(float)
    shared = defaultdict(float)  # (p, q) with p < q
    stints = []
    team_of = {}
    prev_end = {}
    score = (0, 0)
    for period, ev in ev_all.groupby("period", sort=True):
        rows = ev.sort_values("eventnum").to_dict("records")
        on = period_starters(rows, [home_id, away_id], prev_end)
        if len(on[home_id]) != 5 or len(on[away_id]) != 5:
            return None
        for t in (home_id, away_id):
            for p in on[t]:
                team_of[p] = t
        last_t = 0
        stint_start, stint_score = 0, score
        for r in rows:
            t_now = clock_to_elapsed(r["pctimestring"], period)
            dt = max(0, t_now - last_t)
            if dt:
                players = on[home_id] + on[away_id]
                for p in players:
                    secs[p] += dt
                for i in range(10):
                    for j in range(i + 1, 10):
                        a, b = players[i], players[j]
                        shared[(a, b) if a < b else (b, a)] += dt
                last_t = t_now
            sc = parse_score(r["score"])
            if sc is not None:
                score = sc
            if r["eventmsgtype"] == SUB:
                t, out_p, in_p = r["player1_team_id"], r["player1_id"], r["player2_id"]
                if t in on and out_p in on[t] and in_p and in_p not in on[t]:
                    stints.append((gid, tuple(sorted(on[home_id])), tuple(sorted(on[away_id])),
                                   t_now - stint_start + (period - 1) * 0,
                                   score[1] - stint_score[1], score[0] - stint_score[0]))
                    on[t] = [in_p if p == out_p else p for p in on[t]]
                    team_of[in_p] = t
                    stint_start, stint_score = t_now, score
        stints.append((gid, tuple(sorted(on[home_id])), tuple(sorted(on[away_id])),
                       last_t - stint_start, score[1] - stint_score[1], score[0] - stint_score[0]))
        prev_end = {t: list(on[t]) for t in on}
    return secs, shared, team_of, stints


def main():
    pbp = pd.read_parquet(D / "pbp.parquet")
    games = pd.read_parquet(D / "games.parquet")
    games["team_id_home"] = games["team_id_home"].astype(int)
    games["team_id_away"] = games["team_id_away"].astype(int)
    meta = games.set_index("game_id")

    out_games, out_stints, failed = [], [], 0
    for n, (gid, ev) in enumerate(pbp.groupby("game_id", sort=False)):
        if gid not in meta.index:
            failed += 1
            continue
        m = meta.loc[gid]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        res = process_game(gid, ev, int(m["team_id_home"]), int(m["team_id_away"]))
        if res is None:
            failed += 1
            continue
        secs, shared, team_of, stints = res
        total = sum(secs.values()) / 10.0
        if total < 2700:  # incomplete play-by-play
            failed += 1
            continue
        out_games.append({
            "game_id": gid, "season": int(ev["season"].iloc[0]), "playoff": bool(ev["playoff"].iloc[0]),
            "date": m["game_date"], "home": int(m["team_id_home"]), "away": int(m["team_id_away"]),
            "margin": float(m["pts_home"] - m["pts_away"]),
            "secs": dict(secs), "team_of": team_of, "shared": dict(shared), "length": total,
        })
        out_stints += stints
        if n % 500 == 0:
            print(n, len(out_games), failed, flush=True)

    with open(D / "game_graphs.pkl", "wb") as f:
        pickle.dump(out_games, f)
    st = pd.DataFrame(out_stints, columns=["game_id", "home5", "away5", "secs", "pts_home", "pts_away"])
    st = st[st["secs"] > 0]
    st.to_pickle(D / "stints.pkl")
    print("games", len(out_games), "failed", failed, "stints", len(st))
    # sanity: stint points should sum to final margin for most games
    agg = st.groupby("game_id")[["pts_home", "pts_away"]].sum()
    mg = pd.DataFrame(out_games).set_index("game_id")["margin"]
    diff = (agg["pts_home"] - agg["pts_away"] - mg.reindex(agg.index)).abs()
    print("stint margin == final margin:", float((diff < 0.5).mean()))


if __name__ == "__main__":
    main()
