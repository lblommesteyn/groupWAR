"""Luck-adjusted and expected-points stint targets for NBA RAPM.

Re-runs the lineup tracking of nba_build_graphs.process_game, but also tags every
shot and free throw with the stint it happened in, then writes one stint table per
scoring scheme (same columns as stints.pkl, pts_home / pts_away replaced):

  actual        made shots and free throws as they happened (check against stints.pkl)
  ctg           3PA at league-average 3P% x3, FTA at league-average FT%; 2s actual
                (Cleaning the Glass style luck adjustment)
  inpred        3PA and FTA at the shooter's shrunk season 3P% / FT%; 2s actual
                (inpredictable luck-adjusted RAPM style)
  xpts          every FGA at the league make rate of its zone, FTA at league FT%
  xpts_shooter  zone rate plus the shooter's shrunk offset (2s, 3s), FTA at shooter FT%

League and shooter rates are per season.

Usage: python v3/nba_stint_luck.py
"""
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from nba_build_graphs import SUB, clock_to_elapsed, parse_score, period_starters

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "v3" / "data" / "nba"
DIST = re.compile(r"(\d+)'")
K3, K2, KFT = 500.0, 300.0, 100.0


def family(desc):
    d = desc.lower()
    if "dunk" in d:
        return "dunk"
    if "layup" in d or "finger roll" in d:
        return "layup"
    if "hook" in d:
        return "hook"
    return "jump"


def zone(is3, dist, fam):
    if is3:
        if dist is None:
            return "3_unk"
        return "3_corner" if dist <= 23 else ("3_arc" if dist <= 26 else "3_deep")
    if dist is None:
        return f"2_unk_{fam}"
    b = "0-3" if dist <= 3 else ("4-9" if dist <= 9 else ("10-15" if dist <= 15 else "16-23"))
    return f"2_{b}_{fam if dist <= 9 else 'any'}"


def process_game(args):
    gid, ev_all, home_id, away_id = args
    stints, shots = [], []
    prev_end = {}
    score = (0, 0)
    for period, ev in ev_all.groupby("period", sort=True):
        rows = ev.sort_values("eventnum").to_dict("records")
        on = period_starters(rows, [home_id, away_id], prev_end)
        if len(on[home_id]) != 5 or len(on[away_id]) != 5:
            return None
        last_t = 0
        stint_start, stint_score = 0, score
        for r in rows:
            t_now = clock_to_elapsed(r["pctimestring"], period)
            if t_now > last_t:
                last_t = t_now
            sc = parse_score(r["score"])
            if sc is not None:
                score = sc
            et = r["eventmsgtype"]
            if et in (1, 2, 3):
                team = r["player1_team_id"]
                if team in (home_id, away_id):
                    desc = r["homedescription"] if team == home_id else r["visitordescription"]
                    desc = desc if isinstance(desc, str) else ""
                    if et == 3:
                        made = "MISS" not in desc
                        shots.append((len(stints), team == home_id, r["player1_id"], "ft", made, 0, "ft"))
                    else:
                        is3 = "3PT" in desc
                        m = DIST.search(desc)
                        dist = int(m.group(1)) if m else None
                        z = zone(is3, dist, family(desc))
                        shots.append((len(stints), team == home_id, r["player1_id"], "fg3" if is3 else "fg2",
                                      et == 1, 3 if is3 else 2, z))
            if et == SUB:
                t, out_p, in_p = r["player1_team_id"], r["player1_id"], r["player2_id"]
                if t in on and out_p in on[t] and in_p and in_p not in on[t]:
                    stints.append((gid, tuple(sorted(on[home_id])), tuple(sorted(on[away_id])),
                                   t_now - stint_start, score[1] - stint_score[1], score[0] - stint_score[0]))
                    on[t] = [in_p if p == out_p else p for p in on[t]]
                    stint_start, stint_score = t_now, score
        stints.append((gid, tuple(sorted(on[home_id])), tuple(sorted(on[away_id])),
                       last_t - stint_start, score[1] - stint_score[1], score[0] - stint_score[0]))
        prev_end = {t: list(on[t]) for t in on}
    return gid, stints, shots


def main():
    pbp = pd.read_parquet(D / "pbp.parquet")
    games = pd.read_parquet(D / "games.parquet")
    meta = games.drop_duplicates("game_id").set_index("game_id")
    base = pd.read_pickle(D / "stints.pkl")
    keep_games = set(base["game_id"])
    season_of = pbp.groupby("game_id")["season"].first()

    jobs = []
    for gid, ev in pbp.groupby("game_id", sort=False):
        if gid in keep_games and gid in meta.index:
            m = meta.loc[gid]
            jobs.append((gid, ev, int(m["team_id_home"]), int(m["team_id_away"])))
    print("games", len(jobs), flush=True)
    st_rows, sh_rows = [], []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for n, res in enumerate(ex.map(process_game, jobs, chunksize=50)):
            if res is None:
                continue
            gid, stints, shots = res
            off = len(st_rows)
            st_rows += stints
            sh_rows += [(off + s, *rest, season_of[gid]) for s, *rest in shots]
            if n % 1000 == 0:
                print(n, flush=True)
    st = pd.DataFrame(st_rows, columns=["game_id", "home5", "away5", "secs", "pts_home", "pts_away"])
    sh = pd.DataFrame(sh_rows, columns=["stint", "home", "shooter", "kind", "made", "val", "zone", "season"])
    sh["made"] = sh["made"].astype(float)
    sh["val"] = np.where(sh["kind"] == "ft", 1, sh["val"]).astype(float)

    # league rates per season and zone; shooter shrunk offsets per season
    lg = sh.groupby(["season", "zone"])["made"].mean().rename("p_lg")
    sh = sh.join(lg, on=["season", "zone"])
    lg_kind = sh.groupby(["season", "kind"])["made"].mean().rename("p_kind")
    sh = sh.join(lg_kind, on=["season", "kind"])
    sh["resid"] = sh["made"] - sh["p_lg"]
    g = sh.groupby(["season", "shooter", "kind"]).agg(n=("made", "size"), r=("resid", "sum"),
                                                     mk=("made", "sum"), pk=("p_kind", "first"))
    k = g.index.get_level_values("kind").map({"fg2": K2, "fg3": K3, "ft": KFT}).to_numpy()
    g["off"] = g["r"] / (g["n"] + k)                       # zone-adjusted skill
    g["p_raw"] = (g["mk"] + k * g["pk"]) / (g["n"] + k)     # shrunk raw percentage
    sh = sh.join(g[["off", "p_raw"]], on=["season", "shooter", "kind"])

    is3, isft, is2 = sh["kind"] == "fg3", sh["kind"] == "ft", sh["kind"] == "fg2"
    actual = sh["made"] * sh["val"]
    scheme = {
        "actual": actual,
        "ctg": np.where(is2, actual, sh["p_kind"] * sh["val"]),
        "inpred": np.where(is2, actual, sh["p_raw"] * sh["val"]),
        "xpts": np.where(isft, sh["p_kind"], sh["p_lg"] * sh["val"]),
        "xpts_shooter": np.where(isft, sh["p_raw"],
                                 np.clip(sh["p_lg"] + sh["off"], 0.01, 0.99) * sh["val"]),
    }
    n = len(st)
    for name, pts in scheme.items():
        pts = np.asarray(pts, float)
        h = np.bincount(sh["stint"][sh["home"]], weights=pts[sh["home"].to_numpy()], minlength=n)
        a = np.bincount(sh["stint"][~sh["home"]], weights=pts[~sh["home"].to_numpy()], minlength=n)
        out = st.copy()
        out["pts_home"], out["pts_away"] = h, a
        if name == "actual":
            agree = ((out["pts_home"] == st["pts_home"]) & (out["pts_away"] == st["pts_away"])).mean()
            print(f"event points == score-diff points on {agree:.4f} of stints", flush=True)
        out = out[out["secs"] > 0].reset_index(drop=True)
        out.to_pickle(D / f"stints_{name}.pkl")
        print(name, len(out), "mean pts/stint", round(out[["pts_home", "pts_away"]].to_numpy().mean(), 3), flush=True)
    st = st[st["secs"] > 0].reset_index(drop=True)
    same = (len(st) == len(base)) and bool((st["secs"].to_numpy() == base["secs"].to_numpy()).all()
                                           and (st["pts_home"].to_numpy() == base["pts_home"].to_numpy()).all())
    print("stint table identical to stints.pkl:", same, len(st), len(base))
    sh.to_parquet(D / "luck_shots.parquet", index=False)


if __name__ == "__main__":
    main()
