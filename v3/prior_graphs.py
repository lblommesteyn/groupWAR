"""Pre-game graphs: keep each game's players but replace the observed time matrix
with the deployment prior implied by season minutes (no in-game information).

The observed matrix is partly a consequence of the result (starters rest in
blowouts, power-play time follows penalties, goalies are pulled when trailing),
so training or evaluating on it can leak the outcome. This module builds the
leakage-free alternative used in Section 6 and as a robustness variant.
"""
import torch

from core import UnitSpace, full_graph
from rosters import calibrate_prior, minutes_template


def rolling_graph_tensors(S, decay=0.9, min_history=5):
    """Pre-game deployment from each team's previous games in the same season.

    For a game, each team's time matrix is the exponentially weighted average
    (factor `decay` per game) of its earlier shared-time matrices, restricted to
    the players who appear in this game and rescaled so the side's time shares
    sum to five. It reflects how the coach has been deploying these players, but
    contains nothing from the game being predicted. Cross-team exposure is the
    product of time shares, as in the deployment model. Returns the tensor and a
    boolean mask of games where both teams have at least `min_history` prior games.
    """
    from collections import defaultdict
    from dataset import side_players
    games = S.games()
    R = S.R
    order = sorted(range(len(games)), key=lambda k: (str(games[k]["date"]), str(games[k]["game_id"])))
    state = {}  # (team, season) -> dict(secs, shared, length, n)
    A = torch.zeros(len(games), 2 * R, 2 * R)
    ok = torch.zeros(len(games), dtype=torch.bool)
    for k in order:
        g = games[k]
        pos = S.hockey_pos(g) if S.hockey_pos else None
        sides, enough = [], True
        for team in (g["home"], g["away"]):
            st = state.get((team, g["season"]))
            players = side_players(g, team, R, pos)
            W = torch.zeros(R, R)
            if st is None or st["n"] < min_history:
                enough = False
            if st is not None and st["length"] > 0:
                for i, p in enumerate(players):
                    if p is None:
                        continue
                    W[i, i] = float(st["secs"].get(p, 0.0) / st["length"])
                    for j in range(i + 1, R):
                        q = players[j]
                        if q is None:
                            continue
                        v = float(st["shared"].get((p, q) if p < q else (q, p), 0.0) / st["length"])
                        W[i, j] = W[j, i] = v
                tot = torch.diagonal(W).sum()
                if tot > 0:
                    W = W * (5.0 / tot)
            sides.append(W)
        mh, ma = torch.diagonal(sides[0]), torch.diagonal(sides[1])
        cross = -mh.unsqueeze(-1) * ma.unsqueeze(0)
        A[k] = torch.cat([torch.cat([sides[0], cross], 1), torch.cat([cross.T, sides[1]], 1)], 0)
        ok[k] = enough
        # update state with this game's observed deployment
        for team in (g["home"], g["away"]):
            key = (team, g["season"])
            st = state.setdefault(key, {"secs": defaultdict(float), "shared": defaultdict(float),
                                        "length": 0.0, "n": 0})
            for d in (st["secs"], st["shared"]):
                for kk in d:
                    d[kk] *= decay
            st["length"] = st["length"] * decay + g["length"]
            st["n"] += 1
            mine = [p for p, t in g["team_of"].items() if t == team]
            ms = set(mine)
            for p in mine:
                st["secs"][p] += g["secs"].get(p, 0.0)
            for (p, q), v in g["shared"].items():
                if p in ms and q in ms:
                    st["shared"][(p, q)] += v
    return A, ok


def prior_graph_tensors(S, X, A, M, meta, lag=0, chunk=512):
    space = UnitSpace(S.name)
    R = S.R
    q = minutes_template(S, (X, A, M, None, meta, None))
    theta = calibrate_prior(space, q).cpu()
    mpg_col = "toi_pg" if S.name == "nhl" else "mpg"
    mpg = {(int(p), int(s)): v for p, s, v in zip(S.ps["player_id"], S.ps["season"], S.ps[mpg_col])}
    games = S.games()
    from dataset import side_players
    ranks = torch.zeros(len(games), 2 * R, dtype=torch.long)
    for k, g in enumerate(games):
        pos = S.hockey_pos(g) if S.hockey_pos else None
        for side, team in enumerate((g["home"], g["away"])):
            players = side_players(g, team, R, pos)
            groups = [range(R)] if S.name == "nba" else [range(12), range(12, R)]
            for grp in groups:
                idx = list(grp)
                key = [-(mpg.get((p, g["season"] - lag), 0.0) if p is not None else -1e9) for p in
                       (players[i] for i in idx)]
                order = sorted(range(len(idx)), key=lambda j: key[j])
                for rank, j in enumerate(order):
                    ranks[k, side * R + idx[j]] = idx[0] + rank
    out = torch.zeros_like(A)
    Ms = [Mg.cpu() for Mg in space.groups]
    for s in range(0, len(games), chunk):
        rk = ranks[s:s + chunk]
        mask = M[s:s + chunk]
        sides = []
        for side in range(2):
            th = theta[rk[:, side * R:(side + 1) * R]]
            th = th - 1e4 * (1 - mask[:, side * R:(side + 1) * R])
            Wa = torch.zeros(len(rk), R, R)
            mins = []
            for Mg in Ms:
                w = torch.softmax(th @ Mg.T, -1)
                Wa = Wa + (Mg.T.unsqueeze(0) * w.unsqueeze(1)) @ Mg
                mins.append(w @ Mg)
            if len(Ms) == 2:
                mf, md = mins
                Wa = Wa + mf.unsqueeze(-1) * md.unsqueeze(1) + md.unsqueeze(-1) * mf.unsqueeze(1)
            sides.append(Wa)
        out[s:s + chunk] = full_graph(sides[0], sides[1]).cpu()
    return out
