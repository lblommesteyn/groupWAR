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
