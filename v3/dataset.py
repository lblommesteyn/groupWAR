"""Turn per-game on-court time records into fixed-size graph tensors.

A game record (both sports) is a dict with keys
  game_id, season, playoff, home, away, margin, length,
  secs: {player: seconds}, team_of: {player: team}, shared: {(p, q): seconds}
and optionally pos: {player: position code} (hockey).
"""
import numpy as np
import torch


def side_players(g, team, R, pos=None, f_slots=12):
    """Top-R players by time for one team. For hockey the first 12 slots are
    forwards and the last 6 defencemen (each ordered by time)."""
    ps = [p for p, t in g["team_of"].items() if t == team]
    ps.sort(key=lambda p: -g["secs"].get(p, 0.0))
    if pos is None:
        return ps[:R] + [None] * (R - len(ps[:R]))
    fw = [p for p in ps if pos.get(p) in ("C", "L", "R")][:f_slots]
    dm = [p for p in ps if pos.get(p) == "D"][:R - f_slots]
    return fw + [None] * (f_slots - len(fw)) + dm + [None] * (R - f_slots - len(dm))


def game_matrix(g, players):
    n = len(players)
    A = np.zeros((n, n), dtype=np.float32)
    L = g["length"]
    half = n // 2
    for i, p in enumerate(players):
        if p is None:
            continue
        A[i, i] = g["secs"].get(p, 0.0) / L
        for j in range(i + 1, n):
            q = players[j]
            if q is None:
                continue
            key = (p, q) if p < q else (q, p)
            v = g["shared"].get(key, 0.0) / L
            sign = 1.0 if (i < half) == (j < half) else -1.0
            A[i, j] = A[j, i] = sign * v
    return A


def build_tensors(games, R, feat, n_feat, lag=0, hockey_pos=None):
    """feat: dict (player, season) -> np.array(n_feat). Missing -> zeros.
    lag=1 uses the previous season's features (strictly pre-game information)."""
    G = len(games)
    N = 2 * R
    X = np.zeros((G, N, n_feat), dtype=np.float32)
    A = np.zeros((G, N, N), dtype=np.float32)
    M = np.zeros((G, N), dtype=np.float32)
    have = np.zeros((G, N), dtype=np.float32)
    for k, g in enumerate(games):
        pos = hockey_pos(g) if hockey_pos else None
        players = side_players(g, g["home"], R, pos) + side_players(g, g["away"], R, pos)
        A[k] = game_matrix(g, players)
        for i, p in enumerate(players):
            if p is None:
                continue
            M[k, i] = 1.0
            v = feat.get((p, g["season"] - lag))
            if v is not None:
                X[k, i] = v
                have[k, i] = 1.0
    y = np.array([g["margin"] for g in games], dtype=np.float32)
    return (torch.tensor(X), torch.tensor(A), torch.tensor(M), torch.tensor(y),
            float((have * M).sum() / M.sum()))
