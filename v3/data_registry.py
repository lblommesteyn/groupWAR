"""Sport-specific loaders exposing a common interface used by training and search."""
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import build_tensors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "v3" / "data"


class Sport:
    name: str
    R: int
    scale: float
    feature_cols: list

    def __init__(self):
        self.ps = self.load_player_seasons()
        base = self.ps[(self.ps["minutes_total"] >= self.min_minutes)]
        self.mu = base[self.feature_cols].mean()
        self.sd = base[self.feature_cols].std().replace(0, 1.0)
        Z = ((self.ps[self.feature_cols] - self.mu) / self.sd).fillna(0.0).to_numpy(np.float32)
        self.feat = {(int(p), int(s)): Z[i] for i, (p, s) in
                     enumerate(zip(self.ps["player_id"], self.ps["season"]))}

    def vec(self, pid, season):
        return self.feat.get((int(pid), int(season)))

    def tensors(self, lag=0):
        games = self.games()
        X, A, M, y, cov = build_tensors(games, self.R, self.feat, len(self.feature_cols), lag,
                                        self.hockey_pos)
        meta = [{k: g[k] for k in ("game_id", "season", "playoff", "home", "away", "margin")}
                for g in games]
        return X, A, M, y, meta, cov

    hockey_pos = None


class NBA(Sport):
    name, R, scale, min_minutes = "nba", 12, 10.0, 200

    def __init__(self):
        from nba_features import FEATURES
        self.feature_cols = FEATURES
        super().__init__()

    def load_player_seasons(self):
        ps = pd.read_parquet(DATA / "nba" / "player_seasons.parquet")
        ps["minutes_total"] = ps["Minutes"]
        return ps

    @lru_cache(maxsize=1)
    def games(self):
        with open(DATA / "nba" / "game_graphs.pkl", "rb") as f:
            return pickle.load(f)


class NHL(Sport):
    name, R, scale, min_minutes = "nhl", 18, 2.0, 200

    def __init__(self):
        from nhl_features import FEATURES
        self.feature_cols = FEATURES
        super().__init__()
        self.pos = dict(zip(self.ps["player_id"], self.ps["pos"]))

    def load_player_seasons(self):
        ps = pd.read_parquet(DATA / "nhl" / "player_seasons.parquet")
        return ps

    def hockey_pos(self, g):
        return g.get("pos") or self.pos

    @lru_cache(maxsize=1)
    def games(self):
        with open(DATA / "nhl" / "game_graphs.pkl", "rb") as f:
            return pickle.load(f)


def load_sport(name):
    return {"nba": NBA, "nhl": NHL}[name]()
