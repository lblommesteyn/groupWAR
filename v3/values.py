"""Additive player-value scorer (replaces the graph network as the main scorer).

Player value v_p = b . x_p + r_p from one weighted ridge regression on stints
(box-score prior block b, per-player residual r, RAPM penalty), fitted on all
regular-season data up to the case season. For the NBA the stint table ends in
2022-23, so 2023-24 enters as pbpstats five-man lineup totals, each treated as
one long stint against an average opponent. Older seasons are down-weighted by
`decay` per season.

A roster's strength is sum_k q_k v_(k): the minutes-by-rank template q
(average on-court share by slot in real games) applied to its players sorted
by value ("coach" order), or by club minutes per game ("club" order, the naive
deployment, kept as a sensitivity check). With an additive model the coach
order is the optimal deployment, so no adversarial step is needed; the
interaction test (interact_test.py) finds no unit or matchup effects to exploit.

Win probability against an opponent is sigmoid(kappa * (S_r - S_o)) at a
neutral site, with kappa fitted out of sample: each season's games are rated by
a model fitted without that season, using rolling pre-game minute shares.

Uncertainty: `n_boot` refits on games resampled with replacement give value
draws; scores are reported per draw.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.linear_model import LogisticRegression, Ridge

from data_registry import load_sport
from rapm import PER, load_stints
from rosters import minutes_template

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "v3" / "data"
CACHE = ROOT / "v3" / "models" / "values"

# settings chosen on the held-out-season validation (stint_ridge.py)
DEFAULTS = {"nba": {"alpha": 1e5, "box_scale": 3.0, "stints": "stints.pkl"},
            "nhl": {"alpha": 1e5, "box_scale": 0.0, "stints": "stints_xg.pkl"}}


def nba_lineup_rows(season_label):
    """pbpstats lineup totals for one season as stint-like rows (home5 = the
    lineup, away5 = empty, i.e. an average opponent)."""
    s = f"{season_label - 1}-{str(season_label)[2:]}"
    rows = []
    for f in sorted((DATA / "nba" / "lineups").glob(f"{s}_*.json")):
        for r in json.loads(f.read_text()):
            secs = r.get("SecondsPlayed") or 0
            if secs < 5:
                continue
            ids = tuple(sorted(int(x) for x in r["EntityId"].split("-")))
            if len(ids) != 5:
                continue
            rows.append({"game_id": f"L{s}", "home5": ids, "away5": (), "secs": float(secs),
                         "pts_home": r["Points"] or 0, "pts_away": r["OpponentPoints"] or 0,
                         "season": season_label, "playoff": False})
    return pd.DataFrame(rows)


class ValueModel:
    """Builds the stint design once; `fit(seed)` refits with optional game-level
    bootstrap weights and returns (b, r, replacement)."""

    def __init__(self, sport, season, decay=0.75, alpha=None, box_scale=None, exclude_season=None,
                 lineup_seasons=None):
        self.sport, self.season = sport, season
        cfg = DEFAULTS[sport]
        self.alpha = alpha if alpha is not None else cfg["alpha"]
        self.c = box_scale if box_scale is not None else cfg["box_scale"]
        S = load_sport(sport)
        self.S = S
        F = self.F = len(S.feature_cols)
        st = load_stints(sport, cfg["stints"])
        st = st[~st["playoff"] & (st["secs"] >= 5) & (st["season"] <= season)]
        if exclude_season is not None:
            st = st[st["season"] != exclude_season]
        zero = np.zeros(F, np.float32)
        fv = lambda p, s: S.vec(p, s) if S.vec(p, s) is not None else zero
        st = st.assign(is_lineup=False)
        if sport == "nba":
            extra = [ls for ls in (lineup_seasons if lineup_seasons is not None else [season])
                     if ls not in set(st["season"]) and ls != exclude_season]
            if extra:
                opp_curve = self._opponent_curve(st, fv, decay)
                mpg = {(p, s_): m for p, s_, m in zip(S.ps["player_id"], S.ps["season"], S.ps["mpg"])}
                for ls in extra:
                    lr = nba_lineup_rows(ls)
                    if not len(lr):
                        continue
                    # lineup totals do not say who the opponent was; bench units face
                    # bench units, so add the expected opponent unit value for a lineup
                    # of this usage level (measured on seasons with full stints)
                    use = np.array([sum(mpg.get((p, ls), 0.0) for p in u) for u in lr["home5"]])
                    opp = np.interp(use, *opp_curve)
                    secs = lr["secs"].to_numpy()
                    lr["pts_home"] = lr["pts_home"] + opp * secs / PER[sport]
                    st = pd.concat([st, lr.assign(is_lineup=True)], ignore_index=True)
        self.opp_curve = getattr(self, "opp_curve", None)
        players = sorted({p for t in st["home5"] for p in t} | {p for t in st["away5"] for p in t})
        ix = {p: i for i, p in enumerate(players)}
        rows, cols, vals = [], [], []
        B = np.zeros((len(st), F), np.float32)
        for r, (h, a, s) in enumerate(zip(st["home5"], st["away5"], st["season"])):
            for p in h:
                rows.append(r); cols.append(ix[p]); vals.append(1.0); B[r] += fv(p, s)
            for p in a:
                rows.append(r); cols.append(ix[p]); vals.append(-1.0); B[r] -= fv(p, s)
        Xid = csr_matrix((vals, (rows, cols)), shape=(len(st), len(players)))
        self.X = hstack([csr_matrix(B * self.c), Xid]).tocsr() if self.c > 0 else Xid
        if st["is_lineup"].any():
            # separate intercept for lineup rows (they mix home and away time)
            self.X = hstack([self.X, csr_matrix(st["is_lineup"].to_numpy(float)[:, None] * 30.0)]).tocsr()
        if sport == "nhl":
            # manpower-state terms (5v4, 4v5, 6v5, ...) so power-play time is not
            # credited to the skaters on the ice; lightly penalised via scaling
            self.X = hstack([self.X, state_dummies(st)]).tocsr()
        self.y = (st["pts_home"] - st["pts_away"]).to_numpy() / st["secs"].to_numpy() * PER[sport]
        self.w = st["secs"].to_numpy() * decay ** (season - st["season"].to_numpy())
        self.game_codes, self.games = pd.factorize(st["game_id"])
        self.players = players
        cur = S.ps[S.ps["season"] == season]
        lo, hi = cur["minutes_total"].quantile([0.10, 0.30])
        self.repl_ids = cur[(cur["minutes_total"] >= lo) & (cur["minutes_total"] <= hi)]["player_id"].tolist()

    def _opponent_curve(self, st, fv, decay):
        """Mean opponent unit value (stint-only fit) by own-unit usage, as bin
        centres and values for interpolation."""
        sub = ValueModel.__new__(ValueModel)
        sub.__init__(self.sport, self.season, decay=decay, alpha=self.alpha, box_scale=self.c,
                     lineup_seasons=[])
        b, r, repl = sub.fit()
        S = self.S
        mpg = {(p, s_): m for p, s_, m in zip(S.ps["player_id"], S.ps["season"], S.ps["mpg"])}
        u, o, w = [], [], []
        for h, a, s_, sec in zip(st["home5"], st["away5"], st["season"], st["secs"]):
            for own, opp in ((h, a), (a, h)):
                u.append(sum(mpg.get((p, s_), 0.0) for p in own))
                o.append(sum(value_of(S, p, s_, b, r, repl) for p in opp))
                w.append(sec)
        u, o, w = map(np.array, (u, o, w))
        edges = np.quantile(u, np.linspace(0, 1, 21))
        cen, val = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (u >= lo) & (u <= hi)
            cen.append(np.average(u[m], weights=w[m])); val.append(np.average(o[m], weights=w[m]))
        self.opp_curve = (np.array(cen), np.array(val))
        return self.opp_curve

    def fit(self, seed=None):
        w = self.w
        if seed is not None:  # game-level bootstrap as row reweighting
            rng = np.random.default_rng(seed)
            counts = np.bincount(rng.integers(0, len(self.games), len(self.games)), minlength=len(self.games))
            w = w * counts[self.game_codes]
        m = Ridge(alpha=self.alpha, fit_intercept=True).fit(self.X, self.y, sample_weight=w)
        F = self.F
        b = m.coef_[:F] * self.c if self.c > 0 else np.zeros(F)
        off = F if self.c > 0 else 0
        r = dict(zip(self.players, m.coef_[off:off + len(self.players)]))
        vals = [value_of(self.S, p, self.season, b, r, 0.0) for p in self.repl_ids]
        return b, r, float(np.mean(vals))


def state_dummies(st, scale=30.0):
    """One-hot manpower state per stint (skaters home, skaters away), clipped to 3..6."""
    nh = st["home5"].map(len).clip(3, 6).to_numpy()
    na = st["away5"].map(len).clip(3, 6).to_numpy()
    code = (nh - 3) * 4 + (na - 3)
    code[code == 2 * 4 + 2] = -1  # 5v5 is the baseline
    rows = np.where(code >= 0)[0]
    return csr_matrix((np.full(len(rows), scale), (rows, code[rows])), shape=(len(st), 16))


def value_of(S, p, season, b, r, repl):
    if p is None or p < 0:
        return repl
    x = S.vec(p, season)
    if x is None and p not in r:
        return repl
    return (float(x @ b) if x is not None else 0.0) + r.get(p, 0.0)


def fit_kappa(sport, decay=0.75):
    """Out-of-sample win-probability slope: leave-one-season-out values applied
    to that season's games with rolling pre-game minute shares."""
    from dataset import side_players
    from prior_graphs import rolling_graph_tensors
    S = load_sport(sport)
    games = S.games()
    A, ok = rolling_graph_tensors(S)
    seasons = sorted({g["season"] for g in games})
    xs, ys = [], []
    for s in seasons:
        b, r, repl = ValueModel(sport, max(seasons), decay=decay, exclude_season=s, lineup_seasons=[]).fit()
        for k, g in enumerate(games):
            if g["season"] != s or g["playoff"] or not ok[k] or g["margin"] == 0:
                continue
            pos = S.hockey_pos(g) if S.hockey_pos else None
            tot = []
            for side, team in enumerate((g["home"], g["away"])):
                pl = side_players(g, team, S.R, pos)
                share = np.diagonal(A[k].numpy())[side * S.R:(side + 1) * S.R]
                tot.append(sum(sh * value_of(S, p, s, b, r, repl) for p, sh in zip(pl, share) if p is not None))
            xs.append(tot[0] - tot[1]); ys.append(int(g["margin"] > 0))
        print(f"  kappa fold {s}: {len(xs)} games so far", flush=True)
    lr = LogisticRegression().fit(np.array(xs)[:, None], np.array(ys))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


class ValueEvaluator:
    """Drop-in for rosters.Evaluator: score_rosters, war, war_many, mpg, R, S, season."""

    def __init__(self, sport, season, n_boot=20, order="coach", decay=0.75, **_):
        self.sport, self.season, self.order_mode = sport, season, order
        self.S = load_sport(sport)
        self.R = self.S.R
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"{sport}_{season}_d{decay}_b{n_boot}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                state = pickle.load(fh)
        else:
            print("fitting value model", flush=True)
            vm = ValueModel(sport, season, decay=decay)
            main = vm.fit()
            draws = [vm.fit(seed=1000 + k) for k in range(n_boot)]
            kf = CACHE / f"{sport}_kappa_d{decay}.json"
            if kf.exists():
                kappa = json.loads(kf.read_text())["kappa"]
            else:
                kappa, icpt = fit_kappa(sport, decay)
                kf.write_text(json.dumps({"kappa": kappa, "home_intercept": icpt}))
            state = {"main": main, "draws": draws, "kappa": kappa}
            with open(f, "wb") as fh:
                pickle.dump(state, fh)
        self.kappa = state["kappa"]
        self._models = [state["main"]] + state["draws"]
        self.q = minutes_template(self.S).numpy()
        ps = self.S.ps
        cur = ps[ps["season"] == season]
        self.mpg = dict(zip(cur["player_id"], cur["toi_pg"] if sport == "nhl" else cur["mpg"]))
        self.pos = dict(zip(ps["player_id"], ps.get("pos", ps.get("position"))))
        self._vcache = {}

    def _value(self, p, m=0):
        key = (p, m)
        if key not in self._vcache:
            b, r, repl = self._models[m]
            self._vcache[key] = value_of(self.S, p, self.season, b, r, repl)
        return self._vcache[key]

    def deploy(self, roster, m=0, order=None):
        """Players in slot order and their template minute shares."""
        order = order or self.order_mode
        key = (lambda p: -self._value(p, m)) if order == "coach" else (lambda p: -self.mpg.get(p, 0.0))
        if self.sport == "nba":
            ids = sorted(roster, key=key)
        else:
            fw = sorted([p for p in roster if self.pos.get(p) != "D"], key=key)
            dm = sorted([p for p in roster if self.pos.get(p) == "D"], key=key)
            ids = fw + dm
        return ids, self.q[:len(ids)]

    def strength(self, roster, m=0, order=None):
        ids, q = self.deploy(roster, m, order)
        return float(sum(qi * self._value(p, m) for p, qi in zip(ids, q)))

    def score_rosters(self, rosters, opponents, mode=None, model=0):
        order = "club" if mode == "club" else self.order_mode
        so = [self.strength(o, model, order) for o in opponents]
        out = np.zeros((len(rosters), len(opponents)), np.float32)
        for i, r in enumerate(rosters):
            s = self.strength(r, model, order)
            out[i] = 1 / (1 + np.exp(-self.kappa * (s - np.array(so))))
        return torch.tensor(out)

    def score_draws(self, roster, opponents, mode=None):
        """Mean score against the opponents under each bootstrap draw."""
        return [float(self.score_rosters([roster], opponents, mode, model=m).mean())
                for m in range(1, len(self._models))]

    def war_many(self, rosters, opponents, mode=None):
        out = []
        for r in rosters:
            base = float(self.score_rosters([r], opponents, mode).mean())
            d = {}
            for p in r:
                knocked = [(-999 if x == p else x) for x in r]  # replacement-level stand-in
                if self.sport == "nhl" and self.pos.get(p) == "D":
                    self.pos[-999] = "D"
                else:
                    self.pos[-999] = "F"
                d[p] = base - float(self.score_rosters([knocked], opponents, mode).mean())
            out.append((base, d))
        return out

    def war(self, roster, opponents, mode=None):
        return self.war_many([roster], opponents, mode)[0]

    def values_table(self, players):
        return {p: self._value(p) for p in players}
