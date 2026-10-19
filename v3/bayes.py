"""Bayesian hierarchical player-value model (main scorer).

Each stint gives two rows, one per attacking side, with the scoring rate of the
attacking side as the outcome (points per 48 minutes in the NBA, expected goals
per 60 in the NHL), weighted by stint seconds:

  y = mu + home + sum_{attackers} o_{p,s} - sum_{defenders} d_{p,s} + e,
  e ~ N(0, sigma^2 / secs).

Player-season offence and defence follow a random walk with an age curve and a
position-group mean:

  o_{p,s} = a_p + sum_{first_p < k <= s} u_{p,k} + age(p, s) . c + grp(p) . g
  a_p ~ N(0, tau_a^2),  u_{p,k} ~ N(0, tau_rw^2)       (and the same for d)

age(p, s) is a hinge basis in age (linear around 27, extra slopes after 31 and
before 24); c and g have wide priors. The model is linear and Gaussian, so the
posterior is exact: precision A = X'WX / sigma^2 + Lambda. The prior scales
(tau_a, tau_rw for offence and defence) and sigma^2 are set by maximising the
marginal likelihood on the training data (empirical Bayes); nothing is tuned on
held-out games. Posterior draws come from the Cholesky factor of A.

A player's value is o + d (net rate per 48 or per 60) relative to replacement
(the mean of players in the 10th to 30th percentile of minutes that season).
For a season past the data, each player gets one more random-walk step and the
age curve at his new age.

NBA 2023-24 enters as pbpstats lineup totals with no opponent recorded: the
opponent's expected offence and defence by own-lineup usage (measured on full
stints) are added back, as in values.py.

  python v3/bayes.py validate --sport nba --test-season 2023 --std-days 14
  python v3/bayes.py fit --sport nba --season 2024
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, diags
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from data_registry import load_sport
from rapm import PER, load_stints

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "v3" / "models" / "bayes"
STINTS = {"nba": "stints.pkl", "nhl": "stints_xg.pkl"}
SIDES = ("o", "d")
WIDE = 3.0  # prior sd for age-curve and position-group coefficients


def _birth(S, sport):
    col = "birthdate" if sport == "nba" else "birthDate"
    bd = pd.to_datetime(S.ps[col], errors="coerce")
    out = {}
    for p, b in zip(S.ps["player_id"], bd):
        if pd.notna(b) and p not in out:
            out[p] = b
    return out


def age_basis(birth, season):
    if birth is None:
        return np.zeros(3)
    age = (pd.Timestamp(f"{season}-02-01") - birth).days / 365.25
    return np.array([age - 27.0, max(0.0, age - 31.0), max(0.0, 24.0 - age)]) / 5.0


def _groups(S, sport):
    ps = S.ps.sort_values("season")
    out = {}
    if sport == "nba":
        for p, g, f, c in zip(ps["player_id"], ps["isG"], ps["isF"], ps["isC"]):
            v = np.array([g, f, c], float)
            if np.isfinite(v).all() and v.sum() > 0:
                out[p] = v / v.sum()
        default = np.full(3, 1 / 3)
    else:
        for p, d in zip(ps["player_id"], ps["isD"]):
            if np.isfinite(d):
                out[p] = np.array([1 - d, d], float)
        default = np.array([0.7, 0.3])
    return out, default


def state_code(n_att, n_def):
    """Manpower state of the attacking side; 5v5 (and 6v6 artefacts) = -1."""
    a, d = min(max(n_att, 3), 6), min(max(n_def, 3), 6)
    c = (a - 3) * 4 + (d - 3)
    return -1 if c == 2 * 4 + 2 else c


class BayesModel:
    """Design over attack rows, built once; `fit(mask)` solves for any subset of rows."""

    def __init__(self, sport, seasons=None, lineup_seasons=(), opp_curve=None):
        t0 = time.time()
        self.sport = sport
        S = self.S = load_sport(sport)
        self.birth = _birth(S, sport)
        self.grp, self.grp_default = _groups(S, sport)
        st = load_stints(sport, STINTS[sport])
        gdate = {g["game_id"]: pd.Timestamp(g["date"]) for g in S.games()}
        st = st[~st["playoff"] & (st["secs"] >= 5)].copy()
        if seasons is not None:
            st = st[st["season"].isin(seasons)]
        st["date"] = st["game_id"].map(gdate)
        st["lineup"] = False
        if sport == "nba" and lineup_seasons:
            from values import nba_lineup_rows
            mpg = {(p, s): m for p, s, m in zip(S.ps["player_id"], S.ps["season"], S.ps["mpg"])}
            extra = []
            for ls in lineup_seasons:
                lr = nba_lineup_rows(ls)
                use = np.array([sum(mpg.get((p, ls), 0.0) for p in u) for u in lr["home5"]])
                # unknown opponent: add back its expected defence (to our scoring)
                # and remove its expected offence (from what we allowed)
                lr["opp_o"] = np.interp(use, opp_curve[0], opp_curve[1])
                lr["opp_d"] = np.interp(use, opp_curve[0], opp_curve[2])
                lr["date"] = pd.Timestamp(f"{ls}-06-30")
                lr["lineup"] = True
                extra.append(lr)
            st = pd.concat([st] + extra, ignore_index=True)
        self.st = st.reset_index(drop=True)

        # attack rows: (attackers, defenders, points, home flag)
        n = len(self.st)
        H, A_ = self.st["home5"].tolist(), self.st["away5"].tolist()
        secs = self.st["secs"].to_numpy(float)
        ph, pa = self.st["pts_home"].to_numpy(float), self.st["pts_away"].to_numpy(float)
        lin = self.st["lineup"].to_numpy(bool)
        if lin.any():
            oo = self.st["opp_o"].fillna(0).to_numpy(float)
            od = self.st["opp_d"].fillna(0).to_numpy(float)
            ph = ph + np.where(lin, od * secs / PER[sport], 0.0)
            pa = pa - np.where(lin, oo * secs / PER[sport], 0.0)
        seas = self.st["season"].to_numpy(int)
        self.row_season = np.concatenate([seas, seas])
        self.row_date = np.concatenate([self.st["date"].to_numpy(), self.st["date"].to_numpy()])
        self.row_lineup_def = np.concatenate([np.zeros(n, bool), lin])  # lineup rows whose attackers are unknown
        self.y = np.concatenate([ph, pa]) / np.concatenate([secs, secs]) * PER[sport]
        self.w = np.concatenate([secs, secs])

        # player-season columns (vectorised: one entry per player per stint side)
        ln_h = np.fromiter((len(t) for t in H), int, n)
        ln_a = np.fromiter((len(t) for t in A_), int, n)
        hp = np.fromiter((p for t in H for p in t), np.int64, ln_h.sum())
        ap_ = np.fromiter((p for t in A_ for p in t), np.int64, ln_a.sum())
        hi = np.repeat(np.arange(n), ln_h)
        ai = np.repeat(np.arange(n), ln_a)
        # (row, player, season, side 0=o 1=d, sign)
        R_ = np.concatenate([hi, n + hi, n + ai, ai])
        Pp = np.concatenate([hp, hp, ap_, ap_])
        Ss = seas[np.concatenate([hi, hi, ai, ai])]
        Sd = np.concatenate([np.zeros_like(hi), np.ones_like(hi), np.zeros_like(ai), np.ones_like(ai)])
        V = np.where(Sd == 0, 1.0, -1.0)
        key = (Pp * 100 + (Ss - 2000)) * 2 + Sd
        codes, uniq = pd.factorize(key, sort=True)
        P = len(uniq)
        self.ps_keys = {(int(k // 200), int(k // 2 % 100) + 2000, SIDES[int(k % 2)]): j for j, k in enumerate(uniq)}
        rows, cols, vals = [R_], [codes], [V]
        # nuisance columns: intercept, home attack, lineup rows (mixed home/away), NHL states
        nuis = ["mu", "home", "lineup"]
        nr = 2 * n
        home = np.concatenate([np.where(lin, 0.5, 1.0), np.where(lin, 0.5, 0.0)])
        allr = np.arange(nr)
        rows += [allr, allr]; cols += [np.full(nr, P), np.full(nr, P + 1)]; vals += [np.ones(nr), home]
        lr_rows = np.where(np.concatenate([lin, lin]))[0]
        rows.append(lr_rows); cols.append(np.full(len(lr_rows), P + 2)); vals.append(np.ones(len(lr_rows)))
        if sport == "nhl":
            nuis += [f"state{k}" for k in range(16)]
            na = np.concatenate([ln_h, ln_a]); nd = np.concatenate([ln_a, ln_h])
            c = (np.clip(na, 3, 6) - 3) * 4 + (np.clip(nd, 3, 6) - 3)
            keep = c != 2 * 4 + 2
            rows.append(allr[keep]); cols.append(P + 3 + c[keep]); vals.append(np.ones(keep.sum()))
        self.nuis = nuis
        self.X = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(nr, P + len(nuis)))
        ps_keys = self.ps_keys

        # parameters and the map T: params -> player-season effects (+ nuisance)
        players = sorted({k[0] for k in ps_keys})
        self.players = players
        self.pseasons = {}
        for (p, s, side) in ps_keys:
            self.pseasons.setdefault(p, set()).add(s)
        self.first = {p: min(v) for p, v in self.pseasons.items()}
        self.seasons = sorted(set(seas))
        self.smax = max(self.seasons)
        self.pidx = {}
        k = 0
        self.block = {}
        self._ia, self._irw = {}, {}
        for side in SIDES:
            start = k
            ia, irw = [], []
            for p in players:
                self.pidx[(p, "a", side)] = k; ia.append(k); k += 1
            for p in players:
                for s in self.seasons:
                    if s > self.first[p]:
                        self.pidx[(p, s, side)] = k; irw.append(k); k += 1
            self._ia[side], self._irw[side] = np.array(ia, int), np.array(irw, int)
            self.block[(side, "player")] = (start, k)
            self.pidx[("age", side)] = k; k += 3
            G = len(self.grp_default)
            self.pidx[("grp", side)] = k; k += G
        self.nuis_start = k
        k += len(nuis)
        self.K = k
        trows, tcols, tvals = [], [], []
        for (p, s, side), c in ps_keys.items():
            for j, v in self._theta_terms(p, s, side):
                trows.append(c); tcols.append(j); tvals.append(v)
        for j in range(len(nuis)):
            trows.append(P + j); tcols.append(self.nuis_start + j); tvals.append(1.0)
        self.T = csr_matrix((tvals, (trows, tcols)), shape=(P + len(nuis), self.K))
        print(f"  {sport}: {n} stints -> {nr} rows, {len(players)} players, {P} player-season cols, "
              f"{self.K} params ({time.time() - t0:.0f}s)", flush=True)

    def _theta_terms(self, p, s, side):
        """(param index, coefficient) pairs for player p's effect in season s."""
        out = []
        if (p, "a", side) in self.pidx:
            out.append((self.pidx[(p, "a", side)], 1.0))
            for k in self.seasons:
                if self.first[p] < k <= s and (p, k, side) in self.pidx:
                    out.append((self.pidx[(p, k, side)], 1.0))
        ab = age_basis(self.birth.get(p), s)
        j = self.pidx[("age", side)]
        out += [(j + i, ab[i]) for i in range(3) if ab[i] != 0]
        g = self.grp.get(p, self.grp_default)
        j = self.pidx[("grp", side)]
        out += [(j + i, g[i]) for i in range(len(g)) if g[i] != 0]
        return out

    def prior_precision(self, h):
        lam = np.empty(self.K)
        for side in SIDES:
            lam[self._ia[side]] = 1 / h[f"tau_a_{side}"] ** 2
            lam[self._irw[side]] = 1 / h[f"tau_rw_{side}"] ** 2
            j = self.pidx[("age", side)]; lam[j:j + 3] = 1 / WIDE ** 2
            j = self.pidx[("grp", side)]; G = len(self.grp_default); lam[j:j + G] = 1 / WIDE ** 2
        lam[self.nuis_start:] = 1e-6
        return lam

    def gram(self, mask):
        """Sufficient statistics on a subset of rows."""
        w = self.w * mask
        Xw = self.X.multiply(w[:, None]).tocsr()
        M = (self.X.T @ Xw).tocsr()
        G = (self.T.T @ M @ self.T).toarray()
        c = self.T.T @ (Xw.T @ self.y)
        return {"G": G, "c": np.asarray(c).ravel(), "yy": float(w @ self.y ** 2),
                "n": int((w > 0).sum()), "logw": float(np.log(w[w > 0]).sum())}

    @staticmethod
    def log_marginal(gs, lam, s2):
        A = gs["G"] / s2 + np.diag(lam)
        L = cho_factor(A, lower=True, check_finite=False)
        b = gs["c"] / s2
        mu = cho_solve(L, b, check_finite=False)
        logdetA = 2 * np.log(np.diag(L[0])).sum()
        return (-0.5 * gs["n"] * np.log(2 * np.pi * s2) + 0.5 * gs["logw"] - 0.5 * gs["yy"] / s2
                + 0.5 * b @ mu - 0.5 * logdetA + 0.5 * np.log(lam).sum())

    def empirical_bayes(self, gs, init=None):
        keys = ["tau_a_o", "tau_a_d", "tau_rw_o", "tau_rw_d", "sigma2"]
        x0 = np.log(init if init is not None else
                    ([2.0, 2.0, 1.0, 1.0, 2e4] if self.sport == "nba" else [0.3, 0.3, 0.15, 0.15, 30.0]))
        def f(x):
            h = dict(zip(keys, np.exp(x)))
            try:
                return -self.log_marginal(gs, self.prior_precision(h), h["sigma2"])
            except np.linalg.LinAlgError:
                return 1e18
        t0 = time.time()
        r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 0.5, "maxiter": 400})
        h = dict(zip(keys, np.exp(r.x)))
        print(f"  EB: {', '.join(f'{k} {v:.4g}' for k, v in h.items())}  logml {-r.fun:.1f} "
              f"({r.nit} it, {time.time() - t0:.0f}s)", flush=True)
        return h

    def posterior(self, gs, h):
        lam = self.prior_precision(h)
        A = gs["G"] / h["sigma2"] + np.diag(lam)
        L = cho_factor(A, lower=True, check_finite=False)
        mu = cho_solve(L, gs["c"] / h["sigma2"], check_finite=False)
        return mu, L

    def draws(self, mu, L, n, seed=0):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((self.K, n))
        return mu[:, None] + solve_triangular(L[0], z, lower=True, trans="T", check_finite=False)

    def theta(self, beta, p, s, side, h=None, rng=None):
        """Effect of player p in season s for one parameter vector; with `rng`, the
        random-walk steps past the data (and a_p for unseen players) are drawn."""
        v = sum(beta[j] * c for j, c in self._theta_terms(p, s, side))
        if rng is not None and h is not None:
            if p not in self.first:
                v += rng.normal(0, h[f"tau_a_{side}"])
            steps = max(0, s - self.smax) if p in self.first else 0
            if steps:
                v += rng.normal(0, h[f"tau_rw_{side}"] * np.sqrt(steps))
        return v

    def net(self, beta, p, s, h=None, rng=None):
        return self.theta(beta, p, s, "o", h, rng) + self.theta(beta, p, s, "d", h, rng)

    def opponent_curve(self, beta, mpg):
        """Mean opponent offence and defence sums by own-lineup usage (for lineup rows)."""
        u, oo, od, w = [], [], [], []
        cache = {}
        def val(p, s, side):
            k = (p, s, side)
            if k not in cache:
                cache[k] = self.theta(beta, p, s, side)
            return cache[k]
        for h_, a_, s, sec in zip(self.st["home5"], self.st["away5"], self.st["season"], self.st["secs"]):
            for own, opp in ((h_, a_), (a_, h_)):
                u.append(sum(mpg.get((p, s), 0.0) for p in own))
                oo.append(sum(val(p, s, "o") for p in opp))
                od.append(sum(val(p, s, "d") for p in opp))
                w.append(sec)
        u, oo, od, w = map(np.array, (u, oo, od, w))
        edges = np.quantile(u, np.linspace(0, 1, 21))
        cen, vo, vd = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (u >= lo) & (u <= hi)
            cen.append(np.average(u[m], weights=w[m]))
            vo.append(np.average(oo[m], weights=w[m])); vd.append(np.average(od[m], weights=w[m]))
        return np.array(cen), np.array(vo), np.array(vd)


# ---------------------------------------------------------------- evaluation

def game_table(S):
    from dataset import side_players
    from prior_graphs import rolling_graph_tensors
    games = S.games()
    A, ok = rolling_graph_tensors(S)
    R = S.R
    G = []
    for k, g in enumerate(games):
        if not ok[k]:
            continue
        pos = S.hockey_pos(g) if S.hockey_pos else None
        sides = []
        for side, team in enumerate((g["home"], g["away"])):
            pl = side_players(g, team, R, pos)
            share = np.diagonal(A[k].numpy())[side * R:(side + 1) * R]
            sides.append([(p, float(sh)) for p, sh in zip(pl, share) if p is not None])
        G.append((g["season"], g["playoff"], g["margin"], sides, pd.Timestamp(g["date"])))
    return G


def validate(sport, test_season, std_days=14, out=None):
    m = BayesModel(sport, seasons=[s for s in load_stints(sport, STINTS[sport])["season"].unique()
                                   if s <= test_season])
    G = game_table(m.S)
    train = m.row_season < test_season
    gs = m.gram(train.astype(float))
    h = m.empirical_bayes(gs)
    mu, _ = m.posterior(gs, h)
    test_dates = sorted({x[4] for x in G if x[0] == test_season})
    start = test_dates[0]
    fits = {}

    def beta_at(date):
        k = (date - start).days // std_days
        if k not in fits:
            cut = np.datetime64(start + pd.Timedelta(days=k * std_days))
            mask = (m.row_season < test_season) | ((m.row_season == test_season) & (m.row_date < cut))
            fits[k] = m.posterior(m.gram(mask.astype(float)), h)[0]
        return fits[k]

    strength = []
    vc = {}
    for season, _, _, sides, date in G:
        beta = beta_at(date) if season == test_season else mu
        key = id(beta)
        v = []
        for side in sides:
            tot = 0.0
            for p, sh in side:
                kk = (key, p, season)
                if kk not in vc:
                    vc[kk] = m.net(beta, p, season)
                tot += sh * vc[kk]
            v.append(tot)
        strength.append(v[0] - v[1])
    strength = np.array(strength)
    gs_ = np.array([x[0] for x in G]); gp = np.array([x[1] for x in G]); gm = np.array([x[2] for x in G])
    win = (gm > 0).astype(int)
    trg = (gs_ < test_season) & ~gp & (gm != 0)
    lr = LogisticRegression().fit(strength[trg, None], win[trg])
    res = {"sport": sport, "test_season": test_season, "std_days": std_days, "hyper": h}
    for name, msk in (("test_regular", (gs_ == test_season) & ~gp), ("test_playoffs", (gs_ == test_season) & gp)):
        if msk.sum():
            p = lr.predict_proba(strength[msk, None])[:, 1]
            res[name] = {"n": int(msk.sum()), "auc": roc_auc_score(win[msk], p),
                         "logloss": log_loss(win[msk], p), "acc": float(((p > 0.5) == win[msk]).mean())}
    t = res["test_regular"]
    print(f"{sport} test {test_season}: auc {t['auc']:.4f} logloss {t['logloss']:.4f} acc {t['acc']:.3f} "
          f"(n {t['n']}, {len(fits)} refits)", flush=True)
    if out:
        (ROOT / "v3" / "results" / f"bayes_{sport}_{out}.json").write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- case fits

def fit_case(sport, season, n_draws=100):
    """Posterior values for the case season, plus an out-of-sample kappa."""
    S = load_sport(sport)
    lineup = []
    curve = None
    if sport == "nba":
        stint_seasons = set(load_stints(sport, STINTS[sport])["season"])
        if season not in stint_seasons:
            m0 = BayesModel(sport)
            gs0 = m0.gram(np.ones(len(m0.y)))
            h0 = m0.empirical_bayes(gs0)
            mu0, _ = m0.posterior(gs0, h0)
            mpg = {(p, s): v for p, s, v in zip(S.ps["player_id"], S.ps["season"], S.ps["mpg"])}
            curve = m0.opponent_curve(mu0, mpg)
            print("  opponent curve (usage, opp o, opp d):",
                  np.round(curve[0][[0, 10, -1]], 1), np.round(curve[1][[0, 10, -1]], 2),
                  np.round(curve[2][[0, 10, -1]], 2), flush=True)
            lineup = [season]
    m = BayesModel(sport, lineup_seasons=lineup, opp_curve=curve)
    gs = m.gram(np.ones(len(m.y)))
    h = m.empirical_bayes(gs)
    mu, L = m.posterior(gs, h)
    B = m.draws(mu, L, n_draws, seed=1)

    ps = S.ps
    cur = ps[ps["season"] == season]
    cand = sorted(set(cur["player_id"]) | set(m.players))
    lo, hi = cur["minutes_total"].quantile([0.10, 0.30])
    repl_ids = cur[(cur["minutes_total"] >= lo) & (cur["minutes_total"] <= hi)]["player_id"].tolist()

    def table(beta, rng=None):
        v = {p: m.net(beta, p, season, h, rng) for p in cand}
        repl = float(np.mean([v[p] for p in repl_ids]))
        return v, repl

    main = table(mu)
    rng = np.random.default_rng(7)
    draws = [table(B[:, k], rng) for k in range(n_draws)]
    od = {p: (m.theta(mu, p, season, "o"), m.theta(mu, p, season, "d")) for p in cand}

    # kappa: leave-one-season-out values on that season's games (rolling pre-game shares)
    G = game_table(S)
    xs, ys = [], []
    for s in sorted({g[0] for g in G}):
        mask = (m.row_season != s).astype(float)
        mu_s, _ = m.posterior(m.gram(mask), h)
        vc = {}
        for season_g, playoff, margin, sides, _ in G:
            if season_g != s or playoff or margin == 0:
                continue
            tot = []
            for side in sides:
                t = 0.0
                for p, sh in side:
                    if p not in vc:
                        vc[p] = m.net(mu_s, p, s)
                    t += sh * vc[p]
                tot.append(t)
            xs.append(tot[0] - tot[1]); ys.append(int(margin > 0))
        print(f"  kappa fold {s}: {len(xs)} games", flush=True)
    lr = LogisticRegression().fit(np.array(xs)[:, None], np.array(ys))
    kappa = float(lr.coef_[0, 0])
    print(f"  kappa {kappa:.4f}", flush=True)
    state = {"main": main, "draws": draws, "kappa": kappa, "hyper": h, "od": od,
             "post_sd": {p: float(np.std([d[0][p] - d[1] for d in draws])) for p in cand}}
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{sport}_{season}.pkl", "wb") as f:
        pickle.dump(state, f)
    return state


def _evaluator_base():
    from values import ValueEvaluator
    return ValueEvaluator


class BayesEvaluator(_evaluator_base()):
    """ValueEvaluator interface on the posterior: model 0 = posterior mean,
    models 1.. = posterior draws (used for intervals and P(best beats actual))."""

    def __init__(self, sport, season, order="coach", **_):
        from rosters import minutes_template
        f = CACHE / f"{sport}_{season}.pkl"
        if not f.exists():
            fit_case(sport, season)
        with open(f, "rb") as fh:
            state = pickle.load(fh)
        self.sport, self.season, self.order_mode = sport, season, order
        self.S = load_sport(sport)
        self.R = self.S.R
        self.kappa = state["kappa"]
        F = len(self.S.feature_cols)
        zero = np.zeros(F)
        self._models = [(zero, v, repl) for v, repl in [state["main"]] + state["draws"]]
        self.value_sd = state["post_sd"]
        self.od = state["od"]
        self.q = minutes_template(self.S).numpy()
        ps = self.S.ps
        cur = ps[ps["season"] == season]
        self.mpg = dict(zip(cur["player_id"], cur["toi_pg"] if sport == "nhl" else cur["mpg"]))
        self.pos = dict(zip(ps["player_id"], ps.get("pos", ps.get("position"))))
        self._vcache = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "fit"])
    ap.add_argument("--sport", default="nba")
    ap.add_argument("--test-season", type=int, default=None)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--std-days", type=int, default=14)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "validate":
        validate(a.sport, a.test_season or (2023 if a.sport == "nba" else 2025), a.std_days, a.out)
    else:
        fit_case(a.sport, a.season or (2024 if a.sport == "nba" else 2025), a.draws)


if __name__ == "__main__":
    main()
