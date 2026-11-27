"""IGGY v0: how a team redistributes minutes when players are absent (pregame information only).

Unit: (game, team). Candidates = players with a game for that team in the 30 days before the date. Rotation player =
candidate averaging >= 10 minutes over the team's previous 10 games of the SAME season (0 for missed games); the
first 10 games of each season are not covered (no in-season rotation to start from). An ABSENCE = a rotation
player with no minutes in this game; the absent set is the scenario input (in live use it comes from the injury
report). Target: each non-absent candidate's minutes, rescaled to 240 regulation minutes in overtime games.
Everything about a game is computed from games strictly before its date.

Predictors (all projected onto the feasible set: absent players 0, 0 <= m_i <= cap_i = min(44, max(B0_i, 10) + 14), sum = 240, by water-filling):
  B0 recent     mean minutes over the team's last 10 games when he played ("minutes when available")
  B1 +usage     gradient boosting on B0, availability rate, start rate, rank, team absent minutes
  B2 +history   + his mean minutes above B0 in past games (365 days) where each currently absent teammate was out
  B3 +roles     + role similarity to the absent players (box-score role vectors: per-36 rates, available all seasons)
Selection on the validation season only. Chronological split: train 2021-2024 (end years), validate 2025, test 2026
and 2027 to date. Strata: no absence / exactly one / two or more rotation absences.
Team forecasts (2025-26 blocks with saved point-in-time DUNCAN ratings, spm_ratings.pkl): margin = sum of rating x
minutes share; logistic calibration fit on 2024-25 games per minutes scheme; schemes: naive (B0 incl. absent players,
no availability info), B0 (absent removed), selected IGGY, realised minutes (diagnostic upper reference only).
Output: results/iggy/minutes.json, results/iggy/examples.json, results/iggy/failures.json   python iggy_minutes.py (PC)
"""
import json, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from masked_paths import ROOT
from provenance import provenance

LAT = ROOT / 'results/latent'
OUT = ROOT / 'results/iggy'
CAP = 44.0
ROLE = ['pts', 'fga', 'fg3a', 'fta', 'oreb', 'dreb', 'ast', 'stl', 'blk', 'tov']


def project(pred, cap, total=240.0):
    """Feasible minutes: 0 <= m <= cap, sum = total (water-filling scale on the uncapped players)."""
    m = np.clip(pred, 0, None).astype(float); fixed = np.zeros(len(m), bool)
    if m.sum() <= 0:
        m = np.ones(len(m))
    for _ in range(20):
        free = ~fixed; rem = total - m[fixed].sum()
        s = m[free].sum()
        if s <= 0:
            break
        m[free] *= rem / s
        over = free & (m > cap)
        if not over.any():
            break
        m[over] = cap[over]; fixed |= over
    short = total - m.sum()
    if short > 1e-6:                # every player at his limit: raise the limits evenly (never beyond 48)
        room = np.minimum(48.0, cap + short) - m
        m += room * short / max(room.sum(), 1e-9)
    return m


def starters():
    st = set()
    for f in ('ms2_stints.pkl', 'ms2_stints_2325.pkl', 'ms2_stints_2526.pkl', 'ms2_stints_2627.pkl'):
        d = pickle.load(open(LAT / f, 'rb')); first = {}
        for s in d['stints']:
            g = s['game_id']
            if g not in first or (s.get('period', 1), s['t0']) < first[g][0]:
                first[g] = ((s.get('period', 1), s['t0']), s['lineup'])
        for g, (_, lu) in first.items():
            for side in lu:
                for p in side:
                    st.add((g, int(p)))
    return st


def build():
    gl = pd.read_parquet(LAT / 'gamelogs.parquet')
    gl = gl[~gl.playoff & (gl['min'] > 0)].copy(); gl['d'] = pd.to_datetime(gl.game_date)
    ST = starters()
    gl['start'] = [((g, int(p)) in ST) for g, p in zip(gl.game_id, gl.player_id)]
    tot = gl.groupby(['game_id', 'team_id'])['min'].transform('sum')
    gl['min240'] = np.where(tot > 0, gl['min'] * 240.0 / tot, 0.0)
    rows = []
    for t, g in gl.groupby('team_id'):
        games = g.drop_duplicates('game_id').sort_values('d')[['game_id', 'd']].values
        gix = {gid: i for i, (gid, _) in enumerate(games)}
        pids = np.sort(g.player_id.unique()); pix = {p: j for j, p in enumerate(pids)}
        M = np.full((len(games), len(pids)), np.nan); S = np.zeros_like(M); Y = np.zeros_like(M)
        for r in g.itertuples():
            i, j = gix[r.game_id], pix[r.player_id]; M[i, j] = r.min; S[i, j] = r.start; Y[i, j] = r.min240
        dates = games[:, 1].astype('datetime64[ns]')
        sea = np.array([d.year + (d.month >= 8) for d in pd.to_datetime(games[:, 1])])
        for k in range(10, len(games)):
            if sea[k - 10] != sea[k]:      # same-season windows only: the first 10 games of a season are not covered
                continue
            prev = M[k - 10:k]; played = ~np.isnan(prev)
            lo = np.searchsorted(dates, dates[k] - np.timedelta64(30, 'D'))
            cand = np.flatnonzero((~np.isnan(M[lo:k])).any(0))
            for j in cand:
                n_ = played[:, j].sum()
                base = float(np.nanmean(prev[:, j])) if n_ else 0.0
                rows.append((games[k, 0], games[k, 1], t, int(pids[j]), base, float(np.nansum(prev[:, j])) / 10.0, n_ / 10.0,
                             float(S[k - 10:k, j][played[:, j]].mean()) if n_ else 0.0, float(Y[k, j]), bool(~np.isnan(M[k, j]))))
    D = pd.DataFrame(rows, columns=['gid', 'd', 'team', 'pid', 'base', 'mins10', 'avail', 'startr', 'y', 'played'])
    D['d'] = pd.to_datetime(D.d)
    D['season'] = D.d.dt.year + (D.d.dt.month >= 8)
    D['rot'] = D.mins10 >= 10
    D['absent'] = D.rot & ~D.played
    # role vectors: per-36 rates over each player's previous 40 games (any team), via cumulative sums
    pg = {}
    for p, h in gl.sort_values('d').groupby('player_id'):
        c = np.vstack([np.zeros((1, len(ROLE) + 1)), np.cumsum(h[['min'] + ROLE].values.astype(float), 0)])
        pg[p] = (h.d.values, c)
    return D, gl, pg


def add_features(D, pg):
    """team absent minutes, rank, prior-absence response, role similarity (all from earlier games)."""
    D = D.sort_values(['d', 'gid', 'team']).reset_index(drop=True)
    grp = D.groupby(['gid', 'team'])
    D['abs_min'] = (D.base * D.absent).groupby([D.gid, D.team]).transform('sum')
    D['rank'] = grp.base.rank(ascending=False)
    D['n_abs'] = grp.absent.transform('sum')
    # role vectors per player at each date: previous 40 games per-36
    def role(p, d):
        if p not in pg:
            return None
        ds, c = pg[p]; k = np.searchsorted(ds, np.datetime64(d)); s_ = c[k] - c[max(0, k - 40)]
        return None if s_[0] < 100 else 36 * s_[1:] / s_[0]
    rv = {}
    for p, d in D[['pid', 'd']].drop_duplicates().itertuples(index=False):
        rv[(p, d)] = role(p, d)
    allv = np.array([v for v in rv.values() if v is not None]); mu, sd = allv.mean(0), allv.std(0) + 1e-9
    # prior-absence response: when teammate j was out (rotation, absent) in the past 365 days, i's minutes - i's base
    hist = defaultdict(list)   # (team, i, j) -> [(d, delta)]
    sim, resp = np.zeros(len(D)), np.full(len(D), np.nan)
    for (gid, team), g in D.groupby(['gid', 'team'], sort=False):
        d = g.d.iloc[0]; absent = g[g.absent]
        absv = [(r.pid, r.base, rv.get((r.pid, d))) for r in absent.itertuples()]
        for idx, r in zip(g.index, g.itertuples()):
            if r.absent:
                continue
            vi = rv.get((r.pid, d))
            if vi is not None and absv:
                w = [(b_, float(((vi - mu) / sd) @ ((vj - mu) / sd)) / len(mu)) for _, b_, vj in absv if vj is not None]
                if w:
                    sim[idx] = sum(b_ * s_ for b_, s_ in w) / max(sum(b_ for b_, _ in w), 1e-9)
            past = [x for j, _, _ in absv for (dd, x) in hist.get((team, r.pid, j), []) if (d - dd).days <= 365]
            if past:
                resp[idx] = float(np.mean(past))
        for r in g.itertuples():   # record this game's responses for later games
            if not r.absent and r.played:
                for j in absent.pid:
                    hist[(team, r.pid, j)].append((d, r.y - r.base))
    D['sim'] = sim; D['resp'] = resp; D['has_resp'] = ~np.isnan(resp); D['resp0'] = np.nan_to_num(resp)
    return D


FEATS = {'B1': ['base', 'avail', 'startr', 'rank', 'abs_min', 'n_abs'],
         'B2': ['base', 'avail', 'startr', 'rank', 'abs_min', 'n_abs', 'resp0', 'has_resp'],
         'B3': ['base', 'avail', 'startr', 'rank', 'abs_min', 'n_abs', 'resp0', 'has_resp', 'sim']}


def predict_all(D, models):
    P = {'B0': D.base.values.copy()}
    for k, mdl in models.items():
        P[k] = mdl.predict(D[FEATS[k]].values)
    out = {k: np.zeros(len(D)) for k in P}
    for (gid, team), g in D.groupby(['gid', 'team'], sort=False):
        ix = g.index.values; avail = ~g.absent.values
        cap = np.minimum(CAP, np.maximum(g.base.values[avail], 10.0) + 14.0)   # realistic individual limits
        for k, v in P.items():
            out[k][ix[avail]] = project(v[ix[avail]], cap)
    return out


def main():
    D, gl, pg = build()
    D = add_features(D, pg)
    tr, va = D.season <= 2024, D.season == 2025
    te = D.season >= 2026
    fit_rows = tr & ~D.absent
    models = {k: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
              .fit(D.loc[fit_rows, f].values, D.loc[fit_rows, 'y'].values) for k, f in FEATS.items()}
    P = predict_all(D, models)
    y = D.y.values; ok = ~D.absent.values

    def errs(mask):
        return {k: dict(mae=round(float(np.mean(np.abs(v[mask] - y[mask]))), 3),
                        rmse=round(float(np.sqrt(np.mean((v[mask] - y[mask]) ** 2))), 3)) for k, v in P.items()}
    strata = {'none': D.n_abs == 0, 'single': D.n_abs == 1, 'multi': D.n_abs >= 2}
    res = dict(provenance=provenance(__file__, ['results/latent/gamelogs.parquet', 'results/latent/ms2_stints.pkl',
                                                'results/latent/ms2_stints_2325.pkl', 'results/latent/ms2_stints_2526.pkl',
                                                'results/latent/ms2_stints_2627.pkl']),
               config=dict(cap=CAP, rotation='>= 10 min over last 10 team games', candidates='played for team in prior 30 days',
                           split='train <=2024, validate 2025, test 2026-2027 (season end years)'),
               rows=dict(train=int(tr.sum()), validate=int(va.sum()), test=int(te.sum())), validate={}, test={})
    for name, S in strata.items():
        res['validate'][name] = dict(rows=int((va & S & ok).sum()), **errs((va & S & ok).values))
        res['test'][name] = dict(rows=int((te & S & ok).sum()), **errs((te & S & ok).values))
    # absorbed-minutes error in absence games: predicted vs actual gain over B0 for non-absent candidates, per team-game
    def absorbed(mask, k):   # total absolute minutes error per team-game (non-absent players)
        e = [np.abs(P[k][g.index.values[~g.absent.values]] - y[g.index.values[~g.absent.values]]).sum()
             for _, g in D[mask].groupby(['gid', 'team'])]
        return round(float(np.mean(e)), 2)
    sel = min(('B0', 'B1', 'B2', 'B3'), key=lambda k: res['validate']['single'][k]['mae'] + res['validate']['multi'][k]['mae'])
    res['selected'] = sel
    res['test_team_game_abs_error_sum'] = {k: {s: absorbed((te & strata[s]).values, k) for s in ('single', 'multi')} for k in P}
    # roles: does B3 beat B2 on test absence games? paired team-game bootstrap of MAE difference
    rng = np.random.default_rng(0)
    for a_, b_ in (('B3', 'B2'), ('B2', 'B1'), (sel, 'B0')):
        m = (te & (D.n_abs >= 1)).values & ok
        keys = D.loc[m, 'gid'].values
        ug, gi = np.unique(keys, return_inverse=True)
        da = np.bincount(gi, np.abs(P[a_][m] - y[m]) - np.abs(P[b_][m] - y[m]), len(ug)); cn = np.bincount(gi, minlength=len(ug))
        bs = [da[i].sum() / cn[i].sum() for i in (rng.integers(0, len(ug), len(ug)) for _ in range(2000))]
        res[f'mae_{a_}_minus_{b_}_test_absence'] = dict(mean=round(float(da.sum() / cn.sum()), 4), ci=np.percentile(bs, [2.5, 97.5]).round(4).tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    for k, v in P.items():
        D[f'pred_{k}'] = v
    D.to_parquet(OUT / 'minutes_predictions.parquet')
    # team forecasts
    res['team_forecast'] = team_forecasts(D, sel)
    # examples and failures (test absence games)
    ex, fails = [], []
    T = D[te & (D.n_abs >= 1)]
    for (gid, team), g in list(T.groupby(['gid', 'team']))[:40]:
        ex.append(dict(gid=gid, team=int(team), date=str(g.d.iloc[0].date()), absent=[int(p) for p in g[g.absent].pid],
                       players=[dict(pid=int(r.pid), base=round(r.base, 1), pred=round(getattr(r, f'pred_{sel}'), 1), actual=round(r.y, 1))
                                for r in g[~g.absent].sort_values('y', ascending=False).itertuples()]))
    T = T[~T.absent].assign(err=lambda x: (x[f'pred_{sel}'] - x.y))
    for r in T.reindex(T.err.abs().sort_values(ascending=False).index).head(40).itertuples():
        fails.append(dict(gid=r.gid, team=int(r.team), pid=int(r.pid), date=str(r.d.date()), base=round(r.base, 1),
                          pred=round(getattr(r, f'pred_{sel}'), 1), actual=round(r.y, 1), n_abs=int(r.n_abs), played=bool(r.played)))
    json.dump(res, open(OUT / 'minutes.json', 'w'), indent=1)
    json.dump(ex, open(OUT / 'examples.json', 'w'), indent=1); json.dump(fails, open(OUT / 'failures.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'provenance'}, indent=1))


def team_forecasts(D, sel):
    """2025-26 games inside saved DUNCAN blocks: logistic on 2024-25 margins per minutes scheme, test on 2025-26."""
    R = pickle.load(open(LAT / 'spm_ratings.pkl', 'rb'))
    blocks = sorted((b0, r['spm_lpit_h']) for b0, r in R.items() if 'spm_lpit_h' in r)
    home = {}
    gl = pd.read_parquet(LAT / 'gamelogs.parquet')
    for gid, g in gl.groupby('game_id'):
        h = g[g.matchup.str.contains('vs.')]
        if len(h):
            home[gid] = (int(h.team_id.iloc[0]), h.wl.iloc[0] == 'W')
    out = defaultdict(lambda: defaultdict(list))
    for (gid, team), g in D[D.season.isin([2025, 2026])].groupby(['gid', 'team']):
        if gid not in home:
            continue
        d = str(g.d.iloc[0].date()); blk = [b for b in blocks if b[0] <= d]
        if not blk or (pd.Timestamp(d) - pd.Timestamp(blk[-1][0])).days > 31:
            continue
        r = blk[-1][1]; pix = {int(p): i for i, p in enumerate(r['pids'])}; n = len(pix); th = r['theta']
        def val(mins):
            sh = 5 * mins / max(mins.sum(), 1e-9)
            o = sum(s * th[pix[p]] for p, s in zip(g.pid, sh) if p in pix)
            dd = sum(s * th[n + pix[p]] for p, s in zip(g.pid, sh) if p in pix)
            return o, dd
        schemes = {'naive': g.base.values, 'B0': g.pred_B0.values, 'iggy': g[f'pred_{sel}'].values,
                   'realised_DIAGNOSTIC': g.y.values}
        for k, mins in schemes.items():
            out[(gid, d)][k].append((team == home[gid][0], val(mins)))
    # point-in-time ratings exist for the 2025-26 blocks only, so calibration is ROLLING within 2025-26: block j is
    # scored with a logistic fit on the games of blocks < j (chronological); the first block is only used for fitting
    bidx = {b0: i for i, (b0, _) in enumerate(blocks)}
    rowsg = []
    for (gid, d), sch in out.items():
        if any(len(v) != 2 for v in sch.values()):
            continue
        blk = [b for b in blocks if b[0] <= d][-1][0]
        m = {}
        for k, v in sch.items():
            (hh, (ho, hd)), (aa, (ao, ad)) = sorted(v, key=lambda t: not t[0])
            m[k] = (ho + ad) - (ao + hd)
        rowsg.append((bidx[blk], int(home[gid][1]), m))
    res = dict(calibration='rolling within 2025-26 (fit on earlier blocks only)', games_total=len(rowsg))
    keys = list(rowsg[0][2]) if rowsg else []
    lls = {k: [] for k in keys}; tested = 0
    for j in range(1, len(blocks)):
        fit = [r for r in rowsg if r[0] < j]; te = [r for r in rowsg if r[0] == j]
        if len(fit) < 50 or not te:
            continue
        yf = np.array([r[1] for r in fit]); yt = np.array([r[1] for r in te]); tested += len(te)
        for k in keys:
            lr = LogisticRegression(C=1e4).fit(np.array([r[2][k] for r in fit])[:, None], yf)
            p = lr.predict_proba(np.array([r[2][k] for r in te])[:, None])[:, 1]
            lls[k].append(-np.log(np.clip(np.where(yt == 1, p, 1 - p), 1e-9, 1)))
    res['games_tested'] = tested
    lls = {k: np.concatenate(v) for k, v in lls.items() if v}
    for k, v in lls.items():
        res[k] = round(float(v.mean()), 5)
    rng = np.random.default_rng(1)
    for a_, b_ in (('iggy', 'B0'), ('iggy', 'naive'), ('B0', 'naive'), ('realised_DIAGNOSTIC', 'iggy')):
        if a_ in lls and b_ in lls:
            d_ = lls[a_] - lls[b_]
            res[f'{a_}_minus_{b_}'] = dict(mean=round(float(d_.mean()), 5), ci=np.percentile([d_[rng.integers(0, len(d_), len(d_))].mean() for _ in range(2000)], [2.5, 97.5]).round(5).tolist())
    return res


if __name__ == '__main__':
    import sys
    if '--team-only' in sys.argv:      # rerun the team-forecast step from the saved minute predictions
        D = pd.read_parquet(OUT / 'minutes_predictions.parquet'); res = json.load(open(OUT / 'minutes.json'))
        res['team_forecast'] = team_forecasts(D, res['selected']); res['team_forecast_provenance'] = provenance(__file__, ['results/iggy/minutes_predictions.parquet', 'results/latent/spm_ratings.pkl'])
        json.dump(res, open(OUT / 'minutes.json', 'w'), indent=1); print(json.dumps(res['team_forecast'], indent=1))
    else:
        main()
