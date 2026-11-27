"""Corrected calibration check (AUDIT.md I1): the frozen live calibration (calibrate_live.py -> live_calib_v2.json)
was fit with ms_fit.Pregame, which keeps only players who PLAYED in each game (ms_fit.py:84, gl['min'] > 0): a
realised-roster margin. Live forecasts use pregame minutes (DARKO x_minutes + injury report), so the frozen slopes
may be overconfident. This measures it; it does not change the deployed model.

Per test block of 2024-25 and 2025-26 (msb_* cutoffs), ratings = DUNCAN-2 settings (lambda 3000, half luck target,
SPM prior) fit on stints before the block start; DARKO = point-in-time o + d DPM at the block start. Two minute schemes
for each game, both weighting each player by his mean minutes over his previous 10 games (any team) before the date:
  realised    players with minutes in THIS game (the old Pregame; uses who played)
  pregame     players with a game for this team in the 30 days before the date (availability-blind: no injury info,
              no knowledge of who suits up). Known before tip.
Calibration: logistic fit on 2024-25 games, scored on 2025-26 games (chronological), separately per scheme and model;
also the frozen live_calib_v2.json applied to the pregame margins of 2025-26 (in-sample for that file's fit).
Output: results/corrected/calibration.json     python corrected_calibration.py   (PC)
"""
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import ms_spm as sp
from calibrate_live import our_margin, darko_margin
from ms_stint_eval import load_darko_od
from masked_paths import ROOT
from provenance import provenance

OUT = ROOT / 'results/corrected'
LAT = ROOT / 'results/latent'


def main():
    gl = pd.read_parquet(LAT / 'gamelogs.parquet')
    gl['d'] = pd.to_datetime(gl.game_date)
    gl = gl.sort_values('d')
    played = gl[gl['min'] > 0].copy()
    played['prev10'] = played.groupby('player_id')['min'].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean()).fillna(12.0)
    by_player = {p: (g.d.values, g['min'].values) for p, g in played.groupby('player_id')}
    by_team = {t: g[['d', 'player_id']].values for t, g in played.groupby('team_id')}
    games = {}
    for gid, g in gl.groupby('game_id'):
        h = g[g.matchup.str.contains('vs.')]; a = g[~g.matchup.str.contains('vs.')]
        if len(h) and len(a):
            games[gid] = (g.d.iloc[0], int(h.team_id.iloc[0]), int(a.team_id.iloc[0]), h, a)

    def prev10(p, d):
        if p not in by_player:
            return 0.0
        ds, ms = by_player[p]; k = np.searchsorted(ds, np.datetime64(d)); return float(ms[max(0, k - 10):k].mean()) if k else 0.0

    def shares(team, d, rows=None):
        if rows is not None:              # realised: who played in this game
            ps = [int(p) for p in rows[rows['min'] > 0].player_id]
        else:                              # pregame: played for this team in the 30 days before d
            arr = by_team.get(team)
            if arr is None:
                return None
            lo = np.datetime64(d) - np.timedelta64(30, 'D')
            ps = sorted({int(p) for dd, p in arr if lo <= dd < np.datetime64(d)})
        w = [(p, prev10(p, d)) for p in ps]; w = [(p, m) for p, m in w if m > 0]; tot = sum(m for _, m in w)
        return [(p, 5 * m / tot) for p, m in w] if tot > 0 else None

    Dt = sp.Data(); Dk = load_darko_od()
    X = defaultdict(list)                 # (season, scheme, model) -> margins ; (season) -> wins
    W = defaultdict(list)
    for f in sorted(LAT.glob('msb_*.npz')):
        season, b0 = int(f.stem.split('_')[1]), f.stem.split('_')[2]
        if season not in (2025, 2026):
            continue
        cut = sp.dnum(b0); y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
        th, b = Dt.ridge(np.flatnonzero(Dt.DAY < cut), y, m, 3000.0, cut)
        P = np.load(f)
        for g, win in zip(P['gid'], P['win']):
            if g not in games:
                continue
            d, ht, at, hrows, arows = games[g]
            T = {'realised': (shares(ht, d, hrows), shares(at, d, arows)), 'pregame': (shares(ht, d), shares(at, d))}
            if any(x is None for v in T.values() for x in v):
                continue
            for sc, TT in T.items():
                X[(season, sc, 'ours')].append(our_margin(Dt.pix, th, b, TT))
                X[(season, sc, 'darko')].append(darko_margin(Dk, b0, TT))
            W[season].append(int(win))
        print(b0, len(W[season]), flush=True)
    res = dict(provenance=provenance(__file__, ['results/latent/gamelogs.parquet', 'results/latent/ms2_stints.pkl',
                                                'results/latent/ms2_stints_2325.pkl', 'results/latent/ms2_stints_2526.pkl',
                                                'results/benchmarks/darko_history.json', 'results/latent/live_calib_v2.json']),
               games=dict(fit_2025=len(W[2025]), test_2026=len(W[2026])), arms={})
    ll = lambda p, yy: float(-np.mean(np.log(np.clip(np.where(yy == 1, p, 1 - p), 1e-9, 1))))
    y25, y26 = np.array(W[2025]), np.array(W[2026])
    frozen = json.load(open(LAT / 'live_calib_v2.json'))
    for sc in ('realised', 'pregame'):
        for mdl in ('ours', 'darko'):
            x25, x26 = np.array(X[(2025, sc, mdl)]), np.array(X[(2026, sc, mdl)])
            lr = LogisticRegression(C=1e4).fit(x25[:, None], y25)
            p26 = lr.predict_proba(x26[:, None])[:, 1]
            e = dict(slope_fit_2025=round(float(lr.coef_[0, 0]), 4), test_ll_2026=round(ll(p26, y26), 5),
                     test_mean_p_vs_rate=[round(float(p26.mean()), 4), round(float(y26.mean()), 4)])
            if sc == 'pregame':
                pf = 1 / (1 + np.exp(-(frozen[mdl]['intercept'] + frozen[mdl]['slope'] * x26)))
                e['frozen_live_calib_ll_2026'] = round(ll(pf, y26), 5); e['frozen_slope'] = frozen[mdl]['slope']
                # reliability: mean predicted vs observed in 5 bins (frozen calibration on pregame margins)
                q = np.quantile(pf, np.linspace(0, 1, 6)); bins = np.clip(np.searchsorted(q, pf, side='right') - 1, 0, 4)
                e['frozen_reliability_bins'] = [[round(float(pf[bins == k].mean()), 3), round(float(y26[bins == k].mean()), 3), int((bins == k).sum())] for k in range(5)]
            res['arms'][f'{sc}_{mdl}'] = e
    # paired ours - darko log loss on 2025-26 under each scheme (chronological calibration), game bootstrap
    rng = np.random.default_rng(0)
    for sc in ('realised', 'pregame'):
        lo = {}
        for mdl in ('ours', 'darko'):
            x25, x26 = np.array(X[(2025, sc, mdl)]), np.array(X[(2026, sc, mdl)])
            p = LogisticRegression(C=1e4).fit(x25[:, None], y25).predict_proba(x26[:, None])[:, 1]
            lo[mdl] = -np.log(np.clip(np.where(y26 == 1, p, 1 - p), 1e-9, 1))
        d = lo['ours'] - lo['darko']
        res[f'{sc}_ours_minus_darko_ll'] = dict(mean=round(float(d.mean()), 5),
                                                ci=np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5]).round(5).tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'calibration.json', 'w'), indent=1)
    print(json.dumps(res, indent=1)[:3000])


if __name__ == '__main__':
    main()
