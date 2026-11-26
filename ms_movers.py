"""Team-changers test: do our ratings, frozen before a move, predict how a player does on his new team better than
DARKO's? Box-score fitting can't help here: the new team's lineups and role are unseen when the ratings are fixed.

Events: trade deadlines 2021-03-25, 2022-02-10, 2023-02-09, 2024-02-08, 2025-02-06, 2026-02-05 (window: rest of that
regular season + playoffs) and the 2025 and 2026 offseasons (cutoff = opening night; window: first 45 days).
Movers: >= MIN_PRE possessions for one team in the 120 days before the cutoff and >= MIN_POST for another team in the
window. Ratings frozen at the cutoff: spm_lpit_h (lambda 1000) and DARKO point-in-time (o/d DPM at the cutoff).
Each model's stint prediction is mapped to points per 100 with a linear fit on the 30 days before the cutoff (as in
ms_trk), then scored on post-cutoff stints that include a mover:
  stint R2 (possession-weighted), and per mover: actual vs predicted on-court team net per 100 (offense rows minus
  defense rows) -> possession-weighted MSE and correlation. 95% CIs from resampling movers.
Output: results/latent/ms_movers.json     python ms_movers.py   (PC)
"""
import json
from collections import defaultdict
from datetime import date, timedelta
import numpy as np
import pandas as pd

import ms_spm as sp
import ms_stint_eval as se
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
EVENTS = [('2021 deadline', '2021-03-25', None), ('2022 deadline', '2022-02-10', None), ('2023 deadline', '2023-02-09', None),
          ('2024 deadline', '2024-02-08', None), ('2025 deadline', '2025-02-06', None), ('2026 deadline', '2026-02-05', None),
          ('2025 offseason', '2025-10-21', 45), ('2026 offseason', '2026-10-20', 45)]
MIN_PRE, MIN_POST = 300.0, 150.0
LAM = 1000.0


def team_ids():
    fs = [r'C:\Users\LukeB\dev\groupWAR\v3\data\nba\games.parquet'] + [OUT / f'games_{t}.parquet' for t in ('2325', '2526', '2627')]
    g = pd.concat([pd.read_parquet(f)[['game_id', 'team_id_home', 'team_id_away']] for f in fs]).drop_duplicates('game_id')
    return {str(r.game_id): (int(r.team_id_home), int(r.team_id_away)) for r in g.itertuples()}


def main():
    Dt = sp.Data(live=True); n = Dt.n; TM = team_ids(); Dk = se.load_darko_od()
    gid = np.array(Dt.gids)[Dt.GI]
    has = np.array([g in TM for g in gid])
    home_t = np.array([TM.get(g, (0, 0))[0] for g in gid]); away_t = np.array([TM.get(g, (0, 0))[1] for g in gid])
    off_team = np.where(Dt.H == 1, home_t, away_t); def_team = np.where(Dt.H == 1, away_t, home_t)
    season = np.array([Dt.G[g]['season'] for g in Dt.gids])[Dt.GI]
    res = {'events': {}}; allm = []
    for name, cut_s, days in EVENTS:
        cut = sp.dnum(cut_s)
        if cut > Dt.DAY.max():
            continue
        if not days:  # rest of that season (incl. playoffs)
            s = season[np.flatnonzero(Dt.DAY >= cut)[0]]; post = (Dt.DAY >= cut) & (season == s)
        else:
            post = (Dt.DAY >= cut) & (Dt.DAY < cut + days)
        pre = (Dt.DAY < cut) & (Dt.DAY >= cut - (120 if not days else 300)) & has
        post &= has
        # possessions by (player, team)
        def poss_by(mask):
            c = defaultdict(float)
            for i in np.flatnonzero(mask):
                for p in Dt.O[i]:
                    c[(p, off_team[i])] += Dt.W[i]
                for p in Dt.D[i]:
                    c[(p, def_team[i])] += Dt.W[i]
            return c
        a, b = poss_by(pre), poss_by(post)
        last_team = {}
        for (p, t), w in a.items():
            if w >= MIN_PRE and w > last_team.get(p, (0, 0))[1]:
                last_team[p] = (t, w)
        movers = {}
        for (p, t), w in b.items():
            if p in last_team and t != last_team[p][0] and w >= MIN_POST:
                movers[p] = (last_team[p][0], t)
        if len(movers) < 5:
            continue
        # frozen ratings
        th, bb = sp.fit_arm(Dt, 'spm_lpit_h', cut, LAM)
        o, d = se.darko_vec(Dk, Dt.pids, cut_s)
        vrow = np.flatnonzero((Dt.DAY >= cut - sp.VAL_DAYS) & (Dt.DAY < cut))
        if len(vrow) < 2000:  # offseason: use the last 30 days with games
            last = Dt.DAY[Dt.DAY < cut].max(); vrow = np.flatnonzero((Dt.DAY > last - sp.VAL_DAYS) & (Dt.DAY <= last))
        ours = lambda r: bb[0] + bb[1] * Dt.H[r] + th[Dt.O[r]].sum(1) + th[n + Dt.D[r]].sum(1)
        dk = lambda r: [o[Dt.O[r]].sum(1), d[Dt.D[r]].sum(1), Dt.H[r]]
        c_o = se.lin_fit([ours(vrow), Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow]); c_d = se.lin_fit(dk(vrow), Dt.Y[vrow], Dt.W[vrow])
        mv = set(movers)
        rows = np.array([i for i in np.flatnonzero(post) if mv & set(Dt.O[i]) or mv & set(Dt.D[i])])
        po, pdk = se.lin_pred(c_o, [ours(rows), Dt.H[rows]]), se.lin_pred(c_d, dk(rows))
        y, w = Dt.Y[rows], Dt.W[rows]
        ybar = np.average(Dt.Y[vrow], weights=Dt.W[vrow])
        r2 = lambda p: 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - ybar) ** 2)
        # per mover on-court net
        for p in movers:
            on_o = np.array([p in Dt.O[i] for i in rows]); on_d = np.array([p in Dt.D[i] for i in rows])
            def net(v):
                return (np.sum(w[on_o] * v[on_o]) / max(w[on_o].sum(), 1)) - (np.sum(w[on_d] * v[on_d]) / max(w[on_d].sum(), 1))
            allm.append(dict(event=name, pid=int(Dt.pids[p]), poss=float(w[on_o].sum() + w[on_d].sum()), actual=net(y),
                             ours=net(po), darko=net(pdk),
                             sse_ours=float(np.sum(w[on_o | on_d] * (y[on_o | on_d] - po[on_o | on_d]) ** 2)),
                             sse_darko=float(np.sum(w[on_o | on_d] * (y[on_o | on_d] - pdk[on_o | on_d]) ** 2)),
                             sst=float(np.sum(w[on_o | on_d] * (y[on_o | on_d] - ybar) ** 2))))
        res['events'][name] = dict(cut=cut_s, movers=len(movers), stints=int(len(rows)), r2_ours=round(float(r2(po)), 5),
                                   r2_darko=round(float(r2(pdk)), 5))
        print(name, res['events'][name], flush=True)
    M = pd.DataFrame(allm)
    rng = np.random.default_rng(0); k = len(M)
    def stats(df):
        wt = df.poss.values
        e_o = np.average((df.actual - df.ours) ** 2, weights=wt); e_d = np.average((df.actual - df.darko) ** 2, weights=wt)
        cw = lambda a, b: np.cov(a, b, aweights=wt)[0, 1] / np.sqrt(np.cov(a, a, aweights=wt)[0, 1] * np.cov(b, b, aweights=wt)[0, 1])
        return dict(mse_ours=e_o, mse_darko=e_d, r_ours=cw(df.actual, df.ours), r_darko=cw(df.actual, df.darko),
                    stint_r2_ours=1 - df.sse_ours.sum() / df.sst.sum(), stint_r2_darko=1 - df.sse_darko.sum() / df.sst.sum())
    base = stats(M); boots = [stats(M.iloc[rng.integers(0, k, k)]) for _ in range(2000)]
    ci = lambda f: np.percentile([f(b) for b in boots], [2.5, 97.5]).round(5).tolist()
    res['movers'] = int(k)
    res['pooled'] = {kk: round(float(v), 5) for kk, v in base.items()}
    res['ci'] = dict(mse_ours_minus_darko=ci(lambda s: s['mse_ours'] - s['mse_darko']),
                     r_ours_minus_darko=ci(lambda s: s['r_ours'] - s['r_darko']),
                     stint_r2_ours_minus_darko=ci(lambda s: s['stint_r2_ours'] - s['stint_r2_darko']))
    res['players'] = allm
    json.dump(res, open(OUT / 'ms_movers.json', 'w'), indent=1)
    print(json.dumps({k_: res[k_] for k_ in ('movers', 'pooled', 'ci')}, indent=1))


if __name__ == '__main__':
    main()
