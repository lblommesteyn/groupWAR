"""Trade-deadline role-change test: does the latent-skill model predict what happens to traded players?

Natural experiment: a player traded at the deadline gets new teammates, a new role and a new usage share overnight.
For each deadline in the v3 lineup data (2020-02-06, 2021-03-25, 2022-02-10, 2023-02-09) fit each arm on every
stint before the deadline (recency-weighted, as ms_latent.py), then predict the rest of that regular season.

Traded players: >= MIN_PRE possessions on court for one team before the deadline and >= MIN_POST for a different
team after it. Control: every other player with >= MIN_POST possessions after the deadline.
Tests, on stints after the deadline whose ten players were all seen in training:
  1. usage share: player's share of his lineup's usage (FGA + 0.44 FTA + TOV) on the new team.
     usage arm prediction = softmax(u) over the realised post-deadline lineups; baseline = his pre-deadline share.
  2. on-court offense: possession-weighted points per 100 of his team's offense while he is on the floor,
     predicted from the realised lineups by each arm (additive / usage + load / full), vs actual.
     Error = possession-weighted squared error over players; player-bootstrap 95% CIs for differences vs additive.
Output: results/latent/trade_test.json.   Usage (PC): python trade_test.py
"""
import json, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

import latent_skill as ls
import ms_latent as ml
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
DEADLINES = {2020: '2020-02-06', 2021: '2021-03-25', 2022: '2022-02-10', 2023: '2023-02-09'}
MIN_PRE, MIN_POST = 300.0, 150.0
ARMS = {'additive': (False, False, None), 'usage': (True, False, None), 'full': (True, True, 1000.0)}
V = 4.0  # ridge prior variance chosen in most ms_latent blocks


def team_rows(S, G, gids, gm):
    """ms_latent rows + offense/defense team ids."""
    R = ml.rows(S, G, gids)
    out = []
    for r in R:
        h, a = gm[r[0]]
        out.append(r + ((h, a) if r[3] == 1 else (a, h),))
    return R, out


@torch.no_grad()
def outputs(mdl, pidx, R):
    O = torch.tensor([[pidx[p] for p in r[1]] for r in R]); D = torch.tensor([[pidx[p] for p in r[2]] for r in R])
    H = torch.tensor([r[3] for r in R], dtype=torch.float64)
    o = mdl(O, D, H)
    sh = torch.softmax(mdl.u[O], 1).numpy()
    return o['yhat'].numpy(), sh


def boot(err, base, w, n=2000, seed=0):
    rng = np.random.default_rng(seed); k = len(err)
    d = [(w[b] * (err[b] - base[b])).sum() / w[b].sum() for b in rng.integers(0, k, (n, k))]
    return float((w * (err - base)).sum() / w.sum()), [float(x) for x in np.percentile(d, [2.5, 97.5])]


def main():
    D_ = pickle.load(open(OUT / 'ms_stints.pkl', 'rb')); S, G = D_['stints'], D_['games']
    gp = pd.read_parquet(ml.ROOT.parent / 'groupWAR/v3/data/nba/games.parquet').drop_duplicates('game_id').set_index('game_id')
    gm = {g: (int(gp.loc[g, 'team_id_home']), int(gp.loc[g, 'team_id_away'])) for g in G if g in gp.index}
    res = {'deadlines': {}}
    pooled = defaultdict(list)
    for season, dl in DEADLINES.items():
        train = sorted((g for g in G if G[g]['date'] < dl and g in gm), key=lambda g: G[g]['date'])
        post = [g for g in G if G[g]['date'] >= dl and G[g]['season'] == season and not G[g]['playoff'] and g in gm]
        pre_season = [g for g in train if G[g]['season'] == season and not G[g]['playoff']]
        _, Rpre = team_rows(S, G, pre_season, gm)
        _, Rpost = team_rows(S, G, post, gm)
        # player -> team possessions and usage before / after
        poss = {k: defaultdict(lambda: defaultdict(float)) for k in ('pre', 'post')}
        use = {k: defaultdict(lambda: [0.0, 0.0]) for k in ('pre', 'post')}
        for k, RR in (('pre', Rpre), ('post', Rpost)):
            for r in RR:
                tot = sum(r[6])
                for p, c in zip(r[1], r[6]):
                    poss[k][p][r[7][0]] += r[5]; use[k][p][0] += c; use[k][p][1] += tot
        main_team = lambda k, p: max(poss[k][p].items(), key=lambda t: t[1]) if poss[k].get(p) else (None, 0.0)
        traded, control = [], []
        for p in poss['post']:
            tp, np_ = main_team('post', p); tq, nq = main_team('pre', p)
            if np_ < MIN_POST:
                continue
            if tq is not None and nq >= MIN_PRE and tq != tp:
                traded.append(p)
            elif tq == tp and nq >= MIN_PRE:
                control.append(p)
        print(f'{season} deadline {dl}: train {len(train)} games, post {len(post)} games, traded {len(traded)}, control {len(control)}', flush=True)
        Rtr = ml.rows(S, G, train)
        wm = ml.decay(Rtr, G, dl)
        fits = {}
        for arm, (usage, fit_, lw) in ARMS.items():
            fits[arm] = ls.fit(Rtr, ml.ex_for(Rtr, V), usage, fit_, l2w=lw or ls.L2_W, wmult=wm)
            print(f'  fit {arm}  g {float(torch.nn.functional.softplus(fits[arm][0].g0)):.3f}', flush=True)
        pidx = fits['additive'][1]
        Rk = [r for r in Rpost if all(p in pidx for p in r[1] + r[2])]
        y = np.array([r[4] for r in Rk]); w = np.array([r[5] for r in Rk])
        pred = {}; share = None
        for arm, (mdl, px) in fits.items():
            pred[arm], sh = outputs(mdl, px, Rk)
            if arm == 'usage':
                share = sh
        # per player: on-court offense actual vs predicted, and usage share
        for grp, players in (('traded', traded), ('control', control)):
            ps = set(players)
            acc = defaultdict(lambda: defaultdict(float))
            for i, r in enumerate(Rk):
                tot = sum(r[6])
                for j, p in enumerate(r[1]):
                    if p not in ps or r[7][0] != main_team('post', p)[0]:
                        continue
                    a = acc[p]; a['w'] += w[i]; a['y'] += w[i] * y[i]
                    for arm in pred:
                        a[arm] += w[i] * pred[arm][i]
                    a['u_act'] += r[6][j]; a['u_tot'] += tot; a['u_pred'] += tot * share[i, j]
            for p, a in acc.items():
                if a['w'] < MIN_POST or a['u_tot'] <= 0:
                    continue
                u0 = use['pre'][p][0] / max(use['pre'][p][1], 1e-9)
                pooled[grp].append(dict(season=season, pid=int(p), w=a['w'], y=a['y'] / a['w'],
                                        **{arm: a[arm] / a['w'] for arm in pred},
                                        share_act=a['u_act'] / a['u_tot'], share_pred=a['u_pred'] / a['u_tot'], share_pre=u0))
        res['deadlines'][dl] = dict(train_games=len(train), post_games=len(post), traded=len(traded), control=len(control))
    for grp, L in pooled.items():
        w = np.array([d['w'] for d in L]); y = np.array([d['y'] for d in L])
        err = {arm: (np.array([d[arm] for d in L]) - y) ** 2 for arm in ARMS}
        sa = np.array([d['share_act'] for d in L]); sp = np.array([d['share_pred'] for d in L]); s0 = np.array([d['share_pre'] for d in L])
        r = dict(n_players=len(L),
                 offense_wmse={arm: float((w * e).sum() / w.sum()) for arm, e in err.items()},
                 offense_vs_additive={arm: boot(err[arm], err['additive'], w) for arm in ARMS if arm != 'additive'},
                 share_mae_model=float(np.average(np.abs(sp - sa), weights=w)), share_mae_no_change=float(np.average(np.abs(s0 - sa), weights=w)),
                 share_vs_no_change=boot(np.abs(sp - sa), np.abs(s0 - sa), w),
                 share_change_corr=float(np.corrcoef(sp - s0, sa - s0)[0, 1]))
        res[grp] = r
        print(f"\n{grp}: {len(L)} players")
        print('  on-court offense wMSE', {k: round(v, 2) for k, v in r['offense_wmse'].items()}, ' vs additive', r['offense_vs_additive'])
        print(f"  usage share MAE: model {r['share_mae_model']:.4f}  no-change {r['share_mae_no_change']:.4f}  diff {r['share_vs_no_change']}"
              f"  corr(pred change, actual change) {r['share_change_corr']:.3f}", flush=True)
    res['players'] = pooled
    json.dump(res, open(OUT / 'trade_test.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
