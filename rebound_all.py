"""Rebound-adjusted value of every tracked shot, and per-game player rebounding sums.

For every tracked shot (results/rebounds/shots_release.pkl; landing bank and P(OREB) model cross-fitted by game
fold, trained on misses only):
  p_oreb    P(offensive rebound | miss, positions at release)   GBM on base + geometry + control features
  v_miss    p_oreb * E[pts | OREB] - (1 - p_oreb) * E[opp pts | DREB]       (means from training folds)
  xpts      tracking xFG * points                     xpts_reb = xpts + (1 - xFG) * v_miss
Per game, per player (results/rebounds/player_game.pkl), merged into metrics2 descriptions with REB=1:
  r_off_n / r_def_n          misses on the floor (offense / defense)
  r_pos_off / r_pos_def      positioning: dP(OREB) from their presence (removal), on offense / defense
  r_x_* / r_got_*            expected and actual individual rebounds
  r_sh_n, r_sh_xpts, r_sh_xreb, r_sh_vmiss   shooter: shots, xpts, rebound-adjusted xpts, value of their misses
"""
import json, pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import rebound_model as rm
import rebound_credit as rc
from masked_paths import ROOT

OUT = ROOT / 'results/rebounds'


def per_shot(args):
    r, bank = args
    w = rm.landing_weights(r, bank)
    C, posn, share = rc.per_shot((r, bank, w))
    return rm.base_feats(r), rm.geometry(r), rm.control_feats(r, w), posn, share


def main():
    S = [r for r in pickle.load(open(OUT / 'shots_release.pkl', 'rb')) if r['dist'] is not None]
    xq = {}
    for a, x in zip(map(json.loads, open(ROOT / 'results/shots/alignment.jsonl')), map(json.loads, open(ROOT / 'results/embed/shots_xfg.jsonl'))):
        if x.get('xb') is not None:
            xq[(a['game_id'], a['event_id'])] = (x['xb'], 3 if x['three'] else 2)
    S = [r for r in S if (r['game_id'], r['event_id']) in xq]
    G = np.array([r['game_id'] for r in S]); games = np.array(sorted(set(G)))
    fold_of = {g: i % 5 for i, g in enumerate(np.random.default_rng(0).permutation(games))}
    F = np.array([fold_of[g] for g in G])
    miss = np.array([not r['made'] for r in S])
    XB = np.zeros((len(S), 9)); XG = np.zeros((len(S), 9)); XC = np.zeros((len(S), 4))
    POS = np.zeros((len(S), 10)); SH = np.zeros((len(S), 10))
    with ProcessPoolExecutor(8) as ex:
        for f in range(5):
            bank = rm.landing_bank([r for r, ff, m in zip(S, F, miss) if ff != f and m])
            idx = np.where(F == f)[0]
            for i, (b, g, c, p, s) in zip(idx, ex.map(per_shot, [(S[i], bank) for i in idx], chunksize=64)):
                XB[i], XG[i], XC[i], POS[i], SH[i] = b, g, c, p, s
            print('fold', f, flush=True)
    X = np.hstack([XB, XG, XC])
    y = np.array([r['oreb'] if r['oreb'] is not None else -1 for r in S])
    pa = np.array([r.get('pts_after', np.nan) for r in S], float)
    p = np.zeros(len(S)); eo = np.zeros(len(S)); ed = np.zeros(len(S))
    for f in range(5):
        tr = (F != f) & miss
        mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50).fit(X[tr], y[tr])
        p[F == f] = mdl.predict_proba(X[F == f])[:, 1]
        eo[F == f] = np.nanmean(pa[tr & (y == 1)]); ed[F == f] = np.nanmean(pa[tr & (y == 0)])
    xb = np.array([xq[(r['game_id'], r['event_id'])][0] for r in S]); pts = np.array([xq[(r['game_id'], r['event_id'])][1] for r in S])
    v_miss = p * eo - (1 - p) * ed
    xpts = xb * pts; xreb = xpts + (1 - xb) * v_miss
    np.savez(OUT / 'shots_value.npz', game=G, event=np.array([r['event_id'] for r in S]), p_oreb=p, v_miss=v_miss,
             xpts=xpts, xpts_reb=xreb, made=~miss, POS=POS, SH=SH, X=X)
    from sklearn.metrics import roc_auc_score
    print(f'shots {len(S)} misses {miss.sum()} OREB AUC on misses {roc_auc_score(y[miss], p[miss]):.4f}')
    print(f'xpts mean {xpts.mean():.3f} sd {xpts.std():.3f}; xpts_reb mean {xreb.mean():.3f} sd {xreb.std():.3f}; '
          f'corr {np.corrcoef(xpts, xreb)[0, 1]:.3f}; v_miss sd {v_miss.std():.3f}')
    # per-game player sums; slope converts zone control to P(OREB) (as in rebound_credit)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression().fit(XC[miss][:, 2:3], y[miss]); pc = lr.predict_proba(XC[:, 2:3])[:, 1]
    slope = lr.coef_[0, 0] * pc * (1 - pc)
    pg = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for i, r in enumerate(S):
        g = r['game_id']
        sh = pg[g][r['shooter']]
        sh['r_sh_n'] += 1; sh['r_sh_xpts'] += xpts[i]; sh['r_sh_xreb'] += xreb[i]; sh['r_sh_vmiss'] += v_miss[i]
        if not miss[i]:
            continue
        for k, q in enumerate(list(r['off']) + list(r['de'])):
            o = k < 5; a = pg[g][q]; side = 'off' if o else 'def'
            a[f'r_{side}_n'] += 1
            a[f'r_pos_{side}'] += slope[i] * POS[i, k]
            a[f'r_x_{side}'] += (p[i] if o else 1 - p[i]) * SH[i, k]
            a[f'r_got_{side}'] += float(r['rebounder'] == q)
    pickle.dump({g: {q: dict(v) for q, v in d.items()} for g, d in pg.items()}, open(OUT / 'player_game.pkl', 'wb'))
    print('player_game games', len(pg))


if __name__ == '__main__':
    main()
