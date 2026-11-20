"""Player rebounding credit, split into positioning (opportunity) and conversion (efficiency).

For every tracked miss (results/rebounds/rebounds.pkl), with the landing distribution cross-fitted by game fold:
  C        P(offense reaches the landing zone first), probabilistic control (rebound_model)
  pos_i    positioning: attacker i -> C - C(without i); defender j -> C(without j) - C   (zone control they add)
  share_i  expected individual share: within-team probability of being first to each landing cell (softmin of
           arrival times, temperature 0.3 s), weighted by the landing distribution and the team's control
  x_i      expected individual rebound = P(team gets it) * share_i, with P(team) = observed-calibrated logistic of C
  got_i    1 if the PBP credits player i with the rebound
Per player: positioning per miss on the floor (ΔP of an offensive rebound, via the logistic slope), expected and
actual rebounds, conversion = (actual - expected) / (expected + K) shrunk; split-half (odd/even games) reliability.
Output: results/rebounds/player_credit.json
"""
import json, pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.linear_model import LogisticRegression

import rebound_model as rm
from masked_paths import ROOT

OUT = ROOT / 'results/rebounds'
TEMP, K_CONV = 0.3, 15.0


def per_shot(args):
    r, bank = args[:2]
    w = args[2] if len(args) > 2 else rm.landing_weights(r, bank)
    pos, vel, pe = r['pos'].astype(float), r['vel'].astype(float), r['pe'].astype(float)
    m = w > 1e-4; q = rm.GRID[m]; wq = w[m] / w[m].sum()
    mu = rm.arrival(pos, vel, q); var = rm.S0 ** 2 + (pe[:, None] / rm.V_REF) ** 2 + (rm.K_DIST * mu) ** 2

    def ctrl(oi, di):
        mo, vo = rm.clark(mu[oi], var[oi]); md, vd = rm.clark(mu[di], var[di])
        return rm.Phi((md - mo) / np.sqrt(vo + vd))
    O, D = list(range(5)), list(range(5, 10))
    cq = ctrl(O, D); C = wq @ cq
    rem = np.array([wq @ ctrl([o for o in O if o != i], D) for i in O] + [wq @ ctrl(O, [d for d in D if d != j]) for j in D])
    posn = np.concatenate([C - rem[:5], rem[5:] - C])
    share = np.zeros(10)
    for team, tc in ((O, cq), (D, 1 - cq)):
        e = np.exp(-(mu[team] - mu[team].min(0)) / TEMP); s = e / e.sum(0)
        share[team] = (s * tc * wq).sum(1) / max((tc * wq).sum(), 1e-9)
    return C, posn, share


def main():
    R = [r for r in pickle.load(open(OUT / 'rebounds.pkl', 'rb')) if r['dist'] is not None]
    G = np.array([r['game_id'] for r in R])
    games = np.array(sorted(set(G)))
    fold_of = {g: i % 5 for i, g in enumerate(np.random.default_rng(0).permutation(games))}
    F = np.array([fold_of[g] for g in G])
    C = np.zeros(len(R)); POS = np.zeros((len(R), 10)); SH = np.zeros((len(R), 10))
    with ProcessPoolExecutor(8) as ex:
        for f in range(5):
            bank = rm.landing_bank([r for r, ff in zip(R, F) if ff != f])
            idx = np.where(F == f)[0]
            for i, (c, p, s) in zip(idx, ex.map(per_shot, [(R[i], bank) for i in idx], chunksize=64)):
                C[i], POS[i], SH[i] = c, p, s
            print('fold', f, flush=True)
    y = np.array([r['oreb'] for r in R])
    lr = LogisticRegression().fit(C[:, None], y)
    pt = lr.predict_proba(C[:, None])[:, 1]
    slope = lr.coef_[0, 0] * pt * (1 - pt)  # dP/dC at each shot
    print(f'logit slope on C {lr.coef_[0, 0]:.2f}; mean dP/dC {slope.mean():.3f}')
    np.savez(OUT / 'credit_arrays.npz', C=C, POS=POS, SH=SH, F=F)
    acc = defaultdict(lambda: defaultdict(float))
    odd = {g: i % 2 for i, g in enumerate(sorted(games))}
    for i, r in enumerate(R):
        ids = list(r['off']) + list(r['de'])
        for k, p in enumerate(ids):
            o = k < 5; h = f'_{odd[r["game_id"]]}'
            pr = pt[i] if o else 1 - pt[i]
            for suf in ('', h):
                a = acc[(p, 'off' if o else 'def')]
                a['n' + suf] += 1
                a['pos' + suf] += slope[i] * POS[i, k]
                a['x' + suf] += pr * SH[i, k]
                a['got' + suf] += float(r['rebounder'] == p)
    names = {}
    G_ = pickle.load(open(ROOT / 'results/embed/player_games.pkl', 'rb'))
    for d in G_.values():
        names.update(d['players'])
    out = {}
    for (p, side), a in acc.items():
        out.setdefault(str(p), {'name': names.get(p, str(p))})[side] = dict(
            n=a['n'], positioning=100 * a['pos'] / a['n'], exp_reb=a['x'], reb=a['got'],
            conversion=(a['got'] - a['x']) / (a['x'] + K_CONV),
            half=[dict(n=a[f'n_{h}'], positioning=100 * a[f'pos_{h}'] / max(a[f'n_{h}'], 1),
                       conversion=(a[f'got_{h}'] - a[f'x_{h}']) / (a[f'x_{h}'] + K_CONV),
                       reb_rate=a[f'got_{h}'] / max(a[f'n_{h}'], 1), exp_rate=a[f'x_{h}'] / max(a[f'n_{h}'], 1)) for h in (0, 1)])
    json.dump(out, open(OUT / 'player_credit.json', 'w'), indent=1)
    for side in ('off', 'def'):
        q = [v[side] for v in out.values() if side in v and min(v[side]['half'][0]['n'], v[side]['half'][1]['n']) >= 150]
        for m in ('positioning', 'conversion', 'reb_rate', 'exp_rate'):
            a = np.array([x['half'][0][m] for x in q]); b = np.array([x['half'][1][m] for x in q])
            print(f'{side} {m:12s} split-half r {np.corrcoef(a, b)[0, 1]:+.2f} (n {len(q)})')
        ranked = sorted([(v['name'], v[side]) for v in out.values() if side in v and v[side]['n'] >= 400],
                        key=lambda t: -t[1]['positioning'])
        print(f'{side} positioning TOP', [f"{n} {s['positioning']:.1f}" for n, s in ranked[:8]])
        print(f'{side} positioning BOTTOM', [f"{n} {s['positioning']:.1f}" for n, s in ranked[-5:]])
        rc = sorted(ranked, key=lambda t: -t[1]['conversion'])
        print(f'{side} conversion TOP', [f"{n} {s['conversion']:+.2f} ({s['reb']:.0f}/{s['exp_reb']:.0f})" for n, s in rc[:8]])
        print(f'{side} conversion BOTTOM', [f"{n} {s['conversion']:+.2f} ({s['reb']:.0f}/{s['exp_reb']:.0f})" for n, s in rc[-5:]])


if __name__ == '__main__':
    main()
