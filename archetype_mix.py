"""Fluid archetypes: cluster every (player, half-court possession) into an offensive and a defensive archetype, then
describe each player as a mix that moves with context.

Input results/archetypes/poss_{2026,2027}.npz (archetypes.py). Features are standardised (touches clipped at 3,
distances log-scaled); a diagonal Gaussian mixture (K_OFF offense, K_DEF defense) is fit on a sample of
player-possessions and gives every player-possession soft membership probabilities.
Per player (min MIN_POSS half-court possessions on that side of the ball):
  mix           mean membership over all possessions (the player's archetype distribution), with its entropy
  by_season     the same per season;  by_month: per calendar month (>= 60 possessions)
  with / without each of the 3 most frequent teammates (>= 120 possessions each way), and with / without another
                ball-dominant teammate on the floor (one whose own mix is >= 35% "lead handler")
  stickiness    chance the player is in the same archetype on his next possession of the same game, against the
                chance implied by his overall mix (how much he switches within games)
Output: results/archetypes/mix.json      python archetype_mix.py   (Mac)
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent
A = ROOT / 'results/archetypes'
K_OFF, K_DEF = 6, 5
MIN_POSS = 400


def load():
    parts = [dict(np.load(A / f'poss_{s}.npz')) | {'season': np.full(len(np.load(A / f'poss_{s}.npz')['gid']), s)} for s in (2026, 2027)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k not in ('off_f', 'def_f')}, \
        list(parts[0]['off_f']), list(parts[0]['def_f'])


def prep(F, names):
    F = F.astype(float).copy()
    for j, n in enumerate(names):
        if n == 'touches':
            F[:, j] = np.minimum(F[:, j], 3)
        if n.startswith('d_'):
            F[:, j] = np.log1p(F[:, j])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    return (F - mu) / sd


def fit(Z, k, seed=0, n=200_000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(n, len(Z)), replace=False)
    gm = GaussianMixture(k, covariance_type='diag', n_init=4, random_state=seed).fit(Z[idx])
    return gm, gm.predict_proba(Z)


def describe(gm, Fraw, R, names):
    out = []
    lab = R.argmax(1)
    for c in range(gm.n_components):
        m = R[:, c] / R[:, c].sum()
        prof = {n: round(float(m @ Fraw[:, j]), 3) for j, n in enumerate(names)}
        out.append(dict(id=c, share=round(float(R[:, c].mean()), 3), profile=prof))
    return out


def main():
    D, OFF_F, DEF_F = load()
    N = len(D['gid']); print(N, 'possessions')
    res = {'off': {}, 'def': {}}
    for side, key, feats, k in (('off', 'fo', OFF_F, K_OFF), ('def', 'fd', DEF_F, K_DEF)):
        Fraw = D[key].reshape(-1, len(feats))                      # [N*5, F]
        pid = (D['off'] if side == 'off' else D['deff']).reshape(-1)
        Z = prep(Fraw, feats)
        gm, R = fit(Z, k)
        res[side]['archetypes'] = describe(gm, Fraw, R, feats)
        res[side]['R'], res[side]['pid'] = R.astype(np.float32), pid
        print(side); [print(' ', a['id'], a['share'], a['profile']) for a in res[side]['archetypes']]
    # per-possession context
    poss = np.repeat(np.arange(N), 5)
    month = np.array([d[:7] for d in D['date']])
    out = {'off': {}, 'def': {}}
    for side in ('off', 'def'):
        R, pid = res[side]['R'], res[side]['pid']
        mates = D['off'] if side == 'off' else D['deff']           # same-team five on that possession
        by_p = defaultdict(list)
        for r, p in enumerate(pid):
            by_p[int(p)].append(r)
        overall = {p: R[rows].mean(0) for p, rows in by_p.items() if len(rows) >= MIN_POSS}
        lead = None
        if side == 'off':  # the archetype with the highest on-ball share = lead handler
            lead = int(np.argmax([a['profile']['onball'] for a in res['off']['archetypes']]))
            ball_dom = {p for p, m in overall.items() if m[lead] >= 0.35}
        for p, mix in overall.items():
            rows = np.array(by_p[p]); P_ = poss[rows]; Rp = R[rows]
            e = dict(n=int(len(rows)), mix=mix.round(3).tolist(),
                     entropy=round(float(-(mix * np.log(mix + 1e-12)).sum() / np.log(len(mix))), 3),
                     by_season={str(s): R[rows[D['season'][P_] == s]].mean(0).round(3).tolist()
                                for s in (2026, 2027) if (D['season'][P_] == s).sum() >= 150})
            bm = {}
            for mth in sorted(set(month[P_])):
                sel = month[P_] == mth
                if sel.sum() >= 60:
                    bm[mth] = Rp[sel].mean(0).round(3).tolist()
            e['by_month'] = bm
            # teammates
            five = mates[P_]                                       # [n, 5]
            cnt = defaultdict(int)
            for row in five:
                for q in row:
                    if q != p:
                        cnt[int(q)] += 1
            ctx = []
            for q, c in sorted(cnt.items(), key=lambda t: -t[1])[:3]:
                w = (five == q).any(1)
                if w.sum() >= 120 and (~w).sum() >= 120:
                    a_, b_ = Rp[w], Rp[~w]
                    ctx.append(dict(mate=q, with_n=int(w.sum()), without_n=int((~w).sum()), with_=a_.mean(0).round(3).tolist(),
                                    without=b_.mean(0).round(3).tolist(),
                                    se=np.sqrt(a_.var(0) / len(a_) + b_.var(0) / len(b_)).round(3).tolist()))
            e['mates'] = ctx
            if side == 'off':
                w = np.array([any(int(q) in ball_dom and q != p for q in row) for row in five])
                if w.sum() >= 120 and (~w).sum() >= 120:
                    a_, b_ = Rp[w], Rp[~w]
                    e['with_handler'] = dict(with_n=int(w.sum()), without_n=int((~w).sum()), with_=a_.mean(0).round(3).tolist(),
                                             without=b_.mean(0).round(3).tolist(),
                                             se=np.sqrt(a_.var(0) / len(a_) + b_.var(0) / len(b_)).round(3).tolist())
            # stickiness: same hard archetype on the next possession of the same game
            g = D['gid'][P_]; order = np.argsort(D['t0'][P_], kind='stable')
            lab = Rp.argmax(1)[order]; gg = g[order]
            same = (lab[1:] == lab[:-1])[gg[1:] == gg[:-1]]
            hard = np.bincount(Rp.argmax(1), minlength=len(mix)) / len(Rp)
            e['stick'] = [round(float(same.mean()), 3) if len(same) else None, round(float((hard ** 2).sum()), 3)]
            out[side][str(p)] = e
        res[side].pop('R'); res[side].pop('pid')
        res[side]['lead'] = lead
    json.dump(dict(archetypes={s: res[s]['archetypes'] for s in res}, lead_handler=res['off']['lead'],
                   features={'off': OFF_F, 'def': DEF_F}, players=out), open(A / 'mix.json', 'w'))
    print({s: len(out[s]) for s in out}, 'players')


if __name__ == '__main__':
    main()
