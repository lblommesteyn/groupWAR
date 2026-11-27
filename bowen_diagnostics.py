"""Non-human diagnostics for what BOWEN's positioning deviation can mean (no labels are invented here).

1. Scheme follows the new team, LEAVE-SELF-OUT (AUDIT W3): bowen_eval.py correlated a mover's 2026-27 scheme shift
   with his new team's 2026-27 mean, which INCLUDES him. Recomputed with the team mean excluding the player.
2. Tendency vs situation: per possession-defender mean deviation (out-of-fold). Share of variance explained by
   (a) the situation features of that possession (BORIS defensive features: distance to rim, paint, on-handler,
   distance to man, between man and basket, speed) and (b) additionally the defender's identity (player-season
   effects, ridge). A large identity share with stable split-halves = a player tendency; whether it is QUALITY needs
   the human review.
3. Cross-situation consistency: does a player's mean deviation on on-ball possessions correlate with his mean
   deviation on help / rotation possessions (same season)? A trait that holds across situations vs a situational one.
4. Outcome link (from bowen_value.json): do deviations of the shooter's defender predict points beyond location xFG?
Output: results/bowen_review/diagnostics.json     python bowen_diagnostics.py   (Mac)
"""
import json
from collections import defaultdict
import numpy as np
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack

from masked_paths import ROOT
from provenance import provenance

W = ROOT / 'results/world'


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return None, int(ok.sum())
    r = float(np.corrcoef(a[ok], b[ok])[0, 1]); n = int(ok.sum())
    se = 1 / np.sqrt(max(n - 3, 1)); lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
    return dict(r=round(r, 3), ci=[round(float(lo), 3), round(float(hi), 3)], n=n), n


def main():
    res = {}
    # 1. leave-self-out
    Pl = json.load(open(W / 'bowen_players.json'))
    P = {(int(k.rsplit('_', 1)[0]), int(k.rsplit('_', 1)[1])): v for k, v in Pl.items()}
    tsum = defaultdict(lambda: [0.0, 0.0])
    for (p, s), v in P.items():
        tsum[(v['team'], s)][0] += v['scheme_depth'] * v['frames']; tsum[(v['team'], s)][1] += v['frames']
    both = [p for (p, s) in P if s == 2026 and (p, 2027) in P]
    movers = [p for p in both if P[(p, 2026)]['team'] != P[(p, 2027)]['team']]
    def team_mean_excl(team, s, p):
        a, b = tsum[(team, s)]; v = P[(p, s)] if P.get((p, s), {}).get('team') == team else None
        if v:
            a, b = a - v['scheme_depth'] * v['frames'], b - v['frames']
        return a / b if b > 0 else np.nan
    new_incl = [tsum[(P[(p, 2027)]['team'], 2027)][0] / tsum[(P[(p, 2027)]['team'], 2027)][1] for p in movers]
    new_excl = [team_mean_excl(P[(p, 2027)]['team'], 2027, p) for p in movers]
    old_excl = [team_mean_excl(P[(p, 2026)]['team'], 2026, p) for p in movers]
    own = [P[(p, 2027)]['scheme_depth'] for p in movers]
    res['scheme_follows_new_team'] = dict(movers=len(movers), incl_self=corr(new_incl, own)[0], leave_self_out=corr(new_excl, own)[0],
                                          old_team_leave_self_out=corr(old_excl, own)[0],
                                          note='scheme spot is conditioned on a team-season embedding, so a team-level '
                                               'correlation is expected by construction even leave-self-out')
    # 2 + 3. possession-level deviation
    z = np.load(W / 'pred.npz')
    Pp = [np.load(ROOT / f'results/archetypes/poss_{s}.npz', allow_pickle=True) for s in (2026, 2027)]
    fd = np.concatenate([p['fd'] for p in Pp]); dfn = list(Pp[0]['def_f'])
    poss, k = z['poss'], z['k'].astype(int)
    dev = np.hypot(*(z['y'].astype(np.float32) - z['ms'].astype(np.float32)).T)
    key = poss.astype(np.int64) * 5 + k; uk, inv = np.unique(key, return_inverse=True)
    md = np.bincount(inv, dev) / np.bincount(inv); nf = np.bincount(inv)
    Pi, Ki = (uk // 5).astype(int), (uk % 5).astype(int); ok = nf >= 10
    Pi, Ki, md = Pi[ok], Ki[ok], md[ok]
    X = fd[Pi, Ki].astype(float); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.column_stack([X, X ** 2])
    pid = z['meta_deff'][Pi, Ki]; season = z['meta_season'][Pi]
    ps = np.array([f'{a}_{b}' for a, b in zip(pid, season)]); ups, pix = np.unique(ps, return_inverse=True)
    half = np.array([hash(g) % 2 for g in z['meta_gid'][Pi]])
    def r2_split(feats_fn):
        out = []
        for h in (0, 1):
            tr, te = half != h, half == h
            m = feats_fn(tr, te); out.append(m)
        return round(float(np.mean(out)), 4)
    yc = md - md.mean()
    def sit_only(tr, te):
        r = Ridge(alpha=1.0).fit(X[tr], yc[tr]); p_ = r.predict(X[te]); return 1 - np.sum((yc[te] - p_) ** 2) / np.sum((yc[te] - yc[tr].mean()) ** 2)
    Id = csr_matrix((np.ones(len(pix)), (np.arange(len(pix)), pix)), shape=(len(pix), len(ups)))
    def sit_id(tr, te):
        A = hstack([csr_matrix(X), Id]).tocsr()
        r = Ridge(alpha=50.0).fit(A[tr], yc[tr]); p_ = r.predict(A[te]); return 1 - np.sum((yc[te] - p_) ** 2) / np.sum((yc[te] - yc[tr].mean()) ** 2)
    res['variance_explained_heldout_games'] = dict(possession_defenders=int(len(md)), situation_features=r2_split(sit_only),
                                                    situation_plus_player_identity=r2_split(sit_id))
    oh, pa, dr = (fd[Pi, Ki, dfn.index(c)] for c in ('on_handler', 'paint', 'd_rim'))
    sit = np.where(oh >= 0.4, 'onball', np.where(pa >= 0.5, 'interior', np.where(dr >= 22, 'perimeter', 'help')))
    agg = defaultdict(lambda: defaultdict(list))
    for v, s_, p_ in zip(md, sit, ps):
        agg[p_][s_].append(v)
    pairs = {}
    for a_, b_ in (('onball', 'help'), ('onball', 'perimeter'), ('help', 'interior'), ('perimeter', 'help')):
        xs, ys = [], []
        for p_, d in agg.items():
            if len(d[a_]) >= 30 and len(d[b_]) >= 30:
                xs.append(np.mean(d[a_])); ys.append(np.mean(d[b_]))
        pairs[f'{a_}_vs_{b_}'] = corr(xs, ys)[0]
    res['cross_situation_consistency'] = pairs
    try:
        res['outcome_link'] = {k_: v for k_, v in json.load(open(W / 'bowen_value.json')).items() if k_ != 'players'}
    except FileNotFoundError:
        res['outcome_link'] = 'bowen_value.json missing'
    res['provenance'] = provenance(__file__, ['results/world/pred.npz', 'results/world/bowen_players.json', 'results/world/bowen_value.json'])
    out = ROOT / 'results/bowen_review'; out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / 'diagnostics.json', 'w'), indent=1)
    print(json.dumps({k_: v for k_, v in res.items() if k_ != 'provenance'}, indent=1))


if __name__ == '__main__':
    main()
