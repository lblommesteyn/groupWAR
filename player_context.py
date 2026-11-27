"""Contextual player evaluation: role-adjusted ratings, situational splits, lineup / teammate fit.

Ratings: DUNCAN-2, the live model (every game through yesterday, lambda 3000; the live mover pull is not applied here), offense and defense per 100
possessions (defense sign flipped so + is good). Teammates and opponents are already adjusted for by the ridge.
Roles: Gaussian mixture on the cross-season-stable tracking style features (both seasons pooled, minutes-weighted),
separately for offense and defense. Role-adjusted = the rating against players in the same role (percentile and
gap to the role's minutes-weighted mean).
Situational splits (tracked shots, both seasons): points over expected per 100 shots by shot type (xFG from defender
geometry at release); defense as nearest defender at release, same buckets. Matchup difficulty: minutes-weighted
offensive rating of the players a defender is matched to (tracking matchup frames), and the reverse for offense.
Fit (stints from 2025-26 and 2026-27): residuals of the additive model regressed on
  (a) teammate pairs (offense and defense pair indicators, ridge) -> pair effect per 100 possessions together;
  (b) offensive / defensive role-pair counts -> which role combinations beat their parts, with game-bootstrap CIs and
      a held-out check (fit on 2025-26, score on 2026-27).
Output: results/context/players.json, results/context/roles.json.     python player_context.py   (PC)
"""
import json, pickle, urllib.request
from collections import defaultdict
from datetime import date
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from sklearn.mixture import GaussianMixture

import embed_eval as ee
import metrics2 as m2
import ms_spm as sp
from ms_trk2 import season_feats, pool, T27
from masked_paths import ROOT

OUT = ROOT / 'results/context'
OFF_F = ['onball_share', 'hold_per_touch', 'pot_ast36', 'drives36', 'cs_share', 'cut_rate', 'screen_rate',
         'paint_share', 'crash_rate', 'spacing', 'own_touch', 'pass_rate']
DEF_F = ['rim_presence', 'dmatch', 'onball_def', 'help', 'matchup_usage', 'nd_rate']
K_OFF, K_DEF = 7, 5
LAM = 3000.0        # DUNCAN-2 (court-live/HOLDOUT.md)
PAIR_LAM = 1500.0
SIG = 115.0


def darko():
    req = urllib.request.Request('https://darko.app/__data.json', headers={'User-Agent': 'Mozilla/5.0 (research script)'})
    p = json.load(urllib.request.urlopen(req, timeout=60))
    d = [n for n in p['nodes'] if n and n.get('type') == 'data'][-1]['data']

    def dec(i):
        v = d[i]
        return {k: dec(j) for k, j in v.items()} if isinstance(v, dict) else [dec(j) for j in v] if isinstance(v, list) else v
    r = dec(0); cols = dict(zip(r['players']['keys'], r['players']['values']))
    return {int(row['nba_id']): row for row in (dict(zip(cols, v)) for v in zip(*cols.values()))}


def roles(F, names, k, feats, seed=0):
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    gm = GaussianMixture(k, covariance_type='diag', n_init=10, random_state=seed).fit(Z)
    lab = gm.predict(Z)
    cen = [(Z[lab == c].mean(0) if (lab == c).any() else np.zeros(Z.shape[1])) for c in range(k)]
    desc = []
    for c in range(k):
        o = np.argsort(-np.abs(cen[c]))[:3]
        desc.append(', '.join(f"{'high' if cen[c][j] > 0 else 'low'} {feats[j]}" for j in o))
    return gm, (F.mean(0), F.std(0) + 1e-9), lab, desc, cen


def ridge_sparse(rows_cols, y, w, ncol, lam):
    r, c = zip(*rows_cols) if rows_cols else ((), ())
    X = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(y), ncol))
    sw = np.sqrt(w)
    Xw = csr_matrix(X.multiply(sw[:, None])); yw = y * sw
    return lsqr(Xw, yw, damp=np.sqrt(lam))[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Dt = sp.Data(live=True); n = Dt.n; cut = sp.dnum(str(date.today()))
    y = Dt.luck_target(cut, 0.5)
    m_ = Dt.spm_prior(cut, y); th, b = Dt.ridge(np.flatnonzero(Dt.DAY < cut), y, m_, LAM, cut)
    rows = np.flatnonzero(Dt.DAY < cut)
    pred = b[0] + b[1] * Dt.H + th[Dt.O].sum(1) + th[n + Dt.D].sum(1)
    resid = y - pred
    season = np.array([Dt.G[Dt.gids[g]]['season'] for g in Dt.GI])
    off_poss = np.bincount(Dt.O[rows].ravel(), np.repeat(Dt.W[rows], 5), n)
    rec_rows = rows[season[rows] >= 2026]
    off_poss_rec = np.bincount(Dt.O[rec_rows].ravel(), np.repeat(Dt.W[rec_rows], 5), n)
    def_poss_rec = np.bincount(Dt.D[rec_rows].ravel(), np.repeat(Dt.W[rec_rows], 5), n)

    # ---- tracking (both seasons)
    PGa = {g: d for g, d in pickle.load(open(ee.E / 'player_games.pkl', 'rb')).items() if d['stints']}
    Sa = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    PGb = {g: d for g, d in pickle.load(open(T27 / 'player_games.pkl', 'rb')).items() if d['stints']}
    Sb = [json.loads(l) for l in open(T27 / 'shots_xfg.jsonl')]
    EVb = pickle.load(open(T27 / 'event_counts.pkl', 'rb'))
    fa = season_feats(PGa, Sa, m2.EV, sorted(PGa)); fb = season_feats(PGb, Sb, EVb, sorted(PGb))
    TF = pool([fa, fb])
    allf = {**{p: v for p, v in fa.items()}, **{p: v for p, v in fb.items()}}

    # ---- roles
    elig = [p for p in TF if TF[p]['minutes'] >= 500]
    role = {}
    out_roles = {}
    for side, feats, k in (('off', OFF_F, K_OFF), ('def', DEF_F, K_DEF)):
        F = np.array([[TF[p][f] for f in feats] for p in elig])
        gm, (mu, sd), lab, desc, cen = roles(F, elig, k, feats)
        allp = [p for p in TF if TF[p]['minutes'] >= 150]
        Z = (np.array([[TF[p][f] for f in feats] for p in allp]) - mu) / sd
        lab_all = gm.predict(Z); prob = gm.predict_proba(Z).max(1)
        for p, l, q in zip(allp, lab_all, prob):
            role.setdefault(p, {})[side] = (int(l), float(q))
        out_roles[side] = [dict(id=c, desc=desc[c], centroid={f: round(float(v), 2) for f, v in zip(feats, cen[c])},
                                n=int((lab == c).sum())) for c in range(k)]
        print(side, 'roles'); [print(' ', c, desc[c], (lab == c).sum()) for c in range(k)]

    # ---- ratings, role-adjusted
    ratings = {}
    for p, i in Dt.pix.items():
        if off_poss[i] < 300:
            continue
        o, d = float(th[i]), float(-th[n + i])
        ratings[p] = dict(off=o, deff=d, tot=o + d, poss=float(off_poss[i]), sd=float(SIG / np.sqrt(off_poss[i] * 0.25 + LAM)))

    def role_stats(side, key):
        grp = defaultdict(list)
        for p, r in ratings.items():
            if p in role and side in role[p]:
                grp[role[p][side][0]].append((p, r[key], TF[p]['minutes']))
        out = {}
        for c, v in grp.items():
            vals = np.array([x[1] for x in v]); w = np.array([x[2] for x in v])
            mean = float(np.average(vals, weights=w))
            for p, x, _ in v:
                out[p] = dict(pct=float((vals < x).mean() * 100), vs_role=x - mean, peers=len(v))
        return out
    ro, rd = role_stats('off', 'off'), role_stats('def', 'deff')
    tot_all = np.array([r['tot'] for r in ratings.values()])

    # ---- matchup difficulty (tracking matchup frames)
    guard, guarded_by = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
    for PG in (PGa, PGb):
        for g in PG.values():
            for (q, o_), c in g['pairs'].items():
                if o_ in Dt.pix:
                    guard[q][0] += c * th[Dt.pix[o_]]; guard[q][1] += c
                if q in Dt.pix:
                    guarded_by[o_][0] += c * -th[n + Dt.pix[q]]; guarded_by[o_][1] += c

    # ---- situational splits (tracked shots, both seasons)
    def bucket(s):
        if s['three']:
            t = s.get('touch_s')
            return 'catch-and-shoot 3' if t is not None and t <= 1.2 else 'off-dribble 3' if t is not None and t >= 2.0 else 'other 3'
        return 'rim (<6 ft)' if s['dist'] < 6 else 'floater/short (6-14 ft)' if s['dist'] < 14 else 'long 2 (14+ ft)'
    off_split = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))  # n, pts, xpts, var
    def_split = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    for s in Sa + Sb:
        if not s['trk_ok'] or s['xb'] is None:
            continue
        v = 3 if s['three'] else 2; x = s['xb']; k = bucket(s)
        a = np.array([1, v * s['made'], v * x, v * v * x * (1 - x)])
        off_split[s['shooter']][k] += a
        if s.get('nd_id') and s.get('d1') is not None and s['d1'] <= 6:
            def_split[s['nd_id']][k] += a

    def splits(tab):
        out = []
        for k, (cnt, pts, xp, var) in sorted(tab.items(), key=lambda t: -t[1][0]):
            if cnt >= 15:
                out.append(dict(kind=k, n=int(cnt), pts=round(pts / cnt, 3), xpts=round(xp / cnt, 3),
                                poe100=round(100 * (pts - xp) / cnt, 1), se100=round(100 * np.sqrt(var) / cnt, 1)))
        return out

    # ---- fit (a): teammate pairs
    rr = np.flatnonzero((season >= 2026) & (Dt.DAY < cut))
    col = {}; trip = []; together = defaultdict(float)
    for j, i in enumerate(rr):
        for side, five in ((0, Dt.O[i]), (1, Dt.D[i])):
            for a_, b_ in combinations(sorted(five), 2):
                c = col.setdefault((side, a_, b_), len(col)); trip.append((j, c))
                if side == 0:
                    together[(a_, b_)] += Dt.W[i]
    coef = ridge_sparse(trip, resid[rr], Dt.W[rr], len(col), PAIR_LAM)
    pair = {}
    for (side, a_, b_), c in col.items():
        pair.setdefault((a_, b_), [0.0, 0.0])[side] = coef[c]
    pair_eff = {k: (v[0] - v[1], SIG / np.sqrt(together[k] + PAIR_LAM)) for k, v in pair.items() if together[k] > 0}
    mates = defaultdict(list)
    for (a_, b_), (e, se_) in pair_eff.items():
        if together[(a_, b_)] >= 300:
            for p, q in ((a_, b_), (b_, a_)):
                mates[Dt.pids[p]].append((Dt.pids[q], float(e), float(se_), float(together[(a_, b_)])))

    # ---- fit (b): role-pair synergy, with held-out check
    rid = lambda i, side: role.get(Dt.pids[i], {}).get(side, (None,))[0]
    ko, kd = K_OFF, K_DEF
    oc = {c: j for j, c in enumerate(combinations_with(ko))}; dc = {c: len(oc) + j for j, c in enumerate(combinations_with(kd))}

    def role_X(idx):
        trip = []
        for j, i in enumerate(idx):
            ro_ = [rid(p, 'off') for p in Dt.O[i]]; rd_ = [rid(p, 'def') for p in Dt.D[i]]
            for a_, b_ in combinations(range(5), 2):
                if ro_[a_] is not None and ro_[b_] is not None:
                    trip.append((j, oc[tuple(sorted((ro_[a_], ro_[b_])))]))
                if rd_[a_] is not None and rd_[b_] is not None:
                    trip.append((j, dc[tuple(sorted((rd_[a_], rd_[b_])))]))
        r, c = zip(*trip)
        return csr_matrix((np.ones(len(r)), (r, c)), shape=(len(idx), len(oc) + len(dc)))

    def fit_roles(idx, lam=2e4):
        X = role_X(idx); sw = np.sqrt(Dt.W[idx])
        return lsqr(csr_matrix(X.multiply(sw[:, None])), resid[idx] * sw, damp=np.sqrt(lam))[0], X
    tr = rr[season[rr] == 2026]; te = rr[season[rr] == 2027]
    beta_tr, _ = fit_roles(tr); Xte = role_X(te)
    yt, wt = resid[te], Dt.W[te]
    sse0 = np.sum(wt * yt ** 2); sse1 = np.sum(wt * (yt - Xte @ beta_tr) ** 2)
    held = dict(train='2025-26', test='2026-27 to date', r2_gain=float(1 - sse1 / sse0), test_rows=int(len(te)))
    beta, _ = fit_roles(rr)
    gs = Dt.GI[rr]; ug = np.unique(gs); rng = np.random.default_rng(0); boots = []
    by_g = defaultdict(list)
    for j, g in enumerate(gs):
        by_g[g].append(rr[j])
    for _ in range(100):
        pick = rng.choice(ug, len(ug))
        idx = np.concatenate([by_g[g] for g in pick])
        boots.append(fit_roles(idx)[0])
    boots = np.array(boots)
    syn = dict(held_out=held, off=[], deff=[])
    for (a_, b_), j in oc.items():
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        syn['off'].append(dict(roles=[a_, b_], effect=float(beta[j]), ci=[float(lo), float(hi)]))
    for (a_, b_), j in dc.items():
        lo, hi = np.percentile(-boots[:, j], [2.5, 97.5])
        syn['deff'].append(dict(roles=[a_, b_], effect=float(-beta[j]), ci=[float(lo), float(hi)]))
    print('role-pair held-out', held, flush=True)

    # ---- assemble
    dk = darko()
    players = []
    for p, r in ratings.items():
        i = Dt.pix[p]
        if off_poss_rec[i] < 500 and p not in dk:
            continue
        dd = dk.get(p, {})
        e = dict(id=p, name=dd.get('player_name') or str(p), team=dd.get('team_name'), pos=dd.get('position'),
                 age=round(dd['age'], 1) if dd.get('age') else None,
                 off=round(r['off'], 2), deff=round(r['deff'], 2), tot=round(r['tot'], 2), sd=round(r['sd'], 2),
                 tot_pct=round(float((tot_all < r['tot']).mean() * 100), 1),
                 poss_recent=int(off_poss_rec[i]), def_poss_recent=int(def_poss_rec[i]),
                 darko=dict(dpm=dd.get('dpm'), o=dd.get('o_dpm'), d=dd.get('d_dpm')) if dd else None,
                 trk_minutes=round(TF[p]['minutes']) if p in TF else 0)
        if p in role:
            for side, key, rs in (('off', 'off_role', ro), ('def', 'def_role', rd)):
                if side in role[p]:
                    e[key] = dict(id=role[p][side][0], conf=round(role[p][side][1], 2),
                                  **({k: round(v, 2) for k, v in rs[p].items()} if p in rs else {}))
        if p in TF:
            e['style'] = {f: round(TF[p][f], 3) for f in OFF_F + DEF_F}
        if guard.get(p, [0, 0])[1] > 500:
            e['guards_off_rating'] = round(guard[p][0] / guard[p][1], 2)
        if guarded_by.get(p, [0, 0])[1] > 500:
            e['guarded_by_def_rating'] = round(guarded_by[p][0] / guarded_by[p][1], 2)
        e['shots'] = splits(off_split.get(p, {})); e['defended'] = splits(def_split.get(p, {}))
        m = sorted(mates.get(p, []), key=lambda t: -t[1])
        e['mates'] = [dict(id=q, name=dk.get(q, {}).get('player_name') or str(q), effect=round(ef, 2), se=round(s_, 2),
                           poss=int(ps)) for q, ef, s_, ps in m]
        players.append(e)
    json.dump(dict(generated=str(date.today()), ratings_through=max(Dt.G[g]['date'] for g in Dt.gids),
                   roles=out_roles, synergy=syn, players=players), open(OUT / 'players.json', 'w'))
    print(len(players), 'players written', flush=True)


def combinations_with(k):
    return [(a, b) for a in range(k) for b in range(a, k)]


if __name__ == '__main__':
    main()
