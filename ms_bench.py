"""Multi-season DARKO benchmark: our stint model vs DARKO point-in-time, 2020-21 .. 2025-26 (about 6,500 games).

The 423-game tracking test cannot separate us from calibrated DARKO (CI crosses zero), so this repeats the comparison
on six seasons of play-by-play without tracking. It tests the backbone of our model (recency-weighted multi-season
stint ratings), not the tracking embedding, which only exists for 2025-26.

Rolling origin: in each test season, 30-day blocks starting 30 days after opening night (playoffs included). For a
block starting b0:
  additive / usage   ms_latent.run_arm on every stint before b0 (half-life 365 days); prior variance chosen and the
                     logistic calibration fit on the last 30 days before b0, from a fit that ends where they start
  darko_cal          seconds-weighted DARKO DPM per side from each player's latest value dated before b0 (calibration
                     games: before the start of the 30-day window), same logistic calibration on the same games
  blend              mean of the usage and darko_cal logits
Both sides use the game's realised stints (who played, for how long), so neither arm knows anything the other does not.
Players DARKO has not rated yet get -2.0 (replacement-ish); our model drops stints with players it has never seen.
Output: results/latent/msb_<season>_<b0>.npz, results/latent/ms_bench.json
  python ms_bench.py [season ...]      (PC; seasons as end years, e.g. 2021 2022; no args = summary only)
"""
import bisect, json, pickle, sys
from collections import defaultdict
from datetime import date, timedelta
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import ms_latent as ml
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
DARKO = ROOT / 'results/benchmarks/darko_history.json'
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
UNRATED = -2.0
ARMS = {'additive': (False, False), 'usage': (True, False)}
PRIOR_ARMS = {'prior_add': False, 'prior_usage': True}
INFLATE = [1.0, 2.0, 4.0]


def prior_file(season):
    return ROOT / 'results/groupwar_t' / ('prior_2026_v2.json' if season == 2026 else f'prior_pit_{season}.json')


def prior_ex(pids, P, inflate):
    """Prior mean/variance per player (o block, then d block in RAPM sign: + = allows more); rookies if unseen."""
    n = len(pids); m = np.zeros(2 * n); v = np.zeros(2 * n)
    ro, rov, rd, rdv = P['rookie']
    for i, p in enumerate(pids):
        e = P['players'].get(str(p))
        o, ov, d, dv = (e[0], e[1], e[2], e[3]) if e else (ro, rov, rd, rdv)
        m[i], v[i], m[n + i], v[n + i] = o, ov * inflate, -d, dv * inflate
    return dict(pids=pids, m=m, v=np.maximum(v, 1e-3), Z=np.zeros((n, 1)))


def prior_margins(S, G, P, fit_g, pred_g, ref, usage, inflate):
    """Fit the season's stints in fit_g under the preseason prior (empty: prior only); predicted margins for pred_g."""
    import latent_skill as ls
    Rf, Rp = ml.rows(S, G, fit_g), ml.rows(S, G, pred_g)
    pids = sorted({p for r in Rf + Rp for p in r[1] + r[2]})
    ex = prior_ex(pids, P, inflate)
    if Rf:
        mdl, pidx = ls.fit(Rf, ex, usage, False, wmult=ml.decay(Rf, G, ref))
    else:
        mdl, pidx = ls.Model(len(pids), ex['m'], ex['v'], ex['Z'], usage, False), {p: i for i, p in enumerate(pids)}
    pm = ml.margins(mdl, pidx, Rp)
    return np.array([pm.get(g, 0.0) for g in pred_g])


def run_prior_arm(S, G, P, season, train, val_tr, val, test, b0, usage):
    win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
    ins = lambda gs: [g for g in gs if G[g]['season'] == season]
    vtr, va, tr = ins(val_tr), ins(val), ins(train)
    best = None
    for k in INFLATE:
        x = prior_margins(S, G, P, vtr, va, (G[vtr[-1]]['date'] if vtr else b0), usage, k)
        cal = LogisticRegression(C=1e4).fit(x[:, None], win(va))
        l = float(-np.mean(np.log(np.clip(cal.predict_proba(x[:, None])[np.arange(len(va)), win(va)], 1e-9, 1))))
        if best is None or l < best[0]:
            best = (l, k, cal)
    _, k, cal = best
    x = prior_margins(S, G, P, tr, test, b0, usage, k)
    return cal.predict_proba(x[:, None])[:, 1], k


def load():
    S, G = [], {}
    for f in ('ms_stints.pkl', 'ms_stints_2325.pkl', 'ms_stints_2526.pkl'):
        d = pickle.load(open(OUT / f, 'rb')); S += d['stints']; G.update(d['games'])
    secs = defaultdict(lambda: (defaultdict(float), defaultdict(float)))
    for st in S:
        h, a = st['lineup']; dt = st['t1'] - st['t0']
        for p in h:
            secs[st['game_id']][0][p] += dt
        for p in a:
            secs[st['game_id']][1][p] += dt
    return S, G, secs


def load_darko():
    out = {}
    for p, h in json.load(open(DARKO)).items():
        if 'dpm' not in h:
            continue
        rows = sorted((d, v) for d, v, f in zip(h['date'], h['dpm'], h.get('future_game', [0] * len(h['date'])))
                      if v is not None and not f)
        out[int(p)] = ([r[0] for r in rows], [r[1] for r in rows])
    return out


def darko_strength(D, secs, gids, asof):
    x = []
    for g in gids:
        v = []
        for side in secs[g]:
            tot = sum(side.values()) / 5 or 1.0
            s = 0.0
            for p, t in side.items():
                if p in D:
                    j = bisect.bisect_left(D[p][0], asof) - 1
                    r = D[p][1][j] if j >= 0 else UNRATED
                else:
                    r = UNRATED
                s += t * r
            v.append(s / tot)
        x.append(v[0] - v[1])
    return np.array(x)


def blocks(G, season):
    ds = sorted(G[g]['date'] for g in G if G[g]['season'] == season)
    b = [date.fromisoformat(ds[0]) + timedelta(days=30)]
    end = date.fromisoformat(ds[-1]) + timedelta(days=1)
    while b[-1] < end:
        b.append(min(b[-1] + timedelta(days=30), end))
    return [str(d) for d in b]


def run_season(S, G, secs, D, season):
    win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
    B = blocks(G, season)
    for b0, b1 in zip(B[:-1], B[1:]):
        f = OUT / f'msb_{season}_{b0}.npz'
        if f.exists() and (not prior_file(season).exists() or 'prior_usage' in np.load(f)):
            continue
        train = sorted((g for g in G if G[g]['date'] < b0), key=lambda g: G[g]['date'])
        vcut = str(date.fromisoformat(b0) - timedelta(days=ml.VAL_DAYS))
        val_tr = [g for g in train if G[g]['date'] < vcut]; val = [g for g in train if G[g]['date'] >= vcut]
        test = sorted((g for g in G if G[g]['season'] == season and b0 <= G[g]['date'] < b1), key=lambda g: (G[g]['date'], g))
        if not test:
            continue
        if f.exists():  # model arms already done; add the prior arms
            P = dict(np.load(f)); assert list(P['gid']) == test
        else:
            P = {'win': win(test), 'gid': np.array(test), 'margin': np.array([G[g]['final'][0] - G[g]['final'][1] for g in test], float)}
            for arm, (usage, fit_) in ARMS.items():
                P[arm], info = ml.run_arm(S, G, train, val_tr, val, test, b0, usage, fit_)
        if prior_file(season).exists():
            PR = json.load(open(prior_file(season)))
            for arm, usage in PRIOR_ARMS.items():
                P[arm], P[arm + '_inflate'] = run_prior_arm(S, G, PR, season, train, val_tr, val, test, b0, usage)
        cal = LogisticRegression(C=1e4).fit(darko_strength(D, secs, val, vcut)[:, None], win(val))
        P['darko_cal'] = cal.predict_proba(darko_strength(D, secs, test, b0)[:, None])[:, 1]
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        P['blend'] = 1 / (1 + np.exp(-(lg(P['usage']) + lg(P['darko_cal'])) / 2))
        if 'prior_usage' in P:
            P['blend_prior'] = 1 / (1 + np.exp(-(lg(P['prior_usage']) + lg(P['darko_cal'])) / 2))
        np.savez(f, **P)
        print(season, b0, len(test), 'games', ' '.join(f'{k} {roc_auc_score(P["win"], P[k]):.3f}' for k in
                                                     ('additive', 'usage', 'prior_add', 'prior_usage', 'darko_cal', 'blend') if k in P), flush=True)


def summary():
    rows = defaultdict(list)
    for f in sorted(OUT.glob('msb_*.npz')):
        rows[int(f.name.split('_')[1])].append(dict(np.load(f)))
    ALL = ['additive', 'usage', 'prior_add', 'prior_usage', 'rapm', 'spm_prior', 'spm', 'spm_luck', 'spm_lpit', 'spm_lpit_h',
           'spm_bio', 'spm_sq', 'kal', 'kal_bio', 'kal_sq', 'kal_sq_fitr', 'kal_sq_fitg', 'darko_cal', 'blend', 'blend_prior']
    ll = lambda w, p: -(w * np.log(np.clip(p, 1e-6, 1)) + (1 - w) * np.log(np.clip(1 - p, 1e-6, 1)))
    rng = np.random.default_rng(0)
    res = {}

    def one(P, name):
        arms = [a for a in ALL if a in P]
        w = P['win']; bs = rng.integers(0, len(w), (2000, len(w)))
        r = dict(n=int(len(w)))
        for a in arms:
            r[a] = dict(auc=float(roc_auc_score(w, P[a])), log_loss=float(ll(w, P[a]).mean()))
        for a in [a for a in arms if a != 'darko_cal']:
            d = ll(w, P[a]) - ll(w, P['darko_cal'])
            r[a]['d_ll_vs_darko'] = [float(x) for x in np.percentile([d[b].mean() for b in bs], [2.5, 97.5])]
            r[a]['d_auc_vs_darko'] = [float(x) for x in np.percentile(
                [roc_auc_score(w[b], P[a][b]) - roc_auc_score(w[b], P['darko_cal'][b]) for b in bs[:500]], [2.5, 97.5])]
        lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        if 'margin' in P:  # calibration-free margin check: correlation of each arm's logit with the final margin
            M = P['margin']
            for a in arms:
                r[a]['margin_corr'] = float(np.corrcoef(lgt(P[a]), M)[0, 1])
                if a != 'darko_cal':
                    dc = [np.corrcoef(lgt(P[a][b]), M[b])[0, 1] - np.corrcoef(lgt(P['darko_cal'][b]), M[b])[0, 1] for b in bs[:500]]
                    r[a]['d_corr_vs_darko'] = [float(x) for x in np.percentile(dc, [2.5, 97.5])]
        res[name] = r
        print(f"{name:7s} n {r['n']:5d} " + ' '.join(f"{a} {r[a]['auc']:.3f}/{r[a]['log_loss']:.4f}/r{r[a].get('margin_corr', 0):.3f}" for a in arms)
              + ' | dll vs darko: ' + ' '.join(f"{a} {np.round(r[a]['d_ll_vs_darko'], 4)}" for a in arms if a != 'darko_cal'))
    allP = defaultdict(list)
    full = [a for a in ALL if all(a in p for Ps in rows.values() for p in Ps)]
    for s, Ps in sorted(rows.items()):
        P = {k: np.concatenate([p[k] for p in Ps]) for k in ['win', 'margin'] + [a for a in ALL if all(a in p for p in Ps)]}
        one(P, str(s))
        for k in ['win', 'margin'] + full:
            allP[k].append(P[k])
    if allP:
        one({k: np.concatenate(v) for k, v in allP.items()}, 'pooled')
        pr = [s for s, Ps in sorted(rows.items()) if all('prior_usage' in p for p in Ps)]
        if pr and len(pr) < len(rows):  # pooled over the seasons that have the prior arms
            one({k: np.concatenate([np.concatenate([p[k] for p in rows[s]]) for s in pr])
                 for k in ['win', 'margin'] + [a for a in ALL if all(a in p for s in pr for p in rows[s])]}, 'pooled_prior')
    json.dump(res, open(OUT / 'ms_bench.json', 'w'), indent=1)


def redo_darko(S, G, secs, D):
    """Recompute darko_cal and blend in saved blocks (e.g. after the DARKO history was extended)."""
    win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    for f in sorted(OUT.glob('msb_*.npz')):
        P = dict(np.load(f)); b0 = f.stem.split('_')[2]
        vcut = str(date.fromisoformat(b0) - timedelta(days=ml.VAL_DAYS))
        val = [g for g in G if vcut <= G[g]['date'] < b0]
        cal = LogisticRegression(C=1e4).fit(darko_strength(D, secs, val, vcut)[:, None], win(val))
        P['darko_cal'] = cal.predict_proba(darko_strength(D, secs, list(P['gid']), b0)[:, None])[:, 1]
        P['blend'] = 1 / (1 + np.exp(-(lg(P['usage']) + lg(P['darko_cal'])) / 2))
        if 'prior_usage' in P:
            P['blend_prior'] = 1 / (1 + np.exp(-(lg(P['prior_usage']) + lg(P['darko_cal'])) / 2))
        np.savez(f, **P)


if __name__ == '__main__':
    if sys.argv[1:] == ['redarko']:
        S, G, secs = load(); redo_darko(S, G, secs, load_darko()); summary(); sys.exit()
    seasons = [int(x) for x in sys.argv[1:]]
    if seasons:
        S, G, secs = load(); D = load_darko()
        print('darko players', len(D), flush=True)
        for s in seasons:
            run_season(S, G, secs, D, s)
    summary()
