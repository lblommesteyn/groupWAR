"""Two-season tracking prior on the current season: does tracking (2025-26 Game Pulse archive + 2026-27 CourtSketch
download, converted by convert_courtsketch.py) improve 2026-27 player ratings over the box + half-luck model?

Weekly blocks: cutoffs 2026-10-28, 11-04, 11-11, 11-18; each tests the games up to the next cutoff (last to 11-25).
Every arm is refit at the validation start (cutoff - 30 days) to pick lambda + calibration, then at the cutoff.
  spm_lpit_h   box/minutes SPM prior + half point-in-time luck target (the live model)
  trk_prev     + tracking prior from 2025-26 tracked games only (a season-ahead prior: no current tracking)
  trk_both     + tracking prior from 2025-26 games and 2026-27 tracked games before the cutoff
  darko_cal    DARKO point-in-time, calibrated the same way
Tracking prior (as ms_trk.trk_adjust): per player, minutes-weighted pooled describe2 features across seasons;
RidgeCV of (RAPM - box prior), offense and defense separately, weights min(minutes, 1500); prediction added to the
box prior. Scored on game win (log loss, paired bootstrap vs spm_lpit_h and DARKO) and held-out stint pts/100 R2.
Inputs on the PC: results/embed + results/metrics (2025-26), results/trk2627/{player_games.pkl, shots_xfg.jsonl,
event_counts.pkl} (2026-27).   Output: results/latent/ms_trk2.json     python ms_trk2.py   (PC)
"""
import json, pickle
from collections import defaultdict
from datetime import date, timedelta
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV

import embed_eval as ee
import metrics2 as m2
import ms_spm as sp
import ms_stint_eval as se
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
T27 = ROOT / 'results/trk2627'
TRK = list(dict.fromkeys(ee.TRK_OFF + ee.TRK_DEF + m2.NEW_OFF + m2.NEW_DEF))
CUTS = ['2026-10-28', '2026-11-04', '2026-11-11', '2026-11-18', '2026-11-26']
ARMS = ['spm_lpit_h', 'trk_prev', 'trk_both']


def season_feats(PG, shots, EV, gids):
    old = m2.EV; m2.EV = EV
    try:
        return m2.describe2(PG, gids, shots) if gids else {}
    finally:
        m2.EV = old


def pool(ds):
    out = {}
    for p in set().union(*ds):
        parts = [d[p] for d in ds if p in d and d[p].get('minutes', 0) > 0 and all(np.isfinite(d[p].get(f, np.nan)) for f in TRK)]
        if not parts:
            continue
        m = np.array([q['minutes'] for q in parts])
        out[p] = {f: float(np.dot(m, [q[f] for q in parts]) / m.sum()) for f in TRK}
        out[p]['minutes'] = float(m.sum())
    return out


class Trk:
    def __init__(self, Dt):
        self.Dt = Dt
        PG = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
        self.a = (({g: d for g, d in PG.items() if d['stints']}), [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')], m2.EV)
        PG = pickle.load(open(T27 / 'player_games.pkl', 'rb'))
        self.b = (({g: d for g, d in PG.items() if d['stints']}), [json.loads(l) for l in open(T27 / 'shots_xfg.jsonl')],
                  pickle.load(open(T27 / 'event_counts.pkl', 'rb')))
        self.fa = season_feats(self.a[0], self.a[1], self.a[2], sorted(self.a[0]))  # all of 2025-26
        self._c = {}

    def feats(self, cut, current):
        if not current:
            return pool([self.fa])
        cut_s = str(date.fromordinal(cut))
        gb = sorted(g for g, d in self.b[0].items() if d['date'] < cut_s)
        return pool([self.fa, season_feats(self.b[0], self.b[1], self.b[2], gb)]), len(gb)

    def adjust(self, cut, y, m, current):
        key = (cut, id(y), current)
        if key in self._c:
            return self._c[key]
        Dt = self.Dt
        r = self.feats(cut, current); d, ng = (r if current else (r, 0))
        rows = np.flatnonzero(Dt.DAY < cut)
        th, _ = Dt.ridge(rows, y, np.zeros(2 * Dt.n), sp.LAM_TARGET, cut)
        ps = [p for p in d if p in Dt.pix]
        idx = np.array([Dt.pix[p] for p in ps]); n = Dt.n
        F = np.array([[d[p][f] for f in TRK] for p in ps]); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        w = np.minimum([d[p]['minutes'] for p in ps], 1500)
        adj = np.zeros(2 * n); info = dict(players=len(ps), games_2627=ng)
        for blk, sign in ((0, 1), (1, -1)):
            res = sign * (th[blk * n + idx] - m[blk * n + idx])
            reg = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(F, res, sample_weight=w)
            adj[blk * n + idx] = sign * reg.predict(F)
            info[f'r2_{blk}'] = round(float(reg.score(F, res, sample_weight=w)), 3)
        self._c[key] = (adj, info)
        return adj, info


def fit(Dt, T, arm, cut, lam):
    y = Dt.luck_target(cut, 0.5)
    m = Dt.spm_prior(cut, y); info = {}
    if arm != 'spm_lpit_h':
        adj, info = T.adjust(cut, y, m, current=arm == 'trk_both'); m = m + adj
    th, b = Dt.ridge(np.flatnonzero(Dt.DAY < cut), y, m, lam, cut)
    return th, b, info


def rowpred(Dt, th, b, rows):
    return b[0] + b[1] * Dt.H[rows] + th[Dt.O[rows]].sum(1) + th[Dt.n + Dt.D[rows]].sum(1)


def main():
    Dt = sp.Data(live=True); G = Dt.G; gix = {g: i for i, g in enumerate(Dt.gids)}
    T = Trk(Dt); Dk = se.load_darko_od()
    win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
    ll = lambda p, y: -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-9, 1))
    games = defaultdict(list); stints = defaultdict(list); Y = []; SY = []; SW = []; SG = []; log = {}
    for b0, b1 in zip(CUTS[:-1], CUTS[1:]):
        test = [g for g in Dt.gids if b0 <= G[g]['date'] < b1 and gix[g] in Dt.rows_of]
        vcut_s = str(date.fromisoformat(b0) - timedelta(days=sp.VAL_DAYS)); cut, vcut = sp.dnum(b0), sp.dnum(vcut_s)
        val = [g for g in Dt.gids if vcut_s <= G[g]['date'] < b0 and gix[g] in Dt.rows_of]
        trow = np.concatenate([Dt.rows_of[gix[g]] for g in test]).astype(int)
        vrow = np.concatenate([Dt.rows_of[gix[g]] for g in val]).astype(int)
        Y.append(win(test)); SY.append(Dt.Y[trow]); SW.append(Dt.W[trow]); SG.append(Dt.GI[trow])
        for arm in ARMS:
            best = None
            for lam in sp.LAMS:
                th, b, _ = fit(Dt, T, arm, vcut, lam)
                x = Dt.margins(th, b, [gix[g] for g in val])
                cal = LogisticRegression(C=1e4).fit(x[:, None], win(val))
                l = ll(cal.predict_proba(x[:, None])[:, 1], win(val)).mean()
                if best is None or l < best[0]:
                    best = (l, lam, cal, th, b)
            _, lam, cal, thv, bv = best
            th, b, info = fit(Dt, T, arm, cut, lam)
            games[arm].append(cal.predict_proba(Dt.margins(th, b, [gix[g] for g in test])[:, None])[:, 1])
            c = se.lin_fit([rowpred(Dt, thv, bv, vrow), Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow])
            stints[arm].append(se.lin_pred(c, [rowpred(Dt, th, b, trow), Dt.H[trow]]))
            log[f'{b0} {arm}'] = dict(lam=lam, **info)
            print(b0, arm, 'lam', lam, info, flush=True)
        # DARKO, point-in-time at each date, same calibration protocol
        o, d = se.darko_vec(Dk, Dt.pids, vcut_s); thd = np.concatenate([o, -d])
        xv = Dt.margins(thd, np.array([0.0, 0.0]), [gix[g] for g in val])
        cal = LogisticRegression(C=1e4).fit(xv[:, None], win(val))
        o, d = se.darko_vec(Dk, Dt.pids, b0); tht = np.concatenate([o, -d])
        games['darko_cal'].append(cal.predict_proba(Dt.margins(tht, np.array([0.0, 0.0]), [gix[g] for g in test])[:, None])[:, 1])
        fv = [o_[Dt.O[vrow]].sum(1) for o_ in [se.darko_vec(Dk, Dt.pids, vcut_s)[0]]] + [se.darko_vec(Dk, Dt.pids, vcut_s)[1][Dt.D[vrow]].sum(1), Dt.H[vrow]]
        ft = [o[Dt.O[trow]].sum(1), d[Dt.D[trow]].sum(1), Dt.H[trow]]
        stints['darko_cal'].append(se.lin_pred(se.lin_fit(fv, Dt.Y[vrow], Dt.W[vrow]), ft))
        print(b0, len(test), 'test games', len(val), 'val games', flush=True)
    y = np.concatenate(Y); sy = np.concatenate(SY); sw = np.concatenate(SW); sg = np.concatenate(SG)
    rng = np.random.default_rng(0); n = len(y)
    L = {a: ll(np.concatenate(v), y) for a, v in games.items()}
    R = {a: np.concatenate(v) for a, v in stints.items()}
    ug = np.unique(sg); gi = np.searchsorted(ug, sg)
    sst = np.bincount(gi, sw * (sy - np.average(sy, weights=sw)) ** 2, len(ug))
    sse = {a: np.bincount(gi, sw * (sy - R[a]) ** 2, len(ug)) for a in R}
    bg = rng.integers(0, len(ug), (2000, len(ug))); bgm = rng.integers(0, n, (4000, n))
    res = {'games': int(n), 'stints': int(len(sy)), 'log': log, 'arms': {}}
    for a in L:
        e = dict(ll=round(float(L[a].mean()), 4), stint_r2=round(float(1 - sse[a].sum() / sst.sum()), 5))
        for k in ('spm_lpit_h', 'darko_cal'):
            if k != a:
                e[f'dll_vs_{k}'] = np.percentile([(L[a][i] - L[k][i]).mean() for i in bgm], [2.5, 97.5]).round(4).tolist()
                e[f'dr2_vs_{k}'] = np.percentile([(sse[k][i].sum() - sse[a][i].sum()) / sst[i].sum() for i in bg], [2.5, 97.5]).round(5).tolist()
        res['arms'][a] = e
        print(a, e, flush=True)
    json.dump(res, open(OUT / 'ms_trk2.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
