"""Phase 2: does tracking add anything on top of the Phase-1 model (box prior + stints)? 2025-26 blocks after tracking
starts (cutoffs 2026-02-18 .. 2026-05-19, about 490 test games), same games and calibration as ms_bench.

Two uses of tracking, neither as extra game-outcome features:
  trk_x      a cleaner target. Every tracked shot's actual points are replaced by its tracking expected points
             (xFG from defender geometry at release x shot value, results/rebounds/shots_value.npz). Shots carry the
             ten players on the floor, so the per-shot difference (xpts - pts) is summed by (game, offense five,
             defence five) and spread over the matching stint rows by possessions. Untracked games keep actual points.
  trk_prior  a better prior. At each cutoff, the plain multi-season RAPM minus the box prior (what the box score misses)
             is regressed, minutes-weighted with leave-one-out ridge, on each player's tracking descriptions from
             tracked games before the cutoff (metrics2.describe2: shot making/quality, gravity, drives, passing,
             contests, rim protection, matchups, closeouts, help ...). The fitted residual is added to the box prior.
             Target is player-level and stable (every possession of up to seven seasons), not a few hundred games.
  trk_both   both.
Compared with spm (Phase 1) and darko_cal on the same games: log loss, AUC, margin correlation, and the stint-level
held-out R2 of points per 100 (Phase 3 test). Output: results/latent/ms_trk.json   (PC: python ms_trk.py)
"""
import json, pickle
from collections import defaultdict
from datetime import date, timedelta
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score

import embed_eval as ee
import metrics2 as m2
import ms_spm as sp
import ms_stint_eval as se
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
TRK = list(dict.fromkeys(ee.TRK_OFF + ee.TRK_DEF + m2.NEW_OFF + m2.NEW_DEF))
ARMS = ['trk_prior', 'trk_x', 'trk_both', 'trk_prior_l']  # trk_prior_l: tracking prior + half luck-adjusted target (best Phase 1 base)
BLOCKS = ['2026-02-18', '2026-03-20', '2026-04-19', '2026-05-19']
MIN_TRK_GAMES = 30


class TData(sp.Data):
    def __init__(self):
        super().__init__()
        sv = np.load(ROOT / 'results/rebounds/shots_value.npz')
        rel = pickle.load(open(ROOT / 'results/rebounds/shots_release.pkl', 'rb'))
        delta = defaultdict(float)
        for k, r in enumerate(rel):
            delta[(r['game_id'], tuple(sorted(r['off'])), tuple(sorted(r['de'])))] += sv['xpts'][k] - (3 if r['three'] else 2) * bool(r['made'])
        P = np.array(self.pids)
        by_key = defaultdict(list)
        for i in range(len(self.Y)):
            by_key[(self.gids[self.GI[i]], tuple(sorted(P[self.O[i]])), tuple(sorted(P[self.D[i]])))].append(i)
        self.YX = self.Y.copy(); hit = 0
        for key, d in delta.items():
            rows = by_key.get(key)
            if not rows:
                continue
            hit += 1
            w = self.W[rows]
            self.YX[rows] += 100 * d * (w / w.sum()) / w
        print(f'tracked shot keys {len(delta)}, matched to stint rows {hit} ({hit / len(delta):.1%}); '
              f'target sd in tracked games {np.std(self.Y[self.YX != self.Y]):.1f} -> {np.std(self.YX[self.YX != self.Y]):.1f}', flush=True)
        self.PG = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
        self.PG = {g: d for g, d in self.PG.items() if d['stints']}
        self.shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
        self._tc = {}

    def trk_adjust(self, cut, y, m):
        """Tracking regression of (RAPM - box prior) at cutoff `cut` -> additive adjustment to m (same layout)."""
        key = (cut, id(y))
        if key in self._tc:
            return self._tc[key]
        cut_s = str(date.fromordinal(cut))
        gids = sorted(g for g, d in self.PG.items() if d['date'] < cut_s)
        adj = np.zeros(2 * self.n); info = dict(games=len(gids))
        if len(gids) >= MIN_TRK_GAMES:
            d = m2.describe2(self.PG, gids, self.shots)
            rows = np.flatnonzero(self.DAY < cut)
            th, _ = self.ridge(rows, y, np.zeros(2 * self.n), sp.LAM_TARGET, cut)
            ps = [p for p in d if p in self.pix and d[p].get('minutes', 0) > 0 and all(np.isfinite(d[p].get(f, np.nan)) for f in TRK)]
            idx = np.array([self.pix[p] for p in ps]); n = self.n
            F = np.array([[d[p][f] for f in TRK] for p in ps]); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
            w = np.minimum([d[p]['minutes'] for p in ps], 1500)
            for blk, sign in ((0, 1), (1, -1)):
                res = sign * (th[blk * n + idx] - m[blk * n + idx])
                reg = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(F, res, sample_weight=w)
                adj[blk * n + idx] = sign * reg.predict(F)
                info[f'alpha_{blk}'] = float(reg.alpha_); info[f'r2_{blk}'] = float(reg.score(F, res, sample_weight=w))
            info['players'] = len(ps)
        self._tc = {k: v for k, v in self._tc.items() if k[0] >= cut - 40}
        self._tc[key] = (adj, info)
        return adj, info


def fit_arm(Dt, arm, cut, lam):
    y = Dt.YX if arm in ('trk_x', 'trk_both') else Dt.luck_target(cut, 0.5) if arm in ('trk_prior_l', 'spm_lpit_h') else Dt.Y
    rows = np.flatnonzero(Dt.DAY < cut)
    m = Dt.spm_prior(cut, y)
    info = {}
    if arm in ('trk_prior', 'trk_both', 'trk_prior_l'):
        adj, info = Dt.trk_adjust(cut, y, m); m = m + adj
    th, b = Dt.ridge(rows, y, m, lam, cut)
    return th, b, info


def main():
    Dt = TData(); G = Dt.G; gix = {g: i for i, g in enumerate(Dt.gids)}
    win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
    Dk = se.load_darko_od()
    allP = defaultdict(list); stint = defaultdict(list); log = {}
    for b0 in BLOCKS:
        f = OUT / f'msb_2026_{b0}.npz'
        P = dict(np.load(f)); test = list(P['gid'])
        vcut_s = str(date.fromisoformat(b0) - timedelta(days=sp.VAL_DAYS)); cut, vcut = sp.dnum(b0), sp.dnum(vcut_s)
        val = [g for g in Dt.gids if vcut_s <= G[g]['date'] < b0]
        test_rows = np.concatenate([Dt.rows_of[gix[g]] for g in test]).astype(int)
        val_rows = np.flatnonzero((Dt.DAY >= vcut) & (Dt.DAY < cut))
        rowpred = {}
        for arm in ['trk_prior', 'trk_prior_l', 'spm', 'spm_lpit_h']:  # trk_x / trk_both: done, both hurt
            best = None
            for lam in sp.LAMS:
                th, b, _ = (fit_arm(Dt, arm, vcut, lam) if arm not in ('spm', 'spm_lpit_h') else (*sp.fit_arm(Dt, arm, vcut, lam), {}))
                x = Dt.margins(th, b, [gix[g] for g in val])
                cal = LogisticRegression(C=1e4).fit(x[:, None], win(val))
                l = -np.mean(np.log(np.clip(cal.predict_proba(x[:, None])[np.arange(len(val)), win(val)], 1e-9, 1)))
                if best is None or l < best[0]:
                    best = (l, lam, cal, th, b)
            _, lam, cal, thv, bv = best
            th, b, info = (fit_arm(Dt, arm, cut, lam) if arm not in ('spm', 'spm_lpit_h') else (*sp.fit_arm(Dt, arm, cut, lam), {}))
            if arm in ('spm', 'spm_lpit_h'):
                pass  # keep the Phase-1 game predictions already saved
            else:
                P[arm] = cal.predict_proba(Dt.margins(th, b, [gix[g] for g in test])[:, None])[:, 1]
            log[f'{b0}_{arm}'] = dict(lam=lam, **info)
            pv = bv[0] + bv[1] * Dt.H[val_rows] + thv[Dt.O[val_rows]].sum(1) + thv[Dt.n + Dt.D[val_rows]].sum(1)
            pt = b[0] + b[1] * Dt.H[test_rows] + th[Dt.O[test_rows]].sum(1) + th[Dt.n + Dt.D[test_rows]].sum(1)
            c = se.lin_fit([pv, Dt.H[val_rows]], Dt.Y[val_rows], Dt.W[val_rows])
            rowpred[arm] = se.lin_pred(c, [pt, Dt.H[test_rows]])
            if arm in ARMS:
                print(b0, arm, 'lam', lam, info, flush=True)
        feats = []
        for asof, rows in ((vcut_s, val_rows), (b0, test_rows)):
            o, d = se.darko_vec(Dk, Dt.pids, asof)
            feats.append([o[Dt.O[rows]].sum(1), d[Dt.D[rows]].sum(1), Dt.H[rows]])
        rowpred['darko'] = se.lin_pred(se.lin_fit(feats[0], Dt.Y[val_rows], Dt.W[val_rows]), feats[1])
        np.savez(f, **P)
        for k in ['win', 'margin', 'spm', 'darko_cal'] + ARMS + [a for a in ('spm_lpit', 'spm_lpit_h') if a in P]:
            allP[k].append(P[k])
        for k, v in rowpred.items():
            stint[k].append(v)
        stint['y'].append(Dt.Y[test_rows]); stint['yx'].append(Dt.YX[test_rows]); stint['w'].append(Dt.W[test_rows])
        stint['g'].append(Dt.GI[test_rows])
        print(b0, len(test), 'games', ' '.join(f'{k} {roc_auc_score(P["win"], P[k]):.3f}' for k in ['spm', 'darko_cal'] + ARMS if len(set(P['win'])) > 1), flush=True)
    A = {k: np.concatenate(v) for k, v in allP.items()}; w = A['win']; M = A['margin']
    ll = lambda p: -(w * np.log(np.clip(p, 1e-6, 1)) + (1 - w) * np.log(np.clip(1 - p, 1e-6, 1)))
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    rng = np.random.default_rng(0); bs = rng.integers(0, len(w), (2000, len(w)))
    res = dict(n=int(len(w)), games={}, log=log)
    for k in [a for a in A if a not in ('win', 'margin')]:
        e = dict(auc=float(roc_auc_score(w, A[k])), log_loss=float(ll(A[k]).mean()), margin_corr=float(np.corrcoef(lg(A[k]), M)[0, 1]))
        for ref in ('spm', 'spm_lpit_h', 'darko_cal'):
            if k != ref and ref in A:
                d = ll(A[k]) - ll(A[ref])
                e[f'd_ll_vs_{ref}'] = [float(x) for x in np.percentile([d[b].mean() for b in bs], [2.5, 97.5])]
                dc = [np.corrcoef(lg(A[k][b]), M[b])[0, 1] - np.corrcoef(lg(A[ref][b]), M[b])[0, 1] for b in bs[:1000]]
                e[f'd_corr_vs_{ref}'] = [float(x) for x in np.percentile(dc, [2.5, 97.5])]
        res['games'][k] = e
        print(f"{k:10s} auc {e['auc']:.3f} ll {e['log_loss']:.4f} r {e['margin_corr']:.3f} | vs spm dll {np.round(e.get('d_ll_vs_spm', []), 4)} "
              f"dr {np.round(e.get('d_corr_vs_spm', []), 3)} | vs darko dll {np.round(e.get('d_ll_vs_darko_cal', []), 4)} dr {np.round(e.get('d_corr_vs_darko_cal', []), 3)}")
    S = {k: np.concatenate(v) for k, v in stint.items()}
    games = np.unique(S['g']); gpos = {g: i for i, g in enumerate(games)}; gi = np.array([gpos[g] for g in S['g']])
    bsg = rng.integers(0, len(games), (1000, len(games)))
    res['stint'] = {}
    for tgt in ('y', 'yx'):
        y, ww = S[tgt], S['w']
        sst = np.bincount(gi, ww * (y - np.average(y, weights=ww)) ** 2, len(games))
        sse = {k: np.bincount(gi, ww * (y - S[k]) ** 2, len(games)) for k in ['spm', 'spm_lpit_h', 'darko', 'trk_prior', 'trk_prior_l']}
        res['stint'][tgt] = {}
        for k in sse:
            e = dict(r2=float(1 - sse[k].sum() / sst.sum()))
            for ref in ('spm', 'spm_lpit_h', 'darko'):
                if k != ref:
                    dd = [(sse[ref][b].sum() - sse[k][b].sum()) / sst[b].sum() for b in bsg]
                    e[f'd_r2_vs_{ref}'] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            res['stint'][tgt][k] = e
        print(f'stint R2 vs {tgt}:', ' '.join(f"{k} {v['r2']:.5f}" for k, v in res['stint'][tgt].items()),
              '| trk_prior_l - darko', np.round(res['stint'][tgt]['trk_prior_l']['d_r2_vs_darko'], 5), '- lpit_h', np.round(res['stint'][tgt]['trk_prior_l']['d_r2_vs_spm_lpit_h'], 5), '| spm - darko', np.round(res['stint'][tgt]['spm']['d_r2_vs_darko'], 5))
    json.dump(res, open(OUT / 'ms_trk.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
