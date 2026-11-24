"""Beyond additive ratings, and beyond realised minutes. Run after ms_kalman.py and ms_spm.py.

1. Lineup fit. The Kalman pass saved every row's one-step-ahead prediction (made before that game's update), so its
   residuals are honest out-of-sample misses of the additive model. Lineup-composition features built from each
   player's point-in-time box profile (latest grid cut before the row) describe what sums of player ratings cannot:
   usage concentration (max, top two, squared total), spacing (count of volume three-point shooters, weakest
   shooter), offensive rebounding against defensive rebounding, spacing against rim protection, size gap, passing x
   spacing. A ridge and a shallow boosted model learn residual ~ features on rows before the cutoff; the block is
   predicted as kal_sq snapshot + fit. Arms: kal_sq_fitr (ridge), kal_sq_fitg (boosting).
2. Pregame. Every other arm (ours and DARKO) is scored with the game's realised stints. Here both only know who was
   available (players who played) and project each one's minutes as the mean of his previous 10 games (12 if none);
   team strength = projected-minute-share-weighted ratings. Arms: kal_sq_pre, spm_lpit_h_pre, darko_pre.
3. Stint-level R2 (as ms_stint_eval) for spm_lpit_h, spm_bio, spm_sq, kal, kal_bio, kal_sq, the fit arms, and DARKO.
Game arms are written into results/latent/msb_*.npz (ms_bench.py summary picks them up); stint R2 and pregame
paired CIs go to results/latent/ms_fit.json.
  python ms_fit.py      (PC)
"""
import json, pickle
from collections import defaultdict
from datetime import date, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV

import ms_spm as sp
import ms_kalman as mk
from ms_stint_eval import load_darko_od, darko_vec, lin_fit, lin_pred
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
RIDGE_ARMS = ['spm_lpit_h', 'spm_bio', 'spm_sq']
KAL_ARMS = ['kal', 'kal_bio', 'kal_sq']
TRAIN_DAYS = 730


def lineup_features(Dt, cuts):
    """Per row composition features from point-in-time per-36 box profiles (+ height)."""
    ci = np.searchsorted(cuts, Dt.DAY, side='right') - 1
    ci = np.maximum(ci, 0)
    Fx = np.zeros((len(Dt.Y), 17))
    for k, c in enumerate(cuts):
        rows = np.flatnonzero(ci == k)
        if not len(rows):
            continue
        F, mins = Dt.box_features(c)
        hgt = Dt.bio_features(c, mins)[:, 1]
        use = F[:, 1] + 0.44 * F[:, 3] + F[:, 9]
        fg3a, ast, oreb, dreb, blk, stl = F[:, 2], F[:, 6], F[:, 4], F[:, 5], F[:, 8], F[:, 7]
        O, D = Dt.O[rows], Dt.D[rows]
        u = np.sort(use[O], 1)
        sh = fg3a[O]
        Fx[rows] = np.column_stack([
            u[:, -1], u[:, -1] + u[:, -2], u.sum(1) ** 2,                        # usage concentration
            (sh >= 5).sum(1), ((sh >= 5).sum(1) >= 4), sh.min(1),                # spacing
            oreb[O].sum(1) * dreb[D].sum(1), sh.sum(1) * blk[D].sum(1),          # boards; spacing vs rim protection
            (hgt[O].mean(1) - hgt[D].mean(1)) ** 2, ast[O].max(1) * sh.sum(1),    # size gap; passing x spacing
            blk[D].max(1), (blk[D] >= 1.5).sum(1), stl[D].sum(1) * u.sum(1),
            u.sum(1), sh.sum(1), oreb[O].sum(1), hgt[O].mean(1) - hgt[D].mean(1)])  # raw sums (boosting only)
    return Fx


def snap_pred(Dt, theta, b, rows):
    return b[0] + b[1] * Dt.H[rows] + theta[Dt.O[rows]].sum(1) + theta[Dt.n + Dt.D[rows]].sum(1)


def game_margins(Dt, yhat_rows, rows):
    out = defaultdict(float)
    for r, v in zip(rows, yhat_rows):
        out[Dt.GI[r]] += (1 if Dt.H[r] == 1 else -1) * Dt.W[r] * v / 100
    return out


def calibrated(xv, wv, xt):
    cal = LogisticRegression(C=1e4).fit(np.asarray(xv)[:, None], wv)
    return cal.predict_proba(np.asarray(xt)[:, None])[:, 1]


class Pregame:
    """Availability-only team strength: projected minutes from each player's previous 10 games."""

    def __init__(self, Dt):
        gl = pd.read_parquet(OUT / 'gamelogs.parquet')
        gl = gl[gl['min'] > 0].sort_values('game_date')
        gl['proj'] = gl.groupby('player_id')['min'].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        gl['proj'] = gl['proj'].fillna(12.0)
        self.by_game = {g: d for g, d in gl.groupby('game_id')}
        self.Dt = Dt

    def teams(self, gid):
        """[(home [(pid, share)], away [...])] or None. Home = the team whose matchup has 'vs.'."""
        d = self.by_game.get(gid)
        if d is None:
            return None
        out = []
        for home in (True, False):
            t = d[d.matchup.str.contains('vs.') == home]
            tot = t.proj.sum()
            if tot <= 0:
                return None
            out.append([(int(p), 5 * m / tot) for p, m in zip(t.player_id, t.proj)])
        return out

    def ours(self, theta, b, gid):
        T = self.teams(gid)
        if T is None:
            return None
        n, pix = self.Dt.n, self.Dt.pix
        val = lambda side, k: sum(s * theta[k * n + pix[p]] for p, s in side if p in pix)
        h, a = T
        return (b[1] + val(h, 0) + val(a, 1)) - (val(a, 0) + val(h, 1))

    def darko(self, Dk, asof, gid):
        T = self.teams(gid)
        if T is None:
            return None
        def net(side):
            s = 0.0
            for p, w in side:
                if p in Dk:
                    j = np.searchsorted(Dk[p][0], asof) - 1
                    o, d = Dk[p][1][j] if j >= 0 else (-1.0, -1.0)
                else:
                    o, d = -1.0, -1.0
                s += w * (o + d)
            return s
        return net(T[0]) - net(T[1])


def main():
    Dt = sp.Data(); G = Dt.G; gix = {g: i for i, g in enumerate(Dt.gids)}
    win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
    B = mk.blocks(); cuts = np.array(mk.grid_cuts(Dt, B))
    snaps = pickle.load(open(OUT / 'kal_snap.pkl', 'rb'))
    Fx = lineup_features(Dt, cuts)
    Dk = load_darko_od(); PG = Pregame(Dt)
    acc = defaultdict(list); pre = defaultdict(list)
    for f, season, b0, vcut in B:
        P = dict(np.load(f)); test = list(P['gid'])
        if 'kal_sq_hp' not in P:
            continue
        val = [g for g in Dt.gids if vcut <= G[g]['date'] < b0]
        cut, vc = sp.dnum(b0), sp.dnum(vcut)
        test_rows = np.concatenate([Dt.rows_of[gix[g]] for g in test if gix[g] in Dt.rows_of]).astype(int)
        val_rows = np.flatnonzero((Dt.DAY >= vc) & (Dt.DAY < cut))
        hp = tuple(P['kal_sq_hp'])
        one = np.load(OUT / f'kal_onestep_{hp[0]}_{int(hp[1])}.npy')
        preds = {}
        # --- base predictions at the validation start and the block start
        base = {}
        for arm in KAL_ARMS:
            h = tuple(P[arm + '_hp'])
            base[arm] = [snap_pred(Dt, *snaps[arm][h][c], rows) for c, rows in ((vc, val_rows), (cut, test_rows))]
        ridge_fit = {}
        for arm in RIDGE_ARMS:
            lam = float(P[arm + '_lam'])
            for c in (vc, cut):
                ridge_fit[(arm, c)] = sp.fit_arm(Dt, arm, c, lam)
            base[arm] = [snap_pred(Dt, *ridge_fit[(arm, c)], rows) for c, rows in ((vc, val_rows), (cut, test_rows))]
        # --- lineup fit on one-step residuals before each cutoff
        for kind in ('r', 'g'):
            fits = []
            for c, rows in ((vc, val_rows), (cut, test_rows)):
                tr = np.flatnonzero((Dt.DAY < c) & (Dt.DAY >= c - TRAIN_DAYS) & ~np.isnan(one))
                res = Dt.shotq_target(c, 0.5)[tr] - one[tr]
                if kind == 'r':
                    X = Fx[:, :13]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                    mdl = RidgeCV(alphas=[1e2, 1e3, 1e4, 1e5]).fit((X[tr] - mu) / sd, res, sample_weight=Dt.W[tr])
                    fits.append(mdl.predict((X[rows] - mu) / sd))
                else:
                    mdl = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                                        min_samples_leaf=3000, l2_regularization=10.0)
                    mdl.fit(Fx[tr], res, sample_weight=Dt.W[tr])
                    fits.append(mdl.predict(Fx[rows]))
            base[f'kal_sq_fit{kind}'] = [base['kal_sq'][0] + fits[0], base['kal_sq'][1] + fits[1]]
        # --- game probabilities for the fit arms (same calibration protocol)
        for arm in ('kal_sq_fitr', 'kal_sq_fitg'):
            mv = game_margins(Dt, base[arm][0], val_rows); mt = game_margins(Dt, base[arm][1], test_rows)
            P[arm] = calibrated([mv[gix[g]] for g in val], win(val), [mt.get(gix[g], 0.0) for g in test])
        np.savez(f, **P)
        # --- pregame arms
        sets = {}
        for c_day, gs in ((vcut, val), (b0, test)):
            th, b = snaps['kal_sq'][hp][sp.dnum(c_day)]
            th2, b2 = ridge_fit[('spm_lpit_h', sp.dnum(c_day))]
            rows = []
            for g in gs:
                xo, xs, xd = PG.ours(th, b, g), PG.ours(th2, b2, g), PG.darko(Dk, c_day, g)
                if None not in (xo, xs, xd):
                    rows.append((g, xo, xs, xd))
            sets[c_day] = rows
        V, T = sets[vcut], sets[b0]
        if V and T:
            wv, wt = win([r[0] for r in V]), win([r[0] for r in T])
            for k, name in ((1, 'kal_sq_pre'), (2, 'spm_lpit_h_pre'), (3, 'darko_pre')):
                pre[name].append(calibrated([r[k] for r in V], wv, [r[k] for r in T]))
            pre['win'].append(wt); pre['gid'].append(np.array([gix[r[0]] for r in T]))
        # --- stint-level R2 accumulators (linear map fit on the validation window, as ms_stint_eval)
        for arm, (pv, pt) in base.items():
            c = lin_fit([pv, Dt.H[val_rows]], Dt.Y[val_rows], Dt.W[val_rows])
            preds[arm] = lin_pred(c, [pt, Dt.H[test_rows]])
        feats = []
        for asof, rows in ((vcut, val_rows), (b0, test_rows)):
            o, d = darko_vec(Dk, Dt.pids, asof)
            feats.append([o[Dt.O[rows]].sum(1), d[Dt.D[rows]].sum(1), Dt.H[rows]])
        preds['darko'] = lin_pred(lin_fit(feats[0], Dt.Y[val_rows], Dt.W[val_rows]), feats[1])
        acc['season'].append(np.full(len(test_rows), season)); acc['g'].append(Dt.GI[test_rows])
        acc['y'].append(Dt.Y[test_rows]); acc['w'].append(Dt.W[test_rows])
        for k, v in preds.items():
            acc[k].append(v)
        print(season, b0, len(test), 'games', flush=True)
    A = {k: np.concatenate(v) for k, v in acc.items()}
    rng = np.random.default_rng(0); res = {}
    games = np.unique(A['g']); gpos = {g: i for i, g in enumerate(games)}; gi = np.array([gpos[g] for g in A['g']])
    names = [k for k in A if k not in ('season', 'g', 'y', 'w')]
    y, w = A['y'], A['w']
    sse = {k: np.bincount(gi, w * (y - A[k]) ** 2, len(games)) for k in names}
    sst = np.bincount(gi, w * (y - np.average(y, weights=w)) ** 2, len(games))
    bs = rng.integers(0, len(games), (1000, len(games)))
    res['stint_r2'] = {}
    for k in names:
        e = dict(r2=float(1 - sse[k].sum() / sst.sum()))
        for ref in ('darko', 'kal_sq', 'spm_lpit_h'):
            if k != ref:
                dd = [(sse[ref][b].sum() - sse[k][b].sum()) / sst[b].sum() for b in bs]
                e[f'd_vs_{ref}'] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
        e['by_season'] = {str(s): float(1 - sse[k][np.unique(gi[A['season'] == s])].sum() / sst[np.unique(gi[A['season'] == s])].sum())
                          for s in sorted(set(A['season']))}
        res['stint_r2'][k] = e
        print(f"stint R2 {k:14s} {e['r2']:.5f} vs darko {np.round(e.get('d_vs_darko', []), 5)} vs kal_sq {np.round(e.get('d_vs_kal_sq', []), 5)}")
    if pre:
        Q = {k: np.concatenate(v) for k, v in pre.items()}
        wq = Q['win']; ll = lambda p: -(wq * np.log(p) + (1 - wq) * np.log(1 - p))
        res['pregame'] = {'n': int(len(wq))}
        L = {k: ll(np.clip(Q[k], 1e-9, 1 - 1e-9)) for k in ('kal_sq_pre', 'spm_lpit_h_pre', 'darko_pre')}
        bq = rng.integers(0, len(wq), (2000, len(wq)))
        for k, l in L.items():
            e = dict(ll=float(l.mean()))
            if k != 'darko_pre':
                d = [(l[b] - L['darko_pre'][b]).mean() for b in bq]
                e['d_ll_vs_darko'] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            res['pregame'][k] = e
        print('pregame', res['pregame'])
    json.dump(res, open(OUT / 'ms_fit.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
