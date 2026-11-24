"""Phase 1 of the DARKO chase: a box-score + minutes prior (SPM) and a luck-adjusted target for the stint ratings,
scored on the ms_bench blocks (2020-21 .. 2025-26, same games and calibration as darko_cal).

Ratings: sparse ridge on stint-side rows (offense five +1 at o, defense five +1 at d; y = points per 100 possessions,
weight = possessions x recency decay, half-life HL days), shrunk toward a prior mean m:
    min  sum w (y - b - b_h home - X theta)^2 + lam |theta - m|^2
Arms (all fit only on data before the cutoff):
  rapm          m = 0
  spm_prior     the box prior alone (theta = m, no plus-minus update)
  spm           m = SPM prediction from each player's box score (decayed per-36 rates, shrunk toward league rates,
                true shooting, minutes per game, log total minutes). SPM weights are learned at each cutoff by
                minutes-weighted ridge of the plain-RAPM o and d (same cutoff) on the box features, as BPM does.
  spm_luck      spm, with the target luck-adjusted: 3PM and FTM replaced by their expectation from each on-court
                shooter's shrunk 3P% / FT% in EARLIER seasons (unattributed attempts at league rates). Used for both
                the SPM's RAPM target and the final fit.
Per block: lambda chosen on the last 30 days before b0 (fit with cutoff = start of that window), logistic calibration
on those games, then refit at b0 and predict the block from its realised stints (as ms_latent / darko_cal).
Adds keys to results/latent/msb_*.npz; saves spm_luck ratings per block to results/latent/spm_ratings.pkl.
  python ms_spm.py [season ...]   (PC; then python ms_bench.py for the summary)
"""
import pickle, sys
from collections import defaultdict
from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import LogisticRegression, Ridge

from masked_paths import ROOT

OUT = ROOT / 'results/latent'
HL = 365.0
LAMS = [1000.0, 3000.0, 10000.0, 30000.0]
LAM_TARGET = 3000.0
K_MIN = 250.0     # pseudo-minutes shrinking box rates toward league
K3, KFT = 300.0, 100.0  # pseudo-attempts for shooter 3P% / FT%
VAL_DAYS = 30
BOX = ['pts', 'fga', 'fg3a', 'fta', 'oreb', 'dreb', 'ast', 'stl', 'blk', 'tov', 'pf']
SC = 1000.0       # intercept columns scaled so the ridge penalty on them is negligible
ARMS = ['rapm', 'spm_prior', 'spm', 'spm_luck', 'spm_lpit', 'spm_lpit_h']  # lpit: point-in-time luck (full / half)


def dnum(s):
    return date.fromisoformat(str(s)[:10]).toordinal()


class Data:
    def __init__(self):
        S, G = [], {}
        for f in ('ms2_stints.pkl', 'ms2_stints_2325.pkl', 'ms2_stints_2526.pkl'):
            d = pickle.load(open(OUT / f, 'rb')); S += d['stints']; G.update(d['games'])
        self.G = G
        pids = sorted({p for s in S for side in s['lineup'] for p in side})
        self.pids, self.pix, n = pids, {p: i for i, p in enumerate(pids)}, len(pids)
        self.n = n
        # shooter 3P% / FT% from earlier seasons only
        by = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
        for s in S:
            se = G[s['game_id']]['season']
            for p, v in s['shoot'].items():
                by[p][se] += v
        tot = np.sum([v for d in by.values() for v in d.values()], axis=0)
        self.L3, self.LFT = tot[1] / tot[0], tot[3] / tot[2]

        def rates(p, se):
            c = sum((v for s2, v in by[p].items() if s2 < se), np.zeros(4))
            return (c[1] + K3 * self.L3) / (c[0] + K3), (c[3] + KFT * self.LFT) / (c[2] + KFT)
        cache = {}
        O, D, H, Y, YL, W, GI, DAY = [], [], [], [], [], [], [], []
        SR, SP, S3, SF = [], [], [], []          # per row: on-court offense shooter attempts (row, player, 3PA, FTA)
        UNA3, UNAF, M3, MFT, EV = [], [], [], [], []  # unattributed attempts, makes; EV: (player, day, 3PA, 3PM, FTA, FTM)
        self.gids = sorted(G, key=lambda g: (G[g]['date'], g)); gix = {g: i for i, g in enumerate(self.gids)}
        for s in S:
            g = s['game_id']; se = G[g]['season']; dy = dnum(G[g]['date'])
            for k in (0, 1):
                poss = s['poss_team'][k]
                if poss <= 0:
                    continue
                off, de = s['lineup'][k], s['lineup'][1 - k]
                e3 = eft = a3 = aft = 0.0
                for p in off:
                    v = s['shoot'].get(p)
                    if v is None:
                        continue
                    SR.append(len(Y)); SP.append(self.pix[p]); S3.append(v[0]); SF.append(v[2])
                    if (p, se) not in cache:
                        cache[(p, se)] = rates(p, se)
                    r3, rft = cache[(p, se)]
                    e3 += v[0] * r3; eft += v[2] * rft; a3 += v[0]; aft += v[2]
                e3 += (s['fg3a'][k] - a3) * self.L3; eft += (s['fta'][k] - aft) * self.LFT
                pts = s['pts'][k]
                O.append([self.pix[p] for p in off]); D.append([self.pix[p] for p in de]); H.append(1 - k)
                Y.append(100 * pts / poss); YL.append(100 * (pts - 3 * (s['fg3m'][k] - e3) - (s['ftm'][k] - eft)) / poss)
                W.append(poss); GI.append(gix[g]); DAY.append(dy)
                UNA3.append(s['fg3a'][k] - a3); UNAF.append(s['fta'][k] - aft); M3.append(s['fg3m'][k]); MFT.append(s['ftm'][k])
            for p, v in s['shoot'].items():
                if p in self.pix:
                    EV.append((self.pix[p], dy, *v))
        self.O, self.D = np.array(O, np.int32), np.array(D, np.int32)
        self.H, self.Y, self.YL, self.W = np.array(H, float), np.array(Y), np.array(YL), np.array(W)
        self.GI, self.DAY = np.array(GI), np.array(DAY)
        m = len(self.Y)
        self.A3 = csr_matrix((S3, (SR, SP)), shape=(m, n)); self.AF = csr_matrix((SF, (SR, SP)), shape=(m, n))
        self.UNA3, self.UNAF, self.M3, self.MFT = map(np.array, (UNA3, UNAF, M3, MFT))
        self.EV = np.array(EV, float)
        cols = np.concatenate([self.O, self.D + n], 1)
        self.X = csr_matrix((np.ones(m * 10), (np.repeat(np.arange(m), 10), cols.ravel())), shape=(m, 2 * n))
        self.rows_of = defaultdict(list)
        for i, gi in enumerate(self.GI):
            self.rows_of[gi].append(i)
        print(f'{len(G)} games, {m} rows, {n} players; league 3P% {self.L3:.3f} FT% {self.LFT:.3f}; '
              f'luck sd {np.std(self.Y - self.YL):.1f} vs y sd {np.std(self.Y):.1f}', flush=True)
        gl = pd.read_parquet(OUT / 'gamelogs.parquet')
        gl = gl[gl.player_id.isin(self.pix) & (gl['min'] > 0)]
        self.gl_p = gl.player_id.map(self.pix).values; self.gl_day = gl.game_date.map(dnum).values
        self.gl_min = gl['min'].values.astype(float); self.gl = {c: gl[c].values.astype(float) for c in BOX}

    def luck_target(self, cut, alpha=1.0):
        """Points per 100 with alpha x (3PM, FTM luck) removed; shooter rates from all events before `cut`."""
        key = ('luck', cut, alpha)
        if not hasattr(self, '_lc'):
            self._lc = {}
        if key not in self._lc:
            self._lc = {k: v for k, v in self._lc.items() if k[1] >= cut - 40}
            e = self.EV[self.EV[:, 1] < cut]; p = e[:, 0].astype(int); n = self.n
            c = [np.bincount(p, e[:, j], n) for j in (2, 3, 4, 5)]
            r3 = (c[1] + K3 * self.L3) / (c[0] + K3); rf = (c[3] + KFT * self.LFT) / (c[2] + KFT)
            e3 = self.A3 @ r3 + self.UNA3 * self.L3; ef = self.AF @ rf + self.UNAF * self.LFT
            self._lc[key] = self.Y - alpha * 100 * (3 * (self.M3 - e3) + (self.MFT - ef)) / self.W
        return self._lc[key]

    def box_features(self, cut):
        """Decayed box features per player from games before day `cut`. Returns F (n x f), decayed minutes."""
        m = self.gl_day < cut
        w = 0.5 ** ((cut - self.gl_day[m]) / HL); p = self.gl_p[m]; n = self.n
        mins = np.bincount(p, w * self.gl_min[m], n); games = np.bincount(p, w, n)
        F = []
        for c in BOX:
            s = np.bincount(p, w * self.gl[c][m], n)
            mu = s.sum() / mins.sum()
            F.append(36 * (s + K_MIN * mu) / (mins + K_MIN))
        pts, fga, fta = (np.bincount(p, w * self.gl[c][m], n) for c in ('pts', 'fga', 'fta'))
        tsa = 2 * (fga + 0.44 * fta); lts = pts.sum() / tsa.sum()
        ts = (pts + 100 * lts) / (tsa + 100)
        F += [ts, F[0] * ts, mins / np.maximum(games, 1e-9), np.log1p(mins)]
        return np.column_stack(F), mins

    def ridge(self, rows, y, m, lam, cut):
        w = self.W[rows] * 0.5 ** ((cut - self.DAY[rows]) / HL); sw = np.sqrt(w)
        X = self.X[rows]
        r = y[rows] - X @ m
        A = csr_matrix(X.multiply(sw[:, None]))
        C = np.column_stack([np.ones(len(rows)), self.H[rows]]) * SC * sw[:, None]
        from scipy.sparse import hstack
        sol = lsqr(hstack([A, csr_matrix(C)]).tocsr(), sw * r, damp=np.sqrt(lam), atol=1e-7, btol=1e-7, iter_lim=2000)[0]
        return m + sol[:2 * self.n], sol[2 * self.n:] * SC

    def spm_prior(self, cut, y):
        key = (cut, id(y))
        if not hasattr(self, '_pc'):
            self._pc = {}
        if key not in self._pc:
            self._pc = {k: v for k, v in self._pc.items() if k[0] >= cut - 40}  # keep this block's cutoffs only
            self._pc[key] = self._spm_prior(cut, y)
        return self._pc[key]

    def _spm_prior(self, cut, y):
        rows = np.flatnonzero(self.DAY < cut)
        th, _ = self.ridge(rows, y, np.zeros(2 * self.n), LAM_TARGET, cut)
        F, mins = self.box_features(cut)
        ok = mins > 50
        mu, sd = F[ok].mean(0), F[ok].std(0) + 1e-9
        Z = (F - mu) / sd; sw = np.minimum(mins[ok], 3000)
        n = self.n; m = np.zeros(2 * n)
        for blk, sign in ((0, 1), (1, -1)):  # defence target: -d (higher = better)
            t = sign * th[blk * n:(blk + 1) * n]
            reg = Ridge(alpha=10.0).fit(Z[ok], t[ok], sample_weight=sw)
            pred = reg.predict(Z)
            low = ok & (mins < 300)
            pred[~ok] = np.average(pred[low], weights=np.maximum(mins[low], 1)) if low.any() else 0.0
            m[blk * n:(blk + 1) * n] = sign * pred
        return m

    def margins(self, theta, b, gis):
        out = np.zeros(len(gis))
        for k, gi in enumerate(gis):
            r = self.rows_of[gi]
            yh = b[0] + b[1] * self.H[r] + theta[self.O[r]].sum(1) + theta[self.n + self.D[r]].sum(1)
            out[k] = np.sum(np.where(self.H[r] == 1, 1, -1) * self.W[r] * yh / 100)
        return out


def fit_arm(Dt, arm, cut, lam):
    y = (Dt.YL if arm == 'spm_luck' else Dt.luck_target(cut, 1.0) if arm == 'spm_lpit'
         else Dt.luck_target(cut, 0.5) if arm == 'spm_lpit_h' else Dt.Y)
    rows = np.flatnonzero(Dt.DAY < cut)
    m = np.zeros(2 * Dt.n) if arm == 'rapm' else Dt.spm_prior(cut, y)
    if arm == 'spm_prior':
        _, b = Dt.ridge(rows, y, m, 1e9, cut)  # intercepts only
        return m, b
    return Dt.ridge(rows, y, m, lam, cut)


def main(seasons):
    Dt = Data(); G = Dt.G; gix = {g: i for i, g in enumerate(Dt.gids)}
    win = lambda gs: np.array([int(G[g]['final'][0] > G[g]['final'][1]) for g in gs])
    rf = OUT / 'spm_ratings.pkl'
    saved = pickle.load(open(rf, 'rb')) if rf.exists() else {}
    for f in sorted(OUT.glob('msb_*.npz')):
        season, b0 = int(f.stem.split('_')[1]), f.stem.split('_')[2]
        if seasons and season not in seasons:
            continue
        P = dict(np.load(f))
        todo = [a for a in ARMS if a not in P]
        if not todo:
            continue
        test = list(P['gid']); vcut_s = str(date.fromisoformat(b0) - timedelta(days=VAL_DAYS))
        val = [g for g in Dt.gids if vcut_s <= G[g]['date'] < b0]
        cut, vcut = dnum(b0), dnum(vcut_s)
        for arm in todo:
            best = None
            for lam in (LAMS if arm not in ('spm_prior',) else [None]):
                th, b = fit_arm(Dt, arm, vcut, lam)
                x = Dt.margins(th, b, [gix[g] for g in val])
                cal = LogisticRegression(C=1e4).fit(x[:, None], win(val))
                l = -np.mean(np.log(np.clip(cal.predict_proba(x[:, None])[np.arange(len(val)), win(val)], 1e-9, 1)))
                if best is None or l < best[0]:
                    best = (l, lam, cal)
            _, lam, cal = best
            th, b = fit_arm(Dt, arm, cut, lam)
            P[arm] = cal.predict_proba(Dt.margins(th, b, [gix[g] for g in test])[:, None])[:, 1]
            P[arm + '_lam'] = np.array(lam if lam else 0.0)
            saved.setdefault(b0, {})[arm] = dict(pids=Dt.pids, theta=th, b=b, lam=lam)
        np.savez(f, **P); pickle.dump(saved, open(rf, 'wb'))
        from sklearn.metrics import roc_auc_score
        print(season, b0, ' '.join(f'{a} {roc_auc_score(P["win"], P[a]):.3f}' for a in ARMS + ['darko_cal'] if len(set(P['win'])) > 1), flush=True)


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]])
