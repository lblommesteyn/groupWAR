"""Every angle for getting tracking into DUNCAN, on one harness, without forcing it.

DEVELOPMENT set: tracking-era blocks (CUTS; 21-day test windows), every fit point-in-time (only tracking, box and
stints from before the cutoff). Many arms are tried here, so a "win" on this set only makes an arm a CANDIDATE:
  candidate = stint-R2 gain over DUNCAN-2 with a game-resampled 95% CI above zero AND game log loss not worse
  (CI of the log-loss difference not entirely above zero).
Candidates are frozen and judged on games after the development set (shadow, no retuning). Pre-declared here, before
any result.

Arms (all on top of DUNCAN-2: lambda 3000, half-strength 3P/FT luck target, SPM box prior, movers pulled 0.5):
  base        DUNCAN-2
  darko       DARKO point-in-time o/d DPM (linear map fit on the 30 days before the cutoff, as for every arm)
  A2_a        tracking shot-quality target, 2-pointers: FG luck vs tracking xFG (location + defenders + touch time,
              shots_xfg.jsonl 'xb') + the shooter's shrunk pre-cutoff residual, removed at strength a
  Aall_.5     the same for every field goal (threes: expectation = tracking xFG instead of the shooter's 3P%)
  Boff_g      involvement-weighted offense: in every stint, offense credit w_j = 1 + g (load_j / lineup mean - 1),
              load = shrunk pre-cutoff on-ball share (BORIS possession features); same weights in prediction
  Bdef_g      the same for defense, load = share of frames as the handler's nearest defender + paint share
  C           role-aware box prior: the SPM regression also gets box x BORIS-mix interactions (offense mix for
              the offense prior, defense mix for the defense prior; players without a mix: interactions 0)
  D           similarity prior for thin samples: ridge from BORIS style (off + def mix) to established players'
              (rating - prior); added to the prior of players with < 1000 decayed minutes, weight exp(-min / 500)
  H           tracking-informed mover pull (ms_movers_trk rule, b 0.15 / 0.15) instead of the flat 0.5
Output: results/trk/bench.json     python trk_bench.py   (PC)
"""
import json, pickle, sys, time
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr, LinearOperator
from sklearn.linear_model import LogisticRegression, Ridge

import ms_spm as sp
import ms_stint_eval as se
from ms_fix import FData
from masked_paths import ROOT

OUT = ROOT / 'results/trk'
LAM = 3000.0
CUTS = ['2026-02-01', '2026-02-24', '2026-03-19', '2026-04-11', '2026-05-04', '2026-11-06']
WIN = 21
POSS = {2026: OUT / 'poss_2026.npz', 2027: OUT / 'poss_2027.npz'}
XFG = [OUT / 'shots_xfg_2026.jsonl', OUT / 'shots_xfg_2027.jsonl']


# ----------------------------------------------------------------------------------------------------- tracking data
class Trk:
    def __init__(self, Dt):
        self.Dt = Dt
        P = [np.load(POSS[s], allow_pickle=True) for s in (2026, 2027)]
        self.off_f = list(P[0]['off_f']); self.def_f = list(P[0]['def_f'])
        cat = lambda k: np.concatenate([p[k] for p in P])
        self.day = np.array([sp.dnum(d) for d in cat('date')]); self.off = cat('off'); self.deff = cat('deff')
        self.fo = cat('fo'); self.fd = cat('fd')
        self.mixes = json.load(open(ROOT / 'results/archetypes/mix.json'))['players']
        self._match_shots()

    def loads(self, cut, K=200.0):
        """per player index: (offense on-ball load, defense load), shrunk to league, possessions before cut."""
        Dt = self.Dt; n = Dt.n; m = self.day < cut
        out = []
        oh, pa = self.def_f.index('on_handler'), self.def_f.index('paint')
        for ids, F, val in ((self.off, self.fo, lambda f: f[:, :, self.off_f.index('onball')]),
                            (self.deff, self.fd, lambda f: f[:, :, oh] + f[:, :, pa]),
                            (self.deff, self.fd, lambda f: f[:, :, oh]),
                            (self.deff, self.fd, lambda f: f[:, :, pa])):
            v = val(F[m]).ravel().astype(float); p = ids[m].ravel()
            ix = np.array([Dt.pix.get(int(q), -1) for q in p]); ok = ix >= 0
            s = np.bincount(ix[ok], v[ok], n); c = np.bincount(ix[ok], minlength=n).astype(float)
            mu = v[ok].mean()
            out.append((s + K * mu) / (c + K) / mu)       # relative load, 1 = league average
        return out

    def mix(self, cut, side):
        mo = sp.date.fromordinal(int(cut)).isoformat()[:7]
        k = 6 if side == 'off' else 5
        M = np.full((self.Dt.n, k), np.nan)
        for pid, v in self.mixes[side].items():
            ms = [np.array(x) for mth, x in v.get('by_month', {}).items() if mth < mo]
            if ms and int(pid) in self.Dt.pix:
                M[self.Dt.pix[int(pid)]] = np.mean(ms, 0)
        return M

    def _match_shots(self):
        """Tracking xFG per field goal in Dt.SHOT, matched in order within (game, shooter); -1 if unmatched."""
        Dt = self.Dt; S = Dt.SHOT
        X = defaultdict(list)
        for f in XFG:
            for l in open(f):
                r = json.loads(l)
                if r.get('xb') is not None:
                    X[(r['game_id'], int(r['shooter']))].append((int(r['made']), int(r['three']), float(r.get('dist') or 0), float(r['xb'])))
        gid_row = np.array(Dt.gids)[Dt.GI[S[:, 0]]]
        idx = defaultdict(list)
        for i, (g, p) in enumerate(zip(gid_row, S[:, 1])):
            if p >= 0 and (g, int(Dt.pids[p])) in X:
                idx[(g, int(Dt.pids[p]))].append(i)
        xb = np.full(len(S), -1.0); ok = tried = 0
        for key, ii in idx.items():   # in-order alignment: same make, same 2/3, distance within 3 ft, look ahead <= 4
            xs = X[key]; j = 0; tried += len(xs)
            for mk_, t3, dd, xv in xs:
                for jj in range(j, min(j + 5, len(ii))):
                    i = ii[jj]
                    if int(S[i, 4]) == mk_ and int(S[i, 3]) == t3 and abs(S[i, 2] - dd) <= 3:
                        xb[i] = xv; ok += 1; j = jj + 1; break
        bad = tried - ok
        self.xb = xb
        print(f'tracking xFG matched to {ok} field goals ({bad} in mismatched game-shooter groups skipped)', flush=True)

    def shot_target(self, cut, y_base, alpha, threes):
        """y_base with alpha x tracking shot-quality luck removed on matched FGs (2s; and 3s if threes)."""
        Dt = self.Dt; S = Dt.SHOT; n = Dt.n
        i3 = S[:, 3]; mk = S[:, 4].astype(float); pl = S[:, 1]
        m = (self.xb >= 0) & (pl >= 0)
        past = m & (Dt.SHOT_DAY < cut)
        sk = np.zeros(len(S))
        for t, K in ((0, 400.0), (1, 300.0)):
            sel = past & (i3 == t)
            res = np.bincount(pl[sel], (mk - self.xb)[sel], n); cnt = np.bincount(pl[sel], minlength=n)
            s_ = res / (cnt + K); mm = m & (i3 == t); sk[mm] = s_[pl[mm]]
        ex = np.clip(self.xb + sk, 0.01, 0.99)
        y = y_base.copy()
        two = m & (i3 == 0)
        y -= alpha * 100 * np.bincount(S[two, 0], 2 * (mk - ex)[two], len(y)) / Dt.W
        if threes:   # replace the shooter-3P% expectation by the tracking one: y += a * 100 * 3 * (ex - p3_shooter) / W
            e = Dt.EV[Dt.EV[:, 1] < cut]; p = e[:, 0].astype(int)
            c0, c1 = np.bincount(p, e[:, 2], n), np.bincount(p, e[:, 3], n)
            r3 = (c1 + sp.K3 * Dt.L3) / (c0 + sp.K3)
            th3 = m & (i3 == 1)
            y += alpha * 100 * np.bincount(S[th3, 0], 3 * (ex[th3] - r3[pl[th3]]), len(y)) / Dt.W
        return y


# ------------------------------------------------------------------------------------------------------ fitting
def design(Dt, wO=None, wD=None):
    n = Dt.n; mrows = len(Dt.Y)
    vo = np.ones((mrows, 5)) if wO is None else wO; vd = np.ones((mrows, 5)) if wD is None else wD
    cols = np.concatenate([Dt.O, Dt.D + n], 1); vals = np.concatenate([vo, vd], 1)
    return csr_matrix((vals.ravel(), (np.repeat(np.arange(mrows), 10), cols.ravel())), shape=(mrows, 2 * n))


def ridge(Dt, X, rows, y, m, cut):
    w = Dt.W[rows] * 0.5 ** ((cut - Dt.DAY[rows]) / sp.HL); sw = np.sqrt(w)
    Xr = X[rows]; r = y[rows] - Xr @ m
    A = csr_matrix(Xr.multiply(sw[:, None])); C = np.column_stack([np.ones(len(rows)), Dt.H[rows]]) * sp.SC * sw[:, None]
    n2 = X.shape[1]
    op = LinearOperator((len(rows), n2 + 2), matvec=lambda v: A @ v[:n2] + C @ v[n2:],
                        rmatvec=lambda u: np.concatenate([A.T @ u, C.T @ u]), dtype=float)
    sol = lsqr(op, sw * r, damp=np.sqrt(LAM), atol=1e-8, btol=1e-8, iter_lim=4000)[0]
    return m + sol[:n2], sol[n2:] * sp.SC


def spm_prior_roles(Dt, cut, y, Mo, Md, X):
    """SPM prior as ms_spm._spm_prior, plus box x role-mix interactions (C)."""
    rows = np.flatnonzero(Dt.DAY < cut)
    th, _ = ridge(Dt, X, rows, y, np.zeros(2 * Dt.n), cut)
    F, mins = Dt.box_features(cut); ok = mins > 50
    mu, sd = F[ok].mean(0), F[ok].std(0) + 1e-9; Z = (F - mu) / sd; sw = np.minimum(mins[ok], 3000)
    n = Dt.n; m = np.zeros(2 * n)
    for blk, sign, M in ((0, 1, Mo), (1, -1, Md)):
        t = sign * th[blk * n:(blk + 1) * n]
        mm = np.where(np.isnan(M), np.nanmean(M, 0), M); mm = mm - np.nanmean(M, 0)
        Zi = np.concatenate([Z] + [Z * mm[:, [k]] for k in range(mm.shape[1])], 1)
        reg = Ridge(alpha=10.0).fit(Z[ok], t[ok], sample_weight=sw)
        res = t - reg.predict(Z)
        reg2 = Ridge(alpha=300.0).fit(Zi[ok][:, Z.shape[1]:], res[ok], sample_weight=sw)
        pred = reg.predict(Z) + reg2.predict(Zi[:, Z.shape[1]:])
        low = ok & (mins < 300)
        pred[~ok] = np.average(pred[low], weights=np.maximum(mins[low], 1)) if low.any() else 0.0
        m[blk * n:(blk + 1) * n] = sign * pred
    return m


# ------------------------------------------------------------------------------------------------------ main
def main():
    Dt = FData(); n = Dt.n; T = Trk(Dt); Dk = se.load_darko_od()
    G = Dt.G; gids = np.array(Dt.gids)
    # orientation check: realised margin from stint rows vs final score
    gm = np.bincount(Dt.GI, np.where(Dt.H == 1, 1, -1) * Dt.Y * Dt.W / 100, len(gids))
    fin = np.array([(G[g]['final'][0] - G[g]['final'][1]) if G[g].get('final') else np.nan for g in gids])
    okf = np.isfinite(fin) & (fin != 0)
    agree = np.mean(np.sign(gm[okf]) == np.sign(fin[okf])); FS = 1 if agree > 0.5 else -1
    print(f'final-score orientation agreement {agree:.3f} -> sign {FS}', flush=True)
    home_win = (FS * fin > 0).astype(float)

    import ms_movers_trk as mvt
    recs = defaultdict(lambda: defaultdict(list)); per_block = {}
    X0 = design(Dt)
    for cut_s in CUTS:
        t0 = time.time(); cut = sp.dnum(cut_s)
        te = np.flatnonzero((Dt.DAY >= cut) & (Dt.DAY < cut + WIN))
        if len(te) < 1000:
            print('skip', cut_s); continue
        val = np.flatnonzero((Dt.DAY >= cut - sp.VAL_DAYS) & (Dt.DAY < cut))
        y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
        # movers (DUNCAN-2): last-120-day team != team in the test window
        rec = np.flatnonzero((Dt.DAY < cut) & (Dt.DAY >= cut - 120)); cnt = defaultdict(float)
        for i in rec:
            for p in Dt.O[i]:
                cnt[(p, Dt.off_team[i])] += Dt.W[i]
        last = {}
        for (p, t), w_ in cnt.items():
            if w_ > last.get(p, (0, 0))[1]:
                last[p] = (t, w_)
        newt = {}
        for i in te:
            for p in Dt.O[i]:
                if p in last and last[p][0] != Dt.off_team[i]:
                    newt[p] = Dt.off_team[i]
        mv = np.zeros(n, bool); mv[list(newt)] = True
        pull = lambda th, mm, k=None: mm + (np.where(np.r_[mv, mv], 0.5, 1.0) if k is None else k) * (th - mm)

        arms = {}
        th, b = ridge(Dt, X0, np.flatnonzero(Dt.DAY < cut), y, m, cut); arms['base'] = (X0, pull(th, m), b)
        trn = np.flatnonzero(Dt.DAY < cut)
        # A: tracking shot-quality target
        for a_ in ():
            yA = T.shot_target(cut, y, a_, threes=False); th, b = ridge(Dt, X0, trn, yA, m, cut)
            arms[f'A2_{a_}'] = (X0, pull(th, m), b)
        # B: involvement-weighted design
        lo, ld, ldh, ldp = T.loads(cut)
        F_, mins_ = Dt.box_features(cut)
        bp = (F_[:, sp.BOX.index('blk')] + F_[:, sp.BOX.index('dreb')]); bp = bp / np.average(bp, weights=np.maximum(mins_, 1))
        for side, L, gs in (('def', ld, (0.6, 1.2, 1.8)), ('defP', ldp, (0.6, 0.9, 1.2, 1.8)), ('defBOX', bp, (0.6, 1.2)), ('defBOXP', (bp + ldp) / 2, (0.6, 1.2))):
            for g_ in gs:
                U = (Dt.O if side == "off" else Dt.D); Lu = L[U]; W_ = np.maximum(1 + g_ * (Lu / Lu.mean(1, keepdims=True) - 1), 0.05)
                X = design(Dt, wO=W_ if side == 'off' else None, wD=None if side == 'off' else W_)
                th, b = ridge(Dt, X, trn, y, m, cut); arms[f'B{side}_{g_}'] = (X, pull(th, m), b)
        # C: role-aware prior
        Mo, Md = T.mix(cut, 'off'), T.mix(cut, 'def')
        mC = spm_prior_roles(Dt, cut, y, Mo, Md, X0); th, b = ridge(Dt, X0, trn, y, mC, cut); arms['C'] = (X0, pull(th, mC), b)
        # D: similarity prior for thin samples
        _, mins = Dt.box_features(cut)
        S_ = np.concatenate([Mo, Md], 1); has = ~np.isnan(S_).any(1)
        thb = arms['base'][1]; est = has & (mins > 2000)
        mD = m.copy()
        if est.sum() > 50:
            for blk in (0, 1):
                r_ = Ridge(alpha=1.0).fit(S_[est], (thb - m)[blk * n:(blk + 1) * n][est], sample_weight=np.minimum(mins[est], 3000))
                thin = has & (mins < 1000)
                mD[blk * n:(blk + 1) * n][thin] += np.exp(-mins[thin] / 500) * r_.predict(S_[thin])
        th, b = ridge(Dt, X0, trn, y, mD, cut); arms['D'] = (X0, pull(th, mD), b)
        # H: tracking-informed mover pull
        th, b = ridge(Dt, X0, trn, y, m, cut)
        k = np.ones(2 * n)
        S_sch = mvt.team_schemes(cut_s); Mx = mvt.pre_mix(T.mixes['off'], cut_s)
        roster = defaultdict(list)
        for (p, t), w_ in cnt.items():
            roster[t].append((p, w_))
        def occ(t, ex):
            v = [(Mx[int(Dt.pids[p])], w_) for p, w_ in roster.get(t, []) if p != ex and int(Dt.pids[p]) in Mx]
            if not v:
                return None
            W2 = np.array([w_ for _, w_ in v]); return (np.array([q for q, _ in v]) * W2[:, None]).sum(0) / W2.sum()
        ds, dr = {}, {}
        for p, tn in newt.items():
            to = last[p][0]
            if to in S_sch and tn in S_sch:
                ds[p] = abs(S_sch[tn] - S_sch[to])
            pm = Mx.get(int(Dt.pids[p])); on, oo = occ(tn, p), occ(to, p)
            if pm is not None and on is not None and oo is not None:
                dr[p] = abs(float(pm @ on) - float(pm @ oo))
        def zs(d):
            if len(d) < 3:
                return {}
            v = np.array(list(d.values())); return {p: (x - v.mean()) / (v.std() + 1e-9) for p, x in d.items()}
        zr, zd = zs(dr), zs(ds)
        for p in newt:
            k[p] = np.clip(0.5 - 0.15 * zr.get(p, 0.0), 0.05, 0.95); k[n + p] = np.clip(0.5 - 0.15 * zd.get(p, 0.0), 0.05, 0.95)
        arms['H'] = (X0, pull(th, m, k), b)
        Lu = ldp[Dt.D]; Xbd = design(Dt, wD=np.maximum(1 + 1.2 * (Lu / Lu.mean(1, keepdims=True) - 1), 0.05))
        th, b = ridge(Dt, Xbd, trn, y, m, cut); arms['BdefP_1.2+H'] = (Xbd, pull(th, m, k), b)

        # score: linear map on the validation window (as ms_shrink_d2), stint SE; game margins -> logistic on val games
        Yt, Wt = Dt.Y[te], Dt.W[te]; ybar = np.average(Dt.Y[val], weights=Dt.W[val])
        gte, gva = Dt.GI[te], Dt.GI[val]
        def game_ll(pv_val, pv_te):
            sg_v = np.where(Dt.H[val] == 1, 1, -1); sg_t = np.where(Dt.H[te] == 1, 1, -1)
            mv_ = np.bincount(gva, sg_v * Dt.W[val] * pv_val / 100, len(gids)); mt_ = np.bincount(gte, sg_t * Dt.W[te] * pv_te / 100, len(gids))
            gv = np.unique(gva); gv = gv[np.isfinite(fin[gv]) & (fin[gv] != 0)]
            gt = np.unique(gte); gt = gt[np.isfinite(fin[gt]) & (fin[gt] != 0)]
            lr = LogisticRegression(C=1e4).fit(mv_[gv, None], home_win[gv])
            p = np.clip(lr.predict_proba(mt_[gt, None])[:, 1], 1e-6, 1 - 1e-6)
            return gt, -(home_win[gt] * np.log(p) + (1 - home_win[gt]) * np.log(1 - p))
        preds = {}
        for an, (X, th, b) in arms.items():
            f = lambda r: b[0] + b[1] * Dt.H[r] + X[r] @ th
            c = se.lin_fit([f(val), Dt.H[val]], Dt.Y[val], Dt.W[val])
            preds[an] = (se.lin_pred(c, [f(val), Dt.H[val]]), se.lin_pred(c, [f(te), Dt.H[te]]))
        o_, d_ = se.darko_vec(Dk, Dt.pids, cut_s)
        dkf = lambda r: [o_[Dt.O[r]].sum(1), d_[Dt.D[r]].sum(1), Dt.H[r]]
        cd = se.lin_fit(dkf(val), Dt.Y[val], Dt.W[val]); preds['darko'] = (se.lin_pred(cd, dkf(val)), se.lin_pred(cd, dkf(te)))
        line = {}
        for an, (pv, pt) in preds.items():
            gt, ll = game_ll(pv, pt)
            R = recs[an]
            R['g'].append(gte); R['se'].append(Wt * (Yt - pt) ** 2); R['st'].append(Wt * (Yt - ybar) ** 2)
            R['gg'].append(gt); R['ll'].append(ll)
            line[an] = round(1 - np.sum(Wt * (Yt - pt) ** 2) / np.sum(Wt * (Yt - ybar) ** 2), 5)
        per_block[cut_s] = dict(rows=int(len(te)), stint_r2=line)
        print(cut_s, len(te), 'rows', f'{time.time() - t0:.0f}s', line, flush=True)

    # pooled, game-resampled CIs vs base and vs darko
    cat = {an: {k: np.concatenate(v) for k, v in R.items()} for an, R in recs.items()}
    g = cat['base']['g']; ug = np.unique(g); gi = np.searchsorted(ug, g)
    gg = cat['base']['gg']; ugg = np.unique(gg); ggi = np.searchsorted(ugg, gg)
    rng = np.random.default_rng(0); B = 2000
    idx = [rng.integers(0, len(ug), len(ug)) for _ in range(B)]
    sst = np.bincount(gi, cat['base']['st'], len(ug))
    per_g = {an: np.bincount(gi, C['se'], len(ug)) for an, C in cat.items()}
    llg = {an: C['ll'] for an, C in cat.items()}          # same game order for every arm
    res = dict(cuts=CUTS, window_days=WIN, per_block=per_block, games=int(len(ugg)), arms={})
    for an in cat:
        e = dict(stint_r2=round(float(1 - per_g[an].sum() / sst.sum()), 6), log_loss=round(float(llg[an].mean()), 5))
        for ref in ('base', 'darko', 'BdefBOX_0.6', 'BdefBOX_1.2'):
            if an == ref or ref not in cat:
                continue
            bs = [(per_g[ref][i].sum() - per_g[an][i].sum()) / sst[i].sum() for i in idx]
            e[f'r2_gain_vs_{ref}_ci'] = np.percentile(bs, [2.5, 97.5]).round(6).tolist()
            d = llg[an] - llg[ref]
            bl = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
            e[f'll_diff_vs_{ref}_ci'] = np.percentile(bl, [2.5, 97.5]).round(5).tolist()
        if an not in ('base', 'darko'):
            e['candidate'] = bool(e['r2_gain_vs_base_ci'][0] > 0 and e['ll_diff_vs_base_ci'][0] <= 0)
        res['arms'][an] = e
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'bench.json', 'w'), indent=1)
    for an, e in sorted(res['arms'].items(), key=lambda t: -t[1]['stint_r2']):
        print(an, e, flush=True)


if __name__ == '__main__':
    main()
