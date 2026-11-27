"""Corrected duncan-3 candidate test (AUDIT.md D5; I2 + I3): the tracking-informed mover pull, fully chronological.

Original: ms_movers_trk.py — movers from future playing time; BOWEN scheme values from a network cross-fitted on game
folds spanning both seasons (post-cutoff games in training); BORIS mixes from a mixture fit on both seasons.
Here, per cutoff (2026 deadline, 2026 offseason):
  movers     transaction-time (corrected_movers.py rule); 'eligible' (>= MIN_POST later possessions) only as a stratum
  scheme     team mean BOWEN scheme depth shift from results/world/chrono_{tag}/pred.npz: a network trained ONLY on
             possessions before the cutoff (world_model.py with WM_CUT; cross-fit within them), 365-day window
  roles      BORIS offensive mixes from chrono_{tag}/boris_mix.json (mixture refit on pre-cutoff possessions)
  rule       frozen as pre-registered (court-live HOLDOUT.md): k = clip(0.5 - b z, 0.05, 0.95), b = 0.15 / 0.15;
             placebo b = -0.15; flat 0.5 (DUNCAN-2) as the reference
Lambda: 3000 (as the original) and point-in-time (as corrected_movers.py). Conditional on realised lineups (stints).
Output: results/corrected/movers_trk.json     python corrected_movers_trk.py   (PC, after the chronological BOWEN runs)
"""
import json
from collections import defaultdict
from datetime import date
import numpy as np

import ms_spm as sp
import ms_stint_eval as se
from corrected_movers import tx_team_at, LAMS_PIT
from ms_fix import FData
from ms_movers import EVENTS, MIN_PRE, MIN_POST
from masked_paths import ROOT
from provenance import provenance

OUT = ROOT / 'results/corrected'
TAGS = {'2026 deadline': 'dl2026', '2026 offseason': 'os2026'}
RIM = np.array([41.75, 0.0], np.float32)


def team_schemes(tag, cut_s):
    z = np.load(ROOT / 'results/world' / f'chrono_{tag}' / 'pred.npz')
    dep = lambda a: np.hypot(a[:, 0].astype(np.float32) - RIM[0], a[:, 1].astype(np.float32) - RIM[1])
    sh = dep(z['ms']) - dep(z['mb']); P = len(z['meta_gid'])
    s_, n_ = np.bincount(z['poss'], sh, P), np.bincount(z['poss'], minlength=P)
    date_, team = z['meta_date'], z['meta_def_team']
    assert (date_ < cut_s).all(), 'chronological pred contains post-cutoff possessions'
    lo = str(np.datetime64(cut_s) - np.timedelta64(365, 'D')); ok = date_ >= lo
    return {int(t): float(s_[ok & (team == t)].sum() / max(n_[ok & (team == t)].sum(), 1)) for t in np.unique(team[ok])}, int(P)


def main():
    Dt = FData(); n = Dt.n; Dk = se.load_darko_od(); season = Dt.season_r; has = Dt.to >= 0; pid_of = np.array(Dt.pids)
    tx = defaultdict(list)
    for r in json.load(open(ROOT / 'results/latent/transactions.json')):
        if r['PLAYER_ID'] and r['Transaction_Type'] in ('Trade', 'Signing', 'AwardOnWaivers', 'Waive'):
            tx[int(r['PLAYER_ID'])].append((r['TRANSACTION_DATE'][:10], r['Transaction_Type'], int(r['TEAM_ID'])))
    for v in tx.values():
        v.sort()
    store = defaultdict(lambda: defaultdict(list)); info = {}
    for name, cut_s, days in EVENTS:
        if name not in TAGS:
            continue
        tag = TAGS[name]; cut = sp.dnum(cut_s)
        post = (Dt.DAY >= cut) & (Dt.DAY < cut + days) if days else (Dt.DAY >= cut) & (season == season[np.flatnonzero(Dt.DAY >= cut)[0]])
        pre = (Dt.DAY < cut) & (Dt.DAY >= cut - (120 if not days else 300)) & has
        post &= has
        a, b, lastg = defaultdict(float), defaultdict(float), {}
        for i in np.flatnonzero(pre):
            for p in Dt.O[i]:
                a[(p, Dt.off_team[i])] += Dt.W[i]
                if Dt.DAY[i] >= lastg.get(p, (0, 0))[0]:
                    lastg[p] = (Dt.DAY[i], Dt.off_team[i])
            for p in Dt.D[i]:
                a[(p, Dt.def_team[i])] += Dt.W[i]
        for i in np.flatnonzero(post):
            for p in Dt.O[i]:
                b[(p, Dt.off_team[i])] += Dt.W[i]
            for p in Dt.D[i]:
                b[(p, Dt.def_team[i])] += Dt.W[i]
        last = {}
        for (p, t), w in a.items():
            if w >= MIN_PRE and w > last.get(p, (0, 0))[1]:
                last[p] = (t, w)
        newt = {}; mv = np.zeros(n, bool); el = np.zeros(n, bool)
        for p, (t0, _) in last.items():
            ld, lt = lastg[p]
            tc = tx_team_at(tx, int(pid_of[p]), lt, date.fromordinal(int(ld)).isoformat(), cut_s)
            if tc is not None and tc != t0:
                newt[p] = tc; mv[p] = True; el[p] = b.get((p, tc), 0) >= MIN_POST
        S, n_pred = team_schemes(tag, cut_s)
        M = {int(k): np.array(v) for k, v in json.load(open(ROOT / 'results/world' / f'chrono_{tag}' / 'boris_mix.json')).items()}
        roster = defaultdict(list)
        for (p, t), w in a.items():
            roster[t].append((p, w))

        def occ(t, ex):
            v = [(M[int(pid_of[p])], w) for p, w in roster.get(t, []) if p != ex and int(pid_of[p]) in M]
            if not v:
                return None
            W_ = np.array([w for _, w in v]); return (np.array([q for q, _ in v]) * W_[:, None]).sum(0) / W_.sum()
        ds, dr = {}, {}
        for p, tn in newt.items():
            to = last[p][0]
            if to in S and tn in S:
                ds[p] = abs(S[tn] - S[to])
            pm, on, oo = M.get(int(pid_of[p])), occ(tn, p), occ(to, p)
            if pm is not None and on is not None and oo is not None:
                dr[p] = abs(float(pm @ on) - float(pm @ oo))

        def zs(d):
            if len(d) < 3:
                return {}
            v = np.array(list(d.values())); return {p: (x - v.mean()) / (v.std() + 1e-9) for p, x in d.items()}
        zr, zd = zs(dr), zs(ds)
        info[name] = dict(cut=cut_s, tx_movers=int(mv.sum()), eligible=int(el.sum()), with_scheme=len(ds), with_role=len(dr),
                          chrono_pred_possessions=n_pred, teams_with_scheme=len(S))
        y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y); trn = np.flatnonzero(Dt.DAY < cut)
        vrow = np.flatnonzero((Dt.DAY >= cut - sp.VAL_DAYS) & (Dt.DAY < cut))
        if len(vrow) < 2000:
            ld_ = Dt.DAY[Dt.DAY < cut].max(); vrow = np.flatnonzero((Dt.DAY > ld_ - sp.VAL_DAYS) & (Dt.DAY <= ld_))
        pr = lambda th, bb, r: bb[0] + bb[1] * Dt.H[r] + th[Dt.O[r]].sum(1) + th[n + Dt.D[r]].sum(1)
        vcut = int(Dt.DAY[vrow].min()); yv = Dt.luck_target(vcut, 0.5); mvv = Dt.spm_prior(vcut, yv); trv = np.flatnonzero(Dt.DAY < vcut)
        best, bsse = None, np.inf
        for L in LAMS_PIT:
            t_, b_ = Dt.ridge(trv, yv, mvv, L, vcut); p_ = pr(t_, b_, vrow); c_ = se.lin_fit([p_, Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow])
            s_ = float(np.sum(Dt.W[vrow] * (Dt.Y[vrow] - se.lin_pred(c_, [p_, Dt.H[vrow]])) ** 2))
            if s_ < bsse:
                best, bsse = L, s_
        info[name]['lam_pit'] = best
        rows = np.flatnonzero(post); Y, W = Dt.Y[rows], Dt.W[rows]; ybar = np.average(Dt.Y[vrow], weights=Dt.W[vrow])
        strata = {'tx': mv[Dt.O[rows]].any(1) | mv[Dt.D[rows]].any(1), 'eligible': el[Dt.O[rows]].any(1) | el[Dt.D[rows]].any(1)}
        preds = {}
        for Ltag, L in (('3000', 3000.0), ('pit', best)):
            th, bb = Dt.ridge(trn, y, m, L, cut)
            for an, bo, bd in (('flat', 0.0, 0.0), ('trk', 0.15, 0.15), ('placebo', -0.15, -0.15)):
                k = np.ones(2 * n)
                for p in newt:
                    k[p] = np.clip(0.5 - bo * zr.get(p, 0.0), 0.05, 0.95); k[n + p] = np.clip(0.5 - bd * zd.get(p, 0.0), 0.05, 0.95)
                t2 = m + k * (th - m); c = se.lin_fit([pr(t2, bb, vrow), Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow])
                preds[f'{an}_{Ltag}'] = se.lin_pred(c, [pr(t2, bb, rows), Dt.H[rows]])
        o_, d_ = se.darko_vec(Dk, Dt.pids, cut_s); dkf = lambda r: [o_[Dt.O[r]].sum(1), d_[Dt.D[r]].sum(1), Dt.H[r]]
        preds['darko'] = se.lin_pred(se.lin_fit(dkf(vrow), Dt.Y[vrow], Dt.W[vrow]), dkf(rows))
        for an, pv in preds.items():
            St = store[an]; St['g'].append(Dt.GI[rows]); St['event'].append(np.full(len(rows), name))
            St['se'].append(W * (Y - pv) ** 2); St['st'].append(W * (Y - ybar) ** 2)
            for k_, v_ in strata.items():
                St[k_].append(v_)
        print(name, info[name], flush=True)
    cat = {an: {f: np.concatenate(v) for f, v in S_.items()} for an, S_ in store.items()}
    ref = cat['darko']; ug, gi = np.unique(ref['g'], return_inverse=True); rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(ug), len(ug)) for _ in range(2000)]
    res = dict(provenance=provenance(__file__, ['results/world/chrono_dl2026/pred.npz', 'results/world/chrono_os2026/pred.npz',
                                                'results/world/chrono_dl2026/boris_mix.json', 'results/world/chrono_os2026/boris_mix.json',
                                                'results/latent/transactions.json', 'results/latent/ms2_stints_2526.pkl',
                                                'results/latent/ms2_stints_2627.pkl']),
               config=dict(rule='pre-registered b 0.15/0.15', evaluation='conditional on realised lineups (stints)'),
               events=info, strata={}, per_event={})
    for strat in ('tx', 'eligible'):
        mk = ref[strat]; st = np.bincount(gi, ref['st'] * mk, len(ug))
        sse = {an: np.bincount(gi, C['se'] * mk, len(ug)) for an, C in cat.items()}
        res['strata'][strat] = {an: dict(r2=round(float(1 - sse[an].sum() / st.sum()), 6),
                                         **{f'gain_vs_{r_}_ci': np.percentile([(sse[r_][i].sum() - sse[an][i].sum()) / st[i].sum() for i in idx], [2.5, 97.5]).round(6).tolist()
                                            for r_ in (f"flat_{an.split('_')[1]}" if '_' in an else 'flat_3000', 'darko') if r_ != an})
                                for an in cat}
        for name in TAGS:
            sel = (ref['event'] == name) & mk
            res['per_event'].setdefault(name, {})[strat] = {an: round(float(1 - C['se'][sel].sum() / ref['st'][sel].sum()), 6) for an, C in cat.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'movers_trk.json', 'w'), indent=1)
    print(json.dumps(res['strata'], indent=1)); print(json.dumps(res['per_event'], indent=1))


if __name__ == '__main__':
    main()
