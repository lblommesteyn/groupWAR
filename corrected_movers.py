"""Corrected movers evaluation (AUDIT.md I2): movers identified at TRANSACTION TIME, everything fit before the cutoff.

Original (ms_shrink_d2.py:54, ms_movers.py:72): a mover = >= MIN_POSS possessions for one team before the cutoff and
>= MIN_POST possessions for ANOTHER team AFTER it, so the set (and the pull applied to it) used future playing time.

Here, per event cutoff c:
  last team     team with the most possessions in the pre window (120 days; 300 for offseason cutoffs) and >= MIN_PRE
  team at c     team of his last game before c, updated by every transaction dated after that game and <= c (results/latent/transactions.json, nba.com player
                movement): Trade / Signing / AwardOnWaivers -> TEAM_ID (receiving team), Waive -> no team
  mover (tx)    team at c exists and differs from the last team. Known before any post-cutoff game.
  eligible      tx movers with >= MIN_POST possessions after c: an evaluation STRATUM (uses the future), reported
                separately, never used to decide whom to pull
  orig          the original future-defined set, for comparison only
Arms (ratings fit on all stints before c, half-strength luck target, SPM prior; stint predictions mapped to points per
100 with a linear fit on the 30 days before c, as ms_shrink_d2):
  nopull_L      no mover pull
  d2_L          flat 0.5 pull toward the prior for tx movers (the deployed DUNCAN-2 rule, transaction-time movers)
  d2orig_L      flat 0.5 pull for the original future-defined movers (reproduces the old number)
  darko         DARKO point-in-time
  L = 3000 (deployed; chosen on 2020-26 blocks, so NOT chronological for these events) and pit (chosen from
  {1000, 3000, 10000} on the 30 days before c with a fit ending 30 days earlier: chronological).
Evaluation is conditional on the realised post-cutoff lineups (stint-level). Not a pregame forecast.
Strata: all post-cutoff stints; stints with a tx mover; with an eligible mover; with an orig mover. Held-out events =
not in the 2021-2023 deadlines used to tune k 0.5. 95% CIs resample games; per-event results are saved too.
Output: results/corrected/movers.json (+ provenance)     python corrected_movers.py   (PC)
"""
import json
from collections import defaultdict
import numpy as np

import ms_spm as sp
import ms_stint_eval as se
from ms_fix import FData, TUNE
from ms_movers import EVENTS, MIN_PRE, MIN_POST
from masked_paths import ROOT
from provenance import provenance

OUT = ROOT / 'results/corrected'
LAMS_PIT = [1000.0, 3000.0, 10000.0]


def tx_team_at(tx, pid, start_team, start_day, cut_s):
    """Team of player pid at the end of cut_s, from start_team, applying his transactions dated after start_day."""
    team = start_team
    for d, typ, t in tx.get(pid, []):
        if d <= start_day or d > cut_s:
            continue
        team = None if typ == 'Waive' else t
    return team


def main():
    Dt = FData(); n = Dt.n; Dk = se.load_darko_od(); season = Dt.season_r; has = Dt.to >= 0
    tx = defaultdict(list)
    for r in json.load(open(ROOT / 'results/latent/transactions.json')):
        if r['PLAYER_ID'] and r['Transaction_Type'] in ('Trade', 'Signing', 'AwardOnWaivers', 'Waive'):
            tx[int(r['PLAYER_ID'])].append((r['TRANSACTION_DATE'][:10], r['Transaction_Type'], int(r['TEAM_ID'])))
    for v in tx.values():
        v.sort()
    pid_of = np.array(Dt.pids)
    store = defaultdict(lambda: defaultdict(list)); ev_info = {}
    for name, cut_s, days in EVENTS:
        cut = sp.dnum(cut_s)
        if cut > Dt.DAY.max():
            continue
        if days:
            post = (Dt.DAY >= cut) & (Dt.DAY < cut + days)
        else:
            s0 = season[np.flatnonzero(Dt.DAY >= cut)[0]]; post = (Dt.DAY >= cut) & (season == s0)
        pre = (Dt.DAY < cut) & (Dt.DAY >= cut - (120 if not days else 300)) & has
        post &= has

        def poss_by(mask):
            c = defaultdict(float); lastday = {}
            for i in np.flatnonzero(mask):
                for p in Dt.O[i]:
                    c[(p, Dt.off_team[i])] += Dt.W[i]
                    if Dt.DAY[i] >= lastday.get(p, (0, 0))[0]:
                        lastday[p] = (Dt.DAY[i], Dt.off_team[i])
                for p in Dt.D[i]:
                    c[(p, Dt.def_team[i])] += Dt.W[i]
            return c, lastday
        (a, lastday), (b, _) = poss_by(pre), poss_by(post)
        last = {}
        for (p, t), w in a.items():
            if w >= MIN_PRE and w > last.get(p, (0, 0))[1]:
                last[p] = (t, w)
        mv_tx = np.zeros(n, bool); mv_orig = np.zeros(n, bool); mv_elig = np.zeros(n, bool)
        from datetime import date
        for p, (t0, _) in last.items():
            # current team at c (as live: DARKO's tm_id) = team of his last game before c + transactions since then;
            # mover = current team differs from the most-possessions team of the pre window
            ld_, lt_ = lastday[p]
            tc = tx_team_at(tx, int(pid_of[p]), lt_, date.fromordinal(int(ld_)).isoformat(), cut_s)
            if tc is not None and tc != t0:
                mv_tx[p] = True
                if b.get((p, tc), 0) >= MIN_POST:
                    mv_elig[p] = True
        for (p, t), w in b.items():
            if p in last and t != last[p][0] and w >= MIN_POST:
                mv_orig[p] = True
        ev_info[name] = dict(cut=cut_s, tx_movers=int(mv_tx.sum()), eligible=int(mv_elig.sum()), orig=int(mv_orig.sum()),
                             orig_not_tx=int((mv_orig & ~mv_tx).sum()), tx_not_orig=int((mv_tx & ~mv_orig).sum()))
        y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
        trn = np.flatnonzero(Dt.DAY < cut)
        vrow = np.flatnonzero((Dt.DAY >= cut - sp.VAL_DAYS) & (Dt.DAY < cut))
        if len(vrow) < 2000:
            ld = Dt.DAY[Dt.DAY < cut].max(); vrow = np.flatnonzero((Dt.DAY > ld - sp.VAL_DAYS) & (Dt.DAY <= ld))
        # point-in-time lambda: fit ending at the start of the validation window, score validation stints
        vcut = int(Dt.DAY[vrow].min())
        yv = Dt.luck_target(vcut, 0.5); mvv = Dt.spm_prior(vcut, yv); trv = np.flatnonzero(Dt.DAY < vcut)
        pr = lambda th, bb, r: bb[0] + bb[1] * Dt.H[r] + th[Dt.O[r]].sum(1) + th[n + Dt.D[r]].sum(1)
        best, best_sse = None, np.inf
        for L in LAMS_PIT:
            th_, bb_ = Dt.ridge(trv, yv, mvv, L, vcut)
            p_ = pr(th_, bb_, vrow); c_ = se.lin_fit([p_, Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow])
            sse = float(np.sum(Dt.W[vrow] * (Dt.Y[vrow] - se.lin_pred(c_, [p_, Dt.H[vrow]])) ** 2))
            if sse < best_sse:
                best, best_sse = L, sse
        ev_info[name]['lam_pit'] = best
        rows = np.flatnonzero(post); Y, W = Dt.Y[rows], Dt.W[rows]
        strata = {k: v[Dt.O[rows]].any(1) | v[Dt.D[rows]].any(1) for k, v in (('tx', mv_tx), ('eligible', mv_elig), ('orig', mv_orig))}
        ybar = np.average(Dt.Y[vrow], weights=Dt.W[vrow])
        preds = {}
        for Ltag, L in (('3000', 3000.0), ('pit', best)):
            th, bb = Dt.ridge(trn, y, m, L, cut)
            for an, mvset in (('nopull', None), ('d2', mv_tx), ('d2orig', mv_orig)):
                k = np.ones(2 * n) if mvset is None else np.where(np.r_[mvset, mvset], 0.5, 1.0)
                t2 = m + k * (th - m)
                c = se.lin_fit([pr(t2, bb, vrow), Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow])
                preds[f'{an}_{Ltag}'] = se.lin_pred(c, [pr(t2, bb, rows), Dt.H[rows]])
        o_, d_ = se.darko_vec(Dk, Dt.pids, cut_s)
        dkf = lambda r: [o_[Dt.O[r]].sum(1), d_[Dt.D[r]].sum(1), Dt.H[r]]
        preds['darko'] = se.lin_pred(se.lin_fit(dkf(vrow), Dt.Y[vrow], Dt.W[vrow]), dkf(rows))
        for an, pv in preds.items():
            S = store[an]
            S['event'].append(np.full(len(rows), name)); S['g'].append(Dt.GI[rows])
            S['se'].append(W * (Y - pv) ** 2); S['st'].append(W * (Y - ybar) ** 2)
            for k_, v_ in strata.items():
                S[k_].append(v_)
        print(name, ev_info[name], flush=True)

    cat = {an: {f: np.concatenate(v) for f, v in S.items()} for an, S in store.items()}
    ref = cat['darko']; held = ~np.isin(ref['event'], list(TUNE))
    rng = np.random.default_rng(0)

    def summarize(mask_name, sel):
        g = ref['g'][sel]; ug, gi = np.unique(g, return_inverse=True)
        idx = [rng.integers(0, len(ug), len(ug)) for _ in range(2000)]
        out = {}
        mk = np.ones(sel.sum(), bool) if mask_name == 'all' else ref[mask_name][sel]
        st = np.bincount(gi, ref['st'][sel] * mk, len(ug))
        sse = {an: np.bincount(gi, C['se'][sel] * mk, len(ug)) for an, C in cat.items()}
        for an in cat:
            e = dict(r2=round(float(1 - sse[an].sum() / st.sum()), 6))
            for r_ in ('nopull_3000', 'd2_3000', 'darko'):
                if r_ != an:
                    bs = [(sse[r_][i].sum() - sse[an][i].sum()) / st[i].sum() for i in idx]
                    e[f'gain_vs_{r_}_ci'] = np.percentile(bs, [2.5, 97.5]).round(6).tolist()
            out[an] = e
        return out
    res = dict(provenance=provenance(__file__, ['results/latent/ms2_stints.pkl', 'results/latent/ms2_stints_2325.pkl',
                                                'results/latent/ms2_stints_2526.pkl', 'results/latent/ms2_stints_2627.pkl',
                                                'results/latent/gamelogs.parquet', 'results/latent/transactions.json',
                                                'results/benchmarks/darko_history.json']),
               config=dict(min_pre=MIN_PRE, min_post=MIN_POST, lams_pit=LAMS_PIT, k_mv=0.5, tune_events=sorted(TUNE),
                           evaluation='conditional on realised post-cutoff lineups (stint level); not pregame'),
               events=ev_info, held={}, per_event={})
    for strat in ('all', 'tx', 'eligible', 'orig'):
        res['held'][strat] = summarize(strat, held)
    for name in ev_info:
        sel = ref['event'] == name
        res['per_event'][name] = {an: round(float(1 - (C['se'][sel] * C['tx'][sel]).sum() / (ref['st'][sel] * ref['tx'][sel]).sum()), 6)
                                  for an, C in cat.items()} if ref['tx'][sel].any() else None
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'movers.json', 'w'), indent=1)
    for strat in ('tx', 'eligible', 'orig', 'all'):
        print('HELD', strat, json.dumps(res['held'][strat]), flush=True)


if __name__ == '__main__':
    main()
