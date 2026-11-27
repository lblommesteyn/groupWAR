"""Movers with tracking: replace DUNCAN-2's uniform mover pull with a per-player pull set by BORIS and BOWEN.

DUNCAN-2 pulls every player who changed teams halfway to his box prior: theta <- m + 0.5 (theta - m).
BOWEN found that a defender's scheme follows the team and his execution follows him; BORIS gives each player's
offensive role mix. So the pull should depend on how different the new context is:
  defense   scheme change   ds_i = | S(new team) - S(old team) |, S = the team's mean BOWEN scheme depth shift
                            (scheme spot minus league-baseline spot, distance to rim, ft), from its possessions in the
                            365 days before the cutoff.
  offense   role crowding   dr_i = | p_i . occ(new) - p_i . occ(old) |, p_i = his BORIS offensive mix from months
                            before the cutoff, occ(T) = possession-weighted mean mix of T's other players (pre-cutoff
                            possessions). How much more or less his roles are already taken on the new team.
Per event, z-score each among the movers who have it (others z = 0), then
  k_off_i = clip(0.5 - b_off z_r,i, 0.05, 0.95),  k_def_i = clip(0.5 - b_def z_s,i, 0.05, 0.95)
so the average pull stays 0.5 and only its allocation across movers changes. b = 0 is DUNCAN-2.
Pre-declared primary arm: b_off = b_def = 0.15. Negative b (pull LESS when the context changes more) is the
placebo direction. Events: the two cutoffs with tracking before them, 2026 deadline and 2026 offseason.
Caveats: BOWEN's network was cross-fitted on both seasons, so its scheme representation saw post-cutoff games
(team scheme values used here are pre-cutoff only); two events, so this is a small test.
Metric: stint R2 on post-cutoff stints (ms_shrink_d2 harness: linear map fit on the 30 days before the cutoff),
all stints and stints with a mover; 95% CI vs DUNCAN-2 by resampling games.
Output: results/latent/ms_movers_trk.json     python ms_movers_trk.py   (PC)
"""
import json
from collections import defaultdict
import numpy as np

import ms_spm as sp
import ms_stint_eval as se
from ms_fix import FData
from ms_movers import EVENTS, MIN_PRE, MIN_POST
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
LAM = 3000.0
EVS = ('2026 deadline', '2026 offseason')
B_OFF = [-0.15, 0.0, 0.15, 0.3]
B_DEF = [-0.15, 0.0, 0.15, 0.3]
RIM = np.array([41.75, 0.0], np.float32)


def team_schemes(cut_s):
    """team id -> mean scheme depth shift (ft) over its possessions in the 365 days before cut_s."""
    if not hasattr(team_schemes, 'z'):
        z = np.load(ROOT / 'results/world/pred.npz')
        dep = lambda a: np.hypot(a[:, 0].astype(np.float32) - RIM[0], a[:, 1].astype(np.float32) - RIM[1])
        sh = dep(z['ms']) - dep(z['mb']); poss = z['poss']
        P = len(z['meta_gid'])
        team_schemes.z = dict(sum=np.bincount(poss, sh, P), n=np.bincount(poss, minlength=P),
                              date=z['meta_date'], team=z['meta_def_team'])
    Z = team_schemes.z
    lo = str(np.datetime64(cut_s) - np.timedelta64(365, 'D'))
    ok = (Z['date'] < cut_s) & (Z['date'] >= lo)
    out = {}
    for t in np.unique(Z['team'][ok]):
        m = ok & (Z['team'] == t)
        out[int(t)] = float(Z['sum'][m].sum() / max(Z['n'][m].sum(), 1))
    return out


def pre_mix(mixes, cut_s):
    """NBA id -> mean BORIS offensive mix over months before the cutoff's month."""
    mo = cut_s[:7]; out = {}
    for pid, v in mixes.items():
        ms = [np.array(m) for k, m in v.get('by_month', {}).items() if k < mo]
        if ms:
            out[int(pid)] = np.mean(ms, 0)
    return out


def main():
    mixes = json.load(open(ROOT / 'results/archetypes/mix.json'))['players']['off']
    Dk = se.load_darko_od()
    Dt = FData(); n = Dt.n; season = Dt.season_r; has = Dt.to >= 0
    pid_of = np.array(Dt.pids)
    store = defaultdict(lambda: defaultdict(list)); info = {}
    for name, cut_s, days in EVENTS:
        if name not in EVS:
            continue
        cut = sp.dnum(cut_s)
        if days:
            post = (Dt.DAY >= cut) & (Dt.DAY < cut + days)
        else:
            s0 = season[np.flatnonzero(Dt.DAY >= cut)[0]]; post = (Dt.DAY >= cut) & (season == s0)
        pre = (Dt.DAY < cut) & (Dt.DAY >= cut - (120 if not days else 300)) & has
        post &= has

        def poss_by(mask):
            c = defaultdict(float)
            for i in np.flatnonzero(mask):
                for p in Dt.O[i]:
                    c[(p, Dt.off_team[i])] += Dt.W[i]
                for p in Dt.D[i]:
                    c[(p, Dt.def_team[i])] += Dt.W[i]
            return c
        a, b = poss_by(pre), poss_by(post)
        last = {}
        for (p, t), w in a.items():
            if w >= MIN_PRE and w > last.get(p, (0, 0))[1]:
                last[p] = (t, w)
        new = {}
        for (p, t), w in b.items():
            if p in last and t != last[p][0] and w >= MIN_POST and w > new.get(p, (0, 0))[1]:
                new[p] = (t, w)
        mv = np.zeros(n, bool); mv[list(new)] = True
        # context features
        S = team_schemes(cut_s); M = pre_mix(mixes, cut_s)
        roster = defaultdict(list)                       # pre-cutoff team -> (player, possessions); new team adds movers' new teams' existing players
        for (p, t), w in a.items():
            roster[t].append((p, w))

        def occ(t, excl):
            v = [(M[int(pid_of[p])], w) for p, w in roster.get(t, []) if p != excl and int(pid_of[p]) in M]
            if not v:
                return None
            W_ = np.array([w for _, w in v]); return (np.array([m for m, _ in v]) * W_[:, None]).sum(0) / W_.sum()
        ds, dr = {}, {}
        for p, (tn, _) in new.items():
            to = last[p][0]
            if to in S and tn in S:
                ds[p] = abs(S[tn] - S[to])
            pm = M.get(int(pid_of[p]))
            on, oo = occ(tn, p), occ(to, p)
            if pm is not None and on is not None and oo is not None:
                dr[p] = abs(float(pm @ on) - float(pm @ oo))

        def zs(d):
            if len(d) < 3:
                return {}
            v = np.array(list(d.values())); mu, sd = v.mean(), v.std() + 1e-9
            return {p: (x - mu) / sd for p, x in d.items()}
        zr, zd = zs(dr), zs(ds)
        info[name] = dict(movers=int(mv.sum()), with_scheme=len(ds), with_role=len(dr),
                          scheme_change_ft=dict(mean=float(np.mean(list(ds.values()))) if ds else None),
                          role_change=dict(mean=float(np.mean(list(dr.values()))) if dr else None))
        print(name, info[name], flush=True)

        y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
        trn = np.flatnonzero(Dt.DAY < cut)
        vrow = np.flatnonzero((Dt.DAY >= cut - sp.VAL_DAYS) & (Dt.DAY < cut))
        if len(vrow) < 2000:
            ld = Dt.DAY[Dt.DAY < cut].max(); vrow = np.flatnonzero((Dt.DAY > ld - sp.VAL_DAYS) & (Dt.DAY <= ld))
        rows = np.flatnonzero(post); Y, W = Dt.Y[rows], Dt.W[rows]
        mrow = mv[Dt.O[rows]].any(1) | mv[Dt.D[rows]].any(1)
        ybar = np.average(Dt.Y[vrow], weights=Dt.W[vrow])
        pr = lambda th, bb, r: bb[0] + bb[1] * Dt.H[r] + th[Dt.O[r]].sum(1) + th[n + Dt.D[r]].sum(1)
        th, bb = Dt.ridge(trn, y, m, LAM, cut)
        preds = {}
        for bo in B_OFF:
            for bd in B_DEF:
                k = np.ones(2 * n)
                for p in new:
                    k[p] = np.clip(0.5 - bo * zr.get(p, 0.0), 0.05, 0.95)
                    k[n + p] = np.clip(0.5 - bd * zd.get(p, 0.0), 0.05, 0.95)
                t2 = m + k * (th - m)
                c = se.lin_fit([pr(t2, bb, vrow), Dt.H[vrow]], Dt.Y[vrow], Dt.W[vrow])
                preds[f'off{bo:+.2f}_def{bd:+.2f}'] = se.lin_pred(c, [pr(t2, bb, rows), Dt.H[rows]])
        o_, d_ = se.darko_vec(Dk, Dt.pids, cut_s)
        dkf = lambda r: [o_[Dt.O[r]].sum(1), d_[Dt.D[r]].sum(1), Dt.H[r]]
        cd = se.lin_fit(dkf(vrow), Dt.Y[vrow], Dt.W[vrow]); preds['darko'] = se.lin_pred(cd, dkf(rows))
        for an, pv in preds.items():
            St = store[an]
            St['g'].append(Dt.GI[rows]); St['mrow'].append(mrow)
            St['se'].append(W * (Y - pv) ** 2); St['st'].append(W * (Y - ybar) ** 2)
        print(name, len(rows), 'post rows,', int(mrow.sum()), 'with a mover', flush=True)

    cat = {an: {f: np.concatenate(v) for f, v in S_.items()} for an, S_ in store.items()}
    base = 'off+0.00_def+0.00'
    g = cat[base]['g']; ug = np.unique(g); gi = np.searchsorted(ug, g); rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(ug), len(ug)) for _ in range(2000)]
    res = dict(events=info, base='DUNCAN-2 (uniform pull 0.5)', primary='off+0.15_def+0.15', arms={})
    for an, S_ in cat.items():
        e = {}
        for lab, mk in (('all', np.ones(len(g), bool)), ('movers', S_['mrow'])):
            sse = np.bincount(gi, S_['se'] * mk, len(ug)); sst = np.bincount(gi, S_['st'] * mk, len(ug))
            sb = np.bincount(gi, cat[base]['se'] * mk, len(ug))
            e[lab] = round(float(1 - sse.sum() / sst.sum()), 6)
            if an != base:
                bs = [(sb[i].sum() - sse[i].sum()) / sst[i].sum() for i in idx]
                e[lab + '_vs_duncan2_ci'] = np.percentile(bs, [2.5, 97.5]).round(6).tolist()
        res['arms'][an] = e
    for an in sorted(res['arms'], key=lambda a: -res['arms'][a]['movers']):
        print(an, res['arms'][an], flush=True)
    json.dump(res, open(OUT / 'ms_movers_trk.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
