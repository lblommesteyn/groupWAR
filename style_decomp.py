"""How much of each style metric is the player and how much is the team (scheme + context)? Two-way fixed effects
(AKM, as in worker/firm wage decompositions), identified by players who changed teams during the archive.

Units: (player, team, half). Each team's games alternate odd/even (half); a mover's games are split the same way
inside each of his team spells. describe2 is run on each slice of games, so every metric is computed exactly as in the
rating model (shrunk rates). Per metric and half, weighted least squares fits
    x[p, t] = a[p] + psi[t] + e        (weights = minutes in the slice)
on the connected set. Halves use disjoint games, so their estimation errors are independent and covariances across
halves are unbiased (this removes limited-mobility bias):
    reliable variance  cov(x_odd, x_even)
    player share       cov(a_odd, a_even) / reliable
    team share         cov(psi_odd, psi_even) / reliable
    sorting            [cov(a_odd, psi_even) + cov(a_even, psi_odd)] / reliable   (do good-X players land on X teams?)
Team = coach scheme + teammates + context; within one season the coach is not separable from the team.
CIs: bootstrap over players (team effects refit each draw). Also an event-study check for movers:
    slope of (x_new - x_old) on (psi_new - psi_old), psi estimated from the other half of games.
Output: results/style_decomp/decomp.json
  EPV=gravity+coverage HOLES=1 python style_decomp.py
"""
import json, os, pickle
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
import embed_eval as ee
import metrics2 as m2
from masked_paths import ROOT

OUT = ROOT / 'results/style_decomp'
MIN_SPELL_GAMES = 4
MIN_MINUTES = 40
NBOOT = 300


def slices(G):
    team_games = defaultdict(list); pteam = defaultdict(dict)
    for g, v in G.items():
        if not v['stints']:
            continue
        team_games[v['home']].append((v['date'], g)); team_games[v['away']].append((v['date'], g))
        home, away = v['stints'][0]['lineup']
        on = defaultdict(set)
        for s in v['stints']:
            for p in s['lineup'][0]:
                on[p].add(v['home'])
            for p in s['lineup'][1]:
                on[p].add(v['away'])
        for p, ts in on.items():
            if len(ts) == 1:
                pteam[p][g] = (v['date'], next(iter(ts)))
    spells = defaultdict(lambda: defaultdict(list))
    for p, d in pteam.items():
        for g, (dt, t) in d.items():
            spells[p][t].append((dt, g))
    movers = {p: {t: sorted(gs) for t, gs in sp.items()} for p, sp in spells.items()
              if sum(len(gs) >= MIN_SPELL_GAMES for gs in sp.values()) >= 2}
    out = []  # (kind, key, half, gids)
    for t, gs in team_games.items():
        gs = sorted(gs)
        for h in (0, 1):
            out.append(('team', t, h, [g for i, (_, g) in enumerate(gs) if i % 2 == h]))
    for p, sp in movers.items():
        for t, gs in sp.items():
            if len(gs) >= MIN_SPELL_GAMES:
                for h in (0, 1):
                    out.append(('mover', (p, t), h, [g for i, (_, g) in enumerate(gs) if i % 2 == h]))
    return out, movers, pteam


def fit_fe(rows, feats):
    """rows: list of (pid, team, w, xvec). Returns a [P], psi [T] per feature (dict), index maps."""
    P = {p: i for i, p in enumerate(sorted({r[0] for r in rows}))}
    T = {t: i for i, t in enumerate(sorted({r[1] for r in rows}))}
    n = len(rows); w = np.sqrt(np.array([r[2] for r in rows]))
    ri = np.repeat(np.arange(n), 2)
    ci = np.ravel([[P[r[0]], len(P) + T[r[1]]] for r in rows])
    A = csr_matrix((np.repeat(w, 2), (ri, ci)), shape=(n, len(P) + len(T)))
    X = np.array([r[3] for r in rows])
    res = {}
    for j, f in enumerate(feats):
        y = X[:, j]
        sol = lsqr(A, w * (y - np.average(y, weights=w ** 2)), damp=1e-3, atol=1e-10, btol=1e-10)[0]
        a, psi = sol[:len(P)], sol[len(P):]
        res[f] = (a, psi)
    return res, P, T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
    shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    feats = list(dict.fromkeys(ee.BOX + ee.TRK_OFF + ee.TRK_DEF + m2.NEW_OFF + m2.NEW_DEF + (m2.HOLES_OFF + m2.HOLES_DEF if m2.HOLES else [])
                               + (['grav'] if 'gravity' in m2.EPV else [])
                               + (['cov_drop', 'cov_switch', 'cov_up'] if 'coverage' in m2.EPV else [])))
    sl, movers, pteam = slices(G)
    print('slices', len(sl), 'movers', len(movers), flush=True)
    cache = OUT / 'slices.pkl'
    if cache.exists():
        U = pickle.load(open(cache, 'rb'))
    else:
        U = {}  # (pid, team, half) -> (minutes, {feat: value})
        for i, (kind, key, h, gids) in enumerate(sl):
            d = m2.describe2(G, gids, shots)
            if kind == 'team':
                t = key
                for p, f in d.items():
                    if p in movers:
                        continue  # movers come from their own spell slices
                    teams = {pteam[p].get(g, (0, None))[1] for g in gids} - {None}
                    if teams == {t}:
                        U[(p, t, h)] = (f['minutes'], {k: f.get(k, np.nan) for k in feats})
            else:
                p, t = key
                if p in d:
                    U[(p, t, h)] = (d[p]['minutes'], {k: d[p].get(k, np.nan) for k in feats})
            if i % 20 == 0:
                print(f'{i}/{len(sl)}', flush=True)
        pickle.dump(U, open(cache, 'wb'))
    feats = [f for f in feats if all(np.isfinite(v[1].get(f, np.nan)) for v in U.values())]
    # keep units observed in both halves with enough minutes in each
    keys = sorted({(p, t) for (p, t, h) in U if (p, t, 0) in U and (p, t, 1) in U
                   and min(U[(p, t, 0)][0], U[(p, t, 1)][0]) >= MIN_MINUTES})
    # largest set of teams connected through movers
    par = {}
    fnd = lambda a: a if par.setdefault(a, a) == a else fnd(par[a])
    for p in movers:
        ts = [t for pp, t in keys if pp == p]
        for t in ts[1:]:
            par[fnd(ts[0])] = fnd(t)
    comp = defaultdict(set)
    for _, t in keys:
        comp[fnd(t)].add(t)
    big = max(comp.values(), key=len)
    keys = [k for k in keys if k[1] in big]
    mv = sorted({p for p, t in keys if p in movers and sum(k[0] == p for k in keys) >= 2})
    print('units', len(keys), 'movers usable', len(mv), 'teams', len({t for _, t in keys}), flush=True)

    def decomp(keys_):
        rows = {h: [(p, t, U[(p, t, h)][0], [U[(p, t, h)][1][f] for f in feats]) for p, t in keys_] for h in (0, 1)}
        fit = {h: fit_fe(rows[h], feats) for h in (0, 1)}
        out = {}
        for j, f in enumerate(feats):
            x = {h: np.array([r[3][j] for r in rows[h]]) for h in (0, 1)}
            w = np.array([min(r[2], s[2]) for r, s in zip(rows[0], rows[1])])
            a = {}; ps = {}
            for h in (0, 1):
                (res, P, T) = fit[h]; aa, pp = res[f]
                a[h] = np.array([aa[P[p]] for p, t in keys_]); ps[h] = np.array([pp[T[t]] for p, t in keys_])
            cv = lambda u, v: np.cov(u, v, aweights=w)[0, 1]
            rel = cv(x[0], x[1])
            out[f] = dict(total=float(np.mean([cv(x[h], x[h]) for h in (0, 1)])), reliable=float(rel),
                          player=float(cv(a[0], a[1]) / rel), team=float(cv(ps[0], ps[1]) / rel),
                          sorting=float((cv(a[0], ps[1]) + cv(a[1], ps[0])) / rel))
        return out, fit

    base, fit = decomp(keys)
    # event study: movers' change vs change in team effect estimated on the other half
    ev = {}
    for j, f in enumerate(feats):
        dx, dpsi = [], []
        for h in (0, 1):
            res, P, T = fit[1 - h]; _, pp = res[f]
            for p in mv:
                sp = sorted([(movers[p][t][0][0], t) for pp_, t in keys if pp_ == p])
                (t_old, t_new) = (sp[0][1], sp[-1][1])
                dx.append(U[(p, t_new, h)][1][f] - U[(p, t_old, h)][1][f]); dpsi.append(pp[T[t_new]] - pp[T[t_old]])
        dx, dpsi = np.array(dx), np.array(dpsi)
        ev[f] = float(np.polyfit(dpsi, dx, 1)[0]) if np.std(dpsi) > 0 else None
    # bootstrap over players
    rng = np.random.default_rng(0)
    players = sorted({p for p, _ in keys}); byp = defaultdict(list)
    for k in keys:
        byp[k[0]].append(k)
    B = defaultdict(lambda: defaultdict(list))
    for b in range(NBOOT):
        draw = rng.choice(len(players), len(players))
        # duplicate units get distinct pids so the fit treats them as separate players
        kb2 = []
        U_b = {}
        cnt = defaultdict(int)
        for i in draw:
            p = players[i]; cnt[p] += 1; tag = (p, cnt[p])
            for _, t in byp[p]:
                kb2.append((tag, t))
                for h in (0, 1):
                    U_b[(tag, t, h)] = U[(p, t, h)]
        Uo = dict(U); U.clear(); U.update(U_b)
        try:
            r, _ = decomp(kb2)
        finally:
            U.clear(); U.update(Uo)
        for f, v in r.items():
            for k in ('player', 'team', 'sorting'):
                B[f][k].append(v[k])
        if b % 50 == 0:
            print('boot', b, flush=True)
    names = {}
    for v in G.values():
        names.update(v['players'])
    res = dict(n_units=len(keys), n_teams=len(big), n_movers=len(mv), movers=[names.get(p, str(p)) for p in mv], metrics={})
    for f in feats:
        d = base[f]
        for k in ('player', 'team', 'sorting'):
            d[k + '_ci'] = [float(np.percentile(B[f][k], 2.5)), float(np.percentile(B[f][k], 97.5))]
        d['reliability'] = d['reliable'] / d['total']
        d['mover_slope'] = ev[f]
        res['metrics'][f] = d
    json.dump(res, open(OUT / 'decomp.json', 'w'), indent=1)
    print(f"{'metric':22s} {'rel':>5s} {'player':>18s} {'team':>18s} {'sorting':>7s} {'mover':>6s}")
    for f, d in sorted(res['metrics'].items(), key=lambda kv: -kv[1]['team']):
        print(f"{f:22s} {d['reliability']:5.2f} {d['player']:5.2f} [{d['player_ci'][0]:4.2f},{d['player_ci'][1]:4.2f}] "
              f"{d['team']:5.2f} [{d['team_ci'][0]:4.2f},{d['team_ci'][1]:4.2f}] {d['sorting']:7.2f} "
              f"{d['mover_slope'] if d['mover_slope'] is None else round(d['mover_slope'], 2)!s:>6s}")


if __name__ == '__main__':
    main()
