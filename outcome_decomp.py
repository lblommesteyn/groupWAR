"""Where does winning come from? Stint scoring margin split into player talent, coach, franchise, team-season and
five-man-lineup synergy (v3 stints, regular seasons 2019-20..2022-23; results/decomp/stints.pkl + games.parquet).

y = margin per 48 minutes in the stint (home minus away), weight = stint seconds. Home-minus-away design:
  player     persistent talent (one column per player, +1 home / -1 away)
  aging      player-season deviation from his persistent talent
  coach      head coach at that game (coaches.csv; mid-season changes assigned in listed order by game count, so a
             firing gives same roster / different coach)
  franchise  team identity across seasons and coaches
  team_season   that team in that season beyond the above (roster context, health, chemistry)
  lineup     the specific five-man unit beyond its players (synergy)
  home       intercept
Each group is a ridge-penalised random effect; penalties chosen on held-out games. Then the model is fit separately on
odd and even games; for each group, the covariance between its odd-fit and even-fit contribution across all stints
(seconds weighted) is its reliable share (noise in the two fits is independent, so it cancels). Shares are reported
relative to the reliable variance of the full prediction. Held-out R2 increments, group by group, as a check.
Output: results/decomp/outcome_decomp.json
  python outcome_decomp.py
"""
import json, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr
from masked_paths import ROOT

D = ROOT / 'results/decomp'
ABBR = {'BRK': 'BKN', 'CHO': 'CHA', 'PHO': 'PHX'}
GROUPS = ['player', 'aging', 'coach', 'franchise', 'team_season', 'lineup']


def build():
    st = pd.read_parquet(D / 'stints.parquet') if (D / 'stints.parquet').exists() else pickle.load(open(D / 'stints.pkl', 'rb'))
    g = pd.read_parquet(D / 'games.parquet')
    g = g[g.season_id.astype(str).str.startswith('2')].copy()
    g['season'] = g.season_id.astype(int) % 10000 + 1  # 22019 -> 2020 (basketball-reference season label)
    g = g.sort_values('game_date')
    co = pd.read_csv(D / 'coaches.csv'); co['team'] = co.team.replace(ABBR)
    # coach for each (team, season, k-th game of the season)
    plan = defaultdict(list)
    for r in co.itertuples():
        plan[(r.team, r.season)].append((r.coach, r.games))
    gc = {}
    cnt = defaultdict(int)
    for r in g.itertuples():
        for side, t in (('h', r.team_abbreviation_home), ('a', r.team_abbreviation_away)):
            k = (t, r.season); i = cnt[k]; cnt[k] += 1
            c, acc = None, 0
            for name, n in plan.get(k, []):
                acc += n; c = name
                if i < acc:
                    break
            gc[(r.game_id, side)] = c
    gi = g.set_index('game_id')
    st = st[st.game_id.isin(gi.index) & (st.secs > 0)].reset_index(drop=True)
    gm = gi.loc[st.game_id]
    st['season'] = gm.season.values; st['th'] = gm.team_abbreviation_home.values; st['ta'] = gm.team_abbreviation_away.values
    st['ch'] = [gc[(x, 'h')] for x in st.game_id]; st['ca'] = [gc[(x, 'a')] for x in st.game_id]
    st = st[st.ch.notna() & st.ca.notna()].reset_index(drop=True)
    st['y'] = (st.pts_home - st.pts_away) / st.secs * 2880
    return st


def design(st):
    cols = {gname: {} for gname in GROUPS}
    rows, cidx, vals = defaultdict(list), defaultdict(list), defaultdict(list)

    def add(gname, i, key, v):
        c = cols[gname].setdefault(key, len(cols[gname]))
        rows[gname].append(i); cidx[gname].append(c); vals[gname].append(v)
    for i, r in enumerate(st.itertuples(index=False)):
        for sgn, five, team, coach in ((1, r.home5, r.th, r.ch), (-1, r.away5, r.ta, r.ca)):
            for p in five:
                add('player', i, p, sgn); add('aging', i, (p, r.season), sgn)
            add('coach', i, coach, sgn); add('franchise', i, team, sgn); add('team_season', i, (team, r.season), sgn)
            add('lineup', i, (team, tuple(sorted(five))), sgn)
    n = len(st)
    X = {gname: csr_matrix((vals[gname], (rows[gname], cidx[gname])), shape=(n, len(cols[gname]))) for gname in GROUPS}
    return X, cols


def fit(X, y, w, lam, mask, groups):
    """ridge with a separate penalty per group, on rows in mask. Returns coefficients per group and intercept."""
    sw = np.sqrt(w[mask])
    blocks = [X[gname][mask] / np.sqrt(lam[gname]) for gname in groups]
    A = hstack(blocks + [csr_matrix(np.ones((mask.sum(), 1)))]).tocsr()
    A = csr_matrix(A.multiply(sw[:, None]))
    sol = lsqr(A, sw * y[mask], damp=1.0, atol=1e-8, btol=1e-8, iter_lim=3000)[0]
    out, o = {}, 0
    for gname, b in zip(groups, blocks):
        k = b.shape[1]; out[gname] = sol[o:o + k] / np.sqrt(lam[gname]); o += k
    return out, sol[-1]  # intercept is damped too, but it is small relative to the data


def contrib(X, coef, groups):
    return {gname: X[gname] @ coef[gname] for gname in groups}


def wr2(y, p, w):
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)


def main():
    st = build()
    y = st.y.values.astype(float); w = st.secs.values.astype(float)
    print(len(st), 'stints;', st.game_id.nunique(), 'games; seasons', sorted(st.season.unique()), flush=True)
    X, cols = design(st)
    print({gname: len(c) for gname, c in cols.items()}, flush=True)
    gid = st.game_id.astype(int).values
    odd = gid % 2 == 1
    # penalties: coordinate search on odd games, validate on even, starting from RAPM-like values
    lam = dict(player=3000.0, aging=20000.0, coach=20000.0, franchise=20000.0, team_season=20000.0, lineup=20000.0)
    rng = np.random.default_rng(0)
    tr = odd & (rng.random(len(y)) < 2)  # all odd games
    # inner validation: split odd games by game id parity of id//2
    inner_tr = odd & ((gid // 2) % 2 == 0); inner_va = odd & ((gid // 2) % 2 == 1)

    def val(l, groups):
        c, b = fit(X, y, w, l, inner_tr, groups)
        p = sum(contrib(X, c, groups).values()) + b
        return wr2(y[inner_va], p[inner_va], w[inner_va])
    for sweep in range(2):
        for gname in GROUPS:
            best = None
            for f in (0.1, 0.3, 1, 3, 10):
                l = dict(lam); l[gname] = lam[gname] * f
                v = val(l, GROUPS)
                if best is None or v > best[0]:
                    best = (v, l[gname])
            lam[gname] = best[1]
            print(f'sweep {sweep} {gname}: lambda {lam[gname]:.0f} inner R2 {best[0]:.5f}', flush=True)
    # nested held-out R2 (train odd, test even) adding groups one at a time
    nested = {}
    for k in range(1, len(GROUPS) + 1):
        gs = GROUPS[:k]
        c, b = fit(X, y, w, lam, odd, gs)
        p = sum(contrib(X, c, gs).values()) + b
        nested['+'.join(gs)] = float(wr2(y[~odd], p[~odd], w[~odd]))
        print('held-out R2', '+'.join(gs), f'{nested["+".join(gs)]:.5f}', flush=True)
    # split-half variance shares
    C = {}
    for name, m in (('odd', odd), ('even', ~odd)):
        c, b = fit(X, y, w, lam, m, GROUPS)
        C[name] = contrib(X, c, GROUPS)
    cov = lambda a, b: np.cov(a, b, aweights=w)[0, 1]
    tot = cov(sum(C['odd'].values()), sum(C['even'].values()))
    shares = {gname: float(cov(C['odd'][gname], C['even'][gname]) / tot) for gname in GROUPS}
    cross = {}
    for i, a in enumerate(GROUPS):
        for b_ in GROUPS[i + 1:]:
            cross[f'{a}|{b_}'] = float((cov(C['odd'][a], C['even'][b_]) + cov(C['even'][a], C['odd'][b_])) / tot)
    print('reliable variance of the full prediction (per-48 margin^2):', round(float(tot), 2))
    print('shares:', json.dumps({k: round(v, 3) for k, v in shares.items()}))
    print('covariances (x2):', json.dumps({k: round(v, 3) for k, v in cross.items() if abs(v) > 0.01}))
    # same at the team-season level (what decides a team's net rating over a season)
    keyts = st.th.values + st.season.astype(str).values
    res = dict(n_stints=len(st), lam=lam, nested_heldout_r2=nested, shares=shares, cross=cross, reliable_var=float(tot),
               n_cols={gname: len(c) for gname, c in cols.items()})
    json.dump(res, open(D / 'outcome_decomp.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
