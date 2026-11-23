"""Player vs coach vs franchise vs roster context: variance decomposition of player style stats, 2018-19..2024-25.

Data: v3 player_seasons (one team per player-season, from the PC: results/decomp/player_seasons.parquet) and head
coaches by team-season (results/decomp/coaches.csv, basketball-reference; the majority coach is the season's coach).
Player-seasons with >= 500 minutes; each stat has its season mean removed (league trends) and is standardised.

Model  x[p,s] = a[p] + coach[c] + franchise[f] + u[team-season] + e[p,s]
  a   the player (persistent across seasons and teams)
  coach, franchise   persistent scheme effects; identified because coaches move and franchises change coaches
  u   this roster in this season beyond its coach and franchise (teammates, role, fit)
  e   noise, the player's own drift, his role on this particular team
Estimator (pairwise covariances, so noise never enters): with leave-one-out estimates
  r_team[p,s] = x - a_hat[p] (a_hat from p's OTHER seasons),  r_play[p,s] = x - team_hat (from p's teammates only)
  player    mean r_play[p,s] r_play[p,s'] over same-player pairs, s != s'
  team-season total   mean r_team r_team over teammate pairs (same team-season, different players)
  coach     same coach, different franchise, different player, different season
  franchise same franchise, different coach, different player, different season
  coach+franchise    same coach and franchise, different player, different season
  context u = team-season total - coach+franchise
  sorting   2 mean a_hat * team_hat (do players with high X land on teams that push X?)
Shares are of total variance (1 after standardising); what is left is noise + drift + player-specific role.
Uncertainty: jackknife over franchises.
Output: results/decomp/season_decomp.json
  python season_decomp.py
"""
import json
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from masked_paths import ROOT

D = ROOT / 'results/decomp'
MIN_MIN = 500
ABBR = {'BRK': 'BKN', 'CHO': 'CHA', 'PHO': 'PHX'}
STATS = ['mpg', 'usage', 'ts', 'pts36', 'rim_share', 'smid_share', 'lmid_share', 'c3_share', 'a3_share', 'rim_acc',
         'fg3_acc', 'fta_rate', 'ast36', 'ast3_share', 'astrim_share', 'oreb36', 'dreb36', 'stl36', 'blk36', 'tov36', 'pf36']


def load():
    ps = pd.read_parquet(D / 'player_seasons.parquet')
    ps = ps[ps.Minutes >= MIN_MIN].copy()
    co = pd.read_csv(D / 'coaches.csv'); co['team'] = co.team.replace(ABBR)
    co = co.sort_values('games').groupby(['season', 'team']).tail(1).set_index(['season', 'team']).coach
    ps['coach'] = [co.get((s, t)) for s, t in zip(ps.season, ps.team)]
    assert ps.coach.notna().all(), ps[ps.coach.isna()][['season', 'team']].drop_duplicates()
    ps['ts_key'] = ps.team + ps.season.astype(str)
    return ps.reset_index(drop=True)


def components(df, x):
    """x: standardised stat aligned with df rows. Returns the variance components."""
    n = len(df); pid = df.player_id.values; ts = df.ts_key.values
    by_ts = defaultdict(list); by_p = defaultdict(list)
    for i in range(n):
        by_ts[ts[i]].append(i); by_p[pid[i]].append(i)
    a_hat = np.zeros(n); t_hat = np.zeros(n)
    for _ in range(4):  # alternate leave-one-out player and team-season estimates
        resid_t = x - a_hat
        for g in by_ts.values():
            s = resid_t[g].sum()
            for i in g:
                t_hat[i] = (s - resid_t[i]) / (len(g) - 1) if len(g) > 1 else 0.0
        resid_p = x - t_hat
        new = np.zeros(n)
        for g in by_p.values():
            s = resid_p[g].sum()
            for i in g:
                new[i] = (s - resid_p[i]) / (len(g) - 1) if len(g) > 1 else 0.0
        a_hat = new
    r_team = x - a_hat; r_play = x - t_hat
    has_hist = np.array([len(by_p[p]) > 1 for p in pid])
    out = {}
    out['player'] = np.mean([r_play[i] * r_play[j] for g in by_p.values() for i, j in combinations(g, 2)])
    tm = [(i, j) for g in by_ts.values() for i, j in combinations(g, 2) if has_hist[i] and has_hist[j]]
    out['team_season'] = np.mean([r_team[i] * r_team[j] for i, j in tm])
    # cross-season pairs of different players, grouped by coach / franchise relation (subsample for speed)
    rng = np.random.default_rng(0)
    co = df.coach.values; fr = df.team.values; se = df.season.values
    idx = np.flatnonzero(has_hist)
    acc = defaultdict(list)
    by_co = defaultdict(list); by_fr = defaultdict(list)
    for i in idx:
        by_co[co[i]].append(i); by_fr[fr[i]].append(i)
    for groups, key in ((by_co, 'coach'), (by_fr, 'franchise')):
        for g in groups.values():
            g = np.array(g)
            for i, j in combinations(g, 2):
                if se[i] == se[j] or pid[i] == pid[j]:
                    continue
                same_c, same_f = co[i] == co[j], fr[i] == fr[j]
                if key == 'coach' and same_c and not same_f:
                    acc['coach'].append(r_team[i] * r_team[j])
                elif key == 'franchise' and same_f and not same_c:
                    acc['franchise'].append(r_team[i] * r_team[j])
                elif key == 'franchise' and same_f and same_c:
                    acc['coach_franchise'].append(r_team[i] * r_team[j])
    for k in ('coach', 'franchise', 'coach_franchise'):
        out[k] = float(np.mean(acc[k])) if acc[k] else np.nan
        out['n_' + k] = len(acc[k])
    out['context'] = out['team_season'] - out['coach_franchise']
    out['sorting'] = 2 * float(np.mean((a_hat * t_hat)[has_hist]))
    out['residual'] = 1 - out['player'] - out['team_season'] - out['sorting']
    out['_a_hat'] = a_hat
    return out


def team_level(df, x, a_hat):
    """Team style = minute-weighted mean over the roster. Roster prediction = the same mean of what those players did
    in their OTHER seasons (a_hat). Remaining team residual is split into coach / franchise by cross-season pairs."""
    has = np.array([a_hat[i] != 0 for i in range(len(df))])
    w = df.Minutes.values.astype(float)
    T = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # sum w x, sum w, sum w a_hat (players with history), sum w (with history)
    for i, k in enumerate(df.ts_key.values):
        t = T[k]; t[0] += w[i] * x[i]; t[1] += w[i]
        if has[i]:
            t[2] += w[i] * a_hat[i]; t[3] += w[i]
    meta = df.drop_duplicates('ts_key').set_index('ts_key')[['team', 'coach', 'season']]
    keys = [k for k in T if T[k][3] > 0.6 * T[k][1]]
    X = np.array([T[k][0] / T[k][1] for k in keys]); R = np.array([T[k][2] / T[k][3] for k in keys])
    X = X - X.mean(); R = R - R.mean()
    b = np.cov(X, R)[0, 1] / np.var(R, ddof=1)
    roster = np.corrcoef(X, R)[0, 1] ** 2
    e = X - b * R; vX = X.var()
    co = meta.loc[keys, 'coach'].values; fr = meta.loc[keys, 'team'].values; se = meta.loc[keys, 'season'].values
    acc = defaultdict(list)
    for i, j in combinations(range(len(keys)), 2):
        if se[i] == se[j]:
            continue
        sc, sf = co[i] == co[j], fr[i] == fr[j]
        if sc and not sf:
            acc['coach'].append(e[i] * e[j])
        elif sf and not sc:
            acc['franchise'].append(e[i] * e[j])
        elif sf and sc:
            acc['coach_franchise'].append(e[i] * e[j])
    out = dict(roster=float(roster), n_team_seasons=len(keys))
    for k in ('coach', 'franchise', 'coach_franchise'):
        out[k] = float(np.mean(acc[k]) / vX) if acc[k] else np.nan
        out['n_' + k] = len(acc[k])
    out['other'] = 1 - out['roster'] - out['coach_franchise']
    return out


def main():
    ps = load()
    print(len(ps), 'player-seasons;', ps.player_id.nunique(), 'players;', ps.coach.nunique(), 'coaches', flush=True)
    mv = ps.groupby('coach').team.nunique(); print('coaches with 2+ franchises:', ', '.join(mv[mv > 1].index))
    res = {}
    keys = ['player', 'team_season', 'coach', 'franchise', 'coach_franchise', 'context', 'sorting', 'residual']
    for st in STATS:
        x = ps[st].astype(float)
        x = (x - ps.groupby('season')[st].transform('mean')).values
        x = x / x.std()
        full = components(ps, x)
        team = team_level(ps, x, full.pop('_a_hat'))
        jk = []
        for f in sorted(ps.team.unique()):
            m = (ps.team != f).values
            xs = x[m]
            sub = ps[m].reset_index(drop=True); xs = (xs - xs.mean()) / xs.std()
            c = components(sub, xs); c.update({'T_' + k: v for k, v in team_level(sub, xs, c.pop('_a_hat')).items()})
            jk.append(c)
        nj = len(jk)
        se = {k: float(np.sqrt((nj - 1) / nj * np.sum((np.array([j[k] for j in jk]) - np.mean([j[k] for j in jk])) ** 2))) for k in keys}
        full.update({'T_' + k: v for k, v in team.items()})
        keys2 = keys + ['T_roster', 'T_coach', 'T_franchise', 'T_coach_franchise', 'T_other']
        se = se | {k: float(np.sqrt((nj - 1) / nj * np.nansum((np.array([j[k] for j in jk]) - np.nanmean([j[k] for j in jk])) ** 2))) for k in keys2[len(keys):]}
        res[st] = {k: float(full[k]) for k in keys2} | {k + '_se': se[k] for k in keys2} | {k: full[k] for k in full if k.startswith('n_')}
        r = res[st]
        print(f"{st:13s} player {r['player']:.2f}±{r['player_se']:.2f}  team-season {r['team_season']:.2f}±{r['team_season_se']:.2f} "
              f"[coach {r['coach']:.2f}±{r['coach_se']:.2f} franchise {r['franchise']:.2f}±{r['franchise_se']:.2f} "
              f"coach+fr {r['coach_franchise']:.2f} context {r['context']:.2f}±{r['context_se']:.2f}] sorting {r['sorting']:.2f} "
              f"resid {r['residual']:.2f}", flush=True)
        print(f"{'':13s} TEAM style: roster {r['T_roster']:.2f}±{r['T_roster_se']:.2f} coach {r['T_coach']:.2f}±{r['T_coach_se']:.2f} "
              f"franchise {r['T_franchise']:.2f}±{r['T_franchise_se']:.2f} coach+fr {r['T_coach_franchise']:.2f}±{r['T_coach_franchise_se']:.2f} "
              f"other {r['T_other']:.2f}", flush=True)
    json.dump(res, open(D / 'season_decomp.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
