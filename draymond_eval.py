"""BOWEN (formerly DRAYMOND) decomposition checks, from results/world/pred.npz (world_model.py, cross-fitted).

Per defender-frame: baseline spot mb (league), scheme spot ms (league + his team's scheme), actual y.
  scheme shift  = ms - mb       (what the scheme asks for beyond the league norm)
  execution gap = y - ms        (where he is versus where the scheme says he should be)
Also signed along the depth axis (distance to the rim): positive = further from the rim than expected.
Checks
  1. reliability: split-half (odd / even games) correlation of each player's mean execution gap, per season, and
     season-to-season correlation for players in both seasons;
  2. scheme belongs to teams: team-season mean scheme depth shift, correlation across seasons for the same team;
  3. movers: for players who changed teams between seasons, does his execution gap follow him (corr across
     seasons) while his scheme shift follows the team (corr with the new team's mean, vs the old team's)?
Output: results/world/draymond_eval.json, results/world/draymond_players.json   python draymond_eval.py  (PC)
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
W = ROOT / 'results/world'
RIM = np.array([41.75, 0.0])
MIN_FR = 4000   # defender-frames per player-season (~ 1,600 s)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else None


def main():
    z = np.load(W / 'pred.npz')
    poss, k = z['poss'], z['k'].astype(int)
    y, ms, mb = z['y'].astype(np.float32), z['ms'].astype(np.float32), z['mb'].astype(np.float32)
    pid = z['meta_deff'][poss, k]; season = z['meta_season'][poss]; team = z['meta_def_team'][poss]; gid = z['meta_gid'][poss]
    depth = lambda p: np.hypot(*(p - RIM).T)
    ex = np.hypot(*(y - ms).T); bl = np.hypot(*(y - mb).T); sh = np.hypot(*(ms - mb).T)
    ex_d = depth(y) - depth(ms); sh_d = depth(ms) - depth(mb)
    res = dict(frames=int(len(y)), mean_ft=dict(baseline_err=float(bl.mean()), execution_err=float(ex.mean()), scheme_shift=float(sh.mean())))
    # share of baseline error explained by scheme, by situation: distance of the defender's spot to the rim
    near = depth(ms) < 12
    res['scheme_gain_ft'] = dict(near_rim=float(bl[near].mean() - ex[near].mean()), perimeter=float(bl[~near].mean() - ex[~near].mean()))
    odd = np.array([int(g[-1]) % 2 for g in gid])
    # per player-season aggregates
    key = pid.astype(np.int64) * 10000 + season
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    agg = lambda v, m=None: np.bincount(inv, v if m is None else v * m, len(uk)) / np.maximum(np.bincount(inv, None if m is None else m, len(uk)), 1)
    E, ED, SD = agg(ex), agg(ex_d), agg(sh_d)
    Eo, Ee = agg(ex, odd), agg(ex, 1 - odd)
    EDo, EDe = agg(ex_d, odd), agg(ex_d, 1 - odd)
    ok = cnt >= MIN_FR
    ut, ti = np.unique(team, return_inverse=True)
    pair, pc = np.unique(inv.astype(np.int64) * len(ut) + ti, return_counts=True)
    best = {}
    for pr, c in zip(pair, pc):
        i_, t_ = divmod(int(pr), len(ut))
        if c > best.get(i_, (0, 0))[0]:
            best[i_] = (c, int(ut[t_]))
    P = {}
    for i in np.flatnonzero(ok):
        p, s = int(uk[i] // 10000), int(uk[i] % 10000)
        P[(p, s)] = dict(frames=int(cnt[i]), exec_ft=float(E[i]), exec_depth=float(ED[i]), scheme_depth=float(SD[i]),
                         half=(float(Eo[i]), float(Ee[i])), half_d=(float(EDo[i]), float(EDe[i])),
                         team=best[i][1])
    rel = {}
    for s in (2026, 2027):
        ps = [v for (p, s_), v in P.items() if s_ == s]
        rel[str(s)] = dict(players=len(ps), exec_ft=corr([v['half'][0] for v in ps], [v['half'][1] for v in ps]),
                           exec_depth=corr([v['half_d'][0] for v in ps], [v['half_d'][1] for v in ps]))
    both = [p for (p, s) in P if s == 2026 and (p, 2027) in P]
    movers = [p for p in both if P[(p, 2026)]['team'] != P[(p, 2027)]['team']]
    stayers = [p for p in both if P[(p, 2026)]['team'] == P[(p, 2027)]['team']]
    res['reliability_split_half'] = rel
    res['cross_season'] = dict(players=len(both),
                               exec_ft=corr([P[(p, 2026)]['exec_ft'] for p in both], [P[(p, 2027)]['exec_ft'] for p in both]),
                               exec_depth=corr([P[(p, 2026)]['exec_depth'] for p in both], [P[(p, 2027)]['exec_depth'] for p in both]))
    # team-level scheme depth (frame-weighted mean over the team's defenders), per season
    tsd = defaultdict(lambda: [0.0, 0.0])
    for (p, s), v in P.items():
        tsd[(v['team'], s)][0] += v['scheme_depth'] * v['frames']; tsd[(v['team'], s)][1] += v['frames']
    T = {k_: a / b for k_, (a, b) in tsd.items()}
    teams = sorted({t for t, s in T if (t, 2026) in T and (t, 2027) in T})
    res['team_scheme_depth_cross_season_r'] = corr([T[(t, 2026)] for t in teams], [T[(t, 2027)] for t in teams])
    # movers: execution follows player? scheme shift follows team?
    def mv_stats(group):
        return dict(n=len(group),
                    exec_depth_r=corr([P[(p, 2026)]['exec_depth'] for p in group], [P[(p, 2027)]['exec_depth'] for p in group]),
                    exec_ft_r=corr([P[(p, 2026)]['exec_ft'] for p in group], [P[(p, 2027)]['exec_ft'] for p in group]),
                    scheme_r_own_prev=corr([P[(p, 2026)]['scheme_depth'] for p in group], [P[(p, 2027)]['scheme_depth'] for p in group]),
                    scheme_r_new_team=corr([T[(P[(p, 2027)]['team'], 2027)] for p in group], [P[(p, 2027)]['scheme_depth'] for p in group]),
                    scheme_r_old_team=corr([T[(P[(p, 2026)]['team'], 2026)] for p in group], [P[(p, 2027)]['scheme_depth'] for p in group]))
    res['movers'] = mv_stats(movers); res['stayers'] = mv_stats(stayers)
    print(json.dumps(res, indent=1))
    json.dump(res, open(W / 'draymond_eval.json', 'w'), indent=1)
    json.dump({f'{p}_{s}': v for (p, s), v in P.items()}, open(W / 'draymond_players.json', 'w'))


if __name__ == '__main__':
    main()
