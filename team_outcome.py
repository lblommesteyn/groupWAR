"""Team-season level: how much of a team's net rating is its roster's talent, and what is left for the coach?

Player talent for season s comes from a players-only RAPM fit on the OTHER three seasons (leave-season-out), so a team
gets no credit for its own results. For each stint, resid = margin per 48 - (home roster - away roster - home edge).
Team-season "beyond roster" = seconds-weighted mean of resid from the team's side.
  reliable variance     covariance of odd-game and even-game values across team-seasons (noise cancels)
  roster share          1 - reliable var(beyond roster) / reliable var(net rating)
  coach / franchise     mean product of beyond-roster values over team-season pairs in different seasons:
                        same coach & other franchise / same franchise & other coach / same coach & franchise
  mid-season changes    same roster, new coach: beyond-roster before vs after the change (stints of that team only)
Caveat: leave-season-out talent misses player development and injuries, which land in "beyond roster".
Output: results/decomp/team_outcome.json
"""
import json
from collections import defaultdict
from itertools import combinations
import numpy as np
from outcome_decomp import build, design, fit, contrib
from masked_paths import ROOT

D = ROOT / 'results/decomp'
LAM = 90000.0  # players-only ridge penalty (CV in outcome_decomp.py)


def main():
    st = build(); X, cols = design(st)
    y = st.y.values.astype(float); w = st.secs.values.astype(float); se = st.season.values
    resid = np.zeros(len(y))
    for s in sorted(set(se)):
        tr = se != s
        c, b = fit(X, y, w, dict(player=LAM), tr, ['player'])
        p = contrib(X, c, ['player'])['player'] + b
        resid[se == s] = (y - p)[se == s]
        print('season', s, 'done', flush=True)
    odd = st.game_id.astype(int).values % 2 == 1
    acc = defaultdict(lambda: np.zeros((3, 2, 2)))  # key -> [net|beyond][half][sum, w]
    coach_of = {}
    for i, r in enumerate(st.itertuples(index=False)):
        h = int(odd[i])
        for sgn, team, coach in ((1, r.th, r.ch), (-1, r.ta, r.ca)):
            k = (team, r.season)
            a = acc[k]; a[0, h] += [sgn * y[i] * w[i], w[i]]; a[1, h] += [sgn * resid[i] * w[i], w[i]]
            coach_of.setdefault(k, defaultdict(float))[coach] += w[i]
    keys = sorted(acc)
    val = lambda k, j, h: acc[k][j, h, 0] / acc[k][j, h, 1]
    net = np.array([[val(k, 0, h) for h in (0, 1)] for k in keys]); bey = np.array([[val(k, 1, h) for h in (0, 1)] for k in keys])
    rel_net = np.cov(net[:, 0], net[:, 1])[0, 1]; rel_bey = np.cov(bey[:, 0], bey[:, 1])[0, 1]
    print(f'team-seasons {len(keys)}; reliable sd of net rating {np.sqrt(rel_net):.2f} per 48, beyond roster {np.sqrt(max(rel_bey, 0)):.2f}')
    print(f'roster talent (from other seasons) explains {1 - rel_bey / rel_net:.0%} of reliable net-rating variance')
    # pairs across seasons; beyond-roster value is the full-season mean
    full = {k: (acc[k][1, :, 0].sum() / acc[k][1, :, 1].sum()) for k in keys}
    main_coach = {k: max(v, key=v.get) for k, v in coach_of.items()}
    m = np.mean(list(full.values()))
    pr = defaultdict(list)
    for a, b in combinations(keys, 2):
        if a[1] == b[1]:
            continue
        sc, sf = main_coach[a] == main_coach[b], a[0] == b[0]
        kind = 'coach_only' if sc and not sf else 'franchise_only' if sf and not sc else 'both' if sc and sf else 'neither'
        pr[kind].append((full[a] - m) * (full[b] - m))
    rng = np.random.default_rng(0)
    out = dict(team_seasons=len(keys), reliable_var_net=float(rel_net), reliable_var_beyond=float(rel_bey),
               roster_share=float(1 - rel_bey / rel_net), pairs={})
    for kind, v in pr.items():
        v = np.array(v); bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        out['pairs'][kind] = dict(n=len(v), cov=float(v.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                  share_of_net=float(v.mean() / rel_net))
        print(f'{kind:15s} n {len(v):5d} cov {v.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] '
              f'= {v.mean() / rel_net:+.0%} of net-rating variance')
    # mid-season coaching changes
    ev = []
    for (team, season), cs in coach_of.items():
        if len([c for c, s in cs.items() if s > 3 * 2880]) < 2:
            continue
        m_ = ((st.th == team) | (st.ta == team)).values & (se == season)
        idx = np.flatnonzero(m_)
        seq = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        order = []
        for i in idx:
            r = st.iloc[i]; side = 1 if r.th == team else -1; c = r.ch if side == 1 else r.ca
            if c not in order:
                order.append(c)
            q = seq[c]; q[0] += side * resid[i] * w[i]; q[1] += w[i]; q[2] += side * y[i] * w[i]; q[3] += w[i]
        row = dict(team=team, season=int(season), coaches=[dict(coach=c, minutes=seq[c][1] / 60 / 2, beyond=seq[c][0] / seq[c][1],
                                                                net=seq[c][2] / seq[c][3]) for c in order])
        ev.append(row)
        print(team, season, ' -> '.join(f"{c['coach']} ({c['minutes']:.0f} min: net {c['net']:+.1f}, beyond roster {c['beyond']:+.1f})" for c in row['coaches']))
    out['midseason'] = ev
    json.dump(out, open(D / 'team_outcome.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
