"""Uncertainty-aware roster search with the latent usage/load model, validated on real deadline trades.

At each deadline (trade_test.DEADLINES) fit the additive and usage arms on every stint before it (as trade_test.py).
Swap value of putting candidate C in player R's place on team T:
  rebuild T's current-season pre-deadline stints with C wherever R played (same opponents, same possessions),
  predict T's net points per 100 (offense minus defense) with each arm; value = swapped minus original.
  The usage arm also re-predicts usage shares and the load cost in every rebuilt lineup.
  SD from the Gaussian posterior of the player ratings (diagonal precision: prior 1/V + possessions / s2),
  scaled by R's share of T's possessions.
Validation: for every team that traded at a deadline, pair outgoing and incoming rotation players by minutes,
  predict the swap value of the actual trade, and compare with the team's realised change in net rating
  (rest of season minus pre-deadline, same season). Correlation + game-free player bootstrap CI.
Demo (last deadline): for each team, replace its weakest rotation player; top candidates by the usage arm,
  and which candidates the load term moves most relative to additive.
Output: results/latent/roster_search.json.   Usage (PC): python roster_search.py
"""
import json, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

import latent_skill as ls
import ms_latent as ml
import trade_test as tt
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
V3 = ROOT.parent / 'groupWAR/v3/data/nba'
ROT_N = 8          # rotation size for the demo
TOP = 5


class Team:
    """T's pre-deadline stints of the current season as index tensors (offense rows then defense rows)."""
    def __init__(self, rows, T, pidx):
        rs = [r for r in rows if T in r[7] and all(p in pidx for p in r[1] + r[2])]
        self.off = torch.tensor([r[7][0] == T for r in rs])
        self.O = torch.tensor([[pidx[p] for p in r[1]] for r in rs]); self.D = torch.tensor([[pidx[p] for p in r[2]] for r in rs])
        self.H = torch.tensor([r[3] for r in rs], dtype=torch.float64)
        self.w = torch.tensor([r[5] for r in rs], dtype=torch.float64); self.y = torch.tensor([r[4] for r in rs], dtype=torch.float64)
        self.n = len(rs)

    def mine(self):
        """T's player slots: offense lineup on offense rows, defense lineup on defense rows."""
        return torch.where(self.off[:, None], self.O, self.D)

    @torch.no_grad()
    def net(self, mdl, swaps=()):
        O, D = self.O.clone(), self.D.clone()
        for r, c in swaps:
            O = torch.where(self.off[:, None] & (O == r), torch.full_like(O, c), O)
            D = torch.where(~self.off[:, None] & (D == r), torch.full_like(D, c), D)
        yh = mdl(O, D, self.H)['yhat']
        w = self.w
        return float((w * yh)[self.off].sum() / w[self.off].sum() - (w * yh)[~self.off].sum() / w[~self.off].sum())

    def actual(self):
        w, y = self.w, self.y
        return float((w * y)[self.off].sum() / w[self.off].sum() - (w * y)[~self.off].sum() / w[~self.off].sum())

    def share(self, i):
        """fraction of T's offense / defense possessions with player index i on the floor"""
        m = (self.mine() == i).any(1)
        w = self.w
        return float(w[m & self.off].sum() / w[self.off].sum()), float(w[m & ~self.off].sum() / w[~self.off].sum())


@torch.no_grad()
def posterior_var(mdl, pidx, R, wm):
    n = len(pidx)
    O = torch.tensor([[pidx[p] for p in r[1]] for r in R]); D = torch.tensor([[pidx[p] for p in r[2]] for r in R])
    H = torch.tensor([r[3] for r in R], dtype=torch.float64); y = torch.tensor([r[4] for r in R], dtype=torch.float64)
    w = torch.tensor([r[5] for r in R], dtype=torch.float64) * torch.tensor(np.asarray(wm, np.float64))
    res = y - mdl(O, D, H)['yhat']
    s2 = float((w * res ** 2).sum() / len(R))
    po = torch.zeros(n, dtype=torch.float64).index_add_(0, O.reshape(-1), w.repeat_interleave(5)) / s2
    pd_ = torch.zeros(n, dtype=torch.float64).index_add_(0, D.reshape(-1), w.repeat_interleave(5)) / s2
    return (1 / (po + 1 / mdl.v[:n])).numpy(), (1 / (pd_ + 1 / mdl.v[n:])).numpy()


def swap_sd(var, team, r, c):
    vo, vd = var
    fo, fd = team.share(r)
    return float(np.sqrt(fo ** 2 * (vo[c] + vo[r]) + fd ** 2 * (vd[c] + vd[r])))


def main():
    D_ = pickle.load(open(OUT / 'ms_stints.pkl', 'rb')); S, G = D_['stints'], D_['games']
    gp = pd.read_parquet(V3 / 'games.parquet').drop_duplicates('game_id').set_index('game_id')
    gm = {g: (int(gp.loc[g, 'team_id_home']), int(gp.loc[g, 'team_id_away'])) for g in G if g in gp.index}
    abbr = {}
    for _, r in gp.iterrows():
        abbr[int(r['team_id_home'])] = r['team_abbreviation_home']; abbr[int(r['team_id_away'])] = r['team_abbreviation_away']
    pl = pd.read_parquet(V3 / 'players.parquet'); name = dict(zip(pl['person_id'].astype(int), pl['display_first_last']))
    try:
        from nba_api.stats.static import players as _np
        name.update({int(q['id']): q['full_name'] for q in _np.get_players()})
    except ImportError:
        pass
    nm = lambda p: name.get(int(p), str(p))
    res = {'validation': [], 'demo': {}}
    for season, dl in tt.DEADLINES.items():
        train = sorted((g for g in G if G[g]['date'] < dl and g in gm), key=lambda g: G[g]['date'])
        pre = [g for g in train if G[g]['season'] == season and not G[g]['playoff']]
        post = [g for g in G if G[g]['date'] >= dl and G[g]['season'] == season and not G[g]['playoff'] and g in gm]
        _, Rpre = tt.team_rows(S, G, pre, gm); _, Rpost = tt.team_rows(S, G, post, gm)
        Rtr = ml.rows(S, G, train); wm = ml.decay(Rtr, G, dl)
        fits = {arm: ls.fit(Rtr, ml.ex_for(Rtr, tt.V), u, f, l2w=ls.L2_W, wmult=wm) for arm, (u, f, _) in tt.ARMS.items() if arm != 'full'}
        pidx = fits['additive'][1]; inv = {i: p for p, i in pidx.items()}
        var = posterior_var(*fits['additive'], Rtr, wm)
        # team possessions per player before / after
        poss = {k: defaultdict(lambda: defaultdict(float)) for k in ('pre', 'post')}
        for k, RR in (('pre', Rpre), ('post', Rpost)):
            for r in RR:
                for p in r[1]:
                    poss[k][p][r[7][0]] += r[5]
        main_team = lambda k, p: max(poss[k][p].items(), key=lambda t: t[1]) if poss[k].get(p) else (None, 0.0)
        teams = sorted({t for r in Rpre for t in r[7]})
        print(f'{season} deadline {dl}: {len(teams)} teams', flush=True)
        # ---- validation on the actual trades
        for T in teams:
            outg = sorted([p for p in poss['pre'] if main_team('pre', p)[0] == T and main_team('pre', p)[1] >= tt.MIN_PRE
                           and main_team('post', p)[0] not in (T,)], key=lambda p: -main_team('pre', p)[1])
            inc = sorted([p for p in poss['post'] if main_team('post', p)[0] == T and main_team('post', p)[1] >= tt.MIN_POST
                          and main_team('pre', p)[0] not in (T, None) and main_team('pre', p)[1] >= tt.MIN_PRE and p in pidx],
                         key=lambda p: -main_team('pre', p)[1])
            outg = [p for p in outg if p in pidx]
            k = min(len(outg), len(inc))
            if k == 0:
                continue
            tm_pre = Team(Rpre, T, pidx); tm_post = Team(Rpost, T, pidx)
            sw = [(pidx[o], pidx[i]) for o, i in zip(outg[:k], inc[:k])]
            v = dict(season=season, team=abbr.get(T, T), out=[nm(p) for p in outg[:k]], inc=[nm(p) for p in inc[:k]],
                     realised=tm_post.actual() - tm_pre.actual(), post_poss=float(tm_post.w[tm_post.off].sum()))
            for arm, (mdl, _) in fits.items():
                v[f'pred_{arm}'] = tm_pre.net(mdl, sw) - tm_pre.net(mdl)
            v['sd'] = float(np.sqrt(sum(swap_sd(var, tm_pre, r, c) ** 2 for r, c in sw)))
            res['validation'].append(v)
        # ---- demo at the last deadline
        if season == max(tt.DEADLINES):
            mdl_u, mdl_a = fits['usage'][0], fits['additive'][0]
            season_poss = defaultdict(float)
            for r in Rpre:
                for p in r[1]:
                    season_poss[p] += r[5]
            pool = [pidx[p] for p, x in season_poss.items() if x >= 2 * tt.MIN_PRE and p in pidx]
            vals = {}  # team -> (replaced, {cand: (add, use, sd)})
            for T in teams:
                tm = Team(Rpre, T, pidx)
                if tm.n == 0:
                    continue
                rot = sorted({int(i) for i in tm.mine().unique()}, key=lambda i: -sum(tm.share(i)))[:ROT_N]
                worst = min(rot, key=lambda i: (mdl_a.th[i] - mdl_a.th[len(pidx) + i]).item())  # lowest additive net rating
                ba, bu = tm.net(mdl_a), tm.net(mdl_u)
                fo, fd = tm.share(worst)
                vals[T] = (worst, fo, {c: (tm.net(mdl_a, [(worst, c)]) - ba, tm.net(mdl_u, [(worst, c)]) - bu, swap_sd(var, tm, worst, c))
                                       for c in pool if c not in rot and main_team('pre', inv[c])[0] != T})
            # team-specific load effect: (usage - additive) swap value per unit of replaced minutes, minus its league mean
            per = defaultdict(list)
            for T, (_, fo, cv) in vals.items():
                for c, (a, u, _) in cv.items():
                    per[c].append((u - a) / max(fo, 1e-3))
            mean_load = {c: float(np.mean(v)) for c, v in per.items()}
            gen = {c: float(np.mean([vals[T][2][c][0] / max(vals[T][1], 1e-3) for T in vals if c in vals[T][2]])) for c in per}
            order = sorted(gen, key=lambda c: -gen[c])
            role = set(order[30:200])  # skip the top 30 (stars any team would take); role-player tier
            for T, (worst, fo, cv) in vals.items():
                rows = []
                for c, (a, u, sd) in cv.items():
                    rows.append(dict(player=nm(inv[c]), add=a, use=u, sd=sd, team_fit=(u - a) / max(fo, 1e-3) - mean_load[c],
                                     role=c in role, generic_rank=order.index(c) + 1))
                top = sorted([r for r in rows if r['role']], key=lambda r: -r['use'])[:TOP]
                fitr = sorted([r for r in rows if r['role']], key=lambda r: -r['team_fit'])
                res['demo'][abbr.get(T, str(T))] = dict(replace=nm(inv[worst]), replace_share=fo, top_role=top,
                                                        best_fit=fitr[:3], worst_fit=fitr[-3:])
    V = res['validation']
    y = np.array([v['realised'] for v in V])
    print(f'\nvalidation: {len(V)} team-deadline trades')
    res['summary'] = {}
    rng = np.random.default_rng(0); B = rng.integers(0, len(V), (2000, len(V)))
    for arm in ('additive', 'usage'):
        x = np.array([v[f'pred_{arm}'] for v in V])
        r = float(np.corrcoef(x, y)[0, 1]); rb = [np.corrcoef(x[b], y[b])[0, 1] for b in B]
        slope = float(np.polyfit(x, y, 1)[0])
        res['summary'][arm] = dict(corr=r, corr_ci=[float(q) for q in np.percentile(rb, [2.5, 97.5])], slope=slope,
                                   sd_pred=float(x.std()), sd_realised=float(y.std()))
        print(f'  {arm:9s} corr(pred, realised) {r:.3f} CI {np.round(np.percentile(rb, [2.5, 97.5]), 3)}  slope {slope:.2f}  sd pred {x.std():.2f} realised {y.std():.2f}')
    xs = np.array([v['pred_usage'] - v['pred_additive'] for v in V])
    print(f'  load adjustment (usage - additive): mean {xs.mean():+.2f} sd {xs.std():.2f}; corr with realised residual '
          f'{np.corrcoef(xs, y - np.array([v["pred_additive"] for v in V]))[0, 1]:.3f}')
    big = sorted(V, key=lambda v: -abs(v['pred_usage']))[:10]
    print('\nlargest predicted trades:')
    for v in big:
        print(f"  {v['season']} {v['team']}: in {', '.join(v['inc'])} / out {', '.join(v['out'])}  pred {v['pred_usage']:+.1f} ± {v['sd']:.1f}  "
              f"(additive {v['pred_additive']:+.1f})  realised {v['realised']:+.1f}")
    print(f'\ndemo at {tt.DEADLINES[max(tt.DEADLINES)]}:')
    for T in sorted(res['demo']):
        d = res['demo'][T]
        print(f"  {T} replace {d['replace']} ({d['replace_share']:.0%} of poss). best role players: "
              + '; '.join(f"{c['player']} {c['use']:+.1f}±{c['sd']:.1f}" for c in d['top_role']))
        print(f"      fits {T} better than average (load): " + ', '.join(f"{c['player']} {c['team_fit']:+.2f}" for c in d['best_fit'])
              + "  | worse: " + ', '.join(f"{c['player']} {c['team_fit']:+.2f}" for c in d['worst_fit']))
    json.dump(res, open(OUT / 'roster_search.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
