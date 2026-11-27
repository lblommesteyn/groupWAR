"""DUNCAN and IGGY value outputs, defined separately, with ranges. Does not touch the live forecasting model.

Common basis (all outputs): DUNCAN-2 ratings from the live run (live_ratings_{date}.npz: offense o, defense d per 100
possessions, d flipped so + = better, centred on the season's possession-weighted average player), points per win
from team-seasons, replacement level as in duncan_score.py.

1 ABILITY        net = o + d per 100 possessions vs an average player. Uncertainty: game-cluster bootstrap of the ridge
                 (B refits with Poisson(1) game weights, same lambda / prior / luck target): sd of each rating.
2 SEASON-TO-DATE ability x possessions played this season: (net - repl) x poss / 100 / ppw. This is "what his
                 current ability estimate is worth over the minutes he has played", NOT a measure of what happened.
3 PROJECTED ROS  (net - repl) x E[possessions left] / 100 / ppw,
                 E[poss left] = availability x minutes-when-available x team games left x possessions per minute;
                 availability = games played / team games over his last 82 team games (shrunk to the league mean,
                 30 pseudo-games); minutes-when-available = mean over his last 10 games played.
4 IGGY roster    for player i on team T, per game, in points then wins:
  (a) fixed-minutes substitution: his minutes are played by a replacement-level player, or by a named teammate j:
      (net_i - net_j) x poss_i / 100 / ppw
  (b) roster change with redistribution: the IGGY minutes baseline (selected rotation model, results/iggy) predicts
      T's rotation with i available and with i out; value = sum_k net_k (poss_k with i - poss_k without i) / 100 / ppw
  depth/redistribution effect = (b) - (a, replacement level). It comes only from WHO absorbs the minutes and how good
      they are; it is not "fit" (no synergy terms are used; synergy stays experimental).
Ranges: rating uncertainty (+-1 bootstrap sd), replacement level (-1.0 / -2.5), availability (+-0.1), minutes (+-20%),
points per win (bootstrap over team-seasons). Unmodelled: injury risk beyond the availability rate, trades, rating
drift over the season, coach changes, the minutes model's own error (reported as its MAE), playoff value.
Output: results/value/players.json, results/value/iggy_examples.json, results/value/audit.json
  python value_outputs.py [--date YYYY-MM-DD] [--boot 20]   (PC)
"""
import argparse, json, pickle
from collections import defaultdict
from datetime import date
import numpy as np
import pandas as pd

import ms_spm as sp
from duncan_score import points_per_win, team_ids
from ms_fix import FData
from masked_paths import ROOT
from provenance import provenance

LAT = ROOT / 'results/latent'
OUT = ROOT / 'results/value'
SEASON = 2027


def boot_sd(cut_s, B):
    """Game-cluster bootstrap of the DUNCAN-2 ridge (before the mover pull): per-player sd of o and d."""
    Dt = FData(); cut = sp.dnum(cut_s); y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
    rows = np.flatnonzero(Dt.DAY < cut); rng = np.random.default_rng(0)
    W0 = Dt.W.copy(); TH = []
    for b in range(B):
        gw = rng.poisson(1.0, len(Dt.gids)).astype(float)
        Dt.W = W0 * gw[Dt.GI]
        th, _ = Dt.ridge(rows, y, m, 3000.0, cut); TH.append(th)
        print('boot', b, flush=True)
    Dt.W = W0
    TH = np.array(TH); n = Dt.n
    return {int(p): (float(TH[:, i].std()), float(TH[:, n + i].std())) for i, p in enumerate(Dt.pids)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--date', default=str(date.today())); ap.add_argument('--boot', type=int, default=20)
    a = ap.parse_args()
    sc = json.load(open(ROOT / 'results/duncan' / f'score_{a.date}.json'))
    repl, ppw = sc['replacement'], sc['points_per_win']
    # points-per-win uncertainty: bootstrap team-seasons
    G = {}
    for f in ('ms2_stints.pkl', 'ms2_stints_2325.pkl', 'ms2_stints_2526.pkl'):
        G.update(pickle.load(open(LAT / f, 'rb'))['games'])
    TM = team_ids(); acc = defaultdict(lambda: [0, 0, 0.0])
    for g, m_ in G.items():
        if m_['playoff'] or g not in TM or not m_.get('final'):
            continue
        h, aw = TM[g]; sh, sa = m_['final']
        for t, d_ in ((h, sh - sa), (aw, sa - sh)):
            r = acc[(t, m_['season'])]; r[0] += 1; r[1] += d_ > 0; r[2] += d_
    v = np.array([(w / n, d_ / n) for n, w, d_ in acc.values() if n >= 50]); rng = np.random.default_rng(1)
    ppw_bs = [1 / np.polyfit(v[i, 1], v[i, 0], 1)[0] for i in (rng.integers(0, len(v), len(v)) for _ in range(2000))]
    ppw_ci = np.percentile(ppw_bs, [2.5, 97.5]).round(1).tolist()
    cache = OUT / f'boot_sd_{a.date}_{a.boot}.json'
    if cache.exists():
        SD = {int(k): tuple(v) for k, v in json.load(open(cache)).items()}
    else:
        SD = boot_sd(a.date, a.boot); OUT.mkdir(parents=True, exist_ok=True); json.dump({str(k): v for k, v in SD.items()}, open(cache, 'w'))
    gl = pd.read_parquet(LAT / 'gamelogs.parquet'); gl = gl[~gl.playoff].copy(); gl['d'] = pd.to_datetime(gl.game_date)
    gl = gl[gl.d < pd.Timestamp(a.date)].sort_values('d')
    # availability: games played / team games over his last 82 team games (team = his current team)
    tg = gl.drop_duplicates(['game_id', 'team_id']).sort_values('d').groupby('team_id').game_id.apply(list).to_dict()
    lg_av = 0.85
    players = []
    for r in sc['players']:
        p = r['pid']; h = gl[gl.player_id == p]
        if h.empty:
            continue
        team = int(h.team_id.iloc[-1]); last82 = tg.get(team, [])[-82:]
        played = h.game_id.isin(last82).sum()
        avail = (played + 30 * lg_av) / (len(last82) + 30)
        mpg = float(h['min'].tail(10).mean())
        team_left = max(82 - int(gl[(gl.team_id == team) & (gl.season == SEASON)].game_id.nunique()), 0)
        so, sd_ = SD.get(p, (np.nan, np.nan)); sd_net = float(np.hypot(so, sd_))
        ppm = r['poss'] / max(h[h.season == SEASON]['min'].sum(), 1) if r['poss'] else 2.05
        e_poss = avail * mpg * team_left * ppm      # rescaled to 240 team minutes per game below
        war = lambda net, rp, poss, pw=ppw: (net - rp) * poss / 100 / pw
        net = r['duncan']
        row = dict(pid=p, name=r['name'], team=r['team'],
                   ability=dict(o=r['o'], d=r['d'], net=net, net_sd=round(sd_net, 2)),
                   season_to_date=dict(possessions=r['poss'], war=round(war(net, repl, r['poss']), 2),
                                       range=[round(war(net - sd_net, -1.0, r['poss']), 2), round(war(net + sd_net, -2.5, r['poss']), 2)]),
                   projection=dict(availability=round(avail, 3), minutes_when_available=round(mpg, 1), team_games_left=team_left,
                                   expected_possessions=round(e_poss), war_ros=round(war(net, repl, e_poss), 2),
                                   ranges={'rating_+-1sd': [round(war(net - sd_net, repl, e_poss), 2), round(war(net + sd_net, repl, e_poss), 2)],
                                           'replacement_-1.0_to_-2.5': [round(war(net, -1.0, e_poss), 2), round(war(net, -2.5, e_poss), 2)],
                                           'availability_+-0.1': [round(war(net, repl, e_poss * max(avail - 0.1, 0) / avail), 2), round(war(net, repl, e_poss * min(avail + 0.1, 1) / avail), 2)],
                                           'minutes_+-20pct': [round(war(net, repl, e_poss * 0.8), 2), round(war(net, repl, e_poss * 1.2), 2)],
                                           'points_per_win_ci': [round(war(net, repl, e_poss, ppw_ci[1]), 2), round(war(net, repl, e_poss, ppw_ci[0]), 2)]},
                                   old_war_ros_no_availability=r['war_ros']))
        players.append(row)
    # team minutes must add to 240 per game: missed games' minutes are played by teammates (IGGY B0 rule: rescale
    # expected minutes within the team, capped as in iggy_minutes.project). Recompute projections with the rescaled minutes.
    from iggy_minutes import project
    byteam = defaultdict(list)
    for x in players:
        byteam[x['team']].append(x)
    for tm, xs in byteam.items():
        em = np.array([x['projection']['availability'] * x['projection']['minutes_when_available'] for x in xs])
        if em.sum() <= 0:
            continue
        cap = np.minimum(44.0, np.maximum(np.array([x['projection']['minutes_when_available'] for x in xs]), 10.0) + 14.0)
        m240 = project(em * 240.0 / em.sum(), cap) if em.sum() < 240 else project(em, cap)
        for x, mm, e0 in zip(xs, m240, em):
            pj = x['projection']; f = mm / max(e0, 1e-9)
            pj['expected_minutes_per_team_game'] = round(float(mm), 2)
            pj['team_minutes_rescale'] = round(float(f), 3)
            for k in ('expected_possessions',):
                pj[k] = round(pj[k] * f)
            pj['war_ros'] = round(pj['war_ros'] * f, 2)
            pj['ranges'] = {k: [round(v[0] * f, 2), round(v[1] * f, 2)] for k, v in pj['ranges'].items()}
    players.sort(key=lambda x: -(x['season_to_date']['war'] + x['projection']['war_ros']))
    # verified calculations: recompute by hand for every player; accounting identities
    for x in players:
        net = x['ability']['net']; std = (net - repl) * x['season_to_date']['possessions'] / 100 / ppw
        assert abs(std - x['season_to_date']['war']) < 0.006, (x['name'], std, x['season_to_date']['war'])
        pj = x['projection']; ros = (net - repl) * pj['expected_possessions'] / 100 / ppw
        assert abs(ros - pj['war_ros']) < 0.05 + 0.01 * abs(ros), (x['name'], ros, pj['war_ros'])
    team_min = {tm: sum(x['projection']['expected_minutes_per_team_game'] for x in xs) for tm, xs in byteam.items()}
    tot_old = sum(r['war_82'] for r in sc['players']); tot_new = sum(x['season_to_date']['war'] + x['projection']['war_ros'] for x in players)
    std_tot = sum(x['season_to_date']['war'] for x in players)
    audit = dict(date=a.date, replacement=repl, points_per_win=ppw, points_per_win_ci=ppw_ci,
                 units='ratings per 100 possessions; possessions = mean of offensive and defensive possessions on court; '
                       'wins = points / points_per_win (+1 point per game over 82 games = 82 / ppw wins)',
                 league_total_war_82=dict(old_no_availability=round(tot_old), new_with_availability=round(tot_new),
                                          implied_by_replacement=round(30 * (41 - 82 * (0.5 + 5 * repl * 0.99 / ppw)), 1),
                                          note='replacement team: five replacement players, ~99 possessions per game'),
                 bootstrap_refits=a.boot,
                 checks=dict(per_player_recompute='passed (season-to-date and projection recomputed from definitions)',
                             team_minutes_per_game=dict(min=round(min(team_min.values()), 1), max=round(max(team_min.values()), 1)),
                             season_to_date_total=round(std_tot, 1),
                             note='league total over a full season should be close to implied_by_replacement; residual gaps come from '
                                  'players without a current rating, rookies yet to play, and the ROS centring (ratings are centred on '
                                  'season-to-date possessions)'),
                 unmodelled=['injury risk beyond the availability rate', 'trades and role changes', 'rating drift during the season',
                             'coach changes', 'minutes-model error (see results/iggy/minutes.json MAE)', 'playoff value',
                             'rating covariance between teammates (bootstrap sd is per player)'])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(audit=audit, players=players), open(OUT / 'players.json', 'w'), indent=1)
    json.dump(dict(audit, provenance=provenance(__file__, [f'results/duncan/score_{a.date}.json', f'results/latent/live_ratings_{a.date}.npz',
                                                            'results/latent/gamelogs.parquet'])), open(OUT / 'audit.json', 'w'), indent=1)
    iggy_examples(a.date, players, repl, ppw)
    print(json.dumps(audit, indent=1)); print([(x['name'], x['ability'], x['projection']['war_ros']) for x in players[:5]])


def iggy_examples(day, players, repl, ppw):
    """(a) fixed-minutes substitution vs (b) roster change with redistribution, ONE minutes model for both scenarios:
    B2 (iggy_minutes.py; statistically equal to the rule-selected B3, and needs no role vectors at scenario time),
    retrained exactly as there. Scenario: each team's candidates at its latest covered game, with that game's absences
    still out; 'with i' vs 'i also out' (absent minutes, count and absence-history features updated)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from iggy_minutes import project, FEATS
    D = pd.read_parquet(ROOT / 'results/iggy/minutes_predictions.parquet')
    tr = (D.season <= 2024) & ~D.absent
    mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(D.loc[tr, FEATS['B2']].values, D.loc[tr, 'y'].values)
    info = {x['pid']: x for x in players}
    net = lambda p: info[p]['ability']['net'] if p in info else repl
    cur = D[D.season == SEASON]
    out = []
    for team, g0 in cur.groupby('team'):
        g = g0[g0.gid == g0.sort_values('d').gid.iloc[-1]]
        av = g[~g.absent].copy()
        if len(av) < 8:
            continue
        hist = D[(D.team == team) & (D.d < g.d.iloc[0])]
        def predict(frame):
            cap = np.minimum(44.0, np.maximum(frame.base.values, 10.0) + 14.0)
            return project(mdl.predict(frame[FEATS['B2']].values), cap)
        with_i = predict(av)
        order = np.argsort(-with_i)
        for idx in list(order[:2]) + ([order[6]] if len(order) > 6 else []):
            i = int(av.pid.values[idx]); rest = av.drop(av.index[idx]).copy()
            rest['abs_min'] = rest.abs_min + av.base.values[idx]; rest['n_abs'] = rest.n_abs + 1
            past = hist[(hist.pid == i) & hist.absent].gid.unique()       # earlier games where i was out
            if len(past):
                resp = hist[hist.gid.isin(past) & ~hist.absent & hist.played].assign(dl=lambda x: x.y - x.base).groupby('pid').dl.mean()
                r_ = rest.pid.map(resp)
                old = np.where(rest.has_resp, rest.resp0, np.nan)
                comb = np.where(np.isnan(old), r_.values, np.where(np.isnan(r_.values), old, (old + np.nan_to_num(r_.values)) / 2))
                rest['has_resp'] = ~np.isnan(comb); rest['resp0'] = np.nan_to_num(comb)
            without = predict(rest)
            pr = rest.pid.values.astype(int)
            v_with = sum(net(int(p)) * m for p, m in zip(av.pid.values.astype(int), with_i)) * 2.05 / 100
            v_without = sum(net(int(p)) * m for p, m in zip(pr, without)) * 2.05 / 100
            b_pts = v_with - v_without
            a_pts = (net(i) - repl) * with_i[idx] * 2.05 / 100
            gain = without - np.delete(with_i, idx); j = int(pr[np.argmax(gain)])
            a_named = (net(i) - net(j)) * with_i[idx] * 2.05 / 100
            out.append(dict(team=int(team), as_of_game=str(g.gid.iloc[0]), pid=i, name=info.get(i, {}).get('name', str(i)), net=net(i),
                            minutes_with=round(float(with_i[idx]), 1), past_games_without_him=int(len(past)),
                            per_game_points=dict(a_fixed_minutes_vs_replacement=round(a_pts, 2),
                                                 a_fixed_minutes_vs_top_absorber=round(a_named, 2),
                                                 top_absorber=dict(pid=j, name=info.get(j, {}).get('name', str(j)), net=net(j), extra_minutes=round(float(gain.max()), 1)),
                                                 b_roster_change_redistributed=round(b_pts, 2),
                                                 depth_effect_b_minus_a=round(b_pts - a_pts, 2)),
                            per_82_games_wins=dict(a=round(82 * a_pts / ppw, 2), b=round(82 * b_pts / ppw, 2)),
                            redistribution={info.get(int(p), {}).get('name', str(int(p))): round(float(d_), 1) for p, d_ in zip(pr, gain) if abs(d_) >= 1}))
    json.dump(dict(model='B2 minutes model (iggy_minutes.py), same model for both scenarios; 2.05 possessions per minute',
                   note='(a) fixed-minutes substitution; (b) roster change with the minutes redistributed by the same minutes '
                        'model; b - a = depth/redistribution effect (who absorbs the minutes and how good they are). No synergy '
                        'terms, so this is not "fit".', examples=out), open(OUT / 'iggy_examples.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
