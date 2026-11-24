"""Point-in-time version of export_prior_v2: prior for season TARGET (argv[1]) from seasons before it only.

Export a player prior from the v3 Bayesian random-walk model (run on the PC in groupWAR/v3).

Fits on all stint seasons + 2023-24 lineup totals (as fit_case does), then for each player draws
theta(p, 2026, side) with the random-walk steps past the data and the age curve at the new age.
Output: v3/results/prior_2026.json  {players: {pid: [o_mean, o_var, d_mean, d_var, last_season, n_seasons]},
                                       rookie: [o_mean, o_var, d_mean, d_var], hyper, units}
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bayes import BayesModel, STINTS, load_stints
from data_registry import load_sport

TARGET = int(sys.argv[1])
t0 = time.time()
S = load_sport('nba')
stint_seasons = set(load_stints('nba', STINTS['nba'])['season'])
use_st = sorted(s for s in stint_seasons if s < TARGET)
lineup = [s for s in (2024, 2025) if s not in stint_seasons and s < TARGET]
curve = None
if lineup:
    m0 = BayesModel('nba', seasons=use_st)
    gs0 = m0.gram(np.ones(len(m0.y))); h0 = m0.empirical_bayes(gs0); mu0, _ = m0.posterior(gs0, h0)
    mpg = {(p, s): v for p, s, v in zip(S.ps['player_id'], S.ps['season'], S.ps['mpg'])}
    curve = m0.opponent_curve(mu0, mpg)
m = BayesModel('nba', seasons=use_st, lineup_seasons=lineup, opp_curve=curve)
gs = m.gram(np.ones(len(m.y)))
h = m.empirical_bayes(gs)
mu, L = m.posterior(gs, h)
B = m.draws(mu, L, 200, seed=1)
print('fit done', round(time.time() - t0), 's; smax', m.smax, 'hyper', {k: float(v) for k, v in h.items()}, flush=True)

rng = np.random.default_rng(11)
players = sorted(m.first)
seasons_of = {}
for (p, s, side) in m.ps_keys:
    seasons_of.setdefault(p, set()).add(s)
out = {}
for p in players:
    o = np.array([m.theta(B[:, k], p, TARGET, 'o', h, rng) for k in range(B.shape[1])])
    d = np.array([m.theta(B[:, k], p, TARGET, 'd', h, rng) for k in range(B.shape[1])])
    ss = sorted(seasons_of.get(p, []))
    out[int(p)] = [float(o.mean()), float(o.var()), float(d.mean()), float(d.var()),
                   int(ss[-1]) if ss else None, len(ss)]
# rookie baseline: first-season posterior means across players who entered after the first data season
first_vals = [(m.theta(mu, p, s0, 'o'), m.theta(mu, p, s0, 'd')) for p, s0 in m.first.items() if s0 > min(m.first.values())]
fo, fd = np.array(first_vals).T
rookie = [float(fo.mean()), float(fo.var() + h['tau_a_o'] ** 2 * 0), float(fd.mean()), float(fd.var())]
res = dict(players=out, rookie=rookie, hyper={k: float(v) for k, v in h.items()}, smax=int(m.smax),
           target=TARGET, units='points per 48 minutes on court (o: own scoring, d: opponent scoring prevented)')
Path(__file__).resolve().parent.joinpath('results', f'prior_pit_{TARGET}.json').write_text(json.dumps(res))
print('exported', len(out), 'players; rookie', np.round(rookie, 2), 'total', round(time.time() - t0), 's')
