"""Noise controls for the append test: per-game columns with the same shape as gravity / coverage but no signal.

noise1_pg.pkl  one offensive column (nz, nz_n): every (game, player) gravity record's rate is replaced by a rate
               drawn from the pooled gravity rates of all records (a global shuffle), times that player's own
               possession count, so the shrinkage and scale match gravity exactly.
noise3_pg.pkl  three defensive columns (nz_drop, nz_switch, nz_up, nz3_n): coverage records shuffled globally across
               (game, player) slots, so counts and rates have coverage's distribution but belong to someone else.
  python make_noise.py
"""
import pickle
import numpy as np
from masked_paths import ROOT

E = ROOT / 'results/epv'


def main():
    rng = np.random.default_rng(12345)
    G = pickle.load(open(E / 'gravity_pg.pkl', 'rb'))
    slots = [(g, p) for g, c in G.items() for p in c]
    rate = np.array([G[g][p]['grav'] / max(G[g][p]['grav_n'], 1) for g, p in slots]); rng.shuffle(rate)
    out = {}
    for (g, p), r in zip(slots, rate):
        n = G[g][p]['grav_n']; out.setdefault(g, {})[p] = dict(nz=float(r * n), nz_n=float(n))
    pickle.dump(out, open(E / 'noise1_pg.pkl', 'wb'))
    C = pickle.load(open(E / 'coverage_pg.pkl', 'rb'))
    slots = [(g, p) for g, c in C.items() for p in c]
    vals = [C[g][p] for g, p in slots]; perm = rng.permutation(len(vals))
    out = {}
    for (g, p), j in zip(slots, perm):
        v = vals[j]
        out.setdefault(g, {})[p] = dict(nz_drop=v['pnr_drop'], nz_switch=v['pnr_switch'], nz_up=v['pnr_up'], nz3_n=v['pnr_n'])
    pickle.dump(out, open(E / 'noise3_pg.pkl', 'wb'))
    print('noise1 slots', len(rate), 'noise3 slots', len(vals))


if __name__ == '__main__':
    main()
