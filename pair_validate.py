"""Do teammate-pair fit effects (player_context.py) replicate? Fit the pair ridge on 2025-26 residuals and on
2026-27 residuals separately (same additive DUNCAN fit), correlate effects for pairs with >= MIN possessions in both.
  python pair_validate.py   (PC)"""
from collections import defaultdict
from datetime import date
from itertools import combinations
import numpy as np
import ms_spm as sp
from player_context import ridge_sparse, LAM, PAIR_LAM

Dt = sp.Data(live=True); n = Dt.n; cut = sp.dnum(str(date.today()))
y = Dt.luck_target(cut, 0.5); th, b = sp.fit_arm(Dt, 'spm_lpit_h', cut, LAM)
resid = y - (b[0] + b[1] * Dt.H + th[Dt.O].sum(1) + th[n + Dt.D].sum(1))
season = np.array([Dt.G[g]['season'] for g in Dt.gids])[Dt.GI]
eff = {}
for s in (2026, 2027):
    rr = np.flatnonzero(season == s); col = {}; trip = []; tog = defaultdict(float)
    for j, i in enumerate(rr):
        for side, five in ((0, Dt.O[i]), (1, Dt.D[i])):
            for a, c in combinations(sorted(five), 2):
                trip.append((j, col.setdefault((side, a, c), len(col))))
                if side == 0:
                    tog[(a, c)] += Dt.W[i]
    coef = ridge_sparse(trip, resid[rr], Dt.W[rr], len(col), PAIR_LAM)
    e = defaultdict(lambda: [0.0, 0.0])
    for (side, a, c), k in col.items():
        e[(a, c)][side] = coef[k]
    eff[s] = {k: (v[0] - v[1], tog[k]) for k, v in e.items()}
for MIN in (300, 800):
    common = [k for k in eff[2026] if k in eff[2027] and eff[2026][k][1] >= MIN and eff[2027][k][1] >= MIN]
    a = np.array([eff[2026][k][0] for k in common]); c = np.array([eff[2027][k][0] for k in common])
    print(f'pairs with {MIN}+ possessions in both seasons: {len(common)}, cross-season r = {np.corrcoef(a, c)[0, 1]:.3f}')
