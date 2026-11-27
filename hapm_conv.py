"""Were ms_hapm's warm-started held-out fits converged? For the 2024 deadline, 15 movers: compare the HAPM value
from the fast fit (atol 1e-6, 400 iterations, warm start) against a tight fit (atol 1e-9, 5000 iterations, cold).
  python hapm_conv.py   (PC)"""
import numpy as np
from scipy.sparse.linalg import lsqr
import ms_spm as sp
from ms_fix import FData
from ms_hapm import design, WINDOW, LAM

Dt = FData(); n = Dt.n; cut = sp.dnum('2024-02-08')
y = Dt.luck_target(cut, 0.5); m = Dt.spm_prior(cut, y)
trn = np.flatnonzero((Dt.DAY < cut) & (Dt.DAY >= cut - WINDOW))
A, w, sw = design(Dt, trn, cut); r0 = sw * (y[trn] - Dt.X[trn] @ m)
full = lsqr(A, r0, damp=np.sqrt(LAM), atol=1e-9, btol=1e-9, iter_lim=5000)[0]
pos = np.bincount(Dt.O[trn].ravel(), np.repeat(w, 5), n)
js = np.argsort(-pos)[:15]
On, Dn = Dt.O[trn], Dt.D[trn]
out = []
for j in js:
    in_o = (On == j).any(1); in_d = (Dn == j).any(1); keep = ~(in_o | in_d)
    vals = []
    for kw in (dict(atol=1e-6, btol=1e-6, iter_lim=400, x0=full.copy()), dict(atol=1e-9, btol=1e-9, iter_lim=5000)):
        r = lsqr(A[keep], r0[keep], damp=np.sqrt(LAM), **kw); sol, it = r[0], r[2]
        th, bb = m + sol[:2 * n], sol[2 * n:] * sp.SC; th[j] = 0; th[n + j] = 0
        ro = trn[in_o]; pred = bb[0] + bb[1] * Dt.H[ro] + th[Dt.O[ro]].sum(1) + th[n + Dt.D[ro]].sum(1)
        vals.append((float(np.average(y[ro] - pred, weights=w[in_o])), int(it)))
    out.append(vals)
    print(Dt.pids[j], 'fast %+.3f (%d it)  tight %+.3f (%d it)' % (vals[0][0], vals[0][1], vals[1][0], vals[1][1]), flush=True)
d = np.array([[a[0][0], a[1][0]] for a in out])
print('max abs diff %.4f, corr %.4f, sd of values %.3f' % (np.abs(d[:, 0] - d[:, 1]).max(), np.corrcoef(d.T)[0, 1], d[:, 1].std()))
