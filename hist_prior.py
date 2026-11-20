"""Multi-season prior for the GroupWAR-T rating backbone.

prior_2026.json comes from groupWAR v3 bayes.py (2019-20..2023-24, random walk projected to 2025-26,
units points/48 min; treated as points/100 possessions, pace ~100). In our RAPM design the offense
block is +own scoring and the defense block is +opponent scoring, so the defense prior is -d.

prior_rapm: minimise sum w (y - X theta)^2 + sum_i (s2 / v_i) (theta_i - m_i)^2, with s2 the
row noise scale estimated from the data and (m_i, v_i) from history (rookie baseline if unseen).
"""
import json
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import lsqr
import embed_eval as ee
from masked_paths import ROOT

import os
PRIOR = json.load(open(ROOT / 'results/groupwar_t' / os.environ.get('PRIOR_FILE', 'prior_2026.json')))
TAU_A = (PRIOR['hyper']['tau_a_o'], PRIOR['hyper']['tau_a_d'])


def prior_vectors(pids, inflate=1.0):
    n = len(pids)
    m = np.zeros(2 * n); v = np.zeros(2 * n); known = np.zeros(n, bool)
    ro, rov, rd, rdv = PRIOR['rookie']
    for i, p in enumerate(pids):
        e = PRIOR['players'].get(str(p))
        if e is not None:
            m[i], v[i], m[n + i], v[n + i] = e[0], e[1] * inflate, -e[2], e[3] * inflate
            known[i] = True
        else:
            m[i], v[i], m[n + i], v[n + i] = ro, rov + TAU_A[0] ** 2, -rd, rdv + TAU_A[1] ** 2
    return m, np.maximum(v, 1e-3), known


def prior_rapm(R, pidx, m, v, offset=None):
    y = np.array([r[4] for r in R]); w = np.array([r[5] for r in R]); hm = np.array([r[3] for r in R], float)
    if offset is not None:
        y = y - offset
    X = ee.Lin().ind(R, pidx)
    y0 = y - X @ m
    C = np.column_stack([np.ones(len(R)), hm])
    b0 = np.linalg.lstsq(C * np.sqrt(w)[:, None], y0 * np.sqrt(w), rcond=None)[0]
    res = y0 - C @ b0
    s2 = float(np.sum(w * res ** 2) / len(R))
    D = np.sqrt(v / s2)  # theta - m = D z, penalty |z|^2 = sum s2/v (theta-m)^2
    sw = np.sqrt(w)
    z = lsqr(diags(sw) @ X @ diags(D), res * sw, damp=1.0, atol=1e-8, btol=1e-8, iter_lim=5000)[0]
    return b0, m + D * z


def net(theta, pids):
    n = len(pids)
    return {p: theta[i] - theta[n + i] for i, p in enumerate(pids)}
