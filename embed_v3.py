"""Player embeddings v3: reliability shrinkage, minutes-aware priors, compression.

1. Reliability: per metric, split-half correlation r on odd/even games inside the window at the
   median minutes m_h gives k = m_h (1 - r) / r (Spearman-Brown); rho(m) = m / (m + k).
2. Minutes-aware prior: shrink toward mu(m) = minutes-weighted linear fit of the metric on log minutes,
   so low-minute players regress toward what low-minute players look like, not the league mean.
3. Compression: off/def quality (description->impact map, learned on the map window) + 8 PCs of
   stable style metrics (rho at median minutes >= 0.6) + log minutes.
"""
import numpy as np
import embed_eval as ee
import metrics2 as m2

ALL = ee.BOX + ee.TRK_OFF + ee.TRK_DEF + m2.NEW_OFF + m2.NEW_DEF
N_PC = 8


def reliability(G, gids, shots, cols):
    odd = [g for i, g in enumerate(sorted(gids, key=lambda g: G[g]['date'])) if i % 2]
    even = [g for i, g in enumerate(sorted(gids, key=lambda g: G[g]['date'])) if not i % 2]
    da, db = m2.describe2(G, odd, shots), m2.describe2(G, even, shots)
    both = [p for p in da if p in db and da[p]['minutes'] >= 150 and db[p]['minutes'] >= 150]
    mh = float(np.median([(da[p]['minutes'] + db[p]['minutes']) / 2 for p in both]))
    k = {}
    for c in cols:
        a = np.array([da[p].get(c, np.nan) for p in both], float); b = np.array([db[p].get(c, np.nan) for p in both], float)
        ok = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 20 and a[ok].std() > 0 and b[ok].std() > 0 else 0.0
        r = float(np.clip(r, 0.02, 0.98))
        k[c] = mh * (1 - r) / r  # minutes at which reliability = 0.5 (for the full window: 2x half-window minutes)
    return k, mh


def shrunk_matrix(desc, cols, pids, k):
    """Standardized, reliability-shrunk features toward a minutes-conditional prior."""
    m = np.array([desc.get(p, {}).get('minutes', 0.0) for p in pids])
    X = np.array([[desc.get(p, {}).get(c, np.nan) for c in cols] for p in pids], float)
    lm = np.log1p(m)
    Z = np.zeros_like(X)
    for j, c in enumerate(cols):
        x = X[:, j]; ok = np.isfinite(x) & (m > 0)
        if ok.sum() < 20:
            continue
        w = m[ok]
        D = np.column_stack([np.ones(ok.sum()), lm[ok]])
        beta = np.linalg.lstsq(D * np.sqrt(w)[:, None], x[ok] * np.sqrt(w), rcond=None)[0]
        mu = beta[0] + beta[1] * lm
        rho = m / (m + k[c])
        v = np.where(np.isfinite(x), mu + rho * (x - mu), mu)  # unseen players: minutes prior (m=0 -> low-minute prior)
        sd = np.sqrt(np.average((v[ok] - np.average(v[ok], weights=w)) ** 2, weights=w))
        Z[:, j] = (v - np.average(v[ok], weights=w)) / (sd if sd > 0 else 1)
    lmz = (lm - np.average(lm, weights=np.maximum(m, 1))) / (lm.std() + 1e-9)
    return Z, lmz


def build(G, shots, win_games, map_games, R_win, R_map, pids, pidx, desc):
    """Return dict of embedding matrices aligned with pids: raw, shrunk, comp."""
    k, mh = reliability(G, win_games, shots, ALL)
    Zraw = ee.matrix(desc, ALL, pids)
    Zs, lmz = shrunk_matrix(desc, ALL, pids, k)
    # supervised quality factors on shrunk features
    lam = ee.cv_lambda(R_map, Zs, pidx)
    mdl = ee.Lin().fit(R_map, Zs, pidx, lam)
    F_ = len(ALL); wo, wd = mdl.beta[2:2 + F_], mdl.beta[2 + F_:]
    offq, defq = Zs @ wo, -(Zs @ wd)
    # style PCs on stable metrics (reliability at window-median minutes >= 0.6)
    mins = np.array([desc.get(p, {}).get('minutes', 0.0) for p in pids])
    med = np.median(mins[mins > 0])
    rel = np.array([med / (med + k[c]) for c in ALL])
    stable = [j for j in range(len(ALL)) if rel[j] >= 0.6]
    if len(stable) < 20:  # short windows: fall back to the 30 most reliable metrics
        stable = sorted(np.argsort(-rel)[:30].tolist())
    S = Zs[:, stable]; w = np.maximum(mins, 1e-3)
    Sc = S - np.average(S, axis=0, weights=w)
    U, sv, Vt = np.linalg.svd(Sc * np.sqrt(w)[:, None] / np.sqrt(w.sum()), full_matrices=False)
    pcs = Sc @ Vt[:N_PC].T
    pcs = pcs / (pcs.std(0) + 1e-9)
    q = np.column_stack([offq, defq]); q = q / (q.std(0) + 1e-9)
    comp = np.column_stack([q, pcs, lmz]).astype(np.float32)
    info = dict(k=k, stable=[ALL[j] for j in stable], explained=(sv[:N_PC] ** 2 / (sv ** 2).sum()).round(3).tolist(),
                loadings={f'pc{i}': sorted(zip([ALL[j] for j in stable], Vt[i].round(2)), key=lambda t: -abs(t[1]))[:5]
                          for i in range(N_PC)})
    return dict(raw=Zraw.astype(np.float32), shrunk=np.column_stack([Zs, lmz]).astype(np.float32), comp=comp), info
