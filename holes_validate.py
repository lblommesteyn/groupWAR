"""Replicates the paper's section 4.2 on the NBA archive: does the field predict the next 3 s?

Samples: every 10th half-court frame (0.5 Hz) from results/holes. Targets in (t, t + 3 s], offense team only:
  shot        any aligned shot release
  open_shot   closest defender > 6 ft at release
  rim_shot    release within 6 ft of the rim
  shot_q      regression on the tracking xFG x points of the next shot (frames with a shot in the window)
Feature sets: baseline (16 geometry features as in the paper), + Voronoi, + probabilistic field, field only.
Models: logistic / ridge and HistGradientBoosting, 5-fold GroupKFold by game.
"""
import gzip, json, sys, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from masked_paths import ROOT, ARCHIVE
HOOP_X = 41.75  # (not imported from holes_field: torch + sklearn OpenMP deadlock on macOS)

H = ROOT / 'results/holes'
FIELD = ['A', 'A_empty', 'A_occ', 'A_ball', 'A_rim', 'A_paint', 'A_corner3', 'A_arc3', 'H_tot', 'H_max', 'C_mean']


def timestamps(base):
    T = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for line in gzip.open(z.open(base + 'tracking.jsonl.gz'), 'rt'):
            f = json.loads(line)
            if f.get('period'):
                T.append(f['timestamp_ms'])
    return base, T


def hull(p):
    try:
        return ConvexHull(p).volume
    except Exception:
        return 0.0


def baseline(pos, vel, ball, meta):
    off, de = pos[:, :5], pos[:, 5:]
    dmat = np.linalg.norm(off[:, :, None] - de[:, None], axis=-1)  # [N,5,5]
    nd = dmat.min(2)
    hold = meta[:, 2]; ids = meta[:, 5:10]
    hi = np.argmax(ids == hold[:, None], 1); has = (ids == hold[:, None]).any(1)
    hnd = np.where(has, nd[np.arange(len(nd)), hi], nd.min(1))
    rim = lambda x, y: np.hypot(x - HOOP_X, y)
    sc = meta[:, 4].astype(float); sc = np.where(sc < 0, 24000, sc) / 1000
    sp = np.linalg.norm(vel, axis=-1)
    return np.column_stack([
        ball[:, 0], ball[:, 1], rim(ball[:, 0], ball[:, 1]), sc, hnd, nd.mean(1), nd.max(1), (nd > 6).sum(1),
        [hull(p) for p in off], [hull(p) for p in de], off.std(1).sum(1), off[:, :, 0].mean(1), de[:, :, 0].mean(1),
        sp[:, :5].mean(1), sp[:, 5:].mean(1), rim(off[:, :, 0], off[:, :, 1]).min(1)])


def main():
    al = defaultdict(list)
    feats = {(r['game_id'], r['event_id']): r for r in map(json.loads, open(ROOT / 'results/shots/features.jsonl'))}
    xq = {}  # shots_xfg.jsonl is row-aligned with alignment.jsonl (as in tracking_metrics)
    for a, x in zip(map(json.loads, open(ROOT / 'results/shots/alignment.jsonl')), map(json.loads, open(ROOT / 'results/embed/shots_xfg.jsonl'))):
        assert a['game_id'] == x['game_id'] and a['shooter'] == x['shooter']
        f = feats.get((a['game_id'], a['event_id']))
        if f and f.get('trk_ok'):
            al[a['game_id']].append(f)
            xq[(a['game_id'], a['event_id'])] = x['xb'] * (3 if x['three'] else 2) if x.get('xb') is not None else np.nan
    gids = sorted(f.stem for f in H.glob('*.npz'))
    if len(sys.argv) > 1:
        gids = gids[:int(sys.argv[1])]
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = {n.rsplit('tracking.jsonl.gz', 1)[0].rstrip('/').rsplit('/', 1)[1]: n.rsplit('tracking.jsonl.gz', 1)[0]
                 for n in zz.namelist() if n.endswith('tracking.jsonl.gz')}
    with ProcessPoolExecutor(8) as ex:
        TS = {b.rstrip('/').rsplit('/', 1)[1]: T for b, T in ex.map(timestamps, [bases[g] for g in gids])}
    X0, XV, XF, Y, G = [], [], [], defaultdict(list), []
    for g in gids:
        d = np.load(H / f'{g}.npz'); T = TS[g]
        s = np.arange(0, len(d["meta"]), 10)
        meta = d['meta'][s]; t = meta[:, 1]; off = meta[:, 5:10]
        shots = sorted((T[f['release_idx']], f['shooter'], f['d1'], f['dist_trk'],
                        xq.get((g, f['event_id']), np.nan))
                       for f in al[g] if f['release_idx'] < len(T))
        st = np.array([x[0] for x in shots])
        for k in range(len(s)):
            j0, j1 = np.searchsorted(st, t[k], 'right'), np.searchsorted(st, t[k] + 3000, 'right')
            nxt = [x for x in shots[j0:j1] if x[1] in off[k]]
            Y['shot'].append(bool(nxt)); Y['open_shot'].append(any(x[2] > 6 for x in nxt))
            Y['rim_shot'].append(any(x[3] < 6 for x in nxt)); Y['shot_q'].append(nxt[0][4] if nxt else np.nan)
        pos, vel, ball = (d[k][s].astype(np.float32) for k in ('pos', 'vel', 'ball'))
        X0.append(baseline(pos, vel, ball, meta)); XV.append(d['A_vor'][s].astype(float)[:, None])
        XF.append(np.column_stack([d[k][s].astype(float) for k in FIELD])); G += [g] * len(s)
    X0, XV, XF, G = np.vstack(X0), np.vstack(XV), np.vstack(XF), np.array(G)
    Y = {k: np.array(v, float) for k, v in Y.items()}
    sets = {'baseline': X0, '+voronoi': np.hstack([X0, XV]), '+field': np.hstack([X0, XF]), 'field_only': XF}
    print(f'frames {len(G)} games {len(set(G))} rates', {k: round(float(np.nanmean(v)), 3) for k, v in Y.items() if k != 'shot_q'})
    res = {}
    cv = GroupKFold(5)
    for tgt, y in Y.items():
        reg = tgt == 'shot_q'
        m = np.isfinite(y)
        for mname in ('linear', 'gbm'):
            for sname, X in sets.items():
                Xm, ym, gm = X[m], y[m], G[m]
                pred = np.zeros(len(ym))
                for tr, te in cv.split(Xm, ym, gm):
                    if reg:
                        mdl = make_pipeline(StandardScaler(), Ridge(1.0)) if mname == 'linear' else HistGradientBoostingRegressor(max_iter=300)
                        pred[te] = mdl.fit(Xm[tr], ym[tr]).predict(Xm[te])
                    else:
                        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)) if mname == 'linear' else HistGradientBoostingClassifier(max_iter=300)
                        pred[te] = mdl.fit(Xm[tr], ym[tr]).predict_proba(Xm[te])[:, 1]
                sc = r2_score(ym, pred) if reg else roc_auc_score(ym, pred)
                res[f'{tgt}|{mname}|{sname}'] = float(sc)
                print(f'{tgt:10s} {mname:6s} {sname:11s} {"R2" if reg else "AUC"} {sc:.4f}', flush=True)
    json.dump(res, open(H.parent / 'holes_validate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
