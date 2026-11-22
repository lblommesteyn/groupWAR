"""EPV v2 labels: subtract shot luck from possession points (control variate).

  y2 = possession points - sum over the possession's shots of (made - xFG) * shot points
  xFG: tracking shot model (shots_value xpts / pts) when the shot passed tracking QC, else location-only GBM.
  Unbiased for possession points when xFG is calibrated; keeps free throws, and-ones and putbacks.
  kind 0 = possession had >= 1 matched shot, 1 = none (y2 = y), 2 = frame after the possession's last shot (dropped).
  (First attempt used xpts_reb of the next shot; its miss value nets out the opponent's next possession, so it
  measured a different quantity and under-predicted points by 0.16 per possession. Kept below for reference only.)

OLD DESCRIPTION:

v1 labelled every 1 Hz frame with the possession's actual points, so most of the label variance was make/miss luck
(R2 vs constant 0.07). v2 keeps v1's frames (results/epv/data) and relabels each one:

  next tracked shot by the offense in this possession, released after the frame  ->  y2 = xpts_reb of that shot
      (tracking xFG * pts + (1 - xFG) * value of the miss, which carries the offensive-rebound continuation)
  frame after the possession's last tracked shot (ball in flight, putback scramble)  ->  dropped (kind 2)
  shot without a tracking-quality value (trk_ok false / location mismatch, 27% of shots)  ->  y2 = location-only
      xFG (distance, 2/3, shot type; GBM cross-fitted by game fold) * pts + (1 - xFG) * mean value of a miss (kind 3)
  possession with no tracked shot (turnover, non-shooting foul FTs, untracked shot)  ->  y2 = actual points (kind 1)

Shot release time = tracking timestamp at release_idx (results/shots/features.jsonl), same frame filter as epv_data.
Output: results/epv/labels2/{gid}.npz  y2 float32, kind int8 (0 shot-smoothed, 1 actual, 2 drop), shot_t (release ms
of the labelling shot or -1), shot_pid.   Usage: python epv2_label.py [workers]
"""
import json, sys, zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import holes_validate as hv
from masked_paths import ROOT, ARCHIVE

D = ROOT / 'results/epv/data'
OUT = ROOT / 'results/epv/labels2'
GAP_MS = 3000  # a shot may be stamped slightly after the last 1 Hz frame of the possession


def game(args):
    gid, base, shots = args
    d = np.load(D / f'{gid}.npz')
    meta, y = d['meta'], d['y']
    _, T = hv.timestamps(base)
    rel = []  # (t_release, shooter, xpts_reb)
    for ri, sh, xr, kd in shots:
        if ri < len(T):
            rel.append((T[ri], sh, xr, kd))
    rel.sort()
    rt = np.array([r[0] for r in rel]) if rel else np.zeros(0)
    n = len(y)
    y2 = y.astype(np.float32).copy(); kind = np.ones(n, np.int8); st = np.full(n, -1, np.int64); sp = np.zeros(n, np.int64)
    by_poss = defaultdict(list)
    for i in range(n):
        by_poss[meta[i, 0]].append(i)
    for k, idx in by_poss.items():
        off = set(meta[idx[0], 3:8].tolist())
        t0, t1 = meta[idx[0], 1], meta[idx[-1], 1]
        lo, hi = np.searchsorted(rt, t0 - 1000), np.searchsorted(rt, t1 + GAP_MS, 'right')
        ps = [rel[j] for j in range(lo, hi) if rel[j][1] in off]
        if not ps:
            continue  # kind 1: actual points
        luck = sum(s[2] for s in ps)
        for i in idx:
            t = meta[i, 1]
            nxt = next((s for s in ps if s[0] >= t), None)
            y2[i] = y[i] - luck
            if nxt is None:
                kind[i] = 2
            else:
                kind[i] = 0; st[i] = nxt[0]; sp[i] = nxt[1]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'{gid}.npz', y2=y2, kind=kind, shot_t=st, shot_pid=sp)
    return gid, n, int((kind == 0).sum()), int((kind == 2).sum()), len(rel)


def main():
    sv = np.load(ROOT / 'results/rebounds/shots_value.npz')
    xr = {(g, int(e)): float(v) for g, e, v in zip(sv['game'], sv['event'], sv['xpts_reb'])}
    xp = {(g, int(e)): float(v) for g, e, v in zip(sv['game'], sv['event'], sv['xpts'])}
    from sklearn.ensemble import HistGradientBoostingClassifier
    F = [json.loads(l) for l in open(ROOT / 'results/shots/features.jsonl')]
    F = [r for r in F if r.get('release_idx') is not None]
    sub = {s: j for j, s in enumerate(sorted({r['subtype'] for r in F}))}
    X = np.array([[r['dist_pbp'], r['three'], sub[r['subtype']]] for r in F], float)
    yb = np.array([r['made'] for r in F], int)
    games = sorted({r['game_id'] for r in F}); fold = {g: i % 5 for i, g in enumerate(games)}
    fd = np.array([fold[r['game_id']] for r in F]); px = np.zeros(len(F))
    for k in range(5):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, categorical_features=[2]).fit(X[fd != k], yb[fd != k])
        px[fd == k] = m.predict_proba(X[fd == k])[:, 1]
    vmiss = float(np.mean(sv['v_miss']))
    by = defaultdict(list); nfb = 0
    for r, p in zip(F, px):
        key = (r['game_id'], r['event_id']); pts = 3 if r['three'] else 2
        if key in xp:
            xfg = xp[key] / pts
        else:
            xfg = p; nfb += 1
        by[r['game_id']].append((r['release_idx'], r['shooter'], (float(r['made']) - xfg) * pts, 0))
    print('location-fallback shots', nfb, 'mean v_miss', round(vmiss, 3), flush=True)
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = {n.rsplit('tracking.jsonl.gz', 1)[0].rstrip('/').rsplit('/', 1)[1]: n.rsplit('tracking.jsonl.gz', 1)[0]
                 for n in zz.namelist() if n.endswith('tracking.jsonl.gz')}
    tasks = [(f.stem, bases[f.stem], by.get(f.stem, [])) for f in sorted(D.glob('*.npz')) if f.stem in bases]
    tot = np.zeros(3, int); shots = 0
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        for i, (g, n, ns, nd, nr) in enumerate(ex.map(game, tasks, chunksize=2), 1):
            tot += [n, ns, nd]; shots += nr
            if i % 100 == 0:
                print(i, tot, flush=True)
    print(f'frames {tot[0]}  with shots (kind 0) {tot[1]} ({tot[1] / tot[0]:.1%})  dropped {tot[2]} ({tot[2] / tot[0]:.1%})  shots {shots}')


if __name__ == '__main__':
    main()
