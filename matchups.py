"""Who guards whom: defensive assignments for every half-court frame (Franks, Miller, Bornn & Goldsberry 2015 style).

Model: defender d guarding attacker j stands near mu_j = a * O_j + b * ball + (1 - a - b) * hoop (half-court frame,
offense attacks +x, hoop at holes_field.HOOP_X, 0); position ~ N(mu_j, s^2 I). (a, b, s) by EM on a sample of games
with frame-independent soft assignments. Assignments over time: per defender and possession, an HMM with
P(stay) = RHO per 0.2 s frame, Viterbi path. Frames come from the holes pipeline (results/holes/{gid}.npz, 5 Hz).
Per frame and defender: assigned attacker slot and the posterior share of the Viterbi choice (confidence).

Products:
  results/matchups/{gid}.npz   assign [T, 5] int8 (attacker slot 0-4 for defenders 0-4), conf [T, 5] float16
  results/matchups/player.json per defender: possessions, share of frames on the ball handler, switches per 100
  possessions, matchup difficulty (time-weighted ORAPM of the men he guards), help (mean distance from his man);
  per attacker: how often he draws the opponent's best defender (by DRAPM); split-half reliability of each.
Usage: python matchups.py
"""
import json, csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from masked_paths import ROOT

H = ROOT / 'results/holes'
OUT = ROOT / 'results/matchups'
HOOP = np.array([41.75, 0.0])
RHO = 0.97        # P(defender keeps his man over 0.2 s)


def frames(gid):
    d = np.load(H / f'{gid}.npz')
    return d['pos'].astype(np.float32), d['ball'].astype(np.float32), d['meta']


def em(sample, iters=30):
    a, b, s = 0.6, 0.1, 4.0
    O, Dd, B = [], [], []
    for g in sample:
        pos, ball, _ = frames(g)
        O.append(pos[:, :5]); Dd.append(pos[:, 5:]); B.append(ball)
    O, Dd, B = np.concatenate(O)[::3], np.concatenate(Dd)[::3], np.concatenate(B)[::3]
    for _ in range(iters):
        mu = a * O[:, None] + b * B[:, None, None] + (1 - a - b) * HOOP          # [N, 1, 5, 2] -> broadcast below
        mu = mu.reshape(len(O), 1, 5, 2)
        d2 = ((Dd[:, :, None] - mu) ** 2).sum(-1)                              # [N, 5 def, 5 att]
        R = np.exp(-(d2 - d2.min(-1, keepdims=True)) / (2 * s ** 2)); R /= R.sum(-1, keepdims=True)
        # M step: weighted least squares for (a, b) in D - hoop = a (O - hoop) + b (ball - hoop)
        X1 = (O[:, None] - HOOP)[:, :, :, :]                                   # [N, 1, 5, 2]
        X1 = np.broadcast_to(X1, (len(O), 5, 5, 2)); X2 = np.broadcast_to((B - HOOP)[:, None, None], (len(O), 5, 5, 2))
        Y = np.broadcast_to((Dd - HOOP)[:, :, None], (len(O), 5, 5, 2)); W = R[..., None]
        A = np.array([[(W * X1 * X1).sum(), (W * X1 * X2).sum()], [(W * X1 * X2).sum(), (W * X2 * X2).sum()]])
        c = np.array([(W * X1 * Y).sum(), (W * X2 * Y).sum()])
        a, b = np.linalg.solve(A, c)
        mu = HOOP + a * (O[:, None] - HOOP).reshape(len(O), 1, 5, 2) + b * (B - HOOP)[:, None, None]
        s = float(np.sqrt((R * ((Dd[:, :, None] - mu) ** 2).sum(-1)).sum() / (2 * R.sum())))
    return float(a), float(b), float(s)


def viterbi(ll, rho):
    """ll [T, 5] log-likelihoods for one defender -> path [T], posterior-ish confidence [T]"""
    T = len(ll); ls, lc = np.log(rho), np.log((1 - rho) / 4)
    trans = np.full((5, 5), lc); np.fill_diagonal(trans, ls)
    dp = ll[0].copy(); bp = np.zeros((T, 5), int)
    for t in range(1, T):
        m = dp[:, None] + trans
        bp[t] = m.argmax(0); dp = m.max(0) + ll[t]
    path = np.zeros(T, int); path[-1] = dp.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    p = np.exp(ll - ll.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return path, p[np.arange(T), path]


def game(args):
    gid, (a, b, s), base, shots = args
    pos, ball, meta = frames(gid)
    O, Dd = pos[:, :5], pos[:, 5:]
    mu = HOOP + a * (O - HOOP) + b * (ball - HOOP)[:, None]                      # [T, 5 att, 2]
    ll = -((Dd[:, :, None] - mu[:, None]) ** 2).sum(-1) / (2 * s ** 2)          # [T, 5 def, 5 att]
    assign = np.zeros((len(pos), 5), np.int8); conf = np.zeros((len(pos), 5), np.float16)
    k = meta[:, 0]
    starts = np.r_[0, np.where(np.diff(k) != 0)[0] + 1, len(k)]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        for dd in range(5):
            p, c = viterbi(ll[s0:s1, dd], RHO)
            assign[s0:s1, dd] = p; conf[s0:s1, dd] = c
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'{gid}.npz', assign=assign, conf=conf)
    # per-game aggregates
    acc = defaultdict(lambda: defaultdict(float))
    holder = meta[:, 2]; offi = meta[:, 5:10]; defi = meta[:, 10:15]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        for dd in range(5):
            dp_ = int(defi[s0, dd]); A = acc[('d', dp_)]
            A['poss'] += 1; A['frames'] += s1 - s0
            seg = assign[s0:s1]
            for t in np.where(np.diff(seg[:, dd]) != 0)[0] + 1:
                j, kk = seg[t - 1, dd], seg[t, dd]
                lo, hi = max(1, t - 5), min(len(seg), t + 6)
                # a true switch: some other defender moves from kk to j within 1 s (the two exchange men)
                exch = any(((seg[u - 1, e] == kk) & (seg[u, e] == j)) for e in range(5) if e != dd for u in range(lo, hi))
                A['switch' if exch else 'help'] += 1
            mine = seg[:, dd]
            A['double'] += float(((seg == mine[:, None]).sum(1) >= 2).sum())
            men = offi[np.arange(s0, s1), assign[s0:s1, dd]]
            A['on_ball'] += float((men == holder[s0:s1]).sum())
            for m_ in men:
                acc[('dm', dp_)][int(m_)] += 1
            dist = np.linalg.norm(Dd[s0:s1, dd] - O[np.arange(s0, s1), assign[s0:s1, dd]], axis=1)
            A['dist'] += float(dist.sum())
        for j in range(5):
            ap_ = int(offi[s0, j])
            guards = defi[s0:s1][assign[s0:s1] == j]
            for g_ in guards:
                acc[('am', ap_)][int(g_)] += 1
    # shots: who was assigned to the shooter at release
    shot_rows = []
    if shots:
        import holes_validate as hv
        _, T = hv.timestamps(base)
        t = meta[:, 1]
        for ev, ri, sh, made, xb, three, nd in shots:
            if ri >= len(T):
                continue
            f = np.searchsorted(t, T[ri], 'right') - 1
            if f < 0 or T[ri] - t[f] > 300 or sh not in list(offi[f]):
                continue
            j = list(offi[f]).index(sh)
            on = [dd for dd in range(5) if assign[f, dd] == j]
            if not on:
                continue
            dd = max(on, key=lambda e: float(conf[f, e]))
            shot_rows.append(dict(game_id=gid, event_id=ev, shooter=int(sh), assigned=int(defi[f, dd]), nearest=nd,
                                  n_assigned=len(on), made=made, xb=xb, three=three))
    return gid, {f'{k[0]}|{k[1]}': dict(v) for k, v in acc.items()}, shot_rows


def main():
    gids = sorted(f.stem for f in H.glob('00*.npz'))
    rng = np.random.default_rng(0)
    a, b, s = em(rng.choice(gids, 40, replace=False))
    print(f'EM: defender target = {a:.2f} x man + {b:.2f} x ball + {1 - a - b:.2f} x hoop, sd {s:.2f} ft', flush=True)
    dates = {}
    for l in open(ROOT / 'results/shots/alignment.jsonl'):
        r = json.loads(l); dates[r['game_id']] = r['date']
    import zipfile
    from masked_paths import ARCHIVE
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = {n.rsplit('tracking.jsonl.gz', 1)[0].rstrip('/').rsplit('/', 1)[1]: n.rsplit('tracking.jsonl.gz', 1)[0]
                 for n in zz.namelist() if n.endswith('tracking.jsonl.gz')}
    feats = {(r['game_id'], r['event_id']): r for r in map(json.loads, open(ROOT / 'results/shots/features.jsonl'))}
    by = defaultdict(list)
    for al, x in zip(map(json.loads, open(ROOT / 'results/shots/alignment.jsonl')), map(json.loads, open(ROOT / 'results/embed/shots_xfg.jsonl'))):
        f = feats.get((al['game_id'], al['event_id']))
        if f and f.get('trk_ok') and f.get('release_idx') is not None and x.get('xb') is not None:
            by[al['game_id']].append((f['event_id'], f['release_idx'], f['shooter'], int(f['made']), float(x['xb']), bool(f['three']),
                                      int(f['nd_id']) if f.get('nd_id') else None))
    per, shots_all = {}, []
    with ProcessPoolExecutor(8) as ex:
        for gid, agg, sr in ex.map(game, [(g, (a, b, s), bases.get(g), by.get(g, [])) for g in gids], chunksize=4):
            per[gid] = agg; shots_all += sr
    with open(OUT / 'shots.jsonl', 'w') as fo:
        for r in shots_all:
            fo.write(json.dumps(r) + '\n')
    agree = np.mean([r['assigned'] == r['nearest'] for r in shots_all if r['nearest']])
    print(f'shots with an assigned defender: {len(shots_all)}; assigned == nearest defender at release: {agree:.2f}', flush=True)
    # ratings of the men guarded / guarding (2025-26 RAPM table)
    orapm, drapm, name = {}, {}, {}
    for r in csv.DictReader(open(ROOT / 'results/rapm/players.csv')):
        orapm[int(r['player_id'])] = float(r['ORAPM']); drapm[int(r['player_id'])] = float(r['DRAPM']); name[int(r['player_id'])] = r['name']
    order = sorted(per, key=lambda g: (dates.get(g, ''), g))
    half = {g: i % 2 for i, g in enumerate(order)}
    tot = [defaultdict(lambda: defaultdict(float)) for _ in range(3)]  # all, half 0, half 1
    for g, agg in per.items():
        for key, v in agg.items():
            for T_ in (tot[0], tot[1 + half[g]]):
                for kk, x in v.items():
                    T_[key][kk] += x
    def defender_stats(T_):
        out = {}
        for key, v in T_.items():
            kind, p = key.split('|')
            if kind != 'd' or v['poss'] < 300:
                continue
            men = T_.get(f'dm|{p}', {})
            w = sum(men.values())
            diff = sum(c * orapm.get(int(m), 0.0) for m, c in men.items()) / max(w, 1)
            out[int(p)] = dict(poss=v['poss'], on_ball=v['on_ball'] / v['frames'], switches_100=100 * v.get('switch', 0) / v['poss'],
                               help_100=100 * v.get('help', 0) / v['poss'], double=v.get('double', 0) / v['frames'],
                               difficulty=diff, dist_ft=v['dist'] / v['frames'])
        return out
    D0, D1, D2 = defender_stats(tot[0]), defender_stats(tot[1]), defender_stats(tot[2])
    rel = {}
    for k in ('on_ball', 'switches_100', 'help_100', 'double', 'difficulty', 'dist_ft'):
        c = [p for p in D1 if p in D2]
        rel[k] = float(np.corrcoef([D1[p][k] for p in c], [D2[p][k] for p in c])[0, 1])
    print('defender metrics split-half r:', {k: round(v, 2) for k, v in rel.items()})
    for k, lab in (('difficulty', 'toughest assignments (avg ORAPM of man guarded)'), ('on_ball', 'most time on the ball handler'),
                   ('switches_100', 'most true switches per 100 poss'), ('help_100', 'most help rotations per 100 poss'),
                   ('double', 'most time doubling')):
        top = sorted(D0, key=lambda p: -D0[p][k])[:10]
        print(f'{lab}: ' + ', '.join(f"{name.get(p, p)} {D0[p][k]:.2f}" for p in top))
    # shot defense: points vs expected on shots by the man he was assigned to, vs the nearest-defender version
    K = 100.0
    def shot_def(rows, key):
        acc = defaultdict(lambda: [0.0, 0])
        for r in rows:
            if r[key]:
                acc[r[key]][0] += (r['made'] - r['xb']) * (3 if r['three'] else 2); acc[r[key]][1] += 1
        return {p: (v[0] / (v[1] + K), v[1]) for p, v in acc.items()}
    shot_rel = {}
    for key in ('assigned', 'nearest'):
        A0 = shot_def([r for r in shots_all if half.get(r['game_id']) == 0], key); A1 = shot_def([r for r in shots_all if half.get(r['game_id']) == 1], key)
        c = [p for p in A0 if p in A1 and A0[p][1] >= 60 and A1[p][1] >= 60]
        shot_rel[key] = float(np.corrcoef([A0[p][0] for p in c], [A1[p][0] for p in c])[0, 1])
    SD = shot_def(shots_all, 'assigned')
    for p, (v, n) in SD.items():
        if p in D0:
            D0[p]['shot_pts_vs_exp'] = v; D0[p]['shots_faced'] = n
    print('shot defense split-half r (points vs expected per shot, shrunk):', {k: round(v, 2) for k, v in shot_rel.items()})
    best = sorted([p for p in D0 if D0[p].get('shots_faced', 0) >= 200], key=lambda p: D0[p]['shot_pts_vs_exp'])[:10]
    print('best shot defense vs assigned man (pts/shot vs expected): ' + ', '.join(f"{name.get(p, p)} {D0[p]['shot_pts_vs_exp']:+.3f}" for p in best))
    json.dump(dict(em=dict(a=a, b=b, s=s), reliability=rel, shot_reliability=shot_rel, defenders={str(p): v for p, v in D0.items()}),
              open(OUT / 'player.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
