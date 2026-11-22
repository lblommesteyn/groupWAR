"""Possession library for the replayer + play search site (~/code/possession-lab/public/data).

Library: every Charlotte half-court possession (offense or defense; capped) + a league-wide random sample.
A possession = the longest continuous run (gaps <= 400 ms) of one holes possession k, >= 5 s of half-court frames.
Per possession: 5 Hz frames (10 players + ball, offense attacks +x), holder per frame, EPV v3 per frame (cross-fitted,
results/epv/out3/epv3_cf.npz when present,
linearly recalibrated to actual points per fold using the other folds), outcome (points; last shot from results/shots/features.jsonl).

Play search: windows ending every 1 s (full 2 s history) are embedded with the MAE encoder (mae_long, no identity;
mean of "now" tokens + mean of all tokens, also for the y-mirrored window). Shortlist 60 by cosine (best of
direct/mirrored), rerank by role-aligned path distance: best one-to-one matching of attackers to attackers and
defenders to defenders (mean distance over the 2 s), averaged with the ball's distance -> feet. Keep top 8 from
other possessions.
Usage: python poss_lab.py [n_cha] [n_league]
"""
import json, os, sys, random, itertools
from collections import defaultdict
import numpy as np

from masked_paths import ROOT

H = ROOT / 'results/holes'
E = ROOT / 'results/epv'
OUT = ROOT.parent / 'possession-lab/public/data'
CHA = 1610612766
TEAMS = dict(zip(range(1610612737, 1610612767), 'ATL BOS CLE NOP CHI DAL DEN GSW HOU LAC LAL MIA MIL MIN BKN NYK ORL IND PHI PHX POR SAC SAS OKC TOR UTA MEM WAS DET CHA'.split()))
SCALE = np.array([47.0, 25.0, 20.0, 20.0], np.float32)
PERMS = np.array(list(itertools.permutations(range(5))))


def short(n):
    p = n.split(' ')
    return (p[0][0] + '. ' + ' '.join(p[1:])) if len(p) > 1 else n


def game_frames(g):
    h = np.load(H / f'{g}.npz')
    meta = h['meta']; t = meta[:, 1]
    pos, vel, ball = h['pos'].astype(np.float32), h['vel'].astype(np.float32), h['ball'].astype(np.float32)
    bv = np.zeros_like(ball); bv[1:] = (ball[1:] - ball[:-1]) / np.maximum(np.diff(t)[:, None] / 1000, 0.1)
    brk = np.r_[True, (meta[1:, 0] != meta[:-1, 0]) | (np.diff(t) > 400)]
    bv[brk] = 0
    return meta, pos, vel, ball, bv, brk


def main():
    n_cha = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n_lg = int(sys.argv[2]) if len(sys.argv) > 2 else 700
    import pickle
    G = pickle.load(open(ROOT / 'results/embed/player_games.pkl', 'rb'))
    v3 = np.load(E / 'v3.npz')
    games3 = list(v3['games']); gi, fi = v3['gi'], v3['fi']
    present = sorted(set(gi.tolist()))
    off, o = {}, 0
    for k in present:
        g = games3[k]; off[g] = o; o += len(np.load(H / f'{g}.npz')['meta'])
    cf = E / 'out3/epv3_cf.npz'
    epv_s = None
    if cf.exists():
        c = np.load(cf); epv_s = c['epv'].astype(float); fd, yy, kd = c['fold'], c['y'], c['kind']
        # recalibrate: per fold, linear map EPV -> actual points fit on the other folds' out-of-fold EPV (kind != 2)
        raw = epv_s.copy()
        for k in np.unique(fd):
            m = (fd != k) & (kd != 2)
            b, a = np.polyfit(raw[m], yy[m], 1)
            epv_s[fd == k] = a + b * raw[fd == k]
            print(f'fold {k}: actual = {a:+.3f} + {b:.3f} * EPV', flush=True)
        ok = kd != 2
        print('mean EPV raw %.3f calibrated %.3f actual %.3f' % (raw[ok].mean(), epv_s[ok].mean(), yy[ok].mean()), flush=True)
    print('EPV v3 cross-fit', 'found' if epv_s is not None else 'missing', flush=True)
    frame2s = {}  # (g, j) -> v3 sample
    for s, (k, f) in enumerate(zip(gi, fi)):
        g = games3[k]; frame2s[(g, int(f - off[g]))] = s
    # shots: release times
    import holes_validate as hv, zipfile
    from masked_paths import ARCHIVE
    shots = defaultdict(list)
    for l in open(ROOT / 'results/shots/features.jsonl'):
        r = json.loads(l)
        if r.get('release_idx') is not None:
            shots[r['game_id']].append(r)

    team_of = {}
    for g, d in G.items():
        for st in d['stints']:
            for p in st['lineup'][0]:
                team_of[(g, p)] = d['home']
            for p in st['lineup'][1]:
                team_of[(g, p)] = d['away']
    cand = defaultdict(list)  # 'cha' / 'lg' -> (g, j0, j1)
    for g in off:
        if g not in G:
            continue
        meta, *_ , brk = game_frames(g)
        runs = np.split(np.arange(len(meta)), np.flatnonzero(brk)[1:])
        best = {}
        for r in runs:
            k = int(meta[r[0], 0])
            if len(r) >= 25 and (k not in best or len(r) > best[k][1] - best[k][0]):
                best[k] = (r[0], r[-1] + 1)
        is_cha = CHA in (G[g]['home'], G[g]['away'])
        for k, (a, b) in best.items():
            cand['cha' if is_cha else 'lg'].append((g, a, b))
    random.seed(0)
    lib = random.sample(cand['cha'], min(n_cha, len(cand['cha']))) + random.sample(cand['lg'], min(n_lg, len(cand['lg'])))
    lib.sort()
    print('candidates', {k: len(v) for k, v in cand.items()}, 'library', len(lib), flush=True)

    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = {n.rsplit('tracking.jsonl.gz', 1)[0].rstrip('/').rsplit('/', 1)[1]: n.rsplit('tracking.jsonl.gz', 1)[0]
                 for n in zz.namelist() if n.endswith('tracking.jsonl.gz')}
    rel_t = {}
    from concurrent.futures import ProcessPoolExecutor
    gl_ = sorted({g for g, _, _ in lib})
    with ProcessPoolExecutor(8) as ex:
        for g, (_, T) in zip(gl_, ex.map(hv.timestamps, [bases[g] for g in gl_], chunksize=4)):
            rel_t[g] = [(T[r['release_idx']], r) for r in shots[g] if r['release_idx'] < len(T)]
    print('shot times for', len(gl_), 'games', flush=True)

    poss, WX, Wref = [], [], []
    cache = {}
    for pi, (g, a, b) in enumerate(lib):
        if g not in cache:
            cache = {g: game_frames(g)}
        meta, pos, vel, ball, bv, brk = cache[g]
        d = G[g]; names = d['players']
        ids = meta[a, 5:15].tolist(); t = meta[a:b, 1]
        offt = team_of.get((g, ids[0])); deft = d['away'] if offt == d['home'] else d['home']
        hold = [ids.index(h) if h in ids else -1 for h in meta[a:b, 2]]
        samp = [frame2s.get((g, j)) for j in range(a, b)]
        s0 = next((s for s in samp if s is not None), None)
        gl = v3['glob'][s0] if s0 is not None else np.full(5, np.nan)
        pts = int(v3['y'][s0]) if s0 is not None else None
        sh = [r for tt, r in rel_t[g] if t[0] - 500 <= tt <= t[-1] + 1500 and r['shooter'] in ids[:5]]
        if sh:
            r = sh[-1]
            outcome = f"{short(names.get(r['shooter'], '?'))} {'made' if r['made'] else 'missed'} {'3' if r['three'] else '2'}PT {r['subtype'].lower()}"
            shot_f = int(np.searchsorted(t, max(tt for tt, rr in rel_t[g] if rr is r)))
        else:
            outcome = 'No shot (turnover, foul or reset)' if not pts else 'Free throws'; shot_f = None
        ep = [None if s is None or epv_s is None else round(float(epv_s[s]), 3) for s in samp]
        F = np.concatenate([pos[a:b], ball[a:b, None]], 1)
        poss.append(dict(id=pi, g=g, date=d['date'], off=TEAMS.get(offt, '?'), de=TEAMS.get(deft, '?'), cha=CHA in (offt, deft),
                         per=int(meta[a, 3]), clock=None if np.isnan(gl[1]) else float(gl[1]),
                         margin=None if np.isnan(gl[3]) else int(gl[3]), pts=pts, outcome=outcome, shot=shot_f,
                         n=[short(names.get(p, str(p))) for p in ids], dur=round((t[-1] - t[0]) / 1000, 1),
                         f=np.round(F.reshape(len(F), -1), 1).tolist(), h=hold, e=ep))
        Xf = np.concatenate([np.concatenate([pos[a:b], vel[a:b]], -1), np.concatenate([ball[a:b], bv[a:b]], -1)[:, None]], 1)
        for j in range(9, b - a, 5):
            WX.append(Xf[j - 9:j + 1]); Wref.append((pi, j))
    WX = np.array(WX, np.float32); print('windows', len(WX), flush=True)

    import torch
    sys.path.insert(0, str(ROOT)); import jepa
    net = jepa.Net(1, 'mae_long'); net.load_state_dict(torch.load(ROOT / 'results/jepa/mae_long.pt', map_location='cpu')); net.eval()

    def emb(X):
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 512):
                x = torch.from_numpy(X[s:s + 512] / SCALE)
                h = net.encode(x).view(len(x), 10, 11, -1)
                z = torch.cat([h[:, 9].mean(1), h.mean((1, 2))], 1)
                out.append(torch.nn.functional.normalize(z, dim=1).numpy())
        return np.concatenate(out)
    Xm = WX.copy(); Xm[..., 1] *= -1; Xm[..., 3] *= -1
    Z, Zm = emb(WX), emb(Xm)
    wp = np.array([p for p, _ in Wref])
    P = WX[..., :2]  # [W, 10, 11, 2]
    Pm = Xm[..., :2]
    nbr = defaultdict(dict)
    for s in range(0, len(Z), 1024):
        S1, S2 = Z[s:s + 1024] @ Z.T, Z[s:s + 1024] @ Zm.T
        S = np.maximum(S1, S2); S[wp[s:s + 1024, None] == wp[None]] = -9
        top = np.argpartition(-S, 60, 1)[:, :60]
        for qi, row in zip(range(s, s + len(top)), top):
            res = []
            for flip, PP in ((0, P), (1, Pm)):
                A = P[qi][None]  # [1, 10, 11, 2]
                B = PP[row]      # [K, 10, 11, 2]
                Do = np.linalg.norm(A[:, :, :5, None] - B[:, :, None, :5], axis=-1).mean(1)   # [K, 5, 5]
                Dd = np.linalg.norm(A[:, :, 5:10, None] - B[:, :, None, 5:10], axis=-1).mean(1)
                bo = Do[:, np.arange(5)[None], PERMS].sum(-1).min(-1) / 5
                bd = Dd[:, np.arange(5)[None], PERMS].sum(-1).min(-1) / 5
                bb = np.linalg.norm(A[:, :, 10] - B[:, :, 10], axis=-1).mean(1)
                res.append((bo * 5 + bd * 5 + bb) / 11)
            dist = np.minimum(res[0], res[1]); fl = (res[1] < res[0]).astype(int)
            o = np.argsort(dist)[:8]
            pi, j = Wref[qi]
            nbr[pi][j] = [[int(Wref[row[k]][0]), int(Wref[row[k]][1]), round(float(dist[k]), 1), int(fl[k])] for k in o]
        print('search', s, flush=True)

    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'p').mkdir(exist_ok=True)
    index = []
    for p in poss:
        p['w'] = {str(j): v for j, v in nbr[p['id']].items()}
        json.dump(p, open(OUT / 'p' / f"{p['id']}.json", 'w'), separators=(',', ':'))
        index.append({k: p[k] for k in ('id', 'g', 'date', 'off', 'de', 'cha', 'per', 'clock', 'margin', 'pts', 'outcome', 'n', 'dur')}
                     | dict(e0=next((x for x in p['e'] if x is not None), None), ns=len(p['w'])))
    json.dump(dict(possessions=index, hoop_x=41.75, epv=epv_s is not None), open(OUT / 'index.json', 'w'), separators=(',', ':'))
    print('wrote', len(poss), 'possessions')


if __name__ == '__main__':
    main()
