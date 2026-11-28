"""BOWEN prospective test: frozen models scored on games captured live after training (protocol: BOWEN_PROSPECTIVE.md).

Frozen models: results/world/frozen/{variant}/world_fold{0,1}.pt (downloaded from the Modal volume, sha256 in the
protocol). Nothing is retrained. For a new game every model's two fold networks both predict, and the readout is their
equal-weight ensemble: density = mean of the two fold densities, spot = mean of the two fold spots.

  python bowen_prospective.py gate                 reproduction gate: re-predict saved held-out possessions, must match pred.npz
  python bowen_prospective.py inputs GID ...       input check on converted live games (no predictions, no errors)
  python bowen_prospective.py score [--live DIR]   predict + score every eligible live game (only after the protocol is committed)

Live games: court-live data/tracking/{gid}/tracking.jsonl.gz (convert_live.py) and data/schedule/2026-27.json for teams.
Possessions are cut by traj_export.game itself (the training code), fed through a temporary zip with the archive layout.
"""
import hashlib, importlib, io, json, math, os, sys, tempfile, zipfile
from pathlib import Path
import numpy as np

from masked_paths import ROOT

W = ROOT / 'results/world'
FROZEN = W / 'frozen'
LIVE_REPO = Path.home() / 'code/court-live'
VARIANTS = {'A_live': ('bowen_live', {'WM_LIVE': '1', 'WM_MIX': '1'}),
            'B_live_mix3': ('bowen_live_mix3', {'WM_LIVE': '1', 'WM_MIX': '3'}),
            'V3': ('bowen_v3_hier', {'WM_LIVE': '1', 'WM_MIX': '3', 'WM_HIER': '1'})}
FIRST_DATE = '2026-11-27'      # first live-captured date; training data end 2026-11-22


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(env):
    for k in ('WM_LIVE', 'WM_MIX', 'WM_HIER', 'WM_DIR', 'WM_CUT'):
        os.environ.pop(k, None)
    os.environ.update(env)
    import world_model
    wm = importlib.reload(world_model); wm.DEV = 'cpu'
    return wm


def maps(wm):
    """Training index maps, rebuilt exactly as world_model.load() does (and checked against it)."""
    Dd, n_ts = wm.load()
    ts = sorted({(int(t), int(s)) for t, s in zip(Dd['def_team'], Dd['season'])})
    M = dict(ts={k: i + 1 for i, k in enumerate(ts)},
             tm={t: i + 1 for i, t in enumerate(sorted(set(int(t) for t in Dd['def_team'])))},
             df={p: i + 1 for i, p in enumerate(sorted(set(int(p) for p in Dd['deff'].ravel())))}, n_ts=n_ts)
    assert all(M['ts'][(int(t), int(s))] == v for t, s, v in zip(Dd['def_team'][:5000], Dd['season'][:5000], Dd['ts'][:5000]))
    return Dd, M


def nets(wm, variant, M):
    import torch
    d, _ = VARIANTS[variant]; out = []
    wm.N_TEAM, wm.N_DEF = len(M['tm']) + 1, len(M['df']) + 1
    for f in (0, 1):
        n = wm.Net(M['n_ts']); n.load_state_dict(torch.load(FROZEN / d / f'world_fold{f}.pt', map_location='cpu')); n.eval()
        out.append(n)
    return out


def run(wm, net, Dd, idx):
    import torch
    with torch.no_grad():
        return wm.predict(net, Dd, idx)[0]


# ---------------------------------------------------------------- gate
def gate(n_per_fold=150, tol_spot=0.05, tol_nll=0.02):
    """world_fold{f}.pt predicted fold f's games in training; re-predict a sample and compare with the saved pred.npz.
    float16 storage: spots within 0.05 ft, NLL within 0.02 (both are float16 rounding at these magnitudes)."""
    rep = {}
    for v, (d, env) in VARIANTS.items():
        wm = module(env); Dd, M = maps(wm); N = nets(wm, v, M)
        S = np.load(W / d / 'pred.npz')
        key = lambda p, t, k: p.astype(np.int64) * 600 + t.astype(np.int64) * 5 + k.astype(np.int64)
        sk = key(S['poss'], S['t'], S['k']); o = np.argsort(sk); sk = sk[o]
        rng = np.random.default_rng(1); r = {}
        for f in (0, 1):
            idx = np.sort(rng.choice(np.flatnonzero(S['fold'] == f), n_per_fold, replace=False))
            p = run(wm, N[f], Dd, idx)
            k2 = key(p['poss'], p['t'], p['k']); j = np.searchsorted(sk, k2)
            found = (j < len(sk)) & (sk[np.minimum(j, len(sk) - 1)] == k2); i = o[np.minimum(j, len(sk) - 1)]
            cols = ['ms', 'mb', 'nll_s', 'nll_b'] + (['mp', 'nll_p'] if 'mp' in p else [])
            dev = {c: float(np.max(np.abs(p[c][found].astype(np.float64) - S[c][i[found]].astype(np.float64)))) for c in cols}
            r[f'fold{f}'] = dict(possessions=int(len(idx)), rows=int(len(k2)), matched=int(found.sum()), max_abs_diff=dev,
                                 ok=bool(found.all() and all(dev[c] <= (tol_spot if c[0] == 'm' else tol_nll) for c in cols)))
        rep[v] = dict(r, ok=all(x['ok'] for x in r.values()))
        print(v, json.dumps(rep[v]), flush=True)
    out = dict(tolerances=dict(spot_ft=tol_spot, nll=tol_nll), results=rep, ok=all(x['ok'] for x in rep.values()),
               weights={f'{d}/world_fold{f}.pt': sha(FROZEN / d / f'world_fold{f}.pt') for d, _ in VARIANTS.values() for f in (0, 1)})
    (W / 'bowen_prospective').mkdir(exist_ok=True)
    json.dump(out, open(W / 'bowen_prospective/gate.json', 'w'), indent=1)
    print('GATE', 'PASS' if out['ok'] else 'FAIL')
    return out['ok']


# ---------------------------------------------------------------- live games
def schedule():
    return {g['gid']: g for g in json.load(open(LIVE_REPO / 'data/schedule/2026-27.json'))}


def live_possessions(gid, track_root):
    """traj_export.game on a converted live game (temporary zip in the archive layout)."""
    import traj_export
    g = schedule()[gid]
    meta = dict(game=dict(homeTeam=dict(teamId=g['home_id']), awayTeam=dict(teamId=g['away_id'])))
    base = f'live/{g["date"]}/{gid}/'
    with tempfile.TemporaryDirectory() as td:
        zp = Path(td) / 'g.zip'
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:
            z.writestr(base + 'metadata.json', json.dumps(meta))
            z.write(Path(track_root) / gid / 'tracking.jsonl.gz', base + 'tracking.jsonl.gz')
        _, rows = traj_export.game((str(zp), base))
    return g, rows


def dataset(games, desc, M, season=2027):
    rows, gids, dates = [], [], []
    for g, rs in games:
        rows += rs; gids += [g['gid']] * len(rs); dates += [g['date']] * len(rs)
    N, T = len(rows), 60
    D = dict(pos=np.full((N, T, 10, 2), np.nan, np.float16), ball=np.full((N, T, 3), np.nan, np.float16),
             holder=np.full((N, T), -1, np.int8), sc=np.full((N, T), np.nan, np.float16), length=np.zeros(N, np.int16))
    for k, r in enumerate(rows):
        t = len(r['pos']); D['length'][k] = t
        D['pos'][k, :t] = r['pos']; D['ball'][k, :t] = r['ball']; D['holder'][k, :t] = r['holder']; D['sc'][k, :t] = r['sc']
    ds = desc['all'] if 'all' in desc else desc[str(season)]
    D['off'] = np.array([r['off'] for r in rows], np.int64); D['deff'] = np.array([r['deff'] for r in rows], np.int64)
    D['off_team'] = np.array([r['off_team'] for r in rows], np.int64); D['def_team'] = np.array([r['def_team'] for r in rows], np.int64)
    D['desc'] = np.array([[ds.get(str(p), [0.0] * 11) for p in row] for row in D['off']], np.float32)
    D['ts'] = np.array([M['ts'].get((int(t), season), 0) for t in D['def_team']])      # 0 = untrained row (zeros)
    D['tm'] = np.array([M['tm'].get(int(t), 0) for t in D['def_team']])
    D['df'] = np.vectorize(lambda p: M['df'].get(int(p), 0))(D['deff']) if N else np.zeros((0, 5), int)
    D['gid'] = np.array(gids); D['date'] = np.array(dates)
    return D


def inputs(gids, track_root):
    """Input-only comparison with the training season 2026-27 (no model output)."""
    tr = np.load(W / 'traj_2027.npz')

    def summ(gid_arr, L, hold, sc, pos):
        n = len(L); g = len(set(gid_arr.tolist()))
        m = np.arange(60)[None] < L[:, None]
        return dict(games=g, possessions=n, poss_per_game=round(n / max(g, 1), 1), mean_frames=round(float(L.mean()), 1) if n else None,
                    holder_known=round(float((hold[m] >= 0).mean()), 3) if n else None,
                    shot_clock_known=round(float(np.isfinite(sc.astype(float))[m].mean()), 3) if n else None,
                    all_ten_located=round(float(np.isfinite(pos.astype(float)).all(-1).all(-1)[m].mean()), 3) if n else None)
    out = dict(training_2027=summ(tr['gid'], tr['length'], tr['holder'], tr['sc'], tr['pos']), live={})
    desc = json.load(open(FROZEN / 'bowen_v3_hier/desc.json'))
    wm = module(VARIANTS['V3'][1]); _, M = maps(wm)
    G = [live_possessions(g, track_root) for g in gids]
    D = dataset(G, desc, M)
    out['live'] = summ(D['gid'], D['length'], D['holder'], D['sc'], D['pos'])
    out['live']['defenders_with_trained_embedding'] = round(float((D['df'] > 0).mean()), 3) if len(D['df']) else None
    out['live']['attackers_with_descriptor'] = round(float((np.abs(D['desc']).sum(-1) > 0).mean()), 3) if len(D['desc']) else None
    out['live']['team_season_known'] = round(float((D['ts'] > 0).mean()), 3) if len(D['ts']) else None
    print(json.dumps(out, indent=1))
    return out


def eligible_games(track_root, min_poss=30):
    sch = schedule(); out, skipped = [], {}
    for gid, g in sorted(sch.items(), key=lambda x: (x[1]['date'], x[0])):
        if g['date'] < FIRST_DATE:
            continue
        f = Path(track_root) / gid / 'tracking.jsonl.gz'
        if not f.exists():
            continue
        g_, rows = live_possessions(gid, track_root)
        if len(rows) < min_poss:
            skipped[gid] = f'{len(rows)} possessions'; continue
        out.append((g_, rows))
    return out, skipped


def readouts(track_root):
    """Ensemble predictions of every model on every eligible live game (live frames, all ten located)."""
    desc = json.load(open(FROZEN / 'bowen_v3_hier/desc.json'))
    G, skipped = eligible_games(track_root)
    R, base = {}, None
    for v, (d, env) in VARIANTS.items():
        wm = module(env); _, M = maps(wm); N = nets(wm, v, M)
        D = dataset(G, desc, M); idx = np.arange(len(D['gid']))
        P = [run(wm, n, D, idx) for n in N]
        assert all(np.array_equal(P[0][c], P[1][c]) for c in ('poss', 't', 'k', 'live'))
        ens = lambda c: -np.logaddexp(-P[0][c].astype(np.float64), -P[1][c].astype(np.float64)) + math.log(2)
        spot = lambda c: (P[0][c].astype(np.float64) + P[1][c].astype(np.float64)) / 2
        reads = {'scheme': ('ms', 'nll_s'), 'league': ('mb', 'nll_b')}
        if 'mp' in P[0]:
            reads['scheme_plus_player'] = ('mp', 'nll_p')
        for r, (mc, nc) in reads.items():
            R[f'{v}:{r}'] = dict(spot=spot(mc), nll=ens(nc))
        if base is None:
            base = dict(y=P[0]['y'].astype(np.float64), live=P[0]['live'].astype(bool), gid=D['gid'][P[0]['poss']], key=(P[0]['poss'], P[0]['t'], P[0]['k']))
        else:
            assert all(np.array_equal(a, b) for a, b in zip(base['key'], (P[0]['poss'], P[0]['t'], P[0]['k'])))
    return R, base, G, skipped


def score(track_root):
    R, b, G, skipped = readouts(track_root)
    lv = b['live']; gid = b['gid'][lv]; ug, gi = np.unique(gid, return_inverse=True)
    res = {}
    for m, r in R.items():
        e = np.hypot(*(b['y'] - r['spot']).T)[lv]
        res[m] = dict(median=round(float(np.median(e)), 2), p90=round(float(np.percentile(e, 90)), 2),
                      share_over_15ft=round(float((e > 15).mean()), 4), mean_nll=round(float(r['nll'][lv].mean()), 4))
    # per-game sums for the paired game bootstrap (frame-weighted mean, as in the retrospective evaluation)
    cnt = np.bincount(gi)
    gs = {m: np.bincount(gi, weights=r['nll'][lv]) for m, r in R.items()}
    rng = np.random.default_rng(0); B = rng.integers(0, len(ug), (2000, len(ug)))
    comps = {'V3:scheme_plus_player - B_live_mix3:scheme': ('V3:scheme_plus_player', 'B_live_mix3:scheme'),   # primary
             'V3:scheme - B_live_mix3:scheme': ('V3:scheme', 'B_live_mix3:scheme'),
             'V3:scheme_plus_player - V3:scheme': ('V3:scheme_plus_player', 'V3:scheme'),
             'B_live_mix3:scheme - A_live:scheme': ('B_live_mix3:scheme', 'A_live:scheme'),
             'B_live_mix3:scheme - B_live_mix3:league': ('B_live_mix3:scheme', 'B_live_mix3:league')}
    diffs = {}
    for name, (a, c) in comps.items():
        d = gs[a] - gs[c]; pt = d.sum() / cnt.sum()
        bs = d[B].sum(1) / cnt[B].sum(1)
        diffs[name] = dict(diff=round(float(pt), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
    out = dict(label='prospective (games after the frozen models were trained and the protocol was committed)',
               games=int(len(ug)), first=min(g['date'] for g, _ in G), last=max(g['date'] for g, _ in G),
               live_frames=int(lv.sum()), skipped=skipped, results=res, differences=diffs,
               weights={f'{d}/world_fold{f}.pt': sha(FROZEN / d / f'world_fold{f}.pt') for d, _ in VARIANTS.values() for f in (0, 1)})
    json.dump(out, open(W / 'bowen_prospective/score.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'score':
        if not (W / 'bowen_prospective/gate.json').exists() or not json.load(open(W / 'bowen_prospective/gate.json'))['ok']:
            sys.exit('gate has not passed')
        score(sys.argv[sys.argv.index('--live') + 1] if '--live' in sys.argv else LIVE_REPO / 'data/tracking')
    if cmd == 'gate':
        sys.exit(0 if gate() else 1)
    if cmd == 'inputs':
        inputs(sys.argv[2:], LIVE_REPO / 'data/tracking')
