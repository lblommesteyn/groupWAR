"""Per-(game, player) raw counts for the embedding experiment.

Outputs results/embed/player_games.pkl: {game_id: dict(date, stints, box, trk, profile)}
and results/embed/shots_xfg.parquet-ish jsonl with cross-fitted xFG per attempt.

Everything is stored as additive sums so any date window can be aggregated later.
"""
import gzip, json, pickle, re, zipfile
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from masked_paths import ROOT, ARCHIVE
from shot_align import OUT as SHOTS, HOOP_X, load_frames
from shot_model import ACTIONS, SUBTYPES, TRK, SPLIT
import rapm

OUT = ROOT / 'results/embed'
PREFIX = 'NBA-2025-26-all-Tracking/games/'
HBINS = 11  # 3-ft bins of distance to hoop, last bin open


# ---------------------------------------------------------------- xFG (cross-fitted)
def loc_x(r):
    d = r['desc'].lower()
    return [np.hypot(r['pbp_x'], r['pbp_y']), abs(r['pbp_x']), r['pbp_y'], r['three']] + \
           [r['subtype'] == s for s in SUBTYPES] + [a in d for a in ACTIONS]


def hgb():
    return HistGradientBoostingClassifier(learning_rate=0.08, max_iter=400, max_leaf_nodes=15,
                                          min_samples_leaf=80, l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=30, random_state=0)


def crossfit(X, y, g, pre):
    """OOF predictions for pre-split rows (5 game folds); post-split rows use a model fit on all pre rows."""
    p = np.full(len(y), np.nan)
    idx = np.flatnonzero(pre)
    for tr, te in GroupKFold(5).split(idx, groups=g[idx]):
        p[idx[te]] = hgb().fit(X[idx[tr]], y[idx[tr]]).predict_proba(X[idx[te]])[:, 1]
    post = np.flatnonzero(~pre)
    if len(post):
        p[post] = hgb().fit(X[idx], y[idx]).predict_proba(X[post])[:, 1]
    return p


def build_xfg():
    allr = [json.loads(l) for l in open(SHOTS / 'alignment.jsonl')]
    feat = {(r['game_id'], r['event_id']): r for r in map(json.loads, open(SHOTS / 'features.jsonl'))}
    y = np.array([r['made'] for r in allr], float)
    g = np.array([r['game_id'] for r in allr])
    pre = np.array([r['date'] < SPLIT for r in allr])
    XA = np.array([loc_x(r) for r in allr], float)
    xa = crossfit(XA, y, g, pre)
    cols = sum(TRK.values(), [])
    ok = np.array([bool(feat.get((r['game_id'], r['event_id']), {}).get('trk_ok')) for r in allr])
    XB = np.array([loc_x(r) + [feat[(r['game_id'], r['event_id'])].get(c, np.nan) for c in cols]
                   for r, o in zip(allr, ok) if o], float)
    xb = np.full(len(allr), np.nan)
    xb[ok] = crossfit(XB, y[ok], g[ok], pre[ok])
    rows = []
    for r, a, b, o in zip(allr, xa, xb, ok):
        f = feat.get((r['game_id'], r['event_id']), {})
        rows.append(dict(game_id=r['game_id'], date=r['date'], shooter=r['shooter'], made=r['made'],
                         three=r['three'], dist=float(np.hypot(r['pbp_x'], r['pbp_y'])), xa=float(a),
                         xb=float(b) if o else None, trk_ok=bool(o),
                         **({k: f.get(k) for k in ['d1', 'touch_s', 'shooter_speed', 'nd_id', 'def_ids',
                                                'def_dists', 'rim_def_ids', 'shooter_side', 'hoop_sign',
                                                'period']} if o else {})))
    with open(OUT / 'shots_xfg.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print('xfg done', len(rows), 'trk', int(ok.sum()))
    return rows


# ---------------------------------------------------------------- per game
def secs(iso):
    m = re.match(r'PT(\d+)M([\d.]+)S', iso or '')
    return int(m.group(1)) * 60 + float(m.group(2)) if m else 0.0


def box_and_profile(z, base):
    stats, prof, names, pbp = {}, {}, {}, {}
    for line in gzip.open(z.open(base + 'raw/stats.jsonl.gz'), 'rt'):
        r = json.loads(line); e = r['event']; m = r['metadata']
        if e.startswith('PlayerStats') and e.endswith('#show') and 'personId' in m:
            k = (r['timestamp'], r['updatedTime'])
            if m['personId'] not in stats or k >= stats[m['personId']][0]:
                stats[m['personId']] = (k, m)
        elif e == 'PlayerProfile#show':
            for pid, p in (m.get('players') or {}).items():
                prof[int(pid)] = dict(ht=p.get('ht'), wt=p.get('wt'), pos=p.get('pos'))  # no season stats (leak)
        elif e == 'Teams#show':
            for side in ('home', 'away'):
                for p in (m.get(side) or {}).get('players', []):
                    names[(side, p['nameI'])] = p['personId']
        elif e == 'PlayByPlay#show':
            pbp[m['id']] = m
    box = {}
    for pid, (_, m) in stats.items():
        box[pid] = dict(sec=secs(m.get('minIso')) if m.get('minIso') else 60.0 * float(m.get('min') or 0), fga=m.get('fga', 0), fgm=m.get('fgm', 0), tpa=m.get('tpa', 0),
                        tpm=m.get('tpm', 0), fta=m.get('fta', 0), ftm=m.get('ftm', 0), pts=m.get('pts', 0),
                        ast=m.get('ast', 0), oreb=m.get('oreb', 0), dreb=m.get('dreb', 0), stl=m.get('stl', 0),
                        blk=m.get('blk', 0), tov=m.get('tov', 0), pf=m.get('pf', 0),
                        rim_a=0, rim_m=0, smid_a=0, lmid_a=0, c3_a=0, c3_m=0, a3_a=0, a3_m=0, ast3=0, astrim=0)
    for m in pbp.values():
        if m['eventType'] not in ('2pt', '3pt') or m['personId'] not in box:
            continue
        b = box[m['personId']]; made = 'MISS' not in m['description']
        d = np.hypot(m['locX'], m['locY']) / 10; three = m['eventType'] == '3pt'
        if three:
            z3 = 'c3' if m['locY'] / 10 < 8.75 else 'a3'
            b[z3 + '_a'] += 1; b[z3 + '_m'] += made
        elif d < 4:
            b['rim_a'] += 1; b['rim_m'] += made
        elif d < 14:
            b['smid_a'] += 1
        else:
            b['lmid_a'] += 1
        a = re.search(r'\(([^()]+?) \d+ AST\)', m['description'])
        if made and a:
            ap = names.get((m.get('team'), a.group(1).strip()))
            if ap in box:
                box[ap]['ast3'] += three; box[ap]['astrim'] += (not three and d < 4)
    return box, prof


def attack_dirs(shots):
    """(period, side) -> hoop sign, from aligned shots; fill missing periods from same half."""
    v = defaultdict(Counter)
    for s in shots:
        v[(s['period'], s['shooter_side'])][s['hoop_sign']] += 1
    out = {k: c.most_common(1)[0][0] for k, c in v.items()}
    for p in range(1, 9):
        for side in ('home', 'away'):
            if (p, side) not in out:
                other = 'away' if side == 'home' else 'home'
                for q in ([3 - p] if p <= 2 else [x for x in range(3, 9) if x != p]):
                    if (q, side) in out:
                        out[(p, side)] = out[(q, side)]; break
                else:
                    if (p, other) in out:
                        out[(p, side)] = -out[(p, other)]
    return out


def frame_pass(z, base, dirs):
    T, P, B, players, on = load_frames(z, base)
    off = defaultdict(lambda: dict(n=0, onball=0, speed=0.0, nspeed=0, ball_dist=0.0,
                                   cnt=np.zeros(HBINS), dsum=np.zeros(HBINS)))
    de = defaultdict(lambda: dict(n=0, dmatch=0.0, hoop=0.0, paint=0, speed=0.0, nspeed=0, onball=0))
    pairs = Counter()
    poss_team, poss_t = None, -1e18
    for i in range(0, len(T)):
        h, a = on[i]
        if len(h) != 5 or len(a) != 5 or not np.isfinite(B[i, 0]):
            continue
        pos = players[i]
        vis = [p for p in h + a if p in pos]
        if vis:
            dist = [np.hypot(*(np.array(pos[p]) - B[i, :2])) for p in vis]
            j = int(np.argmin(dist))
            if dist[j] <= 4:
                poss_team, poss_t = ('home' if vis[j] in h else 'away'), T[i]
        if i % 5 or poss_team is None or T[i] - poss_t > 3000:
            continue
        offense, defense = (h, a) if poss_team == 'home' else (a, h)
        sgn = dirs.get((P[i], poss_team))
        if sgn is None or B[i, 0] * sgn < 5 or not all(p in pos for p in offense + defense):
            continue
        hoop = np.array([sgn * HOOP_X, 0.0])
        O = np.array([pos[p] for p in offense]); D = np.array([pos[p] for p in defense])
        bdist = np.linalg.norm(O - B[i, :2], axis=1); hb = int(np.argmin(bdist))
        od = np.linalg.norm(O[:, None] - D[None], axis=2)
        prev = i - 5 if i >= 5 and T[i] - T[i - 5] <= 1100 and P[i - 5] == P[i] else None
        for k, p in enumerate(offense):
            s = off[p]
            if k == hb:
                s['onball'] += 1; continue
            s['n'] += 1
            hd = np.linalg.norm(O[k] - hoop); b = min(int(hd / 3), HBINS - 1)
            s['cnt'][b] += 1; s['dsum'][b] += od[k].min(); s['ball_dist'] += bdist[k]
            if prev is not None and p in players[prev]:
                s['speed'] += np.linalg.norm(O[k] - players[prev][p]); s['nspeed'] += 1
        for k, q in enumerate(defense):
            s = de[q]; m = int(np.argmin(od[:, k]))
            s['n'] += 1; s['dmatch'] += od[m, k]; s['hoop'] += np.linalg.norm(D[k] - hoop)
            s['paint'] += int(abs(D[k, 0] - hoop[0]) < 19 and abs(D[k, 1]) < 8); s['onball'] += int(m == hb)
            pairs[(q, offense[m])] += 1
            if prev is not None and q in players[prev]:
                s['speed'] += np.linalg.norm(D[k] - players[prev][q]); s['nspeed'] += 1
    return {k: dict(v) for k, v in off.items()}, {k: dict(v) for k, v in de.items()}, dict(pairs)


def game(task):
    base, shots = task
    out = rapm.parse_game(base)
    with zipfile.ZipFile(ARCHIVE) as z:
        box, prof = box_and_profile(z, base)
        dirs = attack_dirs(shots)
        off, de, pairs = frame_pass(z, base, dirs)
    return out['game_id'], dict(date=out['date'], stints=out['stints'], home=out['home'], away=out['away'],
                                final=out['final'], players=out['players'], box=box, profile=prof,
                                trk_off=off, trk_def=de, pairs=pairs)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    shots = build_xfg()
    by = defaultdict(list)
    for s in shots:
        if s['trk_ok']:
            by[s['game_id']].append(s)
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = sorted(n.rsplit('tracking.jsonl.gz', 1)[0] for n in zz.namelist() if n.endswith('tracking.jsonl.gz'))
    tasks = [(b, by.get(b.rstrip('/').rsplit('/', 1)[1], [])) for b in bases]
    res = {}
    with ProcessPoolExecutor(10) as ex:
        for i, (gid, d) in enumerate(ex.map(game, tasks, chunksize=2), 1):
            res[gid] = d
            if i % 50 == 0:
                print(i, 'games', flush=True)
    pickle.dump(res, open(OUT / 'player_games.pkl', 'wb'))
    print('done', len(res))
