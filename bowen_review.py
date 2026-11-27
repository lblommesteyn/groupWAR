"""BOWEN human-review set: ~100 stratified, BLINDED half-court possessions with tracking replays.

What BOWEN outputs (labels used from now on): EXPECTED POSITIONING (the scheme-conditioned spot mu_scheme and the
league-baseline spot mu_league for defender k, from a network that sees all other players, the ball and the clock but
not k) and POSITIONING DEVIATION (k's actual spot minus the expected spot). Whether a deviation is an execution error,
an acceptable alternative, a deliberate assignment, or a tendency is exactly what this review is for.

Sampling (out-of-fold predictions only, results/world/pred.npz cross-fit by game):
  eligible   half-court possessions >= 15 frames (6 s) with all ten players located
  situation  from the defender's tracking features on that possession (results/archetypes/poss_*.npz):
             on-ball (on_handler >= 0.4), interior (paint >= 0.5), perimeter stay-home (mean distance to rim >= 22 ft),
             help / rotation (the rest)
  deviation  tertile of the defender's mean |actual - expected (scheme)| within his situation: low / ordinary / high
  8 per situation x tertile cell (96) + 4 random ordinary possessions = 100; both seasons; one defender per possession
Outputs
  results/bowen_review/sample.json   what reviewers see: replay frames (players, ball, holder), highlighted defender,
                                     period, shot clock. NO BOWEN values, no stratum, no game id or date.
  results/bowen_review/key.json      PRIVATE (never published): item -> possession, defender, situation, tertile,
                                     deviation summaries, expected spots per frame.
  python bowen_review.py   (Mac)
"""
import hashlib, json
import numpy as np

from masked_paths import ROOT
from provenance import provenance

W = ROOT / 'results/world'
OUT = ROOT / 'results/bowen_review'
RIM = np.array([41.75, 0.0])
PER_CELL = 8


def main():
    rng = np.random.default_rng(20261127)
    z = np.load(W / 'pred.npz')
    T = [np.load(W / f'traj_{s}.npz') for s in (2026, 2027)]
    P = [np.load(ROOT / f'results/archetypes/poss_{s}.npz', allow_pickle=True) for s in (2026, 2027)]
    pos = np.concatenate([t['pos'] for t in T]); ball = np.concatenate([t['ball'] for t in T]); hold = np.concatenate([t['holder'] for t in T])
    sc = np.concatenate([t['sc'] for t in T]); per = np.concatenate([t['period'] for t in T]); L = np.concatenate([t['length'] for t in T])
    fd = np.concatenate([p['fd'] for p in P]); dfn = list(P[0]['def_f'])
    poss, k, t = z['poss'], z['k'].astype(int), z['t'].astype(int)
    y, ms, mb = z['y'].astype(np.float32), z['ms'].astype(np.float32), z['mb'].astype(np.float32)
    dev = np.hypot(*(y - ms).T)
    key = poss.astype(np.int64) * 5 + k
    uk, inv = np.unique(key, return_inverse=True)
    mdev = np.bincount(inv, dev) / np.bincount(inv); nfr = np.bincount(inv)
    P_, K_ = (uk // 5).astype(int), (uk % 5).astype(int)
    full = nfr >= 15
    f = fd[P_, K_]
    oh, pa, dr = f[:, dfn.index('on_handler')], f[:, dfn.index('paint')], f[:, dfn.index('d_rim')]
    sit = np.where(oh >= 0.4, 'on-ball', np.where(pa >= 0.5, 'interior', np.where(dr >= 22, 'perimeter stay-home', 'help / rotation')))
    picks = []
    for s_ in ('on-ball', 'interior', 'perimeter stay-home', 'help / rotation'):
        cand = np.flatnonzero(full & (sit == s_))
        q = np.quantile(mdev[cand], [1 / 3, 2 / 3])
        terc = np.digitize(mdev[cand], q)
        for tt, lab in enumerate(('low', 'ordinary', 'high')):
            c = cand[terc == tt]; picks += [(i, s_, lab) for i in rng.choice(c, PER_CELL, replace=False)]
    used = {p for p, _, _ in picks}
    extra = [i for i in rng.permutation(np.flatnonzero(full)) if i not in used][:4]
    picks += [(i, str(sit[i]), 'random') for i in extra]
    order = rng.permutation(len(picks))
    sample, keyd = [], {}
    for rank, j in enumerate(order):
        i, s_, lab = picks[j]; p, kk = int(P_[i]), int(K_[i]); n = int(L[p])
        item = hashlib.sha256(f'{p}-{kk}-bowen'.encode()).hexdigest()[:10]
        sample.append(dict(id=item, n=rank + 1, defender=5 + kk, period=int(per[p]),
                           shot_clock=[round(float(x), 1) if np.isfinite(x) else None for x in sc[p, :n]],
                           players=np.round(pos[p, :n].astype(float), 1).tolist(),
                           ball=np.round(ball[p, :n, :2].astype(float), 1).tolist(),
                           holder=[int(h) for h in hold[p, :n]]))
        fr = np.flatnonzero((poss == p) & (k == kk)); order_t = np.argsort(t[fr]); fr = fr[order_t]
        keyd[item] = dict(poss=p, slot=kk, gid=str(z['meta_gid'][p]), date=str(z['meta_date'][p]), season=int(z['meta_season'][p]),
                          defender_pid=int(z['meta_deff'][p, kk]), def_team=int(z['meta_def_team'][p]), situation=s_, stratum=lab,
                          mean_deviation_ft=round(float(mdev[i]), 2),
                          mean_depth_deviation_ft=round(float(np.mean(np.hypot(*(y[fr] - RIM).T) - np.hypot(*(ms[fr] - RIM).T))), 2),
                          mean_scheme_shift_ft=round(float(np.mean(np.hypot(*(ms[fr] - mb[fr]).T))), 2),
                          frames=[int(x) for x in t[fr]], expected_scheme=np.round(ms[fr], 1).tolist(),
                          expected_league=np.round(mb[fr], 1).tolist(), fold=int(z['fold'][p]))
    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(n_items=len(sample), per_cell=PER_CELL, frame_hz=2.5,
                cells={f'{s_}|{lab}': sum(1 for _, a, b in picks if a == s_ and b == lab) for _, s_, lab in picks},
                provenance=provenance(__file__, ['results/world/pred.npz', 'results/world/traj_2026.npz', 'results/world/traj_2027.npz']))
    def clean(o):   # browsers reject NaN in JSON: missing tracking values become null
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, dict):
            return {k_: clean(v) for k_, v in o.items()}
        return o
    json.dump(clean(dict(meta={k_: v for k_, v in meta.items() if k_ != 'cells'}, items=sample)), open(OUT / 'sample.json', 'w'), allow_nan=False)
    json.dump(dict(meta=meta, items=keyd), open(OUT / 'key.json', 'w'))
    print(len(sample), 'items;', meta['cells'])


if __name__ == '__main__':
    main()
