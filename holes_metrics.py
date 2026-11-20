"""Per-game, per-player sums from the holes field (results/holes/*.npz) -> results/holes/player_game.pkl.

Offense (frames on court on offense):  h_off_n, h_open (excess space within 5 ft), h_reach (space only they reach
  first), h_hat (field value at their spot), h_teamA (team A); on the ball: h_hold_n, h_hold_A, and per touch
  (same holder, contiguous) h_touches, h_create (A at the end of the touch minus A at its start).
Defense: h_def_n, h_cover (space they currently deny), h_oppA (opponent A); guarding the holder (nearest defender
  to the ball handler): h_onball_n, h_onball_A.
metrics2.describe2 adds these to the event counts, so they flow into the descriptions and embeddings.
"""
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from masked_paths import ROOT

H = ROOT / 'results/holes'


def game(f):
    d = np.load(f)
    meta = d['meta']; ids = meta[:, 5:15]; hold = meta[:, 2]; t = meta[:, 1]; k = meta[:, 0]
    A = d['A'].astype(float); op = d['open_i'].astype(float); rc = d['off_reach_i'].astype(float)
    ha = d['H_at_i'].astype(float); cov = d['def_cover_j'].astype(float); pos = d['pos'].astype(float)
    S = defaultdict(lambda: defaultdict(float))
    for j in range(5):
        for p, a, o, r, h in zip(ids[:, j], A, op[:, j], rc[:, j], ha[:, j]):
            s = S[int(p)]; s['h_off_n'] += 1; s['h_open'] += o; s['h_reach'] += r; s['h_hat'] += h; s['h_teamA'] += a
        for p, a, c in zip(ids[:, 5 + j], A, cov[:, j]):
            s = S[int(p)]; s['h_def_n'] += 1; s['h_cover'] += c; s['h_oppA'] += a
    # holder and on-ball defender
    for i in range(len(A)):
        hp = int(hold[i])
        if hp == 0 or hp not in ids[i, :5]:
            continue
        S[hp]['h_hold_n'] += 1; S[hp]['h_hold_A'] += A[i]
        hj = int(np.where(ids[i, :5] == hp)[0][0])
        dj = int(np.argmin(np.linalg.norm(pos[i, 5:] - pos[i, hj], axis=1)))
        q = int(ids[i, 5 + dj]); S[q]['h_onball_n'] += 1; S[q]['h_onball_A'] += A[i]
    # touches: contiguous frames with the same holder in the same possession (gaps <= 400 ms)
    i = 0
    while i < len(A):
        hp = int(hold[i])
        if hp == 0 or hp not in ids[i, :5]:
            i += 1; continue
        j = i
        while j + 1 < len(A) and hold[j + 1] == hp and k[j + 1] == k[i] and t[j + 1] - t[j] <= 400:
            j += 1
        if t[j] - t[i] >= 400:
            S[hp]['h_touches'] += 1; S[hp]['h_create'] += A[j] - A[i]
        i = j + 1
    return f.stem, {p: dict(v) for p, v in S.items()}


if __name__ == '__main__':
    fs = sorted(f for f in H.glob('*.npz'))
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(game, fs, chunksize=4))
    pickle.dump(res, open(H / 'player_game.pkl', 'wb'))
    print('games', len(res))
