"""Movement fingerprints: how much does knowing WHO a player is help reconstruct HOW he moves?

Two masked-reconstruction encoders trained identically on games before TRAIN_BEFORE (jepa.py, 40 epochs):
  mae_long  scene only;   mae_id  scene + learned player-identity embeddings.
Test windows (games on/after TRAIN_BEFORE, never trained on): hide one player for the whole 4 s window
(all 20 frames) and reconstruct his path from the other nine players and the ball. Error = mean distance (ft)
between reconstructed and true positions. Identity gain = error(mae_long) - error(mae_id), per window.
Per player (>= MIN_N held-out windows): mean gain on offense and on defense, split-half reliability
(odd vs even test games), and nearest neighbours of the identity embeddings ("moves like").
Output: results/jepa/mae_ident.json, mae_ident_windows.npz.   Usage (PC): python mae_ident.py
"""
import json, os
import numpy as np
import torch

import jepa as jp

TRAIN_BEFORE = os.environ.get('TRAIN_BEFORE', '2026-04-01')
MIN_N = 150


@torch.no_grad()
def recon_error(net, X, ids, slot, use_id, bs=512):
    """mean position error (ft) over the 20 frames when player `slot` is hidden, for each window"""
    keep = np.ones((20, 11), bool); keep[:, slot] = False
    keep_t = torch.from_numpy(keep.ravel()).to(jp.DEV); tg = ~keep_t
    ti = torch.arange(20, device=jp.DEV).repeat_interleave(11)[tg]; ai = torch.arange(11, device=jp.DEV).repeat(20)[tg]
    out = np.zeros(len(X), np.float32)
    for i in range(0, len(X), bs):
        b = np.arange(i, min(i + bs, len(X)))
        x, pid = jp.to_dev(X, ids, b, False)
        pid = pid if use_id else None
        ctx = net.encode(x, pid, keep_t)
        q = net.emb.query(len(b), ti, ai, pid) + net.mask_tok
        pr = net.head(net.pred(torch.cat([ctx, q], 1))[:, ctx.shape[1]:])
        tgt = x.flatten(1, 2)[:, tg]
        d = ((pr[..., :2] - tgt[..., :2]) * jp.SCALE[:2].to(jp.DEV)).norm(dim=-1)  # ft
        out[b] = d.mean(1).cpu().numpy()
    return out


def main():
    data = jp.load()
    n_players = len(data['pids'])
    nets = {}
    for v in ('mae_long', 'mae_id'):
        net = jp.Net(n_players, v).to(jp.DEV); net.load_state_dict(torch.load(jp.J / f'{v}.pt', map_location=jp.DEV)); net.eval()
        nets[v] = net
    te = np.where(data['dates'][data['game']] >= TRAIN_BEFORE)[0]
    X, ids, game = data['X'][te], data['ids'][te], data['game'][te]
    E = {v: np.zeros((len(te), 10), np.float32) for v in nets}
    for slot in range(10):
        for v, net in nets.items():
            E[v][:, slot] = recon_error(net, X, ids, slot, v == 'mae_id')
        print(f'slot {slot}: error scene-only {E["mae_long"][:, slot].mean():.2f} ft, with identity {E["mae_id"][:, slot].mean():.2f} ft', flush=True)
    gain = E['mae_long'] - E['mae_id']
    res = {'n_windows': int(len(te)),
           'error_ft': {v: dict(offense=float(e[:, :5].mean()), defense=float(e[:, 5:].mean())) for v, e in E.items()},
           'gain_ft': dict(offense=float(gain[:, :5].mean()), defense=float(gain[:, 5:].mean()))}
    # per player
    rows = {}
    for side, sl in (('off', slice(0, 5)), ('def', slice(5, 10))):
        pid = ids[:, sl].ravel(); g = gain[:, sl].ravel(); e0 = E['mae_long'][:, sl].ravel()
        gm = np.repeat(game, 5)
        for p in np.unique(pid):
            m = pid == p
            if m.sum() < MIN_N:
                continue
            odd = m & (gm % 2 == 1); even = m & (gm % 2 == 0)
            r = rows.setdefault(int(data['pids'][p]), {})
            r[side] = dict(n=int(m.sum()), gain=float(g[m].mean()), base_err=float(e0[m].mean()),
                           half=[float(g[odd].mean()) if odd.any() else None, float(g[even].mean()) if even.any() else None])
    for side in ('off', 'def'):
        h = np.array([r[side]['half'] for r in rows.values() if side in r and None not in r[side]['half']], float)
        res[f'split_half_r_{side}'] = float(np.corrcoef(h[:, 0], h[:, 1])[0, 1]) if len(h) > 5 else None
    # identity embeddings: nearest neighbours
    emb = np.load(jp.J / 'player_emb_mae_id.npy')
    have = [i for i, p in enumerate(data['pids']) if int(p) in rows]
    Z = emb[have]; Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    S = Z @ Z.T; np.fill_diagonal(S, -1)
    nn = {int(data['pids'][have[i]]): [int(data['pids'][have[j]]) for j in np.argsort(-S[i])[:5]] for i in range(len(have))}
    res['players'] = rows; res['neighbours'] = nn
    print(f"\nheld-out windows {len(te)}: identity gain offense {res['gain_ft']['offense']:.3f} ft, defense {res['gain_ft']['defense']:.3f} ft")
    print(f"split-half r of per-player gain: offense {res['split_half_r_off']}, defense {res['split_half_r_def']}")
    json.dump(res, open(jp.J / 'mae_ident.json', 'w'), indent=1)
    np.savez(jp.J / 'mae_ident_windows.npz', err_scene=E['mae_long'], err_id=E['mae_id'], ids=ids, game=game)


if __name__ == '__main__':
    main()
