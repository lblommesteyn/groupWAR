"""Apply the long-trained MAE encoder (results/jepa/mae_long.pt) to shots and players.

Shots: results/jepa/shot_windows.npz (shot_windows.py) -> results/jepa/shot_emb.npz  E [N, 384] =
  [mean latent of the release frame, mean over all tokens, shooter token at release]; windows shorter than 10
  frames are encoded with their true frame indices (last frame = index 9).
Players: every window in windows.npz; each attacker / defender token latent averaged over the 10 context frames,
  projected to the top K principal directions (fit on windows of games before TRAIN_BEFORE, separately for offense
  and defense), summed per (game, player) -> results/jepa/player_game.pkl {gid: {pid: {mae_o0.., mae_off_n, ..}}}
  in the event-count format metrics2 merges (MAE=1).
Usage (PC): python mae_apply.py
"""
import os, pickle
import numpy as np
import torch

import jepa as jp

K = 8
TRAIN_BEFORE = os.environ.get('TRAIN_BEFORE', '2026-04-01')


def load_net(n_players):
    net = jp.Net(n_players, 'mae_long').to(jp.DEV)
    net.load_state_dict(torch.load(jp.J / 'mae_long.pt', map_location=jp.DEV)); net.eval()
    return net


@torch.no_grad()
def enc(net, x, t0):
    """x [B, T, 11, 4] normalised; frames carry time indices t0 .. t0+T-1 -> [B, T, 11, D]"""
    T = x.shape[1]
    h = net.emb(x, None, torch.arange(t0, t0 + T, device=x.device)).flatten(1, 2)
    return net.norm(net.enc(h)).view(x.shape[0], T, 11, jp.D)


def shots(net):
    s = np.load(jp.J / 'shot_windows.npz')
    X, L, ss = s['X'], s['L'].astype(int), s['shooter_slot'].astype(int)
    E = np.zeros((len(X), 3 * jp.D), np.float32)
    for l in np.unique(L):
        rows = np.where(L == l)[0]
        for i in range(0, len(rows), 1024):
            b = rows[i:i + 1024]
            x = (torch.from_numpy(X[b, 10 - l:].astype(np.float32)) / jp.SCALE).to(jp.DEV)
            h = enc(net, x, 10 - l)
            last = h[:, -1]
            sh = last[torch.arange(len(b)), torch.from_numpy(ss[b]).to(jp.DEV)]
            E[b] = torch.cat([last.mean(1), h.mean((1, 2)), sh], 1).cpu().numpy()
    np.savez(jp.J / 'shot_emb.npz', E=E, game_id=s['game_id'], event_id=s['event_id'], L=L)
    print('shot embeddings', E.shape, flush=True)


def players(net, data):
    X, ids, game = data['X'], data['ids'], data['game']
    tok = np.zeros((len(X), 10, jp.D), np.float32)
    for i in range(0, len(X), 1024):
        b = np.arange(i, min(i + 1024, len(X)))
        x = (torch.from_numpy(X[b, :10].astype(np.float32)) / jp.SCALE).to(jp.DEV)
        tok[b] = enc(net, x, 0)[:, :, :10].mean(1).cpu().numpy()
    tr = data['dates'][game] < TRAIN_BEFORE
    out = {}
    for side, sl in (('o', slice(0, 5)), ('d', slice(5, 10))):
        T = tok[:, sl].reshape(-1, jp.D)
        mu = T[np.repeat(tr, 5)].mean(0)
        _, sv, Vt = np.linalg.svd(T[np.repeat(tr, 5)][::7] - mu, full_matrices=False)
        print(side, 'explained', (sv[:K] ** 2 / (sv ** 2).sum()).round(3), flush=True)
        P = ((T - mu) @ Vt[:K].T).reshape(len(X), 5, K)
        pid = ids[:, sl]
        for w in range(len(X)):
            g = str(data['games'][game[w]])
            for a in range(5):
                c = out.setdefault(g, {}).setdefault(int(data['pids'][pid[w, a]]), {})
                for k in range(K):
                    c[f'mae_{side}{k}'] = c.get(f'mae_{side}{k}', 0.0) + float(P[w, a, k])
                c[f'mae_{"off" if side == "o" else "def"}_n'] = c.get(f'mae_{"off" if side == "o" else "def"}_n', 0) + 1
    pickle.dump(out, open(jp.J / 'player_game.pkl', 'wb'))
    print('player-games', sum(len(v) for v in out.values()), flush=True)


def main():
    data = jp.load()
    net = load_net(len(data['pids']))
    shots(net)
    players(net, data)


if __name__ == '__main__':
    main()
