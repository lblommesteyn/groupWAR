"""Defensive world model: where should each defender be, under the league baseline and under his team's scheme?

Data: results/world/traj_{2026,2027}.npz (traj_export.py), desc.json (world_desc.py). Offense attacks (+41.75, 0).
For defender k at frame t the decoder sees everything except k's own position:
  5 attackers (x, y, vx, vy, holder flag, 11-dim descriptor), ball (x, y, z), the other 4 defenders (x, y, vx, vy),
  shot clock and time into the possession, plus a conditioning token = [scheme code z, team-season embedding].
Scheme code z (8 dims) comes from an encoder over the WHOLE possession of the other four defenders and the five
attackers (k excluded, so one player cannot define his own scheme), as a small VAE posterior (KL weight BETA).
Conditioning dropout: with probability P_DROP the conditioning token is replaced by a learned "no scheme" token, so
the same network also predicts the league baseline. Output: 2-D Gaussian (mean, log-variance) for k's position.
Cross-fitting: games split into 2 folds; each fold is predicted by the model trained on the other.
Writes results/world/pred_{season}.npz: for every (possession, frame, defender) with all ten players located:
  actual, scheme-conditioned mean, baseline mean, log-variances, plus z per (possession, defender).
  python world_model.py   (PC GPU; ~1 h)
"""
import json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
W = ROOT / 'results/world'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
D, ZD, TD, BETA, P_DROP = 128, 8, 8, 0.01, 0.3
STEPS, BATCH, FRAMES = 12000, 192, 6
RIM = 41.75


def load():
    desc = json.load(open(W / 'desc.json'))
    parts = []
    for s in (2026, 2027):
        z = dict(np.load(W / f'traj_{s}.npz'))
        ds = desc[str(s)]
        z['desc'] = np.array([[ds.get(str(p), [0.0] * 11) for p in row] for row in z['off']], np.float32)
        z['season'] = np.full(len(z['gid']), s)
        parts.append(z)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ts = sorted({(int(t), int(s)) for t, s in zip(out['def_team'], out['season'])})
    tix = {k: i + 1 for i, k in enumerate(ts)}
    out['ts'] = np.array([tix[(int(t), int(s))] for t, s in zip(out['def_team'], out['season'])])
    return out, len(ts) + 1


class Net(nn.Module):
    def __init__(self, n_ts):
        super().__init__()
        self.att = nn.Linear(4 + 1 + 11, D); self.dfn = nn.Linear(4, D); self.bal = nn.Linear(3, D)
        self.glob = nn.Linear(2, D); self.typ = nn.Embedding(5, D)
        enc = lambda L: nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 4 * D, 0.1, batch_first=True), L)
        self.frame = enc(3)
        self.zenc = enc(2); self.zgru = nn.GRU(D, D, batch_first=True, bidirectional=True)
        self.zhead = nn.Linear(2 * D, 2 * ZD)
        self.tse = nn.Embedding(n_ts, TD); nn.init.zeros_(self.tse.weight)
        self.cond = nn.Linear(ZD + TD, D); self.nocond = nn.Parameter(torch.zeros(D))
        self.out = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 3))

    def tokens(self, pos, vel, ball, hold, desc, others, glob):
        # pos/vel [B, 10, 2]; others: bool mask of defenders to include [B, 5]
        A = torch.cat([pos[:, :5], vel[:, :5], hold.unsqueeze(-1), desc], -1)
        a = self.att(A) + self.typ.weight[0]
        d = self.dfn(torch.cat([pos[:, 5:], vel[:, 5:]], -1)) + self.typ.weight[1]
        b = (self.bal(ball) + self.typ.weight[2]).unsqueeze(1)
        g = (self.glob(glob) + self.typ.weight[3]).unsqueeze(1)
        toks = torch.cat([a, d, b, g], 1)                                   # [B, 12, D]
        pad = torch.cat([torch.zeros_like(others[:, :1]).repeat(1, 5), ~others, torch.zeros_like(others[:, :2])], 1)
        return toks, pad

    def scheme(self, P, V, Bl, H, De, G, L, k):
        # whole possession [B, T, ...]; k = target defender slot [B]
        B_, T = P.shape[:2]
        oth = torch.ones(B_, 5, dtype=torch.bool, device=P.device); oth[torch.arange(B_), k] = False
        o = oth.unsqueeze(1).expand(B_, T, 5).reshape(-1, 5)
        toks, pad = self.tokens(P.reshape(-1, 10, 2), V.reshape(-1, 10, 2), Bl.reshape(-1, 3), H.reshape(-1, 5),
                                De.unsqueeze(1).expand(B_, T, 5, 11).reshape(-1, 5, 11), o, G.reshape(-1, 2))
        h = self.zenc(toks, src_key_padding_mask=pad)
        h = (h * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True)
        h = h.view(B_, T, D)
        packed = nn.utils.rnn.pack_padded_sequence(h, L.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, hn = self.zgru(packed)
        mu, lv = self.zhead(torch.cat([hn[0], hn[1]], -1)).chunk(2, -1)
        return mu, lv.clamp(-6, 3)

    def decode(self, pos, vel, ball, hold, desc, glob, k, cond):
        B_ = pos.shape[0]
        oth = torch.ones(B_, 5, dtype=torch.bool, device=pos.device); oth[torch.arange(B_), k] = False
        toks, pad = self.tokens(pos, vel, ball, hold, desc, oth, glob)
        q = (cond + self.typ.weight[4]).unsqueeze(1)
        h = self.frame(torch.cat([toks, q], 1), src_key_padding_mask=torch.cat([pad, torch.zeros_like(pad[:, :1])], 1))
        o = self.out(h[:, -1])
        return o[:, :2] + torch.tensor([RIM - 12.0, 0.0], device=pos.device), o[:, 2].clamp(-4, 6)


def tens(Dd, idx):
    P = torch.tensor(Dd['pos'][idx].astype(np.float32)); Bl = torch.tensor(Dd['ball'][idx].astype(np.float32))
    V = torch.zeros_like(P); V[:, 1:] = (P[:, 1:] - P[:, :-1]) / 0.4
    H = torch.tensor(Dd['holder'][idx].astype(np.int64))
    Ho = F.one_hot(H.clamp(min=0), 5).float() * (H >= 0).unsqueeze(-1).float()
    T = P.shape[1]; L = torch.tensor(Dd['length'][idx].astype(np.int64))
    sc = torch.tensor(np.nan_to_num(Dd['sc'][idx].astype(np.float32), nan=12.0)) / 24.0
    tt = torch.arange(T).float().unsqueeze(0).expand(len(idx), T) * 0.4 / 24.0
    G = torch.stack([sc, tt], -1)
    ok = torch.isfinite(P).all(-1).all(-1) & (torch.arange(T).unsqueeze(0) < L.unsqueeze(1))
    P = torch.nan_to_num(P); V = torch.nan_to_num(V).clamp(-30, 30); Bl = torch.nan_to_num(Bl)
    De = torch.tensor(Dd['desc'][idx]); TS = torch.tensor(Dd['ts'][idx])
    return [x.to(DEV) for x in (P, V, Bl, Ho, De, G, L, ok, TS)]


def step(net, batch, train=True):
    P, V, Bl, Ho, De, G, L, ok, TS = batch
    B_ = P.shape[0]
    k = torch.randint(0, 5, (B_,), device=DEV)
    mu, lv = net.scheme(P, V, Bl, Ho, De, G, L, k)
    z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if train else mu
    kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1).mean()
    cond = net.cond(torch.cat([z, net.tse(TS)], -1))
    if train:
        drop = torch.rand(B_, device=DEV) < P_DROP
        cond = torch.where(drop.unsqueeze(-1), net.nocond.expand_as(cond), cond)
    # sample FRAMES valid frames per possession
    w = ok.float() + 1e-6
    ti = torch.multinomial(w, FRAMES, replacement=True)                  # [B, FRAMES]
    bi = torch.arange(B_, device=DEV).unsqueeze(1).expand_as(ti)
    sel = lambda X: X[bi, ti].reshape(-1, *X.shape[2:])
    kk = k.unsqueeze(1).expand_as(ti).reshape(-1)
    m, lvar = net.decode(sel(P), sel(V), sel(Bl), sel(Ho), De.unsqueeze(1).expand(B_, FRAMES, 5, 11).reshape(-1, 5, 11), sel(G), kk,
                         cond.unsqueeze(1).expand(B_, FRAMES, D).reshape(-1, D))
    y = sel(P)[torch.arange(len(kk), device=DEV), 5 + kk]
    valid = sel(ok.float())
    nll = ((((y - m) ** 2).sum(-1) / (2 * lvar.exp()) + lvar) * valid).sum() / valid.sum().clamp(min=1)
    return nll + BETA * kl, nll.item(), kl.item()


@torch.no_grad()
def predict(net, Dd, idx):
    """For every possession in idx, every valid frame and defender: scheme-conditioned and baseline predictions."""
    net.eval(); out = []; zs = []
    for s in range(0, len(idx), 64):
        b = idx[s:s + 64]
        P, V, Bl, Ho, De, G, L, ok, TS = tens(Dd, b)
        B_, T = P.shape[:2]
        for k in range(5):
            kk = torch.full((B_,), k, device=DEV)
            mu, _ = net.scheme(P, V, Bl, Ho, De, G, L, kk)
            c_s = net.cond(torch.cat([mu, net.tse(TS)], -1)); c_b = net.nocond.expand_as(c_s)
            fl = lambda X: X.reshape(B_ * T, *X.shape[2:])
            kf = kk.unsqueeze(1).expand(B_, T).reshape(-1)
            Dx = De.unsqueeze(1).expand(B_, T, 5, 11).reshape(-1, 5, 11)
            ms, ls = net.decode(fl(P), fl(V), fl(Bl), fl(Ho), Dx, fl(G), kf, c_s.unsqueeze(1).expand(B_, T, D).reshape(-1, D))
            mb, lb = net.decode(fl(P), fl(V), fl(Bl), fl(Ho), Dx, fl(G), kf, c_b.unsqueeze(1).expand(B_, T, D).reshape(-1, D))
            okf = fl(ok)
            y = fl(P)[:, 5 + k]
            sel = okf.nonzero().squeeze(-1)
            zs.append((b, k, mu.cpu().numpy().astype(np.float16)))
            out.append(dict(poss=np.repeat(b, T)[sel.cpu().numpy()].astype(np.int32), t=np.tile(np.arange(T), B_)[sel.cpu().numpy()].astype(np.int8), k=np.full(len(sel), k, np.int8),
                            y=y[sel].cpu().numpy().astype(np.float16), ms=ms[sel].cpu().numpy().astype(np.float16), ls=ls[sel].cpu().numpy().astype(np.float16),
                            mb=mb[sel].cpu().numpy().astype(np.float16), lb=lb[sel].cpu().numpy().astype(np.float16)))
    res = {key: np.concatenate([o[key] for o in out]) for key in out[0]}
    Z = np.zeros((len(Dd['gid']), 5, ZD), np.float16)
    for b, k, z in zs:
        Z[b, k] = z
    return res, Z


def main():
    torch.manual_seed(0); np.random.seed(0)
    Dd, n_ts = load()
    N = len(Dd['gid']); print(N, 'possessions', n_ts, 'team-seasons', DEV, flush=True)
    games = np.unique(Dd['gid']); rng = np.random.default_rng(0)
    fold_of = {g: int(f) for g, f in zip(games, rng.integers(0, 2, len(games)))}
    fold = np.array([fold_of[g] for g in Dd['gid']])
    preds = []
    for f in (0, 1):
        tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f)
        va = rng.choice(te, 3000, replace=False)
        net = Net(n_ts).to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS)
        t0 = time.time()
        for it in range(STEPS):
            net.train()
            loss, nll, kl = step(net, tens(Dd, rng.choice(tr, BATCH)))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            if it % 1000 == 0 or it == STEPS - 1:
                net.eval()
                with torch.no_grad():
                    v = [step(net, tens(Dd, va[i:i + BATCH]), train=False)[1] for i in range(0, 1500, BATCH)]
                print(f'fold {f} step {it} train nll {nll:.3f} kl {kl:.2f} val nll {np.mean(v):.3f} ({time.time() - t0:.0f}s)', flush=True)
        torch.save(net.state_dict(), W / f'world_fold{f}.pt')
        p, Zf = predict(net, Dd, te); preds.append(p); Zall = Zf if f == 0 else Zall + Zf
        e_s = np.hypot(*(p['y'].astype(float) - p['ms'].astype(float)).T); e_b = np.hypot(*(p['y'].astype(float) - p['mb'].astype(float)).T)
        print(f'fold {f}: held-out mean error scheme {e_s.mean():.2f} ft, baseline {e_b.mean():.2f} ft', flush=True)
    P = {k: np.concatenate([p[k] for p in preds]) for k in preds[0]}
    meta = {k: Dd[k] for k in ('gid', 'date', 'season', 'off', 'deff', 'off_team', 'def_team', 'ts')}
    np.savez_compressed(W / 'pred.npz', **P, z=Zall, fold=fold, **{'meta_' + k: v for k, v in meta.items()})
    print('saved', len(P['y']), 'defender-frames', flush=True)


if __name__ == '__main__':
    main()
