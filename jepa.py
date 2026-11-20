"""Tracking self-supervised pretraining ablation: JEPA variants vs masked reconstruction vs autoregressive world model.

Data: results/jepa/windows.npz (jepa_data.py): 20 frames x 11 agents (5 attackers, 5 defenders, ball) x (x,y,vx,vy).
Tokens: agent-frame pairs. Embedding = Linear(feat) + team type (O/D/ball) + frame index (+ player identity, *_id arms).
Encoder: transformer (D=128, 4 layers, 4 heads). Training games = first 70% by date (probe test games never seen).
Variants:
  (JEPA losses add VICReg variance/covariance terms on pooled context and predicted latents against collapse)
  jepa_time   context = frames 0-9; predict EMA-target-encoder latents of frames 10-19 (all agents)
  jepa_agent  context = all frames minus 1-2 random players; predict their target latents
  jepa_mix    50/50 of the two masks             jepa_mix_id  same + learned player-identity embeddings
  mae         same masks as jepa_mix, predict raw (normalised) coordinates of masked tokens instead of latents
  world       causal over frames; predict next-frame position change of every agent (Gaussian NLL)
  random      untrained encoder (control)
Probes (frozen encoder on frames 0-9, all agents; features = mean latent of frame 9 + mean over all tokens):
  shot3 / open3 / rim3 (AUC), shot_q / A_now / A_fut (R^2), linear (logistic / ridge), test = last 30% of games
  by date. Compared with the 16-feature geometry baseline and baseline + holes field on the same windows.
Usage (PC): python jepa.py [variant ...]      -> results/jepa/{variant}.pt, probe_{variant}.json
"""
import copy, json, math, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from masked_paths import ROOT

J = ROOT / 'results/jepa'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
D, LAYERS, HEADS, PRED_LAYERS = 128, 4, 4, 2
EPOCHS, BS, LR = 12, 256, 3e-4
SCALE = torch.tensor([47.0, 25.0, 20.0, 20.0])


def load():
    d = np.load(J / 'windows.npz')
    return {k: d[k] for k in d.files}


class Embed(nn.Module):
    def __init__(self, n_players, use_id):
        super().__init__()
        self.lin = nn.Linear(4, D); self.team = nn.Embedding(3, D); self.time = nn.Embedding(20, D)
        self.pid = nn.Embedding(n_players + 1, D) if use_id else None
        nn.init.zeros_(self.pid.weight) if use_id else None
        self.register_buffer('team_idx', torch.tensor([0] * 5 + [1] * 5 + [2]))

    def forward(self, x, ids=None, t_idx=None):
        """x [B, T, 11, 4] normalised -> tokens [B, T, 11, D]"""
        B, T = x.shape[:2]
        t_idx = torch.arange(T, device=x.device) if t_idx is None else t_idx
        h = self.lin(x) + self.team(self.team_idx)[None, None] + self.time(t_idx)[None, :, None]
        if self.pid is not None and ids is not None:
            pid = torch.cat([ids, torch.full_like(ids[:, :1], self.pid.num_embeddings - 1)], 1)  # ball slot
            h = h + self.pid(pid)[:, None]
        return h

    def query(self, B, t_idx, a_idx, ids=None):
        """embeddings of positions to predict (no content): team + time (+ identity)"""
        q = self.team(self.team_idx[a_idx]) + self.time(t_idx)
        q = q[None].expand(B, -1, -1)
        if self.pid is not None and ids is not None:
            pid = torch.cat([ids, torch.full_like(ids[:, :1], self.pid.num_embeddings - 1)], 1)
            q = q + self.pid(pid[:, a_idx])
        return q


def tfm(layers):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(D, HEADS, 4 * D, 0.0, batch_first=True, norm_first=True), layers)


class Net(nn.Module):
    def __init__(self, n_players, variant):
        super().__init__()
        self.variant = variant
        self.emb = Embed(n_players, variant.endswith('_id'))
        self.enc = tfm(LAYERS); self.norm = nn.LayerNorm(D)
        self.pred = tfm(PRED_LAYERS); self.mask_tok = nn.Parameter(torch.zeros(D))
        out = 4 if variant == 'mae' else 4 if variant == 'world' else D
        self.head = nn.Linear(D, out)

    def encode(self, x, ids=None, keep=None, causal=False):
        """x [B,T,11,4]; keep: bool [T*11] of tokens to feed (same for the batch). -> [B, n_keep, D]"""
        h = self.emb(x, ids).flatten(1, 2)
        if keep is not None:
            h = h[:, keep]
        mask = None
        if causal:
            T = x.shape[1]; tt = torch.arange(T, device=x.device).repeat_interleave(11)
            mask = tt[None, :] > tt[:, None]
        return self.norm(self.enc(h, mask=mask))


VIC = 1.0


def vicreg(z):
    """anti-collapse: per-dimension std >= 1 across the batch, and decorrelated dimensions (VICReg terms)"""
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    cov = (z.T @ z) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return Fn.relu(1 - std).mean() + (off ** 2).sum() / z.shape[1]


def masks(rng, kind):
    """-> keep [220] bool (context tokens), tgt [220] bool"""
    keep = np.ones((20, 11), bool)
    if kind == 'time':
        keep[10:] = False
    else:
        hide = rng.choice(10, rng.integers(1, 3), replace=False); keep[:, hide] = False
    return keep.ravel(), ~keep.ravel()


def batches(n, bs, rng):
    p = rng.permutation(n)
    for i in range(0, n - bs + 1, bs):
        yield p[i:i + bs]


def to_dev(X, ids, b, flip):
    x = torch.from_numpy(X[b].astype(np.float32)) / SCALE
    if flip:
        x[..., 1] *= -1; x[..., 3] *= -1
    return x.to(DEV, non_blocking=True), torch.from_numpy(ids[b].astype(np.int64)).to(DEV)


def train(variant, data, tr):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    n_players = len(data['pids'])
    net = Net(n_players, variant).to(DEV)
    if variant == 'random':
        return net
    tgt = copy.deepcopy(net).requires_grad_(False) if variant.startswith('jepa') else None
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    steps = EPOCHS * (len(tr) // BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.05)
    X, ids = data['X'], data['ids']
    step, t0 = 0, time.time()
    for ep in range(EPOCHS):
        tot = 0.0
        for b in batches(len(tr), BS, rng):
            x, pid = to_dev(X, ids, tr[b], rng.random() < 0.5)
            pid = pid if variant.endswith('_id') else None
            if variant == 'world':
                h = net.encode(x[:, :-1], pid, causal=True)                       # frames 0..18
                out = net.head(h).view(x.shape[0], 19, 11, 4)
                dxy = (x[:, 1:, :, :2] - x[:, :-1, :, :2])                         # next-frame change
                mu, logs = out[..., :2], out[..., 2:].clamp(-8, 3)
                loss = (0.5 * ((dxy - mu * 0.05) / (0.05 * logs.exp())) ** 2 + logs).mean()
            else:
                kind = 'time' if variant == 'jepa_time' else 'agent' if variant == 'jepa_agent' else ('time' if rng.random() < 0.5 else 'agent')
                keep, tg = masks(rng, kind)
                keep_t, tg_t = torch.from_numpy(keep).to(DEV), torch.from_numpy(tg).to(DEV)
                ctx = net.encode(x, pid, keep_t)
                ti = torch.arange(20, device=DEV).repeat_interleave(11)[tg_t]; ai = torch.arange(11, device=DEV).repeat(20)[tg_t]
                q = net.emb.query(x.shape[0], ti, ai, pid) + net.mask_tok
                z = net.pred(torch.cat([ctx, q], 1))[:, ctx.shape[1]:]
                pr = net.head(z)
                if variant == 'mae':
                    target = x.flatten(1, 2)[:, tg_t]
                    loss = Fn.mse_loss(pr, target)
                else:
                    with torch.no_grad():
                        target = Fn.layer_norm(tgt.encode(x, pid)[:, tg_t], (D,))
                    loss = Fn.smooth_l1_loss(pr, target) + VIC * vicreg(ctx.mean(1)) + VIC * vicreg(pr.mean(1))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if tgt is not None:
                m = 0.996 + (1 - 0.996) * step / steps
                with torch.no_grad():
                    for a, b_ in zip(tgt.parameters(), net.parameters()):
                        a.mul_(m).add_(b_.detach(), alpha=1 - m)
            tot += loss.item()
        if variant.startswith('jepa'):
            with torch.no_grad():
                e = net.encode(x[:, :10], pid).mean(1)
            print(f'    pooled-latent std across batch {e.std(0).mean().item():.3f}', flush=True)
        print(f'  {variant} epoch {ep} loss {tot / max(1, len(tr) // BS):.4f} {time.time() - t0:.0f}s', flush=True)
    return net


@torch.no_grad()
def embed_all(net, data, use_id):
    net.eval(); X, ids = data['X'], data['ids']; out = []
    for i in range(0, len(X), 1024):
        b = np.arange(i, min(i + 1024, len(X)))
        x, pid = to_dev(X, ids, b, False)
        h = net.encode(x[:, :10], pid if use_id else None, causal=net.variant == 'world')
        h = h.view(len(b), 10, 11, D)
        out.append(torch.cat([h[:, 9].mean(1), h.mean((1, 2))], 1).cpu().numpy())
    return np.concatenate(out)


def probe(E, data, tr, te):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    res = {}
    sets = {'baseline': data['base'], 'baseline+field': np.hstack([data['base'], data['field']])}
    if E is not None:
        sets.update({'emb': E, 'baseline+emb': np.hstack([data['base'], E]),
                     'baseline+field+emb': np.hstack([data['base'], data['field'], E])})
    for tgt in ('shot3', 'open3', 'rim3', 'shot_q', 'A_now', 'A_fut'):
        y = data[tgt].astype(float); ok = np.isfinite(y)
        a, b = tr[ok[tr]], te[ok[te]]
        for name, Xs in sets.items():
            Xs = np.nan_to_num(Xs.astype(np.float64))
            if tgt in ('shot3', 'open3', 'rim3'):
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(Xs[a], y[a])
                sc = roc_auc_score(y[b], m.predict_proba(Xs[b])[:, 1])
            else:
                m = make_pipeline(StandardScaler(), Ridge(10.0)).fit(Xs[a], y[a])
                sc = r2_score(y[b], m.predict(Xs[b]))
            res[f'{tgt}|{name}'] = float(sc)
    return res


def main():
    variants = sys.argv[1:] or [ 'jepa_time', 'jepa_agent', 'jepa_mix', 'jepa_mix_id', 'mae', 'world']
    data = load()
    games = data['game']; ng = games.max() + 1
    cut = int(0.7 * ng); tr = np.where(games < cut)[0]; te = np.where(games >= cut)[0]
    print(f'windows {len(games)} train {len(tr)} test {len(te)} device {DEV}', flush=True)
    if not (J / 'probe_handcrafted.json').exists():
        json.dump(probe(None, data, tr, te), open(J / 'probe_handcrafted.json', 'w'), indent=1)
    for v in variants:
        net = train(v, data, tr)
        torch.save(net.state_dict(), J / f'{v}.pt')
        E = embed_all(net, data, v.endswith('_id'))
        res = probe(E, data, tr, te)
        if v.endswith('_id'):
            np.save(J / f'player_emb_{v}.npy', net.emb.pid.weight.detach().cpu().numpy()[:-1])
        json.dump(res, open(J / f'probe_{v}.json', 'w'), indent=1)
        print(v, {k: round(x, 4) for k, x in res.items() if '|emb' in k or 'baseline+emb' in k}, flush=True)


if __name__ == '__main__':
    main()
