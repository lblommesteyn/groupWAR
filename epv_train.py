"""EPV transformer (runs on the PC GPU).

Input: 11 tokens (5 offense, 5 defense, ball) x 8 features + 5 globals. No player identity.
Model: per-token MLP embed + type embedding, global token, 4-layer transformer encoder (d=128),
head on the global token -> softmax over possession points {0,1,2,3,4}; EPV = expectation.
Augmentation: random reflection y -> -y (court is symmetric across the long axis).
Split: games before TRAIN_END train (game-level 10% validation for early stopping); later games test.

  python epv_train.py --data data --out out
Writes out/epv.pt, out/metrics.json, out/epv_all.npz (EPV for every sample, game_id per sample).
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_END = '2026-03-01'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load(data_dir, dates):
    toks, globs, ys, gids, metas = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(data_dir, '*.npz'))):
        g = os.path.basename(f)[:-4]
        d = np.load(f)
        toks.append(d['tok'].astype(np.float32)); globs.append(d['glob']); ys.append(d['y'].astype(np.int64))
        metas.append(d['meta']); gids += [g] * len(d['y'])
    return (np.concatenate(toks), np.concatenate(globs), np.concatenate(ys), np.array(gids), np.concatenate(metas))


def normalize(tok, glob):
    tok = tok.copy(); glob = glob.copy()
    tok[..., 0] /= 47.0; tok[..., 1] /= 25.0; tok[..., 2:4] /= 15.0; tok[..., 4] /= 10.0; tok[..., 7] /= 47.0
    glob[:, 0] /= 24.0; glob[:, 1] /= 720.0; glob[:, 2] = np.minimum(glob[:, 2], 5) / 4.0
    glob[:, 3] /= 15.0; glob[:, 4] /= 24.0
    return np.nan_to_num(tok), np.nan_to_num(glob)


class EPV(nn.Module):
    def __init__(self, d=128, layers=4, heads=8, n_out=5):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.typ = nn.Embedding(3, d)  # offense, defense, ball
        self.glob = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))
        self.register_buffer('types', torch.tensor([0] * 5 + [1] * 5 + [2]))

    def forward(self, tok, glob):
        h = self.tok(tok) + self.typ(self.types)
        x = torch.cat([self.glob(glob).unsqueeze(1), h], 1)
        return self.head(self.enc(x)[:, 0])


def epv_of(logits):
    return (torch.softmax(logits, -1) * torch.arange(logits.shape[-1], device=logits.device)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data'); ap.add_argument('--dates', default='dates.json')
    ap.add_argument('--out', default='out'); ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--bs', type=int, default=1024); ap.add_argument('--lr', type=float, default=3e-4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dates = json.load(open(a.dates))
    tok, glob_, y, gid, meta = load(a.data, dates)
    tok, glob_ = normalize(tok, glob_)
    gdate = np.array([dates[g] for g in gid])
    games = np.unique(gid[gdate < TRAIN_END]); rng = np.random.default_rng(0)
    val_games = set(rng.choice(games, max(1, len(games) // 10), replace=False))
    is_val = np.array([g in val_games for g in gid])
    tr = np.flatnonzero((gdate < TRAIN_END) & ~is_val); va = np.flatnonzero((gdate < TRAIN_END) & is_val)
    te = np.flatnonzero(gdate >= TRAIN_END)
    print(f'samples train {len(tr)} val {len(va)} test {len(te)} | mean pts {y[tr].mean():.3f}', flush=True)
    T, Gl, Y = torch.tensor(tok), torch.tensor(glob_), torch.tensor(y)
    model = EPV().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)

    def evaluate(idx):
        model.eval(); ll, se, n = 0.0, 0.0, 0
        with torch.no_grad():
            for s in range(0, len(idx), 8192):
                b = torch.tensor(idx[s:s + 8192])
                lg = model(T[b].to(DEV), Gl[b].to(DEV)); yb = Y[b].to(DEV)
                ll += F.cross_entropy(lg, yb, reduction='sum').item()
                se += ((epv_of(lg) - yb.float()) ** 2).sum().item(); n += len(b)
        return ll / n, se / n

    best, t0 = 1e9, time.time()
    for ep in range(a.epochs):
        model.train(); perm = tr[np.random.permutation(len(tr))]
        for s in range(0, len(perm), a.bs):
            b = torch.tensor(perm[s:s + a.bs])
            tb, gb = T[b].to(DEV), Gl[b].to(DEV)
            flip = (torch.rand(len(b), device=DEV) < 0.5)
            tb = tb.clone(); tb[flip, :, 1] *= -1; tb[flip, :, 3] *= -1
            loss = F.cross_entropy(model(tb, gb), Y[b].to(DEV))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        vl, vmse = evaluate(va)
        print(f'epoch {ep} val ce {vl:.4f} mse {vmse:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if vl < best:
            best = vl; torch.save(model.state_dict(), os.path.join(a.out, 'epv.pt'))
    model.load_state_dict(torch.load(os.path.join(a.out, 'epv.pt')))
    # baselines on test: constant mean, and clock-only (shot clock + time in possession) via the same CE metric
    p_const = np.bincount(y[tr], minlength=5) / len(tr)
    const_ce = float(-np.log(p_const[y[te]] + 1e-9).mean()); const_mse = float(((y[te] - (p_const * np.arange(5)).sum()) ** 2).mean())
    tce, tmse = evaluate(te)
    res = dict(train=len(tr), val=len(va), test=len(te), test_ce=tce, test_mse=tmse, const_ce=const_ce, const_mse=const_mse,
               r2_vs_const=1 - tmse / const_mse)
    print(json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    model.eval(); ev = []
    with torch.no_grad():
        for s in range(0, len(y), 8192):
            ev.append(epv_of(model(T[s:s + 8192].to(DEV), Gl[s:s + 8192].to(DEV))).cpu().numpy())
    np.savez_compressed(os.path.join(a.out, 'epv_all.npz'), epv=np.concatenate(ev), gid=gid, meta=meta, y=y)


if __name__ == '__main__':
    main()
