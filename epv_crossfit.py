"""Per-block, cross-fitted EPV (runs on the PC GPU).

For each rolling block start b: games before b are split into 5 game folds; each fold's samples get EPV
from a model trained on the other 4 folds (history only). Games at/after b are never used for training.
Output: out_cf/epv_{b}.npz with epv for every pre-b sample (+ gid, meta, y). Also a 'final' run on all
games (for the case study) with the same 5-fold scheme.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from epv_train import EPV, load, normalize, epv_of, DEV

BLOCKS = ['2026-03-01', '2026-03-15', '2026-03-29', '2026-04-12', '2026-04-26', '2026-05-10', 'final']
EPOCHS, BS, LR = 12, 1024, 3e-4


def fit(T, Gl, Y, tr, va, seed):
    torch.manual_seed(seed)
    m = EPV().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (len(tr) // BS + 1), pct_start=0.1)
    best, state = 1e9, None
    for ep in range(EPOCHS):
        m.train(); perm = tr[np.random.permutation(len(tr))]
        for s in range(0, len(perm), BS):
            b = torch.tensor(perm[s:s + BS]); tb = T[b].to(DEV).clone(); gb = Gl[b].to(DEV)
            flip = torch.rand(len(b), device=DEV) < 0.5; tb[flip, :, 1] *= -1; tb[flip, :, 3] *= -1
            loss = F.cross_entropy(m(tb, gb), Y[b].to(DEV))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        m.eval(); ce = 0.0
        with torch.no_grad():
            for s in range(0, len(va), 8192):
                b = torch.tensor(va[s:s + 8192]); ce += F.cross_entropy(m(T[b].to(DEV), Gl[b].to(DEV)), Y[b].to(DEV), reduction='sum').item()
        ce /= len(va)
        if ce < best:
            best, state = ce, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    return m, best


def predict(m, T, Gl, idx):
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 8192):
            b = torch.tensor(idx[s:s + 8192]); out.append(epv_of(m(T[b].to(DEV), Gl[b].to(DEV))).cpu().numpy())
    return np.concatenate(out)


def main():
    dates = json.load(open('dates.json'))
    tok, glob_, y, gid, meta = load('data', dates)
    tok, glob_ = normalize(tok, glob_)
    T, Gl, Y = torch.tensor(tok), torch.tensor(glob_), torch.tensor(y)
    gdate = np.array([dates[g] for g in gid])
    os.makedirs('out_cf', exist_ok=True)
    blocks = sys.argv[1:] or BLOCKS
    for b in blocks:
        f = f'out_cf/epv_{b}.npz'
        if os.path.exists(f):
            continue
        hist = np.flatnonzero(gdate < b) if b != 'final' else np.arange(len(y))
        games = np.unique(gid[hist]); rng = np.random.default_rng(0); fold_of = dict(zip(games, rng.integers(0, 3, len(games))))
        fold = np.array([fold_of[g] for g in gid[hist]])
        epv = np.full(len(hist), np.nan, np.float32); ces = []
        for k in range(3):
            te = hist[fold == k]; rest = hist[fold != k]
            rg = np.unique(gid[rest]); vg = set(rng.choice(rg, max(1, len(rg) // 10), replace=False))
            isv = np.array([g in vg for g in gid[rest]])
            m, ce = fit(T, Gl, Y, rest[~isv], rest[isv], seed=k)
            epv[fold == k] = predict(m, T, Gl, te)
            te_ce = F.cross_entropy(torch.zeros(1)) if False else None
            ces.append(ce)
        mse = float(((epv - y[hist]) ** 2).mean()); c = float(((y[hist] - y[hist].mean()) ** 2).mean())
        print(f'{b}: {len(hist)} samples, {len(games)} games, oof r2 {1 - mse / c:.4f}, val ce {np.mean(ces):.4f}', flush=True)
        np.savez_compressed(f, epv=epv, gid=gid[hist], meta=meta[hist], y=y[hist])


if __name__ == '__main__':
    main()
