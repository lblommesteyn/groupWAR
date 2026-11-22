"""EPV v2 training (runs on the PC GPU). Same frames and network body as v1 (epv_train.EPV), scalar regression head.

  --mode split : train on games before TRAIN_END (10% game-level validation), test after. Two arms with identical
                 setup: target = actual possession points (v1-style) vs target = y2 (shot-quality label, epv2_label.py).
                 Both are scored against ACTUAL points on test frames (the honest metric) and against y2.
  --mode cf    : 5 game folds over all games; every frame gets EPV from the model that never saw its game (y2 arm).
                 Writes out2/epv_cf.npz (epv, gid, meta, y, y2, kind) for credit assignment.
Frames with kind 2 (after the possession's last shot) are excluded from training and testing.
  python epv2_train.py --mode split|cf --data data --labels labels2 --out out2
"""
import argparse, glob, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from epv_train import EPV, normalize, DEV

TRAIN_END = '2026-03-01'


def load(data_dir, lab_dir):
    toks, globs, ys, y2s, kinds, gids, metas = [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(data_dir, '*.npz'))):
        g = os.path.basename(f)[:-4]
        lf = os.path.join(lab_dir, g + '.npz')
        if not os.path.exists(lf):
            continue
        d, l = np.load(f), np.load(lf)
        toks.append(d['tok'].astype(np.float32)); globs.append(d['glob']); ys.append(d['y'].astype(np.float32))
        y2s.append(l['y2']); kinds.append(l['kind']); metas.append(d['meta']); gids += [g] * len(d['y'])
    return (np.concatenate(toks), np.concatenate(globs), np.concatenate(ys), np.concatenate(y2s),
            np.concatenate(kinds), np.array(gids), np.concatenate(metas))


def predict(m, T, Gl, idx):
    m.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), 8192):
            b = torch.tensor(idx[s:s + 8192]); out.append(m(T[b].to(DEV), Gl[b].to(DEV))[:, 0].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(T, Gl, Y, tr, va, epochs, bs, lr, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = EPV(n_out=1).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (len(tr) // bs + 1), pct_start=0.1)
    best, state, t0 = 1e9, None, time.time()
    for ep in range(epochs):
        m.train(); perm = tr[np.random.permutation(len(tr))]
        for s in range(0, len(perm), bs):
            b = torch.tensor(perm[s:s + bs]); tb = T[b].to(DEV).clone(); gb = Gl[b].to(DEV)
            flip = torch.rand(len(b), device=DEV) < 0.5; tb[flip, :, 1] *= -1; tb[flip, :, 3] *= -1
            loss = F.mse_loss(m(tb, gb)[:, 0], Y[b].to(DEV))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        p = predict(m, T, Gl, va); vl = float(((p - Y[va].numpy()) ** 2).mean())
        print(f'  epoch {ep} val mse {vl:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    return m


def r2(p, y, base):
    return 1 - ((p - y) ** 2).mean() / ((base - y) ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='split'); ap.add_argument('--data', default='data'); ap.add_argument('--labels', default='labels2')
    ap.add_argument('--dates', default='dates.json'); ap.add_argument('--out', default='out2')
    ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--bs', type=int, default=1024); ap.add_argument('--lr', type=float, default=3e-4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dates = json.load(open(a.dates))
    tok, gl, y, y2, kind, gid, meta = load(a.data, a.labels)
    tok, gl = normalize(tok, gl)
    T, Gl = torch.tensor(tok), torch.tensor(gl)
    keep = kind != 2
    rng = np.random.default_rng(0)
    if a.mode == 'split':
        gdate = np.array([dates[g] for g in gid])
        games = np.unique(gid[gdate < TRAIN_END]); val_games = set(rng.choice(games, len(games) // 10, replace=False))
        is_val = np.isin(gid, list(val_games))
        tr = np.flatnonzero(keep & (gdate < TRAIN_END) & ~is_val); va = np.flatnonzero(keep & (gdate < TRAIN_END) & is_val)
        te = np.flatnonzero(keep & (gdate >= TRAIN_END))
        print(f'train {len(tr)} val {len(va)} test {len(te)}', flush=True)
        res = dict(train=len(tr), val=len(va), test=len(te))
        for arm, Y in (('actual', y), ('y2', y2)):
            print('arm', arm, flush=True)
            m = fit(T, Gl, torch.tensor(Y), tr, va, a.epochs, a.bs, a.lr)
            p = predict(m, T, Gl, te)
            res[arm] = dict(r2_actual=float(r2(p, y[te], y[tr].mean())), r2_y2=float(r2(p, y2[te], y2[tr].mean())),
                            mse_actual=float(((p - y[te]) ** 2).mean()), mean_pred=float(p.mean()), mean_actual=float(y[te].mean()),
                            sd_pred=float(p.std()))
            # calibration: actual points by decile of predicted EPV
            q = np.quantile(p, np.linspace(0, 1, 11)); bins = np.clip(np.searchsorted(q, p, 'right') - 1, 0, 9)
            res[arm]['deciles'] = [[float(p[bins == k].mean()), float(y[te][bins == k].mean())] for k in range(10)]
            print(json.dumps({k: v for k, v in res[arm].items() if k != 'deciles'}), flush=True)
            torch.save(m.state_dict(), os.path.join(a.out, f'epv_{arm}.pt'))
        json.dump(res, open(os.path.join(a.out, 'metrics_split.json'), 'w'), indent=1)
    else:
        games = np.unique(gid); fold_of = dict(zip(rng.permutation(games), np.arange(len(games)) % 5))
        fd = np.array([fold_of[g] for g in gid]); epv = np.full(len(y), np.nan, np.float32)
        Y2 = torch.tensor(y2)
        for k in range(5):
            trg = np.unique(gid[fd != k]); vg = set(rng.choice(trg, len(trg) // 10, replace=False)); isv = np.isin(gid, list(vg))
            tr = np.flatnonzero(keep & (fd != k) & ~isv); va = np.flatnonzero(keep & (fd != k) & isv)
            print(f'fold {k}: train {len(tr)} val {len(va)}', flush=True)
            m = fit(T, Gl, Y2, tr, va, a.epochs, a.bs, a.lr, seed=k)
            te = np.flatnonzero(fd == k); epv[te] = predict(m, T, Gl, te)
            torch.save(m.state_dict(), os.path.join(a.out, f'epv_cf{k}.pt'))
            ok = keep & (fd == k)
            print(f'fold {k} r2 actual {r2(epv[ok], y[ok], y[keep & (fd != k)].mean()):.4f}', flush=True)
        np.savez_compressed(os.path.join(a.out, 'epv_cf.npz'), epv=epv, gid=gid, meta=meta, y=y, y2=y2, kind=kind, fold=fd)


if __name__ == '__main__':
    main()
