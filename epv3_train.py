"""EPV v3 (PC GPU): value of a half-court moment from its last 2 s of movement (5 Hz), fine-tuned from the MAE.

Input per sample: window [10, 11, 4] (5 attackers, 5 defenders, ball; x, y, vx, vy; offense attacks +x), frames
left-padded by repeating the earliest one when the possession has < 2 s of half-court history, + 5 globals
(shot clock, game clock, period, margin, seconds into possession). No player identity.
Model: jepa.Net('mae_long') encoder (D=128, 4 layers) over the 110 agent-frame tokens; head on [mean of the "now"
tokens, ball token now, mean of all tokens, globals MLP] -> scalar EPV. Target: y2 (possession points, shot luck
removed; epv2_label.py). Frames after the possession's last shot (kind 2) are excluded.

  --mode split : train before 2026-03-01 (10% game validation), test after. Arms 'pre' (MAE init) and 'scratch'
                 (same architecture, random init). Scored against actual points and y2 on all test frames, and on
                 the moments that also have a v1 frame, next to EPV v2 (court-epv/out2/epv_y2.pt) on those moments.
  --mode cf    : 'pre' arm, 5 game folds identical to EPV v2 -> out3/epv3_cf.npz (EPV for every sample)
  python epv3_train.py --mode split|cf
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'C:\Users\LukeB\dev\court-model'); sys.path.insert(0, r'C:\Users\LukeB\dev\court-epv')
import jepa

DEV = torch.device('cuda')
EPV_DIR = r'C:\Users\LukeB\dev\court-epv'
TRAIN_END = '2026-03-01'
SCALE = torch.tensor([47.0, 25.0, 20.0, 20.0], device=DEV)
GSCALE = torch.tensor([24.0, 720.0, 4.0, 15.0, 24.0], device=DEV)


class V3(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.net = jepa.Net(1, 'mae_long')
        if pretrained:
            sd = torch.load(os.path.join(r'C:\Users\LukeB\dev\court-model\results\jepa', 'mae_long.pt'), map_location='cpu')
            missing = self.net.load_state_dict(sd, strict=False); print('loaded MAE', missing, flush=True)
        D = jepa.D
        self.glob = nn.Sequential(nn.Linear(5, D), nn.GELU(), nn.Linear(D, D))
        self.head = nn.Sequential(nn.LayerNorm(4 * D), nn.Linear(4 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, 1))

    def forward(self, x, g):
        h = self.net.encode(x).view(len(x), 10, 11, -1)
        z = torch.cat([h[:, 9].mean(1), h[:, 9, 10], h.mean((1, 2)), self.glob(g)], -1)
        return self.head(z)[:, 0]


class Data:
    def __init__(self, path):
        d = np.load(path)
        self.F = torch.from_numpy(d['F']).to(DEV)  # [Nf, 11, 4] f16
        self.fi = torch.from_numpy(d['fi']).to(DEV); self.L = torch.from_numpy(d['L']).to(DEV)
        self.g = torch.from_numpy(np.nan_to_num(d['glob'])).to(DEV)
        for k in ('y', 'y2', 'kind', 'gi', 'v1idx', 'meta'):
            setattr(self, k, d[k])
        self.games = d['games']
        self.offs = torch.arange(-9, 1, device=DEV)

    def batch(self, idx, flip=False):
        i = torch.as_tensor(idx, device=DEV)
        fi, L = self.fi[i], self.L[i]
        w = torch.maximum(fi[:, None] + self.offs[None], (fi - L + 1)[:, None])
        x = self.F[w].float() / SCALE
        if flip:
            f = torch.rand(len(i), device=DEV) < 0.5; x[f, ..., 1] *= -1; x[f, ..., 3] *= -1
        g = self.g[i].clone(); g[:, 2] = g[:, 2].clamp(max=5); g = g / GSCALE
        return x, g


def predict(m, D, idx, bs=4096):
    m.eval(); out = []
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        for s in range(0, len(idx), bs):
            x, g = D.batch(idx[s:s + bs]); out.append(m(x, g).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(D, Y, tr, va, pretrained, epochs, seed=0, bs=512):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = V3(pretrained).to(DEV)
    enc = list(m.net.parameters()); rest = list(m.glob.parameters()) + list(m.head.parameters())
    lr_enc = 1e-4 if pretrained else 3e-4
    opt = torch.optim.AdamW([{'params': enc, 'lr': lr_enc}, {'params': rest, 'lr': 1e-3}], weight_decay=0.01)
    steps = epochs * (len(tr) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [lr_enc, 1e-3], total_steps=steps, pct_start=0.05)
    Yt = torch.from_numpy(Y).to(DEV)
    best, state, t0 = 1e9, None, time.time()
    for ep in range(epochs):
        m.train(); perm = tr[rng.permutation(len(tr))]
        for s in range(0, len(perm) - bs + 1, bs):
            b = perm[s:s + bs]; x, g = D.batch(b, flip=True)
            with torch.autocast('cuda', torch.bfloat16):
                p = m(x, g)
            loss = F.mse_loss(p.float(), Yt[torch.as_tensor(b, device=DEV)])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        pv = predict(m, D, va); vl = float(((pv - Y[va]) ** 2).mean())
        print(f'  epoch {ep} val mse {vl:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    return m


def r2(p, y, base):
    return float(1 - ((p - y) ** 2).mean() / ((base - y) ** 2).mean())


def v2_predict(v1idx):
    """EPV v2 (split model, same train dates) on v1 frames v1idx."""
    from epv_train import normalize
    from epv2_train import load as load2
    import epv_train
    tok, gl, y, y2, kind, gid, meta = load2(os.path.join(EPV_DIR, 'data'), os.path.join(EPV_DIR, 'labels2'))
    tok, gl = normalize(tok[v1idx], gl[v1idx])
    m = epv_train.EPV(n_out=1).to(DEV); m.load_state_dict(torch.load(os.path.join(EPV_DIR, 'out2', 'epv_y2.pt'))); m.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(tok), 8192):
            out.append(m(torch.tensor(tok[s:s + 8192]).to(DEV), torch.tensor(gl[s:s + 8192]).to(DEV))[:, 0].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='split'); ap.add_argument('--data', default=os.path.join(EPV_DIR, 'v3.npz'))
    ap.add_argument('--out', default=os.path.join(EPV_DIR, 'out3')); ap.add_argument('--epochs', type=int, default=4)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = Data(a.data); dates = json.load(open(os.path.join(EPV_DIR, 'dates.json')))
    gname = D.games[D.gi]; keep = D.kind != 2
    print('samples', len(D.y), 'kept', keep.sum(), flush=True)
    if a.mode == 'split':
        gdate = np.array([dates[g] for g in D.games])[D.gi]
        rng = np.random.default_rng(0); trg = np.unique(gname[gdate < TRAIN_END])
        vg = set(rng.choice(trg, len(trg) // 10, replace=False)); isv = np.isin(gname, list(vg))
        tr = np.flatnonzero(keep & (gdate < TRAIN_END) & ~isv); va = np.flatnonzero(keep & (gdate < TRAIN_END) & isv)
        te = np.flatnonzero(keep & (gdate >= TRAIN_END)); same = te[D.v1idx[te] >= 0]
        print(f'train {len(tr)} val {len(va)} test {len(te)} (same-moment {len(same)})', flush=True)
        res = dict(train=len(tr), val=len(va), test=len(te), same=len(same))
        base, base2 = D.y[tr].mean(), D.y2[tr].mean()
        p2 = v2_predict(D.v1idx[same])
        res['v2_same'] = dict(r2_actual=r2(p2, D.y[same], base), r2_y2=r2(p2, D.y2[same], base2))
        print('v2 on same moments', res['v2_same'], flush=True)
        for arm, pre in (('pre', True), ('scratch', False)):
            print('arm', arm, flush=True)
            m = fit(D, D.y2, tr, va, pre, a.epochs)
            p = predict(m, D, te); ps = p[np.isin(te, same)]
            res[arm] = dict(r2_actual=r2(p, D.y[te], base), r2_y2=r2(p, D.y2[te], base2), mean_pred=float(p.mean()),
                            mean_y=float(D.y[te].mean()), sd_pred=float(p.std()),
                            same_r2_actual=r2(ps, D.y[same], base), same_r2_y2=r2(ps, D.y2[same], base2),
                            # smoothness: mean |change| between consecutive 5 Hz frames of the same possession
                            step_abs=float(np.mean(np.abs(np.diff(p))[(np.diff(te) == 1)])))
            print(json.dumps(res[arm]), flush=True)
            torch.save(m.state_dict(), os.path.join(a.out, f'epv3_{arm}.pt'))
            json.dump(res, open(os.path.join(a.out, 'metrics_split.json'), 'w'), indent=1)
    else:
        games = D.games; rng = np.random.default_rng(0)
        fold_of = dict(zip(rng.permutation(games), np.arange(len(games)) % 5)); fd = np.array([fold_of[g] for g in gname])
        epv = np.full(len(D.y), np.nan, np.float32)
        for k in range(5):
            trg = np.unique(gname[fd != k]); vg = set(rng.choice(trg, len(trg) // 10, replace=False)); isv = np.isin(gname, list(vg))
            tr = np.flatnonzero(keep & (fd != k) & ~isv); va = np.flatnonzero(keep & (fd != k) & isv)
            print(f'fold {k}: train {len(tr)} val {len(va)}', flush=True)
            m = fit(D, D.y2, tr, va, True, a.epochs, seed=k)
            te = np.flatnonzero(fd == k); epv[te] = predict(m, D, te)
            ok = keep & (fd == k)
            print(f'fold {k} r2 actual {r2(epv[ok], D.y[ok], D.y[keep & (fd != k)].mean()):.4f}', flush=True)
            torch.save(m.state_dict(), os.path.join(a.out, f'epv3_cf{k}.pt'))
        np.savez_compressed(os.path.join(a.out, 'epv3_cf.npz'), epv=epv, gi=D.gi, games=D.games, meta=D.meta, y=D.y, y2=D.y2,
                            kind=D.kind, fold=fd, fi=np.load(a.data)['fi'], L=D.L)


if __name__ == '__main__':
    main()
