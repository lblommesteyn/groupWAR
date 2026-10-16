"""Do lineup interactions add anything beyond additive player values?

1. Fit the additive box-prior RAPM (stint_ridge) on training-season stints.
2. Train an antisymmetric network on the residual stint margin rate:
     g = rho(sum_h phi(x)) - rho(sum_a phi(x)) + psi([sum_h phi, sum_a phi]) - psi([sum_a phi, sum_h phi])
   (the first part is unit composition, the second is the matchup between units).
3. Report held-out-season weighted MSE of the residual against predicting zero.
   A shuffled control trains the same network with player features permuted
   across stints, which keeps marginals but destroys composition.

  python v3/interact_test.py --sport nba --test-season 2023
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge

from data_registry import load_sport
from rapm import PER, load_stints

ROOT = Path(__file__).resolve().parents[1]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Interact(nn.Module):
    def __init__(self, F, h=64, d=16):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(F, h), nn.GELU(), nn.Linear(h, d))
        self.rho = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
        self.psi = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, Xh, Xa):
        sh, sa = self.phi(Xh).sum(1), self.phi(Xa).sum(1)
        comp = self.rho(sh) - self.rho(sa)
        match = self.psi(torch.cat([sh, sa], -1)) - self.psi(torch.cat([sa, sh], -1))
        return (comp + match).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="nba")
    ap.add_argument("--test-season", type=int, default=2023)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1e5)
    ap.add_argument("--box-scale", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stints", default="stints.pkl")
    args = ap.parse_args()
    torch.manual_seed(0)

    S = load_sport(args.sport)
    F = len(S.feature_cols)
    zero = np.zeros(F, np.float32)
    fv = lambda p, s: S.vec(p, s - args.lag) if S.vec(p, s - args.lag) is not None else zero
    st = load_stints(args.sport, args.stints)
    st = st[~st["playoff"] & (st["secs"] >= 5)].reset_index(drop=True)
    players = sorted({p for t in st["home5"] for p in t} | {p for t in st["away5"] for p in t})
    ix = {p: i for i, p in enumerate(players)}
    U = 6 if args.sport == "nhl" else 5
    Xh = np.zeros((len(st), U, F), np.float32)
    Xa = np.zeros((len(st), U, F), np.float32)
    Mh = np.zeros((len(st), U), np.float32)
    Ma = np.zeros((len(st), U), np.float32)
    rows, cols, vals = [], [], []
    for r, (h, a, s) in enumerate(zip(st["home5"], st["away5"], st["season"])):
        for k, p in enumerate(h[:U]):
            Xh[r, k] = fv(p, s); Mh[r, k] = 1
            rows.append(r); cols.append(ix[p]); vals.append(1.0)
        for k, p in enumerate(a[:U]):
            Xa[r, k] = fv(p, s); Ma[r, k] = 1
            rows.append(r); cols.append(ix[p]); vals.append(-1.0)
    Xid = csr_matrix((vals, (rows, cols)), shape=(len(st), len(players)))
    B = (Xh * Mh[..., None]).sum(1) - (Xa * Ma[..., None]).sum(1)
    y = (st["pts_home"] - st["pts_away"]).to_numpy() / st["secs"].to_numpy() * PER[args.sport]
    w = st["secs"].to_numpy().astype(float)
    tr = (st["season"] < args.test_season).to_numpy()
    te = (st["season"] == args.test_season).to_numpy()
    c = args.box_scale
    X = hstack([csr_matrix(B * c), Xid]).tocsr()
    if args.sport == "nhl":
        from values import state_dummies
        X = hstack([X, state_dummies(st)]).tocsr()
    lin = Ridge(alpha=args.alpha).fit(X[tr], y[tr], sample_weight=w[tr])
    resid = y - lin.predict(X)
    wm = lambda e, m: float(np.average(e[m] ** 2, weights=w[m]))
    print(f"{args.sport}: stints train {tr.sum()} test {te.sum()}", flush=True)
    print(f"  additive model: test wMSE {wm(y - lin.predict(X), te):.2f}  (predict-zero {wm(y, te):.2f})", flush=True)

    out = {"additive_test": wm(y - lin.predict(X), te), "zero_test": wm(y, te)}
    for label, shuffle in (("interaction", False), ("shuffled_control", True)):
        Xh_t, Xa_t = torch.tensor(Xh * Mh[..., None]), torch.tensor(Xa * Ma[..., None])
        if shuffle:
            g = torch.Generator().manual_seed(1)
            Xh_t = Xh_t[torch.randperm(len(st), generator=g)]
            Xa_t = Xa_t[torch.randperm(len(st), generator=g)]
        R_t = torch.tensor(resid, dtype=torch.float32)
        W_t = torch.tensor(w, dtype=torch.float32)
        tri = np.where(tr)[0]
        rng = np.random.default_rng(0)
        rng.shuffle(tri)
        va, fit_idx = tri[: len(tri) // 10], tri[len(tri) // 10:]
        net = Interact(F).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)

        def loss_on(idx):
            net.eval()
            tot, ws = 0.0, 0.0
            with torch.no_grad():
                for s in range(0, len(idx), 8192):
                    b = torch.tensor(idx[s:s + 8192])
                    p = net(Xh_t[b].to(DEV), Xa_t[b].to(DEV)).cpu()
                    tot += float((W_t[b] * (R_t[b] - p) ** 2).sum()); ws += float(W_t[b].sum())
            return tot / ws

        best, state = 1e18, None
        for ep in range(args.epochs):
            net.train()
            perm = torch.tensor(fit_idx)[torch.randperm(len(fit_idx))]
            for s in range(0, len(perm), 4096):
                b = perm[s:s + 4096]
                p = net(Xh_t[b].to(DEV), Xa_t[b].to(DEV))
                l = (W_t[b].to(DEV) * (R_t[b].to(DEV) - p) ** 2).sum() / W_t[b].sum().to(DEV)
                opt.zero_grad(); l.backward(); opt.step()
            v = loss_on(va)
            if v < best:
                best, state = v, {k: x.clone() for k, x in net.state_dict().items()}
        net.load_state_dict(state)
        t = loss_on(np.where(te)[0])
        out[label] = {"test_wmse": t, "val_wmse": best, "val_zero": wm(resid, va)}
        print(f"  + {label:17s}: test wMSE {t:.2f}  (residual predict-zero {wm(resid, te):.2f})", flush=True)
    (ROOT / "v3" / "results" / f"interact_{args.sport}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
