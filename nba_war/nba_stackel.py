"""
NBA Stackelberg model — adapted from mie368stackel.py
Key difference: n=10 (5 per team) instead of n=36 (18 per team)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import RGCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_PLAYERS = 10
N_HALF = 5


class GCN_DeepSet_AntiSym_Invariant(nn.Module):
    """GCN model — identical to hockey version, just n=10."""
    def __init__(self, node_in_dim, gcn_hidden, gcn_layers,
                 deepset_phi_dim, deepset_rho_dim,
                 n_nodes=N_PLAYERS, vector_size=128, n_half=N_HALF,
                 use_edge_agg=False, dropout=0.1):
        super().__init__()
        self.N = n_nodes
        self.nh = n_half

        self.embedder = nn.Linear(node_in_dim, vector_size)
        self.embedder2 = nn.Linear(vector_size, vector_size)

        layers = []
        for i in range(gcn_layers):
            in_d = vector_size if i == 0 else gcn_hidden
            layers.append(RGCNConv(in_d, gcn_hidden, num_relations=2))
            layers.append(nn.Linear(gcn_hidden, gcn_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(gcn_hidden, gcn_hidden))

        self.gcn = nn.ModuleList(layers)

        self.phi = nn.Sequential(
            nn.Linear(gcn_hidden, deepset_phi_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(deepset_phi_dim, deepset_phi_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.rho = nn.Sequential(
            nn.Linear(deepset_phi_dim, deepset_rho_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(deepset_rho_dim, 1)
        )

        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(gcn_hidden)
        self.drop = nn.Dropout(dropout)
        self.sig = nn.Tanh()

    def forward(self, A, X, home_mask, away_mask=None):
        B, N, _ = X.shape
        assert N == self.N
        device = X.device
        home_mask = home_mask.to(device).float()
        if away_mask is None:
            away_mask = (1.0 - home_mask).to(device)
        else:
            away_mask = away_mask.to(device).float()

        H = self.relu(self.embedder(X))
        H = self.embedder2(H).reshape(B * N, -1)

        mask = A != 0
        b, i, j = mask.nonzero(as_tuple=True)
        edge_index = torch.stack([b * N + i, b * N + j], dim=0).long()
        edge_weight = A[mask]
        edge_type = (A[mask] > 0).long()
        agg = torch.zeros_like(H)

        for layer in self.gcn:
            if not isinstance(layer, (nn.Linear, nn.Dropout, nn.ReLU)):
                H = H + agg
                agg = torch.zeros_like(H)
                H2 = layer(H, edge_index, edge_type)
                row, col = edge_index
                msg = H2[col] * edge_weight.abs().unsqueeze(-1)
                agg.index_add_(0, row, msg)
                agg = self.relu(self.norm(agg))
                agg = self.drop(agg)
            else:
                agg = layer(agg)

        H = (H + agg).reshape(B, N, -1)
        phi_input = H
        phi_out = self.phi(phi_input.view(B * N, -1)).view(B, N, -1)

        home_sum = (phi_out * home_mask.unsqueeze(-1)).sum(dim=1)
        away_sum = (phi_out * away_mask.unsqueeze(-1)).sum(dim=1)

        home_scalar = self.rho(home_sum).squeeze(-1)
        away_scalar = self.rho(away_sum).squeeze(-1)

        return 0.5 + 0.5 * self.sig(home_scalar - away_scalar)


class ConstraintProjector:
    def __init__(self, n=N_PLAYERS, n_half=N_HALF, tl_target=1, br_target=1,
                 neg_scale=1.25, max_iters=1, tol=1e-3):
        self.n = n
        self.n_half = n_half
        self.tl_target = tl_target
        self.br_target = br_target
        self.neg_scale = neg_scale
        self.max_iters = max_iters
        self.tol = tol

        nh = n_half
        self.tl_mask = torch.zeros(n, n, dtype=torch.bool)
        self.tl_mask[:nh, :nh] = 1
        self.tl_mask.fill_diagonal_(0)

        self.br_mask = torch.zeros(n, n, dtype=torch.bool)
        self.br_mask[nh:, nh:] = 1
        self.br_mask.fill_diagonal_(0)

        self.tr_mask = torch.zeros(n, n, dtype=torch.bool)
        self.tr_mask[:nh, nh:] = 1

        self.bl_mask = torch.zeros(n, n, dtype=torch.bool)
        self.bl_mask[nh:, :nh] = 1

        self.pos_mask = (self.tl_mask | self.br_mask).float()
        self.neg_mask = (self.tr_mask | self.bl_mask).float()

    def project(self, A):
        X = A.clone().float().to("cpu")
        nh = self.n_half
        if len(X.shape) == 3:
            for _ in range(self.max_iters):
                X_prev = X.clone()
                X.diagonal(dim1=1, dim2=2).zero_()
                X[:, :nh, :nh] *= self.tl_target / (X[:, :nh, :nh].sum() + 1e-8)
                X[:, nh:, nh:] *= self.br_target / (X[:, nh:, nh:].sum() + 1e-8)
                row_pos = (X * self.pos_mask.unsqueeze(0)).sum(dim=1, keepdim=True)
                row_neg = (X * self.neg_mask.unsqueeze(0)).sum(dim=1, keepdim=True)
                scale = -self.neg_scale * row_pos / (row_neg + 1e-8)
                X = X * (self.neg_mask.unsqueeze(0) * scale + (1 - self.neg_mask.unsqueeze(0)))
                row_pos = (X * self.pos_mask.unsqueeze(0)).sum(dim=2, keepdim=True)
                row_neg = (X * self.neg_mask.unsqueeze(0)).sum(dim=2, keepdim=True)
                scale = -self.neg_scale * row_pos / (row_neg + 1e-8)
                X = X * (self.neg_mask.unsqueeze(0) * scale + (1 - self.neg_mask.unsqueeze(0)))
                X = 0.5 * (X + X.transpose(1, 2))
                if torch.norm(X - X_prev) < self.tol:
                    break
            X[:, :nh, :nh] /= X[:, :nh, :nh].sum() * (1 / nh ** 2)
            X[:, nh:, nh:] /= X[:, nh:, nh:].sum() * (1 / nh ** 2)
            X[:, :nh, nh:] /= -X[:, :nh, nh:].sum() * (1 / nh ** 2)
            X[:, nh:, :nh] /= -X[:, nh:, :nh].sum() * (1 / nh ** 2)
            return X.to(device)
        else:
            for _ in range(self.max_iters):
                X_prev = X.clone()
                X.fill_diagonal_(0)
                X[:nh, :nh] *= self.tl_target / (X[:nh, :nh].sum() + 1e-8)
                X[nh:, nh:] *= self.br_target / (X[nh:, nh:].sum() + 1e-8)
                row_pos = (X * self.pos_mask).sum(dim=1, keepdim=True)
                row_neg = (X * self.neg_mask).sum(dim=1, keepdim=True)
                scale = -self.neg_scale * row_pos / (row_neg + 1e-8)
                X = X * (self.neg_mask * scale + (1 - self.neg_mask))
                row_pos = (X * self.pos_mask).sum(dim=0, keepdim=True)
                row_neg = (X * self.neg_mask).sum(dim=0, keepdim=True)
                scale = -self.neg_scale * row_pos / (row_neg + 1e-8)
                X = X * (self.neg_mask * scale + (1 - self.neg_mask))
                X = 0.5 * (X + X.transpose(0, 1))
                if torch.norm(X - X_prev) < self.tol:
                    break
            X[:nh, :nh] /= X[:nh, :nh].sum() * (1 / nh ** 2)
            X[nh:, nh:] /= X[nh:, nh:].sum() * (1 / nh ** 2)
            X[:nh, nh:] /= -X[:nh, nh:].sum() * (1 / nh ** 2)
            X[nh:, :nh] /= -X[nh:, :nh].sum() * (1 / nh ** 2)
            return X


projector = ConstraintProjector(N_PLAYERS, N_HALF)


class StackelbergParamX(nn.Module):
    def __init__(self, n, mask_leader, mask_follower, S_init, n_half=N_HALF,
                 tl_target=400 * 60, br_target=400 * 60,
                 neg_scale=1.25, max_iters=100, tol=1e-3):
        super().__init__()
        self.n = n
        self.n_half = n_half
        self.tl_target = tl_target
        self.br_target = br_target
        self.neg_scale = neg_scale
        self.max_iters = max_iters
        self.tol = tol
        self.mask_leader = mask_leader
        self.mask_follower = mask_follower
        self.S = nn.Parameter(S_init)

        nh = n_half
        self.tl_mask = torch.zeros(n, n, dtype=torch.bool)
        self.tl_mask[:nh, :nh] = 1
        self.tl_mask.fill_diagonal_(0)
        self.br_mask = torch.zeros(n, n, dtype=torch.bool)
        self.br_mask[nh:, nh:] = 1
        self.br_mask.fill_diagonal_(0)
        self.tr_mask = torch.zeros(n, n, dtype=torch.bool)
        self.tr_mask[:nh, nh:] = 1
        self.bl_mask = torch.zeros(n, n, dtype=torch.bool)
        self.bl_mask[nh:, :nh] = 1
        self.pos_mask = (self.tl_mask | self.br_mask).float()
        self.neg_mask = (self.tr_mask | self.bl_mask).float()
        self.relu = nn.ReLU()

    def build_X(self):
        X = self.S.clone()
        nh = self.n_half
        X[:, :nh, :nh] = self.relu(X[:, :nh, :nh])
        X[:, nh:, nh:] = self.relu(X[:, nh:, nh:])
        X[:, :nh, nh:] = -self.relu(-X[:, :nh, nh:])
        X[:, nh:, :nh] = -self.relu(-X[:, nh:, :nh])
        return projector.project(X)

    def forward_leader(self, mask):
        X_full = self.S.clone()
        X = X_full * (self.mask_leader - mask) + X_full.detach() * (1 - (self.mask_leader - mask))
        return X

    def forward_follower(self):
        X_full = self.build_X()
        X = X_full * self.mask_follower + X_full.detach() * (1 - self.mask_follower)
        return X


def stackelberg(Y_fixed, model, mask, avg=None, opt_leader=None, opt_follower=None):
    """Stackelberg adversarial optimization — identical logic, n=10."""
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    n = N_PLAYERS
    n_2 = N_HALF

    mask_leader = torch.zeros(n, n).to(device).int().unsqueeze(0)
    mask_leader[:, :n_2, :n_2] = True
    mask_follower = 1 - mask_leader

    B = Y_fixed.shape[0]
    if avg is None:
        S_init = torch.ones(B, n, n)
        S_init[:] /= S_init.sum()
        S_init[:, n_2:, :n_2] *= -1.25
        S_init[:, :n_2, n_2:] *= -1.25
    else:
        S_init = avg.clone()

    steps = 100 if avg is None else 50
    S_init[mask == 1] = 0

    stack = StackelbergParamX(n, mask_leader, mask_follower, S_init).to(device)
    wd = 1e-4
    opt_leader = torch.optim.Adamax([stack.S], lr=1e-5, weight_decay=wd, betas=(0.9, 0.99), eps=1e-5)
    opt_follower = torch.optim.Adamax([stack.S], lr=5e-4, weight_decay=wd, betas=(0.8, 0.98), eps=1e-5)
    sched_l = torch.optim.lr_scheduler.CosineAnnealingLR(opt_leader, T_max=steps)
    sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_follower, T_max=steps // 5)

    # home_mask: first 5 players are home team
    home_mask = torch.tensor([1] * N_HALF + [0] * N_HALF, dtype=torch.bool).unsqueeze(0)
    Y_fixed = Y_fixed.to(device)
    mask = mask.to(device)

    prev_loss = None
    patience = 0
    for step in range(steps):
        X_leader = stack.forward_leader(mask)
        loss = torch.stack([-m(X_leader, Y_fixed, home_mask).mean() for m in model]).mean()
        opt_leader.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([stack.S], max_norm=1)
        opt_leader.step()

        loss_val = loss.item()
        if prev_loss is not None and abs(loss_val - prev_loss) < 1e-5:
            patience += 1
            if patience >= 5:
                break
        else:
            patience = 0
        prev_loss = loss_val

        if (step + 1) % 5 == 0:
            X_follower = stack.forward_follower()
            loss = torch.stack([m(X_follower, Y_fixed, home_mask).mean() for m in model]).mean()
            opt_follower.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([stack.S], max_norm=1)
            opt_follower.step()
            sched_f.step()

        sched_l.step()

    X_final = stack.build_X().detach()
    with torch.no_grad():
        final_preds = torch.stack([m(X_final, Y_fixed, home_mask) for m in model])
    return final_preds.mean(dim=0).to("cpu"), X_final.to("cpu"), opt_leader, opt_follower


class PeriodDataset(Dataset):
    def __init__(self, X, A, y, home_mask=None, away_mask=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.A = torch.stack(A).to(dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        if home_mask is not None:
            self.home_mask = torch.tensor(home_mask, dtype=torch.bool)
            self.away_mask = torch.tensor(away_mask, dtype=torch.bool)
        else:
            self.home_mask = torch.tensor([1] * N_HALF + [0] * N_HALF, dtype=torch.bool)
            self.away_mask = ~self.home_mask

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.home_mask, self.away_mask, self.y[idx]
