"""Shared v3 components: graph scorer, lineup-mixture deployments, Stackelberg
evaluation, replacement-level knockouts.

Conventions
-----------
A roster side has R slots (NBA R=12, NHL R=18). A game graph has N=2R nodes,
home side first. A is an N x N matrix of on-court time fractions:
  A[i,i]  = fraction of game time player i is on court/ice,
  A[i,j]  = fraction of game time i and j are on together (same team, >= 0),
  A[i,j]  = -(fraction of time i and j face each other) (cross team, <= 0).
Padded slots have zero features, zero time and node_mask = 0.
"""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────── scorer ───────────────────────────────

class RelGraphScorer(nn.Module):
    """Relational graph convolution (separate teammate / opponent channels),
    DeepSet pooling per side, antisymmetric head with a learned home term."""

    def __init__(self, f_in, hidden=128, layers=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(f_in + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.w_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.w_pos = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.w_neg = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.phi = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.home_adv = nn.Parameter(torch.zeros(1))

    def side_values(self, X, A, node_mask, R):
        """Return rho(home side), rho(away side), each (B,)."""
        minutes = torch.diagonal(A, dim1=1, dim2=2).unsqueeze(-1)
        H = self.embed(torch.cat([X, minutes], -1))
        off = A - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
        A_pos, A_neg = off.clamp(min=0), (-off).clamp(min=0)
        for ws, wp, wn, ln in zip(self.w_self, self.w_pos, self.w_neg, self.norms):
            msg = ws(H) + wp(A_pos @ H) + wn(A_neg @ H)
            H = H + self.drop(F.relu(ln(msg)))
        # weight each node's contribution by its on-court time so that
        # unused players contribute nothing
        P = self.phi(H) * (minutes * node_mask.unsqueeze(-1))
        zh, za = P[:, :R].sum(1), P[:, R:].sum(1)
        return self.rho(zh).squeeze(-1), self.rho(za).squeeze(-1)

    def forward(self, X, A, node_mask, R, home=True):
        vh, va = self.side_values(X, A, node_mask, R)
        logit = vh - va + (self.home_adv if home else 0.0)
        return torch.sigmoid(logit)


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, X, A, node_mask, R, home=False):
        return torch.stack([m(X, A, node_mask, R, home=home) for m in self.models]).mean(0)

    def per_model(self, X, A, node_mask, R, home=False):
        return torch.stack([m(X, A, node_mask, R, home=home) for m in self.models])


# ─────────────────────────── deployments ───────────────────────────

class UnitSpace:
    """Enumerates on-court units for one roster side.

    NBA: units are 5-player subsets of the 12 slots.
    NHL: forward trios from the 12 forward slots and D pairs from the 6 D slots,
         mixed independently (line and pair deployment chosen separately).
    """

    def __init__(self, sport):
        self.sport = sport
        if sport == "nba":
            self.R = 12
            combos = list(itertools.combinations(range(12), 5))
            self.groups = [self._incidence(combos, 12)]
            self.group_slots = [list(range(12))]
        else:
            self.R = 18
            trios = list(itertools.combinations(range(12), 3))
            pairs = list(itertools.combinations(range(12, 18), 2))
            self.groups = [self._incidence(trios, 18), self._incidence(pairs, 18)]
            self.group_slots = [list(range(12)), list(range(12, 18))]

    @staticmethod
    def _incidence(combos, R):
        M = torch.zeros(len(combos), R)
        for k, c in enumerate(combos):
            M[k, list(c)] = 1.0
        return M.to(DEV)

    def prior_logits(self, slot_minutes):
        """Lineup prior from a per-slot minutes template q (B, R): independent
        inclusion odds, log w_l = sum_{i in l} log(q_i / (1 - q_i))."""
        q = slot_minutes.clamp(1e-3, 1 - 1e-3)
        lo = torch.log(q / (1 - q))
        return [lo @ M.T for M in self.groups]

    def within(self, logits):
        """Lineup-mixture logits -> within-team time matrix (B, R, R)."""
        B = logits[0].shape[0]
        A = torch.zeros(B, self.R, self.R, device=DEV)
        mins = []
        for lg, M in zip(logits, self.groups):
            w = torch.softmax(lg, -1)
            A = A + torch.einsum("bl,li,lj->bij", w, M, M)
            mins.append(w @ M)
        if len(self.groups) == 2:  # forward-D cross terms under independence
            mf, md = mins
            A = A + mf.unsqueeze(-1) * md.unsqueeze(1) + md.unsqueeze(-1) * mf.unsqueeze(1)
        return A


def full_graph(A_home, A_away):
    """Assemble the N x N signed matrix from two within-team matrices.
    Cross-team exposure is the product of on-court fractions."""
    mh = torch.diagonal(A_home, dim1=1, dim2=2)
    ma = torch.diagonal(A_away, dim1=1, dim2=2)
    cross = -mh.unsqueeze(-1) * ma.unsqueeze(1)
    top = torch.cat([A_home, cross], -1)
    bot = torch.cat([cross.transpose(1, 2), A_away], -1)
    return torch.cat([top, bot], 1)


def kl_logits(lg, prior):
    w = torch.softmax(lg, -1)
    return (w * (torch.log_softmax(lg, -1) - torch.log_softmax(prior, -1))).sum(-1)


def stackelberg_eval(model, Xh, Xa, mh_template, ma_template, maskh, maska, space,
                     mode="stackelberg", outer=40, inner=5, lr=0.3, lam=0.05,
                     return_deploy=False):
    """Score home roster Xh (B,R,F) against away roster Xa (B,R,F).

    mode = "prior"       : both sides use the prior lineup mixture (no optimisation)
           "stackelberg" : leader (home) commits, follower (away) best-responds;
                           follower takes `inner` steps per leader step
           "worst"       : home fixed at prior, away minimises (pure worst case)
    lam is the KL penalty keeping each deployment near its prior.
    Returns P(home win) at a neutral site, shape (B,).
    """
    R = space.R
    node_mask = torch.cat([maskh, maska], 1)
    X = torch.cat([Xh, Xa], 1)
    ph = space.prior_logits(mh_template * maskh)
    pa = space.prior_logits(ma_template * maska)
    # padded slots can never be deployed
    ph = [p - 1e4 * (1 - maskh) @ M.T for p, M in zip(ph, space.groups)]
    pa = [p - 1e4 * (1 - maska) @ M.T for p, M in zip(pa, space.groups)]

    def score(lh, la):
        A = full_graph(space.within(lh), space.within(la))
        return model(X, A, node_mask, R, home=False)

    if mode == "prior":
        with torch.no_grad():
            s = score(ph, pa)
        return (s, (ph, pa)) if return_deploy else s

    lh = [p.clone().requires_grad_(mode == "stackelberg") for p in ph]
    la = [p.clone().requires_grad_(True) for p in pa]
    opt_f = torch.optim.Adam(la, lr=lr)
    opt_l = torch.optim.Adam(lh, lr=lr) if mode == "stackelberg" else None
    for _ in range(outer):
        for _ in range(inner):  # follower best response
            s = score([x.detach() for x in lh], la)
            loss = (s + lam * sum(kl_logits(x, p) for x, p in zip(la, pa))).sum()
            opt_f.zero_grad()
            loss.backward()
            opt_f.step()
        if opt_l is not None:  # leader step at the follower's response (Danskin)
            s = score(lh, [x.detach() for x in la])
            loss = (-s + lam * sum(kl_logits(x, p) for x, p in zip(lh, ph))).sum()
            opt_l.zero_grad()
            loss.backward()
            opt_l.step()
    for _ in range(inner):  # final follower response to the committed leader
        s = score([x.detach() for x in lh], la)
        loss = (s + lam * sum(kl_logits(x, p) for x, p in zip(la, pa))).sum()
        opt_f.zero_grad()
        loss.backward()
        opt_f.step()
    with torch.no_grad():
        s = score([x.detach() for x in lh], [x.detach() for x in la])
    if return_deploy:
        return s, ([x.detach() for x in lh], [x.detach() for x in la])
    return s


def win_prob_to_margin(p, scale):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return scale * math.log(p / (1 - p))
