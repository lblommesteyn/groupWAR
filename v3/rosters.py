"""Roster evaluation: ensemble loading, slot ordering, deployment prior,
batched Stackelberg scoring, replacement-level knockouts (WAR)."""
import json
from pathlib import Path

import numpy as np
import torch

from core import DEV, Ensemble, RelGraphScorer, UnitSpace, stackelberg_eval
from data_registry import load_sport

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "v3" / "models"


def load_ensemble(sport, tag):
    d = MODELS / sport / tag
    cfg = json.loads((d / "config.json").read_text())
    ms = []
    for k in range(cfg["folds"]):
        m = RelGraphScorer(cfg["n_feat"], cfg["hidden"], cfg["layers"]).to(DEV)
        m.load_state_dict(torch.load(d / f"fold{k}.pt", map_location="cpu"))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        ms.append(m)
    return Ensemble(ms), cfg


def minutes_template(S, X_A_M=None):
    """Average on-court fraction by slot rank in real regular-season games."""
    X, A, M, y, meta, _ = X_A_M or S.tensors()
    d = torch.diagonal(A, dim1=1, dim2=2)
    R = S.R
    reg = torch.tensor([not m["playoff"] for m in meta])
    q = torch.cat([d[reg, :R], d[reg, R:]], 0).mean(0)
    return q


def calibrate_prior(space, q, iters=200):
    """Per-slot log-odds theta such that the lineup prior's marginal minutes
    equal the template q (iterative proportional fitting on the marginals)."""
    theta = torch.log(q / (1 - q + 1e-6)).to(DEV).unsqueeze(0)
    for _ in range(iters):
        m = torch.zeros_like(theta)
        for M in space.groups:
            w = torch.softmax(theta @ M.T, -1)
            m = m + w @ M
        theta = theta + 0.5 * torch.log(q.to(DEV).clamp(1e-4) / m.clamp(1e-4))
    return theta.squeeze(0)


class Evaluator:
    def __init__(self, sport, tag="main", season=None, mode="stackelberg", lam=0.05,
                 outer=40, inner=5):
        self.S = load_sport(sport)
        self.sport = sport
        self.model, self.cfg = load_ensemble(sport, tag)
        self.space = UnitSpace(sport)
        self.R = self.S.R
        self.season = season
        self.mode, self.lam, self.outer, self.inner = mode, lam, outer, inner
        q = minutes_template(self.S)
        theta = calibrate_prior(self.space, q)
        # prior_logits(q) uses log(q/(1-q)); feed the calibrated odds through q*
        self.q_star = torch.sigmoid(theta).detach()
        self.q = q
        ps = self.S.ps
        cur = ps[(ps["season"] == season) & (ps["minutes_total"] >= self.S.min_minutes)]
        lo, hi = cur["minutes_total"].quantile([0.10, 0.30])
        repl = cur[(cur["minutes_total"] >= lo) & (cur["minutes_total"] <= hi)]
        Z = ((repl[self.S.feature_cols] - self.S.mu) / self.S.sd).fillna(0).to_numpy(np.float32)
        self.replacement = torch.tensor(Z.mean(0))
        self.mpg = dict(zip(cur["player_id"], cur["toi_pg"] if sport == "nhl" else cur["mpg"]))
        self.pos = dict(zip(ps["player_id"], ps.get("pos", ps.get("position"))))

    # ── roster tensors ──
    def order(self, roster):
        """Slot order: NBA by minutes per game; NHL forwards then D, each by TOI."""
        key = lambda p: -self.mpg.get(p, 0.0)
        if self.sport == "nba":
            return sorted(roster, key=key)
        fw = sorted([p for p in roster if self.pos.get(p) != "D"], key=key)
        dm = sorted([p for p in roster if self.pos.get(p) == "D"], key=key)
        assert len(fw) == 12 and len(dm) == 6, (len(fw), len(dm))
        return fw + dm

    def tensor(self, roster, knock=None):
        ids = self.order(roster)
        X = torch.zeros(self.R, len(self.S.feature_cols))
        for i, p in enumerate(ids):
            v = self.S.vec(p, self.season)
            # players with no league data in the feature season (e.g. playing in
            # Europe) are represented at replacement level
            X[i] = torch.tensor(v) if v is not None else self.replacement
        if knock is not None:
            X[ids.index(knock)] = self.replacement
        return X, ids

    # ── scoring ──
    @torch.no_grad()
    def _noop(self):
        return None

    def score_tensors(self, Xh, Xo_list, mode=None, per_model=False, chunk=512):
        """Xh (B,R,F); Xo_list list of (R,F). Returns (B, n_opp) win probabilities."""
        mode = mode or self.mode
        B = Xh.shape[0]
        out = torch.zeros(B, len(Xo_list))
        mask = torch.ones(1, self.R, device=DEV)
        q = self.q_star.to(DEV).unsqueeze(0)
        for j, Xo in enumerate(Xo_list):
            for s in range(0, B, chunk):
                xb = Xh[s:s + chunk].to(DEV)
                b = xb.shape[0]
                xo = Xo.to(DEV).unsqueeze(0).expand(b, -1, -1)
                with torch.enable_grad():
                    p = stackelberg_eval(self.model, xb, xo, q.expand(b, -1), q.expand(b, -1),
                                         mask.expand(b, -1), mask.expand(b, -1), self.space,
                                         mode=mode, lam=self.lam, outer=self.outer, inner=self.inner)
                out[s:s + b, j] = p.detach().cpu()
        return out

    def score_rosters(self, rosters, opponents, mode=None):
        Xh = torch.stack([self.tensor(r)[0] for r in rosters])
        Xo = [self.tensor(o)[0] for o in opponents]
        return self.score_tensors(Xh, Xo, mode)

    def war(self, roster, opponents, mode=None):
        """Replacement-level knockout: drop in mean win probability (vs the opponent
        set) when player i's features are replaced by the replacement-level vector."""
        base_X, ids = self.tensor(roster)
        Xs = [base_X] + [self.tensor(roster, knock=p)[0] for p in ids]
        Xo = [self.tensor(o)[0] for o in opponents]
        s = self.score_tensors(torch.stack(Xs), Xo, mode).mean(1)
        return float(s[0]), {p: float(s[0] - s[i + 1]) for i, p in enumerate(ids)}
