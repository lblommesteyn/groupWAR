"""Paris 2024 roster search with the anchored graph scorer (ratings backbone + GCN fit term).

  python v3/anchored_paris.py --team USA --tag main --arm anchored --out anchored
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from anchored import RSCALE, AnchoredScorer
from core import DEV, Ensemble
from rosters import Evaluator
from run_case import setup
from search import Quota, run_search
from values import value_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v3" / "results"


class AnchoredEvaluator(Evaluator):
    def __init__(self, tag, arm, mode="prior", season=2024, outer=40, inner=5):
        super().__init__("nba", "main_pre", season=season, mode=mode, outer=outer, inner=inner)
        d = ROOT / "v3" / "models" / "nba" / f"anchored_{tag}_{arm}"
        cfg = json.loads((d / "config.json").read_text())
        ms = []
        for k in range(cfg["folds"]):
            m = AnchoredScorer(cfg["n_feat"], use_gcn=(arm != "ratings"), hide_rating=cfg.get("hide_rating", False)).to(DEV)
            m.load_state_dict(torch.load(d / f"fold{k}.pt", map_location="cpu"))
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            ms.append(m)
        self.model = Ensemble(ms)
        self.drop_idx = [self.S.feature_cols.index(c) for c in cfg.get("drop_feats", [])]
        state = pickle.load(open(ROOT / "v3" / "models" / "values" / f"nba_{season}_d0.75_b20.pkl", "rb"))
        self.b, self.r, self.repl = state["main"]

    def rating(self, p):
        if p is None or p < 0:
            return self.repl
        return value_of(self.S, p, self.season, self.b, self.r, self.repl) if p in self.r else 0.0

    def order(self, roster):  # coach order: by rating
        return sorted(roster, key=lambda p: -self.rating(p))

    def tensor(self, roster, knock=None):
        ids = self.order(roster)
        X = torch.zeros(self.R, len(self.S.feature_cols) + 1)
        for i, p in enumerate(ids):
            v = self.S.vec(p, self.season)
            X[i, :-1] = torch.tensor(v) if v is not None else self.replacement
            X[i, -1] = self.rating(p) / RSCALE
        if knock is not None:
            i = ids.index(knock)
            X[i, :-1] = self.replacement
            X[i, -1] = self.repl / RSCALE
        if self.drop_idx:
            X[:, self.drop_idx] = 0.0
        return X, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="USA")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--arm", default="anchored")
    ap.add_argument("--mode", default="prior")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-locks", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="", help="comma-separated names left out of the pool (injured, withdrawn)")
    ap.add_argument("--outer", type=int, default=40)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--screen", type=int, default=128)
    args = ap.parse_args()
    ev = AnchoredEvaluator(args.tag, args.arm, mode=args.mode, outer=args.outer, inner=args.inner)
    pool, actual, locks, opps, qpos, names = setup(ev, "paris", args.team, 500)
    excl = {x.strip() for x in args.exclude.split(",") if x.strip()}
    dropped = [names.get(p) for p in pool if names.get(p) in excl]
    pool = [p for p in pool if names.get(p) not in excl]
    print("excluded:", dropped, "missing:", sorted(excl - set(dropped)), flush=True)
    ev.mpg.update({p: 0.0 for o in opps + [actual] for p in o if p < 0})
    if args.no_locks:
        locks = []
    quota = Quota("nba", qpos)
    res = {"args": vars(args), "actual": actual, "runs": []}
    for seed in range(args.seeds):
        t0 = time.time()
        roster, s, _ = run_search(ev, pool, opps, locks, quota, seed=seed, screen=args.screen)
        res["runs"].append({"seed": seed, "roster": roster, "score": s, "minutes": (time.time() - t0) / 60})
        print(f"seed {seed}: {s:.4f}", [names.get(p, p) for p in roster], flush=True)
    best = max(res["runs"], key=lambda r: r["score"])
    res["best"] = best["roster"]
    res["best_names"] = [names.get(p, p) for p in best["roster"]]
    res["best_score"] = best["score"]
    res["actual_score"] = float(ev.score_rosters([actual], opps).mean())
    s_b, war_b = ev.war(best["roster"], opps)
    res["war_best"] = {names.get(k, k): v for k, v in war_b.items()}
    # how far each fit term moves the roster: score with the GCN fit term vs ratings only, per player
    res["ratings"] = {names.get(p, p): ev.rating(p) for p in sorted(set(pool) | set(actual))}
    (OUT / f"paris_{args.team}_{args.out}.json").write_text(json.dumps(res, indent=1, default=int))
    print("best", best["score"], "actual", res["actual_score"])
    print("best roster:", res["best_names"])


if __name__ == "__main__":
    main()
