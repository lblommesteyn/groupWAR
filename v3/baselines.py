"""E3: baseline rosters and a random null, scored under the model objective.

  python v3/baselines.py --case paris --team USA --tag main

Baselines (same pool, quotas and locks as the search):
  rapm_topk     players ranked by RAPM fit on the seasons before the feature season
  minutes_topk  players ranked by minutes played in the feature season (coach usage)
  random null   N quota-valid random rosters
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

import run_case as RC
from rapm import fit as fit_rapm
from rosters import Evaluator
from search import Quota, random_valid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v3" / "results"


def topk(order, locks, quota, sport):
    roster = list(locks)
    if sport == "nba":
        def need(r):
            g = sum("Guard" in str(quota.pos.get(p, "")) for p in r)
            c = sum("Center" in str(quota.pos.get(p, "")) for p in r)
            return max(0, quota.min_g - g), max(0, quota.min_c - c)
        for p in order:
            if len(roster) == 12:
                break
            if p in roster:
                continue
            ng, nc = need(roster)
            slots = 12 - len(roster)
            isg, isc = "Guard" in str(quota.pos.get(p, "")), "Center" in str(quota.pos.get(p, ""))
            if ng + nc >= slots and not ((ng and isg) or (nc and isc)):
                continue
            roster.append(p)
    else:
        def cnt(r, k):
            return sum(quota.pos.get(p) == k for p in r)
        for p in order:
            if len(roster) == 18:
                break
            if p in roster:
                continue
            k = quota.pos.get(p)
            nd = 6 - cnt(roster, "D")
            nf = 12 - (len(roster) - cnt(roster, "D"))
            if k == "D":
                if nd > 0:
                    roster.append(p)
                continue
            if nf <= 0:
                continue
            nl = max(0, quota.min_hc - cnt(roster, "C"))
            nr = max(0, quota.min_hw - cnt(roster, "W"))
            if nl + nr >= nf and not ((nl and k == "C") or (nr and k == "W")):
                continue
            roster.append(p)
    assert quota.ok(roster), "top-k fill failed"
    return roster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--search-out", default="main")
    ap.add_argument("--scorer", default="value", choices=["value", "gnn"])
    args = ap.parse_args()
    sport = "nba" if args.case == "paris" else "nhl"
    season = 2024 if sport == "nba" else 2025
    if args.scorer == "value":
        from values import ValueEvaluator
        ev = ValueEvaluator(sport, season)
    else:
        ev = Evaluator(sport, args.tag, season=season)
    mm = 500 if sport == "nba" else 300
    pool, actual, locks, opps, qpos, names = RC.setup(ev, args.case, args.team, mm)
    ev.mpg.update({p: 0.0 for o in opps + [actual] for p in o if p < 0})
    quota = Quota(sport, qpos)

    rapm_seasons = [2021, 2022, 2023] if sport == "nba" else [2023, 2024, 2025]
    rp, _, _ = fit_rapm(sport, rapm_seasons)
    rmap = dict(zip(rp["player_id"], rp["rapm"]))
    ps = ev.S.ps
    cur = ps[ps["season"] == season]
    mins = dict(zip(cur["player_id"], cur["minutes_total"]))
    by_rapm = sorted(pool, key=lambda p: -rmap.get(p, -99))
    by_min = sorted(pool, key=lambda p: -mins.get(p, 0))

    rosters = {"actual": actual,
               "rapm_topk_locked": topk(by_rapm, locks, quota, sport),
               "rapm_topk_unlocked": topk(by_rapm, [], quota, sport),
               "minutes_topk_locked": topk(by_min, locks, quota, sport)}
    srch = OUT / f"{args.case}_{args.team}_{args.search_out}.json"
    if srch.exists():
        rosters["model_best"] = json.loads(srch.read_text())["best"]
    res = {"rosters": rosters, "scores": {}, "prior_scores": {}, "overlap_actual": {}}
    names_list = list(rosters)
    S = ev.score_rosters([rosters[k] for k in names_list], opps)
    # comparison deployment: no adversary for the GNN, club-minutes rotation for the value model
    alt = "club" if args.scorer == "value" else "prior"
    P = ev.score_rosters([rosters[k] for k in names_list], opps, mode=alt)
    for k, s, p in zip(names_list, S, P):
        res["scores"][k] = float(s.mean())
        res["prior_scores"][k] = float(p.mean())
        res["overlap_actual"][k] = len(set(rosters[k]) & set(actual))

    rng = random.Random(0)
    null = [random_valid(pool, locks, quota, rng) for _ in range(args.n_null)]
    ns = ev.score_rosters(null, opps).mean(1).numpy()
    npr = ev.score_rosters(null, opps, mode=alt).mean(1).numpy()
    res["null"] = {"mean": float(ns.mean()), "sd": float(ns.std()),
                   "q": {str(q): float(np.quantile(ns, q)) for q in (0.05, 0.5, 0.95, 0.99)}}
    res["percentile"] = {k: float((ns < v).mean()) for k, v in res["scores"].items()}
    # does the adversarial layer change rankings? (E6) Spearman across null rosters
    from scipy.stats import spearmanr
    res["rank_corr_stackelberg_vs_prior"] = float(spearmanr(ns, npr).correlation)
    res["null_scores"] = ns.tolist()
    res["null_prior_scores"] = npr.tolist()
    res["rapm_seasons"] = rapm_seasons
    (OUT / f"E3_{args.case}_{args.team}.json").write_text(json.dumps(res, indent=1, default=int))
    for k in names_list:
        print(f"{k:22s} score {res['scores'][k]:.4f}  prior {res['prior_scores'][k]:.4f}  "
              f"pct {res['percentile'][k]:.3f}  overlap {res['overlap_actual'][k]}")
    print("null", res["null"], "rank corr", res["rank_corr_stackelberg_vs_prior"])


if __name__ == "__main__":
    main()
