"""Run roster searches and audits for one case study.

  python v3/run_case.py --case paris --team USA --tag main --seeds 5 --out main
  python v3/run_case.py --case paris --team USA --tag main --seeds 5 --no-locks --out unlocked
  python v3/run_case.py --case 4nf --team CAN --tag main --seeds 5 --out main

Writes results/<case>_<team>_<out>.json with: searched rosters per seed, scores,
per-fold-model scores, WAR, the actual roster's score and WAR, and the
distance-to-selection of every actual-roster and model-roster player.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import cases as C
from rosters import Evaluator
from search import Quota, run_search

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v3" / "results"
OUT.mkdir(exist_ok=True, parents=True)


def setup(ev, case, team, min_minutes):
    ps = ev.S.ps
    season = ev.season
    if case == "paris":
        idx = C.name_index(ps, ["name", "name_bio"])
        country = ps.set_index("player_id")["country"].to_dict()
        for n, c in C.NBA_COUNTRY_OVERRIDE.items():
            if C.norm(n) in idx:
                country[idx[C.norm(n)]] = c
        cur = ps[ps["season"] == season]
        nat = {"USA": "USA", "Canada": "Canada", "France": "France"}
        pool = [int(p) for p, m in zip(cur["player_id"], cur["Minutes"])
                if country.get(p) == nat[team] and m >= min_minutes]
        actual, miss = C.resolve(C.PARIS[team], idx)
        # players with no NBA record (e.g. Melvin Ejim) enter at replacement level
        placeholders = {}
        for k, n in enumerate(miss):
            pid = -(1000 + k)
            actual.append(pid)
            placeholders[pid] = n
        locks, _ = C.resolve(C.NBA_LOCKS[team], idx)
        opps = []
        for o in C.NBA_OPP[team]:
            if o in C.PARIS:
                r, _ = C.resolve(C.PARIS[o], idx, strict=False)
                r += [-(k + 1) for k in range(12 - len(r))]  # unresolved -> replacement level
            elif o == "France":
                fr = [int(p) for p in cur["player_id"] if country.get(p) == "France"]
                r = C.by_minutes(ps, season, fr, 12, "Minutes")
            else:
                r = C.by_minutes(ps, season, cur[cur["team"] == o]["player_id"].tolist(), 12, "Minutes")
            r += [-(k + 100) for k in range(12 - len(r))]
            opps.append(r)
        pos = dict(zip(ps["player_id"], ps["position"]))
        pos.update({pid: C.NON_NBA_POS.get(n, "Forward") for pid, n in placeholders.items()})
        quota_pos = pos
        names = dict(zip(ps["player_id"], ps["name"].fillna(ps["name_bio"])))
        names.update(placeholders)
    else:
        ps = ps.copy()
        idx = C.name_index(ps, ["name"])
        cur = ps[ps["season"] == season]
        pool = [int(p) for p, m, b, ps_ in zip(cur["player_id"], cur["minutes_total"], cur["birthCountry"], cur["pos"])
                if b == team and m >= min_minutes and ps_ != "G"]

        def dressed(t):
            f, _ = C.resolve(C.FOUR_NATIONS[t]["F"], idx, strict=(t == team))
            d, _ = C.resolve(C.FOUR_NATIONS[t]["D"], idx, strict=(t == team))
            return (C.by_minutes(ps, season, f, 12, "minutes_total")
                    + C.by_minutes(ps, season, d, 6, "minutes_total"))
        actual = dressed(team)
        locks, _ = C.resolve(C.NHL_LOCKS[team], idx)
        opps = [dressed(o) for o in C.NHL_OPP[team]]
        cur_role = ps[ps["season"] == season]
        quota_pos = {int(p): ("D" if po == "D" else ("C" if fo >= 15 else "W"))
                     for p, po, fo in zip(cur_role["player_id"], cur_role["pos"], cur_role["fo60"])}
        names = dict(zip(ps["player_id"], ps["name"]))
    pool = sorted(set(pool) | set(actual) | set(locks))
    return pool, actual, locks, opps, quota_pos, names


def distance_to_selection(ev, best, best_s, players, pool_roster_check, quota, opps):
    """For each player not in `best`: the score of the best quota-valid roster
    obtained by swapping that player into `best`, minus best_s (<= 0 means the
    model prefers `best`). For players in `best`: score of the best valid roster
    without them (forced out, best replacement from the pool) minus best_s."""
    out = {}
    for p in players:
        cands = []
        if p in best:
            for q in pool_roster_check:
                if q in best or not quota.same_group(p, q):
                    continue
                new = [q if x == p else x for x in best]
                if quota.ok(new):
                    cands.append(new)
        else:
            for i, x in enumerate(best):
                if not quota.same_group(x, p):
                    continue
                new = best[:i] + [p] + best[i + 1:]
                if quota.ok(new):
                    cands.append(new)
        if not cands:
            out[p] = None
            continue
        s = ev.score_rosters(cands, opps).mean(1).numpy()
        out[p] = float(s.max() - best_s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["paris", "4nf"])
    ap.add_argument("--team", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-locks", action="store_true")
    ap.add_argument("--no-quota", action="store_true")
    ap.add_argument("--mode", default="stackelberg")
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--drop-opp", type=int, default=None)
    ap.add_argument("--min-minutes", type=float, default=None)
    ap.add_argument("--audit", action="store_true", help="WAR + distance-to-selection")
    ap.add_argument("--screen", type=int, default=128, help="greedy shortlist size (0 = no screen)")
    ap.add_argument("--scorer", default="value", choices=["value", "gnn"],
                    help="value: additive player-value model (main); gnn: graph scorer (appendix)")
    ap.add_argument("--order", default="coach", choices=["coach", "club"],
                    help="value scorer: rotation by model value (coach) or by club minutes (club)")
    ap.add_argument("--decay", type=float, default=0.75, help="value scorer: weight per season back")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sport = "nba" if args.case == "paris" else "nhl"
    season = 2024 if sport == "nba" else 2025
    if args.scorer == "value":
        from values import ValueEvaluator
        ev = ValueEvaluator(sport, season, order=args.order, decay=args.decay)
    else:
        ev = Evaluator(sport, args.tag, season=season, mode=args.mode, lam=args.lam)
    mm = args.min_minutes or (500 if sport == "nba" else 300)
    pool, actual, locks, opps, qpos, names = setup(ev, args.case, args.team, mm)
    # negative ids are replacement-level placeholders for unknown opponents
    ev.mpg.update({p: 0.0 for o in opps + [actual] for p in o if p < 0})
    if args.drop_opp is not None:
        opps = [o for k, o in enumerate(opps) if k != args.drop_opp]
    if args.no_locks:
        locks = []
    quota = Quota(sport, qpos, enforce=not args.no_quota)
    print(f"{args.case} {args.team}: pool {len(pool)}, actual {len(actual)}, locks {len(locks)}", flush=True)
    print("  actual quota-valid:", quota.ok(actual), flush=True)

    res = {"args": vars(args), "pool_size": len(pool), "names": {str(k): v for k, v in names.items()
                                                                 if k in set(pool) | {p for o in opps for p in o}},
           "actual": actual, "locks": locks, "opponents": opps, "runs": []}
    actual_scores = ev.score_rosters([actual], opps)
    pm = None
    for seed in range(args.seeds):
        t0 = time.time()
        roster, s, hist = run_search(ev, pool, opps, locks, quota, seed=seed, screen=args.screen)
        res["runs"].append({"seed": seed, "roster": roster, "score": s,
                            "history": [list(map(lambda v: v if not isinstance(v, (np.integer,)) else int(v), h))
                                        for h in hist],
                            "overlap_actual": len(set(roster) & set(actual)),
                            "minutes": time.time() - t0})
        print(f"seed {seed}: score {s:.4f}, overlap {len(set(roster) & set(actual))}/{len(actual)}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        (OUT / f"{args.case}_{args.team}_{args.out}.json").write_text(json.dumps(res, indent=1, default=int))

    best = max(res["runs"], key=lambda r: r["score"])
    res["best"] = best["roster"]
    res["best_score"] = best["score"]
    res["actual_score"] = float(actual_scores.mean())
    res["actual_score_by_opp"] = actual_scores[0].tolist()
    # per-fold-model scores (uncertainty) for best and actual, in the same deployment
    for key, r in [("best", best["roster"]), ("actual", actual)]:
        if hasattr(ev, "score_draws"):  # bootstrap draws of the value model
            res[f"{key}_per_model"] = ev.score_draws(r, opps)
            continue
        X, _ = ev.tensor(r)
        per = []
        for m in ev.model.models:
            full = ev.model
            ev.model = type(full)([m])
            per.append(float(ev.score_tensors(X.unsqueeze(0), [ev.tensor(o)[0] for o in opps]).mean()))
            ev.model = full
        res[f"{key}_per_model"] = per
    if hasattr(ev, "values_table"):
        res["values"] = {str(k): v for k, v in ev.values_table(sorted(set(pool) | set(actual))).items()}
        res["rotation_best"] = [[int(p), float(q)] for p, q in zip(*ev.deploy(best["roster"]))]
        res["rotation_actual"] = [[int(p), float(q)] for p, q in zip(*ev.deploy(actual))]
    if args.audit:
        s_b, war_b = ev.war(best["roster"], opps)
        s_a, war_a = ev.war(actual, opps)
        res["war_best"] = {str(k): v for k, v in war_b.items()}
        res["war_actual"] = {str(k): v for k, v in war_a.items()}
        players = sorted(set(actual) | set(best["roster"]))
        res["distance"] = {str(k): v for k, v in distance_to_selection(
            ev, best["roster"], best["score"], players, pool, quota, opps).items()}
    (OUT / f"{args.case}_{args.team}_{args.out}.json").write_text(json.dumps(res, indent=1, default=int))
    print("best", best["score"], "actual", res["actual_score"])
    print("best roster:", [names.get(p, p) for p in best["roster"]])


if __name__ == "__main__":
    main()
