"""RAPM baseline from the NBA 2025-26 tracking archive.

Stints: lineups come from the `Lineup#show` stream in raw/stats.jsonl.gz (the same stream that
populates tracking on_court; its timestamps equal the PBP substitution timestamps). A stint is a
maximal run of PBP events (sorted by event timestamp, then PBP id) sharing one 10-man lineup.
Points: change in the PBP running score (scoreHome/scoreAway) at each event, attributed to the
lineup on court at the event timestamp (handles waived baskets, which `pts` does not).
Possessions per team: FGA + 0.44*FTA - OREB(player, not TEAM) + TOV; stint poss = mean of both teams.
Tracking on_court is used only as a validation (--validate-tracking): it is stale in coverage-warning
games, whereas the Lineup stream is not. Note the O/D model (two rows per stint, same weight, same
lambda) gives O-D coefficients exactly equal to the net-margin ridge, so RAPM == RAPM_net_model.

Usage: python rapm.py [--validate-tracking]
"""
import bisect, gzip, json, sys, zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from masked_paths import ARCHIVE as ZIP  # TRK_ARCHIVE env selects the season
OUT = Path(__file__).parent / "results/rapm"
SPLIT_DATE = "2026-04-15"
LAMBDAS = [10, 30, 100, 300, 1000, 2000, 3000, 5000, 10000, 30000]
RNG = np.random.default_rng(0)


def jl(z, name):
    return [json.loads(x) for x in gzip.decompress(z.read(name)).splitlines() if x.strip()]


def parse_game(prefix, validate=False):
    z = zipfile.ZipFile(ZIP)
    meta = json.loads(z.read(prefix + "metadata.json"))
    cov = json.loads(z.read(prefix + "coverage.json"))
    g = meta["game"]
    gid, date = meta["game_id"], meta["game_date"]
    rows = jl(z, prefix + "raw/stats.jsonl.gz")
    pbp, lineups = {}, {}
    for r in rows:
        if r["event"] == "PlayByPlay#show":
            m = r["metadata"]
            key = (r["updatedTime"], r["createdTime"])
            if m["id"] not in pbp or key >= pbp[m["id"]][0]:
                pbp[m["id"]] = (key, r["timestamp"], m)
        elif r["event"] == "Lineup#show":
            m = r["metadata"]
            if len(m.get("home") or []) == 5 and len(m.get("away") or []) == 5:
                k = (r["timestamp"], r["createdTime"])
                lineups[k] = (tuple(sorted(m["home"])), tuple(sorted(m["away"])))
    lk = sorted(lineups)
    lts = [k[0] for k in lk]
    lus = [lineups[k] for k in lk]
    evs = sorted(((ts, i, m) for i, (_, ts, m) in pbp.items()), key=lambda e: (e[0], e[1]))

    def lineup_at(ts):  # last lineup with timestamp <= ts
        j = bisect.bisect_right(lts, ts) - 1
        return lus[j] if j >= 0 else None

    stints, cur, prev = [], None, (0, 0)
    pts_field = [0, 0]
    diag = Counter()
    for ts, i, m in evs:
        et = m["eventType"]
        side = {"home": 0, "away": 1}.get(m.get("team"))
        try:
            score = (int(m["scoreHome"]), int(m["scoreAway"]))
        except (KeyError, ValueError, TypeError):
            score = prev
        d = (score[0] - prev[0], score[1] - prev[1])
        prev = score
        lu = lineup_at(ts)
        if et in ("2pt", "3pt", "freethrow") and side is not None:
            pts_field[side] += int(m.get("pts") or 0)
        if d != (0, 0) or et in ("2pt", "3pt", "freethrow"):
            diag["score_events"] += 1
            if lu is None:
                diag["score_events_no_lineup"] += 1
        if lu is None:
            diag["pts_lost"] += abs(d[0]) + abs(d[1])
            continue
        if cur is None or cur["lineup"] != lu or cur["period"] != m.get("period"):
            cur = dict(game_id=gid, date=date, period=m.get("period"), lineup=lu, t0=ts, t1=ts,
                       pts=[0, 0], fga=[0, 0], fta=[0, 0], oreb=[0, 0], tov=[0, 0], n_ev=0,
                       use={})  # per player: [fga, fta, tov, pts, [shot event ids]]
            stints.append(cur)
        cur["t1"] = ts
        cur["n_ev"] += 1
        cur["pts"][0] += d[0]
        cur["pts"][1] += d[1]
        if side is None:
            continue
        if m.get("personId") and et in ("2pt", "3pt", "freethrow", "rebound", "turnover"):
            # independent check: the PBP actor must be on court for his side in the attributed lineup
            diag["actor_checked"] += 1
            diag["actor_on_court"] += m["personId"] in lu[side]
        pid = m.get("personId")
        if pid and pid in lu[side] and et in ("2pt", "3pt", "freethrow", "turnover"):
            u = cur["use"].setdefault(pid, [0, 0, 0, 0, []])
            if et in ("2pt", "3pt"):
                u[0] += 1; u[4].append(i)
            elif et == "freethrow":
                u[1] += 1
            else:
                u[2] += 1
            u[3] += max(d[side], 0)
        if et in ("2pt", "3pt"):
            cur["fga"][side] += 1
        elif et == "freethrow":
            cur["fta"][side] += 1
        elif et == "rebound" and m["subType"] == "offensive" and m.get("personId"):
            cur["oreb"][side] += 1
        elif et == "turnover":
            cur["tov"][side] += 1
    for s in stints:
        pt = [s["fga"][k] + 0.44 * s["fta"][k] - s["oreb"][k] + s["tov"][k] for k in (0, 1)]
        s["poss_team"] = pt
        s["poss"] = (pt[0] + pt[1]) / 2
    final = (g["homeTeam"]["score"], g["awayTeam"]["score"])
    stint_pts = (sum(s["pts"][0] for s in stints), sum(s["pts"][1] for s in stints))
    warn = bool(cov.get("periods_without_coordinate_observations") or
                cov.get("period_clock_labels_without_coordinate_observations"))
    out = dict(game_id=gid, date=date, home=g["homeTeam"]["teamId"], away=g["awayTeam"]["teamId"],
               home_tri=g["homeTeam"]["teamTricode"], away_tri=g["awayTeam"]["teamTricode"],
               final=final, pbp_final=prev, stint_pts=stint_pts, pts_field=tuple(pts_field),
               coverage_warning=warn, n_lineup_events=len(lus), diag=dict(diag), stints=stints,
               players={int(k): v["name"] for k, v in meta.get("players", {}).items()})
    if validate:  # compare tracking on_court at each scoring event timestamp with the Lineup stream
        tr = jl(z, prefix + "tracking.jsonl.gz")
        tts = [r["timestamp_ms"] for r in tr]
        agree = n = unk = 0
        for ts, i, m in evs:
            if m["eventType"] not in ("2pt", "3pt", "freethrow"):
                continue
            j = bisect.bisect_right(tts, ts + 150) - 1  # tracking lineup lags events by ~0.25s at most
            r = tr[j] if j >= 0 else None
            if r is None or not r["quality"].get("lineup_known") or not r.get("on_court"):
                unk += 1
                continue
            n += 1
            agree += (tuple(sorted(r["on_court"]["home"])), tuple(sorted(r["on_court"]["away"]))) == lineup_at(ts)
        out["tracking_check"] = dict(n=n, agree=agree, unknown=unk)
    return out


def phase(gid):
    return {"002": "regular", "004": "playoffs", "005": "play-in"}.get(gid[:3], "other")


def build(validate):
    z = zipfile.ZipFile(ZIP)
    prefixes = sorted({n.rsplit("/", 1)[0] + "/" for n in z.namelist()
                       if n.endswith("/metadata.json") and "/games/" in n})
    with Pool(10) as p:
        games = p.starmap(parse_game, [(x, validate) for x in prefixes], chunksize=4)
    return sorted(games, key=lambda g: (g["date"], g["game_id"]))


# ---------------------------------------------------------------- ridge
def design(stints, pidx, net):
    """net=True: one row per stint, y = home margin/100, +1 home / -1 away.
    net=False: two rows per stint (each side on offense), cols [offense P | defense P]."""
    P = len(pidx)
    from scipy.sparse import csr_matrix
    r, c, v, y, w, h = [], [], [], [], [], []
    k = 0
    for s in stints:
        hp, ap = s["lineup"]
        if net:
            for pl, sg in [(x, 1) for x in hp] + [(x, -1) for x in ap]:
                if pl in pidx:
                    r.append(k); c.append(pidx[pl]); v.append(sg)
            y.append(100 * (s["pts"][0] - s["pts"][1]) / s["poss"]); w.append(s["poss"]); h.append(1.0); k += 1
        else:
            for off, de, side in ((hp, ap, 0), (ap, hp, 1)):
                for pl in off:
                    if pl in pidx:
                        r.append(k); c.append(pidx[pl]); v.append(1)
                for pl in de:
                    if pl in pidx:
                        r.append(k); c.append(P + pidx[pl]); v.append(1)
                y.append(100 * s["pts"][side] / s["poss"]); w.append(s["poss"]); h.append(1.0 if side == 0 else 0.0); k += 1
    X = csr_matrix((v, (r, c)), shape=(k, P if net else 2 * P))
    return X, np.array(y), np.array(w), np.array(h)


def ridge(X, y, w, h, lam):
    """Weighted ridge; unpenalized intercept + home column. Returns (b0, bh, beta)."""
    from scipy.sparse import hstack, csr_matrix
    A = hstack([csr_matrix(np.c_[np.ones(len(y)), h]), X]).tocsr()
    Aw = A.multiply(w[:, None]).tocsr()
    G = (A.T @ Aw).toarray()
    pen = np.full(G.shape[0], lam); pen[:2] = 0
    if net_only_intercept := (h.min() == h.max()):  # net model: h is constant -> drop dup column
        pen[1] = 1e9
    G[np.diag_indices_from(G)] += pen
    b = np.linalg.solve(G, Aw.T @ y)
    if net_only_intercept:
        b[0] += b[1]; b[1] = 0.0
    return b[0], b[1], b[2:]


def predict_margin(stints, pidx, fit_net=None, fit_od=None):
    """Predicted home margin per 100 for each stint."""
    P = len(pidx)
    out = []
    for s in stints:
        hp, ap = s["lineup"]
        hi = [pidx[x] for x in hp if x in pidx]; ai = [pidx[x] for x in ap if x in pidx]
        if fit_net is not None:
            b0, _, be = fit_net
            out.append(b0 + be[hi].sum() - be[ai].sum())
        else:
            b0, bh, be = fit_od
            o, d = be[:P], be[P:]
            out.append((b0 + bh + o[hi].sum() + d[ai].sum()) - (b0 + o[ai].sum() + d[hi].sum()))
    return np.array(out)


def wmetrics(pred, y, w):
    mse = np.sum(w * (y - pred) ** 2) / w.sum()
    ybar = np.sum(w * y) / w.sum()
    sst = np.sum(w * (y - ybar) ** 2) / w.sum()
    pm = np.sum(w * pred) / w.sum()
    cov = np.sum(w * (pred - pm) * (y - ybar)); vp = np.sum(w * (pred - pm) ** 2)
    corr2 = cov ** 2 / (vp * sst * w.sum()) if vp > 0 else 0.0
    return dict(rmse=float(np.sqrt(mse)), r2=float(1 - mse / sst), corr2=float(corr2))


def main():
    validate = "--validate-tracking" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    games = build(validate)
    names = {}
    for g in games:
        names.update(g["players"])

    # ------------------------------------------------ reconciliation
    rec = dict(n_games=len(games))
    ok = lambda a, b: tuple(a) == tuple(b)
    rec["pbp_final_matches_metadata"] = sum(ok(g["pbp_final"], g["final"]) for g in games)
    rec["stint_points_match_final"] = sum(ok(g["stint_pts"], g["final"]) for g in games)
    rec["pts_field_sum_matches_final"] = sum(ok(g["pts_field"], g["final"]) for g in games)
    rec["rate_stint_points_match_final"] = rec["stint_points_match_final"] / len(games)
    cw = [g for g in games if g["coverage_warning"]]
    rec["coverage_warning_games"] = len(cw)
    rec["coverage_warning_games_reconciled"] = sum(ok(g["stint_pts"], g["final"]) for g in cw)
    tot = Counter()
    for g in games:
        tot.update(g["diag"])
    rec["score_events"] = tot["score_events"]
    rec["score_events_without_lineup"] = tot["score_events_no_lineup"]
    rec["pbp_actor_in_attributed_lineup_rate"] = tot["actor_on_court"] / max(tot["actor_checked"], 1)
    rec["pbp_actor_events_checked"] = tot["actor_checked"]
    rec["games_without_pbp"] = [g["game_id"] for g in games if not g["stints"]]
    rec["rate_stint_points_match_final_games_with_pbp"] = rec["stint_points_match_final"] / max(1, len(games) - len(rec["games_without_pbp"]))
    rec["mismatched_games"] = [dict(game_id=g["game_id"], final=g["final"], pbp_final=g["pbp_final"],
                                    stint_pts=g["stint_pts"], pts_field=g["pts_field"])
                               for g in games if not ok(g["stint_pts"], g["final"])][:30]
    rec["pts_field_mismatch_examples"] = [dict(game_id=g["game_id"], final=g["final"], pts_field=g["pts_field"])
                                          for g in games if not ok(g["pts_field"], g["final"])][:10]
    if validate:
        for lab, flag in (("no_coverage_warning", False), ("coverage_warning", True)):
            tc = Counter()
            for g in games:
                if g["coverage_warning"] == flag:
                    tc.update(g["tracking_check"])
            rec["tracking_on_court_vs_lineup_stream_" + lab] = dict(tc, agree_rate=tc["agree"] / max(tc["n"], 1))
    print(json.dumps({k: v for k, v in rec.items() if "examples" not in k and "mismatched" not in k}, indent=1))

    # ------------------------------------------------ stints
    stints = [s for g in games for s in g["stints"]]  # the 10 games without PBP contribute no stints
    n_all = len(stints)
    pts_drop = sum(abs(s["pts"][0]) + abs(s["pts"][1]) for s in stints if s["poss"] <= 0)
    stints = [s for s in stints if s["poss"] > 0]
    print(f"stints {n_all}, with poss>0 {len(stints)} (dropped pts {pts_drop}), poss {sum(s['poss'] for s in stints):.0f}")

    train = [s for s in stints if s["date"] < SPLIT_DATE and phase(s["game_id"]) == "regular"]
    test = [s for s in stints if s["date"] >= SPLIT_DATE]
    pidx = {p: i for i, p in enumerate(sorted({x for s in train for side in s["lineup"] for x in side}))}

    # ------------------------------------------------ time-ordered CV for lambda (expanding window)
    dates = sorted({s["date"] for s in train})
    cuts = [dates[int(len(dates) * f)] for f in (0.55, 0.7, 0.85)]
    cv = {}
    for lam in LAMBDAS:
        errs = {"net": [0, 0], "od": [0, 0]}
        for i, c in enumerate(cuts):
            end = cuts[i + 1] if i + 1 < len(cuts) else "9999"
            tr = [s for s in train if s["date"] < c]; va = [s for s in train if c <= s["date"] < end]
            pi = {p: j for j, p in enumerate(sorted({x for s in tr for side in s["lineup"] for x in side}))}
            ya = np.array([100 * (s["pts"][0] - s["pts"][1]) / s["poss"] for s in va]); wa = np.array([s["poss"] for s in va])
            fn = ridge(*design(tr, pi, True), lam)
            fo = ridge(*design(tr, pi, False), lam)
            for key, pr in (("net", predict_margin(va, pi, fit_net=fn)), ("od", predict_margin(va, pi, fit_od=fo))):
                errs[key][0] += np.sum(wa * (ya - pr) ** 2); errs[key][1] += wa.sum()
        cv[lam] = {k: float(np.sqrt(a / b)) for k, (a, b) in errs.items()}
        print(f"lambda {lam:>6}: CV wRMSE net {cv[lam]['net']:.3f}  O/D {cv[lam]['od']:.3f}")
    lam_net = min(cv, key=lambda l: cv[l]["net"]); lam_od = min(cv, key=lambda l: cv[l]["od"])

    # ------------------------------------------------ holdout
    fn = ridge(*design(train, pidx, True), lam_net)
    fo = ridge(*design(train, pidx, False), lam_od)
    yt = np.array([100 * (s["pts"][0] - s["pts"][1]) / s["poss"] for s in test]); wt = np.array([s["poss"] for s in test])
    gt = np.array([s["game_id"] for s in test])
    # team net rating baseline from training stints (team id via game home/away)
    gteam = {g["game_id"]: (g["home"], g["away"]) for g in games}
    tm = defaultdict(lambda: [0.0, 0.0])
    for s in train:
        h, a = gteam[s["game_id"]]
        tm[h][0] += s["pts"][0] - s["pts"][1]; tm[h][1] += s["poss"]
        tm[a][0] += s["pts"][1] - s["pts"][0]; tm[a][1] += s["poss"]
    ytr = np.array([100 * (s["pts"][0] - s["pts"][1]) / s["poss"] for s in train]); wtr = np.array([s["poss"] for s in train])
    hca = float(np.sum(wtr * ytr) / wtr.sum())
    nr = {t: 100 * v[0] / v[1] for t, v in tm.items()}
    preds = {
        "rapm_net": predict_margin(test, pidx, fit_net=fn),
        "rapm_od": predict_margin(test, pidx, fit_od=fo),
        "zero": np.zeros(len(test)),
        "home_court_only": np.full(len(test), hca),
        "team_net_rating": np.array([hca + nr.get(gteam[s["game_id"]][0], 0) - nr.get(gteam[s["game_id"]][1], 0) for s in test]),
    }
    hold = {k: wmetrics(p, yt, wt) for k, p in preds.items()}
    # secondary: aggregate stints to game level (poss-weighted mean prediction vs actual game margin/100)
    ug0 = sorted(set(gt)); gw = np.array([wt[gt == g].sum() for g in ug0])
    gy = np.array([np.sum(wt[gt == g] * yt[gt == g]) / wt[gt == g].sum() for g in ug0])
    hold_game = {k: wmetrics(np.array([np.sum(wt[gt == g] * p[gt == g]) / wt[gt == g].sum() for g in ug0]), gy, gw)
                 for k, p in preds.items()}
    for k, m in hold_game.items():
        print(f"game-level {k:>16}: wRMSE {m['rmse']:.2f}  R2 {m['r2']:+.3f}  corr^2 {m['corr2']:.3f}")
    ug = np.unique(gt); gi = {g: np.where(gt == g)[0] for g in ug}
    boot = defaultdict(list)
    for _ in range(1000):
        idx = np.concatenate([gi[g] for g in RNG.choice(ug, len(ug))])
        m = {k: wmetrics(p[idx], yt[idx], wt[idx]) for k, p in preds.items()}
        for k in m:
            boot[k + ".rmse"].append(m[k]["rmse"]); boot[k + ".r2"].append(m[k]["r2"])
        for k in ("rapm_net", "rapm_od"):
            boot[k + ".rmse_minus_team_net_rating"].append(m[k]["rmse"] - m["team_net_rating"]["rmse"])
            boot[k + ".rmse_minus_zero"].append(m[k]["rmse"] - m["zero"]["rmse"])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
    for k, m in hold.items():
        print(f"holdout {k:>16}: wRMSE {m['rmse']:.3f} [{ci[k+'.rmse'][0]:.3f},{ci[k+'.rmse'][1]:.3f}]  "
              f"R2 {m['r2']:+.4f} [{ci[k+'.r2'][0]:+.4f},{ci[k+'.r2'][1]:+.4f}]  corr^2 {m['corr2']:.4f}")
    for k in ci:
        if "minus" in k:
            print(f"  {k}: 95% CI {ci[k][0]:+.3f}, {ci[k][1]:+.3f}")

    # ------------------------------------------------ full-sample fit for players.csv
    pall = {p: i for i, p in enumerate(sorted({x for s in stints for side in s["lineup"] for x in side}))}
    P = len(pall)
    b0n, _, bn = ridge(*design(stints, pall, True), lam_net)
    b0, bh, bo = ridge(*design(stints, pall, False), lam_od)
    poss = Counter()
    for s in stints:
        for side in s["lineup"]:
            for x in side:
                poss[x] += s["poss"]
    rows = sorted(((p, names.get(p, ""), poss[p], bo[i], -bo[P + i], bo[i] - bo[P + i], bn[i]) for p, i in pall.items()),
                  key=lambda r: -r[5])
    with open(OUT / "players.csv", "w") as f:
        f.write("player_id,name,possessions,ORAPM,DRAPM,RAPM,RAPM_net_model\n")
        for r in rows:
            f.write(f'{r[0]},"{r[1]}",{r[2]:.1f},{r[3]:.3f},{r[4]:.3f},{r[5]:.3f},{r[6]:.3f}\n')
    q = [r for r in rows if r[2] >= 1000]
    print(f"\nTop 15 RAPM (O+D, min 1000 poss of {len(rows)} players)")
    for r in q[:15]:
        print(f"  {r[1]:<26} poss {r[2]:7.0f}  O {r[3]:+5.2f}  D {r[4]:+5.2f}  RAPM {r[5]:+5.2f}  net-model {r[6]:+5.2f}")
    print("Bottom 15")
    for r in q[-15:]:
        print(f"  {r[1]:<26} poss {r[2]:7.0f}  O {r[3]:+5.2f}  D {r[4]:+5.2f}  RAPM {r[5]:+5.2f}  net-model {r[6]:+5.2f}")

    summary = dict(
        reconciliation=rec, n_stints=len(stints), n_stints_before_poss_filter=n_all,
        points_in_dropped_zero_poss_stints=pts_drop, n_possessions=float(sum(s["poss"] for s in stints)),
        n_players=P, lambda_net=lam_net, lambda_od=lam_od, cv_rmse=cv, cv_cuts=cuts,
        full_fit=dict(intercept_od=float(b0), home_offense_bonus=float(bh), intercept_net=float(b0n)),
        holdout=dict(split_date=SPLIT_DATE, n_train_stints=len(train), n_test_stints=len(test),
                     n_test_games=int(len(ug)), train_poss=float(wtr.sum()), test_poss=float(wt.sum()),
                     metrics=hold, game_level_metrics=hold_game, bootstrap_95ci=ci, n_boot=1000,
                     test_players_unseen_in_train_share_of_poss=float(
                         sum(s["poss"] for s in test for side in s["lineup"] for x in side if x not in pidx) / (10 * wt.sum()))),
        possessions_formula="per team FGA + 0.44*FTA - OREB(player only) + TOV; stint poss = mean of the two teams",
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
