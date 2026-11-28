"""Pregame availability benchmark. Protocol (fixed before results): AVAIL_BENCH.md. Development data only.

Arms on identical eligible games and DARKO ratings: P_none (recent minutes), P0 (proxy of the live shares() rule:
recent minutes x fixed status share, normalised to 5 on court), B2 (frozen IGGY B2 applied with the pregame Out set),
L1 / L2 (new development models). Reports: results/injury_hist/verified.json (fetch_injury_history.py --verify).
  python avail_bench.py               (PC: B2 needs the stint files for start rates)
  python avail_bench.py --smoke OUT   (laptop code check, start rate 0; never reported)
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

import iggy_minutes as ig
import injuries as inj
from fetch_injury_history import SUF
from masked_paths import ROOT
from provenance import provenance

OUT = ROOT / 'results/avail'
ST = ['none', 'available', 'probable', 'questionable', 'doubtful', 'out']
FIXED = dict(none=1.0, **inj.AVAIL)
HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)   # = iggy_minutes


def match(rows, team, ck, first=None, last=None):
    """Exact: first + last name key. Fallback (fixed before results, counted): same surname key and one first name a
    prefix of the other (Alexandre / Alex, Christopher / Chris); never across different first names (Jalen / Jaylin)."""
    ck = SUF.sub('', ck)
    for i, (t, kb, kf, st) in enumerate(rows):
        if t == team and kf and ck.startswith(kf) and len(ck) - len(kf) >= 2 and kb.endswith(ck[len(kf):]):
            return i, st
    if first and last and len(last) >= 4:
        for i, (t, kb, kf, st) in enumerate(rows):
            if t == team and kf and kb.endswith(last) and (kf.startswith(first) or first.startswith(kf)) and min(len(kf), len(first)) >= 3:
                return i, st
    return None, 'none'


def report_state(V):
    """(date, team) -> 'valid' | 'not_submitted'; date -> parsed rows (valid reports only)."""
    state, rows = {}, {}
    for d, v in V.items():
        if not v['valid']:
            continue
        rows[d] = [tuple(x) for x in v['parsed']]
        for t in v['teams_in_report']:
            state[(d, t)] = 'not_submitted' if t in v['not_yet_submitted'] else 'valid'
    return state, rows


def b2_pregame(D):
    """Frozen B2 features with the PREGAME absent set (rotation & listed Out). Past responses use realised absences
    of earlier games (as frozen); the current game's absent set comes only from the report."""
    D = D.copy()
    D['absent'] = D.rot & (D.status == 'out')
    grp = D.groupby(['gid', 'team'])
    D['abs_min'] = (D.base * D.absent).groupby([D.gid, D.team]).transform('sum')
    D['rank'] = grp.base.rank(ascending=False)
    D['n_abs'] = grp.absent.transform('sum')
    hist = defaultdict(list); resp = np.full(len(D), np.nan)
    for (gid, team), g in D.groupby(['gid', 'team'], sort=False):
        d = g.d.iloc[0]; absent = g.pid[g.absent.values].tolist()
        for idx, r in zip(g.index, g.itertuples()):
            if r.absent:
                continue
            past = [x for j in absent for (dd, x) in hist.get((team, r.pid, j), []) if (d - dd).days <= 365]
            if past:
                resp[idx] = float(np.mean(past))
        real_abs = g.pid[(g.rot & ~g.played).values].tolist()
        for r in g.itertuples():           # recorded after this game's prediction, from realised absences
            if r.played:
                for j in real_abs:
                    hist[(team, r.pid, j)].append((d, r.y - r.base))
    D['resp0'] = np.nan_to_num(resp); D['has_resp'] = ~np.isnan(resp)
    return D


def shares(D, m):
    """live_forecast.shares: 5 x minutes / team total."""
    tot = pd.Series(m).groupby([D.gid.values, D.team.values]).transform('sum').values
    return np.where(tot > 0, 5 * m / np.maximum(tot, 1e-9), 0.0)


def project_team(D, v, absent):
    out = np.zeros(len(D))
    for _, g in D.groupby(['gid', 'team'], sort=False):
        ix = g.index.values; ok = ~absent[ix]
        cap = np.minimum(ig.CAP, np.maximum(g.base.values[ok], 10.0) + 14.0)
        out[ix[ok]] = ig.project(v[ix[ok]], cap)
    return out


def reproduce(D0, b2, tol=0.005):
    """Gate (AVAIL_BENCH.md): the reconstructed B2, scored as frozen (realised absences), must reproduce the saved
    benchmark results/iggy/minutes.json: identical test row counts per stratum and B0/B2 MAE within `tol`; unique
    game/team/player keys on both sides; and the saved per-row predictions (required; PC) matched one-to-one within 0.01.
    Reads the reference outputs only; never writes to results/iggy."""
    saved = json.loads((ROOT / 'results/iggy/minutes.json').read_text())['test']
    P = {'B0': D0.base.values.copy(), 'B2': b2.predict(D0[ig.FEATS['B2']].values)}
    out = {k: np.zeros(len(D0)) for k in P}
    for _, g in D0.groupby(['gid', 'team'], sort=False):
        ix = g.index.values; av = ~g.absent.values
        cap = np.minimum(ig.CAP, np.maximum(g.base.values[av], 10.0) + 14.0)
        for k in P:
            out[k][ix[av]] = ig.project(P[k][ix[av]], cap)
    te = (D0.season >= 2026) & ~D0.absent; y = D0.y.values
    strata = {'none': D0.n_abs == 0, 'single': D0.n_abs == 1, 'multi': D0.n_abs >= 2}
    res, ok = {}, True
    for s, m in strata.items():
        mk = (te & m).values
        r = dict(rows=int(mk.sum()), saved_rows=saved[s]['rows'])
        ok &= r['rows'] == r['saved_rows']
        for k in P:
            r[k] = round(float(np.mean(np.abs(out[k][mk] - y[mk]))), 3); r[k + '_saved'] = saved[s][k]['mae']
            ok &= abs(r[k] - r[k + '_saved']) <= tol
        res[s] = r
    f = ROOT / 'results/iggy/minutes_predictions.parquet'
    K = ['gid', 'team', 'pid']
    dup_rec = int(D0.duplicated(K).sum())
    if f.exists():
        S = pd.read_parquet(f)[K + ['pred_B2']]
        dup_saved = int(S.duplicated(K).sum())
        res['keys'] = dict(reconstructed_rows=len(D0), reconstructed_duplicates=dup_rec, saved_rows=len(S), saved_duplicates=dup_saved)
        ok &= dup_rec == 0 and dup_saved == 0
        if dup_rec == 0 and dup_saved == 0:
            M = D0[K].assign(rec=out['B2']).merge(S, on=K, how='outer', indicator=True, validate='one_to_one')
            both = M._merge == 'both'
            res['per_row'] = dict(rows_both=int(both.sum()), only_reconstructed=int((M._merge == 'left_only').sum()),
                                  only_saved=int((M._merge == 'right_only').sum()),
                                  max_abs_diff=round(float(np.abs(M.rec[both] - M.pred_B2[both]).max()), 6))
            ok &= res['per_row']['only_reconstructed'] == 0 and res['per_row']['only_saved'] == 0 and res['per_row']['max_abs_diff'] <= 0.01
        else:
            res['per_row'] = 'not compared: duplicate game/team/player keys'
    else:
        res['keys'] = dict(reconstructed_rows=len(D0), reconstructed_duplicates=dup_rec)
        res['per_row'] = 'MISSING: saved per-row predictions required for the real run'
        ok = False
    res['tolerance_mae'] = tol; res['passed'] = bool(ok)
    return res


def main():
    smoke = '--smoke' in sys.argv
    out_dir = Path(sys.argv[sys.argv.index('--smoke') + 1]) if smoke else OUT
    if smoke:
        ig.starters = lambda: set()
    # 1. frozen IGGY build + frozen B2 training (realised absences, seasons <= 2024), exactly as iggy_minutes.main
    D0, gl, pg = ig.build()
    D0 = ig.add_features(D0, pg)
    tr0 = (D0.season <= 2024) & ~D0.absent
    b2 = HistGradientBoostingRegressor(**HGB).fit(D0.loc[tr0, ig.FEATS['B2']].values, D0.loc[tr0, 'y'].values)
    gate = reproduce(D0, b2)
    print('B2 reproduction gate:', json.dumps(gate), flush=True)
    if not gate['passed'] and not smoke:
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(dict(protocol='AVAIL_BENCH.md', reproduction_gate=gate, stopped='reconstructed B2 does not reproduce the saved benchmark'),
                  open(OUT / 'avail_bench.json', 'w'), indent=1)
        raise SystemExit('B2 reproduction gate failed: no pregame results computed')
    # 2. pregame report status for candidate rows from 2022-23 on
    V = json.loads((ROOT / 'results/injury_hist/verified.json').read_text())
    state, rows = report_state(V)
    D = D0[D0.season >= 2023][['gid', 'd', 'team', 'pid', 'base', 'mins10', 'avail', 'startr', 'y', 'played', 'season', 'rot']]
    D = D.sort_values(['d', 'gid', 'team']).reset_index(drop=True)
    names = {int(p): inj.key(n) for p, n in zip(gl.player_id, gl.player_name)}
    parts = {int(p): (inj.key(n.split(' ', 1)[0]), SUF.sub('', inj.key(n.split(' ', 1)[1])) if ' ' in n else '')
             for p, n in zip(gl.player_id, gl.player_name)}
    ds = D.d.dt.strftime('%Y-%m-%d').values
    D['rstate'] = [('no_report' if d not in rows else state.get((d, int(t)), 'absent_from_report')) for d, t in zip(ds, D.team)]
    stt, used, n_alias = [], defaultdict(set), 0
    for d, t, p, s_ in zip(ds, D.team, D.pid, D.rstate):
        if s_ != 'valid':
            stt.append('none'); continue
        i, st = match(rows[d], int(t), names.get(int(p), ''), *parts.get(int(p), (None, None)))
        if i is not None and match(rows[d], int(t), names.get(int(p), ''))[0] is None:
            n_alias += 1
        if i is not None:
            used[d].add(i)
        stt.append(st)
    D['status'] = stt
    unmatched = defaultdict(int)
    for d, rr in rows.items():
        playing = {int(t) for t in D.team[ds == d]}
        for i, (t, kb, kf, st) in enumerate(rr):
            if t in playing and state.get((d, t)) == 'valid' and i not in used[d]:
                unmatched[st] += 1
    # 3. arms
    for s in ST[1:]:
        D['st_' + s] = (D.status == s).astype(float)
    P = {'P_none': D.base.values.copy(), 'P0': D.base.values * D.status.map(FIXED).values}
    Db = b2_pregame(D)
    P['B2'] = project_team(Db, b2.predict(Db[ig.FEATS['B2']].values), Db.absent.values)
    lst = D.rot & D.status.isin(['out', 'doubtful']); q = D.rot & (D.status == 'questionable')
    D['out_min'] = (D.base * lst).groupby([D.gid, D.team]).transform('sum') - D.base * lst
    D['q_min'] = (D.base * q).groupby([D.gid, D.team]).transform('sum') - D.base * q
    D['n_out'] = lst.groupby([D.gid, D.team]).transform('sum') - lst
    D['rank'] = Db['rank']; D['resp0'] = Db.resp0; D['has_resp'] = Db.has_resp.astype(float)
    F1 = ['base', 'mins10', 'avail', 'rank'] + ['st_' + s for s in ST[1:]]
    F2 = F1 + ['out_min', 'q_min', 'n_out', 'resp0', 'has_resp']
    trn = ((D.season <= 2024) & (D.rstate == 'valid')).values
    for k, f in (('L1', F1), ('L2', F2)):
        m = HistGradientBoostingRegressor(**HGB).fit(D.loc[trn, f].values, D.y.values[trn])
        P[k] = project_team(D, np.clip(m.predict(D[f].values), 0, None), np.zeros(len(D), bool))
    # 4. ratings (identical for all arms) and eligible games
    H = json.loads((ROOT / 'results/benchmarks/darko_history.json').read_text())
    DK = {int(p): (np.array(h['date'], dtype='datetime64[D]'), np.array([np.nan if x is None else x for x in h['dpm']], float))
          for p, h in H.items() if h and h.get('date')}
    rating = np.full(len(D), -1.0)
    for i, (p, d) in enumerate(zip(D.pid.values, D.d.values.astype('datetime64[D]'))):
        h = DK.get(int(p))
        if h is None:
            continue
        k = np.searchsorted(h[0], d) - 1
        while k >= 0 and np.isnan(h[1][k]):
            k -= 1
        if k >= 0:
            rating[i] = h[1][k]
    home = {g: mu.split(' vs. ')[0] for g, mu in zip(gl.game_id, gl.matchup) if ' vs. ' in mu}
    abbr = dict(zip(gl.team_id, gl.team_abbreviation))
    pts = gl.groupby(['game_id', 'team_id']).pts.sum()
    tg = D.groupby(['gid', 'team']).agg(rstate=('rstate', 'first'), season=('season', 'first')).reset_index()
    games = []
    for gid, g in tg.groupby('gid'):
        if len(g) != 2 or gid not in home:
            continue
        h = g[g.team.map(abbr) == home[gid]]
        if len(h) != 1:
            continue
        ht, at = int(h.team.iloc[0]), int(g[g.team != h.team.iloc[0]].team.iloc[0])
        games.append(dict(gid=gid, h=ht, a=at, season=int(h.season.iloc[0]), eligible=bool((g.rstate == 'valid').all()),
                          margin=float(pts[(gid, ht)] - pts[(gid, at)])))
    G = pd.DataFrame(games)
    E = G[G.eligible & (G.season >= 2025)].reset_index(drop=True)
    yw = (E.margin > 0).values.astype(int); va = (E.season == 2025).values; te = (E.season >= 2026).values
    res = dict(protocol='AVAIL_BENCH.md', smoke=smoke, evidence='retrospective development evidence, not prospective validation',
               reproduction_gate=gate)
    vv = [v for v in V.values() if v['valid']]
    res['coverage'] = dict(dates=len(V), dates_valid=len(vv), dates_refetched=sum(v.get('refetched', False) for v in V.values()),
                           min_minutes_asof_before_first_tip=min(v['minutes_before_first_tip'] for v in vv),
                           team_games_by_state={str(s): {k: int(c) for k, c in g.rstate.value_counts().items()}
                                                for s, g in tg[tg.season >= 2023].groupby('season')},
                           games=dict(validation=int((G.season == 2025).sum()), validation_eligible=int(va.sum()),
                                      test=int((G.season >= 2026).sum()), test_eligible=int(te.sum())),
                           listed_rows_matched=int(sum(len(u) for u in used.values())), matched_by_first_name_fallback=n_alias,
                           listed_rows_unmatched_by_status=dict(unmatched))
    tab = {}
    for name, msk in (('train_2023_2024', (D.season <= 2024).values), ('test_2026_2027', (D.season >= 2026).values)):
        t_ = {}
        for s in ST:
            mk = msk & (D.status == s).values & (D.rstate == 'valid').values & D.rot.values
            if mk.sum() >= 20:
                pl = D.played.values[mk]
                t_[s] = dict(n=int(mk.sum()), p_play=round(float(pl.mean()), 3),
                             minutes_vs_recent_if_played=round(float(np.mean(D.y.values[mk][pl] / np.maximum(D.base.values[mk][pl], 1))), 3),
                             fixed_share=FIXED[s])
        tab[name] = t_
    res['status_table_rotation_players'] = tab
    mk = D.gid.isin(set(E.gid[te])).values
    res['minutes_mae_test'] = {k: round(float(np.mean(np.abs(shares(D, v)[mk] * 48 - D.y.values[mk]))), 3) for k, v in P.items()}
    LL, TF = {}, {}
    for k, v in P.items():
        s = pd.Series(shares(D, v) * rating).groupby([D.gid.values, D.team.values]).sum()
        x = np.array([s[(g, h)] - s[(g, a)] for g, h, a in zip(E.gid, E.h, E.a)])
        lr = LogisticRegression(C=1e6).fit(x[va, None], yw[va]); p = np.clip(lr.predict_proba(x[:, None])[:, 1], 1e-6, 1 - 1e-6)
        LL[k] = -(yw * np.log(p) + (1 - yw) * np.log(1 - p))
        li = LinearRegression().fit(x[va, None], E.margin.values[va])
        TF[k] = dict(log_loss=round(float(LL[k][te].mean()), 5),
                     margin_rmse=round(float(np.sqrt(np.mean((li.predict(x[te, None]) - E.margin.values[te]) ** 2))), 3),
                     slope=round(float(lr.coef_[0, 0]), 4))
    rng = np.random.default_rng(0); n = int(te.sum()); idx = [rng.integers(0, n, n) for _ in range(2000)]
    for a_, b_, role in (('B2', 'P0', 'primary'), ('P0', 'P_none', 'secondary'), ('L1', 'P0', 'secondary'), ('L2', 'B2', 'secondary')):
        d_ = (LL[a_] - LL[b_])[te]
        ci = np.percentile([d_[i].mean() for i in idx], [2.5, 97.5])
        TF[f'{a_}_minus_{b_}'] = dict(role=role, mean=round(float(d_.mean()), 5), ci=ci.round(5).tolist(), improves=bool(ci[1] < 0))
    res['team_forecast_test'] = TF
    res['provenance'] = provenance(__file__, ['results/latent/gamelogs.parquet', 'results/injury_hist/verified.json',
                                              'results/benchmarks/darko_history.json', 'results/latent/ms2_stints.pkl'])
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_dir / 'avail_bench.json', 'w'), indent=1)
    print('wrote', out_dir / 'avail_bench.json', '(smoke: code check only, not for reporting)' if smoke else '')
    if not smoke:
        print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
