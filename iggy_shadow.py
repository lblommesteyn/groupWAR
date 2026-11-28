"""Shadow IGGY minutes forecast (protocol: court-live HOLDOUT.md amendment + IGGY_SHADOW.md). Never changes DUNCAN-2.

After the PC's pm forecast (forecasts/{date}-pm.json) exists, re-forecast the same games with the SAME ratings
(that run's live_ratings_{date}.npz and DARKO table), the SAME injury report and the SAME frozen calibration; only the
playing time changes: frozen IGGY B2 minutes instead of DARKO projected minutes x status share.
  minutes: B2 (results/iggy_shadow/b2_frozen.pkl, trained exactly as iggy_minutes / avail_bench: seasons <= 2024,
           realised absences, same hyperparameters and seed), features built by the frozen iggy_minutes.build code
           for a game today, absent set = rotation players listed Out on the report (matched as avail_bench.match),
           projected to 240 team minutes, share = minutes / 48 (= 5 x minutes / team total, as live shares()).
  teams whose report state is not valid (NOT YET SUBMITTED or absent from the report) get an empty absent set and
  are flagged 'report_state'.
Writes {LIVE}/forecasts/shadow/{date}-pm-iggy.json (model_version iggy-b2-shadow-1; never overwritten).
  python iggy_shadow.py train                     (PC, once; needs the avail_bench reproduction gate passed)
  python iggy_shadow.py [--date D] [--live DIR]   (PC daily, after live_forecast.py --run pm)
"""
import argparse, gzip, hashlib, json, pickle
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import iggy_minutes as ig
import injuries as inj
from avail_bench import HGB, match, b2_pregame, project_team
from calibrate_live import our_margin
from fetch_injury_history import SUF, parse_text, pdf_text
from masked_paths import ROOT

MODEL_VERSION = 'iggy-b2-shadow-1'
OUT = ROOT / 'results/iggy_shadow'
MODEL = OUT / 'b2_frozen.pkl'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def train():
    gate = json.loads((ROOT / 'results/avail/avail_bench.json').read_text()).get('reproduction_gate', {})
    if not gate.get('passed'):
        raise SystemExit('avail_bench reproduction gate has not passed: B2 is not frozen')
    D0, gl, pg = ig.build(); D0 = ig.add_features(D0, pg)
    tr = (D0.season <= 2024) & ~D0.absent
    b2 = HistGradientBoostingRegressor(**HGB).fit(D0.loc[tr, ig.FEATS['B2']].values, D0.loc[tr, 'y'].values)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pickle.dumps(b2); MODEL.write_bytes(raw)
    json.dump(dict(sha256=sha(raw), rows=int(tr.sum()), feats=ig.FEATS['B2'], hgb=HGB, trained_utc=datetime.now(timezone.utc).isoformat(timespec='seconds')),
              open(OUT / 'b2_frozen.json', 'w'), indent=1)
    print('saved', MODEL, sha(raw))


def darko_rows(path):
    p = json.loads(gzip.open(path).read())
    d = [n for n in p['nodes'] if n and n.get('type') == 'data'][-1]['data']

    def dec(i):
        v = d[i]
        if isinstance(v, dict):
            return {k: dec(j) for k, j in v.items()}
        if isinstance(v, list):
            return [dec(j) for j in v]
        return v
    r = dec(0); cols = dict(zip(r['players']['keys'], r['players']['values']))
    return [dict(zip(cols, v)) for v in zip(*cols.values())]


def today_rows(day, games):
    """Frozen iggy_minutes.build rows for today's games: one placeholder game per team today (a dummy player, so the
    frozen code builds candidate rows from the previous 10 / 30-day windows; the dummy is never a candidate)."""
    gl = pd.read_parquet(ig.LAT / 'gamelogs.parquet')
    gl = gl[pd.to_datetime(gl.game_date) < pd.Timestamp(day)]
    dummy = []
    for g in games:
        for t in (g['home_id'], g['away_id']):
            dummy.append(dict(game_id=g['gid'], team_id=t, player_id=-1, min=1.0, game_date=day, playoff=False,
                              **{c: 0 for c in ig.ROLE}))
    gl2 = pd.concat([gl, pd.DataFrame(dummy)], ignore_index=True)
    orig = pd.read_parquet; ig.BENCHMARK_CUTOFF = str((pd.Timestamp(day) + pd.Timedelta(days=1)).date())
    pd.read_parquet = lambda *a, **k: gl2
    try:
        D, gl_, pg = ig.build()
    finally:
        pd.read_parquet = orig
    D = D[D.pid != -1]
    keep_from = pd.Timestamp(day) - pd.Timedelta(days=400)        # responses look back 365 days
    return D[D.d >= keep_from].sort_values(['d', 'gid', 'team']).reset_index(drop=True), gl


def report_status(D, day, pdf, gl):
    rows, nys, _, _, teams = parse_text(pdf_text(pdf), day)
    state = {int(t): ('not_submitted' if t in nys else 'valid') for t in teams}
    names = {int(p): inj.key(n) for p, n in zip(gl.player_id, gl.player_name)}
    parts = {int(p): (inj.key(n.split(' ', 1)[0]), SUF.sub('', inj.key(n.split(' ', 1)[1])) if ' ' in n else '')
             for p, n in zip(gl.player_id, gl.player_name)}
    today = D.d == pd.Timestamp(day)
    st = ['none'] * len(D); rs = {}
    for i in np.flatnonzero(today.values):
        t = int(D.team.iat[i]); rs[t] = state.get(t, 'absent_from_report')
        if rs[t] == 'valid':
            st[i] = match(rows, t, names.get(int(D.pid.iat[i]), ''), *parts.get(int(D.pid.iat[i]), (None, None)))[1]
    D['status'] = st
    return D, rs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', nargs='?')
    ap.add_argument('--date', default=str(date.today())); ap.add_argument('--live', default=r'C:\Users\LukeB\dev\court-live')
    ap.add_argument('--out', help='write here instead of {live}/forecasts/shadow (smoke tests)')
    ap.add_argument('--ratings', help='ratings npz (default results/latent/live_ratings_{date}.npz)')
    ap.add_argument('--model', default=str(MODEL))
    ap.add_argument('--smoke', action='store_true', help='laptop code check: start rate 0, model trained in place, no ratings check; never used')
    a = ap.parse_args()
    if a.smoke:
        ig.starters = lambda: set()
    if a.cmd == 'train':
        return train()
    live, day = Path(a.live), a.date
    out = Path(a.out) if a.out else live / 'forecasts' / 'shadow' / f'{day}-pm-iggy.json'
    if out.exists():
        print('shadow forecast exists; preserving', out); return
    base_f = live / 'forecasts' / f'{day}-pm.json'
    if not base_f.exists():
        print('no pm forecast for', day, '- shadow not run'); return
    braw = base_f.read_bytes(); base = json.loads(braw)
    if a.smoke:
        D0, _, pg0 = ig.build(); D0 = ig.add_features(D0, pg0); tr = (D0.season <= 2024) & ~D0.absent
        mraw = pickle.dumps(HistGradientBoostingRegressor(**HGB).fit(D0.loc[tr, ig.FEATS['B2']].values, D0.loc[tr, 'y'].values))
    else:
        mraw = Path(a.model).read_bytes()
    b2 = pickle.loads(mraw)
    pdf_name = base['injury_report'].rsplit('/', 1)[1]; pdf = (live / 'data' / 'injury' / pdf_name).read_bytes()
    z = np.load(a.ratings or ROOT / f'results/latent/live_ratings_{day}.npz', allow_pickle=True)
    pix = {int(p): i for i, p in enumerate(z['pids'])}; th, b = z['theta'], z['b']
    h = sha(th.tobytes() + np.asarray(b).tobytes())[:12]
    if h != base['ratings_hash'] and not a.smoke:
        raise SystemExit(f'ratings hash {h} differs from the pm forecast {base["ratings_hash"]}')
    dkx = {int(r['nba_id']): r for r in darko_rows(live / 'data' / 'darko' / f'{day}.json.gz')}
    cal = base['calibration']
    prob = lambda k, x: float(1 / (1 + np.exp(-(cal[k]['intercept'] + cal[k]['slope'] * x))))
    v = lambda x: -1.0 if x is None else float(x)
    dpm = lambda p: v(dkx[p]['o_dpm']) + v(dkx[p]['d_dpm']) if p in dkx else -1.0

    games = [g for g in base['games'] if 'ours_inj_p' in g]
    D, gl = today_rows(day, games)
    D, rstate = report_status(D, day, pdf, gl)
    Db = b2_pregame(D)
    td = (Db.d == pd.Timestamp(day)).values
    Dt = Db[td].reset_index(drop=True)
    pred = b2.predict(Dt[ig.FEATS['B2']].values)
    M = {'iggy': project_team(Dt, pred, Dt.absent.values),                                  # frozen B2 (primary)
         'iggy0': project_team(Dt, pred, (Dt.absent | (Dt.status == 'out')).values)}      # + every listed-Out player 0 (variant)
    res = []
    for g in games:
        r = dict(gid=g['gid'], home_id=g['home_id'], away_id=g['away_id'], code=g.get('code'),
                 report_state=dict(home=rstate.get(g['home_id']), away=rstate.get(g['away_id'])),
                 rotation_out=[int(p) for p in Dt.pid[(Dt.gid == g['gid']) & Dt.absent]],
                 listed_out=[int(p) for p in Dt.pid[(Dt.gid == g['gid']) & (Dt.status == 'out')]])
        for arm, m in M.items():
            side = {}
            for k, t in (('home', g['home_id']), ('away', g['away_id'])):
                s = Dt[(Dt.gid == g['gid']) & (Dt.team == t)]
                side[k] = [(int(p), float(x) / 48.0) for p, x in zip(s.pid, m[s.index.values]) if x > 0]
            if not side['home'] or not side['away']:
                r['skipped'] = 'no IGGY candidates (first 10 games of the season or no history)'; break
            T = (side['home'], side['away'])
            mo = our_margin(pix, th, b, T); md = sum(w * dpm(p) for p, w in T[0]) - sum(w * dpm(p) for p, w in T[1])
            r.update({f'ours_{arm}_margin': round(float(mo), 3), f'darko_{arm}_margin': round(float(md), 3),
                      f'ours_{arm}_p': round(prob('ours', mo), 4), f'darko_{arm}_p': round(prob('darko', md), 4),
                      f'home_minutes_{arm}': [(p, round(w, 4)) for p, w in side['home']],
                      f'away_minutes_{arm}': [(p, round(w, 4)) for p, w in side['away']]})
        res.append(r)
    doc = dict(date=day, model_version=MODEL_VERSION, generated_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               base_forecast=base_f.name, base_forecast_sha256=sha(braw), b2_sha256=sha(mraw), ratings_hash=h,
               injury_report=base['injury_report'], injury_report_sha256=sha(pdf), calibration=cal, games=res)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(doc, f, indent=1)
    print('wrote', out, len([r for r in res if 'ours_iggy_p' in r]), 'games')


if __name__ == '__main__':
    main()
