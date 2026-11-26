"""Daily pregame forecasts for the live head-to-head: our frozen model (spm_lpit_h) against DARKO.

Both models get the same playing time: DARKO's own projected minutes (x_minutes) and team assignments from its
daily ratings table, normalised to 5 players on court. So the comparison is ratings against ratings.
  ours  = home edge + sum(share * (our offense)) of home + away's defense terms ... (calibrate_live.our_margin)
  darko = sum(share * (o_dpm + d_dpm)) home - away
Win probability: the logistic maps frozen by calibrate_live.py; blend = mean of the two probabilities.
Ratings: spm_lpit_h refit every day on all games before today, lambda fixed at the last backtest block's choice.

Writes {LIVE}/forecasts/{date}.json and {LIVE}/data/darko/{date}.json.gz (the raw DARKO table, for audit).
  python live_forecast.py [--date YYYY-MM-DD] [--live C:\\Users\\LukeB\\dev\\court-live]   (PC)
"""
import argparse, gzip, hashlib, json, time, urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np

import injuries as inj
import ms_spm as sp
from calibrate_live import our_margin
from masked_paths import ROOT

OUT = ROOT / 'results/latent'
ARM = 'spm_lpit_h'


def darko_table(save_to):
    req = urllib.request.Request('https://darko.app/__data.json', headers={'User-Agent': 'Mozilla/5.0 (research script)'})
    raw = urllib.request.urlopen(req, timeout=60).read()
    save_to.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(save_to, 'wb') as f:
        f.write(raw)
    p = json.loads(raw)
    d = [n for n in p['nodes'] if n and n.get('type') == 'data'][-1]['data']

    def dec(i):
        v = d[i]
        if isinstance(v, dict):
            return {k: dec(j) for k, j in v.items()}
        if isinstance(v, list):
            return [dec(j) for j in v]
        return v
    r = dec(0)
    cols = dict(zip(r['players']['keys'], r['players']['values']))
    rows = [dict(zip(cols, v)) for v in zip(*cols.values())]
    return rows, r.get('ratingsThrough')


def schedule(day):
    from nba_api.stats.endpoints import scoreboardv2
    gh = scoreboardv2.ScoreboardV2(game_date=day, league_id='00', timeout=60).get_data_frames()[0]
    out = []
    for _, g in gh.drop_duplicates('GAME_ID').iterrows():
        out.append(dict(gid=str(g.GAME_ID), home_id=int(g.HOME_TEAM_ID), away_id=int(g.VISITOR_TEAM_ID),
                        status=str(g.GAME_STATUS_TEXT).strip(), code=str(g.GAMECODE)))
    return out


def shares(rows, team, avail=None):
    a = lambda p: avail.get(p, (None, 1.0))[1] if avail else 1.0
    t = [(int(r['nba_id']), float(r['x_minutes'] or 0) * a(int(r['nba_id']))) for r in rows if r['tm_id'] == team and (r['x_minutes'] or 0) > 0]
    t = [(p, m) for p, m in t if m > 0]
    tot = sum(m for _, m in t)
    return [(p, 5 * m / tot) for p, m in t] if tot > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=str(date.today()))
    ap.add_argument('--live', default=r'C:\Users\LukeB\dev\court-live')
    ap.add_argument('--run', default='am', choices=['am', 'pm'], help='pm: second run before evening tips, latest injury report')
    a = ap.parse_args()
    live = Path(a.live); day = a.date
    cal = json.load(open(OUT / 'live_calib.json'))
    prob = lambda k, x: float(1 / (1 + np.exp(-(cal[k]['intercept'] + cal[k]['slope'] * x))))

    games = schedule(day)
    if not games:
        print(day, 'no games'); return
    dk, dk_through = darko_table(live / 'data' / 'darko' / f'{day}.json.gz')
    dkx = {int(r['nba_id']): r for r in dk}
    rep_url, rep = inj.latest(day, save_dir=live / 'data' / 'injury')
    avail, missed = inj.match(rep, dk)
    print('injury report', rep_url, len(rep), 'rows', len(avail), 'matched', 'unmatched:', missed, flush=True)

    # our ratings: lambda frozen at the most recent backtest block's choice
    last = sorted(OUT.glob('msb_*.npz'), key=lambda f: f.stem.split('_')[2])[-1]
    lam = float(np.load(last)[ARM + '_lam'])
    rf = OUT / f'live_ratings_{day}.npz'
    if a.run == 'pm' and rf.exists():  # same day: reuse the morning's ratings
        z = np.load(rf, allow_pickle=True)
        pix = {int(p): i for i, p in enumerate(z['pids'])}; th, b = z['theta'], z['b']; through = str(z['through'])
    else:
        Dt = sp.Data(live=True); cut = sp.dnum(day)
        th, b = sp.fit_arm(Dt, ARM, cut, lam); pix = Dt.pix
        through = max(Dt.G[g]['date'] for g in Dt.gids if sp.dnum(Dt.G[g]['date']) < cut)
        np.savez(rf, pids=np.array(Dt.pids), theta=th, b=b, through=through)
    h = hashlib.sha256(th.tobytes() + np.asarray(b).tobytes()).hexdigest()[:12]

    v = lambda x: -1.0 if x is None else float(x)
    net = lambda side: sum(w * (v(dkx[p]['o_dpm']) + v(dkx[p]['d_dpm'])) for p, w in side)
    out = []
    for g in games:
        H, A = shares(dk, g['home_id']), shares(dk, g['away_id'])
        if not H or not A:
            out.append({**g, 'skipped': 'no projected minutes'}); continue
        Hi, Ai = shares(dk, g['home_id'], avail) or H, shares(dk, g['away_id'], avail) or A
        mo, md = our_margin(pix, th, b, (H, A)), net(H) - net(A)
        mi, mdi = our_margin(pix, th, b, (Hi, Ai)), net(Hi) - net(Ai)
        po, pd_, poi, pdi = prob('ours', mo), prob('darko', md), prob('ours', mi), prob('darko', mdi)
        adj = sorted({p for p, _ in H + A if p in avail and avail[p][1] < 1}, key=lambda p: avail[p][1])
        out.append({**g, 'ours_margin': round(float(mo), 3), 'darko_margin': round(float(md), 3),
                    'ours_p': round(po, 4), 'darko_p': round(pd_, 4), 'blend_p': round((po + pd_) / 2, 4),
                    'ours_inj_margin': round(float(mi), 3), 'darko_inj_margin': round(float(mdi), 3),
                    'ours_inj_p': round(poi, 4), 'darko_inj_p': round(pdi, 4), 'blend_inj_p': round((poi + pdi) / 2, 4),
                    'injuries': [[p, dkx[p]['player_name'], avail[p][0]] for p in adj],
                    'home_minutes': H, 'away_minutes': A, 'home_minutes_inj': Hi, 'away_minutes_inj': Ai,
                    'unrated_by_us': [p for p, _ in H + A if p not in pix]})
    doc = dict(date=day, generated_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), model=ARM,
               lam=lam, ratings_through=through, ratings_hash=h, darko_ratings_through=dk_through,
               calibration=cal, run=a.run, injury_report=rep_url, games=out)
    f = live / 'forecasts' / (f'{day}.json' if a.run == 'am' else f'{day}-pm.json'); f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(f, 'w'), indent=1)
    for g in out:
        if 'ours_p' in g:
            print(f"{g['code']:>18} ours {g['ours_p']:.3f} -> {g['ours_inj_p']:.3f}  darko {g['darko_p']:.3f} -> {g['darko_inj_p']:.3f}  out/limited: {', '.join(n for _, n, _ in g['injuries'][:4])}")
    print('wrote', f, 'ratings through', through, 'darko through', dk_through)


if __name__ == '__main__':
    main()
