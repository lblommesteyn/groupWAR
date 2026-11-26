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


def shares(rows, team):
    t = [(int(r['nba_id']), float(r['x_minutes'] or 0)) for r in rows if r['tm_id'] == team and (r['x_minutes'] or 0) > 0]
    tot = sum(m for _, m in t)
    return [(p, 5 * m / tot) for p, m in t] if tot > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=str(date.today()))
    ap.add_argument('--live', default=r'C:\Users\LukeB\dev\court-live')
    a = ap.parse_args()
    live = Path(a.live); day = a.date
    cal = json.load(open(OUT / 'live_calib.json'))
    prob = lambda k, x: float(1 / (1 + np.exp(-(cal[k]['intercept'] + cal[k]['slope'] * x))))

    games = schedule(day)
    if not games:
        print(day, 'no games'); return
    dk, dk_through = darko_table(live / 'data' / 'darko' / f'{day}.json.gz')
    dkx = {int(r['nba_id']): r for r in dk}

    # our ratings: lambda frozen at the most recent backtest block's choice
    last = sorted(OUT.glob('msb_*.npz'), key=lambda f: f.stem.split('_')[2])[-1]
    lam = float(np.load(last)[ARM + '_lam'])
    Dt = sp.Data(live=True); cut = sp.dnum(day)
    th, b = sp.fit_arm(Dt, ARM, cut, lam)
    through = max(Dt.G[g]['date'] for g in Dt.gids if sp.dnum(Dt.G[g]['date']) < cut)
    np.savez(OUT / f'live_ratings_{day}.npz', pids=np.array(Dt.pids), theta=th, b=b)
    h = hashlib.sha256(th.tobytes() + np.asarray(b).tobytes()).hexdigest()[:12]

    out = []
    for g in games:
        H, A = shares(dk, g['home_id']), shares(dk, g['away_id'])
        if not H or not A:
            out.append({**g, 'skipped': 'no projected minutes'}); continue
        mo = our_margin(Dt.pix, th, b, (H, A))
        v = lambda x: -1.0 if x is None else float(x)
        net = lambda side: sum(w * (v(dkx[p]['o_dpm']) + v(dkx[p]['d_dpm'])) for p, w in side)
        md = net(H) - net(A)
        po, pd_ = prob('ours', mo), prob('darko', md)
        out.append({**g, 'ours_margin': round(float(mo), 3), 'darko_margin': round(float(md), 3),
                    'ours_p': round(po, 4), 'darko_p': round(pd_, 4), 'blend_p': round((po + pd_) / 2, 4),
                    'home_minutes': H, 'away_minutes': A,
                    'unrated_by_us': [p for p, _ in H + A if p not in Dt.pix]})
    doc = dict(date=day, generated_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), model=ARM,
               lam=lam, ratings_through=through, ratings_hash=h, darko_ratings_through=dk_through,
               calibration=cal, games=out)
    f = live / 'forecasts' / f'{day}.json'; f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(f, 'w'), indent=1)
    for g in out:
        if 'ours_p' in g:
            print(f"{g['code']:>18} ours {g['ours_p']:.3f} ({g['ours_margin']:+.1f})  darko {g['darko_p']:.3f} ({g['darko_margin']:+.1f})")
    print('wrote', f, 'ratings through', through, 'darko through', dk_through)


if __name__ == '__main__':
    main()
