"""Archived official NBA injury reports, one per game date, as they stood BEFORE every tip that day.

For each regular-season date (2022-23 .. 2026-27 to date) the target time is min(5:30 PM ET, earliest tip - 45 min),
close to the live pm run (5:45 PM) while never after a tip. Walks back from the target over both URL formats the league
has used (hourly `_05PM` through 2024-25, 15-minute `_05_30PM` from 2025-26), one request per second, and keeps the
newest report found. Download-time rows (injuries.parse) are not used: the space-less live parser misses the older spaced layout.
File names are not publication times: an hourly file `05PM` holds the report AS OF 5:30 PM (header
"Injury Report: MM/DD/YY 05:30 PM"). So every report is validated by its header as-of time: valid only if the header
parses and as-of <= target (<= first tip - 45 min). `--verify` re-reads every saved PDF, walks further back when a
report is too late, and records coverage: status tokens in the text, parsed rows, rows without a team, and teams
marked NOT YET SUBMITTED (their players are unknown, not unlisted).
Output: results/injury_hist/reports.json {date: {url, target, first_tip, rows}}; results/injury_hist/verified.json
  python fetch_injury_history.py            download
  python fetch_injury_history.py --verify   validate as-of times (re-fetching if needed) and record coverage
"""
import json, re, sys, time, urllib.error, urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from nba_api.stats.endpoints import scheduleleaguev2

import injuries as inj
from masked_paths import ROOT

ET = ZoneInfo('America/New_York')
OUT = ROOT / 'results/injury_hist'
CUTOFF = '2026-11-27'   # prospective games stay out of development data


def tips():
    first = {}
    for y in range(2023, 2028):
        df = scheduleleaguev2.ScheduleLeagueV2(season=f'{y - 1}-{str(y)[2:]}', league_id='00').get_data_frames()[0]
        df = df[df.gameId.str.startswith('002')]
        for t in df.gameDateTimeUTC:
            et = datetime.fromisoformat(t.replace('Z', '+00:00')).astimezone(ET)
            d = et.date().isoformat()
            if d < CUTOFF and (d not in first or et < first[d]):
                first[d] = et
        time.sleep(2)
    return first


def urls(t):
    h12 = t.strftime('%I'); ap = t.strftime('%p')
    out = [inj.URL.format(d=t.date().isoformat(), t=f'{h12}_{t.strftime("%M")}{ap}')]
    if t.minute == 0:
        out.append(inj.URL.format(d=t.date().isoformat(), t=f'{h12}{ap}'))
    return out


def get(u):
    try:
        b = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'court-live-research/1.0'}), timeout=30).read()
        return b if b[:4] == b'%PDF' else None
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise


STATUS = re.compile(r'\b(Available|Out|Doubtful|Questionable|Probable)\b')
SUF = re.compile(r'(jr|sr|ii|iii|iv)$')
HEAD = re.compile(r'Injury Report:\s*(\d\d)/(\d\d)/(\d\d)\s+(\d\d):(\d\d)\s*([AP]M)')


def pdf_text(path_or_bytes):
    import io, pdfplumber
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with pdfplumber.open(src) as pdf:
        return '\n'.join(p.extract_text() or '' for p in pdf.pages)


def asof(text):
    """Header as-of time (ET) or None."""
    m = HEAD.search(text)
    if not m:
        return None
    mo, dd, yy, hh, mi, ap = m.groups(); h = int(hh) % 12 + (12 if ap == 'PM' else 0)
    return datetime(2000 + int(yy), int(mo), int(dd), h, int(mi), tzinfo=ET)


GDATE = re.compile(r'\b(\d\d)/(\d\d)/(\d{4})\b')


def parse_text(text, day):
    """Both layouts (spaced through 2024-25, space-less from 2025-26). Reports also list later dates' games, so only
    rows under game date `day` are kept. Returns rows (team_id, key before the comma, key after it, status), teams
    marked NOT YET SUBMITTED for `day`, the count of status words under `day`, and the dates seen."""
    rows, team, nys, n_tok, cur, seen, teams = [], None, set(), 0, None, set(), set()
    for line in text.splitlines():
        g = GDATE.search(line)
        if g:
            cur = f'{g.group(3)}-{g.group(1)}-{g.group(2)}'; seen.add(cur)
        if cur != day:
            continue
        lk = inj.key(line); best = -1
        for tk, tid in inj.TEAMS.items():
            i = lk.rfind(tk)
            if i > best:
                best, team_ = i, tid
        if best >= 0:
            team = team_; teams.add(team_)
        if 'NOT YET SUBMITTED' in line.upper().replace('  ', ' ') or 'notyetsubmitted' in lk:
            if best >= 0:
                nys.add(team_)
        start = 0
        for m in STATUS.finditer(line):
            n_tok += 1
            seg = line[start:m.start()]; start = m.end()
            if ',' not in seg:
                continue
            before, first = seg.rsplit(',', 1)
            kb = SUF.sub('', re.sub(r'[^a-z]', '', inj.key(before)))
            rows.append((team, kb, inj.key(first), m.group(1).lower()))
    return rows, sorted(nys), n_tok, sorted(seen), sorted(teams)


def walk(target, floor=None):
    """Newest report whose HEADER as-of time is <= target. Returns (url, bytes, asof) or None."""
    t = target.replace(minute=target.minute - target.minute % 15, second=0, microsecond=0)
    for _ in range(64):
        for u in urls(t):
            b = get(u); time.sleep(1.0)
            if b:
                a = asof(pdf_text(b))
                if a is not None and a <= target:
                    return u, b, a
        t -= timedelta(minutes=15)
        if t.hour < 6:
            break
    return None


def verify():
    f = OUT / 'reports.json'; res = json.loads(f.read_text()); V = {}
    for d, r in sorted(res.items()):
        target = datetime.fromisoformat(r['target'])
        u, b, a = r['url'], None, None
        if u:
            b = (OUT / 'pdf' / u.rsplit('/', 1)[1]).read_bytes(); a = asof(pdf_text(b))
        refetched = False
        if u is None or a is None or a > target:
            hit = walk(target); refetched = True
            if hit:
                u, b, a = hit; (OUT / 'pdf' / u.rsplit('/', 1)[1]).write_bytes(b)
            else:
                u, b, a = None, None, None
        if b is None:
            V[d] = dict(valid=False, reason='no report with as-of <= target', target=r['target'], first_tip=r['first_tip'])
        else:
            rows, nys, n_tok, seen, teams = parse_text(pdf_text(b), d)
            V[d] = dict(valid=True, url=u, asof=a.isoformat(), target=r['target'], first_tip=r['first_tip'],
                        minutes_before_first_tip=round((datetime.fromisoformat(r['first_tip']) - a).total_seconds() / 60),
                        refetched=refetched, status_tokens=n_tok, rows=len(rows),
                        rows_without_team=sum(t is None for t, *_ in rows), not_yet_submitted=nys, teams_in_report=teams, game_dates_in_report=seen,
                        parsed=[list(x) for x in rows])
        if refetched:
            print(d, 'refetched ->', V[d].get('asof'), flush=True)
    (OUT / 'verified.json').write_text(json.dumps(V, indent=0))
    ok = [v for v in V.values() if v['valid']]
    print(dict(dates=len(V), valid=len(ok), refetched=sum(v.get('refetched', False) for v in V.values()),
               min_lead_min=min(v['minutes_before_first_tip'] for v in ok), tokens=sum(v['status_tokens'] for v in ok),
               rows=sum(v['rows'] for v in ok), no_team=sum(v['rows_without_team'] for v in ok),
               dates_with_not_submitted=sum(bool(v['not_yet_submitted']) for v in ok)))


def main():
    (OUT / 'pdf').mkdir(parents=True, exist_ok=True)
    f = OUT / 'reports.json'
    res = json.loads(f.read_text()) if f.exists() else {}
    first = tips()
    print(len(first), 'game dates', flush=True)
    for d, tip in sorted(first.items()):
        if d in res:
            continue
        target = min(datetime.fromisoformat(d + 'T17:30').replace(tzinfo=ET), tip - timedelta(minutes=45))
        t = target.replace(minute=target.minute - target.minute % 15, second=0, microsecond=0)
        hit = None
        for _ in range(48):                      # up to 12 hours back
            for u in urls(t):
                b = get(u); time.sleep(1.0)
                if b:
                    hit = (u, b); break
            if hit or t.hour < 6:
                break
            t -= timedelta(minutes=15)
        if hit:
            u, b = hit
            (OUT / 'pdf' / u.rsplit('/', 1)[1]).write_bytes(b)
            rows = inj.parse(b)
        else:
            u, rows = None, []
        res[d] = dict(url=u, target=target.isoformat(), first_tip=tip.isoformat(), rows=rows)
        print(d, u and u.rsplit('_', 2)[-2:], len(rows), flush=True)
        if len(res) % 20 == 0:
            f.write_text(json.dumps(res))
    f.write_text(json.dumps(res))


if __name__ == '__main__':
    verify() if '--verify' in sys.argv else main()
