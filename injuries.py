"""Official NBA injury report -> per-player availability for the live forecasts.

The league posts PDFs at ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{HH}_{MM}{AM|PM}.pdf
(a missing time returns 403). latest() walks back from now in 15-minute steps (at most MAX_PROBES requests, one per
second) to the newest report of the day, saves the PDF next to the forecasts, and parses it.
Text extraction drops spaces ("MiamiHeat", "Antetokounmpo,Giannis"), so players are matched to DARKO's table on a
letters-only key (first + last) and team. Status -> expected share of normal minutes: AVAIL below.
  python injuries.py [YYYY-MM-DD]     prints the parsed report (PC or Mac)
"""
import re, sys, time, unicodedata, urllib.error, urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
URL = 'https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{t}.pdf'
AVAIL = {'out': 0.0, 'doubtful': 0.15, 'questionable': 0.6, 'probable': 0.95, 'available': 1.0}
MAX_PROBES = 40
ROW = re.compile(r"([A-Za-z.'\-]+(?:[A-Za-z.'\-]*)?,[A-Za-z.'\-]+)\s+(Available|Out|Doubtful|Questionable|Probable)\b")


def key(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'(?<=[a-z])\.?\s*(jr|sr|ii|iii|iv)\.?(?=$|[\s,])', '', s)  # suffixes, with or without a space
    return re.sub(r'[^a-z]', '', s)


def stamp(dt):
    h = dt.strftime('%I'); return f"{h}_{dt.strftime('%M')}{dt.strftime('%p')}"


def latest(day, save_dir=None, now=None):
    """Newest report for `day` (YYYY-MM-DD) published by `now` (ET). Returns (url, rows) or (None, [])."""
    now = now or datetime.now(ET)
    now = min(now, datetime.fromisoformat(day + 'T23:45').replace(tzinfo=ET))  # past day: start from its last slot
    t = now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0)
    for _ in range(MAX_PROBES):
        u = URL.format(d=day, t=stamp(t))
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'court-live-research/1.0'})
            body = urllib.request.urlopen(req, timeout=30).read()
            if body[:4] == b'%PDF':
                if save_dir:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    (Path(save_dir) / u.rsplit('/', 1)[1]).write_bytes(body)
                return u, parse(body)
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):
                raise
        t -= timedelta(minutes=15)
        if t.date().isoformat() < day:
            break
        time.sleep(1.0)
    return None, []


def parse(pdf_bytes):
    import io
    import pdfplumber
    rows, team = [], None
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
    teams = TEAMS
    for line in text.splitlines():
        for tk in re.findall(r'[A-Z][A-Za-z0-9]+', line):
            if key(tk) in teams:
                team = teams[key(tk)]
        for name, status in ROW.findall(line):
            last, first = name.split(',', 1)
            rows.append(dict(name=f'{first} {last}', key=key(first + ' ' + last), team_id=team, status=status.lower()))
    return rows


def match(rows, darko_rows):
    """{nba_id: (status, availability)} for report rows matched to DARKO players (name key, then team check)."""
    by = {}
    for r in darko_rows:
        by.setdefault(key(r['player_name']), []).append(r)
    out, miss = {}, []
    for r in rows:
        c = by.get(r['key'], [])
        if len(c) > 1 and r['team_id']:
            c = [x for x in c if x['tm_id'] == r['team_id']] or c
        if c:
            out[int(c[0]['nba_id'])] = (r['status'], AVAIL[r['status']])
        else:
            miss.append(r['name'])
    return out, miss


TEAMS = {key(n): i for n, i in {
    'Atlanta Hawks': 1610612737, 'Boston Celtics': 1610612738, 'Brooklyn Nets': 1610612751, 'Charlotte Hornets': 1610612766,
    'Chicago Bulls': 1610612741, 'Cleveland Cavaliers': 1610612739, 'Dallas Mavericks': 1610612742, 'Denver Nuggets': 1610612743,
    'Detroit Pistons': 1610612765, 'Golden State Warriors': 1610612744, 'Houston Rockets': 1610612745, 'Indiana Pacers': 1610612754,
    'LA Clippers': 1610612746, 'Los Angeles Lakers': 1610612747, 'Memphis Grizzlies': 1610612763, 'Miami Heat': 1610612748,
    'Milwaukee Bucks': 1610612749, 'Minnesota Timberwolves': 1610612750, 'New Orleans Pelicans': 1610612740,
    'New York Knicks': 1610612752, 'Oklahoma City Thunder': 1610612760, 'Orlando Magic': 1610612753,
    'Philadelphia 76ers': 1610612755, 'Phoenix Suns': 1610612756, 'Portland Trail Blazers': 1610612757,
    'Sacramento Kings': 1610612758, 'San Antonio Spurs': 1610612759, 'Toronto Raptors': 1610612761, 'Utah Jazz': 1610612762,
    'Washington Wizards': 1610612764}.items()}


if __name__ == '__main__':
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.now(ET).date().isoformat()
    u, rows = latest(day, now=datetime.now(ET) if len(sys.argv) < 3 else datetime.fromisoformat(sys.argv[2]).replace(tzinfo=ET))
    print(u, len(rows), 'rows')
    from collections import Counter
    print(Counter(r['status'] for r in rows), sum(r['team_id'] is None for r in rows), 'without team')
    for r in rows[:12]:
        print(r)
