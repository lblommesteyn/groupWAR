"""Fetch point-in-time DARKO DPM history (per game) for every player in the archive, from darko.app player pages."""
import json, pickle, time, urllib.request
from masked_paths import ROOT

OUT = ROOT / 'results/benchmarks/darko_history.json'


def decode(payload):
    d = [n for n in payload['nodes'] if n and n.get('type') == 'data'][-1]['data']
    def dec(i):
        v = d[i]
        if isinstance(v, dict): return {k: dec(j) for k, j in v.items()}
        if isinstance(v, list): return [dec(j) for j in v]
        return v
    return dec(0)


def main():
    import sys
    if len(sys.argv) > 1:  # a file of player ids, one per line (e.g. every player in the multi-season stints)
        pids = sorted({int(l) for l in open(sys.argv[1]) if l.strip()})
    else:
        G = pickle.load(open(ROOT / 'results/embed/player_games.pkl', 'rb'))
        pids = sorted({p for d in G.values() for p in d['box']})
    res = json.load(open(OUT)) if OUT.exists() else {}
    for k, p in enumerate(pids):
        if str(p) in res:
            continue
        req = urllib.request.Request(f'https://darko.app/player/{p}/__data.json', headers={'User-Agent': 'Mozilla/5.0 (research script)'})
        try:
            r = decode(json.load(urllib.request.urlopen(req, timeout=30)))
            h = r.get('history') or {}
            keys, vals = h.get('keys', []), h.get('values', [])
            cols = {kk: vals[i] for i, kk in enumerate(keys)}
            keep = [c for c in ('date', 'season', 'future_game', 'seconds_played', 'dpm', 'o_dpm', 'd_dpm', 'box_dpm', 'bayes_rapm_total') if c in cols]
            res[str(p)] = {c: cols[c] for c in keep}
        except Exception as e:
            res[str(p)] = {'error': str(e)[:200]}
        if k % 25 == 0:
            json.dump(res, open(OUT, 'w')); print(k, len(pids), flush=True)
        time.sleep(1.0)
    json.dump(res, open(OUT, 'w'))
    ok = sum('dpm' in v for v in res.values()); print('done', ok, 'of', len(res), 'with history')


if __name__ == '__main__':
    main()
