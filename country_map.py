"""Player nationality from PlayerProfile.hcc ('School/Country'); latest profile wins."""
import gzip, json, pickle, zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from masked_paths import ARCHIVE, ROOT

def one(base):
    out = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        for line in gzip.open(z.open(base + 'raw/stats.jsonl.gz'), 'rt'):
            r = json.loads(line)
            if r['event'] == 'PlayerProfile#show':
                for pid, p in (r['metadata'].get('players') or {}).items():
                    h = p.get('hcc') or ''
                    if '/' in h:
                        out[int(pid)] = h.rsplit('/', 1)[1].strip()
    return out

if __name__ == '__main__':
    with zipfile.ZipFile(ARCHIVE) as zz:
        bases = sorted(n.rsplit('tracking.jsonl.gz', 1)[0] for n in zz.namelist() if n.endswith('tracking.jsonl.gz'))
    C = {}
    with ProcessPoolExecutor(10) as ex:
        for d in ex.map(one, bases, chunksize=8):
            C.update(d)
    pickle.dump(C, open(ROOT / 'results/groupwar_t/country.pkl', 'wb'))
    print(len(C), Counter(C.values()).most_common(25))
