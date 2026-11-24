"""DARKO history loader by column (shared by darko_parts.py and ms_diag.py)."""
import json
import ms_bench as mb


def load(col):
    out = {}
    for p, h in json.load(open(mb.DARKO)).items():
        if col not in h:
            continue
        rows = sorted((d, v) for d, v, f in zip(h['date'], h[col], h.get('future_game', [0] * len(h['date']))) if v is not None and not f)
        out[int(p)] = ([r[0] for r in rows], [r[1] for r in rows])
    return out
