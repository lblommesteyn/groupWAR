"""Chronological tracking representations for the corrected movers test (AUDIT.md I3).

For each cutoff, using ONLY tracking possessions and box scores dated before it:
  BORIS     offensive Gaussian mixture refit on pre-cutoff player-possessions (archetype_mix.prep / fit, same K);
            per-player mix = mean membership over his pre-cutoff possessions (>= 60)
  BOWEN     attacker descriptors (world_desc.py recipe: 3PA/36, shrunk 3P%, FTA/36, AST/36, rim attempts/36 + the
            6 BORIS shares) from pre-cutoff box scores and the pre-cutoff BORIS mix, one row per player
Outputs results/world/chrono_{tag}/: boris_mix.json {pid: [6]}, desc.json {"all": {pid: [11]}}, meta.json.
world_model.py then trains with WM_CUT / WM_DIR set (games before the cutoff only, cross-fit within them).
  python chrono_reps.py   (Mac: tracking archives' derived files live here)
"""
import json, pickle
from collections import defaultdict
import numpy as np

import archetype_mix as am
from masked_paths import ROOT
from provenance import provenance

CUTS = {'dl2026': '2026-02-05', 'os2026': '2026-10-20'}
SRC = [ROOT / 'results/embed/player_games.pkl', ROOT.parent / 'court-model-2627/results/embed/player_games.pkl']


def main():
    D, OFF_F, _ = am.load()
    for tag, cut in CUTS.items():
        sel = D['date'] < cut
        Fraw = D['fo'][sel].reshape(-1, len(OFF_F)); pid = D['off'][sel].reshape(-1)
        gm, R = am.fit(am.prep(Fraw, OFF_F), am.K_OFF)
        by = defaultdict(list)
        for r, p in enumerate(pid):
            by[int(p)].append(r)
        mix = {str(p): R[rows].mean(0).round(4).tolist() for p, rows in by.items() if len(rows) >= 60}
        tot = defaultdict(lambda: np.zeros(7))
        for f in SRC:
            if not f.exists():
                continue
            for g in pickle.load(open(f, 'rb')).values():
                if g['date'] >= cut:
                    continue
                for p, b in g['box'].items():
                    tot[p] += [b['sec'], b['tpa'], b['tpm'], b['fta'], b['ast'], b.get('rim_a', 0), b['fga']]
        lg3 = sum(v[2] for v in tot.values()) / max(sum(v[1] for v in tot.values()), 1)
        rows = {}
        for p, v in tot.items():
            m = max(v[0] / 60, 1)
            rows[p] = [36 * v[1] / m, (v[2] + 150 * lg3) / (v[1] + 150), 36 * v[3] / m, 36 * v[4] / m, 36 * v[5] / m] + \
                      (mix.get(str(p)) or [np.nan] * am.K_OFF)
        A = np.array(list(rows.values()), float); mu = np.nanmean(A, 0); sd = np.nanstd(A, 0) + 1e-9
        desc = {'all': {str(p): np.nan_to_num((np.array(r) - mu) / sd).round(3).tolist() for p, r in rows.items()}}
        out = ROOT / 'results/world' / f'chrono_{tag}'; out.mkdir(parents=True, exist_ok=True)
        json.dump(mix, open(out / 'boris_mix.json', 'w')); json.dump(desc, open(out / 'desc.json', 'w'))
        meta = dict(cut=cut, tracked_possessions=int(sel.sum()), tracked_games=int(len(np.unique(D['gid'][sel]))),
                    players_with_mix=len(mix), players_with_desc=len(rows),
                    provenance=provenance(__file__, ['results/archetypes/poss_2026.npz', 'results/archetypes/poss_2027.npz',
                                                     'results/embed/player_games.pkl']))
        json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
        print(tag, {k: v for k, v in meta.items() if k != 'provenance'}, flush=True)


if __name__ == '__main__':
    main()
