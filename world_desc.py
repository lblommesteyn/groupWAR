"""Attacker descriptors per player-season for the world model: shooting threat and on-ball role.
From the tracking seasons' box scores (player_games.pkl, both workspaces) and the Flux offense mix (archetype_mix).
  3PA per 36, shrunk 3P% (K = 150 attempts), FTA per 36, AST per 36, rim attempts per 36, + 6 Flux mix shares.
Output: results/world/desc.json {season: {pid: [11 floats]}}, standardised per column.   python world_desc.py (Mac)
"""
import json, pickle
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = {2026: ROOT / 'results/embed/player_games.pkl', 2027: ROOT.parent / 'court-model-2627/results/embed/player_games.pkl'}


def main():
    mix = json.load(open(ROOT / 'results/archetypes/mix.json'))['players']['off']
    out = {}
    for season, f in SRC.items():
        G = pickle.load(open(f, 'rb')); tot = defaultdict(lambda: np.zeros(7))
        for g in G.values():
            for p, b in g['box'].items():
                tot[p] += [b['sec'], b['tpa'], b['tpm'], b['fta'], b['ast'], b.get('rim_a', 0), b['fga']]
        lg3 = sum(v[2] for v in tot.values()) / max(sum(v[1] for v in tot.values()), 1)
        rows = {}
        for p, v in tot.items():
            m = max(v[0] / 60, 1)
            fl = mix.get(str(p), {}).get('by_season', {}).get(str(season)) or mix.get(str(p), {}).get('mix') or [np.nan] * 6
            rows[p] = [36 * v[1] / m, (v[2] + 150 * lg3) / (v[1] + 150), 36 * v[3] / m, 36 * v[4] / m, 36 * v[5] / m] + list(fl)
        A = np.array(list(rows.values()), float); mu = np.nanmean(A, 0); sd = np.nanstd(A, 0) + 1e-9
        out[str(season)] = {str(p): np.nan_to_num((np.array(r) - mu) / sd).round(3).tolist() for p, r in rows.items()}
        print(season, len(rows), 'players')
    json.dump(out, open(ROOT / 'results/world/desc.json', 'w'))


if __name__ == '__main__':
    main()
