import json, numpy as np, pandas as pd
from masked_paths import ROOT
D = pd.read_parquet(ROOT / 'results/iggy/minutes_predictions.parquet'); sel = json.load(open(ROOT / 'results/iggy/minutes.json'))['selected']
T = D[(D.season >= 2026) & ~D.absent].copy(); T['err'] = T[f'pred_{sel}'] - T.y; T['ae'] = T.err.abs()
tot = T.ae.sum()
cats = {
 'did not play, not in the absent set (DNP or unlisted injury)': (~T.played) & (T[f'pred_{sel}'] >= 5),
 'played with no recent minutes (new arrival / returning from injury)': T.played & (T.base == 0),
 'big minutes swing: actual >= predicted + 12': T.err <= -12,
 'big minutes cut: actual <= predicted - 12 (blowout, foul trouble, in-game injury)': T.played & (T.err >= 12),
}
out = {k: dict(rows=int(m.sum()), share_of_abs_error=round(float(T.ae[m].sum() / tot), 3), mae=round(float(T.ae[m].mean()), 2)) for k, m in cats.items()}
out['all non-absent test rows'] = dict(rows=int(len(T)), mae=round(float(T.ae.mean()), 2))
# blowout check: team-games by final margin
gl = pd.read_parquet(ROOT / 'results/latent/gamelogs.parquet')[['game_id', 'team_id', 'plus_minus']].drop_duplicates(['game_id', 'team_id'])
pm = gl.groupby(['game_id', 'team_id']).plus_minus.first()
T['margin'] = [abs(pm.get((g, t), np.nan) or 0) for g, t in zip(T.gid, T.team)]
for lo, hi in ((0, 10), (10, 20), (20, 99)):
    m = (T.margin >= lo) & (T.margin < hi); out[f'final margin {lo}-{hi}'] = dict(rows=int(m.sum()), mae=round(float(T.ae[m].mean()), 2))
json.dump(out, open(ROOT / 'results/iggy/failure_summary.json', 'w'), indent=1); print(json.dumps(out, indent=1))
