"""Does tracking at release improve shot-make prediction on later games?

A: location + PBP shot type/action words + shooter history (empirical Bayes, earlier dates only)
B: A + hand-built defender/shooter/spacing features at release
Train on games before SPLIT, test on games from SPLIT on, same shots for every model.
"""
import json, re
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from shot_align import OUT

SPLIT = '2026-04-01'
ACTIONS = ['driving', 'pullup', 'running', 'floating', 'step back', 'turnaround', 'fadeaway', 'cutting',
           'finger roll', 'tip', 'putback', 'reverse', 'alley oop', 'bank', 'hook', 'dunk', 'layup']
SUBTYPES = ['Jump Shot', 'Layup', 'DUNK', 'Hook']
TRK = {
    'defender': ['d1', 'd2', 'n_def_4', 'n_def_6', 'n_def_10', 'def_cos', 'def_between', 'close_speed', 'd1_prev'],
    'shooter_motion': ['shooter_speed', 'shooter_toward_hoop', 'touch_s', 'ball_z_release'],
    'spacing': ['mate_mean_dist', 'off_hull', 'mates_open', 'def_hull', 'def_paint'],
}
K = 60  # prior strength in attempts


def zone(r):
    d = np.hypot(r['pbp_x'], r['pbp_y'])
    if r['three']:
        return 'c3' if r['pbp_y'] < 9 else 'ab3'
    return 'rim' if d < 4 else 'paint' if d < 10 else 'mid'


def shooter_priors(all_rows):
    """Per-shot shooter residual vs league zone rate, using only earlier dates."""
    train = [r for r in all_rows if r['date'] < SPLIT]
    league = defaultdict(lambda: [0, 0])
    for r in train:
        z = league[zone(r)]; z[0] += r['made']; z[1] += 1
    lr = {z: m / n for z, (m, n) in league.items()}
    hist = defaultdict(lambda: np.zeros(4))  # (resid2, n2, resid3, n3)
    pri = {}
    for date in sorted({r['date'] for r in all_rows}):
        day = [r for r in all_rows if r['date'] == date]
        for r in day:
            h = hist[r['shooter']]
            pri[(r['game_id'], r['event_id'])] = (h[0] / (h[1] + K), h[2] / (h[3] + K), np.log1p(h[1] + h[3]))
        for r in day:
            h = hist[r['shooter']]; i = 2 if r['three'] else 0
            h[i] += r['made'] - lr[zone(r)]; h[i + 1] += 1
    return pri


def base_x(r, pri):
    d = r['desc'].lower()
    x = [np.hypot(r['pbp_x'], r['pbp_y']), abs(r['pbp_x']), r['pbp_y'], r['three']]
    x += [r['subtype'] == s for s in SUBTYPES] + [a in d for a in ACTIONS]
    x += list(pri[(r['game_id'], r['event_id'])])
    return x


def fit_eval(Xtr, ytr, Xte, seed=0):
    m = HistGradientBoostingClassifier(learning_rate=0.04, max_iter=1500, max_leaf_nodes=15, min_samples_leaf=80,
                                       l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                       n_iter_no_change=60, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def boot_diff(y, pa, pb, games, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(games, return_inverse=True)
    la = -(y * np.log(pa) + (1 - y) * np.log(1 - pa)); lb = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    sa, sb, c = np.bincount(inv, la), np.bincount(inv, lb), np.bincount(inv)
    d = []
    for _ in range(n):
        g = rng.integers(len(ug), size=len(ug))
        d.append((sb[g].sum() - sa[g].sum()) / c[g].sum())
    return float(np.mean(lb - la)), [float(v) for v in np.percentile(d, [2.5, 97.5])]


def metrics(y, p):
    return dict(log_loss=float(log_loss(y, p)), brier=float(brier_score_loss(y, p)), auc=float(roc_auc_score(y, p)))


if __name__ == '__main__':
    allrows = [json.loads(l) for l in open(OUT / 'alignment.jsonl')]
    pri = shooter_priors(allrows)
    rows = [json.loads(l) for l in open(OUT / 'features.jsonl')]
    rows = [r for r in rows if r['trk_ok']]
    tr = np.array([r['date'] < SPLIT for r in rows])
    y = np.array([r['made'] for r in rows], float)
    games = np.array([r['game_id'] for r in rows])
    XA = np.array([base_x(r, pri) for r in rows], float)
    trk_cols = sum(TRK.values(), [])
    XT = np.array([[r.get(c, np.nan) for c in trk_cols] for r in rows], float)
    print(f'train {tr.sum()} shots / {len(set(games[tr]))} games; test {(~tr).sum()} / {len(set(games[~tr]))} games')

    seeds = range(3)
    def run(X):
        return np.mean([fit_eval(X[tr], y[tr], X[~tr], s) for s in seeds], axis=0)

    yt, gt = y[~tr], games[~tr]
    preds = {'league_rate': np.full(len(yt), y[tr].mean()), 'A_location_type_shooter': run(XA)}
    preds['A_minus_shooter'] = run(XA[:, :-3])
    preds['B_plus_all_tracking'] = run(np.hstack([XA, XT]))
    idx = {g: [trk_cols.index(c) for c in cs] for g, cs in TRK.items()}
    for g, ii in idx.items():
        preds[f'A_plus_{g}'] = run(np.hstack([XA, XT[:, ii]]))
    res = {'split': SPLIT, 'n_train': int(tr.sum()), 'n_test': int((~tr).sum()),
           'test_games': int(len(set(gt))), 'models': {k: metrics(yt, p) for k, p in preds.items()}, 'vs_A': {}}
    for k, p in preds.items():
        if k != 'A_location_type_shooter':
            dm, ci = boot_diff(yt, preds['A_location_type_shooter'], p, gt)
            res['vs_A'][k] = dict(logloss_change=dm, ci95=ci)
    # by shot class
    three = np.array([r['three'] for r in rows])[~tr]
    res['by_class'] = {}
    for name, msk in [('two', ~three), ('three', three)]:
        res['by_class'][name] = {k: metrics(yt[msk], preds[k][msk]) for k in ['A_location_type_shooter', 'B_plus_all_tracking']}
        res['by_class'][name]['B_vs_A'] = boot_diff(yt[msk], preds['A_location_type_shooter'][msk], preds['B_plus_all_tracking'][msk], gt[msk])
    json.dump(res, open(OUT / 'shot_model.json', 'w'), indent=1)
    np.savez(OUT / 'shot_model_preds.npz', y=yt, games=gt, **preds)
    for k, v in res['models'].items():
        extra = res['vs_A'].get(k)
        print(f"{k:28s} ll={v['log_loss']:.4f} brier={v['brier']:.4f} auc={v['auc']:.4f}",
              f"Δll vs A={extra['logloss_change']:+.4f} [{extra['ci95'][0]:+.4f},{extra['ci95'][1]:+.4f}]" if extra else '')
    for name, v in res['by_class'].items():
        print(name, {k: round(m['log_loss'], 4) for k, m in v.items() if k != 'B_vs_A'}, 'Δ', v['B_vs_A'])
