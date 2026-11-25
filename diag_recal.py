"""Our margins beat DARKO's but our log loss doesn't: is it the probability mapping? Re-map each arm's logit
with an expanding-window logistic fit on all EARLIER test blocks (point-in-time), and compare.
  python diag_recal.py   (PC)"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from masked_paths import ROOT
OUT = ROOT / 'results/latent'
arms = ['spm_lpit_h', 'kal', 'darko_cal']
B = []
for f in sorted(OUT.glob('msb_*.npz'), key=lambda f: f.stem.split('_')[2]):
    P = np.load(f)
    if all(a in P for a in arms):
        B.append({k: P[k] for k in arms + ['win']})
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
ll = lambda p, y: -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-9, 1))
res = {a: ([], []) for a in arms}; Y = []
for i, b in enumerate(B):
    hist = B[:i]; n = sum(len(h['win']) for h in hist)
    Y.append(b['win'])
    for a in arms:
        orig = b[a]
        if n >= 500:
            x = np.concatenate([lg(h[a]) for h in hist]); y = np.concatenate([h['win'] for h in hist])
            m = LogisticRegression(C=1e4).fit(x[:, None], y)
            new = m.predict_proba(lg(orig)[:, None])[:, 1]
        else:
            new = orig
        res[a][0].append(ll(orig, b['win'])); res[a][1].append(ll(new, b['win']))
rng = np.random.default_rng(0)
O = {a: np.concatenate(res[a][0]) for a in arms}; N = {a: np.concatenate(res[a][1]) for a in arms}
print(len(O['darko_cal']), 'games')
for a in arms:
    print(f'{a:12s} 30-day cal {O[a].mean():.4f}  expanding cal {N[a].mean():.4f}')
for a in arms[:-1]:
    for lab, X in (('expanding both', (N[a], N['darko_cal'])), ('ours expanding vs darko orig', (N[a], O['darko_cal']))):
        d = X[0] - X[1]; bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        print(f'  {a} {lab}: gap {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f} {np.percentile(bs, 97.5):+.4f}]')
