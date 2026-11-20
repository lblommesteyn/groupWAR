"""GroupWAR-T case study: 2027 World Cup rosters (Canada, USA) from 2025-26 NBA players.

Model: anchored scorer (tracking-prior RAPM backbone + box/tracking node features), full-season fit,
6 folds x 3 seeds. Baseline: ratings-only RAPM (v3-style backbone).
Deployment: v3 lineup mixture over the 792 five-man units, prior from a rotation-minutes template
(slots ordered by rating); final scores also under the Stackelberg leader-follower game (v3 core).
Search: greedy best single swap (v3 phase 2) from the top-12-by-rating start.
WAR_i = score(roster) - score(roster with i replaced by a replacement-level player).
"""
import itertools, json, pickle, sys
import numpy as np
import torch
import torch.nn.functional as F

import embed_eval as ee
import groupwar_t as gw

R = 12
TEMPLATE = np.array([.80, .78, .72, .68, .62, .45, .36, .25, .17, .10, .05, .02], np.float32)  # sums to 5
MIN_MINUTES = 300
FIX = {'Joel Embiid': 'USA', 'Kyrie Irving': 'USA', 'Chris Boucher': 'Canada'}
CASES = {'Canada': ['USA', 'France', 'Germany'], 'USA': ['Canada', 'France', 'Germany']}
OUT = gw.OUT / 'case'


# ---------------------------------------------------------------- deployment (v3 core.UnitSpace / full_graph)
COMBOS = list(itertools.combinations(range(R), 5))
INC = torch.zeros(len(COMBOS), R)
for k, c in enumerate(COMBOS):
    INC[k, list(c)] = 1.0


def within(logits):
    w = torch.softmax(logits, -1)
    return (INC.T.unsqueeze(0) * w.unsqueeze(1)) @ INC


def prior_logits(q):
    q = q.clamp(1e-3, 1 - 1e-3)
    return torch.log(q / (1 - q)) @ INC.T


def full_graph(Ah, Aa):
    mh, ma = torch.diagonal(Ah, dim1=1, dim2=2), torch.diagonal(Aa, dim1=1, dim2=2)
    cross = -mh.unsqueeze(-1) * ma.unsqueeze(1)
    return torch.cat([torch.cat([Ah, cross], -1), torch.cat([cross.transpose(1, 2), Aa], -1)], 1)


# ---------------------------------------------------------------- models
def fit_models(X, A, M, Rt, y, idx, use_gcn, seeds=3, folds=6, epochs=40):
    X, A, M, Rt = (torch.tensor(v) for v in (X, A, M, Rt))
    yt = torch.sigmoid(torch.tensor(y) / gw.SCALE)
    models = []
    for s in range(seeds):
        fs = np.array_split(np.random.default_rng(s).permutation(idx), folds)
        for k in range(folds):
            torch.manual_seed(100 * s + k)
            val = torch.tensor(fs[k]); tr = torch.tensor(np.concatenate([fs[j] for j in range(folds) if j != k]))
            m = gw.Anchored(X.shape[-1], use_gcn, True)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3 if use_gcn else 3e-2, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            best, st = 1e9, None
            for ep in range(epochs):
                m.train(); perm = tr[torch.randperm(len(tr))]
                for i in range(0, len(perm), 64):
                    b = perm[i:i + 64]
                    loss = F.mse_loss(m(X[b], A[b], M[b], Rt[b]), yt[b])
                    if m.last_fit is not None:
                        loss = loss + 0.01 * (m.last_fit ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                sch.step(); m.eval()
                with torch.no_grad():
                    vl = F.mse_loss(m(X[val], A[val], M[val], Rt[val]), yt[val]).item()
                if vl < best:
                    best, st = vl, {kk: v.clone() for kk, v in m.state_dict().items()}
            m.load_state_dict(st); m.eval(); models.append(m)
    return models


class Scorer:
    def __init__(self, models, feats, rating, rep):
        self.models, self.feats, self.rating, self.rep = models, feats, rating, rep
        self.nf = len(next(iter(feats.values())))

    def side(self, roster):
        """roster: list of pids (None = replacement). Slots ordered by rating."""
        key = lambda p: self.rating.get(p, self.rep[1]) if p is not None else self.rep[1]
        ps = sorted(roster, key=lambda p: -key(p))
        X = np.stack([self.feats.get(p, self.rep[0]) if p is not None else self.rep[0] for p in ps])
        r = np.array([key(p) for p in ps], np.float32) / gw.RSCALE
        n = len(ps); q = np.zeros(R, np.float32); q[:n] = TEMPLATE[:n] * 5 / TEMPLATE[:n].sum()
        Xp = np.zeros((R, self.nf), np.float32); Xp[:n] = X; rp = np.zeros(R, np.float32); rp[:n] = r
        mask = np.zeros(R, np.float32); mask[:n] = 1
        return Xp, rp, q, mask, ps

    def score(self, home_rosters, away_roster, mode='prior', return_deploy=False):
        """Mean P(home beats away) at a neutral site for a batch of home rosters."""
        H = [self.side(r) for r in home_rosters]; Aw = self.side(away_roster)
        B = len(H)
        Xh = torch.tensor(np.stack([h[0] for h in H])); rh = torch.tensor(np.stack([h[1] for h in H]))
        qh = torch.tensor(np.stack([h[2] for h in H])); mh = torch.tensor(np.stack([h[3] for h in H]))
        Xa = torch.tensor(Aw[0]).expand(B, -1, -1); ra = torch.tensor(Aw[1]).expand(B, -1)
        qa = torch.tensor(Aw[2]).expand(B, -1); ma = torch.tensor(Aw[3]).expand(B, -1)
        X, rat, M = torch.cat([Xh, Xa], 1), torch.cat([rh, ra], 1), torch.cat([mh, ma], 1)
        ph = prior_logits(qh) - 1e4 * (1 - mh) @ INC.T
        pa = prior_logits(qa) - 1e4 * (1 - ma) @ INC.T

        def f(lh, la):
            A = full_graph(within(lh), within(la))
            return torch.stack([m(X, A, M, rat, home=False) for m in self.models]).mean(0)  # neutral site

        if mode == 'prior':
            with torch.no_grad():
                return f(ph, pa).numpy()
        # Stackelberg (v3 core.stackelberg_eval): leader=home, follower=away, KL to prior
        lam, lr = 0.05, 0.3
        kl = lambda lg, pr: (torch.softmax(lg, -1) * (torch.log_softmax(lg, -1) - torch.log_softmax(pr, -1))).sum(-1)
        lh, la = ph.clone().requires_grad_(True), pa.clone().requires_grad_(True)
        of, ol = torch.optim.Adam([la], lr=lr), torch.optim.Adam([lh], lr=lr)
        for _ in range(20):
            for _ in range(5):
                l = (f(lh.detach(), la) + lam * kl(la, pa)).sum(); of.zero_grad(); l.backward(); of.step()
            l = (-f(lh, la.detach()) + lam * kl(lh, ph)).sum(); ol.zero_grad(); l.backward(); ol.step()
        for _ in range(5):
            l = (f(lh.detach(), la) + lam * kl(la, pa)).sum(); of.zero_grad(); l.backward(); of.step()
        with torch.no_grad():
            return f(lh, la).numpy()

    def total(self, rosters, opps, mode='prior'):
        return np.mean([self.score(rosters, o, mode) for o in opps], 0)


def search(sc, pool, opps, start, max_iter=12):
    cur = list(start); best = float(sc.total([cur], opps)[0]); hist = [best]
    for it in range(max_iter):
        bench = [p for p in pool if p not in cur]
        cands = [(i, b) for i in range(R) for b in bench]
        rosters = [cur[:i] + [b] + cur[i + 1:] for i, b in cands]
        vals = np.concatenate([sc.total(rosters[j:j + 512], opps) for j in range(0, len(rosters), 512)])
        k = int(np.argmax(vals))
        if vals[k] <= best + 1e-5:
            break
        cur, best = rosters[k], float(vals[k]); hist.append(best)
    return cur, best, hist


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(ee.E / 'player_games.pkl', 'rb'))
    shots = [json.loads(l) for l in open(ee.E / 'shots_xfg.jsonl')]
    G = {g: d for g, d in G.items() if d['stints']}
    names = {}
    mins = {}
    for d in G.values():
        names.update(d['players'])
        for p, b in d['box'].items():
            mins[p] = mins.get(p, 0) + b['sec'] / 60
    country = pickle.load(open(gw.OUT / 'country.pkl', 'rb'))
    for p, n in names.items():
        if n in FIX:
            country[p] = FIX[n]
    allg = sorted(G, key=lambda g: G[g]['date'])
    mid = G[allg[len(allg) // 2]]['date']
    h1 = [g for g in allg if G[g]['date'] < mid]; h2 = [g for g in allg if G[g]['date'] >= mid]
    import rolling_prior as rp
    cache = OUT / 'tables_hist.pkl'
    if cache.exists():
        t1, t2, tf = pickle.load(open(cache, 'rb'))
    else:
        conv = lambda t: (t[0]['rapm'], t[0]['rapm_hist_trk'], {'box_trk': t[1]['comp']})  # rapm_t slot = new backbone
        t1, t2, tf = (conv(rp.tables(G, shots, a, b)) for a, b in ((h2, h1), (h1, h2), (allg, h2)))
        pickle.dump((t1, t2, tf), open(cache, 'wb'))
    recs = gw.game_records(G)
    A, ok = gw.rolling_A(recs)
    which = {g: t1 for g in h1} | {g: t2 for g in h2}
    M, Rts, Xs = gw.tensors(recs, {r['game_id']: which[r['game_id']] for r in recs})
    y = np.array([r['margin'] for r in recs], np.float32)
    idx = np.flatnonzero(ok)
    print(f'training on {len(idx)} games', flush=True)
    rapm, rapm_t, feats = tf
    F_ = feats['box_trk']
    zero_f = {p: np.zeros(1, np.float32) for p in F_}
    mdl_t = fit_models(Xs['box_trk'], A, M, Rts['rapm_t'], y, idx, True)
    mdl_r = fit_models(np.zeros_like(Xs['box_trk'][..., :1]), A, M, Rts['rapm'], y, idx, False)
    # replacement level: players with 300-600 minutes
    reps = [p for p in F_ if 300 <= mins.get(p, 0) <= 600]
    rep_t = (np.mean([F_[p] for p in reps], 0).astype(np.float32), float(np.mean([rapm_t[p] for p in reps])))
    rep_r = (np.zeros(1, np.float32), float(np.mean([rapm[p] for p in reps])))
    scorers = {'groupwar_t_hist': Scorer(mdl_t, F_, rapm_t, rep_t), 'rapm_v3style': Scorer(mdl_r, zero_f, rapm, rep_r)}

    def pool_of(c):
        return [p for p, k in country.items() if k == c and mins.get(p, 0) >= MIN_MINUTES and p in F_]

    results = {}
    for team, opp_names in CASES.items():
        pool = pool_of(team)
        for sname, sc in scorers.items():
            opps = [sorted(pool_of(o), key=lambda p: -sc.rating.get(p, -99))[:R] for o in opp_names]
            start = sorted(pool, key=lambda p: -sc.rating.get(p, -99))[:R]
            roster, val, hist = search(sc, pool, opps, start)
            stack = float(np.mean([sc.score([roster], o, 'stackelberg')[0] for o in opps]))
            prior = float(sc.total([roster], opps)[0])
            war = {}
            for i, p in enumerate(roster):
                rr = roster[:i] + [None] + roster[i + 1:]
                war[names.get(p, str(p))] = round(prior - float(sc.total([rr], opps)[0]), 4)
            gwar = round(sum(war.values()), 4)
            results[f'{team}/{sname}'] = dict(
                roster=[names.get(p, str(p)) for p in sorted(roster, key=lambda p: -sc.rating.get(p, 0))],
                start=[names.get(p, str(p)) for p in start], prior_score=prior, stackelberg_score=stack,
                swaps=len(hist) - 1, war=dict(sorted(war.items(), key=lambda kv: -kv[1])), groupwar=gwar,
                opponents=opp_names, pool_size=len(pool),
                top_bench=[names.get(p, str(p)) for p in sorted([p for p in pool if p not in roster],
                                                                key=lambda p: -sc.rating.get(p, -99))[:5]])
            r = results[f'{team}/{sname}']
            print(f"\n== {team} / {sname}: pool {len(pool)}, prior {prior:.3f}, stackelberg {stack:.3f}, swaps {r['swaps']}, GroupWAR {gwar}")
            print('  roster:', r['roster'])
            print('  WAR:', r['war'])
            print('  left out (top by rating):', r['top_bench'])
    # player value table for reference
    table = sorted([(names.get(p, str(p)), country.get(p), round(rapm.get(p, 0), 2), round(rapm_t.get(p, 0), 2), int(mins.get(p, 0)))
                    for p in F_ if mins.get(p, 0) >= MIN_MINUTES], key=lambda r: -r[3])
    json.dump(dict(results=results, top_players=table[:60]), open(OUT / 'case_study_hist.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
