"""Roster search: tournament elimination followed by batched greedy 1-swap,
under positional quotas and optional hard locks."""
import random

import numpy as np
import torch


class Quota:
    """NBA: 12 players, >= min_g guards, >= min_c centres (multi-position players count
    for each listed position). NHL: 12 forwards + 6 D, >= min_l LW, >= min_r RW."""

    def __init__(self, sport, pos, min_g=4, min_c=2, min_l=3, min_r=3, enforce=True):
        self.sport, self.pos, self.enforce = sport, pos, enforce
        self.min_g, self.min_c, self.min_l, self.min_r = min_g, min_c, min_l, min_r

    def is_d(self, p):
        return self.pos.get(p) == "D"

    def ok(self, roster):
        if self.sport == "nba":
            if len(roster) != 12:
                return False
            if not self.enforce:
                return True
            g = sum("Guard" in str(self.pos.get(p, "")) for p in roster)
            c = sum("Center" in str(self.pos.get(p, "")) for p in roster)
            return g >= self.min_g and c >= self.min_c
        d = sum(self.is_d(p) for p in roster)
        if len(roster) != 18 or d != 6:
            return False
        if not self.enforce:
            return True
        l = sum(self.pos.get(p) == "L" for p in roster)
        r = sum(self.pos.get(p) == "R" for p in roster)
        return l >= self.min_l and r >= self.min_r

    def same_group(self, a, b):
        return self.sport == "nba" or self.is_d(a) == self.is_d(b)


def random_valid(pool, locks, quota, rng, tries=2000):
    R = 12 if quota.sport == "nba" else 18
    free = [p for p in pool if p not in locks]
    for _ in range(tries):
        rng.shuffle(free)
        if quota.sport == "nba":
            team = list(locks) + free[:R - len(locks)]
        else:
            nd = 6 - sum(quota.is_d(p) for p in locks)
            nf = 12 - (len(locks) - (6 - nd))
            team = (list(locks) + [p for p in free if quota.is_d(p)][:nd]
                    + [p for p in free if not quota.is_d(p)][:nf])
        if quota.ok(team):
            return team
    raise RuntimeError("could not sample a valid roster")


def score(ev, rosters, opponents):
    return ev.score_rosters(rosters, opponents).mean(1)


def tournament(ev, pool, opponents, locks, quota, seed=0, verbose=True, max_rounds=25):
    rng = random.Random(seed)
    R = ev.R
    remaining = [p for p in pool if p not in locks]
    target = 2 * (R - len(locks))
    best = (-1, None)
    rnd = 0
    while len(remaining) > target and rnd < max_rounds:
        rnd += 1
        rng.shuffle(remaining)
        teams, used = [], set()
        for _ in range(max(2, len(remaining) // (R - len(locks)))):
            avail = [p for p in remaining if p not in used]
            try:
                t = random_valid(avail, locks, quota, rng, tries=200)
            except RuntimeError:
                break
            teams.append(t)
            used.update(p for p in t if p not in locks)
        if len(teams) < 2:
            break
        s = score(ev, teams, opponents).numpy()
        order = np.argsort(-s)
        if s[order[0]] > best[0]:
            best = (float(s[order[0]]), teams[order[0]])
        keep = set()
        for k in order[: max(1, len(teams) // 3)]:
            keep.update(p for p in teams[k] if p not in locks)
        drop = set()
        for k in order[max(1, len(teams) // 3):]:
            _, war = ev.war(teams[k], opponents)
            cands = sorted([p for p in teams[k] if p not in locks], key=lambda p: war[p])
            n_drop = 2 if quota.sport == "nba" else 4
            drop.update(cands[:n_drop])
        remaining = [p for p in remaining if p in keep or p not in drop]
        if verbose:
            print(f"  round {rnd}: {len(teams)} teams, best {s[order[0]]:.4f}, "
                  f"{len(remaining)} remain", flush=True)
    return remaining, best


def greedy(ev, start, pool, opponents, locks, quota, max_iters=40, verbose=True, tol=1e-4):
    cur = list(start)
    cur_s = float(score(ev, [cur], opponents)[0])
    history = [(0, cur_s, None, None)]
    for it in range(1, max_iters + 1):
        cands, moves = [], []
        for i, out in enumerate(cur):
            if out in locks:
                continue
            for inn in pool:
                if inn in cur or not quota.same_group(out, inn):
                    continue
                new = cur[:i] + [inn] + cur[i + 1:]
                if quota.ok(new):
                    cands.append(new)
                    moves.append((out, inn))
        if not cands:
            break
        s = score(ev, cands, opponents).numpy()
        k = int(np.argmax(s))
        if s[k] <= cur_s + tol:
            break
        cur, cur_s = cands[k], float(s[k])
        history.append((it, cur_s, moves[k][0], moves[k][1]))
        if verbose:
            print(f"  greedy {it}: {cur_s:.4f}  out {moves[k][0]} in {moves[k][1]}  "
                  f"({len(cands)} candidates)", flush=True)
    return cur, cur_s, history


def run_search(ev, pool, opponents, locks, quota, seed=0, verbose=True):
    remaining, best = tournament(ev, pool, opponents, locks, quota, seed, verbose)
    rng = random.Random(seed + 1)
    # start greedy from the best tournament team restricted to survivors where possible
    survivors = list(locks) + remaining
    try:
        start = random_valid(survivors, locks, quota, rng)
        s0 = float(score(ev, [start], opponents)[0])
        if best[1] is not None and best[0] > s0:
            start = best[1]
    except RuntimeError:
        start = best[1]
    return greedy(ev, start, pool, opponents, locks, quota, verbose=verbose)
