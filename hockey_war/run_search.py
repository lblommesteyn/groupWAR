# Data Manipulation
import numpy as np
import pandas as pd
# Shuffling
import random
# Tabu search
from collections import deque
# Project model
from mie368stackel import stackelberg, GCN_DeepSet_AntiSym_Invariant
# Tensor operations
import torch
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Team USA player IDs
usa = [20240008479314, 20240008480801, 20240008479318, 20240008481559,
       20240008478403, 20240008477404, 20240008478398, 20240008475184,
       20240008477946, 20240008476468, 20140008475754, 20240008476389,
       20240008479325, 20240008476958, 20240008479323,
       20240008478460, 20240008478396, 20240008482105]

# Load models
torch.cuda.set_device(0)
modells = []
for i in range(6):
    model = GCN_DeepSet_AntiSym_Invariant(
        275, 128, 6, 128, 128, 36, 128, use_edge_agg=False, dropout=5e-2
    )
    new_state_dict = {}
    for k, v in torch.load(f'model_{i}_30.pth', map_location=device).items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    modells.append(model.to(device))
print("Models loaded.")


class TabuSearchLineup:
    def __init__(
        self,
        embeddings,
        embeddings2,
        poolA,
        poolB,
        sizeA=12,
        sizeB=6,
        tabu_size=50,
        candidates_per_swap=5,
        num_low_usage=2,
        seed=123,
        year=2024,
    ):
        self.poolA = poolA['player_id2'].unique()
        self.poolB = poolB['player_id2'].unique()
        self.sizeA = sizeA
        self.sizeB = sizeB
        self.rng = random.Random(seed)
        self.embeddings = embeddings
        self.embeddings2 = embeddings2
        self.tabu = deque(maxlen=tabu_size)
        self.year = year
        self.candidates_per_swap = candidates_per_swap
        self.num_low_usage = num_low_usage

        # Load APM results once
        self.results = pd.concat([
            pd.read_csv("data/apm_minutes.csv"),
            pd.read_csv("apm_minutes2025.csv")
        ])
        self.results['year'] = self.results['player_id'] // 1e10

        # Cache opponent tensors — same 3 opponents evaluated every call
        self._away_cache = {}

    def hash_lineup(self, lineupA, lineupB):
        return tuple(sorted(lineupA)), tuple(sorted(lineupB))

    def evaluate(self, lineupA, lineupB):
        if self.year == 2025 or self.year == 2024:
            teams = ['EDM', 'FLA', 'CAR', 'DAL']
        elif self.year == 2021:
            teams = ['MTL', "TBL", 'VGK', 'NYI']
        elif self.year == 2017:
            teams = ['PIT', 'NSH', 'OTT', 'WSH']
        elif self.year == 2013:
            teams = ['LAK', 'NYR', 'CHI', 'MTL']
        teams = ['USA'] + teams[:2]

        results = self.results
        year = self.year

        hometeams = []
        # FIX: collect d1s from every opponent team, not just the last one
        all_scores = []

        homex = self.embeddings[
            self.embeddings['player_id2'].isin(lineupA + lineupB)
        ].drop_duplicates()

        players = homex['player_id2'].values
        P = len(players)
        zero_masks = torch.zeros(P, 36, 36, dtype=torch.int32)

        for awayteam in teams:
            cache_key = (awayteam, year)
            if cache_key not in self._away_cache:
                if awayteam == 'USA':
                    awayf = results[
                        results['player_id'].isin(usa) &
                        results['positionCode'].isin(['C', 'L', 'R'])
                    ].sort_values(by='difference', ascending=False).iloc[:12].reset_index()
                    awayd = results[
                        results['player_id'].isin(usa) &
                        results['positionCode'].isin(['D'])
                    ].sort_values(by='difference', ascending=False).iloc[:6].reset_index()
                else:
                    awayf = results[
                        (results['year'] == year) &
                        results['team'].isin([awayteam]) &
                        results['positionCode'].isin(['C', 'L', 'R'])
                    ].sort_values(by='difference', ascending=False).iloc[:12].reset_index()
                    awayd = results[
                        (results['year'] == year) &
                        results['team'].isin([awayteam]) &
                        results['positionCode'].isin(['D'])
                    ].sort_values(by='difference', ascending=False).iloc[:6].reset_index()

                ids = list(awayf['player_id']) + list(awayd['player_id'])
                awayx = (
                    self.embeddings2[self.embeddings2['player_id2'].isin(ids)]
                    .drop_duplicates(subset=["player_id2"])
                    .reset_index(drop=True)
                ).copy()
                awayx = awayx.sort_values(
                    by='positionCode',
                    key=lambda x: x.isin(['C', 'L', 'R']),
                    ascending=False
                ).iloc[:, :-1]

                if len(awayx) < 18:
                    rows = [[0] * len(awayx.columns) for _ in range(18 - len(awayx))]
                    awayx = pd.concat([awayx, pd.DataFrame(rows, columns=awayx.columns)])

                # Use pd.concat to align columns (homex/awayx may differ by positionCode),
                # then split into home and away tensors with matching feature dim
                combined = pd.concat([homex, awayx])
                combined_np = combined.iloc[:, 2:-2].fillna(0).to_numpy()
                combined_t = torch.tensor(combined_np, dtype=torch.float32)
                combined_t = torch.cat((combined_t[:, :99], combined_t[:, 101:]), dim=-1)
                n_home = len(homex)
                home_base = combined_t[:n_home]   # (N_home, F)
                away_base = combined_t[n_home:]    # (18, F)
                self._away_cache[cache_key] = (home_base, away_base)

            home_base, away_tensor = self._away_cache[cache_key]

            # Baseline: single sample
            rep_base = torch.cat([home_base, away_tensor], dim=0).unsqueeze(0)  # (1, 36, F)
            main, avg3, opt_l, opt_f = stackelberg(
                rep_base, modells, torch.zeros((1, 36, 36), dtype=torch.int32)
            )

            # All P knockouts vectorized — build in batch, no copies of homex
            home_batch = home_base.unsqueeze(0).expand(P, -1, -1).clone()  # (P, N_home, F)
            for i in range(P):
                home_batch[i, i, :] = -2.5
            away_expanded = away_tensor.unsqueeze(0).expand(P, -1, -1)
            reps = torch.cat([home_batch, away_expanded], dim=1)  # (P, 36, F)

            d1, avg2, ig, ig2 = stackelberg(reps, modells, zero_masks)

            d1s = [(main - d1[i]) for i in range(P)]
            all_scores.append(d1s)
            hometeams.append(main)

        # FIX: average player contributions across all opponent teams
        # all_scores: list of T lists, each containing P tensors of shape (1,)
        all_d1s = torch.stack([torch.stack(s) for s in all_scores])  # (T, P, 1)
        avg_d1s = all_d1s.mean(dim=0).squeeze(-1)                     # (P,)

        return (
            torch.stack(hometeams).mean().item(),
            avg_d1s,
            homex['player_id2'].unique()
        )

    def evaluate_batch(self, candidates, chunk_size=32):
        """
        Batched evaluation of B candidate lineups. Assumes cache is pre-populated
        (call self.evaluate(...) at least once beforehand).
        candidates: list of (lineupA, lineupB) tuples, each yielding 18 players total.
        Returns: (B,) tensor of mean scores across the 3 opponent teams.
        """
        if self.year == 2025 or self.year == 2024:
            teams = ['EDM', 'FLA', 'CAR', 'DAL']
        elif self.year == 2021:
            teams = ['MTL', "TBL", 'VGK', 'NYI']
        elif self.year == 2017:
            teams = ['PIT', 'NSH', 'OTT', 'WSH']
        elif self.year == 2013:
            teams = ['LAK', 'NYR', 'CHI', 'MTL']
        teams = ['USA'] + teams[:2]

        B = len(candidates)
        all_scores = torch.zeros(B)

        for start in range(0, B, chunk_size):
            chunk = candidates[start:start + chunk_size]
            cB = len(chunk)

            # Build fresh home tensor per candidate (same slicing as evaluate())
            home_tensors = []
            for lineupA, lineupB in chunk:
                ids = list(lineupA) + list(lineupB)
                homex = self.embeddings[
                    self.embeddings['player_id2'].isin(ids)
                ].drop_duplicates(subset=['player_id2'])
                home_np = homex.iloc[:, 2:-2].fillna(0).to_numpy()
                home_t = torch.tensor(home_np, dtype=torch.float32)
                home_t = torch.cat((home_t[:, :99], home_t[:, 101:]), dim=-1)
                # Pad to 18 rows if needed (shouldn't happen but safe)
                if home_t.size(0) < 18:
                    pad = torch.zeros(18 - home_t.size(0), home_t.size(1))
                    home_t = torch.cat([home_t, pad], dim=0)
                home_tensors.append(home_t[:18])
            home_batch = torch.stack(home_tensors)  # (cB, 18, F)

            # Accumulate over opponents
            chunk_scores = torch.zeros(cB)
            zero_masks = torch.zeros(cB, 36, 36, dtype=torch.int32)
            for awayteam in teams:
                _, away_tensor = self._away_cache[(awayteam, self.year)]
                away_expanded = away_tensor.unsqueeze(0).expand(cB, -1, -1)
                reps = torch.cat([home_batch, away_expanded], dim=1)  # (cB, 36, F)
                scores_t, _, _, _ = stackelberg(reps, modells, zero_masks)
                chunk_scores = chunk_scores + scores_t.squeeze(-1).cpu()
            chunk_scores = chunk_scores / len(teams)
            all_scores[start:start + cB] = chunk_scores

        return all_scores

    def propose_swaps(self, lineupA, lineupB, usageA, usageB):
        neighbors = []
        lowA_idx = usageA['order'].iloc[:self.num_low_usage]
        benchA = list(set(self.poolA) - set(lineupA))
        for idx in lowA_idx:
            p_out = idx
            idx = lineupA.index(p_out)
            bench_sorted = sorted(
                benchA, key=lambda p: self.dist.loc[p_out, p], reverse=True
            )
            for p_in in bench_sorted[:self.candidates_per_swap]:
                move = ("A", p_out, p_in)
                if move not in self.tabu:
                    newA = lineupA.copy()
                    newA[idx] = p_in
                    neighbors.append((newA, lineupB, move))

        lowB_idx = usageB['order'].iloc[:self.num_low_usage]
        benchB = list(set(self.poolB) - set(lineupB))
        for idx in lowB_idx:
            p_out = idx
            idx = lineupB.index(p_out)
            bench_sorted = sorted(
                benchB, key=lambda p: self.dist.loc[p_out, p], reverse=True
            )
            for p_in in bench_sorted[:self.candidates_per_swap]:
                move = ("B", p_out, p_in)
                if move not in self.tabu:
                    newB = lineupB.copy()
                    newB[idx] = p_in
                    neighbors.append((lineupA, newB, move))

        return neighbors

    def run(self, max_iters=200, initialA=None, initialB=None, verbose=True):
        initialA = self.poolA['player_id2'].tolist()[:12]
        initialB = self.poolB['player_id2'].tolist()[:6]

        lineupA, lineupB = initialA[:], initialB[:]

        best_score, usage, order = self.evaluate(lineupA, lineupB)
        best_lineup = (lineupA[:], lineupB[:])

        if verbose:
            print(f"Init score: {best_score:.4f}")

        for it in range(max_iters):
            best_score, usage, order = self.evaluate(lineupA, lineupB)
            # FIX: usage is (P,) tensor — don't index with [0]
            usage = pd.DataFrame({'usage': usage.tolist(), 'order': order.tolist()})
            neighbors = self.propose_swaps(
                lineupA, lineupB,
                usage[usage['order'].isin(self.poolA)].sort_values(by='usage', ascending=True),
                usage[usage['order'].isin(self.poolB)].sort_values(by='usage', ascending=True)
            )
            if not neighbors:
                if verbose:
                    print("No new neighbors possible.")
                break

            scored = []
            for newA, newB, move in neighbors:
                s2, usage, order = self.evaluate(newA, newB)
                usage = pd.DataFrame({'usage': usage.tolist(), 'order': order.tolist()})
                scored.append((s2, usage[usage['order'].isin(self.poolA)]['order'].tolist(),
                                usage[usage['order'].isin(self.poolB)]['order'].tolist(), move))

            scored.sort(key=lambda x: x[0], reverse=True)
            s2, newA, newB, move = scored[0]

            lineupA, lineupB = newA, newB
            self.tabu.append(move)

            if s2 > best_score:
                best_score = s2
                best_lineup = (newA[:], newB[:])

            if verbose:
                print(f"Iter {it:3d}:  score={s2:.4f}  best={best_score:.4f}   move={move}")

        return best_lineup, best_score

    def run_tournament(self, roster, verbose=True, forw=[], defe=[]):
        forwards = roster[~roster['positionCode'].isin(['D', 'G'])]['player_id'].tolist()
        defense = roster[roster['positionCode'] == 'D']['player_id'].tolist()
        forwards = [f for f in forwards if f in self.embeddings['player_id2'].unique() and f not in forw]
        defense = [f for f in defense if f in self.embeddings['player_id2'].unique() and f not in defe]

        remaining_forwards = forwards[:]
        remaining_defense = defense[:]

        round_num = 0
        removed = []
        max_iters = 25

        while (
            (len(remaining_forwards) + len(remaining_defense)) > (36 - len(forw) - len(defe))
            and round_num < max_iters
        ):
            round_num += 1
            total_players = len(remaining_forwards) + len(remaining_defense)
            num_teams = min(
                len(remaining_forwards) // (12 - len(forw)),
                len(remaining_defense) // (6 - len(defe))
            )

            if verbose:
                print(f"\n=== Round {round_num} | {num_teams} teams | {total_players} players remaining ===")
                print(f"Crosby in Pool: {(self.year * 1e10 + 8471675) in remaining_forwards}")

            self.rng.shuffle(remaining_forwards)
            self.rng.shuffle(remaining_defense)

            teams = []
            for i in range(num_teams):
                f_start = i * (12 - len(forw))
                d_start = i * (6 - len(defe))
                if (f_start + (12 - len(forw)) > len(remaining_forwards) or
                        d_start + (6 - len(defe)) > len(remaining_defense)):
                    break
                teamA = forw + remaining_forwards[f_start:f_start + 12 - len(forw)]
                teamB = defe + remaining_defense[d_start:d_start + 6 - len(defe)]
                teams.append((teamA, teamB))

            teams_formed = len(teams)
            kept_f = remaining_forwards[teams_formed * (12 - len(forw)):]
            kept_d = remaining_defense[teams_formed * (6 - len(defe)):]

            if len(teams) < 2:
                break

            results = []
            for i, (teamA, teamB) in enumerate(teams):
                score, usage, order = self.evaluate(teamA, teamB)
                usage_df = pd.DataFrame({'usage': usage.tolist(), 'order': order.tolist()})
                results.append((score, usage_df, teamA, teamB))
                if verbose:
                    print(f"Team {i + 1:2d}: score={score:.4f}")

            results.sort(key=lambda x: x[0], reverse=True)
            if (len(remaining_forwards) + len(remaining_defense)) > 100:
                top_half = results[:len(results) // 3]
                bottom_half = results[len(results) // 3:]
            else:
                top_half = results[:len(results) // 2]
                bottom_half = results[len(results) // 2:]

            survivors_f, survivors_d = [], []
            for _, _, teamA, teamB in top_half:
                survivors_f += teamA
                survivors_d += teamB

            for score, usage_df, teamA, teamB in bottom_half:
                f_usage = usage_df[usage_df['order'].isin(teamA)].sort_values(by='usage', ascending=False)
                d_usage = usage_df[usage_df['order'].isin(teamB)].sort_values(by='usage', ascending=False)

                keep_f = (f_usage['order'].iloc[:-4]
                          if (len(f_usage) > 4 and len(remaining_forwards) > 24 - len(forw))
                          else f_usage['order'])
                keep_d = (d_usage['order'].iloc[:-2]
                          if (len(d_usage) > 2 and len(remaining_defense) > 12 - len(defe))
                          else d_usage['order'])

                # FIX: don't append empty lists — only append entries that are actually removed
                if len(f_usage) > 4 and len(remaining_forwards) > 24 - len(forw):
                    removed += [[f, round_num] for f in f_usage['order'].iloc[-4:]]
                if len(d_usage) > 2 and len(remaining_defense) > 12 - len(defe):
                    removed += [[f, round_num] for f in d_usage['order'].iloc[-2:]]

                survivors_f += keep_f.tolist()
                survivors_d += keep_d.tolist()

            survivors_f += kept_f
            survivors_d += kept_d

            combined = survivors_f + survivors_d
            remaining_forwards = [p for p in combined if p in forwards]
            remaining_defense = [p for p in combined if p in defense]

            if verbose:
                print(f"After round {round_num}: {len(remaining_forwards)}F + {len(remaining_defense)}D = "
                      f"{len(remaining_forwards) + len(remaining_defense)} players")

        final_players = remaining_forwards + remaining_defense + forw + defe
        if verbose:
            print(f"\nTournament complete — {len(final_players)} players remain.")
        return final_players, removed

    def run_greedy(self, lineup, locked_forw, locked_defe, all_forwards, all_defense, verbose=True, max_iters=10):
        """
        Greedy 1-player swap refinement. Takes the tournament result as starting point
        and iteratively swaps players if it improves the team score.
        """
        # Tournament may return more than 18 — trim to exact roster size
        n_forw = 12 - len(locked_forw)
        n_defe = 6 - len(locked_defe)
        lineupA = locked_forw + [p for p in lineup if p in all_forwards and p not in locked_forw][:n_forw]
        lineupB = locked_defe + [p for p in lineup if p in all_defense and p not in locked_defe][:n_defe]

        # Pool = all eligible players NOT in current lineup (respecting position)
        pool_forw = [p for p in all_forwards if p not in lineupA]
        pool_defe = [p for p in all_defense if p not in lineupB]

        best_score, _, _ = self.evaluate(lineupA, lineupB)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Greedy refinement — starting score: {best_score:.4f}")
            print(f"Pool: {len(pool_forw)}F + {len(pool_defe)}D available for swaps")
            print('='*60)

        for iteration in range(1, max_iters + 1):
            # Build all swap candidates (batch across slots & positions)
            cand_lineups = []
            cand_meta = []
            for i, p_out in enumerate(lineupA):
                if p_out in locked_forw:
                    continue
                for p_in in pool_forw:
                    cand_lineups.append((lineupA[:i] + [p_in] + lineupA[i+1:], lineupB))
                    cand_meta.append(('F', i, p_out, p_in))
            for i, p_out in enumerate(lineupB):
                if p_out in locked_defe:
                    continue
                for p_in in pool_defe:
                    cand_lineups.append((lineupA, lineupB[:i] + [p_in] + lineupB[i+1:]))
                    cand_meta.append(('D', i, p_out, p_in))

            if not cand_lineups:
                break
            if verbose:
                print(f"  [iter {iteration}/{max_iters}] evaluating {len(cand_lineups)} candidates...", flush=True)

            scores = self.evaluate_batch(cand_lineups)
            best_idx = int(scores.argmax().item())
            best_swap_score = float(scores[best_idx].item())

            if best_swap_score <= best_score:
                if verbose:
                    print(f"  [iter {iteration}] no improvement ({best_swap_score:.4f} vs {best_score:.4f}), converged", flush=True)
                break

            pos, slot, p_out, p_in = cand_meta[best_idx]
            if pos == 'F':
                lineupA = lineupA[:slot] + [p_in] + lineupA[slot+1:]
                pool_forw = [p for p in pool_forw if p != p_in] + [p_out]
            else:
                lineupB = lineupB[:slot] + [p_in] + lineupB[slot+1:]
                pool_defe = [p for p in pool_defe if p != p_in] + [p_out]
            best_score = best_swap_score

            if verbose:
                print(f"Iter {iteration}: swapped {pos} score={best_score:.4f} "
                      f"out={p_out} in={p_in}", flush=True)
            pd.DataFrame({'player_id2': lineupA + lineupB}).to_csv(
                f"Canada2{self.year}_greedy_iter{iteration}.csv", index=False
            )

        if verbose:
            print(f"\nGreedy complete — final score: {best_score:.4f}")
        return lineupA + lineupB, best_score


# Main loop
for year in [2025, 2024]:
    print(f"\n{'='*60}")
    print(f"Running tournament for year {year}")
    print('='*60)

    roster = pd.concat([
        pd.read_csv("data/headshots2.csv"),
        pd.read_csv("headshots22025.csv")
    ]).fillna(0)
    roster['player_id'] = roster['id'] + roster['year'] * 1e10

    with open('data/yearly_embeddings_v2.pkl', 'rb') as file:
        embeddings = pickle.load(file)
    with open('yearly_embeddings_v22025.pkl', 'rb') as file:
        embeddings25 = pickle.load(file)
    embeddings25['year'] = 2025
    embeddings25[[col for col in embeddings25.columns if col not in ['year', 'player_id']]] *= 82 / 54

    for col in embeddings.columns:
        if col not in ['year', 'player_id']:
            embeddings[col] = embeddings[col].astype(float)
            if col in embeddings25.columns:
                embeddings25[col] = embeddings25[col].astype(float)
                embeddings25.loc[:, col] -= embeddings[col].mean()
                embeddings25.loc[:, col] /= embeddings[col].std()
            embeddings.loc[:, col] -= embeddings[col].mean()
            embeddings.loc[:, col] /= embeddings[col].std()

    embeddings = pd.concat([embeddings, embeddings25]).fillna(0)

    usa_ids = [20240008479314, 20240008480801, 20240008479318, 20240008481559,
               20240008478403, 20240008477404, 20240008478398, 20240008475184,
               20240008477946, 20240008476468, 20240008475754, 20240008476389,
               20240008479325, 20240008476958, 20240008479323,
               20240008478460, 20240008478396, 20240008482105]

    roster_year = roster[roster['birthCountry'].isin(['CAN']) & (roster['year'] == year)]
    us_roster = embeddings[(embeddings['player_id'] + embeddings['year'] * 1e10).isin(usa_ids)].copy()

    embeddings_yr = embeddings[embeddings['year'].isin([year, year - 1, year - 2])]
    embeddings_yr = embeddings_yr.groupby(['player_id'])[embeddings_yr.columns[:]].mean().reset_index(drop=True)
    embeddings_yr['year'] = year
    embeddings_yr['player_id2'] = embeddings_yr['player_id'] + embeddings_yr['year'] * 1e10
    embeddings_yr['positionCode'] = embeddings_yr['player_id2'].map(
        dict(zip(roster['id'] + roster['year'] * 1e10, roster['positionCode']))
    )

    embeddings2 = embeddings_yr.copy()
    embeddings_yr = embeddings_yr[embeddings_yr['player_id2'].isin(roster_year['player_id'].unique().tolist())]

    poolB = embeddings_yr[embeddings_yr['player_id2'].isin(
        roster_year[roster_year['positionCode'] == "D"]['player_id'].unique().tolist()
    )]
    poolA = embeddings_yr[embeddings_yr['player_id2'].isin(
        roster_year[~roster_year['positionCode'].isin(['D', 'G'])]['player_id'].unique().tolist()
    )]

    us_roster['player_id2'] = us_roster['player_id'] + us_roster['year'] * 1e10
    us_roster['positionCode'] = us_roster['player_id2'].map(
        dict(zip(roster['id'] + roster['year'] * 1e10, roster['positionCode']))
    )
    embeddings2 = pd.concat([embeddings2, us_roster])

    solver = TabuSearchLineup(
        embeddings=embeddings_yr,
        embeddings2=embeddings2,
        poolA=poolA,
        poolB=poolB,
        sizeA=12,
        sizeB=6,
        tabu_size=50,
        candidates_per_swap=4,
        num_low_usage=2,
        seed=42,
        year=year
    )

    locked_forw = [f + year * 1e10 for f in [8478402, 8477492, 8477933, 8471675]]
    locked_defe = [d + year * 1e10 for d in [8480069, 8477447]]

    best_lineup, remainder = solver.run_tournament(
        roster_year,
        verbose=True,
        forw=locked_forw,
        defe=locked_defe
    )

    # Save tournament-only result as a checkpoint so we have SOMETHING if greedy times out
    roster_year[roster_year['player_id'].isin(best_lineup)].sort_values(by='positionCode').to_csv(
        f"Canada2{year}_tournament.csv", index=False
    )
    print(f"Tournament checkpoint saved to Canada2{year}_tournament.csv", flush=True)

    # Greedy refinement from tournament result
    all_forwards = poolA['player_id2'].tolist()
    all_defense = poolB['player_id2'].tolist()
    best_lineup, _ = solver.run_greedy(
        best_lineup,
        locked_forw=locked_forw,
        locked_defe=locked_defe,
        all_forwards=all_forwards,
        all_defense=all_defense,
        verbose=True
    )

    print("Best lineup:", best_lineup)
    roster_year[roster_year['player_id'].isin(best_lineup)].sort_values(by='positionCode').to_csv(
        f"Canada2{year}.csv", index=False
    )

    # ── Individual player WAR scores ─────────────────────────────────────────
    final_forw = [p for p in best_lineup if p in set(all_forwards)]
    final_defe = [p for p in best_lineup if p in set(all_defense)]
    team_score, player_wars, player_ids = solver.evaluate(final_forw, final_defe)

    name_map = dict(zip(
        roster['player_id'],
        roster['firstName'].str.extract(r"'default': '([^']+)'")[0].fillna('') + ' ' +
        roster['lastName'].str.extract(r"'default': '([^']+)'")[0].fillna('')
    ))
    pos_map  = dict(zip(roster['player_id'], roster['positionCode']))
    team_map = dict(zip(roster['player_id'], roster['team']))

    war_df = pd.DataFrame({
        'player_id':  list(player_ids),
        'war':        player_wars.tolist(),
        'team_score': team_score,
    })
    war_df['name']         = war_df['player_id'].map(name_map)
    war_df['positionCode'] = war_df['player_id'].map(pos_map)
    war_df['team']         = war_df['player_id'].map(team_map)
    war_df.sort_values('war', ascending=False).to_csv(
        f'Canada_Individual{year}.csv', index=False
    )

    removed_df = pd.DataFrame(remainder, columns=['pid', 'round']).sort_values(by='round')
    removed_df['name'] = removed_df['pid'].map(
        dict(zip(roster['player_id'], roster['firstName'] + roster['lastName']))
    )
    removed_df.to_csv(f'Canada_Removed{year}.csv', index=False)
    print(f"Results saved to Canada2{year}.csv, Canada_Individual{year}.csv, Canada_Removed{year}.csv")
