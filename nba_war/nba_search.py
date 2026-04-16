"""
NBA Olympic Roster Search — Phase 4
Tournament + greedy search for optimal 12-player FIBA rosters.
Mirrors run_search.py from hockey, adapted for basketball.

Roster: 12 players (5 starters)
Positions: Guards (G) / Forwards+Centers (F+C)
"""

import numpy as np
import pandas as pd
import random
import pickle
from collections import deque
from pathlib import Path
import torch

from nba_stackel import stackelberg, GCN_DeepSet_AntiSym_Invariant, N_PLAYERS, N_HALF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
DATA_DIR = Path("data")
MODELS_DIR = Path("models")

# ── FIBA opponent teams (top teams at 2024 Paris Olympics) ──
OPPONENTS = {
    "USA":    ["Serbia", "France", "Australia"],
    "Canada": ["USA",    "Serbia", "Australia"],
}

# ── Locked-in stars ──
LOCKED = {
    2024: {
        "USA":    {"G": [203999, 201939],        # Curry, Holiday
                   "F": [2544,   201142, 203954]}, # LeBron, Durant, AD
        "Canada": {"G": [1629029, 203500],        # SGA, Murray
                   "F": []},
    }
}

# ── Load models ──
modells = []
for i in range(6):
    model = GCN_DeepSet_AntiSym_Invariant(
        node_in_dim=13,    # 13 feature cols: FGA..APM (excl. player_id, year)
        gcn_hidden=128, gcn_layers=6,
        deepset_phi_dim=128, deepset_rho_dim=128,
        n_nodes=N_PLAYERS, vector_size=128, n_half=N_HALF,
        dropout=5e-2
    )
    ckpt = MODELS_DIR / f"nba_model_{i}_30.pth"
    state = torch.load(ckpt, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    modells.append(model.to(device))
print("Models loaded.")


class NBATabuSearchLineup:
    def __init__(self, embeddings, embeddings_opp, pool_g, pool_f,
                 country, year, seed=42):
        self.embeddings = embeddings        # home country players
        self.embeddings_opp = embeddings_opp  # all players (for opponent lookups)
        self.pool_g = pool_g["player_id2"].unique()
        self.pool_f = pool_f["player_id2"].unique()
        self.country = country
        self.year = year
        self.rng = random.Random(seed)
        self._away_cache = {}

        self.apm = pd.read_csv(DATA_DIR / "nba_apm.csv")

    def evaluate(self, lineupG, lineupF):
        """
        Evaluate a 12-player roster (guards + forwards/centers) against
        3 FIBA opponents. Returns (mean_score, per_player_war, player_ids).
        """
        opponent_teams = OPPONENTS[self.country]
        year = self.year

        homex = self.embeddings[
            self.embeddings["player_id2"].isin(lineupG + lineupF)
        ].drop_duplicates()

        players = homex["player_id2"].values
        P = len(players)
        zero_masks = torch.zeros(P, N_PLAYERS, N_PLAYERS, dtype=torch.int32)

        # Build home tensor
        home_np = homex.iloc[:, 2:-1].to_numpy()
        home_base_raw = torch.tensor(home_np, dtype=torch.float32)

        hometeams = []
        all_scores = []

        for awayteam in opponent_teams:
            cache_key = (awayteam, year)
            if cache_key not in self._away_cache:
                # Get top 12 players for this national team
                away_ids = self._get_fiba_roster(awayteam, year)
                awayx = (
                    self.embeddings_opp[self.embeddings_opp["player_id2"].isin(away_ids)]
                    .drop_duplicates(subset=["player_id2"])
                    .reset_index(drop=True)
                ).copy().iloc[:, :-1]

                # Pad to 5 (we only use 5 starters in GCN evaluation)
                if len(awayx) < N_HALF:
                    rows = [[0] * len(awayx.columns) for _ in range(N_HALF - len(awayx))]
                    awayx = pd.concat([awayx, pd.DataFrame(rows, columns=awayx.columns)])
                awayx = awayx.iloc[:N_HALF]

                # Align columns via concat
                combined = pd.concat([homex.iloc[:N_HALF], awayx])
                combined_np = combined.iloc[:, 2:-1].fillna(0).to_numpy()
                combined_t = torch.tensor(combined_np, dtype=torch.float32)
                home_t = combined_t[:N_HALF]
                away_t = combined_t[N_HALF:]
                self._away_cache[cache_key] = (home_t, away_t)

            home_5, away_tensor = self._away_cache[cache_key]

            # Baseline: best starting 5 from home lineup vs away starting 5
            rep_base = torch.cat([home_5, away_tensor], dim=0).unsqueeze(0)
            main, _, _, _ = stackelberg(
                rep_base, modells, torch.zeros((1, N_PLAYERS, N_PLAYERS), dtype=torch.int32)
            )

            # Player knockouts — replace each player with zeros
            home_batch = home_5.unsqueeze(0).expand(P, -1, -1).clone()
            # For knockout: if player is in starting 5, zero them; else no change
            for i, pid in enumerate(players):
                if pid in homex["player_id2"].values[:N_HALF]:
                    idx_in_5 = list(homex["player_id2"].values[:N_HALF]).index(pid)
                    home_batch[i, idx_in_5, :] = -2.5

            away_expanded = away_tensor.unsqueeze(0).expand(P, -1, -1)
            reps = torch.cat([home_batch, away_expanded], dim=1)
            d1, _, _, _ = stackelberg(reps, modells, zero_masks)

            d1s = [(main - d1[i]) for i in range(P)]
            all_scores.append(d1s)
            hometeams.append(main)

        all_d1s = torch.stack([torch.stack(s) for s in all_scores]).mean(dim=0).squeeze(-1)
        return torch.stack(hometeams).mean().item(), all_d1s, homex["player_id2"].unique()

    def _get_fiba_roster(self, country, year):
        """Get top 12 players for a FIBA opponent by APM."""
        apm = self.apm.copy()
        headshots = pd.read_csv(DATA_DIR / "nba_headshots.csv")
        eligible = headshots[headshots["birthCountry"] == country]["PERSON_ID"].tolist()
        top = apm[apm["player_id"].isin(eligible)].nlargest(12, "APM")["player_id"].tolist()
        return top

    def run_tournament(self, roster, verbose=True, locked_g=[], locked_f=[]):
        """Tournament elimination — same structure as hockey."""
        guards = roster[roster["isG"] == 1]["player_id2"].tolist()
        forwards = roster[roster["isG"] == 0]["player_id2"].tolist()
        guards = [p for p in guards if p in self.pool_g and p not in locked_g]
        forwards = [p for p in forwards if p in self.pool_f and p not in locked_f]

        n_g = 4 - len(locked_g)   # need 4 guards total (minus locked)
        n_f = 8 - len(locked_f)   # need 8 forwards/centers total (minus locked)
        target = n_g + n_f

        remaining_g = guards[:]
        remaining_f = forwards[:]
        removed = []
        round_num = 0
        max_iters = 25

        while (len(remaining_g) + len(remaining_f)) > target and round_num < max_iters:
            round_num += 1
            num_teams = min(len(remaining_g) // n_g, len(remaining_f) // n_f)

            if verbose:
                print(f"\n=== Round {round_num} | {num_teams} teams | "
                      f"{len(remaining_g)}G + {len(remaining_f)}F remaining ===")

            self.rng.shuffle(remaining_g)
            self.rng.shuffle(remaining_f)

            teams = []
            for i in range(num_teams):
                tg = locked_g + remaining_g[i * n_g:(i + 1) * n_g]
                tf = locked_f + remaining_f[i * n_f:(i + 1) * n_f]
                if len(tg) + len(tf) == 12:
                    teams.append((tg, tf))

            if len(teams) < 2:
                break

            results = []
            for i, (tg, tf) in enumerate(teams):
                score, usage, order = self.evaluate(tg, tf)
                usage_df = pd.DataFrame({"usage": usage.tolist(), "order": order.tolist()})
                results.append((score, usage_df, tg, tf))
                if verbose:
                    print(f"Team {i+1:2d}: score={score:.4f}")

            results.sort(key=lambda x: x[0], reverse=True)

            if (len(remaining_g) + len(remaining_f)) > 60:
                top_half = results[:len(results) // 3]
                bottom_half = results[len(results) // 3:]
            else:
                top_half = results[:len(results) // 2]
                bottom_half = results[len(results) // 2:]

            surv_g, surv_f = [], []
            for _, _, tg, tf in top_half:
                surv_g += [p for p in tg if p not in locked_g]
                surv_f += [p for p in tf if p not in locked_f]

            for _, usage_df, tg, tf in bottom_half:
                g_usage = usage_df[usage_df["order"].isin(tg)].sort_values("usage", ascending=False)
                f_usage = usage_df[usage_df["order"].isin(tf)].sort_values("usage", ascending=False)

                if len(g_usage) > 2 and len(remaining_g) > n_g * 2:
                    removed += [[p, round_num] for p in g_usage["order"].iloc[-2:]]
                    surv_g += g_usage["order"].iloc[:-2].tolist()
                else:
                    surv_g += g_usage["order"].tolist()

                if len(f_usage) > 3 and len(remaining_f) > n_f * 2:
                    removed += [[p, round_num] for p in f_usage["order"].iloc[-3:]]
                    surv_f += f_usage["order"].iloc[:-3].tolist()
                else:
                    surv_f += f_usage["order"].tolist()

            # Add leftover teams not formed
            surv_g += remaining_g[len(teams) * n_g:]
            surv_f += remaining_f[len(teams) * n_f:]

            remaining_g = [p for p in surv_g if p in guards]
            remaining_f = [p for p in surv_f if p in forwards]

            if verbose:
                print(f"After round {round_num}: {len(remaining_g)}G + {len(remaining_f)}F remaining")

        final = locked_g + remaining_g[:n_g] + locked_f + remaining_f[:n_f]
        if verbose:
            print(f"\nTournament complete — {len(final)} players remain.")
        return final, removed

    def run_greedy(self, lineup, locked_g, locked_f, all_guards, all_forwards, verbose=True):
        """Greedy 1-player swap refinement from tournament result."""
        lineupG = [p for p in lineup if p in all_guards]
        lineupF = [p for p in lineup if p in all_forwards]
        pool_g = [p for p in all_guards if p not in lineupG]
        pool_f = [p for p in all_forwards if p not in lineupF]

        best_score, _, _ = self.evaluate(lineupG, lineupF)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Greedy refinement — starting score: {best_score:.4f}")
            print('='*60)

        improved = True
        iteration = 0
        while improved:
            improved = False
            iteration += 1
            best_swap = None

            for i, p_out in enumerate(lineupG):
                if p_out in locked_g:
                    continue
                for p_in in pool_g:
                    new_g = lineupG[:i] + [p_in] + lineupG[i+1:]
                    score, _, _ = self.evaluate(new_g, lineupF)
                    if score > best_score:
                        best_score = score
                        best_swap = ("G", i, p_out, p_in, new_g, lineupF)

            for i, p_out in enumerate(lineupF):
                if p_out in locked_f:
                    continue
                for p_in in pool_f:
                    new_f = lineupF[:i] + [p_in] + lineupF[i+1:]
                    score, _, _ = self.evaluate(lineupG, new_f)
                    if score > best_score:
                        best_score = score
                        best_swap = ("F", i, p_out, p_in, lineupG, new_f)

            if best_swap is not None:
                pos, idx, p_out, p_in, lineupG, lineupF = best_swap
                if pos == "G":
                    pool_g = [p for p in pool_g if p != p_in] + [p_out]
                else:
                    pool_f = [p for p in pool_f if p != p_in] + [p_out]
                improved = True
                if verbose:
                    print(f"Iter {iteration}: swapped {pos} score={best_score:.4f}  out={p_out}  in={p_in}")

        if verbose:
            print(f"\nGreedy complete — final score: {best_score:.4f}")
        return lineupG + lineupF, best_score


# ── Main ──
if __name__ == "__main__":
    with open(DATA_DIR / "nba_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    headshots = pd.read_csv(DATA_DIR / "nba_headshots.csv")

    # Add player_id2 to raw embeddings so opponent lookups work
    embeddings["player_id2"] = embeddings["player_id"] + embeddings["year"] * 1e10

    for year in [2024]:
        for country, birth_country in [("USA", "USA"), ("Canada", "Canada")]:
            print(f"\n{'='*60}")
            print(f"Running tournament for {country} {year}")
            print('='*60)

            # Use most recent available season if target season not in data
            target_season = f"{year-1}-{str(year)[2:]}"
            available_seasons = headshots["season"].unique().tolist()
            season = target_season if target_season in available_seasons else sorted(available_seasons)[-1]

            roster = headshots[
                (headshots["birthCountry"] == birth_country) &
                (headshots["season"] == season)
            ].copy()
            roster["player_id2"] = roster["PERSON_ID"] + year * 1e10

            # Use most recent available embedding year if target year not in data
            avail_years = embeddings["year"].unique().tolist()
            emb_year = year if year in avail_years else max(avail_years)
            emb = embeddings[embeddings["year"] == emb_year].copy()
            emb["year"] = year   # relabel so player_id2 construction is consistent
            emb["player_id2"] = emb["player_id"] + year * 1e10
            emb["isG"] = emb["player_id2"].map(
                dict(zip(roster["player_id2"], roster["isG"]))
            )
            emb = emb[emb["player_id2"].isin(roster["player_id2"])]

            # Filter to above-average contributors (games_played > 0 = above mean, z-scored)
            emb_active = emb[emb["games_played"] > 0]
            pool_g = emb_active[emb_active["isG"] == 1]
            pool_f = emb_active[emb_active["isG"] == 0]
            print(f"Pool after activity filter: {len(pool_g)}G + {len(pool_f)}F")

            locked_g = [pid + year * 1e10 for pid in LOCKED[year][country]["G"]]
            locked_f = [pid + year * 1e10 for pid in LOCKED[year][country]["F"]]

            solver = NBATabuSearchLineup(
                embeddings=emb,
                embeddings_opp=embeddings,
                pool_g=pool_g,
                pool_f=pool_f,
                country=country,
                year=year,
                seed=42
            )

            best_lineup, remainder = solver.run_tournament(
                roster, verbose=True,
                locked_g=locked_g, locked_f=locked_f
            )

            all_guards = pool_g["player_id2"].tolist()
            all_forwards = pool_f["player_id2"].tolist()
            best_lineup, _ = solver.run_greedy(
                best_lineup, locked_g, locked_f,
                all_guards, all_forwards, verbose=True
            )

            name_map = dict(zip(
                roster["player_id2"],
                roster["PLAYER_FIRST_NAME"] + " " + roster["PLAYER_LAST_NAME"]
            ))
            result_df = pd.DataFrame({"player_id2": best_lineup})
            result_df["name"] = result_df["player_id2"].map(name_map)
            result_df.to_csv(f"NBA_{country}_{year}.csv", index=False)
            print(f"\nSaved to NBA_{country}_{year}.csv")
            print(result_df["name"].tolist())

            removed_df = pd.DataFrame(remainder, columns=["pid", "round"])
            removed_df["name"] = removed_df["pid"].map(name_map)
            removed_df.to_csv(f"NBA_{country}_{year}_removed.csv", index=False)
