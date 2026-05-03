"""
NBA Olympic Roster Search — V2
Changes from v1:
  - NODE_IN_DIM = 20 (v2 embeddings)
  - Opponents updated: USA vs Canada/France/OKC, Canada vs USA/France/OKC
  - Loads v2 models from models/v2/
  - Holiday (201950) removed from locked (selected by merit)
  - Davis (203076) added to locked for USA
"""

import numpy as np
import pandas as pd
import random
import pickle
from pathlib import Path
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nba_stackel import stackelberg, GCN_DeepSet_AntiSym_Invariant, N_PLAYERS, N_HALF
from nba_search import NBATabuSearchLineup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent / "data"
MODELS_DIR = _HERE.parent / "models" / "v2"

# ── V2: Updated opponents ──
# USA evaluates against Canada, France, OKC
# Canada evaluates against USA, France, OKC
OPPONENTS_V2 = {
    "USA":    ["Canada", "France", "OKC"],
    "Canada": ["USA", "France", "OKC"],
}

# ── V2: Updated locked players ──
# Holiday removed from locked (was 201950)
# Davis added (203076)
LOCKED_V2 = {
    2024: {
        "USA":    {"G": [201939],                    # Curry
                   "F": [2544, 201142, 203076]},     # LeBron, Durant, Davis
        "Canada": {"G": [1628983, 1627750],          # SGA, Murray
                   "F": []},
    }
}

# ── Load V2 models (20-dim input) ──
modells_v2 = []
for i in range(6):
    model = GCN_DeepSet_AntiSym_Invariant(
        node_in_dim=20,    # V2: 20 features
        gcn_hidden=128, gcn_layers=6,
        deepset_phi_dim=128, deepset_rho_dim=128,
        n_nodes=N_PLAYERS, vector_size=128, n_half=N_HALF,
        dropout=5e-2
    )
    ckpt = MODELS_DIR / f"nba_model_v2_{i}_30.pth"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
    else:
        print(f"  Warning: {ckpt} not found, using random init")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    modells_v2.append(model.to(device))
print("V2 models loaded.")


class NBASearchV2(NBATabuSearchLineup):
    """V2 search: updated opponents and v2 models."""

    def _get_fiba_roster(self, team, year):
        """
        V2: Handle national team lookups (Canada, USA, France) by filtering
        headshots by birthCountry, plus NBA club lookups (OKC) by team abbrev.
        """
        country_map = {
            "Canada": "Canada",
            "USA": "USA",
            "France": "France",
        }

        if team in country_map:
            # National team: get top 12 players from that country
            headshots = pd.read_csv(DATA_DIR / "nba_headshots.csv")
            country_players = headshots[
                headshots["birthCountry"] == country_map[team]
            ]["PERSON_ID"].tolist()

            # Get their APM, return top 10
            apm_df = self.apm[self.apm["player_id"].isin(country_players)]
            return apm_df.nlargest(10, "APM")["player_id"].tolist()
        else:
            # NBA club team (OKC, etc.)
            return super()._get_fiba_roster(team, year)


if __name__ == "__main__":
    with open(DATA_DIR / "nba_embeddings_v2.pkl", "rb") as f:
        embeddings = pickle.load(f)

    headshots = pd.read_csv(DATA_DIR / "nba_headshots.csv")
    embeddings["player_id2"] = embeddings["player_id"] + embeddings["year"] * 1e10

    # Monkey-patch opponents and models for v2
    import nba_search
    nba_search.OPPONENTS = OPPONENTS_V2
    nba_search.modells = modells_v2

    for year in [2024]:
        for country, birth_country in [("USA", "USA"), ("Canada", "Canada")]:
            print(f"\n{'='*60}")
            print(f"V2 Search: {country} {year}")
            print('='*60)

            target_season = f"{year-1}-{str(year)[2:]}"
            available_seasons = headshots["season"].unique().tolist()
            season = target_season if target_season in available_seasons else sorted(available_seasons)[-1]

            roster = headshots[
                (headshots["birthCountry"] == birth_country) &
                (headshots["season"] == season)
            ].copy()
            roster["player_id2"] = roster["PERSON_ID"] + year * 1e10

            avail_years = embeddings["year"].unique().tolist()
            emb_year = year if year in avail_years else max(avail_years)
            emb = embeddings[embeddings["year"] == emb_year].copy()
            emb["year"] = year
            emb["player_id2"] = emb["player_id"] + year * 1e10
            emb["isG"] = emb["player_id2"].map(
                dict(zip(roster["player_id2"], roster["isG"]))
            )
            emb = emb[emb["player_id2"].isin(roster["player_id2"])]

            locked_g = [pid + year * 1e10 for pid in LOCKED_V2[year][country]["G"]]
            locked_f = [pid + year * 1e10 for pid in LOCKED_V2[year][country]["F"]]

            emb_active = emb[
                (emb["games_played"] > -1) |
                emb["player_id2"].isin(locked_g + locked_f)
            ]
            pool_g = emb_active[emb_active["isG"] == 1]
            pool_f = emb_active[emb_active["isG"] == 0]
            print(f"Pool: {len(pool_g)}G + {len(pool_f)}F")

            solver = NBASearchV2(
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

            name_map = dict(zip(
                roster["player_id2"],
                roster["PLAYER_FIRST_NAME"] + " " + roster["PLAYER_LAST_NAME"]
            ))

            pd.DataFrame({
                "player_id2": best_lineup,
                "name": [name_map.get(p, "") for p in best_lineup],
            }).to_csv(_HERE / f"outputs/NBA_v2_{country}_{year}_tournament.csv", index=False)

            all_guards = pool_g["player_id2"].tolist()
            all_forwards = pool_f["player_id2"].tolist()
            best_lineup, _ = solver.run_greedy(
                best_lineup, locked_g, locked_f,
                all_guards, all_forwards, verbose=True
            )

            result_df = pd.DataFrame({"player_id2": best_lineup})
            result_df["name"] = result_df["player_id2"].map(name_map)
            result_df.to_csv(_HERE / f"outputs/NBA_v2_{country}_{year}.csv", index=False)
            print(f"\nSaved to outputs/NBA_v2_{country}_{year}.csv")
            print(result_df["name"].tolist())
