"""
Patch nba_headshots.csv + nba_embeddings.pkl to include Olympic stars missing
from common_player_info. Their position flags need to be set manually (common_player_info
is the source of those flags in the original pipeline — so players missing from it
end up with isG/isF/isC = 0).
"""

import sqlite3
import pickle
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"

# (country, isG, isF, isC) per missing star.
MISSING = {
    2544:    ("USA",    0, 1, 0),   # LeBron James - F
    1626164: ("USA",    1, 0, 0),   # Devin Booker - G
    201950:  ("USA",    1, 0, 0),   # Jrue Holiday - G
    203954:  ("USA",    0, 0, 1),   # Joel Embiid - C
    1628389: ("USA",    0, 1, 1),   # Bam Adebayo - F/C
    202695:  ("USA",    0, 1, 0),   # Kawhi Leonard - F
    1627750: ("Canada", 1, 0, 0),   # Jamal Murray - G
    1628983: ("Canada", 1, 0, 0),   # SGA - G
    1628415: ("Canada", 0, 1, 0),   # Dillon Brooks - F
    1629652: ("Canada", 1, 0, 0),   # Luguentz Dort - G
    1629638: ("Canada", 1, 0, 0),   # Nickeil Alexander-Walker - G
    1629076: ("Canada", 0, 1, 0),   # Oshae Brissett - F
}


def main():
    hs = pd.read_csv(DATA / "nba_headshots.csv")
    print(f"Existing headshots: {len(hs)} rows, {hs['PERSON_ID'].nunique()} players")

    # Names from `player` table
    con = sqlite3.connect(DATA / "nba.sqlite")
    cur = con.cursor()
    names = {}
    for pid in MISSING:
        cur.execute("SELECT first_name, last_name FROM player WHERE id = ?", (pid,))
        r = cur.fetchone()
        if r:
            names[pid] = (r[0], r[1])
    con.close()

    # Drop any stale rows for these players (may exist with wrong flags)
    hs = hs[~hs["PERSON_ID"].isin(MISSING.keys())].copy()
    print(f"After dropping stale star rows: {len(hs)} rows")

    seasons = ["2021-22", "2022-23"]
    new_rows = []
    for pid, (country, isG, isF, isC) in MISSING.items():
        if pid not in names:
            print(f"  skip {pid}: no player record")
            continue
        fn, ln = names[pid]
        pos = ("Guard" if isG else
               "Forward-Center" if (isF and isC) else
               "Forward" if isF else "Center")
        for season in seasons:
            new_rows.append({
                "PERSON_ID": pid,
                "PLAYER_FIRST_NAME": fn,
                "PLAYER_LAST_NAME": ln,
                "birthCountry": country,
                "height": "",
                "weight": 0,
                "POSITION": pos,
                "from_year": 2017.0,
                "to_year": 2023.0,
                "isG": isG,
                "isF": isF,
                "isC": isC,
                "season": season,
            })

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([hs, new_df], ignore_index=True)
    combined.to_csv(DATA / "nba_headshots.csv", index=False)
    print(f"Appended {len(new_rows)} rows ({len(MISSING)} players x {len(seasons)} seasons).")
    print(f"Final headshots: {len(combined)} rows, {combined['PERSON_ID'].nunique()} players.")

    # Patch embeddings — isG/isF/isC are features the GCN reads
    with open(DATA / "nba_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    patched = 0
    for pid, (_, isG, isF, isC) in MISSING.items():
        mask = emb["player_id"] == pid
        if mask.sum() == 0:
            continue
        emb.loc[mask, "isG"] = isG
        emb.loc[mask, "isF"] = isF
        emb.loc[mask, "isC"] = isC
        patched += 1
    with open(DATA / "nba_embeddings.pkl", "wb") as f:
        pickle.dump(emb, f)
    print(f"Patched position flags for {patched} players in nba_embeddings.pkl.")


if __name__ == "__main__":
    main()
