"""Case-study definitions: eligibility pools, actual rosters, opponents, locks."""
import unicodedata

import pandas as pd


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


# ── NBA: Paris 2024 (features from 2023-24, season label 2024) ──
PARIS = {
    "USA": ["LeBron James", "Stephen Curry", "Kevin Durant", "Joel Embiid", "Jayson Tatum",
            "Devin Booker", "Anthony Edwards", "Anthony Davis", "Bam Adebayo", "Jrue Holiday",
            "Tyrese Haliburton", "Derrick White"],
    "Canada": ["Shai Gilgeous-Alexander", "Jamal Murray", "Nickeil Alexander-Walker",
               "Andrew Nembhard", "RJ Barrett", "Luguentz Dort", "Dillon Brooks", "Melvin Ejim",
               "Trey Lyles", "Kelly Olynyk", "Dwight Powell", "Khem Birch"],
}
NBA_LOCKS = {"USA": ["Stephen Curry", "LeBron James", "Kevin Durant"],
             "Canada": ["Shai Gilgeous-Alexander", "Jamal Murray"]}
NBA_OPP = {"USA": ["Canada", "France", "OKC"], "Canada": ["USA", "France", "OKC"]}
NBA_COUNTRY_OVERRIDE = {"Joel Embiid": "USA"}  # represented USA; listed country Cameroon

# ── NHL: 2025 4 Nations Face-Off (features from 2024-25 games before 2025-02-12) ──
FOUR_NATIONS = {
    "CAN": {"F": ["Sam Bennett", "Anthony Cirelli", "Sidney Crosby", "Brandon Hagel", "Seth Jarvis",
                  "Travis Konecny", "Nathan MacKinnon", "Brad Marchand", "Mitch Marner",
                  "Connor McDavid", "Brayden Point", "Sam Reinhart", "Mark Stone"],
            "D": ["Drew Doughty", "Cale Makar", "Josh Morrissey", "Colton Parayko", "Travis Sanheim",
                  "Shea Theodore", "Devon Toews", "Thomas Harley"]},
    "USA": {"F": ["Matt Boldy", "Kyle Connor", "Jack Eichel", "Jake Guentzel", "Jack Hughes",
                  "Chris Kreider", "Dylan Larkin", "Auston Matthews", "J.T. Miller", "Brock Nelson",
                  "Brady Tkachuk", "Matthew Tkachuk", "Vincent Trocheck"],
            "D": ["Brock Faber", "Adam Fox", "Noah Hanifin", "Quinn Hughes", "Jake Sanderson",
                  "Jaccob Slavin", "Zach Werenski", "Charlie McAvoy"]},
    "SWE": {"F": ["Viktor Arvidsson", "Jesper Bratt", "Leo Carlsson", "Joel Eriksson Ek",
                  "Filip Forsberg", "Adrian Kempe", "Elias Lindholm", "William Nylander",
                  "Gustav Nyquist", "Elias Pettersson", "Lucas Raymond", "Rickard Rakell",
                  "Mika Zibanejad"],
            "D": ["Rasmus Andersson", "Jonas Brodin", "Rasmus Dahlin", "Mattias Ekholm",
                  "Gustav Forsling", "Victor Hedman", "Erik Karlsson"]},
    "FIN": {"F": ["Sebastian Aho", "Joel Armia", "Aleksander Barkov", "Mikael Granlund", "Erik Haula",
                  "Roope Hintz", "Kaapo Kakko", "Patrik Laine", "Artturi Lehkonen", "Anton Lundell",
                  "Eetu Luostarinen", "Mikko Rantanen", "Teuvo Teravainen"],
            "D": ["Henri Jokiharju", "Esa Lindell", "Olli Maatta", "Niko Mikkola", "Nikolas Matinpalo",
                  "Urho Vaakanainen", "Juuso Valimaki"]},
}
NHL_LOCKS = {"CAN": ["Connor McDavid", "Nathan MacKinnon", "Sidney Crosby", "Cale Makar"]}
NHL_OPP = {"CAN": ["USA", "SWE", "FIN"]}


def name_index(ps, cols):
    idx = {}
    for c in cols:
        for pid, n in zip(ps["player_id"], ps[c]):
            if isinstance(n, str):
                idx.setdefault(norm(n), int(pid))
    return idx


def resolve(names, idx, strict=True):
    out, missing = [], []
    for n in names:
        k = norm(n)
        if k in idx:
            out.append(idx[k])
        else:
            missing.append(n)
    if missing and strict:
        print("  unresolved:", missing)
    return out, missing


def by_minutes(ps, season, ids, k, key):
    cur = ps[(ps["season"] == season) & ps["player_id"].isin(ids)]
    return cur.sort_values(key, ascending=False)["player_id"].astype(int).tolist()[:k]
