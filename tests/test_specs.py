from __future__ import annotations

from pathlib import Path

from groupwar.data import load_players_csv, prepare_player_pool
from groupwar.specs import NBA, NHL


ROOT = Path(__file__).resolve().parents[1]


def test_nhl_positions_are_grouped() -> None:
    players = load_players_csv(ROOT / "examples" / "nhl_players.csv")
    prepared = prepare_player_pool(NHL, players)
    assert prepared["lineup_group"].value_counts().to_dict() == {"forwards": 15, "defense": 8}


def test_nba_hybrid_positions_are_grouped() -> None:
    players = load_players_csv(ROOT / "examples" / "nba_players.csv")
    prepared = prepare_player_pool(NBA, players)
    assert prepared["lineup_group"].value_counts().to_dict() == {"guards": 5, "wings": 5, "bigs": 5}

