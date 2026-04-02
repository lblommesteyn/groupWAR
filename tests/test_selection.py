from __future__ import annotations

from pathlib import Path

from groupwar.data import load_players_csv
from groupwar.scoring import WeightedColumnScorer
from groupwar.selection import LineupOptimizer
from groupwar.specs import NBA, NHL


ROOT = Path(__file__).resolve().parents[1]


def test_tabu_returns_top_ranked_nhl_groups() -> None:
    players = load_players_csv(ROOT / "examples" / "nhl_players.csv")
    scorer = WeightedColumnScorer(players, id_column="player_id", score_column="war")
    optimizer = LineupOptimizer(NHL, players, scorer, ranking_column="war", seed=7)
    result = optimizer.run_tabu(max_iters=15)

    selected = result.to_frame(players, id_column="player_id")
    forwards = selected[selected["lineup_group"] == "forwards"]["player_id"].tolist()
    defense = selected[selected["lineup_group"] == "defense"]["player_id"].tolist()

    expected_forwards = (
        players[~players["positionCode"].isin(["D", "LD", "RD"])]
        .sort_values("war", ascending=False)
        .head(12)["player_id"]
        .tolist()
    )
    expected_defense = players[players["positionCode"].isin(["D", "LD", "RD"])].sort_values("war", ascending=False).head(6)["player_id"].tolist()

    assert forwards == expected_forwards
    assert defense == expected_defense
    assert result.score is not None


def test_tournament_builds_nba_rotation() -> None:
    players = load_players_csv(ROOT / "examples" / "nba_players.csv")
    scorer = WeightedColumnScorer(players, id_column="player_id", score_column="impact_score")
    optimizer = LineupOptimizer(NBA, players, scorer, ranking_column="impact_score", seed=11)
    result = optimizer.run_tournament(max_rounds=10)

    assert len(result.lineup["guards"]) == 4
    assert len(result.lineup["wings"]) == 4
    assert len(result.lineup["bigs"]) == 4
    assert result.score is not None
