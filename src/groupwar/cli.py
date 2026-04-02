from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_players_csv, load_synergy_csv, prepare_player_pool
from .scoring import WeightedColumnScorer
from .selection import LineupOptimizer
from .specs import LEAGUES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="groupwar")
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser("optimize", help="optimize a league-specific lineup")
    optimize.add_argument("--league", required=True, choices=sorted(LEAGUES))
    optimize.add_argument("--players", required=True, help="CSV containing player rows")
    optimize.add_argument("--output", required=True, help="output CSV for selected players")
    optimize.add_argument("--score-column", help="metric column used by the weighted scorer")
    optimize.add_argument("--strategy", choices=("tabu", "tournament"), default="tournament")
    optimize.add_argument("--player-id-column", help="override the league default id column")
    optimize.add_argument("--position-column", help="override the league default position column")
    optimize.add_argument("--synergy", help="optional CSV with columns player_id_a, player_id_b, bonus")
    optimize.add_argument("--locked", help="optional JSON mapping group names to locked player ids")
    optimize.add_argument("--feature-columns", help="comma-separated feature columns for swap ranking")
    optimize.add_argument("--seed", type=int, default=123)
    optimize.add_argument("--max-iters", type=int, default=100)
    return parser


def _load_locked(path: str | None) -> dict[str, list[object]]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    spec = LEAGUES[args.league].with_overrides(
        player_id_column=args.player_id_column,
        position_column=args.position_column,
    )
    players = load_players_csv(args.players)
    players = prepare_player_pool(spec, players)

    score_column = args.score_column or spec.default_score_column
    synergy = load_synergy_csv(args.synergy) if args.synergy else None
    scorer = WeightedColumnScorer(
        players,
        id_column=spec.player_id_column,
        score_column=score_column,
        synergy_frame=synergy,
    )
    feature_columns = args.feature_columns.split(",") if args.feature_columns else None
    optimizer = LineupOptimizer(
        spec,
        players,
        scorer,
        feature_columns=feature_columns,
        seed=args.seed,
        ranking_column=score_column,
    )
    locked = _load_locked(args.locked)

    if args.strategy == "tabu":
        result = optimizer.run_tabu(max_iters=args.max_iters, initial_lineup=optimizer.initial_lineup(locked=locked))
    else:
        result = optimizer.run_tournament(locked=locked, max_rounds=args.max_iters)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_frame(players, id_column=spec.player_id_column).to_csv(output_path, index=False)

    print(f"league={spec.code}")
    print(f"score={result.score}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
