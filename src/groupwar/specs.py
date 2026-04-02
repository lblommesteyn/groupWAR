from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable

import pandas as pd


def _tokenize_position(raw_position: object) -> set[str]:
    text = str(raw_position or "").upper().strip()
    if not text:
        return set()
    tokens = {token for token in re.split(r"[^A-Z0-9]+", text) if token}
    tokens.add(text)
    return tokens


@dataclass(frozen=True)
class PositionGroup:
    name: str
    lineup_size: int
    positions: tuple[str, ...]
    elimination_size: int = 1

    def exact_match(self, raw_position: object) -> bool:
        text = str(raw_position or "").upper().strip()
        return text in {position.upper() for position in self.positions}

    def matches(self, raw_position: object) -> bool:
        tokens = _tokenize_position(raw_position)
        eligible = {position.upper() for position in self.positions}
        return bool(tokens & eligible)


@dataclass(frozen=True)
class LeagueSpec:
    code: str
    display_name: str
    groups: tuple[PositionGroup, ...]
    position_column: str
    player_id_column: str = "player_id"
    default_score_column: str = "score"
    feature_exclude: tuple[str, ...] = ("player_id", "player_name", "name", "lineup_group")

    @property
    def team_size(self) -> int:
        return sum(group.lineup_size for group in self.groups)

    def with_overrides(
        self,
        *,
        player_id_column: str | None = None,
        position_column: str | None = None,
    ) -> "LeagueSpec":
        return replace(
            self,
            player_id_column=player_id_column or self.player_id_column,
            position_column=position_column or self.position_column,
        )

    def group_for_position(self, raw_position: object) -> str | None:
        for group in self.groups:
            if group.exact_match(raw_position):
                return group.name
        for group in self.groups:
            if group.matches(raw_position):
                return group.name
        return None

    def annotate_groups(self, players: pd.DataFrame) -> pd.DataFrame:
        annotated = players.copy()
        annotated["lineup_group"] = annotated[self.position_column].map(self.group_for_position)
        return annotated

    def required_group_names(self) -> list[str]:
        return [group.name for group in self.groups]

    def empty_lineup(self) -> dict[str, list[object]]:
        return {group.name: [] for group in self.groups}


NHL = LeagueSpec(
    code="nhl",
    display_name="National Hockey League",
    position_column="positionCode",
    player_id_column="player_id",
    default_score_column="war",
    groups=(
        PositionGroup("forwards", 12, ("C", "L", "R", "LW", "RW", "F"), elimination_size=4),
        PositionGroup("defense", 6, ("D", "LD", "RD"), elimination_size=2),
    ),
    feature_exclude=("player_id", "player_name", "name", "positionCode", "lineup_group"),
)

NBA = LeagueSpec(
    code="nba",
    display_name="National Basketball Association",
    position_column="position",
    player_id_column="player_id",
    default_score_column="impact_score",
    groups=(
        PositionGroup("guards", 4, ("PG", "SG", "G", "PG/SG", "SG/PG"), elimination_size=2),
        PositionGroup("wings", 4, ("SF", "PF", "F", "SF/PF", "PF/SF"), elimination_size=2),
        PositionGroup("bigs", 4, ("C", "FC", "F/C", "PF/C", "C/F"), elimination_size=1),
    ),
    feature_exclude=("player_id", "player_name", "name", "position", "lineup_group"),
)

LEAGUES: dict[str, LeagueSpec] = {
    NHL.code: NHL,
    NBA.code: NBA,
}


def known_leagues() -> Iterable[str]:
    return LEAGUES.keys()
