from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class LineupEvaluation:
    score: float
    usage: pd.DataFrame


class LineupScorer(Protocol):
    def evaluate(self, lineup: dict[str, list[object]]) -> LineupEvaluation:
        ...


class WeightedColumnScorer:
    def __init__(
        self,
        players: pd.DataFrame,
        *,
        id_column: str,
        score_column: str,
        synergy_frame: pd.DataFrame | None = None,
    ) -> None:
        self.players = players.set_index(id_column, drop=False)
        self.id_column = id_column
        self.score_column = score_column
        self.synergy_frame = synergy_frame

    def evaluate(self, lineup: dict[str, list[object]]) -> LineupEvaluation:
        selected_ids = [player_id for group_ids in lineup.values() for player_id in group_ids]
        selected = self.players.loc[selected_ids].copy()
        group_lookup = {
            player_id: group_name
            for group_name, group_ids in lineup.items()
            for player_id in group_ids
        }
        selected["group_name"] = selected[self.id_column].map(group_lookup)
        selected["usage"] = selected[self.score_column].astype(float)

        total_score = float(selected["usage"].sum())
        if self.synergy_frame is not None and not self.synergy_frame.empty:
            active = set(selected_ids)
            for row in self.synergy_frame.itertuples(index=False):
                left_id = getattr(row, "player_id_a")
                right_id = getattr(row, "player_id_b")
                if left_id in active and right_id in active:
                    bonus = float(getattr(row, "bonus"))
                    total_score += bonus
                    selected.loc[selected[self.id_column] == left_id, "usage"] += bonus / 2.0
                    selected.loc[selected[self.id_column] == right_id, "usage"] += bonus / 2.0

        usage = selected[[self.id_column, "group_name", "usage"]].rename(
            columns={self.id_column: "player_id"}
        )
        return LineupEvaluation(score=total_score, usage=usage.reset_index(drop=True))
