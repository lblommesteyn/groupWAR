from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
import random

import pandas as pd

from .data import build_group_pools, compute_pairwise_distance, infer_feature_columns, prepare_player_pool
from .scoring import LineupEvaluation, LineupScorer
from .specs import LeagueSpec


@dataclass
class SelectionResult:
    lineup: dict[str, list[object]]
    score: float | None
    removed: list[dict[str, Any]]
    rounds_completed: int

    def to_frame(self, players: pd.DataFrame, *, id_column: str) -> pd.DataFrame:
        order = {
            player_id: (group_name, slot)
            for group_name, group_ids in self.lineup.items()
            for slot, player_id in enumerate(group_ids)
        }
        frame = players[players[id_column].isin(order)].copy()
        frame["lineup_group"] = frame[id_column].map(lambda player_id: order[player_id][0])
        frame["lineup_slot"] = frame[id_column].map(lambda player_id: order[player_id][1])
        return frame.sort_values(["lineup_group", "lineup_slot"]).reset_index(drop=True)


class LineupOptimizer:
    def __init__(
        self,
        spec: LeagueSpec,
        players: pd.DataFrame,
        scorer: LineupScorer,
        *,
        feature_columns: list[str] | None = None,
        seed: int = 123,
        tabu_size: int = 50,
        candidates_per_swap: int = 5,
        low_usage_candidates: int = 2,
        ranking_column: str | None = None,
    ) -> None:
        self.spec = spec
        self.players = prepare_player_pool(spec, players)
        self.scorer = scorer
        self.rng = random.Random(seed)
        self.tabu_moves: deque[tuple[str, object, object]] = deque(maxlen=tabu_size)
        self.candidates_per_swap = candidates_per_swap
        self.low_usage_candidates = low_usage_candidates
        self.ranking_column = ranking_column
        self.group_pools = build_group_pools(spec, self.players)

        chosen_features = feature_columns
        if chosen_features is None:
            chosen_features = infer_feature_columns(self.players, exclude=spec.feature_exclude)
        self.distance_frame = compute_pairwise_distance(
            self.players,
            id_column=spec.player_id_column,
            feature_columns=chosen_features,
        )
        self.player_index = self.players.set_index(spec.player_id_column, drop=False)

    def _clone_lineup(self, lineup: dict[str, list[object]]) -> dict[str, list[object]]:
        return {group_name: list(group_ids) for group_name, group_ids in lineup.items()}

    def _sort_ids(self, ids: list[object], *, descending: bool = True) -> list[object]:
        if not self.ranking_column:
            return list(ids)
        ordered = self.player_index.loc[ids].sort_values(self.ranking_column, ascending=not descending)
        return ordered[self.spec.player_id_column].tolist()

    def initial_lineup(self, *, locked: dict[str, list[object]] | None = None) -> dict[str, list[object]]:
        locked = locked or {}
        lineup = self.spec.empty_lineup()
        for group in self.spec.groups:
            pinned = list(locked.get(group.name, []))
            available = [
                player_id
                for player_id in self._sort_ids(
                    self.group_pools[group.name][self.spec.player_id_column].tolist()
                )
                if player_id not in pinned
            ]
            needed = max(group.lineup_size - len(pinned), 0)
            lineup[group.name] = pinned + available[:needed]
        self._validate_lineup(lineup)
        return lineup

    def _validate_lineup(self, lineup: dict[str, list[object]]) -> None:
        for group in self.spec.groups:
            count = len(lineup.get(group.name, []))
            if count != group.lineup_size:
                raise ValueError(
                    f"group '{group.name}' expected {group.lineup_size} players, received {count}"
                )

    def evaluate(self, lineup: dict[str, list[object]]) -> LineupEvaluation:
        self._validate_lineup(lineup)
        return self.scorer.evaluate(lineup)

    def _rank_bench(self, player_out: object, bench: list[object]) -> list[object]:
        if self.distance_frame is not None and player_out in self.distance_frame.index:
            return sorted(
                bench,
                key=lambda candidate: float(self.distance_frame.loc[player_out, candidate]),
                reverse=True,
            )
        return self._sort_ids(bench)

    def propose_swaps(
        self,
        lineup: dict[str, list[object]],
        usage: pd.DataFrame,
    ) -> list[tuple[dict[str, list[object]], tuple[str, object, object]]]:
        neighbors: list[tuple[dict[str, list[object]], tuple[str, object, object]]] = []
        for group in self.spec.groups:
            selected = list(lineup[group.name])
            group_usage = (
                usage[usage["group_name"] == group.name]
                .sort_values("usage", ascending=True)
                .head(self.low_usage_candidates)
            )
            bench = [
                player_id
                for player_id in self.group_pools[group.name][self.spec.player_id_column].tolist()
                if player_id not in selected
            ]
            for player_out in group_usage["player_id"].tolist():
                candidate_bench = self._rank_bench(player_out, bench)[: self.candidates_per_swap]
                for player_in in candidate_bench:
                    move = (group.name, player_out, player_in)
                    if move in self.tabu_moves:
                        continue
                    candidate = self._clone_lineup(lineup)
                    slot = candidate[group.name].index(player_out)
                    candidate[group.name][slot] = player_in
                    neighbors.append((candidate, move))
        return neighbors

    def run_tabu(
        self,
        *,
        max_iters: int = 100,
        initial_lineup: dict[str, list[object]] | None = None,
    ) -> SelectionResult:
        lineup = initial_lineup or self.initial_lineup()
        best_lineup = self._clone_lineup(lineup)
        best_eval = self.evaluate(lineup)

        for _ in range(max_iters):
            current_eval = self.evaluate(lineup)
            neighbors = self.propose_swaps(lineup, current_eval.usage)
            if not neighbors:
                break

            scored_neighbors: list[tuple[float, dict[str, list[object]], tuple[str, object, object]]] = []
            for candidate, move in neighbors:
                candidate_eval = self.evaluate(candidate)
                scored_neighbors.append((candidate_eval.score, candidate, move))

            scored_neighbors.sort(key=lambda item: item[0], reverse=True)
            score, lineup, move = scored_neighbors[0]
            self.tabu_moves.append(move)
            if score > best_eval.score:
                best_eval = self.evaluate(lineup)
                best_lineup = self._clone_lineup(lineup)

        return SelectionResult(
            lineup=best_lineup,
            score=best_eval.score,
            removed=[],
            rounds_completed=max_iters,
        )

    def run_tournament(
        self,
        *,
        locked: dict[str, list[object]] | None = None,
        max_rounds: int = 25,
    ) -> SelectionResult:
        locked = {group.name: list((locked or {}).get(group.name, [])) for group in self.spec.groups}
        remaining = {
            group.name: [
                player_id
                for player_id in self.group_pools[group.name][self.spec.player_id_column].tolist()
                if player_id not in locked[group.name]
            ]
            for group in self.spec.groups
        }
        removed: list[dict[str, Any]] = []
        rounds_completed = 0

        while (
            sum(len(remaining[group.name]) + len(locked[group.name]) for group in self.spec.groups)
            > self.spec.team_size
            and rounds_completed < max_rounds
        ):
            rounds_completed += 1
            needs = {
                group.name: max(group.lineup_size - len(locked[group.name]), 0)
                for group in self.spec.groups
            }
            team_counts = [len(remaining[name]) // need for name, need in needs.items() if need > 0]
            num_teams = min(team_counts) if team_counts else 0
            if num_teams < 2:
                break

            shuffled = {name: list(player_ids) for name, player_ids in remaining.items()}
            for player_ids in shuffled.values():
                self.rng.shuffle(player_ids)

            teams: list[dict[str, list[object]]] = []
            leftovers: dict[str, list[object]] = {}
            for group_name, need in needs.items():
                cutoff = num_teams * need
                leftovers[group_name] = shuffled[group_name][cutoff:]

            for index in range(num_teams):
                lineup = {group.name: list(locked[group.name]) for group in self.spec.groups}
                for group in self.spec.groups:
                    need = needs[group.name]
                    start = index * need
                    stop = start + need
                    lineup[group.name].extend(shuffled[group.name][start:stop])
                teams.append(lineup)

            scored: list[tuple[float, dict[str, list[object]], LineupEvaluation]] = []
            for lineup in teams:
                evaluation = self.evaluate(lineup)
                scored.append((evaluation.score, lineup, evaluation))
            scored.sort(key=lambda item: item[0], reverse=True)

            total_candidates = sum(len(ids) for ids in remaining.values())
            divisor = 3 if total_candidates > 100 else 2
            top_count = max(1, len(scored) // divisor)
            next_remaining = {group.name: [] for group in self.spec.groups}

            for _, lineup, _ in scored[:top_count]:
                for group in self.spec.groups:
                    next_remaining[group.name].extend(
                        [player_id for player_id in lineup[group.name] if player_id not in locked[group.name]]
                    )

            for _, lineup, evaluation in scored[top_count:]:
                for group in self.spec.groups:
                    team_players = lineup[group.name]
                    usage = (
                        evaluation.usage[evaluation.usage["player_id"].isin(team_players)]
                        .sort_values("usage", ascending=False)
                    )
                    available_count = len(remaining[group.name]) + len(locked[group.name])
                    removable = max(available_count - group.lineup_size, 0)
                    drop_count = min(group.elimination_size, removable, len(usage))
                    keep = usage["player_id"].iloc[: len(usage) - drop_count].tolist()
                    dropped = usage["player_id"].iloc[len(usage) - drop_count :].tolist()
                    next_remaining[group.name].extend(
                        [player_id for player_id in keep if player_id not in locked[group.name]]
                    )
                    removed.extend(
                        {
                            "player_id": player_id,
                            "round": rounds_completed,
                            "group_name": group.name,
                        }
                        for player_id in dropped
                    )

            for group in self.spec.groups:
                next_remaining[group.name].extend(leftovers[group.name])
                remaining[group.name] = list(dict.fromkeys(next_remaining[group.name]))

        final_lineup = self.spec.empty_lineup()
        for group in self.spec.groups:
            candidates = locked[group.name] + remaining[group.name]
            if self.ranking_column:
                unlocked = [player_id for player_id in candidates if player_id not in locked[group.name]]
                ordered = locked[group.name] + self._sort_ids(unlocked)
            else:
                ordered = candidates
            final_lineup[group.name] = ordered[: group.lineup_size]

        score = None
        if all(len(final_lineup[group.name]) == group.lineup_size for group in self.spec.groups):
            score = self.evaluate(final_lineup).score

        return SelectionResult(
            lineup=final_lineup,
            score=score,
            removed=removed,
            rounds_completed=rounds_completed,
        )
