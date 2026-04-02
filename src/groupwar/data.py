from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .specs import LeagueSpec


def load_players_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_synergy_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_player_pool(spec: LeagueSpec, players: pd.DataFrame) -> pd.DataFrame:
    annotated = spec.annotate_groups(players)
    prepared = annotated[annotated["lineup_group"].notna()].copy()
    prepared = prepared.drop_duplicates(subset=[spec.player_id_column]).reset_index(drop=True)
    return prepared


def build_group_pools(spec: LeagueSpec, players: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prepared = prepare_player_pool(spec, players)
    return {
        group.name: prepared[prepared["lineup_group"] == group.name].copy().reset_index(drop=True)
        for group in spec.groups
    }


def infer_feature_columns(
    players: pd.DataFrame,
    *,
    exclude: Iterable[str],
) -> list[str]:
    excluded = set(exclude)
    columns: list[str] = []
    for column in players.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(players[column]):
            columns.append(column)
    return columns


def compute_pairwise_distance(
    players: pd.DataFrame,
    *,
    id_column: str,
    feature_columns: Iterable[str],
) -> pd.DataFrame | None:
    columns = list(feature_columns)
    if not columns:
        return None
    matrix = players.loc[:, columns].to_numpy(dtype=float)
    distances = np.sqrt(np.sum((matrix[:, None, :] - matrix[None, :, :]) ** 2, axis=-1))
    ids = players[id_column].tolist()
    return pd.DataFrame(distances, index=ids, columns=ids)
