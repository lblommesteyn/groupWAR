# groupWAR

`groupWAR` is a cleaned-up package extracted from the original notebook snapshot and reshaped into a reusable lineup optimization toolkit.

The repository now has two layers:

- a generic optimizer/scoring layer that works across leagues
- an optional ML layer containing the legacy graph + Stackelberg machinery, generalized so it is no longer hardcoded to the NHL 18-vs-18 setting

## What Changed

The legacy zip was notebook-heavy and NHL-specific:

- business logic lived inside notebooks
- search code depended on global variables
- the graph solver assumed 36 players and two hardcoded NHL roster buckets
- there was no clean extension point for NBA work

This refactor moves the reusable pieces into a package with:

- league specs for `nhl` and `nba`
- a generic lineup optimizer for tabu search and tournament-style pruning
- scorer interfaces so the optimizer is not tied to one model family
- optional ML modules for graph/Stackelberg evaluation when trained weights and feature tensors are available

## Install

Base package:

```bash
python -m pip install --user -e .
```

With test tooling:

```bash
python -m pip install --user -e .[dev]
```

With the graph model extras:

```bash
python -m pip install --user -e .[ml]
```

## Quick Start

Use the packaged optimizer with the included synthetic examples.

NHL:

```bash
python -m groupwar.cli optimize ^
  --league nhl ^
  --players examples/nhl_players.csv ^
  --score-column war ^
  --output outputs/nhl_selected.csv
```

NBA:

```bash
python -m groupwar.cli optimize ^
  --league nba ^
  --players examples/nba_players.csv ^
  --score-column impact_score ^
  --output outputs/nba_selected.csv
```

The CLI uses the generic weighted scorer by default so the package works immediately for both sports. When trained graph models and matchup features are available, the optional ML modules in `groupwar.ml` can be used to plug the legacy counterfactual scorer back in.

## Package Layout

```text
src/groupwar/
  cli.py           Command-line entry point
  data.py          Player loading, grouping, distance helpers
  scoring.py       Generic scorer interfaces and weighted scorer
  selection.py     Tabu and tournament lineup optimization
  specs.py         NHL/NBA league definitions
  ml/
    graph.py       Graph model extracted from the notebooks
    stackelberg.py Stackelberg adjacency optimizer generalized for any even matchup size
```

## Data Contract

For the base optimizer, a player table is enough:

- player id column, defaulted per league
- position column, defaulted per league
- score column, chosen at the CLI

Optional columns:

- `player_name` or `name`
- feature columns for distance-based swap ranking

The examples show the expected shape.

## NHL and NBA Defaults

Default lineup groups are editable in code and intentionally explicit:

- NHL: `12` forwards and `6` defense
- NBA: `4` guards, `4` wings, and `4` bigs

The NBA default is a 12-player rotation rather than a strict 5-man starting lineup so the optimizer mirrors the broader roster-selection use case from the original hockey workflow.

## Validation

Run:

```bash
python -m pytest
```
