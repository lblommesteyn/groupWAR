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

## Where The Data Comes From

This repository includes small example CSVs in `examples/`, but it does not bundle the full raw research datasets, intermediate parquet/pickle files, or trained model checkpoints. To reproduce the original hockey and NBA workflow, you need to build those inputs yourself from the upstream data sources below.

### Hockey

The hockey pipeline was built from NHL public endpoints:

- play-by-play: `https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play`
- shift charts: `https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}`
- roster and player metadata: `https://api-web.nhle.com/v1/roster/{team}/{season}`

From those feeds, the original workflow constructs lineup stints, co-deployment graphs, APM tables, and player feature embeddings. In the legacy project layout, those derived artifacts end up as files such as:

- season play-by-play and shift dumps
- `apm_minutes.csv`
- roster or headshot tables used to map player IDs, positions, and nationality
- period or stint-level tensors used for GCN training

### NBA

There are two supported acquisition paths for the NBA side:

- Kaggle SQLite dump: the legacy NBA pipeline expects `data/nba.sqlite` from the `wyattowalsh/basketball` dataset on Kaggle
- direct API pull: an alternate script path uses `nba_api` to pull `stats.nba.com` game logs, play-by-play, game rotations, shot charts, and player metadata

The SQLite path is the cleaner starting point if you want to reproduce the training tables in one place. The API path is useful when you want to refresh the dataset without depending on the Kaggle dump.

### Not Included In Git

The following are intentionally not packaged here:

- large raw game-event dumps
- derived parquet or pickle artifacts
- trained `.pth` model weights
- league-specific output CSVs from ad hoc search runs

If you want a reproducible setup, store those files under a local `data/` directory and keep only lightweight examples in git.

## NHL and NBA Defaults

Default lineup groups are editable in code and intentionally explicit:

- NHL: `12` forwards and `6` defense
- NBA: `4` guards, `4` wings, and `4` bigs

The NBA default is a 12-player rotation rather than a strict 5-man starting lineup so the optimizer mirrors the broader roster-selection use case from the original hockey workflow.

## Awesome TODO

Ideas worth building because they would make `groupWAR` materially more interesting, not just more polished:

- Opponent-aware roster co-design: optimize a lineup against an entire distribution of opponents, then expose which players are "meta-stable" versus only good against a narrow matchup class.
- Counterfactual substitution simulator: learn the value of replacing one player with an archetype rather than a specific name, so the system can answer questions like "what kind of player are we actually missing?"
- Chemistry as a latent variable: infer hidden pair and trio effects from observed lineups/rotations instead of relying only on explicit player features, then surface unexplained chemistry as a first-class output.
- Robust optimization under uncertainty: optimize not just for expected score but for downside protection when player projections are noisy, creating lineups that are resilient to model error.
- Time-aware lineup policy search: choose different optimal groups for early game, late game, trailing, leading, or special-situation contexts rather than producing one static "best roster."
- Cross-sport representation learning: align NHL and NBA player embeddings into a shared "role space" so ideas like connectors, finishers, stabilizers, and suppressors become sport-agnostic building blocks.
- Fatigue and load propagation: treat repeated high-leverage usage as a cost that changes downstream matchup quality, allowing the optimizer to reason about rotation sustainability instead of one-shot peak output.
- Adversarial coach model: train a second policy that reacts to your chosen lineup with substitutions or matchup targeting, turning the search into a real strategic game instead of a fixed-response evaluation.
- Budget and contract frontier search: add salary, age, and term constraints so the package can generate efficient front-office tradeoffs instead of only best-on-paper competitive lineups.
- Tournament bracket optimization: optimize a roster for a path of opponents rather than one matchup, which is a genuinely different problem and much closer to playoff or international tournament decision-making.
- Explainable lineup narratives: generate a compact explanation of why each selected player survived the search, including which opponent styles or internal chemistry structures they protect.
- Archetype market generator: simulate a synthetic free-agent or trade market by asking which archetypes would most improve a roster if added, then rank them by marginal team-level counterfactual value.

## Validation

Run:

```bash
python -m pytest
```
