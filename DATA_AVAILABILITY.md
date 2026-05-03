# Data Availability

This repository contains source code, synthetic examples, compact derived NBA
artifacts, and curated result tables. It does not contain large raw data dumps,
downloaded third-party databases, or trained model checkpoints.

## Included

- `examples/nhl_players.csv`: synthetic NHL-style player table for package
  smoke tests
- `examples/nba_players.csv`: synthetic NBA-style player table for package
  smoke tests
- `nba_war/data/`: compact derived NBA artifacts used for inspection and
  development
- `results/`: curated CSV tables used in the manuscript summaries

## Not Included

- raw NHL play-by-play and shift-chart dumps
- raw NBA SQLite database or downloaded Kaggle archive
- full raw NBA API pull outputs
- trained `.pth` model checkpoints
- large intermediate tensors generated during model training

These are excluded to keep the repository small and to avoid redistributing
data sources with separate access terms.

## NHL Sources

The hockey workflow was built from NHL public endpoints:

- play-by-play:
  `https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play`
- shift charts:
  `https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}`
- roster and player metadata:
  `https://api-web.nhle.com/v1/roster/{team}/{season}`

Derived hockey artifacts include stint-level lineups, co-deployment graphs,
adjusted plus-minus estimates, player features, and trained scoring models.

## NBA Sources

The primary NBA workflow expects the Kaggle `wyattowalsh/basketball` SQLite
database:

<https://www.kaggle.com/datasets/wyattowalsh/basketball>

Place the SQLite file at:

```text
nba_war/data/nba.sqlite
```

An alternate acquisition path uses `nba_api` to pull public NBA endpoints for
game logs, play-by-play, rotations, shot charts, and player metadata.

## External Benchmark Sources

The manuscript also references public roster and player-value sources for
validation and comparison:

- Hockey Canada 2025 4 Nations roster page
- NHL.com 4 Nations roster page
- FIBA 2024 men's Olympic roster tracker
- NBA.com USA Basketball roster release
- Canadian Olympic Committee and Canada Basketball roster release
- Dunks and Threes Estimated Plus-Minus documentation

These links are cited in the manuscript bibliography.
