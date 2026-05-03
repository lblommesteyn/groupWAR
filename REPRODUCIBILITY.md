# Reproducibility Notes

This repository is organized as a lightweight research artifact for the
manuscript "National-Team Selection Under Pressure: Graph-Scored Search and
Adversarial Evaluation."

The repository supports three levels of reproduction.

## Level 1: Package Smoke Tests

These checks require only the repository contents.

```bash
python -m pip install -e .[dev]
python -m pytest
python -m groupwar.cli optimize --league nhl --players examples/nhl_players.csv --score-column war --output outputs/nhl_selected.csv
python -m groupwar.cli optimize --league nba --players examples/nba_players.csv --score-column impact_score --output outputs/nba_selected.csv
```

This verifies:

- league-specific roster constraints
- position-group assignment
- tournament and tabu selection logic
- CSV input/output behavior

## Level 2: Inspect Manuscript Results

Curated result tables are stored in `results/`. These are lightweight CSV
exports of the case-study outputs used in the manuscript. They are intended for
audit and inspection of reported numbers, not for retraining the GCN from raw
events.

Start with:

```text
results/results_summary_statistics.csv
results/results_hockey_canada_2025.csv
results/results_hockey_canada_2024.csv
results/results_basketball_usa_2024.csv
results/results_basketball_canada_2024.csv
```

## Level 3: Rebuild Case Studies

Full rebuilding requires external data and generated model artifacts.

### NBA

1. Download the Kaggle `wyattowalsh/basketball` dataset.
2. Place the SQLite database at `nba_war/data/nba.sqlite`.
3. Build derived tables:

   ```bash
   cd nba_war
   python nba_from_sqlite.py
   ```

4. Train models:

   ```bash
   python nba_train.py
   ```

5. Run search:

   ```bash
   python nba_search.py
   ```

The alternate API acquisition path starts with `python nba_data_pull.py` and
then `python nba_process.py`.

### Hockey

The hockey workflow uses NHL public API data for play-by-play, shifts, rosters,
and metadata. The hockey-specific scripts are under `hockey_war/`. The
repository includes the generic optimizer package, the hockey search scripts,
and curated result exports, but does not include large raw NHL event dumps or
trained hockey checkpoints.

The minimum external ingredients are:

- season-level play-by-play
- shift charts
- roster and player metadata
- adjusted plus-minus tables
- trained graph-scoring checkpoints

The exact source endpoints are listed in `DATA_AVAILABILITY.md`.

## Determinism

The package examples use explicit random seeds. Full model training can still
vary across hardware, PyTorch versions, CUDA kernels, and data refresh timing.
For manuscript comparison, use the curated result tables under `results/`.

## Known Limits

- The repository is not a frozen archival release unless it is tagged or
  archived separately.
- Large raw inputs and trained models are excluded from git.
- The NBA scripts preserve research workflow code and are less polished than
  the installable `groupwar` package.
