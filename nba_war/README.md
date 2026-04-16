# nba_war

This folder contains the league-specific NBA research scripts that sit beside the packaged `groupwar` library.

## What Is Here

- `nba_from_sqlite.py`: builds the NBA pipeline from `data/nba.sqlite`
- `nba_data_pull.py`: alternate data acquisition path using `nba_api`
- `nba_process.py`: turns pulled NBA data into lineup, APM, and embedding artifacts
- `nba_train.py`: trains the 6-model GCN ensemble
- `nba_search.py`: runs the Olympic roster search
- `nba_stackel.py`: NBA-specific graph and Stackelberg utilities
- `data/`: small derived files that are safe to keep in git

## What You Still Need Locally

This repo does not include the large raw inputs or trained weights:

- `data/nba.sqlite` from the Kaggle `wyattowalsh/basketball` dataset
- any large downloaded archives
- `models/nba_model_*.pth` checkpoints produced by training
- large raw API pull outputs if you use the `nba_api` route

## Data Sources

There are two paths:

1. Kaggle SQLite path
   - Download the `wyattowalsh/basketball` dataset from Kaggle.
   - Place the SQLite file at `nba_war/data/nba.sqlite`.
   - Run `python nba_from_sqlite.py`.

2. `nba_api` path
   - Install `nba_api`.
   - Run `python nba_data_pull.py`.
   - Then run `python nba_process.py`.

## Typical Order

1. Build data with `nba_from_sqlite.py` or `nba_data_pull.py` plus `nba_process.py`
2. Train models with `nba_train.py`
3. Run the search with `nba_search.py`
