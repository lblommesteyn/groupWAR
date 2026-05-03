# groupWAR

Code and lightweight reproducibility materials for:

**National-Team Selection Under Pressure: Graph-Scored Search and Adversarial Evaluation**

Repository URL: <https://github.com/lblommesteyn/groupWAR>

This repository supports the paper's roster-selection case studies. It contains
an installable Python package for constrained group selection, legacy NBA
research scripts, small example inputs, and curated result tables. Large raw
game-event dumps, downloaded databases, and trained model checkpoints are not
stored in git.

## Repository Contents

```text
src/groupwar/        Installable optimizer package
src/groupwar/ml/     Optional graph and Stackelberg model components
examples/            Small synthetic NHL and NBA player tables for smoke tests
hockey_war/          Hockey case-study search and graph-scoring scripts
nba_war/             NBA data, training, and roster-search research scripts
nba_war/data/        Compact derived NBA artifacts kept for inspection
nba_war/v2/          Later NBA feature, search, rotation, and RAPM analyses
results/             Curated manuscript result tables
tests/               Package regression tests
```

The package layer is intentionally generic. League definitions in
`src/groupwar/specs.py` specify roster groups and position constraints. The
selection logic in `src/groupwar/selection.py` then runs tournament-style
pruning or tabu refinement against a scorer interface. The default scorer is a
weighted column scorer so the package can be tested without trained neural
models.

## Install

From the repository root:

```bash
python -m pip install -e .[dev]
```

Optional graph-model dependencies:

```bash
python -m pip install -e .[ml]
```

The optional ML dependency set installs PyTorch and PyTorch Geometric. It is not
needed for the package tests or the example optimizer runs.

## Quick Verification

Run the test suite:

```bash
python -m pytest
```

Run the included NHL example:

```bash
python -m groupwar.cli optimize --league nhl --players examples/nhl_players.csv --score-column war --output outputs/nhl_selected.csv
```

Run the included NBA example:

```bash
python -m groupwar.cli optimize --league nba --players examples/nba_players.csv --score-column impact_score --output outputs/nba_selected.csv
```

These commands use synthetic examples and do not reproduce the trained GCN
results from the paper. They verify the installable package, league constraints,
and search machinery.

## Manuscript Result Tables

Curated result tables used for the manuscript summaries are in `results/`.
They include selected roster summaries, evaluation scores, tournament phase
outputs, greedy-iteration summaries, model-performance summaries, and rotation
analysis tables.

See [results/README.md](results/README.md) for the table manifest.

## Full Reproduction Scope

The paper's full end-to-end workflow requires external data and trained model
artifacts that are too large or too license-dependent for git:

- NHL public API event, shift, roster, and metadata pulls
- Kaggle `wyattowalsh/basketball` SQLite data for the NBA workflow
- trained GCN checkpoint files
- large intermediate tensors and raw event dumps

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact inputs and the intended
reproduction path.

## NBA Workflow

The NBA-specific scripts are preserved under `nba_war/`:

```text
nba_from_sqlite.py   Build derived tables from Kaggle SQLite
nba_data_pull.py     Alternate acquisition path using nba_api
nba_process.py       Build lineup, APM, and embedding artifacts
nba_train.py         Train the GCN ensemble
nba_search.py        Run the Olympic roster search
nba_stackel.py       NBA graph and Stackelberg utilities
v2/                  Later case-study, rotation-scoring, and RAPM scripts
```

Place the external SQLite file at `nba_war/data/nba.sqlite` before running the
SQLite path. Model checkpoints are expected under `nba_war/models/` after
training and are ignored by git.

## Hockey Workflow

The hockey-specific scripts are preserved under `hockey_war/`:

```text
run_search.py       Canada hockey roster search workflow
mie368stackel.py    Hockey graph model and Stackelberg utilities
submit_search.sh    Cluster submission helper
```

The hockey workflow expects locally generated NHL event data, APM tables,
eligibility tables, and trained checkpoint files. Those large inputs remain
outside git; the manuscript-facing result exports are in `results/`.

## Citation

If using the repository directly, cite the repository URL above or use the
metadata in [CITATION.cff](CITATION.cff). A formal paper citation can replace
the repository citation once the manuscript is published.
