# V2 Changes

## 1. Bigger Embeddings (13-dim → 20-dim)

**`nba_process_v2.py`**

Added features:
- `shot_zone_*_pct` (5 cols): proportion of FGA from each zone (Above the Break 3, Paint, Mid-Range, Left Corner 3, Right Corner 3)
- `AST` (1 col): total assists from play-by-play
- `AST_rate` (1 col): assists per minute on court

Why: Jaden raised that shot zone distributions + assist locations help the model infer compatibility for novel (never-before-seen) player pairings, since the adjacency matrix can't capture chemistry for players who haven't shared the court.

## 2. Updated Opponents

**`nba_search_v2.py`**

- USA evaluates against: Canada, France, OKC
- Canada evaluates against: USA, France, OKC

Why: National teams (Canada, France) are the actual Olympic opponents. OKC provides a third opponent with strong data coverage.

## 3. RAPM Correlation

**`rapm_correlation.py`**

Compares our Ridge APM against publicly available RAPM. Outputs Pearson/Spearman correlations and rank-order comparison for top-50 players.

## 4. Case Studies

**`case_studies.py`**

Three detailed analyses:
1. **Embiid → Green swap**: Why adversarial evaluation prefers switchable defense over paint scoring
2. **Hockey 2024 constraints**: Unconstrained (10C/0R, all negative WAR) vs constrained (balanced, all positive WAR)
3. **Holiday consistency**: Why a non-locked two-way guard is selected every iteration

## Running

```bash
# Step 1: Build v2 embeddings
python v2/nba_process_v2.py

# Step 2: Train v2 models
python v2/nba_train_v2.py

# Step 3: Run v2 search
python v2/nba_search_v2.py

# Step 4: RAPM correlation
python v2/rapm_correlation.py

# Step 5: Case studies
python v2/case_studies.py
```

## File Structure

```
v2/
├── nba_process_v2.py    # Expanded embeddings (20-dim)
├── nba_train_v2.py      # Training with NODE_IN_DIM=20
├── nba_search_v2.py     # Updated opponents + locks
├── rapm_correlation.py  # APM vs public RAPM
├── case_studies.py      # Three detailed analyses
├── README.md            # This file
└── outputs/             # Generated results
```
