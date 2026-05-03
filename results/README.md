# Result Table Manifest

The CSV files in this directory are lightweight exports used to audit the
manuscript's reported case-study summaries.

## Summary Tables

- `results_summary_statistics.csv`: one-row-per-case summary of final score,
  roster size, overlap, position balance, and WAR range
- `results_model_performance.csv`: model-performance summary values
- `results_training_data_summary.csv`: training data coverage summary

## Final Rosters

- `results_hockey_canada_2025.csv`: Canada 2025 4 Nations hockey roster output
- `results_hockey_canada_2024.csv`: Canada 2024 hockey diagnostic roster output
- `results_basketball_usa_2024.csv`: USA 2024 basketball roster output
- `results_basketball_canada_2024.csv`: Canada 2024 basketball roster output

## Search Diagnostics

- `results_evaluation_scores.csv`: final evaluation score components
- `results_tournament_phase.csv`: tournament-stage outputs
- `results_greedy_iterations.csv`: greedy refinement iteration summaries
- `results_rotation_analysis.csv`: basketball stochastic-rotation analysis

These files are curated exports. They are not a replacement for raw data,
trained checkpoints, or the full end-to-end data build.
