# Figures (from the counterfactual_ball report)

Shared here so they're accessible from this repo.

## `report/` — TikZ / pgfplots figures rendered from the report
Rendered from `report-latex/findings.tex` (each `tikzpicture` cropped to its own
PNG at 200 DPI via the LaTeX `preview` package). These are the inline TikZ figures
from the paper, not screenshots.

| File | Figure |
|------|--------|
| `01-pipeline-diagram.png` | End-to-end prototype pipeline |
| `02-adversarial-adjacency-matrix.png` | Schematic of the adversarial adjacency matrix |
| `03-stochastic-rotation-template.png` | Basketball stochastic-rotation template |
| `04-apm-vs-rapm-scatter.png` | Our Ridge APM vs. public RAPM (r = 0.81) |
| `05-rapm-rank-comparison.png` | Rank comparison, top-50 (38/50 overlap) |
| `06-groupwar-vs-rapm-nba-n80.png` | Group WAR vs. public RAPM, NBA (n = 80) |
| `07-groupwar-vs-rapm-all-n153.png` | Group WAR vs. public RAPM, all rosters (n = 153) |
| `08-hockey-war-constrained-vs-unconstrained.png` | Canada 2024 individual WAR: constrained vs. unconstrained |

## Presentation slides
`slide-13.png`, `slide-14.png` — exported presentation slides.

## Raw LaTeX
The full report/presentation sources are in [`../report-latex/`](../report-latex/).
Rebuild the report figures with `pdflatex findings.tex`.
