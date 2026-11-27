# BOWEN v2/v3 (2026-11-27, Modal A10G)

Same seed and game folds as v1; all out-of-fold; 17.2M common defender-frames (13.4M live, 4.1M live early season).
Eval: `bowen_v2_eval.py` -> `results/world/bowen_v2_eval.json`; game-bootstrap of NLL: `bowen_v2_boot.py` (900 games).

| Live frames | median ft | p90 | p99 | >15 ft | x p90 | y p90 | NLL |
|---|---|---|---|---|---|---|---|
| v1 all frames, single spot | 2.95 | 10.77 | 26.8 | 5.6% | 7.95 | 5.99 | 3.434 |
| A live-only, single spot | 2.95 | 10.65 | 27.2 | 5.5% | 7.83 | 6.00 | 3.447 |
| B live-only, 3-option mixture | 2.94 | 11.30 | 33.1 | 6.6% | 8.20 | 6.26 | 3.120 |
| V3 hierarchy, scheme readout | 2.97 | 11.21 | 33.1 | 6.5% | 8.14 | 6.23 | 3.109 |
| V3 hierarchy, scheme + player | 2.90 | 10.86 | 32.8 | 6.3% | 7.85 | 6.07 | 3.057 |

NLL differences (95% game-bootstrap CI): A−v1 +0.013 [+0.011, +0.015]; B−v1 −0.314 [−0.325, −0.303];
V3 scheme−B −0.011 [−0.012, −0.010]; V3 scheme+player−B −0.064 [−0.065, −0.062].
Learned sigmas: team 0.111, team-season 0.114, defender 0.089.

Reading:
- Live-only training alone does almost nothing; dead-ball misses were not contaminating live predictions.
- The mixture is a much better *distribution* (−0.31 nats) but its point readout (most likely option) is worse in the
  tail: when it picks the wrong option (sink vs stay home) the miss is large. Use the mixture for deviation scoring
  (likelihood), not the single spot. Mixture mean/components are not saved yet.
- Hierarchy: team/team-season levels add little (−0.011); the defender level is the real gain (−0.052 over scheme),
  biggest on Interior and On-ball. Early season same pattern (3.173 -> 3.164 -> 3.113).
- Perimeter stay-home depth misses remain the main failure (x p90 ~19 ft for mixtures).
