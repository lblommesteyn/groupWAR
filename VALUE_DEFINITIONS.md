# DUNCAN and IGGY value outputs: definitions (2026-11-27)

Code `value_outputs.py` → `results/value/players.json`, `iggy_examples.json`, `audit.json` (with provenance). The
frozen live forecasting model is not changed. Example date: ratings through 2026-11-24 (`live_ratings_2026-11-25.npz`).

## Common basis

| Quantity | Definition | Audit |
|---|---|---|
| Rating unit | points per 100 possessions vs an average player; O = offense, D = defense with + = fewer points allowed, net = O + D | DUNCAN-2 ridge θ (after the mover pull), centred on the season's possession-weighted average player |
| Possessions | a player's on-court possessions = mean of his team's offensive and defensive possessions in each stint he plays | O applies to offensive and D to defensive possessions; using their mean for net is exact when they are equal, and they differ by under 1% over a season |
| Points → wins | wins = points / ppw. ppw = 1 / slope of win% on average margin per game over 210 team-seasons = **36.0** (95% CI 34.6–37.5, team-season bootstrap) | +1 point per game over 82 games = 82 / 36 = 2.3 wins |
| Replacement level | possession-weighted mean net rating of players outside the top 390 by possessions over two seasons (the minimum-contract pool) = **−1.64** (a definition, not an estimate) | a replacement team (five such players, ~99 possessions per game) wins 82 × (0.5 + 5 × −1.64 × 0.99 / 36) ≈ 22.5 games |

## The four outputs (kept separate)

1. **Current ability** (`ability`): O, D, net per 100, with **net_sd** from a game-cluster bootstrap of the ridge
   (20 refits, Poisson game weights, same λ / prior / luck target). Answers: how good is he now?
2. **Season-to-date contribution** (`season_to_date.war`): (net − replacement) × possessions played this season /
   100 / ppw. Answers: what are the minutes he has played worth *at his current ability estimate*. It is not a
   measure of what actually happened on the court.
3. **Projected rest of season** (`projection.war_ros`): (net − replacement) × expected possessions / 100 / ppw.
   Expected minutes per team game = availability × minutes when available, **rescaled within the team so the team's
   players add up to 240 per game** (missed minutes are played by teammates; IGGY B0 rule with the same caps).
   availability = games played over his last 82 team games, shrunk to 0.85 with 30 pseudo-games.
4. **Marginal value to a roster (IGGY)** (`iggy_examples.json`), per game, with one minutes model (B2 from
   `iggy_minutes.py`) and one reference scenario (the team's candidates at its latest covered game, that game's
   absences still out):
   - **(a) fixed-minutes substitution:** his predicted minutes are played by a replacement-level player (or by the
     teammate who would absorb the most minutes): (net_i − net_sub) × his minutes × 2.05 poss/min / 100.
   - **(b) roster change with predicted redistribution:** the same minutes model re-predicts the rotation with him
     out (absent-minutes, absence-count and absence-history features updated):
     Σ_k net_k × (minutes_k with − minutes_k without) × 2.05 / 100.
   - **depth/redistribution effect = (b) − (a vs replacement).** It comes only from who absorbs the minutes and how
     good they are. It is **not "fit"**: no synergy terms are used. Skill-pair synergy (`pop_synergy.py`) stays
     experimental and is not part of any value output.

## Sensitivity (per player in `players.json → projection.ranges`)

rating ±1 bootstrap sd; replacement level −1.0 / −2.5; availability ±0.1; minutes ±20%; points per win over its CI.

## Unmodelled uncertainty

Injury risk beyond the availability rate; trades and role changes; rating drift during the season; coach changes;
minutes-model error (test MAE 4.5 min per player in single-absence games, `results/iggy/minutes.json`); playoff value;
covariance of teammates' rating errors (the bootstrap sd is per player, so team sums treat errors as independent);
possessions per minute fixed at 2.05 in IGGY scenarios.

## Verified calculations (`audit.json → checks`)

- Every player's season-to-date WAR and projection are recomputed from the definitions above (asserted in code).
- Team expected minutes per game add up to 240 (min/max in `checks.team_minutes_per_game`).
- League totals (see examples below): the projection without availability overshot (618 vs 555 implied by
  replacement level); availability without team rescaling undershot (458), because missed minutes vanished. The
  rescaled version is reported below.

## Examples

| Player | Net per 100 (±sd) | Season-to-date WAR (range) | Exp. min/game | Projected ROS WAR | ROS range: rating ±1sd · replacement −1.0/−2.5 · availability ±0.1 · minutes ±20% |
|---|---|---|---|---|---|
| Nikola Jokić (DEN) | +7.25 (±0.43) | 2.63 (2.31–3.01) | 39.9 | 11.72 | 11.2–12.3 · 10.9–12.9 · 10.3–13.2 · 9.4–14.1 |
| Victor Wembanyama (SAS) | +9.87 (±0.73) | 3.30 (2.91–3.76) | 25.8 | 10.58 | 9.9–11.2 · 10.0–11.4 · 9.3–11.8 · 8.5–12.7 |
| Shai Gilgeous-Alexander (OKC) | +6.55 (±0.78) | 2.59 (2.14–3.11) | 29.0 | 8.37 | 7.6–9.2 · 7.7–9.2 · 7.3–9.4 · 6.7–10.0 |
| Derrick White (BOS) | +3.83 (±0.62) | 1.51 (1.16–1.92) | 32.7 | 6.20 | 5.5–6.9 · 5.5–7.2 · 5.5–6.8 · 5.0–7.4 |
| Jalen Johnson (ATL) | -1.10 (±0.75) | 0.15 (-0.23–0.59) | 36.3 | 0.73 | -0.3–1.7 · -0.1–1.9 · 0.6–0.8 · 0.6–0.9 |

IGGY, per game (points; wins per 82 games), same minutes model in both scenarios:

| Player | Minutes | (a) fixed minutes vs replacement | (a) vs top absorber | (b) roster change, redistributed | depth effect b − a | Main absorbers |
|---|---|---|---|---|---|---|
| Jayson Tatum | 33.9 | +2.89 (+6.6 W) | +1.38 vs Mike Conley | +1.59 (+3.6 W) | -1.30 | Mike Conley +6.2, Hugo González +4.2, Luka Garza +3.9 |
| Derrick White | 30.5 | +3.42 (+7.8 W) | +2.07 vs Mike Conley | +2.18 (+5.0 W) | -1.24 | Mike Conley +4.4, Baylor Scheierman +4.3, Jordan Walsh +3.3 |
| Donovan Mitchell | 31.1 | +2.94 (+6.7 W) | +2.65 vs Mario Hezonja | +2.52 (+5.7 W) | -0.42 | Mario Hezonja +5.2, Norchad Omier +4.9, Craig Porter Jr. +4.8 |
| James Harden | 31.9 | +1.63 (+3.7 W) | +1.33 vs Mario Hezonja | +1.20 (+2.7 W) | -0.43 | Mario Hezonja +5.1, Craig Porter Jr. +5.1, Norchad Omier +4.9 |
| Jalen Johnson | 33.7 | +0.37 (+0.8 W) | +0.46 vs Corey Kispert | -0.11 (-0.2 W) | -0.48 | Corey Kispert +6.2, Asa Newell +5.2, RayJ Dennis +3.8 |

League totals over a full season: projection without availability 618; with availability and team minutes rescaled to 240 **518**; implied by replacement level 554.7. The remaining gap (~7%) is players not yet rated this season (they play some of the minutes but carry no projection) and the ROS centring. Team minutes per game: {'min': 240.0, 'max': 240.0}.

Reading the examples: the depth effect is negative for every star shown. Their minutes would go to teammates well above replacement level, so the fixed-minutes-vs-replacement number overstates what losing them costs a real roster. That gap is about depth, not fit.
