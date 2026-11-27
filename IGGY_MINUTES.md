# IGGY v0: minutes redistribution when players are absent (2026-11-27)

Code `iggy_minutes.py` (+ `iggy_failures.py`); outputs `results/iggy/` (minutes.json, examples.json, failures.json,
failure_summary.json, minutes_predictions.parquet on the PC). Provenance in minutes.json.

## Task and rules

- **Unit:** (game, team), regular season. **Candidates:** players with a game for the team in the previous 30 days.
- **Rotation player:** ≥ 10 min/game over the team's previous 10 games *of the same season* (0 for games missed).
- **Absence:** a rotation player who did not play. The absent set is the scenario input, as an injury report would
  give it pregame; late scratches mean this is slightly optimistic.
- **Pregame only:** every feature uses games before the date. The first 10 games of each season are not covered
  (there is no in-season rotation to start from; no preseason data).
- **Feasible rotation:** absent players 0; 0 ≤ m ≤ min(44, max(recent, 10) + 14); total 240 regulation minutes
  (overtime games rescaled); water-filling projection.
- **Split (season end years):** train 2021-24, select on 2025, test 2026 + 2027 to date.

## Results (test, mean absolute minutes error per non-absent candidate)

| Model | No absence (7,496) | One absence (11,442) | Two or more (15,328) |
|---|---|---|---|
| B0 recent minutes, renormalised | 5.42 | 5.64 | 6.47 |
| B1 + availability, start rate, rank, absent minutes | 4.04 | 4.52 | 5.66 |
| B2 + own history in past games without the same teammate | 4.04 | **4.50** | 5.58 |
| B3 + role similarity (box-score role vectors) | 4.03 | 4.51 | **5.58** |

- B2 − B1 on absence games: −0.050 min (95% CI −0.065..−0.036). Absence history helps, a little.
- B3 − B2: +0.002 (−0.007..+0.010). **Player roles do not improve predictions** at this stage.
- Selected by the pre-set rule (validation single + multi MAE): B3, by 0.01 over B2. They are statistically
  indistinguishable, so **B2 is the working baseline**: simpler, and it needs no role vectors in scenarios.

## Team forecasts (2025-26, 822 games; rolling calibration fit only on earlier blocks)

| Minutes used with point-in-time DUNCAN ratings | Log loss |
|---|---|
| naive: recent minutes, absent players still in | 0.5885 |
| B0: absent removed, renormalised (simple additive baseline) | 0.5772 |
| **IGGY (selected rotation)** | **0.5708** |
| realised minutes (*diagnostic upper reference only, not pregame*) | 0.5741 |

IGGY − B0 = −0.0063 (CI −0.0108..−0.0019). Realised minutes are not an upper bound here: they include garbage time
(the trailing team empties its bench), so they partly encode the result.

## Failure analysis (test, non-absent candidates; `failure_summary.json`)

| Failure | Rows | Share of total error | MAE |
|---|---|---|---|
| predicted ≥ 5 min but did not play (coach's decision, or an injury below the rotation threshold / not reported) | 2,630 | 14.4% | 9.2 |
| big upward swing (actual ≥ predicted + 12) | 1,502 | 14.9% | 16.6 |
| big cut (actual ≤ predicted − 12: blowout, foul trouble, in-game injury) | 623 | 5.9% | 15.8 |
| no recent minutes (new arrival, return from injury) | 98 | 0.8% | 13.1 |

Blowouts matter little on average (MAE 4.83 for games decided by 0–10, 5.02 for 20+). Next fixes, in order:
injury-report availability for non-rotation players; a returning-player rule; a playing-time prior for new arrivals.
No synergy terms until this baseline is improved (per the goal).

## Example predictions

`results/iggy/examples.json`: 40 test team-games with each player's recent minutes, prediction and actual minutes.
`failures.json`: the 40 largest errors.
