# Pregame availability benchmark: protocol (recorded 2026-11-28, before any results)

Development evidence only (HOLDOUT.md rule 3: games before 2026-11-27). Nothing here changes live DUNCAN-2 or its
forecasts. Code: `fetch_injury_history.py` (reports), `avail_bench.py` (benchmark). Output `results/avail/`.

## Question

Does IGGY's frozen B2 minutes recipe, fed only pregame information (the official injury report published before
tip), improve team forecasts over the live system's injury rule? Until now IGGY was evaluated only conditional on
observed absences (IGGY_MINUTES.md, AUDIT.md "data blockers").

## Inputs (identical for every arm)

- **Reports:** for each date, the newest official report whose **header as-of time** is <= min(5:30 PM ET, first
  tip that day − 45 min). File names are not times (an hourly `05PM` file is the 5:30 PM report). `--verify` checks
  every report and re-fetches earlier ones when needed. The as-of time is not the posting time; posting lag is unknown,
  and the 45-minute margin is meant to cover it.
- **Rows:** only rows under that day's game date (reports also list the next day's games).
- **Team-game report state:** `valid` (team appears under the date and is not NOT YET SUBMITTED), `not_submitted`,
  `absent_from_report`, or `no_report` (no valid report for the date). Only `valid` team-games count. A player on a
  `valid` team who is not listed is **unlisted** (status none), which is different from missing.
- **Candidates and history:** frozen IGGY definitions (`iggy_minutes.build`): players with a game for the team in the
  prior 30 days, same-season previous-10 history, first 10 team games not covered.
- **Ratings:** point-in-time DARKO DPM, the latest value dated before the game (`darko_history.json`; point-in-time
  status unverified, AUDIT I6). Players with no value get −1, as in `live_forecast.py`.
- **Margin:** sum over the team of share × DPM, home − away, where share = 5 × minutes / team total (same as
  `live_forecast.shares`). Win probability: logistic fitted per arm on the validation season's eligible games.

## Arms

| Arm | Minutes | Status |
|---|---|---|
| P_none | recent minutes (B0 base), no report | reference |
| **P0** | recent minutes × fixed share (out 0, doubtful 0.15, questionable 0.6, probable 0.95, else 1), normalised to 5 on court | **proxy** of the live rule (`shares()`): live uses DARKO projected minutes, which are not archived, so this is not the current live system |
| **B2** | frozen IGGY B2 (`iggy_minutes.FEATS['B2']`, same training seasons ≤ 2024, same rows, hyperparameters and seed), applied pregame: absent set = rotation players listed **Out**; features recomputed with that set; minutes projected as frozen | existing recipe; the only arm that tests B2 |
| L1 | new: gradient boosting of minutes (zeros included) on recent minutes, availability, rank, status; trained 2022-23 + 2023-24 | new development model |
| L2 | new: L1 + listed teammates' minutes, counts and own past response | new development model |

L1 and L2 are new models and are not evidence about B2.

## Eligible games

Validation 2024-25 and test 2025-26 + 2026-27 games before 2026-11-27, where both team-games are `valid`, both teams
have candidate rows, and home and away are identified. Every arm is scored on exactly this set.

## Comparisons (fixed now)

- **Primary:** B2 − P0, test log loss, paired game bootstrap (2,000 draws, seed 0), 95% CI. B2 "improves" only if
  the CI is entirely below 0.
- **Secondary (reported, not decisive):** P0 − P_none (does the report help under fixed shares); L1 − P0; L2 − B2;
  margin RMSE; minutes MAE (all candidates, zeros included) per arm; status table (how often listed rotation players
  play, by status, train vs test).
- **Coverage (reported):** dates with and without a valid report, re-fetched reports, minimum lead before first tip,
  team-games by report state, listed rows not matched to any candidate (by status).
- No arm, rule or eligibility filter is changed after seeing test results; all results, including negative ones, are
  kept in `results/avail/` and summarised in IGGY_MINUTES.md and AUDIT.md.

## Where it runs

B2's frozen features include the start rate from the PC's stint files (`ms2_stints*.pkl`), so the full benchmark runs
on the PC. A laptop smoke test (start rate set to 0) only checks the code and writes to the scratchpad; its numbers are
not reported.
