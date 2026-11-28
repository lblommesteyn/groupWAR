# IGGY shadow forecast: protocol (recorded 2026-11-28, before the first shadow forecast)

**Prospective, separate from the DUNCAN-2 test.** Version `iggy-b2-shadow-1`, registered in court-live HOLDOUT.md.
It writes its own files (`forecasts/shadow/{date}-pm-iggy.json`) and never changes, replaces or delays a DUNCAN-2
forecast. Code: `iggy_shadow.py`. The retrospective pregame benchmark (AVAIL_BENCH.md) is separate development evidence.

## Question

Given the same ratings, the same injury report and the same frozen calibration, does IGGY's frozen B2 playing time
give better pregame win probabilities than the live rule (DARKO projected minutes × status share, normalised)?

## What the shadow does

After the PC's evening forecast (`forecasts/{date}-pm.json`) is written, for the same games:
- **Ratings:** that run's DUNCAN-2 ratings (`live_ratings_{date}.npz`; the shadow stops if the hash differs from the
  forecast's `ratings_hash`) and the DARKO table saved by that run (`data/darko/{date}.json.gz`; missing values −1, as
  live). Calibration: the forecast's own frozen maps.
- **Report:** the same PDF as the pm forecast, parsed with the benchmark parser (`fetch_injury_history.parse_text`,
  game date only) and matched with `avail_bench.match`. A team that is NOT YET SUBMITTED or absent from the report
  gets no absences and is flagged.
- **Minutes, primary `iggy`:** frozen B2 exactly as the benchmark arm: candidate rows from the frozen
  `iggy_minutes.build` code for a game today, absent set = rotation players (≥ 10 minutes over the last 10 games)
  listed Out, B2 features with that set, projection to 240 minutes, share = minutes / 48.
- **Minutes, variant `iggy0` (new, not B2):** the same, but every player listed Out is set to 0, rotation or not.
  Added after a code check on 2026-11-27 inputs showed frozen B2 gives non-rotation players listed Out up to about 26
  minutes per game (summed over both teams). No outcome was looked at; this variant is scored separately and is not
  evidence about B2.

## Model file

`results/iggy_shadow/b2_frozen.pkl`, trained once on the PC (`python iggy_shadow.py train`) exactly as
`iggy_minutes` / `avail_bench` (seasons ≤ 2024, realised absences, same hyperparameters and seed). It may be trained
only after the AVAIL_BENCH reproduction gate passes, since that gate is what shows the PC rebuild matches the
saved B2. Its SHA-256 is added to this file and to HOLDOUT.md before the first shadow forecast. Every shadow file
records that hash, the pm forecast's SHA-256 and the report PDF's SHA-256.

## Scoring (fixed now)

- **Games:** every game with a pm PC forecast and a shadow forecast that both have a valid pre-tip receipt
  (`forecast-receipt.yml`, same rule as the live scorer). Days the PC fails have no shadow, and those games are not
  scored. Skipped games (no IGGY candidates) are listed.
- **Primary:** log loss of `ours_iggy_p` − `ours_inj_p` (same DUNCAN-2 ratings, only the minutes differ), paired game
  bootstrap (2,000 draws, seed 0), 95% CI. IGGY "improves" only if the CI is entirely below 0.
- **Secondary (reported, not decisive):** `darko_iggy_p` − `darko_inj_p`; the same two for `iggy0`; Brier scores;
  margin RMSE against final margins.
- **When:** one decisive analysis at 400 scored games. Before that only counts are reported (forecasts written,
  receipts, skipped), not outcomes. A season-end update is reported as additional evidence and does not replace it.
- No rule, model or calibration is changed after any outcome is seen. Negative results are kept and summarised in
  IGGY_MINUTES.md and AUDIT.md.

## Limitations

- The calibration maps were fit for DUNCAN-2 with DARKO minutes; IGGY minutes shift the margins they receive. Using
  them unchanged is a choice fixed in advance, not tuned.
- IGGY needs 10 same-season team games, so it covers the season from now on, not the first games.
- The shadow runs only on the PC (start rates come from its stint files; ratings from its morning fit).
