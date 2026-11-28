# BOWEN prospective test: protocol (recorded 2026-11-28, before any live game was converted or scored)

**Prospective.** The models are frozen as already trained; no part of this test has seen the games it is scored on.
Nothing here touches DUNCAN-2 or its forecasts. Code: `bowen_prospective.py`. Output: `results/world/bowen_prospective/`.

## Question

Do the BOWEN v2/v3 results from the retrospective cross-fit (BOWEN_V2.md) hold on new games, captured live after the
models were trained? Retrospective (live frames, 900 games, 95% game bootstrap): B − A = −0.31 NLL, V3 scheme − B =
−0.011, V3 scheme + player − B = −0.064.

## Frozen models

Trained on Modal (app `bowen-v2-v3`, data through 2026-11-22), downloaded from volume `bowen-world` to
`results/world/frozen/` (git-ignored; the volume keeps the originals). SHA-256:

| File | SHA-256 |
|---|---|
| bowen_live/world_fold0.pt | 13da0fae5501fadf85bc0ea25c7377e30ae44385029ed5b7b9e9d5f36e920b9d |
| bowen_live/world_fold1.pt | 51c4462dcb8feab51f92fddba36bfd0be0f0d0b8235f5e4e0dc11d776aa60141 |
| bowen_live_mix3/world_fold0.pt | b2a757cec8b8183d4c795b7c64bf48ed6297976958fa65f0d5513dd42909b90c |
| bowen_live_mix3/world_fold1.pt | 2a547cfd7e061229a1aca57bdb46bedec159e8c0b028d0d70f0977e4c875bcf1 |
| bowen_v3_hier/world_fold0.pt | a2d4fdc895aac84c6e7917dd23bcea1c45642bc1c791c4b887b120d182a0cdcc |
| bowen_v3_hier/world_fold1.pt | 2f463ead6f898e2b3f86f9526a99a0e1e08bc52d8102871261221f4d96e7dfb9 |

Player descriptors: `desc.json` as trained (sha256 prefix 6b9989c16887ffab). v1 is not included: its weights are on
the PC only. It may be added only before the first score is computed, after its own hash and gate.

## Reproduction gate (passed 2026-11-28, `results/world/bowen_prospective/gate.json`)

For each model, each fold network re-predicted 150 randomly chosen possessions of the fold it held out in training;
every row matched the saved `pred.npz` one-to-one (30,335 rows per model). Largest differences: spots 0.031 ft, NLL
0.008, both float16 storage rounding (tolerances 0.05 ft, 0.02). Possession cutting for live games is the training
code itself (`traj_export.game`, through a temporary zip); on two archived games it gave identical possessions.

## Readouts on a new game

Both fold networks predict, and the readout is their equal-weight ensemble: density = average of the two densities
(NLL = −log of the mean likelihood), spot = average of the two spots (each the most likely mixture component, as
retrospectively). Team-season, team and defender embeddings are the trained ones (2026-27 rows from games through
11-22); a defender or team with no trained row gets the zero row (league + scheme only). Readouts: A scheme, B scheme,
B league (no-scheme token), V3 scheme, V3 scheme + player.

## Games and frames

- Regular-season games on or after 2026-11-27 with live-captured positions, converted by court-live `convert_live.py`
  (GitHub capture plus PC capture when present), with at least 30 possessions after `traj_export.game`. Games below
  that are listed as skipped with their count.
- Frames: all ten players located, live play only (same dead-ball rule as training and BOWEN_V2.md).
- Pipeline fixes found by the input check (counts only, no model output) are allowed before the first score and
  are recorded here; the models and these rules are not changed.

## Comparisons (fixed now)

- **Primary:** V3 scheme + player − B scheme, mean live-frame NLL, paired game bootstrap (2,000 draws, seed 0), 95% CI.
  "Holds" only if the CI is entirely below 0.
- **Secondary (reported, not decisive):** V3 scheme − B; V3 scheme + player − V3 scheme; B − A; B scheme − B league;
  median error, 90th percentile, share > 15 ft per readout.
- **When:** one decisive analysis once 150 eligible games exist (about two weeks of captures). Before that only
  pipeline counts (games captured, converted, eligible, possessions) are looked at, not model errors or NLL. A
  season-end update is reported as additional evidence and does not replace the 150-game result.
- All results, including negative ones, are kept and summarised in BOWEN_V2.md and AUDIT.md.

## Limitations

- **Feed source.** The 2026-27 training data came from the CourtSketch archive; live games come from the Game Pulse
  companion feed. Conversion is validated against the 2025-26 Game Pulse archive (100% field match, replay_fixture.py),
  not against CourtSketch. The input check compares possession counts, lengths, holder and shot-clock coverage with
  the training season before scoring.
- **Embedding staleness.** Team and defender terms stop learning at 11-22; players new to a team or the league get
  only the league + scheme part. This is part of what is tested, not corrected.
- **What NLL measures.** Prediction of where defenders actually stood, not whether the scheme codes match coaching
  intent (that is the separate human review).
