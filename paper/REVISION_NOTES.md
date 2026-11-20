# Revision notes: JQAS DGJQAS.2026.0109 to JSA submission

Internal document for the authors. It maps every comment from the JQAS decision to what changed.
Items marked RESULT are filled from `v3/results/` once the runs finish.

## Code audit findings (not raised by reviewers, fixed before rerunning)

1. `evaluate()` in both `hockey_war/run_search.py` and `nba_war/nba_search.py` cached the
   home roster tensor with the opponent tensors, so every call after the first scored the first
   roster ever evaluated. The tournament phase ranked identical scores and the hockey WAR tables
   knocked players out of the wrong roster. v3 builds every roster tensor fresh.
2. The NBA scorer in the repo used only the first five rows of the twelve-player roster, and the
   NBA greedy loop could never accept a swap. The stochastic-rotation code behind the submitted
   basketball results is not in the repo. v3 implements deployment over all 792 lineups.
3. Hockey features were 275-dimensional in code but described as 14 in the paper. v3 uses 26
   documented hockey features and 27 basketball features (Section 4).
4. No positional quota code existed. v3 implements and reports the quotas.
5. The output head had no home-advantage term although the paper described one. v3 learns one
   and removes it for neutral-site evaluation.
6. Every NBA training graph had the same uniform adjacency, so the GCN never saw on-court time.
   v3 trains on real shared-time matrices from reconstructed lineups.
7. Several case-study numbers (Embiid to Green, Holiday, hockey 0.4562/0.4428) were hardcoded in
   `nba_war/v2/case_studies.py` ("simulated results"), and `rapm_correlation.py` falls back to
   random noise when its input file is absent. None of these numbers are reused.
8. Data facts corrected: Kaggle play-by-play ends in 2022-23 (the paper said 2023-24); Doughty
   did play at the 4 Nations (Harley replaced Theodore); "Braeden Bowman (VGK)" could not be matched to any NHL player in the roster data.

## Associate Editor priorities

| AE item | Change |
|---|---|
| 1. Role of expert judgment | Reframed as an audit (Introduction, Section 3 "What is and is not claimed"). Quotas are stated as the only positional knowledge; every locked analysis is repeated unlocked (Section 7); quota removal is a sensitivity analysis (Section 8). |
| 2. Evaluation | Out-of-sample scorer validity against home-only, RAPM-sum and linear-feature baselines (Section 6). Actual rosters scored under the model objective with per-model spread. RAPM top-k and minutes top-k baselines under identical constraints, plus a random-roster null (Section 7). Seed and refit uncertainty, sensitivity grid (Section 8). Regular-season-to-international shift tested on playoff and high-talent subsets (Section 6). |
| 3. Transparency | Problem setup section defines deployment, score, constraints, locks, notation before use. Method section gives every algorithmic step and hyperparameter. Related work added. |
| 4. Presentation | No bullet or numbered lists in the body. Every table and figure is referenced in the text. |

## Reviewer 1

| Comment | Change |
|---|---|
| No related work | Section 2 (team formation, player value and interaction, GNNs and Stackelberg games), including both suggested references. |
| Figures/tables never referenced | All referenced. |
| Kipf & Welling "symmetry" unclear | Replaced by explicit statement of permutation invariance and antisymmetry in Section 5.2. |
| Adversarial evaluation explained before deployment defined | Deployment defined in Section 3 before the game; game in Section 5.4. |
| Phase A/B unclear (top third, usage, locked) | Section 5.6 states team formation, survival rule, elimination by replacement-level WAR, greedy candidate set and stopping rule. Locks defined in Section 3. |
| Sentinel vector / all-positive WAR | Replacement-level vector defined (10th-30th minutes percentile); WAR in win-probability units (Section 5.5). |
| y-hat undefined | Defined in Section 3. |
| Constrained vs unconstrained | Defined in Section 3. |
| Evaluation thin; compare predicted values, distance to selection, roster vs opponents | Section 7: actual roster score, distance-to-selection for every player, per-opponent scores, baselines. |
| Stochastic rotation should be in methods | Generalized into the deployment prior (Section 5.3). |
| "Closer to what wins single elimination" unsupported | Claim removed. |
| Bullet-point limitations/next steps read as LLM-generated | Rewritten as prose; Next Steps section removed. |
| Figure 6 legend overlap | Figure replaced. |
| Case study 4 one sentence | Case studies rebuilt from real outputs with figures. |
| Why no spatial data for hockey | Hockey now has shot location (xG, shot distance) and zone-start share. |
| 5:1 cadence | Reversed to the Stackelberg-correct order (follower 5 steps per leader step, Danskin), justified in 5.4, sensitivity in Section 8. |
| "NHL-calibre" wording | Removed. |
| Hockey vs basketball mismatch | Both sports now use roster-sized graphs and unit mixtures; the earlier mismatch is gone. |
| Positional data in 5-man combinations | Units are unrestricted; quotas apply to the roster. Stated in 5.3. |
| RAPM correlations pooled across sports | All comparisons are within sport. |
| Current-year vs time-weighted embeddings | Single-season features; prior-season (lag 1) variant reported in Section 6. |
| Payoff/risk sentences | Removed. |
| Feature vector passed to GCN for hockey | Section 4 now parallel for both sports. |
| Graph sizes before model details | Section 5.1 first. |
| "Blend across opponents" | Equal-weight mean, Equation 2. |
| Define GCN acronym; paragraph break in 4.3 | Fixed / section removed. |

## Reviewer 2

| Comment | Change |
|---|---|
| Expert judgment critiqued but relied upon | Opening no longer claims experts are opaque; the method is positioned as an audit (Introduction). |
| Quotas added instead of diagnosing failure | Quotas are stated as minimum balance requirements; the unconstrained solution is reported and analysed (Section 8). |
| "Any real coach would reject" | Removed. |
| Locks drive agreement; verify unconstrained picks the elites | Every analysis repeated without locks; selection frequency of each locked player reported (Section 7). |
| Interactions not identifiable from data | Shuffled-graph control measures how much the interaction structure contributes out of sample (Section 6); limitation discussed (Section 9). |
| Sensitivity and uncertainty | Section 8 and per-model spread in Section 7. |
| Are top rosters separated? Score of actual roster? | Section 7. |
| RAPM top-k baseline | Section 7. |
| Stochastic rotation and reduced-usage roles | Deployment prior matches observed minutes by rank, and the deployment game can move minutes; discussed in Section 9. |
| Improves on coaches? | Not claimed (Section 3, Section 9). |
