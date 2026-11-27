# BOWEN validation: what the outputs mean, and the human-review handoff (2026-11-27)

## What BOWEN sees, and what its outputs may be called

`world_model.py` predicts where defender *k* is at each frame (a 2-D Gaussian) from: the five attackers (position,
velocity, ball-holder flag, an 11-number descriptor of shooting threat and role), the ball, the other four defenders,
the shot clock and time into the possession, plus a conditioning token = an 8-dim **scheme code** (VAE over the
*whole possession* of the other four defenders and the attackers) and a **team-season embedding**. With probability
P_DROP the token is replaced by a "no scheme" token, which gives the league-baseline spot.

It does **not** see: defender *k*'s own position or identity, the coaching call, the matchup assignment, the
score/time situation, or anything outside the tracked frames.

So the outputs are, and from now on are labelled:

- **expected positioning**: the scheme-conditioned spot (and the league-baseline spot)
- **positioning deviation**: actual spot minus expected spot (distance, and signed depth toward the rim)

Not yet justified: "execution quality", "mistake", "out of position". A deviation can be an error, an acceptable
alternative (sinking vs staying attached), a deliberate assignment the model cannot see, or a player habit.

Known limitations of the residual's meaning:
1. **Hindsight in the expected spot.** The scheme encoder is bidirectional over the whole possession
   (`world_model.py:54,72-78`), so it sees how teammates reacted later, including to *k*'s own mistakes. Fix: a
   causal encoder (frames ≤ t). Not yet run (GPU queue).
2. **No identity, so tendencies are absorbed into the deviation.** Player averages are stable (split-half 0.72–0.83),
   which shows a tendency, not quality.
3. **Team scheme is built in.** "Scheme follows the new team" is r 0.887 even leave-self-out, but the spot is
   conditioned on a team-season embedding, so this is largely by construction.

## Non-human diagnostics (`bowen_diagnostics.py` → `results/bowen_review/diagnostics.json`)

| Check | Result | Reading |
|---|---|---|
| Scheme follows the new team, leave-self-out (72 movers) | r 0.887 (CI 0.83–0.93); including self 0.905; old team −0.36 | Not driven by including the player; still by construction |
| Possession-level deviation explained, held-out games | situation features 17.1%; + player identity 17.7% | Mostly situation and noise per possession; identity adds little per possession |
| Same player across situations | on-ball vs help r 0.38 (0.32–0.44); on-ball vs perimeter 0.01; help vs interior −0.02; perimeter vs help 0.20 | Partly a trait, partly situational |
| Deviation → points beyond location xFG (43.5k shots, `bowen_value.json`) | coefficients ≈ 0 in both folds | No link to points scored |

## Review package

- **Page:** https://claude.ai/artifact/QH6grxDyArSL8Eta1f1Rik (private; share from its Share menu with the
  reviewer as **Contributor** so they can save labels)
- **Set:** `results/bowen_review/sample.json`, 100 possessions: 4 situations (on-ball, interior, perimeter
  stay-home, help/rotation) × 3 deviation tertiles × 8, plus 4 random. Out-of-fold predictions only. Random order,
  no BOWEN values, no strata, no game ids or dates on the page.
- **Key (private, never published):** `results/bowen_review/key.json`
- **Rubric:** on the page ("How to review"). Fields: situation, assignment (attacker 1–5 / zone / switched / can't
  tell), acceptable alternatives, execution quality 1–5 judged against the acceptable options, uncertainty plus reasons,
  notes.
- **Storage:** `reviews/<reviewer>/items/<item>`. Each reviewer sees only their own labels; the owner reads all. If
  saving fails, the page keeps labels in the browser and "Copy my labels" exports them.
- **Comparison:** `bowen_review_compare.py`, with the interpretation rule fixed before any labels exist.

## Handoff

- **Owner:** Luke. Recruit at least two reviewers, ideally one coach or scout and one analyst (Jaden counts as an
  analyst), so inter-rater agreement can be measured.
- **Effort:** about 1–2 minutes per possession, so 2–3 hours per reviewer for 100.
- **Trigger for the comparison:** when two or more reviewers have finished, ask Claude to "run the BOWEN review
  comparison". Claude exports the labels with the ArtifactData tool to `results/bowen_review/labels.json` and runs
  `bowen_review_compare.py`.
- **Until then:** all BOWEN outputs stay labelled expected positioning / positioning deviation, on the site and the
  player page.
