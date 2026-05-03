# hockey_war

This folder contains the hockey-specific research scripts used by the
manuscript case studies.

## Files

- `run_search.py`: Canada roster search workflow for the hockey case studies
- `mie368stackel.py`: graph model and Stackelberg utilities used by the hockey
  search script
- `submit_search.sh`: cluster submission helper used for long search runs

## External Inputs

The scripts expect locally generated NHL data and trained model checkpoints.
Those large artifacts are intentionally not stored in git. See
`../DATA_AVAILABILITY.md` and `../REPRODUCIBILITY.md` for the source endpoints
and reproduction notes.

Typical local-only inputs include:

- NHL play-by-play and shift data
- adjusted plus-minus tables
- player metadata and eligibility tables
- trained `model_*_30.pth` checkpoint files

The curated outputs used in the manuscript are stored in `../results/`.
