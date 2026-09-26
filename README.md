# Kaggle Competitions

Working notes and code for three Kaggle competitions. Each directory holds a
`GUIDE.md` — the full analysis, every measurement taken from the actual data rather
than quoted from the competition page — and, where the work is finished, a
`RESULTS.md`.

Competition data is not in this repository. See [Data](#data).

---

## [LANL Earthquake Prediction](LANL_EQ_Prediction/) — complete

Predicting time-to-failure of laboratory earthquakes from a 3.74 MHz acoustic
stream. Closed 2019, entered by late submission.

**[→ RESULTS.md](LANL_EQ_Prediction/RESULTS.md)**

| | |
|---|---|
| Best private MAE | **2.65160** |
| Constant-median floor | 2.8423 |
| Competition winner | 2.26589 |
| Metric | MAE |

The training file has 629,145,480 rows and **15 independent observations** — the
`time_to_failure` sawtooth contains only 15 complete stick-slip cycles. Every
decision follows from that.

Two findings worth the click:

- **The leaderboard lies, measurably.** Across four submissions,
  `corr(CV mean, public LB) = +0.9986` and `corr(public LB, private LB) = −0.7884`.
  The best public score had the worst private score. This is the competition whose
  winner finished 354th on the public board.
- **Four well-motivated improvements produced nothing.** Model-specific feature
  selection made the private score *worse*; 6× more training windows changed the CV
  to four decimal places; a physics-motivated b-value feature was redundant by
  construction; frequency-domain features had the strongest correlations of anything
  tried (+0.562) and added 0.034 of R². The problem is information-limited, and
  section 7 of RESULTS.md measures why.

```
baseline.py   constant predictor + leave-one-earthquake-out CV harness   CV 2.842
gbm.py        27 window features, cached per stride; LightGBM            CV 2.217
select.py     greedy backward elimination, per model                     CV 2.123
models.py     GBM / NuSVR / Ridge, log-target and blend wrappers         CV 2.036
```

---

## [BirdCLEF+ 2026](BirdCLEF/) — modelled locally, not yet submitted

Identifying 234 species (birds, amphibians, mammals, reptiles, insects) from 5-second
windows of passive acoustic monitoring in the Brazilian Pantanal. Closed June 2026.

**[→ RESULTS.md](BirdCLEF/RESULTS.md)**

| | |
|---|---|
| Best local CV | **0.8881** |
| Zero-shot BirdNET | 0.6168 |
| Competition winner | 0.96574 |
| Metric | macro-averaged ROC-AUC, skipping classes with no true positives |
| Constraint | **CPU notebook ≤ 90 minutes, GPU disabled, no internet** |

The dataset fact that shapes everything: `taxonomy.csv` defines 234 classes,
`train.csv` contains 206, and **28 classes have no focal training audio at all** — all
25 insect sonotypes and 3 amphibians. 72 non-bird classes are 31% of a macro-averaged
metric and share 750 focal clips, while 162 bird classes have 34,799.

Two findings worth the click:

- **BirdNET's embeddings are not bird-specific, its classifier is.** Zero-shot, the 72
  non-bird classes score exactly 0.5000 — they have no output column. A linear probe
  on the penultimate layer takes them to **0.8553**, close to what the same probe gets
  on birds. That removes the need for the separate non-bird detector the analysis had
  called for. Blending the logits back in helps only on the 28 bird columns; doing it
  everywhere *costs* 0.066, because rank-averaging a constant is dilution.
- **333 hours of focal audio made it worse.** Four hours of extraction, 131,943
  segments, every configuration below the 0.8756 you get by ignoring it — and the
  sweep is monotone toward discarding it. The loss is concentrated on birds
  (0.9098 → 0.8021), which is the tell: the embeddings already encode focal bird
  audio, so the extra rows contribute domain shift and little else.

---

## [Enveda CASMI 2026](MoleculeID-from-Mass-Spectra/) — live, closes 14 Dec 2026

Predicting 2D molecular structure (SMILES) from tandem mass spectra. Up to 25 ranked
candidates per molecule.

| | |
|---|---|
| Metric | MRR@25, matched on InChIKey14 after RDKit tautomer canonicalisation |
| Data | 2,539,608 spectra, 275,810 unique structures |
| Constraint | 9 h CPU or GPU, no internet |

[GUIDE.md](MoleculeID-from-Mass-Spectra/GUIDE.md) covers the three novelty classes
(spectral-library retrieval, database retrieval, de novo generation) and the approach
ladder for each. The measurement that changes the preprocessing:

> Test spectra have a median of **230 peaks**. The training libraries whose chemistry
> best matches the test set — `riken` (plant metabolites), `massbank`, `msdial` — have
> medians of **9, 11 and 11**. They ship heavily thresholded.

Symmetric cosine similarity therefore fails against exactly the references that matter
most, because the test spectrum carries ~220 peaks the reference never recorded.

---

## Data

Nothing under `data/` is committed. The three datasets total about 33 GB, and
redistributing them would breach the competition rules and dataset licences (LANL:
competition rules; BirdCLEF: CC BY-NC-SA 4.0; Enveda: CC BY-NC 4.0).

```bash
./fetch_data.sh              # all three
./fetch_data.sh molecule     # or one: molecule | birdclef | lanl
```

Requires Kaggle credentials at `~/.kaggle/kaggle.json` and each competition's rules
accepted on kaggle.com first, or every request returns 403.

The script uses `curl -C -` rather than the Kaggle CLI: it resumes, and it streams to
disk instead of buffering the whole file in memory. Kaggle serves large single files
zipped even when the URL ends in `.csv`, so the script checks the `PK` magic bytes and
expands when needed.

## Environment

Python 3.10, CPU only, no GPU anywhere. Full install, per-step RAM and wall-clock
figures, and the two macOS caveats are in **[SETUP.md](SETUP.md)**.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Runs on Linux and on macOS Apple Silicon. Intel Macs can run LANL and MoleculeID but
not BirdCLEF — `ai-edge-litert` ships no Intel-Mac wheel.

Everything is written to a 15 GB laptop's budget, because that is what it was built
on. Reading LANL's 9.1 GB CSV naively exhausts RAM, so `baseline.py` works from
memory-mapped `.npy` caches; loading Enveda's `train.parquet` whole peaks near 18 GB,
so `library.py` streams it one row group at a time. `guard.sh` kills a runaway job
before it takes the machine with it:

```bash
MIN_AVAIL_MB=2500 ./guard.sh python3 -u MoleculeID-from-Mass-Spectra/library.py
```
