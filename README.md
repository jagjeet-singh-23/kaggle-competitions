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

## [BirdCLEF+ 2026](BirdCLEF/) — submitted, 0.92681

Identifying 234 species (birds, amphibians, mammals, reptiles, insects) from 5-second
windows of passive acoustic monitoring in the Brazilian Pantanal. Closed June 2026.

**[→ RESULTS.md](BirdCLEF/RESULTS.md)**

| | |
|---|---|
| Private LB, Perch v2 kernel | **0.92681** |
| Private LB, same pipeline on BirdNET | 0.87008 |
| First submission | 0.83575 |
| Local CV, 75 of 234 classes, Perch vs BirdNET | 0.9144 vs 0.8880 |
| Competition winner | 0.96574 |
| Metric | macro-averaged ROC-AUC, skipping classes with no true positives |
| Constraint | **CPU notebook ≤ 90 minutes, GPU disabled, no internet** |

The dataset fact that shapes everything: `taxonomy.csv` defines 234 classes,
`train.csv` contains 206, and **28 classes have no focal training audio at all** — all
25 insect sonotypes and 3 amphibians. 72 non-bird classes are 31% of a macro-averaged
metric and share 750 focal clips, while 162 bird classes have 34,799.

Seven levers were tried on BirdNET after the first submission, then the encoder was
swapped:

| lever | result |
|---|---|
| 333 h focal audio, where labels already existed | −0.05 |
| focal audio for the 30 classes that had none | **+0.021** |
| 177 h in-domain pseudo-labels | +0.004, 1.6σ |
| nonlinear head, 24 configurations | 24/24 below linear |
| all 6,522 BirdNET logits instead of 157 | −0.012 |
| temporal smoothing across a recording | **+0.013** |
| focal calls mixed into soundscape backgrounds | ±0.000 |
| **Perch v2 in place of BirdNET, frozen, everything else unchanged** | **+0.057** |

Three findings worth the click:

- **The ceiling was BirdNET's, not the frozen encoder's.** Every attack on BirdNET's
  features failed, which looked like a ceiling only fine-tuning could lift. Swapping
  in Perch v2, still frozen, added +0.057 private. Its classifier knows 203 of the 234
  classes, against BirdNET's 157, so classes with no prediction route fell from 30 to
  2. The gain came from Perch's logits: a head on its embedding alone beats BirdNET's
  by only +0.008.
- **The local CV moved by 0.02 between machines.** Ranking with `argsort(argsort())`
  broke ties among fold-constant predictions in whatever order the platform's sort
  left them. The same blend scored 0.8881 on x86 and 0.8684 on ARM; with ties averaged
  it is 0.8794 on both. The leaderboard was never affected.
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
| Public LB | **0.213** |
| Leader | 0.425 |
| Metric | MRR@25, matched on InChIKey14 after RDKit tautomer canonicalisation |
| Constraint | kernels-only, 9 h, no internet |

**[→ RESULTS.md](MoleculeID-from-Mass-Spectra/RESULTS.md)**

Measured on a held-out split that simulates the three novelty classes:

| held-out class | library search | + analog propagation |
|---|---|---|
| 1 — spectra exist in another library | 0.9390 | **0.9420** |
| 2 — structure listed, no spectra | 0.0000 | **0.0676** |
| 3 — structure absent entirely | 0.0000 | 0.0000 |
| mean | 0.3130 | **0.3365** |

Class 3 is 0 *by construction* and cannot be otherwise — the structure is banned from
the candidate list, so no retrieval method reaches it. Class 2 was 0 for the same
reason until analog propagation: score a candidate by how similar it is to the
best-matching structure that *does* have spectra, inside a 5 ppm neutral-mass window.

Three findings worth the click:

- **The visible test set cannot measure anything.** It ships with answers, and the
  submission scores **0.9975** against them. All 400 visible molecules were located in
  `train.parquet` by matching `(precursor_mz, adduct, num_peaks)` — it is entirely
  class 1, while the hidden test is a mix of all three. Tuning against that number is
  tuning against a set that does not contain the problem.
- **Class 3 is 73.5% of the test, and three submissions say its candidate pool is
  finished.** Four submissions with known per-class scores solve the class mix
  exactly. Better features lifted class 2 by 43% relative and moved real class 3 by
  nothing (0.0617 → 0.0610). A PubChem build of 8.4M structures moved held-out
  coverage 8.0% → 9.5% and cost ranking quality 0.366 → 0.269. ChEMBL, 2.6M structures
  of exactly the right chemistry, lifted held-out class 3 by 32% and took the
  leaderboard from **0.213 to 0.187** — which solves to real class 3 falling 0.0610 →
  0.0352, on a pool that is a strict superset. Coverage cannot fall, so quality did,
  at 0.16 per doubling against the split's 0.065. As elasticities against pool size:
  coverage +0.50, quality −0.86. An expansion pays only if nearly every structure it
  adds is one some query needs, and no general database is like that.

- **A 0.5% key mismatch was leaking the split.** The dataset ships an `inchikey14`
  column; the metric scores on the key RDKit derives from the SMILES. They disagree on
  amide/imide tautomers, so structures held out by dataset key stayed reachable, and
  class 3 scored 0.0050 when it must be 0. Canonicalising everything through RDKit
  merged 1,148 structures and dropped class 1 from 0.9643 to its honest 0.9390.

- **One analog seed beats a hundred.** Using only the single best spectral match as a
  seed scores 0.0587 on class 2; ten seeds give 0.0535 and a hundred give 0.0510.
  Seeds past the first are worse matches whose analogs dilute the ranking — and
  *summing* their evidence instead of taking the max drops class 1 from 0.9415 to
  0.8882.

The measurement that shapes the preprocessing: test spectra have a median of **230
peaks**, while the libraries whose chemistry best matches them — `riken`, `massbank`,
`msdial` — have medians of **9, 11 and 11**. Symmetric cosine therefore fails against
exactly the references that matter most.

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
