# Enveda CASMI 2026 — Results

Predicting 2D molecular structure (SMILES) from tandem mass spectra, up to 25 ranked
candidates per molecule. Metric is MRR@25, matched on the first block of the InChIKey
after RDKit tautomer canonicalisation. Competition is live, closes 14 December 2026.

Rung 1 (spectral library search) and Rung 2 (analog propagation) are built and
measured. Nothing has been submitted.

| held-out class | molecules | library search | + analog propagation |
|---|---|---|---|
| 1 — spectra exist in another library | 200 | 0.9390 | **0.9420** |
| 2 — structure in the candidate list, no spectra | 200 | 0.0000 | **0.0676** |
| 3 — structure absent entirely | 200 | 0.0000 | 0.0000 |
| mean | | 0.3130 | **0.3365** |

Class 3 is 0 *by construction* and must stay there — its structure is banned from the
candidate list, so no retrieval method can reach it. Class 2 was also 0 for library
search by construction, and Rung 2 is what moves it.

Whether 0.3365 resembles a leaderboard score depends on the hidden test's class mix,
which is not published. The mean weights the three equally, which is an assumption,
not a measurement.

---

## 1. The visible test set cannot be used to measure anything

`test.parquet` ships with answers. Scoring the submission against them gives
**0.9975**. That number is worthless: all 400 visible molecules were located in
`train.parquet` by matching `(precursor_mz, adduct, num_peaks)`, so the visible test
is entirely class 1, and a library search retrieves what it already has.

The hidden test is a mix of all three novelty classes. Any tuning against 0.9975 is
tuning against a set that does not contain the problem.

That is why section 2 exists.

## 2. A held-out split that simulates the three classes

`split.py` holds out `inchikey14` groups from the index:

- **class 1** — structure has spectra in ≥2 source libraries. One library's spectra
  become queries; the others stay in the index.
- **class 2** — every spectrum of the structure is held out, the structure stays in
  the candidate list.
- **class 3** — every spectrum is held out *and* the structure is banned from
  candidates.

Grouping is on the structure, never on spectra. The median structure has 4 spectra
(max 1,084), so splitting by row would leave three collision energies of the same
molecule in the index and one in the query set — class 1 wearing a class 3 label.

Class 1 at 0.9390 is the optimistic end of class 1. Holding out one *library* means
the query and the surviving reference differ by source, but often not by instrument
or collision energy. A real class 1 molecule may be further from its nearest
reference than this measures.

## 3. The bug this split found

The first run returned class 2 = 0.0150 and class 3 = 0.0050. Both must be 0 by
construction, so a non-zero number is not a lucky hit — it is a leak.

Cause: the dataset ships an `inchikey14` column, and `split.py` grouped and banned on
it, but the competition metric scores on the key **RDKit** derives from the SMILES.
Those disagree on **0.5% of rows** — almost all amide/imide tautomers, where the two
standardisers pick different forms:

```
col=CPYBTQCZANHDQS  rdkit=MXNULNSJRBWSTM  COC(=O)c1cccc(NC(=O)Cn2[nH]cc3c(=O)ncnc2-3)c1
col=AFSQLDVCZNYCBV  rdkit=VEDFLRCGIUMWPR  CC(C)(NC(=O)CC1CC(=O)NC1=O)c1cccc(C(F)(F)F)c1
```

So a structure held out by dataset key could still sit in the index under a row whose
dataset key differed but whose RDKit key matched, and a "banned" class 3 structure was
still reachable. Canonicalising every structure through RDKit
(`library.canonical_keys`, cached in `data/key14.npz`) merged **1,148** structures —
265,475 became 264,327 — and both classes went to exactly 0.

Class 1 fell from 0.9643 to 0.9390 in the same change. The leaky split had been
flattering it.

`search.py` now asserts `class 3 == 0`, so this cannot come back silently.

## 4. Both sides of a cosine must be curated the same way

Test spectra carry a median of **230 peaks**. The libraries whose chemistry best
matches the test set — `riken`, `massbank`, `msdial` — have medians of **9, 11 and
11**. They ship heavily thresholded.

`evaluate()` draws its queries out of the index, so it compares trimmed against
trimmed. `submit()` originally fed raw test spectra straight in, which is a different
comparison: a 230-peak query is normalised over ~220 peaks the reference never
recorded, so its cosine against exactly the most relevant references is low. Both
paths now run through `library.trim`.

This is a patch, not a solution. Symmetric cosine remains the wrong similarity for
this asymmetry, and a reverse or hybrid score that only charges for reference peaks
absent from the query is the obvious next step.

## 5. Rung 2: analog propagation reaches class 2

Library search can only return a structure that *has spectra* in the index. Class 2
molecules have none by definition, so they score 0 no matter how good the similarity
function gets. Class 2 and 3 together are 69% of the problem.

Class 2 is the reachable half — the structure is in the candidate list, it just has no
spectra. Fragmentation is mostly a property of the scaffold, so a spectrum that
matches structure S is evidence for molecules that *look like* S:

```
score(c) = spectral(c) + max over seeds s of [ spectral(s) * tanimoto(c, s) ]
```

The candidate pool is every structure whose neutral mass fits the query precursor
under any adduct of the right ion mode. The first term is library search unchanged,
so class 1 cannot regress; the second is the only term a class 2 molecule can score
on.

**One seed beats many.** Sweeping the number of spectral matches used as seeds:

| seeds | aggregation | class 1 | class 2 | mean |
|---|---|---|---|---|
| **1** | max | **0.9415** | **0.0587** | **0.3334** |
| 10 | max | 0.9399 | 0.0535 | 0.3311 |
| 25 | max | 0.9399 | 0.0512 | 0.3304 |
| 100 | max | 0.9399 | 0.0510 | 0.3303 |
| 100 | sum | 0.8882 | 0.0500 | 0.3127 |

Seeds beyond the first are worse spectral matches, and their analogs only dilute the
ranking. Summing across seeds is worse than taking the max, and badly so at 100 seeds
— accumulated weak evidence outranks the correct answer.

**The mass window should be proportional, not fixed.** The index is filtered to
|precursor error| ≤ 10 ppm, and 10 ppm is 0.004 Da at m/z 400 but 0.0125 Da at 1250,
so a constant window is too loose for light molecules and too tight for heavy ones.
Switching to ppm and sweeping, at one seed:

| window | median pool | class 1 | class 2 | mean |
|---|---|---|---|---|
| 20 ppm | 385 | 0.9419 | 0.0586 | 0.3335 |
| 10 ppm | 206 | 0.9420 | 0.0629 | 0.3350 |
| **5 ppm** | **115** | **0.9420** | **0.0676** | **0.3365** |
| 2 ppm | 36 | 0.9421 | 0.0399 | 0.3273 |

5 ppm is an interior optimum, not the end of a monotone trend: at 2 ppm the window
starts excluding the true structure faster than it excludes wrong ones. That shape is
what makes it a real choice rather than a tuning artifact.

For scale, a median pool of 115 competing for 25 slots would give MRR ≈ 0.033 if the
answer were placed at random. 0.0676 is about twice that — Tanimoto to the best
spectral match is genuinely informative, and genuinely weak.

**Restricting to the common adducts hurts, and the reason matters.** Keeping only
`[M+H]+`/`[M-H]-` cuts the median pool from 115 to 23 but drops class 2 to 0.0614.
Those two adducts are 95% of the *real test*, but the held-out queries are drawn from
the index, whose adduct mix is the library's, not the test's. So this measurement
understates what the real submission gets: `test.parquet` ships an `adduct` column, so
`analog.py --submit` opens the window for one adduct rather than five, and its pool is
tighter than anything the held-out split can show.

## 6. What it cost

| step | output | peak RAM | wall clock |
|---|---|---|---|
| `library.py` | 1,888,517 spectra, 1.3 GB | 4.9 GB | ~25 min |
| `canonical_keys` (once, cached) | 267,280 keys | 1.8 GB | ~20 min |
| `split.py` | 3 × 200 molecules | 1.8 GB | ~20 min (incl. the above) |
| `search.py` evaluate | 1,958 queries | ~2.5 GB | **3 s** |
| `search.py --submit` | 400 molecules | ~2.5 GB | ~30 s |
| `analog.py --build` | 264,327 fingerprints + masses, 153 MB | ~2.5 GB | ~2 min |
| `analog.py` evaluate | 600 molecules × 8 configs | ~2.7 GB | ~6 min |

`train.parquet` is 2.9 GB on disk, 6.4 GB as Arrow, and roughly 18 GB once pandas
boxes each spectrum's peak lists into Python objects — which froze a 15 GB laptop
twice and cost two reboots. `library.py` reads one row group at a time, takes the
peak arrays as numpy views over Arrow's flat value buffer rather than via
`to_pylist()`, keeps the top 64 peaks, and writes fixed-width byte columns. Peak
memory 4.9 GB, and it resumes from whatever shards are on disk.

The 3-second search is the precursor-mass window: candidates come from a
`searchsorted` over a sorted precursor array, so a query scores against a few hundred
spectra rather than 1.9 million.

## 7. What is next

Ranked by expected value:

1. **Predict fingerprints from the spectrum.** Analog propagation reaches class 2
   only through a structure that already has spectra — it cannot score a candidate
   whose neighbourhood the library never sampled. A model mapping spectrum to
   fingerprint scores every candidate in the mass window directly, and 0.0676 against
   a random-placement 0.033 says how much room is left.
2. **A reverse or hybrid cosine**, per section 4. The peak-count asymmetry is the
   clearest remaining mismatch between the data and the scoring function, and it caps
   class 1 as well as class 2.
3. **Class 3 needs de novo generation** and nothing short of it. It is 33% of the
   problem and structurally unreachable by retrieval — the assertion in `analog.py`
   exists to keep that honest.
4. **Adduct-aware scoring.** The index keeps all ten adducts but the cosine ignores
   whether query and reference share one, so a `[M+Na]+` spectrum is compared against
   an `[M+H]+` reference on equal terms.

## 8. Files

```
metric.py     MRR@25 exactly as scored; self-checks against the rules' examples
library.py    row-group-streamed index over train.parquet -> data/index/
              canonical_keys(): RDKit key14 per structure, cached
split.py      held-out novelty split on canonical keys  -> data/splits.npz
search.py     Rung 1: precursor-windowed cosine search   0.9390 / 0      / 0
analog.py     Rung 2: mass-windowed Tanimoto propagation 0.9420 / 0.0676 / 0
```

Self-checks that need no competition data:

```bash
python3 metric.py              # metric vs the glucose example in the rules
python3 library.py --selftest  # spectrum curation
python3 analog.py --selftest   # mass windows, Tanimoto, and the class 3 ban
python3 search.py --selftest   # search and aggregation on a synthetic index
```

`metric.py` is the one to trust on a new machine: RDKit is pinned to 2026.3.3 because
tautomer canonicalisation is version-dependent, and a different version silently
changes what counts as a correct answer.
