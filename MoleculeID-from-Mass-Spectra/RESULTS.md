# Enveda CASMI 2026 — Results

Predicting 2D molecular structure (SMILES) from tandem mass spectra, up to 25 ranked
candidates per molecule. Metric is MRR@25, matched on the first block of the InChIKey
after RDKit tautomer canonicalisation. Competition is live, closes 14 December 2026.

Rung 1 — spectral library search — is built and measured. Nothing has been submitted.

| held-out class | molecules | MRR@25 |
|---|---|---|
| 1 — spectra exist in another library | 200 | **0.9390** |
| 2 — structure in the candidate list, no spectra | 200 | 0.0000 |
| 3 — structure absent entirely | 200 | 0.0000 |
| mean | | 0.3130 |

Classes 2 and 3 are 0 *by construction*, and that is the point of measuring them: it
is the ceiling of what library search alone can reach. Whether 0.3130 resembles a
leaderboard score depends on the hidden test's class mix, which is not published. The
mean above weights the three equally, which is an assumption, not a measurement.

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

## 5. What it cost

| step | output | peak RAM | wall clock |
|---|---|---|---|
| `library.py` | 1,888,517 spectra, 1.3 GB | 4.9 GB | ~25 min |
| `canonical_keys` (once, cached) | 267,280 keys | 1.8 GB | ~20 min |
| `split.py` | 3 × 200 molecules | 1.8 GB | ~20 min (incl. the above) |
| `search.py` evaluate | 1,958 queries | ~2.5 GB | **3 s** |
| `search.py --submit` | 400 molecules | ~2.5 GB | ~30 s |

`train.parquet` is 2.9 GB on disk, 6.4 GB as Arrow, and roughly 18 GB once pandas
boxes each spectrum's peak lists into Python objects — which froze a 15 GB laptop
twice and cost two reboots. `library.py` reads one row group at a time, takes the
peak arrays as numpy views over Arrow's flat value buffer rather than via
`to_pylist()`, keeps the top 64 peaks, and writes fixed-width byte columns. Peak
memory 4.9 GB, and it resumes from whatever shards are on disk.

The 3-second search is the precursor-mass window: candidates come from a
`searchsorted` over a sorted precursor array, so a query scores against a few hundred
spectra rather than 1.9 million.

## 6. What is next

Ranked by expected value:

1. **Classes 2 and 3 are 69% of the problem and currently score 0.** Library search
   cannot touch them by definition. Class 2 needs retrieval against the candidate
   structure list — predicted fingerprints or formula-constrained scoring. Class 3
   needs de novo generation.
2. **A reverse/hybrid cosine**, per section 4. The peak-count asymmetry is the single
   clearest mismatch between the data and the current scoring function.
3. **Analog propagation** — rank near-neighbour structures of a good spectral match,
   not just exact hits. This is what lifts class 2 without a full retrieval model, and
   is what the published baseline notebook does.
4. **Use all 10 test adducts properly.** The index keeps them, but scoring ignores
   adduct compatibility between query and reference.

## 7. Files

```
metric.py     MRR@25 exactly as scored; self-checks against the rules' examples
library.py    row-group-streamed index over train.parquet -> data/index/
              canonical_keys(): RDKit key14 per structure, cached
split.py      held-out novelty split on canonical keys  -> data/splits.npz
search.py     precursor-windowed cosine library search   class 1 0.9390
```

Self-checks that need no competition data:

```bash
python3 metric.py              # metric vs the glucose example in the rules
python3 library.py --selftest  # spectrum curation
python3 search.py --selftest   # search and aggregation on a synthetic index
```

`metric.py` is the one to trust on a new machine: RDKit is pinned to 2026.3.3 because
tautomer canonicalisation is version-dependent, and a different version silently
changes what counts as a correct answer.
