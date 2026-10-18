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

## 7. The leaderboard says class 1 is 13% of the problem, not 33%

Submitted as a kernel (kernels-only competition). **Public LB 0.140.** Private is not
revealed until the competition closes on 14 December 2026.

The held-out split predicted 0.3365 by weighting the three classes equally. That
weighting was flagged as an assumption, and the leaderboard says it was wrong.

Our per-class numbers are known, so one leaderboard score constrains the test's class
mix. With `LB = 0.9420·f1 + 0.0676·f2 + 0·f3`:

| if class 2 is | then class 1 is | class 3 is |
|---|---|---|
| 0% | 14.9% | 85.1% |
| 20% | 13.4% | 66.6% |
| 33% | 12.5% | 54.5% |
| 45% | 11.6% | 43.4% |

**Class 1 is only 12–15% of the hidden test regardless of how the rest splits.** This
is CASMI — a challenge about identifying unknowns — so a test set dominated by
molecules with no spectra in any library is exactly what it should be. The equal-mix
assumption was the naive reading.

The consequence is blunt: **library search is worth at most 0.14 of this metric, and
we are already getting nearly all of it.** Class 1 at 0.9420 cannot go much higher,
and every additional point of it buys about 0.13 of a point on the leaderboard.

For scale, the leader is at 0.425. Assuming they also solve class 1 at ~0.94, that is
0.14 from class 1 and **0.285 from the other 85%** — an average near 0.335 on classes
2 and 3, against our 0.0676 and 0.0000. They are not doing better retrieval; they are
doing structure elucidation. That is the whole competition, and it is the thing this
codebase does not yet do at all.

## 8. Class 3 is coverage-limited, and the held-out split cannot see it

Four submissions, each a controlled change:

| | change | public LB |
|---|---|---|
| v1 | library search only | 0.140 |
| v2 | + analog propagation, fingerprint scoring | 0.158 |
| v3 | + COCONUT candidate pool | 0.199 |
| v4 | + mass-defect features, hidden 2048 | **0.213** |

The first two have no class-3 capability at all, which pins the class mix exactly:
**class 1 14.0%, class 2 12.6%, class 3 73.5%.**

The v4 gain decomposes cleanly, and the decomposition is the finding:

| | held-out delta | share | LB contribution |
|---|---|---|---|
| class 1 | +0.0115 | 14.0% | +0.0016 |
| class 2 | +0.1026 | 12.6% | +0.0129 |
| class 3 | +0.0071 | 73.5% | **−0.0005** |
| | | | total +0.0140 (actual +0.0140) |

Class 2 paid for all of it. Solving each submission for the *real* class-3 MRR shows
it did not move: **0.0617 → 0.0610**, while the held-out split reported +0.0071.

So the split misleads about class 3 in both directions. It understates the level (by
2.8x at v3, 2.1x at v4) and it reports improvements that do not exist.

The reason is that real class 3 is **coverage-limited, not ranking-limited**. A better
fingerprint cannot find a molecule that is not in the candidate pool at all. Our
proxy class 3 is partly a ranking problem, because the 8% COCONUT covers are in the
pool and rank better with a better model — reality has no such headroom.

Backing out the numbers needs the ranking quality among molecules that *are* in the
pool. The obvious estimate — held-out class 3 divided by its coverage, 0.0288/0.08 —
is computed from the 16 molecules of 200 that COCONUT happens to hold, and it is
worthless at that sample size. Measuring it properly is section 9.

Either way the shape of the problem is fixed: model work only touches class 1 and
class 2, which are 26.5% of the metric between them. Database coverage is the only
lever that moves the other 73.5%.

## 9. Separating coverage from dilution

A bigger database buys coverage and costs dilution — more candidates competing for 25
slots. Both are measurable without downloading anything, by subsampling the COCONUT
pool to a fraction f: coverage falls to about f while the pool shrinks by the same
factor, so sweeping f traces the two curves at once (`dilute.py`).

The ordinary split reads coverage straight off the sweep, and it is linear:

| f | median pool | class 3 covered | class 1 | class 2 | class 3 |
|---|---|---|---|---|---|
| 0 | 115 | 0.0% | 0.8836 | 0.4050 | 0.0000 |
| 0.25 | 148 | 2.5% | 0.8696 | 0.3980 | 0.0123 |
| 0.5 | 170 | 5.0% | 0.8638 | 0.3907 | 0.0199 |
| 1 | 217 | 8.0% | 0.8512 | 0.3859 | 0.0293 |

But ranking quality still lands on 16 molecules at f=1 and fewer below, so the same
sweep on a **split whose class 3 is drawn only from structures COCONUT already has**
— coverage 100% by construction, so class-3 MRR *is* the ranking quality, on 200
molecules instead of 16:

| f | median pool | class 3 covered | class 1 | class 2 | quality |
|---|---|---|---|---|---|
| 0.125 | 91 | 13.0% | 0.8874 | 0.4930 | 0.556 |
| 0.25 | 108 | 25.5% | 0.8867 | 0.4888 | 0.596 |
| 0.5 | 141 | 51.5% | 0.8812 | 0.4860 | 0.539 |
| 1 | 203 | 100.0% | 0.8678 | 0.4765 | **0.492** |

**Ranking quality is 0.492, not the 0.360 the 16-molecule estimate gave.** That moves
the implied real coverage from 17% down to 0.0610/0.492 = **12.4%**.

Subsampling can only shrink the pool, so it measures dilution over one doubling where
an expansion needs six. Widening the neutral-mass window instead adds candidates
without removing the answer, and reaches pool sizes a real expansion would produce.
Those candidates have the wrong mass. Nothing in the ranking looks at mass, so the
reasoning at the time was that they compete on the same footing as the right-mass ones
a bigger database would add. **That conclusion is wrong, and section 10 measures the
real slope at seven times steeper.** A candidate inside the query's 5 ppm window shares
its molecular mass, which constrains its formula and therefore most of its
composition; a candidate 160 ppm away does not. Real isomers are the confusable kind,
and a database supplies those exclusively.

| ppm | median pool | class 1 | class 2 | quality |
|---|---|---|---|---|
| 5 | 203 | 0.8678 | 0.4765 | 0.4917 |
| 20 | 606 | 0.8584 | 0.4292 | 0.5117 |
| 80 | 2393 | 0.8344 | 0.3446 | 0.4769 |
| 160 | 4711 | 0.8241 | 0.3163 | 0.4561 |

What this does measure honestly is **class 2, at −0.037 per doubling**: it rides on
analog propagation, and a wider pool is more structures competing to be somebody's
analog, whatever their mass. That is against a pool growing on the index side too; a
database grows only the external side, and the subsample sweep puts that at −0.012 per
doubling for class 2 and −0.018 for class 1.

Protecting class 2 by denying external structures the analog term (`PROP_INTERNAL=1`)
does not pay: class 2 gained 0.005 and class 3 lost 0.017, which at the solved class
mix is −0.0105 of leaderboard.

## 10. PubChem is twenty times larger and scores worse

A REST probe of all 200 held-out class-3 skeletons found **88.7% of them in PubChem**
against COCONUT's 8.0%, so: 124M compounds parsed on a Kaggle notebook, filtered to
the mass range and to the most natural-product-like 8% by RDKit's NP-likeness score,
deduplicated on key14. 8,390,560 structures, 4.4 GB, six hours.

| pool | structures | median window | class 1 | class 2 | class 3 | quality |
|---|---|---|---|---|---|---|
| COCONUT | 712k | 217 | 0.8512 | 0.3859 | **0.0293** | 0.366 |
| PubChem | 8.63M | 1030 | 0.8485 | 0.3581 | 0.0121 | 0.186 |
| both | 8.81M | 1104 | 0.8409 | 0.3525 | 0.0256 | 0.269 |

**Coverage rose from 8.0% to 9.5% and ranking quality fell from 0.366 to 0.269.**
Twelve times the structures bought nineteen percent more coverage and cost twenty-six
percent of the quality. Coverage has to grow faster than quality falls; it grows
slower.

The real test does not rescue it. Classes 1 and 2 are real in the held-out split, so
their losses transfer directly: −0.0103 and −0.0334, which at the solved mix is
−0.0057 of leaderboard. Class 3 would then have to reach 0.0687 to pay for that, and
at a quality of 0.28 that needs **24.5% real coverage against 12.4% today**. Scaling
the measured proxy gain by COCONUT's own real-to-proxy ratio of 1.55 gives 14.7%.

Two filters were tried and both are the wrong axis:

- **NP-likeness.** The held-out class 3 structures score a median of **−1.43** on it.
  They are not natural-product-like at all — they are the drugs, pesticides and
  metabolites that reference spectral libraries are made of. A threshold keeping the
  most natural-product-like 8% of PubChem keeps 6.5% of them, which is worse than
  random. Nothing else was removing them: all 200 pass the mass, fragment and length
  filters.
- **CID order.** Deposit order is a decent proxy for "long-studied compound", and
  class 1 obliges with a median lowest CID of 22M. Classes 2 and 3 have medians of
  119M and 92M — the rarer a structure, the later it was deposited. CID ≤ 100M covers
  49.5% of class 3.

The lever is real and the database was wrong for it. A candidate pool needs density in
the chemistry that reaches spectral libraries, not size: COCONUT reaches 8.0% carrying
217 candidates a query, PubChem 6.5% carrying 1030, which is four times less coverage
per candidate. ChEMBL and the curated metabolite databases are the shape to try next —
one to two million structures of exactly that chemistry, where the dilution is small
enough that a modest coverage gain pays.

## 11. The leaderboard kills the coverage lever

ChEMBL went to the leaderboard as one controlled change, the pool, at the same gamma
as the 0.213 run. It scored **0.187**.

| | held-out class 3 | public LB |
|---|---|---|
| v4, COCONUT | 0.0288 | 0.213 |
| v8, COCONUT + ChEMBL | 0.0393 | **0.187** |

Held-out class 3 rose 32% and the leaderboard fell 0.026. Solving for the real
per-class scores, with the split's class 1 and class 2 deltas carried over (−0.0090
and −0.0460, worth −0.0071 at the solved mix):

```
0.735 * real class 3 = 0.187 - (0.1682 - 0.0071) = 0.0259
real class 3         = 0.0610 -> 0.0352          (-42%)
```

**Real coverage cannot have fallen** — the union pool is a strict superset of COCONUT,
so every structure that was reachable still is. What fell is ranking quality, and it
fell much faster than the split said:

| | pool multiplier | coverage | quality |
|---|---|---|---|
| held-out split | ×4.58 | ×2.13 | ×0.61 (−0.065/doubling) |
| real test | ×4.58 | ×2.13 (assumed) | ×0.27 (−0.16/doubling) |

That is the whole result, and it is best read as two elasticities against pool size:
**coverage +0.50, quality −0.86.** An expansion pays only when coverage grows almost
as fast as the pool does, which means almost every structure it adds is one some query
needs. No general database is like that; COCONUT reaches 8.0% of held-out class 3
carrying 217 candidates, ChEMBL 11.5% carrying 898, PubChem 6.5% carrying 1030.

So the coverage lever is exhausted, and section 8's projections — 50% coverage for
0.298, 100% for 0.453 — are dead. They assumed quality held while coverage grew, and
at −0.16 per doubling the arithmetic never closes: reaching 50% coverage needs a pool
around 4× larger, by which point quality is 0.17 and class 3 is 0.085, against the
0.096 that breaking even would need.

Three submissions now say the same thing from different directions. Better features
moved class 2 and not class 3 (section 8). A twenty-times-larger pool moved neither
(section 10). A well-chosen pool moved held-out class 3 up and the real score down
(here). What is left is the term the three have in common: how well a candidate is
ranked *against the isomers that share its mass*. That is a scoring problem, not a
retrieval one.

## 12. What is next

Ranked by expected value, with the measurements that rank them:

1. **Rank better among same-mass isomers.** Section 11 makes this the only lever with
   headroom: quality falls 0.16 per doubling of pool on the real test, so every
   candidate carried is expensive and the way to profit is to score them better rather
   than to carry fewer or more. The measurement to run first is the cheapest one —
   restrict the held-out evaluation to candidates sharing the answer's molecular
   formula and see what class 3 becomes. That is the ceiling a perfect formula filter
   would reach, and it says whether formula prediction is worth building.

   *Measured* (`FORMULA_ORACLE=1 python3 dilute.py`, COCONUT, gamma 32):

   | split | oracle | class1 | class2 | class3 | quality | kept |
   |---|---|---|---|---|---|---|
   | ordinary | off | 0.8984 | 0.3398 | 0.0288 | 0.360 | 100% |
   | ordinary | on | 0.9529 | 0.4077 | 0.0331 | 0.413 | 13.1% |
   | covered | off | 0.9029 | 0.3978 | 0.4714 | 0.471 | 100% |
   | covered | on | 0.9535 | 0.4533 | 0.5417 | 0.542 | 16.4% |

   Removing the 84-87% of candidates with the wrong formula raises quality only 15%:
   the ranking already pushes them down, and the confusion that costs MRR is between
   isomers of one formula. Projected onto the real test the ceiling is about +0.02
   (0.213 -> ~0.233), and a filter that has to infer the formula from fragments gets
   less. Worth building, but no longer obviously first.

   *Then built (item 2) and measured, and it captures ~5% of that ceiling.*
   `formula.explained` scores the fraction of intensity whose fragments are
   sub-formulas of the candidate's ion; `DELTAS=0,4,8,16,32 python3 dilute.py`
   gives LB 0.1898 / 0.1904 / **0.1907** / 0.1901 / 0.1834. The reason is
   discrimination, measured on covered class 3 (answer vs 30 wrong-formula
   candidates from the same window, n=190):

   | fragment tol | answer explains | wrong explains | AUC |
   |---|---|---|---|
   | 0.002 Da | 0.727 | 0.658 | 0.590 |
   | 0.005 Da | 0.827 | 0.774 | 0.608 |
   | 0.01 Da | 0.862 | 0.819 | 0.587 |

   Weighting peaks by m/z, (m/z)^2 or keeping only peaks above half the precursor
   leaves AUC at 0.60-0.61. Small fragments are sub-formulas of almost every
   formula in the window, and the true formula itself leaves 15-25% of intensity
   unexplained (isotopes, noise, rearrangements). +0.0009 is inside the noise, so
   it is not shipped; `formula.py` stays as the measurement.

   *Reverse/hybrid cosine (item 3) also measured, and lost on held-out:*
   `SIM=rev` LB 0.1700, `SIM=hybrid` 0.1840, against cosine 0.1898. Hybrid lifts
   class 1 0.8984 -> 0.9043 but costs class 2 0.3398 -> 0.2880: analog propagation
   weights analogs by seed score, and reverse inflates references with few peaks.
   The held-out queries are library spectra (<=64 peaks) and cannot show the
   230-peak asymmetry of the real test, so this is not a verdict on the test, but
   there is no held-out evidence to ship it. `SIM` defaults to `cos`.
2. **A formula-consistency term.** A 5 ppm window admits several molecular formulas,
   and only one of them can explain the fragments: a candidate's formula must contain
   every fragment's. Scoring candidates by the fraction of spectrum intensity their
   formula can account for prunes the part of the window that is not isomeric, without
   a new model and without touching coverage. Item 1 measures its ceiling.
3. **A reverse or hybrid cosine**, per section 4. Test spectra carry a median of 230
   peaks against a library median of 9 to 11, which is the clearest remaining mismatch
   between the data and the scoring function. It caps class 1 and class 2 directly —
   26.5% of the metric between them — and sharpens item 1.
4. **Adduct-aware scoring.** The index keeps all ten adducts but the cosine ignores
   whether query and reference share one, so a `[M+Na]+` spectrum is compared against
   an `[M+H]+` reference on equal terms.

Candidate-pool work is closed. Three databases were measured — COCONUT, PubChem
(8.4M), ChEMBL (2.6M) — and the two expansions both lost, the second of them on the
leaderboard.

De novo generation is off this list. Earlier versions of it said class 3 was
"structurally unreachable by retrieval", which was an inference from a held-out split
that bans the answer by construction, not a measurement. The leaderboard says real
class 3 already scores 0.0610 by retrieval, and that the public databases hold most of
it.

## 13. Files

```
metric.py     MRR@25 exactly as scored; self-checks against the rules' examples
library.py    row-group-streamed index over train.parquet -> data/index/
              canonical_keys(): RDKit key14 per structure, cached
split.py      held-out novelty split on canonical keys  -> data/splits.npz
              C3_COVERED=1 draws class 3 from structures the external pool has
search.py     Rung 1: precursor-windowed cosine search   0.9390 / 0      / 0
analog.py     Rung 2: mass-windowed Tanimoto propagation 0.9420 / 0.0676 / 0
fingerprint.py  spectrum -> Morgan bits; binned m/z, neutral loss, mass defect
fpsearch.py   Rung 3: the three scores combined         0.8512 / 0.3859 / 0.0293
external.py   COCONUT as a candidate source, sharded    +448k structures
pubchem.py    PubChem as a candidate source; picks its NP-likeness threshold
              from a strided sample to land on a row count. Measured worse than
              COCONUT (section 10) and not shipped
chembl.py     ChEMBL as a candidate source; 2.6M structures, eight minutes
dilute.py     coverage against dilution: subsample the pool, or widen the window
              CASMI_EXT takes several directories, so two pools can be compared
```

Self-checks that need no competition data:

```bash
python3 metric.py              # metric vs the glucose example in the rules
python3 library.py --selftest  # spectrum curation
python3 analog.py --selftest   # mass windows, Tanimoto, and the class 3 ban
python3 search.py --selftest   # search and aggregation on a synthetic index
python3 fingerprint.py --selftest  # featuriser and a single-molecule overfit
```

`metric.py` is the one to trust on a new machine: RDKit is pinned to 2026.3.3 because
tautomer canonicalisation is version-dependent, and a different version silently
changes what counts as a correct answer.

## 14. One adduct window: 0.236

`submit()` paired the first spectrum's precursor with the adducts of *every* spectrum
of the molecule. 108 of the 400 test molecules carry two adducts, so each of them also
searched a window ~22 Da away from its real neutral mass, filled only with decoys.
Taking precursor and adduct from the same spectrum (v9) moved the leaderboard
**0.213 -> 0.236**, the largest single step in the project, from a quarter of the
molecules losing a decoy window.

The held-out eval could not see this. It opened every adduct of the polarity for every
query (median pool 217) while the kernel used the test's adduct column. `dilute.py
TRUE_ADDUCT=1` recovers the held-out query's adduct from the answer's mass and opens
only that window, which is what the kernel does:

| eval | pool | class1 | class2 | class3 | LB |
|---|---|---|---|---|---|
| every adduct | 217 | 0.8984 | 0.3398 | 0.0288 | 0.1898 |
| true adduct | 48 | 0.9262 | 0.3802 | 0.0323 | 0.2013 |

Gamma re-tuned on the true-adduct eval: 16 / 32 / 64 / 128 / 256 give 0.1910 /
0.2013 / 0.2057 / **0.2068** / 0.2041, trading class 1 (0.93 -> 0.78) for class 2
(0.31 -> 0.54). On the leaderboard gamma 64 (v10) scored **0.235** against 0.236: the
held-out gain of +0.004 is inside the noise of 400 molecules, and the class 1 loss
and class 2/3 gain cancel. Gamma stays 32.

Two lessons. Every eval/test mismatch is a place a bug can hide, and this one was worth
more than all the modelling since v4. And the real test rewards a smaller pool even
more than held-out does: a fix touching 27% of molecules gained 0.023.

## 15. Held-out does not predict scoring weights

A gated gamma — keep 32 when the best spectral match averages cosine >= 0.7 per
spectrum, use 128 otherwise — was the best held-out result since the eval was fixed:
0.2013 -> **0.2168**, class 1 0.926 -> 0.919, class 2 0.380 -> 0.492 (gates 0.5-0.9
give 0.2130-0.2172, a plateau). On the leaderboard (v11) it scored **0.236**, exactly
v9. With v10 that is two weight changes in a row that held-out ranked +0.004 and
+0.016 and the leaderboard ranked 0. Gating is kept in `dilute.py` (`GATES`) and taken
out of `submit()`.

Two reasons the held-out eval cannot see this:

* **The local `test.parquet` is a placeholder.** Every one of its spectra has an
  identical twin in `enveda-180`, and `submit()` ranks the twin's structure first for
  97.5% of molecules; scored on that file MRR would be ~0.97, not 0.236. Kaggle swaps
  in the hidden test when it scores. Nothing about the hidden test (adduct mix, mass
  range, match strength) can be read off the local file.
* **Held-out class 1 is not drawn from the test's chemistry.** The dummy test is all
  `enveda-180`, so the hidden test is very likely Enveda in-house data too. Held-out
  class 2 and 3 queries are 70-74% `enveda-180`, but class 1 queries contain none:
  pluskal_ms2 39%, gnps 22%, riken 12%, mona 11%. The class 1 side of every gamma
  trade-off was measured on public-library spectra.

The next eval change is therefore a class 1 drawn from `enveda-180` queries whose
structure also has spectra elsewhere, before any more weights are tuned against it.

## 16. The gamma gain lives in class 2, and needs the answer in the index

Section 15's eval change, built: `C1_LIB=enveda-180 SPLIT_OUT=data/splits_c1env.npz
python3 split.py` draws class 1 from the 2,255 multi-library structures that
`enveda-180` measured and holds out the `enveda-180` spectra. The fingerprint model was
retrained without this split's spectra (`SPLITS=... FP_MODEL=data/fpmodel_c1env.pt
python3 fingerprint.py --train --holdout`, hidden 2048, defect, 6 epochs, val tanimoto
0.307): `fpmodel.pt` excludes only `splits.npz`'s rows, and 95% of the new split's
query spectra were in its training set. The same holds for `splits_cov3.npz` (90% of
its queries), so class 2 and 3 numbers measured on that split with `fpmodel.pt` are
inflated.

| | class 1 | class 2 | class 3 | predicted LB | actual LB |
|---|---|---|---|---|---|
| gamma 32 | 0.8984 | 0.3781 | 0.0195 | 0.1877 | 0.236 (v9) |
| gamma 64 | 0.8918 | 0.4635 | 0.0204 | 0.1983 | 0.235 (v10) |
| gated 0.7 / 128 | 0.8726 | 0.5102 | 0.0211 | 0.2019 | 0.236 (v11) |

An Enveda class 1 does not fix it. Class 1 barely moves; the predicted gain is class 2
(0.378 -> 0.510, +0.017 of LB), and the leaderboard shows none of it.

Held-out class 2 structures are always in the candidate list, because they come from
the index's own 264k structures. With them banned as well (`C2_BANNED=1`), so a class 2
answer is reachable only through COCONUT (18 of 200):

| | class 1 | class 2 | class 3 | predicted LB |
|---|---|---|---|---|
| gamma 32 | 0.8984 | 0.0360 | 0.0195 | 0.1446 |
| gamma 64 | 0.8918 | 0.0365 | 0.0204 | 0.1445 |
| gated 0.7 / 128 | 0.8726 | 0.0362 | 0.0211 | 0.1422 |

The three now tie within 0.0024, as the leaderboard does. That is weak evidence: class
2 is flat by construction once its answers are mostly unreachable, and the absolute
level (0.145) is further from 0.236 than the unbanned eval's (0.188). The truth sits
between the two variants. What it settles is narrower: **a held-out gain that comes
from class 2 with the answer in the index does not reach the leaderboard.** A scoring
change must improve the `C2_BANNED=1` eval, or class 1 / class 3, to be worth a
submission.

## 17. Leak-free covered class 3: ranking quality is 0.40, not 0.56

> **Gamma here is 128, not 32** (section 24): `dilute.py`'s default gate sent every
> molecule to `GAMMA_HI`. Reproduced exactly on 2026-10-17; at a real gamma 32 this
> model scores class 1 0.9058, class 3 0.3492, `C2_BANNED=1` LB 0.3869.

One split for both open questions: `C1_LIB=enveda-180 C3_COVERED=1
SPLIT_OUT=data/splits_c1cov3.npz python3 split.py` (class 1 from enveda-180, class 3 all
in COCONUT, 200 molecules each). The model was retrained without its 6,590 held-out
spectra (`FP_MODEL=data/fpmodel_c1cov3.pt`, hidden 2048, defect, 6 epochs, val tanimoto
0.316, still rising at epoch 6). Same split, same pool (median 53, one adduct window),
gamma 32, only the model differs:

| model | class 1 | class 2 | class 3 quality | predicted LB |
|---|---|---|---|---|
| `fpmodel.pt` (leaky) | 0.8836 | 0.5395 | 0.5574 | 0.6014 |
| `fpmodel_c1cov3.pt` | 0.8838 | 0.4408 | **0.3970** | 0.4710 |
| `fpmodel_c1cov3.pt`, `C2_BANNED=1` | 0.8838 | 0.0302 | 0.3970 | 0.4193 |

The leak inflated covered class 3 by +0.16 and class 2 by +0.10; class 1 does not use
the predicted fingerprint for its answer and is unchanged. **Every class-3 number
measured with `fpmodel.pt` on a non-`splits.npz` split overstated the model**, including
section 11's 0.492. Honest isomer ranking among same-mass COCONUT candidates is 0.40 MRR,
so the fingerprint model has more headroom than the leaky numbers suggested. Predicted LB
here is meaningless (class 3 is 100% covered by construction); only the class columns count.

## 18. Six more epochs raise tanimoto, not class 3

`FP_INIT=data/fpmodel_c1cov3.pt FP_COSINE=1 FP_MODEL=data/fpmodel_c1cov3_e12.pt
python3 fingerprint.py --train --holdout`: section 17's model continued for 6 epochs with
lr cosine-decayed to 0 (12 in all). Val tanimoto 0.316 -> 0.329, loss 0.0595 -> 0.0530.
Same split and pool as section 17, gamma 32:

| model | class 1 | class 2 | class 3 quality | `C2_BANNED=1` LB |
|---|---|---|---|---|
| 6 epochs | 0.8838 | 0.4408 | 0.3970 | 0.4193 |
| 12 epochs, cosine | 0.8851 | 0.4760 | 0.3885 | 0.4134 |

Class 3 does not move (-0.009 on 200 molecules, inside noise); the only gain is class 2
with the answer in the index, which section 16 showed does not reach the leaderboard. It
fails the submission rule. A better average fingerprint is not a better isomer ranking:
same-mass candidates share most bits, and mean tanimoto is dominated by the bits they
share. Training longer on the same objective is closed; the next fingerprint change has to
target the bits that separate isomers.

## 19. Re-weighting bits at inference does not rank isomers better

`d_b` = fraction of 5 ppm pools (19,526 pools around random non-banned structures) in
which bit b varies; median 0.343. Weighted tanimoto with `w_b = d_b ** alpha`, section
17's model and split, gamma 32 (alpha 0 reproduces 0.3970 exactly):

| weighting | class 1 | class 2 | class 3 quality |
|---|---|---|---|
| none | 0.8838 | 0.4408 | 0.3970 |
| global, alpha 0.5 | 0.8766 | 0.4322 | 0.3920 |
| global, alpha 1 | 0.8586 | 0.4241 | 0.3907 |
| global, alpha 2 | 0.8301 | 0.3994 | 0.3719 |
| pool-local (only bits that vary in this pool) | 0.8616 | 0.4465 | 0.3973 |

Global weights hurt every class: a bit that is constant in most pools is decisive in the
few where it varies, and a global weight mutes exactly those. Pool-local masking, which
keeps every bit that can separate this pool's candidates and nothing else, leaves class 3
unchanged. The scoring formula is not the bottleneck; the predicted values of the
separating bits are. Closed at inference; the next step changes the model (a ranking loss
over the same pools). Code reverted.

## 20. The fingerprint ceiling is 0.95: the gap is all prediction

If the predicted fingerprint were exactly the answer's, only exact bit collisions in the
5 ppm pool could outrank it (ties scored as expected reciprocal rank under random order):

| set | bits | median pool | answer's bits unique in pool | ceiling MRR |
|---|---|---|---|---|
| section 17 class 3 (200) | 1574 kept | 34 | 90.0% | 0.9479 |
| section 17 class 3 (200) | all 2048 | 34 | 90.0% | 0.9479 |
| 20k random structures | 1574 kept | 47 | 93.5% | 0.9569 |

The model reaches 0.397 of a possible 0.948. Collisions cost 0.05 and the 474 dropped
bits cost nothing; the rest is the accuracy of the predicted separating bits.

## 21. A ranking loss over the pool lifts class 3: 0.397 -> 0.449

> **Gamma here is 128, not 32** (section 24). Val pool-MRR does not depend on gamma.

`FP_RANK=1` adds cross-entropy over the spectrum's own 5 ppm pool (index + COCONUT,
split's banned structures excluded, up to 64 candidates) on soft tanimoto / tau (0.05),
answer as target, to the BCE. Fine-tuned from section 17's model for 2 epochs, cosine lr
(`FP_INIT=data/fpmodel_c1cov3.pt FP_RANK=1 FP_COSINE=1 FP_MODEL=data/fpmodel_c1cov3_rank.pt
python3 fingerprint.py --train --holdout --epochs=2`, 50 min/epoch). Val pool-MRR (2,000
held-out spectra, answer's rank in its own pool) 0.531 -> 0.615 -> 0.630; val tanimoto
0.316 -> 0.325. Same split and pool as section 17, gamma 32:

| model | class 1 | class 2 | class 3 quality | `C2_BANNED=1` LB |
|---|---|---|---|---|
| BCE, 6 epochs | 0.8838 | 0.4408 | 0.3970 | 0.4193 |
| + ranking loss, 2 epochs | 0.8869 | 0.6530 | **0.4486** | **0.4583** |

The first change that moves class 3 on a leak-free split, and it passes the submission
rule: the `C2_BANNED=1` eval rises and class 1 does not fall. Val pool-MRR moved +0.10 on
2,000 independent spectra, so the +0.052 on 200 class 3 molecules is not noise. Not tuned:
lambda 1, tau 0.05, 2 epochs (pool-MRR still rising). The class 2 jump (+0.21) is the
in-index kind section 16 showed does not reach the leaderboard; it is not counted.

## 22. The shipped model with the ranking loss

> **Gamma here is 128, not 32** (section 24), while v12 ships gamma 32 with no gate.
> Reproduced for `fpmodel.pt`; at a real gamma 32 it scores class 1 0.9262, class 3
> 0.0322, `C2_BANNED=1` LB 0.1574. The leaderboard gain of v12 stands; the held-out
> estimate made for it was measured at the wrong gamma.

`fpmodel.pt` (v9's model, excludes `splits.npz`) fine-tuned exactly as section 21:
`SPLITS=data/splits.npz FP_INIT=data/fpmodel.pt FP_RANK=1 FP_COSINE=1
FP_MODEL=data/fpmodel_rank.pt python3 fingerprint.py --train --holdout --epochs=2`
(~60 min/epoch). Val pool-MRR 0.507 -> 0.647 -> 0.651. On `splits.npz`, for which this
model is leak-free (gamma 32, one adduct window):

| model | class 1 | class 2 | class 3 (8% covered) | predicted LB | `C2_BANNED=1` LB |
|---|---|---|---|---|---|
| `fpmodel.pt` (v9) | 0.8308 | 0.5105 | 0.0356 | 0.2068 | 0.1465 |
| `fpmodel_rank.pt` | 0.8621 | 0.6740 | 0.0568 | 0.2474 | 0.1677 |

Every class rises, and the `C2_BANNED=1` eval by +0.021. Class 1 gains too (+0.031):
a fingerprint that ranks the answer above its isomers also helps when the library match
is weak. Class 3 is 16 covered molecules here, so its number is noise; section 21 is the
class 3 evidence.

**Leaderboard: v12 = 0.252** (v9/v11 0.236), the first gain since the adduct-window fix.
The per-class estimate made before submitting was +0.006 to +0.011 (class 1 from the
Enveda split, class 3 scaled by 12-15% coverage, class 2 only through COCONUT); the
actual +0.016 is above it, so at least one of those three was too conservative. The
`C2_BANNED=1` delta (+0.021) was the closer predictor this time.

## 23. Softer ranking loss (tau 0.1) is worse; the rest of the sweep is open

> **Gamma here is 128, not 32** (section 24).

Section 21's setup with `FP_TAU=0.1` (2 epochs, cosine, same init, same split, gamma 32):

| tau | val pool-MRR (ep 1 / ep 2) | class 1 | class 2 | class 3 quality | `C2_BANNED=1` LB |
|---|---|---|---|---|---|
| 0.05 (section 21) | 0.615 / 0.630 | 0.8869 | 0.6530 | 0.4486 | 0.4583 |
| 0.1 | 0.595 / 0.601 | 0.8630 | 0.6093 | 0.4352 | 0.4456 |

Every column falls, class 1 by 0.024, so it fails the submission rule. A softer softmax
over the pool weakens the signal that pushes the answer above its isomers. Not run:
tau 0.025, lambda 0.3 / 3, 4 epochs, lower lr, RANK_K (a first tau 0.025 run was killed
by the memory guard before its first epoch; the machine was shared with Firefox and a
1.7 GB server). `FP_LR` and `FP_RANK_K` are now env knobs (defaults 1e-3 / 64 unchanged).
The plan for these runs is in PLAN_MAC.md.

## 24. The eval ran at gamma 128; at the shipped gamma 32 no tuned tau beats the baseline

**Eval bug.** `dilute.py` gained gating in 3c810a7 (2026-10-10) with a default gate of 2.
A molecule keeps `GAMMA` only when its mean cosine per spectrum reaches the gate, which a
cosine never does, so every eval without an explicit `GATES` scored at `GAMMA_HI` = 128
whatever `GAMMA` said. Sections 17-23 are affected (§17 and §22 reproduced exactly at
gamma 128); §15-16 set the gate and are not. `submit()` ships gamma 32 with no gate, so
the held-out numbers and the submissions used different gammas. The default is now 0
(no gate): `GAMMA` takes effect.

**Sweep on the Mac** (Apple M4, 16 GB, CPU, 12.5 min/epoch against ~50 on the Linux box).
Section 21's setup, one knob changed at a time. Epoch 0 pool-MRR 0.5313 and the baseline's
epochs 0.6151 / 0.6294 match Linux, but its class 3 at gamma 128 came out 0.4283 against
0.4486, so the rows are compared with the Mac baseline. Val pool-MRR is the last epoch:

| model | knobs | val pool-MRR | val tanimoto |
|---|---|---|---|
| mac | tau 0.05 (section 21) | 0.6294 | 0.324 |
| t025 | tau 0.025 | 0.6616 | 0.291 |
| t0125 | tau 0.0125 | 0.6833 | 0.238 |
| t00625 | tau 0.00625 | 0.7066 | 0.175 |
| t025_l3 | tau 0.025, lambda 3 | 0.6666 | 0.285 |
| best4 | tau 0.00625, 4 epochs | 0.7123 | 0.178 |
| lr4 | lr 1e-4 | 0.5951 at epoch 1, stopped | |

`C2_BANNED=1` LB (class 1 / class 3) by gamma, no gate:

| gamma | mac | t0125 | t00625 | best4 |
|---|---|---|---|---|
| 16 | **0.3936** (0.912 / 0.356) | 0.3837 | 0.3623 | 0.3820 |
| **32 (shipped)** | **0.4322** (0.918 / 0.407) | 0.4180 (0.911 / 0.389) | 0.3948 | 0.4104 (0.908 / 0.380) |
| 64 | 0.4422 | **0.4497** | 0.4342 | 0.4486 |
| 128 | 0.4459 (0.902 / 0.428) | 0.4675 (0.934 / 0.452) | 0.4558 | **0.4721** (0.919 / 0.461) |
| 256 | 0.4505 | 0.4707 | 0.4687 | **0.4843** (0.933 / 0.475) |

A sharper tau lowers val tanimoto (0.32 -> 0.18): the predicted fingerprint stops being
calibrated and the `GAMMA * tanimoto` term shrinks, so it needs a larger gamma to weigh
the same. Tau and gamma are one knob. At gamma 128 tau looks like the biggest lever found
(+0.026 LB); at the shipped 32 every tuned tau loses to the baseline. Held-out also asks
for more gamma for every model, the baseline included, which §15-16 showed does not reach
the leaderboard. By the submission rule at the shipped gamma nothing beats v12: **keep
v12.** Lower lr is worse at any gamma (stopped by the epoch-1 rule).

Open: whether a sharper tau plus a matching gamma is real can only be shown on the
leaderboard (a t0125 or best4 fine-tune on `splits.npz`, shipped at gamma 128). A score
that does not depend on the tanimoto scale (rank or z-score within the pool) would
let model and gamma be tuned apart.

**The leaderboard candidate, built and gated out.** `fpmodel.pt` fine-tuned at tau 0.0125
on `splits.npz` (`FP_TAU=0.0125`, 2 epochs, `data/fpmodel_rank_t0125.pt`): epoch 0
pool-MRR 0.5073 (as Linux), then 0.714 / 0.719 against v12's 0.651. On `splits.npz`,
`C2_BANNED=1`, no gate:

| model | gamma | class 1 | class 3 (16 covered) | `C2_BANNED=1` LB | LB |
|---|---|---|---|---|---|
| v12 (`fpmodel_rank.pt`) | **32 (shipped)** | **0.9215** | 0.0504 | **0.1709** | 0.2393 |
| v12 | 128 | 0.8621 | 0.0550 | 0.1664 | 0.2461 |
| tau 0.0125 | 32 | 0.9311 | 0.0444 | 0.1674 | 0.2316 |
| tau 0.0125 | 64 | 0.9226 | 0.0494 | 0.1700 | 0.2461 |
| tau 0.0125 | 128 | 0.9099 | 0.0569 | 0.1740 | 0.2567 |

v12 at 128 reproduces §22. No row passes the rule against v12 as shipped: at 128 the
`C2_BANNED=1` LB is +0.003 (16 class 3 molecules, noise) while class 1 falls 0.012; at 32
and 64 it is lower. The plain LB's +0.017 at 128 is index-reachable class 2 (§16). Not
submitted; v12 stays.

## 25. Ranking molecules instead of spectra does not help

Inference scores the mean prediction over a molecule's spectra (`fpsearch.query_fp`), and a
held-out query is up to 4 spectra of one structure, but the ranking loss ranked each spectrum
alone. `FP_MOL=4` splits each structure's training spectra into random groups of up to 4
(every spectrum still once per epoch), keeps BCE per spectrum, and applies the ranking loss to
each group's mean sigmoid. A new val metric, `mol-MRR`, is pool-MRR on the mean over up to 4
spectra of 2,000 val structures, the way the eval queries. The init model already gains +0.07
from averaging (0.5313 -> 0.6025). Section 21's knobs otherwise; gamma 32, no gate:

| model | val pool-MRR | val mol-MRR | class 1 | class 3 | `C2_BANNED=1` LB |
|---|---|---|---|---|---|
| Mac baseline (per spectrum) | **0.6294** | **0.7153** | **0.9178** | 0.4066 | 0.4322 |
| `FP_MOL=4` | 0.6172 | 0.7036 | 0.9159 | **0.4243** | **0.4446** |
| `FP_MOL=4 FP_RANK=4` | 0.6081 | 0.7006 | 0.9104 | 0.4139 | 0.4365 |
| tau 0.0125 (§24, reference) | 0.6833 | 0.7544 | 0.9114 | 0.3894 | 0.4180 |

`FP_MOL=4` passes the gate on the 200-molecule eval (+0.018 class 3), but its own target
metric, mol-MRR on 2,000 molecules, falls 0.012, so the second look points the other way and
it is not a winner. Restoring the ranking gradient (lambda 4, since a batch holds 64 molecule
terms instead of 256 spectrum terms) does not lift it. Ranking every spectrum alone raises the
averaged prediction more than ranking the average does: each spectrum must place the answer
first on its own, and a batch draws four times as many pools. One upside: val tanimoto stays
calibrated (0.326), unlike a sharper tau. An epoch takes half the time (6.5 min), which locates
the training cost: `Pools.batch`, a Python loop per row, not the network.

`fingerprint.py` also checkpoints a resumable state every `FP_CKPT_MIN` minutes (default 5) to
`FP_MODEL + ".state"` (model, optimizer, schedule, rng, the epoch's order and position; written
via fsync + rename). Rerunning the same command resumes from it. A run killed with `kill -9`
mid-epoch and resumed ends with weights bit-identical to an uninterrupted run.

## 26. Training twice as fast; the 200-molecule eval moves 0.02 between identical runs

**Speed.** A ranking-loss step on the Mac: model forward/backward 50 ms (46%), building the
candidate tensor 56 ms (51%), the per-row pool loop 1 ms (1%). The cost was unpacking all
2,048 bits of 256 x 64 candidates and indexing the 1,574 kept ones every batch. `Pools.packed`
now repacks every candidate's kept bits once (134 MB, 3 s) and `unpack_bytes` unpacks them in
torch: 56 -> 11 ms, same values. An epoch takes 6.4 min instead of 12.3. Torch threads stay
at 4: 6-10 threads are slower on the M4 (its efficiency cores).

**Noise floor.** The new tensor is contiguous where the old one (numpy's `[..., bits]`) was
laid out bits-first, so the einsum sums in a different order and gradients differ in the last
float bit. Retraining section 21's Mac baseline with nothing else changed:

| run | val pool-MRR | val mol-MRR | class 3 / `C2_BANNED=1` LB, gamma 32 | gamma 128 |
|---|---|---|---|---|
| `fpmodel_c1cov3_mac.pt` | 0.6294 | 0.7153 | 0.4066 / 0.4322 | 0.4283 / 0.4459 |
| `fpmodel_c1cov3_mac_repro.pt` | 0.6297 | 0.7163 | 0.4164 / 0.4385 | 0.4535 / 0.4645 |

Rounding alone moves 200-molecule class 3 by 0.010-0.025 and its LB by 0.006-0.019, while the
2,000-sample val metrics move 0.001. This is the Linux-vs-Mac class 3 gap of section 24
(0.4486 vs 0.4283). **A class 3 or LB difference under ~0.025 between two trainings is not
evidence**; val pool-MRR / mol-MRR is. Sections 24-25 already decided on val MRR (tau 0.0125,
`FP_MOL`); their eval differences at gamma 32 (-0.014, +0.012) are within this noise, so the
"keep v12" decision rests on no eval gain being shown, not on a measured loss.

## 27. Normalising the fingerprint term within the pool: sharp tau wins on the leak-free split

Section 24's sharp-tau models lost at gamma 32 because their predicted tanimoto has 2.6x less
spread within a pool (median std: tau 0.05 0.041, tau 0.0125 0.025, tau 0.00625 0.016), so the
same gamma weighs them less. `FP_NORM` (`fpsearch.normalise`, used by `dilute.py` and
`submit()`) rescales each molecule's pool tanimoto before gamma: `std` t / std(t), `z`
(t - mean) / std, `rank` the percentile. Default "" is v12's raw score. `dilute.py` sweeps
`GAMMAS` and `NORMS` in one run. Leak-free split, `C2_BANNED=1` LB (class 1 / class 3):

| scoring | mac | mac_repro | t0125 | t00625 | best4 | mol4 |
|---|---|---|---|---|---|---|
| raw, gamma 32 (shipped) | 0.4322 (0.918/0.407) | 0.4385 | 0.4180 | 0.3948 | 0.4104 | 0.4446 |
| std, gamma 1.31 | 0.4157 | 0.4230 | 0.4354 | 0.4350 | **0.4477** | 0.4295 |
| **std, gamma 2.62** | 0.4300 (0.914/0.404) | 0.4430 (0.914/0.422) | 0.4498 | 0.4515 | **0.4681** (0.914/0.457) | 0.4452 |
| std, gamma 5.24 | 0.4370 | 0.4498 | 0.4622 | 0.4631 | **0.4782** (0.925/0.468) | 0.4435 |

`z` is 0.001-0.004 below `std` everywhere, `rank` 0.02-0.07 below. At std gamma 2.62 the baseline
scores what raw 32 gives it (0.430 / 0.443 vs 0.432 / 0.439), so that is the shipped weight.
There best4 beats the mean of the two baseline trainings by +0.032 LB and +0.044 class 3 with
class 1 unchanged, 2-3x the gap between the two baseline trainings (0.013 / 0.018), and the
order best4 > t0125 ~ t00625 > baseline ~ mol4 holds at every gamma and every norm. It is also
val pool-MRR's order, so for the first time the eval and the val metric agree on tau.

**On `splits.npz` it does not pass.** Ship fine-tunes of `fpmodel.pt` (tau 0.0125 2 epochs,
`fpmodel_rank_t0125.pt`; tau 0.00625 4 epochs, `fpmodel_rank_best4.pt`, val pool-MRR 0.733 /
mol-MRR 0.765 against v12's 0.651):

| model | scoring | class 1 | `C2_BANNED=1` LB |
|---|---|---|---|
| v12 | raw 32 (shipped) | **0.9215** | 0.1709 |
| v12 | std 2.62 | 0.8930 | 0.1710 |
| tau 0.0125 | std 2.62 | 0.9142 | **0.1728** |
| tau 0.00625, 4 ep | std 2.62 | 0.9075 | 0.1647 |

Class 3 here is 16 covered molecules, so the comparison is class 1, and the std norm costs
every model class 1 on this split (v12 -0.029), where it costs none on the leak-free split.
The two splits' class 1 differ: public-library spectra here, Enveda spectra there (§15-16,
the hidden test is likely Enveda). Under the rule neither model passes (class 1 falls; best4's
LB too); tau 0.0125 at std 2.62 is closest (+0.002 LB, class 1 -0.007, the size of retraining
noise). Only the leaderboard can settle it. `submit()` runs locally with
`FP_MODEL=data/fpmodel_rank_t0125.pt FP_NORM=std GAMMA=2.62`.

**Leaderboard: v13 (tau 0.0125 at std gamma 2.62) scored 0.245, against v12's 0.252.** The
leak-free split predicted a gain (+0.015 `C2_BANNED=1` LB over the mean baseline, +0.03 class 3)
and `splits.npz` predicted +0.002 with class 1 -0.007; the real -0.007 sits inside the
retraining noise of §26 but on the losing side, so the class 3 gain the leak-free split shows for
sharp tau plus normalisation does not reach the hidden test. `splits.npz` class 1 falling was the
warning that pointed the right way. v12 stays best and is the one to select for the final
ranking. The repo's `kernel/casmi_submit.py` and `kernel/code/` are v13; resubmitting v12 means
removing the two `os.environ` lines and shipping `data/fpmodel_rank.pt` again.

## 28. A popularity prior from COCONUT's metadata: class 3 0.407 -> 0.467

Isomers inside the 5 ppm window are what retrieval loses to (§10-11), and the spectrum
separates them badly. Molecules that get measured as reference standards are the
documented ones, and COCONUT records documentation per entry. The October 2026 release CSV
(CC0, `coconut_csv-10-2026.zip`, 739k rows) is reduced by `coconut_meta.py` to one
row per InChIKey14 (`data/coconut_meta/meta.npz`, 480k keys): counts of synonyms,
organisms, collections and DOIs, CAS and name flags, annotation level, NP-likeness.
`prior.py` loads one feature (`PRIOR`), counts as log1p; `dilute.py` adds `w * prior` to
every candidate (`PRIOR_WS`), keys COCONUT lacks get `PRIOR_FILL` (0).

Class 3 answers of the leak-free split have a median of 6 synonyms; COCONUT's median is 1
and 15% of it has 5 or more. Leak-free split, `fpmodel_c1cov3_mac.pt`, raw gamma 32,
`C2_BANNED=1` LB (class 1 / class 3):

| feature | w 0.25 | w 0.5 | w 1 | w 2 |
|---|---|---|---|---|
| none (w 0) | 0.4322 (0.918/0.407) | | | |
| **syn** | 0.4677 | **0.4788** (0.913/0.470) | **0.4861** (0.905/0.482) | 0.4706 |
| coll | 0.4461 | 0.4544 | 0.4674 | 0.4714 (0.901/0.464) |
| doi | 0.4419 | 0.4477 | 0.4634 | 0.4377 |
| cas | 0.4419 | 0.4522 | 0.4567 | 0.4593 |
| name | 0.4417 | 0.4478 | 0.4490 | 0.4509 |
| org | 0.4429 | 0.4464 | 0.4452 | 0.4143 |
| annot | 0.4465 | 0.4497 | 0.4299 | 0.4098 |
| npl | 0.4293 | 0.4288 | 0.4113 | 0.3906 |

The second training of the same config (`mac_repro`) agrees: class 3 0.416 -> 0.473 (w 0.5)
/ 0.485 (w 0.75), class 1 0.911 -> 0.907 / 0.906. That is +0.06 class 3, three times the
§26 noise, and the largest single gain this eval has shown for a model-free change.

The cost is class 1: its answers are rarely in COCONUT (1% on the leak-free split), so
every documented COCONUT isomer gains on them. Filling missing keys with a positive value
protects class 1 and gives back class 3 (fill 1: class 3 0.458 at w 0.5; fill 3.4: 0.421),
because the index's wrong structures in a class 3 pool are the ones it boosts. A gate does
better: `PRIOR_GATES` skips molecules whose best library match (mean cosine per spectrum)
reaches the gate. On `splits.npz` with v12's model (public-library class 1):

| w | gate | class 1 | `C2_BANNED=1` LB |
|---|---|---|---|
| 0 | - | 0.9215 | 0.1709 |
| 0.5 | none | 0.9110 | 0.1703 |
| **0.5** | **0.8** | **0.9186** | 0.1713 |
| 0.75 | 0.8 | 0.9182 | 0.1709 |
| 1 | 0.8 | 0.9183 | 0.1695 |

On the leak-free split the gate leaves class 1 at 0.913 (its losses are weak matches) and
costs class 3 0.003 at w 0.5 (0.467). Class 3 of `splits.npz` is 16 molecules the leaky
model trained on and does not move. **Shipped as v14: v12 model, raw gamma 32, `PRIOR=syn
PRIOR_W=0.5 PRIOR_GATE=0.8`** (`submit()` reads the same knobs; the kernel ships
`coconut_meta.npz`). The smaller weight because held-out answers come from public spectral
libraries and may be better documented than the hidden test's; the leaderboard decides.
The local `test.parquet` cannot check this: every molecule in it has a library match at
cosine ~1.0, so the gate skips all of them.

**A pool-cache bug the gate sweep found.** `external.merge` cached the merged pool in one
`data/pool/` and rebuilt it in place whenever the row count changed, which it does for
every banned set. A training run starting during an eval with `C2_BANNED=1` rewrote the
files under the eval's memory maps, and the eval printed class 2 and 3 as 0.0000 from that
point. Each pool now has its own directory named by a hash of the shard list and the index
keys, built in a temporary directory and renamed into place. Earlier evals ran one at a
time or against a training run that had already loaded its pool; their numbers reproduce.

`FP_SEED` (training order, pool negatives, dropout; not the val structures) and ensembles
(`FP_MODEL=a.pt,b.pt`: the mean of the members' probabilities, `fpsearch.Ensemble`) are in
for the next section. The two rounding-only trainings `mac` + `mac_repro` ensembled score
their average (0.4380 against 0.4322 / 0.4385): their predictions differ by 0.011 per bit,
too correlated to gain.
