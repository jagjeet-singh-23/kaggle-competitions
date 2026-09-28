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
Those candidates have the wrong mass, which nothing in the ranking looks at — every
term scores a structure, not its mass — so they compete on the same footing:

| ppm | median pool | class 1 | class 2 | quality |
|---|---|---|---|---|
| 5 | 203 | 0.8678 | 0.4765 | 0.4917 |
| 20 | 606 | 0.8584 | 0.4292 | 0.5117 |
| 80 | 2393 | 0.8344 | 0.3446 | 0.4769 |
| 160 | 4711 | 0.8241 | 0.3163 | 0.4561 |

**A 23x pool costs class 3 fifteen thousandths.** Ranking quality is −0.011 per
doubling; class 1 is −0.010 and class 2 is −0.037. Class 3 is nearly immune to pool
size, because the predicted fingerprint separates the right structure from arbitrary
ones sharply, while class 2 rides on analog propagation and a wider pool is exactly
more structures competing to be somebody's analog.

That last number is measured under window widening, which grows the index side of the
pool too. A database expansion grows only the external side, and the subsample sweep
puts that at −0.012 per doubling for class 2 and −0.018 for class 1.

Protecting class 2 by denying external structures the analog term (`PROP_INTERNAL=1`)
does not pay: class 2 gained 0.005 and class 3 lost 0.017, which at the solved class
mix is −0.0105 of leaderboard.

So the arithmetic for an expansion of m times the current pool:

```
class 1  0.804 - 0.018*log2(m)         real values, solved from the leaderboard
class 2  0.442 - 0.012*log2(m)
quality  0.492 - 0.011*log2(m)
LB       0.140*class1 + 0.126*class2 + 0.735*coverage*quality
```

At m = 84 (PubChem after filtering, ~60M structures) the dilution costs 0.026 of
leaderboard and quality falls to 0.422. **Coverage only has to reach 22.7% to break
even**, against 12.4% today:

| real class 3 coverage | class 3 | predicted LB |
|---|---|---|
| 12.4% (now, COCONUT) | 0.061 | 0.213 |
| 22.7% | 0.096 | 0.213 — breakeven |
| 50% | 0.211 | 0.298 |
| 70% | 0.295 | 0.360 |
| 100% | 0.422 | 0.453 |

A REST probe of all 200 held-out class-3 skeletons found **88.7% of them in PubChem**
against COCONUT's 8.0%. Those molecules all have reference spectra somewhere, so they
are known compounds and the real figure will be lower — but breakeven needs the
coverage multiplier to be 1.8x and the probe says 11x.

## 10. What is next

Ranked by expected value:

1. **Predict fingerprints from the spectrum.** This is no longer one option among
   several — section 7 says classes 2 and 3 are ~87% of the metric and we score
   0.0676 and 0.0000 on them. Analog propagation reaches class 2 only through a
   structure that already has spectra, so it cannot score a candidate whose
   neighbourhood the library never sampled. A spectrum-to-fingerprint model scores
   every candidate in the mass window directly. Everything else on this list is worth
   a fraction of what this is worth.
2. **A reverse or hybrid cosine**, per section 4. The peak-count asymmetry is the
   clearest remaining mismatch between the data and the scoring function, and it caps
   class 1 as well as class 2.
3. **Class 3 needs de novo generation** and nothing short of it. It may be half the
   hidden test (section 7) and is structurally unreachable by retrieval — the
   assertion in `analog.py` exists to keep that honest.
4. **Adduct-aware scoring.** The index keeps all ten adducts but the cosine ignores
   whether query and reference share one, so a `[M+Na]+` spectrum is compared against
   an `[M+H]+` reference on equal terms.

## 11. Files

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
