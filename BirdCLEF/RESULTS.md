# BirdCLEF+ 2026 — Results

Identifying 234 species from 5-second windows of passive acoustic monitoring in the
Brazilian Pantanal. Metric is macro-averaged ROC-AUC over classes with at least one
true positive. Competition closed June 2026; winner 0.96574.

**Private leaderboard 0.92681**, from a linear head on frozen Perch v2 embeddings and
logits, blended with Perch's own logits on the 203 classes it knows, cold-start heads
on focal audio for the 2 classes that have neither, and temporal smoothing across the
windows of each recording (section 14). The same pipeline on BirdNET scored 0.87008.

Unless a table says public or private, a number is 5-fold cross-validation over the
739 labelled soundscape segments, grouped by recording file. Section 13 explains why
numbers before it moved by up to 0.02 between machines.

---

## 1. The local metric is not the leaderboard metric

234 classes are defined. Only **75** have a positive in the labelled soundscape data,
so every local score is a macro-average over 75 columns, not 234. It is comparable
between the models below and not comparable to the leaderboard in absolute terms.

The 75 split into 28 species BirdNET knows and 47 it does not — and that split turns
out to explain almost every result on this page.

## 2. Scores

| Model | all (75) | birds (28) | others (47) |
|---|---|---|---|
| zero-shot BirdNET, no training | 0.6168 | 0.8128 | 0.5000 |
| head on embeddings | 0.8756 | 0.9098 | 0.8553 |
| head on emb + logits | 0.8774 | 0.9078 | 0.8593 |
| head + zero-shot, everywhere | 0.8100 | 0.9188 | 0.7451 |
| **head + zero-shot, bird columns only** | **0.8881** | 0.9156 | 0.8717 |

The head is one-vs-rest logistic regression, `C=0.01`, `class_weight="balanced"`,
on 1,024-dim embeddings with 739 rows. Heavily regularised on purpose.

*Correction: rows 4 and 5 were inflated or deflated by how `rank()` broke ties. With
ties averaged they are 0.8769 and 0.8794. Section 13 explains.*

## 3. The finding that mattered: the embeddings are not bird-specific

BirdNET is a bird classifier. Its 6,522 output classes contain 157 of the 162
competition birds and **none** of the 72 amphibians, insects, mammals and one reptile.
Zero-shot, those 72 classes score exactly 0.5000 — by construction, they get a
constant.

Those non-bird classes are 31% of the metric, and 28 of them have no focal training
audio at all. The obvious plan was a separate non-bird detector.

That was unnecessary. A linear probe on the *penultimate* layer takes the non-birds
from **0.5000 → 0.8553**, within 0.05 of what the same probe does on the birds
BirdNET was actually trained to recognise. The pooled 1,024-dim layer is a general
bioacoustic representation — frequency structure, pulse rate, harmonic spacing — and
insects are not out of distribution for it even though they are out of distribution
for the classifier head bolted on top.

Reaching that layer needs `experimental_preserve_all_tensors=True`, which costs about
25% throughput (24 ms vs 18 ms per chunk) and is the only way to get it out of TFLite.

## 4. Blending zero-shot helps only where BirdNET has an opinion

Row 4 of the table is the mistake worth keeping. Blending BirdNET's logits into the
head everywhere *drops* the score from 0.8756 to 0.8100. For the 47 classes BirdNET
has never heard, its logits are a constant, and rank-averaging a constant into a real
prediction is pure dilution: 0.8553 → 0.7451.

Restricting the blend to the 28 bird columns keeps the gain where it is real
(0.9098 → 0.9156) and leaves the rest alone. That is the 0.8881.

*Correction: the dilution above was not real. argsort ranked BirdNET's constant
column as a ramp in row order, so the blend mixed in noise. With ties averaged,
blending everywhere costs 0.0025, not 0.078 (section 13).*

## 5. Negative result: 333 hours of focal audio did not help

`train_audio/` is 35,549 focal recordings, 333 hours — 178× more rows than the
labelled soundscape data. Extracting BirdNET embeddings for it took about four hours
and produced 131,943 segments (`embed_train.py`, 6 windows per clip).

Combining it with the soundscape data made every configuration worse:

| focal per class | soundscape weight | focal rows | all | birds | others |
|---|---|---|---|---|---|
| 0 *(soundscape only)* | — | 0 | **0.8756** | 0.9098 | 0.8553 |
| 30 | 1 | 6,271 | 0.8209 | 0.8161 | 0.8237 |
| 30 | 10 | 6,271 | 0.8257 | 0.8132 | 0.8331 |
| 30 | 100 | 6,271 | 0.8265 | 0.8021 | 0.8410 |
| 150 | 10 | 28,237 | 0.8249 | 0.8186 | 0.8287 |

Two things say this is real and not a tuning failure:

- **The trend is monotone in the wrong direction.** Raising the soundscape weight
  1 → 10 → 100 improves the score 0.8209 → 0.8257 → 0.8265, heading back toward the
  0.8756 you get by discarding focal audio entirely. The sweep's own best answer is
  "use less of it".
- **More of it does not help.** 28,237 focal rows (0.8249) is no better than 6,271
  (0.8257).

The damage is concentrated on birds: 0.9098 → 0.8021, against 0.8553 → 0.8410 for
non-birds. That direction is the explanation. The embeddings were produced by a
network trained on focal bird recordings, so for birds the focal data adds supervision
the representation already encodes, in the wrong acoustic domain — close, clean,
single-caller, versus distant, overlapping and noisy. For the non-birds BirdNET never
trained on, there is only 750 focal clips to do damage with, and 28 classes have none
at all.

A separate ridge experiment (`rung1c.py`, no `--logistic`) shows focal audio *helping*,
0.705 → 0.748. That is not a contradiction: plain ridge on one-hot targets has no
class balancing, is much weaker on 739 rows against 234 classes, and is starved enough
that even domain-shifted data is an improvement. It never gets near 0.8756. The
comparison that counts holds the model fixed.

## 6. What this cost, and what made it affordable

| step | output | peak RAM | wall clock |
|---|---|---|---|
| `embed_labelled.py` | 739 segments | < 1 GB | ~2 min |
| `embed_train.py` | 131,943 segments, 251 MB | < 1 GB | ~4 h |
| `rung1c.py` ridge sweep, 24 configs | — | < 1 GB | ~2 min |
| `rung1c.py --logistic`, 5 configs | — | < 1 GB | ~35 min |

The ridge sweep is 24 fits over 131,943 × 1,024 and still runs in two minutes, because
the focal data is in every training fold and never in validation. Its Gram matrix
`X'X` and cross-product `X'Y` are therefore constant, accumulated in one streaming
pass over the shards, and each configuration is then one 1,024 × 1,024 solve on
`Gf + w·Gs`. Peak memory is one shard, about 32 MB.

`embed_train.py` writes a shard every 2,000 clips and resumes from what is on disk,
which is what let the four-hour run survive two unrelated machine restarts.

## 7. Leaderboard, and where the 0.052 gap went

Submitted as a kernel (this competition is kernels-only, so a CSV upload is never
accepted):

| | |
|---|---|
| public | 0.83593 |
| **private** | **0.83575** |
| local CV | 0.8881 |
| top-20 floor | 0.95426 |

(Section 8 improves this to 0.85710 private / 0.86983 public.)

Public and private differ by 0.00018 — the opposite of LANL, where the best public
score had the worst private one. A large test set makes the ranking stable.

The CV-to-leaderboard gap is not noise and not overfitting. It is 30 classes:

| BirdNET column? | labelled positive? | classes | what they get |
|---|---|---|---|
| yes | yes | 28 | head + zero-shot blend |
| yes | no | 129 | zero-shot only |
| no | yes | 47 | head only |
| **no** | **no** | **30** | **0.5 constant** |

Local CV only ever averages the 75 classes with a labelled positive, so it never sees
the 30 that fall through both routes. The leaderboard does. Their cost is arithmetic:

```
(204 x 0.885 + 30 x 0.50) / 234 = 0.8356      vs 0.83575 actual
```

That accounts for essentially the whole gap. And every one of those 30 classes —
18 amphibians, 5 birds, 4 mammals, 3 insects — **has focal training audio available**.
None of them is data-starved; they are simply not being trained.

This is also the limit of the section 5 negative result. Focal audio hurt classes that
already had soundscape data. For these 30 the alternative is a coin flip, so the same
data can only help. Generalising "focal audio hurts" to them would be wrong.

## 8. Rung 2: cold-start heads, and how well the proxy predicted them

The 30 classes from section 7 now get a head fitted on focal audio instead of a
constant. They cannot be validated directly — having no labelled positive is the
definition of the cohort — so `coldstart.py` measures a proxy: the 19 classes that
are *also* absent from BirdNET and *also* have focal audio, but do have labels. Their
labels are used only to score, never to fit.

| focal segments/class | rows | mean AUC on the 19 | above 0.5 | above 0.7 |
|---|---|---|---|---|
| **30** | 6,271 | **0.7385** | 18/19 | 15/19 |
| 150 | 28,237 | 0.7175 | 17/19 | 11/19 |
| 400 | 64,970 | 0.7202 | 18/19 | 10/19 |

30 segments per class beats 150 and 400, matching Rung 1c: more focal audio does not
help anywhere it has been tried.

**Submitted result:**

| | public | private |
|---|---|---|
| v1 | 0.83593 | 0.83575 |
| v3, with cold-start heads | **0.86983** | **0.85710** |
| gain | +0.03390 | +0.02135 |
| predicted from the proxy | +0.0306 | +0.0306 |

The gain is real and is the largest single improvement so far. The proxy was accurate
on public (+0.0339 against +0.0306) and **over-predicted private by about 30%**.
Inverting `gain = 30/234 · (v − 0.5)` says the 30 classes actually scored ~0.764 on
public and ~0.667 on private, against the proxy's 0.7385. So the 19 proxy classes are
somewhat easier than the 30 they stand in for — worth remembering before trusting the
same method again.

Public and private also diverged for the first time: 0.0127 apart, against 0.00018 on
v1. The cold-start heads are the new variance. They are fitted on at most 30 focal
segments per class for species with no in-domain data at all, so how well any one of
them transfers depends on which recordings land in which split.

This is also the limit of the section 5 negative result. Focal audio hurt classes that
already had soundscape data. For these 30 the incumbent was a coin flip, and the same
data is worth +0.021. Generalising "focal audio hurts" across both would have been
wrong.

The same experiment says explicitly *not* to touch the 129 classes that fall back to
zero-shot. Measured on the mapped-and-labelled cohort, zero-shot scores 0.8128 against
a focal head's 0.7809, and a 50/50 blend gains nothing. That would have been a
regression.

## 9. Negative result: pseudo-labelling 177 hours of in-domain audio bought ~0.004

`train_soundscapes/` holds 10,658 recordings and 66 are labelled. The other 10,592 —
177 hours, 127,104 windows — are from the scored domain and were completely unused.
This looked like the largest remaining lever, because every failure up to here had
been a domain-gap failure and this data has no gap to cross. Extraction took 209
minutes.

It did not deliver. Two cohorts, measured independently:

| cohort | share of metric | baseline | with pseudo-labels | gain |
|---|---|---|---|---|
| 75 classes with labels | 32% | 0.8756 | 0.8799 | +0.0043 |
| 129 mapped, unlabelled | 55% | 0.8128 | 0.8174 | +0.0046 |

Combined macro effect is about **+0.004** — against +0.021 measured on the
leaderboard for the far cheaper cold-start change in section 8.

Both numbers are weak on their own terms. On the 75 classes only 25 of them improve,
so the positive mean comes from a few large gains against many small losses. On the
129-class proxy the gain is 1.6 sigma at the blend weight that maximises it, and the
sweep shows why that weight matters:

| weight on the taught head | gain | sem | sigma |
|---|---|---|---|
| 0.1 | +0.0046 | 0.0029 | 1.6 |
| 0.3 | +0.0113 | 0.0099 | 1.1 |
| 0.5 | +0.0128 | 0.0173 | 0.7 |

A larger blend weight buys a larger point estimate and loses more than it gains in
certainty. The two cohorts agreeing at ~+0.004 is the only reassuring part.

### Three hypotheses formed and discarded along the way

Worth recording, because all three came from reading means without measuring spread.

1. **"More unlabelled data helps."** Pool sizes 3 through 7 gave a clean monotone
   rise, 0.8331 to 0.8390.
2. **"More unlabelled data hurts."** Extending to 11 and 22 shards reversed it,
   0.8370 then 0.8256, and a site-composition story was constructed to explain it.
3. **"Top-k concentrates on a few loud recordings as the pool grows."** Measured: the
   top-25 spans 13.7 files at one shard and 15.4 at twenty-two. It spreads out, not in.

The paired comparison settles it. Between 7 and 22 shards the difference is −0.0134
with a standard error of 0.0085 — **1.6 sigma, and the per-class standard deviation
of the difference is 0.0448, more than three times the mean difference.** The pool
size was never doing anything. Two opposite trends were read out of the same noise
before anyone checked the spread.

The full pool is used regardless. Picking 7 shards because a 28-class estimate liked
it is exactly the overfitting that cost this repo its LANL private score.

## 10. Negative result: the head is not the bottleneck, the representation is

Three data sources in a row returned little or nothing (sections 5, 8, 9). "Capacity"
was the explanation offered, but that word hides two claims with very different price
tags:

- **head capacity** — the linear probe is too weak, and a nonlinear head fixes it with
  the backbone frozen. Cheap to test.
- **representation** — BirdNET's features have said all they can, and only fine-tuning
  the backbone helps. Expensive, and as it turns out, unavailable.

`mlp.py` settles the cheap one. Every configuration lost:

| hidden | best of 8 configs | vs linear 0.8774 |
|---|---|---|
| 64 | 0.8602 | −0.0172 |
| 256 | 0.8709 | −0.0065 |
| 512 | 0.8727 | −0.0047 |

**24 of 24 configurations below the linear probe**, and the shape of the failure is
the informative part: the deficit shrinks monotonically as the hidden layer grows. A
wider trunk is not learning something new, it is slowly re-learning the linear
solution from below. Head capacity is not the constraint.

### The experiment had to be redesigned twice first

The obvious version of this test — swap the linear layer for an MLP and retrain —
does not answer the question, because it changes two things at once. Two bugs found
before any number here could be trusted:

1. **Learning rate.** At `lr=1e-3` a full-batch model needs thousands of steps to fit
   a trivially separable toy problem (AUC 0.810 at 200 steps, 0.997 at 3000). The
   first grid used 300 and 800 epochs, so it would have compared two under-trained
   models and blamed the MLP for having more parameters to move. A self-check on the
   toy problem caught it.

2. **Joint optimisation is a handicap, not a treatment.** A linear layer over 234
   outputs is 234 *independent* problems — there is no shared trunk for them to
   benefit from. Optimising all 276k parameters together converges far worse per class
   than 234 dedicated solves: jointly trained, the linear model scores **0.67** against
   sklearn's 0.8774 on identical folds and features, while a single-class torch fit
   reproduces sklearn to three decimals (0.9978 vs 0.9974). The original design
   justified joint training as letting classes "share statistical strength", which for
   a linear layer is not a thing that exists.

The fix is two stages: train the trunk jointly to learn features, throw it away except
for its hidden layer, then fit the final classifier with the *same per-class sklearn
call the baseline uses*. `hidden=0` is an identity trunk and must reproduce 0.8774
exactly, which is asserted rather than hoped for.

Two further hypotheses about the 0.67 were formed and measured wrong before the real
cause was found: that decoupled weight decay was mis-scaled by the loss reduction
(`wd=0` gives the same 0.6731, and Adam is scale-invariant anyway), and that the L2
strength simply needed matching to sklearn's `C=0.01` (the matched value is ~3.6e-4,
which was already in the swept range). Train AUC was 1.0000 at every setting, which
should have been the first thing looked at: AUC is invariant to uniformly scaling the
weights, so weight decay can barely move it.

### Why fine-tuning is not the answer here either

Checked rather than assumed:

- BirdNET V2.4 ships as `.tflite`, an **inference-only** format. There is no gradient
  path through it.
- The upstream releases publish GUI installers, not a bare trainable checkpoint, and
  BirdNET-Analyzer's own "training" feature trains a classifier *on embeddings* —
  precisely the thing sections 8, 9 and this one have exhausted.
- No CUDA device is available (`torch.cuda.is_available()` is False), and no
  TensorFlow is installed.

So the remaining levers are not about the head or the backbone. They are about giving
the frozen encoder **different audio** — mixing focal calls into soundscape
backgrounds produces genuinely new embedding vectors, and is directly motivated by
section 5's finding that focal audio fails on domain rather than on content.

## 11. Temporal smoothing: the first gain since the cold-start heads

Everything in sections 5, 9 and 10 attacked the features. This does not touch them.
A soundscape is continuous, so the 12 windows of a recording are not 12 independent
problems — which is how every earlier version of this kernel treated them.

Each window's scores are blended with the strongest score nearby in the same file:

```
score = 0.2 * own + 0.8 * max over +/-6 windows of the same recording
```

| | local CV |
|---|---|
| no smoothing | 0.8774 |
| k=6, max, alpha=0.2 | **0.8930** |

All 24 swept configurations beat the baseline, with a monotone structure — larger k
better, max over mean, lower alpha better — plateauing at k=6, which for a 12-window
recording is the whole file.

**A suspicious result, checked.** alpha=0 — discarding the window's own prediction
entirely — scores 0.8926, almost the best. That suggests the labels might simply be
file-level, in which case this exploits the annotation scheme rather than acoustics.
Measured: of the 370 (file, class) pairs where a class appears at all, **48.6% are
labelled across all 12 windows** and 51.4% vary, with the class on in 45% of windows
where it does. So it is half prior and half acoustics. It is still legitimate and
transferable — the metric ranks every test row against every other, so knowing a
species is present in a recording is most of the available signal, and the test set
is annotated by the same process.

The sweep also found a real bug: for a file shorter than the smoothing window, a
computed end index goes negative and numpy reads it as an offset from the end rather
than an empty slice. It crashed at k=3. The self-check now covers one- and two-row
files at every k.

### Submitted

| | public | private | gain (private) | predicted |
|---|---|---|---|---|
| v1 head + zero-shot | 0.83593 | 0.83575 | — | — |
| v3 + cold-start heads | 0.86983 | 0.85710 | +0.02135 | +0.0306 |
| **v4 + smoothing** | **0.88572** | **0.87008** | **+0.01298** | +0.0156 |

**0.83575 → 0.87008, +0.0343 in total.**

The prediction was well calibrated this time — +0.0156 locally against +0.0159 public
and +0.0130 private — where the cold-start prediction over-shot private by 30%. The
difference is the instrument. Smoothing was measured directly on the scored classes
with real labels; the cold-start gain was estimated on a 19-class proxy standing in
for 30 classes that cannot be validated at all. Proxy cohorts predict the direction
but not the magnitude.

## 12. Negative result: mixing focal calls into soundscapes changed nothing

Section 10 left exactly one data lever unblocked. Everything else asks the frozen
encoder for more from the same audio; this hands it *different* audio, so it gets
genuinely new embedding vectors back. It is also the direct test of section 5's
explanation — that 333 hours of focal audio failed on **domain** rather than content.
Mixing a focal call into a real soundscape background keeps the content and replaces
the domain.

6,180 mixtures, 30 per class across 206 classes, 9.5 minutes to generate.

**The first attempt was far worse than doing nothing, and the reason was a real bug.**
Backgrounds were chosen as the quietest 10% of unlabelled windows *by BirdNET's
strongest mapped logit*. Measured afterwards: those windows have a median RMS of
0.0193, and the **loudest** 10% have 0.0192 — a ratio of 0.99. Low BirdNET confidence
is not acoustic quiet. Those are wind, rain and insect noise: loud broadband maskers
that BirdNET cannot name. Calls were then mixed as much as 5 dB *below* them.

| cold-start cohort (19 scoreable stand-ins) | mean AUC | vs raw focal |
|---|---|---|
| raw focal | 0.7385 | — |
| mixtures, backgrounds by confidence, SNR −5..15 dB | 0.6578 | −0.0807 |
| mixtures, backgrounds by **energy**, SNR 5..25 dB | 0.7339 | −0.0047 |

Fixing the background selection recovered the entire deficit. It bought nothing
beyond that: corrected mixtures are statistically indistinguishable from raw focal
audio (−0.2 sigma, better on 7 of 19 classes).

| 75 labelled classes (real labels, no proxy) | mean AUC | vs labelled only |
|---|---|---|
| labelled only | **0.8756** | — |
| + mixtures, weight 0.1 | 0.8310 | −0.0446 |
| + raw focal (section 5) | 0.8265 | −0.0491 |

Mixtures beat raw focal here by 0.0045 — the direction section 5 predicted, and far
too small to matter. Both remain far below simply not using out-of-domain audio.

So the domain hypothesis is not refuted, it is just worth almost nothing. Replacing
the acoustic domain of focal audio recovers a sliver of what focal audio loses, and
never enough to make it useful where in-domain labels already exist.

**BirdCLEF is blocked at 0.87008.** Seven levers, and the pattern is the whole
result:

| lever | result |
|---|---|
| 333 h focal audio, where labels already existed | −0.05 |
| focal audio for the 30 classes that had none | **+0.021** |
| 177 h in-domain pseudo-labels | +0.004, 1.6σ |
| nonlinear head, 24 configurations | 24/24 below linear |
| all 6,522 BirdNET logits instead of 157 | −0.012 |
| temporal smoothing across a recording | **+0.013** |
| focal calls mixed into soundscape backgrounds | ±0.000 |

Every attempt to extract more from the features failed. Both changes that left the
features alone worked — one gave a prediction to classes that had none, the other
used structure in the output space. The frozen representation is the ceiling, and the
only thing that lifts it is fine-tuning, which is unavailable here.

*That conclusion was about one representation. Section 14 swaps BirdNET for Perch v2,
still frozen, and the ceiling moves.*

## 13. The local CV was not platform-independent

Moving from the Linux machine (x86, numpy 2.2) to a Mac mini (ARM, numpy 2.5)
reproduced zero-shot exactly, 0.6168, but not the head:

| | Linux | Mac |
|---|---|---|
| head on emb+logits | 0.8774 | 0.8755 |
| head + zero-shot, birds only | **0.8881** | **0.8684** |

The audio and the model were the same. Both differences come from how a sort breaks
ties, which numpy does differently on the two architectures.

**`rank()` broke ties in sort order.** In cross-validation a class absent from a
training fold predicts a 0.5 constant for that whole fold. That is 29 of the 75 scored
columns and 4,278 rows. `argsort(argsort())` gives those tied rows distinct ranks in
whatever order the sort leaves them, and AUC then scores that order as if it were a
prediction. Ranking must not change AUC; on the Mac it did:

| | AUC |
|---|---|
| raw head | 0.8755 |
| rank by argsort | 0.8655 |
| rank with ties averaged | 0.8755 |
| blend, argsort ranks, Mac / Linux | 0.8684 / 0.8881 |
| blend, ties averaged | **0.8794** |
| blend, 50 random tie orders | 0.8762 – 0.8818 |

Linux's 0.8881 sits above every random tie order and the Mac's 0.8684 below every one.
Neither is a measurement of the model. Two claims in this file were the same artifact:

- Section 4's "others 0.8593 → 0.8717 from rank normalisation". Rank normalisation
  cannot move AUC, and with ties averaged it does not.
- Section 4's "blending a constant is pure dilution" (0.8100). argsort turns a
  constant column into a ramp in row order, so the blend was mixing in noise, not a
  constant. With ties averaged, blending everywhere scores 0.8769 against 0.8794 for
  birds only. The restriction still helps, by 0.0025 rather than by 0.078.

The leaderboard numbers stand. At test time every head is fitted on all 739 rows, so
there are no fold constants and nothing to tie.

**GroupKFold's default folds depend on the platform too.** 59 of the 66 labelled files
have exactly 12 segments. GroupKFold assigns groups in size order, so the assignment
is decided by an unstable argsort over ties. Across 20 equally valid tie-breaks the
head scores 0.8643–0.8888, sd 0.0064. That spread is wider than several effects
reported above, including section 9's +0.004.

Fixes:
- `head.rank` uses `rankdata`, and `head.py --selftest` asserts that ranking leaves
  AUC unchanged.
- New comparisons run on 10 seeded shuffles (`GroupKFold(shuffle=True)`) and are
  paired on identical folds.
- `smooth.py`, `pseudo.py`, `rung1c.py` and `coldstart.py` still rank by argsort.
  Their local numbers carry the artifact; only the deltas confirmed on the
  leaderboard (smoothing) are trustworthy.

**A smaller bug in the same area.** `rung1c.focal_subsample` measured its cap in
appended shard chunks, not rows. "30 per class" gave 17 classes more than 30 rows,
up to 59, so `focal_cap30.npz` holds 6,271 rows where at most 6,180 were intended.

## 14. Perch v2: the ceiling was BirdNET's

The CLEF 2026 working notes contain the controlled comparison this repo never ran. The
same ridge probe on frozen embeddings scored 0.798 private on BirdNET and 0.862 on
Perch v2 (DS@GT, Table 6). `perch_head.py` reruns kernel v4's pipeline with only the
encoder changed.

**The model.** Perch v2 takes exactly one competition window, 5 s at 32 kHz, so there
is no resampling and no two-chunk fold. It returns a 1,536-dim embedding and 14,795
logits. The ONNX export used (`perch.py`) was checked against Google's SavedModel
on 12 windows of a labelled soundscape:

- embeddings matched within 3.8e-6 and logits within 9e-5;
- the top-20 logits were identical on every window;
- it runs at 42–48 ms per window on an M4, against 150 ms for TensorFlow;
- the CoreML provider was ten times slower and drifted by up to 0.19 in the logits.

**Coverage.** Perch has exact scientific-name matches for 203 of the 234 classes: all
162 birds, 32 of 35 amphibians, 6 of 8 mammals and 3 of 28 insects. BirdNET matched
157, all birds.

| 10 seeds, paired | BirdNET | Perch v2 |
|---|---|---|
| zero-shot | 0.6168 | 0.7367 |
| head on emb+logits | 0.8762 | 0.8838 |
| + zero-shot blend on mapped columns | 0.8803 | 0.9077 |
| **+ smoothing (kernel v4)** | **0.8880** | **0.9144** |

At the kernel-v4 stage Perch gains **+0.0264**. It wins all 10 seeds and 52 of 75
classes, at 2.9σ over classes.

The gain does not come from the embedding. A head on Perch's embedding beats one on
BirdNET's by only +0.008, at 1.0σ. It comes from Perch's own classifier:

- On the 28 birds both models know, zero-shot rises from 0.8128 to 0.9005.
- On the 18 labelled classes only Perch knows, zero-shot scores 0.8631, where
  BirdNET's constant scores 0.5.

That is also where the leaderboard should gain most, because local CV cannot see it:

| route, all 234 classes | BirdNET | Perch |
|---|---|---|
| mapped + labelled: head + blend | 28 | 46 |
| mapped, unlabelled: zero-shot | 129 | 157 |
| unmapped + labelled: head only | 47 | 29 |
| unmapped, unlabelled: cold-start | 30 | 2 |

28 of the 30 cold-start classes now have a Perch column: 17 amphibians, 5 birds,
3 insects and 3 mammals. On private, the cold-start heads they replace scored about
0.667. Two classes still need a cold-start head: Uruguay Harlequin Frog (5 clips) and
Black-tailed Marmoset (3 clips).

### The kernel

`kernel_perch/birdclef_perch_submit.py` is kernel v4 with the encoder swapped. Ranks
are tie-aware. The cold-start pool is re-embedded with Perch by
`embed_focal_perch.py`: the same windows recipe and a true cap of 30, giving 5,802
windows in 7 minutes. On Kaggle's CPU Perch runs at 202 ms per window, so about 600
test files take 24 minutes and the whole kernel about 35.

Two deployment surprises:

- **The Kaggle image moved to Python 3.13.** The ONNX dataset's `onnxruntime` wheel
  is cp312, so the first run failed. The assets dataset now ships the cp313 wheel, and
  pip picks whichever wheel fits. Kernel v4's `ai-edge-litert` wheel is also cp312, so
  re-running v4 today would fail the same way.
- **The Kaggle CLI names its upload state file after the full source path.** From a
  long directory that name passes macOS's 255-character limit and the upload aborts
  part-way. Stage uploads from a short path.

### Submitted

| | public | private |
|---|---|---|
| v1 BirdNET head + zero-shot | 0.83593 | 0.83575 |
| v3 + cold-start heads | 0.86983 | 0.85710 |
| v4 + smoothing | 0.88572 | 0.87008 |
| **Perch v1: v4 on Perch v2** | **0.92423** | **0.92681** |
| gain over v4 | +0.03851 | **+0.05673** |

The leaderboard gain is more than twice the local one, +0.057 against +0.026. Local
CV only scores the 75 labelled classes. The rest of the gain sits in the 157
zero-shot classes and the 28 former cold-start classes, which local CV cannot see.
This time the estimate from the routes table pointed the right way and the 75-class
number undersold it.

Private is now above public. Since v3 the order had been reversed, with a gap of
0.013–0.016, and section 8 attributed that gap to the cold-start heads. Removing 28
of the 30 removed it.

For scale:

| | private |
|---|---|
| DS@GT, frozen Perch v2 + MLP probes | 0.918 |
| this kernel, one frozen model, no GPU | **0.927** |
| public Perch/ProtoSSM pipeline, forked by hundreds | ~0.940 |
| 13th place, six CNNs + Perch | 0.955 |
| winner | 0.96574 |

## 15. Where this stands and what is next

The remaining gap to the top is about 0.04. Every team above 0.95 used one of these:
a trained spectrogram CNN, a temporal head over the 12 windows, or pseudo-labels from
a strong teacher. Ranked by expected value:

1. **A spectrogram CNN/SED branch blended with Perch.** This is the step to 0.95.
   DS@GT reached 0.920 with the Perch probe alone and 0.917 with HGNetV2-B0 SED
   alone; the blend scored 0.929. 13th place blended six CNNs (0.953) with Perch and
   reached 0.955. It needs GPU training, either on Kaggle or on the Mac's MPS. Section
   10's "fine-tuning unavailable" applied to BirdNET's `.tflite`, not to training a
   CNN from scratch.
2. **Pseudo-labels from this kernel.** At 0.927 it is now a strong teacher. Section 9
   failed with a 0.87-class teacher, and 13th place saw the same: a weak self-teacher
   lowered their CNN from 0.881 to 0.864, while strong labels raised it to 0.934. The
   pseudo-labels matter most as training data for item 1. For a linear head on frozen
   features, a late entrant's study found them redundant.
3. **A temporal head over the 12 windows (ProtoSSM).** Public pipelines use one.
   Paper 348's ablation credits temporal aggregation with only +0.0013, and the max
   smoother already captures most of it. Small.
4. **Re-tune smoothing for Perch.** k and alpha were chosen on BirdNET. Locally,
   smoothing adds +0.0067 on Perch against +0.0077 on BirdNET. Cheap, small.

## 16. SED pilot: no-go with either teacher

Section 15's item 1, at pilot scale: one fold of an HGNetV2-B0 SED, trained on
public pseudo-labels, rank-blended with the Perch kernel. The go bar was 0.9318
private, Perch plus 0.005.

| submission | public | private |
|---|---|---|
| Perch kernel (section 14) | 0.92423 | **0.92681** |
| SED alone | 0.87753 | 0.86072 |
| Perch + SED, rank blend w=0.5 | 0.91718 | 0.91148 |

**No-go.** The blend is 0.0153 below Perch alone. A 0.86 model given half the
weight dilutes a 0.93 one. SED alone sits at BirdNET's level (0.870), and 0.056
below DS@GT's HGNetV2-B0 SED (0.917). SED also drops 0.017 from public to private,
while Perch gains 0.003.

**The likely cause is the teacher.** The public pseudo-label file also holds the
teacher's out-of-fold predictions on our 739 labelled windows. They score
**0.8275**, below Perch (0.9144) and BirdNET (0.8880) on the same windows. This is
suggestive, not proof: the teacher's labels on unlabelled windows may come from
models trained on all the labelled files. Still, SED alone at 0.861 sits next to
13th place's CNN trained on a weak teacher (0.864), not next to their CNN trained
on a strong one (0.934). The second suspect is data scale. The pilot used 12,000
pseudo windows, 13% of the ~92,700 that 13th place used, and one fold. Phase B
separates the two: it keeps the same 12,000 windows and swaps only the targets for
the Perch kernel's own (`pseudo_targets_B.npy`).

The fold-0 validation AUC climbed from 0.55 to 0.970 and was useless as a guide.
The teacher was trained on the labelled files, so its pseudo-labels carry their
labels into training.

**Pilot data** (`sed_cache.py`, 8.4 GB, int16):

| source | count |
|---|---|
| pseudo-labelled windows, chosen by seed only | 12,000 of 127,104 |
| labelled soundscape windows, true labels, folds ≠ 0 train | 739 |
| focal clips, ≤ 30 per class, first 15 s, random 5 s crop | 5,313 (18.7 h) |

**Training** (`sed_train.py`, Kaggle T4, torch 2.11 / timm 1.0.29 preinstalled):
20 epochs, batch 64, AdamW 5e-4, OneCycle, mixup, EMA 0.999.

| | |
|---|---|
| epoch 0 | 509 s, loader wait 83% (cold reads from the dataset mount) |
| epochs 1–19 | ~75 s, 238 samples/s, loader wait < 1% |
| peak GPU memory | 3.22 GB |
| total GPU time | ~33 min, plus a ~16 min queue |

**Inference** on Kaggle CPU (`sed_infer.py`): SED takes 41–49 ms per window and
Perch 169–198. The 600-file projection is 33–34 min, so the 90-minute limit
leaves room for 2–3 more SED models.

**ONNX Runtime bug.** With the default graph optimizations, ORT 1.24.4 changed the
trained model's outputs by up to 0.078. At initialisation the same graph matched
torch, so the export selftest passed. With `ORT_DISABLE_ALL` it matches torch to
8e-7, at +11% CPU time. The selftest now also checks that the CPU path reproduces
the GPU predictions from training (1e-6).

**Provenance.** The pseudo-labels are `pseudo_labels_v2s_oof_raw.csv` from Kaggle
dataset `aliozanmemetoglu/raw-pseudos`. Its licence is "unknown", so they were used
only in this private experiment. Kernels:

| kernel | version | role |
|---|---|---|
| `birdclef26-sed-train` | v1 | training |
| `birdclef26-perch-sed-blend` | v1 | blend, submission 57434345 |
| `birdclef26-perch-sed-blend` | v2 | SED alone, submission 57434639 |

Datasets: `birdclef-sed-cache` (07:35 UTC), and `birdclef-kernel-code` and
`birdclef-sed-models` (08:33 UTC), all on 2026-10-20.

### Phase B: the Perch kernel as teacher, same windows

Phase B changed one thing: the targets of the 12,000 pseudo windows. For each
class, A's values were re-assigned in the order of the Perch kernel's own scores
for that class (`teacher.py`, quantile mapping). Prevalence and confidence stay
A's, and only the ranking changes. Windows, labelled and focal data, fold, seed,
recipe and W_SED are the pilot's.

| submission | public | private |
|---|---|---|
| SED alone, A (public teacher) | 0.87753 | 0.86072 |
| SED alone, B (Perch teacher) | 0.85621 | 0.87352 |
| blend w=0.5, A | 0.91718 | 0.91148 |
| blend w=0.5, B | 0.89834 | 0.91027 |

**No-go, and the teacher is not the main cause.**
- B's SED alone gains 0.0128 on private but loses 0.0213 on public. The two
  splits disagree by more than the gain.
- The B blend is 0.0012 below A on private and 0.0188 below on public. It is still
  0.017 below Perch alone.
- The rule's go-bars were +0.02 for SED alone and 0.9318 for the blend. Both were
  missed.
- Its third case, "B within 0.01 of A", is missed by 0.003 on private. With public
  moving the other way, this reads as no teacher effect beyond noise.

Data scale is the next suspect, then the recipe.

**The two teachers order the windows very differently.** Per-class Spearman(A, B)
over the 12,000 windows:

| | Spearman(A, B) |
|---|---|
| q25 / median / q75 | 0.155 / 0.262 / 0.385 |
| head classes (75) | median 0.389 |
| zero-shot classes (157) | median 0.203 |
| cold-start classes (2) | median 0.536 |

The 10 lowest are all zero-shot birds, with ρ from −0.115 to 0.010. The SED was
trained on two very different orderings of the same windows, and its LB barely
moved. At this scale, the 12,000 pseudo windows do not decide the SED's score.

**Other details.**
- Fold-0 validation AUC ended at 0.9729 (A: 0.9697). Both are inflated: the
  Perch teacher's heads were fit on all labelled files, fold 0 included.
- The teacher's smoothed scores tie on about 70 of the 12,000 windows per class,
  never on a whole row. Tied windows share the mean of their A values, so sorted B
  differs from sorted A by a median of 1.2e-4 and a maximum of 0.057.
- The teacher reproduces `kernel_perch` to 1.1e-16 on 20 unlabelled files.

**Cost.**

| step | time |
|---|---|
| Perch over all 127,104 unlabelled windows, Mac CPU | ~1.5 h (42 ms/window), 43 resumable shards |
| B targets, including a fresh embed of the 739 labelled windows | a few minutes (not timed) |
| training, Kaggle T4 | epoch 0 262 s, then ~74 s per epoch, ~28 min in all |

Perch's scores now exist for every unlabelled window, so B targets for all 127,104
cost minutes, not hours.

| kernel | version | role |
|---|---|---|
| `birdclef26-sed-train-b` | v1 | training on B targets |
| `birdclef26-perch-sed-blend` | v3 | B blend, submission 57441058 |
| `birdclef26-perch-sed-blend` | v4 | B SED alone, submission 57441268 |

Datasets: `birdclef-sed-targets` (13:18 UTC) and a `birdclef-sed-models` version
holding both ONNX files (13:54 UTC), on 2026-10-20.

## 17. Files

```
birdnet.py            BirdNET V2.4 wrapper; logits + penultimate embedding
perch.py              Perch v2 ONNX wrapper; logits + embedding, verified vs SavedModel
embed_labelled.py     739 labelled soundscape segments  -> data/labelled.npz
embed_train.py        131,943 focal segments, sharded   -> data/emb_train/
embed_focal_perch.py  Perch focal pool, 30 per class    -> data/focal_cap30_perch.npz
cv.py                 macro ROC-AUC, zero-shot baseline
head.py               linear head + blends; tie-aware rank()
perch_head.py         BirdNET vs Perch, 10 seeds paired  0.8880 -> 0.9144
rung1c.py             focal + soundscape                 (negative)
coldstart.py          cold-start heads on focal audio
pseudo.py             pseudo-labelling unlabelled soundscapes
mlp.py                nonlinear heads                    (negative)
smooth.py             temporal smoothing sweep
mixup.py, mixeval.py  focal calls mixed into soundscapes (negative)
kernel/               BirdNET kernel, v1-v4              0.87008 private
kernel_perch/         Perch kernel                       0.92681 private
sed_model.py          log-mel front-end + HGNetV2-B0 SED head, ONNX export
sed_cache.py          int16 pilot cache: pseudo, labelled, focal windows
sed_train.py          one SED fold on Kaggle GPU
sed_infer.py          SED on CPU via ONNX Runtime; rank blend
kernel_sed_train/     SED training kernel
teacher.py            kernel_perch as functions; Perch over unlabelled; B targets
kernel_sed_train_b/   SED training kernel on B targets
kernel_blend/         Perch + SED kernel                 0.91148 / 0.91027 private (no-go)
```

Self-checks that need no competition data: `python3 rung1c.py --selftest`,
`python3 smooth.py --selftest`, `python3 head.py --selftest`, `python3 sed_model.py --selftest`,
`python3 sed_train.py --selftest`. `python3 teacher.py --selftest` needs the competition
data and the Perch ONNX.
