# BirdCLEF+ 2026 — Results

Identifying 234 species from 5-second windows of passive acoustic monitoring in the
Brazilian Pantanal. Metric is macro-averaged ROC-AUC over classes with at least one
true positive. Competition closed June 2026; winner 0.96574.

**Private leaderboard 0.87008**, from a linear head on frozen BirdNET embeddings,
blended with BirdNET's own logits on the species it knows, cold-start heads on focal
audio for the 30 classes that have neither, and temporal smoothing across the windows
of each recording.

All numbers below are 5-fold cross-validation over the 739 labelled soundscape
segments, grouped by recording file. Nothing here has been submitted.

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

## 12. Where this stands and what is next

Rung 1b, 0.8881, is the current best and has **not been submitted** — there is no
inference kernel yet, and the competition's binding constraint is a CPU-only notebook
under 90 minutes with no internet. BirdNET at 24 ms per 3-second chunk against the
hidden test set is the thing that has to be measured before any of this is a score.

Ranked by expected value, with the arithmetic from section 7:

1. ~~Train the 30 constant classes on focal audio.~~ **Done, section 8: +0.021
   private.** The remaining headroom in that cohort is real but smaller — they are at
   ~0.667 on private against ~0.885 for the rest.
2. ~~Pseudo-label the 10,592 unlabelled soundscape files.~~ **Done, section 9:
   about +0.004, and weakly supported.** The in-domain data is real but a linear
   probe on frozen features cannot extract much more from it than BirdNET already
   provides. This is an argument for fine-tuning, not against the data.
2. **Energy-based window selection.** `starts_for()` spreads windows uniformly; 12% of
   clips run past a minute with the call anywhere in them. This is the most likely
   source of label noise in the focal set — and worth retrying section 5 after fixing,
   since a cleaner focal set is the one thing that might change that result.
4. ~~Fine-tune BirdNET rather than freezing it.~~ **Not available, section 10.** The
   model ships as inference-only `.tflite`, upstream publishes no bare trainable
   checkpoint, and there is no GPU or TensorFlow here.

## 13. Files

```
birdnet.py          BirdNET V2.4 wrapper; logits + penultimate embedding
embed_labelled.py   739 labelled soundscape segments  -> data/labelled.npz
embed_train.py      131,943 focal segments, sharded   -> data/emb_train/
cv.py               macro ROC-AUC, zero-shot baseline
head.py             Rung 1b: linear head + blends          0.8881
rung1c.py           Rung 1c: focal + soundscape            0.8265  (negative)
kernel/rung0.py     submission kernel skeleton
```

Self-checks that need no competition data: `python3 rung1c.py --selftest`.
