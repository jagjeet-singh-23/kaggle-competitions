# BirdCLEF+ 2026 — Results

Identifying 234 species from 5-second windows of passive acoustic monitoring in the
Brazilian Pantanal. Metric is macro-averaged ROC-AUC over classes with at least one
true positive. Competition closed June 2026; winner 0.96574.

**Best local score: 0.8881**, from a linear head on frozen BirdNET embeddings blended
with BirdNET's own logits on the species it already knows.

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

## 7. Where this stands and what is next

Rung 1b, 0.8881, is the current best and has **not been submitted** — there is no
inference kernel yet, and the competition's binding constraint is a CPU-only notebook
under 90 minutes with no internet. BirdNET at 24 ms per 3-second chunk against the
hidden test set is the thing that has to be measured before any of this is a score.

Ranked by expected value:

1. **Build the submission kernel.** An unsubmitted 0.8881 is worth nothing, and the
   90-minute budget is a real risk that no local number addresses.
2. **Use the 1,478 labelled soundscape segments, not 739.** `embed_labelled.py`
   currently keeps one label set per segment after de-duplication. This is the only
   in-domain data that exists and the head is regularised to `C=0.01` precisely
   because there is so little of it.
3. **Energy-based window selection.** `starts_for()` spreads windows uniformly; 12% of
   clips run past a minute with the call anywhere in them. This is the most likely
   source of label noise in the focal set — and worth retrying section 5 after fixing,
   since a cleaner focal set is the one thing that might change that result.
4. **Fine-tune BirdNET rather than freezing it.** Everything here is a linear probe.
   The domain gap in section 5 is exactly what fine-tuning addresses.

## 8. Files

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
