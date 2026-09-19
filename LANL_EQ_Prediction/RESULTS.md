# LANL Earthquake Prediction — Results

Predicting time-to-failure of laboratory earthquakes from a 4 MHz acoustic stream.
Kaggle competition, closed June 2019; entered via late submission, so scores are real
but no leaderboard position is awarded.

| | |
|---|---|
| **Best private MAE** | **2.65160** |
| Constant-median floor | 2.8423 (measured, not quoted) |
| Competition winner | 2.26589 |
| Signal captured | **32%** of the 0.576 MAE gap between floor and winner |
| Submissions used | 4 |
| Total compute | ~25 minutes on a 15 GB laptop, no GPU |

The headline result is not the score. It is that **four separate, well-motivated
improvements all produced nothing**, and the reason why is measurable.

---

## 1. Problem

A rock sample in a double direct shear rig fails in repeated stick-slip cycles.
A continuous acoustic signal is recorded. Given a 150,000-sample window
(40.1 ms), predict the seconds remaining until the next failure. Scored on MAE.

The training file is one continuous 9.1 GB stream: **629,145,480 rows**, two
columns. The test set is 2,624 disjoint 150,000-row segments.

---

## 2. The finding that governs everything

`time_to_failure` is a sawtooth. Counting resets across the full stream:

```
16 resets -> 17 segments -> only 15 complete cycles
   (cycle 0 is truncated at the file start, 1.468s;
    cycle 16 never reaches failure, ttf 11.619 -> 9.760)

cycle duration: min 7.059s  max 16.107s  mean 9.612s  std 3.824s
```

**The effective sample size is 15, not 629 million.** Cycle length varies by a
factor of 2.3 with a coefficient of variation near 40%, which is what the
organisers meant by providing "considerably more aperiodic" data than their
earlier work.

Every result below follows from this one number.

### Supporting measurements

| Measurement | Value | Why it matters |
|---|---|---|
| Sampling rate | **3.74 MHz** | Derived, not documented: `ttf` jumps once per *exactly* 4096 rows by 1.0955 ms |
| `ttf` structure | piecewise-constant over 4096-row blocks | The acquisition system timestamped whole batches; targets are stale by up to 1.1 ms |
| Acoustic distribution | mostly within ±25 counts; `\|x\|>100` in 0.119% of samples | The precursor lives in continuous low-level emission, not in spikes |
| `ttf` precision | needs float64 | Cast to float32 and 99% of row-to-row decrements collapse to zero, destroying the block structure |

---

## 3. Validation design

Leave-one-earthquake-out, 15 folds. Cycles 0 and 16 stay in every training set but
are never validation folds.

Windows are 150,000 rows to match the test segments; the target is the `ttf` of the
last row. Windows straddling a reset are dropped (16 of them) because they contain
the failure itself, which no test segment does. That leaves **4,178 windows**.

Random K-fold would leak: neighbouring windows share rows and sit on the same
sawtooth, so a shuffled split scores far better than the private leaderboard ever
will.

**Every number in this document is per-fold.** The mean alone is misleading, as
section 6 shows.

---

## 4. Submissions

All four scored against the real leaderboard.

| # | Model | Features | CV mean | CV std | Public | Private |
|---|---|---|---|---|---|---|
| 1 | `log_nusvr` | 7 (gbm-selected) | 2.0356 | 0.8316 | 1.60825 | 2.70127 |
| 2 | `ridge` | 7 (gbm-selected) | 2.2090 | **0.6796** | 1.81966 | 2.65632 |
| 3 | **`blend(log_nusvr, ridge)`** | 7 (gbm-selected) | 2.0965 | 0.7609 | 1.67898 | **2.65160** |
| 4 | `log_nusvr` | 5 (own-selected) | **2.0154** | 0.8371 | **1.59476** | 2.70235 |

Public LB rank for submission 1: **3233 / 4519** (public median was 1.5125).

---

## 5. Progression

| Stage | What | CV mean | CV std |
|---|---|---|---|
| Rung 0 | Constant median | 2.8423 | 0.7066 |
| Rung 1 | 16 hand-built features + LightGBM | 2.2172 | 0.8036 |
| Rung 2 | Backward elimination to 7 features | 2.1228 | 0.8036 |
| Rung 3 | Model class + log target + blending | 2.0356 | 0.8316 |
| Rung 4 | Four further ideas | no gain | no gain |

Feature families, all computed on one window:

- **Energy** — std of 100-sample (26.7 µs) and 1,000-sample (267 µs) chunks, then
  aggregated. Chunking first makes these robust; a global std is dominated by rare
  spikes.
- **Amplitude** — high quantiles and max of `|x|`.
- **Rate** — fraction of samples above 20, 50, 100 counts.

Correlation with `ttf`, strongest first: `c1k_std_q95` −0.431, `c100_std_q95`
−0.431, `frac_gt20` −0.421 … `abs_max` −0.188, `c1k_std_slope` −0.011.

The best predictors are moderately-elevated activity, not the largest spike.
`abs_max` — the single biggest event in the window — is nearly useless.

---

## 6. Principal finding: CV mean and the public leaderboard both mislead

Across the four submissions:

```
corr(CV mean, public LB)  = +0.9986      (ordering is perfectly monotone)
corr(CV std,  private LB) = +0.8596
corr(CV mean, private LB) = -0.8068      <- better CV mean, WORSE private
corr(public,  private LB) = -0.7884      <- better public,  WORSE private
```

The best public score (1.59476) has the worst private score (2.70235). The model
with the worst CV mean (`ridge`, 2.2090 — only 0.63 better than a constant) has
among the best private scores, because its worst fold was the best of any model
(3.7105 vs 3.89–3.96).

This reproduces, on our own numbers, the effect the competition is known for: the
winning team finished **354th** on the public leaderboard, second place 668th,
eighth place 1771st. Public LB was 13% of the test data.

**CV mean and public LB measure the same thing** — how well the model fits these
15 cycles — and that quantity anti-correlates with generalisation here.

CV was also optimistic in absolute terms: `log_nusvr` scored 2.0356 in CV and
2.70127 on private, a gap of 0.67, *after* leave-one-earthquake-out.

---

## 7. What did not work

Four ideas, each defensible in advance, each worth writing down so nobody repeats them.

### 7.1 Model-specific feature selection — made private worse

Backward elimination run separately for `log_nusvr` produced a genuinely different
subset (only 2 of 7 features shared with the GBM subset). It kept all three
`frac_gt*` rate features, which GBM had discarded early.

The reason is structural: the three thresholds correlate 0.83–0.97, so axis-aligned
tree splits treat them as redundant, while an RBF kernel can combine them smoothly.
Those three thresholds are three points on the amplitude-frequency curve, and their
slope falls from 4.37 far from failure to 2.75 near it — the b-value decrease that
seismology associates with impending failure.

Real effect, correct physics. Result: CV mean −0.020, public −0.013,
**private +0.001**.

### 7.2 Six times more training windows — literally no change

Stride reduced from 150,000 to 25,000, giving **25,064 windows** instead of 4,178.

| Model | 4,178 windows | 25,064 windows |
|---|---|---|
| `gbm` | 2.2170 / 0.7978 | 2.2205 / 0.7969 |
| `ridge` | 2.1961 / 0.7404 | 2.1948 / 0.7417 |
| `log_nusvr` | **2.0750** / 0.8355 | **2.0750** / 0.8468 |

`log_nusvr` is identical to four decimal places. Overlapping windows add rows, not
information — the direct experimental confirmation that the sample size is 15.

### 7.3 Explicit b-value feature — redundant by construction

Added `(log10 frac_gt20 − log10 frac_gt100) / log10(5)`. It correlates 0.310 with
`ttf`. It changed nothing (`gbm` 2.2172 → 2.2170) because it is a deterministic
function of two features already present.

### 7.4 Spectral features — strongest correlations, negligible gain

Ten frequency-domain features (8 log-spaced FFT band energies, spectral centroid,
95% rolloff) — the first features in the set that measure *rate of oscillation*
rather than *amplitude*.

They produced the strongest individual correlations found anywhere in this work:

```
spec_rolloff95  +0.562        best time-domain feature: -0.431
spec_centroid   +0.552
```

The sign says the spectrum shifts *downward* approaching failure, consistent with
larger cracks radiating lower frequencies. But:

```
corr(spec_rolloff95, frac_gt20) = -0.727

in-sample R2:   time-domain 17 features = 0.4320
                spectral 10 features    = 0.3541
                all 27 features         = 0.4659    (+0.034)
```

Ten features from an entirely different domain add 0.034 of R². They observe the
same underlying process through a different lens.

### Conclusion

The problem is **information-limited**, not feature-limited or model-limited.
Roughly 2.65 is what 15 earthquakes support with window statistics. Closing the gap
to 2.2659 requires something qualitatively different.

---

## 8. Limitations

**The private leaderboard was consulted four times.** The rule "CV fold-std predicts
private" is derived from exactly those four probes. In a live competition this is
impossible — the private leaderboard only appears after the deadline — and building
a rule from four points is the same overfitting that produced the shakeup.

The defensible version of the rule, stated a priori: *when two models have similar
CV means, prefer the one with the smaller fold spread.* The indefensible version:
*pick the one that scored 2.65.*

Other limitations:

- Feature subsets were selected on the same CV they were then scored on, so those CV
  numbers are optimistic. Elimination traces are printed in full in `GUIDE.md`; both
  are flat from k=14 down to k=4, meaning the choice of k is noise.
- `NuSVR` hyperparameters (`nu=0.7, C=1.0`) were never tuned.
- No frequency-domain feature beyond the basic FFT summary was tried (no MFCC,
  wavelet, or autocorrelation features).

---

## 9. Reproducing

```bash
./fetch_data.sh                      # from the repo root; needs Kaggle rules accepted

python3 baseline.py                  # constant + LOEO harness       CV 2.842
python3 gbm.py                       # 16 features + LightGBM        CV 2.217
python3 select.py --model=gbm        # backward elimination -> 7     CV 2.123
python3 models.py                    # 8 model variants
python3 models.py --submit=blend_log_nusvr_ridge
```

| File | Role |
|---|---|
| `baseline.py` | Window extraction, cycle boundaries, `loeo` CV harness, constant predictor |
| `gbm.py` | 27 features, cached per stride; LightGBM |
| `select.py` | Greedy backward elimination, per model |
| `models.py` | Model zoo (GBM / NuSVR / Ridge, log-target and blend wrappers), submission writer |
| `GUIDE.md` | Full working notes, every measurement, all traces |

Everything routes through one interface, so a new model is one function:

```python
loeo(y, cyc, fit_predict, X)       # fit_predict(X_tr, y_tr, X_va, n_va) -> predictions
```

Each script carries assertions that fail if the split, the feature matrix, or the
fold count breaks.

Reading the 9.1 GB CSV naively exhausts 15 GB of RAM; `baseline.py` reads from
memory-mapped `.npy` caches built once.

---

## 10. What this project demonstrates

Not a competitive score. It demonstrates:

1. **Recovering effective sample size from raw data.** 629M rows, 15 independent
   observations, and every subsequent decision made accordingly.
2. **Building a validation scheme that matches the generating process** rather than
   the row count.
3. **Measuring that the leaderboard lies**, on own submissions, and quantifying the
   direction and size of the lie.
4. **Reporting negative results.** Four improvements that should have worked,
   didn't, with the measurement explaining each failure.
5. **Knowing when to stop.** The remaining 0.39 MAE is not reachable by more
   features or more models, and the evidence for that is in section 7.
