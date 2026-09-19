# LANL Earthquake Prediction

**Status: CLOSED (3 Jun 2019). Late submission open** — score milega, rank nahi.
Host: Los Alamos National Laboratory. Prize tha $50k. 4,516 teams, 28,088 entrants.
Winner private MAE: **2.26589** (The Zoo). 2nd–10th: 2.2967 → 2.3331.

Ye Kaggle ki sabse mashhoor **shakeup** comp hai. 1st place team public LB pe **354th** thi. 2nd place 668th thi. 8th place 1771st thi. Yaani public LB aur private LB me koi rishta nahi tha.

Portfolio ke liye ye comp ki asli value modelling nahi, **validation discipline** hai. "Maine yahan overfitting se bacha" ek behtar story hai "maine yahan bada model banaya" se.

---

## 1. Problem

Lab me rock pe double direct shear experiment chalaya gaya — do fault gouge layers ko constant normal load ke saath shear kiya. Fault repetitive stick-slip cycles me fail hoti hai (lab earthquakes). Continuous acoustic signal record hua.

**Task:** acoustic signal se predict karo ki agla lab earthquake kitne seconds baad aayega.

### Metric

**Mean Absolute Error** predicted vs actual `time_to_failure`.

### Submission

```csv
seg_id,time_to_failure
seg_00030f,0
seg_0012b5,0
```

Har test segment ke liye ek single number — us segment ki **aakhri row** se agle earthquake tak ka time.

---

## 2. Data

```
Total: 10.42 GB, 2,626 files
```

| File | Kya hai |
|---|---|
| `train.csv` | **Ek single continuous segment.** 2 columns: `acoustic_data` (int16), `time_to_failure` (float64). ~629M rows. |
| `test/` | 2,624 chhote segments, har ek **exactly 150,000 rows**. Har segment ke andar continuous hai, par segments aapas me continuous nahi hain. |
| `sample_submission.csv` | 2,624 rows |

Sampling rate maine measure ki: **3.74 MHz** (4096 rows per 1.0955 ms block). 150,000 rows ≈ 0.0401 s of signal.

### Data ke baare me teen cheezein jo sab kuch decide karti hain

**1. Train me sirf 16 earthquakes hain — aur sirf 15 poore cycles.**
`time_to_failure` ek sawtooth hai — linearly ghatta hai, phir earthquake pe 0 hoke reset. Poore 629M rows me 16 resets hain, par pehla aur aakhri segment adhoora hai (Section 2b). Yaani tumhare paas effectively **15 independent samples** hain, 629 million nahi. Ye comp ki poori kahani hai.

**2. `time_to_failure` me artifacts hain.**
Time column continuously nahi ghatta — chhote discrete jumps hain kyunki data acquisition system batches me likhta tha (~4096 rows ka block same-ish timestamp). Isko dhyan se handle karo, warna target noise ban jaata hai.

**3. Test segments train se overlap nahi karte aur unka order unknown hai.**
Isliye koi bhi temporal/sequential trick kaam nahi karti. Har 150k window ko independent treat karna padega.

---

## 2b. EDA — maine chalake nikala

`train.csv` = **9.1 GB, 629,145,480 rows.** Ye numbers actual file se hain.

### 16 resets → sirf 15 poore cycles

| cycle | rows | length | duration |
|---|---|---|---|
| 0 | 0 – 5,656,574 | 5.66M | **1.468s** ← partial (file yahin se shuru hota hai) |
| 1 | 5,656,574 – 50,085,878 | 44.43M | 11.540s |
| 2 | 50,085,878 – 104,677,356 | 54.59M | 14.180s |
| 3 | 104,677,356 – 138,772,453 | 34.10M | 8.856s |
| 4 | 138,772,453 – 187,641,820 | 48.87M | 12.693s |
| 5 | 187,641,820 – 218,652,630 | 31.01M | 8.054s |
| 6 | 218,652,630 – 245,829,585 | 27.18M | 7.059s |
| 7 | 245,829,585 – 307,838,917 | 62.01M | **16.107s** ← sabse lamba |
| 8 | 307,838,917 – 338,276,287 | 30.44M | 7.905s |
| 9 | 338,276,287 – 375,377,848 | 37.10M | 9.637s |
| 10 | 375,377,848 – 419,368,880 | 43.99M | 11.426s |
| 11 | 419,368,880 – 461,811,623 | 42.44M | 11.024s |
| 12 | 461,811,623 – 495,800,225 | 33.99M | 8.828s |
| 13 | 495,800,225 – 528,777,115 | 32.98M | 8.565s |
| 14 | 528,777,115 – 585,568,144 | 56.79M | 14.752s |
| 15 | 585,568,144 – 621,985,673 | 36.42M | 9.459s |
| 16 | 621,985,673 – 629,145,480 | 7.16M | 1.859s ← **truncated**, ttf kabhi 0 tak nahi pahunchta (11.619 → 9.760) |

**Poore cycles: 15.** Cycle 0 aur 16 partial hain — 16 me to earthquake aaya hi nahi.

```
duration: min 7.059  max 16.107  mean 9.612  std 3.824
15 full cycles: [11.54 14.18 8.856 12.693 8.054 7.059 16.107 7.905
                  9.637 11.426 11.024 8.828 8.565 14.752 9.459]
```

**Coefficient of variation ~40%.** Yahi wo "considerably more aperiodic" hai jo hosts ne kaha. Cycle length 7s se 16.1s tak — 2.3× ka faraq. Koi bhi model jo "average cycle length" seekh raha hai, wo ±4s galat hoga. Constant-prediction MAE ~2.9 isi se aata hai.

**CV implication:** tumhare paas **15 usable folds** hain, 16 nahi. Aur fold sizes barabar nahi — cycle 7 (62M rows) cycle 6 (27M rows) se 2.3× bada hai. Leave-one-out karte waqt fold-wise MAE compare karna hai to ye dhyan me rakho.

### `time_to_failure` ka asli structure: 4096-row blocks

Ye wo detail hai jo target ko noisy banati hai:

```
big jumps (>1e-5 s):  har  EXACTLY 4096 rows pe  (mode = median = 4096)
jump size:            1.0955 ms
in-between step:      1.1e-9 s  ← float rounding, real signal nahi
```

Yaani **ttf 4096-row blocks pe piecewise-constant hai.** Data acquisition system 4096 samples ek batch me likhta tha aur poore batch ko ek timestamp deta tha.

`4096 / 1.0955 ms = 3.74 MHz` — ye actual sampling rate hai.

Do practical nateeje:

1. **150,000-row window = 36.6 blocks.** Window ki aakhri row ka ttf us block ke shuru ka timestamp hai, yaani tumhara target **0 se 1.1 ms** tak stale ho sakta hai. MAE ~2.3 ke against 1.1 ms ignorable hai — isliye ise over-engineer mat karna. Par jab tum overlapping windows bana rahe ho, to **block boundaries pe align karo**, warna ek hi target value wale duplicate windows ban jayenge.
2. **Zero-decrement wali baat float32 me hi dikhti hai.** Agar tumne `ttf` ko float32 me cast kiya (2.4 GB vs 5 GB — lalach hoti hai), to ~99% rows ka decrement 0 ho jayega aur fine structure gayab. Window-level **target** ke liye float32 theek hai (0–16s range, ~1e-6 precision), par **boundary detection float64 pe karo**.

### Acoustic signal

Pehle 6M rows pe (cycle 0 + cycle 1 ka hissa):

```
min -4621   max 3252   mean 4.55   std 22.77
|x| > 100:  0.119%
|x| > 500:  0.024%
```

Signal mostly ±25 ke andar rehta hai, spikes bahut rare hain. Isi wajah se **mean/max useless hain aur high quantiles (95th, 99th) + rolling std kaam karte hain** — precursor signal continuous low-level emission me hai, spikes me nahi.

### Cache bana diya

Har baar 9 GB CSV parse karna 5+ minute leta hai. `data/` me npy cache hai:

```python
ac  = np.load('acoustic.npy', mmap_mode='r')   # int16,   1.2 GB
ttf = np.load('ttf.npy',      mmap_mode='r')   # float32, 2.4 GB
```

`mmap_mode='r'` se RAM me load nahi hota — is machine pe **15 GB RAM** hai, aur CSV ko naive read karne se OOM ho jaata hai (maine kar ke dekha). Windows slice karke feature banao, poora array memory me mat lao.

Cycle boundaries (upar wali table se, reuse karo):

```python
BOUNDS = [0, 5656574, 50085878, 104677356, 138772453, 187641820, 218652630,
          245829585, 307838917, 338276287, 375377848, 419368880, 461811623,
          495800225, 528777115, 585568144, 621985673, 629145480]
```

---

## 3. Data

Rules accept ho chuke hain, `train.csv` (9.1 GB) + npy cache local me hai. Baaki (test segments) `fetch_data.sh` se aata hai:

```bash
./fetch_data.sh          # repo root se
```

Kaggle bade single files ko **zip karke** serve karta hai chahe URL me `.csv` likha ho — script `PK` magic check karke khud unzip kar deti hai.

### EDA jo abhi baaki hai

Cycle structure aur ttf quantization Section 2b me ho chuki. Baaki:

1. **Failure se pehle acoustic amplitude** plot karo (per cycle, downsampled). Dekhna ye hai ki precursor kitna pehle shuru hota hai — aur kya wo har cycle me ek jaisa hai. (Nahi hai; yahi problem hai.)
2. **Train windows ka ttf distribution vs test prediction range.** Test ka mean ttf ~4–6s hona chahiye. Agar model 8s predict kar raha hai to distribution shift hai.
3. **Ek 150k window ka raw signal** plot karke dekho ki visually ttf ka andaza lagta hai ya nahi. (Nahi lagta — ye realistic expectation set karta hai aur tumhe 1000-feature rabbit hole se bachata hai.)
4. **Test segments ka acoustic distribution vs train windows.** 2,624 segments hain; agar unka std/quantile distribution train se hatke hai to wo batata hai ki test kis cycle-phase se sample hua.

---

## 4. Approach ladder

### Rung 0 — Constant prediction ✅ done

`baseline.py` me hai. `python3 baseline.py` chalao.

Non-overlapping 150k windows, target = window ki aakhri row ka ttf, earthquake ko cross karne wali 16 windows drop. **4,178 windows.** Leave-one-earthquake-out, 15 folds, predictor = train median:

```
LOEO MAE per fold
  cycle  1: 2.8919      cycle  9: 2.4312
  cycle  2: 3.7847      cycle 10: 2.8580
  cycle  3: 2.3231      cycle 11: 2.7499
  cycle  4: 3.2464      cycle 12: 2.3170
  cycle  5: 2.2521      cycle 13: 2.2898
  cycle  6: 2.2906      cycle 14: 3.9920
  cycle  7: 4.5452      cycle 15: 2.4031
  cycle  8: 2.2594

mean 2.8423   std 0.7066   min 2.2521   max 4.5452
```

Winner 2.2659 pe tha. Yaani poori competition ka total signal **0.58 MAE** ka tha, aur 4,516 teams us 0.58 ke liye lade.

### ⚠️ Rung 0 ne jo bata diya

**`corr(fold MAE, cycle duration) = 0.971`**

Fold ki error lagbhag poori tarah us cycle ki lambai se explain ho jaati hai. Sabse kharab folds — cycle 7 (16.1s → MAE 4.55), cycle 14 (14.8s → 3.99), cycle 2 (14.2s → 3.78) — bilkul wahi teen sabse lambe cycles hain. Sabse achhe folds sabse chhote cycles hain.

Iska matlab ye problem asal me hai: **"ye wala cycle kitna lamba hoga?"** Median predictor chhote cycles pe theek hai aur lambe cycles pe phat jaata hai, kyunki wo 16s tak pahunch hi nahi sakta.

Do practical nateeje:

1. **Jo bhi feature tum banao, uska kaam ye batana hai ki cycle abhi kis phase me hai** — absolute time nahi. "Kitna stress build ho chuka hai" wali baat. Isiliye rolling std aur high quantiles kaam karte hain: microfracturing rate cycle ke saath badhti hai.
2. **Mean fold MAE ko akela mat dekho.** std 0.71 hai. Agar tumhara model mean 2.5 pe laata hai par cycle 7 pe abhi bhi 4.4 hai, to tumne kuch nahi seekha — sirf chhote cycles pe fit kiya hai. Har commit pe **per-fold table** dekho, aur specially cycle 7, 14, 2.

### Window count

```
stride 150,000 (non-overlapping) ->  4,178 windows
stride  50,000                   -> 12,532
stride  25,000                   -> 25,064
```

Stride ghatane se rows badhte hain, **information nahi** — overlapping windows 83% same data share karte hain. LOEO CV isko sambhal leta hai (overlap cycle ke andar rehta hai), par model ko lagega uske paas 25k samples hain jabki asli sample size 15 hai. Regularization usi hisaab se rakho.

### Rung 1 — Window features + GBM ✅ done

`gbm.py` me hai. 16 features (jaan-bujhke — 1000 nahi), LightGBM `objective="mae"`, `num_leaves=8`, `min_child_samples=80`. Rung 0 ka wahi `loeo` harness use karta hai. Poora run **16 seconds** (features cached).

```
                 constant    gbm
mean MAE          2.8423   2.2172     +0.6251
fold std          0.7066   0.8046     ← spread BADH gaya
worst fold        4.5452   3.9162
```

Submission: 2,624 rows, predictions 0.730–10.693, mean 5.351.

> **CV 2.217 ≠ LB 2.217.** Ye train ke cycles pe leave-one-out hai; test experiment ka alag sample hai. Original comp me bahut teams ka CV ~2.0 tha aur private LB 2.4+. Winner 2.2659 pe tha — usse compare karne ke liye asli submission chahiye, CV number nahi.

### ⚠️ Rung 1 ne jo bata diya: model chhote cycles pe bigadta hai

15 me se 12 folds sudhre, **3 bigde**. Wo 3 kaun se?

| cycle | duration | constant | gbm | delta |
|---|---|---|---|---|
| 6 | **7.06s** | 2.291 | 2.551 | **+0.260** |
| 8 | **7.91s** | 2.259 | 2.669 | **+0.410** |
| 5 | **8.05s** | 2.252 | 2.307 | **+0.055** |

Ye dataset ke **teen sabse chhote cycles** hain, exactly. Bigde hue folds ka mean duration **7.67s**, sudhre hue folds ka **11.42s**.

Model mean cycle length ki taraf regress kar raha hai. Lambe cycles pe wo madad hai (constant 16s tak pahunch hi nahi sakta), chhote cycles pe wo nuksan hai (constant already close tha).

Ek aur number: `corr(fold MAE, cycle duration)` **0.971 → 0.597**. Yaani model ne cycle-phase thoda seekha — error ab sirf duration se explain nahi hoti. Par abhi bhi worst folds 7 (16.1s) aur 14 (14.75s) hain. **Error dono extremes pe concentrated hai.**

Yahi tumhara agla kaam define karta hai: mean MAE ko 2.1 pe le jaana aasan hai, par **fold std ko 0.80 se neeche laana** asli progress hai. Jo bhi change karo, wo teen chhote cycles dekho — agar wahan aur bigad raha hai to tum public LB ke liye overfit kar rahe ho.

### Feature importance (full fit)

```
c100_std_mean    355     c1k_std_q95      196
abs_mean         350     c1k_std_std      147
c1k_std_slope    254     frac_gt50        129
c1k_std_max      248     std               82
frac_gt20        214     abs_q95           71
abs_max          214     abs_q999          71
c1k_std_mean     200     frac_gt100        56
c100_std_q95     161     abs_q99           52
```

Top do — 100-sample chunks ke std ka mean, aur mean absolute amplitude — dono **low-level emission rate** naapte hain. Bottom teen quantiles aur `frac_gt100` hain, jo **spikes** naapte hain. Ye directly confirm karta hai jo Rung 0 ne suggest kiya tha: signal continuous microfracturing me hai, bade events me nahi.

Agla step (Rung 2) yahi 16 se **~8 pe** aana hai, badhana nahi.

### Rung 1 ka original plan (reference)

Standard recipe jo 90% teams ne kiya:

1. `train.csv` ko **150,000-row windows** me kaato (test jaisa hi), stride ke saath overlapping windows banao (data augment karne ke liye)
2. Har window ka target = us window ki **aakhri row** ka `time_to_failure`
3. Features nikalo
4. LightGBM / XGBoost / NuSVR fit karo

Feature families (yehi log use karte the):

- **Basic stats:** mean, std, min, max, skew, kurtosis of `acoustic_data`
- **Quantiles:** 1%, 5%, 25%, 50%, 75%, 95%, 99% — aur specially **absolute value ke quantiles**
- **Rolling window stats:** rolling mean/std/min/max at windows 10, 100, 1000 → phir unke stats
- **Peak counts:** kitne samples threshold ke upar (threshold sweep karo)
- **Frequency domain:** FFT ke real/imag parts ke stats, band energies
- **MFCC:** signal ko audio maan ke librosa se MFCC — surprisingly kaam karta hai
- **Hilbert transform / envelope** stats
- **Trend:** window ke andar linear regression ka slope
- **Sub-window stats:** window ko 10 parts me kaato, har part ke stats, phir unka variation

Ye ~200–1000 features ban jaate hain.

### Rung 2 — Feature selection ✅ done

`select.py` — greedy backward elimination usi LOEO CV pe. 16 → 4 tak, har step pe wo feature girao jiske bina sabse achha ho. Runtime **4m24s** (1,400+ LightGBM fits, 11 cores).

```
 k  mean     std    dropped
16  2.2099  0.8082  -
15  2.1610  0.8032  abs_mean        ← asli gain sirf yahan
14  2.1295  0.8156  abs_q95
13  2.1251  0.8127  frac_gt50
12  2.1243  0.8116  c1k_std_q95
11  2.1223  0.8074  frac_gt20
10  2.1250  0.8111  std
 9  2.1262  0.8131  frac_gt100
 8  2.1203  0.8101  abs_q999
 7  2.1192  0.8046  c1k_std_std     ← best
 6  2.1196  0.8103  abs_max
 5  2.1193  0.8130  c1k_std_mean
 4  2.1264  0.8099  c1k_std_slope
```

**Chuni gayi 7 features:**
`abs_q99, abs_max, c1k_std_mean, c1k_std_max, c100_std_mean, c100_std_q95, c1k_std_slope`

Full `n_estimators` pe: **mean 2.1228, std 0.8036, max 3.9335.** Submission: 7 feats, pred 0.741–11.305, mean 5.047.

### Is trace ko imaandari se padho

**1. Gain lagbhag poora ek feature se aaya.** `abs_mean` girane se 2.2099 → 2.1610. Uske baad k=14 se k=5 tak sab kuch **2.1193–2.1295** ke andar hai — 0.01 ka band. Wo flat plateau hai, signal nahi. k=7 ko k=12 ke upar chunna CV noise chunna hai.

Matlab: "7 features best hain" mat kehna. Kehna ye chahiye — **"14 ke baad koi farak nahi padta, toh 7 le liye kyunki chhota model kam overfit karta hai."**

**2. Fold std bilkul nahi hila.** 0.8082 → 0.8036. Ye wo cheez thi jo Rung 1 ne target banayi thi, aur feature selection ne **usko touch nahi kiya**. Toh:

> Feature selection ne mean MAE 0.09 sudhara aur instability zero sudhari.

Ye asli nateeja hai. Fold spread features ki wajah se nahi hai — wo isliye hai ki 15 cycles ki lambai 7s se 16.1s tak hai aur model unka farak samajh nahi pa raha.

**3. Teen chhote cycles me se ek theek hua.**

| cycle | dur | constant | gbm-16 | sel-7 |
|---|---|---|---|---|
| 5 | 8.05s | 2.252 | 2.307 | **2.218** ✅ ab constant se behtar |
| 6 | 7.06s | 2.291 | 2.551 | 2.393 ❌ abhi bhi kharab |
| 8 | 7.91s | 2.259 | 2.669 | 2.616 ❌ abhi bhi kharab |
| 7 | 16.11s | 4.545 | 3.916 | 3.934 — worst fold, jaisa tha waisa |
| 14 | 14.75s | 3.992 | 3.573 | 3.558 |

**4. Selection wahi CV pe hui jispe report kar rahe hain**, toh 2.1192 optimistic hai. Iska matlab ye nahi ki number fake hai — iska matlab ye hai ki k=7 vs k=12 ka 0.005 ka farak bharosemand nahi. Trace poora print isliye hota hai taki tum khud dekh sako ki curve flat hai ya usme knee hai. Yahan flat hai.

**5. Ek chhota bonus signal:** search `n_estimators=250` pe chali aur 16 features pe 2.2099 diya, jabki `PARAMS` (400) ne 2.2172 diya. Kam trees thoda behtar — regularization wali direction sahi hai.

### Ab kya karna hai

Feature selection nichod liya. Progress ab yahan se aayega:

- **Model class badlo.** NuSVR (RBF) is comp me GBM se accha chala tha, aur wo smooth hai — extremes pe regression-to-mean alag tarah se behave karti hai. Yahi sabse sasta agla experiment hai.
- **Target transform.** MAE optimize kar rahe ho par distribution skewed hai. `log1p(ttf)` pe fit karke `expm1` se wapas laana lambe cycles pe madad kar sakta hai.
- **Fold std pe attack.** Cycle 7 (16.1s) aur 14 (14.75s) worst hain. Dekho ki un cycles me kya alag hai — kya wahan precursor der se shuru hota hai?
- **Blend.** MAE pe median-blend GBM + NuSVR + ridge.

Jo bhi karo, har baar **per-fold table** dekho, mean nahi.

### Features actually kya naapte hain

Sampling 3.74 MHz, toh timescales ye hain:

```
1 sample        = 0.267 µs
100 samples     = 26.7 µs     <- c100_* chunks (1,500 per window)
1,000 samples   = 267 µs      <- c1k_*  chunks (150 per window)
150,000 samples = 40.1 ms     <- poora window
```

Dhyan do: poora window **40 ms** ka hai, aur cycle **7–16 seconds** ka. Yaani ek window cycle ka sirf **0.25–0.5%** dekhti hai. Model ek jhalak se poore cycle ka phase batata hai.

**Physics:** fault gouge shear ke under hai. Failure paas aane par grains micro-fracture karte hain, har fracture ek chhota acoustic emission (AE) event hai. Stress badhne ke saath events ki **rate** badhti hai aur unki **size distribution** badalti hai. Signal mostly ±25 counts me rehta hai; precursor continuous low-level emission me hai, bade spikes me nahi.

Do feature families isi se nikalte hain:

| Family | Sawaal | Features |
|---|---|---|
| **Energy** | "kitni zor se hil raha hai" | `c100_std_*`, `c1k_std_*`, `std`, `abs_mean` |
| **Amplitude** | "sabse bade events kitne bade hain" | `abs_q95/q99/q999`, `abs_max` |
| **Rate** | "events kitne baar aate hain" | `frac_gt20/50/100` |

Har feature ttf ke saath kaise badalta hai (medians, 4,178 windows):

| feature | corr(ttf) | far (ttf>8) | near (ttf<2) |
|---|---|---|---|
| `c1k_std_q95` | **-0.431** | 7.03 | 12.91 |
| `c100_std_q95` | **-0.431** | 6.22 | 11.91 |
| `frac_gt20` | **-0.421** | 0.0060 | 0.0191 |
| `c1k_std_mean` | -0.358 | 3.62 | 5.32 |
| `abs_q95` | -0.347 | 10 | 13 |
| `c100_std_mean` | -0.341 | 3.43 | 4.85 |
| `abs_mean` | -0.286 | 5.06 | 5.81 |
| `frac_gt50` | -0.249 | 0.00053 | 0.00243 |
| `abs_q99` | -0.225 | 16 | 28 |
| `std` | -0.216 | 4.42 | 6.95 |
| `c1k_std_std` | -0.201 | 2.42 | 4.47 |
| `abs_q999` | -0.192 | 42 | 68 |
| `abs_max` | -0.188 | 102 | 158 |
| `c1k_std_max` | -0.178 | 21.4 | 34.5 |
| `frac_gt100` | -0.159 | 6.7e-06 | 2.2e-04 |
| `c1k_std_slope` | **-0.011** | 1.5e-05 | -6.5e-04 |

Padhne layak baatein:

- **Sab negative hain** — failure paas aane par har cheez badhti hai. Expected.
- **Sabse strong: `*_std_q95` aur `frac_gt20`** (~-0.43, -0.42). Yaani "thoda-sa upar wale" events, spikes nahi.
- **`abs_max` sabse kamzor amplitude feature hai** (-0.188). Ek sabse bada spike lagbhag bekar hai.
- **`frac_gt20` 3.2× badhta hai** (0.60% → 1.91%), `frac_gt100` **33× badhta hai** (6.7e-6 → 2.2e-4). Rate features ka relative change sabse bada hai — bas wo bahut chhote numbers pe hai.
- **`c1k_std_slope` dead hai** (-0.011). 40 ms window me trend nikalna bekaar hai jab cycle 10 seconds ka ho. Ye feature theory me achhi lagti hai, data me nahi.

### Dono models ne alag features kyun chuni

Correlation matrix se saaf hai:

```
corr(c100_std_mean, c1k_std_mean) = 0.999   <- same cheez, do timescales
corr(abs_mean,      c100_std_mean) = 0.986  <- abs_mean lagbhag duplicate hai
corr(c100_std_mean, c100_std_q95)  = 0.964

frac_gt* aapas me (raw):  0.83 - 0.97
frac_gt* aapas me (log):  0.61 - 0.77   <- log space me kaafi kam
```

**`abs_mean` dono runs me sabse pehle gira** — aur yahi ek drop tha jisne asli gain diya (gbm 2.2099→2.1610, log_nusvr 2.0741→2.0410). Wajah ab saaf hai: wo `c100_std_mean` ka 0.986-correlated duplicate hai, aur weaker version hai (corr with ttf -0.286 vs -0.341). Redundant aur kamzor.

**GBM ne `frac_gt*` phenke, NuSVR ne teeno rakhe.** Kyun:

- Tree axis-aligned splits karta hai. Teen 0.83–0.97 correlated features me se ek pe split karne ke baad baaki do extra kuch nahi dete — GBM ek rakhta hai aur baaki drop kar deta hai.
- RBF kernel scaled space me **distance** pe kaam karta hai, toh wo teeno ko ek saath, smoothly combine kar sakta hai. Aur teen thresholds (20, 50, 100) asal me **amplitude-frequency curve ke teen points** hain — unke beech ka ratio distribution ka **shape** batata hai, sirf level nahi.

Ye shape actually meaningful hai. Crude slope nikala:

```
slope = (log10 frac_gt20 - log10 frac_gt100) / (log10 100 - log10 20)

corr with ttf = +0.310
median far (ttf>8): 4.37      near (ttf<2): 2.75
```

Failure paas aane par slope **4.37 se 2.75 pe girti hai** — yaani bade events chhoton ke mukable zyada common ho jaate hain, distribution flat ho jaati hai. Earthquake seismology me ye **b-value drop** kehlata hai aur failure ka classic precursor mana jaata hai. NuSVR ko teen thresholds dena effectively use ye slope compute karne de raha hai; GBM ko ek dena ye chheen leta hai.

> Isliye agar aage koi ek feature add karni ho, to **ye slope khud** — explicitly compute karke — sabse defensible candidate hai. (Par yaad rakho: Rung 3 ka nateeja ye tha ki CV mean sudharna private pe ulta pad sakta hai.)

### Rung 3 — Model class + target transform ✅ done, aur ISKA nateeja sabse important hai

`models.py` — wahi 7 features, 8 alag models. Runtime 26s.

```
        CV mean   CV std   CV max
constant  2.8423   0.7066   4.5452
gbm       2.1228   0.8036   3.9335
nusvr     2.1040   0.7799   3.8181
ridge     2.2090   0.6796   3.6599
log_gbm   2.1213   0.8092   3.9550
log_nusvr 2.0356   0.8316   3.8864   ← best mean
blend×3   2.1047   0.7770   3.8053
blend_log 2.0645   0.8159   3.8804
log_nusvr+ridge  2.0965   0.7609   3.7565
```

Teen submit kiye. **Ye actual Kaggle scores hain, CV nahi:**

| model | CV mean | CV std | **public** | **private** |
|---|---|---|---|---|
| `log_nusvr` | **2.0356** (best) | 0.8316 (worst) | **1.60825** (best) | **2.70127** (worst) |
| `blend(log_nusvr, ridge)` | 2.0965 | 0.7609 | 1.67898 | **2.65160** (best) |
| `ridge` | 2.2090 (worst) | **0.6796** (best) | 1.81966 (worst) | 2.65632 |

### Ordering ekdum ulti hai

```
public  se rank:  log_nusvr  >  blend  >  ridge
private se rank:  blend      >  ridge  >  log_nusvr
```

**Jo public pe sabse achha tha, wo private pe sabse kharab nikla.** Bilkul wahi cheez jisne is comp me 1st place ko public 354th pe rakha tha — bas ab tumhare apne numbers pe.

Aur ye dekho:

- **CV mean ne public LB predict kiya.** Ordering exactly same: 2.0356 < 2.0965 < 2.2090 → 1.608 < 1.679 < 1.820.
- **CV std ne private LB predict kiya.** Ordering exactly same: 0.6796 < 0.7609 < 0.8316 → 2.6563 ≈ 2.6516 < 2.7013.

Yaani jab maine Rung 1 ke baad kaha tha "mean nahi, fold std dekho" — wo sahi tha, aur ab wo opinion nahi, measured hai. `ridge` ka CV mean sabse kharab tha (2.2090, constant se sirf 0.63 behtar) par uska worst fold sabse achha tha (3.6599), aur private pe wo `log_nusvr` ko 0.045 se haraata hai.

### CV apne aap bhi optimistic tha

```
log_nusvr:  CV 2.0356  ->  private 2.70127   (gap 0.67)
```

15 cycles pe leave-one-out karne ke baad bhi CV private se **0.67 optimistic** tha. Kyun: test segments usi experiment se hain par unka cycle-phase mix alag hai, aur features + model dono usi 15 cycles pe tune hue.

Iska sabak: **CV number ko absolute mat maano, comparison ke liye use karo.** "Mera CV 2.03 hai matlab main winner (2.2659) se behtar hoon" — galat. Sahi: "model A ka CV std model B se kam hai, isliye A safer hai."

### Kahan khade hain, imaandari se

```
public 1.60825 -> rank 3233 / 4519   (public LB ka median 1.5125 tha)
private 2.65160 -> winner 2.26589, 10th place 2.33312
```

Public LB pe median se neeche. Private pe winner se 0.39 door. **~30 minute ke kaam ka yahi realistic outcome hai** — aur ye theek hai. Constant baseline 2.842 se hum 2.6516 pe aaye, yaani available signal (2.842 → 2.266 = 0.576) ka **32%** nikal liya. Baaki 68% ke liye 4,516 teams ne mahine lagaye.

### Ab agar aage badhna hai

- **Overlapping windows** (`windows(stride=25000)` → 25,064 windows). Abhi sirf 4,178 non-overlapping use kar rahe hain.
- **NuSVR ke `nu`/`C` tune karo** — abhi `nu=0.7, C=1.0` guess hai, tuned nahi.
- **Cycle 7 (16.1s) aur 14 (14.75s)** — worst folds, aur teeno models me worst. Wahan kya alag hai?
- **Feature set wapas kholo** — 7 features log_nusvr ke liye chuni gayi thi nahi, `gbm` ke liye chuni gayi thi. Alag model ke liye alag subset best ho sakta hai.
- Jo bhi karo: **CV std dekho, mean nahi.** Ab iske liye evidence hai.

### Rung 4 — Teen "obvious" improvements, teeno dud

Rung 3 ke baad teen cheezein try ki jo theory me pakki lagti thi. **Teeno se kuch nahi mila.** Ye section isliye hai taki tum inhe dobara na karo.

#### 1. Model-specific feature selection — LB pe *bigda*

`log_nusvr` ke liye alag subset chuna (`select.py --model=log_nusvr`). Subset sach me alag nikla — 7 me se sirf 2 common:

```
gbm ke liye (7):        abs_q99, abs_max, c1k_std_mean, c1k_std_max,
                        c100_std_mean, c100_std_q95, c1k_std_slope
log_nusvr ke liye (5):  c100_std_mean, c100_std_q95,
                        frac_gt20, frac_gt50, frac_gt100
```

| | CV mean | CV std | public | private |
|---|---|---|---|---|
| `log_nusvr` + gbm-7 | 2.0356 | 0.8316 | 1.60825 | 2.70127 |
| `log_nusvr` + own-5 | **2.0154** | 0.8371 | **1.59476** | **2.70235** ← bigda |

CV mean 0.02 sudhra, public 0.013 sudhra, **private 0.001 bigda.**
`blend` ke liye bhi selection chalayi: mean 2.0965→2.0702 behtar, std 0.7609→**0.7866 kharab**. Teesri baar wahi pattern.

#### 2. Overlapping windows (6× data) — bilkul zero asar

`stride=25000` → **25,064 windows** (4,178 ki jagah).

| | stride 150k (4,178) | stride 25k (25,064) |
|---|---|---|
| `gbm` | 2.2170 / 0.7978 | 2.2205 / 0.7969 |
| `ridge` | 2.1961 / 0.7404 | 2.1948 / 0.7417 |
| `log_nusvr` | **2.0750** / 0.8355 | **2.0750** / 0.8468 |

`log_nusvr` ka mean **4 decimal tak identical** hai. 6× rows, zero change.

Ye guide ki thesis ka seedha proof hai: **effective sample size 15 earthquakes hai, windows ki ginti nahi.** Overlapping windows rows badhate hain, information nahi. Agar tumne kabhi socha ki "aur data generate kar lo", ye uska jawab hai.

#### 3. b-value feature — redundant by construction

`(log10 frac_gt20 − log10 frac_gt100) / log10(5)` explicitly add ki. `corr(bvalue, ttf) = 0.310`.

| model | 16 feats | 17 feats (+bvalue) |
|---|---|---|
| `gbm` | 2.2172 / 0.8046 | 2.2170 / 0.7978 |
| `ridge` | 2.1964 / 0.7418 | 2.1961 / 0.7404 |
| `log_nusvr` | 2.0741 / 0.8348 | 2.0750 / 0.8355 |

Kuch nahi. Obvious wajah: wo `frac_gt20` aur `frac_gt100` ka **deterministic function** hai, aur dono already set me hain. Model wo relationship khud bana sakta tha.

#### 4. Spectral features — sabse strong correlation, phir bhi bekaar

8 log-spaced FFT band energies + centroid + 95% rolloff. Ye pehli baar **frequency** naap rahe the — baaki saare 17 features time-domain amplitude statistics hain.

Individual correlations sabse strong nikle:

```
spec_rolloff95   +0.562      <- best overall
spec_centroid    +0.552
band0            +0.541
(best time-domain: c1k_std_q95 -0.431)
```

Positive corr ka matlab: failure paas aane par spectrum **neeche** shift hota hai. Bade cracks lower frequency radiate karte hain — physically sahi.

Par CV pe:

| model | 17 feats | 27 feats (+spectral) |
|---|---|---|
| `gbm` | 2.2170 / 0.7978 | 2.2100 / 0.7915 |
| `ridge` | 2.1961 / 0.7404 | 2.1745 / 0.7528 |
| `log_nusvr` | 2.0750 / 0.8355 | 2.0849 / 0.8223 |

Mixed aur marginal. Wajah:

```
corr(spec_rolloff95, frac_gt20)     = -0.727
corr(spec_rolloff95, c100_std_mean) = -0.562

in-sample R2:  time-domain 17 = 0.4320
               spectral 10    = 0.3541
               all 27         = 0.4659   <- sirf +0.034
```

Spectral features **usi cheez** ko dusre lens se naap rahe hain. 10 naye features milke R² me 0.034 add karte hain.

### Rung 4 ka asli sabak

Chaar alag ideas — model-specific selection, 6× data, physics-motivated feature, poora naya feature domain — aur **kisi se bhi meaningful gain nahi**. Har ek wahi underlying signal dobara express kar raha tha.

Problem **information-limited** hai, feature-limited ya model-limited nahi. 15 earthquakes se jitna nikal sakta tha, ~2.65 pe lagbhag nikal chuka hai. Winner ka 2.2659 tak pahunchne ke liye kuch qualitatively alag chahiye — ya to bahut zyada mehnat wali cycle-phase modelling, ya kuch jo hum abhi nahi dekh pa rahe.

> ### ⚠️ Ek imaandar baat: hum ab leaderboard probe kar rahe hain
>
> Humne private LB ko **4 baar** dekh liya. "CV std private predict karta hai" wala rule **unhi 4 probes se** nikla hai.
>
> Live competition me ye possible hi nahi hai — private LB deadline ke baad hi dikhta hai. Aur 4 points pe rule banana bilkul wahi overfitting hai jisse ye comp ne 4,516 teams ko maara.
>
> Isko sikhne ke liye use karo, strategy ke liye nahi. Live comp me sirf ek hi cheez available hoti hai: **CV, aur usme bhi fold spread.** Rule ye rakho: *"jab do models ka CV mean paas ho, tab kam fold-std wala chuno"* — ye a-priori defensible hai. *"Private 2.65 wala chuno"* defensible nahi hai.

### Rung 2 ka original plan (reference)

**Yahi wo jagah hai jahan ye comp jeeti gayi thi.** 1000 features + 15 earthquakes = guaranteed overfit.

Winning solutions ne feature count **drastically** kam kiya — kuch ne 10 se bhi kam features use kiye. Top teams ne genetic algorithms aur permutation importance se selection kiya.

Practical:
- Permutation importance leave-one-earthquake-out CV pe (single split pe nahi)
- Feature stability check: alag folds me feature ki importance consistent hai ya nahi
- Agar ek feature drop karne se CV nahi bigadta, to **drop kar do**
- Target: **<20 features**

Sabse zyada predictive feature families jo ikhatthi hui:
- Absolute acoustic value ka **high quantile** (95th, 99th)
- Rolling std ka mean
- Peak count above threshold

Physical intuition: rock failure se pehle microfracturing badhti hai → continuous low-level acoustic emission badhta hai. Mean/max nahi, **distribution ka upper-middle hissa** batata hai.

### Rung 3 — Model choice

- **NuSVR** (RBF kernel) — is comp me GBM se accha perform kiya, kyunki smooth aur kam variance hai
- **LightGBM** with heavy regularization: shallow depth (3–4), high `min_child_samples`, low learning rate, aggressive `feature_fraction`
- **Simple ridge/linear regression** on 5–10 features — seriously, ye competitive tha
- **Blend** of the above — MAE pe median blending achhi hai

Neural nets (RNN/CNN on raw signal) bahut try hue aur **zyadatar fail** hue. 16 samples pe deep model overfit hi karega. Agar karna hai to heavily regularized 1D-CNN on downsampled signal, aur CV pe skeptical raho.

### Rung 4 — Target clipping

`time_to_failure` non-negative hai. Predictions ko `[0, 16]` (ya train ke max cycle length) pe clip karo. Free MAE improvement.

Aur: MAE optimize kar rahe ho to **median** predict karna behtar hai mean se. Loss function ko MAE/Huber pe set karo, MSE pe nahi.

---

## 5. Validation — yahi poora comp hai

**Public LB sirf 13% test data pe tha.** Private 87% pe. ~2624 segments me 13% ≈ 340 segments. Public LB me itna variance tha ki wo effectively random tha — isiliye 354th → 1st ka shakeup hua.

### Rules

1. **Leave-One-Earthquake-Out CV.** 15 poore cycles = 15 folds (Section 2b). Cycle 0 aur 16 partial hain — unhe train me rakho, fold mat banao. Ek poora cycle hold out karo, baaki pe train. Random KFold **kabhi nahi** — overlapping windows leak karte hain aur CV ko fake-achha dikhate hain.
2. **Fold-wise MAE ka spread dekho, sirf mean nahi.** Agar folds me MAE 1.8 se 3.5 tak hai, to tumhara model unstable hai chahe mean achha ho.
3. **Public LB ko ek 17th fold ki tarah treat karo**, ground truth ki tarah nahi. Agar CV improve ho raha hai aur LB nahi, to CV pe bharosa karo.
4. **Model selection CV pe karo, LB pe nahi.** Ye likhna aasan hai aur karna mushkil — jab tum LB pe 500 rank upar jaate ho to temptation bahut hoti hai.
5. Final submission ke liye **do** submissions chuno: ek best-CV, ek best-LB. (Original comp me 2 allowed the.) Story ye hai ki best-CV wali jeeti.

### Realistic targets

| Approach | Expected private MAE |
|---|---|
| Constant median | **2.842 measured** (LOEO, 15 folds) |
| 16-feature LightGBM (gbm.py) | CV 2.217 |
| Selected 7 features + LightGBM (select.py) | CV 2.123 |
| log_nusvr, 7 feats | CV 2.036 → **private 2.70127** |
| blend(log_nusvr, ridge) | CV 2.097 → **private 2.65160** ← best submitted |
| ridge, 7 feats | CV 2.209 → **private 2.65632** |
| Winner | 2.2659 |

Agar tum 2.35 se neeche aate ho with honest LOEO CV, to tum us waqt top-50 me hote.

---

## 6. Portfolio angle

Ye comp modelling showcase nahi hai. Iska writeup ye hona chahiye:

> "629 million rows, par effectively 16 training samples. Public leaderboard 13% data pe tha aur winner wahan 354th tha. Maine leave-one-earthquake-out CV banaya, 1000 features se 18 pe aaya, aur LB signal ko deliberately ignore kiya. Final MAE X — jo original private LB pe rank Y hoti."

Ye ek strong story hai kyunki ye batati hai ki tum **effective sample size** samajhte ho, aur validation ko metric se upar rakhte ho. Zyadatar portfolio projects yahi miss karte hain.

Practical bonus: ye comp tumhare hardware pe **poora chal sakta hai** — GPU ki zaroorat nahi, LightGBM/sklearn CPU pe theek hai, 10 GB data 171 GB free space me aaram se aata hai.

---

## 7. Gotchas

- `train.csv` naive `pd.read_csv` se ~10 GB RAM khayega. dtypes specify karo ya chunked padho.
- Test segments exactly 150,000 rows ke hain — training windows bhi exactly 150,000 ke rakho, warna feature distributions match nahi karenge.
- Overlapping training windows se sample count badhta hai par **information nahi**. CV me ye leak ki tarah behave karta hai — isiliye earthquake-level grouping zaroori hai.
- `time_to_failure` ke discrete jumps ki wajah se window ke aakhri row ka exact ttf thoda noisy hai. Aakhri kuch rows ka median lena zyada stable hai.
- Earthquake ke bilkul paas (ttf < 0.5s) wale windows me signal alag hai. Kuch teams ne inhe drop kiya kyunki test me aise segments rare the.
- Late submission me LB rank nahi milega, par **private score milega** — original private LB se compare karke apni rank estimate kar sakte ho.

---

## Sources

- [Competition overview](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/overview)
- [Data description](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/data)
- [Final leaderboard](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/leaderboard) — solution writeups linked per team
