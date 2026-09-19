# BirdCLEF+ 2026 — Acoustic Species ID in the Pantanal

**Status: CLOSED (3 Jun 2026). Late submission open** — LB pe ranking nahi milegi, par score milega.
Host: Cornell Lab of Ornithology. Prize tha $50k. 4,094 teams, 152,756 submissions.
Winner private score: **0.96574** (Nikita Babych). Top-12 sab 0.9556–0.9657 me — ekdum tight.

Portfolio ke liye achhi choice: dataset reasonable size (16 GB), metric clean, aur constraint interesting hai.

---

## 1. Problem

Brazil ke Pantanal wetlands ki continuous audio recordings me kaun si species bol rahi hai, wo identify karo.
Sirf birds nahi — **birds, amphibians, mammals, reptiles, insects**. 234 classes.

### Metric

**Macro-averaged ROC-AUC, un classes ko skip karke jinme koi true positive nahi hai.**

Do implications, dono important:

1. **Macro** hai → har class ka barabar weight. Rare species utni hi matter karti hai jitni common. Class imbalance ko ignore karna mahanga hai.
2. **ROC-AUC** hai → sirf **ranking** matter karti hai, calibration nahi. Par ranking **per class, across all rows** hoti hai. Iska matlab ek class ke andar rows ka order theek hona chahiye; alag classes ke scores ka aapas me comparable hona zaroori nahi. Ye post-processing ke liye bada darwaza kholta hai.

### Submission

```csv
row_id,<species_1>,<species_2>,...,<species_234>
BC2026_Test_0001_S05_20250227_010002_5,0.01,0.9,...
```

`row_id` = `[soundscape_filename]_[end_time]`. Har test file 1 minute ki hai → 5-second windows → **12 rows per file**.
~600 test files → ~7,200 rows × 234 columns.

### Code competition constraints — yahi asli challenge hai

- **CPU notebook ≤ 90 minutes**
- **GPU submissions disabled** (technically submit ho jayega par sirf 1 minute runtime milega — yaani useless)
- **Internet disabled**
- Free + publicly available external data allowed, **including pre-trained models**
- Output `submission.csv`

Yaani: ~600 files × 1 min audio = 10 hours of audio, 7200 windows, 234 classes, **90 CPU minutes me**. Model loading + audio decode me hi ~5 min jaayenge (host khud bolte hain). Yahi comp ka defining constraint hai — accuracy nahi, accuracy-per-CPU-second.

---

## 2. Data

```
Total: 16.14 GB, 46,213 files
```

| Path | Kya hai |
|---|---|
| `train_audio/` | Short focal recordings — individual animal sounds. xeno-canto + iNaturalist se. 32 kHz, `.ogg`. Filename = `[collection][file_id].ogg` |
| `train_soundscapes/` | Test jaise hi locations se continuous recordings. **Kuch expert-annotated hain.** |
| `test_soundscapes/` | Submit karne pe ~600 recordings populate hote hain. 1 min each, 32 kHz ogg. Filename: `BC2026_Test_<id>_<site>_<date>_<UTC time>.ogg` |
| `train.csv` | training metadata |
| `train_soundscapes_labels.csv` | `filename`, `start`, `end`, `primary_label` (semicolon-separated species present in that 5s segment) |
| `taxonomy.csv` | 234 rows = 234 submission columns. iNat taxon ID + class (Aves/Amphibia/Mammalia/Insecta/Reptilia) |
| `sample_submission.csv` | 259 columns |
| `recording_location.txt` | Pantanal location info |

### `train.csv` ke relevant columns

| Column | Kya hai |
|---|---|
| `primary_label` | species code. Birds ke liye eBird code, non-birds ke liye iNaturalist taxon ID |
| `secondary_labels` | recordist ne jo aur species mark ki. **Incomplete ho sakti hai** |
| `latitude`, `longitude` | kahan record hua. Kuch birds ke local call "dialects" hote hain |
| `author` | uploader |
| `filename` | audio file |
| `rating` | 1–5 (xeno-canto). Background species hone pe 0.5 ghat jaati hai. 0 = no rating. iNat me rating nahi hoti |
| `collection` | `XC` ya `iNat` |

### Do cheezein jo directly strategy badalti hain

**1. Domain gap.** `train_audio` = focal recordings (ek animal, paas se, clean). `test_soundscapes` = passive acoustic monitoring (dur se, overlapping calls, rain, wind, insects, silence). Ye sabse bada gap hai aur BirdCLEF ke har edition me yahi decide karta hai.

**2. Label gap.** Host khud bolte hain:
> "Some species with occurrences in the hidden test data might only have train samples in the labeled portion of train_soundscapes and not in the train_audio."

Yaani `train_soundscapes_labels.csv` **optional nahi hai**. Kuch species ka wahi akela source hai. Aur ulta bhi: `train_soundscapes` ki saari species test me nahi hain.

**3. Insect sonotypes.** Zyadatar insects species level pe identified nahi hain — sonotypes ki tarah aate hain (`47158son16` = insect sonotype 16). Ye bhi classes hain aur kuch test me hain. Inko normal species ki tarah treat karo.

---

## 2b. Metadata EDA — maine chalake nikala

Ye numbers data page se nahi, actual files se hain.

```
taxonomy.csv                  234 rows  × 5 cols
train.csv                  35,549 rows  × 15 cols
train_soundscapes_labels.csv 1,478 rows × 4 cols  (66 files)
```

### Taxonomy: 234 classes, par balance bilkul nahi

| `class_name` | classes | focal clips in `train.csv` |
|---|---|---|
| Aves | 162 | **34,799** |
| Amphibia | 35 | 451 |
| Insecta | 28 | 199 |
| Mammalia | 8 | 99 |
| Reptilia | 1 | **1** |

Metric **macro** AUC hai — har class ka barabar weight. Par clips ka **97.9%** birds hain. Reptilia (Southern Spectacled Caiman) ke paas poore dataset me **ek** focal clip hai, aur wo 1/234 metric ka utna hi hissa hai jitna 499 clips wali koi bird.

### ⚠️ Sabse bada finding: 28 classes ka focal audio hai hi nahi

```
taxonomy classes:            234
train.csv me unique labels:  206
→ 28 classes: ZERO focal audio
```

Wo 28:

- **25 insect sonotypes** — `47158son01` se `47158son25` tak, poora set
- **3 amphibians** — Guaraní leaf-litter frog, *Chiasmocleis mehelyi*, Southern Orange-legged Leaf Frog

Achhi khabar: **saare 28 `train_soundscapes_labels.csv` me covered hain.** Zero classes aise nahi hain jinka koi training signal na ho.

Buri khabar: macro AUC me ye **28/234 = 12% of the metric** hai, aur inka poora training signal sirf **1,478 labelled soundscape segments** hai.

Iska seedha matlab: `train_soundscapes_labels.csv` optional side-data nahi hai. Agar tumne sirf `train_audio` pe train kiya, to tumhara model 28 classes ke liye random guess karega aur tum **12% metric upfront chhod rahe ho**. Ye akela hi 0.95 aur 0.96 ka fark hai.

### `train_soundscapes_labels.csv` ki reality

```
66 labelled files, 1,478 segments, 75 distinct classes (234 me se)
segments per class:  median 26, min 2, max 666
11 classes ke paas <5 labelled segments
```

Sites bhi skewed hain — `S22` akela 954 segments (65%), baaki 8 sites me 38–120 each:

```
S22 954 | S08 120 | S15 96 | S19 72 | S23 72 | S13 48 | S03 48 | S09 38 | S18 30
```

Yaani in-domain data hai hi bahut kam, aur ek site pe concentrated hai. Isliye **unlabeled `train_soundscapes` pe pseudo-labelling optional nahi, zaroori hai** — labelled portion akele se train nahi hoga.

### `train.csv` ke aur numbers

- `collection`: XC 23,043 · iNat 12,506
- clips per class (206 labelled classes me): median 125, min **1**, max 499. **25 classes ke paas <10 clips, 14 ke paas <5**
- `secondary_labels`: sirf **12.3%** clips me koi secondary label hai, average 0.21 per clip. Host khud bolte hain ye incomplete hai — positives ki tarah use karo par soft weight ke saath
- `rating == 0` (no rating): **36.1%** — ye zyadatar iNat clips hain (iNat rating deta hi nahi). Low-rated clips filter karte waqt dhyan rakhna, warna tum sara iNat data phenk doge, aur non-bird taxa zyadatar iNat se hi aate hain
- `latitude`/`longitude`: **0% missing** — geographic filtering/weighting ke liye poora usable
- 4,017 unique authors → `author` pe group CV karna practical hai

---

## 3. Data download

Rules accept ho chuke hain, API unblocked hai. Repo root me `fetch_data.sh` hai:

```bash
./fetch_data.sh          # metadata pehle, phir 16 GB bulk archive
```

`curl -C -` use karta hai — resumable, aur poori file RAM me buffer nahi karta (kaggle CLI karta hai, aur 3 GB pe laptop restart hone par poori download gayi thi).

### EDA jo abhi baaki hai (audio aane ke baad)

Metadata wali EDA Section 2b me ho chuki. Audio milne pe ye:

1. **Clip duration distribution** in `train_audio` — 5s window sampling strategy isi pe depend karti hai. iNat clips XC se chhoti hoti hain.
2. **Segments per soundscape jisme koi label nahi** — background/silence ka ratio. Ye "nothing" ka prior hai aur negative sampling ke liye chahiye.
3. **Insect sonotypes ka spectral character** — 25 sonotypes ke labelled segments sunke/dekhke confirm karo ki ye narrowband continuous stridulation hain. Agar haan, to inke liye alag (sasta) detector banana banta hai — bird CNN inke liye overkill hai aur CPU budget khata hai.
4. **`train_soundscapes` me total unlabeled duration** — pseudo-labelling budget kitna bada hai.
5. **Test-site (`S05` etc.) overlap** `train_soundscapes` sites ke saath. Labelled sites S03, S08, S09, S13, S15, S18, S19, S22, S23 hain.

---

## 4. Approach ladder

### Rung 0 — Constant submission

Sab cells me 0.5. AUC = 0.5. Format verify karo, 90-min budget me notebook ka skeleton (audio load → dummy predict → write csv) chala ke dekho. Audio decode ka cost measure karo — yahi tumhara budget ka bada hissa hai.

### Rung 1 — Pretrained embeddings + light head ← **yahan se shuru karo**

CPU-only constraint ka seedha jawab: koi bhi bhaari model train mat karo, pehle se maujood bird-audio foundation model ko frozen feature extractor ki tarah use karo.

- **Perch** (Google, bird vocalization classifier) ya **BirdNET** — dono publicly available hain aur Kaggle Datasets me already uploaded milenge
- Embeddings nikalo (ya inke logits directly), upar ek linear / shallow MLP / LightGBM head train karo 234 classes ke liye
- Head training tumhare paas kahin bhi ho sakti hai; inference me sirf frozen encoder + tiny head chalega

Ye single sabse achha effort-to-score ratio hai. BirdCLEF 2023–2025 me ye approach akele hi silver zone tak pahuncha deti thi.

### Rung 2 — Apna mel-spectrogram CNN, CPU ke liye exported

Standard recipe:

- 5s windows → mel-spectrogram (128 mels, 32 kHz, hop ~320)
- Backbone: `efficientnet_b0` / `regnety_002` / `mobilenetv3` — chhota rakho, CPU pe chalna hai
- Loss: BCE with `secondary_labels` bhi positive maan ke (soft labels)
- Mixup / cutmix on spectrograms, background noise augmentation (soundscape se noise-only segments uthao)
- Train GPU pe (Colab/Kaggle GPU), phir **ONNX + OpenVINO** ya **quantized int8** me export karke CPU inference

CPU inference ka budget: 7200 windows. `efficientnet_b0` at 224×224 ≈ 10-20 ms/window single-threaded → 2 min. Yaani ensemble ki gunjaish hai, par unlimited nahi. Batch + multi-thread karo.

### Rung 2b — Non-bird classes ko alag treat karo

Section 2b ka finding: 72 classes (Amphibia + Insecta + Mammalia + Reptilia) ke paas milakar 750 focal clips hain, aur 28 ke paas **zero**. Par macro AUC me ye 72/234 = **31% of the metric** hain.

Bird-focused pretrained models (Perch, BirdNET) in classes pe kuch nahi jaante. Do raaste:

- **Frogs/insects ke liye alag head** un labelled soundscape segments pe, embeddings share karke. Sasta hai — encoder ek hi baar chalta hai.
- **Insect sonotypes ke liye rule-based/cheap detector.** Sonotypes zyadatar narrowband continuous stridulation hote hain — spectral peak tracking se pakde jaate hain, CNN ki zaroorat nahi. 25 classes = 10.7% metric, aur CPU cost lagbhag zero.

Ye wo hissa hai jahan bheed nahi hai, kyunki zyadatar log BirdCLEF ko bird problem samajh ke chalte hain.

### Rung 3 — Domain adaptation (yahan medal banta hai)

Focal → soundscape gap band karna. Priority order:

1. **`train_soundscapes_labels.csv` pe fine-tune.** Ye in-domain labelled data hai. Chhota hai par exactly test distribution ka hai. Isko sabse zyada weight do.
2. **Pseudo-labelling unlabeled `train_soundscapes`.** Model se predict karo, confident predictions ko label maano, retrain. 2-3 rounds. Ye BirdCLEF ka classic winning move hai.
3. **Background mixing.** `train_soundscapes` se noise-only segments uthao aur `train_audio` clips pe overlay karo. Focal clips ko soundscape jaisa banao.
4. **Random 5s crop** focal clips se, sirf pehle 5s nahi — animal poore clip me kahin bhi bol sakta hai.

### Rung 4 — Post-processing (mufat me score)

AUC rank-based hai, aur predictions temporally correlated hain (agar 0:15–0:20 me jaguar hai to 0:20–0:25 me bhi hone ke chances zyada hain).

- **Temporal smoothing:** har file ke andar 12 windows pe smooth karo. `p_smooth = a*p_prev + b*p_curr + a*p_next`, ya file-level max ko blend karo: `0.7*p_window + 0.3*p_file_max`. Ye BirdCLEF me hamesha kaam karta hai.
- **Per-site priors:** filename me site code hai (`S05`). `train_soundscapes` se per-site species distribution nikalo. Jo species us site pe kabhi nahi aayi, uska score daba do.
- **Time-of-day:** filename me UTC time hai. Nocturnal vs diurnal species ka strong prior hai — owls 01:00 pe, zyadatar songbirds dawn pe. Amphibians/insects raat me.
- **Per-class rank normalization:** kyunki metric per-class AUC hai, har class ke predictions ko independently rank me convert kar sakte ho bina kuch khoye.

Ye sab **free** hai — zero extra inference cost. Pehle kar lo, model improve karne se pehle.

### Rung 5 — Distillation / efficiency

CPU budget tight hai, isliye speed hi capacity hai. Agar tum ek bade model ko chhote me distill kar sakte ho, to tum zyada ensemble members chala sakte ho.

Padho: *"Distilling Spectrograms into Tokens: Fast and Lightweight Bioacoustic Classification for BirdCLEF+ 2025"* ([arxiv 2507.08236](https://arxiv.org/pdf/2507.08236)).

### Rung 6 — Ensemble

Alag backbones, alag mel configs, alag seeds. Rank-average karo (AUC metric hai, probability-average ki zaroorat nahi). Budget: 90 min me kitne models fit hote hain, wahi limit hai.

---

## 5. Validation

- **Split by recording, never by window.** Ek recording ke windows train aur val dono me mat jaane do.
- **Group by `author`** bhi consider karo — ek recordist ke clips correlated hote hain (same equipment, same location).
- **Sabse achha val set: labelled `train_soundscapes`.** Yahi test distribution hai. Focal recordings pe CV LB se correlate nahi karega — ye BirdCLEF ki har saal ki kahani hai.
- **Metric exactly replicate karo:** macro AUC, zero-positive classes skip. Sklearn ka default macro average un classes pe crash karega ya galat dega.
- Private LB **66%** test data pe hai, public 34% pe. ~600 files me 34% ≈ 200 files — public LB noisy hai.

---

## 6. Practical notes

- Tumhare machine pe **GPU nahi hai** (`nvidia-smi` missing, torch cu130 installed par driver nahi). Backbone training ke liye Kaggle ke free GPU hours ya Colab use karo. Achhi baat ye hai ki **submission bhi CPU-only hi hai**, toh local dev environment inference ke liye representative hai.
- Audio decoding CPU pe mehnga hai. `soundfile`/`libsndfile` se ogg padho, `librosa.load` se nahi (wo resampling karta hai jo yahan chahiye nahi — sab pehle se 32 kHz hai).
- 16 GB download ek baar hi karna hai. `train_audio` sabse bada hissa hai; agar disk tight ho to pehle sirf metadata + `train_soundscapes` lo.
- Late submission me LB rank nahi milta, par **score milta hai** — portfolio ke liye "private LB equivalent 0.95x" claim karna valid hai, aur winner 0.9657 ke against benchmark hai.
- Working notes award (CLEF conference) deadline nikal chuki hai, par writeup portfolio ke liye likhna worth hai.

---

## Sources

- [Competition overview](https://www.kaggle.com/competitions/birdclef-2026/overview)
- [Data description](https://www.kaggle.com/competitions/birdclef-2026/data)
- [Distilling Spectrograms into Tokens (BirdCLEF+ 2025)](https://arxiv.org/pdf/2507.08236)
- [13th place 2026 writeup](https://zenodo.org/records/21545329)
