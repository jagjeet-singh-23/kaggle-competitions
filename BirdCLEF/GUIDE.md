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

Yaani: ~600 files × 1 min audio = 10 hours of audio, 7,200 windows, 234 classes, **90 CPU minutes me**.

### Budget: maine Kaggle pe chala ke naapa (Rung 0)

Pehle maine socha tha ki yahi comp ka defining constraint hai. **Galat tha.** `kernel/rung0.py` ne actual Kaggle CPU pe naapa:

```
decode :  94.4 ms/file      (mere laptop pe 28 ms — Kaggle CPU ~3.4x dheema)
mel    : 242.2 ms/file      (laptop 56 ms — ~4.3x dheema)

600 files ke liye projected: decode + mel = 3.4 min of 90
bacha: 86.6 min for 7,200 windows  ->  722 ms per window
```

**722 ms per window** bahut zyada hai. CPU pe `efficientnet_b0` ka ek forward pass ~30–60 ms leta hai. Yaani 10+ model ka ensemble aaram se fit ho jaayega.

Host bolte hain test soundscapes load hone me ~5 min lagte hain (mera 60-file sample `train_soundscapes` se tha aur shayad cached tha), toh realistic overhead 5–10 min maano. Phir bhi **~80 min inference ke liye bachte hain.**

Nateeja: budget ko constraint maan ke chhota model mat chuno. Accuracy pe focus karo, speed apne aap fit ho jaayegi.

### Submission notebook-only hai — LANL se bada fark

`kaggle competitions submit -f submission.csv` **kaam nahi karega**. Code competition hai: notebook Kaggle pe chalna chahiye aur wahin se submit hota hai.

Workflow:

```bash
cd kernel
kaggle kernels push -p .                                    # notebook chadhao + run karo
kaggle kernels status jagjeetsingh23/birdclef-2026-rung0    # wait
kaggle kernels output jagjeetsingh23/birdclef-2026-rung0 -p out   # log + submission.csv
```

Aakhri "Submit to competition" wala step **UI se hi hota hai** — notebook page pe Output tab → Submit.

#### ⚠️ Gotcha: mount path push method pe depend karta hai

```
UI me banaya notebook :  /kaggle/input/birdclef-2026
API se push kiya      :  /kaggle/input/competitions/birdclef-2026
```

Isi pe mera pehla run fail hua. Runtime pe resolve karo:

```python
COMP = next(p for p in ("/kaggle/input/birdclef-2026",
                        "/kaggle/input/competitions/birdclef-2026")
            if os.path.exists(f"{p}/taxonomy.csv"))
```

#### `sample_submission.csv` sirf 3 rows ka hai

Wo format example hai, poora grid nahi — ek hi test file ke pehle 3 windows. Apne row_ids test files se khud banao:

```
row_id = f"{filename_without_ogg}_{end_second}"     # end_second = 5, 10, ... 60
```

Rerun me `test_soundscapes/` populate hoti hai; uske bahar wo khaali hai (sirf `readme.txt`), isliye code ko dono case handle karne padenge.

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

Buri khabar: macro AUC me ye **28/234 = 12% of the metric** hai, aur inka poora training signal sirf **972 labelled 5s segments** hai — kul milakar 81 minute audio, aur ek class (`47158son05`) ke paas sirf **15 second**.

Iska seedha matlab: `train_soundscapes_labels.csv` optional side-data nahi hai. Agar tumne sirf `train_audio` pe train kiya, to tumhara model 28 classes ke liye random guess karega aur tum **12% metric upfront chhod rahe ho**. Ye akela hi 0.95 aur 0.96 ka fark hai.

### `train_soundscapes_labels.csv` ki reality

> ⚠️ **File me har row do baar hai.** 1,478 rows me se **739 exact duplicates** hain.
> Dedup zaroor karo, warna tumhara sample weighting aur CV dono galat honge.

Dedup ke baad:

```
739 segments, 66 files  =  1.0 ghanta labelled audio (poora)
75 distinct classes (234 me se)
segments per class: median 13, min 1, max 333
16 classes ke paas <5 segments, 25 ke paas <10
species per 5s segment: mean 4.22, median 4, max 10
```

**4.22 species per segment** — ye focal recordings se bilkul ulta hai. Test soundscapes dense multi-label hain; ek model jo "ek clip = ek species" maan ke train hua hai, wo yahan galat prior ke saath aayega.

### Wo 28 zero-focal classes — unka poora training data

Ye literally itna hi hai (deduped 5s segments):

```
47158son05    3      <- 15 seconds of audio, total
47158son12    5      47158son19    5      47158son09    6
47158son02    7      47158son15   12      25073        12
47158son16   12      47158son20   12      47158son18   12
47158son14   12      47158son04   17      47158son08   17
47158son06   18      47158son21   22      47158son01   23
47158son22   24      47158son23   24      47158son24   24
47158son03   33      47158son10   33      47158son13   36
47158son11   36      47158son17   43      47158son07   48
1491113      79      47158son25   84      517063      313
```

28 classes, **972 segments total**, aur ye 12% of the metric hain. `47158son05` ke paas 15 second hai. Ye supervised learning nahi, **few-shot** hai.

### 177 ghante unlabelled in-domain audio

Yahi sabse bada asset hai jo abhi chhua nahi gaya:

```
train_soundscapes: 10,658 files, har ek EXACTLY 60s, 32 kHz  = 178 hours
labelled:              66 files (0.6%)                       =   1 hour
UNLABELLED:        10,592 files                              = 177 hours
```

Format test ke bilkul same hai. **Pseudo-labelling optional nahi hai** — 1 ghante labelled audio se 234 classes nahi seekhenge, par 177 ghante in-domain audio available hai.

### Sites: 23 total, 14 me ek bhi label nahi

| site | files | labelled |
|---|---|---|
| S22 | 3,383 | 40 |
| **S02** | **2,505** | **0** |
| **S01** | **2,341** | **0** |
| S13 | 1,873 | 2 |
| S19 | 76 | 3 |
| S18 | 54 | 2 |
| S15 | 43 | 4 |
| baaki 16 sites | <55 each | 0–4 |

Char sites (S22, S02, S01, S13) me 10,102 files hain — poore dataset ka 95%. Aur **S01 + S02 me 4,846 files hain jinpe ek bhi label nahi**. Wahi pseudo-labelling ka sabse bada target hai.

Labelled sites: `S03 S08 S09 S13 S15 S18 S19 S22 S23`.

### Recording ka time: dawn chorus hai hi nahi

Pantanal UTC-4 hai. Local hour distribution:

```
14:00  1071 |  18:00  1049 |  22:00  1031
15:00  1024 |  19:00  1058 |  23:00   996
16:00  1009 |  20:00   780 |  00:00   266
17:00  1068 |  21:00  1117 |  02:00    95
                             baaki hours: <60 each
```

**Recordings lagbhag poori tarah 14:00–23:00 local hain.** 05:00–09:00 — yaani dawn chorus, jab birds sabse zyada bolte hain — dataset me practically hai hi nahi.

Data page ka apna test example isse match karta hai: `BC2026_Test_0001_S05_20250227_010002` = 01:00 UTC = **21:00 local**.

Iske do bade nateeje:

1. **Test afternoon/evening/night ka hai.** Nocturnal species, amphibians aur insects proportionally zyada important hain. Ye explain karta hai ki 28 insect/amphibian classes itni matter kyun karti hain.
2. **Time-of-day prior ka fayda kam hai** jitna maine pehle socha tha — sab recordings lagbhag ek hi time band se hain, toh usme discriminative power kam hai. Site prior zyada useful hai.

Date range: **2014-01-07 se 2025-11-29** — 11 saal.

### `train_audio` clip durations (3,000 clips ka sample)

```
sab 32 kHz (confirmed)
duration: median 20.9s  mean 33.7s  max 1654s (27 min!)
  10%   5.7s      75%   39.4s
  25%  10.9s      90%   68.9s
                  99%  187.1s
clips <5s: 8.3%   <10s: 22.9%   >60s: 12.1%
total train_audio: ~333 hours
```

8.3% clips 5 second se chhoti hain — un pe padding chahiye. 12% clips 60s+ hain, aur unme se ek random 5s crop lene pe zyadatar chance hai ki us crop me bird bol hi nahi raha. **Random crop naive tarike se mat karo** — energy-based crop selection ya multi-crop averaging chahiye.

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

### Rung 1a — Zero-training BirdNET ✅ done, macro AUC 0.6168

BirdNET V2.4 ko frozen feature extractor ki tarah laga diya. Model MIT licensed hai:

```
https://tuc.cloud/index.php/s/886x39f5N3sdsAM/download/V2.4.zip     # 214 MB
```

(Ye URL `birdnet_analyzer` wheel ke `model.py` me hai — repo me weights nahi hain.)

**Model ke facts jo maine verify kiye:**

```
input   : (batch, 144000) float32  = 3.0 s @ 48 kHz
output  : (batch, 6522)   species logits        tensor index 546
embedding: (batch, 1024)  GLOBAL_AVG_POOL/Mean  tensor index 545
```

Embedding lene ke liye `experimental_preserve_all_tensors=True` chahiye — TFLite intermediate tensors phenk deta hai. Cost ~25% (24 ms vs 18 ms per chunk).

Throughput (mera laptop, 4 threads): **24 ms per 3s chunk = 124x realtime.**
Yaani poora dataset (511 ghante audio) ~4 ghante me process ho jaayega. Kaggle CPU ~3.5x dheema hai, toh 600 test files ka inference ~20 min — 90 min budget me aaram se.

#### Species mapping: 157 of 162 birds free me mil jaate hain

`taxonomy.csv` ke `scientific_name` ko BirdNET ke labels se match kiya:

| taxon | classes | BirdNET me | |
|---|---|---|---|
| Aves | 162 | **157** | 97% |
| Amphibia | 35 | 0 | |
| Insecta | 28 | 0 | |
| Mammalia | 8 | 0 | |
| Reptilia | 1 | 0 | |

**157/234 = 67% classes ka prediction bina kuch train kiye mil jaata hai.** Baaki 77 sab non-bird hain — BirdNET ne inhe kabhi suna hi nahi.

#### Result (739 labelled segments, 75 scored classes)

```
zero-shot BirdNET, macro AUC:  0.6168

  mapped to BirdNET   (28 classes): 0.8128
  not in BirdNET      (47 classes): 0.5000   <- constant, by construction
```

**Jo birds BirdNET jaanta hai, un pe 0.8128 — bina ek bhi training step ke.**

Per-class dekho:

```
1.000  Great Kiskadee               n= 2
1.000  Hyacinth Macaw               n=10
0.994  Turquoise-fronted Amazon     n=14
...
0.545  Nacunda Nighthawk            n=11
0.515  Red Junglefowl               n=12
0.062  Buff-necked Ibis             n= 2    <- n=2, noise
```

> Ye 0.6168 leaderboard se **direct comparable nahi** hai. Locally sirf 75 classes score hoti hain aur unme se sirf 28 birds hain, jabki poore test me 162/234 birds hain. Agar LB pe bhi birds 0.81 aur non-birds 0.5 rahe, to zero-shot roughly **0.67 × 0.81 + 0.33 × 0.50 ≈ 0.71** hoga. Local number sirf apne models ko aapas me compare karne ke liye hai.

Context: winner 0.96574 pe tha. Toh 0.71 se 0.96 tak ka safar baaki hai — par 30 second ke kaam me half-decent baseline khada ho gaya.

#### Code

```
birdnet.py           model wrapper: 32k->48k resample, 3s chunking, logits+embeddings
embed_labelled.py    739 labelled segments -> data/labelled.npz   (50 seconds)
cv.py                exact metric (macro AUC, skip empty classes) + zero-shot baseline
```

Ek detail jo matter karti hai: 5s window 48 kHz pe 240,000 samples ka hai, aur BirdNET chunk 144,000 ka. Ek chunk se poori window cover nahi hoti. **Do overlapping chunks** (0–3s aur 2–5s) leta hoon, 1 second overlap ke saath — isse seam pe koi call kat na jaaye. Logits pe `max` (call dono me se kisi ek me bhi ho to present hai), embeddings pe `mean`.

### Rung 1b — Embeddings pe head ✅ 0.6168 → 0.8881

`head.py`. Sirf **739 labelled in-domain segments** pe, GroupKFold by soundscape file (ek recording ke segments background, mausam aur aksar wahi individuals share karte hain — file ke andar split karna leak hai). Per class one-vs-rest logistic regression, `C=0.01` — 1,024 dimensions vs ~590 training rows per fold.

`train_audio` ke embeddings abhi extract ho rahe hain, toh ye **imaandar floor** hai: sirf in-domain data se ek head kitna kar sakta hai.

```
model                                all   birds  others
zero-shot BirdNET                 0.6168  0.8128  0.5000
head on embeddings                0.8756  0.9098  0.8553
head on emb+logits                0.8774  0.9078  0.8593
head + zero-shot                  0.8100  0.9188  0.7451
head + zero-shot (birds only)     0.8881  0.9156  0.8717   <- best
```

#### Sabse important finding: embeddings non-birds ke liye bhi kaam karti hain

**0.5000 → 0.8553.** BirdNET ne kabhi ek bhi frog, cicada ya jaguar pe train nahi kiya — uska label set 6,522 **birds** ka hai. Phir bhi uski 1024-dim embedding pe ek linear head un 47 non-bird classes ko 0.86 tak pahuncha deta hai.

Matlab wo embedding bird-specific classifier nahi, ek **general bioacoustic representation** hai. Ye is competition ka poora non-bird hissa (31% of the metric) unlock kar deta hai, aur ye baat BirdNET ke docs me kahin nahi likhi.

#### Blend sirf birds pe lagao

Naive rank-blend (`head + zero-shot`) overall **bigaad** deta hai: 0.8774 → 0.8100. Wajah saaf hai — zero-shot non-birds pe constant 0.5 hai, toh use blend karna un 47 classes ko 0.8593 se 0.7451 pe girata hai.

Blend sirf un 157 classes pe lagao jo BirdNET jaanta hai:

```python
sel[:, hit] = 0.5 * sel[:, hit] + 0.5 * rank(Z)[:, hit]
```

Birds 0.9078 → 0.9156, others 0.8593 → 0.8717 (rank normalisation se), overall **0.8881**.

#### Jo abhi baaki hai

- Locally sirf **75 of 234** classes score hoti hain. Baaki 159 ke liye labelled soundscapes me ek bhi positive nahi hai — unke liye head train hi nahi ho sakta. Wo `train_audio` se aayengi (extraction chal rahi hai, ~4.7 ghante).
- 0.8881 LB se comparable nahi hai (75 classes vs 234, aur local mix me non-birds over-represented hain).
- Winner 0.96574 pe tha.

### Rung 1c — train_audio embeddings (running)

```
18 shards x 2,000 clips,  ~16.5 min per shard,  ETA ~4.7 h
shard 0: 7,828 windows  ->  poora ~139k windows
```

Per clip max 6 windows of 5s, evenly spread. Median clip 20.9s hai toh zyadatar poori cover ho rahi hai.

**Shortcut jo maine liya:** windows uniformly spaced hain, energy ke hisaab se nahi. 12% clips 60s+ ki hain aur unme bird kahin bhi ho sakta hai — loudest-window selection isse behtar karegi. Ye pehla obvious improvement hai.

Asli design sawaal jo Rung 1c me aayega: **333 ghante focal audio (saaf, galat domain) aur 1 ghanta soundscape (sahi domain, bahut kam) ko kaise combine karein.** Pehle sabse simple version — dono pe train, soundscape samples pe zyada weight.

### Rung 1 ka original plan (reference)

### Rung 1 ka original plan (reference)

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
