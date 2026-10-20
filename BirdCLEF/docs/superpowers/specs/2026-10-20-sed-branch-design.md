# BirdCLEF: spectrogram SED branch, Perch ke saath blend (design, 2026-10-20)

## Intent, aur kyun

Perch kernel (`jagjeetsingh23/birdclef26-perch-head` v2) **0.92681 private / 0.92423 public** pe hai.
Top ~0.966 hai, toh gap ~0.04 ka hai. CLEF 2026 working notes ke mutabik 0.95 se upar wali har team
ke paas ek **trained spectrogram CNN/SED branch** thi, jo Perch ke saath blend hoti thi (RESULTS §15):

- DS@GT: Perch probe 0.920 + HGNetV2-B0 SED 0.917, blend se 0.929
- 13th place: chhe CNN 0.953 + Perch, blend se 0.955

13th place ke mutabik asli lever **pseudo-labels ki quality** hai, backbone nahi:
- CNN sirf focal audio pe: 0.881
- weak self-teacher ke pseudo-labels ke saath: 0.864
- strong public pseudo-labels ke saath: 0.934
- chaar alag backbones sab 0.934–0.938 ke beech rahe

User ka plan, unke shabdon mein: **"2 ke saath chal kar ek baseline banate, and then 1 karenge aur
dekhenge ki pipeline ke kahan defect hai aur improvement possible h"**. Is project ke do outputs hain:

1. Ek SED branch jo Perch ke saath blend hokar LB badhaye.
2. Ek **controlled experiment**. Training wahi rahegi, sirf pseudo-labels badlenge (public teacher vs
   hamara Perch kernel). Dono ka farak hamare teacher ki kami naapega, aur diagnosis batayega ki woh
   kami kahan hai.

User ki general value (CASMI design se): skills aur samajh, sirf score nahi.

## Goals

1. **Pilot (viability check):** 1 fold, kam epochs, Phase A labels ke saath. Isme naapna hai:
   GPU-time per epoch, Kaggle CPU pe SED ki inference speed (Perch ke saath ek hi kernel mein), aur LB.
2. **Go/no-go:** pilot ka blend Perch akele se **≥ +0.005 private** upar ho (yaani ≥ 0.9318), tabhi
   poora GPU budget lagega. Threshold Claude ka call hai; user ne haan kaha.
3. **Phase A (baseline):** SED ko focal audio + labelled soundscapes + **public pseudo-labels**
   (`aliozanmemetoglu/raw-pseudos`) pe train karna. Folds aur epochs pilot ke numbers se tay honge.
4. **Phase B:** wahi training, sirf pseudo-labels **hamare Perch kernel** ke.
5. **Diagnosis:** A vs B ka LB farak, aur dono label sets ka seedha comparison (neeche "Phase B").
6. RESULTS.md mein naya section: pilot, A, B, diagnosis, aur aage ki improvement hypothesis.

## Non-goals

- Bahut saare backbones ka ensemble. Pehle ek branch kaam kare.
- ProtoSSM jaisa temporal head (RESULTS §15 item 3, chhota lever).
- BirdCLEF 2025 ke in-domain pretrained checkpoints (13th place: +0.007). Ye baad ka lever hai.
- Kisi bhi keemat pe 0.95 tak pahunchna.

## Anti-goals (technically kaam kare tab bhi failure)

- **A aur B ke beech ek se zyada cheez badalna.** Labels ke alawa sab identical rahe: folds, seeds,
  epochs, filtering rule aur augmentation. Warna farak kis wajah se aaya, ye pata nahi chalega.
- **Pseudo-label leakage ko validation inflate karne dena.** Labelled 66 files ke pseudo rows training se bahar rahenge.
- **0.92 ke upar local CV pe knobs tune karna.** 13th place mein offline-vs-LB correlation 0.92 ke upar
  sirf 0.09 tha. Faisle LB pe honge, aur local CV sirf sanity check ke liye hai.
- **Submission kernel ko fragile ya 90 min ke bahut paas le jaana.**
- Pilot fail hone pe bina wajah samjhe GPU jalate rehna.

## Kaam kaise hoga (user ne decide kiya)

- **Training Kaggle GPU pe** hogi. Weekly quota (~30 h) is kaam ko mil sakta hai, **lekin pehle limited
  iterations**, taaki viability pata chale.
- **Backbone:** HGNetV2-B0, SED head ke saath (Claude ki salah, user ne poocha tha aur maana).
- **Labels:** pehle public pseudo-labels se baseline (A), phir apne Perch kernel se (B).
- Kaam alag worktree (`wt-birdclef-sed`, branch `birdclef-sed-branch`) mein hoga. Commit/push sirf
  user ke kehne pe, aur `main` mein merge bhi user ke kehne pe.

## Constraints

- **Submission kernel:** CPU, ≤ 90 min, no internet.
  - Perch branch abhi ~35 min leta hai: ~8 min setup + 202 ms/window × ~7,200 windows ≈ 24 min.
  - Setup mein ~5 min ka unexplained startup hai. Shayad ye `/kaggle/input` ka `os.walk` hai, jo 46k files
    traverse karta hai.
  - SED inference ke liye realistically ≤ ~40 min bachte hain, margin ke saath.
- **Kaggle image Python 3.13 hai.** Saare wheels cp313 ke chahiye. `onnxruntime` 1.24.4 cp313
  `jagjeetsingh23/birdclef-perch-assets` mein pehle se hai. Kernel metadata mein `docker_image`
  pin karne ka option bhi hai.
- **Training kernels internet use kar sakte hain** (timm weights download ke liye). Sirf submission kernel offline hai.
- **Licenses:**
  - competition data CC BY-NC-SA 4.0 hai, toh derived caches private datasets mein rahenge
  - pseudo-label dataset ka license "unknown" hai, toh sirf private experiment mein use hoga aur RESULTS mein note hoga
- **Mac mini:** 16 GB RAM. Heavy jobs `guard.sh` ke andar chalenge.
- **SSD exFAT hai.**
  - Kaggle upload ke liye staging chhote path pe ho (CLI ka state-file naam 255 chars se lamba ho jaata hai).
  - `cp -X` use karo, aur `._*` files upload se pehle hata do.
- **LB submissions:** roz ~99 late-submission slots hain. Memory rule: har submission se pehle user se confirm.

## Data (recon, 2026-10-20)

- **Public pseudo-labels** `pseudo_labels_v2s_oof_raw.csv` (362 MB). SSD pe
  `/Volumes/Extreme SSD/kaggle/BirdCLEF/external/raw-pseudos/` mein hai.
  - 127,896 rows: saari 10,658 soundscapes × 12 windows. `row_id` = `<file>_<end_sec>`.
  - `fold_id`: −1 = unlabelled (127,104 rows), 0–4 = labelled 66 files ke 792 windows (OOF).
  - 234 class columns (taxonomy ke saare), **soft probabilities** (3e-7 se 0.99996), aur extra
    `primary_label` / `primary_label_prob` (argmax).
  - Unlabelled windows: mean 2.39 classes > 0.5 per window, aur 83% windows mein kam se kam ek > 0.5.
    (Labelled data mein asli average 4.22 species per window hai.)
  - Naam se teacher EfficientNetV2-S lagta hai. 13th place ne ise ~0.95-level teacher kaha, aur
    ~92,700 windows use kiye (filter ka rule unke paper mein saaf nahi hai).
- **Focal:** `train_audio`, 35,549 clips, 333 h. Median 20.9 s; 12% clips 60 s se lambe.
- **Labelled soundscapes:** 739 segments (66 files), asli labels.

## Design

### 1. Training data aur cache

- **Inputs:**
  - focal clips: random 5 s crops, primary + secondary labels
  - labelled soundscape segments: asli labels
  - unlabelled soundscape windows: soft pseudo targets
- **Mixing:** sab ek saath concat karke uniform sampling. 13th place ke mutabik strong labels ke saath ye
  ratio-sampling se behtar tha (0.934 vs 0.894).
- **Crop:** focal pe random crop. 13th place: random crop energy-aware aur start-crop dono se behtar tha.
- **Leakage:** labelled 66 files ke pseudo rows (fold 0–4) training se bahar. Un files pe sirf asli labels.
- **Pseudo-window filtering:** ek hi rule, jo A aur B dono teachers pe same tareeke se lage. Rule builder
  tay karega (ya bilkul filter nahi). Rule ka sabse zaroori gun ye hai ki woh dono phases mein identical ho.
- **Cache:** ogg decode training ke dauraan CPU bottleneck hai, isliye har public solution ne WAV ya
  waveform cache banaya tha.
  - Cache hum **khud banayenge**, Mac pe SSD se, aur private Kaggle dataset ke roop mein upload karenge
    (ya Kaggle CPU kernel mein banayenge). Public WAV/cache datasets use nahi honge.
  - Format, sizing aur subset builder ka call hai.
  - Pilot ke liye subset theek hai (capped focal, pseudo windows ka hissa), taaki upload aur disk chhote rahein.

### 2. Model aur training

- **Backbone:** `hgnetv2_b0.ssld_stage2_ft_in1k` (timm), ImageNet init, `in_chans=1`.
- **Head:** SED (attention / frame-level). Exact head builder ka call hai; ttahara ka LSE head bhi ek reference hai.
- **Front-end:** dB-scale log-mel. DS@GT ke mutabik ye sabse bada front-end lever tha (+0.032). Shuruaati
  params ttahara/13th place CNN wale: sr 32k, n_fft 2048, win 626, hop 313, 256 mels, fmin 20, power 2,
  top_db 80.
- **Recipe shuru karne ke liye** (ttahara + 13th place):
  - AdamW lr 5e-4, wd 1e-4, one-cycle with warmup
  - mel mixup, AMP, EMA
  - BCE (ya focal BCE), batch 64, 20–30 epochs
- **Reference cost:** ttahara ka HGNetV2-B0 T4 pe ~30 min/fold leta hai, sirf focal + 66 files pe.

### 3. Export aur submission kernel

- **Export:** ONNX, CPU pe `onnxruntime`.
- **Mel:** ya toh graph ke andar (STFT export) ya kernel mein numpy/torch se. Jo tez aur numerically
  sahi ho, wahi chalega (builder ka call).
- **Kernel:** Perch kernel ka naya version, ya naya kernel. Dono branches ek hi run mein, ≤ 90 min.
  Kernel log mein SED ka ms/window aur kul time projection likha jaayega, jaise Perch ke liye hota hai.
- **Blend:** per-class tie-aware rank blend, SED weight `w`, shuru 0.5 se. Phir kernel v4 wala smoothing
  blend ke baad. Weights LB pe hi tay honge, aur pilot mein sirf ek configuration hoga.
- **Classes:** 2 cold-start classes aur 25 insect sonotypes ko SED mein signal sirf labelled soundscapes
  aur pseudo-labels se milega. 13th place ke mutabik zero-focal taxa ke liye strong pseudo-labels hi
  akela signal tha.

### 4. Pilot aur go/no-go

- **Setup:** 1 fold, kam epochs. Builder epochs aise chune ki pilot ~2 GPU-hours ke andar rahe.
- **Naapna hai:**
  - GPU min/epoch aur peak GPU memory
  - Kaggle CPU pe SED ms/window, aur Perch + SED kernel ka kul time
  - LB: Perch + SED blend. SED akele ka LB bhi, diagnosis ke liye (ek extra slot, Claude ka call)
- **Faisla:**
  - blend private ≥ 0.9318 → Phase A full
  - kam → rukna, wajah samajhna, user ke saath faisla

### 5. Phase B: apne pseudo-labels, aur diagnosis

- **Teacher:** Perch kernel ka pipeline, saare 127,104 unlabelled windows pe. Mac pe ~45 ms/window,
  toh ~1.6 h.
- **Ek known problem:** Perch kernel ka output **per-class ranks** hai, probabilities nahi. Soft targets ke
  liye calibrated probabilities chahiye.
  - Chaar alag sources hain: LR head (`class_weight="balanced"`, isliye inflated), Perch zero-shot
    logits, cold-start heads, aur smoothing.
  - Inhe probability scale pe laane ka tareeka (jaise labelled OOF pe per-class calibration) Phase A ke
    baad, Phase B shuru karne se pehle tay hoga.
  - Ye bhi diagnosis ka hissa hai: public labels mein 2.39 classes > 0.5 per window hain. Hamare mein
    kitni hain, aur asli 4.22 se kitna door hain?
- **Training:** A wala training config bilkul same, sirf label file badlegi.
- **Diagnosis** (sirf LB number nahi):
  - per-class: dono label sets ki agreement. Kaun si classes pe hamara teacher alag hai, aur kahan
    confident par galat hai (labelled 66 files pe asli labels se check).
  - cohort: Perch route (head / zero-shot / cold-start), taxon, site. S01/S02 jaisi sites jinke paas
    ek bhi label nahi hai.
  - calibration: positives per window ka distribution.
  - Output: kam se kam ek concrete hypothesis ki hamare teacher mein kya sudharna hai.

## Claude ke calls (user badal sakta hai)

- HGNetV2-B0 + SED head, dB-mel 256 bins, ImageNet init.
- Go/no-go threshold +0.005 private.
- Pilot mein SED-alone LB bhi lena (ek extra slot).
- Cache khud banana, public WAV/cache datasets nahi.
- Blend 0.5 rank se shuru, smoothing blend ke baad.
- Pseudo-window filtering rule A aur B mein identical.

## Builder pe chhoda gaya

- Cache format (int16 npy / FLAC / WAV), subset size, aur cache kahan bane (Mac ya Kaggle CPU kernel).
- SED head ka exact form, loss (BCE / focal BCE), secondary labels ka weight.
- Pilot ke epochs, aur poore Phase A ke folds/epochs (pilot ke numbers se).
- Pseudo-window filtering threshold (ya koi nahi).
- Mel graph ke andar ya bahar, ONNX opset, batch size.
- Naya kernel banana ya Perch kernel ka version badhana.

## Success

- Pilot ke numbers aur go/no-go faisla, RESULTS mein likhe hue.
- Go hua toh:
  - Phase A ka LB (blend)
  - Phase B ka LB
  - likha hua diagnosis ki hamara teacher kahan aur kyun peeche hai
  - kam se kam ek improvement hypothesis
- Submission kernel 90 min mein aaram se, saare runs reproducible (seeds, dataset versions, kernel versions RESULTS mein).
