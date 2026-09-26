# Enveda CASMI 2026 — Molecule ID from Mass Spectra

**Status: LIVE.** Start 14 Sep 2026 · Entry deadline 7 Dec 2026 · Final submission 14 Dec 2026.
Prize $50k (1st $16k). 747 teams, 4459 entrants. Tu already entered hai.

---

## 1. Problem ek line me

Ek unknown molecule ke MS/MS spectra diye gaye hain. Uska 2D structure (SMILES) predict karo.
Per molecule **25 ranked candidates** de sakte ho, best-guess first, `;` se joined.

### Metric: MRR@25

```
MRR@25 = (1/U) * Σ 1/rank_u
```

`rank_u` = tumhari ranked list me pehla **correct** structure kis position pe hai.
Rank 1 → 1.0, rank 2 → 0.5, rank 25 → 0.04, miss → 0.

**Correct ka matlab:** dono SMILES (tumhara aur answer) RDKit tautomer canonicalization (pinned `2026.03.3`) se guzarte hain, phir InChIKey ka **pehla block** (InChIKey14) nikalta hai, phir compare. Matlab:

- Stereochemistry galat ho to koi penalty nahi.
- Tautomer form galat ho to koi penalty nahi.
- Sirf **atom connectivity** (2D skeleton) matter karta hai.

Ye bada relief hai — search space kaafi chhota ho jaata hai.

**Wrong guess ka koi penalty nahi** sivaay iske ki wo ek rank slot kha jaata hai. Toh hamesha poore 25 bharo.

### Submission format

```csv
molecule_id,smiles
m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```

Reject hoga agar: column missing, empty, nulls, duplicate `molecule_id`, ya 25 se zyada guesses.

### Code competition constraints

- CPU notebook ≤ 9 hours **ya** GPU notebook ≤ 9 hours
- **Internet disabled** — sab kuch Kaggle Datasets me pre-upload karna padega (PubChem subset, COCONUT, pretrained weights, sab)
- Free + publicly available external data + pretrained models allowed
- Output `submission.csv`

---

## 2. Data — jo maine verify kiya

```
train.parquet   3.03 GB   ~2.5M spectra,  ~275k unique structures,  18 cols
test.parquet    4.6 MB    1213 spectra,   400 molecules,            12 cols
sample_submission.csv     400 rows, har row me exactly 25 SMILES
```

### test.parquet — actual numbers (maine chala ke nikale)

| Cheez | Value |
|---|---|
| spectra | 1213 |
| molecules | 400 |
| spectra/molecule | median 3, min 1, max 9 |
| `instrument_type` | **timsTOF only** (100%) |
| `ionization_mode` | positive 987, negative 226 |
| `precursor_mz` | min 245, median 329, max 460 |
| peaks/spectrum | median 230, min 4, max 3259 (heavy tail) |
| `collision_energy_orig_units` | eV only |

Adduct distribution (visible test file):

```
[M+H]+          959
[M-H]-          193
[M+CH2O2-H]-     31
[M+Na]+          22
[M+NH4]+          4
[M+K]+            2
[M+Cl]-           2
```

Official page bolta hai hidden test me 10 adducts aayenge:
`[M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+, [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-`.
Yaani visible file me `[M-H2O+H]+`, `[M-2H2O+H]+`, `[M-H2O-H]-` nahi dikh rahe — code me handle karna.

> ⚠️ **Sabse bada trap:** jo `test.parquet` abhi dikh raha hai wo **training data ke examples** hai. Rerun pe hidden test se replace hoga. Iska matlab local "test" pe score dekhna pure leakage hai. CV khud banana padega (Section 5).

### Columns

**Dono files me shared (12):**

| Column | Kya hai |
|---|---|
| `molecule_id` | prediction is per molecule_id, per spectrum nahi |
| `spectrum_id` | unique per spectrum |
| `ms2_mzs` | fragment m/z ka array |
| `ms2_normalized_intensities` | aligned intensities, base peak = 1.0 |
| `base_peak_intensity` | normalization se pehle ki raw intensity → noise proxy |
| `adduct` | precursor adduct |
| `ionization_mode` | positive / negative |
| `instrument_type` | test me hamesha timsTOF |
| `precursor_mz` | measured precursor m/z |
| `collision_energy_ev` | **list** — `[20]` = single acquisition, `[20,40,60]` = merged |
| `collision_energy_orig` | jaisa source ne record kiya |
| `collision_energy_orig_units` | eV / NCE / V / unknown |

**Sirf train me (6 extra):**

> Correction: data page bolta hai train = "test columns above, plus 6". Actual file me train ke 18 columns me **`molecule_id` aur `spectrum_id` hain hi nahi**. Train me grouping ke liye `inchikey14` use karo — wahi molecule identity hai.

| Column | Kya hai |
|---|---|
| `normalized_smiles` | **label** — RDKit-standardised SMILES |
| `inchikey` / `inchikey14` | full key aur 2D skeleton key. **CV grouping isi pe karo** |
| `molecular_formula` | neutral structure ka formula |
| `ingest_lib` | source library |
| `adduct_orig` | standardisation se pehle ka adduct string |
| `precursor_error_ppm` | measured vs implied precursor ppm error. **Uncleaned — label-quality signal hai** |
| `num_peaks` | peak count |

### Train libraries — kaunsa kitna kaam ka

| `ingest_lib` | Spectra | Structures | Test se kitna relevant |
|---|---|---|---|
| `enveda-180` | 1,153,785 | 182,941 | **Same instrument** (Bruker timsTOF) par synthetic drug-like chemistry — wrong chemical space |
| `pluskal_ms2` | 527,581 | 46,821 | Orbitrap, NCE, consistent protocol |
| `riken` | 347,171 | 15,892 | **Plant specialised metabolites — chemistry match achha** |
| `gnps` | 220,849 | 45,750 | Sabse bada natural-product collection, sabse heterogeneous |
| `massbank` | 101,727 | 9,180 | Curated, multi-instrument |
| `mona` | 92,416 | 11,681 | Community-hosted |
| `spectraverse` | 50,933 | 9,631 | Harmonised aggregation, obscure libs |
| `msdial` | 40,765 | 9,127 | Multi-instrument |
| `drug_plus` | 2,545 | 2,539 | ~1 spectrum/drug, no CE metadata |
| `enveda-np-examples` | 1,151 | 250 | **Test ke sabse kareeb** — same instrument + same pipeline + natural products |
| `masaryk` | 652 | 416 | RECETOX standards |

Core tension: instrument-matched data (`enveda-180`) ki chemistry galat hai, aur chemistry-matched data (`riken`, `gnps`) ka instrument galat hai. Domain adaptation isi comp ka asli kaam hai.

### train.parquet — maine chalake nikala (2,539,608 spectra)

Ye numbers data page se nahi, actual file se hain.

**Unique `inchikey14`: 275,810.** (Page ka 275,810 exactly match karta hai.)
`enveda-np-examples` me actually **1,184** spectra hain, page pe 1,151 likha hai — minor mismatch, ignore kar sakte ho.

**Test ke 10 adducts ka train coverage:**

| Adduct | Train spectra |
|---|---|
| `[M+H]+` | 1,324,585 |
| `[M-H]-` | 465,688 |
| `[M+Na]+` | 178,092 |
| `[M+CH2O2-H]-` | 83,188 |
| `[M+NH4]+` | 53,452 |
| `[M+Cl]-` | 27,511 |
| `[M-H2O+H]+` | 21,682 |
| `[M+K]+` | 18,955 |
| `[M-2H2O+H]+` | 3,439 |
| `[M-H2O-H]-` | **285** ← practically unsupported |

Train ka 85.7% in 10 adducts me hai. Baaki 14.3% (`[2M+Na]+` 190k, `[2M+H]+` 92k, `[2M-H]-` 56k, `[M]+`, `[M+2H]2+`, …) test me kabhi nahi aayega — drop kar sakte ho, ya dimers ko monomer info ke liye use kar sakte ho.

**Test filter lagane pe kitna bachta hai:** `precursor_mz ∈ [245,460]` + test-adduct →
**1,630,439 spectra, 238,673 unique structures.** Yaani aggressive filtering ke baad bhi data kaafi hai. Train ka 64.6% test ke mass range me hai.

**Label quality:** `|precursor_error_ppm| > 10` → train ka **6.9%**. Median |ppm| sirf 1.5 hai, par max 4.7e8 hai — matlab clear garbage rows maujood hain. `precursor_mz` ka max **2,430,333** hai (!). Ye rows drop karo.

**Nulls:** `base_peak_intensity` **1,163,138 rows (45.8%)** me null hai (pre-normalised libraries). `precursor_error_ppm` 4,286 me null.

**Collision energy units:** eV 1,368,908 · NCE 789,066 · unknown 325,078 · V 56,556.
Yaani ~1/3 spectra ki CE ya to unknown hai ya approximate conversion se aayi hai.

**`instrument_type` free text hai** — aur case-inconsistent:
`timsTOF` 1,154,969 · `Orbitrap` 745,562 · `QTOF` 125,936 · `LC-ESI-QTOF` 104,482 · `ESI-QFT` 77,342 · … plus `qTof` 33,777 aur `qtof` 26,111 alag se. Normalize karna padega (lowercase + family mapping), warna instrument ko feature banane pe teen alag "qtof" classes ban jayengi.

### ⚠️ Sabse important finding: peak-count mismatch

Test spectra ka median **230 peaks** hai. Train me:

| Library | median `num_peaks` |
|---|---|
| `masaryk` | 9 |
| `riken` | **9** |
| `msdial` | 11 |
| `massbank` | 11 |
| `pluskal_ms2` | 21 |
| `gnps` | 46 |
| `spectraverse` | 47 |
| `mona` | 57 |
| `drug_plus` | 108 |
| `enveda-180` | 138 |
| `enveda-np-examples` | **141.5** |

Ye instrument gap se bhi bada problem hai. `riken` (plant metabolites — chemistry test se sabse milti-julti) ke spectra me **9 peaks** hain, test me **230**. Wo libraries already heavily thresholded/centroided hain.

Do implications:

1. **Naive cosine similarity fail karegi** riken/massbank ke against, kyunki test spectrum me 220 extra peaks hain jo reference me hai hi nahi. Similarity ko **reference peaks pe conditional** banao (kitne reference peaks test me mile), symmetric nahi.
2. **Test spectra ko train ke level tak thin karna padega** — ya to top-N peaks, ya intensity floor. Agar dono directions me alag preprocessing lagayi to spectra comparable nahi rahenge. Ye tumhara pehla real engineering decision hai.

Ye single observation zyadatar public baselines ko beat karne ka raasta hai, kyunki log aam taur pe same preprocessing dono taraf laga dete hain.

### Teen novelty classes (distribution hidden hai)

| Class | Definition | Attack |
|---|---|---|
| **1** | Public spectral libraries me spectra maujood | Spectral similarity search — train.parquet me hi mil jayega |
| **2** | Structure PubChem/COCONUT me hai, par spectra nahi | Database retrieval — formula se filter, fingerprint se rank |
| **3** | PubChem me bhi nahi — genuinely novel | De novo generation |

Ye teen alag-alag problems hain aur alag-alag machinery chahiye. **Ek pipeline sab solve nahi karega.** Final submission me teeno ke candidates merge karke rank karne padenge.

---

## 3. Pehla kaam: EDA

Download (train 3 GB, curl se stream karo — kaggle CLI poori file RAM me buffer karta hai):

```bash
cd MoleculeID-from-Mass-Spectra/data
U=$(python3 -c "import json;d=json.load(open('$HOME/.kaggle/kaggle.json'));print(d['username']+':'+d['key'])")
curl -L -C - -u "$U" -o train.parquet \
  "https://www.kaggle.com/api/v1/competitions/data/download/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
```

> Note: forum pe host (`inversion`) ne "Please re-download training data" post kiya tha. File creation date `2026-09-15` hai — wahi latest hai.

Pehla EDA pass jo actually decisions badalta hai:

1. **Test ke precursor_mz range (245–460) me train ka kitna hissa hai?** Agar train ka bada chunk is range se bahar hai to filter karo.
2. **Test ke 10 adducts me se har ek ka train coverage.** `[M-2H2O+H]+` shayad bahut rare ho.
3. **`precursor_error_ppm` distribution per library.** Jahan |ppm| > 10–20, wo labels shakki hain → drop.
4. **`collision_energy_ev` null rate per library.** Agar model CE consume karta hai to null-heavy libs ka kya karoge.
5. **`inchikey14` overlap across libs** — asli unique structure count kitna hai (275,810 se kam, kyunki recur karte hain).
6. **Formula distribution** — CHNOPS+halogens ke bahar kitna hai.
7. **`num_peaks` vs `base_peak_intensity`** — noise floor kahan set karein.

RDKit chahiye hoga, abhi installed nahi hai:

```bash
pip install rdkit matchms
```

---

## 3b. Jo maine build kiya (aur jo mila)

### ⚠️ Visible test ke jawab train me hi hain — saare 400

Maine test ke har spectrum ko train se match kiya `(round(precursor_mz,4), adduct, num_peaks)` pe:

```
1213 / 1213 spectra matched   (100%)
400 / 400 molecules recovered
381 unambiguous; baaki 19 peak-list compare karke resolve ho gaye
→ saare 400 ek single consistent structure pe
```

`data/visible_test_answers.csv` me likh diya hai.

Iske do matlab hain, aur dusra zyada important hai:

1. Pipeline end-to-end validate kar sakte ho — asli jawab available hain.
2. **Visible test poora Class 1 hai.** Har structure train me maujood hai. Toh koi bhi library search ispe ~1.0 MRR dega, aur wo number bilkul jhootha hai. Hidden test me Class 1/2/3 ka mix hoga.

> Ispe local score dekhna wahi galti hai jo LANL me public LB dekhna thi. Mat dekhna.

### Metric implement ho gaya — `metric.py`

Rules ke apne examples pe verify kiya: glucose ke dono spellings `WQZGKKKJIJFFOK` pe aate hain, rank-2 hit 0.5 deta hai, rank-26 hit 0 deta hai, invalid SMILES crash nahi karti, aur keto/enol tautomers collide karte hain.

RDKit **2026.03.3** pinned version install hai — wahi jo rules me likha hai. `key14()` 1.1 ms per SMILES leta hai (uncached), toh 400 × 25 = 10,000 candidates ~11 second me score ho jaate hain. `lru_cache` laga hai kyunki common scaffolds baar-baar aate hain.

### Spectral index — `library.py`

```
2,539,608 spectra  →  1,888,517 after filters  (265,875 structures)
disk: 1.3 GB, load: 1.8s / 1.84 GB
```

Filters: `precursor_mz ∈ [150,1250]`, test ke 10 adducts, `|precursor_error_ppm| ≤ 10`, ≥3 peaks.
Har spectrum se top-64 peaks, precursor+2 se upar wale hata ke, 0.5% intensity floor, `sqrt` intensity.

> ### Memory: is file ne meri machine do baar freeze ki
>
> `pd.read_parquet()` poori file pe, peak-list columns ke saath = **~18 GB peak**. Wajah: pandas har spectrum ke do variable-length arrays ko alag numpy objects banata hai — 5 million objects. Machine 15 GB + 2 GB swap ki hai, toh Linux clean OOM-kill nahi karta, swap thrash karke lock ho jaata hai.
>
> Maine chaar tarike naape:
>
> | approach | peak RSS | time |
> |---|---|---|
> | `pd.read_parquet` (poori file) | **~18 GB** | freeze |
> | polars, full `collect` | 9.84 GB | **7.5s** |
> | polars, batched `slice` | 11.21 GB | 211s |
> | pyarrow row-group loop | **4.9 GB** | ~6 min |
>
> Polars I/O pe 50x fast hai par memory usse theek nahi hoti — `slice()` lazy scan pe har baar shuru se re-scan karta hai. Do aur gotchas: `to_pylist()` 20M doubles ko Python float objects me box kar deta hai (4.3 → 1.5 GB fix, Arrow ke flat buffers use karo), aur `explode()` har peak ke liye **baaki saare columns duplicate** karta hai — 300-byte `smiles` × 158 peaks se memory turant khatam.
>
> Jo chala: row-group streaming + Arrow flat buffers + per-row-group shards + fixed-width byte strings.

### CV split — `split.py`

Teen novelty classes simulate karte hain, `inchikey14` pe group karke (spectrum pe kabhi nahi — ek hi molecule ke chaar collision energies me se teen index me aur ek query me daalna Class 1 ko Class 3 ka label pehna dena hai).

```
class 1  structure >=2 libraries me hai. Ek library ke spectra query, baaki index me.
class 2  saare spectra hold out, par structure candidate list me rehta hai.
class 3  saare spectra hold out AUR structure candidate list se bhi hata diya.
```

Result:

```
class 1: 200 molecules,  618 query spectra (3.1 per molecule)
class 2: 200 molecules,  696 query spectra (3.5 per molecule)
class 3: 200 molecules,  681 query spectra (3.4 per molecule)
index rows removed: 3,120 of 1,888,517
peak RSS 0.19 GB
```

3.1–3.5 spectra per molecule real test ke median 3 se match karta hai.

**Ek constraint jo class-1 proxy ko limit karti hai:** 265,875 structures me se sirf **24,893 (9.4%)** ek se zyada library me hain. Yaani class-1 simulate karne ke liye pool chhota hai, aur agar tum 200 se zyada chahiye to wahi 24,893 me se lene padenge.

Spectra per structure: median **4**, max **1067**.

## 4. Approach ladder — sabse sasta pehle

### Rung 0 — Sanity submission (1 ghanta)

Har molecule ko train ke 25 sabse common `inchikey14` structures de do. Score ~0 hoga, par pipeline + submission format verify ho jaayega. Ye skip mat karna — code comp me format bugs sabse zyada time khaate hain.

### Rung 1 — Spectral library search (Class 1 ka 90%)

Test spectrum ko train ke har spectrum se compare karo, best matching structure ka SMILES le lo.

- Similarity: **modified cosine** (precursor mass difference ko allow karta hai) ya plain cosine with m/z tolerance ~0.01 Da / 10 ppm
- Pehle candidates ko `precursor_mz` se filter karo (±10 ppm), tab hi similarity compute karo — warna 2.5M × 1213 comparisons lagenge
- Multiple spectra per molecule → per-molecule score aggregate karo (max ya sum of top-k)

`matchms` me ye sab ready-made hai. Ye akela hi decent MRR de dega agar test me class-1 molecules zyada hain.

### Rung 2 — Analog propagation (public baseline, LB ~0.33+)

Exact match nahi mila to **mass-shifted** match dekho: agar test aur reference spectrum ke fragments ek constant offset pe align ho rahe hain, to structures analogs hain. Reference structure ko us mass shift ke hisaab se modify karke candidate banao.

Public notebook: `prvsiyan/analog-propagation-casmi-2026-baseline`.
Forum pe published top solution: *"[0.339 Top 1 Solution] 4-Channel Mass-Shifted Analog Propagation & Neural Bayes Reranking"* (`haideptry`) — padh lo, ye current public LB top (~0.362) ke bahut kareeb hai.

Yahan pahunchne ke baad tum bheed ke saath ho, aage ka kaam differentiation hai.

### Rung 3 — Formula first (multiplier on everything)

Molecular formula pehle predict karo, phir uspe condition karo. Formula milte hi Class 2 ka search space PubChem ke ~100M se ghatke us formula ke ~10³–10⁵ structures reh jaata hai.

- `precursor_mz` + adduct → exact mass → candidate formulas (CHNOPS + halogens, valence/RDBE rules)
- Fragment m/z ko subformulas se explain karke formulas rank karo (SIRIUS ka core idea)
- Tools padho: **SIRIUS**, **MIST-CF** (dono hosts ne khud recommend kiye hain)
- Internet disabled hai → ya to formula scoring khud implement karo, ya offline weights Kaggle Dataset me daalo

### Rung 4 — Fingerprint prediction + database retrieval (Class 2)

Ye CSI:FingerID ka classic recipe hai, aur Class 2 ke liye single sabse bada lever:

1. Spectrum → molecular fingerprint predict karne wala model train karo (train.parquet me 2.5M labelled spectra hain — kaafi hai)
2. Offline candidate DB banao: PubChem ka formula-indexed subset + COCONUT (natural products)
3. Inference pe: formula se DB filter → har candidate ka true fingerprint compute → predicted fingerprint se Tanimoto/Platt score → rank

Isme kaam:
- Fingerprint choice: Morgan (ECFP4) counts, MACCS, ya CSI-style substructure set
- Encoder: spectrum ko set-of-peaks samjho → Transformer/Set-Transformer over (m/z, intensity, CE) tokens. Neighbouring-peak-differences (neutral losses) as extra channel bahut madad karti hai
- Ye GPU-heavy training hai, par **inference CPU pe chal sakta hai**. Training tum kahin bhi karo, weights Kaggle Dataset me upload.
- Forum me thread hai: *"External structure databases for Class 2 retrieval: is a formula-filtered PubChem subset prize-eligible"* — rules clarify karke hi mehnat lagao.

Dataset ready-made: `Samar Talwar` ne 139K harmonized MassBank spectra + fingerprints post kiye hain.

### Rung 5 — De novo generation (Class 3)

Sirf yahi Class 3 ke liye kaam karega. Sabse mehnga, sabse kam reliable.

- Literature: **MSNovelist**, **MassGenie**, **DiffMS**, **Spec2Mol**
- Practical shortcut: pure generation ki jagah **analog editing** — closest known structure lo aur mass-shift ke hisaab se chhote edits (methylation, hydroxylation, glycosylation) apply karo. Natural products me analogs isi tarah banate hain, aur InChIKey14 match hi chahiye toh exact stereochemistry ki tension nahi.
- Ye Rung 2 ka hi extension hai — isliye Rung 2 achhe se banao.

### Rung 6 — Merge + rerank

Asli score yahan banta hai. Teen sources (library hit, DB retrieval, generated analogs) se candidates aate hain, aur inke scores alag scales pe hain.

- Sabko normalize karke ek learned reranker (LightGBM ya chhota NN) me daalo
- Reranker features: spectral similarity, fingerprint Tanimoto, formula match confidence, candidate ka natural-product-likeness, source library, precursor ppm error, candidate DB me frequency
- **Deduplicate on InChIKey14** — metric wahi compare karta hai. Same skeleton ke 3 tautomers 3 slots kha jayenge, ye pure waste hai.
- Har molecule ke liye poore 25 slots bharo.

---

## 5. Validation — yahi comp jitayega ya haraayega

`test.parquet` me abhi training examples hain. Uspe score dekhna meaningless hai.

**Split `inchikey14` pe karo, spectrum pe kabhi nahi.** Warna same molecule ke doosre collision energy wale spectra leak ho jaayenge.

Teen novelty classes ko alag-alag simulate karo:

- **Class 1 proxy:** held-out molecules jinke spectra train me doosri library me bhi hain
- **Class 2 proxy:** molecules jinke **saare** spectra hold out kar do, par structure candidate DB me rakho
- **Class 3 proxy:** molecules jinka structure candidate DB se bhi nikaal do

Har class ka MRR@25 alag report karo. Class distribution hidden hai, isliye teeno pe robust hona chahiye.

**`enveda-np-examples` ke 250 structures ko akela CV mat banana** — forum thread (`Adarsh`) bolta hai ye LB ke saath correlate nahi karta. Test-distribution ka sanity check hai, validation set nahi.

Test spectra ki cleaning jo already ho chuki hai (train pe *nahi* hui) — same cleaning train pe apply karo:
- Jinka measured precursor mass structure+adduct se inconsistent tha, wo drop
- Base peak < 1000 raw counts → drop
- Precursor + 2 Da se upar ke peaks remove

Baaki common curation (hosts ne khud list kiye, koi mandatory nahi):
relative-intensity floor (0.1–2%), precursor se upar ke peaks drop, min ~5–6 peaks, top-N (128) peaks ya "top-6 per 50 Da window", deisotoping (13C ~1.0033 Da), sqrt/log intensity transform.

---

## 6. External resources (rules pehle check karo)

Hosts ne khud recommend kiye:

- **MIST-CF, SIRIUS** — formula annotation
- **matchms** — spectra cleaning + similarity
- **MassIVE** — billions of unannotated MS/MS, self-supervised pretraining ke liye
- **GeMS / DreaMS** — curated unannotated corpus + **pretrained spectrum-embedding models**. Repository-scale pretraining ka shortcut.
- **FragHub** — merged open MS/MS libraries, kuch extra spectra
- **COCONUT** — natural products DB, Class 2 retrieval ke liye essential
- GNPS "suspect" annotations train me **nahi** hain (labels inferred hain) — external data ke roop me use kar sakte ho, par labels noisy hain

Sab kuch Kaggle Dataset me pre-upload karna padega, internet disabled hai.

---

## 7. Plan — 3 mahine, priority order

| Week | Kaam |
|---|---|
| 1 | EDA + format-valid dummy submission + `inchikey14` CV harness (teeno class proxies) |
| 2 | Modified-cosine library search. Baseline MRR note karo. |
| 3 | Analog propagation (public notebook padhke re-implement, blindly fork mat karo) |
| 4–5 | Formula prediction. Class 2 search space collapse. |
| 6–8 | Fingerprint model train (GPU kahin aur), PubChem+COCONUT offline index build, retrieval |
| 9–10 | Reranker + merge, InChIKey14 dedup |
| 11 | Class 3 analog editing |
| 12 | Ensemble, 9h runtime budget verify, submission hardening |

Highest leverage: **formula prediction** (Rung 3) aur **fingerprint retrieval** (Rung 4). Public LB abhi wahan nahi pahuncha — top ~0.362 zyadatar analog propagation hai.

---

## 8. Gotchas

- Public LB sirf **33%** test data pe hai. Final standings baaki 67% pe. LB chase mat karna.
- `precursor_error_ppm` train me **uncleaned** hai — jaan-bujhke. Label quality filter ki tarah use karo.
- `instrument_type` train me free text hai (dozens of strings, nulls). Sirf `enveda-180` aur `enveda-np-examples` uniformly timsTOF hain.
- `base_peak_intensity` un libraries ke liye null hai jinhone pre-normalised intensities bheji.
- `collision_energy_ev` use karo, `collision_energy_orig` nahi — wahi test ke convention se aligned hai. NCE→eV conversion approximate hai (nominal Thermo formula).
- 9-hour runtime me 400 molecules × (2.5M spectra search + DB retrieval + generation) fit karna padega. Inference cost budget karo, nahi to submission timeout.
- Ek molecule ka jawab ek hai, par spectra multiple. Per-spectrum predict karke aggregate karo — per-molecule ek hi spectrum use karke mat chhodo.

---

## Sources

- [Competition overview](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)
- [Data description](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)
- [Analog Propagation baseline notebook](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline)
