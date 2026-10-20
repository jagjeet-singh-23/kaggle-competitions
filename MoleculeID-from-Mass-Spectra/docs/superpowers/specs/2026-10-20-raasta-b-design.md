# Raasta B: apna multi-evidence engine, seekhte hue (design, 2026-10-20)

## Intent, aur kyun

CASMI 2026 me hum v15 (LB 0.257) pe hain. Public notebooks 0.452 pe hain, aur top 0.495 pe.
Gap ki wajah 2026-10-20 ko public notebooks padh ke mili: unka pipeline kai evidence channels jodta hai
jo hamare paas nahi hain (mass-shifted analogs, learned ranker, gated PubChem channel, forward-model
isomer re-rank). Details memory note `public-pipeline-state` me hain.

User ne **Raasta B** chuna: ye components apne code me khud banana, unka code ya weights fork nahi karna.
Wajah user ke shabdon me: prize money se zyada **skills aur learning**. User jaanna chahta hai ki doosre
log in pipeline decisions tak kaise pahunche, unhe data se kya insights mili, aur hamari data understanding
unki se kahan alag thi. "Sirf unka code use karne se koi learning nahi milegi."

Isliye is project ka asli output **samajh** hai. LB score feedback hai, goal nahi.

## Goals

1. Chaar components apne code me, isi order me (order Claude ka call hai; user badal sakta hai):
   1. mass-shifted analog propagation
   2. learned ranker
   3. gated PubChem channel
   4. apna forward model (spectrum simulation), timsTOF data pe trained
2. Har component ek learning cycle se guzrega (neeche), aur uska asar baaki components se alag naapa jaayega.
3. `LEARNING.md` me har step likha jaayega, taaki 14 Dec ko ek jagah dikhe ki kya seekha.
4. 14 Dec tak hamara best own-code submission final selection me ho.

## Non-goals

- Prize eligibility ya compliance ke liye kaam karna. (Apna code hone se ye side effect me mil sakta hai.)
- Kisi bhi keemat pe 0.452 tak pahunchna.
- Official class 3 (PubChem me bhi nahi) ke liye de novo generation. Is project me nahi hai.

## Anti-goals (technically kaam kare tab bhi failure)

- Public notebooks ka code, trained weights ya prepared datasets apne pipeline me seedha utha lena.
  Unka code **padhna** theek hai, aur woh learning ka hissa hai. Banana apne style me, apni samajh se hai.
  Public databases (ChEBI, LIPID MAPS, PubChem) unke source se khud banana theek hai.
- Noise ke andar knobs tune karna (v10–v16 wala pattern).
- User ka andaza liye bina experiment chala dena.
- Aisa component ship karna jiska "kyun" `LEARNING.md` me likha na ho.

## Kaam kaise hoga (user ne decide kiya)

User ne do tareeke chune:

1. **Claude banata hai, user samajhta hai.** Code Claude likhta hai. Har decision ke saath Claude batata hai:
   - data me kya dikha,
   - public notebooks me kisne ye cheez kaise pakdi, aur unke ledger ka kaunsa step tha,
   - hamari pehli soch usse kahan alag thi.
2. **Pehle andaza, phir data.** Har experiment se pehle Claude ek seedha, number wala sawaal poochta hai.
   User andaza likhta hai, phir data se check hota hai. Galat andaza sabse kaam ki jagah hai, aur uski wajah
   dhoondhi jaati hai.

**Har component ka learning cycle** (cycle ka shape user ne confirm kiya):

1. **Insight dhoondhna:** public notebooks/ledgers se nikaalna ki component kyun bana.
2. **User ka andaza:** experiment se pehle.
3. **Apne data pe check:** unka number maan lene se pehle khud ginna.
4. **Banana:** apne code me.
5. **Measure:** pehle held-out pe, phir zaroorat ho to LB pe.
6. **Likhna:** `LEARNING.md` me.

`LEARNING.md` ka format Claude ka call hai. Har entry me ye hoga: unka insight (source ke saath), user ka
andaza, measured number, fark ki wajah, aur hamari pehle ki belief.

## Constraints

- **Machine:** Mac mini M4, 16 GB, CPU + Apple GPU (MPS). MPS is repo ke code pe ab tak test nahi hua.
  Power cuts se reboot hota hai, isliye jobs `data/jobs/` me rakhni hain, `caffeinate -ims` ke neeche chalani
  hain, aur resumable honi chahiye. `guard.sh` kabhi weaken nahi karna. Bade GPU training ke liye Kaggle ka
  free GPU (~30 ghante/hafta) backup hai.
- **Kaggle kernel:** internet off, 9 ghante. Abhi CPU pe ~66 min chalta hai. Forward model ke liye GPU
  kernel (T4) lag sakta hai.
- **Deadline:** entry 7 Dec, final submission 14 Dec 2026 (~8 hafte).
- **LB:** public LB lagbhag 132 molecules pe hai, to ek molecule rank 1 pe aane se ~0.0076 milta hai.
  Noise ±0.005–0.007. Submissions do tarah ki hongi:
  - **Ship submission:** final selection ka candidate. Iske liye prediction ≥ +0.005 chahiye (memory
    `eval-overstates-gains`).
  - **Probe submission (user ne 2026-10-20 ko learning ke liye allow kiya):** jab kisi component ka asar
    offline naapa hi na ja sake (jaise gated PubChem), ya kisi belief ko test karna ho (jaise coverage
    model). Score aane se pehle prediction `LEARNING.md` me likhni hai. Probe tab tak final selection me
    nahi jaayega jab tak woh best na nikle.
  - Dono tarah ki submissions me har push se pehle user se confirm karna hai.
- **Eval rules (HANDOFF):** har eval `C2_BANNED=1` ke saath aur uske bina. Class 3 ko non-`enveda-180`
  answers pe padhna. Covered splits ke class 3 deltas ko coverage se scale karna. RDKit 2026.3.3 pinned.
- **Git:** commit sirf user ke kehne pe, aur files path se stage karni hain. Models, logs aur competition
  data git me nahi jaate.

## Components

### 1. Mass-shifted analog propagation

**Insight (public):** jo molecule kisi library me nahi hai, uska koi rishtedaar (methyl, hydroxyl, glycoside
etc.) aksar library me hota hai, bas alag mass pe. Rishtedaar ke fragments query se do tarah match karte hain:
- seedhe, jab fragment me shared core hai aur modification nahi,
- Δmass se shifted, jab fragment me modification wala hissa hai.

prvsiyan ke ledger me library-only 0.151 tha. Entropy library + mass-shifted analogs + GBM ranker se 0.233
hua, aur uske turant baad 0.245.

**Hamara abhi ka haal:** `analog.py` ke seeds sirf ±10 ppm precursor window se aate hain, yaani sirf
isomers. Mass-shifted search kabhi bana hi nahi.

**Kya banana hai** (unka design, reference ke liye; hamare numbers builder tune karega):
- Analog index: har structure ka ek representative spectrum. timsTOF libraries (`enveda-180`,
  `enveda-np-examples`) ko pehle choose karna, phir sabse zyada peaks wala. Neutral mass se sorted, aur
  polarity ke hisaab se alag.
- Similarity: `max(direct, Δ neutral mass se shifted)`. Unke yahan entropy similarity hai. Hamara cosine
  comparison ke liye rakhna hai, kyunki entropy vs cosine khud ek seekhne wala sawaal hai.
- Window: query ke neutral mass ke ±200 Da. Top N analogs (unka N = 80).
- Candidate score: `max_a sim(a)^p × Tanimoto(c, a)` (unka p = 3).
- `dilute.py` / `fpsearch.py` me ek naya term. Ranker (component 2) aane tak iska weight haath se rakhna
  padega. Held-out pe chhota sweep chalega, sirf ye dekhne ke liye ki signal hai ya nahi, weight tune
  karne ke liye nahi.

**Leak check:** `split.py` pehle se held-out query ke spectra index se hata deta hai, aur class 3 ka answer
candidates se ban hota hai. Rishtedaar index me rehte hain, aur yahi asli test jaisa scenario hai. Builder ko
verify karna hai ki analog index bhi `drop` maanta hai, yaani query ka apna spectrum analog nahi ban sakta.

**Pehle andaze ke sawaal** (examples, final sawaal Claude experiment ke waqt poochega):
- Held-out class 3 (non-`enveda-180`) ke kitne answers ka koi analog ±200 Da me sim > X ke saath milta hai?
- Isse class 3 MRR kitna badhega?

**Measure:** `splits_big.npz` (covered) aur `splits_bigord.npz` (ordinary, non-env-180 subset), `C2_BANNED=1`
ke saath aur uske bina. Class 1 girna nahi chahiye. LB pe tabhi jaayega jab prediction ≥ +0.005 ho.
Warna ranker ke saath jaayega.

**Coverage model khud test pe hai (Claude ka call):** memory rule class 3 deltas ko coverage ~0.12 se scale
karta hai. Is hisaab se same pool ke andar re-ranking LB pe mushkil se dikhni chahiye. Lekin prvsiyan ka pool
bhi training ∪ COCONUT, ±10 ppm hai, yaani hamare jaisa, aur usne 0.335 banaya, jabki hum 0.257 pe hain. Matlab
0.12 wala decomposition shayad galat hai. Shayad hidden test pe hamare ranker ki quality held-out se kaafi kam
hai, aur coverage zyada hai. Isliye:
- Is component ka LB prediction ordinary split (`splits_bigord.npz`, non-env-180 class 3) se banega, kyunki
  uska class 3 coverage aur quality dono saath pakadta hai. Covered split ke delta ko 0.12 se scale karna
  conservative lower bound hai.
- Agar LB scaled prediction se kaafi zyada hile, to ye `LEARNING.md` me ek entry hogi: hamara coverage model
  galat tha, aur kyun.

### 2. Learned ranker

**Insight (hamara apna + public):** haath se rakhe weights held-out se LB pe transfer nahi hote (RESULTS
§15–16, v10/v11/v13). Public pipelines ye problem ek GBM se solve karte hain jo simulated queries pe train
hota hai: prvsiyan ke 31 features × 8 HistGBM, aur engine 1 ke 160 features × 4 LightGBM.

**Kya banana hai:** ek ranker jo har candidate ke features leta hai aur ek score deta hai:
- library similarity (cosine/entropy, direct/shifted),
- analog features (component 1),
- predicted-fingerprint fit,
- prior,
- mass error,
- pool size,
- formula-explained fraction, etc.

Training rows `split.py` jaise simulated queries se banengi: held-out structures, structure pe grouped.

**Ek zaroori shart:** jo fingerprint model feature banata hai, usne ranker ke training queries nahi dekhe
hone chahiye. Warna feature training me overconfident hoga aur test pe nahi. Ise kaise ensure karna hai
(k-fold, alag split, ya `splits_big` ke leak-free models) builder ka call hai, par shart todni nahi hai.

Library (LightGBM lambdarank ya binary classifier) builder ka call hai.

**MetFrag-lite isi component ka feature hai (Claude ka call):** prvsiyan ke ledger me MetFrag-lite ne +0.021
diya (0.245 → 0.266). Hamara `formula.py` (fraction of intensity explained) iska kamzor rishtedaar hai
(AUC 0.6). Bond-cleavage fragments (1–2 cuts, H-shifts) ka score ranker ke features me jaata hai, alag
component nahi hai.

### 3. Gated PubChem channel

**Insight:** PubChem ko pool me merge karna dono taraf haara. Hamara v16 −0.014, aur unka 0.335 → 0.205.
Kaam karne wala tareeka:
- PubChem ka alag candidate list,
- sirf weak library match wale molecules ke liye,
- list ke fixed slots pe insert,
- rank 1 pe promotion sirf jab fingerprint margin (`rel`) bada ho.

Unke yahan dusra engine galat promotions ko veto karta hai.

**Hamare paas pehle se:** `data/ext_pubpop2/` (16M, 2+ synonyms) aur `data/pubchem_meta/counts.npz`.

**Measure:** held-out PubChem ka dilution price nahi kar sakta (3 baar haara). Fixed slots nuksaan ko
structurally limit karte hain, isliye worst-case loss seedha calculate karna hai. Iska asli measurement
ek probe submission hai, jiski prediction (worst case aur expected) score aane se pehle likhi jaayegi.

**Rank-1 promotion pehle nahi (Claude ka call):** public pipeline me galat promotions ko dusra engine veto
karta hai, aur hamare paas dusra engine nahi hai. Ek galat promotion ek molecule ka 0.5 tak kha sakta hai.
Isliye pehla version sirf fixed slots ka hoga. Promotion tabhi aayega jab koi veto (jaise ranker ki
confidence) maujood ho. prvsiyan ke ledger me fixed slots akele +0.000 the (0.331 → 0.331), aur prior ke saath
0.340. Isliye is component se bada LB gain expect nahi karna hai. Iska learning value ye samajhna hai ki
merge kyun haarta hai aur gating kyun bachata hai.

### 4. Apna forward model (timsTOF)

**Insight (public):** galat winners me se zyada tar sahi answer ka hi formula share karte hain (prvsiyan:
98%). Fingerprint aur analog Tanimoto isomers ko alag nahi kar paate. Forward model har candidate ka spectrum
predict karta hai, aur same-formula group ke andar order theek karta hai. GLACIER ne ICEBERG ko haraya, aur
unka andaza hai ki iski ek wajah instrument match hai.

**Decide hua (user ne Claude ke suggestion pe haan kaha):**
- Apna forward model plain PyTorch me banana.
- Use `enveda-180` ke timsTOF spectra (1.15M) pe train karna, kyunki test bhi timsTOF hai.
- Mac GPU (MPS) ya Kaggle GPU pe chalana.
- Pretrained GLACIER sirf **comparison** ke liye use hoga, ye dekhne ke liye ki hamara model kitna peeche hai
  aur kyun. Ship nahi hoga.

**Builder ke liye:** architecture builder ka call hai. Learning ke liye simple shuru karna better hai, jaise
fingerprint → binned spectrum (NEIMS-style). Fragment-level model baad me, agar waqt ho. ms-pred DGL pe
bana hai, aur jahan tak hum jaante hain DGL MPS support nahi karta, isliye GLACIER comparison shayad Kaggle
GPU ya CPU pe chalega. Ye verify karna hai.

**Eval (Claude ka call):** held-out class 3 ke non-`enveda-180` queries Orbitrap/QTOF ke hain, isliye woh
timsTOF model ko galat instrument pe naapenge. Is component ke do eval sets honge:
- **`enveda-np-examples`** (250 NP, timsTOF, test jaisa pipeline): forward model ki training se bahar
  rakhna. Spectrum prediction quality aur same-formula re-rank dono yahan naapne hain. GUIDE.md ka forum
  note kehta hai ki ye set LB ke saath correlate nahi karta. Isliye ise sirf forward model ki quality ke liye
  use karna hai, LB prediction ke liye nahi.
- **`enveda-180` ke held-out answers:** instrument sahi hai, chemistry galat hai.

Pretrained GLACIER inhi dono sets pe comparison hoga. LB slot tab bhi +0.005 gate se hi milega. Agar gate
pass na ho, to bhi component learning ke liye poora hai.

**Waqt kam pada to** isi component ko chhota karna ya chhodna hai, pehle teen ko nahi.

## Reserve list: agar waqt bache (Claude ka call)

Agar koi component jaldi khatam ho ya signal na de, to bacha waqt is order me jaayega:
1. **ChEBI + LIPID MAPS** candidates, unke source se khud banake. Sasta hai, aur public ledger me +0.005 tha.
2. **Analog-edit candidate generation:** mass-shifted analog ko Δmass ke hisaab se modify karke naye
   structures banana. Isse coverage badhta hai, re-ranking nahi. Engine 1 ka hissa hai.
3. **Richer fingerprint:** 2048 Morgan bits ki jagah kai fingerprint types (unka 6,930 bits), `f·z` scoring,
   merged + single spectrum model pair. Unke ledger me ye 0.299 → 0.335 tak le gaya.
4. **Do engines ka reciprocal-rank fusion.**

## Waqt ka rough plan (Claude ka call)

| hafta | kaam |
|---|---|
| 1 | component 1 |
| 2–3 | component 2 |
| 4 | component 3 |
| 5–7 | component 4 |
| 8 | final selection, buffer |

Ye andaza hai, commitment nahi.

## Builder pe chhoda gaya

- Saare numbers: analog window, N, p, similarity tolerance, weights, thresholds, slot positions.
- Numba vs numpy.
- Index ko 64 se zyada peaks ke saath dobara banana (pehle 64 pe naapna).
- Ranker library aur objective.
- Forward model architecture.
- File layout (naye `.py` files ya maujooda files me), aur kernel me integration ka tareeka.
- Har din ki Kaggle submission limit check karna.

## Success

- 14 Dec ko `LEARNING.md` me har component ka insight, user ka andaza, measured number, fark ki wajah, aur
  hamari purani belief likhi ho.
- Har component held-out pe alag naapa gaya ho. Jo +0.005 scaled se upar gaye, woh LB pe alag naape gaye hon.
- Final selection hamara best own-code submission ho.
