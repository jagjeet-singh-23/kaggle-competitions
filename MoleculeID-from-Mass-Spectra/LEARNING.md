# Learning log: Raasta B

Har entry ek experiment hai: doosron ka insight, mera andaza (experiment se **pehle** likha gaya), asli number,
fark ki wajah, aur hamari purani belief jo isse badli. Design: `docs/superpowers/specs/2026-10-20-raasta-b-design.md`.

---

## 1. Kya hidden-jaise molecules ke rishtedaar library me hain? (2026-10-20)

**Unka insight.** prvsiyan ka analog-propagation baseline library me query ke *rishtedaar* dhoondhta hai: ±200 Da
ke andar, fragments seedhe ya mass-difference se shift karke match. Unke ledger me library-only 0.151 se is channel
(aur ek ranker) ke saath 0.233/0.245 hua, jo poore ledger ka sabse bada step tha.

**Hamari purani belief.** `analog.py` seeds sirf query ke ±10 ppm precursor window se leta tha, yaani sirf
isomers, aur unme bhi sirf ek seed (RESULTS §5: "one seed beats many"). Hum maante the ki analog propagation
class 2 ka hi kaam hai, aur class 3 sirf predicted fingerprint aur bade database se pahunchega.

**Sawaal.** `splits_bigord` ke 302 non-`enveda-180` class 3 answers me se kitne % ke liye ±200 Da ke top-80
rishtedaaron me koi aisa structure hai jiska answer se Morgan Tanimoto ≥ 0.5 ho (A)? Aur aaj ke ek same-mass seed
ke liye ye % kitna hai (B)?

**Andaza (user, experiment se pehle).** A = 22%, B = 8%.

**Measured.** Entropy similarity, `C2_BANNED=1`, 302 answers (`data/jobs/shift_stats.sh`, `shift_examples.sh`):

| kis set me rishtedaar dhoondha | Tanimoto ≥ 0.5 | ≥ 0.7 |
|---|---|---|
| **A**: ±200 Da ke top-80 spectral rishtedaaron me sabse kareebi | **66.2%** | 34.8% |
| unme se sirf top-1 (sabse achha spectral match) | 33.1% | 16.2% |
| **B**: aaj ka ek same-mass seed | **17.2%** | 8.3% |
| control: ±200 Da ke 80 *random* structures | 9.6% | 6.0% |

Cosine se A 54.3% hai, yaani entropy 12 points aage hai. `enveda-180` ke 698 answers pe A 72.9% hai, lekin un pe
class 3 MRR 0 hai (wo synthetic screening compounds pool me hain hi nahi, §31).

Lipids alag karke (answer ke SMILES me lagaatar 8 aliphatic carbon, ek mota proxy):

| | n | A | top-1 | random 80 |
|---|---|---|---|---|
| long chain | 35 | 97.1% | 74.3% | 60.0% |
| baaki | 267 | **62.2%** | 27.7% | **3.0%** |

**Andaze se fark.** A: 22% socha tha, aaya 66% (lipids hata ke 62%), yaani 3 guna. B: 8% socha tha, aaya 17%,
yaani 2 guna. Dono andaze ek hi taraf galat the: library hamari soch se kaafi zyada "pariwaron" me bani hai.

**Fark ki wajah.**
1. *Pariwar mass ke kadmon me alag hote hain.* Jin 200 answers ka kareebi rishtedaar mila, unme sabse aam mass
   difference: ±14.02 Da (CH₂) 35 baar, ±28.03 (C₂H₄) 14, 0.00 (isomer) 11, ±15.99 (O, hydroxylation) 5,
   +2.02 (H₂, ek double bond) 4, +57.02 (C₂H₃NO, glycine) 3, −4.03 (2 H₂) 3. Ye wahi badlav hain jo natural
   products aur metabolism me hote hain. Top-12 differences sirf 79 pairs hain, baaki 121 ek lambi poonch hai.
2. *Same-mass seed inme se sirf "0.00" wale dekh sakta tha.* Baaki sab rishtedaar ±10 ppm window ke bahar hain, isliye
   B, A se 4 guna chhota hai. Fark seed ki *quality* ka nahi hai: hamari window rishtedaaron tak pahunchti hi nahi thi.
3. *Signal asli hai, sirf "80 molecules me koi to milega" nahi.* Non-lipids pe random 80 se sirf 3% milte hain, aur
   spectral rishtedaaron se 62%, yaani 20 guna.
4. *Lipids pe Morgan fingerprint andha hai.* Long-chain answers pe random 80 bhi 60% de dete hain. Radius-2 Morgan
   chain ki lambai nahi dekhta, isliye C16 aur C18 chain wale phospholipid ka Tanimoto **1.00** aata hai. Ye hamare
   fingerprint term ki bhi limitation hai, sirf analog term ki nahi: lipid candidates aapas me tie hote hain, aur
   unhe sirf precursor mass alag karta hai. Public pipeline ka 6,930-bit pair shayad isi ko pakadta hai.
5. *Top-1 rishtedaar sabse kareebi nahi hota.* Top-1 se 33% milte hain, top-80 se 66%. §5 ka "one seed beats many"
   yahan seedha laagu nahi hota. Lekin "80 me koi kareebi hai" ek oracle hai, kyunki ise jaanne ke liye answer pata
   hona chahiye. Bina answer jaane scoring sahi rishtedaar chun paati hai ya nahi, ye experiment 2 (MRR) batayega.
6. *Jinke koi rishtedaar nahi* (Tanimoto < 0.3) unme ek furochromone, ek triazine amide, ek sulfonamide-oxadiazole aur
   ek streptomycin jaisa aminoglycoside the. Ye rare scaffolds hain jinka library me koi pariwar nahi hai.

**Rasta me ek bug mila.** Pehli run me entropy similarity 1.38 tak gayi, jo ho hi nahi sakta. Wajah: positive
representatives ke 30% aur negative ke 51% spectra me ek hi 0.01 Da bin me 2+ peaks the. Vectorised code query ke ek
peak ko dono se milata tha, yani do baar ginta tha. Task 2 me maine ise "rare" maan ke chhoda tha, bina data
dekhe. Fix: bin ke andar peaks pehle jodo, phir milao (`shifted._runs`); test pehle fail hua, phir pass. Pehli run ke
numbers void hain (A tab 65.6% tha, isliye nateeja nahi badla, lekin MRR pe 1 se upar ki similarity asar daalti).
Seekh: "rare hai" ek claim hai, aur claim ka number nikaalna padta hai.

**Purani soch kahan se aayi.** RESULTS §5 me seeds query ki ±10 ppm precursor window se aate the: candidate pool
aur seeds ek hi window se. Wahan seed 2, 3, ... same mass ke *badtar* spectral matches the, yaani zyadatar
alag scaffold wale isomers, aur unhone ranking ko dilute kiya. Humne usse "zyada seeds = noise" seekha. Asli seekh
thi "same mass ke zyada seeds = noise". Candidate ka mass precursor se match hona chahiye; seed ka nahi. Mass-shifted
matching ye do windows alag kar deta hai.

**Belief jo badli.** Class 3 ke liye spectral library khaali haath nahi hai: non-lipid held-out answers me se 62% ka
ek kareebi rishtedaar ±200 Da me spectra ke saath maujood hai. Lekin "rishtedaar maujood hai" aur "answer rank 1
pe aayega" alag baatein hain. Aur hidden test ke class 3 molecules library se door bhi ho sakte hain, isliye LB
prediction me coverage ka lower bound bhi rakhenge (experiment 2).

---

## 2. Kya rishtedaar answer ko upar kheench paate hain? (2026-10-20)

**Unka insight.** Wahi channel (experiment 1), ab score me: candidate `c` ko `SW · max_a sim_a³ · T(c, a)` milta
hai, jahan `a` ±200 Da ke top-80 spectral rishtedaar hain. prvsiyan ke ledger me library-only 0.151 se 0.233/0.245.

**Hamari purani belief.** Class 3 ka signal sirf predicted fingerprint (`32·T(c, fp)`) aur prior se aata hai.

**Sawaal.** v15 config (`GAMMA=32`, COCONUT synonym prior w 0.5 gate 0.8) pe, sabse achhe `SW` pe
`splits_bigord` ka non-`enveda-180` class 3 MRR 0.135 se kitna hoga? Ceiling ≈ 0.25, kyunki 302 me se sirf 76
answers pool me hain (unpe aaj 0.510, baaki 226 pe 0.009). Aur class 1 (aaj 0.927) pe kya hoga?

**Andaza (user, experiment se pehle).** Class 3: **0.185**. Class 1: gir sakta hai, badhega nahi. Best `SW`: experiment
hi batayega.

**Measured.** Poori tables RESULTS §32 me hain. Entropy, p 3, `C2_BANNED=1`:

| SW | 0 | 4 | 8 | **16** | 32 | 64 |
|---|---|---|---|---|---|---|
| non-env class 3 | 0.1355 | 0.1398 | 0.1428 | **0.1507** | 0.1488 | 0.1494 |
| class 1 (bigord) | 0.9269 | 0.9249 | 0.9238 | 0.9232 | 0.9148 | 0.9110 |
| covered class 3 (`splits_big`) | 0.4819 | 0.4967 | 0.5095 | 0.5234 | 0.5305 | 0.5375 |

Per-molecule ranks (SW 0 → 16): non-env me sirf 65 answers kisi bhi weight pe top 25 me hain. 16 upar gaye aur 3
neeche. 25 se neeche wala koi answer top 5 tak nahi pahuncha.

| non-env answers jo top 25 me hain | n | MRR SW 0 → 16 |
|---|---|---|
| jinka Tanimoto ≥ 0.5 rishtedaar hai (experiment 1) | 43 | 0.645 → **0.750** |
| jinka nahi | 22 | 0.599 → 0.602 |

**Andaze se fark.**
- *Class 3:* andaza **0.185** (+0.05) tha, asli **0.1507** (+0.015), yaani predicted gain ka 30%. Dilchasp baat ye hai
  ki tumhara 0.185 lagbhag ek *perfect* channel ka number hai. Agar top 25 me pahunche har us answer ko, jiska
  rishtedaar hai, rank 1 mil jaata, to 0.135 + 43 × (1 − 0.645)/302 = **0.186** aata. Channel ne us headroom ka
  30% liya.
- *Class 1:* "gir sakta hai, badhega nahi" sahi nikla. SW 16 pe girawat −0.004 hai, SW 64 pe −0.016. `splits_big`
  pe class 1 flat raha.

**Fark ki wajah.**
1. *Channel re-rank karta hai, rescue nahi.* Ye term pool ke andar sirf order badalta hai. Jo answer pehle se top 25
   me tha, wahi upar khisakta hai (5 answers rank 2–5 se 1 pe aaye). Rank 25 ke neeche wale answers ke upar
   fingerprint aur prior ke itne points waale competitors hain ki ek rishtedaar ka vote unhe paar nahi kara paata.
2. *Experiment 1 ka oracle sahi dhang se predict karta hai.* Jinka rishtedaar hai unme +0.105, jinka nahi unme ~0.
   `splits_big` pe bina rishtedaar wale thoda *girte* hain (0.547 → 0.540): rishtedaar galat ho, to term galat
   pariwar ko upar kheenchta hai.
3. *Class 1 me near-ties ki adla-badli hoti hai.* `splits_bigord` pe 12 answers rank 1 se 2–5 pe gire aur 9 ulta
   upar aaye. Mera andaza (naapa nahi hai): answer ke same-Morgan isomers ko bhi wahi analog points milte hain, to
   tie todne ka kaam baaki terms pe aa jaata hai.
4. *Entropy vs cosine.* Experiment 1 me entropy se 12 points zyada rishtedaar mile, lekin MRR me fark sirf 0.006
   (non-env) aur 0 (covered) raha. Rishtedaar dhoondh lena aur use ranking me kaam laana alag cheezein hain. Upar
   ke ranks pe jo rishtedaar maayne rakhte hain, wo dono similarities dhoondh leti hain.

**LB andaza (SW 16).** Plan ka formula +0.011 deta hai. Coverage-scaled (hidden test ka ~12% pool me):
+0.0035 se +0.005. Ye ship threshold (+0.005) pe ya uske neeche hai, isliye ye probe hai, ship nahi. v15 (0.257)
ke upar probe ka andaza: **0.260–0.262** coverage-scaled, 0.268 plan ke formula se. LB noise ±0.003–0.005 hai, to
0.260 ka score bhi "koi fark nahi" se alag nahi hoga. Sirf ≥ 0.262 hi kuch kehta hai.

**Belief jo badli.** prvsiyan ke ledger me is channel ka bada step (0.151 → 0.233) ek library-only baseline pe tha,
aur usi step me ek ranker bhi juda tha. Hamare pipeline me predicted fingerprint aur COCONUT prior shayad pehle se
wahi kaam ka kuch hissa kar rahe hain, isliye bacha hua gain chhota hai (ye andaza hai, naapa nahi). Ek channel ka
faayda sirf channel pe nahi, is pe bhi tay hota hai ki baaki pipeline pehle se kya kar raha hai. Agla bada sawaal:
non-env ke 237 answers jo top 25 me hain hi nahi, unme se 226 pool me hi nahi hain. Unke liye ranker ya re-ranking
kaam nahi aayega; wahan pool chahiye, aur pool badhana teen baar LB pe haara hai (§31).

## 3. Kya held-out ka gain LB tak pahuncha? (2026-10-20)

**Sawaal.** v17 = v15 + shifted analogs (`SHIFT_W=16`, entropy, p 3). v15 0.257 pe hai. Held-out se do LB andaze
bante hain. Plan ka formula ordinary split ka non-env class 3 leta hai, jahan coverage 76/302 = 0.25 hai: +0.011.
Coverage-scaled rule covered split ke delta ko 0.12 se guna karta hai: +0.0035. Kaunsa sahi hai?

**Andaza (pehle se likha, entry 2).** 0.260–0.262 coverage-scaled, 0.268 plan ke formula se. Sirf ≥ 0.262 kuch
kehta hai.

**Measured.** v17 **0.264** (submission 57439024), yaani v15 se **+0.007**. Rounding ke saath ye +0.006 se +0.008
ke beech hai.

| model | Δ andaza | LB andaza | asli − andaza |
|---|---|---|---|
| coverage-scaled (covered delta × 0.12) | +0.0035 | 0.2605 | +0.0035 |
| beech wala (non-env ke 76 covered answers, coverage 0.12) | +0.005 | 0.262 | +0.002 |
| plan ka formula (non-env, coverage 0.25) | +0.011 | 0.268 | −0.004 |
| **asli** | **+0.007** | **0.264** | |

**Andaze se fark.** Asli number dono models ke beech gira, aur dono se ek LB-noise (±0.003–0.005) ke andar hai.
Isliye LB kisi bhi model ko galat sabit nahi karta.

**Coverage ulta nikaalo.** Plan ke formula me class 1 aur 2 ka hissa lagbhag zero hai (−0.0005 + 0.0004). To class 3
ka asli gain ≈ (0.007 + 0.0002) / 0.735 = **+0.0098**. Ab dekho kitne coverage pe ye gain aata:
- Non-env ke har covered answer pe gain +0.060 tha (0.0152 × 302/76). Is hisaab se hidden coverage ≈ **0.16**.
- `splits_big` ke covered gain (+0.0415) ke hisaab se ≈ **0.24**.
- LB ka ±0.004 noise jodo to pehla 0.07–0.25 ho jaata hai aur dusra 0.11–0.37.

Dono point estimates 0.12 se upar jhukte hain. HANDOFF pehle se kehta tha ki asli coverage shayad 12.4% se kuch zyada
hai, kyunki wo number leaky model se aaya tha. Lekin ek LB point coverage ka sawaal tay nahi karta: 0.12 galat
sabit nahi hua, bas thoda kam lagta hai.

**Ye probe kya keh sakta tha aur kya nahi.** Do models ke andaze 0.0075 alag the, aur noise ±0.004 hai. Aisa probe
tabhi faisla deta jab result kisi kinare pe girta: ≤ 0.259 hota to coverage-scaled sahi, ≥ 0.269 hota to plan ka
formula. Beech me girne pe sirf itna pata chalta hai ki channel ka asar positive hai. Coverage ko sach me test karna
ho to aisa change chahiye jahan dono models ke andaze noise ke 3–4 guna alag hon.

**Jo pakka hua.**
- Gain noise ka ~1.5–2 guna hai, aur ≥ 0.262 wali pehle se likhi shart paar hui.
- v12 (+0.016) ke baad ye pehla LB step hai jo noise se bahar hai. v14 (+0.002) aur v15 (+0.003) noise ke andar the.
  v13 (−0.007) aur v16 (−0.014) held-out pe jeete the aur LB pe ulte gire.

**Kyun transfer hua (andaza, teen cases pe, rule nahi).** v13 ne global scoring badla (normalisation, tau). v16 ne
pool badla. Dono me har molecule ke har candidate ka score hila. Shifted term pool ko nahi chhoota, aur sirf un
molecules ko hilata hai jinka sach me ek spectral rishtedaar hai (entry 2: bina rishtedaar wale ~0 hile). Mera
andaza hai ki jo change kam molecules ko, ek khaas wajah se, chhoota hai, uska held-out → LB transfer zyada
bharosemand hota hai.

**Ek side observation.** Scoring ~2.5 ghante chala (18:36 → ~21:00), jabki v15 ka ~70 min tha aur v17 ka commit run
84 min ka. Naya kharcha sirf shifted search hai, jo spectra ki ginti ke saath badhta hai. To shayad hidden test me
placeholder se zyada spectra hain, ya Kaggle ki queue thi. Ye naapa nahi hai. 9 ghante ki limit abhi door hai, lekin
component 2–4 jo bhi time jodenge, wo ~2.5 ghante pe judega, 84 min pe nahi.

**Belief jo badli.** HANDOFF ka rule kehta tha ki same pool ke andar ranking ka kaam LB pe mushkil se dikhega, kyunki
ship ke liye covered quality pe +0.057 chahiye. Is channel ne covered pe +0.0415 diya aur LB pe phir bhi +0.007.
Ranking ka kaam LB pe dikh sakta hai, khaas kar jab wo sirf sahi molecules ko chhoota ho.

Component 2 (ranker) ke liye ye achha sanket hai, lekin usme wahi khatra bhi hai: ranker har candidate ka score
badlega, bilkul v13 ki tarah. Uska LB andaza dono models se banana aur noise ke saath padhna hoga.

**Final selection.** Pehle se likhi shart (≥ 0.262) ke hisaab se v17 final selection me jaata hai. Sujhaav: v17
pehla, v15 dusra. v15 bina channel wala hedge hai, agar private split pe channel ka asar ulta nikle.

## 4. Kya seekha hua ranker haath ke weights se behtar jodta hai? (2026-10-24)

**Unka insight.** Haath se rakhe weights held-out se LB tak transfer nahi hote (v10, v11, v13). Public pipelines
weights khud nahi chunte, ek GBM ko simulated queries pe sikhate hain: prvsiyan ke 31 features × 8 HistGBM, aur
engine 1 ke 160 features × 4 LightGBM.

**Hamari purani belief.** v17 ka total, `spectral + analog + 32·T(fp) + 16·shift + 0.5·prior`, har weight alag-alag
held-out pe chuna gaya tha. Weights ek doosre pe nirbhar hain, jaise kab fingerprint pe bharosa karna hai aur kab
spectral match pe, lekin is tarah chunne me wo kabhi saath me nahi dekhe gaye.

**Sawaal.** v17 ke top-100 candidates pe ek GBM ranker (4 seeds, 5-fold OOF, folds structure pe) chalega, MetFrag-lite
aur formula ke bina. Wo v17 ke terms aur unke rank/gap versions dekhta hai. Ranker koi candidate nahi jodta, isliye
upar ki seema ceiling hai: v17 ke top 100 me jo answer hai wo rank 1 pe aa jaaye.

| reading | v17 | ceiling |
|---|---|---|
| bigord non-env class 3 (302) | 0.1507 | 0.238 |
| big covered class 3 (1000) | 0.5234 | 0.930 |
| bigord class 1 (400) | 0.9232 | 0.998 |

**Andaza (user, experiment se pehle).**
1. bigord non-env class 3: **0.1665**
2. big covered class 3: **0.58**
3. bigord class 1: **0.9253**
4. Linear GBM se kitna peeche: GBM ka v17 ke upar gain linear ke gain se **22% zyada** (Claude ki samajh: linear
   GBM ke gain ka ~82% leta hai)
5. LightGBM `lambdarank` binary GBM se: **aage hoga**

**Measured.** 5-fold OOF, folds structure pe, 4 seeds. Bracket me v17 ke upar paired-bootstrap 95% interval hai.

| reading | v17 | GBM | linear | lambdarank | andaza |
|---|---|---|---|---|---|
| bigord non-env class 3 | 0.1507 | 0.1632 [+0.004, +0.023] | 0.1622 | **0.1662** [+0.005, +0.028] | 0.1665 |
| big covered class 3 | 0.5234 | 0.6248 [+0.087, +0.114] | 0.6262 | **0.6666** [+0.126, +0.159] | 0.58 |
| bigord class 1 | 0.9232 | 0.9218 | **0.9040** [−0.029, −0.010] | 0.9283 | 0.9253 |
| big class 1 | 0.9151 | 0.9220 | 0.9038 [−0.023, −0.001] | 0.9224 | |

Lambdarank ke saath answers kaise hile (`data/jobs/exp3_read.py`):

| | upar | neeche | rank 1 pe aaye | rank 1 khoya | top 25 me aaye | top 25 se bahar |
|---|---|---|---|---|---|---|
| big covered class 3 (1000) | 398 | 53 | 208 | 15 | 16 | 4 |
| bigord non-env class 3 (302) | 24 | 5 | 6 | 2 | 5 | 1 |

**Andaze se fark.**
1. *Non-env class 3:* andaza 0.1665 tha, lambdarank 0.1662 laaya. Ye lagbhag bilkul sahi baitha. GBM 0.1632 pe ruka.
2. *Covered class 3:* andaza +0.057 tha, asli +0.101 (GBM) se +0.143 (lambdarank), yaani 2–2.5 guna.
3. *Class 1:* andaza +0.002 tha. Disha sahi nikli: lambdarank +0.005, GBM lagbhag 0.
4. *Linear vs GBM:* andaza tha ki GBM ka gain 22% zyada hoga. Class 3 pe aisa nahi hua, wahan dono barabar rahe (covered +0.103 vs +0.101, non-env +0.0115 vs +0.0125). Fark class 1 me dikha: linear ne class 1 −0.011 aur −0.019 giraya, GBM ne nahi.
5. *Lambdarank binary GBM se aage:* sahi. Teeno readings pe aage raha.

**Fark ki wajah.**
1. *Covered split me jagah thi, non-env me nahi.* Covered ke 51% answers pehle se rank 2–25 me the (ceiling 0.93), jabki non-env ke sirf 9.3% (ceiling 0.238). Lambdarank ne covered ki bachi jagah ka 35% bhara aur non-env ka 18%.
2. *Ranker ne seekha ki spectral match kab dhokha deta hai.* Sabse zyada upar gaye answers ke saamne aksar kisi aur structure ka 0.4–0.9 wala spectral match tha (notebook). v17 `spec` ko seedha total me jodta hai, to woh candidate jeet jaata tha.
3. *Linear model se samajh aata hai ki trees kyun chahiye* (mera andaza, upar ke numbers pe tika, alag se naapa nahi). Linear sirf ek weight seekh sakta hai: "spec kam bharosemand hai". Isse class 3 utna hi jeeta jitna GBM se, lekin class 1 gira, aur sabse zyada wahan jahan library match beech ka tha (`top_spec < 0.8`: −0.018 aur −0.031). Trees seekh sakte hain ki *kab* spec pe bharosa karna hai, molecule ke sabse achhe match (`top_spec`) ke hisaab se. Trees ka faayda class 3 ke gain me nahi, class 1 ko bachane me hai.
4. *Lambdarank seedha ranking pe train hota hai.* Binary logloss har row ko alag dekhta hai, aur jab ~60 me se ek hi positive ho to zyada mehnat negatives ko neeche dhakelne me jaati hai. Lambdarank sirf query ke andar ka order sudhaarta hai. Usne class 1 ke strong matches (`top_spec ≥ 0.8`) pe bhi +0.016–0.018 diya.

**Documentation strata** (lambdarank, answer ke COCONUT synonyms):

| | 0–1 | 2–4 | 5–19 | 20+ |
|---|---|---|---|---|
| big covered | 0.420 → 0.513 (n 246) | 0.483 → 0.554 (168) | 0.507 → 0.702 (314) | 0.660 → 0.834 (272) |
| bigord non-env | 0.038 → 0.037 (244) | 0.444 → 0.482 (15) | 0.488 → 0.617 (20) | 0.857 → 0.935 (23) |

Documented answers zyada jeete, lekin 0–1 synonym wale answers bhi covered pe +0.093 jeete (GBM +0.062, linear +0.029). Ranker poori tarah prior pe nahi tika. Non-env ka 0–1 stratum zyada tar pool me hai hi nahi, isliye wahan kuch nahi hila, aur non-env ka saara gain 58 documented answers se aaya.

**LB ka pehla andaza** (lambdarank; final andaza Task 7 me, saare features ke saath):
- plan formula: +0.014
- coverage-scaled (0.12): +0.015
- coverage 0.16 pe (entry 3 ka back-solve): +0.019
- pessimistic, agar hidden covered answers 0–1 stratum jaise hon: +0.010

Is baar dono LB models lagbhag ek hi number dete hain, kyunki non-env ka gain covered gain ka 11% hai, jo coverage 0.12 ke kareeb hai. Shifted analogs me ye 37% tha, isliye wahan dono models 3 guna alag the.

**Belief jo badli.** Weights ko ek-ek karke tune karna kaafi nahi tha. Har term kab kaam karta hai, ye seekhne se covered class 3 pe +0.10–0.14 mila. Ye v12 ke baad kisi bhi ranking change se bada hai. Khatra wahi hai jo entry 3 me likha tha: ranker har candidate ka score badalta hai, jaise v13 ne kiya tha, aur v13 LB pe ulta gira tha. Isliye LB ka faisla Task 7 me dono models aur strata ke saath hoga.

## 5. MetFrag-lite kitna jodta hai, aur ranker kis cheez pe tika hai? (2026-10-24)

**Unka insight.** prvsiyan ke pipeline me MetFrag-lite (candidate ke bonds tod ke peaks samjhana) ne LB 0.245 → 0.266
kiya, +0.021. Formula check (peak ka formula candidate ke formula ka sub-formula hai ya nahi) isomers ko alag nahi kar
sakta, kyunki unka formula ek hai; bond todna kar sakta hai, kyunki isomers alag tootte hain.

**Hamari purani belief.** RESULTS §12: `formula` ka pool ke andar AUC ~0.6 tha, aur wo v17 me nahi gaya. Experiment 3
me lambdarank ne frag aur formula ke bina big covered class 3 ko 0.5234 → 0.6666 kiya.

**Sawaal.** (1) Pool ke andar answer ko baaki candidates se alag karne me akele `frag` ka AUC? (2) Saare features ke
saath (frag aur formula bhi) big covered class 3 OOF kitna? (3) Kaunsa group hatane se class 3 sabse zyada girega?

**Andaza (user, experiment se pehle).**
1. `frag` ka pool AUC: **0.72**
2. big covered class 3, saare features (lambdarank): **0.69** (experiment 3 ke 0.6666 se +0.023)
3. Sabse zyada nirbharta: **prior**

**Measured.** Lambdarank, 5-fold OOF. Lambdarank deterministic hai, isliye 1 aur 4 seeds ke OOF bilkul ek jaise
aaye (max diff 0.0), aur ablation 1 seed pe hi chala.

| reading | v17 | bina frag/formula | saare features | andaza |
|---|---|---|---|---|
| big covered class 3 | 0.5234 | 0.6666 | **0.6882** (+0.0217 [+0.011, +0.031]) | 0.69 |
| bigord non-env class 3 | 0.1507 | 0.1662 | **0.1775** (+0.0113 [+0.003, +0.021]) | |
| bigord class 1 | 0.9232 | 0.9283 | 0.9366 | |
| big class 1 | 0.9151 | 0.9224 | 0.9299 | |

Pool ke andar AUC (answer vs baaki candidates):

| | big covered c3 | bigord non-env c3 | sirf exact isomers ke saamne (covered) |
|---|---|---|---|
| `frag` | **0.777** | 0.829 | **0.734** |
| `frag_mean` | 0.780 | 0.845 | 0.745 |
| `formula` | 0.542 | 0.540 | 0.500 |
| `fp` | 0.861 | 0.858 | 0.800 |
| `prior` | 0.758 | 0.789 | 0.747 |
| `v17` | 0.900 | 0.906 | 0.850 |

Ek group hatane pe class 3 kitna gira (full model ke saamne):

| hataya | big covered c3 | bigord non-env c3 | bigord class 1 |
|---|---|---|---|
| frag | **−0.0206** | **−0.0076** | −0.0059 |
| prior | −0.0143 | −0.0016 | −0.0030 |
| shift | −0.0119 | −0.0025 | −0.0112 |
| v17 | −0.0091 | −0.0047 | −0.0007 |
| pool | −0.0086 | −0.0040 | −0.0040 |
| fp | −0.0057 | −0.0041 | **−0.0272** |
| formula, mass, spec | ~0 | ~0 | ~0 |

**Andaze se fark.**
1. *frag ka AUC:* andaza 0.72 tha. Saare candidates ke saamne asli 0.777 aaya, sirf isomers ke saamne 0.734. Isomers
   wala test hi insight ko jaanchta hai, aur wo andaze ke bahut paas raha.
2. *Covered class 3:* andaza 0.69 tha (+0.023), asli 0.6882 (+0.0217). Ye lagbhag bilkul sahi raha.
3. *Sabse zyada nirbharta:* andaza `prior` tha. Class 3 pe sabse bada drop `frag` hatane se aaya (covered −0.021,
   non-env −0.008). `prior` covered pe doosre number pe raha (−0.014), lekin non-env pe lagbhag 0. Class 1 pe sab
   kuch `fp` pe tika hai (−0.027).

**Fark ki wajah.**
1. *Ablation importance nahi naapta, ye naapta hai ki kya kahin aur nahi hai.* `fp` ka akela AUC sabse uncha hai
   (0.861), phir bhi use hatane se class 3 sirf −0.006 gira, kyunki `v17` ke total aur shift terms me fp pehle se
   hai. `prior` bhi `v17` ke total me baitha hai (0.5·prior). `frag` ki jaankari kisi aur feature me nahi hai,
   isliye use hatane ka nuksaan poora dikhta hai.
2. *Isomers sach me alag tootte hain.* Exact isomers ke saamne `formula` ka AUC 0.500 hai, jo by construction aana
   hi tha. `frag` wahan bhi 0.734 raha. Notebook ke paanchon molecules me answer aur bina-frag wala winner ka
   `formula` ek tha, aur answer ke fragments ne zyada peaks samjhaye. Jaise lipid me 283.264 (stearate anion) sirf
   answer ki chain se banta hai, aur amino acid me 56.05, 84.04 aur 120.07 sirf answer ke tukdon se.
3. *frag ne sabse kam documented answers ki sabse zyada madad ki.* Covered pe 0–1 synonyms wale answers experiment 3
   me 0.513 the, ab 0.545 hain (+0.032). 20+ synonyms wale 0.834 se 0.845 hue (+0.011). Ye popularity se alag signal
   hai, aur hidden test ke kam documented answers ke liye achhi khabar hai.
4. *GBM ko frag se kam mila* (covered +0.015, non-env +0.004 jiska interval 0 cross karta hai). Lambdarank ko zyada
   mila. Mera andaza hai ki ye wahi wajah hai jo entry 4 me thi: `frag` query ke andar order ka signal hai, aur
   lambdarank seedha order pe train hota hai. Ye alag se naapa nahi gaya.

**LB andaza** (lambdarank, saare features, v17 ke upar):
- plan formula: +0.023
- coverage-scaled (0.12): +0.018
- coverage 0.16 pe: +0.023
- pessimistic, agar hidden covered answers 0–1 stratum jaise hon: +0.015

Chaaron andaze ship rule (+0.005) se upar hain. Khatra v13 wala hai, kyunki ranker har candidate ka score badalta hai.
Faisla Task 7 me kernel ke saath hoga.

**Belief jo badli.** prvsiyan ka insight seedha jaanch me sahi nikla: bond todne se isomers alag hote hain aur formula
check se nahi. Doosri seekh ye hai ki "ranker kis pe tika hai" ka jawab ablation se nahi milta. Ablation batata hai ki
kya irreplaceable hai, aur jo signal `v17` ke total me pehle se ghula hai (fp, prior), uska nuksaan wahan nahi
dikhta.

## 6. Kya seekhe hue ranker ka gain LB tak pahunchega? (2026-10-24)

**Sawaal.** v18 = v17 + lambdarank ranker (`ranker_v1.npz`: saare 27 features, MetFrag-lite ke saath, 500 trees,
numpy me export) jo har molecule ke v17 wale top 100 candidates ko dobara order karta hai. v17 abhi 0.264 pe hai.

**Claude ke LB andaze** (RESULTS §33, entry 5):

| model | Δ andaza | LB andaza |
|---|---|---|
| plan formula (bigord non-env, coverage 0.25) | +0.023 | 0.287 |
| coverage-scaled (covered delta × 0.12) | +0.018 | 0.282 |
| coverage 0.16 pe (entry 3 ka back-solve) | +0.023 | 0.287 |
| pessimistic (hidden covered answers 0–1 synonym stratum jaise) | +0.015 | 0.279 |

**Pre-registered threshold.**
- v18 final selection me tabhi jaayega jab LB **≥ 0.269** ho, yaani v17 + 0.005. Ye ship rule hai, aur LB ke noise
  (±0.003–0.005) se bahar hai.
- 0.264–0.268 ka matlab hai ki gain noise se alag nahi hua.
- 0.264 se neeche ka matlab hai ki transfer v13 ki tarah toota.

**Khatra, pehle se likha.**
1. *v13 jaisa badlav.* Ranker har candidate ka score badalta hai. Held-out pe ye class 3 ke 44%, class 2 ke 37% aur
   class 1 ke 8.8% top-1 badalta hai (placeholder pe 55/400). v13 ka chetavni wala signal, ki class 1 gir raha tha,
   yahan nahi hai: class 1 dono splits pe badha (+0.013, +0.015).
2. *Fingerprint ensemble ka fark.* Ranker 2-seed (bigord) aur 4-seed (big) held-out ensembles ke features pe
   seekha. Kernel 4-seed production ensemble chalata hai. Agar `fp` ka scale badla to raw `fp` features thode khisak
   sakte hain. Rank, gap aur z features is se bache rehte hain.
3. *Placeholder kuch nahi batata.* Uske spectra train se hain, isliye wo quality nahi naapta (HANDOFF).

**Runtime.** Features aur scoring Mac pe 18 s lete hain (34k rows). Kaggle pe ~1 min, v17 ke 84 min + ~2.5 h ke
upar.

**Andaza (user, push se pehle).** LB **0.273**, yaani v17 se +0.009. Ye threshold (0.269) se upar hai, lekin Claude ke
pessimistic andaze (0.279) se bhi neeche.

**Measured.** v18 **0.257** (submission 57599879), yaani v17 se **−0.007**. Rounding ke saath ye −0.006 se −0.008
ke beech hai. Pehle se likhi shart ke hisaab se ye teesri category hai: **transfer v13 ki tarah toota**. v18 final
selection me nahi jaata. v17 pehla aur v15 dusra rehta hai, jaise pehle tha.

| model | Δ andaza | LB andaza | asli − andaza |
|---|---|---|---|
| plan formula | +0.023 | 0.287 | −0.030 |
| coverage-scaled | +0.018 | 0.282 | −0.025 |
| coverage 0.16 pe | +0.023 | 0.287 | −0.030 |
| pessimistic | +0.015 | 0.279 | −0.022 |
| user | +0.009 | 0.273 | −0.016 |
| **asli** | **−0.007** | **0.257** | |

Saare andaze ek hi taraf galat hain, aur LB noise (±0.004) se 4–7 guna door. Ye noise nahi hai: held-out ka
**sign** hi galat nikla. User ka andaza sabse kam galat tha, kyunki wo sabse savdhaan tha. Lekin wo bhi sign nahi
pakad paaya.

**Kernel theek chala.** Kernel 88.7 min chala. Features 95 s me bane aur scoring 4 s me hui, 33,569 rows pe. 400 me
se 399 lists v17 se alag order me thi. Placeholder pe kernel ki file local smoke file se 396/400 lists pe mili: 1 ka
top-1 alag tha aur 3 ke top-25 sets alag the. Rows ki ginti same thi, to pools same the. Ye Linux aur Mac ka chhota
float fark hai, jo LB ke −0.007 ko samjha nahi sakta. Score aane me ~85 min lage (21:36 → 23:02). To v17 ke 2.5 ghante
shayad Kaggle ki queue the, zyada spectra nahi.

**Kya ulta-solve hota hai.** LB ≈ 0.140·c1 + 0.126·c2 + 0.735·c3. Ek LB point teen unknowns ko alag nahi kar
sakta, wo sirf do kinare deta hai:
- Agar class 1 aur 2 held-out jaise chale (+0.0038), to asli class 3 ~**0.015** gira: (−0.007 − 0.0038) / 0.735.
- Agar class 3 bilkul na hila, to class 1 ko ~**0.05** girna padta (0.007 / 0.140). Ye held-out ke +0.013 ke ulta hai.

LB ye nahi batata ki inme se kaunsa hua. Ise alag karne ke liye ya to probe chahiye, ya aisa local test jo hidden test
ke zyada kareeb ho.

**Global badlav ka record.**

| ver | badlav | held-out | LB |
|---|---|---|---|
| v12 | fp model + in-pool ranking loss, gamma 32 | jeeta | **+0.016** |
| v13 | fp term normalise, tau tez | class 3 jeeta, class 1 gira | −0.007 |
| v16 | PubChem pool + prior w 1 | jeeta | −0.014 |
| v17 | shifted analogs, sirf rishtedaar wale molecule hile | +0.0035 se +0.011 | **+0.007** |
| v18 | learned ranker, har candidate dobara score | +0.015 se +0.023 | −0.007 |

Entry 3 ka andaza tha ki jo badlav sirf kuch molecules ko, kisi khaas wajah se, hilata hai, wo zyada bharose se
transfer hota hai. v18 is andaze ke saath fit baithta hai. Lekin v12 bhi har candidate ko hilata tha aur transfer
hua. Ek fark dikhta hai:
- v12 ka model *train library* ke spectra pe seekha.
- v13 ke tau/gamma, v16 ka pool aur v18 ke 500 trees *held-out query lists* pe chune ya seekhe gaye.

Jo cheez held-out lists pe fit hoti hai, wo held-out ki khaasiyat bhi seekh leti hai. Ye abhi teen cases pe andaza
hai, rule nahi.

**Bootstrap interval ne kya nahi dekha.** Covered class 3 ka +0.165 [+0.148, +0.181] tha, aur interval zero se bahut
door tha. Interval sirf held-out ke andar ka sampling noise naapta hai. Held-out aur hidden test ka fark wo nahi
naapta, aur yahan wahi fark sabse bada nikla. Kitna bhi tight interval ho, wo transfer ki guarantee nahi deta.

**Kyun toota: andaze, jaanch ke saath.**
1. *Artifact ya ensemble ka fark* (pehle se likha khatra 2). Ranker 2-seed aur 4-seed held-out ensembles pe seekha,
   jabki kernel production ensemble chalata hai. Jaanch (local, free): shipped `ranker_v1.npz` ko production
   ensemble ke saath un held-out queries pe chalao jo na ranker ne dekhi na fp model ne (`splits.npz` ki queries,
   ranker ki tables wale structures hata ke). Agar wahan bhi gain dikhe, to bug ya mismatch wajah nahi hai.
2. *Popularity.* Ranker `prior` pe tika hai (ablation: covered −0.014). v17 ka prior sirf top_spec < 0.8 pe aur w
   0.5 pe lagta tha, aur v16 ka prior w 1 gira tha. Agar hidden answers held-out se kam documented hain, to ye
   nuksaan karega. Iske khilaf ek baat: held-out ke 0–1 synonym stratum me bhi gain tha (+0.125).
3. *Class 1.* Ranker class 1 ke 8.8% top-1 badalta hai. Probe ka idea: ranker sirf top_spec < 0.8 pe chalao, taaki
   class 1 jaise molecules v17 pe rahein. Isse LB ka nuksaan class 1 aur class 3 me baant sakte hain.

**Belief jo badli.** Entry 3 ke baad maine likha tha ki ranking ka kaam LB pe dikh sakta hai. Isme ab ek shart
judti hai: jo ranking *held-out lists pe seekhi* jaati hai, uska held-out gain LB ka andaza nahi hai, chahe wo kitna
bhi bada aur tight ho. Component 3–4 ke andaze isi hisaab se padhne honge.

**Final selection.** Badla nahi: v17 pehla, v15 dusra. `kernel/casmi_submit.py` wapas v17 pe hai. `RANKER` knob
`fpsearch` me hai, aur default me off hai.

## 7. Kya shipped ranker un queries pe bhi jeetta hai jo usne kabhi nahi dekhi? (2026-10-25)

**Sawaal.** v18 LB pe −0.007 gira (entry 6). Do kahaniyan hain, aur dono ka ilaaj alag hai:
- **A: artifact ya ensemble ka fark.** Shipped `ranker_v1.npz` production ensemble ke features pe theek se kaam nahi
  karta. Wo held-out ensembles (`big` 4-seed, `bigord` 2-seed) pe seekha tha. Ye theek kiya ja sakta hai.
- **B: held-out aur hidden ka fark.** Ranker sahi chalta hai, lekin held-out queries hidden test jaisi nahi hain. Is
  case me held-out pe koi bhi seekhi hui ranking LB ka andaza nahi deti.

**Test (local, free).** `splits.npz` production ensemble ka apna held-out split hai: production ensemble ne iski rows
nahi dekhi. Isme 600 structures hain, har class ke 200.

*Sudhaar, review ke baad:* pehle yahan "1,958 queries, 45 overlap" likha tha. Wo spectra ki ginti thi. Structures
ke hisaab se 14 ranker ke training splits me bhi hain. Unhe hata do.

Bachi queries pe shipped `ranker_v1.npz` ko production ensemble ke features ke saath chalao. Phir use experiment 4 ke
out-of-fold numbers se milao, ek jaise readings pe.

*Sudhaar:* pehle yahan "bilkul kernel ki tarah" likha tha. Ye poora sach nahi hai. Table eval protocol se banti hai:
`C2_BANNED=1`, `TRUE_ADDUCT=1` (adduct answer ke mass se chuna jaata hai), aur banned index structures. Kernel inme
se kuch nahi karta. Training tables bhi isi protocol se bani thi, to ye test sirf split aur ensemble ka fark alag
karta hai. Kernel ke protocol ka fark isme nahi aata.

Main reading hai *class 3 in-list*: class 3 ki wo queries jinka answer v17 ke top 100 me hai. Out-of-fold
reference: `big` covered +0.165 / ceiling 0.93 ≈ **+0.177**. Class 1 ka reference +0.013 hai.

**Pre-registered rule.**
- splits.npz pe class 3 in-list gain ≥ reference ka aadha (≥ +0.09): artifact theek hai, aur kahani **B** hai.
- Gain +0.09 se kam ho, ya negative: kahani **A**. Production features pe ranker toot-ta hai, aur feature shift
  table batayegi kahan.
- Class 1 alag se padhna hai, kyunki LB ka ulta-solve class 1 ke ~0.05 girne ko bhi allow karta hai (entry 6).

**Claude ka andaza.** Kahani B. Production ensemble aur `big` ensemble ek hi config ke 4 seeds hain, sirf alag rows
chhodi gayi hain, to `fp` ke features ek jaise hone chahiye. Class 3 in-list **+0.13** (+0.09 se +0.17), aur class 1
+0.01 ± 0.01.

**Andaza (user, reading se pehle).** Class 3 in-list **+0.113**, class 1 negligible. Ye bhi kahani B hai (≥ +0.09),
lekin Claude se thoda neeche: reference (+0.177) ka 64%, jabki Claude ka andaza 73% tha.

**Power note (table banne ke baad, reading se pehle).** `build_jobs` ki unit structure hai, spectrum nahi.
`splits.npz` me har class ke 200 structures hain, aur class 3 me sirf **16** ka answer v17 ke top 100 me hai.
(Uska class 3 coverage 8% hai, jabki `big` ka class 3 covered hi chuna gaya tha.)

Experiment 4 me covered class 3 ka interval 1,000 queries pe ±0.0165 tha. Usse per-query delta ka SD ~0.27 nikalta
hai. n = 16 pe 95% interval lagbhag **±0.14** hoga. Isliye class 3 in-list reading sirf kinaron pe kuch kehti hai:
- Upper bound +0.09 se neeche (measured ≲ −0.05): kahani A.
- Lower bound +0.09 se upar (measured ≳ +0.23): kahani B.
- Beech me: reading kuch tay nahi karti. Tab dono andaze, +0.113 aur +0.13, bhi test nahi hote.

**Class 1 (199 in-list)** asli faisla karne wali reading ban jaati hai. Interval ~±0.02 hoga. Entry 6 ka dusra
kinara, class 1 ka ~0.05 girna, isme saaf dikhega. Feature-shift table rows pe hai, isliye kam n se kam prabhavit
hoti hai.

**Measured** (`data/jobs/xfer_read.py`; 586 ansuni structures, 42,757 rows):

| reading | n | v17 | ranker | delta [95%] | out-of-fold reference |
|---|---|---|---|---|---|
| class 1 | 193 | 0.9380 | 0.8124 | **−0.1256** [−0.164, −0.088] | +0.0148 big, +0.0134 bigord |
| class 2 | 196 | 0.0444 | 0.0573 | +0.0128 [+0.003, +0.026] | +0.0132, +0.0155 |
| class 3 in-list | 15 | 0.6307 | 0.6991 | +0.0684 [+0.007, +0.148] | +0.1773 big, +0.1125 bigord |
| class 3 non-env | 54 | 0.1752 | 0.1942 | +0.0190 [+0.002, +0.040] | +0.0268 bigord |

**Andazon se fark.**
- Class 3 in-list beech me gira, aur power note ke hisaab se kuch tay nahi karta. Dono andaze (+0.113, +0.13) interval
  ke andar hain, isliye na sahi sabit hote hain na galat.
- Class 1 ne faisla kiya. Tumhara andaza "negligible" tha aur Claude ka +0.01. Asli −0.126 hai, interval zero se
  bahut door.

**LB se mel.** 0.140 × (−0.126) + 0.126 × (+0.013) = −0.016. LB −0.007 tha, to asli class 3 ~**+0.012** badha hoga:
(−0.007 + 0.016) / 0.735. Ye coverage-scaled range me baithta hai: 0.13 × 0.068 ≈ +0.009 se 0.12 × 0.177 ≈ +0.021
tak. Ek kahani teeno numbers samjha deti hai: class 3 ka gain transfer hua, aur class 1 ke collapse ne use kha liya.
Ye tabhi sach hai jab hidden class 1 `splits.npz` ki class 1 jaisi ho. Ye maana gaya hai, naapa nahi.

**Na A, na B: query ka shift, fp ka nahi** (`xfer_c1.py`, `xfer_qmap.py`).
- *fp features nahi khiske.* Class 1 answers ka `fp` 0.347 / 0.333 / 0.330 hai, aur `fp_z` 2.9 / 3.2 / 3.2
  (splits, big, bigord). Ek-ek feature ko big ke marginal pe quantile-map karke dekha, 14 features pe. Koi bhi class 1
  ko 0.82 se upar nahi le gaya. To shipped artifact ya ensemble toota nahi hai. Kahani A galat hai.
- *Queries khiski.* `splits.npz` ki class 1 me 61% queries ka top_spec ≥ 0.9 hai, jabki big aur bigord me 12.5%.
  Wahan answer ka `spec` median 0.93 hai, aur training me 0.77.
- *Kya palta.* 36 queries ne rank 1 khoya. Har baar naya top 1 ek external candidate tha: spec 0, `internal` 0, v17
  20 vs 28. 36 me se **34** me sirf `internal` badalne se answer wapas upar aa jaata hai. Ranker external candidates
  ko inaam deta hai, kyunki class 3 ka answer hamesha external hota hai. Class 3 training weight ka 73.5% hai, aur
  training me top_spec ≥ 0.9 wali queries bhi weight ke hisaab se zyadatar class 3 ki thi
  (0.735 × 4.5–7.3% vs 0.140 × 12.5%). To ranker ne seekha: "strong library match ho tab bhi external pe daao lagao".
  Jab class 1 me strong matches 5 guna zyada hon, ye rule class 1 ko tod deta hai.

Ye kahani B ka ek naapa ja sakne wala roop hai. Held-out splits ki class 1 hidden test jaisi nahi thi, aur iska asar
local test me dikh gaya. LB tak jaane ki zaroorat nahi padi.

**Belief jo badli.**
1. Out-of-fold CV ne class 1 pe +0.015 kaha, kyunki CV ki class 1 bhi wahi khaas class 1 thi. Ek alag split
   (`splits.npz`) ne jhooth pakad liya. Agle har learned component ko kam se kam ek aise split pe padhna hai jo training
   splits se alag tarah bana ho.
2. Class ke hisaab se weights (MIX) ek query-type feature (`internal`) ke saath milke "kaun si class hai" ka andaza
   training ke mix se seekh lete hain. Jab test me wo mix badle, ranker ka andaza galat ho jaata hai.

**Agla (local, free).** v17 khud prior ko top_spec < 0.8 pe gate karta hai. Ranker ko bhi sirf top_spec < 0.8 pe
chalao, aur baaki ko v17 ke order pe chhod do. Ye isi table pe naapa ja sakta hai. Lekin isse pehle andaza likhna hoga.

**Sudhaar, final review ke baad: reference.** `splits.npz` ek ordinary split hai, `big` jaisa covered split nahi.
Iska like-for-like reference bigord ka in-list gain **+0.1125** hai, `big` ka +0.177 nahi. Us hisaab se shart
(aadha) +0.056 hoti, aur measured +0.068 use paar karta. Lekin n = 15 pe interval [+0.007, +0.148] dono references
ko dhak leta hai, to nateeja wahi rehta hai: class 3 ki reading kuch tay nahi karti, aur faisla class 1 se hua.
Tumhara andaza +0.113 bilkul isi bigord reference ke barabar tha.

## 8. Kya top_spec < 0.8 pe gate karne se ranker class 1 ko chhodkar class 3 jeet leta hai? (2026-10-25)

**Sawaal.** Entry 7 me class 1 un queries pe toota jahan library match strong tha: khoye hue cases me top_spec 0.94
tha. v17 khud prior ko top_spec < 0.8 pe gate karta hai. Ab ranker ko bhi usi gate pe chalao. top_spec ≥ 0.8 wali
queries v17 ke order pe rahengi, aur baaki ranker ke order pe.

**Threshold pehle se tay hai.** 0.8 v17 ka `PRIOR_GATE` hai. Ise `splits.npz` pe sweep nahi kiya jaayega. Agar kiya,
to `splits.npz` test set nahi rahega, tuning set ban jaayega. Isiliye ek hi number chalega.

**Readings.** Wahi 586 ansune molecules aur wahi shipped `ranker_v1.npz` (`data/jobs/xfer_gate.py`). Saath me
out-of-fold reference bhi usi gate ke saath. Power entry 7 jaisi hai: class 1 n = 193 (±~0.04), class 3 in-list
n = 15 (±~0.07).

**Pre-registered rule.** Gated ranker tabhi LB probe (v19) ke laayak hai jab dono shartein poori hon:
- class 1 ka delta ≥ −0.01, aur uska interval −0.03 se neeche na jaaye;
- class 3 in-list delta > 0.

LB ka andaza: 0.140 × Δc1 + 0.126 × Δc2 + 0.735 × 0.13 × Δc3_inlist.

**Andaza (user, reading se pehle).** **+0.06**. Claude ise class 3 in-list ka andaza maan raha hai, kyunki class 1 ke
liye +0.06 ceiling ke paas (0.998) hota. Class 1 ka andaza nahi diya gaya.

**Claude ka andaza.**
- Class 1 **−0.01** (−0.03 se +0.01). Gate khoye hue strong-match cases wapas laata hai. Lekin kamzor match wali
  class 1 queries me bhi `internal` wali galti ho sakti hai. Out-of-fold me unka gain +0.007 tha.
- Class 3 in-list **+0.05**. Ungated +0.068 tha, aur gate kuch strong-match class 3 queries ka gain bhi le jaayega.
- Iska LB andaza ≈ −0.0014 + 0.0016 + 0.735 × 0.13 × 0.05 ≈ **+0.005**, jo ship rule ke bilkul kinare pe hai.

**Measured** (`data/jobs/xfer_gate.py`):

| reading | n | v17 pe rakhi gayi (top_spec ≥ 0.8) | ungated delta | **gated delta** [95%] | gated out-of-fold (big / bigord) |
|---|---|---|---|---|---|
| class 1 | 193 | 150 | −0.1256 | **−0.0187** [−0.035, −0.007] | +0.0052 / +0.0034 |
| class 2 | 196 | 18 | +0.0128 | +0.0128 [+0.003, +0.026] | +0.0127 / +0.0122 |
| class 3 in-list | 15 | 1 | +0.0684 | **+0.0684** [+0.007, +0.148] | +0.1398 / +0.1091 |
| class 3 non-env | 54 | 2 | +0.0190 | +0.0190 [+0.002, +0.040] | +0.1301 / +0.0260 |

**Andazon se fark.**
- Class 3 in-list: user ne +0.06 kaha, Claude ne +0.05, aur asli +0.068 aaya. Gate ne class 3 in-list ki 15 me se
  sirf 1 query ko chhua. Isliye class 3 lagbhag waise hi raha, aur dono andaze noise ke andar sahi hain.
- Class 1: Claude ka −0.01 (−0.03 se +0.01) tha. Asli −0.0187 range ke andar hai, lekin shart ke bahar.

**Rule ka faisla: gated ranker v19 probe ke laayak nahi.** Class 1 ka delta −0.0187 hai, jo −0.01 se neeche hai, aur
interval −0.035 tak jaata hai. LB formula 0.140 × (−0.0187) + 0.126 × 0.0128 + 0.735 × 0.13 × 0.0684 = +0.0055 deta
hai, jo ship rule ke theek kinare pe hai. Lekin uska class 3 hissa sirf 15 queries se aata hai: 95% range +0.001 se
+0.014 tak. Pehle se likhi shart class 1 ke nuksaan ko is noise ke saamne daao pe lagane se rokti hai. Wo shart
jaan-boojh ke likhi gayi thi.

**Naya: kamzor match wali class 1 bhi tootti hai.** 193 me se 150 queries gate se v17 pe chali gayi. Bachi 43 queries
pe poora nuksaan −0.0187 × 193 / 43 ≈ **−0.084 per query** hai. Out-of-fold me inhi queries pe gain tha:
+0.0052 × 400 / 203 ≈ +0.010. To gate sirf lakshan dabata hai. Bimari, ki ranker library candidates ko neecha dikhata
hai, kamzor matches me bhi maujood hai. Entry 7 ka `internal` flip yahan bhi wajah hai ya nahi, ye abhi naapa nahi
gaya.

`splits.npz` ki class 1 me 78% queries top_spec ≥ 0.8 pe hain. big aur bigord me ye ~49% hai.

**Belief jo badli.** Gate ek patch hai, ilaaj nahi. Asli ilaaj training me hai:
- ranker ko aisi class 1 dikhani hogi jo `splits.npz` jaisi ho;
- ya `internal` ke saath class ka andaza lagane ka raasta band karna hoga.
Jo bhi badlav ho, use `splits.npz` pe padhna hai, aur uska threshold `splits.npz` pe tune nahi karna hai.

## 9. Kamzor match wali class 1 me bhi kya `internal` hi top 1 chheenta hai? (2026-10-25)

**Sawaal.** Entry 8 me gate ke baad bhi 43 class 1 queries (top_spec < 0.8) pe ranker ne ~−0.084 per query
gawaya. Entry 7 me strong matches pe 36 me se 34 khoye hue top-1 (94%) sirf `internal` badalne se wapas aa gaye the.
Kya kamzor matches me bhi yahi wajah hai, ya wahan `fp` aur `prior` jaise dusre features hain?

**Test** (`data/jobs/xfer_weak.py`). Wahi 586 ansune molecules aur shipped `ranker_v1.npz`. Sirf class 1 ki
top_spec < 0.8 wali queries jahan v17 answer ko rank 1 pe rakhta tha aur ranker nahi. Har case me naye top-1 ko
answer ki ek feature value do, ek baar me ek feature. Gino ki kitne cases me sirf wahi badlav answer ko wapas upar
laata hai.

**Power note (reading se pehle).** 43 queries ka kul nuksaan −0.084 × 43 ≈ −3.6 RR hai. Ek khoya top-1 aam taur pe
0.5–0.9 RR ka hota hai, to ye ~5–8 cases honge. Itne n pe ek percentage ±30% tak hil sakta hai. Ye reading sirf
moti baat kahegi: `internal` zyadatar cases me hai (≥ 2/3), ya kuch hi me (≤ 1/3).

**Andaza (user, reading se pehle).** **73**. Claude ise 73% maan raha hai, kyunki cases 43 se zyada nahi ho sakte.

**Claude ka andaza.** **~60%**. `internal` ab bhi main wajah hai. Lekin kamzor match me answer ka `spec` kam hai, to
`fp`, `prior` ya `shift` jaise features bhi akele flip kar sakte hain. Isliye strong matches ke 94% se kam.

**Measured** (`data/jobs/xfer_weak.py`). 43 molecules me se 10 ne RR khoya aur 1 ne paaya. **5** ne rank 1 khoya, aur
kul −3.61 RR me se −2.83 inhi 5 se aaya. Swap test me **5 me se 5** sirf `internal` badalne se wapas aa gaye. Kisi
aur akele feature ne ek bhi case nahi palta.

Paanchon case me naya top 1 ek hi kism ka tha: external candidate, `spec` 0, v17 kam. Paanch me se chaar me uska
`prior` zyada tha, yaani wo jaana-maana COCONUT compound tha (3.1–3.6 vs 0 ya 1.4).

**Andazon se fark.** User ne 73% kaha, Claude ne ~60%, asli 100% (5/5). Power note ke bucket me "zyadatar cases me
(≥ 2/3)" aaya. User ka andaza usi bucket me tha, Claude ka beech wale me. n = 5 pe 100% aur 73% me fark noise hai.
Lekin bucket ka fark asli hai.

**Kul tasveer (entries 7–9).** Class 1 ke 41 khoye hue top-1 me se **39** sirf `internal` se palat-te hain, strong aur
kamzor dono match me. Ranker ne `internal = 0` ko "ye class 3 ka answer ho sakta hai" ka signal bana liya hai. Ye
signal itna bhari hai ki 0.3–0.9 ka library match aur v17 ke 3–11 points ki badhat bhi use nahi rok paati. Bimari ek
hi feature me hai, poore model me nahi.

**Agla qadam (abhi andaza, plan nahi).** Ranker v2 me `internal` ko class ka proxy banne se rokna hai. Teen raaste:
- (a) `internal` hata do. Lekin `has_spec` lagbhag wahi baat batata hai, to signal wahan chala ja sakta hai.
- (b) LightGBM ka monotone constraint: `internal` aur `has_spec` score ko kabhi neeche na le jaayein. Isse class 3 ka
  "library compound se bacho" wala signal bhi band ho sakta hai.
- (c) Class 1 ko `splits.npz` jaisi banaana. Iske liye training me ek aur split chahiye, taaki `splits.npz` test hi
  rahe.

Har raasta `splits.npz` ke 586 molecules pe, bina tuning ke, padha jaayega. Class 1 ki shart entry 8 wali hogi.

## 10. Kya ranker ranks 2+ pe, v17 ka rank 1 lock karke, class 1 bachaate hue class 3 jeet-ta hai? (2026-11-21)

**Sawaal.** Public kernels ka sabse pakka sabak (2026-11-21 review): jo seekhi hui cheez rank 1 hilaati hai wo LB pe
haarti hai, aur jo sirf ranks 2–25 ko chhooti hai ("leader-safe") wo tie karti hai ya thoda jeet-ti hai. Unka neural
scorer rank 1 pe w 0.5 pe −0.024 gira, aur sirf ranks 2–25 pe (w_mid 0.75) +0.004 utha. Entries 7–9 me class 1 ke
41 me se 39 khoye top-1 `internal` ki wajah se the. Kya rank 1 lock karna hi ranker v2 ka kaam kar deta hai?

**Design (pehle se tay, ek hi variant).** Har query me v17 ka rank 1 wahi rehta hai. Shipped `ranker_v1.npz` v17 ke
top 100 ke baaki 99 ko order karta hai. Koi top_spec gate nahi. Ranks 1–2 lock wala variant nahi chalega, taaki
`splits.npz` pe kuch tune na ho. Wahi 586 ansune molecules, aur out-of-fold reference bhi isi lock ke saath.

**Pre-registered rule** (entry 8 wala). Ye v19 LB probe ke laayak tabhi hai jab dono shartein poori hon:
- class 1 ka delta ≥ −0.01, aur uska interval −0.03 se neeche na jaaye;
- class 3 in-list delta > 0.

LB ka andaza: 0.140 × Δc1 + 0.126 × Δc2 + 0.735 × 0.13 × Δc3_inlist. Ship rule: ≥ +0.005.

**Pehle ke numbers jo andaze se pehle dikhaye gaye** (entries 7–9, kuch naya nahi padha):
- Ungated delta: class 1 −0.126, class 2 +0.013, class 3 in-list +0.068. Gated: class 1 −0.019, baaki wahi.
- Ungated class 1 ka nuksaan −24.2 RR tha. Strong match ki 150 queries ka −20.6 RR (36 khoye top-1) tha, aur weak
  match ki 43 queries ka −3.61 RR. Weak me se −2.83 RR 5 khoye top-1 se aaya aur −0.78 RR baaki ranks ke hilne se.
- v17 class 1 = 0.938, to rank 1 ke bahar sirf ~6% RR bachta hai.

**Andaza (user, reading se pehle).**
- Class 1 **+0.023**.
- Class 3 in-list ungated ke +0.068 se **neeche**.
- Class 2 lagbhag **zero**.
- Rule **pass** hoga.

**Claude ka andaza** (user ke baad likha, reading se pehle).
- **Class 1 −0.005** (−0.015 se +0.003). Lock 41 top-1 wapas laata hai. Lekin jahan v17 ka leader galat hai, wahan
  answer aksar library candidate hota hai, aur `internal` wali bimari use externals ke neeche dhakelegi. Weak hisse ka
  ranks 2+ wala −0.004 isi ka namoona hai. Gain mushkil hai: lock ke baad answer ka best rank 2 (0.5) hai.
  User ke +0.023 ke liye ~4.4 RR ka net gain chahiye, matlab ~9 answers ka rank ≥ 3 se rank 2 pe aana.
- **Class 2 +0.008.** Ungated ke +0.013 ka kuch hissa rank 1 se aaya hoga.
- **Class 3 in-list +0.03** (−0.02 se +0.08). User ki tarah neeche, kyunki answer ko rank 1 laana band hai.
- **Rule:** class 1 shart pass, class 3 > 0 shayad pass. Lekin LB andaza 0.140 × (−0.005) + 0.126 × 0.008 +
  0.0955 × 0.03 ≈ **+0.003** hai, jo ship rule (+0.005) se neeche hai.

**Measured** (`data/jobs/xfer_lock.py`). Har query me top 1 v17 jaisa hi raha; script ye assert karti hai. Ungated
column entries 7–8 se hubahu milta hai.

| reading | n | v17 | ungated delta | **locked delta** [95%] | up / down | RR up / down | locked out-of-fold (big / bigord) |
|---|---|---|---|---|---|---|---|
| class 1 | 193 | 0.9380 | −0.1256 | **−0.0070** [−0.015, −0.001] | 4 / 8 | +0.21 / −1.56 | −0.0014 / −0.0020 |
| class 2 | 196 | 0.0444 | +0.0128 | +0.0025 [+0.000, +0.007] | 3 / 0 | +0.50 / 0 | +0.0042 / +0.0067 |
| class 3 in-list | 15 | 0.6307 | +0.0684 | **+0.0197** [+0.002, +0.043] | 4 / 1 | +0.33 / −0.03 | +0.0442 / +0.0241 |
| class 3 non-env | 54 | 0.1752 | +0.0190 | +0.0055 [+0.000, +0.012] | 4 / 1 | +0.33 / −0.03 | — / +0.0057 |

**Rule ka faisla: dono shartein pass.** Class 1 −0.0070 hai (≥ −0.01), aur interval −0.015 tak hi jaata hai
(−0.03 se upar). Class 3 in-list +0.0197 hai (> 0).

**Lekin LB pe ye dikhega nahi.** 0.140 × (−0.0070) + 0.126 × 0.0025 + 0.0955 × 0.0197 = **+0.0012**. Ye ek molecule
(~0.008) ka saatvan hissa hai. Ship rule (+0.005) se bahut neeche, aur LB noise (±0.003–0.005) ke andar. Isliye v19
probe nahi: LB +0.001 aur 0 me fark nahi kar sakta.

**Andazon se fark.**
- Class 1: user +0.023, Claude −0.005 (−0.015 se +0.003), asli −0.0070. User ka sign ulta nikla. Lock ke baad gain ke
  liye answer ko rank ≥ 3 se rank 2 pe aana tha. Aise sirf 4 cases hue, kul +0.21 RR, jabki +0.023 ke liye ~4.4 RR
  chahiye tha. 8 cases neeche gaye (−1.56 RR).
- Class 3 in-list: dono ne "neeche" kaha (Claude +0.03), asli +0.0197. Dono sahi.
- Class 2: user "lagbhag zero" sahi. Claude ka +0.008 interval [+0.000, +0.007] se thoda bahar.
- Rule: user ne "pass" kaha, sahi. Claude ka LB andaza +0.003 tha, asli +0.001. Dono ship rule se neeche.

**Asli seekh: ranker ki class 3 jeet zyadatar rank 1 pe thi.**
- Lock ne class 3 in-list ka gain ~70% kaata: +0.068 → +0.020. Bade out-of-fold sets pe bhi yahi hua: big +0.177 →
  +0.044 (n = 930), bigord +0.113 → +0.024. Ye n = 15 ka noise nahi hai.
- Out-of-fold class 1 ka +0.015 bhi poora rank 1 se tha. Lock ke baad wo teeno sets pe thoda negative hai
  (−0.001 se −0.007, down cases up se ~2 guna). Ranks 2+ pe bhi ranker library answers ko thoda neeche karta hai.
  Ye `internal` wali kahani se mel khaata hai, lekin swap test se naapa nahi gaya.
- Matlab ranker ki value aur uska nuksaan ek hi jagah hain: rank 1 ke move. Field ke leader-safe layers isliye
  sirf tie ya +0.004 dete hain. Lock risk hata deta hai, lekin uske saath value ka ~3/4 hissa bhi chala jaata hai.

**Belief jo badli.** Rank 1 lock ranker v2 ka shortcut nahi hai. Wo safe hai, lekin LB pe kuch nahi deta. Ranker
ka faayda tabhi milega jab rank 1 ke move class 1 ke liye safe ho jaayein, yaani `internal` ka proxy band ho
(entry 9 ke raaste a–c). Aur tab bhi upper bound chhota hai: poora ungated class 3 gain bhi 0.0955 × 0.068 ≈
+0.0065 LB hai. Class 3 pe ranking ka har gain coverage (~0.13) se guna hota hai. Isliye agla lever coverage hai
(entry 11), ranking nahi.

## 11. Held-out class 3 ke kitne answers PubChem me hain, lekin hamare pool me nahi? (2026-11-21)

**Sawaal.** Public pipeline ka sabse bada naapa hua asar PubChem channel hai: PubChem-only candidates hatane pe
0.464 → 0.409. Wo PubChem ko pool me nahi milate, alag fixed slots pe rakhte hain. Channel ki pehli shart coverage
hai: kitne answers PubChem me hain jo hamare pool me nahi hain.

Yaad rahe: official class 3 PubChem me hota hi nahi (HANDOFF). Hamari "class 3" = official class 2 + 3, to PubChem
sirf official class 2 wala hissa jeet sakta hai.

**Andaza (user, 2026-11-21, kuch bhi padhne se pehle).** **30–37%** (beech 33.5%). Sawaal tha: "class 3 held-out ke
kitne % answers PubChem se aa sakte hain jo hamare pool me nahi hain?"

**Reading kaise hogi** (andaza likhne ke baad tay ki gayi; user badal sakta hai):
- Denominator: `splits_bigord.npz` ki class 3 ke 302 non-`enveda-180` answers (RESULTS §31 wala set).
  `enveda-180` screening compounds hain, test jaise nahi. Inme se 76 pehle se COCONUT me hain, to ceiling
  226 / 302 = 75% hai.
- Numerator: answer ka key14 PubChem me hai, aur v17 ke pool me nahi hai.
- Do readings: `ext_pubpop2` (MIN_SYN ≥ 2, 15.96M, jo Kaggle pe chal sakta hai) aur poora PubChem (`CID-SMILES.gz`).

HANDOFF me ek milta-julta number pehle se hai: `splits_bigord.npz` ki **saari** class 3 pe pubpop2 ne coverage
7.6% → 69% kiya, lekin uska zyadatar hissa `enveda-180` tha. User ka andaza ye number yaad dilaane se pehle diya gaya.

**Measured** (`data/jobs/pubcov.py`, 26 s). Pehle ek check: saari 1000 class 3 pe COCONUT 7.6%, pop2 68.9% aur dono
milke 69.4% aaye. Ye §31 se hubahu milta hai.

| class 3 | n | COCONUT (v17 pool) | pop2 | **pop2, COCONUT me nahi** | NP-PubChem, COCONUT me nahi | dono milke |
|---|---|---|---|---|---|---|
| non-`enveda-180` | 302 | 76 (25.2%) | 207 (68.5%) | **136 (45.0%)**, 95% ~[39%, 51%] | 31 (10.3%) | 212 (70.2%) |
| saari (check) | 1000 | 76 (7.6%) | 689 (68.9%) | 618 (61.8%) | 32 (3.2%) | 694 (69.4%) |

pop2-only answers ke synonyms:
- non-env (136): 2 synonym wale 18, 3–4 wale 32, 5–9 wale 47, 10+ wale 39. Median **6**.
- saari (618): 2 synonym wale 399. Median **2**, yaani zyadatar `enveda-180` ke vendor-ID wale Enamine compounds.

**Andaze se fark.** User ka andaza 30–37% tha, asli **45.0%** [39%, 51%]. Andaza interval se neeche raha, yaani
held-out coverage soch se zyada nikli. Claude ne B ka andaza nahi diya tha.

Poore PubChem (`CID-SMILES.gz`) wali dusri reading nahi chali. Local pe uske key14 nahi hain, aur ~120M SMILES ko
parse karne me lagbhag ek ghanta lagta. Purana REST probe (`pubchem.py`) `splits.npz` ke 200 class 3 skeletons me
se 88.7% PubChem me paata hai.

**Ye number LB ke liye kya kehta hai, aur kya nahi.**
- Held-out non-env answers public libraries ke documented compounds hain (median 6 synonyms). Hidden test pe ye number
  upper bound hai. Hidden pe COCONUT ki coverage ~12% (§9) hai, jabki yahan 25%. Agar pop2 bhi usi anupaat (~0.5) me
  gire, to hidden pe pop2-only coverage ~20–25% hogi. Ye andaza hai, naapa nahi.
- Field ka ablation (PubChem-only hataane pe −0.055) class 3 pe coverage × MRR ≈ 0.055 / 0.735 ≈ 0.075 maangta hai.
  Misaal ke taur pe 22% coverage × 0.34 MRR. Matlab unke channel me aaya answer aksar top 3 me hota hoga.
- v16 ne dikhaya ki coverage akele kuch nahi deti: pool me milaane se dilution ne gain kha liya (§31). Channel ka asli
  sawaal ab ye hai ki fixed slots pe answer kis rank pe aata hai, aur main list ke kitne answers ek slot neeche
  khisakte hain.

**Belief jo badli.** Held-out pe coverage bottleneck nahi hai. 302 me se 136 answers pop2 me hain jo hamare pool me
nahi. Rukawat channel ke andar ki ranking aur main list ka nuksaan hai. Agla qadam channel simulation hai, aur uske
liye naye andaze chahiye.

## 12. PubChem channel fixed slots pe: answer kitna upar aata hai, aur main list kitna khoti hai? (2026-11-21)

**Sawaal.** Entry 11: non-env class 3 ke 45% answers pop2 me hain aur COCONUT me nahi. v16 ne inhe pool me milaaya
tha aur dilution ne gain kha liya. Field ka tareeka alag hai: PubChem ki ek alag list, jiske proposals main list ke
fixed slots pe baithte hain. Is tarah PubChem candidate main list ke candidates se mukaabla nahi karta.

**Field ka rule** (Gluzdov ke notebook ka code, 2026-11-21 padha):
- Strong library match (`lib_max ≥ 0.9`) ya PubChem proposal na ho: list waise hi rehti hai.
- Warna rel = PubChem leader ka fingerprint score − main pool ka sabse acha score.
  - rel > REL_TH (600, unki units me): aggressive slots 2, 4, 6, 8, 10, 24.
  - warna gentle slots 4, 8, 12, 16, 20, 24.
- Rank 1 promotion (S > 6 aur pop ≥ 5, ya rel > 150) is entry me nahi hai.

**Design (pehle se tay).**
- `splits_bigord.npz`, v17 settings, `C2_BANNED=1`, `TRUE_ADDUCT=1`, bigord ensemble (s0, s1).
- **Main list:** v17 ka order, purani COCONUT table (`splits_bigord_dcd7dba4683f.npz`) se.
- **Channel:** ek nayi table jiska pool COCONUT + pop2 hai (v16 jaisa). Channel me wahi candidates jo na index
  structure hain (`internal` 0) aur na COCONUT me (key14). Unka order us table ke v17 total se, aur top 6 liye
  jaayenge. Ye candidates main list me ho hi nahi sakte, isliye duplicate nahi banenge.
- **Gate:** top_spec ≥ 0.9 pe list waise hi rehti hai (field ka LIB_TAU).
- **Slots, primary reading:** rel = channel leader ka `fp` − main top 25 ka sabse bada `fp`. Dono same scale pe hain.
  rel > 0 pe aggressive slots, warna gentle. 0 hamara apna, bina tune kiya kinara hai, kyunki field ka 600 hamari
  units me translate nahi hota. Saath me "hamesha gentle" aur "hamesha aggressive" bhi padhe jaayenge, sirf seemaon ki
  tarah. Inme se koi chunna nahi hai.
- **Readings:**
  - class 1 (400), class 2 (400) aur class 3 non-env (302) ka delta, paired bootstrap;
  - entry 11 ke 136 covered answers me se kitne channel ke top 6 me aate hain.

**LB andaza (pre-registered).** 0.140 × Δc1 + [0 … 0.126] × Δc2 + 0.735 × 0.5 × Δc3_nonenv.
- Class 2 ka term ek range hai. Plain eval me class 2 ka answer pehle se index candidate hai, to channel use kuch nahi
  deta. `C2_BANNED` eval poora gain deta hai. Hidden sach in dono ke beech hai (HANDOFF).
- 0.5 hidden aur held-out coverage ka anupaat hai: hidden pe COCONUT ~12% (§9), yahan 25%. Ye moota andaza hai.
- Ship rule: ≥ +0.005. Lekin §31 ke baad held-out pe jeeti pool cheez LB slot ka sabse kamzor saboot hai, isliye
  koi bhi LB probe ek learning probe hoga.

**Andaza (user, design tay hone se pehle).**
1. Class 3 non-env: **0.23**. Reading se pehle user ne saaf kiya ki ye MRR ka level hai, Δ nahi. v17 isi split pe
   0.1507 hai (entry 10 ka bigord reference), to andaza **Δ +0.079** hai.
2. Class 1: **−0.002**.
3. 136 me se **67%** (~91) channel ke top 6 me.

**Claude ka andaza** (user ke baad, reading se pehle).
- **Top 6 me ~50%** (40–60%, ~68 answers). §31 me merged pool pe pop2-only answers ka MRR ~0.30 tha
  (0.18 × 226 / 136). Channel me COCONUT aur index se mukaabla nahi hota, isliye unka rank behtar hoga.
- **Class 3 non-env Δ +0.05** (+0.03 se +0.08). Lagbhag aadhe hits channel ka #1 honge (slot 2 ya 4), baaki slots
  4–24 pe. COCONUT wale 76 answers ko slots thoda neeche dhakelenge, ~−0.005.
- **Class 1 Δ −0.004** (−0.010 se 0). Rank 1 locked hai. Nuksaan sirf un answers ka hai jo rank 2+ pe hain aur jahan
  top_spec < 0.9 hai.
- **Class 2 Δ +0.08** (`C2_BANNED`). §31 me merged pop2 ne class 2 ko 0.028 → 0.27 kiya tha. Slots ise kam karenge.
- **LB:** 0.140 × (−0.004) + [0 … 0.126 × 0.08] + 0.735 × 0.5 × 0.05 ≈ **+0.018 … +0.028**. Ye v16 ke held-out
  andaze jaisa hi bada number hai, aur v16 LB pe −0.014 gira tha. Isliye is andaze pe bharosa kam hai.

**Measured** (`data/jobs/channel_read.py`; channel table `pop2_table.py`).
- Channel table: 1800 queries, 175,242 rows, queries same kram me (asserted).
- Pehla build mere bug se gira. `spawn` workers ne main guard ke bina script dobara chalayi, aur memory guard ne
  parent ko maar diya. Guard ke saath dusra build saaf chala.
- Reading 20 s chali.

Branches (1800 queries): aggressive 1035, gentle 615, untouched 148, no channel 2.

| reading | n | v17 | **primary Δ** [95%] | up / down | hamesha gentle | hamesha aggressive |
|---|---|---|---|---|---|---|
| class 1 | 400 | 0.9232 | **−0.0035** [−0.006, −0.002] | 0 / 13 (−1.41 RR) | −0.0006 | −0.0135 |
| class 2 (`C2_BANNED`) | 400 | 0.0348 | +0.1699 [+0.149, +0.191] | 186 / 10 | +0.0959 | +0.1906 |
| **class 3 non-env** | 302 | 0.1507 | **+0.0878** [+0.069, +0.110] → 0.2385 | 82 / 15 (−1.06 RR) | +0.0515 | +0.0906 |
| class 3 saari (`enveda-180` samet) | 1000 | 0.0455 | +0.1708 | 459 / 15 | +0.0950 | +0.1860 |

Channel ke top 6 me answer: 136 covered me se **89 (65.4%)**. Channel ke rank ke hisaab se: #1 pe 55, #2 pe 11,
#3 pe 6, #4 pe 6, #5 pe 7, #6 pe 4. Table ki channel rows me kahin bhi: 120.

**Andazon se fark.**

| | user | Claude | asli |
|---|---|---|---|
| class 3 non-env | level 0.23 (Δ +0.079) | Δ +0.05 (+0.03 se +0.08) | **0.2385 (Δ +0.088)**, user ka andaza interval ke andar |
| class 1 | −0.002 | −0.004 (−0.010 se 0) | −0.0035, dono theek |
| top 6 me | 67% | ~50% (40–60%) | **65.4%**, user lagbhag hubahu |
| class 2 | — | +0.08 | +0.170 |

User ke dono bade andaze sahi nikle, Claude ke dono kam pade. Claude ki galti: hits ka 62% (89 me se 55) channel ka
#1 hai. Channel me answer sirf doosre pop2 decoys se ladta hai, COCONUT aur index se nahi. §31 ke merged-pool MRR
se andaza lagana isi wajah se neeche gaya.

**LB andaza (pre-registered formula).** 0.140 × (−0.0035) + [0 … 0.126 × 0.170] + 0.735 × 0.5 × 0.0878 =
**+0.032 … +0.053**. Ship rule (+0.005) bahut aaram se paar hota hai. Agar hidden coverage held-out ka sirf chauthai
ho (0.25), aur class 2 ka term 0 ho, tab bhi +0.016 banta hai.

**Ise kitna maanein.** §31 ne sikhaaya ki pool expansion held-out pe jeet ke LB pe haarti hai. Is baar teen
cheezein alag hain:
1. *Dilution ka raasta band hai.* Main list ka order nahi badalta, sirf slots jitna khisakta hai. Ye nuksaan
   naapa gaya: class 1 −0.0035, class 3 non-env ke 15 down cases −1.06 RR. v16 me hidden dilution held-out se ~6 guna
   tha, kyunki 16M decoys answer se seedha ladte the. Yahan wo ladai channel ke andar band hai.
2. *Field ka alag saboot.* Unka ablation (PubChem-only hataane pe −0.055) aur hamara +0.032 … +0.053 ek hi dayre me
   hain. Do alag pipeline, do alag test (LB ablation aur held-out), ek hi jawab.
3. *Risk jo baaki hai.* Held-out answers documented hain (median 6 synonyms). Hidden pe pop2 ki coverage kitni hai,
   ye naapa nahi gaya. 0.5 ek moota andaza hai. Class 2 ka term `C2_BANNED` ki wajah se upar ki taraf jhuka hai.

**Belief jo badli.** Pool expansion band nahi hai, bas galat tareeke se ki gayi thi. Coverage asli hai (45%), aur
channel ke andar ranking achhi hai (65% top 6 me, 40% channel ka #1). v16 ka nuksaan coverage ka nahi, merge ka
tha. Ye v17 ke baad pehli cheez hai jiska LB andaza noise se kaafi upar hai. Agla qadam v19 kernel hai: v17 + ye
channel, bilkul isi rule ke saath (gate 0.9, rel > 0, slots field wale, koi promotion nahi). Push sirf user ki haan ke
baad.

## 13. Kya PubChem channel LB pe bhi utna hi deta hai? (v19, 2026-11-21)

**Sawaal.** Entry 12 ka LB andaza +0.032 … +0.053 tha. v19 = v17 + channel, bilkul entry 12 ke rule ke saath:
gate 0.9, rel > 0 pe aggressive slots, warna gentle, rank 1 promotion nahi. Kernel me main list ka code v17 ka hi
hai. Channel off pe local file v17 ki tested file se byte-for-byte milti hai, aur channel ka pool COCONUT + pop2 hai.

**Pre-registered padhna** (v17 = 0.264; LB noise ±0.003–0.005; ek molecule ≈ 0.008):
- **v19 ≥ 0.274** (+0.010 ya zyada): channel hidden pe bhi kaam karta hai. v19 final selection me v17 ki jagah lega.
- **0.259 … 0.274:** saboot nahi. Hidden pe pop2 ki coverage held-out se bahut kam hai, ya gain noise me chhup gaya.
  Final selection v17 hi rahegi.
- **v19 < 0.259:** channel nuksaan karta hai. Slots ne hidden ke main-list answers ko neeche dhakela. Entry 12 me ye
  nuksaan ~−0.002 tha (class 1 0.140 × −0.0035, class 3 down cases 0.735 × 0.5 × −0.0035), to ye uska ~3 guna hoga.

Back-solve: Δ = 0.140 × Δc1 + 0.126 × Δc2 + 0.735 × Δc3. Δc1 ≈ −0.0035 aur Δc2 ∈ [0, +0.17] maan ke, hidden class 3
ka gain = (Δ − 0.140 × (−0.0035) − Δc2_term) / 0.735.

**Andaza (user, push se pehle).** **0.284** (Δ +0.020).

**Claude ka andaza** (user ke baad, push se pehle). **0.289** (0.264 se 0.314), yaani Δ +0.025. Entry 12 ka formula
+0.032 … +0.053 deta hai. Usse neeche rakhne ki do wajahein hain:
- pool wali cheezon ka held-out andaza LB pe hamesha zyada nikla hai (v16, ChEMBL);
- field ke −0.055 me rank 1 promotion aur behtar fingerprint models bhi shaamil the, jo hamare channel me nahi hain.

Neeche ka kinara (0.264) asli sambhavna hai: agar hidden ke official class 2 answers pop2 me kam hain.

**Local checks (push se pehle).**
- Channel off: `kernel/code` se bani file v17 ki tested file se byte-for-byte milti hai (`kcode_v19.sh`).
- Channel on, placeholder test: 381 me se 381 untouched, kyunki ye molecules train se hain. 19 lists badli, aur
  `chan_check.py` pass hui.
- Stress run, gate band karke saare 400 pe: rank 1 aur v17 ka kram har list me bacha, naye candidates sirf slots
  pe. 8 min chala, jisme channel ka hissa ~1–2 min tha.

**Nateeja (v19, submission 58670384).** **Score nahi aaya: hidden rerun crash hua** ("Notebook Threw Exception";
API ka `error_description`: "Your notebook hit an unhandled error while rerunning your code"). Rerun ka log Kaggle
nahi dikhata, isliye cause sabooton se dhoondha:
- v17 isi hidden data pe chala tha, to crash v19 ke naye, data pe nirbhar code me tha. Wahan ek hi aisi jagah hai
  jo hidden pe alag chal sakti hai: `submit` ka `assert cands[:1] == lead`.
- v17 ki list **khaali** hoti hai jab precursor window me na koi library spectrum ho, na index/COCONUT structure.
  Tab v17 `"C"` likhta hai. Agar pop2 me usi mass pe structures hon, to `slot_merge([], ch, …)` channel ka item
  rank 1 pe rakhta hai, aur assert `[] != ['X']` pe fail hota hai.
- Held-out (splits_bigord) me 1800 me se **3** molecules aise the: sab class 3, sab non-env. Hidden ke ~294
  class 3 pe kam se kam ek aane ki sambhavna ≈ 1 − 0.997^294 ≈ 59%.
- **Reproduce hua:** `data/jobs/chan_repro.py` asli `fpsearch.submit` chalata hai, ek aise molecule ke saath jo
  414.959 m/z pe hai (main pool 0, library rows 0, pop2 pool 85) →
  `AssertionError: empty_window: the channel moved rank 1`.

Placeholder test ke saare molecules train se hain, isliye unki list kabhi khaali nahi hoti. Gate band karke stress
run me bhi ye case nahi aaya.

**Fix (v19b).** `assert not lead or cands[:1] == lead`. Khaali list me bachane ko rank 1 hai hi nahi, to channel
use bharta hai, jahan v17 ka `"C"` pakka 0 score deta tha. Saath me `slot_merge([], …)` ka module-level assert.
Verification:
- channel off pe v17 ki tested file se byte-for-byte same;
- channel on, placeholder pe `chan_check.py` pass (381 / 12 / 7, pehle jaisa);
- repro ab chalta hai, aur `empty_window` ko pop2 ke 6 candidates milte hain.

Upar ke pre-registered bands aur dono andaze v19b pe waise hi laagu hain. Fix sirf khaali-list molecules ko badalta
hai: hidden pe ~1 molecule, jahan v17 ka score 0 tha.

**Sabak.** Hidden rerun me per-molecule `assert` ek molecule ki gadbad ko poore 400 ke score ki maut bana deta hai.
Jo code sirf unseen structures pe chalta hai (khaali list, gate < 0.9), use placeholder test kabhi nahi chalata.
Push se pehle ek synthetic "khaali window" molecule wala repro chalana chahiye.
