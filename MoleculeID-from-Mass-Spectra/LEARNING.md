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
