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
