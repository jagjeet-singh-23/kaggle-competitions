# Mass-shifted analog propagation (Raasta B, component 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Har query molecule ke liye ±200 Da ke andar library ke "rishtedaar" dhoondhna (fragments seedhe ya Δmass se shifted match), aur pool candidates ko `max_a sim(a)^p × Tanimoto(c, a)` se score karna. Saath me learning cycle ke do experiments chalana.

**Architecture:** Naya module `shifted.py` me teen cheezein hain: similarity (entropy aur cosine, direct ya shifted), ek representative-spectrum index (`ShiftIndex`), aur candidate term (`analog_term`). Similarity numpy me vectorised hai: query peaks 0.01 Da bin axis pe scatter hote hain, aur har reference ke bins `+ shift` pe gather hote hain. Ye `Searcher.score` wala hi tareeka hai. `dilute.py` me naya term env knobs ke peeche aata hai, aur analog lists ek baar compute hoke disk pe cache hoti hain. `fpsearch.submit` me wahi term aata hai, aur default weight 0 rehne tak shipped output nahi badalta.

**Tech Stack:** Python 3.12, numpy ≥ 2 (`np.bitwise_count`). Koi nayi dependency nahi: numba, pytest aur lightgbm venv me nahi hain, aur is component ko unki zaroorat nahi. Tests repo ke pattern me module ke `selftest()` me hain (`python3 shifted.py --selftest`).

**Spec:** `docs/superpowers/specs/2026-10-20-raasta-b-design.md` (component 1, learning cycle, constraints)

## Global Constraints

- Har heavy run `MIN_AVAIL_MB=2500 ../guard.sh` ke neeche aur `caffeinate -ims` ke saath. Job scripts `data/jobs/` me. `guard.sh` kabhi weaken nahi karna.
- Har eval `C2_BANNED=1` ke saath aur uske bina. Class 3 non-`enveda-180` answers pe padhna.
- RDKit 2026.3.3 pinned.
- Commit sirf user ki haan ke baad, files path se stage karke (`git add -A` kabhi nahi). Models, logs, caches aur competition data git me nahi.
- Kaggle push se pehle prediction `LEARNING.md` me likhni hai aur user se confirm karna hai.
- Experiment se pehle user ka andaza lena hai (design: "pehle andaza, phir data"). Ye steps sirf is session me chal sakte hain.
- Default behaviour nahi badalna: `SHIFT_W`/`SHIFT_WS` ka default 0 hai, aur us par output purane jaisa hi rehna chahiye.

## Review Focus

1. **Window me koi reference nahi:** `search` khaali list lautaye, aur term saare candidates ko 0 de. Crash nahi hona chahiye. Task 2 me test.
2. **Trim ke baad query spectrum me 0 peaks (kernel ke test spectra):** woh spectrum skip ho. Task 2 me test.
3. **Negative-mode query:** sirf negative representatives search hon. Task 2 me test.
4. **Structure jiska mass `structures.npz` me −1 hai (unparseable) ya key missing hai:** index se bahar rahe. Task 2 me test.
5. **Purana cache naye settings ke saath padh lena:** cache path settings, split file aur index size sab pe depend kare. Task 3 me test.

---

### Task 1: Similarity kernels

**Files:**
- Create: `shifted.py` (module docstring repo ki style me: kyun, aur RESULTS/design ka reference)

**Interfaces:**
- Produces:
  - `TOL_DA = 0.01`
  - `pair_sim(q_mz, q_it, r_mz, r_it, shift=0.0, kind="entropy") -> float`. Ye ek pair ki reference implementation hai. Intensities sqrt-scaled aati hain, jaise `library.trim` aur `spectra_of` deti hain. Peaks tab match hote hain jab `rint(mz / TOL_DA)` barabar ho, aur reference ke m/z pe `shift` pehle joda jaata hai. Return `max(direct, shifted)` hai, aur `shift == 0` pe sirf direct.
  - `entropy_p(it_sqrt: np.ndarray) -> np.ndarray`, row-wise (1-D ya 2-D). Steps: `it²` → sum 1 pe normalise → agar spectral entropy `S < 3` ho to `p ** (0.25 + 0.25·S)` aur dobara normalise. Padding zeros zero hi rehte hain.

- [ ] **Step 1: Failing selftest likho** (`shifted.py` ka `selftest()`, `--selftest` flag)

```python
# hand-computed: q = {100: .5, 200: .5}, r = {100: .5, 300: .5} (intensities given as sqrt)
s = np.sqrt
assert abs(pair_sim([100, 200], s([.5, .5]), [100, 300], s([.5, .5])) - 0.5) < 1e-6
assert abs(pair_sim([100, 200], s([.5, .5]), [100, 300], s([.5, .5]), kind="cos") - 0.5) < 1e-6
assert abs(pair_sim([100, 200, 250], [1, .7, .3], [100, 200, 250], [1, .7, .3]) - 1.0) < 1e-6
assert pair_sim([100, 200], [1, 1], [400, 500], [1, 1]) == 0.0
# relative +CH2 (14.01565): every reference fragment carries the modification
q = ([114.0157, 214.0157, 300.0], [1, .8, .5]); r = ([100.0, 200.0, 285.9843], [1, .8, .5])
assert pair_sim(*q, *r) < 0.2                      # direct: nothing lines up
assert pair_sim(*q, *r, shift=14.01565) > 0.99     # shifted by the neutral-mass difference
assert pair_sim([], [], [100], [1]) == 0.0
p = entropy_p(np.array([[1.0, 1.0, 0.0]])); assert np.allclose(p, [[.5, .5, 0]])
```

- [ ] **Step 2: Chalao, fail hona chahiye**

Run: `../.venv/bin/python shifted.py --selftest`
Expected: `NameError: name 'pair_sim' is not defined` (ya ImportError)

- [ ] **Step 3: `pair_sim` aur `entropy_p` implement karo**

Entropy similarity ka closed form, jo matched pairs pe hi chalta hai (Task 2 ka vectorised version isi pe tika hai):

```
sim = Σ_matched [(p+q)·ln(p+q) − p·ln p − q·ln q] / ln 4      # p, q = entropy_p of each side
```

Cosine: L2-normalised sqrt intensities ka matched dot product.

- [ ] **Step 4: Chalao, pass hona chahiye**

Run: `../.venv/bin/python shifted.py --selftest`
Expected: `shifted self-checks passed`

---

### Task 2: `ShiftIndex`: representatives aur vectorised search

**Files:**
- Modify: `shifted.py`

**Interfaces:**
- Consumes: Task 1 ka `pair_sim`, `entropy_p` aur `TOL_DA`. `library.load()` ka dict (`mz` (N,64) f32, `it` (N,64) f16 sqrt, `npk`, `precursor`, `ion` bool, `inchikey14` S14, `lib` S24). `analog.DESC` (`key`, `mass`).
- Produces:
  - `WINDOW = 200.0`, `N_ANALOGS = 80`, `TIMS_LIBS = (b"enveda-180", b"enveda-np-examples")`
  - `class ShiftIndex: __init__(self, ix, drop=None, banned=None, masses=None)`
    - `drop` ki rows aur `banned` keys `Searcher.__init__` ki tarah hatati hain.
    - `masses` = `(keys S14, mass f64)`. Default `analog.DESC` se aata hai. Jis key ka mass < 0 ho ya jo missing ho, woh bahar.
    - Har `(key, polarity)` ka ek representative: pehle `TIMS_LIBS`, phir zyada `npk`, phir chhoti row.
    - Neutral mass se sorted, aur polarity ke hisaab se alag arrays.
  - `ShiftIndex.search(self, spectra, neutral, positive, window=WINDOW, n=N_ANALOGS, kind="entropy") -> list[tuple[bytes, float]]`
    - `spectra` = `spectra_of` ka format `[(mz, it_sqrt, precursor), ...]`.
    - Har reference ke liye `max` liya jaata hai: query spectra ke upar, aur direct/shifted ke upar. `shift = neutral − ref_neutral`.
    - Top `n` `(key, sim)` sim ke ghatte order me. Sim 0 wale bahar.

- [ ] **Step 1: Failing selftest likho.** Ek synthetic `ix` aur `masses`, jisme 6 structures hon:
  - A: query ka khud ka structure (row 0 = query, `drop=[0]`)
  - B: +14.01565 Da relative, timsTOF, shifted fragments
  - B': B ka hi non-timsTOF spectrum, zyada peaks ke saath
  - C: 250 Da door (window ke bahar)
  - D: negative mode
  - E: banned
  - F: mass −1

```python
si = ShiftIndex(ix, drop=[0], banned=[b"E"], masses=masses)
got = si.search(spectra_of(ix, [0]), neutral=mA, positive=True)
keys = [k for k, _ in got]
assert keys[0] == b"B" and got[0][1] > 0.99          # shifted relative found, ranked first
assert b"A" not in keys                               # dropped query row never returns itself
assert b"E" not in keys and b"F" not in keys          # banned / unparseable excluded
assert b"C" not in keys                               # outside ±200 Da
assert b"D" not in keys                               # wrong polarity
assert si.rep_lib_of(b"B") == b"enveda-180"          # timsTOF representative preferred
# vectorised path agrees with the reference implementation
for k, v in got:
    assert abs(v - best_pair_sim(k)) < 1e-4           # helper inside selftest using pair_sim
assert si.search([(np.zeros(0, np.float32), np.zeros(0, np.float32), 300.0)], mA, True) == []
assert si.search(spectra_of(ix, [0]), neutral=5000.0, positive=True) == []
```

(`rep_lib_of(key) -> bytes` ek chhota inspection helper hai, isi task me define hota hai.)

- [ ] **Step 2: Chalao, fail hona chahiye**

Run: `../.venv/bin/python shifted.py --selftest`
Expected: `NameError: name 'ShiftIndex' is not defined`

- [ ] **Step 3: `ShiftIndex` implement karo.**
  - Representatives ke bins `(R, 64) int32` me hon, padding −1. Saath me `entropy_p` rows aur L2-normalised cosine rows, float32.
  - Query ko `search.BINS + 2` (200,002) size ke bin axis pe scatter karo. Searcher ki tarah sirf jo likha usi ko clear karna hai.
  - Shift ko bins me `rint(shift / TOL_DA)` banao, aur har reference row ke bins pe add karke gather karo. Jo shifted bins `[0, BINS)` se bahar jaayein, woh kuch match nahi karte (−1 ki tarah).
  - Gather 20k-row chunks me ho, taaki memory 200 MB ke andar rahe.

- [ ] **Step 4: Chalao, pass hona chahiye**

Run: `../.venv/bin/python shifted.py --selftest`
Expected: `shifted self-checks passed`

- [ ] **Step 5: Asli index pe speed check**

Run: ek `data/jobs/shift_speed.py`, jo `ShiftIndex(library.load())` banaye aur 50 random `[M+H]+` rows pe `search` ka time naape.
Expected: build < 3 min. Search per spectrum < 1 s (median). Isse zyada ho to Task 3 ka cache zaroori hai, aur kernel budget (Task 6) ka andaza isi number se lagega. Number `RESULTS.md` §32 ke notes me jaata hai.

---

### Task 3: Candidate term, cache aur `dilute.py` integration

**Files:**
- Modify: `shifted.py` (`analog_term`, `cache_path`)
- Modify: `dilute.py`: `build_jobs` (`:120-148`), `score` (`:168-235`), `main` (`:245-280`), aur knobs block (`:37-110`)

**Interfaces:**
- Consumes: `ShiftIndex`, `search` (Task 2).
- Produces:
  - `analog_term(cand_fp, analog_fp, sims, p=3.0) -> np.ndarray`. `cand_fp` (n, 256) aur `analog_fp` (a, 256) packed uint8 hain. Return `max_a sims[a]**p * tanimoto(c, a)` hai. `a == 0` hone pe zeros.
  - `cache_path(splits_path, settings: dict, n_index_rows: int) -> str`. Path `data/shift_cache/<sha1>.npz` hai. Hash me split file ke bytes, sorted settings aur `n_index_rows` jaate hain.
  - `dilute.py` env knobs:
    - `SHIFT_WS` (comma list, default `"0"`)
    - `SHIFT_P` (3)
    - `SHIFT_N` (80)
    - `SHIFT_WIN` (200)
    - `SHIFT_SIM` (`entropy` | `cos`)
    - `SHIFT_STATS=1`
    - `BY_LIB=1`
  - `jobs` ka har tuple 9 elements ka hota hai. Naya element hai `analogs: list[(key, sim)]`, jo `SHIFT_WS` me koi weight > 0 ya `SHIFT_STATS=1` hone par hi compute hota hai.

- [ ] **Step 1: `analog_term` aur `cache_path` ke failing selftests likho**

```python
a = np.packbits(np.array([[1, 1, 0, 0] + [0] * 2044], np.uint8), axis=1)
c = np.packbits(np.array([[1, 1, 0, 0] + [0] * 2044, [1, 0, 1, 0] + [0] * 2044], np.uint8), axis=1)
assert np.allclose(analog_term(c, a, np.array([0.5]), p=3.0), [0.125, 0.125 / 3])
assert np.allclose(analog_term(c, a[:0], np.zeros(0)), [0, 0])
s1 = cache_path(path, {"sim": "entropy", "win": 200, "n": 80}, 10)
assert s1 == cache_path(path, {"n": 80, "win": 200, "sim": "entropy"}, 10)
assert s1 != cache_path(path, {"sim": "cos", "win": 200, "n": 80}, 10)
assert s1 != cache_path(path, {"sim": "entropy", "win": 200, "n": 80}, 11)
```

- [ ] **Step 2: Chalao, fail hona chahiye.** `../.venv/bin/python shifted.py --selftest` → `NameError`.

- [ ] **Step 3: Implement karo, phir selftest pass karao.**

- [ ] **Step 4: `dilute.py` integrate karo.**
  - `build_jobs`:
    - `ShiftIndex(ix, drop=sp["drop"], banned=sp["banned"])` banao, yaani wahi `C2_BANNED` ke baad wala set jo `Searcher` ko milta hai, `del ix` se pehle.
    - `neutral = pm − ADDUCTS[add[0]][0]`. `SHIFT` active ho to `assert TRUE_ADDUCT`.
    - Analog lists cache se padho. Cache na ho to compute karke likho.
    - **Leak assertion:** class 2/3 job ki apni key uske analogs me kabhi na ho.
  - `score(..., sw=0.0)`: pool candidates pe `sw * analog_term(an.fp[pool], fp_of(analog keys), sims, SHIFT_P)` jodo. Analog ke fingerprints `an.row` se aate hain, jo keys `an` me nahi hain unhe skip karo.
  - `main`: `SHIFT_WS` ko baaki sweeps jaisa ek aur loop banao. Print me `  shift {sw:<5g}` prefix ho, jab `len(SHIFT_WS) > 1`.
  - `BY_LIB=1`: class 3 MRR ko query library `enveda-180` vs baaki me split karke ek line print karo: `    class 3 by query lib:  enveda-180 0.xxxx (n N)   other 0.xxxx (n N)`. Query lib `ix["lib"][rows[0]]` se `build_jobs` me record hoti hai.
  - `SHIFT_STATS=1`: class 2 aur class 3 ke liye ek baar print karo (non-`enveda-180` aur `enveda-180` alag). Answer ke fingerprint `analog.DESC` se aate hain, kyunki banned answers `an` me nahi hote.
    - % jinke paas koi analog hai
    - % jinme `max_a Tanimoto(answer, a) ≥ 0.5` aur `≥ 0.7`
    - median top-1 sim
    - wahi Tanimoto stat purane seeds ke liye: `spec` ka top-1 isomer seed, yaani aaj ka `an.propagate`

- [ ] **Step 5: Regression check, `SHIFT_WS=0` pe purana number exactly aana chahiye** (RESULTS §30, ensemble of 4 + prior):

Run:
```bash
B=data/fpmodel_big_rank_s0.pt,data/fpmodel_big_rank_s1.pt,data/fpmodel_big_rank_s2.pt,data/fpmodel_big_rank_s3.pt
caffeinate -ims env SPLITS=data/splits_big.npz FP_MODEL=$B FRACS=1 GAMMA=32 PRIOR=syn PRIOR_WS=0.5 \
  PRIOR_GATES=0.8 C2_BANNED=1 SHIFT_WS=0 MIN_AVAIL_MB=2500 ../guard.sh ../.venv/bin/python dilute.py
```
Expected: class 1 `0.9174`, class 3 `0.4819`, LB column `0.4866`, jaise §30 me hai. Fark aaye to integration ne kuch aur badal diya hai. Aage badhne se pehle wajah dhoondhni hai.

- [ ] **Step 6: Commit (user ki haan ke baad)**

```bash
git add shifted.py dilute.py
git commit -m "Mass-shifted analog search over ±200 Da, in the eval behind SHIFT_WS"
```

---

### Task 4: Learning experiment 1: kya hidden-jaise molecules ke rishtedaar library me hain?

**Files:**
- Create: `LEARNING.md` (format design ke hisaab se: unka insight, user ka andaza, measured, fark ki wajah, hamari purani belief)
- Create: `data/jobs/shift_stats.sh` (git me nahi)

- [ ] **Step 1: User se andaza lo, experiment chalane se pehle.** Sawaal:

> "`splits_bigord` ke non-`enveda-180` class 3 answers (302) me se kitne % ke liye ±200 Da ke top-80 analogs me koi aisa structure hoga jiska answer se Tanimoto ≥ 0.5 ho? Aur aaj ke isomer-only seed ke liye ye % kitna hoga?"

Andaza `LEARNING.md` ki entry 1 me likho, number aane se pehle.

- [ ] **Step 2: Stats chalao**, `SHIFT_STATS=1` aur `SHIFT_WS=0` ke saath. `splits_bigord.npz` ke models `fpmodel_bigord_rank_s0.pt,fpmodel_bigord_rank_s1.pt` hain, `C2_BANNED=1`. Phir yahi `SHIFT_SIM=cos` ke saath bhi.
Expected: dono similarity kinds ke liye class 2/3 stats lines. Pehli run me analog cache banti hai, baad wali cache se padhti hain.

- [ ] **Step 3: User ke saath number aur andaza milao.** Fark ki wajah dhoondho (jaise: kaunsi chemistry ke analogs milte hain, kaunsi ke nahi; 5 answers aur unke top analogs ke SMILES dekho). `LEARNING.md` entry 1 poori karo, aur usme ye bhi likho ki RESULTS §5 ki "one seed beats many" aur same-window seeds wali soch kahan se aayi thi.

---

### Task 5: Learning experiment 2: MRR pe asar, aur LB prediction

**Files:**
- Create: `data/jobs/shift_sweep.sh`
- Modify: `RESULTS.md` (naya §32), `LEARNING.md` (entry 2)

- [ ] **Step 1: User se andaza lo.** Sawaal:

> "Shifted analog term jodne se (sabse achhe weight pe) `splits_bigord` ka non-`enveda-180` class 3 MRR kitna badhega (aaj ~0.135), aur class 1 pe kya hoga?"

Andaza `LEARNING.md` me likho.

- [ ] **Step 2: Sweep chalao.** `SHIFT_WS=0,4,8,16,32,64`, `BY_LIB=1`, aur v15 config (`GAMMA=32 PRIOR=syn PRIOR_WS=0.5 PRIOR_GATES=0.8`). Chaar runs:
  - `splits_bigord`, `C2_BANNED=1` ke saath aur bina
  - `splits_big` (4-seed ensemble), `C2_BANNED=1` ke saath aur bina

Ye weight ko signal dekhne ke liye sweep karna hai, LB ke liye tune karne ke liye nahi (design). Phir best weight pe `SHIFT_SIM=cos` aur `SHIFT_P=1,6` ek-ek baar.
Expected: har weight ki ek line. Ye check karo ki `SHIFT_WS=0` wali line Task 3 Step 5 se match kare.

- [ ] **Step 3: LB prediction banao.**
  - Main prediction: `0.140·Δclass1 + 0.126·Δclass2(C2_BANNED, bigord) + 0.735·Δclass3(bigord, non-env-180)`.
  - Lower bound: `splits_big` ka covered class 3 delta × 0.12.
  - Dono numbers aur unke beech ka fark `RESULTS.md` §32 me likho. Isi fark ka test LB karega (coverage model).

- [ ] **Step 4: `RESULTS.md` §32 likho** (tables §30/§31 ki style me) aur `LEARNING.md` entry 2 likho (andaza vs measured, entropy vs cosine ka learning).

- [ ] **Step 5: Commit (user ki haan ke baad)**

```bash
git add RESULTS.md LEARNING.md
git commit -m "Mass-shifted analogs on held-out: <one-line result>"
```

---

### Task 6: Kernel integration aur LB decision

**Files:**
- Modify: `fpsearch.py`: `submit()` (`:199-266`), knobs block (`:36-45`)
- Modify: `kernel/casmi_submit.py` (env knobs, step log line)
- Modify: `HANDOFF.md` ("Where we stand" aur "Next steps" ko Raasta B ke hisaab se)

**Interfaces:**
- Consumes: `ShiftIndex`, `analog_term` (Tasks 2–3). Task 5 ka chuna hua weight, `p` aur `kind`.
- Produces: `fpsearch.SHIFT_W` (env `SHIFT_W`, default 0), `SHIFT_P`, `SHIFT_SIM`. Kernel inhe `prior` jaise `os.environ` se set karta hai.

- [ ] **Step 1: `submit()` me term jodo.** `ShiftIndex(ix)` ko `del ix` se pehle banao. Molecule ka neutral mass chune hue spectrum `i` ke adduct se aata hai: `pm − ADDUCTS[add[0]][0]`. Phir sirf pool candidates pe `SHIFT_W * analog_term(...)`, gamma term ke baad aur prior se pehle.

- [ ] **Step 2: Local smoke run (placeholder test), do baar**

Run: `SHIFT_W=0 ../.venv/bin/python fpsearch.py --submit` aur `SHIFT_W=<chosen> ../.venv/bin/python fpsearch.py --submit`, alag output paths pe.
Expected:
- `SHIFT_W=0` ki file purane v15 config ke local output se byte-identical ho.
- Dono `validate()` pass karein.
- Runtime log me analog search ka time ho. Kernel budget = ~66 min + 400 molecules × ~3 spectra × Task 2 Step 5 ka per-spectrum time, jo 9 ghante se kaafi kam hona chahiye.
- Placeholder file se quality ke baare me kuch nahi padhna hai (HANDOFF Read first 1).

- [ ] **Step 3: LB decision.**
  - Task 5 ki prediction aur uska type (ship ya probe) `LEARNING.md` me pehle se likha ho.
  - User se confirm karo: version, prediction, aur ye ki probe final selection me nahi jaayega jab tak jeete nahi.
  - Haan mile to HANDOFF ka Kaggle workflow (dataset version → kernel push → submit → poll).

- [ ] **Step 4: Score aane par.**
  - `LEARNING.md` entry 3: prediction vs LB, aur coverage model ke baare me kya pata chala.
  - `RESULTS.md` §32 me LB line, HANDOFF ki version table update.
  - Commit (user ki haan ke baad), files path se.
