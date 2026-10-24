# Learned ranker with MetFrag-lite (Raasta B, component 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Har molecule ki v17 candidate list ko ek seekhe hue model se dobara rank karna. Model har candidate ke ~30 features leta hai, jinme MetFrag-lite fragment evidence bhi hai, aur held-out queries pe train hota hai. Ye queries leak-free fingerprint models se banti hain. Saath me learning cycle ke teen experiments chalane hain.

**Architecture:**
- `metfrag.py`: 1–2 bond cleavage se fragment masses banata hai, aur batata hai ki un fragments se spectrum ki kitni intensity explain hoti hai.
- `features.py`: ek function jo ek molecule ke candidates aur feature matrix banata hai. Ye wahi terms use karta hai jo v17 ka haath se weight kiya score. Eval aur kernel dono yahi ek function bulaate hain.
- `ranker.py`:
  - held-out splits se training tables banata hai;
  - sklearn HistGBM (binary, 4 seeds) train karta hai;
  - grouped k-fold se out-of-fold (OOF) MRR naapta hai;
  - trees ko npz me export karta hai, aur ek numpy predictor rakhta hai taaki kernel ko sklearn na chahiye.
- `fpsearch.submit`: ranker `RANKER` knob ke peeche aata hai. Knob unset ho to output v17 jaisa hi rehta hai.

**Tech Stack:**
- Python 3.12, numpy ≥ 2, RDKit 2026.3.3.
- Offline training: sklearn 1.7.2 (HistGBM, primary), aur LightGBM 4.7.0 `lambdarank` comparison ke liye (macOS pe `brew install libomp` chahiye, SETUP.md).
- Kernel me koi nayi dependency nahi jaayegi.
- Experiment tracking: MLflow. Venv me `mlflow-skinny` client hai, aur server `mlflow_server.sh` (uvx, sqlite, http://127.0.0.1:5050) se chalta hai.
- Exploration ke liye JupyterLab, outputs `nbstripout` se strip hote hain.
- Tests repo ke pattern me har module ke `selftest()` me hain (`python3 <module>.py --selftest`).

**Spec:** `docs/superpowers/specs/2026-10-20-raasta-b-design.md` (component 2, learning cycle, constraints)

## Global Constraints

- **Leak-free shart (spec):** kisi split ki queries ke fingerprint features sirf us model se bane, jo usi split ki rows ke bina train hua ho. `splits_big.npz` ke liye `fpmodel_big_*`, aur `splits_bigord.npz` ke liye `fpmodel_bigord_*`. Code isse assert karta hai. Ranker ka apna eval OOF hai, folds structure (key14) pe grouped.
- **Pool nahi badhna:** ranker koi candidate nahi jodta, sirf v17 ki list ke top `N_RERANK = 100` ko dobara order karta hai.
- **Eval:** tables `C2_BANNED=1` ke saath bante hain, v17 config pe (`GAMMA=32`, prior syn w 0.5 gate 0.8, `SHIFT_WS=16` entropy p 3). Class 3 `splits_bigord` ke non-`enveda-180` answers pe aur `splits_big` ke covered answers pe padhna hai. Class 1 girna nahi chahiye.
- **Heavy runs:** `caffeinate -ims` aur `MIN_AVAIL_MB=2500 ../guard.sh` ke neeche. Job scripts `data/jobs/` me hain aur `data/jobs/env.sh` source karte hain. `guard.sh` kabhi weaken nahi karna.
- **Worktree:** data aur models main checkout se `CASMI_DATA` ke zariye aate hain. `.venv` absolute path se. `python -c` ke saath `-P` lagta hai, script mode me nahi. Lambi shell commands `zsh data/jobs/<x>.sh` se chalti hain.
- **RDKit 2026.3.3 pinned.** Labels `metric.key14` se bante hain (tautomer canonical InChIKey-14), index ki key se nahi, taaki MRR competition ke metric jaisa hi ho.
- **Default nahi badalna:** `RANKER` unset ho to `fpsearch.submit` ka output byte-identical rehna chahiye.
- **Commit** sirf user ki haan ke baad, files path se stage karke. Models, tables, logs aur caches git me nahi jaate.
- **Learning cycle:** har experiment se pehle user ka andaza `LEARNING.md` me likhna hai, data aane se pehle. LB push user confirm karega, aur Kaggle pe likhne wale commands user `!` se khud chalayega.

## Review Focus

1. **Khaali molecule:** jiske paas na spectral match ho na pool, wahan `molecule_rows` 0 rows lautaye aur `submit` crash na kare. Test Task 2 me hai.
2. **Kharab candidate SMILES:** unparseable SMILES, mass table se bahar ka element, ya `MAX_BONDS` se zyada bonds wala molecule. Inpe `fragment_masses` exception na de, aur `explain` 0 ya sirf single-cut evidence de. Test Task 1 me hai.
3. **Ek hi candidate wala molecule:** rank, gap aur z features NaN/inf na banein. Test Task 2 me hai.
4. **Galat split ke liye fingerprint model:** jaise `fpmodel_rank.pt` ko `splits_big` pe chalana. Table build ko mana karna chahiye. Test Task 3 me hai.
5. **Export ke baad feature order badalna:** npz me feature names save hon, aur load karte waqt `features.FEATURES` se mile, warna error aaye. Exported predictor sklearn ke barabar score de. Test Task 4 me hai.

---

### Task 1: MetFrag-lite

**Files:**
- Create: `metfrag.py`. Docstring repo ki style me ho: kyun banaya, prvsiyan ka +0.021 (0.245 → 0.266), aur `formula.explained` (AUC 0.6) se fark.

**Interfaces:**
- Produces:
  - Constants: `MAX_CUTS = 2`, `MAX_BONDS = 64`, `H_SHIFTS = (-2, -1, 0, 1, 2)`, `TOL_DA = 0.01`.
  - `fragment_masses(smiles: str) -> np.ndarray`:
    - Sorted, unique neutral masses.
    - Har heavy atom ka mass uske implicit H ke saath, `formula.MASS` se.
    - 1 ya 2 bonds hatane pe jo connected components bante hain, unke sums, plus poora molecule.
    - `> MAX_BONDS` bonds pe sirf single cuts.
    - Unparseable SMILES ya `formula.MASS` se bahar ka element → `np.zeros(0)`.
  - `explain(frag: np.ndarray, mzs, ints, adduct: str) -> float`, range [0, 1]:
    - Ion masses = `frag + dh·H + carrier`, jahan `dh ∈ H_SHIFTS`.
    - Carriers adduct ki polarity se: proton, aur `[M+Na]+`/`[M+K]+`/`[M+NH4]+`/`[M+Cl]-`/`[M+CH2O2-H]-` ke liye unka apna ion bhi.
    - Electron mass `analog.E` hai.
    - Peak tab explained maana jaata hai jab koi ion mass `TOL_DA` ke andar ho.
    - Return = explained intensity / total intensity. Intensities waisi hi aati hain jaisi `library.trim` deta hai (sqrt-scaled).

- [ ] **Step 1: Failing selftest likho**

```python
f = fragment_masses("CCO")                               # ethanol: CH3, CH2OH, C2H5, OH, CH2, whole
for m in (15.02348, 31.01839, 29.03913, 17.00274, 14.01565, 46.04186):
    assert np.abs(f - m).min() < 1e-4, m
ion = 31.01839 + MASS_H - E                              # protonated CH2OH, dh = 0
assert explain(f, [ion], [1.0], "[M+H]+") == 1.0
assert explain(f, [ion, 500.0], [1.0, 1.0], "[M+H]+") == 0.5
assert explain(f, [ion - 2 * MASS_H], [1.0], "[M+H]+") == 1.0   # dh = -2 is allowed
assert explain(f, [500.0], [1.0], "[M+H]+") == 0.0
assert explain(f, [31.01839 - MASS_H + E], [1.0], "[M-H]-") == 1.0
assert len(fragment_masses("not a smiles")) == 0
assert len(fragment_masses("[Au]")) == 0
assert explain(np.zeros(0), [100.0], [1.0], "[M+H]+") == 0.0
long = fragment_masses("C" * 70)                         # 69 bonds > MAX_BONDS: single cuts only
assert 1 < len(long) <= 2 * 69 + 1
```

- [ ] **Step 2: Chalao, fail hona chahiye.** `../.venv/bin/python metfrag.py --selftest` (worktree me venv ka absolute path) → `NameError`.

- [ ] **Step 3: Implement karo.** 2-cut enumeration ka tareeka builder pe hai, jaise har bond pair pe components nikaalna (prvsiyan) ya bridges aur ring bonds ka khayal rakhna. Shart sirf ye hai ki output exact ho aur speed Step 5 ke target ke andar rahe.

- [ ] **Step 4: Chalao, pass hona chahiye.** Expected: `metfrag self-checks passed`.

- [ ] **Step 5: Asli candidates pe speed check.** `data/jobs/metfrag_speed.py` likho, jo `structures.npz` se 2000 random SMILES le, plus COCONUT ke 2000. Har ek ka `fragment_masses` time naapo: median, p95, aur bonds > 64 wale ka %. Expected: median ≤ 20 ms. Isse zyada aaye to enumeration tez karo, ya `MAX_BONDS` ka ruling ledger me likho. Ye numbers Task 3 ka table time aur Task 7 ka kernel budget tay karte hain.

- [ ] **Step 6: Commit (user ki haan ke baad).** `git add metfrag.py`

---

### Task 2: Candidate features

**Files:**
- Create: `features.py`

**Interfaces:**
- Consumes:
  - `analog.Analog` (`key`, `smiles`, `mass`, `fp`, `row`, `pool()`, `propagate()`, `n_internal`), `analog.BETA`, `analog.ADDUCTS`
  - `shifted.analog_term`
  - `formula.explained`
  - `metfrag.fragment_masses`, `metfrag.explain`
  - `prior.load("syn")` ka dict (log1p synonyms), `prior.FILL`
- Produces:
  - `V17 = dict(gamma=32.0, beta=1.0, sw=16.0, p=3.0, prior_w=0.5, prior_gate=0.8)`, `N_RERANK = 100`
  - `FEATURES: tuple[str, ...]`, is order me:
    - `v17`, `v17_rank`, `v17_gap`
    - `spec`, `spec_gap`, `has_spec`, `prop`
    - `fp`, `fp_rank`, `fp_gap`, `fp_z`
    - `shift3`, `shift1`, `shift_tan`, `shift_top`
    - `prior`, `prior_rank`
    - `frag`, `frag_mean`, `frag_gap`
    - `formula`
    - `ppm`
    - `log_pool`, `top_spec`, `n_spectra`, `in_pool`, `internal`
  - `GROUPS: dict[str, tuple[str, ...]]`: `v17`, `spec` (spec…prop), `fp`, `shift`, `prior`, `frag`, `formula`, `mass` (ppm), `pool` (log_pool…internal).
  - `molecule_rows(an, bits, spec, smiles, pm, add, fpq, spectra, analogs, prior_of, frags=None, cfg=V17) -> tuple[list[bytes], list[str], np.ndarray]`:
    - Return: keys, SMILES, aur `X` (n, len(FEATURES)) float32.
    - Candidates `dilute.score` ke insertion order me aate hain: spec keys, phir propagate keys, phir pool.
    - `v17` = `spec_sum + beta·prop + gamma·fp + sw·shift3 + prior_w·prior`.
      - `spec_sum` = `spec[c]`, spectra ke upar raw sum, jaise `dilute.score` ka `total` leta hai.
      - Prior tabhi judta hai jab `top_spec < prior_gate`, jahan `top_spec = max(spec)/len(spectra)`.
    - Rows `−v17` pe stable sort hoti hain aur `N_RERANK` pe kat jaati hain.
    - `frags` ek `{smiles: fragment masses}` dict hai, jo caller ne pehle se bana kar diya ho. Kisi SMILES ki entry na ho to inline compute karo.
- Feature definitions:
  - `*_rank`: molecule ke andar rank-norm, 0 = sabse achha.
  - `*_gap`: value − molecule ka max.
  - `fp_z`: (t − mean)/std. std < 1e-9 ho to 0.
  - `spec`: `spec[c]/len(spectra)`.
  - `shift1`: `max_a sim_a·T(c, a)`.
  - `shift_tan`: `max_a T(c, a)`.
  - `shift_top`: top relative ka sim, molecule-level.
  - `frag`/`frag_mean`: `explain` ka max aur mean, molecule ke spectra ke upar, har spectrum ke apne adduct (`add[0]`) ke saath.
  - `formula`: `formula.explained` ka mean, spectra ke upar.
  - `ppm`: candidate ke mass ka |neutral − mass|/neutral·1e6. Jo `an` me nahi hai, uske liye 50.
  - `in_pool`: candidate 5 ppm pool me hai.
  - `internal`: `an.row[c] < an.n_internal`.

- [ ] **Step 1: Failing selftest likho.** Ek chhota fake `an` banao: 4 structures, `pool()`/`propagate()` stubs, 2048-bit fps. Usme ek spec-only candidate ho (pool ke bahar), aur prior dict ho.

```python
keys, smi, X = molecule_rows(an, bits, spec, smiles, pm, add, fpq, spectra, analogs, prior_of)
F = {n: i for i, n in enumerate(FEATURES)}
assert X.shape == (4, len(FEATURES)) and np.isfinite(X).all()
# want: one literal per candidate, worked out by hand from the stub's numbers as
# spec_sum + 1.0*prop + 32*fp + 16*shift3 + 0.5*prior; two candidates get equal totals on purpose
assert np.allclose(X[:, F["v17"]], want_sorted)
assert X[0, F["v17_rank"]] == 0 and X[0, F["v17_gap"]] == 0
assert list(keys) == order_by_minus_want_stable    # the tied pair stays in spec -> prop -> pool order
# prior gate: a molecule with top_spec >= 0.8 has v17 without the prior, but the `prior` column is filled
# single candidate: rank/gap/z all 0, finite
# empty: no spec and an empty pool -> X.shape == (0, len(FEATURES))
# N_RERANK: a 150-candidate stub returns 100 rows, and they are v17's top 100
```

- [ ] **Step 2: Chalao, fail.** Phir **Step 3: implement**, aur **Step 4: pass** (`features self-checks passed`).

- [ ] **Step 5: Commit (user ki haan ke baad).** `git add features.py`

---

### Task 3: Training tables, aur v17 ka regression

**Files:**
- Create: `ranker.py` (yahan table wala hissa; model wala Task 4 me)
- Create: `data/jobs/rank_tables.sh` (git me nahi)

**Interfaces:**
- Consumes: `dilute.build_jobs()` (env se configure hota hai: `SPLITS`, `FP_MODEL`, `SHIFT_WS=16`, `C2_BANNED=1`), `analog.Analog(banned=…, external=True)`, `features.molecule_rows`, `metric.key14`.
- Produces:
  - `LEAK_FREE = {"splits_big.npz": "fpmodel_big_", "splits_bigord.npz": "fpmodel_bigord_"}`.
  - `check_leak_free(splits_path: str, fp_model: str) -> None`: comma-separated `fp_model` ke har member ka basename split ke prefix se shuru na ho to `AssertionError`.
  - `build_table(out: str) -> str`:
    - Leak check sabse pehle.
    - Phir unique candidate SMILES ke `fragment_masses` aur `key14` ek `multiprocessing` pool me (spawn), `N_PROC = cpu_count − 2`.
    - Phir har query ke liye `molecule_rows`.
    - Labels: `y = key14(candidate) == key14(answer)`. Ek query me ek se zyada positives ho sakte hain.
  - Table npz ke andar:
    - per row: `X` float32, `y` uint8, `qid` int32;
    - per query: `qkey` S14, `qclass` int8, `env` bool (query lib == `enveda-180`), `split` str;
    - `feature_names`.
  - Table path: `{WORK}/rank_tables/{split_stem}_{sha1[:12]}.npz`. Hash me ye sab jaate hain: split ke bytes, `FP_MODEL` basenames, `V17`, `N_RERANK`, `FEATURES`, metfrag constants, `TABLE_VERSION = 1`.
  - `mrr_by(table, score) -> dict[str, float]`:
    - Har query ki rows `−score` pe stable order hoti hain. rr = 1/rank pehle positive ka, agar rank ≤ 25, warna 0.
    - Keys: `c1`, `c2`, `c3`, `c3_nonenv`, `c3_env`.
    - Har MRR ka denominator class ki **saari** queries hain, unki bhi jinke top-100 me answer nahi hai.
  - CLI: `python3 ranker.py --table` (env se split), `--selftest`.

- [ ] **Step 1: Failing selftest likho:** `check_leak_free` aur `mrr_by`

```python
check_leak_free("data/splits_big.npz", "/x/fpmodel_big_rank_s0.pt,/x/fpmodel_big_rank_s1.pt")    # passes
for bad in ("/x/fpmodel_rank.pt", "/x/fpmodel_big_rank_s0.pt,/x/fpmodel_rank_s1.pt"):
    try: check_leak_free("data/splits_big.npz", bad); raise SystemExit("leak not caught")
    except AssertionError: pass
# mrr_by on a hand table: q0 positive at rank 2, q1 positive at rank 1 with a duplicate positive at 3,
# q2 has no positive -> class mean (0.5 + 1 + 0) / 3
```

- [ ] **Step 2: Fail, Step 3: implement, Step 4: pass.**

- [ ] **Step 5: Dono tables banao** (`rank_tables.sh`, `caffeinate` aur `guard.sh` ke neeche):
  - `SPLITS=$D/splits_bigord.npz FP_MODEL=$E` aur `SPLITS=$D/splits_big.npz FP_MODEL=$B`.
  - Dono `${V17[@]}` env ke saath.
  - Log me teeno stages ka time ho: `build_jobs`, fragments+labels, features.

  Expected: dono table files likhi jaayein. Time ek ghante ke andar ho. `formula` feature akela 30 min se zyada le to use `GROUPS` se hatane ka ruling ledger me likho.

- [ ] **Step 6: v17 ka regression, yahi is task ka gate hai.** `mrr_by(table, X[:, v17])` ko RESULTS §32 ke SW 16 numbers se 4 decimals tak milna chahiye:
  - bigord: c1 `0.9232`, c2 `0.0348`, c3 `0.0455`, c3_nonenv `0.1507`
  - big: c1 `0.9151`, c2 `0.0382`, c3 `0.5234`

  Fark aaye to features ka v17, `dilute` ka v17 nahi hai. Wajah dhoondho: insertion order, ties, labels ya prior gate. Gate pass hone se pehle Task 4 shuru nahi karna.

- [ ] **Step 7: Ceiling print karo.** Har class ke liye kitni queries ka positive top 25 me hai aur kitno ka top 100 me. Ye RESULTS ke notes me jaayega. Ye bataata hai ki ranker zyada se zyada kya jeet sakta hai.

- [ ] **Step 8: Commit (user ki haan ke baad).** `git add ranker.py`

---

### Task 4: GBM ranker, OOF eval, export

**Files:**
- Modify: `ranker.py`

**Interfaces:**
- Consumes: Task 3 ke tables, `dilute.MIX`.
- Produces:
  - `GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)`.
  - `SEEDS`: env `RANK_SEEDS` se, comma list, default `"0,1,2,3"`.
  - `load_tables(paths) -> dict`: tables ko jodta hai, aur `qid` ko global banata hai.
  - `fit(T, rows_mask=None, drop_groups=(), model="gbm", seeds=SEEDS) -> list`:
    - Row weight = `MIX[qclass] / n_queries_with_positive[qclass]`, training rows ke andar gina hua.
    - Jin queries ka koi positive nahi, woh training se bahar.
    - `model="linear"`: `StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000)`, ek hi model.
    - `model="lambdarank"`: `lightgbm.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.03, num_leaves=63, min_child_samples=80, reg_lambda=1.0)`, seeds ke saath.
      - Group = query, aur query ka weight `MIX` wala hi.
      - `lightgbm` sirf isi branch me import hota hai.
      - Export abhi sirf `"gbm"` ka hai. Agar Task 6/7 me lambdarank jeete, to `booster.dump_model()` ke JSON se usi npz format me export likhna hai, round-trip test ke saath, aur ledger me ruling.
  - `cv(T, k=5, drop_groups=(), model="gbm") -> np.ndarray`:
    - OOF score per row.
    - `GroupKFold(k)` groups = `qkey`. Isliye dono splits me common class 1 structures ek hi fold me rehte hain.
  - `export(models, feature_names, path)` aur `class Ranker`:
    - `Ranker.load(path)` load karte waqt npz ke `feature_names` ko `features.FEATURES` (minus dropped) se assert karta hai.
    - `Ranker.score(X) -> np.ndarray`: models ki probabilities ka mean, sirf numpy se. Tree traversal vectorised hai, max depth tak loop.
    - npz me ye hai: har tree ke `feature_idx`, `num_threshold`, `missing_go_to_left`, `left`, `right`, `is_leaf`, `value` concatenated, offsets, aur har model ka `_baseline_prediction`.
  - `paired_bootstrap(rr_a, rr_b, n=1000, seed=0) -> tuple[float, float, float]`: delta, 2.5%, 97.5%.
  - `pool_auc(T, col) -> float`: har query ke andar positive vs negatives ka AUC. Ties 0.5 gine jaate hain. Average sirf un queries pe jinme dono hon.
  - `log_run(name, params: dict, metrics: dict, artifacts=()) -> None`:
    - MLflow me experiment `casmi-ranker` ke andar ek run.
    - Sirf tab jab `MLFLOW_TRACKING_URI` set ho, warna no-op. `mlflow` andar hi import hota hai, taaki kernel aur selftest ko uski zaroorat na pade.
    - Params: tables, model, drop groups, GBM params, seeds, `N_RERANK`, `TABLE_VERSION`, git commit.
    - Metrics: har split ke `mrr_by` (v17 aur ranker), deltas, bootstrap CI, runtime.
  - `--cv` OOF scores ko `{WORK}/rank_oof/<run>.npz` me likhta hai, tables ki row order me. Notebook inhi ko padhta hai.
  - CLI:
    - `--cv TABLES… [--drop=g1,g2] [--model=linear|lambdarank]`: har split ke liye v17 vs ranker ka `mrr_by` print karta hai, aur c3_nonenv (bigord) aur c3 (big) ka bootstrap delta.
    - `--fit TABLES… --out PATH`
    - `--auc TABLES…`

- [ ] **Step 1: Failing selftests likho**

```python
# export round trip: synthetic 3000 rows x len(FEATURES), 4 seeds, max_iter 30
r = Ranker.load(path)
assert np.abs(r.score(X) - np.mean([m.predict_proba(X)[:, 1] for m in models], 0)).max() < 1e-9
# a value exactly on a threshold goes left, as sklearn does: covered by X built from num_threshold values
# feature-name mismatch: an npz whose names are reordered -> Ranker.load raises AssertionError
# GroupKFold: one qkey in two splits never lands in train and test of the same fold
# pool_auc on a hand table: positive above all negatives -> 1.0; tie -> 0.5
# log_run with MLFLOW_TRACKING_URI unset: returns without importing mlflow
#   (assert "mlflow" not in sys.modules afterwards)
# fit(model="lambdarank") on the synthetic table: one booster per seed, scores finite
```

- [ ] **Step 2: Fail, Step 3: implement, Step 4: pass.** Expected: `ranker self-checks passed`.

- [ ] **Step 5: Smoke CV** dono tables pe, `--cv` ke saath, MLflow server chalu karke (`zsh mlflow_server.sh &`, `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`). Learning experiment ke liye abhi number padhne nahi hain, Task 5 me padhenge. Bas ye check karo:
  - ye chalta hai, aur time kitna lagta hai;
  - OOF scores sab finite hain;
  - run MLflow UI me dikhta hai.

- [ ] **Step 6: Commit (user ki haan ke baad).** `git add ranker.py`

---

### Task 5: Learning experiment 3: kya seekha hua ranker haath ke weights se behtar jodta hai?

**Files:**
- Create: `data/jobs/rank_cv.sh`, `notebooks/ranker_explore.ipynb`, `../.gitattributes` (nbstripout)
- Modify: `LEARNING.md` (entry 4)

- [ ] **Step 1: User se andaza lo, run se pehle.** Sawaal:

> "Ranker, abhi MetFrag-lite aur formula ke bina (`--drop=frag,formula`), OOF pe teen numbers kahan le jaayega? (1) bigord non-env class 3, jo v17 pe 0.1507 hai. (2) big covered class 3, jo 0.5234 hai. (3) bigord class 1, jo 0.9232 hai. Ek linear model GBM se kitna peeche rahega? Aur LightGBM `lambdarank` (seedha ranking objective) binary GBM se aage hoga ya peeche?"

Andaza `LEARNING.md` entry 4 me likho.

- [ ] **Step 2: Chalao.** `--cv … --drop=frag,formula` teen baar chalao: GBM, `--model=linear`, aur `--model=lambdarank`. Teeno runs MLflow me log hon. Expected: har split ka v17 vs ranker `mrr_by` table, bootstrap deltas, aur runtime.

- [ ] **Step 3: Exploration notebook banao:** `notebooks/ranker_explore.ipynb`, git me, outputs strip karke.
  - nbstripout pehli baar set karna hai: `nbstripout --install --attributes .gitattributes`. Ye `.git/config` me filter likhta hai (saare worktrees ke liye shared), aur `.gitattributes` banata hai (tracked, user ki haan ke saath commit).
  - Notebook tables aur `rank_oof` padhta hai. Usme ye hon:
    - answer vs baaki candidates ke feature distributions;
    - v17 → ranker rank movement ka scatter;
    - sabse zyada upar aur neeche gaye 10 molecules, unke top-5 candidates ke SMILES aur features ke saath.
  - `jupyter nbconvert --execute --to notebook` se ek baar poora chala ke check karo ki bina error chalta hai.
  - User ise JupyterLab me khol ke aage khoj sakta hai: `.venv/bin/jupyter lab`, repo root se.

- [ ] **Step 4: Padho aur entry 4 likho.**
  - Andaza vs measured.
  - Per-molecule rank movement (upar, neeche, rank 1 pe aaye), `rank_shift.py` ki style me.
  - Class 3 answers ke synonym strata pe split (0–1 synonyms wala stratum test ke sabse kareeb hai, RESULTS §30). Ranker prior pe zyada jhuka to ye stratum girega.
  - Agar class 1 gire to wajah aur ilaaj ka option likho, jaise ranker ko sirf `top_spec < 0.8` wale molecules pe lagana. Iska faisla Task 7 me data ke saath hoga.

---

### Task 6: Learning experiment 4: MetFrag-lite kitna jodta hai, aur ranker kis cheez pe tika hai?

**Files:**
- Create: `data/jobs/rank_ablate.sh`
- Modify: `LEARNING.md` (entry 5), `RESULTS.md` (naya §33)

- [ ] **Step 1: User se andaza lo.** Sawaal:

> "(1) Pool ke andar answer ko baaki candidates se alag karne me `frag` ka AUC kitna hoga? `formula` ka purana AUC 0.6 tha. (2) MetFrag-lite jodne se big covered class 3 OOF kitna badhega? prvsiyan ke yahan +0.021 LB tha, unke pipeline pe. (3) Ranker sabse zyada kis group pe tika hoga: fp, spec, shift, prior ya frag?"

- [ ] **Step 2: Chalao.**
  - `--auc` saare features ke liye.
  - Full features ke saath `--cv`.
  - Group ablation: har group ko ek-ek karke `--drop` karo, 1 seed (`RANK_SEEDS=0`), taaki time ~20 min me rahe.

  Expected:
  - AUC table, jisme `frag` aur `formula` ki line ho;
  - full-feature OOF;
  - har group ke drop pe class 3 aur class 1 ka delta.

- [ ] **Step 3: Notebook me ek section jodo:** `frag` vs `formula`. Answer aur baaki candidates ke distributions, aur 5 molecules jahan `frag` ne answer ko upar kheencha, unke matched fragments ke saath.

- [ ] **Step 4: `RESULTS.md` §33 likho**, §32 ki style me: features, tables, ceiling, CV, AUC, ablation, linear vs GBM vs lambdarank. Phir `LEARNING.md` entry 5 likho.

- [ ] **Step 5: Commit (user ki haan ke baad).** `git add RESULTS.md LEARNING.md notebooks/ranker_explore.ipynb ../.gitattributes`

---

### Task 7: Kernel integration aur LB decision

**Files:**
- Modify:
  - `fpsearch.py`: knobs block, aur `submit()`.
  - `kernel/casmi_submit.py`: env knobs, docstring timing, step log.
  - `HANDOFF.md`.
- Create: `data/jobs/rank_smoke.sh`, `data/jobs/kcode_build.sh`. Dono git me nahi hain. kcode_build `raasta-b-c1` worktree wale script ki tarah hai.

**Interfaces:**
- Consumes: `features.molecule_rows`, `Ranker`, `metfrag.fragment_masses`.
- Produces: `fpsearch` env knobs:
  - `RANKER`: ranker npz ka path, `""` = off.
  - `RANK_BY`: `""`, ya test mode `"v17"`, jo features ke `v17` column se order karta hai.
- Agar dono me se koi set hai, `submit()`:
  - `SHIFT_W`, `GAMMA` aur prior settings ko `features.V17` se assert karta hai.
  - Saare molecules ke liye `molecule_rows` chalata hai, fragments pehle ek `multiprocessing` pool me (unique SMILES).
  - Ek hi `Ranker.score` call me sab score karta hai, har molecule ki rows dobara order karta hai, aur top 25 likhta hai.

- [ ] **Step 1: Implement karo.** Default path (dono knobs unset) bilkul nahi chhuna hai.

- [ ] **Step 2: Smoke** (placeholder test, `rank_smoke.sh`, `python -P` aur `fpsearch.__file__` print):
  - Knobs unset aur `SHIFT_W=16`: output c1 worktree ke `data/jobs/sub_w16.csv` se byte-identical.
  - `RANK_BY=v17`: wahi file byte-identical. Ye saabit karta hai ki kernel ke raaste pe features ka v17, shipped scoring hi hai.
  - `RANKER=data/ranker_v1.npz`: `validate()` pass ho, runtime log ho, aur kitni lists badlin. Placeholder pe quality nahi padhni.
  - Expected: Mac pe ranker ka kharcha. Kaggle ke liye ×3.4, aur v17 ki ~2.5 h scoring pe joda hua andaza (HANDOFF). Ye 9 h se kaafi neeche hona chahiye.

- [ ] **Step 3: Final ranker fit karo.** `--fit` dono tables pe, Task 6 ke chune hue features ke saath, output `data/ranker_v1.npz`. Ye git me nahi jaata, kernel/code me jaata hai.

- [ ] **Step 4: LB prediction aur decision.**
  - Plan formula: `0.140·Δc1 + 0.126·Δc2 + 0.735·Δc3_nonenv`, bigord OOF se.
  - Coverage-scaled: `0.735 · 0.12 · Δc3` (big covered), plus class 1/2 ke terms.
  - v17 ne dikhaya ki asli LB dono ke beech girta hai (LEARNING entry 3). Ranker har candidate ka score badalta hai, jaise v13, isliye transfer ka khatra zyada hai. Ye `LEARNING.md` entry 6 me pehle likhna hai, saath me pre-registered threshold.
  - User se confirm karo: version, prediction, aur ki probe tabhi final selection me jaayega jab threshold paar kare.
  - Haan mile to kcode_build chalao: v17 ka `kernel/code` (c1 worktree se), plus `features.py`, `metfrag.py`, `ranker.py`, `ranker.npz`, aur naya `fpsearch.py`. Folder se local submit `RANKER` ke saath Step 2 wali file se byte-identical aana chahiye.
  - Phir user ko `kaggle datasets version` / `kernels push` / `competitions submit` ke exact commands do.

- [ ] **Step 5: Score aane par.**
  - `LEARNING.md` entry 6: prediction vs LB, aur kya ranker ka transfer v13 jaisa toota.
  - `RESULTS.md` §33 me LB line, HANDOFF ki version table aur "Next steps".
  - Commit (user ki haan ke baad), files path se.
