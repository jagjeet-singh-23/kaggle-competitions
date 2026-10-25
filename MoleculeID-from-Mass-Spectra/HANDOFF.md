# Handoff — CASMI 2026 (as of 2026-10-24)

Read RESULTS.md §11–30 first; this file is the short version plus how to run things.
The next experiments, and how to run them on the Mac, are in **PLAN_MAC.md**.

## Where we stand

- Competition: `enveda-CASMI26-molecule-id-mass-spectra`, MRR@25 on InChIKey14, closes 14 Dec 2026.
- **Best LB 0.264 (v17)** = v15 + mass-shifted analogs (`SHIFT_W=16 SHIFT_P=3 SHIFT_SIM=entropy`, §32, Raasta B
  component 1): spectral relatives within ±200 Da re-rank the pool, no candidates added. Predicted +0.0035
  (coverage-scaled) to +0.011 (plan formula), scored +0.007. **Select v17 for the final ranking, v15 second**
  (v15 is the same pipeline without the channel). Learning log for Raasta B: `LEARNING.md`.
- **v18 0.257 (−0.007), not selected**: v17 plus a learned lambdarank ranker over each molecule's top 100, with
  MetFrag-lite features (§33, Raasta B component 2). It won held-out covered class 3 by +0.165 [+0.148, +0.181] and
  was predicted at +0.015 to +0.023. This is the third change that re-scores every candidate with something fitted
  on held-out query lists and loses on the LB (v13 −0.007, v16 −0.014). Held-out gains from such changes are not
  LB predictions, however tight their intervals (`LEARNING.md` entry 6).
- v15 0.257 = v14 + a 4-seed fingerprint ensemble (§29); **v14 0.254**: v12 (search + analog + predicted-fingerprint tanimoto, gamma 32, one adduct window per
  molecule (§14), fingerprint model fine-tuned with a **ranking loss over the 5 ppm candidate pool**, §21–22)
  plus a **COCONUT synonym-count prior** (`PRIOR=syn PRIOR_W=0.5 PRIOR_GATE=0.8`, §28). The prior gave +0.002;
  the eval scaled by class 3 coverage predicted +0.005 ± 0.003 (§30), so the two agree within the LB's noise.
- Class mix pinned (§7): LB ≈ 0.140·c1 + 0.126·c2 + 0.735·c3. Class 3 is 73.5% of the score.
- Kaggle kernel `jagjeetsingh23/casmi26-search-analog` latest version is **v18 (0.257)**; v17 (0.264) is version 17.
  `kernel/casmi_submit.py` is back at v17: v15's config plus `SHIFT_W=16`, with `RANKER` unset. `kernel/kernel-metadata.json`
  is v15's. The code dataset's latest version is v18's: v17's folder plus `features.py`, `metfrag.py`, `ranker.py`,
  `formula.py`, the new `fpsearch.py` and `ranker_v1.npz`. With `RANKER` unset that `fpsearch.py` writes v17's file
  byte for byte, so pushing this `casmi_submit.py` against it reproduces v17. v18's folder is in the c2 worktree's
  `kernel/code/`, built by `data/jobs/kcode_build.sh`. v17's code dataset version is v15's
  `kernel/code/` with this tree's `fpsearch.py` and `shifted.py` on top (`data/jobs/kcode_build.sh`, which also proves
  the folder reproduces the locally tested SW 16 file). v16 (0.243) was the failed PubChem popularity pool (§31).
  v15 (0.257) is
  v14 plus a 4-seed ensemble (§29). `kernel/code/` (git-ignored) holds v15's files: the repo's code, `fpmodel.pt` =
  `data/fpmodel_rank.pt` (v12's model), `fpmodel_s{1,2,3}.pt` = `data/fpmodel_rank_s{1,2,3}.pt`, `coconut_meta.npz` =
  `data/coconut_meta/meta.npz`; `kernel/casmi_submit.py` sets the `PRIOR*` env and the comma-separated
  `fingerprint.MODEL`. v15 was expected to tie v14 (scaled eval +0.0005) and scored +0.003, inside the LB's noise:
  the ensemble is not shown to be worth +0.003, only not to hurt.
- Ranking loss swept on the Mac (§24): at the shipped gamma 32 nothing beats tau 0.05 / lambda 1 / lr 1e-3 / 2 epochs,
  so v12 stays. Sharper tau wins only at gamma >= 64, which held-out over-rewards (§15–16). A tau 0.0125 fine-tune of
  `fpmodel.pt` (`data/fpmodel_rank_t0125.pt`) fails the gate against v12 on `splits.npz` at raw gamma.
- Pool-normalised fingerprint term (`FP_NORM=std`, §27): sharp tau wins clearly on the leak-free split, but v13
  (tau 0.0125, std gamma 2.62) scored **0.245 vs 0.252**. The leak-free split's class 3 gain did not transfer;
  `splits.npz` class 1 falling (-0.007) was the honest signal.
- **Before 2026-10-17 every `dilute.py` number without an explicit `GATES` was gamma 128, not the `GAMMA` given**
  (§17–23; default gate bug, fixed). Re-measure before comparing against an old number.

| ver | change | LB |
|---|---|---|
| v1 | library search only | 0.140 |
| v2 | + analog propagation, fingerprint scoring | 0.158 |
| v3 | + COCONUT candidate pool | 0.199 |
| v4 | mass-defect features, hidden 2048 | 0.213 |
| ChEMBL pool | bigger candidate pool | 0.187 (pool expansion closed) |
| v9 | adduct-window bug fix | 0.236 |
| v10 | gamma 64 | 0.235 |
| v11 | gated gamma (0.7 / 128) | 0.236 |
| v12 | fpmodel + in-pool ranking loss (§21–22), gamma 32, no gate | **0.252** |
| v13 | tau 0.0125 ranking model, std-normalised fingerprint term, gamma 2.62 (§27) | 0.245 |
| v14 | v12 + COCONUT synonym prior w 0.5, gate 0.8 (§28) | 0.254 |
| v15 | v14 + 4-seed fingerprint ensemble (§29) | 0.257 |
| v16 | v15 + PubChem 2+ synonym pool (16M), prior w 1 (§31) | 0.243 (pool expansion closed again) |
| v17 | v15 + mass-shifted analogs, SW 16, entropy, p 3 (§32) | **0.264** |
| v18 | v17 + learned ranker over the top 100, lambdarank with MetFrag-lite (§33) | 0.257 (held-out +0.165 covered c3 did not transfer) |

## Read first: what looks like a fact but is not

Every item here was discovered the hard way, some of them twice. Do not re-derive them.

**The three different "tests"**

1. **Local `data/test.parquet` is a placeholder.** All 400 molecules have an identical twin (same precursor, peaks,
   intensities) in the train library `enveda-180`; scored against its answers the submission gets **0.9975** (§1).
   Use it only to check that `submit()` runs. Never infer hidden-test properties from it: adduct mix, mass range, how
   many spectra per molecule, match strength, how many lists a change touches. A gated-gamma change looked like a
   no-op on it (1 of 400 lists changed) only because every molecule has a cosine-1.0 twin; that says nothing about how
   many hidden lists a change touches. `GUIDE.md` is Hinglish and predates every result: its numbers about "the test"
   describe this placeholder, and its modelling plan is background, not the plan. What it quotes from the competition
   page (10 test adducts, 25 candidates, metric, tautomer canonicalisation) is fine.
2. **Kaggle scores a different file.** It is a kernels-only competition with internet off: submitting re-runs the
   kernel on the hidden test (~66 min kernel, ~70 min scoring). A kernel that finishes only proves it ran. The only
   evidence of quality is the public LB; the private score is not revealed until 14 Dec 2026.
3. **Held-out classes are not the hidden classes, and this repo's class names are not the competition's.**

The competition's own definitions (Kaggle Data tab, read 2026-10-18):

| official class | definition | reachable by |
|---|---|---|
| 1 — in public spectral libraries | the structure has public reference MS/MS spectra | similarity to the training spectra |
| 2 — known structure, no public spectra | not in any public library, but **in PubChem or COCONUT** | database retrieval |
| 3 — novel structure | **not in PubChem at all** | de novo only |

The hidden class mix is not published. The test molecules are "confirmed natural products, hypothesised natural
products, natural product analogs, and synthetic molecules that might plausibly occur in nature", all on the same Bruker
timsTOF as `enveda-180`. The Data tab describes `enveda-180` as "synthetic drug-like screening compounds, a different
region of chemical space from the test molecules". `enveda-np-examples` (250 common natural products, same instruments
and pipeline as the test) is the closest library to it.

**Everywhere else in this repo, "class 3" means "not found by library search", i.e. official class 2 + official class 3
together**, and "class 2" is the held-out class whose answer is one of the index's own structures. The 14.0 / 12.6 / 73.5
mix (§7–8) is in those terms. Consequences:
- No database can reach official class 3. What any pool expansion can win is official class 2, all of which is in
  PubChem ∪ COCONUT by definition. The question for a pool is how much of the hidden official class 2 a given subset holds.
- COCONUT's ~12% "class 3 coverage" and v3's +0.041 came from official class 2 molecules in COCONUT.
- Held-out class 3 drawn uniformly from the index is ~70% `enveda-180` (698 of 1000 in `splits_bigord.npz`): synthetic
  screening compounds, unlike the test. Read held-out class 3 on its non-`enveda-180` answers (§31).

| | held-out `splits.npz` | covered splits (`cov3`, `c1cov3`) | hidden test |
|---|---|---|---|
| class 1 | structures in 2+ libraries; never holds out `enveda-180` | `c1env`, `c1cov3`: from `enveda-180` | Enveda data; 14.0% of the score |
| class 2 | answer is in the index's own structures, so reachable | same | 12.6% of the score; partly reachable: the truth sits between the plain and the `C2_BANNED=1` eval (§16) |
| class 3 | 8% in COCONUT (16 of 200): its MRR is noise | 100% in COCONUT by construction: "quality" = ranking among candidates in the pool | 73.5% of the score = official class 2 + 3; mostly unreachable; real class 3 MRR was ~0.061 at v3/v4 |

- The class mix is exact: v1 and v2 have no class 3 capability, so the two LB scores pin it at 14.0 / 12.6 / 73.5 (§7–8).
- Real class 3 coverage by COCONUT is ~12%: §9 got 12.4% from real MRR 0.061 over a ranking quality of 0.492. That
  quality came from the leaky `fpmodel.pt`, and a leak-free one is lower, so the true figure is probably somewhat higher.
  Do not quote 12.4% as exact.
- A held-out delta therefore reaches the LB roughly as
  `0.140·Δclass1(Enveda split) + 0.126·Δclass2(C2_BANNED=1) + 0.735·coverage·Δquality(covered split)`, with
  coverage ~0.12. The covered splits' own `LB` column leaves coverage at 1 and so inflates class 3 deltas ~8x (see the
  section below). Predicted absolute levels are not calibrated; use deltas only.

**Conclusions in RESULTS.md that later sections overturned**

- §8 and §11 (early): "model work only touches class 1 and 2, class 3 is coverage-limited". The end of §11 already says
  the real lever is ranking against isomers of the same mass, and v12 confirmed it (+0.016). Which class paid for v12 is
  not known.
- §8 and §9 class 3 quality 0.492, and every covered-split class 2/3 number before §17, used the leaky `fpmodel.pt`.
- §9 reasoned that widening the mass window to 160 ppm mimics a bigger database. §10 showed that wrong: candidates in
  the 5 ppm window are isomers, the confusable kind.
- Pool expansion is closed. ChEMBL scored 0.187 against COCONUT's 0.213, PubChem was worse (§10–11): coverage
  elasticity +0.50, ranking-quality elasticity −0.86 against pool size. Do not add candidates.
- §17–23 numbers are at gamma 128, not the gamma they name (see "Where we stand"). §24 re-measures at gamma 32.

## Next steps (Raasta B, picked 2026-10-20)

Raasta B builds the public pipelines' missing components in our own code, for learning:
`docs/superpowers/specs/2026-10-20-raasta-b-design.md`. Every experiment carries the user's numeric prediction in
`LEARNING.md`, written before the data. LB probes are allowed with a prediction written first, and a probe enters final
selection only if it beats its pre-registered threshold.

1. **(Done, v17 0.264) Mass-shifted analogs**: plan `docs/superpowers/plans/2026-10-20-mass-shifted-analogs.md`,
   §32, `LEARNING.md` entries 1–3. The LB back-solves class 3 coverage to ~0.16 (0.07–0.25 with noise) against the
   0.12 rule below. That leans higher but does not overturn the rule.
2. **(Done, v18 0.257, did not transfer) Learned ranker with MetFrag-lite features**: plan
   `docs/superpowers/plans/2026-10-24-learned-ranker.md`, §33, `LEARNING.md` entries 4–6. MetFrag-lite separates
   exact isomers at AUC 0.734 and is the ranker's most distinctive feature. **Why it reversed** (entry 7, §33
   transfer test): on `splits.npz`'s 586 unseen molecules with the production ensemble, class 1 falls 0.938 → 0.812.
   The ranker prefers external candidates over strong library matches (`internal` flips 34 of 36 lost top-1s), and
   `splits.npz` class 1 has 5× the training splits' share of `top_spec ≥ 0.9`. Class 3 still gains. That predicts
   the LB's −0.007 with real class 3 near +0.012. Next, local and free: the ranker gated to `top_spec < 0.8`, read on
   that table (prediction first). Any retrained ranker must be read on a split built differently from its training
   splits.
3. Gated PubChem channel (fixed slots, not merged into the pool), then our own timsTOF forward model (spec 3–4). Read
   their held-out gains with entry 6 in mind: anything fitted on held-out lists needs an LB probe before belief.

Timing: v18's kernel ran 89 min and its scoring ~85 min. v17's ~2.5 h of scoring was most likely Kaggle's queue.

## Next steps (picked 2026-10-18, after v15; superseded by Raasta B above)

A change is worth an LB slot only if its coverage-scaled prediction is >= +0.005 (see "What the eval can and cannot
tell you"). With class 3 coverage ~0.12, a class 3 ranking change needs **+0.057 covered quality** to get there, so
ranking work alone rarely shows on the LB. Coverage multiplies every gain, which makes it the lever again.

1. **(Picked) Re-open pool expansion, now with a popularity prior.** §10–11 closed PubChem / ChEMBL because isomer
   crowding cut ranking quality (pool x4.6, quality x0.27), but that was before the prior (§28). The classic CASMI
   recipe is PubChem plus a popularity prior: obscure decoys in a crowded window get pushed down.
   - Measure on an **ordinary** (not covered) big split, class 3 1000: its class 3 MRR captures coverage and quality
     together, like the LB. Compare COCONUT vs COCONUT + PubChem, both with the prior.
   - Needs from the Linux box: `data/ext_pubchem/` (NP-filtered PubChem shards, ~8.4M rows, `pubchem.py --build`) and
     `data/ext_chembl/`. The shards hold key/smiles/mass/fp only, no metadata.
   - Popularity for PubChem: synonym counts from PubChem FTP (`Compound/Extras/CID-Synonym-filtered.gz`) joined through
     `CID-InChI-Key.gz` to key14. Public domain, like the shards. Build a `meta.npz` in the same format `prior.py`
     reads (`coconut_meta.py` is the template), or merge it with COCONUT's.
   - Risk: if the hidden class 3 answers are poorly documented, the prior hurts them (§30, 0–1 synonym stratum −0.05).
     A gain here would be large, though, so it will not hide in the LB's noise.
   - **Status 2026-10-18:** a popularity pool is built: `MIN_SYN=2 pubchem.py --build` -> `data/ext_pubpop2/` (15.96M,
     8.5 GB; compressed copy `data/ext_pubpop2_z/` 1.14 GB, uploaded as private dataset `casmi26-pubchem-pop2`).
     Per-CID counts: `pubchem_meta.py` -> `data/pubchem_meta/counts.npz`; `data/pubchem/CID-SMILES.gz` is on disk.
     On `splits_bigord.npz` (ordinary, 400/400/1000) it lifted class 3 coverage 7.6% -> 69% and class 3 0.036 -> 0.284
     (COCONUT prior w 0.5 gate 0.8; class 1 −0.017). **An adversarial review found most of that is `enveda-180`
     screening compounds** (Enamine deposits whose two PubChem synonyms are vendor IDs), which the test is not; no leak.
     Its estimate for the LB: about −0.01 (−0.03..+0.02). Pending before any v16: class 3 on the non-`enveda-180`
     answers, and the dilution of the COCONUT-covered answers on `splits_big.npz`. Kernel v16 files are prepared,
     not pushed.
   - **Result: v16 scored 0.243 vs v15's 0.257 (-0.014), 3.5x the predicted worst case.** Dilution of the
     covered hidden answers back-solves to ~6x what `splits_big.npz` measured (§31). Closed; do not give another
     expansion an LB slot on held-out evidence.
2. **(Closed 2026-10-18) Find out why v13 lost.** Every eval term said v13's config was better, and it scored −0.007.
   It cannot be diagnosed offline: the local test is the placeholder (Read first, 1), and the kernel logs we get are
   from that run, not the hidden one. Submitting the halves would only say which half cost the public LB, and
   neither half is near the shipped config (tau 0.0125 at raw 32 also lost on the leak-free split, 0.418 vs 0.432).
   The lesson stands: a large model or scoring change can win every eval term and still lose.
3. **Contrastive spectrum/molecule model, or test-like augmentation** (230-peak Enveda spectra). Big work, and it only
   touches class 3 ranking, which the LB discounts by 0.12. Do it after 1, when coverage makes ranking worth more.

## What has been tried and closed

- Why v13 (std norm + tau 0.0125) lost: not diagnosable offline; see Next steps 2.
- Pool expansion (PubChem, ChEMBL): dilution beats coverage (§10–11). Re-tried with a popularity-chosen PubChem
  pool and the prior (v16, §31): 0.243 vs 0.257. Held-out splits underprice the hidden test's dilution ~2.5-6x.
- Formula oracle ceiling ≈ +0.02; formula-consistency term captures ~5% of it (AUC 0.6) (§12).
- Reverse / hybrid cosine (`SIM=rev|hyb`): loses on held-out (§12).
- `SPEC_MASS` filter: no effect.
- Gamma 64, gated gamma: held-out gains, zero on LB (§15–16).
- More BCE epochs + cosine: val tanimoto up, class 3 flat (§18).
- Inference-side bit weighting, global or pool-local: no gain (§19).
- Fingerprint ceiling: 0.948 within the pool, so the gap is prediction, not bit collisions (§20).
- Ranking loss tau 0.1: worse in every column (§23).
- Ranking loss tau 0.025 / 0.0125 / 0.00625, lambda 3, 4 epochs: no better than tau 0.05 at gamma 32 (eval
  gaps within the §26 noise); lr 1e-4 worse at any gamma (§24).
- Ranking loss on the molecule-averaged prediction (`FP_MOL=4`, lambda 1 or 4): val mol-MRR below the per-spectrum
  loss; its +0.018 class 3 on 200 molecules disagrees with it (§25).

## What the eval can and cannot tell you (§15–16, §17, §21)

- **Scale class 3 by coverage before comparing with the LB.** The covered splits' `LB` column assumes class 3 is 100%
  in COCONUT; the hidden test is ~12%. Unscaled, every class 3 gain looks ~8x too big: v14's column said +0.044, the
  scaled prediction is +0.005 ± 0.003, the LB gave +0.002 (§30). (An earlier version of this file blamed better-documented
  held-out answers for the whole gap; the coverage scale explains v14 without that.)
- **The LB is noisy for a ranking change: ±0.003–0.005.** Only the ~35 covered hidden class 3 molecules carry it (fewer if
  the public LB is a subset), and a change moves a third of them by ~0.2 each (§30). An LB delta under ~0.005 is noise in
  either direction; submit a change to learn something only if its scaled prediction is ≥ +0.005.
- **How documented the answer is matters for anything that favours known structures.** The synonym prior gains +0.10–0.15
  on answers with 5+ COCONUT synonyms and loses −0.05 on 0–1 (`STRATA=1`, §30). The hidden test's mix is not identifiable
  from one LB point. Do not raise the prior's weight on the eval's word.
- **v13 is still unexplained**: on `splits_big.npz` its config beats v12's in every term and every documentation stratum
  (scaled +0.002), and it scored −0.007, ~3 SE off. Something about the hidden test hurts the std-normalised term or the
  sharp-tau model that no held-out split shows.
- The held-out eval over-rewards scoring-weight changes: the gain is **class 2 whose answer sits in the index's own
  structures**, which does not reach the LB. With class 2 answers banned (`C2_BANNED=1`) gamma variants tie, like the LB.
- **Rule before any submission:** a change must improve the `C2_BANNED=1` eval, or class 1 / class 3, not just ordinary
  class 2, and class 1 must not fall. Absolute predicted LB is not calibrated (0.145–0.188 vs 0.236); use deltas only.
- Class 3 quality needs a **covered** split and a **leak-free** model: `splits_c1cov3.npz` + `fpmodel_c1cov3.pt`
  (BCE baseline 0.397 and leaky 0.557 at gamma 128, §17; the §21 ranking baseline is 0.407 at the shipped gamma 32, §26). Class 3 of `splits.npz` is 16 covered molecules: noise.
- **Retraining the same config moves 200-molecule class 3 by up to 0.025 and its LB by up to 0.019** (float rounding
  alone, §26); val pool-MRR / mol-MRR moves 0.001. On `splits_big.npz` (1000 class 3) four seeds span **0.001** in class 3
  and 0.0009 in the `C2_BANNED=1` column (§30): use that split.
- The v12 estimate from held-out deltas was +0.006..+0.011; the real gain was +0.016. `C2_BANNED=1` delta (+0.021)
  was the closer predictor.

## Models (`data/`, git-ignored; each ~69 MB)

| file | trained excluding | what it is |
|---|---|---|
| `fpmodel.pt` | `splits.npz` rows | v9–v11 shipped model; init for the shipped fine-tune |
| `fpmodel_rank.pt` | `splits.npz` rows | **v12/v14 shipped** (fpmodel.pt + ranking loss 2 ep) |
| `fpmodel_rank_s{1,2,3}.pt` | `splits.npz` rows | §29, same config at `FP_SEED` 1–3; v15 ensemble members |
| `fpmodel_rank_t0125.pt` | `splits.npz` rows | §24, tau 0.0125 fine-tune of fpmodel.pt; v13 shipped (0.245) |
| `fpmodel_rank_best4.pt` | `splits.npz` rows | §27, tau 0.00625 4 ep fine-tune of fpmodel.pt; not shipped |
| `fpmodel_c1cov3.pt` | `splits_c1cov3.npz` rows | leak-free BCE baseline; **init for every tuning run** |
| `fpmodel_c1cov3_rank.pt` | `splits_c1cov3.npz` rows | §21 result (tau 0.05, 2 ep): class 3 0.4486 |
| `fpmodel_c1cov3_t10.pt` | `splits_c1cov3.npz` rows | §23, tau 0.1 (worse); `_e12` is §18 (12 BCE epochs) |
| `fpmodel_c1cov3_{mac,t025,t0125,t00625,t025_l3,best4,lr4}.pt` | `splits_c1cov3.npz` rows | §24 sweep, Mac only; `mac` = §21 rerun |
| `fpmodel_big.pt` | `splits_big.npz` rows | leak-free BCE base for the big split (6 ep, 26 min on the Mac) |
| `fpmodel_big_rank_s{0,1,2,3}.pt` | `splits_big.npz` rows | v12's config on the big split, 4 seeds (§30) |
| `fpmodel_big_t0125.pt` | `splits_big.npz` rows | v13's model (tau 0.0125) on the big split (§30) |
| `fpmodel_c1cov3_s{1,2,3}.pt` | `splits_c1cov3.npz` rows | §29, baseline config at `FP_SEED` 1–3 |
| `fpmodel_c1cov3_{mol4,mol4_l4}.pt` | `splits_c1cov3.npz` rows | §25, molecule-averaged ranking loss (not better) |
| `fpmodel_c1env.pt` | `splits_c1env.npz` rows | §16; not used since §17 |

A model is only valid on the split whose rows it excluded. `fpmodel.pt` saw 90% of `splits_cov3.npz` queries and
95% of `splits_c1env.npz` queries.

## Training cheat-sheet (`fingerprint.py`)

```bash
# BCE model for a split (~22 min/epoch, 6 epochs)
SPLITS=data/<split>.npz FP_MODEL=data/fpmodel_<name>.pt FP_HIDDEN=2048 FP_DEFECT=1 \
  MIN_AVAIL_MB=2500 ../guard.sh python3 fingerprint.py --train --holdout

# ranking-loss fine-tune (~50-60 min/epoch on the Linux box, ~6.4 on the M4 Mac since §26)
SPLITS=data/splits_c1cov3.npz FP_INIT=data/fpmodel_c1cov3.pt FP_MODEL=data/fpmodel_c1cov3_<name>.pt \
  FP_RANK=1 FP_TAU=0.05 FP_COSINE=1 FP_HIDDEN=2048 FP_DEFECT=1 MIN_AVAIL_MB=2500 \
  ../guard.sh python3 fingerprint.py --train --holdout --epochs=2
```
Knobs: `FP_RANK` (lambda), `FP_TAU`, `FP_RANK_K` (candidates per spectrum, 64), `FP_LR` (1e-3), `FP_COSINE`,
`FP_MOL` (rank molecule-averaged groups, §25; 0), `FP_SEED` (order/negatives/dropout, not val; 0), `FP_CKPT_MIN` (resumable checkpoint interval, 5; 0 = off),
`FP_INIT` (asserts same bits/width/defect, so same split), `--epochs`, `--limit`. The log prints `epoch 0 pool-MRR`
for a fine-tune: it must read **0.5313** for `fpmodel_c1cov3.pt` (0.5073 for `fpmodel.pt` on `splits.npz`), then
`mol-MRR` (mean over up to 4 spectra per structure, as the eval queries; 0.6025 for `fpmodel_c1cov3.pt`).
Weights are saved every epoch. **A killed run resumes**: rerun the same command and it continues from
`FP_MODEL.state` (at most `FP_CKPT_MIN` lost; bit-identical to an uninterrupted run). The state file is removed at
the end; a stale one from a different run trips an assert, delete it to start over.

## Eval cheat-sheet

Splits (`split.py`, env `SPLIT_OUT`):
- `data/splits.npz` — ordinary.
- `data/splits_cov3.npz` — `C3_COVERED=1`, class 3 all in COCONUT (leaky with `fpmodel.pt`).
- `data/splits_c1env.npz` — `C1_LIB=enveda-180`, class 1 from Enveda spectra (hidden test is Enveda data).
- `data/splits_c1cov3.npz` — both of the above, 200 per class.
- `data/splits_big.npz` — the same at 400 / 400 / 1000 (`N_PER_CLASS=400,400,1000`). **The one to use**, with
  `STRATA=1`; models `fpmodel_big*.pt`.

`dilute.py` env knobs: `SPLITS`, `FP_MODEL`, `FRACS=1` (full COCONUT), `GAMMA`, `GATES`+`GAMMA_HI` (default gate 0 =
no gate; it was 2 = always `GAMMA_HI` until 2026-10-17), `DELTAS`,
`FORMULA_ORACLE=1`, `TRUE_ADDUCT` (default 1), `SPEC_MASS`, `C2_BANNED=1`, `GAMMAS`/`NORMS` sweeps,
`PRIOR`+`PRIOR_WS`+`PRIOR_GATES`+`PRIOR_FILL` (§28). Search: `SIM=cos|rev|hyb`. A comma-separated `FP_MODEL` is an
ensemble (mean probability). Val MRR of any model/ensemble: `val_mrr.py`.
The merged pool cache is one directory per pool under `data/pool/` (§28); concurrent runs are safe since.

Standard eval (FRACS=1: 1–2 min, 5–6 GB peak; always run with and without `C2_BANNED=1`):
```bash
SPLITS=data/splits_big.npz FP_MODEL=data/fpmodel_big_<name>.pt FRACS=1 GAMMA=32 STRATA=1 \
  MIN_AVAIL_MB=2500 ../guard.sh python3 dilute.py            # add C2_BANNED=1 for the second run
```

## Kaggle submit workflow (~66 min kernel + ~70 min scoring at v15; v17: 84 min + ~2.5 h; v18: 89 min + ~85 min)

Claude Code's auto mode refuses `kaggle datasets version`, `kaggle kernels push` and `kaggle competitions submit`
as writes to a connected app; the user runs them with `!`. Status polls and `kaggle kernels output` are read-only.

1. Copy `fpmodel.pt` (the new shipped model) and any changed `.py` to `kernel/code/`.
2. `kaggle datasets version -p kernel/code -m "..." -q`, poll `kaggle datasets status jagjeetsingh23/casmi26-molecule-id-code`.
3. `kaggle kernels push -p kernel`, poll `kaggle kernels status`.
4. `kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k jagjeetsingh23/casmi26-search-analog -v N -f submission.csv -m "..."`
   (`submission.csv` = `kaggle kernels output`).
5. Poll `kaggle competitions submissions` (line 3 = latest). A completed kernel only proves it ran; the LB is the result.

Local `test.parquet` is a placeholder copied from train: use it only to check `submit()` runs.

## Machine notes

Mac mini (Apple M4, 16 GB; the training machine since 2026-10-17):
- Venv at the repo root: `../.venv/bin/python` (Python 3.12, numpy 2.5, rdkit 2026.3.3, torch 2.14). CPU only.
- Ranking-loss epoch 6.4 min (§26; was 12.5); `dilute.py` FRACS=1 ~25 s. Torch threads 4 is fastest.
- No battery, so a power cut restarts it (`autorestart` is on; no UPS). Keep job scripts in `data/jobs/`,
  not `/tmp` (wiped on reboot), and run everything under `caffeinate -ims`: without it the Mac idle-sleeps
  mid-run (17 min lost on 2026-10-17).

Linux laptop (15.6 GB, swap nearly full):
- One heavy job at a time, always under `../guard.sh` with `MIN_AVAIL_MB=2500`; **never weaken the guard** after a
  kill, free memory or shrink the job. Training peaks at ~2.4 GB; `dilute.py` at ~5–6 GB, so it needs ~8 GB available to start (the guard kills below 2.5 GB free).
  Firefox and any dev server must be closed: they cost two runs (2026-10-13).
- **Closing the lid suspends the laptop, and `systemd-inhibit` does not stop it** (logind ignores inhibitors for the
  lid by default). Needs `HandleLidSwitch=ignore` (sudo, theirs to set) or the lid open.
- `rg` sometimes resolves to grep in this shell: use `/usr/bin/rg`.
- Never `git add -A`; stage by path. Competition data is never committed. Pushing needs the personal gh account:
  `gh auth switch --user jagjeet-singh-23`, push, `gh auth switch --user jagjeet-singh-ambak`.

## Shell and harness traps

- `pkill -f <pattern>` kills the calling shell (exit 144) when the pattern also appears in the command line that runs
  it. It happened three times. Find the PID with `pgrep -af` and `kill <pid>`.
- `ps -e` returned nothing on the Linux box on 2026-09-28 (a shell wrapper swallowed it) and worked on 2026-10-13. That
  broke an "is the old job still running?" check and three duplicate downloaders were started. Use `pgrep -af <pat>`,
  and `fuser <file>` to name every writer of a file.
- zsh aborts the whole command line on an unmatched glob (`ls data/eval_*` before the files exist).
- `;` does not stop a chain when a step fails: a killed run was followed by the next run starting on top of it.
- `!` commands in Claude Code have no TTY, so `sudo` fails there. Run sudo in a real terminal. Never hand a sudo password
  to the assistant.
- Claude Code reaps background shells when the machine is low on memory ("stopped because the system is running low on
  memory"). That says nothing about the command. Do not restart it unprompted.
- `tail -n 1 a b` (not `tail -1`) when given two files. `sleep` in a chained command is blocked by the harness: wait with
  `until <check>; do sleep 15; done` in a background command.
- Linux box: the shell prints a zoxide warning before every command's output. Harmless.

## Working agreements

- Commit and push only when asked, staged by path. Confirm before a Kaggle submission unless the user has said to go ahead.
- Never weaken `guard.sh` after a kill; free memory or shrink the job. Never kill a process the assistant did not start
  without being told which one.
- Before formatting or wiping anything, name the device and confirm it is the removable one.
- The user writes in Hinglish.
