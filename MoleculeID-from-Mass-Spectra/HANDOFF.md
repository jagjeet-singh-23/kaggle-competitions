# Handoff — CASMI 2026 (as of 2026-10-17)

Read RESULTS.md §11–24 first; this file is the short version plus how to run things.
The next experiments, and how to run them on the Mac, are in **PLAN_MAC.md**.

## Where we stand

- Competition: `enveda-CASMI26-molecule-id-mass-spectra`, MRR@25 on InChIKey14, closes 14 Dec 2026.
- **Best LB 0.252 (v12)**: search + analog + predicted-fingerprint tanimoto, gamma 32, one adduct window per
  molecule (§14), with a fingerprint model fine-tuned with a **ranking loss over the 5 ppm candidate pool**
  (§21–22). The loss is what moved it: +0.016 over v9/v11 (0.236).
- Class mix pinned (§7): LB ≈ 0.140·c1 + 0.126·c2 + 0.735·c3. Class 3 is 73.5% of the score.
- Kaggle kernel `jagjeetsingh23/casmi26-search-analog` latest version is **v12**. `kernel/code/` (git-ignored) holds
  the shipped files and equals the repo's `fingerprint.py` / `fpsearch.py` at the time of v12 (no gate, gamma 32);
  `kernel/code/fpmodel.pt` = `data/fpmodel_rank.pt`.
- Ranking loss swept on the Mac (§24): at the shipped gamma 32 nothing beats tau 0.05 / lambda 1 / lr 1e-3 / 2 epochs,
  so v12 stays. Sharper tau wins only at gamma >= 64, which held-out over-rewards (§15–16); untested on the LB.
- **Before 2026-10-17 every `dilute.py` number without an explicit `GATES` was gamma 128, not the `GAMMA` given**
  (§17–23; default gate bug, fixed). Re-measure before comparing against an old number.

| ver | change | LB |
|---|---|---|
| v4 | mass-defect features, hidden 2048 | 0.213 |
| ChEMBL pool | bigger candidate pool | 0.187 (pool expansion closed) |
| v9 | adduct-window bug fix | 0.236 |
| v10 | gamma 64 | 0.235 |
| v11 | gated gamma (0.7 / 128) | 0.236 |
| v12 | fpmodel + in-pool ranking loss (§21–22), gamma 32, no gate | **0.252** |

## What has been tried and closed

- Pool expansion (PubChem, ChEMBL): dilution beats coverage (§10–11).
- Formula oracle ceiling ≈ +0.02; formula-consistency term captures ~5% of it (AUC 0.6) (§12).
- Reverse / hybrid cosine (`SIM=rev|hyb`): loses on held-out (§12).
- `SPEC_MASS` filter: no effect.
- Gamma 64, gated gamma: held-out gains, zero on LB (§15–16).
- More BCE epochs + cosine: val tanimoto up, class 3 flat (§18).
- Inference-side bit weighting, global or pool-local: no gain (§19).
- Fingerprint ceiling: 0.948 within the pool, so the gap is prediction, not bit collisions (§20).
- Ranking loss tau 0.1: worse in every column (§23).
- Ranking loss tau 0.025 / 0.0125 / 0.00625, lambda 3, 4 epochs: worse than tau 0.05 at gamma 32; lr 1e-4 worse at
  any gamma (§24).

## What the eval can and cannot tell you (§15–16, §17, §21)

- The held-out eval over-rewards scoring-weight changes: the gain is **class 2 whose answer sits in the index's own
  structures**, which does not reach the LB. With class 2 answers banned (`C2_BANNED=1`) gamma variants tie, like the LB.
- **Rule before any submission:** a change must improve the `C2_BANNED=1` eval, or class 1 / class 3, not just ordinary
  class 2, and class 1 must not fall. Absolute predicted LB is not calibrated (0.145–0.188 vs 0.236); use deltas only.
- Class 3 quality needs a **covered** split and a **leak-free** model: `splits_c1cov3.npz` + `fpmodel_c1cov3.pt`
  (baseline 0.397; leaky model on the same split said 0.557). Class 3 of `splits.npz` is 16 covered molecules: noise.
- Differences under ~0.015 in class 3 on 200 molecules are noise; val pool-MRR (2,000 spectra) is the tie-breaker.
- The v12 estimate from held-out deltas was +0.006..+0.011; the real gain was +0.016. `C2_BANNED=1` delta (+0.021)
  was the closer predictor.

## Models (`data/`, git-ignored; each ~69 MB)

| file | trained excluding | what it is |
|---|---|---|
| `fpmodel.pt` | `splits.npz` rows | v9–v11 shipped model; init for the shipped fine-tune |
| `fpmodel_rank.pt` | `splits.npz` rows | **v12 shipped** (fpmodel.pt + ranking loss 2 ep) |
| `fpmodel_c1cov3.pt` | `splits_c1cov3.npz` rows | leak-free BCE baseline; **init for every tuning run** |
| `fpmodel_c1cov3_rank.pt` | `splits_c1cov3.npz` rows | §21 result (tau 0.05, 2 ep): class 3 0.4486 |
| `fpmodel_c1cov3_t10.pt` | `splits_c1cov3.npz` rows | §23, tau 0.1 (worse); `_e12` is §18 (12 BCE epochs) |
| `fpmodel_c1cov3_{mac,t025,t0125,t00625,t025_l3,best4,lr4}.pt` | `splits_c1cov3.npz` rows | §24 sweep, Mac only; `mac` = §21 rerun |
| `fpmodel_c1env.pt` | `splits_c1env.npz` rows | §16; not used since §17 |

A model is only valid on the split whose rows it excluded. `fpmodel.pt` saw 90% of `splits_cov3.npz` queries and
95% of `splits_c1env.npz` queries.

## Training cheat-sheet (`fingerprint.py`)

```bash
# BCE model for a split (~22 min/epoch, 6 epochs)
SPLITS=data/<split>.npz FP_MODEL=data/fpmodel_<name>.pt FP_HIDDEN=2048 FP_DEFECT=1 \
  MIN_AVAIL_MB=2500 ../guard.sh python3 fingerprint.py --train --holdout

# ranking-loss fine-tune (~50-60 min/epoch on the Linux box, ~12.5 on the M4 Mac)
SPLITS=data/splits_c1cov3.npz FP_INIT=data/fpmodel_c1cov3.pt FP_MODEL=data/fpmodel_c1cov3_<name>.pt \
  FP_RANK=1 FP_TAU=0.05 FP_COSINE=1 FP_HIDDEN=2048 FP_DEFECT=1 MIN_AVAIL_MB=2500 \
  ../guard.sh python3 fingerprint.py --train --holdout --epochs=2
```
Knobs: `FP_RANK` (lambda), `FP_TAU`, `FP_RANK_K` (candidates per spectrum, 64), `FP_LR` (1e-3), `FP_COSINE`,
`FP_INIT` (asserts same bits/width/defect, so same split), `--epochs`, `--limit`. The log prints `epoch 0 pool-MRR`
for a fine-tune: it must read **0.5313** for `fpmodel_c1cov3.pt` (0.5073 for `fpmodel.pt` on `splits.npz`).
A checkpoint is saved every epoch (weights only: no optimizer state, no epoch counter).

## Eval cheat-sheet

Splits (`split.py`, env `SPLIT_OUT`):
- `data/splits.npz` — ordinary.
- `data/splits_cov3.npz` — `C3_COVERED=1`, class 3 all in COCONUT (leaky with `fpmodel.pt`).
- `data/splits_c1env.npz` — `C1_LIB=enveda-180`, class 1 from Enveda spectra (hidden test is Enveda data).
- `data/splits_c1cov3.npz` — both of the above. **The one to use.**

`dilute.py` env knobs: `SPLITS`, `FP_MODEL`, `FRACS=1` (full COCONUT), `GAMMA`, `GATES`+`GAMMA_HI` (default gate 0 =
no gate; it was 2 = always `GAMMA_HI` until 2026-10-17), `DELTAS`,
`FORMULA_ORACLE=1`, `TRUE_ADDUCT` (default 1), `SPEC_MASS`, `C2_BANNED=1`. Search: `SIM=cos|rev|hyb`.

Standard eval (FRACS=1: 1–2 min, 5–6 GB peak; always run with and without `C2_BANNED=1`):
```bash
SPLITS=data/splits_c1cov3.npz FP_MODEL=data/fpmodel_c1cov3_<name>.pt FRACS=1 GAMMA=32 \
  MIN_AVAIL_MB=2500 ../guard.sh python3 dilute.py            # add C2_BANNED=1 for the second run
```

## Kaggle submit workflow (~66 min kernel + ~70 min scoring)

1. Copy `fpmodel.pt` (the new shipped model) and any changed `.py` to `kernel/code/`.
2. `kaggle datasets version -p kernel/code -m "..." -q`, poll `kaggle datasets status jagjeetsingh23/casmi26-molecule-id-code`.
3. `kaggle kernels push -p kernel`, poll `kaggle kernels status`.
4. `kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k jagjeetsingh23/casmi26-search-analog -v N -f submission.csv -m "..."`
   (`submission.csv` = `kaggle kernels output`).
5. Poll `kaggle competitions submissions` (line 3 = latest). A completed kernel only proves it ran; the LB is the result.

Local `test.parquet` is a placeholder copied from train: use it only to check `submit()` runs.

## Machine notes

Linux laptop (15.6 GB, swap nearly full):
- One heavy job at a time, always under `../guard.sh` with `MIN_AVAIL_MB=2500`; **never weaken the guard** after a
  kill, free memory or shrink the job. Training peaks at ~2.4 GB; `dilute.py` at ~5–6 GB, so it needs ~8 GB available to start (the guard kills below 2.5 GB free).
  Firefox and any dev server must be closed: they cost two runs (2026-10-13).
- **Closing the lid suspends the laptop, and `systemd-inhibit` does not stop it** (logind ignores inhibitors for the
  lid by default). Needs `HandleLidSwitch=ignore` (sudo, theirs to set) or the lid open.
- `rg` sometimes resolves to grep in this shell: use `/usr/bin/rg`.
- Never `git add -A`; stage by path. Competition data is never committed. Pushing needs the personal gh account:
  `gh auth switch --user jagjeet-singh-23`, push, `gh auth switch --user jagjeet-singh-ambak`.
