# Plan — tune the ranking loss on the Mac (written 2026-10-17)

v12 scored LB 0.252 (§21–22). The ranking loss is untuned and training moves from the Linux laptop (memory
guard kills, lid suspends) to an Apple Silicon Mac with 16 GB+. State and cheat-sheets: HANDOFF.md.

## 0. Move to the Mac (~2 GB, none of it is in git)

Copy these into `MoleculeID-from-Mass-Spectra/data/` (same layout):

| path | size | why |
|---|---|---|
| `index/` (21 shards) | 1.3 GB | spectra + peaks (`library.load`) |
| `structures.npz` | 146 MB | candidate fingerprints |
| `key14.npz` | 25 MB | structure keys |
| `ext/` (COCONUT shards) | 260 MB | the external candidate pool |
| `splits_c1cov3.npz`, `splits.npz` | 0.2 MB | held-out splits |
| `fpmodel_c1cov3.pt` | 69 MB | init for every tuning run |
| `fpmodel.pt` | 69 MB | init for the shipped fine-tune (step 3) |
| `fpmodel_c1cov3_rank.pt` | 69 MB | §21 reference to compare against (optional) |

Not needed: `train.parquet` (only `library.build` reads it), `ext_chembl/`, `ext_pubchem/`, `pubchem/`, `chembl/`.
Do not copy `pool/`: its cache signature contains absolute shard paths, so on the Mac it is stale and the first run
rebuilds it from `ext/` (minutes, then cached).

**Done on 2026-10-17: all of the above is on a pen drive (FAT32, label `CASMI`, folder `casmi-data/`, 38 files, 2.0 GB,
checksums in `SHA256SUMS`, steps in its `README.txt`).** On the Mac:

```bash
mkdir -p data
rsync -rtP /Volumes/CASMI/casmi-data/ data/ --exclude README.txt --exclude SHA256SUMS
(cd data && shasum -a 256 -c /Volumes/CASMI/casmi-data/SHA256SUMS)    # every line must say OK
```

Without the drive, over the network (needs SSH on the Linux box, which is off by default):

```bash
rsync -avP <linux-host>:Coding/Personal/ML/MoleculeID-from-Mass-Spectra/data/{index,ext} data/
rsync -avP <linux-host>:Coding/Personal/ML/MoleculeID-from-Mass-Spectra/data/{structures.npz,key14.npz,splits.npz,splits_c1cov3.npz,fpmodel.pt,fpmodel_c1cov3.pt,fpmodel_c1cov3_rank.pt} data/
```
The shipped v12 weights are also in the Kaggle dataset `jagjeetsingh23/casmi26-molecule-id-code` (`fpmodel.pt` there is v12).

Environment: Python 3.12 venv with `numpy>=2` (`np.bitwise_count`), `torch`, `pyarrow`, `pandas`, `kaggle`, and
**`rdkit==2026.3.3`** (the competition pins it). `fingerprint.py` runs on CPU only; do not switch to MPS, it is untested.
`../guard.sh` has a macOS branch (`vm_stat`).

## 1. Check the Mac reproduces the Linux box before spending hours

1. `python3 fingerprint.py --selftest` → `fingerprint self-checks passed`.
2. Start the §21 run (below) and read the first log line: **`epoch 0 pool-MRR 0.5313`** (±0.01 is fine across
   platforms). If it is far off, data or model files are wrong; stop.
3. Note the minutes per epoch (Linux: ~50). Plan the rest of the sweep from that number, not from this document.
4. Keep the Mac awake and on AC: `caffeinate -ims <command>`. Closing the lid still sleeps a laptop; keep it open or
   use `sudo pmset -a disablesleep 1` (undo with `0`). Not tested by me.

## 2. Tuning runs (leak-free split, one at a time)

Common env, from `MoleculeID-from-Mass-Spectra/`:
```bash
export SPLITS=data/splits_c1cov3.npz FP_INIT=data/fpmodel_c1cov3.pt FP_RANK=1 FP_COSINE=1 \
       FP_HIDDEN=2048 FP_DEFECT=1 MIN_AVAIL_MB=2500
FP_MODEL=data/fpmodel_c1cov3_<name>.pt <knob overrides> ../guard.sh python3 fingerprint.py --train --holdout --epochs=2
FP_MODEL=data/fpmodel_c1cov3_<name>.pt FRACS=1 GAMMA=32 ../guard.sh python3 dilute.py                # then again with C2_BANNED=1
```
Baseline to beat (§21, tau 0.05, lambda 1, lr 1e-3, K 64, 2 ep): **val pool-MRR 0.630, class 1 0.8869, class 3 0.4486,
`C2_BANNED=1` LB 0.4583**. Closed: tau 0.1 (0.601 / 0.8630 / 0.4352 / 0.4456).

| # | name | knobs | reason |
|---|---|---|---|
| 1 | t025 | `FP_TAU=0.025` | tau 0.1 got worse, so try the other direction |
| 2 | l3 | `FP_RANK=3` | cross-entropy is ~30× the BCE term at lambda 1; check whether BCE is still anchoring |
| 3 | l03 | `FP_RANK=0.3` | the other side of lambda |
| 4 | lr4 | `FP_LR=1e-4` | a fine-tune from a trained model at a 1e-3 peak may undo some of it |
| 5 | k128 | `FP_RANK_K=128` | pools above 64 candidates are cut at random; median is ~50, the tail is not |
| 6 | best4 | best settings from 1–5 combined, `--epochs=4` | pool-MRR was still rising at epoch 2 |

Not every row needs running: do rows 1 and 4 first, then follow whichever moves. Stop a row early if its epoch-1
val pool-MRR is below the baseline's epoch-1 value (0.615).

**Pick by this rule** (HANDOFF): `C2_BANNED=1` LB not below 0.4583 and class 1 not more than 0.01 below 0.8869; among
those, higher class 3, and under 0.015 apart (200 molecules) prefer the higher val pool-MRR. Six variants on one 200-molecule
eval can pick noise, so confirm the winner on a second look: val pool-MRR must point the same way.
If nothing beats the baseline, keep v12 and stop.

Possible bigger step, needs code: inference averages the predicted fingerprint over a molecule's spectra
(`fpsearch.query_fp`), training ranks each spectrum alone. Rank on the molecule-averaged prediction instead.

## 3. Ship a winner (only if one passes)

```bash
SPLITS=data/splits.npz FP_INIT=data/fpmodel.pt FP_MODEL=data/fpmodel_rank2.pt FP_RANK=1 <winner knobs> FP_COSINE=1 \
  FP_HIDDEN=2048 FP_DEFECT=1 MIN_AVAIL_MB=2500 ../guard.sh python3 fingerprint.py --train --holdout --epochs=<winner>
```
Epoch-0 pool-MRR must read 0.5073. Then eval `fpmodel_rank.pt` and `fpmodel_rank2.pt` on `splits.npz`
(`FRACS=1 GAMMA=32`, with and without `C2_BANNED=1`). Gate: `C2_BANNED=1` LB above 0.1677 (v12's model), class 1 not lower.
Then the Kaggle workflow in HANDOFF.md: `kaggle datasets download -d jagjeetsingh23/casmi26-molecule-id-code -p kernel/code --unzip`,
replace `fpmodel.pt`, `fingerprint.py`, `fpsearch.py`, copy `kernel/` (`kernel-metadata.json`, `casmi_submit.py`) from the repo,
version the dataset, push the kernel, submit. **Ask before spending the day's submission.**

## 4. Housekeeping

- Models and logs stay out of git. Push only `RESULTS.md` (add a section per sweep: a table like §21/§23), `HANDOFF.md`
  (state, best LB, model table) and code, staged by path, never `git add -A`.
- Push needs the personal gh account: `gh auth switch --user jagjeet-singh-23`, `git push origin main`,
  `gh auth switch --user jagjeet-singh-ambak`.
- Numbers in this repo were measured on Linux; the Mac may differ slightly. Compare runs only against a baseline
  measured on the same machine if the epoch-0 check above is off.
- Deadline: entry 7 Dec, final submission 14 Dec 2026.
