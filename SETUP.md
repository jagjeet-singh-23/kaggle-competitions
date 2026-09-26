# Setup

Developed on Linux (Ubuntu, Python 3.10). Runs unchanged on macOS Apple Silicon.
Everything here is CPU-only — no GPU is used or needed.

## Install

```bash
git clone https://github.com/jagjeet-singh-23/kaggle-competitions.git
cd kaggle-competitions
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Kaggle credentials go in `~/.kaggle/kaggle.json` (`chmod 600`). They are read at
runtime and never committed.

```bash
./fetch_data.sh                 # all three competitions, ~33 GB
./fetch_data.sh molecule        # or just one: molecule | birdclef | lanl
```

The rules for each competition must be accepted on kaggle.com first or every
request 403s.

## macOS caveats

There are only two, and one is fatal on the wrong hardware.

**Apple Silicon is required for BirdCLEF.** `ai-edge-litert` publishes
`macosx_12_0_arm64` wheels only — there is no Intel-Mac build. LANL and MoleculeID
have no such constraint and run on either. On an Intel Mac, either run BirdCLEF on
Linux or drive BirdNET through `tensorflow`'s own `tf.lite.Interpreter`, which takes
the same `experimental_preserve_all_tensors` argument.

**`lightgbm` needs no `brew install libomp`.** The wheel vendors its own OpenMP
runtime. Only a source build would need brew.

Nothing else is platform-specific. `fetch_data.sh` and `guard.sh` are POSIX sh/bash
with a `uname` branch for reading free memory, and the only Linux-ism in the Python
was `ru_maxrss`, which reports kilobytes on Linux and bytes on macOS — handled in
`MoleculeID-from-Mass-Spectra/library.py:rss_gb`.

## Resource requirements

| Step | Peak RAM | Disk | Wall clock |
|---|---|---|---|
| `MoleculeID/library.py` (build index) | 4.9 GB | 1.3 GB out, 2.9 GB in | ~25 min |
| `MoleculeID/split.py` | 0.2 GB | 73 KB | ~1 min |
| `MoleculeID/search.py` | ~2.5 GB | — | not yet measured |
| `BirdCLEF/embed_train.py` | < 1 GB | 251 MB out, 16 GB in | ~4 h |
| `BirdCLEF/rung1c.py` | < 1 GB | — | ~20 min |
| `LANL/*.py` | ~3 GB | 9.1 GB in | minutes |

16 GB of RAM is comfortable. The machine this was built on has 15 GB and froze twice
early on, which is why `guard.sh` exists:

```bash
MIN_AVAIL_MB=2500 ./guard.sh python3 -u MoleculeID-from-Mass-Spectra/library.py
```

It polls free memory and kills the child — not the machine — if the job runs away.
Run one heavy job at a time.

## Skipping the four-hour step

`BirdCLEF/embed_train.py` extracts BirdNET embeddings for 333 hours of focal audio
and takes about four hours. The output is only 251 MB, and it is machine-independent.
Copying it to a second machine beats re-running it:

```bash
rsync -a BirdCLEF/data/emb_train/ BirdCLEF/data/labelled.npz  mac:~/kaggle-competitions/BirdCLEF/data/
```

Then `rung1c.py` and `head.py` run in minutes against the 16 GB of source audio never
being downloaded at all.

## BirdNET weights

Not on PyPI and not in this repo. Download V2.4 (MIT licence, kahst/BirdNET-Analyzer)
and put the FP32 tflite plus the labels file in `BirdCLEF/models/`:

```
BirdCLEF/models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite
BirdCLEF/models/BirdNET_GLOBAL_6K_V2.4_Labels.txt
```

Source: https://tuc.cloud/index.php/s/886x39f5N3sdsAM/download/V2.4.zip

## Verifying the install

Every non-trivial module has a self-check that needs no competition data:

```bash
python3 MoleculeID-from-Mass-Spectra/metric.py          # MRR@25 vs the rules' examples
python3 MoleculeID-from-Mass-Spectra/search.py --selftest
python3 BirdCLEF/rung1c.py --selftest
```

`metric.py` is the one to trust on a new machine: it checks the RDKit tautomer
canonicalisation against the glucose example from the competition rules, so a wrong
RDKit version fails loudly rather than silently changing your score.

## Data is not in this repo

33 GB, and redistributing it would breach the competition rules and the CC BY-NC /
CC BY-NC-SA licences it ships under. `.gitignore` excludes `data/`, `models/`,
`*.npy`, `*.parquet`, `*.zip`, `submission.csv` and `kaggle.json`. Use
`fetch_data.sh` to get it.
