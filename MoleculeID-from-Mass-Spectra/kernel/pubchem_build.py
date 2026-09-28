"""Build the PubChem candidate pool on Kaggle, where the NCBI link is not throttled.

This notebook is not a submission. It runs with internet enabled, fetches PubChem's
CID-SMILES.gz, and writes the sharded pool to /kaggle/working so the output can be
attached as a dataset to both the held-out evaluation and the submission kernel.

It exists because NCBI rate-limits a home connection hard: the same 1.49 GB file came
down at 35 KB/s from here after an hour of use, where Kaggle pulls it in a couple of
minutes and the finished shards come back at 6.7 MB/s. Building where the data is
also avoids uploading 5 GB of shards that were derived from a 1.5 GB input.

Settings this notebook needs, which are not in the file:
  * Internet: on (it is off by default for competition notebooks)
  * Attached dataset: jagjeetsingh23/casmi26-molecule-id-code
  * Persistence: none needed; the output is what matters

    TARGET rows -> NP-likeness threshold -> shards, about 40 minutes.
"""
import os
import subprocess
import sys
import time

print("/kaggle/input contains:", flush=True)
for dirpath, dirnames, files in os.walk("/kaggle/input"):
    if dirpath.count("/") - 2 <= 2:
        print(f"  {dirpath}  ({len(files)} files)", flush=True)


def find(marker, root="/kaggle/input"):
    for dirpath, _, files in os.walk(root):
        if marker in files:
            return dirpath
    raise FileNotFoundError(f"{marker} not found anywhere under {root}")


CODE = find("pubchem.py")

# Pinned for the same reason the submission kernel pins it: the metric is an InChIKey
# comparison after tautomer canonicalisation, so a different RDKit produces different
# keys, and the pool's keys have to be the ones the search will look up.
try:
    import rdkit  # noqa: F401
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "rdkit==2026.3.3"])

DATA = "/kaggle/tmp/data"
os.makedirs(f"{DATA}/pubchem", exist_ok=True)
os.environ["CASMI_DATA"] = DATA
os.environ["CASMI_WORK"] = "/kaggle/tmp/work"
os.environ["PUBCHEM_OUT"] = "/kaggle/working/ext_pubchem"
# About 530 bytes a row. 10M leaves room for the 1.8 GB index beside it in the
# submission kernel, and for the peak while the two are merged.
os.environ.setdefault("TARGET", "10000000")
os.environ.setdefault("NPROC", "4")          # Kaggle CPU notebooks have 4 cores
os.makedirs("/kaggle/tmp/work", exist_ok=True)
sys.path.insert(0, CODE)

t0 = time.time()
step = lambda s: print(f"[{(time.time() - t0) / 60:5.1f} min] {s}", flush=True)

SRC = f"{DATA}/pubchem/CID-SMILES.gz"
URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
subprocess.check_call(["curl", "-sS", "--retry", "5", "-o", SRC, URL])
subprocess.check_call(["gzip", "-t", SRC])
step(f"fetched {os.path.getsize(SRC) / 1e9:.2f} GB and the CRC checks out")

import pubchem            # noqa: E402

pubchem.build()
step("shards written")

import glob               # noqa: E402
import numpy as np        # noqa: E402
fs = sorted(glob.glob(f"{pubchem.OUT}/shard_*.npz"))
rows = sum(len(np.load(f)["key"]) for f in fs)
size = sum(os.path.getsize(f) for f in fs)
print(f"{rows:,} structures in {len(fs)} shards, {size / 1e9:.2f} GB")
assert rows > 1_000_000, rows
step("done")
