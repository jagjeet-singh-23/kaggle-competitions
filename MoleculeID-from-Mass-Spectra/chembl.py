"""ChEMBL as a candidate source, because density beats size.

Section 10 of RESULTS.md: a PubChem build of 8.4M structures raised held-out class 3
coverage from 8.0% to 9.5% and cost ranking quality 0.366 -> 0.269, because everything
a database adds inside a 5 ppm window is an isomer of the answer. Coverage has to grow
faster than quality falls. What a pool needs is density in the chemistry that reaches
spectral libraries, and the same section measured what that chemistry is: the held-out
class 3 structures score a median -1.43 on NP-likeness. They are drugs, pesticides and
metabolites, which is what ChEMBL is made of.

ChEMBL is 2.4M structures against COCONUT's 470k, so the pool roughly doubles rather
than growing twelvefold, and the dilution stays near what COCONUT already pays for.

ChEMBL is CC BY-SA 3.0: publicly available, free and equally accessible, which is what
rules section 2.6 asks for.

No multiprocessing and no InChI generation. The file ships `standard_inchi_key`, which
is the expensive half of what pubchem.py had to compute, and 2.4M parses is six
minutes on one core.

    ./fetch:  curl -O https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_chemreps.txt.gz
    python3 chembl.py --build     -> data/ext_chembl/shard_XXX.npz
    python3 chembl.py             -> coverage of the held-out classes
"""
import gzip
import os
import sys
import time

import numpy as np

import library
from analog import FP_BITS

DATA, WORK = library.DATA, library.WORK
SRC = os.environ.get("CHEMBL_SRC", f"{DATA}/chembl/chembl_37_chemreps.txt.gz")
OUT = os.environ.get("CHEMBL_OUT", f"{WORK}/ext_chembl")
MZ_LO, MZ_HI = 100.0, 1300.0
SMILES_MAX = 300
SHARD = 250_000


def rows():
    """(smiles, key14, exact mass) for in-range ChEMBL entries, deduplicated."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")

    seen = set()
    n = kept = 0
    t0 = time.time()
    with gzip.open(SRC, "rt") as f:
        next(f)                                        # header
        for line in f:
            t = line.rstrip("\n").split("\t")
            if len(t) != 4:
                continue
            smi, key = t[1], t[3][:14]
            n += 1
            # Mixtures and salts are not what one precursor mass describes, and the
            # index holds none.
            if "." in smi or not 3 <= len(smi) <= SMILES_MAX or len(key) != 14:
                continue
            kb = key.encode()
            if kb in seen:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            mass = Descriptors.ExactMolWt(m)
            if not MZ_LO <= mass <= MZ_HI:
                continue
            seen.add(kb)
            kept += 1
            yield smi, key, mass, m
            if n % 500_000 == 0:
                print(f"  {n:,} scanned  {kept:,} kept  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)


def build():
    from rdkit.Chem import rdFingerprintGenerator
    os.makedirs(OUT, exist_ok=True)
    if any(f.endswith(".npz") for f in os.listdir(OUT)):
        print(f"{OUT} is not empty; delete it to rebuild")
        return

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    t0 = time.time()
    shard, K, S, M, F, total = 0, [], [], [], [], 0

    def flush():
        if K:
            np.savez(f"{OUT}/shard_{shard:03d}.npz",
                     key=np.array(K, dtype="S14"),
                     smiles=np.array(S, dtype=f"S{SMILES_MAX}"),
                     mass=np.array(M, np.float64), fp=np.stack(F))

    for smi, key, mass, mol in rows():
        K.append(key.encode())
        S.append(smi.encode())
        M.append(mass)
        F.append(np.packbits(gen.GetFingerprintAsNumPy(mol)))
        if len(K) >= SHARD:
            flush()
            total += len(K)
            shard, K, S, M, F = shard + 1, [], [], [], []
    flush()
    total += len(K)
    print(f"\n{total:,} structures in {shard + 1} shards, "
          f"{(time.time() - t0) / 60:.1f} min")


def coverage():
    import external
    external.OUT = OUT
    ext = external.load()["key"]
    sp = np.load(f"{WORK}/splits.npz", allow_pickle=True)
    qkey, qclass = sp["qkey"], sp["qclass"]
    print(f"{len(ext):,} structures in the pool\n")
    for c in (1, 2, 3):
        q = np.unique(qkey[qclass == c])
        hit = int(np.isin(q, ext).sum())
        print(f"  class {c}: {hit:3d}/{len(q)} ({hit / len(q):5.1%}) reachable")
    print("\nCOCONUT reaches 8.0% of class 3 carrying 217 candidates a query; the "
          "PubChem build reached 6.5% carrying 1030. Coverage per candidate carried "
          "is the number to beat, not coverage.")


if __name__ == "__main__":
    build() if "--build" in sys.argv else coverage()
