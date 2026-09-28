"""PubChem as the candidate pool, because class 3 is coverage-limited.

Section 8 of RESULTS.md established that class 3 is 73.5% of the hidden test and is
limited by whether the molecule is in the candidate list at all, not by how well it is
ranked once it is there. The sweeps in dilute.py then measured the two halves of what
a bigger database does:

  * ranking quality among covered molecules, against pool size: -0.011 per doubling,
    measured over a 23x range on 200 molecules. Class 3 is nearly immune to pool size.
  * class 1 and class 2, against external pool size: -0.018 and -0.012 per doubling.

So expansion pays as long as it raises coverage faster than log2 of its own size,
which is a low bar. A REST probe of the 200 held-out class 3 skeletons found 87% of
them in PubChem against 8% in COCONUT.

PubChem is public domain (NCBI claims no copyright) and free, so Section 2.6 of the
rules permits it.

**Why not all 119M compounds.** Not quality -- the sweep says that would be nearly
free. Memory: the kernel has to hold key, smiles, mass and fingerprint for every
candidate, about 460 bytes a row, so 119M is 55 GB against the kernel's 30. The
NP-likeness filter cuts the set to a size that fits while keeping the kind of compound
the test is made of; --pilot measures the survival rate before committing to a run.

**Why the cheap filters come first.** MolFromSmiles runs at 9k/s, InChIKey at 5k/s and
NP-likeness at 5k/s. Parsing all 119M is unavoidable, but only survivors pay the rest.

    python3 pubchem.py --pilot           # survival rate on a strided sample
    python3 pubchem.py --build           # -> data/ext_pubchem/shard_XXX.npz
    python3 pubchem.py                   # coverage of the held-out classes
"""
import gzip
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import library
from analog import FP_BITS

DATA, WORK = library.DATA, library.WORK
SRC = f"{DATA}/pubchem/CID-SMILES.gz"
OUT = os.environ.get("PUBCHEM_OUT", f"{WORK}/ext_pubchem")
MZ_LO, MZ_HI = 100.0, 1300.0
NP_MIN = float(os.environ.get("NP_MIN", 0.0))
SHARD = 500_000
NPROC = int(os.environ.get("NPROC", 8))
CHUNK = 50_000
SMILES_MAX = 250          # storage width, and a free pre-filter on size

_state = {}


def _init():
    from rdkit import Chem, RDConfig, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "NP_Score"))
    import npscorer
    _state.update(Chem=Chem, Desc=Descriptors, npscorer=npscorer,
                  fs=npscorer.readNPModel(),
                  gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,
                                                                fpSize=FP_BITS))


def _work(lines):
    """One chunk of `CID<TAB>SMILES` lines -> (keys, smiles, masses, fps, np), n."""
    Chem, Desc, gen = _state["Chem"], _state["Desc"], _state["gen"]
    npscorer, fs = _state["npscorer"], _state["fs"]
    K, S, M, F, P = [], [], [], [], []
    for ln in lines:
        t = ln.rstrip("\n").split("\t")
        if len(t) != 2:
            continue
        smi = t[1]
        # Mixtures and salts are not what one precursor mass describes, and the index
        # holds none. Dropping them here also saves the parse on ~15% of PubChem.
        if "." in smi or not 3 <= len(smi) <= SMILES_MAX:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        mass = Desc.ExactMolWt(m)
        if not MZ_LO <= mass <= MZ_HI:
            continue
        np_ = npscorer.scoreMol(m, fs)
        if np_ < NP_MIN:
            continue
        try:
            k = Chem.MolToInchiKey(m)[:14]
        except Exception:
            continue
        if len(k) != 14:
            continue
        K.append(k.encode())
        S.append(smi.encode())
        M.append(mass)
        F.append(np.packbits(gen.GetFingerprintAsNumPy(m)))
        P.append(np_)
    if not K:
        return None, len(lines)
    return (np.array(K, dtype="S14"), np.array(S, dtype=f"S{SMILES_MAX}"),
            np.array(M, np.float64), np.stack(F),
            np.array(P, np.float32)), len(lines)


def _chunks(f, n=CHUNK, stride=1):
    buf = []
    for i, ln in enumerate(f):
        if stride > 1 and i % stride:
            continue
        buf.append(ln)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _stream(stride=1, limit=None):
    """Surviving structures, chunk by chunk, with progress."""
    seen = kept = 0
    t0 = time.time()
    with gzip.open(SRC, "rt") as f:
        next(f)                                    # header
        with Pool(NPROC, initializer=_init) as p:
            for res, n in p.imap(_work, _chunks(f, stride=stride), chunksize=1):
                seen += n
                if res is not None:
                    kept += len(res[0])
                    yield res
                if seen % (CHUNK * 40) == 0:
                    el = time.time() - t0
                    print(f"  {seen:,} scanned  {kept:,} kept "
                          f"({kept / max(seen, 1):.1%})  {el / 60:5.1f} min  "
                          f"{seen / max(el, 1) / 1000:.0f}k/s", flush=True)
                if limit and seen >= limit:
                    return


def pilot(n=2_000_000, stride=53):
    """Survival rate on a strided sample.

    Strided rather than the head of the file: PubChem CIDs run roughly in deposit
    order, so the first few million are classic small molecules and would make a
    natural-product filter look far more generous than it is.
    """
    nps = np.concatenate([r[4] for r in _stream(stride=stride, limit=n)])
    print(f"\nNP_MIN={NP_MIN}: {len(nps):,} of {n:,} sampled pass the mass and "
          f"fragment filters ({len(nps) / n:.2%})")
    print(f"\n{'NP_MIN':>8}{'kept':>10}{'of PubChem':>14}{'in memory':>12}")
    for t in (-3, -2, -1, 0, 0.5, 1.0, 1.5, 2.0, 2.5):
        f = float((nps >= t).mean()) * len(nps) / n
        print(f"{t:>8.1f}{f:>9.2%}{int(f * 119e6):>14,}{f * 119e6 * 460 / 1e9:>10.1f} GB")


def build():
    os.makedirs(OUT, exist_ok=True)
    if any(f.endswith(".npz") for f in os.listdir(OUT)):
        print(f"{OUT} is not empty; delete it to rebuild")
        return
    shard, buf, held, n = 0, [], 0, 0
    t0 = time.time()

    def flush(shard, buf):
        if buf:
            np.savez(f"{OUT}/shard_{shard:03d}.npz",
                     **{k: np.concatenate([b[i] for b in buf])
                        for i, k in enumerate(("key", "smiles", "mass", "fp"))})

    for res in _stream():
        buf.append(res[:4])
        held += len(res[0])
        n += len(res[0])
        if held >= SHARD:
            flush(shard, buf)
            shard, buf, held = shard + 1, [], 0
    flush(shard, buf)
    print(f"\n{n:,} structures in {shard + 1} shards, "
          f"{(time.time() - t0) / 60:.1f} min")
    dedupe()


def dedupe():
    """Drop repeated key14s across shards.

    PubChem lists stereoisomers and tautomers under separate CIDs, which collapse onto
    one key14. A duplicate is not harmless the way it was in external.py: the scoring
    loop adds a Tanimoto term per *row* and totals it by key14, so two rows of one
    structure score it twice and push it up the ranking for nothing.
    """
    import glob
    fs = sorted(glob.glob(f"{OUT}/shard_*.npz"))
    keys = [np.load(f)["key"] for f in fs]
    off = np.cumsum([0] + [len(k) for k in keys])
    _, first = np.unique(np.concatenate(keys), return_index=True)
    del keys
    mask = np.zeros(off[-1], bool)
    mask[first] = True
    for i, f in enumerate(fs):
        m = mask[off[i]:off[i + 1]]
        if not m.all():
            with np.load(f) as d:
                out = {k: d[k][m] for k in d.files}
            np.savez(f, **out)
    print(f"deduped to {int(mask.sum()):,} structures "
          f"({off[-1] - int(mask.sum()):,} removed)")


def coverage():
    # The shard layout is external.py's, so pointing that module at this directory is
    # all it takes for analog.py to use PubChem instead of COCONUT:
    #     CASMI_EXT=data/ext_pubchem python3 fpsearch.py
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
    print("\nCOCONUT reached 8.0% of class 3; a REST probe of the same 200 skeletons "
          "found 87% of them somewhere in PubChem, so the gap between that and this "
          "number is what the mass and NP-likeness filters cost.")


if __name__ == "__main__":
    if "--pilot" in sys.argv:
        pilot()
    elif "--build" in sys.argv:
        build()
    elif "--dedupe" in sys.argv:
        dedupe()
    else:
        coverage()
