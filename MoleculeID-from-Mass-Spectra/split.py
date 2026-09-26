"""Held-out validation that simulates the three novelty classes.

The visible test.parquet cannot be used to measure anything. Every one of its 400
molecules was found in train.parquet by matching (precursor_mz, adduct, num_peaks),
so it is entirely class 1 and a library search scores near 1.0 on it. The hidden
test is a mix of all three classes and will score nothing like that.

So the split is built here instead, by holding out inchikey14 groups:

  class 1  structure has spectra in >=2 source libraries. One library's spectra
           become queries, the others stay in the index. Reachable by similarity.
  class 2  every spectrum of the structure is held out, but the structure itself
           stays in the candidate list. Reachable only by database retrieval.
  class 3  every spectrum is held out and the structure is removed from the
           candidate list. Reachable only de novo.

Grouping is on inchikey14, never on spectra: the same molecule measured at four
collision energies would otherwise put three of them in the index and one in the
query set, which is class 1 wearing a class 3 label.

    python3 split.py        -> data/splits.npz
"""
import glob
import os

import numpy as np

from library import canonical_keys, rss_gb

from library import WORK

DIR = os.path.dirname(os.path.abspath(__file__))
SHARDS = f"{WORK}/index"
OUT = f"{WORK}/splits.npz"

N_PER_CLASS = 200        # held-out structures per novelty class
MAX_SPECTRA = 4          # queries per molecule; the real test has median 3


def load_keys():
    """Only the string columns. The peak arrays are 725 MB and not needed to decide
    which rows go where.

    Structures are identified by RDKit's key14, not the shipped inchikey14 column.
    Grouping on the shipped column would split one molecule across two groups on the
    0.5% of rows where the two standardisers disagree, which is how a class 2 or 3
    molecule keeps spectra in the index and stops being novel. See
    library.canonical_keys.
    """
    smiles, libs = [], []
    for f in sorted(glob.glob(f"{SHARDS}/rg_*.npz")):
        with np.load(f) as d:
            smiles.append(d["smiles"])
            libs.append(d["lib"])
    smiles = np.concatenate(smiles)
    return canonical_keys(smiles), np.concatenate(libs)


def build(seed=0):
    key, lib = load_keys()
    n = len(key)
    print(f"{n:,} spectra  {len(np.unique(key)):,} structures  RSS {rss_gb():.2f} GB")

    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    groups = np.split(order, starts[1:])          # row indices per structure
    uniq = ks[starts]

    n_lib = np.array([len(np.unique(lib[g])) for g in groups])
    n_spec = np.array([len(g) for g in groups])
    print(f"structures in >=2 libraries: {int((n_lib >= 2).sum()):,}  "
          f"spectra per structure: median {int(np.median(n_spec))}, max {n_spec.max()}")

    rng = np.random.default_rng(seed)
    multi = np.flatnonzero(n_lib >= 2)
    single = np.flatnonzero((n_lib == 1) & (n_spec >= 1))

    pick = lambda pool, k: rng.choice(pool, min(k, len(pool)), replace=False)
    c1 = pick(multi, N_PER_CLASS)
    rest = np.setdiff1d(np.r_[multi, single], c1)
    c23 = pick(rest, 2 * N_PER_CLASS)
    c2, c3 = c23[:N_PER_CLASS], c23[N_PER_CLASS:]

    query, qclass, qkey, drop = [], [], [], []
    for cls, sel in ((1, c1), (2, c2), (3, c3)):
        for gi in sel:
            g = groups[gi]
            if cls == 1:
                # Hold out one library's spectra; the others remain findable.
                held = rng.choice(np.unique(lib[g]))
                q = g[lib[g] == held]
            else:
                q = g
            q = q[:MAX_SPECTRA] if len(q) <= MAX_SPECTRA else rng.choice(q, MAX_SPECTRA, replace=False)
            query.append(q)
            qclass.append(np.full(len(q), cls, np.int8))
            qkey.append(np.full(len(q), uniq[gi]))
            # Class 1 keeps its other spectra in the index; 2 and 3 lose all of theirs.
            drop.append(q if cls == 1 else g)

    query = np.concatenate(query)
    np.savez(OUT,
             query=query, qclass=np.concatenate(qclass), qkey=np.concatenate(qkey),
             drop=np.unique(np.concatenate(drop)),
             banned=uniq[c3])            # class 3: also absent from the candidate list
    d = np.load(OUT, allow_pickle=True)
    print(f"\n{OUT}")
    for c in (1, 2, 3):
        m = d["qclass"] == c
        print(f"  class {c}: {len(np.unique(d['qkey'][m])):3d} molecules, "
              f"{int(m.sum()):4d} query spectra "
              f"({m.sum() / max(1, len(np.unique(d['qkey'][m]))):.1f} per molecule)")
    print(f"  index rows removed: {len(d['drop']):,} of {n:,}")
    print(f"  structures banned from candidates: {len(d['banned'])}")
    print(f"  peak RSS {rss_gb():.2f} GB")

    assert len(np.intersect1d(d["query"], d["drop"])) == len(d["query"]), \
        "every query spectrum must be removed from the index"
    assert set(d["banned"]) <= set(d["qkey"]), "banned structures must all be queries"


if __name__ == "__main__":
    build()
