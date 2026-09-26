"""Rung 1: spectral library search, scored on the held-out novelty split.

For each query spectrum, take the candidates whose precursor mass agrees, score them
by cosine over tolerance-binned peaks, collapse to one score per structure, and
combine the structures across a molecule's several spectra into one ranked list.

This can only ever solve class 1. A class 2 or 3 molecule has no spectra left in the
index by construction, so its MRR here is 0 and that is the point: it measures how
much of the competition library search alone reaches.

    python3 search.py                 # evaluate on data/splits.npz
    python3 search.py --submit        # write submission.csv for the real test

Memory: the index is 1.84 GB and everything else is per-query scratch, so this sits
around 2.5 GB. Do not run it beside another heavy job.
"""
import os
import sys
import time
from collections import defaultdict

import numpy as np

import library
from metric import mrr, validate

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = f"{DIR}/data"

TOL_DA = 0.01        # peak matching tolerance; timsTOF is well inside this
PPM = 10.0           # precursor window
BINS = 200_000       # 0.01 Da bins up to m/z 2000
TOP_K = 25


class Searcher:
    """Cosine search over the index, with the precursor window applied first.

    The window is what makes this affordable: comparing a query against all 1.9M
    spectra would be pointless as well as slow, since a fragment pattern from a
    different precursor mass cannot be the same molecule.
    """

    def __init__(self, drop=None, banned=None):
        ix = library.load()
        n = len(ix["npk"])
        mask = np.ones(n, bool)
        if drop is not None:
            mask[drop] = False            # held-out spectra leave the index
        if banned is not None and len(banned):
            mask &= ~np.isin(ix["inchikey14"], np.asarray(banned))
        self.ix = {k: v[mask] for k, v in ix.items()}
        del ix

        p = self.ix["precursor"]
        self.order = np.argsort(p, kind="stable")
        self.psorted = p[self.order]
        # Peak intensities are already sqrt-scaled by library.trim; normalise each
        # spectrum so a cosine is a plain dot product.
        it = self.ix["it"].astype(np.float32)
        self.it = it / (np.linalg.norm(it, axis=1, keepdims=True) + 1e-12)
        self.bin = np.rint(self.ix["mz"] / TOL_DA).astype(np.int32)
        self.bin[self.ix["mz"] == 0] = -1          # padding slots match nothing
        self.scratch = np.zeros(BINS + 2, np.float32)
        print(f"index: {len(self.psorted):,} spectra, "
              f"{len(np.unique(self.ix['inchikey14'])):,} structures")

    def candidates(self, precursor):
        w = precursor * PPM * 1e-6
        lo, hi = np.searchsorted(self.psorted, [precursor - w, precursor + w])
        return self.order[lo:hi]

    def score(self, mzs, ints, precursor):
        """Cosine of one query spectrum against every mass-compatible candidate."""
        rows = self.candidates(precursor)
        if not len(rows):
            return rows, np.zeros(0, np.float32)
        q = np.asarray(ints, np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        qb = np.rint(np.asarray(mzs, np.float32) / TOL_DA).astype(np.int32)
        ok = (qb >= 0) & (qb < BINS)

        s = self.scratch
        s[qb[ok]] = q[ok]                          # scatter the query onto the bin axis
        cb = self.bin[rows]
        sim = np.where(cb >= 0, s[np.clip(cb, 0, BINS + 1)], 0.0) * self.it[rows]
        s[qb[ok]] = 0.0                            # clear only what was written
        return rows, sim.sum(axis=1)

    def rank(self, spectra):
        """spectra: list of (mzs, ints, precursor) for one molecule -> top-25 SMILES.

        A structure's score is summed over the query spectra, taking its best row
        within each. Summing rewards a structure that explains several acquisitions;
        taking the max within one spectrum stops a structure with many near-identical
        library entries from winning on count alone.
        """
        total = defaultdict(float)
        best_smiles = {}
        for mzs, ints, pm in spectra:
            rows, sim = self.score(mzs, ints, pm)
            if not len(rows):
                continue
            # Sort by descending similarity, then np.unique's first occurrence of
            # each structure is that structure's best row. Avoids a Python loop
            # over every candidate, which runs into the millions across a split.
            o = np.argsort(-sim, kind="stable")
            keys = self.ix["inchikey14"][rows[o]]
            uk, first = np.unique(keys, return_index=True)
            for k, v, r in zip(uk, sim[o][first], rows[o][first]):
                total[k] += float(v)
                if k not in best_smiles:
                    best_smiles[k] = self.ix["smiles"][r]
        top = sorted(total.items(), key=lambda kv: -kv[1])[:TOP_K]
        return [best_smiles[k].decode() for k, _ in top]


def spectra_of(ix, rows):
    out = []
    for r in rows:
        n = int(ix["npk"][r])
        out.append((ix["mz"][r, :n], ix["it"][r, :n].astype(np.float32),
                    float(ix["precursor"][r])))
    return out


def evaluate():
    sp = np.load(f"{DATA}/splits.npz", allow_pickle=True)
    ix = library.load()
    se = Searcher(drop=sp["drop"], banned=sp["banned"])

    t0 = time.time()
    per_class = defaultdict(dict)
    answers = defaultdict(dict)
    qkey, qclass, query = sp["qkey"], sp["qclass"], sp["query"]
    for k in np.unique(qkey):
        m = qkey == k
        cls = int(qclass[m][0])
        rows = query[m]
        per_class[cls][k] = se.rank(spectra_of(ix, rows))
        # The answer is the held-out structure's own SMILES, taken before the drop.
        answers[cls][k] = ix["smiles"][rows[0]].decode()
    print(f"\nsearched {len(qkey):,} spectra in {time.time() - t0:.0f}s")

    print(f"\n{'class':<8}{'molecules':>11}{'MRR@25':>10}")
    overall = []
    for cls in (1, 2, 3):
        v = mrr(per_class[cls], answers[cls])
        overall.append(v)
        print(f"{cls:<8}{len(answers[cls]):>11}{v:>10.4f}")
    print(f"{'mean':<8}{'':>11}{np.mean(overall):>10.4f}")
    print("\nclass 2 and 3 are 0 by construction: their spectra are not in the index.")
    print("Only the class 1 number says anything about library search itself.")

    assert overall[0] > 0, "library search should find class 1 molecules"
    # Anything above 0 here means a held-out structure is still reachable, i.e. the
    # split leaks. It did: grouping on the shipped inchikey14 column instead of
    # RDKit's key left class 3 at 0.0050. See library.canonical_keys.
    assert overall[2] == 0, f"class 3 leaked: {overall[2]:.4f}"
    return overall


def submit(path=f"{DIR}/submission.csv"):
    """Write submission.csv for the real test set.

    Queries go through library.trim, the same curation the index rows had. This is not
    cosmetic: evaluate() draws its queries out of the index, so it measures
    trimmed-against-trimmed, and feeding raw test spectra here would measure something
    else entirely. Test spectra carry a median of 230 peaks against a library median
    of 9 to 11, so an untrimmed query is normalised over ~220 peaks the reference
    never recorded and scores low against exactly the libraries that matter most.
    """
    import pandas as pd
    te = pd.read_parquet(f"{DATA}/test.parquet")
    se = Searcher()
    rows = []
    for mid, g in te.groupby("molecule_id"):
        spectra = [library.trim(a, b, p) + (p,)
                   for a, b, p in zip(g.ms2_mzs, g.ms2_normalized_intensities,
                                      g.precursor_mz)]
        cands = se.rank(spectra) or ["C"]
        rows.append((mid, ";".join(cands[:TOP_K])))
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    validate(sub)
    sub.to_csv(path, index=False)
    print(f"wrote {path}: {len(sub)} molecules, "
          f"{sub.smiles.str.split(';').str.len().mean():.1f} candidates each")


def selftest():
    """Drive the search over a tiny synthetic index, so the binning and the
    per-molecule aggregation are checked without loading the 1.84 GB real one."""
    rng = np.random.default_rng(0)
    n, P = 300, library.PEAKS
    mz = np.zeros((n, P), np.float32)
    it = np.zeros((n, P), np.float16)
    npk = np.full(n, 8, np.int16)
    for i in range(n):
        mz[i, :8] = np.sort(rng.uniform(50, 300, 8)).astype(np.float32)
        it[i, :8] = rng.uniform(0.1, 1.0, 8).astype(np.float16)
    fake = dict(mz=mz, it=it, npk=npk,
                precursor=np.full(n, 350.0, np.float32),
                ion=np.ones(n, bool),
                inchikey14=np.array([f"KEY{i:011d}" for i in range(n)], dtype="S14"),
                smiles=np.array([b"CCO"] * n, dtype="S300"),
                lib=np.array([b"fake"] * n, dtype="S24"))
    fake["smiles"][7] = b"c1ccccc1"
    real_load, library.load = library.load, lambda: fake
    try:
        se = Searcher()
        # Row 7 queried against itself must come first, and score ~1.0.
        rows, sim = se.score(mz[7, :8], it[7, :8].astype(np.float32), 350.0)
        assert rows[np.argmax(sim)] == 7, "a spectrum must match itself best"
        assert 0.99 < sim.max() <= 1.0 + 1e-5, f"self-cosine was {sim.max():.4f}"
        top = se.rank([(mz[7, :8], it[7, :8].astype(np.float32), 350.0)])
        assert top[0] == "c1ccccc1", top[:2]
        assert len(top) == TOP_K
        # Nothing shares the precursor window 200 Da away.
        assert len(se.score(mz[7, :8], it[7, :8].astype(np.float32), 550.0)[0]) == 0
        # The scratch buffer must be left clean, or the next query is corrupted.
        assert not se.scratch.any(), "scratch buffer leaked between queries"
    finally:
        library.load = real_load
    print("search self-checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--submit" in sys.argv:
        submit()
    else:
        evaluate()
