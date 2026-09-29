"""Rung 2: analog propagation, to reach class 2.

Library search can only return a structure that has spectra in the index, so class 2
and class 3 score exactly 0 -- and together they are 69% of the problem. Class 2 is
the reachable half: the structure is in the candidate list, it just has no spectra.

The idea is that a molecule's fragmentation is mostly a property of its scaffold, so a
spectrum that matches structure S well is evidence for molecules that look like S. So:
take the best spectral matches as seeds, pull every candidate whose neutral mass fits
the query's precursor, and score those candidates by how similar they are to a seed
that scored well.

    score(c) = spectral(c) + BETA * max over seeds s of [ spectral(s) * tanimoto(c,s) ]

The first term is library search unchanged, so class 1 cannot regress. The second is
the new one, and it is the only term a class 2 molecule can score on.

Class 3 stays 0 and must: its structure is banned from the candidate list, so it is
absent from the pool by construction. That is the split doing its job.

    python3 analog.py --build     # cache structure descriptors (~2 min)
    python3 analog.py             # evaluate on data/splits.npz
    python3 analog.py --selftest
"""
import os
import sys

import numpy as np

import library
from metric import mrr
from search import Searcher, spectra_of

from library import DATA, WORK

DIR = os.path.dirname(os.path.abspath(__file__))
DESC = f"{WORK}/structures.npz"
# Where the merged pool is built as memory-mapped .npy. Empty means build it
# in RAM, which is what COCONUT-sized pools did and what the self-checks use.
POOL_CACHE = os.environ.get("POOL_CACHE", f"{WORK}/pool")

FP_BITS = 2048
MASS_PPM = 5.0       # neutral-mass window; swept, with a clear optimum at 5
N_SEEDS = 1          # analog seeds; more seeds only dilute (see RESULTS.md)
BETA = 1.0           # weight on the analog term
PROP_INTERNAL = os.environ.get("PROP_INTERNAL", "0") == "1"
TOP_K = 25

# m/z = M + shift, for the ten adducts the hidden test uses. Monoisotopic, with the
# electron accounted for on the metal adducts.
PROTON, H2O, E = 1.0072765, 18.0105646, 0.00054858
ADDUCTS = {
    "[M+H]+":       (PROTON, True),
    "[M+NH4]+":     (17.0265491 + PROTON, True),
    "[M-H2O+H]+":   (PROTON - H2O, True),
    "[M-2H2O+H]+":  (PROTON - 2 * H2O, True),
    "[M+Na]+":      (22.9897697 - E, True),
    "[M+K]+":       (38.9637069 - E, True),
    "[M-H]-":       (-PROTON, False),
    "[M-H2O-H]-":   (-PROTON - H2O, False),
    "[M+CH2O2-H]-": (46.0054793 - PROTON, False),
    "[M+Cl]-":      (34.96885271 + E, False),
}


def build():
    """Morgan fingerprint and exact mass for every structure in the index, cached.

    Keyed on the canonical RDKit key14, the same identity the metric and the split
    use, so a structure is one row here no matter how many spectra it has.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    ix = library.load()
    key, first = np.unique(ix["inchikey14"], return_index=True)
    smi = ix["smiles"][first]
    del ix
    print(f"{len(key):,} structures", flush=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    fp = np.zeros((len(key), FP_BITS // 8), np.uint8)
    mass = np.zeros(len(key), np.float64)
    for i, s in enumerate(smi):
        m = Chem.MolFromSmiles(s.decode())
        if m is None:
            mass[i] = -1.0          # unparseable: cannot match any precursor window
            continue
        fp[i] = np.packbits(gen.GetFingerprintAsNumPy(m))
        mass[i] = Descriptors.ExactMolWt(m)
        if i % 50_000 == 0 and i:
            print(f"  {i:,}/{len(key):,}", flush=True)
    np.savez(DESC, key=key, smiles=smi, mass=mass, fp=fp)
    print(f"wrote {DESC}  ({os.path.getsize(DESC) / 1e6:.0f} MB, "
          f"{int((mass < 0).sum())} unparseable)")


class _Rows:
    """key14 -> row index, without a Python dict over every structure.

    A dict of 20M byte keys costs about 3 GB in object overhead alone, which is most
    of what a PubChem-scale pool has to spare inside a 30 GB kernel. A sort order plus
    searchsorted answers the same two questions -- `k in rows` and `rows[k]` -- for
    eight bytes a row.
    """

    def __init__(self, key):
        self.order = np.argsort(key, kind="stable")
        self.sorted = key[self.order]

    def _find(self, k):
        i = np.searchsorted(self.sorted, k)
        return (int(self.order[i])
                if i < len(self.sorted) and self.sorted[i] == k else -1)

    def __getitem__(self, k):
        r = self._find(k)
        if r < 0:
            raise KeyError(k)
        return r

    def __contains__(self, k):
        return self._find(k) >= 0


class Analog:
    """Mass-windowed candidate pool plus Tanimoto scoring against spectral seeds."""

    def __init__(self, banned=None, external=True):
        assert os.path.exists(DESC), "run `python3 analog.py --build` first"
        with np.load(DESC) as d:
            key, smiles, mass, fp = d["key"], d["smiles"], d["mass"], d["fp"]
        if banned is not None and len(banned):
            # A class 3 structure is not in the candidate list at all, so it must not
            # be reachable here either. Without this the split would leak again.
            keep = ~np.isin(key, np.asarray(banned))
            key, smiles, mass, fp = key[keep], smiles[keep], mass[keep], fp[keep]
        if external:
            # A public database (COCONUT, CC0) merged in after the ban, so a structure
            # the library does not have can still be a candidate. The ban is not
            # re-applied: a real class 3 molecule is absent from train.parquet and
            # findable only through exactly this route, so letting it back in here is
            # the faithful simulation rather than a leak.
            try:
                import external as ext_db
                # EXT_FRAC subsamples the external structures. A random fraction f
                # cuts coverage to ~f while cutting the pool by the same factor, so
                # sweeping it separates "more candidates found" from "more candidates
                # competing" -- the two halves of what a bigger database does.
                frac = float(os.environ.get("EXT_FRAC", 1.0))
                if frac < 1.0:
                    e = ext_db.load()
                    new = ~np.isin(e["key"], key)
                    new &= np.random.default_rng(0).random(len(new)) < frac
                    key = np.concatenate([key, e["key"][new]])
                    smiles = np.concatenate([smiles,
                                             e["smiles"][new].astype(smiles.dtype)])
                    mass = np.concatenate([mass, e["mass"][new]])
                    fp = np.concatenate([fp, e["fp"][new]])
                    n_new = int(new.sum())
                    del e, new
                else:
                    key, smiles, mass, fp, n_new = ext_db.merge(
                        key, smiles, mass, fp, cache=POOL_CACHE)
                self.external_n = n_new
                print(f"external pool: +{self.external_n:,} structures")
                self.n_internal = len(key) - self.external_n
            except AssertionError:
                pass
        self.external_n = getattr(self, "external_n", 0)
        self.n_internal = getattr(self, "n_internal", len(key))
        self.key, self.smiles, self.mass, self.fp = key, smiles, mass, fp
        self.order = np.argsort(mass, kind="stable")
        self.msorted = mass[self.order]
        self.row = _Rows(key)
        # In one call this materialises a uint8 copy of the whole fingerprint block,
        # 2.6 GB at PubChem scale. A million rows at a time costs nothing and peaks at
        # 256 MB.
        self.popc = np.empty(len(fp), np.int32)
        for i in range(0, len(fp), 1_000_000):
            self.popc[i:i + 1_000_000] = np.bitwise_count(fp[i:i + 1_000_000]).sum(1)
        print(f"analog pool: {len(key):,} structures")

    def pool(self, precursor, adducts, ppm=MASS_PPM):
        """Rows whose neutral mass fits the precursor under any of `adducts`.

        The window is proportional, not a fixed number of daltons. The index is
        already filtered to |precursor_error| <= 10 ppm, and 10 ppm is 0.004 Da at
        m/z 400 but 0.0125 Da at 1250, so a constant window is simultaneously too
        loose for light molecules and too tight for heavy ones.
        """
        out = []
        for a in adducts:
            shift, _ = ADDUCTS[a]
            neutral = precursor - shift
            w = neutral * ppm * 1e-6
            lo, hi = np.searchsorted(self.msorted, [neutral - w, neutral + w])
            out.append(self.order[lo:hi])
        return np.unique(np.concatenate(out)) if out else np.zeros(0, int)

    def tanimoto(self, rows, seeds):
        """(len(rows), len(seeds)) Tanimoto over packed fingerprints."""
        a, b = self.fp[rows], self.fp[seeds]
        inter = np.bitwise_count(a[:, None, :] & b[None, :, :]).sum(-1)
        return inter / np.maximum(self.popc[rows][:, None]
                                  + self.popc[seeds][None, :] - inter, 1)

    def propagate(self, spec, precursor, adducts, n_seeds=N_SEEDS, agg="max",
                  ppm=MASS_PPM, stat=None):
        """spec: {key: spectral score}. Returns {key: analog score}.

        agg="max" trusts the single best explanation for a candidate; agg="sum"
        accumulates weak evidence from many seeds, which rewards a candidate sitting
        in a dense neighbourhood of decent matches rather than next to one good one.
        """
        rows = self.pool(precursor, adducts, ppm)
        if PROP_INTERNAL:
            # Analog propagation is what class 2 lives on, and it is the one term
            # that a bigger database degrades: widening the pool 23x cost class 2
            # 0.037 per doubling while class 3 lost 0.011. An external structure can
            # take analog score off a seed but rarely deserves it -- it has no
            # spectra anywhere, so nothing vouches for its neighbourhood. Class 3
            # does not need this term either: it is reached by the predicted
            # fingerprint, which scores the full pool in fpsearch regardless.
            rows = rows[rows < self.n_internal]
        if stat is not None:
            stat.append(len(rows))
        if not len(rows) or not spec:
            return {}
        top = sorted(spec.items(), key=lambda kv: -kv[1])[:n_seeds]
        seeds = [(self.row[k], v) for k, v in top if k in self.row]
        if not seeds:
            return {}
        si = np.array([s for s, _ in seeds])
        sv = np.array([v for _, v in seeds], np.float64)
        w = self.tanimoto(rows, si) * sv
        best = w.max(axis=1) if agg == "max" else w.sum(axis=1)
        return {self.key[r]: float(b) for r, b in zip(rows, best) if b > 0}


def rank(se, an, spectra, adducts, **kw):
    """Merge library search with analog propagation into one top-25 SMILES list."""
    spec, smiles = se.score_structures(spectra)
    extra = an.propagate(spec, spectra[0][2], adducts, **kw)
    total = dict(spec)
    for k, v in extra.items():
        total[k] = total.get(k, 0.0) + BETA * v
        smiles.setdefault(k, an.smiles[an.row[k]])
    top = sorted(total.items(), key=lambda kv: -kv[1])[:TOP_K]
    return [smiles[k].decode() if isinstance(smiles[k], bytes) else smiles[k]
            for k, _ in top]


def evaluate():
    sp = np.load(f"{WORK}/splits.npz", allow_pickle=True)
    ix = library.load()
    se = Searcher(drop=sp["drop"], banned=sp["banned"])
    an = Analog(banned=sp["banned"])

    # The index does not store adduct, so a held-out query only knows its ion mode and
    # has to try all five adducts of that mode. The real test ships the adduct, so
    # submit() is strictly better informed than this measurement.
    pos = [a for a, (_, p) in ADDUCTS.items() if p]
    neg = [a for a, (_, p) in ADDUCTS.items() if not p]

    qkey, qclass, query = sp["qkey"], sp["qclass"], sp["query"]
    # Cache each molecule's spectra and spectral scores once; the sweep below only
    # changes how analogs are propagated, and the library search is the slow half.
    jobs = []
    answers = {1: {}, 2: {}, 3: {}}
    base = {1: {}, 2: {}, 3: {}}
    for k in np.unique(qkey):
        m = qkey == k
        cls, rows = int(qclass[m][0]), query[m]
        spectra = spectra_of(ix, rows)
        add = pos if bool(ix["ion"][rows[0]]) else neg
        spec, smiles = se.score_structures(spectra)
        jobs.append((cls, k, spec, smiles, float(spectra[0][2]), add))
        base[cls][k] = [smiles[q].decode() for q, _ in
                        sorted(spec.items(), key=lambda kv: -kv[1])[:TOP_K]]
        answers[cls][k] = ix["smiles"][rows[0]].decode()
    del ix, se

    # seeds=1/max won the first sweep: extra seeds are worse spectral matches whose
    # analogs only dilute the ranking. Hold that and attack the pool instead, which
    # is the other half of the problem -- 530 candidates competing for 25 slots.
    # "primary" keeps only [M+H]+ / [M-H]-, which is 95% of the real test's adducts
    # and cuts the pool roughly fivefold.
    primary = {True: ["[M+H]+"], False: ["[M-H]-"]}
    print(f"\n{'adducts':<12}{'ppm':>7}{'pool':>7}{'class1':>9}{'class2':>9}"
          f"{'class3':>9}{'mean':>9}")
    best, got = (-1, None), None
    for mode in ("all-in-mode", "primary"):
        for ppm in (20.0, 10.0, 5.0, 2.0):
            sizes, preds = [], {1: {}, 2: {}, 3: {}}
            for cls, k, spec, smiles, pm, add in jobs:
                use = add if mode == "all-in-mode" else primary[add is pos]
                extra = an.propagate(spec, pm, use, n_seeds=1, agg="max",
                                     ppm=ppm, stat=sizes)
                total = dict(spec)
                for key_, v in extra.items():
                    total[key_] = total.get(key_, 0.0) + BETA * v
                    smiles.setdefault(key_, an.smiles[an.row[key_]])
                preds[cls][k] = [smiles[q].decode() for q, _ in
                                 sorted(total.items(), key=lambda kv: -kv[1])[:TOP_K]]
            v = [mrr(preds[c], answers[c]) for c in (1, 2, 3)]
            print(f"{mode:<12}{ppm:>7g}{int(np.median(sizes)):>7}{v[0]:>9.4f}"
                  f"{v[1]:>9.4f}{v[2]:>9.4f}{np.mean(v):>9.4f}", flush=True)
            if np.mean(v) > best[0]:
                best, got = (np.mean(v), (mode, ppm)), v

    b = [mrr(base[c], answers[c]) for c in (1, 2, 3)]
    print(f"\nlibrary search alone:  {b[0]:.4f} / {b[1]:.4f} / {b[2]:.4f} "
          f"  mean {np.mean(b):.4f}")
    print(f"best: adducts={best[1][0]} ppm={best[1][1]:g}  mean {best[0]:.4f}  "
          f"({best[0] - np.mean(b):+.4f})")

    assert got[2] == 0, f"class 3 leaked: {got[2]:.4f}"
    assert got[0] >= b[0] - 1e-9, "class 1 must not regress"
    return got


def submit(path=f"{DIR}/submission.csv"):
    """Write submission.csv using library search plus analog propagation.

    Unlike evaluate(), this knows each spectrum's adduct, because test.parquet ships
    it. So the mass window is opened for one adduct rather than all five of an ion
    mode, and the candidate pool is correspondingly tighter than anything the
    held-out measurement could show.
    """
    import pandas as pd
    from metric import validate

    te = pd.read_parquet(f"{DATA}/test.parquet")
    se, an = Searcher(), Analog()
    rows = []
    for mid, g in te.groupby("molecule_id"):
        spectra = [library.trim(a, b, p) + (p,)
                   for a, b, p in zip(g.ms2_mzs, g.ms2_normalized_intensities,
                                      g.precursor_mz)]
        add = [a for a in pd.unique(g.adduct) if a in ADDUCTS] or ["[M+H]+"]
        cands = rank(se, an, spectra, add) or ["C"]
        rows.append((mid, ";".join(cands[:TOP_K])))
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    validate(sub)
    sub.to_csv(path, index=False)
    print(f"wrote {path}: {len(sub)} molecules, "
          f"{sub.smiles.str.split(';').str.len().mean():.1f} candidates each")


def selftest():
    """Check the mass window and the Tanimoto against hand-computable answers."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)

    smis = [b"CC(=O)Oc1ccccc1C(=O)O", b"CC(=O)Oc1ccccc1C(=O)OC", b"CCO", b"c1ccccc1"]
    fp = np.stack([np.packbits(gen.GetFingerprintAsNumPy(
        Chem.MolFromSmiles(s.decode()))) for s in smis])
    mass = np.array([Descriptors.ExactMolWt(Chem.MolFromSmiles(s.decode()))
                     for s in smis])
    key = np.array([f"K{i:013d}" for i in range(len(smis))], dtype="S14")

    an = Analog.__new__(Analog)
    an.key, an.smiles, an.mass, an.fp = key, np.array(smis, dtype="S300"), mass, fp
    an.order = np.argsort(mass, kind="stable")
    an.msorted = mass[an.order]
    an.row = _Rows(key)
    an.popc = np.bitwise_count(fp).sum(1).astype(np.int32)
    an.n_internal = len(key)      # nothing external in the fixture

    # Aspirin is 180.0423; [M+H]+ puts it at 181.0495.
    got = an.pool(180.0423 + PROTON, ["[M+H]+"])
    assert list(got) == [0], (got, mass)
    assert len(an.pool(999.0, ["[M+H]+"])) == 0, "nothing should match a wild mass"
    # [M+Na]+ of the same molecule must find it too, from a different m/z.
    assert list(an.pool(180.0423 + 22.9897697 - E, ["[M+Na]+"])) == [0]

    t = an.tanimoto(np.arange(4), np.array([0]))
    assert abs(t[0, 0] - 1.0) < 1e-12, "a molecule must be identical to itself"
    assert t[1, 0] > t[3, 0], "the methyl ester is a closer analog than benzene"
    assert np.all((t >= 0) & (t <= 1))

    # Propagation must surface the analog of a seed that itself scores well, at a
    # score strictly below the seed's own -- it is similarity-discounted evidence.
    out = an.propagate({key[1]: 1.0}, 180.0423 + PROTON, ["[M+H]+"])
    assert key[0] in out and 0 < out[key[0]] < 1, out

    # A class 3 style ban must remove the structure from the pool entirely, or the
    # split leaks. Rebuild the same object with key[0] banned and re-propagate.
    banned = Analog.__new__(Analog)
    keep = ~np.isin(key, np.array([key[0]]))
    banned.key, banned.mass, banned.fp = key[keep], mass[keep], fp[keep]
    banned.smiles = np.array(smis, dtype="S300")[keep]
    banned.order = np.argsort(banned.mass, kind="stable")
    banned.msorted = banned.mass[banned.order]
    banned.row = _Rows(banned.key)
    banned.popc = np.bitwise_count(banned.fp).sum(1).astype(np.int32)
    banned.n_internal = len(banned.key)
    assert key[0] not in banned.propagate({key[1]: 1.0}, 180.0423 + PROTON, ["[M+H]+"])
    print("analog self-checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--build" in sys.argv:
        build()
    elif "--submit" in sys.argv:
        submit()
    else:
        evaluate()
