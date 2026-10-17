"""Does a bigger candidate database pay for itself?

Section 8 of RESULTS.md leaves one question: class 3 is coverage-limited, so a larger
structure database is the only lever left -- but the last pool expansion cost class 1
and class 2 about 0.02 each, and a PubChem-scale pool is ~100x larger again. Whether
the net is positive is not obvious, and downloading 100M structures to find out is
several hours and 30 GB.

It can be answered without the download. Randomly subsampling the external pool to a
fraction f cuts coverage to ~f and cuts the number of competing candidates by the same
factor. Sweeping f traces both curves at once:

  * class 3 MRR against coverage   -> ranking quality among covered molecules
  * class 1/2 MRR against pool size -> the dilution cost per extra candidate

Extrapolating the second past f=1 is the sanity check the expansion needs. The first
also tests the 0.360 ranking-quality figure section 8 derived from a single point.

    GAMMA=64 MIN_AVAIL_MB=3000 ../guard.sh python3 -u dilution.py
"""
import functools
import os
import sys

import numpy as np

import library
from analog import ADDUCTS, Analog, BETA
import fingerprint
import formula
from fpsearch import load_model, query_fp, TOP_K
from metric import mrr
from search import Searcher, spectra_of

WORK = library.WORK
GAMMA = float(os.environ.get("GAMMA", 64.0))
FRACS = tuple(float(x) for x in os.environ.get("FRACS", "0,0.125,0.25,0.5,1").split(","))
# MODE=ppm holds the database fixed and widens the neutral-mass window instead.
# Subsampling can only shrink the pool, so it measures the dilution slope over
# one doubling and an expansion needs three to seven. Widening the window adds
# candidates without removing the answer, so it reaches pool sizes a real
# database expansion would produce and says whether the slope stays log-linear
# that far out. Those candidates have the wrong mass, which the ranking never
# looks at -- every term scores a structure, not its mass -- so they compete on
# the same footing as the right-mass ones a bigger database would add.
MODE = os.environ.get("MODE", "frac")
PPMS = tuple(float(x) for x in os.environ.get("PPMS", "5,10,20,40,80,160").split(","))

# The solved class mix (RESULTS.md section 7), for turning held-out numbers into a
# leaderboard prediction.
MIX = {1: 0.140, 2: 0.126, 3: 0.735}
# SPLITS=data/splits_cov3.npz points this at the variant whose class 3 is fully
# covered by the external database, where class3 MRR *is* the ranking quality.
SPLITS = os.environ.get("SPLITS", f"{WORK}/splits.npz")
# FORMULA_ORACLE=1 drops every candidate whose formula differs from the answer's: the
# ceiling a perfect formula filter would reach (RESULTS.md section 12, item 1).
ORACLE = os.environ.get("FORMULA_ORACLE") == "1"
# DELTAS sweeps the weight on formula.explained, the fraction of spectrum intensity a
# candidate's formula can account for (section 12, item 2).
# TRUE_ADDUCT=1 (default) opens only the first query spectrum's own adduct window, as
# submit() does with the adduct column the test ships. The index does not store
# adducts, so it is recovered from the answer's mass; 0 opens every adduct of the
# polarity, which is what this eval did before RESULTS.md section 14.
TRUE_ADDUCT = os.environ.get("TRUE_ADDUCT", "1") == "1"
# SPEC_MASS drops spectral matches whose structure cannot have the query's neutral
# mass: the precursor window ignores adducts, so an [M+Na]+ reference of a molecule
# 22 Da lighter sits in an [M+H]+ query's window. "direct" filters only their own
# score and keeps them as analog seeds; "all" drops them as seeds too.
SPEC_MASS = os.environ.get("SPEC_MASS", "0")
SPEC_PPM = float(os.environ.get("SPEC_PPM", 10.0))
# GATES sweeps a gated gamma: a molecule whose best spectral match (mean cosine per
# spectrum) reaches the gate keeps GAMMA, the rest get GAMMA_HI. A strong library
# match is class 1, where a high gamma lets fingerprint-similar decoys outvote it.
# The default 0 is no gate: every molecule gets GAMMA, as submit() does. It was 2 until
# 2026-10-17, which no mean cosine reaches, so every eval silently used GAMMA_HI (128)
# whatever GAMMA said (RESULTS.md section 24).
GATES = tuple(float(x) for x in os.environ.get("GATES", "0").split(","))
GAMMA_HI = float(os.environ.get("GAMMA_HI", 128.0))
DELTAS = tuple(float(x) for x in os.environ.get("DELTAS", "0").split(","))
# C2_BANNED=1 also removes class 2 answers from the candidate list: tests whether the
# held-out class 2 gain from a higher gamma, absent on the leaderboard, needs the
# answer to sit in the index's own structures.
C2_BANNED = os.environ.get("C2_BANNED") == "1"


@functools.lru_cache(maxsize=None)
def formula_of(smi):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    m = Chem.MolFromSmiles(smi)
    return CalcMolFormula(m) if m is not None else None


def build_jobs():
    """Spectral scores and predicted fingerprints, which do not depend on the pool."""
    sp = dict(np.load(SPLITS, allow_pickle=True))
    if C2_BANNED:
        sp["banned"] = np.union1d(sp["banned"], sp["qkey"][sp["qclass"] == 2])
    ix = library.load()
    se = Searcher(drop=sp["drop"], banned=sp["banned"], ix=ix)
    model, bits = load_model()

    pos = [a for a, (_, p) in ADDUCTS.items() if p]
    neg = [a for a, (_, p) in ADDUCTS.items() if not p]
    qkey, qclass, query = sp["qkey"], sp["qclass"], sp["query"]

    jobs, answers = [], {1: {}, 2: {}, 3: {}}
    for k in np.unique(qkey):
        m = qkey == k
        cls, rows = int(qclass[m][0]), query[m]
        spectra = spectra_of(ix, rows)
        add = pos if bool(ix["ion"][rows[0]]) else neg
        if TRUE_ADDUCT:
            mass = formula.counts(ix["smiles"][rows[0]].decode())[1]
            add = [min(add, key=lambda a: abs(mass + ADDUCTS[a][0] - spectra[0][2]))]
        spec, smiles = se.score_structures(spectra)
        jobs.append((cls, k, spec, smiles, float(spectra[0][2]), add,
                     query_fp(model, spectra), spectra))
        answers[cls][k] = ix["smiles"][rows[0]].decode()
    del ix, se, model
    return jobs, answers, sp["banned"], bits


_explained = {}


def explained(j, k, smi, spectra, add):
    """Mean over a molecule's spectra, memoised so a DELTAS sweep computes it once."""
    if (j, k) not in _explained:
        _explained[j, k] = float(np.mean([formula.explained(smi, mz, it, p, add)
                                          for mz, it, p in spectra]))
    return _explained[j, k]


def fits(an, key_, pm, add):
    if key_ not in an.row:
        return False
    m = float(an.mass[an.row[key_]])
    return any(abs(m - (pm - ADDUCTS[a][0])) <= m * SPEC_PPM * 1e-6 for a in add)


def score(an, bits, jobs, answers, gamma=GAMMA, ppm=None, delta=0.0, gate=0.0):
    preds, sizes, kept = {1: {}, 2: {}, 3: {}}, [], []
    for j, (cls, k, spec, smiles, pm, add, fpq, spectra) in enumerate(jobs):
        if SPEC_MASS != "0":
            fit = {q: v for q, v in spec.items() if fits(an, q, pm, add)}
            total = dict(fit)
            if SPEC_MASS == "all":
                spec = fit
        else:
            total = dict(spec)
        sm = dict(smiles)
        kw = {} if ppm is None else {"ppm": ppm}
        for key_, v in an.propagate(spec, pm, add, **kw).items():
            total[key_] = total.get(key_, 0.0) + BETA * v
            sm.setdefault(key_, an.smiles[an.row[key_]])
        pool = an.pool(pm, add, **kw)
        sizes.append(len(pool))
        top_spec = max(spec.values(), default=0.0) / len(spectra)
        g = gamma if top_spec >= gate else GAMMA_HI
        if g and len(pool):
            B = np.unpackbits(an.fp[pool], axis=1)[:, bits].astype(np.float32)
            inter = B @ fpq
            t = inter / np.maximum(B.sum(1) + fpq.sum() - inter, 1e-9)
            for r, v in zip(pool, t):
                key_ = an.key[r]
                total[key_] = total.get(key_, 0.0) + g * float(v)
                sm.setdefault(key_, an.smiles[r])
        if delta:
            for key_ in total:
                smi = sm[key_].decode() if isinstance(sm[key_], bytes) else sm[key_]
                total[key_] += delta * explained(j, key_, smi, spectra, add)
        top = sorted(total.items(), key=lambda kv: -kv[1])
        top = [(sm[q].decode() if isinstance(sm[q], bytes) else sm[q]) for q, _ in top]
        if ORACLE:
            want = formula_of(answers[cls][k])
            n = len(top)
            top = [t for t in top if formula_of(t) == want]
            kept.append(len(top) / max(n, 1))
        preds[cls][k] = top[:TOP_K]
    if kept:
        print(f"  oracle keeps a median {np.median(kept):.1%} of candidates", flush=True)
    return [mrr(preds[c], answers[c]) for c in (1, 2, 3)], int(np.median(sizes))


def coverage(an, jobs):
    """Fraction of class 3 queries whose own structure is somewhere in the pool."""
    pool = set(an.key.tolist())
    keys = [k for cls, k, *_ in jobs if cls == 3]
    return sum(k in pool for k in keys) / len(keys)


def main():
    jobs, answers, banned, bits = build_jobs()
    print(f"\n{'frac':>6}{'delta':>7}{'pool':>8}{'cover3':>9}{'class1':>9}{'class2':>9}"
          f"{'class3':>9}{'quality':>9}{'LB':>8}")
    out = []
    an = Analog(banned=banned, external=True) if MODE == "ppm" else None
    for f in (PPMS if MODE == "ppm" else FRACS):
        if MODE != "ppm":
            os.environ["EXT_FRAC"] = str(f)
            an = Analog(banned=banned, external=f > 0)
        cov = coverage(an, jobs)
        for d, gt in [(d, gt) for d in DELTAS for gt in GATES]:
            v, med = score(an, bits, jobs, answers,
                           ppm=f if MODE == "ppm" else None, delta=d, gate=gt)
            if len(GATES) > 1:
                print(f"  gate {gt:g}", end="")
            # MRR among molecules that are actually in the pool. If the extrapolation
            # in section 8 is sound this is flat across fractions; if it rises with f
            # the model is wrong and coverage buys less than the projection claims.
            q = v[2] / cov if cov else float("nan")
            lb = sum(MIX[c] * v[i] for i, c in enumerate((1, 2, 3)))
            print(f"{f:>6g}{d:>7g}{med:>8}{cov:>9.1%}{v[0]:>9.4f}{v[1]:>9.4f}"
                  f"{v[2]:>9.4f}{q:>9.3f}{lb:>8.4f}", flush=True)
        out.append((f, med, cov, v, q))
        if MODE != "ppm":
            del an

    # Slope of each class against log2(pool). A ranking loses to distractors
    # logarithmically, so this is the coefficient an extrapolation needs, and fitting
    # all three separately keeps class 3 honest: on the covered split its slope is
    # the decay of ranking quality, which is what a larger database has to outrun.
    pts = [(np.log2(m), v) for f, m, _, v, _ in out if f]
    if len(pts) > 1:
        x = np.array([p[0] for p in pts])
        print()
        for i, c in enumerate((1, 2, 3)):
            y = np.array([p[1][i] for p in pts])
            print(f"  class {c}: {np.polyfit(x, y, 1)[0]:+.4f} per doubling of pool")
        y = np.array([sum(MIX[c] * p[1][i] for i, c in enumerate((1, 2, 3)))
                      for p in pts])
        print(f"  LB     : {np.polyfit(x, y, 1)[0]:+.4f} per doubling "
              "(at fixed coverage this is pure cost)")


if __name__ == "__main__":
    main()
