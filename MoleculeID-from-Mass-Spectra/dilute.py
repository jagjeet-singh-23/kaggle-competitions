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
import os
import sys

import numpy as np

import library
from analog import ADDUCTS, Analog, BETA
import fingerprint
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


def build_jobs():
    """Spectral scores and predicted fingerprints, which do not depend on the pool."""
    sp = np.load(SPLITS, allow_pickle=True)
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
        spec, smiles = se.score_structures(spectra)
        jobs.append((cls, k, spec, smiles, float(spectra[0][2]), add,
                     query_fp(model, spectra)))
        answers[cls][k] = ix["smiles"][rows[0]].decode()
    del ix, se, model
    return jobs, answers, sp["banned"], bits


def score(an, bits, jobs, answers, gamma=GAMMA, ppm=None):
    preds, sizes = {1: {}, 2: {}, 3: {}}, []
    for cls, k, spec, smiles, pm, add, fpq in jobs:
        total, sm = dict(spec), dict(smiles)
        kw = {} if ppm is None else {"ppm": ppm}
        for key_, v in an.propagate(spec, pm, add, **kw).items():
            total[key_] = total.get(key_, 0.0) + BETA * v
            sm.setdefault(key_, an.smiles[an.row[key_]])
        pool = an.pool(pm, add, **kw)
        sizes.append(len(pool))
        if gamma and len(pool):
            B = np.unpackbits(an.fp[pool], axis=1)[:, bits].astype(np.float32)
            inter = B @ fpq
            t = inter / np.maximum(B.sum(1) + fpq.sum() - inter, 1e-9)
            for r, v in zip(pool, t):
                key_ = an.key[r]
                total[key_] = total.get(key_, 0.0) + gamma * float(v)
                sm.setdefault(key_, an.smiles[r])
        top = sorted(total.items(), key=lambda kv: -kv[1])[:TOP_K]
        preds[cls][k] = [sm[q].decode() if isinstance(sm[q], bytes) else sm[q]
                         for q, _ in top]
    return [mrr(preds[c], answers[c]) for c in (1, 2, 3)], int(np.median(sizes))


def coverage(an, jobs):
    """Fraction of class 3 queries whose own structure is somewhere in the pool."""
    pool = set(an.key.tolist())
    keys = [k for cls, k, *_ in jobs if cls == 3]
    return sum(k in pool for k in keys) / len(keys)


def main():
    jobs, answers, banned, bits = build_jobs()
    print(f"\n{'frac':>6}{'pool':>8}{'cover3':>9}{'class1':>9}{'class2':>9}"
          f"{'class3':>9}{'quality':>9}{'LB':>8}")
    out = []
    an = Analog(banned=banned, external=True) if MODE == "ppm" else None
    for f in (PPMS if MODE == "ppm" else FRACS):
        if MODE != "ppm":
            os.environ["EXT_FRAC"] = str(f)
            an = Analog(banned=banned, external=f > 0)
        cov = coverage(an, jobs)
        v, med = score(an, bits, jobs, answers,
                       ppm=f if MODE == "ppm" else None)
        if MODE != "ppm":
            del an
        # MRR among molecules that are actually in the pool. If the extrapolation in
        # section 8 is sound this is flat across fractions; if it rises with f the
        # model is wrong and coverage buys less than the projection claims.
        q = v[2] / cov if cov else float("nan")
        lb = sum(MIX[c] * v[i] for i, c in enumerate((1, 2, 3)))
        print(f"{f:>6g}{med:>8}{cov:>9.1%}{v[0]:>9.4f}{v[1]:>9.4f}{v[2]:>9.4f}"
              f"{q:>9.3f}{lb:>8.4f}", flush=True)
        out.append((f, med, cov, v, q))

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
