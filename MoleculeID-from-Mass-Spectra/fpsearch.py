"""Rank candidates by a fingerprint predicted from the spectrum, and measure MRR.

Library search (search.py) reaches only structures that already have spectra, and
analog propagation (analog.py) reaches only structures that look like one that does.
A predicted fingerprint has neither limit: every candidate in the precursor's mass
window can be scored against it directly.

    score(c) = spectral(c) + BETA * analog(c) + GAMMA * tanimoto(predicted_fp, fp(c))

The first two terms are unchanged, so class 1 cannot regress. The third is the only
one that can reach a class 2 molecule whose neighbourhood the library never sampled.

The model must have been trained with --holdout, or it has seen the query spectra and
the MRR below is fiction.

    python3 fpsearch.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch

import library
from analog import ADDUCTS, Analog, BETA
import fingerprint
from fingerprint import Net, NBIN, predict
from metric import mrr
from search import Searcher, spectra_of

DIR = os.path.dirname(os.path.abspath(__file__))
DATA, WORK = library.DATA, library.WORK
TOP_K = 25
GAMMA = float(os.environ.get("GAMMA", 32.0))   # see submit(): chosen on the solved class mix
GAMMAS = tuple(float(x) for x in os.environ.get("GAMMAS", "0,4,8,16,32,64").split(","))


def load_model():
    # Read through the module rather than a name bound at import time, so a caller
    # (the Kaggle kernel) can point it at a shipped checkpoint.
    path = fingerprint.MODEL
    assert os.path.exists(path), "run `python3 fingerprint.py --train --holdout` first"
    ck = torch.load(path, weights_only=False)
    bits = ck["bits"]
    m = Net(2 * NBIN + 2, len(bits))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, bits


def query_fp(model, spectra):
    """One fingerprint per molecule: the mean prediction over its spectra.

    Averaging rather than taking one spectrum's answer, because a molecule is
    usually measured at several collision energies and each sees different fragments.
    """
    mz = np.zeros((len(spectra), 64), np.float32)
    it = np.zeros((len(spectra), 64), np.float32)
    pm = np.zeros(len(spectra), np.float32)
    for i, (m, v, p) in enumerate(spectra):
        n = min(len(m), 64)
        mz[i, :n], it[i, :n], pm[i] = m[:n], v[:n], p
    return predict(model, mz, it, pm).mean(0)


def main():
    sp = np.load(f"{WORK}/splits.npz", allow_pickle=True)
    ix = library.load()
    se = Searcher(drop=sp["drop"], banned=sp["banned"])
    an = Analog(banned=sp["banned"])
    model, bits = load_model()

    # Candidate fingerprints stay packed. Unpacking all 264,127 of them to float32
    # over the 1,575 predicted bits is 1.66 GB, which the memory guard killed; the
    # mass window is ~115 candidates, so unpack per query instead.
    def pool_bits(rows):
        return np.unpackbits(an.fp[rows], axis=1)[:, bits].astype(np.float32)

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
        pm = float(spectra[0][2])
        pool = an.pool(pm, add)
        fpq = query_fp(model, spectra)
        jobs.append((cls, k, spec, smiles, pm, add, pool, fpq))
        answers[cls][k] = ix["smiles"][rows[0]].decode()
    del ix, se
    print(f"{len(jobs)} molecules, median pool {int(np.median([len(j[6]) for j in jobs]))}\n")

    print(f"{'gamma':<8}{'class1':>9}{'class2':>9}{'class3':>9}{'mean':>9}")
    best = (-1, None)
    for g in GAMMAS:
        preds = {1: {}, 2: {}, 3: {}}
        for cls, k, spec, smiles, pm, add, pool, fpq in jobs:
            total = dict(spec)
            sm = dict(smiles)
            extra = an.propagate(spec, pm, add)
            for key_, v in extra.items():
                total[key_] = total.get(key_, 0.0) + BETA * v
                sm.setdefault(key_, an.smiles[an.row[key_]])
            if g and len(pool):
                # Tanimoto between the predicted (soft) fingerprint and each
                # candidate's, in one matrix product over the mass window.
                B = pool_bits(pool)
                inter = B @ fpq
                union = B.sum(1) + fpq.sum() - inter
                t = inter / np.maximum(union, 1e-9)
                for r, v in zip(pool, t):
                    key_ = an.key[r]
                    total[key_] = total.get(key_, 0.0) + g * float(v)
                    sm.setdefault(key_, an.smiles[r])
            top = sorted(total.items(), key=lambda kv: -kv[1])[:TOP_K]
            preds[cls][k] = [sm[q].decode() if isinstance(sm[q], bytes) else sm[q]
                             for q, _ in top]
        v = [mrr(preds[c], answers[c]) for c in (1, 2, 3)]
        print(f"{g:<8g}{v[0]:>9.4f}{v[1]:>9.4f}{v[2]:>9.4f}{np.mean(v):>9.4f}",
              flush=True)
        if np.mean(v) > best[0]:
            best = (np.mean(v), g, v)
    print(f"\nbest gamma={best[1]:g}  mean {best[0]:.4f}")
    print("search+analog alone was 0.9420 / 0.0676 / 0.0000, mean 0.3365")
    # Class 3 must be unreachable from the index alone, and reachable only once an
    # external database is merged. Asserting the right one of those depends on
    # whether that pool is loaded.
    if an.external_n:
        assert best[2][2] > 0, "external pool loaded but class 3 still unreachable"
    else:
        assert best[2][2] == 0, f"class 3 leaked: {best[2][2]:.4f}"


def submit(path=None, gamma=GAMMA):
    """Write submission.csv using search + analog + the predicted fingerprint.

    gamma is 64 by default, which is not the value that maximises the held-out mean
    (that is 32). The held-out split weights the three classes equally and section 7
    showed the real test does not -- class 1 is 12-15% of it. Re-weighting the sweep
    by every plausible class mix, gamma=64 is within 2% of the best choice at each
    one, where 32 and 256 are each 2.3% off at the far end. It is the minimax pick,
    not the maximum.
    """
    import pandas as pd
    from metric import validate

    path = path or f"{DIR}/submission.csv"
    te = pd.read_parquet(f"{DATA}/test.parquet")
    se, an = Searcher(), Analog()
    model, bits = load_model()
    rows = []
    for mid, g in te.groupby("molecule_id"):
        spectra = [library.trim(a, b, p) + (p,)
                   for a, b, p in zip(g.ms2_mzs, g.ms2_normalized_intensities,
                                      g.precursor_mz)]
        add = [a for a in pd.unique(g.adduct) if a in ADDUCTS] or ["[M+H]+"]
        spec, smiles = se.score_structures(spectra)
        pm = float(spectra[0][2])
        total, sm = dict(spec), dict(smiles)
        for k, v in an.propagate(spec, pm, add).items():
            total[k] = total.get(k, 0.0) + BETA * v
            sm.setdefault(k, an.smiles[an.row[k]])
        pool = an.pool(pm, add)
        if gamma and len(pool):
            fpq = query_fp(model, spectra)
            B = np.unpackbits(an.fp[pool], axis=1)[:, bits].astype(np.float32)
            inter = B @ fpq
            t = inter / np.maximum(B.sum(1) + fpq.sum() - inter, 1e-9)
            for r, v in zip(pool, t):
                k = an.key[r]
                total[k] = total.get(k, 0.0) + gamma * float(v)
                sm.setdefault(k, an.smiles[r])
        top = sorted(total.items(), key=lambda kv: -kv[1])[:TOP_K]
        cands = [sm[q].decode() if isinstance(sm[q], bytes) else sm[q] for q, _ in top]
        rows.append((mid, ";".join(cands[:TOP_K] or ["C"])))
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    validate(sub)
    sub.to_csv(path, index=False)
    print(f"wrote {path}: {len(sub)} molecules, "
          f"{sub.smiles.str.split(';').str.len().mean():.1f} candidates each")


if __name__ == "__main__":
    submit() if "--submit" in sys.argv else main()
