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
from fingerprint import MODEL, Net, NBIN, predict
from metric import mrr
from search import Searcher, spectra_of

DATA, WORK = library.DATA, library.WORK
TOP_K = 25
GAMMAS = (0.0, 0.5, 1.0, 2.0, 4.0)


def load_model():
    assert os.path.exists(MODEL), "run `python3 fingerprint.py --train --holdout` first"
    ck = torch.load(MODEL, weights_only=False)
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

    # Candidate fingerprints, restricted to the bits the model predicts.
    cand_bits = np.unpackbits(an.fp, axis=1)[:, bits].astype(np.float32)
    cand_pop = cand_bits.sum(1)

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
                # candidate's, computed in one matrix product over the pool.
                inter = cand_bits[pool] @ fpq
                union = cand_pop[pool] + fpq.sum() - inter
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
    assert best[2][2] == 0, f"class 3 leaked: {best[2][2]:.4f}"


if __name__ == "__main__":
    main()
