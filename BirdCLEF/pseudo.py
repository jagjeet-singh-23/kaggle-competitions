"""Rung 3: pseudo-label the 10,592 unlabelled soundscape recordings.

177 hours of audio from the scored domain, with no labels. Everything that has failed
here failed at the domain gap; this data has no gap to cross.

The teacher must be refitted inside every fold. Pseudo-labels produced by a model that
saw the validation segments carry information about them, so a single teacher trained
on all 739 would leak straight through the pseudo-labels into the student and report a
gain that does not exist. Per fold:

    1. fit the teacher on the training segments only
    2. label the unlabelled pool with it
    3. fit the student on training segments + pseudo-labels
    4. score the student on the held-out segments

The baseline is the teacher scored on the same held-out segments, so the comparison
isolates the pseudo-labels and nothing else.

    python3 pseudo.py               # the 75 classes that have labels
    python3 pseudo.py --zeroshot    # can zero-shot teach an in-domain head?
    python3 pseudo.py --selftest
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from birdnet import species_map
from cv import zero_shot
from embed_soundscape import load as load_unlabelled

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
FOLDS = 5
TOP_K = (25, 100, 400)       # pseudo-positives per class
WEIGHTS = (0.1, 0.3, 1.0)    # weight on a pseudo row relative to a labelled one


def fit(X, y, w=None):
    return LogisticRegression(C=0.01, max_iter=2000,
                              class_weight="balanced").fit(X, y, sample_weight=w)


def pseudo_rows(scores, k):
    """Top-k windows as positives, the bottom k as negatives.

    Top-k rather than a probability threshold: the teacher is fitted on at most a few
    hundred rows and its probabilities are not calibrated, so a fixed threshold moves
    an unpredictable number of rows per class while a rank cut does not.
    """
    o = np.argsort(-scores)
    return o[:k], o[-k:]


def run(Xu, label="unlabelled pool"):
    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    Y, files = d["Y"], d["filename"]
    Xl = d["emb"].astype(np.float32)
    cols = np.where(Y.sum(0) > 0)[0]
    folds = list(GroupKFold(n_splits=FOLDS).split(Xl, groups=files))
    print(f"{len(Xl)} labelled segments, {len(Xu):,} {label}, {len(cols)} classes\n")

    # Baseline: teacher only, exactly the current Rung 1b head on embeddings.
    base = np.zeros((len(Xl), len(cols)))
    teachers = {}
    for fi, (tr, va) in enumerate(folds):
        sc = StandardScaler().fit(Xl[tr])
        A, B, U = sc.transform(Xl[tr]), sc.transform(Xl[va]), sc.transform(Xu)
        teachers[fi] = (sc, A, B, U, tr, va)
        for c, j in enumerate(cols):
            if Y[tr][:, j].sum() == 0:
                base[va, c] = 0.5
                continue
            base[va, c] = fit(A, Y[tr][:, j]).predict_proba(B)[:, 1]
    b = np.array([roc_auc_score(Y[:, j], base[:, c]) for c, j in enumerate(cols)])
    print(f"baseline (teacher only): {b.mean():.4f}\n")

    print(f"{'top-k':<8}{'weight':>8}{'AUC':>9}{'vs base':>9}{'classes up':>12}")
    best = (-1, None)
    for k in TOP_K:
        for w in WEIGHTS:
            P = np.zeros_like(base)
            for fi, (tr, va) in enumerate(folds):
                sc, A, B, U, _, _ = teachers[fi]
                for c, j in enumerate(cols):
                    if Y[tr][:, j].sum() == 0:
                        P[va, c] = 0.5
                        continue
                    t = fit(A, Y[tr][:, j])
                    pos, neg = pseudo_rows(t.predict_proba(U)[:, 1], k)
                    Xs = np.vstack([A, U[pos], U[neg]])
                    ys = np.r_[Y[tr][:, j], np.ones(len(pos)), np.zeros(len(neg))]
                    ws = np.r_[np.ones(len(A)), np.full(len(pos) + len(neg), w)]
                    P[va, c] = fit(Xs, ys, ws).predict_proba(B)[:, 1]
            a = np.array([roc_auc_score(Y[:, j], P[:, c]) for c, j in enumerate(cols)])
            print(f"{k:<8}{w:>8g}{a.mean():>9.4f}{a.mean() - b.mean():>+9.4f}"
                  f"{int((a > b).sum()):>8}/{len(cols)}", flush=True)
            if a.mean() > best[0]:
                best = (a.mean(), (k, w))
    print(f"\nbest: top-k={best[1][0]} weight={best[1][1]:g}  {best[0]:.4f} "
          f"({best[0] - b.mean():+.4f} over {b.mean():.4f})")
    return b.mean(), best


def zeroshot_teacher(Xu):
    """Can BirdNET's own zero-shot logits teach an in-domain head?

    This is the question for the 129 classes that are mapped into BirdNET but have no
    labelled positive. They currently take the raw logits. If a head trained on the
    logits' most confident in-domain detections beats the logits themselves, those 129
    classes -- 55% of the metric -- all improve at once.

    Measured on the 28 classes that are mapped *and* labelled, the only scoreable
    stand-in. Their labels score, never teach.
    """
    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    Y, Xl = d["Y"], d["emb"].astype(np.float32)
    Zl, hit = zero_shot(d["logits"], tax)
    both = np.where(hit & (Y.sum(0) > 0))[0]

    # The unlabelled pool carries the 157 mapped logit columns; line them up with the
    # competition classes the same way zero_shot does.
    idx = species_map(tax)
    col_of = {j: c for c, j in enumerate(np.where(idx >= 0)[0])}
    Lu = Xu["logits"].astype(np.float32)
    Eu = Xu["emb"].astype(np.float32)

    sc = StandardScaler().fit(Eu)
    U, L = sc.transform(Eu), sc.transform(Xl)
    print(f"zero-shot as teacher, measured on {len(both)} mapped+labelled classes\n")
    print(f"{'top-k':<8}{'zero-shot':>11}{'taught head':>13}{'blend':>9}")
    for k in TOP_K:
        zs, st, bl = [], [], []
        for j in both:
            z = roc_auc_score(Y[:, j], Zl[:, j])
            pos, neg = pseudo_rows(Lu[:, col_of[j]], k)
            Xs = np.vstack([U[pos], U[neg]])
            ys = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            p = fit(Xs, ys).predict_proba(L)[:, 1]
            r = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)
            zs.append(z)
            st.append(roc_auc_score(Y[:, j], p))
            bl.append(roc_auc_score(Y[:, j], 0.5 * r(p) + 0.5 * r(Zl[:, j])))
        print(f"{k:<8}{np.mean(zs):>11.4f}{np.mean(st):>13.4f}{np.mean(bl):>9.4f}",
              flush=True)


def selftest():
    """pseudo_rows must pick the extremes, and the fold teacher must never see val."""
    s = np.array([0.9, 0.1, 0.5, 0.99, 0.01])
    pos, neg = pseudo_rows(s, 2)
    assert list(pos) == [3, 0], pos
    assert set(neg) == {4, 1}, neg
    assert len(set(pos) & set(neg)) == 0

    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    files = d["filename"]
    for tr, va in GroupKFold(n_splits=FOLDS).split(d["emb"], groups=files):
        assert not (set(files[tr]) & set(files[va])), "a file spans both sides"
    print("pseudo self-checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        n = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--shards=")), None)
        u = load_unlabelled(max_shards=n)
        print(f"unlabelled pool: {len(u['emb']):,} windows "
              f"from {len(np.unique(u['filename'])):,} files\n")
        if "--zeroshot" in sys.argv:
            zeroshot_teacher(u)
        else:
            run(u["emb"].astype(np.float32))
