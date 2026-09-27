"""Did mixing focal calls into soundscape backgrounds close the domain gap?

Two questions, two instruments.

**The cold-start cohort.** 30 classes have no BirdNET column and no labelled
positive, and are currently served by a head fitted on raw focal audio, worth +0.021
on the leaderboard. They cannot be validated directly, so the stand-in is the same 19
classes coldstart.py uses: also absent from BirdNET, also with focal audio, but with
labels. Raw focal scores 0.7385 there. If the domain was the problem, mixtures should
beat that.

**The labelled classes.** 75 classes have real labels, so this needs no proxy. Raw
focal audio made them *worse* (section 5, 0.8756 -> 0.8265). If the domain was the
problem rather than the content, mixtures should not repeat that.

The mixtures contain no labelled-segment audio -- focal recordings and unlabelled
soundscapes only -- so one set serves every fold and nothing leaks.

    python3 mixeval.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import mixup
from coldstart import cohorts
from rung1c import focal_subsample

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
FOLDS = 5


def fit(X, y, w=None):
    return LogisticRegression(C=0.01, max_iter=2000,
                              class_weight="balanced").fit(X, y, sample_weight=w)


def cohort_auc(Xtr, ytr, Xva, Y, cols):
    """One-vs-rest heads fitted on (Xtr, ytr), scored against real labels."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xva)
    out = []
    for j in cols:
        pos = ytr == j
        if pos.sum() == 0:
            out.append(0.5)
            continue
        out.append(roc_auc_score(Y[:, j], fit(A, pos.astype(int)).predict_proba(B)[:, 1]))
    return np.array(out)


def main():
    c = cohorts()
    Y, emb, proxy = c["Y"], c["emb"], c["proxy"]
    mx = mixup.load()
    Xm, ym = mx["emb"].astype(np.float32), mx["label"].astype(int)
    Xf, yf = focal_subsample(mixup.PER_CLASS, Y.shape[1])
    print(f"{len(Xm):,} mixtures over {len(np.unique(ym))} classes; "
          f"{len(Xf):,} raw focal rows\n")

    print("cold-start cohort: 19 scoreable stand-ins for the 30 classes")
    print(f"{'training data':<26}{'rows':>8}{'mean AUC':>10}{'>0.5':>7}{'>0.7':>7}")
    res = {}
    for name, X, y in (("raw focal", Xf, yf),
                       ("mixtures", Xm, ym),
                       ("focal + mixtures", np.vstack([Xf, Xm]), np.r_[yf, ym])):
        a = cohort_auc(X, y, emb, Y, proxy)
        res[name] = a
        print(f"{name:<26}{len(X):>8,}{a.mean():>10.4f}"
              f"{int((a > 0.5).sum()):>7}{int((a > 0.7).sum()):>7}", flush=True)
    base = res["raw focal"]
    for name, a in res.items():
        if name != "raw focal":
            d = a - base
            sem = d.std(ddof=1) / np.sqrt(len(d))
            print(f"  {name} vs raw focal: {d.mean():+.4f} "
                  f"(sem {sem:.4f}, {d.mean() / sem if sem else 0:.1f} sigma, "
                  f"better on {int((d > 0).sum())}/{len(d)})")

    print("\nlabelled classes: real labels, no proxy")
    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    files = d["filename"]
    cols = np.where((Y.sum(0) > 0) & (Y.sum(0) < len(Y)))[0]
    Ym = np.zeros((len(Xm), Y.shape[1]), np.float32)
    Ym[np.arange(len(Xm)), ym] = 1.0
    print(f"{'training data':<26}{'weight':>8}{'mean AUC':>10}{'vs base':>9}")
    for name, w in (("labelled only", None), ("+ mixtures", 0.1),
                    ("+ mixtures", 0.3), ("+ mixtures", 1.0)):
        P = np.zeros((len(emb), Y.shape[1]), np.float32)
        for tr, va in GroupKFold(n_splits=FOLDS).split(emb, groups=files):
            sc = StandardScaler().fit(emb[tr])
            A, B = sc.transform(emb[tr]), sc.transform(emb[va])
            if w is None:
                Xa, Ya, sw = A, Y[tr], None
            else:
                Xa = np.vstack([A, sc.transform(Xm)])
                Ya = np.vstack([Y[tr], Ym])
                sw = np.r_[np.ones(len(tr)), np.full(len(Xm), w)]
            for j in cols:
                if not 0 < Ya[:, j].sum() < len(Ya):
                    P[va, j] = 0.5
                    continue
                P[va, j] = fit(Xa, Ya[:, j], sw).predict_proba(B)[:, 1]
        a = np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in cols])
        if w is None:
            lb = a
        print(f"{name:<26}{'-' if w is None else w:>8}{a:>10.4f}"
              f"{'' if w is None else f'{a - lb:+9.4f}'}", flush=True)
    print("\nraw focal audio scored 0.8265 here (section 5); labelled-only is 0.8756.")


if __name__ == "__main__":
    main()
