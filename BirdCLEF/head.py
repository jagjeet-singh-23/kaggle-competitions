"""Rung 1b: a linear head on BirdNET embeddings, scored against the zero-shot baseline.

Grouped by soundscape file. Segments from one recording share background, weather
and often the same individuals, so splitting within a file leaks and would read far
better than the leaderboard.

Only the 739 labelled in-domain segments are used here; train_audio embeddings are
still extracting. That makes this the honest floor for what a head can do with
in-domain data alone.

    python3 head.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from birdnet import species_map
from cv import macro_auc, zero_shot

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = f"{DIR}/data"
FOLDS = 5


def fit_predict(Xtr, Ytr, Xva):
    """One-vs-rest logistic regression, skipping classes with no positive in train.

    Heavily regularised on purpose: 1,024 dimensions against ~590 training rows per
    fold. A class absent from the training fold gets a constant, which is 0.5 AUC.
    """
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    P = np.full((len(Xva), Ytr.shape[1]), 0.5)
    for j in np.where(Ytr.sum(0) > 0)[0]:
        m = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")
        P[:, j] = m.fit(Xtr, Ytr[:, j]).predict_proba(Xva)[:, 1]
    return P


def cross_val(X, Y, groups):
    P = np.zeros_like(Y, dtype=float)
    for tr, va in GroupKFold(n_splits=FOLDS).split(X, groups=groups):
        P[va] = fit_predict(X[tr], Y[tr], X[va])
    return P


def rank(P):
    """Per-class rank normalisation. AUC only sees the order within a column, so this
    changes nothing on its own, but it puts columns on one scale before blending."""
    return np.argsort(np.argsort(P, axis=0), axis=0) / max(1, len(P) - 1)


def main():
    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    Y, emb, logits, files = d["Y"], d["emb"].astype(np.float32), d["logits"], d["filename"]

    Z, hit = zero_shot(logits, tax)
    scored = (Y.sum(0) > 0) & (Y.sum(0) < len(Y))
    cols = np.where(scored)[0]
    mapped = hit[cols]
    print(f"{len(Y)} segments, {len(cols)} scored classes "
          f"({mapped.sum()} known to BirdNET, {(~mapped).sum()} not)\n")

    runs = {"zero-shot BirdNET": Z}
    runs["head on embeddings"] = cross_val(emb, Y, files)
    runs["head on emb+logits"] = cross_val(
        np.hstack([emb, logits[:, species_map(tax)[species_map(tax) >= 0]]]), Y, files)
    runs["head + zero-shot"] = rank(runs["head on embeddings"]) + rank(Z)
    # Blending zero-shot into classes BirdNET has never heard only adds noise, so
    # mix it in for the mapped species and leave the rest to the head.
    sel = rank(runs["head on emb+logits"]).copy()
    sel[:, hit] = 0.5 * sel[:, hit] + 0.5 * rank(Z)[:, hit]
    runs["head + zero-shot (birds only)"] = sel

    print(f"{'model':<32}{'all':>8}{'birds':>8}{'others':>8}")
    for name, P in runs.items():
        a, _ = macro_auc(Y, P, per_class=True)
        print(f"{name:<32}{a.mean():>8.4f}{a[mapped].mean():>8.4f}{a[~mapped].mean():>8.4f}")

    base, _ = macro_auc(Y, Z, per_class=True)
    best = max(runs.items(), key=lambda kv: macro_auc(Y, kv[1]))
    print(f"\nbest: {best[0]}  {macro_auc(Y, best[1]):.4f}  "
          f"(zero-shot was {base.mean():.4f})")

    assert macro_auc(Y, runs["head on embeddings"]) > 0.5
    assert macro_auc(Y, best[1]) >= base.mean()


if __name__ == "__main__":
    main()
