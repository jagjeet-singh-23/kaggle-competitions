"""Local evaluation: the competition metric, and the zero-training BirdNET baseline.

The metric is macro-averaged ROC-AUC over the 234 species columns, skipping any
class with no true positive. Locally only 75 classes ever appear in the labelled
segments, so a local score is an average over those 75 and is not comparable to the
leaderboard in absolute terms. It is comparable between our own models, which is
what it is for.

    python3 cv.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from birdnet import species_map

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = f"{DIR}/data"


def macro_auc(Y, P, per_class=False):
    """Competition metric: mean AUC over classes that have at least one positive."""
    keep = (Y.sum(0) > 0) & (Y.sum(0) < len(Y))
    aucs = np.array([roc_auc_score(Y[:, j], P[:, j]) for j in np.where(keep)[0]])
    return (aucs, np.where(keep)[0]) if per_class else aucs.mean()


def load():
    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    return d, tax


def zero_shot(logits, tax):
    """BirdNET's own logits projected onto the 234 competition columns.

    Unmapped classes get a constant, which scores 0.5 AUC for them by construction.
    """
    idx = species_map(tax)
    P = np.full((len(logits), len(idx)), 0.5, np.float32)
    hit = idx >= 0
    P[:, hit] = logits[:, idx[hit]]
    return P, hit


def main():
    d, tax = load()
    Y, logits = d["Y"], d["logits"]
    print(f"{len(Y)} segments, {int((Y.sum(0) > 0).sum())} classes with positives\n")

    P, hit = zero_shot(logits, tax)
    aucs, cols = macro_auc(Y, P, per_class=True)
    print(f"zero-shot BirdNET, macro AUC over {len(cols)} scored classes: {aucs.mean():.4f}")

    # The interesting split: BirdNET knows the birds and has never heard the rest.
    fam = tax.class_name.values[cols]
    mapped = hit[cols]
    print(f"\n  mapped to BirdNET   ({mapped.sum():2d} classes): {aucs[mapped].mean():.4f}")
    print(f"  not in BirdNET      ({(~mapped).sum():2d} classes): {aucs[~mapped].mean():.4f}  (0.5 by construction)")
    print("\n  by taxon:")
    for f in pd.unique(fam):
        m = fam == f
        print(f"    {f:<10} {m.sum():>3d} classes  {aucs[m].mean():.4f}")

    print("\n  best and worst mapped classes:")
    order = np.argsort(-np.where(mapped, aucs, -1))
    for j in list(order[:5]) + list(order[mapped.sum() - 5:mapped.sum()]):
        row = tax.iloc[cols[j]]
        print(f"    {aucs[j]:.3f}  {row.common_name[:38]:<38} n={int(Y[:, cols[j]].sum()):>3d}")

    assert 0.0 <= aucs.mean() <= 1.0
    assert aucs[mapped].mean() > 0.5, "BirdNET should beat chance on species it knows"
    return aucs.mean()


if __name__ == "__main__":
    main()
