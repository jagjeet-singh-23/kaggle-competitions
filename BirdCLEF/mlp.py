"""Is the linear probe the bottleneck, or is the representation?

Sections 5, 8 and 9 of RESULTS.md all end the same way: another source of audio,
another small or negative return. "Capacity" was the conclusion drawn there, but that
covers two claims and only one is expensive to act on.

    head capacity     the linear probe is too weak -> a nonlinear head fixes it,
                      backbone frozen, cheap
    representation    BirdNET's features have said all they can -> only fine-tuning
                      the backbone helps, which needs a trainable checkpoint and a
                      GPU. We have neither: the model ships as .tflite, which is
                      inference-only, and there is no CUDA device here.

So this settles the cheap claim first.

**Why two stages.** The obvious experiment -- swap the linear layer for an MLP and
retrain -- does not answer the question, because it changes two things at once. A
linear layer over 234 outputs is 234 independent problems; there is no shared trunk
for them to benefit from, and optimising all 276k parameters jointly converges far
worse per class than 234 dedicated solves. Measured: jointly trained, the *linear*
model scores 0.67 against sklearn's 0.8774 on identical folds and features, while a
single-class torch fit reproduces sklearn to three decimals. Joint training is a
handicap, not a treatment.

So the trunk is trained jointly to learn features, then thrown away except for its
hidden layer, and the final classifier is fitted by the same per-class sklearn call
the baseline uses. hidden=0 is an identity trunk and must reproduce 0.8774 exactly --
that is the harness self-check, not a hope.

    python3 mlp.py
    python3 mlp.py --selftest
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from birdnet import species_map

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
FOLDS = 5
POS_CLAMP = 50.0     # a class with one positive would otherwise weight 738
C_HEAD = 0.01        # the baseline's regularisation, held fixed across every arm


class Trunk(nn.Module):
    """One hidden layer, trained jointly, kept only for its activations."""

    def __init__(self, d_in, d_out, hidden, dropout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout))
        self.out = nn.Linear(hidden, d_out)

    def forward(self, x):
        return self.out(self.body(x))


def fit_trunk(X, Y, hidden, dropout, wd, epochs, lr=1e-2):
    torch.manual_seed(0)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    pos = Yt.sum(0)
    pw = torch.clamp((len(Yt) - pos) / pos.clamp(min=1), max=POS_CLAMP)
    pw[pos == 0] = 1.0
    m = Trunk(X.shape[1], Y.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(m(Xt), Yt).backward()
        opt.step()
    m.eval()
    return m


def encode(m, X):
    if m is None:
        return X
    with torch.no_grad():
        return m.body(torch.tensor(X)).numpy()


def heads(Htr, Ytr, Hva, cols):
    """The baseline's own classifier, applied to whatever features it is handed."""
    P = np.full((len(Hva), Ytr.shape[1]), 0.5)
    for j in cols:
        y = Ytr[:, j]
        if not 0 < y.sum() < len(y):
            continue
        P[:, j] = LogisticRegression(
            C=C_HEAD, max_iter=2000,
            class_weight="balanced").fit(Htr, y).predict_proba(Hva)[:, 1]
    return P


def features(d, tax):
    idx = species_map(tax)
    return np.hstack([d["emb"].astype(np.float32),
                      d["logits"][:, idx[idx >= 0]].astype(np.float32)])


def evaluate(X, Y, files, cols, hidden, dropout, wd, epochs):
    P = np.zeros((len(X), Y.shape[1]), np.float32)
    for tr, va in GroupKFold(n_splits=FOLDS).split(X, groups=files):
        sc = StandardScaler().fit(X[tr])
        A = sc.transform(X[tr]).astype(np.float32)
        B = sc.transform(X[va]).astype(np.float32)
        trunk = fit_trunk(A, Y[tr], hidden, dropout, wd, epochs) if hidden else None
        P[va] = heads(encode(trunk, A), Y[tr], encode(trunk, B), cols)
    return np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in cols])


def main():
    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    Y, files = d["Y"].astype(np.float32), d["filename"]
    cols = np.where((Y.sum(0) > 0) & (Y.sum(0) < len(Y)))[0]
    X = features(d, tax)
    print(f"{len(X)} segments, {X.shape[1]} features, {len(cols)} scored classes\n")

    anchor = evaluate(X, Y, files, cols, 0, 0.0, 0.0, 0)
    print(f"identity trunk (= sklearn baseline): {anchor:.4f}   expected 0.8774")
    assert abs(anchor - 0.8774) < 0.002, "harness does not reproduce the baseline"

    print(f"\n{'hidden':>7}{'drop':>6}{'wd':>7}{'epochs':>8}{'AUC':>9}{'vs base':>9}")
    best = (anchor, None)
    for hidden in (64, 256, 512):
        for dropout in (0.2, 0.5):
            for wd in (1e-2, 1e-1):
                for epochs in (500, 1500):
                    a = evaluate(X, Y, files, cols, hidden, dropout, wd, epochs)
                    print(f"{hidden:>7}{dropout:>6g}{wd:>7g}{epochs:>8}"
                          f"{a:>9.4f}{a - anchor:>+9.4f}", flush=True)
                    if a > best[0]:
                        best = (a, (hidden, dropout, wd, epochs))
    if best[1] is None:
        print(f"\nNo nonlinear trunk beat the linear probe ({anchor:.4f}).")
        print("The head is not the bottleneck; the representation is.")
    else:
        print(f"\nbest {best[0]:.4f} at hidden={best[1][0]} dropout={best[1][1]} "
              f"wd={best[1][2]} epochs={best[1][3]}  ({best[0] - anchor:+.4f})")


def selftest():
    """The identity trunk must be a no-op, and encode must preserve row count."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 6)).astype(np.float32)
    assert encode(None, X) is X, "hidden=0 must not transform the features"

    Y = np.zeros((30, 4), np.float32)
    Y[:, 0] = (X[:, 0] > 0)
    t = fit_trunk(X, Y, 8, 0.0, 1e-2, 50)
    H = encode(t, X)
    assert H.shape == (30, 8) and np.isfinite(H).all()

    P = heads(X, Y, X, np.arange(4))
    assert P.shape == (30, 4)
    # Class 0 is separable and must be learned; classes with no positive stay at 0.5.
    assert roc_auc_score(Y[:, 0], P[:, 0]) > 0.9
    assert np.allclose(P[:, 1], 0.5), "a class with no positive must stay constant"
    print("mlp self-checks passed")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
