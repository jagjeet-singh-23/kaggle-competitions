"""Exploit the one structure every model here has ignored: time.

Sections 5, 8, 9 and 10 all attacked the features -- more audio, different audio, a
bigger head, more of BirdNET's outputs. Three of the four returned nothing, and
section 10 established why: the frozen representation is the ceiling.

This does not touch the features. A soundscape is continuous, and a bird calling at
t=10s is usually still there at t=15s, so the 12 windows of a recording are not 12
independent problems. Every model in this repo has treated them as though they were.

The labelled recordings are fully covered -- 12 segments each, 99.26% of consecutive
pairs exactly 5 s apart -- and the test recordings have the same shape, so whatever
works here transfers directly.

    python3 smooth.py
    python3 smooth.py --selftest
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from birdnet import species_map
from mlp import heads

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
FOLDS = 5


def neighbour_stat(P, files, starts, k, how):
    """For each row, summarise the k windows either side of it within its own file.

    Files are handled separately and the row order is restored at the end, so this
    never mixes two recordings and never assumes the input is sorted.
    """
    out = np.empty_like(P)
    order = np.lexsort((starts, files))
    for f in np.unique(files):
        idx = order[files[order] == f]
        Q = P[idx]
        n = len(Q)
        acc = []
        for off in range(-k, k + 1):
            # Counted explicitly rather than sliced with computed bounds: when a file
            # is shorter than the window, the naive end index goes negative and numpy
            # reads it as "from the end" instead of "empty".
            src_lo = max(0, off)
            cnt = max(0, min(n, n + off) - src_lo)
            dst_lo = max(0, -off)
            pad = np.full_like(Q, np.nan)
            if cnt:
                pad[dst_lo:dst_lo + cnt] = Q[src_lo:src_lo + cnt]
            acc.append(pad)
        A = np.stack(acc)
        out[idx] = np.nanmax(A, 0) if how == "max" else np.nanmean(A, 0)
    return out


def baseline(X, Y, files, cols):
    P = np.zeros((len(X), Y.shape[1]), np.float32)
    for tr, va in GroupKFold(n_splits=FOLDS).split(X, groups=files):
        sc = StandardScaler().fit(X[tr])
        P[va] = heads(sc.transform(X[tr]), Y[tr], sc.transform(X[va]), cols)
    return P


def score(Y, P, cols):
    return np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in cols])


def main():
    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    Y, files, starts = d["Y"].astype(np.float32), d["filename"], d["start_s"]
    idx = species_map(tax)
    X = np.hstack([d["emb"].astype(np.float32),
                   d["logits"][:, idx[idx >= 0]].astype(np.float32)])
    cols = np.where((Y.sum(0) > 0) & (Y.sum(0) < len(Y)))[0]
    hit = (idx >= 0)[cols]

    P = baseline(X, Y, files, cols)
    b = score(Y, P, cols)
    print(f"baseline (no smoothing): {b:.4f}\n")
    assert abs(b - 0.8774) < 0.002, "baseline drifted"

    r = lambda M: np.argsort(np.argsort(M, axis=0), axis=0) / (len(M) - 1)
    Pr = r(P)
    print(f"{'k':>3}{'how':>6}{'alpha':>7}{'AUC':>9}{'vs base':>9}"
          f"{'birds':>9}{'others':>9}")
    best = (b, None)
    for k in (1, 2, 3):
        for how in ("mean", "max"):
            N = r(neighbour_stat(Pr, files, starts, k, how))
            for alpha in (0.9, 0.7, 0.5, 0.3):
                # alpha keeps the window's own evidence; 1-alpha is its context.
                S = alpha * Pr + (1 - alpha) * N
                a = np.array([roc_auc_score(Y[:, j], S[:, j]) for j in cols])
                print(f"{k:>3}{how:>6}{alpha:>7g}{a.mean():>9.4f}"
                      f"{a.mean() - b:>+9.4f}{a[hit].mean():>9.4f}"
                      f"{a[~hit].mean():>9.4f}", flush=True)
                if a.mean() > best[0]:
                    best = (a.mean(), (k, how, alpha))
    if best[1] is None:
        print(f"\nNo smoother beat the unsmoothed baseline ({b:.4f}).")
    else:
        k, how, alpha = best[1]
        print(f"\nbest: k={k} {how} alpha={alpha:g}  {best[0]:.4f} ({best[0] - b:+.4f})")
    return best


def selftest():
    """Smoothing must respect file boundaries and leave a single-row file alone."""
    P = np.array([[0.0], [1.0], [0.0], [10.0], [20.0]], np.float32)
    files = np.array(["a", "a", "a", "b", "b"])
    starts = np.array([0, 5, 10, 0, 5])

    m = neighbour_stat(P, files, starts, 1, "mean")
    # File "a" alone: [0,1,0] -> [0.5, 1/3, 0.5]. File "b" must not leak in.
    assert np.allclose(m[:3, 0], [0.5, 1 / 3, 0.5]), m[:3, 0]
    assert np.allclose(m[3:, 0], [15.0, 15.0]), m[3:, 0]

    mx = neighbour_stat(P, files, starts, 1, "max")
    assert np.allclose(mx[:3, 0], [1.0, 1.0, 1.0]), mx[:3, 0]
    assert mx[3, 0] == 20.0 and mx[4, 0] == 20.0

    # Unsorted input must give the same answer as sorted input.
    perm = np.array([3, 0, 4, 2, 1])
    m2 = neighbour_stat(P[perm], files[perm], starts[perm], 1, "mean")
    assert np.allclose(m2, m[perm]), "result must not depend on row order"

    # A file shorter than the window is its own neighbourhood, at every k. This is
    # the case that crashed: the computed end index went negative and numpy read it
    # as an offset from the end rather than as an empty slice.
    for k in (1, 2, 3, 5):
        one = neighbour_stat(np.array([[7.0]]), np.array(["z"]), np.array([0]), k, "mean")
        assert one[0, 0] == 7.0, (k, one)
        two = neighbour_stat(np.array([[1.0], [3.0]]), np.array(["z", "z"]),
                             np.array([0, 5]), k, "mean")
        assert np.allclose(two[:, 0], [2.0, 2.0]), (k, two)
    print("smooth self-checks passed")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
