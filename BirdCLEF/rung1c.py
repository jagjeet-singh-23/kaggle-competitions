"""Rung 1c: train on focal audio and soundscape together, weighting the soundscape up.

The two sources disagree about almost everything except the species list. train_audio
is 333 hours of clean, close, single-bird recordings; the scored data is 1 hour of
distant, overlapping, noisy soundscape. Focal audio has 178x the rows and the wrong
distribution, so an unweighted fit is dominated by a domain nobody scores.

So the soundscape rows carry a weight and the sweep finds it. w=0 is focal-only
(transfer with no in-domain data), w=inf is Rung 1b (in-domain only, 739 rows).

Ridge rather than per-class logistic regression, because 234 classes x 5 folds x a
weight sweep of logistic fits on 132k x 1024 is hours, and ridge is one 1024x1024
solve for every class at once. AUC only reads the order within a column, so a
least-squares fit to the one-hot targets ranks about as well as a logistic one.

The trick that makes the sweep free: focal data is in *every* training fold and never
in validation, so its Gram matrix X'X and cross-product X'Y are computed once, in a
single streaming pass over the shards. Each fold and weight is then a 1024x1024
solve on Gf + w*Gs. Peak memory is one shard, about 32 MB.

    python3 rung1c.py
    python3 rung1c.py --selftest
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cv import macro_auc, zero_shot

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = f"{DIR}/data"
SHARDS = f"{DATA}/emb_train"
FOLDS = 5
D = 1024

WEIGHTS = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ALPHAS = [1e2, 1e3, 1e4, 1e5]


def shards():
    fs = sorted(glob.glob(f"{SHARDS}/shard_*.npz"))
    assert fs, f"no shards in {SHARDS}; run `python3 embed_train.py` first"
    return fs


def focal_stats(n_classes):
    """Stream the shards once and return (n, mean, std, Gram, cross-product).

    Gram and cross-product are accumulated on standardised-later data by keeping the
    raw sums as well: standardising afterwards is a rank-1 correction, but it is far
    less error-prone to just take two passes over 251 MB of shards.
    """
    s1 = np.zeros(D, np.float64)
    s2 = np.zeros(D, np.float64)
    n = 0
    for f in shards():
        with np.load(f) as d:
            X = d["emb"].astype(np.float32)[d["label"] >= 0]
        s1 += X.sum(0, dtype=np.float64)
        s2 += (X.astype(np.float64) ** 2).sum(0)
        n += len(X)
    mu = (s1 / n).astype(np.float32)
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)).astype(np.float32)

    G = np.zeros((D, D), np.float64)
    B = np.zeros((D, n_classes), np.float64)
    for f in shards():
        with np.load(f) as d:
            keep = d["label"] >= 0
            X = d["emb"].astype(np.float32)[keep]
            lab = d["label"][keep]
        X = (X - mu) / sd
        G += X.T @ X
        # One-hot targets, accumulated without ever materialising the (n, 234) matrix.
        np.add.at(B, (slice(None), lab), X.T.astype(np.float64))
    return n, mu, sd, G, B


def ridge(G, B, alpha):
    """Solve (G + alpha I) beta = B. No intercept: a per-class constant cannot
    change that class's AUC, which only reads the order within a column."""
    return np.linalg.solve(G + alpha * np.eye(len(G)), B)


def main():
    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    Ys, files = d["Y"].astype(np.float64), d["filename"]
    Xs_raw, logits = d["emb"].astype(np.float32), d["logits"]
    n_classes = Ys.shape[1]

    n_f, mu, sd, Gf, Bf = focal_stats(n_classes)
    Xs = ((Xs_raw - mu) / sd).astype(np.float64)
    print(f"focal {n_f:,} segments   soundscape {len(Xs)} segments "
          f"({n_f / len(Xs):.0f}x)\n")

    Z, hit = zero_shot(logits, tax)
    scored = (Ys.sum(0) > 0) & (Ys.sum(0) < len(Ys))
    mapped = hit[np.where(scored)[0]]

    folds = list(GroupKFold(n_splits=FOLDS).split(Xs, groups=files))
    best = (None, -1)
    print(f"{'w(soundscape)':<16}{'alpha':>9}{'all':>9}{'birds':>9}{'others':>9}")
    for w in WEIGHTS:
        for a in ALPHAS:
            P = np.zeros_like(Ys)
            for tr, va in folds:
                Xt, Yt = Xs[tr], Ys[tr]
                G = Gf + w * (Xt.T @ Xt)
                B = Bf + w * (Xt.T @ Yt)
                P[va] = Xs[va] @ ridge(G, B, a)
            auc, _ = macro_auc(Ys, P, per_class=True)
            if auc.mean() > best[1]:
                best = ((w, a, P), auc.mean())
            print(f"{w:<16g}{a:>9g}{auc.mean():>9.4f}"
                  f"{auc[mapped].mean():>9.4f}{auc[~mapped].mean():>9.4f}")

    (w, a, P), score = best
    print(f"\nbest ridge: w={w:g} alpha={a:g}  {score:.4f}")

    # Same bird-only blend as Rung 1b: zero-shot helps the 157 species BirdNET knows
    # and is a constant for the rest, so mixing it everywhere only adds noise.
    r = lambda M: np.argsort(np.argsort(M, axis=0), axis=0) / max(1, len(M) - 1)
    blend = r(P).copy()
    blend[:, hit] = 0.5 * blend[:, hit] + 0.5 * r(Z)[:, hit]
    ab, _ = macro_auc(Ys, blend, per_class=True)
    az, _ = macro_auc(Ys, Z, per_class=True)
    print(f"  + zero-shot (birds only): {ab.mean():.4f}"
          f"   birds {ab[mapped].mean():.4f}  others {ab[~mapped].mean():.4f}")
    print(f"\nzero-shot BirdNET was {az.mean():.4f}; Rung 1b (soundscape-only head) was 0.8881")

    assert score > az.mean(), "focal+soundscape ridge should beat zero-shot"
    np.savez(f"{DATA}/rung1c.npz", P=P, blend=blend, w=w, alpha=a)


def focal_subsample(per_class, n_classes, seed=0):
    """Up to `per_class` focal segments for each label, streamed from the shards.

    Capping matters twice. A balanced logistic fit on all 131,943 rows is 234 classes
    x 5 folds of lbfgs on a 132k x 1024 dense matrix, which is hours; and the focal
    class counts are very skewed, so an uncapped sample would hand the common species
    thousands of rows and the rare ones a handful.
    """
    rng = np.random.default_rng(seed)
    got = [[] for _ in range(n_classes)]
    for f in shards():
        with np.load(f) as d:
            keep = d["label"] >= 0
            X, lab = d["emb"][keep], d["label"][keep]
        for c in np.unique(lab):
            room = per_class - len(got[c])
            if room <= 0:
                continue
            idx = np.where(lab == c)[0]
            if len(idx) > room:
                idx = rng.choice(idx, room, replace=False)
            got[c].append(X[idx])
    Xs, ys = [], []
    for c, parts in enumerate(got):
        if parts:
            Xs.append(np.concatenate(parts))
            ys.append(np.full(len(Xs[-1]), c))
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


def logistic_run(caps=(0, 30, 150), weights=(10.0,), extra=((30, 1.0), (30, 100.0))):
    """Rung 1b's model exactly -- balanced one-vs-rest logistic -- with focal rows added.

    cap=0 is the anchor and must reproduce Rung 1b's "head on embeddings" row, 0.8756.
    (Rung 1b's headline 0.8881 is that head blended with zero-shot on the bird columns;
    the blend is applied once at the end, not inside the sweep.) Only the classes with
    a positive somewhere in the labelled data are fitted, since the rest cannot be
    scored anyway.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    Ys, files = d["Y"].astype(np.float64), d["filename"]
    Xs, logits = d["emb"].astype(np.float32), d["logits"]

    Z, hit = zero_shot(logits, tax)
    cols = np.where((Ys.sum(0) > 0) & (Ys.sum(0) < len(Ys)))[0]
    mapped = hit[cols]
    folds = list(GroupKFold(n_splits=FOLDS).split(Xs, groups=files))

    configs = [(c, w) for c in caps for w in weights if c] + list(extra)
    configs = [(0, 0.0)] + sorted(set(configs))
    cache = {}
    print(f"\n{'focal/class':<14}{'w':>8}{'focal rows':>12}"
          f"{'all':>9}{'birds':>9}{'others':>9}")
    for cap, w in configs:
        if cap and cap not in cache:
            cache[cap] = focal_subsample(cap, Ys.shape[1])
        Xf, yf = cache.get(cap, (np.zeros((0, D), np.float32), np.zeros(0, int)))

        P = np.full((len(Xs), Ys.shape[1]), 0.5)
        for tr, va in folds:
            Xt = np.vstack([Xf, Xs[tr]])
            sw = np.r_[np.ones(len(Xf)), np.full(len(tr), w or 1.0)]
            sc = StandardScaler().fit(Xt)
            Xt_s, Xv_s = sc.transform(Xt), sc.transform(Xs[va])
            for j in cols:
                # A focal clip labelled species j is a positive for j; the soundscape
                # rows bring their own multi-label truth.
                y = np.r_[(yf == j).astype(float), Ys[tr][:, j]]
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                m = LogisticRegression(C=0.01, max_iter=2000,
                                       class_weight="balanced")
                P[va, j] = m.fit(Xt_s, y, sample_weight=sw).predict_proba(Xv_s)[:, 1]
        auc, _ = macro_auc(Ys, P, per_class=True)
        print(f"{cap:<14}{w:>8g}{len(Xf):>12,}{auc.mean():>9.4f}"
              f"{auc[mapped].mean():>9.4f}{auc[~mapped].mean():>9.4f}", flush=True)
        if cap == 0:
            anchor = auc.mean()
    print(f"\nanchor (cap=0, soundscape only) {anchor:.4f} "
          f"vs Rung 1b 'head on embeddings' 0.8756")
    assert abs(anchor - 0.8756) < 0.01, "anchor drifted; the comparison is not valid"


def selftest():
    """Check the streamed Gram ridge against sklearn on a small dense problem."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 12))
    Y = rng.normal(size=(200, 3))
    for a in (1.0, 50.0):
        mine = ridge(X.T @ X, X.T @ Y, a)
        ref = Ridge(alpha=a, fit_intercept=False).fit(X, Y).coef_.T
        assert np.allclose(mine, ref, atol=1e-8), np.abs(mine - ref).max()

    # Weighted stacking: Gram of the stack must equal the weighted sum of Grams.
    Xs = rng.normal(size=(20, 12))
    Ys = rng.normal(size=(20, 3))
    w = 7.0
    G = X.T @ X + w * (Xs.T @ Xs)
    B = X.T @ Y + w * (Xs.T @ Ys)
    stack = np.vstack([X, Xs])
    sw = np.r_[np.ones(len(X)), np.full(len(Xs), w)]
    ref = Ridge(alpha=3.0, fit_intercept=False).fit(stack, np.vstack([Y, Ys]),
                                                    sample_weight=sw).coef_.T
    assert np.allclose(ridge(G, B, 3.0), ref, atol=1e-8)

    # np.add.at one-hot accumulation must equal an explicit one-hot cross-product.
    lab = rng.integers(0, 3, 200)
    Bo = np.zeros((12, 3))
    np.add.at(Bo, (slice(None), lab), X.T)
    assert np.allclose(Bo, X.T @ np.eye(3)[lab])
    print("rung1c self-checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--logistic" in sys.argv:
        logistic_run()
    else:
        main()
