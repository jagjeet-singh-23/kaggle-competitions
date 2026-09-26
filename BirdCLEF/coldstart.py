"""Rung 2: give the 30 classes that currently emit a 0.5 constant an actual prediction.

234 classes reach a prediction through one of two routes -- a BirdNET output column, or
a labelled soundscape positive to fit a head on. 30 classes have neither and get 0.5.
They are 13% of a macro-averaged metric, and section 7 of RESULTS.md shows they
account for essentially the whole 0.8881 -> 0.83575 CV-to-leaderboard gap.

All 30 have focal training audio, already extracted to data/emb_train/. The question
is whether a head fitted on focal audio alone beats a coin flip on soundscape.

That cannot be measured on the 30 directly -- having no labelled positive is what puts
them here. So it is measured on a proxy: the 19 classes that are also absent from
BirdNET and also have focal audio, but *do* have labelled positives. Same shape, and
scoreable. Their labels are used only to score, never to fit.

This is the question Rung 1c did not ask. There, focal audio competed with soundscape
data and lost. Here there is no soundscape data to compete with and the incumbent is
0.5, so the bar is far lower.

    python3 coldstart.py
    python3 coldstart.py --selftest
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from birdnet import species_map
from rung1c import focal_subsample

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")

CAPS = (30, 150, 400)     # focal segments per class


def cohorts():
    """Split the 234 classes by which prediction route is available to them."""
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    tr = pd.read_csv(f"{DATA}/train.csv", dtype={"primary_label": str})
    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    Y = d["Y"]
    mapped = species_map(tax) >= 0
    labelled = Y.sum(0) > 0
    focal = tax.primary_label.isin(set(tr.primary_label)).values
    return dict(
        tax=tax, Y=Y, emb=d["emb"].astype(np.float32),
        # The classes we actually want to fix: no column, no labels, but focal audio.
        target=np.where(~mapped & ~labelled & focal)[0],
        # Same shape but scoreable, so they stand in for the target cohort.
        proxy=np.where(~mapped & labelled & focal)[0],
    )


def main():
    c = cohorts()
    Y, emb = c["Y"], c["emb"]
    target, proxy = c["target"], c["proxy"]
    print(f"target cohort (currently 0.5 constant): {len(target)} classes")
    print(f"proxy  cohort (same shape, scoreable):  {len(proxy)} classes")
    print(f"  proxy positives per class: min {int(Y[:, proxy].sum(0).min())}, "
          f"median {int(np.median(Y[:, proxy].sum(0)))}, "
          f"max {int(Y[:, proxy].sum(0).max())}\n")

    print(f"{'focal/class':<14}{'rows':>9}{'mean AUC':>10}{'vs 0.5':>9}"
          f"{'>0.5':>7}{'>0.7':>7}")
    best = (None, -1)
    for cap in CAPS:
        Xf, yf = focal_subsample(cap, Y.shape[1])
        sc = StandardScaler().fit(Xf)
        Xf_s, Xs_s = sc.transform(Xf), sc.transform(emb)

        aucs = []
        for j in proxy:
            pos = yf == j
            if pos.sum() == 0:
                aucs.append(0.5)          # no focal rows after capping; stays a constant
                continue
            m = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")
            p = m.fit(Xf_s, pos.astype(int)).predict_proba(Xs_s)[:, 1]
            aucs.append(roc_auc_score(Y[:, j], p))
        a = np.array(aucs)
        print(f"{cap:<14}{len(Xf):>9,}{a.mean():>10.4f}{a.mean() - 0.5:>+9.4f}"
              f"{int((a > 0.5).sum()):>7}{int((a > 0.7).sum()):>7}", flush=True)
        if a.mean() > best[1]:
            best = ((cap, a), a.mean())

    cap, a = best[0]
    print(f"\nbest: {cap} focal segments per class, mean AUC {a.mean():.4f} "
          f"on {len(proxy)} proxy classes")

    # What that buys on the real metric, if the 30 behave like the proxy 19.
    rest = 0.885
    for v in (0.5, a.mean()):
        macro = ((234 - len(target)) * rest + len(target) * v) / 234
        print(f"  30 target classes at {v:.4f} -> macro {macro:.4f}")
    gain = len(target) * (a.mean() - 0.5) / 234
    print(f"  expected leaderboard gain: {gain:+.4f}  (0.83575 -> {0.83575 + gain:.4f})")

    assert a.mean() > 0.5, "focal-only heads must beat the constant they replace"
    np.savez(f"{DATA}/coldstart.npz", proxy=proxy, target=target, auc=a, cap=cap)
    zero_shot_cohort(c, cap)


def zero_shot_cohort(c, cap):
    """The other cohort worth re-examining: 129 classes that fall back to zero-shot.

    They are mapped into BirdNET but have no labelled positive, so they currently take
    BirdNET's logits untouched. A focal-only head is the alternative. Measured on the
    28 classes that are mapped *and* labelled, which is the only scoreable stand-in.
    """
    from cv import zero_shot
    tax, Y, emb = c["tax"], c["Y"], c["emb"]
    d = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    Z, hit = zero_shot(d["logits"], tax)
    both = np.where(hit & (Y.sum(0) > 0))[0]

    Xf, yf = focal_subsample(cap, Y.shape[1])
    sc = StandardScaler().fit(Xf)
    Xf_s, Xs_s = sc.transform(Xf), sc.transform(emb)
    rank = lambda v: np.argsort(np.argsort(v)) / max(1, len(v) - 1)

    zs, fo, bl = [], [], []
    for j in both:
        pos = yf == j
        z = roc_auc_score(Y[:, j], Z[:, j])
        if pos.sum() == 0:
            zs.append(z); fo.append(0.5); bl.append(z)
            continue
        m = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")
        p = m.fit(Xf_s, pos.astype(int)).predict_proba(Xs_s)[:, 1]
        zs.append(z)
        fo.append(roc_auc_score(Y[:, j], p))
        bl.append(roc_auc_score(Y[:, j], 0.5 * rank(p) + 0.5 * rank(Z[:, j])))
    zs, fo, bl = map(np.array, (zs, fo, bl))
    print(f"\n129 zero-shot-only classes: which source is better?")
    print(f"  measured on the {len(both)} mapped+labelled classes")
    print(f"    zero-shot only   {zs.mean():.4f}")
    print(f"    focal-only head  {fo.mean():.4f}")
    print(f"    50/50 blend      {bl.mean():.4f}")
    win = "blend" if bl.mean() > max(zs.mean(), fo.mean()) else (
        "focal" if fo.mean() > zs.mean() else "zero-shot")
    print(f"  -> best for that cohort: {win}")


def selftest():
    """The two cohorts must be disjoint, non-empty, and defined as claimed."""
    c = cohorts()
    tax, Y = c["tax"], c["Y"]
    target, proxy = c["target"], c["proxy"]
    assert len(set(target) & set(proxy)) == 0, "cohorts must be disjoint"
    assert len(target) and len(proxy)
    mapped = species_map(tax) >= 0
    # Target: unmapped and unlabelled -- the two routes that produce a real prediction.
    assert not mapped[target].any(), "target classes must be absent from BirdNET"
    assert Y[:, target].sum() == 0, "target classes must have no labelled positive"
    # Proxy: unmapped like the target, but scoreable.
    assert not mapped[proxy].any(), "proxy must also be absent from BirdNET"
    assert (Y[:, proxy].sum(0) > 0).all(), "every proxy class needs a positive to score"
    print(f"cohorts: {len(target)} target, {len(proxy)} proxy, disjoint")
    print("coldstart self-checks passed")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
