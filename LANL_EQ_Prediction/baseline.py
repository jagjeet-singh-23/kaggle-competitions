"""Rung 0: constant baseline + leave-one-earthquake-out CV harness.

Everything downstream plugs into `loeo`: pass a fit_predict(X_tr, y_tr, X_va) and
it returns per-fold MAE. The constant predictor is the floor to beat.

    python3 baseline.py            # run the CV + self-checks
    python3 baseline.py --submit   # write submission.csv (needs data/test/)
"""
import sys
import numpy as np

DATA = __file__.rsplit("/", 1)[0] + "/data"
WINDOW = 150_000  # test segments are exactly this long

# Row index of each earthquake reset, from the ttf sawtooth. See GUIDE.md §2b.
# 17 cycles; 0 and 16 are partial (16 never reaches failure) so they are not folds.
BOUNDS = np.array([
    0, 5656574, 50085878, 104677356, 138772453, 187641820, 218652630,
    245829585, 307838917, 338276287, 375377848, 419368880, 461811623,
    495800225, 528777115, 585568144, 621985673, 629145480,
])
FULL_CYCLES = np.arange(1, 16)  # the 15 usable folds


def windows(stride=WINDOW):
    """Window end offsets and targets, dropping windows that span an earthquake.

    A window's target is the ttf of its last row, matching how test segments are
    scored. One that straddles a reset contains the failure itself, which no test
    segment does, so it is dropped rather than fed a meaningless target.
    """
    ttf = np.load(f"{DATA}/ttf.npy", mmap_mode="r")
    ends = np.arange(WINDOW, len(ttf) + 1, stride)
    starts = ends - WINDOW
    cyc = np.searchsorted(BOUNDS, starts, side="right") - 1
    keep = cyc == np.searchsorted(BOUNDS, ends - 1, side="right") - 1
    ends, cyc = ends[keep], cyc[keep]
    return ends, np.asarray(ttf[ends - 1], dtype=np.float64), cyc


def loeo(y, cyc, fit_predict, X=None):
    """Leave-one-earthquake-out CV. Returns per-fold MAE, one entry per full cycle.

    Random KFold leaks here: neighbouring windows share rows and sit on the same
    sawtooth, so a shuffled split scores far better than the private LB ever will.
    """
    maes = []
    for c in FULL_CYCLES:
        va = cyc == c
        if not va.any():
            continue
        tr = ~va
        Xtr = None if X is None else X[tr]
        Xva = None if X is None else X[va]
        maes.append(np.abs(fit_predict(Xtr, y[tr], Xva, va.sum()) - y[va]).mean())
    return np.array(maes)


def constant(_Xtr, y_tr, _Xva, n_va):
    """Median minimises MAE, so this is the best possible constant."""
    return np.full(n_va, np.median(y_tr))


def submit(path=f"{DATA}/../submission.csv"):
    import pandas as pd
    _, y, _ = windows()
    pred = np.median(y)
    sub = pd.read_csv(f"{DATA}/sample_submission.csv")
    sub["time_to_failure"] = pred
    sub.to_csv(path, index=False)
    print(f"wrote {path}: {len(sub)} rows, constant {pred:.4f}")


def main():
    ends, y, cyc = windows()
    print(f"windows {len(y):,}  targets {y.min():.3f}–{y.max():.3f}")
    print(f"dropped (span an earthquake): {len(np.arange(WINDOW, 629145480 + 1, WINDOW)) - len(y)}")

    maes = loeo(y, cyc, constant)
    print(f"\nLOEO MAE per fold ({len(maes)} folds):")
    for c, m in zip(FULL_CYCLES, maes):
        print(f"  cycle {c:2d}: {m:.4f}")
    print(f"\nmean {maes.mean():.4f}   std {maes.std():.4f}   "
          f"min {maes.min():.4f}   max {maes.max():.4f}")

    assert len(maes) == 15, f"expected 15 folds, got {len(maes)}"
    starts = ends - WINDOW
    assert (np.searchsorted(BOUNDS, starts, "right")
            == np.searchsorted(BOUNDS, ends - 1, "right")).all(), "window spans a reset"
    assert 2.0 < maes.mean() < 4.0, f"constant MAE {maes.mean():.3f} outside sane range"
    # Spread matters more than the mean: a model is only trustworthy if it holds
    # across cycles, and cycle length varies 7.1s-16.1s.
    assert maes.std() > 0.3, "fold spread suspiciously low, check the split"
    print("\nself-checks passed")


if __name__ == "__main__":
    submit() if "--submit" in sys.argv else main()
