"""Rung 1: window features + LightGBM, scored on the Rung 0 harness.

16 features on purpose. The ladder's next rung is selection, not more features:
1000 features against 15 earthquakes is how this competition ate 4,516 teams.

    python3 gbm.py             # build features (cached), run LOEO, compare to constant
    python3 gbm.py --submit    # refit on everything, write submission.csv
"""
import sys
import numpy as np
import lightgbm as lgb

from baseline import DATA, WINDOW, FULL_CYCLES, windows, loeo, constant

NAMES = [
    "std", "abs_mean", "abs_q95", "abs_q99", "abs_q999", "abs_max",
    "c1k_std_mean", "c1k_std_std", "c1k_std_q95", "c1k_std_max",
    "c100_std_mean", "c100_std_q95",
    "frac_gt20", "frac_gt50", "frac_gt100",
    "c1k_std_slope",
    "bvalue",   # appended last so saved column indices stay valid
] + [f"band{i}" for i in range(8)] + ["spec_centroid", "spec_rolloff95"]

# Log-spaced band edges as a fraction of Nyquist (1.87 MHz at 3.74 MHz sampling).
# Every feature above is a time-domain amplitude statistic, so they all answer
# "how big". These answer "how fast", which is information none of them carry.
BAND_EDGES = np.geomspace(1e-3, 1.0, 9)


def feats(x):
    """17 features from one 150,000-sample window.

    Rung 0 showed the error tracks cycle length, so what matters is how far into
    the stick-slip cycle we are. That shows up as the rate of low-level acoustic
    emission, which is why the spread statistics carry the signal and the mean
    and max do not.
    """
    a = np.abs(x)
    c1k = x[:150_000].reshape(150, 1000).std(axis=1)
    c100 = x[:150_000].reshape(1500, 100).std(axis=1)
    q95, q99, q999 = np.quantile(a, [0.95, 0.99, 0.999])
    return [
        x.std(), a.mean(), q95, q99, q999, a.max(),
        c1k.mean(), c1k.std(), np.quantile(c1k, 0.95), c1k.max(),
        c100.mean(), np.quantile(c100, 0.95),
        (a > 20).mean(), (a > 50).mean(), (a > 100).mean(),
        np.polyfit(np.arange(150), c1k, 1)[0],
        # Slope of the amplitude-frequency curve: how fast event count falls off
        # as the threshold rises. Drops from ~4.4 to ~2.8 approaching failure, the
        # b-value decrease seismology expects. NuSVR was reconstructing this from
        # the three frac_gt features; give it to the trees directly.
        (np.log10((a > 20).mean() + 1e-9) - np.log10((a > 100).mean() + 1e-9)) / np.log10(5.0),
    ] + spectral(x)


def spectral(x):
    """8 log-spaced band energies plus centroid and 95% rolloff.

    Micro-fracturing in a finer, more damaged gouge radiates at higher frequency,
    so the shape of the spectrum should move through the cycle even when the
    total energy does not.
    """
    p = np.abs(np.fft.rfft(x)) ** 2
    p = p[1:]                                  # drop DC
    tot = p.sum() + 1e-30
    f = np.linspace(0, 1, len(p))
    idx = np.searchsorted(f, BAND_EDGES)
    bands = [np.log10(p[idx[i]:idx[i + 1]].sum() / tot + 1e-12) for i in range(8)]
    cdf = np.cumsum(p) / tot
    return bands + [float((f * p).sum() / tot), float(f[np.searchsorted(cdf, 0.95)])]


def build(ends, cache=None):
    """Cached because every window is 150k samples of numpy work."""
    import os
    cache = cache or f"{DATA}/feats_{len(ends)}.npy"
    if os.path.exists(cache):
        X = np.load(cache)
        if X.shape == (len(ends), len(NAMES)):
            return X
    ac = np.load(f"{DATA}/acoustic.npy", mmap_mode="r")
    X = np.array([feats(np.asarray(ac[e - WINDOW:e], dtype=np.float64)) for e in ends])
    np.save(cache, X)
    return X


PARAMS = dict(
    objective="mae",          # metric is MAE, so optimise MAE, not squared error
    n_estimators=400, learning_rate=0.03,
    num_leaves=8, min_child_samples=80,   # 15 effective samples -> keep trees tiny
    colsample_bytree=0.7, subsample=0.7, subsample_freq=1,
    reg_lambda=1.0, verbose=-1, n_jobs=-1,
)


def gbm(X_tr, y_tr, X_va, _n):
    return lgb.LGBMRegressor(**PARAMS).fit(X_tr, y_tr).predict(X_va)


def report(name, maes):
    print(f"\n{name}")
    for c, m in zip(FULL_CYCLES, maes):
        print(f"  cycle {c:2d}: {m:.4f}")
    print(f"  mean {maes.mean():.4f}   std {maes.std():.4f}   max {maes.max():.4f}")
    return maes.mean()


def submit(path=f"{DATA}/../submission.csv"):
    import glob, pandas as pd
    ends, y, _ = windows()
    model = lgb.LGBMRegressor(**PARAMS).fit(build(ends), y)
    rows = []
    for f in sorted(glob.glob(f"{DATA}/test/*.csv")):
        x = pd.read_csv(f, dtype=np.int16).acoustic_data.values.astype(np.float64)
        rows.append((f.rsplit("/", 1)[1][:-4], feats(x)))
    pred = np.clip(model.predict(np.array([r[1] for r in rows])), 0, 16.2)
    pd.DataFrame({"seg_id": [r[0] for r in rows], "time_to_failure": pred}).to_csv(path, index=False)
    print(f"wrote {path}: {len(rows)} rows, pred {pred.min():.3f}-{pred.max():.3f} mean {pred.mean():.3f}")


def main():
    ends, y, cyc = windows()
    X = build(ends)
    print(f"{X.shape[0]:,} windows x {X.shape[1]} features")

    base = report("constant (Rung 0)", loeo(y, cyc, constant))
    got = report("lightgbm (Rung 1)", loeo(y, cyc, gbm, X))
    print(f"\nimprovement over constant: {base - got:+.4f} MAE")

    imp = lgb.LGBMRegressor(**PARAMS).fit(X, y).feature_importances_
    print("\nfeature importance (full fit):")
    for i in np.argsort(-imp):
        print(f"  {NAMES[i]:<16} {imp[i]:>5}")

    assert X.shape == (len(y), 16)
    assert np.isfinite(X).all(), "non-finite feature"
    assert got < base, f"gbm {got:.4f} no better than constant {base:.4f}"


if __name__ == "__main__":
    submit() if "--submit" in sys.argv else main()
