"""Rung 3: model class and target transform, on the 7 features from select.py.

Rung 2 showed feature selection had nothing left to give: mean moved 0.09, fold
std did not move at all. So vary what the model is, not what it eats.

    python3 models.py            # LOEO for every model, per-fold table
    python3 models.py --submit   # refit the winner, write submission.csv
"""
import sys
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

from baseline import DATA, FULL_CYCLES, windows, loeo, constant
from gbm import NAMES, PARAMS, build, feats

CLIP = (0.0, 16.2)  # ttf is non-negative and the longest cycle was 16.107s
NUSVR_CAP = 8000    # RBF SVR is O(n^2); subsample so overlapping windows stay affordable


def _sub(X, y):
    if len(y) <= NUSVR_CAP:
        return X, y
    i = np.random.default_rng(0).choice(len(y), NUSVR_CAP, replace=False)
    return X[i], y[i]


def gbm(X_tr, y_tr, X_va, _n):
    return lgb.LGBMRegressor(**PARAMS).fit(X_tr, y_tr).predict(X_va)


def nusvr(X_tr, y_tr, X_va, _n):
    m = make_pipeline(StandardScaler(), NuSVR(nu=0.7, C=1.0, gamma="scale"))
    return m.fit(*_sub(X_tr, y_tr)).predict(X_va)


def ridge(X_tr, y_tr, X_va, _n):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, y_tr).predict(X_va)


def logged(f):
    """Fit on log1p(ttf). Targets are right-skewed, so this pulls the long cycles in."""
    def g(X_tr, y_tr, X_va, n):
        return np.expm1(f(X_tr, np.log1p(y_tr), X_va, n))
    g.__name__ = f"log_{f.__name__}"
    return g


def blend(*fs):
    """Median, not mean: the metric is MAE, so a single bad member should not drag."""
    def g(X_tr, y_tr, X_va, n):
        return np.median([f(X_tr, y_tr, X_va, n) for f in fs], axis=0)
    g.__name__ = "blend_" + "_".join(f.__name__ for f in fs)
    return g


def clipped(f):
    def g(X_tr, y_tr, X_va, n):
        return np.clip(f(X_tr, y_tr, X_va, n), *CLIP)
    g.__name__ = f.__name__
    return g


MODELS = [gbm, nusvr, ridge, logged(gbm), logged(nusvr),
          blend(gbm, nusvr, ridge), blend(logged(gbm), logged(nusvr), ridge),
          # log_nusvr has the best mean, ridge the tightest spread; pair them.
          blend(logged(nusvr), ridge)]


def main(stride=150_000, featset="all"):
    ends, y, cyc = windows(stride)
    X = build(ends)
    cols = np.arange(X.shape[1]) if featset == "all" else np.load(f"{DATA}/selected_{featset}.npy")
    X = X[:, cols]
    print(f"stride {stride:,} | {X.shape[0]:,} windows x {X.shape[1]} features: "
          + ", ".join(NAMES[c] for c in cols))

    rows = [("constant", loeo(y, cyc, constant))]
    rows += [(f.__name__, loeo(y, cyc, clipped(f), X)) for f in MODELS]

    print("\ncycle " + "".join(f"{n[:11]:>12}" for n, _ in rows))
    for i, c in enumerate(FULL_CYCLES):
        print(f"{c:5d} " + "".join(f"{m[i]:>12.4f}" for _, m in rows))
    print("-" * (6 + 12 * len(rows)))
    for label, fn in (("mean", np.mean), ("std", np.std), ("max", np.max)):
        print(f"{label:>5} " + "".join(f"{fn(m):>12.4f}" for _, m in rows))

    best = min(rows[1:], key=lambda r: r[1].mean())
    print(f"\nbest by mean MAE: {best[0]}  "
          f"mean {best[1].mean():.4f}  std {best[1].std():.4f}  max {best[1].max():.4f}")
    tight = min(rows[1:], key=lambda r: r[1].std())
    print(f"lowest fold std:  {tight[0]}  "
          f"mean {tight[1].mean():.4f}  std {tight[1].std():.4f}  max {tight[1].max():.4f}")

    assert best[1].mean() < rows[0][1].mean(), "nothing beats the constant"
    return best[0]


def submit(name, stride=150_000, featset="all", path=f"{DATA}/../submission.csv"):
    import glob, pandas as pd
    f = clipped(next(m for m in MODELS if m.__name__ == name))
    ends, y, _ = windows(stride)
    Xa = build(ends)
    cols = np.arange(Xa.shape[1]) if featset == "all" else np.load(f"{DATA}/selected_{featset}.npy")
    X = Xa[:, cols]
    rows = []
    for p in sorted(glob.glob(f"{DATA}/test/*.csv")):
        x = pd.read_csv(p, dtype=np.int16).acoustic_data.values.astype(np.float64)
        rows.append((p.rsplit("/", 1)[1][:-4], feats(x)))
    Xt = np.array([r[1] for r in rows])[:, cols]
    pred = f(X, y, Xt, len(rows))
    pd.DataFrame({"seg_id": [r[0] for r in rows],
                  "time_to_failure": pred}).to_csv(path, index=False)
    print(f"wrote {path}: {name}, stride {stride:,}, {len(cols)} feats, {len(rows)} rows, "
          f"pred {pred.min():.3f}-{pred.max():.3f} mean {pred.mean():.3f}")


def _arg(flag, default):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(flag)), default)


if __name__ == "__main__":
    want = _arg("--submit=", None)
    stride, featset = int(_arg("--stride=", 150_000)), _arg("--feats=", "all")
    if want:
        submit(want, stride, featset)
    else:
        main(stride, featset)
