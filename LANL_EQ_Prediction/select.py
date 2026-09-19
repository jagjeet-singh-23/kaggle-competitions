"""Rung 2: cut features until the model stops improving.

Greedy backward elimination on the same leave-one-earthquake-out CV. This is the
rung that decided the original competition: 16 features against 15 earthquakes is
already generous, and the teams that won shipped fewer.

The subset is model-specific. The features that suit a tree ensemble are not the
ones that suit an RBF kernel, so select against the model you intend to ship.

    python3 select.py                       # for gbm (default)
    python3 select.py --model=log_nusvr     # for whichever model you will submit
    python3 select.py --model=X --submit    # refit the chosen subset, write submission.csv

Caveat worth keeping in mind: the subset is chosen on the same CV it is then
scored on, so the reported gain is optimistic. The trace is printed in full so you
can judge whether the curve is flat (gain is noise) or has a real knee.
"""
import sys
import numpy as np

from baseline import DATA, FULL_CYCLES, windows, loeo, constant
from gbm import NAMES, build
from models import MODELS, clipped

MIN_K = 4


def pick(name):
    return next(m for m in MODELS if m.__name__ == name)


def cv(X, y, cyc, cols, fp):
    return loeo(y, cyc, clipped(fp), X[:, cols])


def backward(X, y, cyc, fp):
    """Drop the one feature whose removal helps most, until MIN_K remain."""
    cols = list(range(X.shape[1]))
    m = cv(X, y, cyc, cols, fp)
    trace = [(list(cols), m.mean(), m.std(), None)]
    while len(cols) > MIN_K:
        scored = []
        for c in cols:
            rest = [k for k in cols if k != c]
            mm = cv(X, y, cyc, rest, fp)
            scored.append((mm.mean(), mm.std(), c, rest))
        best = min(scored)
        cols = best[3]
        trace.append((list(cols), best[0], best[1], best[2]))
        print(f"  drop {NAMES[best[2]]:<16} -> {len(cols):2d} feats  "
              f"mean {best[0]:.4f}  std {best[1]:.4f}")
    return trace


def path(name):
    return f"{DATA}/selected_{name}.npy"


def main(name):
    ends, y, cyc = windows()
    X = build(ends)
    fp = pick(name)
    base = loeo(y, cyc, constant)
    print(f"model: {name}")
    print(f"constant: mean {base.mean():.4f}  std {base.std():.4f}")
    print(f"\nbackward elimination from {X.shape[1]} features:")
    trace = backward(X, y, cyc, fp)

    print("\n k  mean     std    dropped")
    for cols, m, s, d in trace:
        print(f"{len(cols):2d}  {m:.4f}  {s:.4f}  {'-' if d is None else NAMES[d]}")

    best = min(trace, key=lambda t: t[1])
    cols, m, s, _ = best
    print(f"\nbest by mean MAE: {len(cols)} features  mean {m:.4f}  std {s:.4f}")
    print("  " + ", ".join(NAMES[c] for c in sorted(cols)))

    tight = min(trace, key=lambda t: t[1] + t[2])
    print(f"\nbest by mean+std: {len(tight[0])} features  "
          f"mean {tight[1]:.4f}  std {tight[2]:.4f}")
    print("  " + ", ".join(NAMES[c] for c in sorted(tight[0])))

    np.save(path(name), np.array(sorted(cols)))
    print(f"\nsaved {path(name)}")

    maes = cv(X, y, cyc, cols, fp)
    print("\nchosen subset, per fold:")
    for c, v in zip(FULL_CYCLES, maes):
        print(f"  cycle {c:2d}: {v:.4f}")
    print(f"  mean {maes.mean():.4f}  std {maes.std():.4f}  max {maes.max():.4f}")

    assert len(cols) >= MIN_K
    assert m <= trace[0][1] + 1e-9, "elimination picked a subset worse than the full set"
    assert maes.mean() < base.mean(), "selected model no better than constant"


def submit(name, out=f"{DATA}/../submission.csv"):
    import glob, pandas as pd
    from gbm import feats
    fp, cols = clipped(pick(name)), np.load(path(name))
    ends, y, _ = windows()
    X = build(ends)[:, cols]
    rows = []
    for p in sorted(glob.glob(f"{DATA}/test/*.csv")):
        x = pd.read_csv(p, dtype=np.int16).acoustic_data.values.astype(np.float64)
        rows.append((p.rsplit("/", 1)[1][:-4], feats(x)))
    Xt = np.array([r[1] for r in rows])[:, cols]
    pred = fp(X, y, Xt, len(rows))
    pd.DataFrame({"seg_id": [r[0] for r in rows],
                  "time_to_failure": pred}).to_csv(out, index=False)
    print(f"wrote {out}: {name}, {len(cols)} feats, {len(rows)} rows, "
          f"pred {pred.min():.3f}-{pred.max():.3f} mean {pred.mean():.3f}")


if __name__ == "__main__":
    model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "gbm")
    if "--submit" in sys.argv:
        submit(model)
    else:
        main(model)
