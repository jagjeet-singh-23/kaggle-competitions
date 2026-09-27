"""Rung 3: predict a molecular fingerprint from a spectrum.

Section 7 of RESULTS.md is the reason this exists. Class 1 is only 12-15% of the
hidden test and library search already recovers nearly all of it; classes 2 and 3 are
the rest and score 0.0676 and 0.0000. The leader's 0.425 implies roughly 0.285 coming
from those two, which is not better retrieval -- it is structure elucidation.

Analog propagation (section 5) can only reach a candidate through some *other*
structure that already has spectra. A fingerprint model has no such limit: it maps the
spectrum straight to a description of the molecule, and every candidate in the mass
window can then be scored against that prediction, whether or not anything like it was
ever measured.

    spectrum -> 2048-bit Morgan fingerprint -> rank candidates by Tanimoto

**Memory.** A dense binned matrix over 1.88M spectra would be 18.9 GB. The peaks are
kept sparse instead (the index already stores at most 64 per spectrum, ~830 MB for
everything needed) and binned into a dense array one batch at a time.

    python3 fingerprint.py --selftest
    python3 fingerprint.py --train        -> data/fpmodel.pt
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import library
from analog import DESC, FP_BITS

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = library.DATA
WORK = library.WORK
MODEL = f"{WORK}/fpmodel.pt"

BIN = 1.0            # dalton; fragment m/z resolution fed to the model
MZ_MAX = 1300.0
NBIN = int(MZ_MAX / BIN)
BIT_LO, BIT_HI = 0.005, 0.995   # drop bits that are almost always on or off

torch.manual_seed(0)


def featurise(mz, it, precursor):
    """(batch, 2 * NBIN + 2) from padded peak arrays.

    Two views of every peak: its m/z, and its neutral loss from the precursor. The
    loss axis is what makes a fragment interpretable across molecules of different
    mass -- losing water is the same event at m/z 300 and at 900, and a model given
    only absolute m/z has to learn that separately for every precursor.
    """
    n = len(mz)
    X = np.zeros((n, 2 * NBIN + 2), np.float32)
    rows = np.repeat(np.arange(n), mz.shape[1])
    m = np.asarray(mz, np.float32).ravel()
    v = np.asarray(it, np.float32).ravel()
    keep = (m > 0) & (v > 0)
    b = (m / BIN).astype(np.int32)
    ok = keep & (b >= 0) & (b < NBIN)
    np.add.at(X, (rows[ok], b[ok]), v[ok])

    loss = np.repeat(np.asarray(precursor, np.float32), mz.shape[1]) - m
    lb = (loss / BIN).astype(np.int32)
    ok = keep & (lb >= 0) & (lb < NBIN)
    np.add.at(X, (rows[ok], NBIN + lb[ok]), v[ok])

    X[:, -2] = np.asarray(precursor, np.float32) / MZ_MAX
    X[:, -1] = (np.asarray(mz, np.float32) > 0).sum(1) / mz.shape[1]
    return X


class Net(nn.Module):
    def __init__(self, d_in, d_out, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def structure_fp():
    """(key -> row) and the unpacked bit matrix for every structure in the index."""
    with np.load(DESC) as d:
        return {k: i for i, k in enumerate(d["key"])}, d["fp"]


def unpack(fp_packed, rows, bits):
    return np.unpackbits(fp_packed[rows], axis=1)[:, bits].astype(np.float32)


def load_training(drop=None):
    """Peaks, precursors and the structure row for every spectrum, minus `drop`."""
    ix = library.load(cols=("mz", "it", "npk", "precursor", "inchikey14"))
    row_of, fp = structure_fp()
    rows = np.array([row_of.get(k, -1) for k in ix["inchikey14"]], np.int32)
    keep = rows >= 0
    if drop is not None and len(drop):
        m = np.ones(len(rows), bool)
        m[drop] = False
        keep &= m
    return (ix["mz"][keep], ix["it"][keep], ix["precursor"][keep], rows[keep], fp)


def choose_bits(fp, rows):
    """Bits that are neither almost always on nor almost always off.

    A bit present in 99.9% of structures carries no information and dominates the
    loss; one present in 0.1% cannot be learned from this much data either.
    """
    f = np.unpackbits(fp[np.unique(rows)], axis=1).mean(0)
    return np.where((f > BIT_LO) & (f < BIT_HI))[0]


def train(epochs=6, batch=256, lr=1e-3, limit=None, quiet=False, drop=None):
    """drop: index rows to exclude. For any measurement on data/splits.npz this MUST
    be splits['drop'], or the model trains on the very spectra it is scored against
    and the MRR is meaningless."""
    mz, it, pm, rows, fp = load_training(drop=drop)
    if limit:
        sel = np.random.default_rng(0).choice(len(mz), limit, replace=False)
        mz, it, pm, rows = mz[sel], it[sel], pm[sel], rows[sel]
    bits = choose_bits(fp, rows)
    if not quiet:
        print(f"{len(mz):,} spectra, {len(np.unique(rows)):,} structures, "
              f"{len(bits)} of {FP_BITS} bits kept", flush=True)

    # Held out by structure, not by spectrum: the same molecule measured four times
    # must not sit on both sides.
    uniq = np.unique(rows)
    rng = np.random.default_rng(0)
    val_struct = set(rng.choice(uniq, max(1, len(uniq) // 20), replace=False).tolist())
    is_val = np.array([r in val_struct for r in rows])
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    if not quiet:
        print(f"train {len(tr):,} spectra / val {len(va):,} "
              f"({len(val_struct):,} held-out structures)", flush=True)

    model = Net(2 * NBIN + 2, len(bits))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            X = torch.tensor(featurise(mz[b], it[b], pm[b]))
            Y = torch.tensor(unpack(fp, rows[b], bits))
            opt.zero_grad()
            l = lossf(model(X), Y)
            l.backward()
            opt.step()
            tot += l.item() * len(b)
        if not quiet:
            print(f"  epoch {ep + 1}/{epochs}  loss {tot / len(order):.4f}  "
                  f"tanimoto {'/'.join(f'{x:.4f}' for x in evaluate(model, mz, it, pm, rows, fp, bits, va))}"
                  f" (model/shuffled/mean-fp)  {(time.time() - t0) / 60:.1f} min", flush=True)
    torch.save({"state": model.state_dict(), "bits": bits}, MODEL)
    if not quiet:
        print(f"wrote {MODEL}")
    return model, bits


def predict(model, mz, it, pm, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(mz), batch):
            X = torch.tensor(featurise(mz[i:i + batch], it[i:i + batch], pm[i:i + batch]))
            out.append(torch.sigmoid(model(X)).numpy())
    return np.concatenate(out)


def tanimoto(P, T):
    inter = (P & T).sum(1)
    union = (P | T).sum(1)
    return float(np.mean(inter / np.maximum(union, 1)))


def evaluate(model, mz, it, pm, rows, fp, bits, idx, n=2000):
    """Mean Tanimoto on held-out structures, against two no-information baselines.

    This metric flatters a model that has learned nothing. Fingerprint bits are
    correlated and many are common, so emitting the *average* fingerprint for every
    spectrum already scores well. Two controls make the number mean something:

      shuffled  the model's own predictions paired with the wrong targets -- what
                the score would be if the spectrum carried no information
      mean-fp   the training-set bit frequencies thresholded, i.e. the best constant
                prediction available

    Only the gap over those says the model reads the spectrum.
    """
    idx = idx[:n]
    P = predict(model, mz[idx], it[idx], pm[idx]) > 0.5
    T = unpack(fp, rows[idx], bits) > 0.5
    perm = np.random.default_rng(0).permutation(len(T))
    const = np.repeat(T.mean(0, keepdims=True) > 0.5, len(T), axis=0)
    return tanimoto(P, T), tanimoto(P[perm], T), tanimoto(const, T)


def selftest():
    """Featurisation must place peaks and losses in the right bins, and the model
    must be able to learn a fingerprint that is a deterministic function of peaks."""
    mz = np.array([[100.0, 300.0, 0.0]], np.float32)
    it = np.array([[1.0, 2.0, 0.0]], np.float32)
    X = featurise(mz, it, np.array([500.0], np.float32))
    assert X.shape == (1, 2 * NBIN + 2)
    assert X[0, 100] == 1.0 and X[0, 300] == 2.0, "m/z bins"
    assert X[0, NBIN + 400] == 1.0, "neutral loss 500-100"
    assert X[0, NBIN + 200] == 2.0, "neutral loss 500-300"
    assert X[0, 0] == 0.0, "padding must not land in bin 0"
    assert abs(X[0, -2] - 500.0 / MZ_MAX) < 1e-6

    # Two peaks in the same bin accumulate rather than overwrite.
    X2 = featurise(np.array([[100.2, 100.4]], np.float32),
                   np.array([[1.0, 1.0]], np.float32), np.array([500.0], np.float32))
    assert X2[0, 100] == 2.0, X2[0, 100]

    # A tiny learnable problem: the bit is on iff a peak sits in bin 200.
    rng = np.random.default_rng(0)
    n = 512
    mz = np.zeros((n, 4), np.float32)
    it = np.ones((n, 4), np.float32)
    has = rng.random(n) < 0.5
    mz[:, 0] = np.where(has, 200.0, 400.0)
    mz[:, 1] = rng.uniform(50, 150, n)
    pm = np.full(n, 600.0, np.float32)
    X = torch.tensor(featurise(mz, it, pm))
    Y = torch.tensor(has.astype(np.float32)).unsqueeze(1)
    m = Net(2 * NBIN + 2, 1, hidden=32)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(150):
        opt.zero_grad()
        lossf(m(X), Y).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        acc = ((torch.sigmoid(m(X)) > 0.5).float() == Y).float().mean()
    assert acc > 0.95, f"must learn a peak-determined bit, got {acc:.2f}"
    print("fingerprint self-checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--train" in sys.argv:
        lim = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), None)
        ep = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--epochs=")), 6)
        drop = None
        if "--holdout" in sys.argv:
            drop = np.load(f"{WORK}/splits.npz", allow_pickle=True)["drop"]
            print(f"excluding {len(drop):,} held-out spectra from training")
        train(epochs=ep, limit=lim, drop=drop)
    else:
        print(__doc__)
