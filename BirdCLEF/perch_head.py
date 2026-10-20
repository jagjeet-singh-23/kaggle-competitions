"""Rung 3: the kernel's pipeline with Perch v2 in place of BirdNET.

Section 12 of RESULTS.md concluded the frozen representation is the ceiling. That was
measured on one representation. The CLEF 2026 working notes report the same ridge
probe at 0.798 private on BirdNET and 0.862 on Perch v2, so the encoder is the
untested lever. Nothing else changes here: the same 739 segments, the same per-class
logistic head, the same zero-shot blend restricted to mapped columns, the same
smoothing as the kernel.

The folds are not GroupKFold's default. 59 of the 66 files have exactly 12 segments,
so its size-ordered assignment is decided by how an unstable argsort breaks ties, and
that differs between x86 and ARM. Across 20 equally valid tie-breaks the BirdNET head
ranges 0.864-0.889 (sd 0.0064), which is larger than most effects in RESULTS.md. Both
encoders are therefore scored on the same 10 seeded shuffles and compared pairwise.

    python3 perch_head.py       -> data/labelled_perch.npz on first run, then CV
"""
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupKFold

import birdnet
import perch
from cv import macro_auc
from embed_labelled import segments
from head import fit_predict, rank
from smooth import neighbour_stat

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
OUT = f"{DATA}/labelled_perch.npz"

SMOOTH_K, SMOOTH_ALPHA = 6, 0.2        # kernel v4
SEEDS = range(10)


def embed():
    """Perch logits and embeddings for the labelled segments, in segments() order."""
    lab = segments()
    net = perch.Perch()
    t0 = time.time()
    L, E, files, starts = [], [], [], []
    for fn, grp in lab.groupby("filename", sort=False):
        y, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32")
        assert sr == perch.SR_IN, sr
        s = grp.start_s.tolist()
        lo, em = net.run(perch.to_windows(y, s))
        L.append(lo)
        E.append(em)
        files += [fn] * len(s)
        starts += s
    L, E = np.concatenate(L), np.concatenate(E)
    np.savez_compressed(OUT, logits=L, emb=E, filename=np.array(files),
                        start_s=np.array(starts))
    print(f"embedded {len(E)} segments in {time.time() - t0:.0f}s -> {OUT}")


def smooth(P, files, starts):
    """The kernel's smoother, with tie-aware ranks: rank, then blend with the +/-k
    window max of the same file."""
    Pr = rank(P)
    return SMOOTH_ALPHA * Pr + (1 - SMOOTH_ALPHA) * rank(
        neighbour_stat(Pr, files, starts, SMOOTH_K, "max"))


def cross_val(X, Y, files, seed):
    P = np.zeros_like(Y, dtype=float)
    for tr, va in GroupKFold(5, shuffle=True, random_state=seed).split(X, groups=files):
        P[va] = fit_predict(X[tr], Y[tr], X[va])
    return P


def pipeline(emb, logits, idx, Y, files, starts, seed):
    """The four stages of the submitted kernel, scored separately."""
    hit = idx >= 0
    Z = np.full((len(logits), len(idx)), 0.5, np.float32)
    Z[:, hit] = logits[:, idx[hit]]
    H = cross_val(np.hstack([emb, logits[:, idx[hit]]]).astype(np.float32), Y, files, seed)
    B = rank(H).copy()
    B[:, hit] = 0.5 * B[:, hit] + 0.5 * rank(Z)[:, hit]
    return {"zero-shot": Z, "head on emb+logits": H,
            "+ zero-shot blend (mapped)": B, "+ smoothing (kernel v4)": smooth(B, files, starts)}


def main():
    if not os.path.exists(OUT):
        embed()
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    bn = np.load(f"{DATA}/labelled.npz", allow_pickle=True)
    pc = np.load(OUT, allow_pickle=True)
    Y, files, starts = bn["Y"].astype(np.float32), bn["filename"], bn["start_s"]
    assert (pc["filename"] == files).all() and (pc["start_s"] == starts).all(), \
        "the two caches must hold the same rows in the same order"

    idx_b, idx_p = birdnet.species_map(tax), perch.species_map(tax)
    enc = {"BirdNET": (bn["emb"].astype(np.float32), bn["logits"], idx_b),
           "Perch v2": (pc["emb"], pc["logits"], idx_p)}

    cols = np.where((Y.sum(0) > 0) & (Y.sum(0) < len(Y)))[0]
    birds = (idx_b >= 0)[cols]           # the 28 / 47 split every earlier table used
    print(f"{len(Y)} segments, {len(cols)} scored classes "
          f"({birds.sum()} BirdNET-mapped birds, {(~birds).sum()} others; "
          f"{(idx_p >= 0)[cols].sum()} Perch-mapped), {len(SEEDS)} fold seeds\n")

    # auc[encoder][stage] -> (seeds, classes)
    auc = {e: {} for e in enc}
    for seed in SEEDS:
        for e, (emb, logits, idx) in enc.items():
            for name, P in pipeline(emb, logits, idx, Y, files, starts, seed).items():
                auc[e].setdefault(name, []).append(macro_auc(Y, P, per_class=True)[0])
        print(f"\r  seed {seed + 1}/{len(SEEDS)}", end="", flush=True)
    print("\n")

    print(f"{'encoder':<10}{'stage':<30}{'all':>8}{'(sd)':>8}{'birds':>8}{'others':>8}")
    for e in enc:
        for name, A in auc[e].items():
            A = np.array(A)
            print(f"{e:<10}{name:<30}{A.mean():>8.4f}{A.mean(1).std():>8.4f}"
                  f"{A[:, birds].mean():>8.4f}{A[:, ~birds].mean():>8.4f}")
        print()

    # Means hide spread (RESULTS section 9), so report paired differences: per seed
    # (does Perch win on every split?) and per class (on how many species?).
    print("Perch - BirdNET, paired on identical folds:")
    for name in auc["BirdNET"]:
        D = np.array(auc["Perch v2"][name]) - np.array(auc["BirdNET"][name])
        per_seed, per_class = D.mean(1), D.mean(0)
        sem = per_class.std(ddof=1) / np.sqrt(len(per_class))
        print(f"  {name:<30}{per_seed.mean():+.4f}  wins {(per_seed > 0).sum()}/{len(D)} seeds,"
              f" {(per_class > 0).sum()}/{len(per_class)} classes,"
              f" {per_seed.mean() / sem:.1f} sigma over classes"
              f"  [birds {D[:, birds].mean():+.4f}, others {D[:, ~birds].mean():+.4f}]")

    # The leaderboard scores all 234 columns; local CV only sees the 75 above. This is
    # how the 234 split between prediction routes under each encoder.
    has_pos = Y.sum(0) > 0
    print(f"\n{'route (234 classes)':<34}{'BirdNET':>9}{'Perch':>7}")
    for label, f in [("mapped + labelled: head + blend", lambda h: h & has_pos),
                     ("mapped, unlabelled: zero-shot", lambda h: h & ~has_pos),
                     ("unmapped + labelled: head only", lambda h: ~h & has_pos),
                     ("unmapped, unlabelled: cold-start", lambda h: ~h & ~has_pos)]:
        print(f"{label:<34}{f(idx_b >= 0).sum():>9}{f(idx_p >= 0).sum():>7}")
    cold = (idx_b < 0) & ~has_pos
    gained = cold & (idx_p >= 0)
    print(f"\nof BirdNET's {cold.sum()} cold-start classes, Perch has a column for "
          f"{gained.sum()}: {tax.class_name[gained].value_counts().to_dict()}")

    # Zero-shot quality is what the 'mapped, unlabelled' route gets on the leaderboard.
    # Its only local proxy is zero-shot AUC on the mapped classes that do have labels.
    # Zero-shot uses no folds, so seed 0 is the whole answer.
    zb, zp = auc["BirdNET"]["zero-shot"][0], auc["Perch v2"]["zero-shot"][0]
    pm = (idx_p >= 0)[cols]
    print(f"zero-shot on the {birds.sum()} BirdNET-mapped: BirdNET {zb[birds].mean():.4f}  "
          f"Perch {zp[birds].mean():.4f}")
    print(f"zero-shot on Perch-mapped, not BirdNET-mapped ({(pm & ~birds).sum()}): "
          f"Perch {zp[pm & ~birds].mean():.4f}  (BirdNET has no column: 0.5)")

    assert abs(zb.mean() - 0.6168) < 1e-3, "BirdNET zero-shot did not reproduce"
    return auc


if __name__ == "__main__":
    main()
