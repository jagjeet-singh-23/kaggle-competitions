"""Perch v2 embeddings for the focal pool the kernel's cold-start heads train on.

Same recipe as focal_cap30.npz (embed_train.py windows, 30 per class, seeded), but
sampled directly instead of out of BirdNET shards, so only the ~6,200 windows that are
used get embedded: about five minutes instead of four hours.

Under Perch only 2 classes still need a cold-start head (no Perch column, no
labelled positive). The other classes' rows are their negatives.

    python3 embed_focal_perch.py      -> data/focal_cap30_perch.npz
"""
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf

from embed_train import starts_for
from perch import Perch, to_windows, SR_IN

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
OUT = f"{DATA}/focal_cap30_perch.npz"
CAP = 30             # beat 150 and 400 on the cold-start proxy (RESULTS section 8)


def main():
    tr = pd.read_csv(f"{DATA}/train.csv", dtype={"primary_label": str})
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    col = {c: i for i, c in enumerate(tax.primary_label)}

    t0 = time.time()
    cand = []                                       # (class, clip row, start seconds)
    for row in tr.itertuples():
        try:
            dur = sf.info(f"{DATA}/train_audio/{row.filename}").duration
        except Exception as e:
            print(f"  skip {row.filename}: {e}", flush=True)
            continue
        cand += [(col[row.primary_label], row.Index, s) for s in starts_for(dur)]
    cand = pd.DataFrame(cand, columns=["label", "src", "start"])
    print(f"{len(cand):,} candidate windows from {cand.src.nunique():,} clips "
          f"in {time.time() - t0:.0f}s")

    # Not groupby().apply(): pandas 3 drops the grouping column from its result.
    rng = np.random.default_rng(0)
    sel = []
    for _, g in cand.groupby("label"):
        sel += list(g.index[rng.choice(len(g), min(CAP, len(g)), replace=False)])
    pick = cand.loc[sel].sort_values(["src", "start"]).reset_index(drop=True)
    print(f"picked {len(pick):,} windows, {pick.label.nunique()} classes, "
          f"{pick.src.nunique():,} clips")

    net = Perch()
    E, lab, clip, start = [], [], [], []
    for n, (c, grp) in enumerate(pick.groupby("src", sort=False)):
        y, sr = sf.read(f"{DATA}/train_audio/{tr.filename[c]}", dtype="float32")
        assert sr == SR_IN, sr
        if y.ndim > 1:
            y = y.mean(axis=1)
        _, em = net.run(to_windows(y, grp.start.tolist()))
        E.append(em.astype(np.float16))
        lab += grp.label.tolist()
        clip += [c] * len(grp)
        start += grp.start.tolist()
        if n % 500 == 0:
            print(f"  {n:,}/{pick.src.nunique():,} clips  {time.time() - t0:.0f}s", flush=True)

    E = np.concatenate(E)
    np.savez_compressed(OUT, emb=E, label=np.array(lab, np.int16),
                        clip=np.array(clip), start_s=np.array(start))
    print(f"{OUT}: emb {E.shape}, {len(set(lab))} classes, {time.time() - t0:.0f}s total")
    assert len(E) == len(pick) and np.isfinite(E.astype(np.float32)).all()


if __name__ == "__main__":
    main()
