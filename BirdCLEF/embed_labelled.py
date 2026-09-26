"""Extract BirdNET logits and embeddings for the labelled soundscape segments.

This is the only labelled in-domain data in the competition: 739 five-second
segments across 66 files, one hour of audio in total. Everything local is validated
against it, so it gets built first and cached.

    python3 embed_labelled.py     -> data/labelled.npz
"""
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf

from birdnet import BirdNET, to_chunks, fold, SR_IN, WIN

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
OUT = f"{DATA}/labelled.npz"


def segments():
    """Deduped labelled segments. The shipped csv has every row twice."""
    lab = pd.read_csv(f"{DATA}/train_soundscapes_labels.csv").drop_duplicates()
    lab["start_s"] = pd.to_timedelta(lab.start).dt.total_seconds().astype(int)
    return lab.sort_values(["filename", "start_s"]).reset_index(drop=True)


def main():
    lab = segments()
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    classes = tax.primary_label.tolist()
    col = {c: i for i, c in enumerate(classes)}

    net = BirdNET()
    t0 = time.time()
    L, E, rows = [], [], []
    for fn, grp in lab.groupby("filename", sort=False):
        y, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32")
        assert sr == SR_IN, sr
        starts = grp.start_s.tolist()
        lo, em = fold(*net.run(to_chunks(y, starts)))
        L.append(lo)
        E.append(em)
        for s, lbl in zip(starts, grp.primary_label.astype(str)):
            rows.append((fn, s, lbl))
        print(f"\r{len(rows):4d}/{len(lab)} segments  {time.time() - t0:5.1f}s", end="", flush=True)
    print()

    L, E = np.concatenate(L), np.concatenate(E)
    meta = pd.DataFrame(rows, columns=["filename", "start_s", "labels"])
    meta["site"] = meta.filename.str.split("_").str[3]

    Y = np.zeros((len(meta), len(classes)), np.float32)
    for i, lbl in enumerate(meta.labels):
        for s in lbl.split(";"):
            if s in col:
                Y[i, col[s]] = 1.0

    np.savez_compressed(OUT, logits=L, emb=E, Y=Y,
                        filename=meta.filename.values, site=meta.site.values,
                        start_s=meta.start_s.values, classes=np.array(classes))
    print(f"\n{OUT}")
    print(f"  logits {L.shape}  emb {E.shape}  Y {Y.shape}")
    print(f"  positives per segment: mean {Y.sum(1).mean():.2f}  max {int(Y.sum(1).max())}")
    print(f"  classes with >=1 positive: {int((Y.sum(0) > 0).sum())} of {len(classes)}")
    print(f"  files {meta.filename.nunique()}  sites {meta.site.nunique()}")
    print(f"  {time.time() - t0:.1f}s total")

    assert L.shape[0] == E.shape[0] == len(meta) == 739, len(meta)
    assert E.shape[1] == 1024 and L.shape[1] == 6522
    assert Y.sum() > 0 and np.isfinite(E).all()


if __name__ == "__main__":
    main()
