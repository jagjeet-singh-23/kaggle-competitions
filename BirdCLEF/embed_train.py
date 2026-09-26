"""Extract BirdNET embeddings for train_audio.

35,549 focal clips, 333 hours. Sampling up to WINDOWS 5-second windows per clip
keeps this near three hours on a laptop instead of ten.

Writes shards so a crash costs one shard, not the whole run:

    python3 embed_train.py            -> data/emb_train/shard_XXX.npz
    python3 embed_train.py --status   -> how far along it is
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

from birdnet import BirdNET, to_chunks, fold, species_map, SR_IN, WIN

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = f"{DIR}/data"
OUT = f"{DATA}/emb_train"
WINDOWS = 6          # per clip; median clip is 20.9 s so most are fully covered
SHARD = 2000         # clips per shard


def starts_for(dur):
    """Up to WINDOWS window starts, evenly spread, always including 0.

    Uniform spacing rather than energy-based selection: 12% of clips run past a
    minute and the call can sit anywhere in them. A louder-window heuristic would
    likely beat this and is the obvious thing to try next.
    """
    n = max(1, int(dur // WIN))
    if n <= WINDOWS:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, WINDOWS)})


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_csv(f"{DATA}/train.csv", dtype={"primary_label": str})
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    col = {c: i for i, c in enumerate(tax.primary_label)}
    keep = species_map(tax)
    keep = keep[keep >= 0]                      # the 157 BirdNET columns we care about

    done = {int(f.split("_")[1][:3]) for f in os.listdir(OUT) if f.endswith(".npz")}
    shards = range((len(tr) + SHARD - 1) // SHARD)
    if "--status" in sys.argv:
        print(f"{len(done)}/{len(list(shards))} shards done")
        return

    net = BirdNET(batch=16)
    t0 = time.time()
    for s in shards:
        if s in done:
            continue
        part = tr.iloc[s * SHARD:(s + 1) * SHARD]
        E, L, lab, fid = [], [], [], []
        for row in part.itertuples():
            try:
                info = sf.info(f"{DATA}/train_audio/{row.filename}")
                y, sr = sf.read(f"{DATA}/train_audio/{row.filename}", dtype="float32")
            except Exception as e:                 # a handful of clips are unreadable
                print(f"  skip {row.filename}: {e}", flush=True)
                continue
            if y.ndim > 1:
                y = y.mean(axis=1)
            st = starts_for(info.duration)
            lo, em = fold(*net.run(to_chunks(y, st)))
            E.append(em.astype(np.float16))
            L.append(lo[:, keep].astype(np.float16))
            lab += [col.get(row.primary_label, -1)] * len(st)
            fid += [row.Index] * len(st)
        np.savez_compressed(f"{OUT}/shard_{s:03d}.npz",
                            emb=np.concatenate(E), logits=np.concatenate(L),
                            label=np.array(lab), clip=np.array(fid))
        el = time.time() - t0
        n = len([f for f in os.listdir(OUT) if f.endswith(".npz")])
        print(f"shard {s:3d}  {sum(len(e) for e in E):6d} windows  "
              f"{el / 60:6.1f} min elapsed  ~{el / 60 / max(1, n - len(done)) * (len(list(shards)) - n):.0f} min left",
              flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
