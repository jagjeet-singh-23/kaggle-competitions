"""Extract BirdNET embeddings for the 10,592 unlabelled soundscape recordings.

`train_soundscapes/` holds 10,658 one-minute recordings and exactly 66 of them are
labelled. The other 10,592 are 177 hours of audio from the scored domain, and nothing
in this repo has touched them.

That matters because the one thing that has consistently failed here is crossing the
domain gap. 333 hours of focal audio made the head worse (RESULTS.md section 5), and
helped only where the alternative was a coin flip (section 8). This data has no gap
to cross -- it is the same recorders, sites and conditions as the test set. It just
has no labels, which is what pseudo-labelling is for.

Sharded every SHARD files so a crash costs one shard, and resumable:

    python3 embed_soundscape.py            -> data/emb_sound/shard_XXX.npz
    python3 embed_soundscape.py --status
    python3 embed_soundscape.py --limit=2000   # enough to test the idea first

Only the 157 BirdNET columns that map to competition species are kept, the same as
embed_train.py: the other 6,365 are species that cannot be scored here.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

from birdnet import BirdNET, to_chunks, fold, species_map, WIN

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
SRC = f"{DATA}/train_soundscapes"
OUT = f"{DATA}/emb_sound"
SHARD = 500          # files per shard; each file is 60 s -> 12 windows
WINDOWS = 12


def unlabelled():
    """Every soundscape recording that is not one of the 66 labelled ones."""
    lab = set(pd.read_csv(f"{DATA}/train_soundscapes_labels.csv").filename)
    fs = sorted(f for f in os.listdir(SRC)
                if f.endswith(".ogg") and f not in lab)
    return fs


def main():
    os.makedirs(OUT, exist_ok=True)
    files = unlabelled()
    limit = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), 0)
    if limit:
        # Evenly spaced rather than the first N, so a partial run still spans all
        # 23 sites instead of whichever ones sort first.
        files = [files[i] for i in np.linspace(0, len(files) - 1, limit).astype(int)]
    n_shards = (len(files) + SHARD - 1) // SHARD

    done = {int(f.split("_")[1][:3]) for f in os.listdir(OUT) if f.endswith(".npz")}
    if "--status" in sys.argv:
        have = sum(int(np.load(f"{OUT}/{f}")["emb"].shape[0])
                   for f in os.listdir(OUT) if f.endswith(".npz"))
        print(f"{len(done)}/{n_shards} shards, {have:,} windows")
        return

    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    idx = species_map(tax)
    keep = idx[idx >= 0]

    net = BirdNET(batch=16)
    t0 = time.time()
    for s in range(n_shards):
        if s in done:
            continue
        part = files[s * SHARD:(s + 1) * SHARD]
        E, L, fi, st = [], [], [], []
        for f in part:
            try:
                y, _ = sf.read(f"{SRC}/{f}", dtype="float32")
            except Exception as e:
                print(f"  skip {f}: {e}", flush=True)
                continue
            if y.ndim > 1:
                y = y.mean(axis=1)
            starts = list(range(0, WINDOWS * WIN, WIN))
            lo, em = fold(*net.run(to_chunks(y, starts)))
            E.append(em.astype(np.float16))
            L.append(lo[:, keep].astype(np.float16))
            fi += [f] * len(starts)
            st += starts
        np.savez_compressed(f"{OUT}/shard_{s:03d}.npz",
                            emb=np.concatenate(E), logits=np.concatenate(L),
                            filename=np.array(fi), start_s=np.array(st))
        el = time.time() - t0
        n = len([f for f in os.listdir(OUT) if f.endswith(".npz")])
        left = (n_shards - n) * el / max(1, n - len(done))
        print(f"shard {s:3d}/{n_shards}  {sum(len(e) for e in E):5d} windows  "
              f"{el / 60:6.1f} min elapsed  ~{left / 60:.0f} min left", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


def load(max_shards=None):
    """Concatenate whatever shards exist. Partial extractions are usable."""
    import glob
    fs = sorted(glob.glob(f"{OUT}/shard_*.npz"))[:max_shards]
    assert fs, f"no shards in {OUT}; run `python3 embed_soundscape.py` first"
    parts = [np.load(f) for f in fs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


if __name__ == "__main__":
    main()
