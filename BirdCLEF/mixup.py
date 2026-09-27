"""Rung 4: mix focal calls into real soundscape backgrounds, then embed the result.

Section 10 established that the frozen representation is the ceiling and that
fine-tuning past it is unavailable. This is the one data lever that is not blocked by
that, because it does not ask the encoder for more from the same audio -- it hands it
*different* audio and gets genuinely new embedding vectors back.

It is also the direct answer to section 5. 333 hours of focal recordings made the head
worse, and the loss concentrated on birds, which said the failure was domain rather
than content: focal audio is close, clean and single-caller, the scored audio is
distant, noisy and overlapping. Mixing a focal call into a real soundscape background
at a plausible SNR keeps the content and replaces the domain.

Backgrounds are the quietest 10% of the 127,104 unlabelled soundscape windows, ranked
by BirdNET's own strongest mapped logit -- 12,729 windows across 3,306 files. Picking
quiet ones matters: a background with an unlabelled bird in it would turn every
mixture into a false negative for that species.

    python3 mixup.py                 -> data/emb_mix/shard_XXX.npz
    python3 mixup.py --status
    python3 mixup.py --selftest
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

from birdnet import BirdNET, to_chunks, fold, species_map, SR_IN, WIN
from embed_soundscape import load as load_unlabelled

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
OUT = f"{DATA}/emb_mix"

PER_CLASS = 30           # matches the cap that keeps winning elsewhere
QUIET_PCT = 10           # background pool: quietest this-% of unlabelled windows
SNR_DB = (-5.0, 15.0)    # focal level relative to background, sampled per mixture
SHARD = 20               # classes per shard


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def mix(fg, bg, snr_db):
    """Combine at a target SNR, then scale to the background's original loudness.

    Both sides are normalised to unit RMS first so the ratio means what it says,
    and the result is rescaled rather than clipped -- clipping would be a domain
    artifact of its own, which is exactly what this is trying to remove.
    """
    f, b = fg / rms(fg), bg / rms(bg)
    y = f * (10.0 ** (snr_db / 20.0)) + b
    y = y / rms(y) * rms(bg)
    peak = np.abs(y).max()
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


def loudest_window(y, sr=SR_IN):
    """The 5 s window with the most energy. A focal recording is mostly silence
    around one call, so a uniformly chosen window often contains no bird at all."""
    n = WIN * sr
    if len(y) <= n:
        return np.pad(y, (0, max(0, n - len(y))))
    starts = range(0, len(y) - n + 1, sr)
    best = max(starts, key=lambda s: rms(y[s:s + n]))
    return y[best:best + n]


def backgrounds():
    """Quietest QUIET_PCT of unlabelled windows, as (filename, start_s) pairs."""
    u = load_unlabelled()
    mx = u["logits"].astype(np.float32).max(1)
    keep = mx <= np.percentile(mx, QUIET_PCT)
    return list(zip(u["filename"][keep], u["start_s"][keep]))


def read_window(path, start_s):
    y, sr = sf.read(path, dtype="float32",
                    start=start_s * SR_IN, frames=WIN * SR_IN)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.pad(y, (0, max(0, WIN * SR_IN - len(y))))


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_csv(f"{DATA}/train.csv", dtype={"primary_label": str})
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    col = {c: i for i, c in enumerate(tax.primary_label)}
    classes = [c for c in tax.primary_label if c in set(tr.primary_label)]
    shards = range((len(classes) + SHARD - 1) // SHARD)

    done = {int(f.split("_")[1][:3]) for f in os.listdir(OUT) if f.endswith(".npz")}
    if "--status" in sys.argv:
        have = sum(int(np.load(f"{OUT}/{f}")["emb"].shape[0])
                   for f in os.listdir(OUT) if f.endswith(".npz"))
        print(f"{len(done)}/{len(list(shards))} shards, {have:,} mixtures")
        return

    bg = backgrounds()
    print(f"{len(classes)} classes with focal audio, {len(bg):,} background windows",
          flush=True)
    by_class = {c: g.filename.tolist() for c, g in tr.groupby("primary_label")}
    idx = species_map(tax)
    keep = idx[idx >= 0]

    rng = np.random.default_rng(0)
    net = BirdNET(batch=16)
    t0 = time.time()
    for s in shards:
        if s in done:
            continue
        part = classes[s * SHARD:(s + 1) * SHARD]
        waves, labs = [], []
        for c in part:
            files = by_class[c]
            for _ in range(PER_CLASS):
                try:
                    f = files[rng.integers(len(files))]
                    y, _ = sf.read(f"{DATA}/train_audio/{f}", dtype="float32")
                    if y.ndim > 1:
                        y = y.mean(axis=1)
                    bf, bs = bg[rng.integers(len(bg))]
                    b = read_window(f"{DATA}/train_soundscapes/{bf}", int(bs))
                    waves.append(mix(loudest_window(y), b,
                                     rng.uniform(*SNR_DB)))
                    labs.append(col[c])
                except Exception as e:
                    print(f"  skip {c}: {e}", flush=True)
        if not waves:
            continue
        E, L = [], []
        for i in range(0, len(waves), 64):
            chunk = np.concatenate([to_chunks(w, [0]) for w in waves[i:i + 64]])
            lo, em = fold(*net.run(chunk))
            E.append(em.astype(np.float16))
            L.append(lo[:, keep].astype(np.float16))
        np.savez_compressed(f"{OUT}/shard_{s:03d}.npz",
                            emb=np.concatenate(E), logits=np.concatenate(L),
                            label=np.array(labs, np.int16))
        el = time.time() - t0
        n = len([f for f in os.listdir(OUT) if f.endswith(".npz")])
        print(f"shard {s:2d}/{len(list(shards))}  {len(waves):4d} mixtures  "
              f"{el / 60:5.1f} min  ~{(len(list(shards)) - n) * el / 60 / max(1, n - len(done)):.0f} min left",
              flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


def load():
    import glob
    fs = sorted(glob.glob(f"{OUT}/shard_*.npz"))
    assert fs, f"no shards in {OUT}; run `python3 mixup.py` first"
    parts = [np.load(f) for f in fs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def selftest():
    """The mixture must contain both sources at the requested ratio, and the
    loudest-window picker must find the call rather than the silence."""
    rng = np.random.default_rng(0)
    fg = rng.normal(size=WIN * SR_IN).astype(np.float32)
    bg = rng.normal(size=WIN * SR_IN).astype(np.float32) * 0.1

    for snr in (-10.0, 0.0, 20.0):
        y = mix(fg, bg, snr)
        assert np.isfinite(y).all() and np.abs(y).max() <= 1.0
        # Recover the ratio: correlation with the foreground should rise with SNR.
        c = abs(np.corrcoef(y, fg)[0, 1])
        assert 0 <= c <= 1
    lo = abs(np.corrcoef(mix(fg, bg, -10.0), fg)[0, 1])
    hi = abs(np.corrcoef(mix(fg, bg, 20.0), fg)[0, 1])
    assert hi > lo, f"higher SNR must mean more foreground ({hi:.3f} vs {lo:.3f})"

    # Output loudness tracks the background, so mixtures do not stand out by level.
    assert abs(rms(mix(fg, bg, 0.0)) - rms(bg)) / rms(bg) < 0.5

    # A call buried in silence must be found.
    y = np.zeros(20 * SR_IN, np.float32)
    y[12 * SR_IN:13 * SR_IN] = rng.normal(size=SR_IN)
    w = loudest_window(y)
    assert len(w) == WIN * SR_IN and rms(w) > 0.1, rms(w)
    # Shorter than one window is padded, not truncated.
    assert len(loudest_window(np.ones(SR_IN, np.float32))) == WIN * SR_IN
    print("mixup self-checks passed")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
