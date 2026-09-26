"""Rung 0: a valid submission, and a measurement of the 90-minute CPU budget.

This notebook predicts nothing (0.5 everywhere, AUC 0.5). Its job is to answer the
questions that decide what model is even affordable here:

  - how long does decoding ~600 one-minute ogg files actually take
  - how long does a mel-spectrogram of every 5-second window take
  - how much of the 90 minutes is left for a model

Submitting something useless early is cheaper than discovering at minute 88 that
the pipeline never fits.
"""
import os
import time
import glob

import numpy as np
import pandas as pd
import soundfile as sf

T0 = time.time()

# A notebook created in the UI mounts the competition at /kaggle/input/<slug>;
# one pushed through the API mounts it at /kaggle/input/competitions/<slug>.
# Resolve at runtime so the same file works either way.
COMP = next(p for p in ("/kaggle/input/birdclef-2026",
                        "/kaggle/input/competitions/birdclef-2026")
            if os.path.exists(f"{p}/taxonomy.csv"))
print(f"data at {COMP}", flush=True)
SR = 32_000
WIN = 5                      # seconds per scored row
CLIP = 60                    # test soundscapes are exactly one minute
N_MELS = 128
HOP = 320                    # 100 frames per second at 32 kHz


def elapsed():
    return time.time() - T0


test = sorted(glob.glob(f"{COMP}/test_soundscapes/*.ogg"))
rerun = len(test) > 0
if not rerun:
    # Outside the rerun the directory holds only a readme, so time the decode
    # against train_soundscapes, which is the same format and length.
    test = sorted(glob.glob(f"{COMP}/train_soundscapes/*.ogg"))[:60]
print(f"rerun={rerun}  files={len(test)}", flush=True)

species = pd.read_csv(f"{COMP}/taxonomy.csv", dtype={"primary_label": str})
species = species.primary_label.tolist()
print(f"{len(species)} species columns", flush=True)


def mel(y):
    """Log-mel via numpy so the timing does not depend on librosa being present."""
    n_fft = 1024
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(y) - n_fft) // HOP
    idx = np.arange(n_fft)[None, :] + HOP * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * win, axis=1)) ** 2
    # Triangular filterbank is overkill for a timing probe; log-average into
    # N_MELS bands, which costs the same order of arithmetic.
    edges = np.linspace(0, spec.shape[1], N_MELS + 1).astype(int)
    banks = np.stack([spec[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    return np.log10(banks + 1e-10)


t_decode = t_feat = 0.0
rows = []
for path in test:
    name = os.path.basename(path)
    t = time.time()
    y, sr = sf.read(path, dtype="float32")
    t_decode += time.time() - t

    t = time.time()
    for w in range(CLIP // WIN):
        seg = y[w * WIN * sr:(w + 1) * WIN * sr]
        if len(seg) >= 1024:
            mel(seg)
        rows.append(f"{name[:-4]}_{(w + 1) * WIN}")
    t_feat += time.time() - t

n = len(test)
print(f"\ndecode : {t_decode:6.1f}s total = {1000 * t_decode / n:6.1f} ms/file", flush=True)
print(f"mel    : {t_feat:6.1f}s total = {1000 * t_feat / n:6.1f} ms/file", flush=True)

# The real test set is ~600 files; scale whatever we just measured up to that.
scale = 600 / n
budget = 90 * 60
used = (t_decode + t_feat) * scale
print(f"\nprojected for 600 files: decode+mel {used / 60:.1f} min of the 90 min budget")
print(f"leaves {(budget - used) / 60:.1f} min for model inference over "
      f"{600 * CLIP // WIN:,} windows")
print(f"  -> {1000 * (budget - used) / (600 * CLIP // WIN):.1f} ms per window", flush=True)

if rerun:
    sub = pd.concat([pd.DataFrame({"row_id": rows}),
                     pd.DataFrame(0.5, index=range(len(rows)), columns=species)], axis=1)
else:
    # Outside the rerun the row_ids above came from train_soundscapes, so fall
    # back to the sample to keep the committed file valid.
    sub = pd.read_csv(f"{COMP}/sample_submission.csv")
    sub[species] = 0.5

sub.to_csv("submission.csv", index=False)
print(f"\nsubmission.csv: {sub.shape[0]:,} rows x {sub.shape[1]} cols")
print(f"total runtime {elapsed() / 60:.1f} min", flush=True)
