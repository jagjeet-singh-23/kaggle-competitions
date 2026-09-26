"""BirdNET V2.4 as a frozen feature extractor.

The model takes 3.0 s at 48 kHz and returns 6,522 species logits plus a 1,024-dim
embedding. The competition scores 5 s windows of 32 kHz audio, so every window is
resampled 32k -> 48k (a clean 3/2 ratio) and covered by two overlapping 3 s chunks.

157 of the 162 bird classes in taxonomy.csv appear in BirdNET's label set, so its
logits are a prediction for two thirds of the competition with no training at all.
The other 77 classes are amphibians, insects, mammals and one reptile, which BirdNET
has never seen; those need a head trained on the embeddings.

Model: https://tuc.cloud/index.php/s/886x39f5N3sdsAM/download/V2.4.zip (MIT licence,
kahst/BirdNET-Analyzer). Place the FP32 tflite and the labels file in models/.
"""
import os

import numpy as np
from scipy.signal import resample_poly

DIR = os.path.dirname(os.path.abspath(__file__))

# A Kaggle kernel has no internet, so the weights arrive as an attached dataset at a
# path that is not next to this file. Everything else about the wrapper is unchanged.
MODELS = os.environ.get("BIRDNET_DIR", f"{DIR}/models")
MODEL = f"{MODELS}/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
LABELS = f"{MODELS}/BirdNET_GLOBAL_6K_V2.4_Labels.txt"

SR_IN = 32_000       # competition audio
SR_BN = 48_000       # what BirdNET expects
CHUNK = 144_000      # 3.0 s at 48 kHz
WIN = 5              # seconds per scored row
EMB_DIM = 1024


class BirdNET:
    """Thin wrapper that returns both the logits and the penultimate embedding.

    TFLite drops intermediate tensors unless asked to keep them, so the embedding
    needs `experimental_preserve_all_tensors`. It costs about 25% throughput
    (24 ms vs 18 ms per chunk) and is the only way to reach the pooled layer.
    """

    def __init__(self, batch=8, threads=os.cpu_count()):
        from ai_edge_litert.interpreter import Interpreter

        self.batch = batch
        self.it = Interpreter(model_path=MODEL, num_threads=threads,
                              experimental_preserve_all_tensors=True)
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        self.emb = self.out - 1           # model/GLOBAL_AVG_POOL/Mean, verified (1, 1024)
        self.it.resize_tensor_input(self.inp, [batch, CHUNK])
        self.it.allocate_tensors()
        self.labels = [l.strip() for l in open(LABELS)]

    def run(self, chunks):
        """chunks: (n, 144000) float32 at 48 kHz -> (logits, embeddings)."""
        n = len(chunks)
        lo, em = [], []
        for i in range(0, n, self.batch):
            part = chunks[i:i + self.batch]
            if len(part) < self.batch:     # pad the tail; the interpreter is fixed-size
                part = np.concatenate([part, np.zeros((self.batch - len(part), CHUNK), np.float32)])
            self.it.set_tensor(self.inp, part.astype(np.float32))
            self.it.invoke()
            lo.append(self.it.get_tensor(self.out).copy())
            em.append(self.it.get_tensor(self.emb).copy())
        return np.concatenate(lo)[:n], np.concatenate(em)[:n]


def to_chunks(y32, starts):
    """Cut 5 s windows out of 32 kHz audio and return two 3 s BirdNET chunks each.

    A 5 s window resampled to 48 kHz is 240,000 samples and a chunk is 144,000, so
    one chunk cannot cover it. Two windows (0-3 s and 2-5 s) cover it with 1 s of
    overlap, and a call near the seam lands whole inside at least one of them.
    """
    out = np.zeros((len(starts) * 2, CHUNK), np.float32)
    for k, s in enumerate(starts):
        seg = y32[s * SR_IN:(s + WIN) * SR_IN]
        if len(seg) < WIN * SR_IN:
            seg = np.pad(seg, (0, WIN * SR_IN - len(seg)))
        y = resample_poly(seg, 3, 2).astype(np.float32)   # 32k -> 48k
        out[2 * k] = y[:CHUNK]
        out[2 * k + 1] = y[-CHUNK:]
    return out


def fold(logits, emb):
    """Collapse the two chunks per window back into one row.

    Max over chunks for logits, because a call present in either half is present in
    the window; mean for embeddings, because they are a description of the audio
    rather than a detection.
    """
    return logits.reshape(-1, 2, logits.shape[-1]).max(axis=1), \
        emb.reshape(-1, 2, emb.shape[-1]).mean(axis=1)


def species_map(taxonomy):
    """Column index into BirdNET's 6,522 outputs for each competition class, or -1.

    Matched on scientific name, which covers 157 of 162 birds and none of the 72
    non-bird classes.
    """
    labels = [l.strip() for l in open(LABELS)]
    bn = {l.split("_")[0].strip().lower(): i for i, l in enumerate(labels)}
    return np.array([bn.get(str(s).strip().lower(), -1) for s in taxonomy.scientific_name])
