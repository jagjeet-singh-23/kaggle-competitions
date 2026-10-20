"""Google Perch v2 as a frozen feature extractor, through an ONNX export.

The model takes exactly one competition window -- 5.0 s at 32 kHz -- and returns
14,795 logits plus a 1,536-dim embedding. Unlike BirdNET there is no resampling and
no two-chunk fold: one window in, one row out.

203 of the 234 competition classes have an exact scientific-name match among the
logits: all 162 birds, 32 of 35 amphibians, 6 of 8 mammals and 3 of 28 insects.
BirdNET matched 157, all of them birds.

Model: the official SavedModel is google/bird-vocalization-classifier/tensorFlow2/
perch_v2_cpu on Kaggle Models. It runs at 150 ms per window on an M4 and needs
TensorFlow. The ONNX export used here is the Kaggle dataset
rishikeshjani/perch-onnx-for-birdclef-2026. It was checked against the official model
on 12 windows of a labelled soundscape: embeddings within 3.8e-6, logits within 9e-5,
identical top-20 logits on every window, at 48 ms per window on CPU. The CoreML
provider was 10x slower and drifted (logits off by up to 0.19), so it is not used.
Place perch_v2.onnx and labels.csv in models/onnx_rishikeshjani/.
"""
import os

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = os.environ.get("PERCH_DIR", f"{DIR}/models/onnx_rishikeshjani")
MODEL = f"{MODELS}/perch_v2.onnx"
LABELS = f"{MODELS}/labels.csv"       # one header line, then 14,795 scientific names

SR_IN = 32_000       # competition audio, and what Perch expects
WIN = 5              # seconds per scored row
WIN_SAMPLES = WIN * SR_IN
EMB_DIM = 1536
N_LABELS = 14_795


def labels():
    with open(LABELS) as f:
        names = [l.strip() for l in f][1:]
    assert len(names) == N_LABELS, len(names)
    return names


class Perch:
    """Thin wrapper returning (logits, embeddings) for a stack of 5 s windows."""

    def __init__(self, batch=16, threads=os.cpu_count()):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(MODEL, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.batch = batch

    def run(self, windows):
        """windows: (n, 160000) float32 at 32 kHz -> (logits (n, 14795), emb (n, 1536))."""
        lo, em = [], []
        for i in range(0, len(windows), self.batch):
            e, l = self.sess.run(["embedding", "label"],
                                 {"inputs": windows[i:i + self.batch].astype(np.float32)})
            lo.append(l)
            em.append(e)
        return np.concatenate(lo), np.concatenate(em)


def to_windows(y32, starts):
    """Cut 5 s windows starting at `starts` (seconds) out of 32 kHz audio, padding the
    tail of a short recording with silence."""
    out = np.zeros((len(starts), WIN_SAMPLES), np.float32)
    for k, s in enumerate(starts):
        seg = y32[s * SR_IN:(s + WIN) * SR_IN]
        out[k, :len(seg)] = seg
    return out


def species_map(taxonomy):
    """Column index into Perch's 14,795 outputs for each competition class, or -1.

    Exact scientific-name match, case-insensitive: 203 of 234.
    """
    pos = {n.lower(): i for i, n in enumerate(labels())}
    return np.array([pos.get(str(s).strip().lower(), -1) for s in taxonomy.scientific_name])
