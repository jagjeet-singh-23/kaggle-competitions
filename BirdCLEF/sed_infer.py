"""CPU inference for the SED branch, and its rank blend with the Perch branch.

The mel front-end runs in torch (the same LogMel module training used, so the two
cannot drift), and the backbone + head run in onnxruntime.

    python3 sed_infer.py --selftest       (SED_CACHE, SED_CKPT pick the files)
"""
import os
import sys

import numpy as np

DEFAULT_CACHE = "/Volumes/Extreme SSD/kaggle/BirdCLEF/jobs/sed-cache"
DEFAULT_CKPT = "/Volumes/Extreme SSD/kaggle/BirdCLEF/jobs/sed-out/fold0/sed_fold0_ema.pt"


def rank(M):
    """Per-column tie-aware rank on a [0, 1] scale (ties get their average rank)."""
    from scipy.stats import rankdata

    return (rankdata(M, axis=0) - 1) / max(1, len(M) - 1)


def blend(perch_rank, sed_prob, w):
    """(1 - w) * Perch's per-class ranks + w * SED's per-class ranks."""
    return (1 - w) * perch_rank + w * rank(sed_prob)


class SEDOnnx:
    """5 s windows -> per-class probabilities: LogMel in torch, the network in ORT."""

    def __init__(self, onnx_path, threads=os.cpu_count(), batch=32):
        import onnxruntime as ort
        import torch

        from sed_model import LogMel

        torch.set_num_threads(threads)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.fe, self.batch, self.torch = LogMel().eval(), batch, torch

    def run(self, windows):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(windows), self.batch):
                mel = self.fe(self.torch.from_numpy(np.ascontiguousarray(
                    windows[i:i + self.batch], dtype=np.float32))).numpy()
                out.append(self.sess.run(["prob"], {"mel": mel})[0])
        return np.concatenate(out)


def selftest():
    import tempfile

    import torch
    from scipy.stats import rankdata

    from sed_model import SED, LogMel, export_onnx

    rng = np.random.default_rng(0)
    P, S = rng.random((50, 5)), rng.random((50, 5))
    S[:10, 0] = 0.5                                  # ties must get their average rank
    assert np.allclose(blend(P, S, 0.0), P)
    want = (rankdata(S, axis=0) - 1) / (len(S) - 1)
    assert np.allclose(blend(P, S, 1.0), want)

    cache = os.environ.get("SED_CACHE", DEFAULT_CACHE)
    model = SED(pretrained=False)
    model.load_state_dict(torch.load(os.environ.get("SED_CKPT", DEFAULT_CKPT), map_location="cpu"))
    model.eval()
    w = np.load(f"{cache}/labelled_audio.npy", mmap_mode="r")[:32].astype(np.float32) / 32767
    with torch.no_grad():
        ref = torch.sigmoid(model(LogMel()(torch.from_numpy(w)))[0]).numpy()
    with tempfile.TemporaryDirectory() as d:
        export_onnx(model, f"{d}/sed.onnx")
        got = SEDOnnx(f"{d}/sed.onnx", threads=4).run(w)
    assert got.shape == (32, 234), got.shape
    diff = np.abs(got - ref).max()
    assert diff < 1e-4, diff
    print(f"sed_infer self-checks passed (max |onnx - torch| = {diff:.1e})")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
