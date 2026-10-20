"""Spectrogram SED branch: the log-mel front-end and an HGNetV2-B0 attention head.

The same LogMel module runs in training (GPU) and in the submission kernel (CPU), so
the front-end cannot drift between the two. It is plain torch.stft plus a mel
filterbank built in numpy; torchaudio is not needed anywhere.

Front-end parameters follow the public HGNetV2-B0 baseline and 13th place's CNN
family: 32 kHz, n_fft 2048, win 626, hop 313, 256 mels from 20 Hz to 16 kHz, power
2, dB with an 80 dB floor. A 5 s window becomes a 256 x 512 image.

    python3 sed_model.py --selftest
"""
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SR = 32_000
N_FFT, WIN, HOP = 2048, 626, 313
N_MELS, F_MIN, F_MAX = 256, 20.0, 16_000.0
TOP_DB = 80.0
N_CLASSES = 234
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX):
    """(n_fft//2 + 1, n_mels) triangular filters on the HTK mel scale, each scaled
    to unit area (Slaney normalisation), as torchaudio builds with norm="slaney"."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = hz(np.linspace(mel(f_min), mel(f_max), n_mels + 2))
    lo, ce, hi = pts[:-2], pts[1:-1], pts[2:]
    f = np.linspace(0, sr / 2, n_fft // 2 + 1)[:, None]
    fb = np.maximum(0, np.minimum((f - lo) / (ce - lo), (hi - f) / (hi - ce)))
    return (fb * 2.0 / (hi - lo)).astype(np.float32)


class LogMel(nn.Module):
    """5 s of 32 kHz audio -> a per-sample standardised dB mel image [B, 1, 256, 512]."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN), persistent=False)
        self.register_buffer("fb", torch.from_numpy(mel_filterbank().T.copy()), persistent=False)

    def forward(self, wave):
        spec = torch.stft(wave, N_FFT, HOP, WIN, window=self.window, center=True,
                          pad_mode="reflect", return_complex=True)
        db = 10 * torch.log10(torch.clamp(self.fb @ spec.abs().pow(2), min=1e-10))
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - TOP_DB)
        db = (db - db.mean(dim=(1, 2), keepdim=True)) / (db.std(dim=(1, 2), keepdim=True) + 1e-6)
        return db[:, None]


class SED(nn.Module):
    """HGNetV2-B0 feature map, averaged over frequency, read by an attention head.

    The clip score is an attention-weighted average of per-frame class logits, so
    a call that fills one second of the window can still carry the whole clip.
    """

    def __init__(self, backbone=BACKBONE, n_classes=N_CLASSES, pretrained=True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=1,
                                          num_classes=0, global_pool="")
        c = self.backbone.num_features
        self.proj = nn.Sequential(nn.Conv1d(c, 512, 1), nn.ReLU(), nn.Dropout(0.3))
        self.att = nn.Conv1d(512, n_classes, 1)
        self.cla = nn.Conv1d(512, n_classes, 1)

    def forward(self, mel):
        h = self.proj(self.backbone.forward_features(mel).mean(2))      # [B, 512, T']
        cla = self.cla(h)
        w = torch.softmax(torch.tanh(self.att(h)), dim=2)
        return (w * cla).sum(2), cla.transpose(1, 2)


def sed_loss(clip_logits, frame_logits, y):
    """Half on the attention-pooled clip, half on the loudest frame."""
    return 0.5 * F.binary_cross_entropy_with_logits(clip_logits, y) + \
        0.5 * F.binary_cross_entropy_with_logits(frame_logits.amax(1), y)


class _Prob(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        return torch.sigmoid(self.model(mel)[0])


def export_onnx(model, path):
    """mel [B, 1, 256, 512] -> prob [B, 234], batch dimension dynamic."""
    x = torch.zeros(2, 1, N_MELS, 512)
    torch.onnx.export(_Prob(model).eval(), (x,), path, input_names=["mel"], output_names=["prob"],
                      dynamic_shapes={"mel": {0: torch.export.Dim("batch")}}, external_data=False)


def selftest():
    import os
    import tempfile

    import onnxruntime as ort

    torch.manual_seed(0)
    fe = LogMel()
    m = fe(torch.zeros(2, 160_000))
    assert m.shape == (2, 1, 256, 512), m.shape
    assert torch.isfinite(m).all(), "silence must not produce inf or nan"

    # A pure tone lands in the mel bin whose centre is nearest its frequency.
    t = torch.arange(160_000) / 32_000
    tone = fe(0.5 * torch.sin(2 * np.pi * 4000 * t)[None])
    peak = int(tone[0, 0].mean(1).argmax())
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    centres = np.linspace(mel(20), mel(16_000), 256 + 2)[1:-1]
    want = int(np.abs(centres - mel(4000)).argmin())
    assert abs(peak - want) <= 1, (peak, want)

    model = SED(pretrained=False).eval()
    x = fe(torch.randn(3, 160_000))
    with torch.no_grad():
        clip, frame = model(x)
    assert clip.shape == (3, 234), clip.shape
    assert frame.shape[0] == 3 and frame.shape[2] == 234, frame.shape
    # Attention pooling is a convex combination of the frame logits.
    assert (clip >= frame.min(1).values - 1e-5).all()
    assert (clip <= frame.max(1).values + 1e-5).all()

    y = torch.zeros(3, 234)
    low = torch.full((3, 234), -20.0)
    assert sed_loss(low, torch.full((3, 7, 234), -20.0), y) < 1e-6

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sed.onnx")
        export_onnx(model, path)
        # One self-contained file: a separate .onnx.data weights file is easy to
        # leave behind when uploading the model to Kaggle.
        assert os.listdir(d) == ["sed.onnx"], os.listdir(d)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        got = sess.run(["prob"], {"mel": x.numpy()})[0]
    diff = np.abs(got - torch.sigmoid(clip).numpy()).max()
    assert diff < 1e-4, diff
    print("sed_model self-checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
