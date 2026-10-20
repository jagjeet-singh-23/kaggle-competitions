"""BirdCLEF+ 2026 submission kernel: the Perch kernel plus an HGNetV2-B0 SED branch.

Kernels-only competition, CPU, no internet, 90 minutes. The Perch half is
kernel_perch/birdclef_perch_submit.py unchanged (0.92681 private). The SED branch
scores the same 5 s windows and is rank-blended in before kernel v4's smoother:

    MODE = "blend":  (1 - W_SED) * rank(Perch) + W_SED * rank(SED)
    MODE = "sed":    rank(SED) alone, for diagnosis

SED pilot: one fold, public pseudo-labels (docs/superpowers/plans/2026-10-20-sed-pilot.md).
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

MODE = "blend"
W_SED = 0.5
SED_ONNX = "sed_fold0_B.onnx"    # pilot A: sed_fold0.onnx

t0 = time.time()
step = lambda s: print(f"[{(time.time() - t0) / 60:5.1f} min] {s}", flush=True)

# Overridable so the whole kernel can be run locally against a fake input tree.
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORKING", "/kaggle/working")


def find(marker, root=INPUT):
    for dirpath, _, files in os.walk(root):
        if marker in files:
            return dirpath
    raise FileNotFoundError(f"{marker} not found under {root}")


COMP = find("taxonomy.csv")
CODE = find("sed_infer.py")
PERCH = find("perch_v2.onnx")
ASSETS = find("focal_cap30_perch.npz")
SEDM = find(SED_ONNX)

# The ONNX file comes from a public dataset. This is the size of the copy that was
# checked against Google's SavedModel (embeddings within 3.8e-6).
assert os.path.getsize(f"{PERCH}/perch_v2.onnx") == 409_148_616, "perch_v2.onnx changed"

try:
    import onnxruntime  # noqa: F401
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index",
                           "--find-links", ASSETS, "--find-links", PERCH,
                           "onnxruntime==1.24.4"])
    step(f"installed onnxruntime offline on Python {sys.version.split()[0]}")

os.environ["BIRDCLEF_DATA"] = COMP
os.environ["PERCH_DIR"] = PERCH
sys.path.insert(0, CODE)

import soundfile as sf                                              # noqa: E402
import torch                                                        # noqa: E402
from perch import Perch, to_windows, species_map, WIN, SR_IN        # noqa: E402
from embed_labelled import segments                                 # noqa: E402
from sed_infer import SEDOnnx, blend, rank                          # noqa: E402
from sklearn.linear_model import LogisticRegression                 # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402

step(f"MODE={MODE} W_SED={W_SED} SED_ONNX={SED_ONNX}  torch {torch.__version__}  comp={COMP}  code={CODE}  "
     f"sed={SEDM}")

tax = pd.read_csv(f"{COMP}/taxonomy.csv", dtype={"primary_label": str})
classes = tax.primary_label.tolist()
col = {c: i for i, c in enumerate(classes)}
idx = species_map(tax)
hit = idx >= 0
keep = idx[hit]
assert hit.sum() == 203, hit.sum()

net = Perch(batch=16)
sed = SEDOnnx(f"{SEDM}/{SED_ONNX}")


def read(path):
    y, _ = sf.read(path, dtype="float32")
    return y if y.ndim == 1 else y.mean(axis=1)


def embed(W):
    lo, em = net.run(W)
    return lo[:, keep], em


# ---- Perch heads: kernel_perch unchanged --------------------------------------------
lab = segments()
L, E, rows, lab_files, lab_starts = [], [], [], [], []
t_emb = time.time()
for fn, grp in lab.groupby("filename", sort=False):
    starts = grp.start_s.tolist()
    lo, em = embed(to_windows(read(f"{COMP}/train_soundscapes/{fn}"), starts))
    L.append(lo)
    E.append(em)
    rows += list(grp.primary_label.astype(str))
    lab_files += [fn] * len(starts)
    lab_starts += starts
Ltr, Etr = np.concatenate(L), np.concatenate(E)
perch_ms = (time.time() - t_emb) / len(Etr) * 1000
step(f"labelled: {Etr.shape} emb at {perch_ms:.0f} ms/window (Perch)")
Y = np.zeros((len(rows), len(classes)), np.float32)
for i, lbl in enumerate(rows):
    for s in lbl.split(";"):
        if s in col:
            Y[i, col[s]] = 1.0
has_pos = Y.sum(0) > 0

sc = StandardScaler().fit(np.hstack([Etr, Ltr]))
Xtr = sc.transform(np.hstack([Etr, Ltr]))
models = {j: LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(Xtr, Y[:, j])
          for j in np.where(has_pos)[0]}
fc = np.load(f"{ASSETS}/focal_cap30_perch.npz")
Xf, yf = fc["emb"].astype(np.float32), fc["label"]
scf = StandardScaler().fit(Xf)
Xf_s = scf.transform(Xf)
cold_models = {j: LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")
               .fit(Xf_s, (yf == j).astype(int))
               for j in range(len(classes)) if not has_pos[j] and not hit[j] and (yf == j).sum() > 0}
assert len(models) + int(hit[~has_pos].sum()) + len(cold_models) == len(classes), \
    "some class would emit a 0.5 constant"
step(f"fitted {len(models)} soundscape heads, {len(cold_models)} cold-start heads")

SMOOTH_K, SMOOTH_ALPHA = 6, 0.2


def smooth(P, files, starts):
    """Kernel v4's smoother: blend each window with the +/-k window max of its file."""
    Pr = rank(P)
    ctx = np.empty_like(Pr)
    order = np.lexsort((starts, files))
    for f in np.unique(files):
        ix = order[files[order] == f]
        Q, n = Pr[ix], len(ix)
        acc = []
        for off in range(-SMOOTH_K, SMOOTH_K + 1):
            src = max(0, off)
            cnt = max(0, min(n, n + off) - src)
            pad = np.full_like(Q, np.nan)
            if cnt:
                dst = max(0, -off)
                pad[dst:dst + cnt] = Q[src:src + cnt]
            acc.append(pad)
        ctx[ix] = np.nanmax(np.stack(acc), 0)
    return SMOOTH_ALPHA * Pr + (1 - SMOOTH_ALPHA) * rank(ctx)


def predict_perch(Lk, E):
    Z = np.full((len(E), len(classes)), 0.5, np.float32)
    Z[:, hit] = Lk
    Zr = rank(Z)
    Xh = sc.transform(np.hstack([E, Lk]))
    Xc = scf.transform(E)
    P = np.full((len(E), len(classes)), 0.5)
    for j in range(len(classes)):
        if j in models:
            p = rank(models[j].predict_proba(Xh)[:, 1])
            P[:, j] = 0.5 * p + 0.5 * Zr[:, j] if hit[j] else p
        elif hit[j]:
            P[:, j] = Zr[:, j]
        elif j in cold_models:
            P[:, j] = rank(cold_models[j].predict_proba(Xc)[:, 1])
    return P


def combine(P_perch, S, files, starts):
    B = blend(P_perch, S, W_SED) if MODE == "blend" else rank(S)
    return smooth(B, np.array(files), np.array(starts))


# ---- test -------------------------------------------------------------------------
test_dir = f"{COMP}/test_soundscapes"
files = sorted(f for f in os.listdir(test_dir) if f.endswith((".ogg", ".wav", ".flac")))
step(f"{len(files)} test soundscapes")

if not files:
    # Public copy of test_soundscapes is empty; exercise both branches on the
    # labelled windows so the scored rerun is not the first run of this path.
    t_sed = time.time()
    S = np.concatenate([sed.run(to_windows(read(f"{COMP}/train_soundscapes/{fn}"), g.start_s.tolist()))
                        for fn, g in lab.groupby("filename", sort=False)])
    sed_ms = (time.time() - t_sed) / len(S) * 1000
    step(f"SED at {sed_ms:.0f} ms/window -> 600 test files: Perch {600 * 12 * perch_ms / 60000:.0f} "
         f"+ SED {600 * 12 * sed_ms / 60000:.0f} min, projected total "
         f"{(time.time() - t0) / 60 + 600 * 12 * (perch_ms + sed_ms) / 60000:.0f} min")
    assert S.shape == (len(Etr), len(classes)) and np.isfinite(S).all()
    assert not np.allclose(S.std(axis=0), 0), "every SED column constant"
    out = combine(predict_perch(Ltr, Etr), S, lab_files, lab_starts)
    assert out.shape == S.shape and np.isfinite(out).all()
    step(f"smoke test on {len(S)} labelled segments passed ({MODE})")
    sub = pd.read_csv(f"{COMP}/sample_submission.csv")
    sub.to_csv(f"{WORK}/submission.csv", index=False)
    step(f"no test audio; wrote sample_submission ({len(sub)} rows)")
    sys.exit(0)

row_ids, Lte, Ete, Ste, te_files, te_starts = [], [], [], [], [], []
t_sed = 0.0
for n, fn in enumerate(files):
    y = read(f"{test_dir}/{fn}")
    stem = os.path.splitext(fn)[0]
    starts = list(range(0, len(y) // SR_IN, WIN))
    if not starts:
        continue
    W = to_windows(y, starts)
    lo, em = embed(W)
    t = time.time()
    Ste.append(sed.run(W))
    t_sed += time.time() - t
    Lte.append(lo)
    Ete.append(em)
    row_ids += [f"{stem}_{s + WIN}" for s in starts]
    te_files += [stem] * len(starts)
    te_starts += starts
    if n % 25 == 0:
        step(f"  {n}/{len(files)} files, {len(row_ids)} windows")
Lte, Ete, Ste = np.concatenate(Lte), np.concatenate(Ete), np.concatenate(Ste)
step(f"embedded {len(row_ids):,} windows; SED {t_sed / len(row_ids) * 1000:.0f} ms/window")
assert len(Ste) == len(Lte) == len(Ete) == len(row_ids), "Perch and SED rows misaligned"

P = combine(predict_perch(Lte, Ete), Ste, te_files, te_starts)
sub = pd.DataFrame(P, columns=classes)
sub.insert(0, "row_id", row_ids)
sub.to_csv(f"{WORK}/submission.csv", index=False)

assert sub.shape[1] == len(classes) + 1, sub.shape
assert sub.row_id.is_unique and np.isfinite(P).all()
print(sub.iloc[:3, :4].to_string())
step(f"wrote submission.csv: {sub.shape[0]:,} rows x {sub.shape[1]} cols ({MODE})")
