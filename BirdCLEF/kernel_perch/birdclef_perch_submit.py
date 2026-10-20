"""BirdCLEF+ 2026 submission kernel: kernel v4's pipeline on Perch v2 instead of BirdNET.

Kernels-only competition, CPU, no internet, 90 minutes. The head is fitted here on
the 739 labelled soundscape segments, so what runs is what the local CV measured.

    embed 739 labelled segments   ~1-2 min
    fit heads                     seconds
    embed + predict the test set  dominates; one Perch call per 5 s window

Local CV, 10 seeded group folds, paired on identical folds (perch_head.py):

    kernel v4 on BirdNET            0.8880
    the same on Perch v2            0.9144   +0.0264, 10/10 seeds, 52/75 classes

Perch has a logit column for 203 of the 234 classes (BirdNET: 157), so the classes
left with neither a column nor a labelled positive fall from 30 to 2. Those two get a
cold-start head on focal audio, as in v3.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

t0 = time.time()
step = lambda s: print(f"[{(time.time() - t0) / 60:5.1f} min] {s}", flush=True)

# Overridable so the whole kernel can be run locally against a fake input tree.
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORKING", "/kaggle/working")

print(f"{INPUT} contains:", flush=True)
for dirpath, _, files in os.walk(INPUT):
    if dirpath.count("/") - INPUT.count("/") <= 2:
        print(f"  {dirpath}  ({len(files)} files)", flush=True)


def find(marker, root=INPUT):
    for dirpath, _, files in os.walk(root):
        if marker in files:
            return dirpath
    raise FileNotFoundError(f"{marker} not found under {root}")


COMP = find("taxonomy.csv")
CODE = find("perch.py")
PERCH = find("perch_v2.onnx")
ASSETS = find("focal_cap30_perch.npz")

# The ONNX file comes from a public dataset. This is the size of the copy that was
# checked against Google's SavedModel (embeddings within 3.8e-6); a changed file
# would be an unverified model.
assert os.path.getsize(f"{PERCH}/perch_v2.onnx") == 409_148_616, "perch_v2.onnx changed"

# onnxruntime is not in the Kaggle image and there is no internet. The ONNX dataset
# ships a cp312 wheel, but the image moved to Python 3.13, so the assets dataset
# carries the cp313 wheel and its pure-python deps; pip picks whichever fits.
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
from perch import Perch, to_windows, species_map, WIN, SR_IN        # noqa: E402
from embed_labelled import segments                                 # noqa: E402
from scipy.stats import rankdata                                    # noqa: E402
from sklearn.linear_model import LogisticRegression                 # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402

step(f"comp={COMP}  code={CODE}  perch={PERCH}  assets={ASSETS}")

tax = pd.read_csv(f"{COMP}/taxonomy.csv", dtype={"primary_label": str})
classes = tax.primary_label.tolist()
col = {c: i for i, c in enumerate(classes)}
idx = species_map(tax)
hit = idx >= 0                       # a property of the taxonomy, not of any audio
keep = idx[hit]                      # the 203 Perch columns; only these are kept
assert hit.sum() == 203, hit.sum()

net = Perch(batch=16)


def embed(y, starts):
    lo, em = net.run(to_windows(y, starts))
    return lo[:, keep], em


# ---- training data: the only labelled in-domain audio in the competition ----------
lab = segments()
L, E, rows, lab_files, lab_starts = [], [], [], [], []
t_emb = time.time()
for fn, grp in lab.groupby("filename", sort=False):
    y, _ = sf.read(f"{COMP}/train_soundscapes/{fn}", dtype="float32")
    starts = grp.start_s.tolist()
    lo, em = embed(y, starts)
    L.append(lo)
    E.append(em)
    rows += list(grp.primary_label.astype(str))
    lab_files += [fn] * len(starts)
    lab_starts += starts
Ltr, Etr = np.concatenate(L), np.concatenate(E)
ms = (time.time() - t_emb) / len(Etr) * 1000
step(f"labelled: {Etr.shape} emb at {ms:.0f} ms/window -> ~600 test files "
     f"would take {600 * 12 * ms / 60000:.0f} min")
Y = np.zeros((len(rows), len(classes)), np.float32)
for i, lbl in enumerate(rows):
    for s in lbl.split(";"):
        if s in col:
            Y[i, col[s]] = 1.0
has_pos = Y.sum(0) > 0
step(f"{int(has_pos.sum())} classes with labelled positives")

sc = StandardScaler().fit(np.hstack([Etr, Ltr]))
Xtr = sc.transform(np.hstack([Etr, Ltr]))
models = {}
for j in np.where(has_pos)[0]:
    models[j] = LogisticRegression(C=0.01, max_iter=2000,
                                   class_weight="balanced").fit(Xtr, Y[:, j])
step(f"fitted {len(models)} soundscape heads")

# ---- cold-start heads for the classes with neither route --------------------------
# Under BirdNET these were 30 classes and worth +0.021 private. Under Perch only 2
# remain; the rest of the focal pool is their negatives.
fc = np.load(f"{ASSETS}/focal_cap30_perch.npz")
Xf, yf = fc["emb"].astype(np.float32), fc["label"]
scf = StandardScaler().fit(Xf)
Xf_s = scf.transform(Xf)
cold_models = {}
for j in range(len(classes)):
    if not has_pos[j] and not hit[j] and (yf == j).sum() > 0:
        cold_models[j] = LogisticRegression(
            C=0.01, max_iter=2000,
            class_weight="balanced").fit(Xf_s, (yf == j).astype(int))
step(f"fitted {len(cold_models)} cold-start heads on focal audio")

sources = {"head": 0, "zero-shot": 0, "cold-start": 0, "constant": 0}
for j in range(len(classes)):
    sources["head" if j in models else "zero-shot" if hit[j]
            else "cold-start" if j in cold_models else "constant"] += 1
step("prediction sources: " + ", ".join(f"{k} {v}" for k, v in sources.items()))
assert sources["constant"] == 0, f"{sources['constant']} classes would emit 0.5"

# Tie-aware ranks. argsort-of-argsort broke ties in platform-dependent order, which
# moved the local blend score by 0.02 between x86 and ARM (head.py selftest).
rank = lambda M: (rankdata(M, axis=0) - 1) / max(1, len(M) - 1)

SMOOTH_K = 6         # +/- windows of context; 12-window recordings make this the file
SMOOTH_ALPHA = 0.2   # weight kept on the window's own evidence


def smooth(P, files, starts):
    """Blend each window's scores with the strongest score nearby in the same file.

    Kernel v4's smoother, +0.013 private on BirdNET. Files are handled separately and
    the row order is restored, so this never mixes two recordings.
    """
    Pr = rank(P)
    ctx = np.empty_like(Pr)
    order = np.lexsort((starts, files))
    for f in np.unique(files):
        ix = order[files[order] == f]
        Q, n = Pr[ix], len(ix)
        acc = []
        for off in range(-SMOOTH_K, SMOOTH_K + 1):
            # Counted explicitly: when a file is shorter than the window, a computed
            # end index goes negative and numpy reads it as "from the end".
            src = max(0, off)
            cnt = max(0, min(n, n + off) - src)
            pad = np.full_like(Q, np.nan)
            if cnt:
                dst = max(0, -off)
                pad[dst:dst + cnt] = Q[src:src + cnt]
            acc.append(pad)
        ctx[ix] = np.nanmax(np.stack(acc), 0)
    return SMOOTH_ALPHA * Pr + (1 - SMOOTH_ALPHA) * rank(ctx)


def predict(Lk, E):
    """(mapped logits, embeddings) -> (n, 234) scores. Every class gets a real one."""
    Z = np.full((len(E), len(classes)), 0.5, np.float32)
    Z[:, hit] = Lk
    Zr = rank(Z)
    Xh = sc.transform(np.hstack([E, Lk]))
    Xc = scf.transform(E)
    P = np.full((len(E), len(classes)), 0.5)
    for j in range(len(classes)):
        if j in models:
            p = rank(models[j].predict_proba(Xh)[:, 1])
            # Blend zero-shot only where Perch has an output column for the class.
            P[:, j] = 0.5 * p + 0.5 * Zr[:, j] if hit[j] else p
        elif hit[j]:
            P[:, j] = Zr[:, j]
        elif j in cold_models:
            P[:, j] = rank(cold_models[j].predict_proba(Xc)[:, 1])
    return P


# ---- test -------------------------------------------------------------------------
test_dir = f"{COMP}/test_soundscapes"
files = sorted(f for f in os.listdir(test_dir) if f.endswith((".ogg", ".wav", ".flac")))
step(f"{len(files)} test soundscapes")

if not files:
    # The public copy of test_soundscapes holds only a readme; the real files appear
    # when Kaggle reruns this against the hidden set. Exercise predict() on the
    # labelled embeddings anyway, so the scored rerun is not the first time it runs.
    P = predict(Ltr, Etr)
    S = smooth(P, np.array(lab_files), np.array(lab_starts))
    assert P.shape == (len(Etr), len(classes)) and np.isfinite(P).all()
    assert S.shape == P.shape and np.isfinite(S).all()
    assert not np.allclose(P.std(axis=0), 0), "every column constant"
    step(f"smoke test on {len(Etr)} labelled segments passed, smoothing ok: {P.shape}")
    sub = pd.read_csv(f"{COMP}/sample_submission.csv")
    sub.to_csv(f"{WORK}/submission.csv", index=False)
    step(f"no test audio; wrote sample_submission ({len(sub)} rows)")
    sys.exit(0)

row_ids, Lte, Ete, te_files, te_starts = [], [], [], [], []
for n, fn in enumerate(files):
    y, sr = sf.read(f"{test_dir}/{fn}", dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    stem = os.path.splitext(fn)[0]
    starts = list(range(0, len(y) // SR_IN, WIN))
    if not starts:
        continue
    lo, em = embed(y, starts)
    Lte.append(lo)
    Ete.append(em)
    row_ids += [f"{stem}_{s + WIN}" for s in starts]
    te_files += [stem] * len(starts)
    te_starts += starts
    if n % 25 == 0:
        step(f"  {n}/{len(files)} files, {len(row_ids)} windows")
Lte, Ete = np.concatenate(Lte), np.concatenate(Ete)
step(f"embedded {len(row_ids):,} windows")

P = smooth(predict(Lte, Ete), np.array(te_files), np.array(te_starts))
step("applied temporal smoothing")
sub = pd.DataFrame(P, columns=classes)
sub.insert(0, "row_id", row_ids)
sub.to_csv(f"{WORK}/submission.csv", index=False)

assert sub.shape[1] == len(classes) + 1, sub.shape
assert sub.row_id.is_unique and np.isfinite(P).all()
print(sub.iloc[:3, :4].to_string())
step(f"wrote submission.csv: {sub.shape[0]:,} rows x {sub.shape[1]} cols")
