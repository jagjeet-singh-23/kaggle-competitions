"""BirdCLEF+ 2026 submission kernel: BirdNET embeddings, a head trained in-kernel.

Kernels-only competition, CPU, no internet, 90 minutes. Nothing is pre-trained and
shipped: the head is fitted here on the 739 labelled soundscape segments, which are
part of the competition data, so what runs is exactly what the local CV measured.

    embed 739 labelled segments   ~40 s
    fit 234 one-vs-rest heads     ~10 s
    embed + predict the test set  dominates; ~48 ms per 5 s window

Local 5-fold CV, grouped by recording file (see ../RESULTS.md):

    zero-shot BirdNET               0.6168
    head on emb+logits              0.8774
    head + zero-shot, birds only    0.8881   <- this

The blend is applied only to the 157 species BirdNET has an output column for.
Mixing its logits into the other 77 is dilution: locally that costs 0.066.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

t0 = time.time()
step = lambda s: print(f"[{(time.time() - t0) / 60:5.1f} min] {s}", flush=True)

print("/kaggle/input contains:", flush=True)
for dirpath, _, files in os.walk("/kaggle/input"):
    if dirpath.count("/") - 2 <= 2:
        print(f"  {dirpath}  ({len(files)} files)", flush=True)


def find(marker, root="/kaggle/input"):
    for dirpath, _, files in os.walk(root):
        if marker in files:
            return dirpath
    raise FileNotFoundError(f"{marker} not found under {root}")


COMP = find("taxonomy.csv")
CODE = find("birdnet.py")
ASSETS = find("BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")

# ai-edge-litert is not in the Kaggle image and there is no internet, so the wheel
# rides along in the assets dataset.
try:
    import ai_edge_litert  # noqa: F401
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index",
                           "--find-links", ASSETS, "--no-deps",
                           "ai-edge-litert==2.2.0"])
    step("installed ai-edge-litert offline")

os.environ["BIRDCLEF_DATA"] = COMP
os.environ["BIRDNET_DIR"] = ASSETS
sys.path.insert(0, CODE)

import soundfile as sf                                            # noqa: E402
from birdnet import BirdNET, to_chunks, fold, species_map, WIN    # noqa: E402
from cv import zero_shot                                          # noqa: E402
from embed_labelled import segments                               # noqa: E402
from sklearn.linear_model import LogisticRegression               # noqa: E402
from sklearn.preprocessing import StandardScaler                  # noqa: E402

step(f"comp={COMP}  code={CODE}  assets={ASSETS}")

tax = pd.read_csv(f"{COMP}/taxonomy.csv", dtype={"primary_label": str})
classes = tax.primary_label.tolist()
col = {c: i for i, c in enumerate(classes)}
idx = species_map(tax)
keep = idx[idx >= 0]                       # the 157 BirdNET columns we feed the head

net = BirdNET(batch=16)

# ---- training data: the only labelled in-domain audio in the competition ----------
lab = segments()
L, E, rows = [], [], []
for fn, grp in lab.groupby("filename", sort=False):
    y, _ = sf.read(f"{COMP}/train_soundscapes/{fn}", dtype="float32")
    starts = grp.start_s.tolist()
    lo, em = fold(*net.run(to_chunks(y, starts)))
    L.append(lo)
    E.append(em)
    rows += list(grp.primary_label.astype(str))
Ltr, Etr = np.concatenate(L), np.concatenate(E)
Y = np.zeros((len(rows), len(classes)), np.float32)
for i, lbl in enumerate(rows):
    for s in lbl.split(";"):
        if s in col:
            Y[i, col[s]] = 1.0
step(f"labelled: {Etr.shape} emb, {int((Y.sum(0) > 0).sum())} classes with positives")

Xtr = np.hstack([Etr, Ltr[:, keep]])
sc = StandardScaler().fit(Xtr)
Xtr = sc.transform(Xtr)
has_pos = Y.sum(0) > 0
models = {}
for j in np.where(has_pos)[0]:
    models[j] = LogisticRegression(C=0.01, max_iter=2000,
                                   class_weight="balanced").fit(Xtr, Y[:, j])
step(f"fitted {len(models)} heads")

# ---- test ------------------------------------------------------------------------
test_dir = f"{COMP}/test_soundscapes"
files = sorted(f for f in os.listdir(test_dir) if f.endswith((".ogg", ".wav", ".flac")))
step(f"{len(files)} test soundscapes")

if not files:
    # The public copy of test_soundscapes holds only a readme; the real files appear
    # when the notebook is rerun against the hidden set. Emit the sample so the
    # submission is still valid and the rerun is what actually scores.
    sub = pd.read_csv(f"{COMP}/sample_submission.csv")
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    step(f"no test audio; wrote sample_submission ({len(sub)} rows)")
    sys.exit(0)

row_ids, Lte, Ete = [], [], []
for n, fn in enumerate(files):
    y, _ = sf.read(f"{test_dir}/{fn}", dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    stem = os.path.splitext(fn)[0]
    starts = list(range(0, len(y) // 32_000, WIN))
    if not starts:
        continue
    lo, em = fold(*net.run(to_chunks(y, starts)))
    Lte.append(lo)
    Ete.append(em)
    row_ids += [f"{stem}_{s + WIN}" for s in starts]
    if n % 25 == 0:
        step(f"  {n}/{len(files)} files, {len(row_ids)} windows")
Lte, Ete = np.concatenate(Lte), np.concatenate(Ete)
step(f"embedded {len(row_ids):,} windows")

Z, hit = zero_shot(Lte, tax)
Xte = sc.transform(np.hstack([Ete, Lte[:, keep]]))

# ---- cold-start heads for the classes with neither route ------------------------
# 30 classes have no BirdNET column and no labelled positive, so they would emit a
# 0.5 constant. That is 13% of a macro-averaged metric and accounted for essentially
# the whole 0.8881 -> 0.83575 gap on the first submission. All of them do have focal
# audio, and a focal-only head scores 0.7385 against 0.5 on a scoreable proxy cohort
# of the same shape (coldstart.py). 30 segments per class beat 150 and 400.
#
# This is deliberately NOT applied to the 129 classes that fall back to zero-shot:
# measured on the mapped-and-labelled cohort, zero-shot is 0.8128 against a focal
# head's 0.7809, and blending the two gains nothing.
fc = np.load(f"{ASSETS}/focal_cap30.npz")
Xf, yf = fc["emb"].astype(np.float32), fc["label"]
scf = StandardScaler().fit(Xf)          # focal heads are embedding-only, so their own
Xf_s, Xte_f = scf.transform(Xf), scf.transform(Ete)
cold = [j for j in range(len(classes))
        if not has_pos[j] and not hit[j] and (yf == j).sum() > 0]
cold_models = {}
for j in cold:
    cold_models[j] = LogisticRegression(C=0.01, max_iter=2000,
                                        class_weight="balanced").fit(Xf_s, (yf == j).astype(int))
step(f"fitted {len(cold_models)} cold-start heads on focal audio")

rank = lambda v: np.argsort(np.argsort(v)) / max(1, len(v) - 1)
P = np.full((len(row_ids), len(classes)), 0.5)
sources = {"head": 0, "zero-shot": 0, "cold-start": 0, "constant": 0}
for j in range(len(classes)):
    sources["head" if j in models else
            "zero-shot" if hit[j] else
            "cold-start" if j in cold_models else "constant"] += 1
    if j in models:
        p = rank(models[j].predict_proba(Xte)[:, 1])
        # Blend zero-shot only where BirdNET actually has an output column for the
        # species; elsewhere its logits are a constant and blending is pure dilution.
        P[:, j] = 0.5 * p + 0.5 * rank(Z[:, j]) if hit[j] else p
    elif hit[j]:
        # No labelled positive to train on, but BirdNET knows this species: its
        # zero-shot logits beat the constant a head would otherwise emit.
        P[:, j] = rank(Z[:, j])
    elif j in cold_models:
        P[:, j] = rank(cold_models[j].predict_proba(Xte_f)[:, 1])
step("predicted by source: " + ", ".join(f"{k} {v}" for k, v in sources.items()))
assert sources["constant"] == 0, f"{sources['constant']} classes still emit 0.5"

sub = pd.DataFrame(P, columns=classes)
sub.insert(0, "row_id", row_ids)
sub.to_csv("/kaggle/working/submission.csv", index=False)

assert sub.shape[1] == len(classes) + 1, sub.shape
assert sub.row_id.is_unique and np.isfinite(P).all()
print(sub.iloc[:3, :4].to_string())
step(f"wrote submission.csv: {sub.shape[0]:,} rows x {sub.shape[1]} cols")
