"""Phase B teacher: the Perch kernel's pipeline, run over the unlabelled soundscapes.

    python3 teacher.py --selftest
    python3 teacher.py --embed      Perch over every unlabelled soundscape -> shards
"""
import os
import sys

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
OUT = os.environ.get("TEACHER_OUT", "/Volumes/Extreme SSD/kaggle/BirdCLEF/jobs/perch-unlabelled")
SHARD = 250           # files per shard


def unlabelled_files():
    """Every train soundscape outside the 66 labelled files, sorted."""
    from embed_labelled import segments

    labelled = set(segments().filename)
    return sorted(f for f in os.listdir(f"{DATA}/train_soundscapes")
                  if f.endswith(".ogg") and not f.startswith("._") and f not in labelled)


def read(path):
    import soundfile as sf

    y, _ = sf.read(path, dtype="float32")
    return y if y.ndim == 1 else y.mean(axis=1)


def embed():
    """Perch over every 5 s window of every unlabelled soundscape, the kernel's way.

    One shard per SHARD files, written to a temp name and renamed, so a run cut
    short by a power cut resumes at the first missing shard."""
    import time

    import pandas as pd

    from perch import Perch, SR_IN, WIN, species_map, to_windows

    os.makedirs(OUT, exist_ok=True)
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    idx = species_map(tax)
    keep = idx[idx >= 0]
    files = unlabelled_files()
    net = Perch(batch=16)
    for k in range(0, len(files), SHARD):
        path = f"{OUT}/shard_{k // SHARD:03d}.npz"
        if os.path.exists(path):
            continue
        t0 = time.time()
        names, starts, L, E = [], [], [], []
        for fn in files[k:k + SHARD]:
            y = read(f"{DATA}/train_soundscapes/{fn}")
            s = list(range(0, len(y) // SR_IN, WIN))
            lo, em = net.run(to_windows(y, s))
            L.append(lo[:, keep])
            E.append(em)
            names += [fn] * len(s)
            starts += s
        tmp = path.replace(".npz", ".tmp.npz")
        np.savez(tmp, filename=np.array(names), start_s=np.array(starts),
                 logits=np.concatenate(L).astype(np.float32), emb=np.concatenate(E).astype(np.float32))
        os.replace(tmp, path)
        print(f"shard {k // SHARD:3d}: {len(names):,} windows, "
              f"{(time.time() - t0) / len(names) * 1000:.0f} ms/window", flush=True)
    print("embed done", flush=True)


# ---- kernel_perch's pipeline, as functions. Kept identical to the kernel (the
# selftest compares the two on real files); the kernel itself is a script. -------

SMOOTH_K, SMOOTH_ALPHA = 6, 0.2


def rank(M):
    from scipy.stats import rankdata

    return (rankdata(M, axis=0) - 1) / max(1, len(M) - 1)


def labelled(net, keep, classes):
    """Perch on the 739 labelled windows, embedded as the kernel embeds them."""
    from embed_labelled import segments
    from perch import to_windows

    col = {c: i for i, c in enumerate(classes)}
    L, E, rows = [], [], []
    for fn, grp in segments().groupby("filename", sort=False):
        lo, em = net.run(to_windows(read(f"{DATA}/train_soundscapes/{fn}"), grp.start_s.tolist()))
        L.append(lo[:, keep])
        E.append(em)
        rows += list(grp.primary_label.astype(str))
    Y = np.zeros((len(rows), len(classes)), np.float32)
    for i, lbl in enumerate(rows):
        for s in lbl.split(";"):
            if s in col:
                Y[i, col[s]] = 1.0
    return np.concatenate(L), np.concatenate(E), Y


def fit(Ltr, Etr, Y, Xf, yf, hit):
    """The kernel's soundscape heads and cold-start heads."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    has_pos = Y.sum(0) > 0
    sc = StandardScaler().fit(np.hstack([Etr, Ltr]))
    Xtr = sc.transform(np.hstack([Etr, Ltr]))
    models = {j: LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(Xtr, Y[:, j])
              for j in np.where(has_pos)[0]}
    scf = StandardScaler().fit(Xf)
    Xf_s = scf.transform(Xf)
    cold = {j: LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")
            .fit(Xf_s, (yf == j).astype(int))
            for j in range(len(hit)) if not has_pos[j] and not hit[j] and (yf == j).sum() > 0}
    assert len(models) + int(hit[~has_pos].sum()) + len(cold) == len(hit), "a class has no source"
    return {"sc": sc, "models": models, "scf": scf, "cold": cold, "hit": hit}


def raw_scores(state, Lk, E):
    """Head and cold-start probabilities, NaN where a class has no such head.
    Row-wise, so it can run in chunks; the ranks are taken later, over all rows."""
    n, k = len(E), len(state["hit"])
    H, C = np.full((n, k), np.nan), np.full((n, k), np.nan)
    Xh = state["sc"].transform(np.hstack([E, Lk]))
    for j, m in state["models"].items():
        H[:, j] = m.predict_proba(Xh)[:, 1]
    Xc = state["scf"].transform(E)
    for j, m in state["cold"].items():
        C[:, j] = m.predict_proba(Xc)[:, 1]
    return H, C


def combine(state, Lk, H, C):
    """The kernel's predict(): ranked heads, blended with ranked zero-shot logits."""
    hit = state["hit"]
    Z = np.full((len(Lk), len(hit)), 0.5, np.float32)
    Z[:, hit] = Lk
    Zr = rank(Z)
    P = np.full(Z.shape, 0.5)
    for j in range(len(hit)):
        if j in state["models"]:
            p = rank(H[:, j])
            P[:, j] = 0.5 * p + 0.5 * Zr[:, j] if hit[j] else p
        elif hit[j]:
            P[:, j] = Zr[:, j]
        elif j in state["cold"]:
            P[:, j] = rank(C[:, j])
    return P


def smooth(P, files, starts):
    """The kernel's smoother. Files are grouped by one sort instead of a scan per
    file, which the kernel can afford on 600 files but not on 10,592."""
    Pr = rank(P)
    ctx = np.empty_like(Pr)
    order = np.lexsort((starts, files))
    cut = np.flatnonzero(files[order][1:] != files[order][:-1]) + 1
    for ix in np.split(order, cut):
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


def quantile_map(scores, ref):
    """Per column, ref's values re-assigned in the order of scores. Rows with tied
    scores share the mean of the values they would have taken."""
    out = np.empty(ref.shape, np.float64)
    for j in range(ref.shape[1]):
        col = np.empty(len(ref))
        col[np.argsort(scores[:, j], kind="stable")] = np.sort(ref[:, j])
        _, g = np.unique(scores[:, j], return_inverse=True)
        out[:, j] = (np.bincount(g, weights=col) / np.bincount(g))[g]
    return out


def _fidelity(files):
    """The teacher must score these files exactly as kernel_perch does."""
    import subprocess
    import tempfile

    import pandas as pd

    from perch import Perch, SR_IN, WIN, species_map, to_windows

    with tempfile.TemporaryDirectory() as d:
        # os.walk does not descend into symlinked directories, so every directory the
        # kernel's find() must reach is real, and only the files are links.
        comp, code, pdir, assets, work = (f"{d}/in/{p}" for p in
                                          ("comp", "code", "perch", "assets", "../work"))
        for p in (f"{comp}/test_soundscapes", code, pdir, assets, work):
            os.makedirs(p)
        for f in ("taxonomy.csv", "sample_submission.csv", "train_soundscapes_labels.csv"):
            os.symlink(f"{DATA}/{f}", f"{comp}/{f}")
        os.symlink(f"{DATA}/train_soundscapes", f"{comp}/train_soundscapes")
        for fn in files:
            os.symlink(f"{DATA}/train_soundscapes/{fn}", f"{comp}/test_soundscapes/{fn}")
        for f in ("perch.py", "embed_labelled.py", "birdnet.py", "cv.py"):
            os.symlink(f"{DIR}/{f}", f"{code}/{f}")
        for f in ("perch_v2.onnx", "labels.csv"):
            os.symlink(f"{DIR}/models/onnx_rishikeshjani/{f}", f"{pdir}/{f}")
        os.symlink(f"{DATA}/focal_cap30_perch.npz", f"{assets}/focal_cap30_perch.npz")
        env = dict(os.environ, KAGGLE_INPUT=f"{d}/in", KAGGLE_WORKING=work)
        subprocess.run([sys.executable, f"{DIR}/kernel_perch/birdclef_perch_submit.py"],
                       env=env, check=True, stdout=subprocess.DEVNULL)
        want = pd.read_csv(f"{work}/submission.csv")

    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    idx = species_map(tax)
    keep = idx[idx >= 0]
    net = Perch(batch=16)
    Ltr, Etr, Y = labelled(net, keep, tax.primary_label.tolist())
    fc = np.load(f"{DATA}/focal_cap30_perch.npz")
    state = fit(Ltr, Etr, Y, fc["emb"].astype(np.float32), fc["label"], idx >= 0)
    L, E, rows, stems, starts = [], [], [], [], []
    for fn in files:
        y = read(f"{DATA}/train_soundscapes/{fn}")
        s = list(range(0, len(y) // SR_IN, WIN))
        lo, em = net.run(to_windows(y, s))
        L.append(lo[:, keep])
        E.append(em)
        stem = os.path.splitext(fn)[0]
        rows += [f"{stem}_{t + WIN}" for t in s]
        stems += [stem] * len(s)
        starts += s
    Lk, E = np.concatenate(L), np.concatenate(E)
    H, C = raw_scores(state, Lk, E)
    P = smooth(combine(state, Lk, H, C), np.array(stems), np.array(starts))
    want = want.set_index("row_id").loc[rows]
    return np.abs(P - want.to_numpy()).max()


def selftest():
    from embed_labelled import segments

    files = unlabelled_files()
    assert len(files) == 10_592, len(files)
    assert files == sorted(files)
    assert not set(files) & set(segments().filename), "a labelled file is in the unlabelled pool"

    rng = np.random.default_rng(0)
    scores, ref = rng.random((40, 3)), rng.random((40, 3))
    scores[[3, 9, 17, 25, 31], 1] = 0.5
    out = quantile_map(scores, ref)
    for j in range(3):
        o = np.argsort(scores[:, j], kind="stable")
        assert (np.diff(out[o, j]) >= -1e-12).all(), "values must follow the scores' order"
        assert np.isclose(out[:, j].sum(), ref[:, j].sum()), "values must be ref's, re-assigned"
    assert np.allclose(np.sort(out[:, 0]), np.sort(ref[:, 0]))
    tied = out[[3, 9, 17, 25, 31], 1]
    assert np.allclose(tied, tied[0]), "tied scores must get equal values"

    diff = _fidelity(files[:20])
    assert diff < 1e-6, diff
    print(f"teacher self-checks passed (teacher vs kernel_perch on 20 files: max diff {diff:.1e})")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--embed" in sys.argv:
        embed()
