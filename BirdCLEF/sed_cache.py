"""Pilot audio cache for the SED branch: int16 windows, with targets kept per teacher.

Decoding ogg inside a GPU training loop starves the GPU, so every window the pilot
trains on is decoded once here and stored as int16 arrays.

The pseudo-label windows are chosen from row_id and a seed alone, never from any
teacher's values. Phase B reuses exactly these windows and only swaps the targets
file (pseudo_targets_A.npy -> pseudo_targets_B.npy), which is what makes A vs B a
one-variable comparison.

The 66 labelled files never contribute pseudo rows (their fold_id is not -1); they
contribute only their true labels.

    python3 sed_cache.py            -> $SED_CACHE (default: the SSD jobs dir)
    python3 sed_cache.py --selftest
"""
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BIRDCLEF_DATA", f"{DIR}/data")
PSEUDO_CSV = os.environ.get(
    "SED_PSEUDO_CSV",
    "/Volumes/Extreme SSD/kaggle/BirdCLEF/external/raw-pseudos/pseudo_labels_v2s_oof_raw.csv")
OUT = os.environ.get("SED_CACHE", "/Volumes/Extreme SSD/kaggle/BirdCLEF/jobs/sed-cache")

N_PSEUDO = 12_000     # pilot: ~9% of the 127,104 unlabelled windows
PER_CLASS = 30        # focal clips per primary label
MAX_SEC = 15.0        # first 15 s of each focal clip; training crops 5 s inside it
SECONDARY = 0.5       # target weight of a secondary label
SEED = 0
SR, WIN_SAMPLES = 32_000, 160_000


def to_int16(y):
    return np.round(np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)


def select_pseudo(csv_path, n, seed):
    """n unlabelled windows, chosen by row_id and seed only, sorted for reading."""
    ids = pd.read_csv(csv_path, usecols=["row_id", "fold_id"])
    pool = np.sort(ids.row_id[ids.fold_id == -1].to_numpy())
    pick = pool[np.sort(np.random.default_rng(seed).choice(len(pool), n, replace=False))]
    df = pd.DataFrame({"row_id": pick})
    parts = df.row_id.str.rsplit("_", n=1, expand=True)
    df["filename"] = parts[0] + ".ogg"
    df["start_s"] = parts[1].astype(int) - 5
    return df.sort_values(["filename", "start_s"]).reset_index(drop=True)


def targets_for(csv_path, row_ids, classes):
    t = pd.read_csv(csv_path, usecols=["row_id"] + classes).set_index("row_id")
    return t.loc[row_ids, classes].to_numpy(np.float16)


def focal_items(train, classes, per_class, seed):
    """Up to per_class clips per primary label, with their in-taxonomy secondaries."""
    import ast

    col = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed)
    rows = []
    for label, g in train.groupby("primary_label", sort=True):
        take = g.iloc[np.sort(rng.choice(len(g), min(per_class, len(g)), replace=False))]
        for r in take.itertuples():
            sec = [col[s] for s in ast.literal_eval(r.secondary_labels) if s in col]
            rows.append((r.filename, col[label], sec))
    return pd.DataFrame(rows, columns=["filename", "label_idx", "secondary_idx"])


def _mono(path, frames=-1):
    y, sr = sf.read(path, dtype="float32", frames=frames)
    assert sr == SR, (path, sr)
    return y if y.ndim == 1 else y.mean(axis=1)


def _windows(path, meta, sub):
    """Write one int16 row per (filename, start_s) of meta, in meta's row order."""
    from perch import to_windows

    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16, shape=(len(meta), WIN_SAMPLES))
    for n, (fn, g) in enumerate(meta.groupby("filename", sort=False)):
        mm[g.index.to_numpy()] = to_int16(to_windows(_mono(f"{DATA}/{sub}/{fn}"), g.start_s.tolist()))
        if n % 1000 == 0:
            print(f"  {path.rsplit('/', 1)[1]}: {n:,} files", flush=True)
    mm.flush()


def main():
    import json

    from embed_labelled import segments

    os.makedirs(OUT, exist_ok=True)
    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    classes = tax.primary_label.tolist()
    col = {c: i for i, c in enumerate(classes)}

    p = select_pseudo(PSEUDO_CSV, N_PSEUDO, SEED)
    p.to_csv(f"{OUT}/pseudo_meta.csv", index=False)
    np.save(f"{OUT}/pseudo_targets_A.npy", targets_for(PSEUDO_CSV, p.row_id.tolist(), classes))
    _windows(f"{OUT}/pseudo_audio.npy", p, "train_soundscapes")

    lab = segments()[["filename", "start_s", "primary_label"]].reset_index(drop=True)
    Y = np.zeros((len(lab), len(classes)), np.float16)
    for i, lbl in enumerate(lab.primary_label.astype(str)):
        for s in lbl.split(";"):
            if s in col:
                Y[i, col[s]] = 1.0
    lab[["filename", "start_s"]].to_csv(f"{OUT}/labelled_meta.csv", index=False)
    np.save(f"{OUT}/labelled_targets.npy", Y)
    _windows(f"{OUT}/labelled_audio.npy", lab, "train_soundscapes")

    tr = pd.read_csv(f"{DATA}/train.csv", dtype={"primary_label": str})
    f = focal_items(tr, classes, PER_CLASS, SEED)
    cap = int(MAX_SEC * SR)
    f["length"] = [min(sf.info(f"{DATA}/train_audio/{fn}").frames, cap) for fn in f.filename]
    f["offset"] = np.concatenate([[0], np.cumsum(f.length.to_numpy())[:-1]])
    mm = np.lib.format.open_memmap(f"{OUT}/focal_audio.npy", mode="w+", dtype=np.int16,
                                   shape=(int(f.length.sum()),))
    T = np.zeros((len(f), len(classes)), np.float16)
    for i, r in enumerate(f.itertuples()):
        y = _mono(f"{DATA}/train_audio/{r.filename}", frames=r.length)
        mm[r.offset:r.offset + len(y)] = to_int16(y)
        T[i, r.secondary_idx] = SECONDARY
        T[i, r.label_idx] = 1.0
        if i % 1000 == 0:
            print(f"  focal_audio.npy: {i:,} clips", flush=True)
    mm.flush()
    f[["filename", "offset", "length"]].to_csv(f"{OUT}/focal_index.csv", index=False)
    np.save(f"{OUT}/focal_targets.npy", T)

    with open(f"{OUT}/dataset-metadata.json", "w") as fh:
        json.dump({"title": "BirdCLEF 2026 SED cache", "id": "jagjeetsingh23/birdclef-sed-cache",
                   "licenses": [{"name": "other"}]}, fh)
    print(f"pseudo {len(p):,}  labelled {len(lab)}  focal clips {len(f):,} "
          f"({f.length.sum() / SR / 3600:.1f} h)")
    for name in sorted(os.listdir(OUT)):
        print(f"  {name:28s} {os.path.getsize(f'{OUT}/{name}') / 1e9:7.3f} GB")


def selftest():
    from embed_labelled import segments

    tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype={"primary_label": str})
    classes = tax.primary_label.tolist()
    ids = pd.read_csv(PSEUDO_CSV, usecols=["row_id", "fold_id"])
    unlabelled = set(ids.row_id[ids.fold_id == -1])
    labelled_files = set(segments().filename)

    a, b = select_pseudo(PSEUDO_CSV, 500, 0), select_pseudo(PSEUDO_CSV, 500, 0)
    assert a.equals(b), "selection must be deterministic"
    assert list(a.columns) == ["row_id", "filename", "start_s"], list(a.columns)
    assert len(a) == 500 and set(a.row_id) <= unlabelled
    assert not set(a.filename) & labelled_files, "a labelled file leaked into pseudo rows"
    assert not a.equals(select_pseudo(PSEUDO_CSV, 500, 1)), "the seed must matter"

    some = a.row_id.tolist()[:3]
    raw = pd.read_csv(PSEUDO_CSV, nrows=200_000, usecols=["row_id"] + classes).set_index("row_id")
    t = targets_for(PSEUDO_CSV, some, classes)
    assert t.dtype == np.float16 and t.shape == (3, 234)
    assert np.allclose(t, raw.loc[some, classes].to_numpy(), atol=1e-3)

    assert to_int16(np.array([0.5, -1.0, 1.2])).tolist() == [16384, -32767, 32767]
    fn = a.filename.iloc[0]
    y, _ = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", frames=160_000)
    assert np.abs(to_int16(y) / 32767 - y).max() <= 1 / 32767

    tr = pd.read_csv(f"{DATA}/train.csv", dtype={"primary_label": str})
    f = focal_items(tr, classes, per_class=30, seed=0)
    assert f.groupby("label_idx").size().max() <= 30
    assert f.equals(focal_items(tr, classes, per_class=30, seed=0))
    print("sed_cache self-checks passed")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
