"""Compact peak index over train.parquet, for spectral library search.

Read one row group at a time. train.parquet is 2.9 GB on disk but 6.4 GB as Arrow,
and pandas turns each spectrum's two peak arrays into separate numpy objects, so
loading it whole peaks near 18 GB and locks up a 15 GB machine. A row group is
0.31 GB, and only the trimmed top-64 peaks are kept, so peak memory stays ~1 GB.

Filtered to the region the test set occupies, with the same curation the test
spectra already had applied, so both sides of a similarity are treated alike.

    python3 library.py        -> data/index.npz
"""
import os
import resource
import sys

import numpy as np
import pyarrow.parquet as pq

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = f"{DIR}/data"
SHARDS = f"{DATA}/index"     # one npz per row group: bounded memory, and resumable

PEAKS = 64           # top-N most intense peaks kept per spectrum
FLOOR = 0.005        # relative intensity floor, as a fraction of the base peak
MZ_LO, MZ_HI = 150.0, 1250.0
PPM_MAX = 10.0       # |precursor_error_ppm| above this is a label-quality problem
MIN_PEAKS = 3

# The ten adducts the hidden test uses. 85.7% of train is already one of these;
# the rest are mostly dimers ([2M+H]+ and friends) that cannot appear in test.
TEST_ADDUCTS = {"[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+",
                "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"}

COLS = ["precursor_mz", "precursor_error_ppm", "adduct", "ionization_mode",
        "inchikey14", "normalized_smiles", "ingest_lib",
        "ms2_mzs", "ms2_normalized_intensities"]


# getrusage reports ru_maxrss in kilobytes on Linux and in bytes on macOS. Reading it
# raw makes every memory figure in this repo wrong by 1024x on one of the two.
_RSS_UNIT = 1 if sys.platform == "darwin" else 1024


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_UNIT / 2 ** 30


def flat(table, name):
    """(values, offsets) of a list<double> column, as numpy views over Arrow buffers."""
    arr = table.column(name).combine_chunks()
    return (np.asarray(arr.values), np.asarray(arr.offsets, dtype=np.int64))


def col(table, name):
    return table.column(name).to_numpy(zero_copy_only=False)


def trim(mzs, ints, precursor):
    """Curate one spectrum the way the test spectra were curated, then keep top-N.

    Peaks above the precursor cannot be fragments of it, low peaks are mostly
    instrument noise, and sqrt on intensity stops the base peak dominating every
    cosine it takes part in.
    """
    mzs = np.asarray(mzs, np.float32)
    ints = np.asarray(ints, np.float32)
    ok = (mzs <= precursor + 2.0) & (ints >= FLOOR)
    mzs, ints = mzs[ok], ints[ok]
    if len(mzs) > PEAKS:
        top = np.argpartition(-ints, PEAKS)[:PEAKS]
        mzs, ints = mzs[top], ints[top]
    order = np.argsort(mzs)
    return mzs[order], np.sqrt(ints[order])


def build():
    pf = pq.ParquetFile(f"{DATA}/train.parquet")
    print(f"{pf.metadata.num_rows:,} spectra in {pf.metadata.num_row_groups} row groups")

    os.makedirs(SHARDS, exist_ok=True)
    n_seen = n_kept = 0
    for g in range(pf.metadata.num_row_groups):
        out = f"{SHARDS}/rg_{g:02d}.npz"
        if os.path.exists(out):
            n_kept += int(np.load(out)["npk"].shape[0])
            continue
        t = pf.read_row_group(g, columns=COLS)
        n_seen += t.num_rows

        pm = t.column("precursor_mz").to_numpy()
        ppm = np.abs(np.nan_to_num(t.column("precursor_error_ppm").to_numpy(
            zero_copy_only=False), nan=0.0))
        ad = t.column("adduct").to_pylist()
        keep = np.where((pm >= MZ_LO) & (pm <= MZ_HI) & (ppm <= PPM_MAX)
                        & np.array([a in TEST_ADDUCTS for a in ad]))[0]

        if len(keep):
            # Read the list columns through Arrow's flat value buffer and offsets.
            # to_pylist() would box every one of ~20M doubles into a Python float
            # and cost gigabytes; this is a view over the buffer.
            a_mz, a_in = flat(t, "ms2_mzs"), flat(t, "ms2_normalized_intensities")
            mz = np.zeros((len(keep), PEAKS), np.float32)
            it = np.zeros((len(keep), PEAKS), np.float16)
            npk = np.zeros(len(keep), np.int16)
            for i, r in enumerate(keep):
                m, v = trim(a_mz[0][a_mz[1][r]:a_mz[1][r + 1]],
                            a_in[0][a_in[1][r]:a_in[1][r + 1]], pm[r])
                npk[i] = len(m)
                mz[i, :len(m)] = m
                it[i, :len(m)] = v
            ok = npk >= MIN_PEAKS
            sel = keep[ok]
            # Fixed-width bytes, not object arrays: 1.9M Python str objects would
            # cost several hundred MB on their own.
            np.savez(out, mz=mz[ok], it=it[ok], npk=npk[ok],
                     precursor=pm[sel].astype(np.float32),
                     ion=col(t, "ionization_mode")[sel] == "positive",
                     inchikey14=col(t, "inchikey14")[sel].astype("S14"),
                     smiles=col(t, "normalized_smiles")[sel].astype("S300"),
                     lib=col(t, "ingest_lib")[sel].astype("S24"))
            n_kept += int(ok.sum())
            del a_mz, a_in, mz, it
        del t
        print(f"  row group {g + 1:2d}/{pf.metadata.num_row_groups}  "
              f"{n_seen:,} seen  {n_kept:,} kept  peak RSS {rss_gb():.1f} GB", flush=True)

    print(f"\n{SHARDS}/: {n_kept:,} spectra, peak RSS {rss_gb():.1f} GB")


def load():
    """Concatenate the shards. About 1.3 GB for the full index."""
    import glob
    files = sorted(glob.glob(f"{SHARDS}/rg_*.npz"))
    assert files, f"no shards in {SHARDS}; run `python3 library.py` first"
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


if __name__ == "__main__":
    build()
