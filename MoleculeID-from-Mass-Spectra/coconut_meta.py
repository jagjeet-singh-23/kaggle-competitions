"""COCONUT metadata per InChIKey14 -> data/coconut_meta/meta.npz (popularity-prior features).
    ../.venv/bin/python coconut_meta.py data/coconut_meta/coconut_csv-10-2026.zip
"""
import sys
import zipfile

import numpy as np
import pandas as pd

src = sys.argv[1]
cols = ["standard_inchi_key", "name", "annotation_level", "np_likeness", "organisms",
        "collections", "dois", "synonyms", "cas"]
with zipfile.ZipFile(src) as z, z.open(z.namelist()[0]) as f:
    df = pd.read_csv(f, usecols=cols, dtype=str, keep_default_na=False)
print(len(df), "rows")


def count(s):
    return s.map(lambda x: 0 if x in ("", "None") else x.count("|") + 1).astype(np.int32)


out = pd.DataFrame({
    "key": df.standard_inchi_key.str[:14],
    "n_syn": count(df.synonyms), "n_org": count(df.organisms), "n_coll": count(df.collections),
    "n_doi": count(df.dois),
    "has_cas": (~df.cas.isin(["", "None"])).astype(np.int8),
    "has_name": (~df.name.isin(["", "None"])).astype(np.int8),
    "annot": pd.to_numeric(df.annotation_level, errors="coerce").fillna(0).astype(np.int8),
    "npl": pd.to_numeric(df.np_likeness, errors="coerce").fillna(0).astype(np.float32),
})
out = out[out.key.str.len() == 14]
# Stereo isomers share a key14: keep the most documented entry's numbers.
agg = out.groupby("key").agg({"n_syn": "max", "n_org": "max", "n_coll": "max", "n_doi": "max",
                              "has_cas": "max", "has_name": "max", "annot": "max", "npl": "mean"})
print(len(agg), "key14s")
print(agg.describe().T[["mean", "50%", "max"]])
np.savez("data/coconut_meta/meta.npz", key=agg.index.values.astype("S14"),
         **{c: agg[c].values for c in agg.columns})
