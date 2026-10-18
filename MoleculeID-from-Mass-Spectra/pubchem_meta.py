"""PubChem popularity per CID -> data/pubchem_meta/counts.npz.

A popularity prior needs a documentation count for every candidate, and a popularity-
selected PubChem pool needs one for every compound before it is parsed. PubChem keeps
two on one scale for all of them: synonyms (CID-Synonym-filtered) and PubMed articles
(CID-PMID). Both are keyed by CID, which CID-SMILES shares, so pubchem.py can filter
on them line by line without computing a single InChIKey first.

The two files are streamed from PubChem's FTP (public domain) and never written to
disk. The Mac's line is ~1 MB/s, so this is ~22 minutes of download.

    ../.venv/bin/python pubchem_meta.py
"""
import gzip
import os
import time
import urllib.request

import numpy as np

FTP = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras"
OUT = "data/pubchem_meta/counts.npz"
MAX_CID = 250_000_000          # PubChem CIDs are below 200M
CAP = np.iinfo(np.uint16).max   # counts saturate here; the prior takes log1p anyway


def lines(name):
    with urllib.request.urlopen(f"{FTP}/{name}") as r, gzip.open(r, "rt", encoding="utf-8",
                                                                 errors="replace") as f:
        yield from f


def counts(name):
    """Rows per CID, saturating at CAP. Both files list one (CID, item) pair per line."""
    c = np.zeros(MAX_CID, np.uint16)
    t0, n = time.time(), 0
    for line in lines(name):
        cid = int(line[:line.index("\t")])
        if c[cid] < CAP:
            c[cid] += 1
        n += 1
        if n % 20_000_000 == 0:
            print(f"  {name}: {n:,} lines, {time.time() - t0:.0f} s", flush=True)
    print(f"{name}: {n:,} lines, {int((c > 0).sum()):,} CIDs, {time.time() - t0:.0f} s", flush=True)
    return c


def load(path=OUT):
    """(n_syn, n_pmid) as dense uint16 arrays indexed by CID."""
    with np.load(path) as d:
        n = int(d["cid"].max()) + 1
        syn, pmid = np.zeros(n, np.uint16), np.zeros(n, np.uint16)
        syn[d["cid"]], pmid[d["cid"]] = d["n_syn"], d["n_pmid"]
    return syn, pmid


def main():
    syn, pmid = counts("CID-Synonym-filtered.gz"), counts("CID-PMID.gz")
    cid = np.flatnonzero((syn > 0) | (pmid > 0)).astype(np.int32)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, cid=cid, n_syn=syn[cid], n_pmid=pmid[cid])
    s = syn[cid]
    print(f"wrote {OUT}: {len(cid):,} CIDs; with >=1/2/5/20 synonyms: "
          + " / ".join(f"{int((s >= t).sum()):,}" for t in (1, 2, 5, 20)), flush=True)


if __name__ == "__main__":
    main()
