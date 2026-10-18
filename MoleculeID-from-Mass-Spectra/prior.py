"""A popularity prior over candidates, from COCONUT's own metadata.

Retrieval loses on the leaderboard to isomers of the answer inside its 5 ppm window
(RESULTS.md sections 10-11), and the spectrum alone separates them badly. Molecules
that end up measured as reference standards are the documented ones: many synonyms,
a CAS number, literature. COCONUT (CC0) records that per entry, so a candidate's
documentation is a feature the spectrum does not carry.

data/coconut_meta/meta.npz is built by data/jobs/coconut_meta.py from the COCONUT
release CSV: one row per InChIKey14, counts of synonyms / organisms / collections /
DOIs, CAS and name flags, annotation level and NP-likeness.
"""
import os

import numpy as np

import library

META = os.environ.get("PRIOR_META", f"{library.WORK}/coconut_meta/meta.npz")
PRIOR = os.environ.get("PRIOR", "")   # feature name, e.g. "syn"; "" turns the term off
# FILL is the value of a candidate COCONUT does not hold. Those are almost all index
# structures, i.e. molecules with reference spectra, so documented by definition.
FILL = float(os.environ.get("PRIOR_FILL", 0))
# W weighs the prior in submit(); GATE skips molecules whose best spectral match (mean
# cosine per spectrum) reaches it, i.e. class 1, where the prior only costs (RESULTS.md 28).
W = float(os.environ.get("PRIOR_W", 0))
GATE = float(os.environ.get("PRIOR_GATE", 9))
COUNTS = {"syn": "n_syn", "org": "n_org", "coll": "n_coll", "doi": "n_doi"}
FLAGS = {"cas": "has_cas", "name": "has_name", "annot": "annot", "npl": "npl"}


def load(feature=PRIOR, path=META):
    """{key14 bytes: value}. Counts are log1p: a molecule with 6,807 synonyms is not
    6,807 times likelier than one with a single name. Keys missing from COCONUT get
    nothing, i.e. 0, the value of an undocumented entry."""
    if not feature:
        return {}
    with np.load(path) as d:
        key = d["key"]
        if feature in COUNTS:
            v = np.log1p(d[COUNTS[feature]].astype(np.float64))
        else:
            v = d[FLAGS[feature]].astype(np.float64)
    return dict(zip(key.tolist(), v.tolist()))
