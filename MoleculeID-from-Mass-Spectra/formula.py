"""How much of a spectrum can a candidate's formula explain?

RESULTS.md section 12: a perfect formula filter is worth about +0.02, and 84-87% of
the candidates in a 5 ppm window have the wrong formula. A fragment is a piece of
the precursor ion, so its formula must be a sub-formula of the ion's. A candidate
whose formula cannot account for the big peaks is the wrong formula, and that can
be checked without predicting the formula at all.

    explained(smiles, mzs, ints, precursor, adducts) -> fraction of intensity in [0, 1]
"""
import functools
import os

import numpy as np

from analog import ADDUCTS, E

MASS = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
        "P": 30.97376163, "S": 31.97207100, "F": 18.99840322, "Cl": 34.96885268,
        "Br": 78.9183371, "I": 126.904473, "Na": 22.9897692809, "K": 38.96370668,
        "Si": 27.9769265325, "B": 11.0093054, "Se": 79.9165213}
# What each adduct adds to (+) or takes from (-) the neutral molecule.
COMP = {"[M+H]+": {"H": 1}, "[M+NH4]+": {"N": 1, "H": 4},
        "[M-H2O+H]+": {"H": -1, "O": -1}, "[M-2H2O+H]+": {"H": -3, "O": -2},
        "[M+Na]+": {"Na": 1}, "[M+K]+": {"K": 1},
        "[M-H]-": {"H": -1}, "[M-H2O-H]-": {"H": -3, "O": -1},
        "[M+CH2O2-H]-": {"C": 1, "H": 1, "O": 2}, "[M+Cl]-": {"Cl": 1}}
TOL = float(os.environ.get("FRAG_TOL", 0.005))     # Da
MAX_SUB = 4_000_000                                 # sub-formula masses kept per ion


@functools.lru_cache(maxsize=200_000)
def counts(smiles):
    """(element counts incl. implicit H, neutral monoisotopic mass) or None."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    c = {}
    for a in m.GetAtoms():
        c[a.GetSymbol()] = c.get(a.GetSymbol(), 0) + 1
        c["H"] = c.get("H", 0) + a.GetTotalNumHs()
    if any(e not in MASS for e in c):
        return None
    return tuple(sorted(c.items())), sum(MASS[e] * n for e, n in c.items())


@functools.lru_cache(maxsize=512)
def sub_masses(ion):
    """Sorted masses of every sub-formula of `ion` (a sorted tuple of counts)."""
    m = np.zeros(1)
    # Biggest counts last, so the array only reaches full size at the final step.
    for e, n in sorted(ion, key=lambda t: t[1]):
        m = (m[:, None] + MASS[e] * np.arange(n + 1)).ravel()
        if len(m) > MAX_SUB:
            return None
    return np.unique(m)


def explained(smiles, mzs, ints, precursor, adducts):
    c = counts(smiles)
    if c is None:
        return 0.0
    comp, mass = c
    # The adduct is the one that puts this candidate's mass on the precursor.
    a = min(adducts, key=lambda a: abs(mass + ADDUCTS[a][0] - precursor))
    ion = dict(comp)
    for e, n in COMP[a].items():
        ion[e] = ion.get(e, 0) + n
    if any(n < 0 for n in ion.values()):
        return 0.0
    sub = sub_masses(tuple(sorted((e, n) for e, n in ion.items() if n)))
    if sub is None:
        return 0.0      # shortcut: formulas too big to enumerate get no evidence
    # A fragment ion of a cation weighs its sub-formula minus an electron.
    target = np.asarray(mzs, np.float64) + (E if ADDUCTS[a][1] else -E)
    i = np.clip(np.searchsorted(sub, target), 1, len(sub) - 1)
    d = np.minimum(np.abs(sub[i] - target), np.abs(sub[i - 1] - target))
    w = np.asarray(ints, np.float64)
    return float(w[d <= TOL].sum() / max(w.sum(), 1e-12))


def selftest():
    # Caffeine [M+H]+ 195.0877; 138.0662 is C6H8N3O+ (a real fragment). 150.5 is
    # nothing CHNO can be near once restricted to caffeine's atoms.
    caf = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    pos = [a for a, (_, p) in ADDUCTS.items() if p]
    f = explained(caf, [138.0662, 195.0877], [1.0, 1.0], 195.0877, pos)
    assert f == 1.0, f
    f = explained(caf, [138.0662, 150.5], [1.0, 3.0], 195.0877, pos)
    assert abs(f - 0.25) < 1e-9, f
    # A fragment heavier than anything in the ion is never explained.
    assert explained(caf, [400.0], [1.0], 195.0877, pos) == 0.0
    print("formula self-checks passed")


if __name__ == "__main__":
    selftest()
