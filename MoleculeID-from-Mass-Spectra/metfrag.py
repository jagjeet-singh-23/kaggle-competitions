"""MetFrag-lite: how much of a spectrum a candidate's own bond cleavages can explain.

The candidates a molecule's spectrum cannot tell apart are mostly isomers inside its
5 ppm window (RESULTS.md sections 10-12). An isomer has the answer's formula, so
`formula.explained`, which only asks whether a peak is *some* sub-formula of the
ion, gives it the same score: that check reached an AUC of 0.6 and no more. Breaking
a candidate's actual bonds asks a sharper question -- is each peak a piece this
particular structure can fall into? -- and two isomers fall apart differently.

prvsiyan's analog-propagation ledger measured this as a ranker feature: +0.021 on the
leaderboard (0.245 -> 0.266), with one or two bonds broken and the fragment allowed
to gain or lose up to two hydrogens. This is the same idea built here, as a feature
for the learned ranker (docs/superpowers/specs/2026-10-20-raasta-b-design.md,
component 2), not a re-scorer of its own.

    python3 metfrag.py --selftest
"""
import sys

import numpy as np

from analog import ADDUCTS, E
from formula import MASS

MAX_CUTS = 2         # bonds broken per fragment
MAX_BONDS = 64       # above this, single cuts only: pairs grow with the square
H_SHIFTS = (-2, -1, 0, 1, 2)   # hydrogens a fragment ion may gain or lose
TOL_DA = 0.01        # as search.py and shifted.py match peaks
H = MASS["H"]
PROTON = H - E
# Charge carriers a fragment may keep besides the proton, per precursor adduct.
CARRIERS = {"[M+Na]+": MASS["Na"] - E, "[M+K]+": MASS["K"] - E,
            "[M+NH4]+": MASS["N"] + 4 * H - E, "[M+Cl]-": MASS["Cl"] + E,
            "[M+CH2O2-H]-": MASS["C"] + H + 2 * MASS["O"] + E}


def _graph(smiles):
    """(atom masses with their implicit H, adjacency [(neighbour, bond index)], bond count)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    w = []
    for a in m.GetAtoms():
        s = a.GetSymbol()
        if s not in MASS:
            return None
        w.append(MASS[s] + a.GetTotalNumHs() * H)
    adj = [[] for _ in w]
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append((j, b.GetIdx()))
        adj[j].append((i, b.GetIdx()))
    return np.array(w), adj, m.GetNumBonds()


def _split(w, adj, skip):
    """Masses of the pieces the graph falls into without bond `skip`, and, for every
    bridge left in it, the two sides that cutting that bridge as well would make.

    One iterative DFS (Tarjan's bridges with subtree mass sums): a second cut only
    separates anything if it is a bridge of what the first cut left, so this finds
    every two-bond fragment through bond `skip` exactly, without trying all pairs.
    """
    n = len(w)
    disc, low, sub = [-1] * n, [0] * n, list(w)
    out, t = [], 0
    for root in range(n):
        if disc[root] >= 0:
            continue
        disc[root] = low[root] = t
        t += 1
        cut = []                                    # subtree masses below this tree's bridges
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            v, via, it = stack[-1]
            for u, e in it:
                if e == skip or e == via:
                    continue
                if disc[u] < 0:
                    disc[u] = low[u] = t
                    t += 1
                    stack.append((u, e, iter(adj[u])))
                    break
                low[v] = min(low[v], disc[u])
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[v])
                    sub[p] += sub[v]
                    if low[v] > disc[p]:
                        cut.append(sub[v])
        whole = sub[root]
        out.append(whole)
        for s in cut:
            out += (s, whole - s)
    return out


def fragment_masses(smiles):
    """Sorted unique neutral masses of the molecule and of every piece one or two bond
    cleavages make; single cleavages only above MAX_BONDS. [] if RDKit cannot parse
    the SMILES or an element has no mass in formula.MASS."""
    g = _graph(smiles)
    if g is None:
        return np.zeros(0)
    w, adj, nb = g
    # The whole molecule, then its connected pieces (a salt has two) and single cuts.
    masses = [w.sum()] + _split(w, adj, -1)
    if MAX_CUTS >= 2 and nb <= MAX_BONDS:
        for i in range(nb):
            masses += _split(w, adj, i)
    return np.unique(np.round(np.array(masses, np.float64), 6))


def explain(frag, mzs, ints, adduct):
    """Fraction of the spectrum's intensity at a fragment ion: a fragment mass, plus or
    minus up to two hydrogens, carrying a proton (or the adduct's own ion) within
    TOL_DA. Intensities are weighed as given, i.e. sqrt-scaled as library.trim
    stores them."""
    frag = np.asarray(frag, np.float64)
    mzs, ints = np.asarray(mzs, np.float64), np.asarray(ints, np.float64)
    tot = ints.sum()
    if not len(frag) or not len(mzs) or tot <= 0:
        return 0.0
    positive = ADDUCTS[adduct][1] if adduct in ADDUCTS else adduct.endswith("+")
    carriers = [PROTON if positive else -PROTON]
    if adduct in CARRIERS:
        carriers.append(CARRIERS[adduct])
    ions = np.sort((frag[:, None, None] + np.array(H_SHIFTS)[None, :, None] * H
                    + np.array(carriers)[None, None, :]).ravel())
    i = np.clip(np.searchsorted(ions, mzs), 1, len(ions) - 1)
    d = np.minimum(np.abs(ions[i] - mzs), np.abs(ions[i - 1] - mzs))
    return float(ints[d <= TOL_DA].sum() / tot)


def _brute(smiles):
    """Every one- and two-bond removal, components by flood fill: the definition
    fragment_masses must agree with, at a cost quadratic in the bonds."""
    w, adj, nb = _graph(smiles)
    out = [w.sum()]
    # No cut at all: a salt's organic ion is a piece too (acetate of sodium acetate).
    cuts = [set()] + [{i} for i in range(nb)]
    if MAX_CUTS >= 2:
        cuts += [{i, j} for i in range(nb) for j in range(i + 1, nb)]
    for drop in cuts:
        seen = [False] * len(w)
        for s in range(len(w)):
            if seen[s]:
                continue
            seen[s], stack, m = True, [s], 0.0
            while stack:
                v = stack.pop()
                m += w[v]
                for u, e in adj[v]:
                    if e not in drop and not seen[u]:
                        seen[u] = True
                        stack.append(u)
            out.append(m)
    return np.unique(np.round(np.array(out), 6))


def selftest():
    from analog import E
    from formula import MASS
    MASS_H = MASS["H"]
    f = fragment_masses("CCO")                               # ethanol: CH3, CH2OH, C2H5, OH, CH2, whole
    for m in (15.02348, 31.01839, 29.03913, 17.00274, 14.01565, 46.04186):
        assert np.abs(f - m).min() < 1e-4, m
    ion = 31.01839 + MASS_H - E                              # protonated CH2OH, dh = 0
    assert explain(f, [ion], [1.0], "[M+H]+") == 1.0
    assert explain(f, [ion, 500.0], [1.0, 1.0], "[M+H]+") == 0.5
    assert explain(f, [ion - 2 * MASS_H], [1.0], "[M+H]+") == 1.0   # dh = -2 is allowed
    assert explain(f, [500.0], [1.0], "[M+H]+") == 0.0
    assert explain(f, [31.01839 - MASS_H + E], [1.0], "[M-H]-") == 1.0
    assert len(fragment_masses("not a smiles")) == 0
    assert len(fragment_masses("[Au]")) == 0
    assert explain(np.zeros(0), [100.0], [1.0], "[M+H]+") == 0.0
    long = fragment_masses("C" * 70)                         # 69 bonds > MAX_BONDS: single cuts only
    assert 1 < len(long) <= 2 * 69 + 1
    exact()
    print("metfrag self-checks passed")


def exact():
    """The bridge-based enumeration equals brute force on rings, fused rings, a sugar
    and a salt, where one cut inside a ring separates nothing and two must."""
    for smi in ("Cn1cnc2c1c(=O)n(C)c(=O)n2C",                  # caffeine
                "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",       # glucose
                "c1ccc2cc3ccccc3cc2c1",                         # anthracene
                "CC(=O)[O-].[Na+]", "C1CC1C2CC2"):
        got, want = fragment_masses(smi), _brute(smi)
        assert len(got) == len(want) and np.allclose(got, want, atol=1e-6), smi


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
