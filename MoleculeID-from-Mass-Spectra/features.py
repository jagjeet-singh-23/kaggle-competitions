"""One molecule's candidates as a feature matrix, for the learned ranker.

v17 ranks a molecule's candidates by a hand-weighted sum: spectral score + analog
propagation + 32 x predicted-fingerprint Tanimoto + 16 x mass-shifted analogs + 0.5 x
COCONUT synonym prior. Weights picked on held-out data have repeatedly failed to
transfer to the leaderboard (RESULTS.md sections 15-16, v10, v11, v13), and public
pipelines learn the combination instead (docs/superpowers/specs/2026-10-20-raasta-b-design.md,
component 2). The ranker needs every term separately, plus how each compares with the
rest of the molecule's list, and it needs them computed exactly as v17 computes them
so the held-out tables and the kernel see the same numbers. This module is that one
place: `molecule_rows` builds the candidates in dilute.score's order, and its `v17`
column is v17's total.

    python3 features.py --selftest
"""
import sys

import numpy as np

import formula
import metfrag
import prior
import shifted
from analog import ADDUCTS

V17 = dict(gamma=32.0, beta=1.0, sw=16.0, p=3.0, prior_w=0.5, prior_gate=0.8)
N_RERANK = 100       # rows per molecule: v17's top candidates, the only ones re-ranked
GROUPS = {
    "v17": ("v17", "v17_rank", "v17_gap"),
    "spec": ("spec", "spec_gap", "has_spec", "prop"),
    "fp": ("fp", "fp_rank", "fp_gap", "fp_z"),
    "shift": ("shift3", "shift1", "shift_tan", "shift_top"),
    "prior": ("prior", "prior_rank"),
    "frag": ("frag", "frag_mean", "frag_gap"),
    "formula": ("formula",),
    "mass": ("ppm",),
    "pool": ("log_pool", "top_spec", "n_spectra", "in_pool", "internal"),
}
FEATURES = sum(GROUPS.values(), ())
PPM_MISSING = 50.0   # a candidate with no mass in `an` (a spectral match outside it)


def _rank(v):
    """0 for the molecule's best value, 1 for its worst; ties in list order."""
    r = np.empty(len(v))
    r[np.argsort(-v, kind="stable")] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def _z(v):
    s = v.std()
    return (v - v.mean()) / s if s >= 1e-9 else np.zeros_like(v)


def molecule_rows(an, bits, spec, smiles, pm, add, fpq, spectra, analogs, prior_of,
                  frags=None, cfg=V17):
    """(keys, SMILES, X) for one molecule's top N_RERANK candidates in v17's order.

    `total` is built with the same additions in the same order as dilute.score and
    fpsearch.submit -- spectral matches, then propagated analogs, then the pool -- so
    its float sums and the stable sort's tie order are v17's to the last bit. `frags`
    maps SMILES to metfrag.fragment_masses, computed by the caller in bulk; a missing
    entry is computed here.
    """
    c = cfg
    total, sm = dict(spec), dict(smiles)
    prop = an.propagate(spec, pm, add)
    for k, v in prop.items():
        total[k] = total.get(k, 0.0) + c["beta"] * v
        sm.setdefault(k, an.smiles[an.row[k]])
    pool = an.pool(pm, add)
    fp_of, s3, s1, s0 = {}, {}, {}, {}
    if len(pool):
        B = np.unpackbits(an.fp[pool], axis=1)[:, bits].astype(np.float32)
        inter = B @ fpq
        t = inter / np.maximum(B.sum(1) + fpq.sum() - inter, 1e-9)
        for r, v in zip(pool, t):
            k = an.key[r]
            fp_of[k] = float(v)
            if c["gamma"]:
                total[k] = total.get(k, 0.0) + c["gamma"] * float(v)
            sm.setdefault(k, an.smiles[r])
    known = [(an.row[a], s) for a, s in analogs if a in an.row]
    if known and len(pool):
        afp = an.fp[np.array([r for r, _ in known])]
        sims = np.array([s for _, s in known])
        t3 = shifted.analog_term(an.fp[pool], afp, sims, c["p"])
        t1 = shifted.analog_term(an.fp[pool], afp, sims, 1.0)
        t0 = shifted.analog_term(an.fp[pool], afp, sims, 0.0)
        for r, v3, v1, v0 in zip(pool, t3, t1, t0):
            k = an.key[r]
            s3[k], s1[k], s0[k] = float(v3), float(v1), float(v0)
            if c["sw"]:
                total[k] = total.get(k, 0.0) + c["sw"] * float(v3)
            sm.setdefault(k, an.smiles[r])
    n_sp = len(spectra)
    top_spec = max(spec.values(), default=0.0) / n_sp
    if c["prior_w"] and top_spec < c["prior_gate"]:
        for k in total:
            total[k] += c["prior_w"] * prior_of.get(k, prior.FILL)
    if not total:
        return [], [], np.zeros((0, len(FEATURES)), np.float32)

    top = sorted(total.items(), key=lambda kv: -kv[1])
    keys = [k for k, _ in top]
    col = {"v17": np.array([v for _, v in top])}
    col["spec"] = np.array([spec.get(k, 0.0) for k in keys]) / n_sp
    col["prop"] = np.array([prop.get(k, 0.0) for k in keys])
    col["fp"] = np.array([fp_of.get(k, 0.0) for k in keys])
    col["shift3"] = np.array([s3.get(k, 0.0) for k in keys])
    col["shift1"] = np.array([s1.get(k, 0.0) for k in keys])
    col["shift_tan"] = np.array([s0.get(k, 0.0) for k in keys])
    col["prior"] = np.array([prior_of.get(k, prior.FILL) for k in keys])
    for name in ("v17", "fp", "prior"):
        col[f"{name}_rank"] = _rank(col[name])
    for name in ("v17", "spec", "fp"):
        col[f"{name}_gap"] = col[name] - col[name].max()
    col["fp_z"] = _z(col["fp"])
    col["has_spec"] = (col["spec"] > 0).astype(np.float64)

    n = min(len(keys), N_RERANK)
    keys = keys[:n]
    col = {name: v[:n] for name, v in col.items()}
    smi = [sm[k].decode() if isinstance(sm[k], bytes) else str(sm[k]) for k in keys]
    rows = np.array([an.row[k] if k in an.row else -1 for k in keys])
    in_pool = set(np.asarray(pool).tolist())
    neutral = pm - ADDUCTS[add[0]][0]
    col["in_pool"] = np.array([float(r in in_pool) for r in rows])
    col["internal"] = ((rows >= 0) & (rows < an.n_internal)).astype(np.float64)
    col["ppm"] = np.array([abs(neutral - float(an.mass[r])) / neutral * 1e6 if r >= 0
                           else PPM_MISSING for r in rows])
    frags = frags if frags is not None else {}
    fr = np.zeros((n, n_sp))
    fo = np.zeros((n, n_sp))
    for i, s in enumerate(smi):
        f = frags[s] if s in frags else metfrag.fragment_masses(s)
        for j, (mz, it, p) in enumerate(spectra):
            fr[i, j] = metfrag.explain(f, mz, it, add[0])
            fo[i, j] = formula.explained(s, mz, it, p, add)
    col["frag"], col["frag_mean"], col["formula"] = fr.max(1), fr.mean(1), fo.mean(1)
    col["frag_gap"] = col["frag"] - col["frag"].max()
    col["shift_top"] = np.full(n, max((s for _, s in known), default=0.0))
    col["log_pool"] = np.full(n, np.log1p(len(pool)))
    col["top_spec"] = np.full(n, top_spec)
    col["n_spectra"] = np.full(n, float(n_sp))
    X = np.column_stack([col[name] for name in FEATURES]).astype(np.float32)
    return [bytes(k) for k in keys], smi, X


def selftest():
    F = {n: i for i, n in enumerate(FEATURES)}
    an = _Stub()
    bits = np.arange(2048)
    fpq = np.zeros(2048, np.float32)
    fpq[[0, 1]] = 1.0
    pm = 47.04914                                    # ethanol [M+H]+
    sp = [(np.array([29.0386, 31.0178]), np.array([1.0, 0.5]), pm)]
    analogs = [(b"B", 0.5)]
    prior_of = {b"C": 2.0, b"D": 1.0}
    keys, smi, X = molecule_rows(an, bits, {b"A": 0.9, b"B": 0.3}, _Stub.SPEC_SMILES, pm, ["[M+H]+"],
                                 fpq, sp, analogs, prior_of)
    assert X.shape == (4, len(FEATURES)) and np.isfinite(X).all()
    # By hand. B: spec 0.3 + 32 x T 1 + 16 x 0.5^3 x T(B, B) 1. C: prop 0.2 + 32 x 1/3 +
    # 16 x 0.125 x 1/3. A: spec 0.9 only, outside the pool. D: prop 0.9 only, which ties A,
    # and A was inserted first (spec before propagate). top_spec 0.9 >= 0.8: no prior.
    assert list(keys) == [b"B", b"C", b"A", b"D"], keys
    assert np.allclose(X[:, F["v17"]], [34.3, 11.5333333, 0.9, 0.9]), X[:, F["v17"]]
    assert X[0, F["v17_rank"]] == 0 and X[0, F["v17_gap"]] == 0
    assert smi == ["CCN", "CCC", "CCO", "COC"], smi   # spectral matches keep the library's SMILES
    assert np.allclose(X[:, F["prior"]], [0, 2.0, 0, 1.0])   # filled even where v17 leaves it out
    assert np.allclose(X[:, F["in_pool"]], [1, 1, 0, 1]) and np.allclose(X[:, F["internal"]], [1, 0, 1, 0])
    assert np.allclose(X[:, F["shift3"]], [0.125, 0.125 / 3, 0, 0])
    assert np.allclose(X[:, F["spec"]], [0.3, 0, 0.9, 0]) and np.allclose(X[:, F["has_spec"]], [1, 0, 1, 0])
    assert np.allclose(X[:, F["top_spec"]], 0.9) and np.allclose(X[:, F["n_spectra"]], 1)
    assert X[2, F["ppm"]] < 1 and X[1, F["ppm"]] > 1   # A is ethanol's own mass; C is off by design
    # top_spec 0.7 < 0.8: the prior joins v17, and D (0.9 + 0.5) now passes A (0.7 + 0).
    keys, _, X = molecule_rows(an, bits, {b"A": 0.7, b"B": 0.3}, _Stub.SPEC_SMILES, pm, ["[M+H]+"],
                               fpq, sp, analogs, prior_of)
    assert list(keys) == [b"B", b"C", b"D", b"A"], keys
    assert np.allclose(X[:, F["v17"]], [34.3, 12.5333333, 1.4, 0.7])
    # One candidate: every within-molecule feature is 0 and finite.
    keys, _, X = molecule_rows(an, bits, {b"A": 0.9}, _Stub.SPEC_SMILES, 999.0, ["[M+H]+"], fpq, sp, [], {})
    assert keys == [b"A"] and np.isfinite(X).all()
    for c in ("v17_rank", "v17_gap", "fp_z", "fp_rank", "spec_gap", "frag_gap", "prior_rank"):
        assert X[0, F[c]] == 0, c
    # Nothing at all: no spectral match, empty pool.
    keys, smi, X = molecule_rows(an, bits, {}, {}, 999.0, ["[M+H]+"], fpq, sp, [], {})
    assert keys == [] and smi == [] and X.shape == (0, len(FEATURES))
    # 150 candidates come back as v17's top N_RERANK.
    big = _Stub(150)
    global N_RERANK
    keep, N_RERANK = N_RERANK, 1000
    all_keys, _, all_X = molecule_rows(big, bits, {}, {}, pm, ["[M+H]+"], fpq, sp, [], {})
    N_RERANK = keep
    keys, _, X = molecule_rows(big, bits, {}, {}, pm, ["[M+H]+"], fpq, sp, [], {})
    assert len(all_keys) == 150 and len(keys) == 100 and keys == all_keys[:100]
    # Within-molecule statistics of the cheap terms use the whole list, so the cut does
    # not move them; frag is computed on the kept rows only, so frag_gap is not compared.
    cols = [i for i, n in enumerate(FEATURES) if n != "frag_gap"]
    assert np.array_equal(X[:, cols], all_X[:100][:, cols])
    assert X[:, F["frag_gap"]].max() == 0 and (X[:, F["frag_gap"]] <= 0).all()
    assert sorted(sum(GROUPS.values(), ())) == sorted(FEATURES)
    print("features self-checks passed")


class _Stub:
    """A four-structure Analog: A and B are index structures, C and D external; the
    5 ppm pool around ethanol holds B, C and D. With n, a pool of n alkanes instead."""
    SPEC_SMILES = {b"A": b"CCO", b"B": b"CCN"}

    def __init__(self, n=None):
        def fp(on):
            b = np.zeros(2048, np.uint8)
            b[list(on)] = 1
            return np.packbits(b)
        if n is None:
            self.key = np.array([b"A", b"B", b"C", b"D"], "S14")
            self.smiles = np.array([b"OCC", b"NCC", b"CCC", b"COC"])
            self.mass = np.array([46.04186, 45.05785, 46.5, 46.04186])
            self.fp = np.array([fp([5]), fp([0, 1]), fp([0, 2]), fp([2, 3])])
            self.n_internal, self._pool, self._prop = 2, np.array([1, 2, 3]), {b"C": 0.2, b"D": 0.9}
        else:
            self.key = np.array([f"K{i:03d}".encode() for i in range(n)], "S14")
            self.smiles = np.array([b"C" * (1 + i % 6) + b"O" for i in range(n)])
            self.mass = np.full(n, 46.04186)
            self.fp = np.array([fp([0] + [2 + i % 40, 50 + i % 7]) for i in range(n)])
            self.n_internal, self._pool, self._prop = 0, np.arange(n), {}
        self.row = {k: i for i, k in enumerate(self.key.tolist())}

    def pool(self, pm, add):
        return self._pool if pm < 900 else np.zeros(0, int)

    def propagate(self, spec, pm, add):
        return dict(self._prop) if pm < 900 else {}


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
