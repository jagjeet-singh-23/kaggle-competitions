"""Mass-shifted analog search: find the query's relatives in the library, not only its isomers.

`analog.py` seeds analog propagation from the best spectral matches inside the query's
own ±10 ppm precursor window, so every seed is an isomer of the answer. A molecule
with no spectra anywhere usually has a relative that does -- one methyl, hydroxyl or
sugar away -- at a different mass. Its fragments line up with the query's in one of
two ways: directly, where a fragment keeps the shared core and loses the modified
part, or shifted by the neutral-mass difference, where the fragment carries the
modification. Scoring both and taking the larger finds relatives a same-mass search
cannot see.

The public analog-propagation baseline measured this as the largest single step of
its ledger: library-only 0.151 -> 0.245 on the leaderboard with a ranker on top.
This repo never had it (docs/superpowers/specs/2026-10-20-raasta-b-design.md).

Intensities arrive sqrt-scaled, as `library.trim` stores them. Entropy similarity
works on the original intensities, so it squares them back first; cosine uses them
as they are, like `search.py`.

    python3 shifted.py --selftest
"""
import sys

import numpy as np

TOL_DA = 0.01        # peaks match when they share a 0.01 Da bin, as in search.py


def entropy_p(it_sqrt):
    """Intensity weights for entropy similarity, row-wise over a 1-D or 2-D array.

    Squares the stored sqrt intensities back, normalises each spectrum to sum 1, and
    applies the entropy weighting of Li et al. (2021): a low-entropy spectrum (S < 3),
    dominated by a few peaks, is flattened by p ** (0.25 + 0.25 S) so its minor peaks
    still count. Padding zeros stay zero.
    """
    a = np.asarray(it_sqrt, np.float64)
    one = a.ndim == 1
    a = np.atleast_2d(a) ** 2
    p = _rows_to_one(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = -np.where(p > 0, p * np.log(p), 0.0).sum(1)
    w = np.where(S < 3.0, 0.25 + 0.25 * S, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(p > 0, p ** w[:, None], 0.0)
    p = _rows_to_one(p).astype(np.float32)
    return p[0] if one else p


def _rows_to_one(a):
    tot = a.sum(1, keepdims=True)
    return np.divide(a, tot, out=np.zeros_like(a), where=tot > 0)


def _binned(mz, w):
    """Peaks merged per TOL_DA bin: (sorted unique bins, summed weights)."""
    b = np.rint(np.asarray(mz, np.float64) / TOL_DA).astype(np.int64)
    u, inv = np.unique(b, return_inverse=True)
    out = np.zeros(len(u))
    np.add.at(out, inv, w)
    return u, out


def _sim(q, r, shift, kind):
    (qb, qv), (rb, rv) = q, _binned(r[0] + shift, r[1])
    _, iq, ir = np.intersect1d(qb, rb, assume_unique=True, return_indices=True)
    a, b = qv[iq], rv[ir]
    if kind == "entropy":
        # Entropy similarity reduces to a sum over matched peaks: an unmatched peak
        # contributes exactly what the normalisation by ln 4 cancels.
        return float(((a + b) * np.log(a + b) - a * np.log(a) - b * np.log(b)).sum()
                     / np.log(4.0))
    return float((a * b).sum())


def pair_sim(q_mz, q_it, r_mz, r_it, shift=0.0, kind="entropy"):
    """Similarity of one query and one reference spectrum, the larger of direct and
    shifted (the reference's m/z plus `shift`). The reference implementation that the
    vectorised index search is checked against."""
    q_mz, q_it = np.asarray(q_mz, np.float64), np.asarray(q_it, np.float64)
    r_mz, r_it = np.asarray(r_mz, np.float64), np.asarray(r_it, np.float64)
    if kind == "entropy":
        qw, rw = entropy_p(q_it).astype(np.float64), entropy_p(r_it).astype(np.float64)
    elif kind == "cos":
        qw = q_it / max(np.linalg.norm(q_it), 1e-12)
        rw = r_it / max(np.linalg.norm(r_it), 1e-12)
    else:
        raise ValueError(f"kind={kind!r}")
    qk, rk = qw > 0, rw > 0
    if not qk.any() or not rk.any():
        return 0.0
    q = _binned(q_mz[qk], qw[qk])
    r = (r_mz[rk], rw[rk])
    s = _sim(q, r, 0.0, kind)
    return max(s, _sim(q, r, float(shift), kind)) if shift else s


WINDOW = 200.0       # Da either side of the query's neutral mass (the public baseline's)
N_ANALOGS = 80       # relatives kept per query molecule
CHUNK = 20_000       # reference rows per gather, ~10 MB of float64 per array
# The test is timsTOF; a relative measured on the same instrument fragments the way the
# query did, so its spectrum represents the structure when there is one.
TIMS_LIBS = (b"enveda-180", b"enveda-np-examples")


class ShiftIndex:
    """One representative spectrum per (structure, polarity), sorted by neutral mass.

    The search scores every representative within WINDOW of the query's neutral mass,
    directly and shifted by the mass difference, in numpy: the query is scattered onto
    the 0.01 Da bin axis once and each reference row gathers from it, as
    `search.Searcher.score` does for the same-mass library search.
    """

    def __init__(self, ix, drop=None, banned=None, masses=None):
        from search import BINS
        self.bins_n = BINS
        n = len(ix["npk"])
        mask = np.ones(n, bool)
        if drop is not None:
            mask[np.asarray(drop, np.int64)] = False     # held-out spectra leave the index
        if banned is not None and len(banned):
            mask &= ~np.isin(ix["inchikey14"], np.asarray(banned))
        if masses is None:
            import analog
            with np.load(analog.DESC) as d:
                masses = (d["key"], d["mass"])
        mkeys, mvals = np.asarray(masses[0]), np.asarray(masses[1], np.float64)
        o = np.argsort(mkeys, kind="stable")
        mkeys, mvals = mkeys[o], mvals[o]
        keys = ix["inchikey14"]
        at = np.clip(np.searchsorted(mkeys, keys), 0, len(mkeys) - 1)
        nm = np.where(mkeys[at] == keys, mvals[at], -1.0)
        mask &= nm > 0                                     # unparseable or unknown mass

        rows = np.where(mask)[0]
        tims = np.isin(ix["lib"][rows], np.array(TIMS_LIBS, ix["lib"].dtype))
        pol = ix["ion"][rows]
        # Best representative first within each (key, polarity): timsTOF, then the
        # richest spectrum, then the lowest row so the choice is deterministic.
        o = np.lexsort((rows, -ix["npk"][rows].astype(np.int64), ~tims, pol, keys[rows]))
        rows, pol = rows[o], pol[o]
        k = keys[rows]
        first = np.ones(len(rows), bool)
        first[1:] = (k[1:] != k[:-1]) | (pol[1:] != pol[:-1])
        rows, pol = rows[first], pol[first]

        self.side = {}
        for positive in (True, False):
            r = rows[pol == positive]
            r = r[np.argsort(nm[r], kind="stable")]
            mz = ix["mz"][r].astype(np.float32)
            it = ix["it"][r].astype(np.float32)
            b = np.rint(mz.astype(np.float64) / TOL_DA).astype(np.int32)
            b[(mz == 0) | (b < 0) | (b >= BINS)] = -1
            c = it / (np.linalg.norm(it, axis=1, keepdims=True) + 1e-12)
            self.side[positive] = {"key": keys[r], "lib": ix["lib"][r], "nm": nm[r],
                                   "mz": mz, "bin": b, "entropy": entropy_p(it), "cos": c}
        self.scratch = np.zeros(BINS + 2)

    def rep_lib_of(self, key):
        for s in self.side.values():
            hit = np.where(s["key"] == key)[0]
            if len(hit):
                return bytes(s["lib"][hit[0]])
        return None

    def _gather(self, bins, w, kind):
        a = np.where(bins >= 0, self.scratch[np.clip(bins, 0, self.bins_n + 1)], 0.0)
        if kind == "cos":
            return (a * w).sum(1)
        b = w.astype(np.float64)
        m = (a > 0) & (b > 0)
        v = np.zeros_like(a)
        am, bm = a[m], b[m]
        v[m] = (am + bm) * np.log(am + bm) - am * np.log(am) - bm * np.log(bm)
        return v.sum(1) / np.log(4.0)

    def search(self, spectra, neutral, positive, window=WINDOW, n=N_ANALOGS,
               kind="entropy"):
        """Top `n` (key, sim) relatives of one molecule, best first; sim > 0 only.

        A reference's sim is the max over the query spectra and over direct/shifted
        matching, with shift = neutral - the reference's neutral mass.
        """
        if kind not in ("entropy", "cos"):
            raise ValueError(f"kind={kind!r}")
        s = self.side[bool(positive)]
        lo, hi = np.searchsorted(s["nm"], [neutral - window, neutral + window])
        if hi <= lo:
            return []
        best = np.zeros(hi - lo)
        for mz, it, _ in spectra:
            mz = np.asarray(mz, np.float64)
            it = np.asarray(it, np.float64)
            if not len(mz):
                continue
            if kind == "entropy":
                w = entropy_p(it).astype(np.float64)
            else:
                w = it / max(np.linalg.norm(it), 1e-12)
            qb = np.rint(mz / TOL_DA).astype(np.int64)
            ok = (w > 0) & (qb >= 0) & (qb < self.bins_n)
            if not ok.any():
                continue
            np.add.at(self.scratch, qb[ok], w[ok])
            for c0 in range(lo, hi, CHUNK):
                c1 = min(c0 + CHUNK, hi)
                rw = s[kind][c0:c1]
                direct = self._gather(s["bin"][c0:c1], rw, kind)
                shift = neutral - s["nm"][c0:c1]
                rmz = s["mz"][c0:c1].astype(np.float64)
                sb = np.rint((rmz + shift[:, None]) / TOL_DA).astype(np.int64)
                sb[(rmz == 0) | (sb < 0) | (sb >= self.bins_n)] = -1
                shifted = self._gather(sb, rw, kind)
                np.maximum(best[c0 - lo:c1 - lo], np.maximum(direct, shifted),
                           out=best[c0 - lo:c1 - lo])
            self.scratch[qb[ok]] = 0.0                     # clear only what was written
        o = np.argsort(-best, kind="stable")[:n]
        o = o[best[o] > 0]
        return [(bytes(s["key"][lo + i]), float(best[i])) for i in o]


def analog_term(cand_fp, analog_fp, sims, p=3.0):
    """max over analogs a of sims[a] ** p * Tanimoto(candidate, a), per candidate.

    Fingerprints are packed uint8 rows, as analog.Analog stores them. A high power on
    the similarity trusts only close relatives: a candidate near a weak match earns
    little, which is what kept many-seed propagation from diluting (RESULTS.md §5).
    """
    cand_fp = np.asarray(cand_fp, np.uint8)
    if not len(analog_fp) or not len(cand_fp):
        return np.zeros(len(cand_fp))
    c = np.unpackbits(cand_fp, axis=1).astype(np.float32)
    a = np.unpackbits(np.asarray(analog_fp, np.uint8), axis=1).astype(np.float32)
    inter = c @ a.T
    tan = inter / np.maximum(c.sum(1)[:, None] + a.sum(1)[None, :] - inter, 1e-9)
    return (tan * np.asarray(sims, np.float64)[None, :] ** p).max(1)


def cache_path(splits_path, settings, n_index_rows):
    """Where a split's analog lists are cached: keyed on the split file's bytes, the
    search settings and the index size, so no setting can read another's lists."""
    import hashlib
    import os
    import library
    h = hashlib.sha1()
    with open(splits_path, "rb") as f:
        h.update(f.read())
    h.update(repr(sorted(settings.items())).encode())
    h.update(str(int(n_index_rows)).encode())
    d = f"{library.WORK}/shift_cache"
    os.makedirs(d, exist_ok=True)
    return f"{d}/{h.hexdigest()}.npz"


def selftest():
    s = np.sqrt
    # Hand-computed: q = {100: .5, 200: .5}, r = {100: .5, 300: .5}. One of two equal
    # peaks matches, which is 0.5 for both measures.
    assert abs(pair_sim([100, 200], s([.5, .5]), [100, 300], s([.5, .5])) - 0.5) < 1e-6
    assert abs(pair_sim([100, 200], s([.5, .5]), [100, 300], s([.5, .5]), kind="cos")
               - 0.5) < 1e-6
    assert abs(pair_sim([100, 200, 250], [1, .7, .3], [100, 200, 250], [1, .7, .3])
               - 1.0) < 1e-6
    assert pair_sim([100, 200], [1, 1], [400, 500], [1, 1]) == 0.0
    # A +CH2 relative (14.01565 Da): every reference fragment carries the modification,
    # so nothing lines up directly and everything does once shifted.
    q = ([114.0157, 214.0157, 300.0], [1, .8, .5])
    r = ([100.0, 200.0, 285.9843], [1, .8, .5])
    assert pair_sim(*q, *r) < 0.2
    assert pair_sim(*q, *r, shift=14.01565) > 0.99
    assert pair_sim([], [], [100], [1]) == 0.0
    p = entropy_p(np.array([[1.0, 1.0, 0.0]]))
    assert np.allclose(p, [[.5, .5, 0]]), p
    _selftest_index()
    # analog_term: identical to the analog -> sim^p; Tanimoto 1/3 -> sim^p / 3.
    a = np.packbits(np.array([[1, 1, 0, 0] + [0] * 2044], np.uint8), axis=1)
    c = np.packbits(np.array([[1, 1, 0, 0] + [0] * 2044, [1, 0, 1, 0] + [0] * 2044],
                             np.uint8), axis=1)
    assert np.allclose(analog_term(c, a, np.array([0.5]), p=3.0), [0.125, 0.125 / 3])
    assert np.allclose(analog_term(c, a[:0], np.zeros(0)), [0, 0])
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        f.write(b"split bytes")
        f.flush()
        s1 = cache_path(f.name, {"sim": "entropy", "win": 200, "n": 80}, 10)
        assert s1 == cache_path(f.name, {"n": 80, "win": 200, "sim": "entropy"}, 10)
        assert s1 != cache_path(f.name, {"sim": "cos", "win": 200, "n": 80}, 10)
        assert s1 != cache_path(f.name, {"sim": "entropy", "win": 200, "n": 80}, 11)
        f.write(b" changed")
        f.flush()
        assert s1 != cache_path(f.name, {"sim": "entropy", "win": 200, "n": 80}, 10)
    print("shifted self-checks passed")


def _fake_ix(rows):
    """A tiny index in library.load()'s layout from (key, lib, positive, peaks)."""
    n = len(rows)
    ix = {"mz": np.zeros((n, 64), np.float32), "it": np.zeros((n, 64), np.float16),
          "npk": np.zeros(n, np.int16), "precursor": np.zeros(n, np.float32),
          "ion": np.zeros(n, bool), "inchikey14": np.zeros(n, "S14"),
          "lib": np.zeros(n, "S24")}
    for i, (key, lib, pos, peaks) in enumerate(rows):
        mz, it = zip(*peaks)
        ix["mz"][i, :len(mz)], ix["it"][i, :len(mz)] = mz, it
        ix["npk"][i], ix["ion"][i] = len(mz), pos
        ix["inchikey14"][i], ix["lib"][i] = key, lib
        ix["precursor"][i] = 400.0          # unused: the index works on neutral masses
    return ix


def _selftest_index():
    from search import spectra_of
    mA, ch2 = 300.0, 14.01565
    query = [(100.0, 1.0), (150.0, .8), (200.0, .6), (250.0, .4)]
    minus = [(m - ch2, v) for m, v in query]           # every fragment carries the CH2
    rows = [(b"A", b"gnps", True, query),                # 0: the query itself, dropped
            (b"B", b"enveda-180", True, minus),          # 1: relative, timsTOF
            (b"B", b"gnps", True, [(90.0 + 7 * i, 1.0) for i in range(6)]),  # 2: B, more peaks
            (b"C", b"gnps", True, query),                # 3: 250 Da away
            (b"D", b"gnps", False, minus),               # 4: negative mode
            (b"E", b"gnps", True, query),                # 5: banned
            (b"F", b"gnps", True, query),                # 6: unparseable, mass -1
            (b"G", b"gnps", True, [(100.0, 1.0), (150.0, .8), (330.0, .6), (380.0, .4)])]
    ix = _fake_ix(rows)
    mass = {b"A": mA, b"B": mA - ch2, b"C": mA + 250, b"D": mA - ch2, b"E": mA + 10,
            b"F": -1.0, b"G": mA + 30}
    masses = (np.array(list(mass), "S14"), np.array(list(mass.values())))
    si = ShiftIndex(ix, drop=[0], banned=[b"E"], masses=masses)
    q = spectra_of(ix, [0])
    for kind in ("entropy", "cos"):
        got = si.search(q, neutral=mA, positive=True, kind=kind)
        keys = [k for k, _ in got]
        assert keys[0] == b"B" and got[0][1] > 0.99, (kind, got)
        assert b"G" in keys and 0 < dict(got)[b"G"] < 0.99, (kind, got)
        assert b"A" not in keys
        assert b"E" not in keys and b"F" not in keys
        assert b"C" not in keys
        assert b"D" not in keys
        # The vectorised search agrees with the one-pair reference implementation.
        rep_row = {b"B": 1, b"G": 7}
        for k, v in got:
            r = spectra_of(ix, [rep_row[k]])[0]
            want = pair_sim(q[0][0], q[0][1], r[0], r[1], shift=mA - mass[k], kind=kind)
            assert abs(v - want) < 1e-4, (kind, k, v, want)
    assert si.rep_lib_of(b"B") == b"enveda-180"
    empty = [(np.zeros(0, np.float32), np.zeros(0, np.float32), 300.0)]
    assert si.search(empty, mA, True) == []
    assert si.search(q, neutral=5000.0, positive=True) == []
    assert [k for k, _ in si.search(q, neutral=mA, positive=True, n=1)] == [b"B"]


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
