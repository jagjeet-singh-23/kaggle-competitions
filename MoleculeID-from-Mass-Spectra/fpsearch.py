"""Rank candidates by a fingerprint predicted from the spectrum, and measure MRR.

Library search (search.py) reaches only structures that already have spectra, and
analog propagation (analog.py) reaches only structures that look like one that does.
A predicted fingerprint has neither limit: every candidate in the precursor's mass
window can be scored against it directly.

    score(c) = spectral(c) + BETA * analog(c) + GAMMA * tanimoto(predicted_fp, fp(c))

The first two terms are unchanged, so class 1 cannot regress. The third is the only
one that can reach a class 2 molecule whose neighbourhood the library never sampled.

The model must have been trained with --holdout, or it has seen the query spectra and
the MRR below is fiction.

    python3 fpsearch.py
"""
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

import library
import prior
import shifted
from analog import ADDUCTS, Analog, BETA
import fingerprint
from fingerprint import Net, NBIN, BIN, predict
from metric import mrr
from search import Searcher, spectra_of

DIR = os.path.dirname(os.path.abspath(__file__))
DATA, WORK = library.DATA, library.WORK
TOP_K = 25
GAMMA = float(os.environ.get("GAMMA", 32.0))   # see submit(): chosen on the solved class mix
GAMMAS = tuple(float(x) for x in os.environ.get("GAMMAS", "0,4,8,16,32,64").split(","))
# FP_NORM normalises the predicted-fingerprint tanimoto within each molecule's pool
# before GAMMA weighs it, so the weight does not depend on the model's tanimoto scale
# (a sharper ranking-loss tau shrinks it 2.6x, RESULTS.md sections 24 and 27): "std"
# t / std(t), "z" (t - mean) / std, "rank" the percentile in the pool; "" leaves t raw
# (v12). A std-normalised gamma of 2.62 weighs the v12-style model as raw 32 does.
FP_NORM = os.environ.get("FP_NORM", "")
# SHIFT_W weighs the mass-shifted analog term in submit(): relatives within ±200 Da,
# SHIFT_W * max_a sim_a^SHIFT_P * tanimoto(c, a) (RESULTS.md section 32). 0 leaves it out.
SHIFT_W = float(os.environ.get("SHIFT_W", 0.0))
SHIFT_P = float(os.environ.get("SHIFT_P", 3.0))
SHIFT_SIM = os.environ.get("SHIFT_SIM", "entropy")
# RANKER re-orders each molecule's top features.N_RERANK candidates by a learned ranker
# (ranker.py --fit, RESULTS.md section 33) instead of v17's total: the path of an exported
# model, "" leaves v17's order. RANK_BY="v17" is the test mode: the same feature rows,
# ordered by their v17 column, which must reproduce the v17 file byte for byte.
RANKER = os.environ.get("RANKER", "")
RANK_BY = os.environ.get("RANK_BY", "")
# CHANNEL_OUT turns on a PubChem channel (LEARNING.md entry 12): a second candidate pool
# (external.OUT-style dirs, COCONUT first so that merge() keeps COCONUT's copy of a shared
# key) scored by the same v17 total. Its best candidates that are neither index structures
# nor COCONUT's take fixed slots in v17's list, as the public pipelines place PubChem
# proposals; rank 1 is never touched. "" leaves v17's lists as they are.
CHANNEL_OUT = os.environ.get("CHANNEL_OUT", "")
CH_N, CH_DEPTH, CH_GATE = 6, 100, 0.9      # proposals, rows searched, library-match gate
CH_AGG, CH_GENTLE = (2, 4, 6, 8, 10, 24), (4, 8, 12, 16, 20, 24)


def normalise(t, how=FP_NORM):
    if not how:
        return t
    if how == "rank":
        r = np.argsort(np.argsort(t, kind="stable"), kind="stable")
        return r / max(len(t) - 1, 1)
    sd = t.std()
    if len(t) < 2 or sd < 1e-9:
        return np.zeros_like(t)
    if how == "std":
        return t / sd
    assert how == "z", f"FP_NORM={how!r}"
    return (t - t.mean()) / sd


class Ensemble(torch.nn.Module):
    """Several checkpoints of one bit set as one model: the mean of their probabilities.

    forward returns logits so that fingerprint.predict, which applies the sigmoid, is
    unchanged. Retraining one config moves the eval by up to 0.025 (RESULTS.md 26), so
    a mean over seeds should be steadier than any one of them.
    """
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        p = torch.stack([torch.sigmoid(m(x)) for m in self.models]).mean(0)
        return torch.logit(p, eps=1e-7)


def load_one(path):
    assert os.path.exists(path), f"{path}: run `python3 fingerprint.py --train --holdout` first"
    ck = torch.load(path, weights_only=False)
    bits = ck["bits"]
    # Rebuild the exact architecture the checkpoint was trained with, and put the
    # featuriser back in the matching mode, rather than assuming the defaults.
    fingerprint.DEFECT = ck.get("defect", False)
    hidden = ck.get("hidden", 1024)
    assert ck.get("bin", BIN) == BIN, "checkpoint used a different m/z bin width"
    m = Net(fingerprint.n_features(), len(bits), hidden=hidden)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, bits, fingerprint.DEFECT


def load_model():
    # Read through the module rather than a name bound at import time, so a caller
    # (the Kaggle kernel) can point it at a shipped checkpoint. A comma-separated
    # FP_MODEL is an ensemble of checkpoints with the same bits and features.
    got = [load_one(p) for p in fingerprint.MODEL.split(",")]
    if len(got) == 1:
        return got[0][:2]
    bits = got[0][1]
    assert all(np.array_equal(b, bits) and d == got[0][2] for _, b, d in got), \
        "ensemble members were trained on different drop sets or features"
    return Ensemble([m for m, _, _ in got]).eval(), bits


def query_fp(model, spectra):
    """One fingerprint per molecule: the mean prediction over its spectra.

    Averaging rather than taking one spectrum's answer, because a molecule is
    usually measured at several collision energies and each sees different fragments.
    """
    mz = np.zeros((len(spectra), 64), np.float32)
    it = np.zeros((len(spectra), 64), np.float32)
    pm = np.zeros(len(spectra), np.float32)
    for i, (m, v, p) in enumerate(spectra):
        n = min(len(m), 64)
        mz[i, :n], it[i, :n], pm[i] = m[:n], v[:n], p
    return predict(model, mz, it, pm).mean(0)


def main():
    sp = np.load(f"{WORK}/splits.npz", allow_pickle=True)
    ix = library.load()
    se = Searcher(drop=sp["drop"], banned=sp["banned"], ix=ix)
    model, bits = load_model()
    import resource
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * library._RSS_UNIT / 2**30
    print(f"peak RSS after loading index, pool and model: {rss():.2f} GB", flush=True)

    # Candidate fingerprints stay packed. Unpacking all 264,127 of them to float32
    # over the 1,575 predicted bits is 1.66 GB, which the memory guard killed; the
    # mass window is ~115 candidates, so unpack per query instead.
    def pool_bits(rows):
        return np.unpackbits(an.fp[rows], axis=1)[:, bits].astype(np.float32)

    pos = [a for a, (_, p) in ADDUCTS.items() if p]
    neg = [a for a, (_, p) in ADDUCTS.items() if not p]
    qkey, qclass, query = sp["qkey"], sp["qclass"], sp["query"]

    jobs, answers = [], {1: {}, 2: {}, 3: {}}
    for k in np.unique(qkey):
        m = qkey == k
        cls, rows = int(qclass[m][0]), query[m]
        spectra = spectra_of(ix, rows)
        add = pos if bool(ix["ion"][rows[0]]) else neg
        spec, smiles = se.score_structures(spectra)
        pm = float(spectra[0][2])
        jobs.append([cls, k, spec, smiles, pm, add, None, query_fp(model, spectra)])
        answers[cls][k] = ix["smiles"][rows[0]].decode()
    # The index and the candidate pool are 1.3 GB and, at PubChem scale, 5 GB. Build
    # the second only once the first is gone, and take the mass windows in a second
    # pass, so the two are never resident together.
    del ix, se
    an = Analog(banned=sp["banned"])
    for j in jobs:
        j[6] = an.pool(j[4], j[5])
    print(f"{len(jobs)} molecules, median pool {int(np.median([len(j[6]) for j in jobs]))}\n")

    print(f"{'gamma':<8}{'class1':>9}{'class2':>9}{'class3':>9}{'mean':>9}")
    best = (-1, None)
    for g in GAMMAS:
        preds = {1: {}, 2: {}, 3: {}}
        for cls, k, spec, smiles, pm, add, pool, fpq in jobs:
            total = dict(spec)
            sm = dict(smiles)
            extra = an.propagate(spec, pm, add)
            for key_, v in extra.items():
                total[key_] = total.get(key_, 0.0) + BETA * v
                sm.setdefault(key_, an.smiles[an.row[key_]])
            if g and len(pool):
                # Tanimoto between the predicted (soft) fingerprint and each
                # candidate's, in one matrix product over the mass window.
                B = pool_bits(pool)
                inter = B @ fpq
                union = B.sum(1) + fpq.sum() - inter
                t = inter / np.maximum(union, 1e-9)
                for r, v in zip(pool, t):
                    key_ = an.key[r]
                    total[key_] = total.get(key_, 0.0) + g * float(v)
                    sm.setdefault(key_, an.smiles[r])
            top = sorted(total.items(), key=lambda kv: -kv[1])[:TOP_K]
            preds[cls][k] = [sm[q].decode() if isinstance(sm[q], bytes) else sm[q]
                             for q, _ in top]
        v = [mrr(preds[c], answers[c]) for c in (1, 2, 3)]
        print(f"{g:<8g}{v[0]:>9.4f}{v[1]:>9.4f}{v[2]:>9.4f}{np.mean(v):>9.4f}",
              flush=True)
        if np.mean(v) > best[0]:
            best = (np.mean(v), g, v)
    print(f"\nbest gamma={best[1]:g}  mean {best[0]:.4f}")
    print("search+analog alone was 0.9420 / 0.0676 / 0.0000, mean 0.3365")
    # Class 3 must be unreachable from the index alone, and reachable only once an
    # external database is merged. Asserting the right one of those depends on
    # whether that pool is loaded.
    if an.external_n:
        assert best[2][2] > 0, "external pool loaded but class 3 still unreachable"
    else:
        assert best[2][2] == 0, f"class 3 leaked: {best[2][2]:.4f}"


def _learned_setup(gamma):
    """The ranker's tables were built at v17's settings (features.V17, dilute's entropy
    shift similarity, the COCONUT synonym prior), so its features mean what they meant
    in training only there. Returns the loaded Ranker, or None for RANK_BY=v17."""
    import features
    from ranker import Ranker
    c = features.V17
    got = (gamma, BETA, SHIFT_W, SHIFT_P, SHIFT_SIM, FP_NORM, prior.PRIOR, prior.W, prior.GATE)
    want = (c["gamma"], c["beta"], c["sw"], c["p"], "entropy", "", "syn", c["prior_w"], c["prior_gate"])
    assert got == want, f"the ranker needs v17's settings {want}, got {got}"
    assert RANK_BY in ("", "v17"), f"RANK_BY={RANK_BY!r}"
    if not RANKER:
        return None
    assert os.path.exists(RANKER), f"RANKER={RANKER} does not exist"
    return Ranker.load(RANKER)


def _once(f):
    """f(), computed on the first call only: the query fingerprint and the shifted-analog
    search are needed by the channel's pool too, and must not run twice."""
    box = []

    def get():
        if not box:
            box.append(f())
        return box[0]
    return get


def _text(s):
    return s.decode() if isinstance(s, bytes) else s


def _score(an, spec, smiles, spectra, pm, add, gamma, bits, fpq, relatives, prior_of):
    """v17's total over one candidate pool: spectral + BETA * analog + gamma * predicted-
    fingerprint tanimoto + SHIFT_W * shifted analogs + the gated prior. fpq and relatives
    are _once callables (relatives None without shifted analogs). Returns the totals, the
    SMILES by key, the pool rows, and each pool key's raw tanimoto."""
    total, sm, fp_raw = dict(spec), dict(smiles), {}
    for k, v in an.propagate(spec, pm, add).items():
        total[k] = total.get(k, 0.0) + BETA * v
        sm.setdefault(k, an.smiles[an.row[k]])
    pool = an.pool(pm, add)
    if gamma and len(pool):
        q = fpq()
        B = np.unpackbits(an.fp[pool], axis=1)[:, bits].astype(np.float32)
        inter = B @ q
        raw = inter / np.maximum(B.sum(1) + q.sum() - inter, 1e-9)
        for r, v, w in zip(pool, normalise(raw), raw):
            k = an.key[r]
            total[k] = total.get(k, 0.0) + gamma * float(v)
            sm.setdefault(k, an.smiles[r])
            fp_raw[k] = float(w)
    if relatives is not None and len(pool):
        rel = [(an.row[a], s) for a, s in relatives() if a in an.row]
        if rel:
            t = shifted.analog_term(an.fp[pool], an.fp[np.array([r for r, _ in rel])],
                                    np.array([s for _, s in rel]), SHIFT_P)
            for r, v in zip(pool, t):
                k = an.key[r]
                total[k] = total.get(k, 0.0) + SHIFT_W * float(v)
                sm.setdefault(k, an.smiles[r])
    if prior_of and max(spec.values(), default=0.0) / len(spectra) < prior.GATE:
        for k in total:
            total[k] += prior.W * prior_of.get(k, prior.FILL)
    return total, sm, pool, fp_raw


def slot_merge(base, ch, slots, n=TOP_K):
    """The public pipelines' merge: channel items at the given 1-based final positions,
    the base list in order everywhere else, cut to n."""
    out, b, c = [], iter(base), iter(ch)
    while len(out) < n:
        nxt = next(c, None) if len(out) + 1 in slots else None
        if nxt is None:
            nxt = next(b, None)
        if nxt is None:
            nxt = next(c, None)
        if nxt is None:
            break
        out.append(nxt)
    return out


assert slot_merge(list("ABC"), list("XY"), (2, 4)) == list("AXBYC")
assert slot_merge(list("ABC"), [], (2, 4)) == list("ABC")
assert slot_merge(list("A"), list("XYZ"), (4,)) == list("AXYZ")
assert slot_merge([], list("XY"), CH_AGG) == list("XY")   # an empty base: the channel leads


def _channel_pool(an):
    """The channel's pool, built with external.OUT pointed at CHANNEL_OUT for the one
    call, and the keys of the main pool's external structures (COCONUT's)."""
    import external
    main_out = external.OUT
    assert CHANNEL_OUT != main_out, "CHANNEL_OUT must differ from the main pool"
    external.OUT = CHANNEL_OUT
    try:
        an2 = Analog()
    finally:
        external.OUT = main_out
    # Analog() swallows a missing shard directory and returns the index alone; a channel
    # without its pool would silently ship v17.
    assert an2.external_n > an.external_n, (an2.external_n, an.external_n)
    return an2, set(an.key[an.n_internal:].tolist())


def _channel(an2, coco, cands, top, fp_raw, spec, smiles, spectra, pm, add, gamma, bits,
             fpq, relatives, prior_of, stats):
    """v17's list with the channel's proposals in fixed slots (LEARNING.md entry 12).

    A strong library match (top_spec >= CH_GATE) keeps the list. Otherwise the channel is
    the second pool's best CH_N structures, within its top CH_DEPTH, that are neither
    index structures nor COCONUT's. rel = the channel leader's raw tanimoto minus the
    best in v17's top 25: positive takes the aggressive slots, else the gentle ones."""
    if max(spec.values(), default=0.0) / len(spectra) >= CH_GATE:
        stats["untouched"] += 1
        return cands
    total, sm, pool, fp2 = _score(an2, spec, smiles, spectra, pm, add, gamma, bits, fpq,
                                  relatives, prior_of)
    ok = {an2.key[r] for r in pool if r >= an2.n_internal} - coco
    ranked = sorted(total.items(), key=lambda kv: -kv[1])[:CH_DEPTH]
    ch = [k for k, _ in ranked if k in ok][:CH_N]
    if not ch:
        stats["no channel"] += 1
        return cands
    rel = fp2.get(ch[0], 0.0) - max((fp_raw.get(k, 0.0) for k, _ in top), default=-np.inf)
    branch = "aggressive" if rel > 0 else "gentle"
    stats[branch] += 1
    return slot_merge(cands, [_text(sm[k]) for k in ch], CH_AGG if rel > 0 else CH_GENTLE)


def submit(path=None, gamma=GAMMA):
    """Write submission.csv using search + analog + the predicted fingerprint.

    gamma is 32. It was 64 for one revision, as the minimax pick across every class
    mix that was plausible before the leaderboard pinned the mix exactly; that
    argument retired with the uncertainty, and this docstring outlived it by a
    revision, which is how a run that was meant to be a gamma-64 control turned out
    to have been 32 all along.

    32 is the maximum under the solved mix (14.0 / 12.6 / 73.5) and it is the maximum
    for both candidate pools, so it is not a pool-specific tuning:

        gamma       16      32      64     128     256
        COCONUT   .1801   .1898   .1893   .1890   .1834
        + ChEMBL  .1791   .1895   .1878   .1868   .1786

    With RANKER or RANK_BY set, each molecule's candidates come from
    features.molecule_rows instead, the code the ranker's tables were built with, and
    are ordered after the loop by one Ranker.score call over every molecule's rows.
    """
    import pandas as pd
    from metric import validate

    learned = bool(RANKER or RANK_BY)
    assert not (learned and CHANNEL_OUT), "the channel is built on v17's order, not the ranker's"
    if learned:
        import features
        ranked = _learned_setup(gamma)
        held, feat_s = [], 0.0
    path = path or f"{DIR}/submission.csv"
    te = pd.read_parquet(f"{DATA}/test.parquet")
    ix = library.load()
    se = Searcher(ix=ix)
    si = shifted.ShiftIndex(ix) if SHIFT_W else None
    del ix                       # Searcher keeps its own masked copy; see main()
    an = Analog()
    an2, coco = _channel_pool(an) if CHANNEL_OUT else (None, None)
    ch_stats = dict(untouched=0, gentle=0, aggressive=0, **{"no channel": 0})
    model, bits = load_model()
    prior_of = prior.load() if prior.W else {}
    rows, clock = [], {"shift": 0.0}
    for mid, g in te.groupby("molecule_id"):
        spectra = [library.trim(a, b, p) + (p,)
                   for a, b, p in zip(g.ms2_mzs, g.ms2_normalized_intensities,
                                      g.precursor_mz)]
        spec, smiles = se.score_structures(spectra)
        # One spectrum's precursor with that spectrum's own adduct: every spectrum
        # of a molecule shares one neutral mass, and pairing the first precursor
        # with another spectrum's adduct opens a window ~22 Da off full of decoys.
        known = [i for i, a in enumerate(g.adduct) if a in ADDUCTS]
        i = known[0] if known else 0
        pm = float(spectra[i][2])
        add = [g.adduct.iloc[i]] if known else ["[M+H]+"]
        if learned:
            pool = an.pool(pm, add)
            fpq = query_fp(model, spectra) if len(pool) else None
            analogs = []
            if len(pool):
                t0 = time.time()
                shift, positive = ADDUCTS[add[0]]
                analogs = si.search(spectra, pm - shift, positive, kind=SHIFT_SIM)
                clock["shift"] += time.time() - t0
            t0 = time.time()
            _, smi, X = features.molecule_rows(an, bits, spec, smiles, pm, add, fpq, spectra,
                                               analogs, prior_of)
            feat_s += time.time() - t0
            held.append((mid, smi, X))
            continue
        def search():
            t0 = time.time()
            shift, positive = ADDUCTS[add[0]]
            got = si.search(spectra, pm - shift, positive, kind=SHIFT_SIM)
            clock["shift"] += time.time() - t0
            return got
        # Both callables are used inside this iteration only, so binding the loop's
        # spectra / pm / add late is safe.
        fpq = _once(lambda: query_fp(model, spectra))
        relatives = _once(search) if si is not None else None
        total, sm, _, fp_raw = _score(an, spec, smiles, spectra, pm, add, gamma, bits, fpq,
                                      relatives, prior_of)
        top = sorted(total.items(), key=lambda kv: -kv[1])[:TOP_K]
        cands = [_text(sm[q]) for q, _ in top]
        if an2 is not None:
            lead = cands[:1]
            cands = _channel(an2, coco, cands, top, fp_raw, spec, smiles, spectra, pm, add,
                             gamma, bits, fpq, relatives, prior_of, ch_stats)
            # An empty v17 list (no library spectrum and no index or COCONUT structure in
            # the window, while pop2 has some) has no rank 1 to keep: the channel fills it,
            # where v17 wrote "C". v19's hidden rerun died on this assert without `lead or`.
            assert not lead or cands[:1] == lead, f"{mid}: the channel moved rank 1"
        rows.append((mid, ";".join(cands[:TOP_K] or ["C"])))
    if learned:
        t0 = time.time()
        X = np.concatenate([x for _, _, x in held] or [np.zeros((0, len(features.FEATURES)), np.float32)])
        s = ranked.score(X) if ranked is not None else X[:, features.FEATURES.index("v17")]
        changed, at = 0, 0
        for mid, smi, x in held:
            o = np.argsort(-s[at:at + len(x)], kind="stable")[:TOP_K]
            at += len(x)
            cands = [smi[j] for j in o]
            changed += cands != smi[:TOP_K]
            rows.append((mid, ";".join(cands or ["C"])))
        print(f"learned order ({RANKER or 'RANK_BY=' + RANK_BY}): features {feat_s:.0f} s, "
              f"scoring {time.time() - t0:.1f} s over {len(X):,} rows; "
              f"{changed} of {len(held)} lists differ from v17's order")
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    validate(sub)
    sub.to_csv(path, index=False)
    print(f"wrote {path}: {len(sub)} molecules, "
          f"{sub.smiles.str.split(';').str.len().mean():.1f} candidates each")
    if si is not None:
        print(f"shifted analogs: weight {SHIFT_W:g}, p {SHIFT_P:g}, {SHIFT_SIM}, "
              f"search {clock['shift']:.0f} s over {len(sub)} molecules")
    if an2 is not None:
        print(f"PubChem channel ({an2.external_n:,} external structures, {len(coco):,} "
              f"COCONUT keys excluded): {ch_stats}")


if __name__ == "__main__":
    submit() if "--submit" in sys.argv else main()
