"""A learned ranker over v17's candidate lists, trained on held-out queries.

v17's weights (32 on the predicted fingerprint, 16 on mass-shifted analogs, 0.5 on
the prior) were each picked on held-out data one at a time, and weights picked that
way have repeatedly failed to transfer (RESULTS.md sections 15-16: v10, v11, v13).
Public pipelines learn the combination from simulated queries instead: prvsiyan's
31 features under eight HistGBMs, engine 1's 160 under four LightGBMs
(docs/superpowers/specs/2026-10-20-raasta-b-design.md, component 2).

This file has two halves. The tables: every held-out query of a split becomes v17's
top N_RERANK candidates as feature rows (features.molecule_rows), labelled by the
competition's own key14. The fingerprint features must come from a model that never
saw the split's queries, or the ranker learns to trust a feature the test cannot
give it, so check_leak_free refuses any other pairing. The model: trained and
scored out of fold, grouped by structure.

    SPLITS=... FP_MODEL=... python3 ranker.py --table
    python3 ranker.py --selftest
"""
import hashlib
import os
import sys
import time

import numpy as np

import features
import library
import metfrag

WORK = library.WORK
K = 25                     # MRR@25
TABLE_VERSION = 1
# The split a fingerprint model family left out. A model is only valid on the split
# whose rows it excluded (HANDOFF, Models); every other pairing leaks.
LEAK_FREE = {"splits_big.npz": "fpmodel_big_", "splits_bigord.npz": "fpmodel_bigord_"}
N_PROC = max(1, (os.cpu_count() or 4) - 2)


def check_leak_free(splits_path, fp_model):
    split = os.path.basename(splits_path)
    assert split in LEAK_FREE, f"leak: no fingerprint model family is known to exclude {split}"
    for m in fp_model.split(","):
        assert os.path.basename(m).startswith(LEAK_FREE[split]), \
            f"leak: {os.path.basename(m)} did not exclude {split}"


def mrr_by(T, score):
    """MRR@25 per class when each query's rows are ordered by -score (stable).

    The denominator is every query of the class, including those with no rows or no
    positive among them, as the competition counts a molecule whose answer is not
    in the list."""
    rr = rr_by_query(T, score)
    cls, env = T["qclass"], T["env"]
    sel = {"c1": cls == 1, "c2": cls == 2, "c3": cls == 3,
           "c3_nonenv": (cls == 3) & ~env, "c3_env": (cls == 3) & env}
    return {k: float(rr[m].mean()) if m.any() else float("nan") for k, m in sel.items()}


def table_path(splits_path, fp_model):
    """{WORK}/rank_tables/<split>_<hash>.npz, the hash over everything a row depends on."""
    h = hashlib.sha1()
    with open(splits_path, "rb") as f:
        h.update(f.read())
    h.update(repr(sorted(os.path.basename(m) for m in fp_model.split(","))).encode())
    h.update(repr((sorted(features.V17.items()), features.N_RERANK, features.FEATURES)).encode())
    h.update(repr((metfrag.MAX_CUTS, metfrag.MAX_BONDS, metfrag.H_SHIFTS, metfrag.TOL_DA)).encode())
    h.update(f"v{TABLE_VERSION}".encode())
    stem = os.path.splitext(os.path.basename(splits_path))[0]
    return f"{WORK}/rank_tables/{stem}_{h.hexdigest()[:12]}.npz"


def build_table(out=None):
    """Rows for every query of the split in the environment (SPLITS, FP_MODEL), at v17's
    settings with C2_BANNED=1: class 2 answers and the index structures whose spectra
    the split held out are then never candidates, an artefact the hidden test has no
    counterpart of. Returns the table's path; an existing table is not rebuilt."""
    import fingerprint
    split, fp_model = os.environ.get("SPLITS", ""), fingerprint.MODEL
    check_leak_free(split, fp_model)
    out = out or table_path(split, fp_model)
    if os.path.exists(out):
        print(f"table exists: {out}", flush=True)
        return out
    import dilute
    import prior
    from analog import Analog
    from metric import key14
    assert dilute.C2_BANNED and dilute.TRUE_ADDUCT, "tables are built with C2_BANNED=1 TRUE_ADDUCT=1"
    assert dilute.SHIFT_WS == (features.V17["sw"],) and dilute.SHIFT_P == features.V17["p"], \
        "tables are built at v17's shifted-analog setting (SHIFT_WS=16, SHIFT_P=3)"
    t0 = time.time()
    jobs, answers, banned, bits = dilute.build_jobs()
    an = Analog(banned=banned, external=True)
    prior_of = prior.load("syn")
    t1 = time.time()
    print(f"[{t1 - t0:6.0f} s] build_jobs: {len(jobs)} queries", flush=True)

    rows, smiles, qkey, qclass, env, ans = [], [], [], [], [], []
    for q, (cls, k, spec, sm, pm, add, fpq, spectra, analogs) in enumerate(jobs):
        _, smi, X = features.molecule_rows(an, bits, spec, sm, pm, add, fpq, spectra,
                                           analogs, prior_of)
        rows.append(X)
        smiles.append(smi)
        qkey.append(k)
        qclass.append(cls)
        env.append(dilute.LIB_OF[k] == b"enveda-180")
        ans.append(answers[cls][k])
        if q % 500 == 0:
            print(f"[{time.time() - t0:6.0f} s]   features {q}/{len(jobs)}", flush=True)
    t2 = time.time()
    print(f"[{t2 - t0:6.0f} s] features: {sum(len(x) for x in rows):,} rows "
          f"({t2 - t1:.0f} s)", flush=True)

    import multiprocessing as mp
    uniq = sorted(set(s for smi in smiles for s in smi) | set(ans))
    with mp.get_context("spawn").Pool(N_PROC) as pool:
        k14 = dict(zip(uniq, pool.map(key14, uniq, chunksize=256)))
    y = np.concatenate([np.array([k14[s] is not None and k14[s] == k14[a] for s in smi], np.uint8)
                        for smi, a in zip(smiles, ans)]) if rows else np.zeros(0, np.uint8)
    t3 = time.time()
    print(f"[{t3 - t0:6.0f} s] labels: key14 of {len(uniq):,} SMILES on {N_PROC} processes "
          f"({t3 - t2:.0f} s)", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, X=np.concatenate(rows) if rows else np.zeros((0, len(features.FEATURES)), np.float32),
             y=y, qid=np.concatenate([np.full(len(x), q, np.int32) for q, x in enumerate(rows)]),
             smiles=np.array([s.encode() for smi in smiles for s in smi]),
             qkey=np.array(qkey, "S14"), qclass=np.array(qclass, np.int8), env=np.array(env),
             answer=np.array([a.encode() for a in ans]),
             split=np.array(os.path.splitext(os.path.basename(split))[0]),
             feature_names=np.array(features.FEATURES))
    print(f"wrote {out}", flush=True)
    return out


def load_table(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def ceiling(T, score):
    """Per class: queries, and how many have a positive in the top 25 / anywhere in the
    N_RERANK rows -- the most a re-ranker of those rows could reach."""
    o = np.argsort(-np.asarray(score, np.float64), kind="stable")
    o = o[np.argsort(T["qid"][o], kind="stable")]
    q = T["qid"][o]
    start = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
    rank = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)])) + 1
    pos = T["y"][o].astype(bool)
    in25, in100 = set(q[pos & (rank <= K)].tolist()), set(q[pos].tolist())
    out = {}
    for c in (1, 2, 3):
        ids = np.flatnonzero(T["qclass"] == c)
        out[c] = (len(ids), sum(i in in25 for i in ids), sum(i in in100 for i in ids))
    return out


# --- the model --------------------------------------------------------------------
# prvsiyan's settings for the same job. early_stopping is off: sklearn turns it on above
# 10,000 rows, and its validation split is random rows, which puts one query's
# candidates on both sides and makes the iteration count vary between runs.
GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80,
           l2_regularization=1.0, early_stopping=False)
LAMBDARANK = dict(objective="lambdarank", n_estimators=500, learning_rate=0.03, num_leaves=63,
                  min_child_samples=80, reg_lambda=1.0, verbose=-1)
SEEDS = tuple(int(s) for s in os.environ.get("RANK_SEEDS", "0,1,2,3").split(","))
MIX = {1: 0.140, 2: 0.126, 3: 0.735}      # dilute.MIX: the solved hidden class mix


def group_of(name):
    return next(g for g, names in features.GROUPS.items() if name in names)


def used_features(drop_groups=()):
    return tuple(n for n in features.FEATURES if group_of(n) not in drop_groups)


def columns(drop_groups=()):
    keep = set(used_features(drop_groups))
    return [i for i, n in enumerate(features.FEATURES) if n in keep]


def load_tables_from(tables):
    """Concatenate tables; qid becomes global, per-query arrays follow."""
    names = [tuple(t["feature_names"].tolist()) for t in tables]
    assert all(n == tuple(features.FEATURES) for n in names), "tables built with other features"
    out, off = {}, 0
    for t in tables:
        n_q = len(t["qclass"])
        t = dict(t, qid=t["qid"] + off,
                 split=np.broadcast_to(np.asarray(t["split"]), (n_q,)).astype("U32"))
        for k in ("X", "y", "qid", "smiles", "qkey", "qclass", "env", "answer", "split"):
            if k in t:
                out.setdefault(k, []).append(t[k])
        off += n_q
    T = {k: np.concatenate(v) for k, v in out.items()}
    T["feature_names"] = np.array(features.FEATURES)
    return T


def load_tables(paths):
    return load_tables_from([load_table(p) for p in paths])


def _weights(T, rows):
    """Row weights MIX[class] / (training queries of that class with a positive), and
    the rows of queries that have one: a query whose answer is not among its rows
    teaches a ranker nothing."""
    qid = T["qid"][rows]
    has = np.zeros(len(T["qclass"]), bool)
    has[qid[T["y"][rows].astype(bool)]] = True
    keep = has[qid]
    rows, qid = rows[keep], qid[keep]
    n = {c: len(np.unique(qid[T["qclass"][qid] == c])) for c in MIX}
    cw = np.zeros(4)
    for c in MIX:
        cw[c] = MIX[c] / max(n[c], 1)
    return rows, cw[T["qclass"][qid]]


def fit(T, rows_mask=None, drop_groups=(), model="gbm", seeds=SEEDS, params=None):
    rows = np.flatnonzero(rows_mask) if rows_mask is not None else np.arange(len(T["y"]))
    rows, w = _weights(T, rows)
    X, y = T["X"][rows][:, columns(drop_groups)], T["y"][rows]
    if model == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        p = params or GBM
        return [HistGradientBoostingClassifier(random_state=s, **p).fit(X, y, sample_weight=w)
                for s in seeds]
    if model == "linear":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        return [m.fit(X, y, logisticregression__sample_weight=w)]
    if model == "lambdarank":
        import lightgbm
        qid = T["qid"][rows]
        assert (np.diff(qid) >= 0).all(), "rows must be grouped by query"
        _, sizes = np.unique(qid, return_counts=True)
        p = {**LAMBDARANK, **(params or {})}
        return [lightgbm.LGBMRanker(random_state=s, **p).fit(X, y, sample_weight=w, group=sizes)
                for s in seeds]
    raise ValueError(f"model={model!r}")


def score(models, X, model="gbm"):
    """Mean over the models: probability for the classifiers, raw score for lambdarank."""
    if model == "lambdarank":
        return np.mean([m.predict(X) for m in models], 0)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], 0)


def folds(T, k=5):
    """Fold per query, grouped by structure, so a class 1 structure held out by both
    splits never trains a model that then scores it."""
    from sklearn.model_selection import GroupKFold
    f = np.empty(len(T["qclass"]), int)
    for i, (_, test) in enumerate(GroupKFold(k).split(np.zeros(len(f)), groups=T["qkey"])):
        f[test] = i
    return f


def cv(T, k=5, drop_groups=(), model="gbm", seeds=SEEDS, params=None):
    """Out-of-fold score for every row."""
    f = folds(T, k)[T["qid"]]
    out = np.zeros(len(T["y"]))
    cols = columns(drop_groups)
    for i in range(k):
        ms = fit(T, f != i, drop_groups, model, seeds, params)
        test = f == i
        out[test] = score(ms, T["X"][test][:, cols], model)
    return out


NODE_FIELDS = ("feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf", "value")


def _lgbm_nodes(tree):
    """One LightGBM tree (dump_model's tree_structure) as HistGBM-style node rows in
    preorder, children by index within the tree."""
    out = []

    def walk(n):
        i = len(out)
        out.append(None)
        if "leaf_value" in n:
            out[i] = (0, 0.0, 0, 0, 0, 1, n["leaf_value"])
            return i
        assert n["decision_type"] == "<=" and n["missing_type"] in ("None", "NaN"), n
        # Missing type None (no NaN in training): LightGBM scores NaN as 0.0.
        go_left = n["default_left"] if n["missing_type"] == "NaN" else 0.0 <= n["threshold"]
        left, right = walk(n["left_child"]), walk(n["right_child"])
        out[i] = (n["split_feature"], n["threshold"], go_left, left, right, 0, 0.0)
        return i

    walk(tree)
    return out


def export(models, feature_names, path, drop_groups=()):
    """sklearn HistGBM or LightGBM trees as flat arrays, so the kernel scores without
    either library. HistGBM scores are probabilities, LightGBM lambdarank's raw sums."""
    cat = {k: [] for k in NODE_FIELDS}
    offsets, tree_model, baseline, n = [0], [], [], 0
    lgbm = hasattr(models[0], "booster_")
    assert all(hasattr(m, "booster_") == lgbm for m in models), "one kind of model per export"
    for mi, m in enumerate(models):
        if lgbm:
            baseline.append(0.0)
            trees = []
            for t in m.booster_.dump_model()["tree_info"]:
                rows = _lgbm_nodes(t["tree_structure"])
                trees.append({k: np.array([r[j] for r in rows]) for j, k in enumerate(NODE_FIELDS)})
        else:
            baseline.append(float(np.ravel(m._baseline_prediction)[0]))
            trees = [it[0].nodes for it in m._predictors]
        for nodes in trees:
            for k in cat:
                cat[k].append(nodes[k])
            n += len(nodes["value"])
            offsets.append(n)
            tree_model.append(mi)
    np.savez(path, **{k: np.concatenate(v) for k, v in cat.items()},
             offsets=np.array(offsets), tree_model=np.array(tree_model),
             baseline=np.array(baseline), feature_names=np.array(feature_names),
             drop_groups=np.array(drop_groups, dtype="U16"),
             link=np.array("identity" if lgbm else "logistic"))


class Ranker:
    """The exported model: numpy only."""

    @classmethod
    def load(cls, path):
        r = cls()
        with np.load(path) as d:
            r.__dict__.update({k: d[k] for k in d.files})
        drop = tuple(r.drop_groups.tolist())
        assert tuple(r.feature_names.tolist()) == used_features(drop), \
            f"{path} was trained on other features than features.FEATURES minus {drop}"
        r.cols = columns(drop)
        r.link = str(r.__dict__.get("link", "logistic"))
        return r

    def score(self, X):
        """X has every column of features.FEATURES; returns the mean over the models of
        the probability (HistGBM) or the raw score (lambdarank), as score() does."""
        X = np.asarray(X, np.float64)[:, self.cols]
        raw = np.repeat(self.baseline[:, None], len(X), 1)
        rows = np.arange(len(X))
        for t, mi in enumerate(self.tree_model):
            lo = self.offsets[t]
            node = np.zeros(len(X), np.int64)
            while True:
                g = lo + node
                inner = ~self.is_leaf[g].astype(bool)
                if not inner.any():
                    break
                x = X[rows, self.feature_idx[g]]
                left = (x <= self.num_threshold[g]) | (np.isnan(x) & self.missing_go_to_left[g].astype(bool))
                node = np.where(inner, np.where(left, self.left[g], self.right[g]), node)
            raw[mi] += self.value[lo + node]
        if self.link == "identity":
            return raw.mean(0)
        return (1.0 / (1.0 + np.exp(-raw))).mean(0)


def rr_by_query(T, score_):
    """Reciprocal rank@25 per query (0 without a positive in the top 25)."""
    score_ = np.asarray(score_, np.float64)
    qid, y = T["qid"], T["y"].astype(bool)
    o = np.argsort(-score_, kind="stable")
    o = o[np.argsort(qid[o], kind="stable")]
    q = qid[o]
    start = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1] if len(q) else np.zeros(0, int)
    rank = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)])) + 1
    rr = np.zeros(len(T["qclass"]))
    hit = y[o] & (rank <= K)
    # Rows are grouped by query with ranks rising, so the first hit per query is the best.
    hq, first = np.unique(q[hit], return_index=True)
    rr[hq] = 1.0 / rank[hit][first]
    return rr


def paired_bootstrap(rr_a, rr_b, n=1000, seed=0):
    """Mean of rr_b - rr_a with a 95% interval from resampling queries."""
    d = np.asarray(rr_b, np.float64) - np.asarray(rr_a, np.float64)
    if not len(d):
        return 0.0, 0.0, 0.0
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pool_auc(T, col):
    """Within-query AUC of one feature, positive against the query's negatives, ties
    half; mean over queries that have both."""
    v = T["X"][:, list(T["feature_names"]).index(col)].astype(np.float64)
    y, qid = T["y"].astype(bool), T["qid"]
    out = []
    for q in np.unique(qid[y]):
        m = qid == q
        p, n = v[m & y], v[m & ~y]
        if len(n):
            out.append(((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean()))
    return float(np.mean(out)) if out else float("nan")


def log_run(name, params, metrics, artifacts=()):
    """One MLflow run in experiment casmi-ranker, only if MLFLOW_TRACKING_URI is set."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    import subprocess
    import mlflow
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    mlflow.set_experiment("casmi-ranker")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({**{k: str(v) for k, v in params.items()}, "git_commit": commit})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
        for a in artifacts:
            mlflow.log_artifact(a)


def subset(T, qmask):
    """The table restricted to the queries in qmask, qid renumbered."""
    new = np.cumsum(qmask) - 1
    rows = qmask[T["qid"]]
    S = {k: T[k][rows] for k in ("X", "y", "smiles") if k in T}
    S["qid"] = new[T["qid"][rows]].astype(np.int32)
    for k in ("qkey", "qclass", "env", "answer", "split"):
        if k in T:
            S[k] = T[k][qmask]
    S["feature_names"] = T["feature_names"]
    return S


def selftest():
    check_leak_free("data/splits_big.npz", "/x/fpmodel_big_rank_s0.pt,/x/fpmodel_big_rank_s1.pt")
    for bad in ("/x/fpmodel_rank.pt", "/x/fpmodel_big_rank_s0.pt,/x/fpmodel_rank_s1.pt",
                "/x/fpmodel_bigord_rank_s0.pt"):
        try:
            check_leak_free("data/splits_big.npz", bad)
            raise SystemExit(f"leak not caught: {bad}")
        except AssertionError:
            pass
    try:
        check_leak_free("data/splits.npz", "/x/fpmodel_rank.pt")
        raise SystemExit("a split without a leak-free model family was accepted")
    except AssertionError:
        pass
    # Hand table. Class 3: q0 answer at rank 2; q1 at rank 1 with a second positive at 3;
    # q2 has no positive. Class 1: q3's answer at rank 26 earns nothing (MRR@25).
    # q4 (class 3, enveda-180) has no rows at all and still counts.
    score = np.r_[[3.0, 2.0, 1.0], [5.0, 4.0, 3.0], [1.0], np.arange(30, 0, -1.0)]
    y = np.r_[[0, 1, 0], [1, 0, 1], [0], np.eye(30)[25]].astype(np.uint8)
    qid = np.r_[[0] * 3, [1] * 3, [2], [3] * 30].astype(np.int32)
    T = {"y": y, "qid": qid, "qclass": np.array([3, 3, 3, 1, 3], np.int8),
         "env": np.array([False, False, True, False, True])}
    m = mrr_by(T, score)
    assert np.isclose(m["c3"], (0.5 + 1 + 0 + 0) / 4), m
    assert np.isclose(m["c3_nonenv"], (0.5 + 1) / 2) and np.isclose(m["c3_env"], 0.0), m
    assert m["c1"] == 0.0 and np.isnan(m["c2"]), m
    # Stable on ties: equal scores keep the table's row order.
    m = mrr_by({**T, "y": np.r_[[0, 1, 0], [0, 0, 0], [0], np.zeros(30)].astype(np.uint8)},
               np.r_[[1.0, 1.0, 1.0], [0, 0, 0], [0], np.zeros(30)])
    assert np.isclose(m["c3"], 0.5 / 4), m
    _selftest_model()
    print("ranker self-checks passed")


def _synthetic(n_q=100, n_rows=30, seed=0):
    """A table whose answer is the row with the largest feature 0 plus noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_q * n_rows, len(features.FEATURES))).astype(np.float32)
    qid = np.repeat(np.arange(n_q, dtype=np.int32), n_rows)
    y = np.zeros(len(X), np.uint8)
    z = (X[:, 0] + 0.3 * rng.normal(size=len(X))).reshape(n_q, n_rows)
    y[np.arange(n_q) * n_rows + z.argmax(1)] = 1
    return {"X": X, "y": y, "qid": qid, "qclass": np.where(np.arange(n_q) % 4, 3, 1).astype(np.int8),
            "env": np.zeros(n_q, bool), "qkey": np.array([f"Q{i:04d}".encode() for i in range(n_q)], "S14"),
            "split": np.array(["syn"] * n_q), "feature_names": np.array(features.FEATURES)}


def _selftest_model():
    import tempfile
    T = _synthetic()
    small = {**GBM, "max_iter": 30}
    models = fit(T, seeds=(0, 1, 2, 3), params=small)
    assert len(models) == 4
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/r.npz"
        export(models, features.FEATURES, path)
        r = Ranker.load(path)
        want = np.mean([m.predict_proba(T["X"])[:, 1] for m in models], 0)
        assert np.abs(r.score(T["X"]) - want).max() < 1e-9
        # Values exactly on a split threshold go left, as sklearn's predictor does.
        nodes = models[0]._predictors[0][0].nodes
        Xt = np.repeat(T["X"][:1], 8, 0).astype(np.float64)
        for i in np.flatnonzero(~nodes["is_leaf"].astype(bool))[:8]:
            Xt[i % 8, nodes["feature_idx"][i]] = nodes["num_threshold"][i]
        want = np.mean([m.predict_proba(Xt)[:, 1] for m in models], 0)
        assert np.abs(r.score(Xt) - want).max() < 1e-9
        # A model whose saved feature names differ from features.FEATURES is refused.
        z = dict(np.load(path))
        z["feature_names"] = z["feature_names"][::-1]
        np.savez(f"{d}/bad.npz", **z)
        try:
            Ranker.load(f"{d}/bad.npz")
            raise SystemExit("a reordered feature list was accepted")
        except AssertionError:
            pass
        # Dropping a group trains on the rest and loads only against those names.
        m2 = fit(T, drop_groups=("frag",), seeds=(0,), params=small)
        export(m2, used_features(("frag",)), f"{d}/nofrag.npz", drop_groups=("frag",))
        assert np.abs(Ranker.load(f"{d}/nofrag.npz").score(T["X"])
                      - m2[0].predict_proba(T["X"][:, columns(("frag",))])[:, 1]).max() < 1e-9
    # Folds by structure: one qkey in two splits lands in the same fold.
    T2 = load_tables_from([T, {**_synthetic(seed=1), "split": np.array(["syn2"] * 100)}])
    f = folds(T2, 5)
    assert all(f[i] == f[i + 100] for i in range(100))          # same qkeys Q0000.. in both
    # pool_auc: above every negative -> 1.0, tied with every negative -> 0.5.
    H = {"X": np.array([[3.0], [1.0], [2.0], [1.0], [1.0]]), "y": np.array([1, 0, 0, 1, 0], np.uint8),
         "qid": np.array([0, 0, 0, 1, 1], np.int32), "feature_names": np.array(["a"])}
    assert pool_auc(H, "a") == (1.0 + 0.5) / 2
    # MLflow is optional: without a tracking URI nothing is imported.
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    log_run("selftest", {"a": 1}, {"m": 0.5})
    assert "mlflow" not in sys.modules
    # LightGBM lambdarank: one booster per seed, finite scores.
    lr = fit(T, model="lambdarank", seeds=(0, 1), params={"n_estimators": 20})
    assert len(lr) == 2 and np.isfinite(score(lr, T["X"], "lambdarank")).all()
    # ... exported to the same flat arrays, scored raw as score() does.
    with tempfile.TemporaryDirectory() as d:
        export(lr, features.FEATURES, f"{d}/lr.npz")
        r = Ranker.load(f"{d}/lr.npz")
        assert np.abs(r.score(T["X"]) - score(lr, T["X"], "lambdarank")).max() < 1e-9
        # Values exactly on a split threshold go left, as LightGBM's do.
        inner = np.flatnonzero(~r.is_leaf[:r.offsets[1]].astype(bool))[:8]
        Xt = np.repeat(T["X"][:1], 8, 0).astype(np.float64)
        for i in inner:
            Xt[i % 8, r.feature_idx[i]] = r.num_threshold[i]
        assert np.abs(r.score(Xt) - score(lr, Xt, "lambdarank")).max() < 1e-9
        # NaN goes where LightGBM sends it: trained without NaN, it scores NaN as 0.0.
        Xn = T["X"][:60].astype(np.float64)
        Xn[::3] = np.nan
        Xn[1::3, :5] = np.nan
        assert np.abs(r.score(Xn) - score(lr, Xn, "lambdarank")).max() < 1e-9
        m3 = fit(T, drop_groups=("frag",), model="lambdarank", seeds=(0,), params={"n_estimators": 20})
        export(m3, used_features(("frag",)), f"{d}/lr_nofrag.npz", drop_groups=("frag",))
        assert np.abs(Ranker.load(f"{d}/lr_nofrag.npz").score(T["X"])
                      - m3[0].predict(T["X"][:, columns(("frag",))])).max() < 1e-9
    # paired bootstrap of identical lists is exactly zero.
    a = np.r_[np.ones(10), np.zeros(10)]
    assert paired_bootstrap(a, a) == (0.0, 0.0, 0.0)


def _table_main():
    T = load_table(build_table())
    v = T["X"][:, features.FEATURES.index("v17")]
    m = mrr_by(T, v)
    print(f"v17 from the table ({T['split']}): " + "  ".join(f"{k} {x:.4f}" for k, x in m.items()))
    for c, (n, a, b) in ceiling(T, v).items():
        print(f"  class {c}: {n} queries, answer in top 25 {a} ({a / max(n, 1):.1%}), "
              f"in top {features.N_RERANK} {b} ({b / max(n, 1):.1%})")


def _args():
    paths = [a for a in sys.argv[2:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    drop = tuple(g for g in opt.get("drop", "").split(",") if g)
    assert all(g in features.GROUPS for g in drop), f"unknown group in {drop}"
    return paths, opt, drop


# Per split, the class 3 reading HANDOFF and the spec name: non-enveda-180 answers on the
# ordinary split, the covered answers on the covered one.
READ = {"splits_bigord": ("c3_nonenv", (3, False)), "splits_big": ("c3", (3, None))}


def _cv_main():
    paths, opt, drop = _args()
    model = opt.get("model", "gbm")
    run = f"{model}_drop-{'+'.join(drop) or 'none'}_{time.strftime('%m%d-%H%M%S')}"
    T = load_tables(paths)
    t0 = time.time()
    oof = cv(T, 5, drop, model)
    took = time.time() - t0
    assert np.isfinite(oof).all(), "non-finite out-of-fold scores"
    v17 = T["X"][:, features.FEATURES.index("v17")]
    metrics = {"cv_seconds": took}
    print(f"{run}: {len(np.unique(T['qid']))} queries, {len(oof):,} rows, cv {took:.0f} s")
    for name in np.unique(T["split"]):
        qm = T["split"] == name
        S, rows = subset(T, qm), qm[T["qid"]]
        a, b = mrr_by(S, v17[rows]), mrr_by(S, oof[rows])
        print(f"  {name:14s} v17    " + "  ".join(f"{k} {x:.4f}" for k, x in a.items()))
        print(f"  {name:14s} ranker " + "  ".join(f"{k} {x:.4f}" for k, x in b.items()))
        ra, rb = rr_by_query(S, v17[rows]), rr_by_query(S, oof[rows])
        reads = [("c1", S["qclass"] == 1)]
        if name in READ:
            key, (c, env) = READ[name]
            m = S["qclass"] == c
            reads.append((key, m & (S["env"] == env) if env is not None else m))
        for key, m in reads:
            d, lo, hi = paired_bootstrap(ra[m], rb[m])
            print(f"  {name:14s} {key:10s} delta {d:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  (n {m.sum()})")
            metrics.update({f"{name}.{key}.delta": d, f"{name}.{key}.lo": lo, f"{name}.{key}.hi": hi})
        metrics.update({f"{name}.v17.{k}": x for k, x in a.items()})
        metrics.update({f"{name}.ranker.{k}": x for k, x in b.items()})
    out = f"{WORK}/rank_oof/{run}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, oof=oof, tables=np.array(paths), model=model, drop=np.array(drop, dtype="U16"))
    print(f"wrote {out}")
    log_run(run, {"tables": [os.path.basename(p) for p in paths], "model": model, "drop": drop,
                  "gbm": GBM if model == "gbm" else LAMBDARANK if model == "lambdarank" else "logreg C=1",
                  "seeds": SEEDS, "n_rerank": features.N_RERANK, "table_version": TABLE_VERSION},
            metrics, artifacts=(out,))


def _fit_main():
    paths, opt, drop = _args()
    model = opt.get("model", "gbm")
    T = load_tables(paths)
    models = fit(T, drop_groups=drop, model=model)
    export(models, used_features(drop), opt["out"], drop_groups=drop)
    r = Ranker.load(opt["out"])
    err = np.abs(r.score(T["X"][:5000]) - score(models, T["X"][:5000][:, columns(drop)], model)).max()
    assert err < 1e-9, err
    print(f"wrote {opt['out']}: {len(models)} models, {len(r.tree_model)} trees, "
          f"numpy vs sklearn max |diff| {err:.1e}")


def _auc_main():
    paths, _, _ = _args()
    T = load_tables(paths)
    for name in np.unique(T["split"]):
        S = subset(T, T["split"] == name)
        aucs = sorted(((pool_auc(S, f), f) for f in features.FEATURES), reverse=True)
        print(f"{name}: within-query AUC of the answer against the other candidates")
        for a, f in aucs:
            print(f"  {f:12s} {a:.3f}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--table" in sys.argv:
        _table_main()
    elif sys.argv[1:2] == ["--cv"]:
        _cv_main()
    elif sys.argv[1:2] == ["--fit"]:
        _fit_main()
    elif sys.argv[1:2] == ["--auc"]:
        _auc_main()
