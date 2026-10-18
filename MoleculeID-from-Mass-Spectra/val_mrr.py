"""Val pool-MRR / mol-MRR of a model or comma-separated ensemble, on the same val structures
train() holds out (rng 0). SPLITS must be the split the models excluded.
    SPLITS=data/splits_c1cov3.npz ../.venv/bin/python val_mrr.py a.pt[,b.pt] ...
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import fingerprint as F  # noqa: E402
import fpsearch as S  # noqa: E402

drop = np.load(os.environ["SPLITS"], allow_pickle=True)["drop"]
mz, it, pm, rows, fp = F.load_training(drop=drop)
uniq = np.unique(rows)
val = set(np.random.default_rng(0).choice(uniq, max(1, len(uniq) // 20), replace=False).tolist())
va = np.where(np.array([r in val for r in rows]))[0]
pools = F.Pools()
for spec in sys.argv[1:]:
    F.MODEL = spec
    model, bits = S.load_model()
    print(f"{spec}: pool-MRR {F.pool_mrr(model, mz, it, pm, rows, bits, va, pools):.4f}  "
          f"mol-MRR {F.mol_mrr(model, mz, it, pm, rows, bits, va, pools):.4f}", flush=True)
