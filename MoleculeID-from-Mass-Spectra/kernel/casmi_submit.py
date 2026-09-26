"""CASMI 2026 submission kernel: build the index, then search + analog propagation.

This competition is kernels-only (`is_kernels_submissions_only: True`), so a CSV
upload is rejected with a 400 no matter how well-formed it is. The submission has to
be produced by a notebook with no internet access, inside a 9-hour budget.

Nothing is precomputed and shipped as a dataset. The whole index is rebuilt here from
train.parquet, which costs about 50 minutes of the 9 hours:

    library.build()      row-group-streamed peak index      ~25 min, 4.9 GB peak
    canonical_keys()     RDKit key14 per structure          ~20 min
    analog.build()       Morgan fingerprint + exact mass    ~2 min
    analog.submit()      search + propagate over 400 mols   ~1 min

Derived artifacts go to /tmp, not /kaggle/working: the index alone is 1.3 GB and
would otherwise be committed as notebook output.

The code comes from an attached dataset rather than being pasted in, so there is one
copy of it and it cannot drift from what the held-out evaluation measured.
"""
import os
import sys
import time

# The competition mounts at a different path depending on whether the kernel was
# created in the browser or pushed through the API, so resolve it rather than guess.
COMP = next(p for p in ("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra",
                        "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra")
            if os.path.exists(f"{p}/test.parquet"))
CODE = next(p for p in ("/kaggle/input/casmi26-molecule-id-code",)
            if os.path.exists(f"{p}/library.py"))

os.environ["CASMI_DATA"] = COMP          # read-only competition files
os.environ["CASMI_WORK"] = "/tmp/casmi"  # writable scratch, not committed as output
os.makedirs("/tmp/casmi", exist_ok=True)
sys.path.insert(0, CODE)

import library            # noqa: E402
import analog             # noqa: E402

t0 = time.time()
step = lambda s: print(f"[{(time.time() - t0) / 60:5.1f} min] {s}", flush=True)

step(f"competition at {COMP}")
library.build()
step(f"index built: {library.SHARDS}")

ix = library.load()                      # also builds and caches the canonical keys
step(f"{len(ix['npk']):,} spectra, {len(set(ix['inchikey14'].tolist())):,} structures")
del ix

analog.build()
step("fingerprints and exact masses built")

analog.submit("/kaggle/working/submission.csv")
step("submission written")

import pandas as pd                      # noqa: E402
sub = pd.read_csv("/kaggle/working/submission.csv")
assert len(sub) == 400 and sub.molecule_id.nunique() == 400, len(sub)
assert sub.smiles.str.split(";").str.len().max() <= 25
print(sub.head(3).to_string())
step("done")
