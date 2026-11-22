"""Helpers for the golden tier.

Each case runs a real entry point in a subprocess with a fully explicit env,
because the modules read env vars at import time. The env is built from a
minimal base, never from os.environ.copy(), so nothing leaks in from the
caller's shell.
"""

import hashlib
import os
import subprocess
import time
from pathlib import Path

import numpy as np

MOD = Path(__file__).resolve().parent.parent
REPO = MOD.parent
PY = "/Users/frustrated-dev/Developer/Personal/python_projects/kaggle-competitions/kaggle-competitions/.venv/bin/python"
D = "/Users/frustrated-dev/Developer/Personal/python_projects/kaggle-competitions/kaggle-competitions/MoleculeID-from-Mass-Spectra/data"
K = "/Users/frustrated-dev/Developer/Personal/python_projects/kaggle-competitions/kaggle-competitions/.claude/worktrees/raasta-b-field/MoleculeID-from-Mass-Spectra/kernel/code"
GUARD = str(REPO / "guard.sh")

# The kernel's v17 env, shared by the submit cases only.
SUBMIT_ENV = {
    "PRIOR": "syn",
    "PRIOR_W": "0.5",
    "PRIOR_GATE": "0.8",
    "PRIOR_META": f"{K}/coconut_meta.npz",
    "FP_MODEL": ",".join(
        f"{K}/{n}" for n in ("fpmodel.pt", "fpmodel_s1.pt", "fpmodel_s2.pt", "fpmodel_s3.pt")
    ),
    "SHIFT_W": "16",
    "SHIFT_P": "3",
    "SHIFT_SIM": "entropy",
    "CASMI_EXT": f"{D}/ext",
}

# Keys that belong to the submit set and must never reach dilute_v17.
SUBMIT_ONLY_KEYS = ("CASMI_EXT", "PRIOR_META", "SHIFT_W", "SHIFT_P", "SHIFT_SIM")

_DILUTE_EXTRA = {
    "FRACS": "1",
    "GAMMA": "32",
    "PRIOR": "syn",
    "PRIOR_WS": "0.5",
    "PRIOR_GATES": "0.8",
    "SHIFT_WS": "16",
    "C2_BANNED": "1",
    "SPLITS": f"{D}/splits_big.npz",
    "FP_MODEL": ",".join(f"{D}/fpmodel_big_rank_s{i}.pt" for i in range(4)),
    "DUMP": f"{D}/golden/dilute_v17.npz",
}

CASES: dict[str, dict] = {
    "submit_v17": {"kind": "submit", "extra": {}, "timeout": 3600},
    "submit_v19": {
        "kind": "submit",
        "extra": {"CHANNEL_OUT": f"{D}/ext:{D}/ext_pubpop2"},
        "timeout": 3600,
    },
    "submit_rankby": {"kind": "submit", "extra": {"RANK_BY": "v17"}, "timeout": 3600},
    "dilute_v17": {"kind": "dilute", "extra": _DILUTE_EXTRA, "timeout": 7200},
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_digest(path) -> str:
    """sha256 over sorted keys with each array's dtype, shape and bytes.

    Zip member timestamps are not part of the digest, so a re-save of the
    same arrays gives the same value.
    """
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for key in sorted(z.files):
            a = np.ascontiguousarray(z[key])
            h.update(key.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def golden_env(**extra) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "MIN_AVAIL_MB": "2500",
    }
    env.update({k: str(v) for k, v in extra.items()})
    return env


def case_env(name: str) -> dict[str, str]:
    case = CASES[name]
    if case["kind"] == "dilute":
        # dilute_v17 takes nothing from the submit set.
        return golden_env(CASMI_DATA=D, PYTHONPATH=str(MOD), **case["extra"])
    return golden_env(CASMI_DATA=D, PYTHONPATH=str(MOD), **SUBMIT_ENV, **case["extra"])


def _command(name: str) -> list[str]:
    if CASES[name]["kind"] == "dilute":
        tail = [PY, "-P", "dilute.py"]
    else:
        out = f"{D}/golden/{name}.csv"
        tail = [PY, "-P", "-c", f"import fpsearch; fpsearch.submit('{out}')"]
    return ["caffeinate", "-ims", GUARD, *tail]


def run_case(name: str) -> str:
    """Run one case in a subprocess and return its digest."""
    case = CASES[name]
    Path(D, "golden").mkdir(parents=True, exist_ok=True)
    log = Path(D, "golden", f"{name}.log")
    out = Path(case["extra"]["DUMP"] if case["kind"] == "dilute" else f"{D}/golden/{name}.csv")
    out.unlink(missing_ok=True)  # never digest a stale output from an earlier run
    with open(log, "w") as f:
        proc = subprocess.run(
            _command(name),
            cwd=MOD,
            env=case_env(name),
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=case["timeout"],
        )
    if proc.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"{name} exited {proc.returncode}; log tail:\n{tail}")
    if case["kind"] == "dilute":
        return npz_digest(case["extra"]["DUMP"])
    return sha256_file(f"{D}/golden/{name}.csv")


def timed_run_case(name: str) -> tuple[str, float]:
    """run_case plus wall time in minutes, rounded to 0.1."""
    t0 = time.monotonic()
    digest = run_case(name)
    return digest, round((time.monotonic() - t0) / 60, 1)
