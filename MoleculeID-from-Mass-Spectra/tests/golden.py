"""Helpers for the golden tier.

Each case runs a real entry point in a subprocess with a fully explicit env,
because the modules read env vars at import time. The env is built from a
minimal base, never from os.environ.copy(), so nothing leaks in from the
caller's shell.

What this gate does not see:

1. Every case reads warm caches in the shared data dir D: key14.npz
   (library.canonical_keys), structures.npz, the pool dirs (the build branch of
   external.merge) and shift_cache. The kernel builds these cold, so changes to
   canonical_keys, the pool build or the shift-cache build are NOT covered.
   run_case fails if a case writes to any of these, so drift cannot go unseen.
2. fpsearch.main's scoring loop is not run by any case.
3. dilute_v17 pins each answer's reciprocal rank only (DUMP holds rr and keys),
   not the order of non-answer candidates.
4. The gate is `pytest -q && pytest -m golden -q`: the fast tier holds
   test_golden_cases_shape, which `-m golden` does not run.
"""

import fcntl
import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MOD = Path(__file__).resolve().parent.parent
REPO = MOD.parent
# The interpreter running pytest also runs the cases.
PY = sys.executable
# The shared data dir lives in the main checkout, which every worktree finds through
# git's common dir. A worktree's own data/ holds job scripts only.
_MAIN = Path(subprocess.run(["git", "-C", str(MOD), "rev-parse", "--path-format=absolute",
                             "--git-common-dir"], capture_output=True, text=True,
                            check=True).stdout.strip()).parent
D = str(_MAIN / MOD.name / "data")
GUARD = str(REPO / "guard.sh")

# Outputs and logs live in a per-worktree dir so two worktrees do not clobber each other.
OUT_DIR = Path(D, "golden", MOD.parent.name)
LOCK_PATH = Path(D, "golden", ".lock")

# Gate inputs under D, pinned by sha256 so a failure is attributed to inputs, not code.
INPUTS: dict[str, str] = {
    "fpmodel_rank.pt": "dc504f9ce7861c0682be3f8cc24b4494099686fbee5242d25d6e440c2a103fc8",
    "fpmodel_rank_s1.pt": "90bf20e2e43e59f8cb39592c28b63b444688408d174fd0091a7635986a85cd1c",
    "fpmodel_rank_s2.pt": "35c719d979f3a57464d636bcb8e54b9ea8544d1feb3e506c52920cd76dd2e187",
    "fpmodel_rank_s3.pt": "2c90a4c7c3e8e0ff891b84b42d807f0623af082ca7313a8c44fdee4d69a6e89b",
    "coconut_meta/meta.npz": "46494de00a0fbaf8f735265192c3ef7a5385fae2d8405b1d7c473eb2f1f42a81",
}
_verified_inputs: set[str] = set()

# The kernel's v17 env, shared by the submit cases only.
SUBMIT_ENV = {
    "PRIOR": "syn",
    "PRIOR_W": "0.5",
    "PRIOR_GATE": "0.8",
    "PRIOR_META": f"{D}/coconut_meta/meta.npz",
    "FP_MODEL": ",".join(
        f"{D}/{n}" for n in ("fpmodel_rank.pt", "fpmodel_rank_s1.pt", "fpmodel_rank_s2.pt", "fpmodel_rank_s3.pt")
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
    "DUMP": str(OUT_DIR / "dilute_v17.npz"),
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


def output_path(name: str) -> Path:
    """The one place a case's output file is named: command, DUMP env, unlink and digest all use it."""
    case = CASES[name]
    if case["kind"] == "dilute":
        return Path(case["extra"]["DUMP"])  # the env var the module writes to
    return OUT_DIR / f"{name}.csv"


def _command(name: str) -> list[str]:
    if CASES[name]["kind"] == "dilute":
        tail = [PY, "-P", "dilute.py"]
    else:
        tail = [PY, "-P", "-c", f"import fpsearch; fpsearch.submit('{output_path(name)}')"]
    return ["caffeinate", "-ims", GUARD, *tail]


def _log_tail(log: Path) -> str:
    return "\n".join(log.read_text(errors="replace").splitlines()[-30:])


def run_in_new_session(cmd, log: Path, label: str, timeout: float, **popen_kw) -> int:
    """Run cmd with output to log; on timeout SIGKILL its whole process group and raise."""
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, **popen_kw)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    raise RuntimeError(f"{label} timed out after {timeout}s; log tail:\n{_log_tail(log)}")


def _verify_inputs() -> None:
    for rel, want in INPUTS.items():
        if rel in _verified_inputs:
            continue
        path = Path(D, rel)
        if not path.is_file() or sha256_file(path) != want:
            raise RuntimeError(f"golden input changed: {path}")
        _verified_inputs.add(rel)


def _cache_snapshot() -> dict:
    """Entries of the pool dirs and mtimes of the index files, which the goldens only read."""
    snap = {}
    for sub in ("pool", "shift_cache"):
        p = Path(D, sub)
        snap[sub] = sorted(e.name for e in p.iterdir()) if p.is_dir() else []
    for f in ("key14.npz", "structures.npz"):
        p = Path(D, f)
        snap[f] = p.stat().st_mtime_ns if p.exists() else None
    return snap


def _assert_caches_unchanged(name: str, before: dict, after: dict) -> None:
    changes = []
    for key in ("pool", "shift_cache"):
        new = sorted(set(after[key]) - set(before[key]))
        if new:
            changes.append(f"new entries in {key}/: {new}")
    for key in ("key14.npz", "structures.npz"):
        if before[key] != after[key]:
            changes.append(f"{key} mtime changed")
    if changes:
        raise RuntimeError(
            f"case {name} built a cache entry: {'; '.join(changes)}; the goldens only pin warm-cache behaviour"
        )


def run_case(name: str) -> str:
    """Run one case in a subprocess and return its digest."""
    case = CASES[name]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another golden run holds D/golden/.lock") from None
        if case["kind"] == "submit":
            _verify_inputs()
        log = OUT_DIR / f"{name}.log"
        out = output_path(name)
        out.unlink(missing_ok=True)  # never digest a stale output from an earlier run
        before = _cache_snapshot()
        code = run_in_new_session(_command(name), log, name, case["timeout"], cwd=MOD, env=case_env(name))
        if code != 0:
            raise RuntimeError(f"{name} exited {code}; log tail:\n{_log_tail(log)}")
        _assert_caches_unchanged(name, before, _cache_snapshot())
    return npz_digest(out) if case["kind"] == "dilute" else sha256_file(out)


def timed_run_case(name: str) -> tuple[str, float]:
    """run_case plus wall time in minutes, rounded to 0.1."""
    t0 = time.monotonic()
    digest = run_case(name)
    return digest, round((time.monotonic() - t0) / 60, 1)
