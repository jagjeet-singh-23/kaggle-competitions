# Refactor Phase 0: Safety Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the tooling and tests that will fail if a later refactor phase changes any output by even one byte, then fix the real bugs the linter finds under that net.

**Architecture:** There are two test tiers.
- *Fast tier:* the existing `selftest()` functions plus unit tests of pure functions. Runs by default in seconds.
- *Golden tier* (`-m golden`, opt-in, about 40 min): characterization tests. Each runs a real entry point (`fpsearch.submit`, `dilute.py`) in a subprocess with explicit env, because modules read env at import. It compares a digest of the output against `tests/golden_manifest.json`.

Two goldens are anchored to shipped artifacts (v17's tested CSV and v19's channel-on CSV), not only to the base commit. Module code stays untouched until the goldens are recorded (Task 3). Task 4 is the only code change.

**Tech Stack:** Python 3.12.12 (root `.venv`, uv-managed), pytest, ruff 0.17.6 (Homebrew), numpy, pandas.

**Spec:** the phase list agreed with the user on 2026-11-21:
- **Phase 0:** pyproject, ruff, pytest, golden-file tests.
- **Phase 1:** Config dataclass.
- **Phase 2:** a single scoring core in place of the duplicates.
- **Phase 3:** types/dataclasses/pathlib/logging, plus parallel `canonical_keys` and shift search.

Every phase must be verified byte-identical. This plan is Phase 0. Phases 1–3 get their own plans once this net exists.

## Global Constraints

- Base commit `5b83c6f` on branch `worktree-refactor`, worktree `.claude/worktrees/refactor`. Paths below are relative to `MoleculeID-from-Mass-Spectra/` (written `MOD`); `<repo>` is the main checkout's root.
- Interpreter: `<repo>/.venv/bin/python` (written `PY`). It is uv-managed: run `uv pip install --dry-run` first and use no pip.
- Shared data: `CASMI_DATA=<repo>/MoleculeID-from-Mass-Spectra/data` (written `D`). Golden outputs are written only under `$D/golden/` (git-ignored).
- Model and meta files for the submit goldens come from `K=<repo>/.claude/worktrees/raasta-b-field/MoleculeID-from-Mass-Spectra/kernel/code`. Only files are read from there. The code under test is always `MOD`: `PYTHONPATH=MOD`, cwd `MOD`, `python -P`.
- Equality is exact: sha256 for CSVs, and per-array equality for `.npz`. There are no tolerances.
- Long runs go through `caffeinate -ims ../guard.sh` with `MIN_AVAIL_MB=2500`, one golden at a time. Never weaken `guard.sh`.
- Code comments and docstrings are in English.
- Commit steps run only after the user's go-ahead for that task.

## Review Focus

1. **Env leaking between tests.** Modules read env at import, so an in-process import in the fast tier fixes `GAMMA`, `CHANNEL_OUT` and the rest for the whole session. Goldens must therefore run in subprocesses with a fully explicit env, not `os.environ.copy()`. Task 3's `golden_env()` starts from a minimal base (`PATH`, `HOME`, `MIN_AVAIL_MB`), and `test_golden_env_is_minimal` pins this.
2. **Nondeterminism.** Torch CPU threads or dict ordering could make a golden flaky, and a flaky golden is worse than none. Task 3 records every golden twice and requires identical digests before writing the manifest.
3. **Stale anchors.** A golden that only matches itself would bless a bug already present at the base. G1 and G2 must match the shipped files' hashes, which are hard-coded in the test rather than in the manifest, so `GOLDEN_UPDATE=1` cannot overwrite them.
4. **Shared cache drift.** Pool and shift caches live in shared `$D`, keyed by settings hash. A golden run must hit the same cache dirs as production runs. The G1 log must show `external pool cached at …/pool/5da426c00a80ea3f`; a new hash means the env differs from the kernel's.
5. **Lint "fixes" that change behaviour.** Adding `zip(strict=True)` (B905) raises where the old code silently truncated. Removing an import can drop a side effect. Task 4 leaves B905 ignored until Phase 3 and proves every edit against the goldens.

---

### Task 1: Tooling (pyproject, pytest, ruff config) with the selftests as the first tests

**Files:**
- Create: `MOD/pyproject.toml`
- Modify: `requirements.txt` (repo root): pin pytest under a "MoleculeID tests" comment
- Create: `MOD/tests/test_selftests.py`

**Interfaces:**
- Produces: `pytest` run from `MOD` collects `tests/`, with `MOD` on `sys.path`. The marker `golden` is registered and deselected by default.

- [ ] **Step 1: Write `pyproject.toml`.** It is tool config only, with no `[build-system]`: the modules stay flat files because the Kaggle kernel ships them as a flat dataset folder.
  - `[project] name = "casmi26-molecule-id"`, `requires-python = ">=3.12"`.
  - `[tool.pytest.ini_options]`: `pythonpath = ["."]`, `testpaths = ["tests"]`, `markers = ["golden: byte-identity tests against recorded outputs (~40 min)"]`, `addopts = "-m 'not golden'"`.
  - `[tool.ruff]`: `target-version = "py312"`, `line-length = 120` (the current max line is 120).
  - `[tool.ruff.lint]`: `select = ["E9", "F", "B"]`, `ignore = ["B905"]`, with the comment `# B905 adds zip(strict=True), a behaviour change; Phase 3`.
- [ ] **Step 2: Install pytest.** Run `uv pip install --dry-run --python $PY pytest`, show the resolved version, then install it and pin that exact version in `requirements.txt`.
- [ ] **Step 3: Write `tests/test_selftests.py`.** It has one test per existing selftest, each calling the module's own function:
  - `test_formula_selftest` → `formula.selftest()`
  - `test_shifted_selftest` → `shifted.selftest()`
  - `test_library_selftest` → `library.selftest()`
  - `test_features_selftest` → `features.selftest()`
- [ ] **Step 4: Run** `cd MOD && $PY -m pytest -q`. Expected: `4 passed`. Then run `ruff check .`. Expected: it reads `pyproject.toml` and reports the Task 4 baseline: F401 ×4, B023 ×4, B007 ×3, F821 ×7, and no B905.
- [ ] **Step 5: Commit** `pyproject.toml`, `tests/test_selftests.py` and `requirements.txt` (after the user's OK).

### Task 2: Unit tests for the pure functions the scoring core will be built from

**Files:**
- Create: `MOD/tests/test_units.py`

**Interfaces:**
- Consumes: `fpsearch.slot_merge(base, ch, slots, n=25) -> list`, `fpsearch._once(f) -> callable`, `fpsearch.normalise(t, how) -> np.ndarray`, `shifted.analog_term(cand_fp, analog_fp, sims, p=3.0) -> np.ndarray`, `metric.reciprocal_rank(pred_smiles, answer_smiles) -> float`, `metric.mrr(preds, answers) -> float`, `metric.validate(df) -> True`.

These pin behaviour that Phase 2 will move. They must pass on the base code unchanged, so there is no red step: the code already exists, and these are characterization tests.

- [ ] **Step 1: Write the tests** with these exact assertions:
  - **`test_slot_merge`**:
    - `slot_merge(list("ABC"), list("XY"), (2, 4)) == list("AXBYC")`
    - `slot_merge(list("ABC"), [], (2, 4)) == list("ABC")`
    - `slot_merge(list("A"), list("XYZ"), (4,)) == list("AXYZ")`
    - `slot_merge([], list("XY"), (2, 4, 6, 8, 10, 24)) == list("XY")`
    - `slot_merge(list("ABCDEFGHIJ"), list("XYZ"), (2, 4, 6), n=5) == list("AXBYC")`
  - **`test_once_calls_f_once`**: a counter-incrementing `f`; `g = _once(f)`; `g(); g()` → counter is 1, and both calls return the same object.
  - **`test_normalise`**:
    - `normalise(t, "")` returns `t` itself (`is`);
    - `normalise(np.array([1., 2., 3.]), "rank")` → `[0, .5, 1]`;
    - `normalise(np.array([5.]), "z")` → `[0.]`.
  - **`test_analog_term`**:
    - `analog_term([[255]], [[255], [15]], [0.5, 1.0], p=3)` → `[0.5]` (tanimoto 1 × 0.125 against 0.5 × 1);
    - `analog_term([[255]], [], [], p=3)` → `[0.]`.
  - **`test_reciprocal_rank`**:
    - `reciprocal_rank(["C", "OCC"], "CCO") == 0.5` (same molecule, different SMILES);
    - `reciprocal_rank(["C"] * 25 + ["CCO"], "CCO") == 0.0` (rank 26 is past K = 25);
    - `mrr({"a": ["CCO"]}, {"a": "CCO", "b": "C"}) == 0.5`.
  - **`test_validate_rejects`**: `pytest.raises(AssertionError)` for a row with 26 `;`-joined guesses, and for a duplicated `molecule_id`.
- [ ] **Step 2: Run** `$PY -m pytest -q`. Expected: `10 passed`. If one fails, the test is wrong, not the code (the base is the shipped code), so fix the expectation and note it in the task report.
- [ ] **Step 3: Commit** `tests/test_units.py` (after the user's OK).

### Task 3: Golden tier: record four outputs from the base and pin them

**Files:**
- Create: `MOD/tests/golden.py` (helpers), `MOD/tests/test_golden.py`, `MOD/tests/golden_manifest.json`

**Interfaces:**
- Produces:
  - `golden.sha256_file(path) -> str`
  - `golden.npz_digest(path) -> str`: sha256 over sorted keys, each key's dtype, shape and bytes, so zip timestamps do not matter.
  - `golden.golden_env(**extra) -> dict[str, str]`
  - `golden.run_case(name) -> str`: runs one case and returns its digest.
  - `CASES: dict[str, dict]` in `golden.py`.
  - Later phases run `$PY -m pytest -m golden` as their identity gate.

The four cases share these settings. The submit env is the kernel's v17 env:
- `PRIOR=syn PRIOR_W=0.5 PRIOR_GATE=0.8 PRIOR_META=$K/coconut_meta.npz`
- `FP_MODEL=$K/fpmodel.pt,$K/fpmodel_s1.pt,$K/fpmodel_s2.pt,$K/fpmodel_s3.pt`
- `SHIFT_W=16 SHIFT_P=3 SHIFT_SIM=entropy CASMI_EXT=$D/ext CASMI_DATA=$D`

| Case | Extra env | Entry point | Output | Expected digest |
|---|---|---|---|---|
| `submit_v17` | none | `fpsearch.submit('$D/golden/submit_v17.csv')` | CSV sha256 | **`2fa67eea992ea4147ea68f85def0d2c4aa9ac7f177697f6a90174f91a2e4c5c9`** (raasta-b-c1 `sub_w16.csv`, v17's tested file) |
| `submit_v19` | `CHANNEL_OUT=$D/ext:$D/ext_pubpop2` | same, `submit_v19.csv` | CSV sha256 | **`2a6cc40956b120c2969ae6899b97977c9ea2e5fca867b9485f9009d8551894e4`** (raasta-b-field `sub_v19_on.csv`) |
| `submit_rankby` | `RANK_BY=v17` | same, `submit_rankby.csv` | CSV sha256 | recorded at base (the `features.molecule_rows` path) |
| `dilute_v17` | `FRACS=1 GAMMA=32 PRIOR=syn PRIOR_WS=0.5 PRIOR_GATES=0.8 SHIFT_WS=16 C2_BANNED=1 SPLITS=$D/splits_big.npz FP_MODEL=$D/fpmodel_big_rank_s0.pt,…_s1.pt,…_s2.pt,…_s3.pt DUMP=$D/golden/dilute_v17.npz`, no `PRIOR_META`/`CASMI_EXT` from the submit set | `python -P dilute.py` | `npz_digest` | recorded at base |

- [ ] **Step 1: Write a failing test of the env contract.** `test_golden_env_is_minimal` (fast tier) checks that `golden_env(X="1")` has exactly the keys `{"PATH", "HOME", "MIN_AVAIL_MB", "X"}`. Run it and confirm it fails with an ImportError, since `tests/golden.py` does not exist yet.
- [ ] **Step 2: Implement `tests/golden.py`.** `run_case` runs `["caffeinate", "-ims", "<repo>/guard.sh", PY, "-P", ...]` with cwd `MOD` and `PYTHONPATH=MOD`, under `check=True` and a 3600 s timeout. It tees stdout to `$D/golden/<case>.log` and returns the digest. Then run the fast tier and expect `11 passed`.
- [ ] **Step 3: Write `tests/test_golden.py`.** It has one `@pytest.mark.golden` test per case, parametrized over `CASES`:
  - The `submit_v17` and `submit_v19` digests are compared against the constants above, hard-coded in the test.
  - The other two are compared against `golden_manifest.json`.
  - With `GOLDEN_UPDATE=1`, a missing manifest entry is written instead of compared. The anchored cases still assert their constants.
- [ ] **Step 4: Record at base,** in the background:
  - Run `GOLDEN_UPDATE=1 $PY -m pytest -m golden -q`.
  - Expected: `4 passed`; `submit_v17` and `submit_v19` match their anchors.
  - The `submit_v17` log shows `external pool cached at …/pool/5da426c00a80ea3f`.
  - Record each case's wall time in the manifest under `"_minutes"`.
- [ ] **Step 5: Check determinism.** Run `$PY -m pytest -m golden -q` again without update. Expected: `4 passed`. A mismatch is a blocker: stop and report which case is flaky.
- [ ] **Step 6: Commit** `tests/golden.py`, `tests/test_golden.py` and `tests/golden_manifest.json` (after the user's OK).

### Task 4: Fix what ruff finds, proven against the net

**Files:**
- Modify:
  - `MOD/dilute.py:23,30` (unused `sys`, `fingerprint`; fpsearch imports fingerprint first, so nothing is lost);
  - `MOD/fpsearch.py:21,31` (unused `defaultdict`, `NBIN`);
  - `MOD/fpsearch.py:422-428`: add `# noqa: B023` with the reason, which the comment above already states: the closures are used within the iteration;
  - the three B007 sites, prefixing the unused loop variable with `_`:
    - `MOD/external.py:214` (`i` → `_i`);
    - `MOD/kernel/casmi_submit.py:31` (`dirnames` → `_dirnames`);
    - `MOD/kernel/pubchem_build.py:25` (`dirnames` → `_dirnames`).
    The goldens don't execute these lines: the pool cache is hit, and the kernel script isn't run. A rename of an unused name is safe on its own.
  - `MOD/pubchem.py:23,28,197-214,312-313`: delete `pilot()` and its `--pilot` flag and docstring lines. It reads the undefined names `nps` and `lens`, so it has raised `NameError` since they were removed, and nothing calls it.

**Interfaces:**
- Consumes: Task 3's golden tier as the gate.

- [ ] **Step 1:** Make the edits above. Each is the smallest change ruff names.
- [ ] **Step 2: Run** `ruff check .`. Expected: `All checks passed!`
- [ ] **Step 3: Run** `$PY -m pytest -q`. Expected: `11 passed`.
- [ ] **Step 4: Run** `$PY -m pytest -m golden -q` in the background. Expected: `4 passed`, with no manifest change.
- [ ] **Step 5: Commit** the modified modules (after the user's OK).

## Follow-up plans (written after Phase 0 lands, each gated by `pytest -m golden`)

- **Phase 1, Config:**
  - Replace the ~90 import-time `os.environ` reads across 15 modules with frozen dataclasses built once from env.
  - Env names stay the same, because the kernel and job scripts set them.
- **Phase 2, one scoring core:**
  - `fpsearch._score`, the inline loop in `fpsearch.main`, `dilute.score` and `features.molecule_rows` each recompute v17's total.
  - Replace them with one core plus per-caller extras.
  - Decide whether `fpsearch.main`'s gamma sweep, which `dilute.py` superseded, is deleted rather than ported. That needs the user's call.
  - Before porting `fpsearch.main`, pin it with a golden or delete it (the user's call).
  - Consider adding a list-level dump to `dilute` as a behaviour-neutral commit, recorded before `dilute.score` is merged.
- **Phase 3, modernise and speed up:**
  - type hints and pathlib, logging instead of print, `zip(strict=True)`;
  - parallel `library.canonical_keys` (about 46 of the kernel's 66 min) and shift search.
  - Its gate also needs a timing check on the kernel.
  - Add a `golden_cold` case before touching `canonical_keys`: `submit_v17` with a temp `CASMI_WORK` (index symlinked), running `analog.build()` then `fpsearch.submit()` as the kernel does, asserting the v17 anchor (about 45-60 min, untested sketch, verify at base first).
  - Repeat the double-run determinism check once parallelism lands.
