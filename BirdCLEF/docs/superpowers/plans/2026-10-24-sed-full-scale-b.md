# SED Full-Scale B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain Phase B's SED on all 127,104 unlabelled windows instead of 12,000, changing nothing else. This tests whether data scale is why the SED branch has not paid.

**Architecture:**
- **Targets.** `teacher.py` already has Perch's scores for every unlabelled window. A new `--targets-full` mode quantile-maps them onto the public teacher's values for the same windows, as Phase B did.
- **Audio cache.** The 127,104 windows of audio are ~41 GB, too big to upload. The training kernel decodes them itself from the competition mount, onto scratch disk, using the pilot's own window code made parallel. It links the pilot cache's focal and labelled files beside them, then runs `sed_train.py` unchanged with `--labels Bfull`.
- **Export and blend.** These are Phase B's.

**Tech Stack:**
- Data and inference: Perch v2 ONNX + onnxruntime 1.24.4 (CPU), scikit-learn, soundfile
- Training: PyTorch 2.11 / timm on a Kaggle T4
- Kaggle CLI 2.2.4

**Spec:** `BirdCLEF/docs/superpowers/specs/2026-10-20-sed-branch-design.md` (sections 1, 4 and 5).
- RESULTS §16 holds the pilot and Phase B results and names data scale as the next suspect.
- The user chose this run over closing BirdCLEF on 2026-10-24 ("karte hain"), on the condition of a stop rule.

**Scope:** One fold, as the pilot did. More folds or an ensemble get their own plan, and only if the decision rule says go.

## Global Constraints

**Carried over.** Everything from the pilot and Phase B plans still holds:
- `$PY`
- the mel front-end, backbone and recipe
- the leakage rule: the 66 labelled files contribute only true labels
- the kernel limits, the staging rules (`cp -X`, delete `._*`, short SSD paths), and `guard.sh`

**Worktree.** `.claude/worktrees/birdclef-sed-full`, branch `worktree-birdclef-sed-full`. `BirdCLEF/data` and `BirdCLEF/models` are symlinks to the SSD.

**One-variable rule against Phase B.** Only the pseudo windows change: from the pilot's 12,000 to all 127,104 unlabelled windows, with no filter. Everything else is Phase B's:
- how targets are made: per class, A's values re-assigned in the order of the Perch kernel's scores, with tied scores sharing a mean; the mapping now runs over the 127,104 windows
- the focal and labelled data, fold 0, seed, 20 epochs, batch 64, recipe, T4
- W_SED = 0.5, the smoother

**Decoded windows equal the pilot's.** For every row the pilot cache also holds, the in-kernel decoder writes bit-identical int16 values.

**No silent subsetting.** If scratch disk cannot hold all 127,104 windows, the kernel stops before decoding and logs free versus needed bytes.

**Budget.**

| item | cost |
|---|---|
| training | ~133k samples per epoch at ~240 samples/s, so ~9 min per epoch, ~3.1 h for 20 epochs |
| decoding | ~20–40 min |
| total | ~4 h of the ~30 h weekly GPU quota, inside Kaggle's 12 h session limit |

**Kaggle.** A new kernel's slug comes from its title, so poll the slug in the push output. Never read `~/.kaggle/access_token`.

**Submissions.** Exactly two: SED alone and the blend, both at W_SED = 0.5. The user's approval of this plan is the approval for those two, and for the dataset and kernel pushes the plan names. Nothing else is submitted without asking.

**Decision rule** (agreed 2026-10-24). LB noise is about ±0.003–0.005. Pilot B scored 0.87352 for SED alone and 0.91027 for the blend.

| SED alone private | blend private | decision |
|---|---|---|
| ≥ 0.90 | ≥ 0.9318 | **go**: a new plan for more folds / ensemble |
| < 0.90 | < 0.9318 | **stop**: BirdCLEF closes at Perch's 0.92681 |
| any other mix | | report both to the user, who decides |

## Review Focus

1. **Decoded windows differing from the pilot's** (start offset, int16 rounding, stereo-to-mono). Expected: identical bytes. Test: Task 2 decodes 3 pilot files with 2 workers and compares them with the pilot's `pseudo_audio.npy` rows.
2. **Bfull rows not matching `full_meta.csv` rows.** Expected:
   - row i of the targets is the teacher's and the public CSV's row for `full_meta.row_id[i]`
   - `check_b` holds on the real arrays
   - 3 rows checked by an independent lookup
   Test: Task 1's run assertions.
3. **The GPU image lacking what the decoder needs, or its disk being too small.** Expected: the launcher imports `soundfile`, pip-installing it if missing (internet is on). It logs free scratch disk and RAM, and stops before decoding if free < 1.05 × 127,104 × 320,000 bytes. Test: Task 2's `stage_full` selftest on a too-small `need`, plus the Task 3 log.
4. **Parallel workers corrupting the memmap.** Two workers writing the same rows, or a write that is never flushed, must not happen. Expected: each file is written by exactly one worker, and the parent re-opens and checks the file. Test: Task 2's 2-worker bit-exact test.
5. **Training reading the pilot's 12k audio instead of the full cache.** The `birdclef-sed-cache` dataset also holds a `pseudo_audio.npy`. Expected: `SED_CACHE` points at the staged scratch directory, and the log says `pseudo 127,104` and names `pseudo_targets_Bfull.npy`. Test: Task 2's staged-directory selftest, plus the Task 3 log check.

---

### Task 1: Targets for all unlabelled windows (`teacher.py --targets-full`)

**Files:**
- Modify: `BirdCLEF/teacher.py`:
  - split `targets()` into `scores()` and `diagnose()`, with no behaviour change to `--targets`
  - add `full_meta()`, `targets_full()` and the `--targets-full` CLI mode

**Interfaces:**
- Consumes:
  - `targets_b(scores, A)` and `check_b(scores, A, B)` from `teacher.py`
  - `targets_for(csv_path, row_ids, classes) -> float16 [n, 234]` and `PSEUDO_CSV` from `sed_cache.py`
- Produces:
  - `scores() -> (state, P, stems, starts)`: P is [127104, 234], in shard order (files sorted, starts ascending); stems are filenames without `.ogg`.
  - `diagnose(A, B, state, classes, tax) -> None`: prints what `targets()` prints today, plus the mean count of classes > 0.5 per window for A and for B.
  - `full_meta(stems, starts) -> pd.DataFrame`: columns `row_id` (`<stem>_<start+5>`), `filename` (`<stem>.ogg`) and `start_s`, in the input order.
  - Files: `{TARGETS_OUT}/full_meta.csv` and `{TARGETS_OUT}/pseudo_targets_Bfull.npy` (float16, [127104, 234]). Rows match each other one to one. `TARGETS_OUT` = env `TARGETS_OUT` or `/Volumes/Extreme SSD/kaggle/BirdCLEF/jobs/sed-targets`.

- [ ] **Step 1: Write the failing test.** In `selftest()`:

```python
m = full_meta(np.array(["BC_x", "BC_x", "BC_y"]), np.array([0, 5, 0]))
assert m.row_id.tolist() == ["BC_x_5", "BC_x_10", "BC_y_5"]
assert m.filename.tolist() == ["BC_x.ogg", "BC_x.ogg", "BC_y.ogg"]
assert m.start_s.tolist() == [0, 5, 0]
```

- [ ] **Step 2: Run** `MIN_AVAIL_MB=2500 ../guard.sh $PY teacher.py --selftest`. Expected: FAIL with `NameError: name 'full_meta' is not defined`.
- [ ] **Step 3: Implement** `full_meta`, `scores` and `diagnose`. Rewrite `targets()` on top of them and add `targets_full()`. In `targets_full()`:
  - `A = targets_for(PSEUDO_CSV, meta.row_id.tolist(), classes)`
  - `B = targets_b(P, A)`, then `check_b(P, A, B)`
  - check 3 rows (first, middle, last) independently: read those `row_id`s from the CSV with a filtered `pd.read_csv` and compare against `A`
  - save both files, then call `diagnose`
- [ ] **Step 4: Run** the selftest. Expected: PASS, with fidelity still 1.1e-16.
- [ ] **Step 5: Regression check for `--targets`.** Run `--targets` with `SED_CACHE` set to a scratch copy of the cache's `pseudo_meta.csv` and `pseudo_targets_A.npy`. Expected: its `pseudo_targets_B.npy` is byte-equal to the shipped one (`cmp`).
- [ ] **Step 6: Run** `MIN_AVAIL_MB=2500 ../guard.sh $PY -u teacher.py --targets-full`. Expected:
  - shape (127104, 234), and `check_b` passes
  - 3 rows verified
  - the diagnosis printed: Spearman quartiles by route, and classes > 0.5 per window for A and B
- [ ] **Step 7: Upload** a new version of `birdclef-sed-targets`, staged at `jobs/sed-targets`. It holds `pseudo_targets_B.npy`, `pseudo_targets_Bfull.npy` and `full_meta.csv`. Expected: status `ready`, and `kaggle datasets files` lists all three.
- [ ] **Step 8: Commit** `teacher.py` with the message "Full-scale B: Perch-teacher targets for all 127,104 unlabelled windows".

### Task 2: In-kernel window cache and training kernel Bfull

**Files:**
- Modify: `BirdCLEF/sed_cache.py`. Add `windows(path, meta, audio_dir, workers=1)` and `stage_full(out, meta, audio_dir, cache_dir, targets_path, workers, need=None)`. `_windows` becomes a call to `windows(path, meta, f"{DATA}/{sub}")`.
- Create: `BirdCLEF/kernel_sed_train_bfull/sed_train_bfull.py` (launcher).
- Create: `BirdCLEF/kernel_sed_train_bfull/kernel-metadata.json`:
  - id and title `jagjeetsingh23/birdclef26-sed-train-bfull` / "BirdCLEF26 SED train Bfull"
  - competition `birdclef-2026`
  - datasets `birdclef-kernel-code`, `birdclef-sed-cache` and `birdclef-sed-targets`
  - T4, GPU, internet on

**Interfaces:**
- Consumes: `full_meta.csv` and `pseudo_targets_Bfull.npy` (Task 1); `to_windows` (`perch.py`); `to_int16` and `_mono` (`sed_cache.py`).
- Produces:
  - **`windows(path, meta, audio_dir, workers=1) -> None`**:
    - writes an int16 `.npy` of shape [len(meta), 160000], in meta's row order
    - each file group is decoded by exactly one worker (`multiprocessing.Pool`), which writes through `np.load(path, mmap_mode="r+")`
  - **`stage_full(out, meta, audio_dir, cache_dir, targets_path, workers, need=None) -> dict`**:
    - checks first that `shutil.disk_usage(out).free >= need`, where `need` defaults to `int(1.05 * len(meta) * 320_000)`; otherwise it raises `RuntimeError` naming free and needed bytes
    - decodes `out/pseudo_audio.npy`
    - symlinks into `out` the six pilot files: `focal_audio.npy`, `focal_index.csv`, `focal_targets.npy`, `labelled_audio.npy`, `labelled_meta.csv`, `labelled_targets.npy`
    - symlinks `out/pseudo_targets_Bfull.npy` to `targets_path`
    - returns `{"rows", "sec", "free_gb"}`

- [ ] **Step 1: Write the failing tests.** In `sed_cache.selftest()`, using the first 3 files of the pilot's `pseudo_meta.csv` in a temp dir:

```python
meta = pd.read_csv(f"{OUT}/pseudo_meta.csv")
three = meta[meta.filename.isin(meta.filename.unique()[:3])]
windows(f"{d}/w.npy", three.reset_index(drop=True), f"{DATA}/train_soundscapes", workers=2)
pilot = np.load(f"{OUT}/pseudo_audio.npy", mmap_mode="r")
assert np.array_equal(np.load(f"{d}/w.npy"), pilot[three.index.to_numpy()])

try:
    stage_full(f"{d}/s", three.reset_index(drop=True), f"{DATA}/train_soundscapes", OUT, tp, 2, need=10**18)
    raise AssertionError("stage_full must refuse when disk is short")
except RuntimeError as e:
    assert "free" in str(e)

info = stage_full(f"{d}/s", three.reset_index(drop=True), f"{DATA}/train_soundscapes", OUT, tp, 2)
assert info["rows"] == len(three)
assert sorted(os.listdir(f"{d}/s")) == sorted(
    ["pseudo_audio.npy", "pseudo_targets_Bfull.npy", "focal_audio.npy", "focal_index.csv",
     "focal_targets.npy", "labelled_audio.npy", "labelled_meta.csv", "labelled_targets.npy"])
assert np.array_equal(np.load(f"{d}/s/pseudo_audio.npy"), pilot[three.index.to_numpy()])
```

Here `tp` is a small `.npy` written in the temp dir. Create `{d}/s` before the first call.

- [ ] **Step 2: Run** `MIN_AVAIL_MB=2500 ../guard.sh $PY sed_cache.py --selftest`. Expected: FAIL with `NameError: name 'windows' is not defined`.
- [ ] **Step 3: Implement** `windows` and `stage_full`, and rewire `_windows`. The worker function is module-level, so macOS spawn can pickle it. The parent creates the file with `open_memmap(mode="w+")` and closes it before the pool starts.
- [ ] **Step 4: Run** the selftest. Expected: PASS.
- [ ] **Step 5: Write the launcher** `sed_train_bfull.py`. Three env overrides exist only for the local test:

  | env var | default |
  |---|---|
  | `KAGGLE_INPUT` | `/kaggle/input` |
  | `SED_SCRATCH` | `/tmp/sedfull` |
  | `SED_SMOKE=1` | unset; when set, appends `--smoke` |

  The launcher:
  - imports `soundfile`, or pip-installs it
  - locates the code dir (`sed_train.py`), competition dir (`taxonomy.csv`), cache dir (`focal_audio.npy`) and targets dir (`pseudo_targets_Bfull.npy`) with an `os.walk` find under `KAGGLE_INPUT`
  - logs free disk at `SED_SCRATCH`'s parent and `MemTotal` (read `/proc/meminfo` if it exists)
  - calls `stage_full(SED_SCRATCH, pd.read_csv(<targets dir>/full_meta.csv), f"{comp}/train_soundscapes", cache_dir, f"{targets dir}/pseudo_targets_Bfull.npy", os.cpu_count())`, logging rows, seconds and windows per second
  - sets `SED_CACHE=SED_SCRATCH`, `SED_OUT=/kaggle/working` (unless already set), and `sys.argv = ["sed_train.py", "--labels", "Bfull"]` (+ `--smoke`)
  - runs `runpy.run_path(<code>/sed_train.py, run_name="__main__")`
- [ ] **Step 6: Run the launcher locally on a 3-file meta.**
  - Build a fake `KAGGLE_INPUT` in the scratchpad: real directories with symlinked files, since `os.walk` does not follow directory links. It holds the code files, the competition's `taxonomy.csv` plus a `train_soundscapes` dir with the 3 files, the six pilot cache files, and a targets dir with a 3-file `full_meta.csv` and matching `pseudo_targets_Bfull.npy` rows.
  - Set `SED_SCRATCH` and `SED_OUT` inside the scratchpad, and `SED_SMOKE=1`.
  - Expected: it stages the directory, and the smoke run of `sed_train.py` exits 0. It prints `pseudo targets: <scratch>/pseudo_targets_Bfull.npy` and `pseudo 36`: 3 files × 12 windows, taken from the real Task 1 outputs.
- [ ] **Step 7: Upload** a new version of `birdclef-kernel-code`, staged at `jobs/kcode`, with `sed_cache.py` added. Expected: `ready`, with `sed_cache.py` in the file list.
- [ ] **Step 8: Commit** `sed_cache.py` and `kernel_sed_train_bfull/` with the message "Full-scale B: decode the unlabelled windows inside the training kernel; kernel Bfull".

### Task 3: Training run, checkpoint, export

**Files:** none in the repo. Outputs go to `jobs/sed-out-bfull/fold0/` and `jobs/sed-models/sed_fold0_Bfull.onnx`.

**Interfaces:**
- Consumes: kernel `birdclef26-sed-train-bfull` (Task 2), and the datasets of Tasks 1–2.
- Produces: `sed_fold0_Bfull.onnx`, in a new version of `birdclef-sed-models` that holds all three ONNX files.

- [ ] **Step 1: Push** `kernel_sed_train_bfull`. Read the slug from the push output and poll that slug in the background.
- [ ] **Step 2: When the kernel finishes, download** to `jobs/sed-out-bfull/fold0/`. The CLI returns the log only after a run ends, with COMPLETE or ERROR. An early ERROR is most likely the disk check.
  - **If the disk check stopped the run:** record the free bytes, and stop and ask the user. A subset changes the experiment.
- [ ] **Step 3: Check the log.** Expected:
  - free disk ≥ 42.7 GB, and decode rows 127,104
  - `pseudo targets: /tmp/sedfull/pseudo_targets_Bfull.npy`
  - `train … (pseudo 127,104, …)`
  - epochs 1+ have `loader_wait_frac` < 0.1. If the loader starved (> 0.3), ledger it. The cost was time, not correctness.

  Record the val_auc trajectory, the seconds per epoch, and the decode time.
- [ ] **Step 4: Export and selftest.** Export `sed_fold0_Bfull.onnx`, then run `SED_CKPT=.../sed-out-bfull/fold0/sed_fold0_ema.pt $PY sed_infer.py --selftest`. Expected: PASS, with GPU predictions reproduced on CPU (< 2e-3). The three ONNX files must have distinct md5s.
- [ ] **Step 5: Upload** a new version of `birdclef-sed-models`. Expected: `ready`, with three ONNX files listed.

### Task 4: Kernel runs, LB, decision, write-up

**Files:**
- Modify: `BirdCLEF/kernel_blend/birdclef_blend_submit.py`. Set `SED_ONNX = "sed_fold0_Bfull.onnx"` and update its comment to name the earlier files.
- Modify: `BirdCLEF/RESULTS.md`:
  - add a new §17, "Full-scale B: does data scale make the SED pay?"
  - renumber Files to §18, and add `kernel_sed_train_bfull/` and the `sed_cache.py` role there

- [ ] **Step 1: Push the blend kernel with `MODE="blend"`.** Wait for COMPLETE. Expected: the log shows `SED_ONNX=sed_fold0_Bfull.onnx` and `smoke test … passed (blend)`.
- [ ] **Step 2: Submit** it with the message "SED full-scale B: 127k Perch-teacher windows, w=0.5".
- [ ] **Step 3: Push with `MODE="sed"`.** Wait for COMPLETE, then submit with the message "SED full-scale B alone". Restore `MODE="blend"` in the repo. Poll until both submissions are scored.
- [ ] **Step 4: Decide** by the decision rule. Write §17 with:
  - the table: Perch, pilot A, pilot B and full B; SED alone and blend; public and private
  - the scale effect, full B minus pilot B
  - the val_auc trajectory, and the decode and GPU time
  - the Task 1 calibration diagnosis
  - the decision
- [ ] **Step 5: Commit** with the message "SED full-scale B: <blend private> private, <decision>". Update the memory note `birdclef-perch-v2-gap.md`. On **stop**, say there that BirdCLEF is closed at 0.92681 and why.
