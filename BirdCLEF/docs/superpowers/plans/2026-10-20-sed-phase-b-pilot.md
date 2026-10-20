# SED Phase B Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the pilot SED with pseudo-labels from our own Perch kernel instead of the public teacher, changing nothing else, and find out whether the teacher is why the pilot was a no-go.

**Architecture:** `teacher.py` runs the Perch kernel's pipeline over all 10,592 unlabelled soundscapes on the Mac. Perch outputs are cached in resumable shards. The kernel's per-class scores are ranks, not probabilities, so each class's ordering comes from Perch while its values come from the public teacher's own values for that class on the same windows (quantile mapping). A and B therefore have identical per-class marginals and differ only in which windows rank high. The result is `pseudo_targets_B.npy`, in the cache's existing row order. Training, export and the blend kernel are the pilot's, with only the targets file and the ONNX filename changed.

**Tech Stack:** As the pilot: Perch v2 ONNX + onnxruntime 1.24.4 (CPU), scikit-learn, PyTorch/timm on a Kaggle T4, Kaggle CLI 2.2.4.

**Spec:** `BirdCLEF/docs/superpowers/specs/2026-10-20-sed-branch-design.md` (section 5, "Phase B"). The pilot results that motivate this are in `BirdCLEF/RESULTS.md` §16.

**Scope:** Pilot scale only: the same 12,000 windows, one fold, 20 epochs. A full-scale run gets its own plan once this result is in.

## Global Constraints

- Everything from the pilot plan (`2026-10-20-sed-pilot.md`) carries over: the worktree, branch, `$PY`, the mel front-end, the backbone, the recipe, the leakage rule, the kernel limits, the staging rules and `guard.sh`.
- **One-variable rule.** Phase B and the pilot differ only in the pseudo targets of the 12,000 windows in `pseudo_meta.csv`. Everything else stays identical:
  - the windows and their order
  - `labelled_targets.npy` and the focal targets
  - fold 0, seed, epochs, batch, recipe, GPU type (T4)
  - W_SED = 0.5 and the smoother
  The one unavoidable difference is GPU nondeterminism.
- **Teacher = kernel_perch, exactly.** The teacher's scores on any set of files must equal what `kernel_perch/birdclef_perch_submit.py` writes for the same files, to 1e-6.
- **Calibration (Claude's call; the spec left the method open).** For class j, the B targets are the pilot's A values for class j over the 12,000 windows, sorted and re-assigned in the order of the teacher's class-j scores. Tied scores get the mean of their values.
  - This keeps prevalence and confidence identical to A, so the comparison isolates ranking quality.
  - Ranking quality is what the 0.8275-vs-0.9144 OOF gap measures.
  - Own calibration (a B2) is a later question.
- Teacher outputs are cached under `/Volumes/Extreme SSD/kaggle/BirdCLEF/jobs/perch-unlabelled/`. The targets go next to A, as `.../jobs/sed-cache/pseudo_targets_B.npy`.
- **Submissions:** Phase B has exactly two, SED alone and the blend, both with W_SED = 0.5. The user's approval of this plan is the approval for those two. Nothing else is submitted without asking.
- **Decision rule.** LB noise is about ±0.003–0.005.
  - If B's SED-alone private beats A's 0.86072 by ≥ 0.02, the teacher was the main cause, and the next plan is full-scale B.
  - If the B blend's private is ≥ 0.9318, that is a go, even at pilot scale.
  - If B is within 0.01 of A, the teacher is not the main cause, and data scale is the next suspect.

## Review Focus

1. **The teacher silently differing from the kernel** (a copied function drifting, a different rank, the smoother's file grouping). Expected: equal scores. Test: Task 1 fidelity test against a local run of `kernel_perch` on the same 20 unlabelled files.
2. **B targets landing on the wrong rows.** The teacher's row order is shard order, and the cache's order is `pseudo_meta.csv`. Expected: B is built by `row_id` lookup. Test: Task 3 asserts every pilot `row_id` is found exactly once, and checks 3 rows by hand against the teacher's scores.
3. **Quantile mapping changing what it must not.** Expected: per class, sorted(B) == sorted(A) and B's order equals the teacher's order. Test: Task 1 unit test, plus a Task 3 assertion on the real arrays.
4. **The training kernel picking up the wrong targets.** Both A and B exist on the Kaggle input mount. Expected: the B kernel loads `pseudo_targets_B.npy` and logs which file it used. Test: Task 3 selftest of `pseudo_targets()`, and the kernel log.
5. **A resumed embedding run duplicating or skipping files after a power cut.** Expected: shards are written atomically (tmp file, then rename), and existing shards are skipped. Test: Task 2 checks 10,592 unique files and 127,104 unique (file, start) rows.

---

### Task 1: The teacher (`teacher.py`)

**Files:**
- Create: `BirdCLEF/teacher.py`

**Interfaces:**
- Produces:
  - `fit(Ltr, Etr, Y, Xf, yf, hit) -> dict`. Holds the scalers, the per-class heads and the cold-start heads, built exactly as in `kernel_perch`.
  - `raw_scores(state, Lk, E) -> (H, C)`. `Lk` is the logits at the 203 mapped columns. H is the head probabilities and C the cold-start probabilities. Chunkable.
  - `combine(state, Lk, H, C) -> P [n, 234]`. The kernel's `predict()`, with ranks over all n rows.
  - `smooth(P, files, starts) -> P [n, 234]`. The kernel's smoother.
  - `quantile_map(scores [n, k], ref [n, k]) -> [n, k]`
- Consumes: `perch.Perch`, `perch.to_windows`, `perch.species_map`, `embed_labelled.segments`, `data/focal_cap30_perch.npz`.

- [ ] **Step 1: Write the failing selftest** (`teacher.py --selftest`).
  - **quantile_map:** on random `scores` with a planted 5-way tie and random `ref`, assert:
    - per column, `np.sort(out) == np.sort(ref)` (up to the tie averaging)
    - `out` is non-decreasing in `scores`
    - the tied rows get equal values
  - **Fidelity:** build a fake `KAGGLE_INPUT` tree in a temp dir. Symlink the competition files and put 20 unlabelled soundscapes (the first 20 sorted) in `test_soundscapes`. Run `kernel_perch/birdclef_perch_submit.py` with `KAGGLE_INPUT`/`KAGGLE_WORKING` pointed at it. Then score the same 20 files with `fit → raw_scores → combine → smooth`, where the labelled windows are embedded the same way the kernel does it. Assert `max |teacher - submission.csv| < 1e-6` on all 234 columns, with rows matched by `row_id`.
- [ ] **Step 2: Run** `MIN_AVAIL_MB=2500 ../guard.sh $PY teacher.py --selftest`. Expected: FAIL (functions missing).
- [ ] **Step 3: Implement** the functions in `teacher.py`, copying the kernel's logic. Do not import the kernel; it is a script.
- [ ] **Step 4: Run the selftest.** Expected: PASS. The fidelity diff is printed.
- [ ] **Step 5: Commit** `BirdCLEF/teacher.py` with the message "Phase B teacher: kernel_perch as functions, fidelity-checked".

### Task 2: Perch over every unlabelled soundscape

**Files:**
- Modify: `BirdCLEF/teacher.py` (`--embed` mode)

**Interfaces:**
- Produces: `jobs/perch-unlabelled/shard_{k:03d}.npz`, 250 files per shard, with keys `filename`, `start_s`, `logits` [n, 203] float32 (the mapped columns) and `emb` [n, 1536] float32. Each shard is written to `*.tmp.npz`, then renamed.
- Consumes: `fit`'s column mapping (`species_map`).

- [ ] **Step 1: Selftest addition.** `unlabelled_files()` returns the sorted soundscapes minus the 66 labelled files. Assert it has 10,592 entries and no overlap with `segments().filename`.
- [ ] **Step 2: Run** the selftest. Expected: FAIL, then implement `unlabelled_files()` and the `--embed` loop, then PASS. Windows are `range(0, len(y) // 32000, 5)`, as in the kernel.
- [ ] **Step 3: Run** in the background: `caffeinate -ims env MIN_AVAIL_MB=2500 ../guard.sh $PY -u teacher.py --embed > "/Volumes/Extreme SSD/kaggle/BirdCLEF/logs/teacher_embed.log" 2>&1`. Expected: ~1.6 h at ~45 ms/window. Log a line per shard with ms/window.
- [ ] **Step 4: Verify** that the shards hold 10,592 unique files and 127,104 unique (filename, start_s) rows. Every `row_id` = `<stem>_<start+5>` must be in the public CSV's `fold_id == -1` set.
- [ ] **Step 5: Commit** `teacher.py` with the message "Phase B: Perch over the 10,592 unlabelled soundscapes, resumable shards".

### Task 3: B targets, and training kernel B

**Files:**
- Modify: `BirdCLEF/teacher.py` (`--targets` mode)
- Modify: `BirdCLEF/sed_train.py`. Add `pseudo_targets(cache, labels, root="/kaggle/input") -> str`, which returns `<cache>/pseudo_targets_<labels>.npy` if it exists, otherwise the directory where `find()` locates that filename under `root`. `main()` uses it and prints the path it loaded.
- Create: `BirdCLEF/kernel_sed_train_b/kernel-metadata.json`, with id and title `jagjeetsingh23/birdclef26-sed-train-b` / "BirdCLEF26 SED train B". The datasets are the pilot's plus `jagjeetsingh23/birdclef-sed-targets`, and the machine is T4.
- Create: `BirdCLEF/kernel_sed_train_b/sed_train_b.py`, a launcher. It finds `sed_train.py` in the code dataset, sets `sys.argv = ["sed_train.py", "--labels", "B"]`, and runs it with `runpy.run_path(..., run_name="__main__")`.

- [ ] **Step 1: Write the failing tests.**
  - `sed_train --selftest`: with temp dirs, `pseudo_targets` returns the cache path when the file is there, and the other dir when it is only under `root`.
  - `teacher --selftest`: `targets_b(P_rows, A)` returns float16 of the same shape, and per column `sort(B) == sort(A)` within float16 rounding.
- [ ] **Step 2: Run both selftests.** Expected: FAIL, then implement, then PASS.
- [ ] **Step 3: Run** `MIN_AVAIL_MB=2500 ../guard.sh $PY -u teacher.py --targets`. It:
  - fits on freshly embedded labelled windows and computes P over all 127,104 rows (raw scores in chunks, ranks global), then smooths per file
  - selects the `pseudo_meta.csv` rows by `row_id`, asserting each is found exactly once
  - writes `pseudo_targets_B.npy`
  - prints the diagnosis: per-class Spearman(A, B) over the 12k windows (quartiles and the 10 lowest, each with its Perch route: head / zero-shot / cold-start), and the 3 hand-checked rows
  Expected: shape (12000, 234), finite, sorted columns equal to A's.
- [ ] **Step 4: Upload.**
  - New dataset `jagjeetsingh23/birdclef-sed-targets` (`pseudo_targets_B.npy` only), staged at `jobs/sed-targets`.
  - New version of `birdclef-kernel-code` with `sed_train.py` added, staged at `jobs/kcode`.
  - Wait for both to be ready.
- [ ] **Step 5: Push** `kernel_sed_train_b`. Read the push output for the real slug, and poll that slug. Expected: COMPLETE in ~33 min GPU plus the queue. The log names `pseudo_targets_B.npy`, and the val_auc trajectory is recorded.
- [ ] **Step 6: Commit** `teacher.py`, `sed_train.py` and `kernel_sed_train_b/` with the message "Phase B: Perch-teacher targets for the pilot windows; training kernel B".

### Task 4: Kernel B runs, LB, decision, write-up

**Files:**
- Modify: `BirdCLEF/kernel_blend/birdclef_blend_submit.py`. Add `SED_ONNX = "sed_fold0_B.onnx"`, used by both `find()` and `SEDOnnx()`.
- Modify: `BirdCLEF/RESULTS.md` §16: add the Phase B rows and the diagnosis.

- [ ] **Step 1: Download** the B checkpoint to `jobs/sed-out-b/fold0/`. Export it to `jobs/sed-models/sed_fold0_B.onnx`. Run `SED_CKPT=.../sed-out-b/fold0/sed_fold0_ema.pt $PY sed_infer.py --selftest`. Expected: PASS, with GPU predictions reproduced on CPU.
- [ ] **Step 2: Upload** a new version of `birdclef-sed-models` holding both ONNX files. Then push the blend kernel twice: `MODE="blend"`, then `MODE="sed"`. Expected for each: COMPLETE, and the smoke log says "passed".
- [ ] **Step 3: Submit both**, as approved with this plan:
  - "SED Phase B pilot: Perch-teacher labels, w=0.5"
  - "SED Phase B pilot alone"
  Poll until both are COMPLETE.
- [ ] **Step 4: Decide** by the Global Constraints rule. Update RESULTS §16 with:
  - the A vs B table
  - the per-class Spearman diagnosis by route
  - teacher embedding time
  - the decision and the next step
  Commit with the message "SED Phase B pilot: <B blend private> private, <decision>". Update the memory note.
