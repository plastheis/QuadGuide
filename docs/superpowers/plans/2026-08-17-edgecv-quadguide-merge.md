# EdgeCV → QuadGuide Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold the EdgeCV tracking library into the QuadGuide repository as a pure relocation, with zero behaviour change, so that later work on the 10-bit pixel path and the classical seeker happens inside one repo.

**Architecture:** EdgeCV lands at `src/edgecv/` as a *second top-level package* alongside `src/quadguide/`, not nested inside it. EdgeCV's 57 modules use only absolute `from edgecv.x import y` imports (verified: zero relative imports), and its backend plugins register via `edgecv.backends` entry points, so this placement requires **no edits under `src/edgecv/` at all**. Touched elsewhere: 27 enumerated lines in the relocated test tree, plus exactly two non-test lines (`tools/track_webcam.py:51` and the adapter's `np.float32` leak).

**Tech Stack:** Python 3.11+, setuptools (src-layout), pytest, ruff, mypy, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-17-edgecv-quadguide-merge-design.md`

## Global Constraints

- **Zero behaviour change.** Every task is a relocation or a packaging fix. If a task requires changing what code *does*, stop and escalate.
- **`requires-python = ">=3.11"`** — QuadGuide's floor, matching the device. EdgeCV drops its 3.10 support.
- **CI matrix: `["3.11", "3.12"]`** on `ubuntu-latest` only. `src/quadguide` imports `fcntl` and `select` and calls `os.sched_setaffinity` — it cannot run on Windows or macOS.
- **`cv2` must NOT be a base dependency.** The device uses apt's GStreamer-enabled `python3-opencv` via `--system-site-packages`; a pip-installed opencv shadows it and breaks `CSICamera`. CI gets `opencv-python-headless` in the `test` extra only.
- **No EdgeCV source file under `src/edgecv/` is edited in this plan.** Verify by diffing against the source of truth — `diff -r /c/Users/plas/projects/EdgeCV/edgecv src/edgecv --exclude="__pycache__"` must print nothing (Task 2 Step 7). Do **not** use `git diff src/edgecv/`: these are newly-added files, so that command is trivially empty once committed and proves nothing.
- **Test edits are capped at the 27 lines enumerated in Task 3** — 14 relocation path-couplings (Steps 3–5a), 4 stale manifest assertions (5b), and 9 CWD-relative manifest paths (5c). No other test may be touched: no new assertions, no fixture changes, no logic changes.
- **Two non-test files are also in scope, and only these two:** `tools/track_webcam.py:51` (Step 5d — hardcodes the old layout) and `src/quadguide/perception/edgecv_adapter.py` (Step 5e — the `np.float32` protocol leak). Neither is under `src/edgecv/`, so neither violates the no-source-edit rule.
- **One deliberate exception to "zero behaviour change":** Step 5e is a real behaviour fix. It is admitted because the merge is what exposed the bug — `tests/unit/test_edgecv_adapter.py:11` calls `pytest.importorskip("edgecv")`, so those 6 tests never ran while EdgeCV lived in a separate repo — and because Task 6's CI gate cannot go green without it. Approved by the human partner on 2026-08-17.
- **`sed -i` rewrites line endings on this box** (CRLF → LF on every file it touches), so a plain `diff -r` against the EdgeCV checkout reports every touched file as wholly changed. Always pass `--strip-trailing-cr` when diffing relocated tests.
- Work happens in the **QuadGuide** repo (`github.com/plastheis/QuadGuide`) on a branch. The EdgeCV repo (`github.com/plastheis/EdgeCV`) is read-only source material until Task 8.

## Baseline facts (measured 2026-08-17, do not re-derive)

| Fact | Value |
|---|---|
| EdgeCV commits | 1 (`b03eae6 kalman`) — no history to preserve |
| EdgeCV modules | 57 `.py` under `edgecv/` |
| EdgeCV relative imports | **0** — all absolute |
| EdgeCV tests | 44 `test_*.py` + 5 support files (`conftest.py`, `__init__.py`, `_acquire_stubs.py`, `_nn_stubs.py`, `_onnx_synth.py`) = 49 `.py`; **306 collected**; on Windows **292 passed, 9 failed, 9 skipped** |
| EdgeCV's 9 Windows failures | **7 are Windows-only** (seqlock/shm — no `os.sched_yield` on Windows, see Task 3 Step 6) and **2 are real stale assertions** repaired in Task 3 Step 5b |
| QuadGuide tests | **276 collected, 6 collection errors on Windows** (all `ModuleNotFoundError: fcntl` — Linux-only, expected) |
| QuadGuide CI | none — no `.github/` on any branch |
| Git LFS | not used by either repo; both commit real blobs |
| Filename collisions | 6 root entries only; zero inside `docs/`, `models/`; `__init__.py` (empty, 0 bytes) only inside `tests/` |

**Where verification runs.** WSL is not installed on this dev box, so QuadGuide's suite cannot fully run locally — its 6 `fcntl` errors are permanent on Windows. Verification is therefore tiered:

- **Local (Windows):** EdgeCV's suite must reach `294 passed, 7 failed, 9 skipped` (the 7 are the documented seqlock/`sched_yield` set). QuadGuide's suite moves from *276 collected* to *282 collected, 6 errors, 1 failure* — the extra 6 are `test_edgecv_adapter.py`, un-hidden because `edgecv` is now importable past its `pytest.importorskip`; the 1 remaining failure is Windows-only `fcntl`.
- **Linux:** the CI workflow added in Task 6 is the first real Linux run of QuadGuide's suite. This is the gate.
- **Device (ROCK 5C):** Task 8 smoke test.

---

## File Structure

**Created:**
- `src/edgecv/**` — 57 modules, moved verbatim
- `tests/edgecv/**` — 44 test files + 3 helper modules + `conftest.py` + `__init__.py`
- `tools/**` — host-only conversion tooling (`convert.py`, `convert_lib/`, `onnx_to_rknn.py`, `track_webcam.py`, `CONVERSION.md`)
- `tests/edgecv/test_package_data.py` — new permanent guard (Task 5)
- `.github/workflows/ci.yml` — new (Task 6)
- `docs/architecture-edgecv.md` — relocated from EdgeCV's `ARCHITECTURE.md`

**Modified:**
- `pyproject.toml` — dependency declaration, extras, package-data, entry points
- `.gitignore` — reconcile two `models/` rule sets
- `requirements.txt` — drop the EdgeCV-install block
- `configs/rk3588.yaml:72` — `model_dir`
- `scripts/firstboot_install.sh`, `scripts/firstboot_install_rpi.sh` — drop EdgeCV clone/install
- `README.md`, `ARCHITECTURE.md` — cross-links

---

## Task 1: Capture the pre-merge baseline

The oracle for a pure relocation is a recorded before-state, not a new test. Capture it before touching anything.

**Files:**
- Create: `$SCRATCH/baseline-edgecv.txt`, `$SCRATCH/baseline-quadguide.txt`, `$SCRATCH/baseline-bench.txt`

Use your scratchpad directory, **not** the repo — these artifacts are not committed. Set it once at the start of the session:

```bash
export SCRATCH="/c/Users/plas/AppData/Local/Temp/claude/C--Users-plas-projects-EdgeCV/scratchpad"
mkdir -p "$SCRATCH"
```

(Any writable directory outside both repos works; the point is that `git status` must stay clean.)

**Interfaces:**
- Produces: three baseline files that Tasks 3 and 6 diff against.

- [ ] **Step 1: Record the EdgeCV suite**

```bash
cd /c/Users/plas/projects/EdgeCV
python -m pytest -q 2>&1 | tail -20 > "$SCRATCH/baseline-edgecv.txt"
cat "$SCRATCH/baseline-edgecv.txt"
```

Expected: `9 failed, 292 passed, 9 skipped`. Record the exact line **and the list of failing test IDs**.

This baseline is **known-red and that is accepted** — see the baseline facts table. Do not attempt to fix anything. Any deviation from 9 failures, or a different set of failing IDs, is what you should report.

- [ ] **Step 2: Record the QuadGuide suite**

```bash
cd /c/Users/plas/projects/QuadGuide
.venv/Scripts/python.exe -m pytest --collect-only -q 2>&1 | tail -12 > "$SCRATCH/baseline-quadguide.txt"
cat "$SCRATCH/baseline-quadguide.txt"
```

Expected: `276 tests collected, 6 errors`. The 6 errors are `ModuleNotFoundError: No module named 'fcntl'` — Linux-only modules on a Windows box. This is the expected Windows baseline, not a regression.

- [ ] **Step 3: Record the bench_tracker oracle**

Two things make this oracle work, and it is useless without both:

- **`edgecv` must be importable.** Pre-merge it is not installed in QuadGuide's venv, so a bare run dies with `ModuleNotFoundError: No module named 'edgecv'`. Put the EdgeCV checkout on the path for the baseline run only.
- **Only three lines are deterministic.** The latency table (`p50`/`p95`/`max`/`mean`) varies run to run and must never be diffed. The locked-fraction and IoU lines are stable — verified identical across repeated same-seed runs.

```bash
cd /c/Users/plas/projects/QuadGuide
PYTHONPATH=/c/Users/plas/projects/EdgeCV .venv/Scripts/python.exe \
    scripts/bench_tracker.py track \
    --frames 150 --width 480 --height 360 --seed 0 --no-plot 2>&1 \
  | grep -E "locked fraction|bbox IoU" > "$SCRATCH/baseline-bench.txt"
cat "$SCRATCH/baseline-bench.txt"
```

Expected: exactly 3 lines — `locked fraction:`, `bbox IoU between variants:`, `bbox IoU vs ground truth:`. If the file is empty or has fewer than 3 lines, the run failed; capture the unfiltered output instead and report it, because Task 6 Step 5 has no oracle without this.

- [ ] **Step 4: Create the working branch**

```bash
cd /c/Users/plas/projects/QuadGuide
git checkout -b merge-edgecv
git status --short   # expect clean
```

- [ ] **Step 5: Commit nothing**

This task produces no repo changes. Confirm with `git status --short` → empty output. Proceed to Task 2.

---

## Task 2: Relocate EdgeCV source, tools, and models

**Files:**
- Create: `src/edgecv/**` (57 modules), `tools/**`
- Create: `models/nanotrack_quant_rk3588/`, `models/nanotrack_quant_rk3566/`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `import edgecv` resolves from `src/edgecv/`; `tools/convert_lib` present for Task 3's conftest path.

- [ ] **Step 1: Copy the source tree**

```bash
cd /c/Users/plas/projects/QuadGuide
cp -r /c/Users/plas/projects/EdgeCV/edgecv src/edgecv
cp -r /c/Users/plas/projects/EdgeCV/tools tools
find src/edgecv -name "*.py" | wc -l    # expect 57
```

- [ ] **Step 2: Copy the committed model blobs**

Only the two `nanotrack_quant_*` directories are tracked in EdgeCV; everything else under its `models/` is gitignored build output.

```bash
cd /c/Users/plas/projects/QuadGuide
cp -r /c/Users/plas/projects/EdgeCV/models/nanotrack_quant_rk3588 models/
cp -r /c/Users/plas/projects/EdgeCV/models/nanotrack_quant_rk3566 models/
ls models/            # expect 4 *.onnx + 2 nanotrack_quant_* dirs
```

- [ ] **Step 3: Purge copied caches**

`cp -r` drags along `__pycache__` and any `.pyc`. Remove them before staging.

```bash
cd /c/Users/plas/projects/QuadGuide
find src/edgecv tools -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find src/edgecv tools -name "*.pyc" -delete
```

- [ ] **Step 4: Reconcile `.gitignore`**

QuadGuide's current model rules are `models/**` + `!models/**/` + `!models/**/*.onnx`. EdgeCV's were `/models/*` plus explicit per-directory `.rknn` negations. The merged block must admit **both** the `.onnx` files and the two `.rknn` directories. Replace QuadGuide's model block with:

```gitignore
# Model blobs are large — keep them out of git, EXCEPT:
#   *.onnx           — the RPi 4B (ONNX/CPU) tracker loads these
#   nanotrack_quant_rk*/*.rknn — the committed per-SoC NPU blobs (from EdgeCV)
# A file cannot be un-ignored while its parent directory is excluded, which is
# why the directory negations come before the file negations.
models/**
!models/**/
!models/**/*.onnx
!models/nanotrack_quant_rk3588/*.rknn
!models/nanotrack_quant_rk3566/*.rknn
```

Also append EdgeCV's build-artifact ignores that QuadGuide lacks:

```gitignore
.mypy_cache/
.ruff_cache/
.eggs/
.venv-convert/
runs/
calib_naive/
calib_nanotrack/
*.rknn
*.onnx
```

**Order matters:** the bare `*.rknn` / `*.onnx` rules must come *before* the `models/**` block above, or they will re-ignore the negated files.

- [ ] **Step 5: Verify git sees exactly the right files**

```bash
cd /c/Users/plas/projects/QuadGuide
git add -A
git status --short | grep -c "^A"                    # count of added files
git status --short | grep "models/" | sort
```

Expected under `models/`: exactly 7 new `.rknn` files (4 in `rk3588`, 3 in `rk3566`). If you see 0, the negation ordering is wrong. If you see stray `.onnx` from EdgeCV's models dir, you copied too much — only the two `nanotrack_quant_*` directories should be there.

- [ ] **Step 6: Verify the package imports**

```bash
cd /c/Users/plas/projects/QuadGuide
PYTHONPATH=src python -c "import edgecv, pathlib; print(pathlib.Path(edgecv.__file__).parent)"
```

Expected: a path ending in `QuadGuide/src/edgecv`. If it prints a path under `projects/EdgeCV`, an old editable install or `.pth` is shadowing — resolve before continuing.

- [ ] **Step 7: Confirm no source file was edited**

```bash
cd /c/Users/plas/projects/QuadGuide
diff -r /c/Users/plas/projects/EdgeCV/edgecv src/edgecv --exclude="__pycache__" && echo "IDENTICAL"
```

Expected: `IDENTICAL`. This is the Global Constraint made checkable.

- [ ] **Step 8: Commit**

```bash
cd /c/Users/plas/projects/QuadGuide
git add -A
git commit -m "refactor: relocate EdgeCV source, tools, and model blobs into QuadGuide

edgecv/ -> src/edgecv/ as a second top-level package. EdgeCV uses only
absolute 'from edgecv.x' imports (zero relative imports), so no source
file is edited; verified byte-identical against the EdgeCV checkout.

Tests move separately in the next commit."
```

---

## Task 3: Relocate the EdgeCV test tree, fix its 14 path couplings and 4 stale assertions

This is the only task that edits test files. The edits are enumerated exhaustively below — there are no others.

**Files:**
- Create: `tests/edgecv/**` (49 test files, 3 helper modules, `conftest.py`, `__init__.py`)
- Modify: `tests/edgecv/conftest.py` (1 line — `tools/` path)
- Modify: 10 test files (1 helper-import prefix each)
- Modify: 3 test files (1 `Path(__file__)` expression each)
- Modify: `tests/edgecv/test_manifests_nn.py` (3 stale assertions) and `tests/edgecv/test_nanotrack_rknn.py` (1 stale assertion)

`test_nanotrack_rknn.py` appears twice — it takes both a path fix (Step 5a) and a stale assertion (Step 5b). Total edited files: 13. Total edited lines: 18.

**Interfaces:**
- Consumes: `src/edgecv/` and `tools/` from Task 2.
- Produces: 306 EdgeCV tests collectable from the QuadGuide repo root.

**Why these edits exist — two unrelated causes, kept separate on purpose:**

- **Steps 3–5a (14 lines), caused by the move.** EdgeCV's tests import shared helpers as an absolute `tests.` package (`from tests._nn_stubs import ...`) and reach for `tools/` and `edgecv/models/manifests/` by relative filesystem path. Nesting one directory deeper shifts both.
- **Step 5b (4 lines), pre-existing rot.** Four assertions contradict the `nanotrack.yaml` shipped beside them. They fail in EdgeCV today, they are platform-independent, and they would keep Task 6's CI gate red forever. Approved for repair by the human partner on 2026-08-17.

- [ ] **Step 1: Copy the test tree**

```bash
cd /c/Users/plas/projects/QuadGuide
mkdir -p tests/edgecv
cp /c/Users/plas/projects/EdgeCV/tests/*.py tests/edgecv/
find tests/edgecv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
ls tests/edgecv/*.py | wc -l          # expect 49
ls tests/edgecv/test_*.py | wc -l     # expect 44
```

The other 5 are `conftest.py`, `__init__.py`, and the three shared helpers `_acquire_stubs.py`, `_nn_stubs.py`, `_onnx_synth.py`.

`tests/edgecv/__init__.py` comes across from EdgeCV (empty, 0 bytes) and is required — QuadGuide's `tests/unit/` and `tests/integration/` both use one, and pytest needs it to import `tests.edgecv._nn_stubs` as a package.

- [ ] **Step 2: Run the suite to see it fail**

```bash
cd /c/Users/plas/projects/QuadGuide
python -m pytest tests/edgecv --collect-only -q 2>&1 | tail -15
```

Expected: collection **errors** — `ModuleNotFoundError: No module named 'tests._nn_stubs'` and friends. This is the failure the next three steps fix. Do not skip this step; seeing the failure first is how you know the fix was necessary.

- [ ] **Step 3: Fix the 10 helper-import prefixes**

Exactly 10 files import helpers via the `tests.` package. Rewrite the prefix:

```bash
cd /c/Users/plas/projects/QuadGuide
sed -i 's/^from tests\./from tests.edgecv./' tests/edgecv/*.py
grep -rn "^from tests\." tests/edgecv/*.py     # expect NO output
grep -rc "^from tests\.edgecv\." tests/edgecv/*.py | grep -v ":0" | wc -l   # expect 10
```

The 10 files, for review: `test_acquire_state_machine.py`, `test_acquire_workers.py`, `test_nanotrack.py`, `test_nanotrack_detector_adapter.py`, `test_nanotrack_rknn.py`, `test_nn_base.py`, `test_nn_onnx.py`, `test_siamfc.py`, `test_verified_acquire_track.py`, `test_yolo.py`.

- [ ] **Step 4: Fix the `conftest.py` tools path**

`tests/conftest.py` put `tools/` on `sys.path` so `import convert_lib` works for 7 conversion tests. It walked up 2 levels; from `tests/edgecv/` it must walk up 3.

Replace the file's path line:

```python
import sys
from pathlib import Path

# tools/ is not an installed package; put it on sys.path so `import convert_lib` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
```

**Change exactly one line** — `parent.parent` → `parents[2]`, because this file
now sits one directory deeper. Leave the comment on a single line: Step 8's
diff-count assertion depends on this being a 1-line change.

- [ ] **Step 5: Fix the 3 filesystem-path expressions, then the 4 stale manifest assertions**

### 5a — path expressions (caused by the move)


`tests/edgecv/test_nanotrack_rknn.py` line ~22 — the manifest moved under `src/`:

```python
MANIFEST = Path(__file__).resolve().parents[2] / "src/edgecv/models/manifests/nanotrack.yaml"
```

`tests/edgecv/test_onnx_to_rknn_scaffold.py` line ~13 — inside the helper that builds the path:

```python
    path = Path(__file__).resolve().parents[2] / "tools" / "onnx_to_rknn.py"
```

`tests/edgecv/test_track_webcam.py` line ~8:

```python
_PATH = Path(__file__).resolve().parents[2] / "tools" / "track_webcam.py"
```

Then confirm none were missed:

```bash
cd /c/Users/plas/projects/QuadGuide
grep -rn "parent\.parent\|parents\[1\]" tests/edgecv/*.py    # expect NO output
```

### 5b — stale manifest assertions (pre-existing rot, NOT caused by the move)

EdgeCV's `nanotrack.yaml` manifest was updated to the v3 backbone and the
yolocrop RKNN blob, but two tests still assert the old names. They fail today in
EdgeCV and would fail on Linux CI, blocking Task 6's gate. The **manifest is
correct** — every file it names exists (`models/nanotrackv3_backbone.onnx`,
`models/nanotrack_quant_rk3588/nanotrack_backbone_yolocrop.rknn`). The tests are
stale. Fix the tests, never the manifest.

Note that pytest stops at a test's first failing assert, which masked the
siblings: fixing only the first line will just surface the next.

In `tests/edgecv/test_manifests_nn.py`, three lines in `test_nanotrack_manifest_loads`:

```python
    assert bb["onnx"]["path"] == "nanotrackv3_backbone.onnx"
    assert bb["rknn"]["path"] == "nanotrack_quant_{target}/nanotrack_backbone_yolocrop.rknn"
    assert bb["rknn"]["quant"] == "int8"                       # already correct — leave
    assert hd["onnx"]["path"] == "nanotrackv3_head.onnx"
    assert hd["rknn"]["path"] == "nanotrack_quant_{target}/nanotrack_head.rknn"   # correct — leave
    assert hd["rknn"]["quant"] == "fp16"                       # already correct — leave
```

In `tests/edgecv/test_nanotrack_rknn.py`, one line in the sorted-paths list:

```python
    assert [Path(p).name for p in paths] == [
        "nanotrack_backbone_yolocrop.rknn",
        "nanotrack_head.rknn",
    ]
```

`yolocrop` still sorts before `head` (`b` < `h`), so the list order is unchanged.

**Change exactly these 4 lines.** Do not "improve" neighbouring assertions, and
do not touch any manifest.

### 5c — CWD-relative manifest paths (a SECOND relocation idiom, 9 lines)

Steps 3–5a caught the `Path(__file__).parents[...]` idiom. A different one also
assumes the old layout: string literals resolved relative to the **current working
directory**, i.e. `"edgecv/models/manifests/..."` meaning repo-root/`edgecv/`. After
Task 2 that directory is `src/edgecv/`. Prefix each with `src/`, changing nothing
else on the line:

| File:line | Change |
|---|---|
| `test_manifests_nn.py:5` | `Path("edgecv/models/manifests")` → `Path("src/edgecv/models/manifests")` |
| `test_convert_yolo.py:38` | `load_manifest("edgecv/...")` → `load_manifest("src/edgecv/...")` |
| `test_nn_base.py:35` | `NNTracker("edgecv/...")` → `NNTracker("src/edgecv/...")` |
| `test_nn_base.py:44` | same |
| `test_nn_onnx.py:26` | `_manifest_with_artifact("edgecv/...")` → `"src/edgecv/..."` |
| `test_nn_onnx.py:39` | same |
| `test_nn_onnx.py:54` | same |
| `test_nn_onnx.py:66` | same |
| `test_pp_precedence.py:21` | `manifest_preprocessing("edgecv/...")` → `"src/edgecv/..."` |

`test_manifests_nn.py:5` is a single module-level constant that causes **6** test
failures on its own — one line, six greens.

Confirm none remain:

```bash
cd /c/Users/plas/projects/QuadGuide
grep -rn '"edgecv/\|Path("edgecv' tests/edgecv/*.py     # expect NO output
```

### 5d — `tools/track_webcam.py` (1 line, not a test)

`tools/track_webcam.py:51` hardcodes the pre-merge layout, which breaks the tool
itself and 2 tests. `_ROOT` is the repo root (`Path(__file__).resolve().parent.parent`):

```python
MANIFESTS_DIR = _ROOT / "src" / "edgecv" / "models" / "manifests"
```

Leave `MODELS_DIR = _ROOT / "models"` on the next line alone — that path is still correct.

### 5e — the `np.float32` protocol leak (1 line, not a test)

`tests/unit/test_edgecv_adapter.py` asserts every bbox component reaching
QuadGuide's tracker protocol is a Python `float`, but MOSSE returns `np.float32`
and the adapter passes it straight through. These 6 tests have **never run** —
line 11 is `pytest.importorskip("edgecv")`, and EdgeCV was never installed in
QuadGuide's venv — so the merge is what exposed this.

Fix it at the boundary that owns the protocol. In
`src/quadguide/perception/edgecv_adapter.py`, in `update()`, the final return:

```python
        b = res.bbox
        return _TrackerOutput(
            _BBox(float(b.x), float(b.y), max(0.0, float(b.w)), max(0.0, float(b.h))),
            self._normalize_confidence(res.confidence),
            health,
            origin_ns,
        )
```

This is the right layer: the module's own docstring calls it "the impedance match
between the two", and it already coerces confidence via `_normalize_confidence`.
Do **not** fix this in `src/edgecv/core/bbox.py` — that is barred by the
no-source-edit constraint, and EdgeCV is entitled to its own numeric types.

(For context, not a task: `struct.pack` accepts `np.float32` happily, so this was
a contract defect, not silent wire corruption.)

- [ ] **Step 6: Run the EdgeCV suite to verify it passes**

```bash
cd /c/Users/plas/projects/QuadGuide
python -m pytest tests/edgecv -q 2>&1 | tail -10
```

Expected on **Windows**: `294 passed, 7 failed, 9 skipped`.

Reconcile that against the baseline (`292 passed, 9 failed, 9 skipped`) as follows:

- **+2 passed / −2 failed** — the two manifest tests you repaired in Step 5b.
- **The remaining 7 failures are Windows-only and must persist unchanged**:
  `test_acquire_channels` (×4), `test_search_roi` (×2), `test_seqlock` (×1). All
  are seqlock-backed. `edgecv/runtime/shm/seqlock.py:23` does
  `_yield = getattr(os, "sched_yield", lambda: None)`, and Windows has no
  `os.sched_yield`, so the yield degrades to a no-op and the reader starves —
  precisely the spurious livelock EdgeCV's ARCHITECTURE §7.3 documents. **These
  are expected. Do not attempt to fix them.** They pass on Linux, which CI proves
  in Task 6.

Any *other* difference is a real regression — investigate before proceeding.

- [ ] **Step 7: Verify QuadGuide's own suite is untouched**

```bash
cd /c/Users/plas/projects/QuadGuide
.venv/Scripts/python.exe -m pytest tests/unit tests/integration tests/hil -q 2>&1 | tail -5
```

Expected: **`282 tests collected, 6 errors`**, with exactly **1 failure**
(`test_loadable_via_tracker_worker_load_tracker`, a Windows-only `fcntl` import).

Do not be alarmed that this exceeds Task 1's `276` baseline. The extra 6 are
`tests/unit/test_edgecv_adapter.py`, which line 11 gates behind
`pytest.importorskip("edgecv")`. EdgeCV was never installed in QuadGuide's venv, so
those 6 tests **had never executed**; Task 2 made `edgecv` importable from `src/`
and un-hid them. That is the merge working as intended, not a regression. One of
the two that failed is repaired in Step 5e; the other is the `fcntl` platform case.

- [ ] **Step 8: Verify the edit surface is exactly 27 lines**

```bash
cd /c/Users/plas/projects/QuadGuide
git add -A
git diff --cached --stat tests/edgecv/ | tail -3
```

Since these are new files, the stat shows additions only. Instead, diff against the source of truth:

**`--strip-trailing-cr` is mandatory here.** `sed -i` rewrites CRLF to LF on every
file it touches on this box, so without it `diff -r` reports whole files as changed
and the count is meaningless.

```bash
diff -r --strip-trailing-cr /c/Users/plas/projects/EdgeCV/tests tests/edgecv \
     --exclude="__pycache__" | grep "^[<>]" | wc -l
```

Expected: **54** (27 changed lines × 2, one `<` and one `>` each) — 14 relocation
couplings (Steps 3–5a) + 4 stale assertions (5b) + 9 CWD-relative paths (5c). If
higher, something beyond the enumerated set was edited; find it and revert it. If
lower, an enumerated edit was missed.

The two non-test files are tracked, so plain `git diff` works for them and each
must show exactly one changed line:

```bash
git diff --stat tools/track_webcam.py src/quadguide/perception/edgecv_adapter.py
```

Expected: `tools/track_webcam.py | 2 +-` and
`src/quadguide/perception/edgecv_adapter.py | 2 +-` (1 insertion, 1 deletion each).

- [ ] **Step 9: Commit**

```bash
cd /c/Users/plas/projects/QuadGuide
git add -A
git commit -m "test: relocate EdgeCV's test tree to tests/edgecv/

306 tests, unchanged except for 27 lines across 16 files.

14 are relocation path-couplings: 10 helper-import prefixes (tests. ->
tests.edgecv.), 3 Path(__file__).parents[] expressions (manifests moved
under src/, tools/ is one level further up), and the conftest sys.path
insert for tools/.

4 repair stale assertions that fail in EdgeCV today and would block CI:
nanotrack.yaml was updated to the v3 backbone and the yolocrop RKNN blob
without updating test_manifests_nn.py (3 lines) or test_nanotrack_rknn.py
(1 line). The manifest is correct — every file it names exists.

No fixture or logic changes; no manifest changes."
```

---

## Task 4: Reconcile `pyproject.toml`

**Files:**
- Modify: `pyproject.toml` (full rewrite)
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `src/edgecv/` and `tests/edgecv/` from Tasks 2–3.
- Produces: `pip install -e .[onnx,ground,test]` installs everything both suites need — the exact command CI runs in Task 6.

**Why:** QuadGuide's `pyproject.toml` declares `dependencies = ["pyyaml"]`, but `src/quadguide` imports `numpy`, `pymavlink`, `pydantic`, `serial`, `fastapi`, and `uvicorn`. The real list has been living in `requirements.txt`, which `pip install -e .` never reads. EdgeCV's CI install step would silently install almost nothing and then fail at import.

- [ ] **Step 1: Prove the problem exists**

```bash
cd /c/Users/plas/projects/QuadGuide
python -m venv /tmp/qg-probe
/tmp/qg-probe/Scripts/python.exe -m pip install -q -e . 2>&1 | tail -3
/tmp/qg-probe/Scripts/python.exe -c "import quadguide.link.mavlink_codec"
```

Expected: `ModuleNotFoundError: No module named 'pymavlink'`. This is the bug. Keep the venv for Step 4.

- [ ] **Step 2: Write the merged `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "quadguide"
version = "0.1.0"
description = "CV-guided quadcopter interceptor: perception, guidance, control, link"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
authors = [{ name = "plastheis" }]
dependencies = [
    "numpy>=2.0",
    "PyYAML>=6.0",
    "pymavlink>=2.4",
    "pyserial>=3.5",
]

[project.optional-dependencies]
# Optional CF-ops accelerators. The numpy reference ops work without these;
# installing them just selects faster FFT/HOG/image paths.
fast = ["scipy>=1.10", "numba>=0.58", "pyFFTW>=0.13"]
onnx = ["onnxruntime>=1.16"]
# rknn-toolkit-lite2 is NOT on PyPI; install it manually on-device.
rknn = []
# Host-only conversion tooling. Not a runtime dependency.
dev = ["torch>=2.5", "onnx>=1.15", "ultralytics>=8.3", "opencv-python>=4.10"]
ground = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "pydantic>=2.0"]
# opencv-python-headless (NOT opencv-python) — GitHub runners have no
# libGL.so.1, so the non-headless wheel fails at import and takes the whole
# collection down. The device uses apt's GStreamer-enabled python3-opencv
# instead; see the cv2 note in the merge spec.
test = [
    "pytest>=8.0",
    "ruff>=0.4",
    "mypy>=1.8",
    "onnxruntime>=1.16",
    "onnx>=1.15",
    "httpx>=0.27",
    "opencv-python-headless>=4.10",
]

[project.entry-points."edgecv.backends"]
mock = "edgecv.backends.mock:MockBackend"
onnx = "edgecv.backends.onnx:OnnxBackend"
rknn = "edgecv.backends.rknn:RknnBackend"

[tool.setuptools.packages.find]
where = ["src"]

# Replaces hatchling's force-include. If this is wrong, NOTHING fails locally
# (manifests resolve via edgecv.__file__ in a source checkout) — only an
# installed wheel breaks, at model-load time on the device. Guarded by
# tests/edgecv/test_package_data.py.
[tool.setuptools.package-data]
"edgecv.models" = ["profiles/*.yaml", "manifests/*.yaml"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
warn_unused_ignores = false
```

Note `opencv-python` moved out of the `fast` extra into `dev` — `fast` is a device-side accelerator group and must not pull a pip opencv onto the SBC.

- [ ] **Step 3: Strip the EdgeCV block from `requirements.txt`**

Delete lines 20–28 (the "EdgeCV tracker library (separate repo)" block and its install instructions — the `pip install -e /home/radxa/EdgeCV` and `edgecv.pth` recipes). Replace the file header comment with a pointer that `pyproject.toml` is now authoritative:

```
# quadguide runtime + dev dependencies.
#
# pyproject.toml is now AUTHORITATIVE for dependencies — this file exists for
# device provisioning convenience and must stay in sync with it.
# Prefer:  pip install -e .[onnx,ground]
#
# NOT listed here on purpose: opencv. The device uses apt's python3-opencv
# (GStreamer-enabled) via a --system-site-packages venv; a pip opencv would
# shadow it and break the CSI camera path.
```

Keep the remaining runtime pins consistent with `pyproject.toml`.

- [ ] **Step 4: Verify a clean install now works**

```bash
cd /c/Users/plas/projects/QuadGuide
rm -rf /tmp/qg-probe && python -m venv /tmp/qg-probe
/tmp/qg-probe/Scripts/python.exe -m pip install -q -e ".[onnx,ground,test]"
/tmp/qg-probe/Scripts/python.exe -c "import quadguide.link.mavlink_codec, edgecv, cv2; print('OK')"
```

Expected: `OK`. This is the exact dependency set CI installs.

- [ ] **Step 5: Verify both suites still collect**

```bash
cd /c/Users/plas/projects/QuadGuide
/tmp/qg-probe/Scripts/python.exe -m pytest --collect-only -q 2>&1 | tail -5
```

Expected: **`592 tests collected, 6 errors`** — 310 from `tests/edgecv` plus 282 from QuadGuide's own suite. The 6 are the Windows `fcntl` collection errors.

**The collected total is environment-dependent, so do not treat a mismatch as a lost test file without checking why.** Three `pytest.importorskip` gates have moved during this merge, each admitting tests that were previously invisible:

- 276 → 282 (Task 2): `tests/unit/test_edgecv_adapter.py:11` gates on `edgecv`, which was never installed in QuadGuide's venv. Those 6 tests had never executed in the project's history.
- 306 → 310 (this task): several EdgeCV modules gate on `onnx` at module level. The repo's `.venv` has `onnxruntime` but **not** `onnx`; the `test` extra installs `onnx>=1.15`, so 4 more modules collect in the probe venv.

Measured in the repo `.venv` (no `onnx`) the figure is 588; in a venv with the `test` extra it is 592. Both are correct for their environment. If you see a number outside these two, then investigate.

Add `--continue-on-collection-errors`, or pytest aborts on the first collection error and reports zero tests.

- [ ] **Step 6: Verify the backend entry points still resolve**

```bash
cd /c/Users/plas/projects/QuadGuide
/tmp/qg-probe/Scripts/python.exe -c "
from edgecv.backends.registry import list_backends, available_backends
print('registered:', sorted(list_backends()))
print('available :', sorted(available_backends()))"
```

Expected: `registered: ['mock', 'onnx', 'rknn']`. `list_backends()` reads the `edgecv.backends` entry-point group directly, so it is the precise check that the group survived the hatchling→setuptools move. If it returns `[]`, the entry points were lost.

`available_backends()` filters to backends that actually load, so `rknn` will be absent on a dev box — that is correct, not a failure.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/plas/projects/QuadGuide
git add pyproject.toml requirements.txt
git commit -m "build: merge EdgeCV's packaging into QuadGuide's pyproject

- declare QuadGuide's real deps (numpy, pymavlink, pyserial); they lived
  only in requirements.txt, which 'pip install -e .' never reads
- carry over EdgeCV's edgecv.backends entry points and extras
- translate hatchling force-include -> setuptools package-data for the
  model manifests/profiles
- requires-python >=3.11; drop hatchling
- opencv-python-headless in [test] only: CI runners lack libGL, and the
  device needs apt's GStreamer-enabled build unshadowed"
```

---

## Task 5: Guard the packaging translation with a permanent test

The `force-include` → `package-data` translation fails **silently** in a source checkout. This task converts that one-time risk into a permanent regression test.

**Files:**
- Create: `tests/edgecv/test_package_data.py`

**Interfaces:**
- Consumes: `[tool.setuptools.package-data]` from Task 4.
- Produces: a test that fails if the manifests or profiles stop shipping.

- [ ] **Step 1: Write the failing test**

Create `tests/edgecv/test_package_data.py`:

```python
"""Guard: model manifests and board profiles must ship as package data.

EdgeCV was built by hatchling with a force-include for
edgecv/models/{manifests,profiles}; the merge translated that to setuptools
package-data. That translation fails SILENTLY in a source checkout, because
manifests resolve via Path(edgecv.__file__).parent, which works whether or
not the YAML was declared as package data. Only an installed wheel breaks,
and only at model-load time on the device.

These tests use importlib.resources, which reads through the package's
declared data rather than the filesystem, so they fail in an installed
environment where the YAML was not shipped.
"""
from importlib import resources

import pytest


def _names(subdir: str) -> set[str]:
    # Anchor on edgecv.models — a REAL package (it has __init__.py) — and reach
    # the data directory with joinpath. Do not use resources.files() directly on
    # "edgecv.models.manifests": that directory has no __init__.py, so it is at
    # best a namespace package and the call is not reliable across environments.
    return {p.name for p in resources.files("edgecv.models").joinpath(subdir).iterdir()}


def test_manifests_ship_as_package_data():
    names = _names("manifests")
    assert {"nanotrack.yaml", "siamfc_generic.yaml", "yolo11n.yaml"} <= names


def test_profiles_ship_as_package_data():
    names = _names("profiles")
    assert {"dev.yaml", "rk3588.yaml"} <= names


@pytest.mark.parametrize("name", ["nanotrack.yaml", "yolo11n.yaml"])
def test_manifest_is_readable_and_nonempty(name):
    text = (resources.files("edgecv.models")
            .joinpath("manifests", name)
            .read_text(encoding="utf-8"))
    assert "artifacts" in text
```

- [ ] **Step 2: Run the test in the source checkout**

```bash
cd /c/Users/plas/projects/QuadGuide
/tmp/qg-probe/Scripts/python.exe -m pytest tests/edgecv/test_package_data.py -v
```

Expected: PASS. A source checkout always passes — that is the whole point, and why Step 3 exists.

- [ ] **Step 3: Run the test against a built wheel (the real check)**

```bash
cd /c/Users/plas/projects/QuadGuide
/tmp/qg-probe/Scripts/python.exe -m pip install -q build
/tmp/qg-probe/Scripts/python.exe -m build --wheel --outdir /tmp/qg-wheel
unzip -l /tmp/qg-wheel/quadguide-0.1.0-*.whl | grep -E "manifests|profiles"
```

Expected: the `.yaml` files listed inside the wheel. If they are absent, the `package-data` key is wrong — fix `pyproject.toml` and rebuild before continuing.

- [ ] **Step 4: Install the wheel clean and run the guard**

```bash
python -m venv /tmp/qg-wheeltest
/tmp/qg-wheeltest/Scripts/python.exe -m pip install -q /tmp/qg-wheel/quadguide-0.1.0-*.whl pytest
/tmp/qg-wheeltest/Scripts/python.exe -m pytest /c/Users/plas/projects/QuadGuide/tests/edgecv/test_package_data.py -v
```

Expected: PASS, now proving the data actually shipped. This satisfies spec success criterion 3.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/plas/projects/QuadGuide
git add tests/edgecv/test_package_data.py
git commit -m "test: guard that model manifests/profiles ship as package data

The hatchling force-include -> setuptools package-data translation fails
silently in a source checkout (manifests resolve via edgecv.__file__), so
only an installed wheel breaks. These read through importlib.resources so
they fail in an environment where the YAML was not shipped."
```

---

## Task 6: Add the CI workflow

QuadGuide has no CI on any branch. This adds it, and the first green run is also the **first Linux execution of QuadGuide's test suite** — the gate the Windows dev box cannot provide.

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the `[onnx,ground,test]` extras from Task 4.

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
    branches: [main, merge-edgecv]
  pull_request:

jobs:
  test:
    # ubuntu only, not incidental: src/quadguide imports fcntl and select and
    # calls os.sched_setaffinity — it cannot run on Windows or macOS.
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e .[onnx,ground,test]
      # Lint/type scope is deliberately src/edgecv only. QuadGuide's code has
      # never been linted or type-checked; widening here would mean fixing an
      # unknown number of pre-existing violations inside a merge whose purpose
      # is to prove nothing changed. Tracked as a follow-up.
      - name: Lint
        run: ruff check src/edgecv tests/edgecv
      - name: Type-check
        run: mypy src/edgecv
      - name: Test
        run: pytest -q

  # The package-data guard CANNOT do its job in the job above. pyproject's
  # [tool.pytest.ini_options] pythonpath = ["src"] puts the source tree first for
  # every pytest run made from this repo, so tests/edgecv/test_package_data.py
  # reads src/edgecv and passes even when the built wheel ships no YAML at all.
  # Proven by negative control in Task 5: deleting the installed wheel's
  # manifests/ directory still gave "4 passed" without isolation.
  #
  # This job is therefore the real gate for spec success criterion 3: build the
  # wheel, install it into a venv that has no source checkout, and run the guard
  # with pythonpath overridden to empty.
  wheel-guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Build wheel
        run: |
          python -m pip install --upgrade pip build
          python -m build --wheel --outdir dist/
      - name: Assert the YAML actually shipped
        run: |
          python - <<'PY'
          import zipfile, glob, sys
          whl = glob.glob("dist/*.whl")[0]
          names = zipfile.ZipFile(whl).namelist()
          yaml = [n for n in names if n.startswith("edgecv/models/") and n.endswith(".yaml")]
          print(f"{whl}: {len(yaml)} manifest/profile YAML files")
          for n in sorted(yaml):
              print("  ", n)
          sys.exit(0 if yaml else "NO YAML IN WHEEL — package-data key is wrong")
          PY
      - name: Install the wheel clean and run the guard
        run: |
          python -m venv /tmp/wheeltest
          /tmp/wheeltest/bin/python -m pip install -q dist/*.whl pytest
          /tmp/wheeltest/bin/python -m pytest -o pythonpath= \
              tests/edgecv/test_package_data.py -q
```

The `-o pythonpath=` is load-bearing, not decorative: without it the guard silently
reads the checkout and this job becomes as vacuous as the one it exists to backstop.

- [ ] **Step 2: Verify the lint scope passes locally first**

```bash
cd /c/Users/plas/projects/QuadGuide
/tmp/qg-probe/Scripts/python.exe -m ruff check src/edgecv tests/edgecv
/tmp/qg-probe/Scripts/python.exe -m mypy src/edgecv
```

Expected: both clean. They were clean in EdgeCV's CI, and no source file changed, so any new failure comes from the 14 test-line edits (most likely an import-order `I` violation in one of the 10 rewritten files). Fix by re-sorting the import block only.

- [ ] **Step 3: Push and let CI run**

```bash
cd /c/Users/plas/projects/QuadGuide
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow (QuadGuide had none)

Adapted from EdgeCV's: matrix drops 3.10 (QuadGuide floor is 3.11), install
adds the ground extra (tests import starlette via FastAPI TestClient), and
lint/type scope stays src/edgecv so the merge is process-neutral too."
git push -u origin merge-edgecv
```

- [ ] **Step 4: Confirm the run is green**

```bash
cd /c/Users/plas/projects/QuadGuide
gh run watch
```

Expected: green on both 3.11 and 3.12, with **zero failures and zero errors**.

Record the actual `N passed, M skipped` line in your report — do not assert a
hardcoded pass count. Collection totals (306 + 276 = 582) are known and additive,
but the pass/skip split is not: the Windows baseline reports more results than
collected items because some tests are parametrized at runtime.

What the Linux run must prove:

- The 6 `fcntl` collection errors seen on Windows are **absent** (Linux has `fcntl`).
- The 7 seqlock/shm failures seen on Windows are **absent** (Linux has `os.sched_yield`).
- The 4 assertions repaired in Task 3 Step 5b **pass**.

Any failure at all means something other than platform is wrong — stop and report it.

This is spec success criteria 1 and 2 satisfied on the platform that matters.

- [ ] **Step 5: Run the bench oracle and diff against baseline**

No `PYTHONPATH` this time — the probe venv has the merged repo installed editable, so `edgecv` resolves from `src/`. Filter to the same three deterministic lines the baseline captured.

```bash
cd /c/Users/plas/projects/QuadGuide
/tmp/qg-probe/Scripts/python.exe scripts/bench_tracker.py track \
    --frames 150 --width 480 --height 360 --seed 0 --no-plot 2>&1 \
  | grep -E "locked fraction|bbox IoU" > "$SCRATCH/after-bench.txt"
diff "$SCRATCH/baseline-bench.txt" "$SCRATCH/after-bench.txt" && echo "IDENTICAL"
```

Expected: `IDENTICAL`. This is the strongest evidence in the plan that the relocation changed no behaviour — the same synthetic sequence, through the same two ONNX model pairs, producing bit-identical tracking geometry before and after the move. If it differs, **stop**: something in the relocation altered tracker behaviour. Satisfies spec success criterion 2.

---

## Task 7: Rewire deployment and relocate documentation

**Files:**
- Modify: `scripts/firstboot_install.sh`, `scripts/firstboot_install_rpi.sh`
- Modify: `configs/rk3588.yaml:72`
- Create: `docs/architecture-edgecv.md`
- Modify: `README.md`, `ARCHITECTURE.md`
- Create: `docs/superpowers/specs/*`, `docs/superpowers/plans/*` (EdgeCV's, unioned)

- [ ] **Step 1: Strip EdgeCV provisioning from the firstboot scripts**

`scripts/firstboot_install.sh` has 17 EdgeCV references and `scripts/firstboot_install_rpi.sh` has 17. Remove from both:

- the `EDGECV_REPO` and `EDGECV_DIR` variable definitions
- the EdgeCV clone/update in step 2, and its failure warnings
- the `pip install -e "$EDGECV_DIR"` line
- the `/home/radxa/EdgeCV` path-mismatch warnings
- the `n_models` counting of `*.rknn` under `$EDGECV_DIR/models`, and its summary line
- **the Git LFS smudge enable** — neither repo has ever used LFS; both commit real blobs. This step was always a no-op.

Renumber the remaining steps (the scripts say "2/5", "3/5" etc.) and update the summary block so it points at `$QG_DIR/models` for both `.onnx` and `.rknn`.

- [ ] **Step 2: Point the config at the merged model directory**

`configs/rk3588.yaml`, in `tracker.params`:

```yaml
    model_dir: /home/radxa/quadguide/models   # resolves *.rknn artifacts (EDGECV_MODEL_DIR)
```

- [ ] **Step 3: Verify the config still parses and the path is the only change**

```bash
cd /c/Users/plas/projects/QuadGuide
python -c "import yaml; d=yaml.safe_load(open('configs/rk3588.yaml')); print(d['tracker']['params']['model_dir'])"
git diff configs/rk3588.yaml
```

Expected: prints the new path; the diff shows exactly one changed line.

- [ ] **Step 4: Relocate the EdgeCV architecture document**

```bash
cd /c/Users/plas/projects/QuadGuide
cp /c/Users/plas/projects/EdgeCV/ARCHITECTURE.md docs/architecture-edgecv.md
cp /c/Users/plas/projects/EdgeCV/docs/superpowers/specs/*.md docs/superpowers/specs/
cp /c/Users/plas/projects/EdgeCV/docs/superpowers/plans/*.md docs/superpowers/plans/
ls docs/superpowers/specs | wc -l    # 11 QuadGuide + 12 EdgeCV = 23
```

Filename collisions were verified to be zero, so no file is overwritten. Confirm nothing was clobbered: `git status --short docs/` should show only additions (`A`), never modifications (`M`).

- [ ] **Step 5: Cross-link the two architecture documents**

At the top of `docs/architecture-edgecv.md`, under the existing `> **Status:**` block, add:

```markdown
> **Note:** EdgeCV was merged into QuadGuide on 2026-08-17 and now lives at
> `src/edgecv/`. This document describes the tracking library; the system it
> runs inside is described in [`ARCHITECTURE.md`](../ARCHITECTURE.md). Paths
> below that read `edgecv/...` are now `src/edgecv/...`.
```

In `ARCHITECTURE.md`, in the section describing the tracker worker, add a pointer:

```markdown
The tracking library itself (trackers, backends, runtime/IPC, fusion) is
documented separately in [`docs/architecture-edgecv.md`](docs/architecture-edgecv.md).
It lives at `src/edgecv/` and was a standalone repository until 2026-08-17.
```

- [ ] **Step 6: Fold EdgeCV's README into QuadGuide's**

Add a `## Tracking library (edgecv)` section to `README.md` carrying over the install-extras table, the RKNN on-device note, and the model-conversion pointer to `tools/CONVERSION.md`. Update the conversion commands to run from the repo root (they are unchanged — `tools/convert.py` kept its path).

- [ ] **Step 7: Commit**

```bash
cd /c/Users/plas/projects/QuadGuide
git add -A
git commit -m "chore: rewire deployment and relocate EdgeCV docs

- firstboot scripts no longer clone/install EdgeCV separately; drop the
  vestigial git-lfs smudge (neither repo ever used LFS)
- rk3588.yaml model_dir -> /home/radxa/quadguide/models
- EdgeCV ARCHITECTURE.md -> docs/architecture-edgecv.md, cross-linked
- union EdgeCV's specs/plans into docs/superpowers (zero collisions)"
```

---

## Task 8: Device migration and EdgeCV repository archive

Ordering is deliberate: the device is verified **before** the old repo is archived, so the rollback path stays intact.

**Files:** none in-repo. This is an ops runbook.

- [ ] **Step 1: Merge to main and confirm CI**

```bash
cd /c/Users/plas/projects/QuadGuide
git checkout main && git merge --no-ff merge-edgecv
git push origin main
gh run watch
```

Expected: green.

- [ ] **Step 2: Remove the shadowing installs on the device**

**This is the migration's main hazard.** The device has EdgeCV installed editable at `~/EdgeCV`, and `requirements.txt` documented an alternative user `.pth`. Either left in place makes the device import the **old** EdgeCV while every test passes green.

On the ROCK 5C:

```bash
pip uninstall -y edgecv
rm -f ~/.local/lib/python3.11/site-packages/edgecv.pth
cd ~/quadguide && git pull && pip install -e .
python -c "import edgecv, pathlib; print(pathlib.Path(edgecv.__file__).parent)"
```

Expected: a path under `~/quadguide/src/edgecv`. **If it prints anything under `~/EdgeCV`, stop** — the shadow is still active and the rest of this task is meaningless.

- [ ] **Step 3: Confirm the models resolved**

```bash
ls ~/quadguide/models/nanotrack_quant_rk3588/*.rknn | wc -l   # expect 4
grep model_dir ~/quadguide/configs/rk3588.yaml
```

- [ ] **Step 4: Smoke-test the service**

```bash
sudo systemctl restart quadguide
journalctl -u quadguide -f --lines=50
```

Expected: all six workers start; no `ModuleNotFoundError`; the tracker worker logs its name. Open the HUD and confirm the camera is live and NanoTrack acquires and locks on a target. This satisfies spec success criterion 4.

- [ ] **Step 5: Remove the stale checkout**

Only after Step 4 passes:

```bash
rm -rf ~/EdgeCV
sudo systemctl restart quadguide   # confirm it still runs with the checkout gone
```

- [ ] **Step 6: Archive the EdgeCV repository**

In the `plastheis/EdgeCV` repo, first replace `README.md` with a pointer:

```markdown
# EdgeCV — merged into QuadGuide

EdgeCV was merged into [QuadGuide](https://github.com/plastheis/QuadGuide) on
2026-08-17 and now lives there at `src/edgecv/`. This repository is archived
and read-only; all development continues in QuadGuide.

Design document: `docs/architecture-edgecv.md` in QuadGuide.
```

Commit and push, then archive via GitHub → Settings → *Archive this repository*. Archiving makes it read-only, stops its Actions from running, and keeps the URL alive so existing links do not 404.

- [ ] **Step 7: Final verification sweep**

```bash
cd /c/Users/plas/projects/QuadGuide
grep -rn "EdgeCV" scripts/ configs/ requirements.txt | grep -v "docs/" | grep -vi "merged\|architecture-edgecv"
```

Expected: no output. Any remaining reference to a separate EdgeCV checkout is a leftover — fix and commit.

---

## Follow-ups (explicitly NOT this plan)

- Measure `ruff check src/quadguide --statistics` and `mypy src/quadguide`, then decide whether to fix violations or configure per-rule ignores, and widen the CI scope.
- Decide whether `tools/` (host-only conversion) and `scripts/` (device/dev ops) should consolidate.
- Reconcile `ARCHITECTURE.md` and `docs/architecture-edgecv.md` into one narrative — deferred until specs 2 and 3 have settled the seeker's shape.
- **Spec 2:** native 1280×800 10-bit mono (uint16) pixel path.
- **Spec 3:** MPCM/Otsu-CC detector, IMM filter, detector-coupled measurement covariance.
