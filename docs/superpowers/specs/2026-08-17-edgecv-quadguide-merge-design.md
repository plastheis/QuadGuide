# EdgeCV → QuadGuide repo merge

**Date:** 2026-08-17
**Status:** design, approved
**Scope:** Spec 1 of 3. Merge only — no behaviour change.

## Context

EdgeCV (`github.com/plastheis/EdgeCV`) is a single-object visual tracking library
consumed by exactly one project: QuadGuide (`github.com/plastheis/QuadGuide`), via
`src/quadguide/perception/edgecv_adapter.py`. The two are developed together,
deployed together onto the same SBC, and released together.

The merge is motivated by upcoming work (specs 2 and 3) that the repo boundary
makes disproportionately expensive:

- **Spec 2** moves the camera path to native 1280×800 10-bit mono (uint16). That
  touches `FrameRing`'s uint8 assumption, QuadGuide's `FrameBuffer`, the HUD
  encode path, MOSSE's grayscale entry point, and every NN preprocess entry
  point. Across a repo boundary that is a coordinated two-sided change with a
  shared-memory ABI bump in the middle; inside one repo it is a single refactor
  gated by one test suite.
- **Spec 3** adds a classical small-target seeker whose IMM filter needs FC body
  rates for LOS derotation. Those live on QuadGuide's bus, which EdgeCV
  architecturally cannot see.

This spec covers **only** the merge. Specs 2 and 3 are written against the merged
repo.

## Goal

Fold EdgeCV into the QuadGuide repository as a **pure relocation with zero
behaviour change**.

### Success criteria

Success is defined negatively and must be demonstrated, not asserted:

1. Both test suites pass. Test-body edits are limited to a **closed, enumerated
   set of 14 path-coupling lines across 13 files** (listed in the implementation
   plan): 10 helper-import prefixes (`from tests._nn_stubs` →
   `from tests.edgecv._nn_stubs`), 3 `Path(__file__).parents[...]` expressions,
   and the `tools/` path insert in `conftest.py`. `git diff tests/` must show no
   assertion, fixture, or logic changes — only those lines.

   These exist because EdgeCV's tests import shared helpers as an absolute
   `tests.` package and reach for `tools/` and `edgecv/models/manifests` by
   relative filesystem path, both of which shift by one directory level in the
   move.
2. `scripts/bench_tracker.py` produces identical output before and after. It is
   deterministic and synthesized ("No camera/hardware required"), which makes it
   the natural regression oracle.
3. A wheel built from the merged repo installs into a clean venv and resolves
   model manifests (see "The packaging trap" below).
4. Device smoke test: `systemctl restart quadguide`, HUD live at the configured
   address, NanoTrack acquires and locks.

If any of these cannot be shown, the merge is not done.

## Findings that shape the design

Established by inspection of both repos:

- **EdgeCV has exactly one commit** (`b03eae6 kalman`). There is no history to
  preserve. No `git subtree`, no `--allow-unrelated-histories`, no
  `git filter-repo`. The merge is a file move plus one commit.
- **QuadGuide has no CI at all.** No `.github/` directory is tracked on any
  branch. EdgeCV has a working x86 workflow. The merge *gains* QuadGuide a
  hardware-free test loop rather than costing it one.
- **Neither repo uses Git LFS.** Both commit real blobs directly: EdgeCV 7
  `.rknn` files (~7 MB) via `.gitignore` negations, QuadGuide 4 `.onnx` files
  (~4 MB) via `!models/**/*.onnx`. `scripts/firstboot_install.sh:67` enables LFS
  smudge "so `git clone` auto-downloads EdgeCV's `*.rknn`" — that step is
  vestigial and is deleted here.
- **Filename collisions are limited to six root entries**: `ARCHITECTURE.md`,
  `README.md`, `pyproject.toml`, `docs/`, `models/`, `tests/`. Four are
  directories whose *contents* do not collide — verified zero overlaps in
  `docs/superpowers/specs`, `docs/superpowers/plans`, and `models/`;
  `__init__.py` is the only overlap under `tests/`.

## Design

### 1. Destination: `src/edgecv/` as a second top-level package

EdgeCV lands at `src/edgecv/`, **not** nested under `src/quadguide/`.

EdgeCV's internal imports are uniformly `from edgecv.x import y`; its backend
plugin entry points are `edgecv.backends.onnx:OnnxBackend` and siblings; the
adapter does `import edgecv`. Landing at `src/edgecv/` means **zero import
rewriting** — `setuptools.packages.find(where=["src"])` discovers both
`quadguide` and `edgecv` automatically, and every one of those references keeps
working unchanged.

Nesting at `src/quadguide/perception/edgecv/` would require rewriting `from
edgecv.` across ~45 source modules and ~45 test files plus the entry points: a
90-file mechanical diff on a change whose entire value is that nothing changed.

Two top-level packages is untidy. That cost is accepted deliberately in exchange
for a merge that is provably identical. Nesting can be revisited in spec 3, when
there is an actual coupling reason to justify the churn.

### 2. File disposition

| From (EdgeCV) | To (QuadGuide) | Note |
|---|---|---|
| `edgecv/**` | `src/edgecv/**` | verbatim; no edits |
| `tests/*.py` | `tests/edgecv/` | subdirectory; `__init__.py` is the only basename collision |
| `tests/conftest.py` | `tests/edgecv/conftest.py` | remains scoped to EdgeCV tests only |
| `tools/` | `tools/` | new top-level. Host-only conversion tooling, kept distinct from `scripts/` (device/dev ops) — different audiences, different install requirements |
| `models/nanotrack_quant_rk3588/`, `models/nanotrack_quant_rk3566/` | `models/` | unions with QuadGuide's `*.onnx`; no path overlap |
| `ARCHITECTURE.md` | `docs/architecture-edgecv.md` | QuadGuide's `ARCHITECTURE.md` stays at root as the system document; the two cross-link |
| `README.md` | folded into QuadGuide's `README.md` as a section | |
| `docs/superpowers/specs/*`, `docs/superpowers/plans/*` | union into QuadGuide's | verified zero collisions |
| `.github/workflows/ci.yml` | `.github/workflows/ci.yml` | adopted; see §4 |
| `.gitignore` | merged | reconcile the two `models/` rule sets into one coherent block |

`.gitattributes` stays QuadGuide's (`*.sh text eol=lf`); EdgeCV has none.

### 3. `pyproject.toml` reconciliation

setuptools (QuadGuide's) wins; hatchling is dropped. Concretely:

- **`requires-python`** becomes `>=3.11` — QuadGuide's floor, matching the
  device's Python. EdgeCV drops its 3.10 support; the CI matrix becomes
  3.11 / 3.12.
- **Package discovery**: `[tool.setuptools.packages.find] where = ["src"]`
  already present; it picks up both packages with no change.
- **Entry points**: `[project.entry-points."edgecv.backends"]` carries over
  verbatim (`mock`, `onnx`, `rknn`).
- **Optional dependencies**: union EdgeCV's `fast` / `onnx` / `rknn` / `dev` /
  `test` with QuadGuide's `ground`. The merged `test` extra is the union of both
  repos' test requirements plus the two CI-specific additions from below:

  ```toml
  test = [
      "pytest>=8.0",                # QuadGuide's floor (>EdgeCV's 7.0)
      "ruff>=0.4", "mypy>=1.8",     # from EdgeCV
      "onnxruntime>=1.16", "onnx>=1.15",   # from EdgeCV
      "httpx>=0.27",                # QuadGuide — FastAPI TestClient
      "opencv-python-headless>=4.10",      # CI only; see below
  ]
  ```
- **ruff config** carries over; `line-length = 100`, `select = ["E","F","I","UP","B"]`.
- **pytest config**: `pythonpath = ["src"]` and `testpaths = ["tests"]` are
  already correct and now cover both packages.

#### Fix: QuadGuide's dependencies are under-declared

`pyproject.toml` currently declares `dependencies = ["pyyaml"]`, but
`src/quadguide` imports `numpy`, `pymavlink`, `pydantic`, `serial` (pyserial),
`fastapi`, `uvicorn`, `yaml`, and `cv2`. The real dependency list has been living
in `requirements.txt`, which `pip install -e .` never reads.

This is a latent bug that the merge forces into the open: EdgeCV's CI install
step is `pip install -e .[onnx,test]`, which on the merged repo would install
`pyyaml` plus extras and then fail QuadGuide's tests at import. The merged
`pyproject.toml` must declare the real runtime dependencies. `requirements.txt`
is then reduced to a pin/comment file or deleted.

#### Fix: cv2 must stay out of the base dependencies

The device deliberately uses **apt's `python3-opencv`** via the venv's
`--system-site-packages`, because PyPI's opencv wheels ship **without GStreamer
support** and `CSICamera` requires it (documented in
`docs/ov9281-bringup-context.md` step 3). A pip-installed opencv would shadow the
apt build and silently break the camera path.

But CI still needs `cv2` importable, and `ubuntu-latest` has no `libGL.so.1`, so
plain `opencv-python` raises `ImportError` at collection and takes the whole
suite down.

Resolution: declare **`opencv-python-headless` in the `test` extra only**. It is
API-identical for everything exercised in CI (`resize`, `cvtColor`, `warpAffine`,
`imencode`). The only GUI cv2 usage anywhere is `tools/track_webcam.py:244-290`,
a host-only dev tool that CI does not run.

#### The packaging trap

Hatchling's `[tool.hatch.build.targets.wheel.force-include]` ships
`edgecv/models/profiles` and `edgecv/models/manifests`. Under setuptools this
must become:

```toml
[tool.setuptools.package-data]
"edgecv.models" = ["profiles/*.yaml", "manifests/*.yaml"]
```

**If this translation is wrong, nothing fails locally.**
`EdgeCVTracker._manifests_dir()` resolves via `Path(edgecv.__file__).parent`, so
a source checkout works and the entire test suite passes green. Only an installed
wheel breaks, and only at model-load time on the device.

This is why success criterion 3 is a wheel-install test and is not optional.

### 4. CI

#### Repository disposition

- **`plastheis/QuadGuide`** becomes the single home. All CI runs there.
- **`plastheis/EdgeCV`** is **archived** (GitHub → Settings → Archive this
  repository) once the merge is verified on the device. Its `README.md` is first
  replaced with a pointer to QuadGuide. Archiving makes it read-only, stops its
  Actions from running, and keeps the URL alive so existing links and clones do
  not 404. With one commit and one consumer, nothing is lost.
- **Ordering:** archive **after** device verification, not before.
  `firstboot_install.sh` still clones EdgeCV until the merge lands, and the
  rollback path must stay intact until the merged build is confirmed good.

#### Workflow

QuadGuide gains `.github/workflows/ci.yml`, adapted from EdgeCV's:

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
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
      - name: Lint
        run: ruff check src/edgecv tests/edgecv
      - name: Type-check
        run: mypy src/edgecv
      - name: Test
        run: pytest -q
```

Notes:

- `ubuntu-latest` is required, not incidental: `src/quadguide` imports `fcntl`
  and `select` and calls `os.sched_setaffinity`.
- The `ground` extra is added to the install because QuadGuide's tests import
  `starlette` via FastAPI's `TestClient`.

#### Lint and type-check scope stays where it is

EdgeCV's CI runs `ruff check edgecv tests` and `mypy edgecv`. QuadGuide's code
has **never** been linted or type-checked. The workflow above therefore keeps the
enforcement scope *identical to today* — `src/edgecv` and `tests/edgecv` only —
so the merge remains process-neutral as well as behaviour-neutral.

Widening ruff/mypy to `src/quadguide` inside this merge would mean fixing an
unknown number of pre-existing violations in a change whose purpose is to prove
that nothing changed. It is filed as a follow-up: measure first
(`ruff check src/quadguide --statistics`), then decide whether to fix or to
configure per-rule ignores.

### 5. Deployment rewiring

- **`scripts/firstboot_install.sh`**: delete the EdgeCV clone (step 2), the
  separate `pip install -e "$EDGECV_DIR"`, the `EDGECV_REPO`/`EDGECV_DIR`
  variables, the vestigial Git-LFS smudge enable, and the `/home/radxa/EdgeCV`
  path warnings. Models now arrive with the QuadGuide clone.
- **`configs/rk3588.yaml`**: `tracker.params.model_dir` moves from
  `/home/radxa/EdgeCV/models` to `/home/radxa/quadguide/models`.
- **`requirements.txt`**: remove the EdgeCV install instructions block (lines
  20–28); reduce or delete the file per §3.
- **`scripts/firstboot_install_rpi.sh`**: same treatment if it carries equivalent
  EdgeCV steps.

#### Migration footgun (must appear in the runbook)

The device currently has EdgeCV installed editable at `~/EdgeCV`, and
`requirements.txt` documents an alternative user `.pth`
(`~/.local/lib/python3.11/site-packages/edgecv.pth`). **Either one left in place
will shadow the merged package** — the device would silently import the old
EdgeCV while every local test passes green.

Removing both is a required migration step, not a cleanup:

```bash
pip uninstall -y edgecv
rm -f ~/.local/lib/python3.11/site-packages/edgecv.pth
rm -rf ~/EdgeCV          # only after the merged build is verified
python -c "import edgecv; print(edgecv.__file__)"   # must be under ~/quadguide/src
```

## Out of scope

- Import rewriting to nest `edgecv` under `quadguide` (§1).
- The 10-bit uint16 pixel path — **spec 2**.
- MPCM/Otsu detector, IMM filter, detector-coupled measurement covariance —
  **spec 3**.
- Reconciling the two ARCHITECTURE documents into a single narrative. They are
  relocated and cross-linked only.
- Extending ruff/mypy coverage to `src/quadguide` (§4).
- Any behaviour change to `configs/rpi4b.yaml` or `configs/config.yaml`.

## Verification plan

| # | Check | Command |
|---|---|---|
| 1 | Both suites pass | `pytest -q` |
| 2 | Lint unchanged in scope | `ruff check src/edgecv tests/edgecv` |
| 3 | Types unchanged in scope | `mypy src/edgecv` |
| 4 | Wheel ships manifests | build wheel, install into clean venv, `load_manifest` resolves |
| 5 | Tracker output identical | `scripts/bench_tracker.py` diffed against a pre-merge run |
| 6 | Device smoke | `systemctl restart quadguide`; HUD live; NanoTrack locks |
| 7 | No shadowed package | `python -c "import edgecv; print(edgecv.__file__)"` on device |

Checks 1–5 run on the dev box; 6–7 on the ROCK 5C.

## Follow-ups (not this spec)

- Measure and then extend ruff/mypy to `src/quadguide`.
- Decide whether `tools/` and `scripts/` should consolidate.
- Reconcile the two architecture documents once specs 2 and 3 have settled the
  seeker's shape.
