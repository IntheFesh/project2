# LIBERO-Plus integration notes (verification log)

**Purpose.** LIBERO-Plus is the primary perturbation suite. This file records the **verified**
facts the repo relies on, and flags everything still unverified as `TODO(verify ...)`. Project
rule #4: *never guess LIBERO-Plus specifics* — verify, or leave a precise seam.

**Single source of truth.** All these constants live in
[`perturb/libero_plus_constants.py`](../perturb/libero_plus_constants.py); downstream code imports
from there, so correcting a value is a one-line edit.

**Verification provenance.** Verified **2026-06-06** against the official repository
`github.com/sylvestf/LIBERO-plus` (paper arXiv:2510.13626) by reading: `README.md`,
`benchmark_scripts/render_single_task.py`, `benchmark_scripts/check_task_suites.py`, and
`libero/libero/benchmark/task_classification.json`. The package is **not installed in this
off-GPU dev container**, so a few details could only be partially observed; those are marked
`TODO(verify on GPU box)` and must be re-confirmed against the *installed* package.

---

## 1. The eval unit is a pre-built task instance (NOT "N inits per base task")

LIBERO-Plus is a drop-in replacement for `libero` (`pip install -e .`). Its perturbations are
**not parametric**: it ships a large set of **pre-built tasks**, each a fixed
`(base task × perturbation category × difficulty)` instance. The mapping
`task → (category, difficulty_level)` lives in `task_classification.json`. **VERIFIED.**

- **Evaluation protocol:** run **each task once** — set `num_trials_per_task = 1` (README:
  "adjusting `num_trials_per_task` from 50 to 1"). **VERIFIED.**
- Consequence for this repo: the eval unit is a *task instance*, and pairing across conditions
  A/B/C is done by running the **same set of task IDs** under each condition and matching on
  `task_id` (not by resampling init states). See `docs/EVALUATION.md`.

## 2. `task_classification.json`

- **Location (in the installed package):** `libero/libero/benchmark/task_classification.json`.
  **VERIFIED** (README also writes it as `.libero/libero/benchmark/...`).
- **Structure:** a JSON **object keyed by suite name** (e.g. `"libero_spatial"`) → **array** of
  entries. **VERIFIED.**
- **Entry fields:** `{"id": int, "name": str, "category": str, "difficulty_level": int}`.
  **VERIFIED** (example: `{"id": 1, "name": "pick_up_the_black_bowl_..._table_1",
  "category": "Background Textures", "difficulty_level": 2}`).
- **`difficulty_level`:** integer in **1..5**. **VERIFIED** — this confirms the repo's L1–L5
  assumption *as a global range*. `TODO(verify on GPU box):` whether every category populates all
  5 levels or only a subset.
- `TODO(verify on GPU box):` is the `id` field **0-based or 1-based** relative to
  `benchmark.get_task(task_id)`? (The observed example used `id: 1`.)

## 3. Perturbation categories (`category` field)

The README lists **seven** dimensions; the JSON `category` strings observed/inferred:

| repo family | LIBERO-Plus `category` | status |
|---|---|---|
| `texture` | `"Background Textures"` | **VERIFIED verbatim** |
| `robot_init` | `"Robot Initial States"` | **VERIFIED verbatim** |
| `viewpoint` | `"Camera Viewpoints"` | `TODO(verify exact string)` |
| `lighting` | `"Light Conditions"` | `TODO(verify exact string)` |
| `noise` | `"Sensor Noise"` | `TODO(verify exact string)` |
| `layout` | `"Objects Layout"` | `TODO(verify exact string)` |
| *(unused by CORE/STRETCH)* | `"Language Instructions"` | `TODO(verify exact string)` |

The 4 inferred strings follow the README's dimension names + the verified naming convention
(title-cased full phrases) but were not individually observed in the JSON. `"Language
Instructions"` is intentionally **not** a CORE/STRETCH family — it is exercised by the
language-conditioning probe (`eval/probe.py`).

## 4. Clean / unperturbed baseline

The observed JSON slice contained **only perturbed** instances. Most likely the clean baseline is
the **original LIBERO suites** (run via standard LIBERO), not a category in this JSON.
`TODO(verify on GPU box):` confirm whether the full JSON contains an `"original"`/`"clean"`
category; if not, the clean set = original LIBERO tasks. `CLEAN_CATEGORY` in the constants module
is `None` to encode this.

## 5. Task enumeration & environment construction API

From `benchmark_scripts/check_task_suites.py` and `render_single_task.py`. **VERIFIED** (mirror
these exactly in `make_perturbed_env`):

```python
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bench = benchmark.get_benchmark_dict()[benchmark_name]()   # benchmark_name = a suite
num_tasks  = bench.get_num_tasks()
task_names = bench.get_task_names()
task        = bench.get_task(task_id)                      # task_id: int index within the suite
init_states = bench.get_task_init_states(task_id)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=128, camera_widths=128)
```

- **Suites** (`get_benchmark_dict()` keys): `libero_object`, `libero_goal`, `libero_spatial`,
  `libero_10`, `libero_90`. **VERIFIED.**
- `TODO(verify on GPU box):` exact derivation of `bddl_file` from a LIBERO-Plus task
  (standard LIBERO uses `os.path.join(get_libero_path("bddl_files"), task.problem_folder,
  task.bddl_file)`; the render script also accepts an explicit `--bddl_file`).

## 6. Install (system + python + assets + config) — see `deploy.py` / `run.sh`

- **apt system packages** (README): `libexpat1`, `libfontconfig1-dev`, `libpython3-stdlib`,
  `libmagickwand-dev` (ImageMagick/Wand backs texture/background perturbations). **VERIFIED.**
- **pip:** `pip install -e .` then `pip install -r extra_requirements.txt`. **VERIFIED.**
- **assets:** download **`assets.zip`** from the HF dataset **`Sylvest/LIBERO-plus`** and unzip so
  `assets/` lands at `<libero-plus>/libero/libero/assets/` (README: "unzip ... to
  `/LIBERO-plus/libero/libero`"). **VERIFIED** (filename + target path).
- **config:** `libero_config_path` at `/root/.libero/config.yaml` (default; confirm in
  `libero/libero/__init__.py`). **VERIFIED** (path); `TODO(verify):` exact YAML keys to patch.

## 7. Note on Condition-C augmentation vs LIBERO-Plus viewpoint perturbation

LIBERO-Plus `Camera Viewpoints` is a genuine **3-D camera move**; a 2-D torchvision transform on a
single frame cannot reproduce it. So `level_to_fraction` / `aligned_magnitude` are a deliberate
**training-side heuristic**, not a claim about LIBERO-Plus internal magnitudes. The honest
alternative (fine-tune C on LIBERO-Plus's released perturbation training data) is documented as an
optional stub in the report's limitations; torchvision augmentation remains the default.
