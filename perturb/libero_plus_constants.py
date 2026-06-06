"""Single source of truth for LIBERO-Plus integration constants.

All values were verified against the official LIBERO-Plus repository
(``github.com/sylvestf/LIBERO-plus``, arXiv:2510.13626) on 2026-06-06 -- see
``docs/LIBERO_PLUS_NOTES.md`` for the verification log and the items still marked
``TODO(verify)`` (to confirm against the *installed* package on the GPU box).

Downstream code MUST import these names instead of hardcoding strings, so a single edit here
fixes everything if a value turns out to differ. Stdlib-only (import-safe off-GPU).
"""

from __future__ import annotations

# ``task_classification.json``: a JSON object keyed by suite name -> list of entries
# {"id": int, "name": str, "category": str, "difficulty_level": int}.  VERIFIED.
TASK_CLASSIFICATION_RELPATH = "libero/libero/benchmark/task_classification.json"

# LIBERO suite names = keys of ``benchmark.get_benchmark_dict()``.  VERIFIED
# (benchmark_scripts/check_task_suites.py).
LIBERO_SUITES: tuple[str, ...] = (
    "libero_object",
    "libero_goal",
    "libero_spatial",
    "libero_10",
    "libero_90",
)

# Difficulty levels. VERIFIED: ``difficulty_level`` is an integer in [1, 5].
# Caveat: levels are heterogeneous task bins, NOT a parametric magnitude of one knob.
MIN_LEVEL = 1
MAX_LEVEL = 5
DIFFICULTY_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)

# Repo family -> LIBERO-Plus ``category`` string in task_classification.json.
#   VERIFIED verbatim : "Background Textures", "Robot Initial States".
#   TODO(verify)      : the other four strings are inferred from the README's 7-dimension list
#                       + the verified naming convention; confirm against the installed JSON.
FAMILY_TO_CATEGORY: dict[str, str] = {
    "texture": "Background Textures",        # VERIFIED
    "robot_init": "Robot Initial States",    # VERIFIED
    "viewpoint": "Camera Viewpoints",        # TODO(verify exact string)
    "lighting": "Light Conditions",          # TODO(verify exact string)
    "noise": "Sensor Noise",                 # TODO(verify exact string)
    "layout": "Objects Layout",              # TODO(verify exact string)
}

# Inverse map (category string -> repo family), for labelling rollout rows from a task's category.
CATEGORY_TO_FAMILY: dict[str, str] = {v: k for k, v in FAMILY_TO_CATEGORY.items()}

# Categories whose exact strings are confirmed verbatim against the JSON.
VERIFIED_CATEGORIES: frozenset[str] = frozenset({"Background Textures", "Robot Initial States"})

# The "Language Instructions" category is intentionally NOT a CORE/STRETCH family; it is
# exercised by the language-conditioning probe (eval/probe.py).  TODO(verify exact string).
LANGUAGE_CATEGORY = "Language Instructions"

# How "clean"/unperturbed is represented. The observed JSON held only perturbed instances, so the
# clean baseline is most likely the ORIGINAL LIBERO suites (run via standard LIBERO), not a
# category here.  ``None`` encodes "clean == original LIBERO suites".
# TODO(verify on GPU box): confirm whether the full JSON has an "original"/"clean" category, and
# whether the ``id`` field is 0- or 1-based relative to ``benchmark.get_task(task_id)``.
CLEAN_CATEGORY: str | None = None
