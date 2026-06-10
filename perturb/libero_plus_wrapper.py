"""Selector over LIBERO-Plus (arXiv 2510.13626) -- task-SELECTION semantics.

LIBERO-Plus is a drop-in replacement for ``libero`` that ships pre-built
``(base task x perturbation category x difficulty)`` task instances; the mapping
``task -> (category, difficulty_level)`` lives in ``task_classification.json``. We therefore do
**not** reimplement perturbations and do **not** sample init states: we *select* pre-built task
IDs by ``(family, level)`` and run each one once (``num_trials_per_task = 1``).

Pure logic (parsing, grouping, selection, distribution labelling) is implemented and unit-tested
here. Constructing the simulator env for a given ``task_id`` is the GPU/simulator seam (it mirrors
LIBERO-Plus ``benchmark_scripts/``). All verified strings/paths come from
``perturb/libero_plus_constants.py``; see ``docs/LIBERO_PLUS_NOTES.md``.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from perturb.libero_plus_constants import (
    CATEGORY_TO_FAMILY,
    FAMILY_TO_CATEGORY,
    MAX_LEVEL,
    MIN_LEVEL,
)

CORE_FAMILIES: tuple[str, ...] = ("viewpoint", "lighting", "texture", "noise")
STRETCH_FAMILIES: tuple[str, ...] = ("layout", "robot_init")
ALL_FAMILIES: tuple[str, ...] = CORE_FAMILIES + STRETCH_FAMILIES


# --------------------------------------------------------------------------- selector key --
@dataclass(frozen=True)
class PerturbSpec:
    """A LIBERO-Plus **selector key**: a ``family`` at a difficulty ``level`` (1-5).

    This is *not* an env-constructor parameter -- the perturbation is baked into each pre-built
    LIBERO-Plus task. A ``PerturbSpec`` is used to *look up* matching pre-built task IDs (via
    :func:`select_perturbed_tasks`) and to label results (:func:`classify_distribution`).
    """

    family: str
    level: int

    def __post_init__(self) -> None:
        if self.family not in ALL_FAMILIES:
            raise ValueError(
                f"unknown perturbation family {self.family!r}; expected one of {ALL_FAMILIES}"
            )
        if not (MIN_LEVEL <= self.level <= MAX_LEVEL):
            raise ValueError(
                f"level must be in [{MIN_LEVEL}, {MAX_LEVEL}], got {self.level!r}"
            )

    @property
    def is_core(self) -> bool:
        """True if this family is part of the CORE suite (viewpoint/lighting/texture/noise)."""
        return self.family in CORE_FAMILIES

    @property
    def category(self) -> str:
        """The LIBERO-Plus ``category`` string this family maps to."""
        return category_for_family(self.family)

    def __str__(self) -> str:
        return f"{self.family}:{self.level}"


def parse_perturb_spec(text: str | None) -> PerturbSpec | None:
    """Parse a ``'family:level'`` string (e.g. ``'viewpoint:4'``); ``None``/empty -> clean (None)."""
    if text is None or text.strip() == "":
        return None
    if ":" not in text:
        raise ValueError(f"perturbation spec must be 'family:level', got {text!r}")
    family, _, level = text.partition(":")
    try:
        level_int = int(level)
    except ValueError as exc:
        raise ValueError(f"level must be an integer in {text!r}") from exc
    return PerturbSpec(family=family.strip(), level=level_int)


# ----------------------------------------------------------------- family <-> category --
def category_for_family(family: str) -> str:
    """Return the LIBERO-Plus ``category`` string for a repo ``family`` (raises if unknown)."""
    try:
        return FAMILY_TO_CATEGORY[family]
    except KeyError as exc:
        raise ValueError(
            f"unknown family {family!r}; expected one of {tuple(FAMILY_TO_CATEGORY)}"
        ) from exc


def family_for_category(category: str) -> str | None:
    """Return the repo ``family`` for a LIBERO-Plus ``category`` (``None`` if unmapped, e.g. Language)."""
    return CATEGORY_TO_FAMILY.get(category)


def level_to_fraction(level: int) -> float:
    """Map a difficulty ``level`` (1-5) to a normalized magnitude in ``(0, 1]`` = ``level/MAX_LEVEL``.

    **Caveat (important):** LIBERO-Plus difficulty levels are *heterogeneous task bins*, NOT a
    clean linear magnitude of a single parameter. ``level/MAX_LEVEL`` is a deliberate
    **training-side heuristic** used only to scale Condition-C augmentation
    (:func:`data.augment.visual_aug.aligned_magnitude`); it makes no claim about LIBERO-Plus's
    internal perturbation magnitudes.
    """
    if not (MIN_LEVEL <= level <= MAX_LEVEL):
        raise ValueError(f"level must be in [{MIN_LEVEL}, {MAX_LEVEL}], got {level!r}")
    return level / MAX_LEVEL


def classify_distribution(spec: PerturbSpec | None, trained_families: Sequence[str]) -> str:
    """Tag an eval ``spec`` as ``"clean"``, ``"in_dist"`` or ``"held_out"``.

    Honesty guard (project rule #3): a family seen during augmentation is *in-distribution*; one
    never seen is *held-out generalization*. Single source of truth for labelling every reported
    perturbation result, so in-dist recovery is never presented as generalization.
    """
    if spec is None:
        return "clean"
    return "in_dist" if spec.family in set(trained_families) else "held_out"


# ---------------------------------------------------------- task_classification.json model --
@dataclass(frozen=True)
class LiberoPlusTask:
    """One pre-built LIBERO-Plus task instance from ``task_classification.json``."""

    suite: str
    id: int
    name: str
    category: str
    difficulty_level: int

    @property
    def uid(self) -> str:
        """Stable identifier ``"<suite>:<id>"`` used in rollout CSVs and env construction."""
        return f"{self.suite}:{self.id}"

    @property
    def family(self) -> str | None:
        """Repo family for this task's category (``None`` for unmapped categories, e.g. Language)."""
        return family_for_category(self.category)


def parse_task_classification(data: Mapping[str, Sequence[Mapping]]) -> list[LiberoPlusTask]:
    """Flatten a loaded ``task_classification.json`` (``{suite: [entry, ...]}``) into task objects.

    Raises:
        ValueError: if an entry is missing a required field.
    """
    tasks: list[LiberoPlusTask] = []
    for suite, entries in data.items():
        for entry in entries:
            try:
                tasks.append(
                    LiberoPlusTask(
                        suite=str(suite),
                        id=int(entry["id"]),
                        name=str(entry["name"]),
                        category=str(entry["category"]),
                        difficulty_level=int(entry["difficulty_level"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed task entry in suite {suite!r}: {entry!r} ({exc})") from exc
    return tasks


def group_by_category_level(
    tasks: Sequence[LiberoPlusTask],
) -> dict[tuple[str, int], list[LiberoPlusTask]]:
    """Group tasks by ``(category, difficulty_level)`` cell (preserving input order within a cell)."""
    groups: dict[tuple[str, int], list[LiberoPlusTask]] = defaultdict(list)
    for task in tasks:
        groups[(task.category, task.difficulty_level)].append(task)
    return dict(groups)


def select_perturbed_tasks(
    tasks: Sequence[LiberoPlusTask],
    family: str,
    level: int,
    n_instances: int,
    *,
    seed: int = 0,
) -> list[str]:
    """Deterministically select up to ``n_instances`` task UIDs for a ``(family, level)`` cell.

    Maps ``family`` -> LIBERO-Plus category, filters ``tasks`` to that ``(category, level)``,
    then deterministically samples (seeded) up to ``n_instances`` of them, clamping to however
    many exist.

    Returns:
        A list of task UIDs (``"<suite>:<id>"``), stable for a given ``seed``.

    Raises:
        ValueError: if ``family`` is unknown, ``level`` is out of range, or ``n_instances`` < 0.
    """
    category = category_for_family(family)
    if not (MIN_LEVEL <= level <= MAX_LEVEL):
        raise ValueError(f"level must be in [{MIN_LEVEL}, {MAX_LEVEL}], got {level!r}")
    if n_instances < 0:
        raise ValueError("n_instances must be >= 0")

    matches = sorted(
        (t for t in tasks if t.category == category and t.difficulty_level == level),
        key=lambda t: (t.suite, t.id),
    )
    if n_instances >= len(matches):
        chosen = matches
    else:
        chosen = random.Random(seed).sample(matches, n_instances)
    return [t.uid for t in chosen]


def parse_task_uid(uid: str) -> tuple[str, int]:
    """Parse a ``"<suite>:<id>"`` task UID into ``(suite, id)``."""
    suite, sep, raw = uid.rpartition(":")
    if not sep:
        raise ValueError(f"task uid must be '<suite>:<id>', got {uid!r}")
    try:
        return suite, int(raw)
    except ValueError as exc:
        raise ValueError(f"task uid must end in an integer id, got {uid!r}") from exc


#: Per-suite episode-length budget (mirrors LeRobot envs/libero.py TASK_SUITE_MAX_STEPS).
LIBERO_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
#: Default if a suite is unknown.
LIBERO_DEFAULT_MAX_STEPS = 300
#: SmolVLA-LIBERO was trained at 256x256.
LIBERO_OBS_HEIGHT = 256
LIBERO_OBS_WIDTH = 256
#: Number of dummy "no-op" steps after reset to let the scene physics settle.
LIBERO_NUM_STEPS_WAIT = 10


def get_libero_dummy_action():
    """No-op 7-D action used during the post-reset settle phase. Matches LeRobot envs/libero.py."""
    return [0, 0, 0, 0, 0, 0, -1]


# Regex to strip LIBERO-Plus task-name suffixes that aren't part of the natural-language
# instruction. The Task NamedTuple builds .language as " ".join(name.split("_")), which leaves
# garbage like "table 1" in the instruction string. The 5 known LIBERO-Plus suffix families:
#   _table_N, _tb_N, _view_N, _language_N, _light_N
# All are integer-indexed scene variants and carry no semantic meaning for the task itself.
_LIBERO_PLUS_SUFFIX_RE = re.compile(
    r"\s+(?:table|tb|view|language|light)\s+\d+\s*$"
)


def _clean_libero_plus_instruction(raw: str) -> str:
    """Strip trailing scene-variant suffix from a LIBERO-Plus task instruction.

    ``"pick up the black bowl ... place it on the plate table 1"``
        -> ``"pick up the black bowl ... place it on the plate"``
    """
    return _LIBERO_PLUS_SUFFIX_RE.sub("", raw).strip()


def _load_libero_plus_init_states(bench, idx):
    """Load init states for task ``idx`` using weights_only=False (trusted pickle).

    Re-implements ``Benchmark.get_task_init_states`` (see
    ``LIBERO-plus/libero/libero/benchmark/__init__.py:192-243``) but passes
    ``weights_only=False`` to ``torch.load``. The suffix routing logic
    (``_table_N``, ``_tb_N``, ``_view_N``, ``_language_N``, ``_light_N``, ``_add_``,
    ``_level``) mirrors LIBERO-Plus exactly so file resolution is identical.
    """
    import os

    import torch
    from libero.libero import get_libero_path

    task = bench.tasks[idx]
    init_states_file = task.init_states_file
    problem_folder = task.problem_folder
    base_dir = get_libero_path("init_states")

    # Routing logic copied from LIBERO-plus benchmark/__init__.py::get_task_init_states.
    init_states_path = None
    if "_language_" in init_states_file:
        rebased = init_states_file.split("_language_")[0] + "." + init_states_file.split(".")[-1]
        init_states_path = os.path.join(base_dir, problem_folder, rebased)
    elif "_view_" in init_states_file:
        rebased = init_states_file.split("_view_")[0] + "." + init_states_file.split(".")[-1]
        init_states_path = os.path.join(base_dir, problem_folder, rebased)
    elif "_table_" in init_states_file:
        rebased = re.sub(r"_table_\d+", "", init_states_file)
        init_states_path = os.path.join(base_dir, problem_folder, rebased)
    elif "_tb_" in init_states_file:
        rebased = re.sub(r"_tb_\d+", "", init_states_file)
        init_states_path = os.path.join(base_dir, problem_folder, rebased)
    elif "_light_" in init_states_file:
        rebased = init_states_file.split("_light_")[0] + "." + init_states_file.split(".")[-1]
        init_states_path = os.path.join(base_dir, problem_folder, rebased)
    elif "_add_" in init_states_file or "_level" in init_states_file:
        init_states_path = os.path.join(base_dir, "libero_newobj", problem_folder, init_states_file)
    else:
        init_states_path = os.path.join(base_dir, problem_folder, init_states_file)

    if not os.path.isfile(init_states_path):
        raise FileNotFoundError(
            f"init states file missing for task idx={idx} name={task.name!r}: {init_states_path}"
        )
    init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
    if "_add_" in init_states_file or "_level" in init_states_file:
        init_states = init_states.reshape(1, -1)
    return init_states


def make_perturbed_env(task_id: str, *, camera_heights: int = LIBERO_OBS_HEIGHT,
                       camera_widths: int = LIBERO_OBS_WIDTH, silent: bool = True,
                       init_state_id: int = 0):
    """Build a LIBERO env for ``task_id="<suite>:<id>"``, ready for SmolVLA rollout.

    Uses LIBERO-Plus official APIs:
      * ``bench.get_task_bddl_file_path(idx)`` for the BDDL file path
      * ``bench.get_task_init_states(idx)`` for the init-states tensor (handles all 5
        suffix-based path rewrites: _table_N, _tb_N, _view_N, _language_N, _light_N)

    Mirrors LeRobot ``envs/libero.py::LiberoEnv`` setup so the policy sees the same observation
    distribution it was trained on:
      1. set_init_state to a fixed init state (controls reproducibility across A/B/C conditions)
      2. step 10 dummy actions so contact physics settles
      3. force relative (delta) controller -- SmolVLA was trained on delta actions

    The returned env carries:
      * ``env._libero_task_instruction``  -- the task's natural-language instruction (cleaned)
      * ``env._libero_suite``             -- suite name (for max_steps lookup)
      * ``env._libero_init_states``       -- the full init-states tensor (len ~50 per task)
    """
    import io, contextlib

    @contextlib.contextmanager
    def _maybe_silence():
        if silent:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                yield
        else:
            yield

    with _maybe_silence():
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        suite, idx = parse_task_uid(task_id)
        bench_dict = benchmark.get_benchmark_dict()
        if suite not in bench_dict:
            raise ValueError(f"unknown suite {suite!r}; available: {sorted(bench_dict)}")
        bench = bench_dict[suite]()
        if not (0 <= idx < bench.get_num_tasks()):
            raise IndexError(
                f"task index {idx} out of range for suite {suite!r} (has {bench.get_num_tasks()})"
            )
        # PyTorch 2.6+ defaults torch.load to weights_only=True, but LIBERO-Plus's
        # get_task_init_states does a bare torch.load on init-states files. The pickle
        # contains numpy ndarrays plus several internal numpy globals; enumerating them
        # all for safe_globals is brittle across numpy versions. Since the init-states
        # files are robosuite training-pipeline artifacts shipped with LIBERO-Plus
        # (not user-supplied), we trust them and load with weights_only=False directly.
        # We re-implement the suffix-resolution logic ourselves to bypass the bare
        # torch.load inside LIBERO-Plus.
        task = bench.get_task(idx)
        bddl = bench.get_task_bddl_file_path(idx)
        init_states = _load_libero_plus_init_states(bench, idx)

        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
        )
    env._libero_task_instruction = _clean_libero_plus_instruction(task.language)
    env._libero_suite = suite
    env._libero_init_states = init_states
    env._libero_init_state_id = init_state_id
    return env


def libero_reset_with_init(env, *, seed: int = 0):
    """Reset ``env`` with a deterministic init state + post-reset settle (mirrors LiberoEnv.reset)."""
    import numpy as np

    env.seed(seed)
    env.reset()
    init = env._libero_init_states[env._libero_init_state_id % len(env._libero_init_states)]
    obs = env.set_init_state(init)
    # Settle the simulation: 10 dummy steps so contacts stabilise (matches LiberoEnv).
    dummy = np.asarray(get_libero_dummy_action(), dtype=np.float32)
    for _ in range(LIBERO_NUM_STEPS_WAIT):
        obs, _r, _d, _i = env.step(dummy)
    # Force RELATIVE (delta-action) controller -- SmolVLA-LIBERO outputs deltas.
    for robot in env.robots:
        robot.controller.use_delta = True
    return obs
