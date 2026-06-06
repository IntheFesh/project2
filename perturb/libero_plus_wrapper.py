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


def make_perturbed_env(task_id: str, **kwargs):
    """Build the LIBERO-Plus simulator env for a pre-built task ``task_id`` (``"<suite>:<id>"``).

    GPU/simulator seam. The perturbation is baked into the task, so there is no ``PerturbSpec``
    argument. Implementation must mirror LIBERO-Plus ``benchmark_scripts/`` (VERIFIED API)::

        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        bench = benchmark.get_benchmark_dict()[suite]()
        task = bench.get_task(task_index)
        env = OffScreenRenderEnv(bddl_file_name=<task bddl>, camera_heights=128, camera_widths=128)

    TODO(verify on GPU box): whether the JSON ``id`` is 0- or 1-based vs ``get_task`` and the exact
    ``bddl_file`` derivation (see docs/LIBERO_PLUS_NOTES.md).
    """
    raise NotImplementedError(
        "Phase 2: construct the LIBERO-Plus env for the pre-built task "
        f"{task_id!r} (install LIBERO-Plus as the `libero` drop-in; mirror benchmark_scripts)."
    )
