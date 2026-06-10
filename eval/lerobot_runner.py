"""Subprocess wrapper around LeRobot's official ``lerobot-eval`` CLI.

Why a wrapper: the project's hand-written rollout path bypassed SmolVLA's normalization
(preprocessor/postprocessor are separate from the policy checkpoint), giving 0% SR. The
official ``lerobot-eval`` correctly wires LiberoProcessorStep + Normalizer/Unnormalizer.
We therefore drive the official CLI and parse its ``eval_info.json``.

Two eval regimes:
  * CLEAN  — original LIBERO 10-task suites, via PYTHONPATH=third_party/LIBERO-orig
             (the installed ``libero`` is LIBERO-Plus = 2402 perturbed variants; the
             original clean benchmark only exists in the separate LIBERO-orig clone).
  * PLUS   — LIBERO-Plus perturbed variants, via ``--env.type=libero_plus``.

Output rows match docs/EVALUATION.md: one row per (condition, task_id, family, level, seed,
episode) carrying a boolean ``success``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBERO_ORIG_PATH = REPO_ROOT / "third_party" / "LIBERO-orig"

# Verified-working CLI knobs (GitHub issue #3264 reproduction + our smoke tests).
_SMOLVLA_NUM_STEPS = 10
_SMOLVLA_N_ACTION_STEPS = 10


def _base_env() -> dict:
    """Environment variables every eval needs (HF mirror, EGL, threads)."""
    import os

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    env["MUJOCO_GL"] = "egl"
    env["OMP_NUM_THREADS"] = "8"
    for proxy in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(proxy, None)
    return env


def build_eval_command(
    *,
    policy_path: str,
    suite: str,
    task_ids: Sequence[int],
    n_episodes: int,
    output_dir: str | Path,
    seed: int,
    is_libero_plus: bool,
    batch_size: int = 1,
    episode_length: int | None = None,
) -> list[str]:
    """Construct the ``lerobot-eval`` argv for one eval call."""
    env_type = "libero_plus" if is_libero_plus else "libero"
    task_ids_str = "[" + ",".join(str(t) for t in task_ids) + "]"
    cmd = [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        f"--policy.num_steps={_SMOLVLA_NUM_STEPS}",
        f"--policy.n_action_steps={_SMOLVLA_N_ACTION_STEPS}",
        f"--env.type={env_type}",
        f"--env.task={suite}",
        f"--env.task_ids={task_ids_str}",
        f"--env.is_libero_plus={'true' if is_libero_plus else 'false'}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        f"--output_dir={output_dir}",
        f"--seed={seed}",
    ]
    if episode_length is not None:
        cmd.append(f"--env.episode_length={episode_length}")
    return cmd


def parse_eval_info(eval_info_path: str | Path) -> list[dict]:
    """Parse ``eval_info.json`` into per-(task_id, episode) success records.

    Returns a list of dicts: ``{"task_id": int, "episode": int, "success": bool}``.
    """
    data = json.loads(Path(eval_info_path).read_text())
    rows: list[dict] = []
    for entry in data.get("per_task", []):
        task_id = int(entry["task_id"])
        successes = entry["metrics"]["successes"]
        for ep_idx, succ in enumerate(successes):
            rows.append({"task_id": task_id, "episode": ep_idx, "success": bool(succ)})
    return rows


def run_eval(
    *,
    policy_path: str,
    suite: str,
    task_ids: Sequence[int],
    n_episodes: int,
    seed: int,
    is_libero_plus: bool,
    output_dir: str | Path | None = None,
    batch_size: int = 1,
    timeout_s: int | None = 600,  # per-call cap; EGL can hang on noise renders
    episode_length: int | None = None,
) -> list[dict]:
    """Run one ``lerobot-eval`` invocation and return parsed per-episode success rows.

    For CLEAN evals (``is_libero_plus=False``) we prepend ``third_party/LIBERO-orig`` to
    PYTHONPATH so the original 10-task benchmark shadows the installed LIBERO-Plus.

    Raises:
        FileNotFoundError: if the run produced no ``eval_info.json`` (eval crashed).
        subprocess.CalledProcessError: if the CLI exits non-zero.
    """
    env = _base_env()
    if not is_libero_plus:
        # Clean: force original LIBERO benchmark via PYTHONPATH isolation.
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(LIBERO_ORIG_PATH) + (":" + existing if existing else "")
        )

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix=f"lerobot_eval_{suite}_seed{seed}_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_eval_command(
        policy_path=policy_path,
        suite=suite,
        task_ids=task_ids,
        n_episodes=n_episodes,
        output_dir=output_dir,
        seed=seed,
        is_libero_plus=is_libero_plus,
        batch_size=batch_size,
        episode_length=episode_length,
    )
    subprocess.run(cmd, env=env, check=True, timeout=timeout_s)

    eval_info = output_dir / "eval_info.json"
    if not eval_info.is_file():
        raise FileNotFoundError(
            f"lerobot-eval produced no eval_info.json at {eval_info} (eval likely crashed)"
        )
    return parse_eval_info(eval_info)
