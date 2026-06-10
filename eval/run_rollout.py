"""Rollout evaluation: run each selected LIBERO-Plus task ONCE and record success (Phase 1+).

Task-SELECTION semantics: the eval set is a list of pre-built task IDs (clean + perturbed cells);
each is run a single trial (``num_trials_per_task = 1``). Output is a per-trial CSV with the exact
schema consumed by ``scripts/analyze_results.py``::

    condition, task_id, family, level, seed, success

Pure logic (eval-plan construction, row/aggregate building, CSV/JSON writing) is implemented and
unit-tested here. Loading the policy and stepping the simulator are the GPU seams
(``_load_policy`` / ``_run_one_trial``), which mirror LIBERO-Plus ``benchmark_scripts/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from eval.metrics import success_rate
from perturb.libero_plus_wrapper import LiberoPlusTask, select_perturbed_tasks

REPO = Path(__file__).resolve().parent.parent

#: Per-trial CSV columns (exact schema consumed by scripts/analyze_results.py).
ROLLOUT_FIELDS: tuple[str, ...] = ("condition", "task_id", "family", "level", "seed", "success")

CLEAN_FAMILY = "clean"
CLEAN_LEVEL = 0


@dataclass(frozen=True)
class EvalItem:
    """One unit of evaluation: a task UID labelled with its family and difficulty level."""

    task_uid: str
    family: str
    level: int


def build_eval_plan(
    tasks: Sequence[LiberoPlusTask],
    families: Sequence[str],
    levels: Sequence[int],
    instances_per_cell: int,
    clean_uids: Sequence[str] = (),
    *,
    seed: int = 0,
) -> list[EvalItem]:
    """Build the (clean + perturbed) list of :class:`EvalItem`s to roll out (pure, deterministic).

    Clean UIDs (original LIBERO tasks) are labelled ``family="clean", level=0``; perturbed task
    UIDs are selected per ``(family, level)`` cell via :func:`select_perturbed_tasks`.
    """
    plan: list[EvalItem] = [EvalItem(uid, CLEAN_FAMILY, CLEAN_LEVEL) for uid in clean_uids]
    for family in families:
        for level in levels:
            for uid in select_perturbed_tasks(tasks, family, level, instances_per_cell, seed=seed):
                plan.append(EvalItem(uid, family, level))
    return plan


def make_row(
    condition: str, task_id: str, family: str, level: int, seed: int, success: bool | int
) -> dict:
    """Construct one per-trial CSV row dict (validated)."""
    if success not in (0, 1, True, False):
        raise ValueError(f"success must be 0/1/bool, got {success!r}")
    return {
        "condition": str(condition),
        "task_id": str(task_id),
        "family": str(family),
        "level": int(level),
        "seed": int(seed),
        "success": int(bool(success)),
    }


def aggregate_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Aggregate per-trial rows to success rate per ``(condition, family, level)`` (pure)."""
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for r in rows:
        groups[(str(r["condition"]), str(r["family"]), int(r["level"]))].append(int(r["success"]))
    out: list[dict] = []
    for (condition, family, level), successes in groups.items():
        out.append({
            "condition": condition, "family": family, "level": level,
            "n": len(successes), "n_success": int(sum(successes)),
            "success_rate": success_rate(successes),
        })
    return out


def write_rows_csv(rows: Sequence[Mapping], path: str | Path) -> Path:
    """Write per-trial rows to a CSV with the canonical :data:`ROLLOUT_FIELDS` header."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROLLOUT_FIELDS))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in ROLLOUT_FIELDS})
    return out


def write_summary(rows: Sequence[Mapping], out_path: str | Path) -> tuple[Path, Path]:
    """Write the per-trial CSV (``out_path``) + an aggregate JSON sidecar. Returns both paths."""
    csv_path = write_rows_csv(rows, out_path)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps({"n_rows": len(rows), "aggregate": aggregate_rows(rows)}, indent=2))
    return csv_path, json_path


# ----------------------------------------------------------------- GPU seams --
def _load_policy(model_config: str, adapter: str | None):
    """Load SmolVLA base; optionally attach a LoRA adapter (Phase 3, B/C/D).

    ``model_config`` is currently ignored (canonical base = ``HuggingFaceVLA/smolvla_libero``);
    Phase 2+ will read ``hf_checkpoint`` from the YAML. Returned policy is on CUDA in eval mode.
    """
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero")
    policy = policy.to(device)
    policy.eval()

    if adapter is not None:
        raise NotImplementedError(f"Phase 3: attach LoRA adapter from {adapter}")
    return policy


#: SmolVLA-LIBERO image resolution (set by policy.config.input_features, verified at runtime).
_SMOLVLA_IMG_SIZE = 256
#: Max tokens for the task instruction (matches policy.config.tokenizer_max_length).
_SMOLVLA_MAX_LANG_TOKENS = 48


def _quat2axisangle(quat_xyzw):
    """Convert a single quaternion ``(x, y, z, w)`` to axis-angle ``(3,)``.

    Exactly mirrors LeRobot ``LiberoProcessorStep._quat2axisangle`` for a single sample.
    """
    import numpy as np
    q = np.asarray(quat_xyzw, dtype=np.float32)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arccos(w))
    axis = q[:3] / den
    return (axis * angle).astype(np.float32)


def _libero_obs_to_smolvla(obs, instruction: str, tokenizer, device: str = "cuda") -> dict:
    """Pack a raw LIBERO obs dict into the SmolVLA input dict.

    Faithfully reproduces LeRobot's ``LiberoProcessorStep`` (env_processor.py) so the policy
    sees the same observation distribution it was trained on:
      * **Images**: HxWx3 uint8 -> (1, 3, H, W) float[0,1] -> resize to 256x256 -> **flip both
        H and W** (180-deg rotation per the HuggingFaceVLA/libero camera convention).
      * **State**: ``eef_pos(3) + axisangle(quat->3) + gripper_qpos(2) = 8``.
      * **Language**: HF tokenizer with ``max_length=48``; attention_mask cast to bool
        (LeRobot's eager_attention_forward expects bool).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    def _prep_image(arr_hw3_uint8):
        t = torch.from_numpy(arr_hw3_uint8.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if t.shape[-1] != _SMOLVLA_IMG_SIZE or t.shape[-2] != _SMOLVLA_IMG_SIZE:
            t = F.interpolate(t, size=(_SMOLVLA_IMG_SIZE, _SMOLVLA_IMG_SIZE),
                              mode="bilinear", align_corners=False)
        # 180-deg rotation: flip both H (dim 2) and W (dim 3). MATCHES LiberoProcessorStep.
        t = torch.flip(t, dims=[2, 3])
        return t.to(device)

    img1 = _prep_image(obs["agentview_image"])
    img2 = _prep_image(obs["robot0_eye_in_hand_image"])

    # State: eef_pos(3) + axisangle(3) + gripper_qpos(2) = 8. MATCHES LiberoProcessorStep.
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_axisangle = _quat2axisangle(obs["robot0_eef_quat"])
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos]).astype(np.float32)
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)

    tok = tokenizer(
        instruction, return_tensors="pt", padding="max_length",
        max_length=_SMOLVLA_MAX_LANG_TOKENS, truncation=True,
    )
    return {
        "observation.images.image": img1,
        "observation.images.image2": img2,
        "observation.state": state_t,
        "observation.language.tokens": tok["input_ids"].to(device),
        "observation.language.attention_mask": tok["attention_mask"].to(device).to(torch.bool),
        "task": [instruction],
    }


def _smolvla_tokenizer(policy):
    """Reach into the SmolVLA policy for the underlying HF tokenizer."""
    return policy.model.vlm_with_expert.processor.tokenizer


def _suite_max_steps(task_uid: str) -> int:
    """Per-suite episode-length budget (mirrors LeRobot TASK_SUITE_MAX_STEPS)."""
    from perturb.libero_plus_wrapper import (
        LIBERO_DEFAULT_MAX_STEPS,
        LIBERO_SUITE_MAX_STEPS,
        parse_task_uid,
    )
    suite, _ = parse_task_uid(task_uid)
    return LIBERO_SUITE_MAX_STEPS.get(suite, LIBERO_DEFAULT_MAX_STEPS)


def _run_one_trial(policy, task_uid: str, seed: int, *, max_steps: int | None = None) -> int:
    """Run ``task_uid`` once with ``policy``; return success (0/1).

    Mirrors LeRobot ``LiberoEnv`` semantics:
      * Reset via ``libero_reset_with_init`` (set_init_state + 10 dummy settle steps + relative ctrl)
      * Per-suite ``max_steps`` (default = LIBERO_SUITE_MAX_STEPS)
      * Success = ``env.check_success()`` checked every step
    """
    import numpy as np
    import torch
    from perturb.libero_plus_wrapper import make_perturbed_env, libero_reset_with_init

    if max_steps is None:
        max_steps = _suite_max_steps(task_uid)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = _smolvla_tokenizer(policy)
    env = make_perturbed_env(task_uid)
    instruction = env._libero_task_instruction
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs = libero_reset_with_init(env, seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()  # clear action-chunk buffer between episodes

        success = False
        for _ in range(max_steps):
            obs_dict = _libero_obs_to_smolvla(obs, instruction, tokenizer, device=device)
            with torch.no_grad():
                action_t = policy.select_action(obs_dict)
            action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
            obs, _reward, done, _info = env.step(action)
            if env.check_success():
                success = True
                break
            if done:
                break
        return int(success)
    finally:
        try:
            env.close()
        except Exception:
            pass


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    """Compose plan -> per-task trial -> CSV/JSON summary (only the trial itself is a GPU seam)."""
    if args.tasks:
        plan = [EvalItem(uid, "unknown", 0) for uid in args.tasks]
    else:
        from data.prepare_libero_subset import (  # local import: avoids loading at module scope
            find_task_classification,
            load_task_classification,
        )
        from perturb.libero_plus_wrapper import parse_task_classification

        src = Path(args.classification) if args.classification else find_task_classification()
        tasks = parse_task_classification(load_task_classification(src))
        plan = build_eval_plan(
            tasks, args.families, args.levels, args.instances_per_cell, args.clean_tasks or [],
            seed=args.seed,
        )

    out = Path(args.out) if args.out else REPO / "analysis" / "runs" / f"{args.condition}_seed{args.seed}.csv"
    print(f"eval plan: {len(plan)} task trials (condition {args.condition}, seed {args.seed})")

    policy = _load_policy(args.model_config, args.adapter)  # GPU seam (raises until Phase 1)
    rows = [
        make_row(args.condition, item.task_uid, item.family, item.level, args.seed,
                 _run_one_trial(policy, item.task_uid, args.seed))
        for item in plan
    ]
    return write_summary(rows, out)


def build_parser() -> argparse.ArgumentParser:
    """Construct the rollout CLI parser."""
    p = argparse.ArgumentParser(description="VLA-Collapse-Recover rollout evaluation (1 trial/task).")
    p.add_argument("--model-config", default="configs/model/smolvla.yaml")
    p.add_argument("--eval-config", default="configs/eval/default.yaml")
    p.add_argument("--condition", default="A", help="Condition label for the CSV (A/B/C/D).")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (B/C/D); None = base (A).")
    p.add_argument("--tasks", nargs="*", default=None, help="Explicit task UIDs (ad-hoc run).")
    p.add_argument("--classification", default=None, help="Path to task_classification.json.")
    p.add_argument("--families", nargs="*", default=["viewpoint", "lighting", "texture", "noise"])
    p.add_argument("--levels", nargs="*", type=int, default=[2, 4])
    p.add_argument("--instances-per-cell", type=int, default=20)
    p.add_argument("--clean-tasks", nargs="*", default=None, help="Clean/original task UIDs.")
    p.add_argument("--seed", type=int, default=0, help="Seed (a LoRA-training seed for B/C/D).")
    p.add_argument("--out", default=None, help="Output CSV path (default: analysis/runs/<auto>.csv).")
    return p


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and dispatch to :func:`run`."""
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
