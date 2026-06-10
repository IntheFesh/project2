"""LoRA fine-tuning of SmolVLA on augmented LIBERO data (Phase 3).

Conditions (differ ONLY in augmentation):
    B = LoRA + standard generic photometric augmentation (LeRobot defaults).
    C = LoRA + perturbation-targeted augmentation aligned to the eval families.

This is a thin, reproducible wrapper over the *verified* ``lerobot-train`` CLI (the same
invocation validated end-to-end in Phase 3 smoke tests). Augmentation is expressed as a LeRobot
``image_transforms.tfs`` dict (data/augment/visual_aug.py) and passed via a single
``--dataset.image_transforms.tfs`` JSON override -- no hand-written training loop, no custom
transform classes. Adapter-only checkpoints (base frozen); GPU at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from data.augment.visual_aug import build_targeted_tfs
from perturb.libero_plus_wrapper import CORE_FAMILIES

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = "/root/autodl-tmp/vcr/hf/lerobot/HuggingFaceVLA/libero"
DEFAULT_EPISODES_JSON = REPO_ROOT / "configs" / "spatial_episodes.json"
POLICY_BASE = "HuggingFaceVLA/smolvla_libero"   # = condition A; LoRA rides on top
LIBERO_ORIG = REPO_ROOT / "third_party" / "LIBERO-orig"


def _train_env() -> dict:
    """Verified env: hf-mirror, proxies unset, offline (data is local), datasets cache off system disk."""
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    env["HF_DATASETS_CACHE"] = "/root/autodl-tmp/vcr/hf/datasets_cache"
    env["PYTHONPATH"] = str(LIBERO_ORIG) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for p in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(p, None)
    return env


def _episodes_arg(episodes_json: Path, smoke: bool) -> str:
    """Build the --dataset.episodes value: spatial subset (or first 5 for smoke)."""
    eps = json.loads(Path(episodes_json).read_text())
    if smoke:
        eps = eps[:5]
    return "[" + ",".join(str(int(e)) for e in eps) + "]"


def build_train_command(
    *, condition: str, aug_families: list[str], out_dir: str, seed: int,
    r: int, alpha: int, steps: int, batch_size: int, dataset_root: str,
    episodes_json: Path, smoke: bool, max_num_transforms: int = 3,
) -> list[str]:
    """Construct the verified ``lerobot-train`` argv for condition B or C."""
    cmd = [
        "lerobot-train",
        f"--policy.path={POLICY_BASE}",
        "--dataset.repo_id=HuggingFaceVLA/libero",
        f"--dataset.root={dataset_root}",
        f"--dataset.episodes={_episodes_arg(episodes_json, smoke)}",
        "--dataset.image_transforms.enable=true",
        "--env.type=libero", "--env.task=libero_spatial",
        "--peft.method_type=LORA", f"--peft.r={r}", f"--peft.lora_alpha={alpha}",
        f"--batch_size={batch_size}", f"--steps={steps}",
        "--log_freq=250", "--eval_freq=1000000",
        f"--save_freq={'50' if smoke else '5000'}",
        "--policy.device=cuda", "--policy.push_to_hub=false", "--wandb.enable=false",
        f"--output_dir={out_dir}", f"--job_name=vcr_{condition}_seed{seed}", f"--seed={seed}",
    ]
    if condition == "C":
        tfs = build_targeted_tfs(aug_families)
        cmd += [
            f"--dataset.image_transforms.tfs={json.dumps(tfs, separators=(',', ':'))}",
            f"--dataset.image_transforms.max_num_transforms={max_num_transforms}",
            "--dataset.image_transforms.random_order=false",
        ]
    # Condition B: leave LeRobot's default tfs (generic photometric) in place via enable=true.
    return cmd


def run(args: argparse.Namespace) -> None:
    """Run LoRA fine-tuning to completion via the verified lerobot-train CLI; adapter saved."""
    if args.condition not in ("B", "C"):
        raise ValueError("--condition must be B or C")
    fams = args.aug_families or list(CORE_FAMILIES)
    out_dir = args.out_dir or f"adapters/{args.condition}_seed{args.seed}_a{args.alpha}"
    cmd = build_train_command(
        condition=args.condition, aug_families=fams, out_dir=out_dir, seed=args.seed,
        r=args.r, alpha=args.alpha,
        steps=(60 if args.smoke_test else args.steps),
        batch_size=args.batch_size, dataset_root=args.dataset_root,
        episodes_json=Path(args.episodes_json), smoke=args.smoke_test,
    )
    print(f"[train_lora] condition={args.condition} families={fams} out={out_dir}")
    print("[train_lora] tfs =", "targeted-C" if args.condition == "C" else "standard-B (LeRobot default)")
    print("[train_lora] CMD:\n  " + " ".join(cmd))
    subprocess.run(cmd, env=_train_env(), check=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA fine-tune SmolVLA on augmented LIBERO data.")
    p.add_argument("--condition", choices=["B", "C"], required=True)
    p.add_argument("--aug-families", nargs="*", default=None,
                   help="For C: families to align augmentation to (default: 4 CORE families).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dataset-root", default=DEFAULT_ROOT)
    p.add_argument("--episodes-json", default=str(DEFAULT_EPISODES_JSON))
    p.add_argument("--out-dir", default=None, help="Default: adapters/{cond}_seed{seed}_a{alpha}.")
    p.add_argument("--smoke-test", action="store_true",
                   help="60 steps + 5 episodes to validate the tfs/CLI wiring before scaling.")
    return p


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
