"""LoRA fine-tuning of SmolVLA on augmented data (Phase 3).

Conditions:
    B = LoRA + standard (generic torchvision) augmentation.
    C = LoRA + perturbation-targeted augmentation (aligned to the eval families).

Saves the adapter only (disk discipline). GPU-only at runtime. Config-driven via
``configs/lora/*.yaml`` and ``configs/model/*.yaml``.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Construct the LoRA-training CLI parser."""
    p = argparse.ArgumentParser(description="LoRA fine-tune SmolVLA on augmented LIBERO data.")
    p.add_argument("--model-config", default="configs/model/smolvla.yaml")
    p.add_argument("--lora-config", default="configs/lora/smolvla_r16.yaml")
    p.add_argument("--condition", choices=["B", "C"], required=False,
                   help="B = standard aug; C = perturbation-targeted aug.")
    p.add_argument("--aug-families", nargs="*", default=None,
                   help="For condition C: eval families to align augmentation to.")
    p.add_argument("--out-dir", default="adapters/", help="Adapter output dir (adapter only).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny subset + few steps to time per-run wall-clock before scaling.")
    return p


def run(args: argparse.Namespace) -> None:
    """Run LoRA fine-tuning to completion and save the adapter."""
    raise NotImplementedError(
        "Phase 3: build augmented dataset, attach LoRA via PEFT, train, save adapter only."
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and dispatch to :func:`run`."""
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
