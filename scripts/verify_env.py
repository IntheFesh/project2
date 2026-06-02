#!/usr/bin/env python3
"""Verify the GPU environment for VLA-Collapse-Recover.

Asserts the rented card actually exposes Blackwell ``sm_120`` with a CUDA-12.8 / PyTorch>=2.7
stack and that LeRobot imports. Designed to FAIL LOUDLY (nonzero exit) when run on a
CPU/no-GPU container -- that is expected off-GPU. Run this on the rented RTX 5090::

    uv sync --extra gpu
    pip install "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot"
    python scripts/verify_env.py
"""

from __future__ import annotations

import sys


def _check(label: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def main() -> int:
    ok_all = True

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _check("torch import", False, f"{exc!r}  (install: uv sync --extra gpu)")
        print("\nCannot continue without torch.")
        return 1

    tv = torch.__version__
    major_minor = tuple(int(x) for x in tv.split("+")[0].split(".")[:2])
    ok_all &= _check("torch >= 2.7", major_minor >= (2, 7), tv)

    cuda_build = getattr(torch.version, "cuda", None)
    ok_all &= _check(
        "torch CUDA build == 12.8 (cu128)",
        cuda_build is not None and cuda_build.startswith("12.8"),
        str(cuda_build),
    )

    avail = torch.cuda.is_available()
    ok_all &= _check("CUDA available", avail, str(avail))

    if avail:
        cap = torch.cuda.get_device_capability()
        ok_all &= _check("device capability == (12, 0) sm_120", cap == (12, 0), str(cap))
        _check("device name", True, torch.cuda.get_device_name(0))
        # Tiny op to catch "no kernel image is available for execution on the device".
        try:
            x = torch.randn(8, 8, device="cuda")
            _ = (x @ x).sum().item()
            ok_all &= _check("cuda matmul smoke", True, "ok")
        except Exception as exc:  # noqa: BLE001
            ok_all &= _check("cuda matmul smoke", False, repr(exc))
    else:
        ok_all &= _check(
            "device capability == (12, 0) sm_120", False, "no CUDA device (expected off-GPU)"
        )

    try:
        import lerobot  # noqa: F401

        _check("lerobot import", True, getattr(lerobot, "__version__", "ok"))
    except Exception as exc:  # noqa: BLE001
        ok_all &= _check(
            "lerobot import", False,
            f"{exc!r}  (pip install 'lerobot[smolvla] @ git+https://github.com/huggingface/lerobot')",
        )

    print("\n" + ("ALL CHECKS PASSED -- sm_120 ready." if ok_all else "ENV NOT READY (see FAILs)."))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
